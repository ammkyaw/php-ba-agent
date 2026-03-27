"""
pipeline/stage13_entrypoints.py — System Entry-Point Catalog (Stage 1.3)
                                   Multi-Language Dispatcher

Runs between Stage 1 (code parsing) and Stage 1.5 (execution paths).
Detects ALL system entry points — not just browser-facing HTTP pages — using
purely static analysis (zero LLM calls).

Entry-point types detected
--------------------------
  http         — Standard browser-facing pages / API handlers
  scheduled    — Cron jobs and scheduled tasks
  cli          — Admin / maintenance CLI scripts
  webhook      — Incoming HTTP callbacks from external systems
  queue_worker — Async queue / message-queue job handlers

Language dispatch
-----------------
  PHP        → pipeline/entrypoints/php_entrypoints  (original Laravel/Symfony logic)
  TypeScript → pipeline/entrypoints/typescript_entrypoints
  Java       → pipeline/entrypoints/java_entrypoints

Framework-specific detection (PHP)
------------------------------------
  Laravel  : app/Console/Commands/, app/Jobs/, app/Listeners/, Kernel.php schedule
  Symfony  : src/Command/, src/MessageHandler/, src/Handler/
  Generic  : shebang scripts, cron dirs, crontab configs, webhook signature patterns

Downstream consumers
--------------------
  Stage 1.5  — augments source-file set with non-HTTP handler files
  Stage 4.5  — tags BusinessFlow.flow_type from evidence_files → ep_type
  Stage 5    — injects background operation descriptions into BRD prompt

Resume behaviour
----------------
If stage13_entrypoints is COMPLETED and entry_point_catalog.json exists, the
stage is skipped and ctx.entry_point_catalog is restored from disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from context import EntryPoint, EntryPointCatalog, Framework, Language, PipelineContext

OUTPUT_FILE = "entry_point_catalog.json"

# Directories to skip during file-tree walks
_SKIP_DIRS = {
    "vendor", "node_modules", ".git", "cache", "logs", "storage",
    "tests", "test", "spec", "stubs", "fixtures", "public", "assets",
}

# Confidence levels
_CONF_CONFIG  = 1.0   # backed by an explicit config / crontab entry
_CONF_PATTERN = 0.8   # matched a strong directory / class pattern
_CONF_HEUR    = 0.6   # matched a filename / content heuristic


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 1.3 entry point. Detects system entry points and populates
    ctx.entry_point_catalog.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If code_map is missing (Stage 1 not run).
    """
    output_path = ctx.output_path(OUTPUT_FILE)

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage13_entrypoints") and Path(output_path).exists():
        ctx.entry_point_catalog = _load_catalog(output_path)
        cat = ctx.entry_point_catalog
        print(
            f"  [stage13] Already completed — "
            f"{cat.total} entry point(s): "
            + ", ".join(f"{k}={len(v)}" for k, v in sorted(cat.by_type.items()))
        )
        return

    if ctx.code_map is None:
        raise RuntimeError("[stage13] ctx.code_map is None — run Stage 1 first.")

    framework = ctx.code_map.framework
    language  = ctx.code_map.language
    root      = Path(ctx.project_path)

    print(
        f"  [stage13] Scanning entry points "
        f"(language={language.value}, framework={framework.value}, root={root.name}) ..."
    )

    eps = _dispatch_detect(root, language, framework)

    # Universal pass: add HTTP entry points from parsed routes (all languages)
    eps += _detect_http_from_routes(ctx.code_map)

    # Final dedup across all sources (non-HTTP by file, HTTP by file+trigger)
    eps = _dedup(eps)

    # Assign sequential IDs
    for i, ep in enumerate(eps, 1):
        ep.ep_id = f"ep_{i:03d}"

    catalog = _build_catalog(eps)
    ctx.entry_point_catalog = catalog

    _save_catalog(catalog, output_path)
    ctx.stage("stage13_entrypoints").mark_completed(output_path)
    ctx.save()

    print(f"  [stage13] Done → {catalog.total} entry point(s)")
    for ep_type, ids in sorted(catalog.by_type.items()):
        eps_of_type = [e for e in eps if e.ep_type == ep_type]
        for ep in eps_of_type[:5]:
            print(f"  [stage13]   [{ep_type}] {ep.name}  ({ep.handler_file})")
        if len(eps_of_type) > 5:
            print(f"  [stage13]   [{ep_type}]  … and {len(eps_of_type)-5} more")


# ─── Language Dispatcher ────────────────────────────────────────────────────────

def _dispatch_detect(root: Path, language: Language, framework: Framework) -> list[EntryPoint]:
    """Route to the correct language-specific entry-point detector."""
    if language in (Language.TYPESCRIPT, Language.JAVASCRIPT):
        from pipeline.entrypoints.typescript_entrypoints import detect as ts_detect  # noqa: PLC0415
        return ts_detect(root, framework)
    if language in (Language.JAVA, Language.KOTLIN):
        from pipeline.entrypoints.java_entrypoints import detect as java_detect  # noqa: PLC0415
        return java_detect(root, framework)
    # PHP (or UNKNOWN — fall back to PHP for backward compat)
    return _detect_all(root, framework)


# ─── PHP Detection Orchestrator ────────────────────────────────────────────────

def _detect_all(root: Path, framework: Framework) -> list[EntryPoint]:
    """Run all detectors, deduplicate by handler_file."""
    results: list[EntryPoint] = []

    if framework == Framework.LARAVEL:
        results += _detect_laravel_commands(root)
        results += _detect_laravel_jobs(root)
        results += _detect_laravel_listeners(root)
        results += _detect_laravel_schedule(root)

    elif framework == Framework.SYMFONY:
        results += _detect_symfony_commands(root)
        results += _detect_symfony_handlers(root)

    # Generic passes run for all frameworks
    results += _detect_generic_cli_scripts(root)
    results += _detect_webhook_files(root)
    results += _detect_cron_dirs(root)
    results += _detect_cron_configs(root)
    results += _detect_raw_queue_workers(root)

    return _dedup(results)


# ─── Universal HTTP Entry-Point Pass ───────────────────────────────────────────

def _detect_http_from_routes(code_map) -> list[EntryPoint]:
    """
    Convert CodeMap routes into HTTP EntryPoints.

    Called after language-specific detection so that all three parsers
    (PHP, TypeScript, Java) automatically get HTTP entries without each
    language module needing to duplicate the logic.

    One EntryPoint per unique (method, path) pair.  When multiple routes
    share the same handler_file (e.g. a Next.js route.ts with GET + POST),
    the _dedup pass downstream keeps the first one encountered — the file
    still maps to ep_type "http" in the ep_type_map used by stage45.
    """
    if not code_map or not code_map.routes:
        return []

    eps: list[EntryPoint] = []
    for r in code_map.routes:
        handler_file = r.get("file", "")
        if not handler_file:
            continue
        method  = (r.get("method") or "GET").upper()
        path    = r.get("path") or "/"
        handler = r.get("handler") or ""
        kind    = r.get("kind") or r.get("source") or "http"

        # Build a human-readable name
        path_label = path.rstrip("/") or "/"
        name = f"{method} {path_label}"
        if handler and handler not in (method, "handler"):
            name = f"{name}  [{handler}]"

        eps.append(EntryPoint(
            ep_type      = "http",
            handler_file = handler_file,
            name         = name,
            trigger      = f"HTTP {method} {path_label}",
            confidence   = _CONF_CONFIG,
        ))

    return eps


# ─── Laravel Detectors ─────────────────────────────────────────────────────────

def _detect_laravel_commands(root: Path) -> list[EntryPoint]:
    """Scan app/Console/Commands/*.php for Artisan CLI commands."""
    eps: list[EntryPoint] = []
    cmd_dir = root / "app" / "Console" / "Commands"
    if not cmd_dir.is_dir():
        return eps

    for php_file in cmd_dir.glob("**/*.php"):
        src = _safe_read(php_file)
        if not src:
            continue
        # Extract $signature = 'emails:send'; or const COMMAND_NAME
        sig = _extract_regex(src, r"""\$signature\s*=\s*['"]([^'"]+)['"]""")
        name = _humanize(sig) if sig else _humanize(php_file.stem)
        trigger = f"php artisan {sig}" if sig else f"php artisan {_to_slug(php_file.stem)}"
        eps.append(EntryPoint(
            ep_type      = "cli",
            handler_file = _rel(php_file, root),
            name         = name,
            trigger      = trigger,
            confidence   = _CONF_PATTERN,
        ))
    return eps


def _detect_laravel_jobs(root: Path) -> list[EntryPoint]:
    """Scan app/Jobs/*.php for ShouldQueue implementations."""
    eps: list[EntryPoint] = []
    jobs_dir = root / "app" / "Jobs"
    if not jobs_dir.is_dir():
        return eps

    for php_file in jobs_dir.glob("**/*.php"):
        src = _safe_read(php_file)
        if not src or "ShouldQueue" not in src:
            continue
        # Extract queue name hint
        queue_name = _extract_regex(src, r"""\$queue\s*=\s*['"]([^'"]+)['"]""") or "default"
        name = _humanize(php_file.stem)
        eps.append(EntryPoint(
            ep_type      = "queue_worker",
            handler_file = _rel(php_file, root),
            name         = name,
            trigger      = f"Queue: {queue_name}  (php artisan queue:work)",
            confidence   = _CONF_PATTERN,
        ))
    return eps


def _detect_laravel_listeners(root: Path) -> list[EntryPoint]:
    """Scan app/Listeners/*.php for queued event listeners."""
    eps: list[EntryPoint] = []
    listeners_dir = root / "app" / "Listeners"
    if not listeners_dir.is_dir():
        return eps

    for php_file in listeners_dir.glob("**/*.php"):
        src = _safe_read(php_file)
        if not src or "ShouldQueue" not in src:
            continue
        name = _humanize(php_file.stem)
        eps.append(EntryPoint(
            ep_type      = "queue_worker",
            handler_file = _rel(php_file, root),
            name         = f"{name} (Listener)",
            trigger      = "Queue: default  (event-driven)",
            confidence   = _CONF_PATTERN,
        ))
    return eps


def _detect_laravel_schedule(root: Path) -> list[EntryPoint]:
    """
    Parse app/Console/Kernel.php and routes/console.php for scheduled tasks.
    Extracts ->command(...)->cron(...) and ->call(...) patterns.
    """
    eps: list[EntryPoint] = []
    candidates = [
        root / "app" / "Console" / "Kernel.php",
        root / "routes" / "console.php",
    ]
    # Laravel 9+ also stores schedule in app/Providers/AppServiceProvider.php
    for provider in (root / "app" / "Providers").glob("*.php"):
        candidates.append(provider)

    _FLUENT_CRON = {
        "->everyMinute()":    "* * * * *",
        "->everyFiveMinutes()": "*/5 * * * *",
        "->everyTenMinutes()":  "*/10 * * * *",
        "->hourly()":         "0 * * * *",
        "->daily()":          "0 0 * * *",
        "->dailyAt(":         "0 HH * * *",   # partial — overridden below
        "->weekly()":         "0 0 * * 0",
        "->monthly()":        "0 0 1 * *",
        "->yearly()":         "0 0 1 1 *",
        "->twiceDaily(":      "0 1,13 * * *",
    }

    for path in candidates:
        if not path.exists():
            continue
        src = _safe_read(path)
        if not src or ("schedule" not in src and "Schedule" not in src):
            continue

        # Find ->command('X')->...->cron('expr') chains
        for m in re.finditer(
            r"""->command\(\s*['"]([^'"]+)['"]\s*\)([^;]{0,200}?)"""
            r"""(?:->cron\(\s*['"]([^'"]+)['"]\s*\))?""",
            src, re.DOTALL
        ):
            cmd, chain, cron_expr = m.group(1), m.group(2), m.group(3) or ""

            # Infer schedule from fluent methods when ->cron() is absent
            if not cron_expr:
                for fluent, expr in _FLUENT_CRON.items():
                    if fluent in chain:
                        cron_expr = expr
                        break

            # Extract dailyAt time
            da = re.search(r"dailyAt\(\s*'(\d{1,2}:\d{2})'", chain)
            if da:
                h, mi = da.group(1).split(":")
                cron_expr = f"{mi} {h} * * *"

            name = _humanize(cmd.replace(":", " "))
            eps.append(EntryPoint(
                ep_type      = "scheduled",
                handler_file = _rel(path, root),
                name         = f"Scheduled: {name}",
                schedule     = cron_expr or "see Kernel.php",
                trigger      = f"php artisan {cmd}",
                confidence   = _CONF_CONFIG,
            ))

        # Find ->call(...) closures / invokable (harder to name, use heuristic)
        for m in re.finditer(
            r"""->call\(\s*['"]([^'"]+)['"]""",
            src
        ):
            func_name = m.group(1)
            name = _humanize(func_name.replace(":", " ").replace("/", " ").replace("\\", " "))
            eps.append(EntryPoint(
                ep_type      = "scheduled",
                handler_file = _rel(path, root),
                name         = f"Scheduled Closure: {name}",
                schedule     = "see schedule definition",
                trigger      = f"Closure call: {func_name[:60]}",
                confidence   = _CONF_HEUR,
            ))

    return eps


# ─── Symfony Detectors ─────────────────────────────────────────────────────────

def _detect_symfony_commands(root: Path) -> list[EntryPoint]:
    """Scan src/Command/ for Symfony console commands."""
    eps: list[EntryPoint] = []
    for cmd_dir in [root / "src" / "Command",
                    root / "src" / "Console" / "Command"]:
        if not cmd_dir.is_dir():
            continue
        for php_file in cmd_dir.glob("**/*.php"):
            src = _safe_read(php_file)
            if not src:
                continue
            if "extends Command" not in src and "AbstractCommand" not in src:
                continue
            # Extract NAME constant or #[AsCommand(name: '...')]
            cmd_name = (
                _extract_regex(src, r"""const\s+NAME\s*=\s*['"]([^'"]+)['"]""")
                or _extract_regex(src, r"""AsCommand\s*\(\s*name\s*:\s*['"]([^'"]+)['"]""")
                or _extract_regex(src, r"""\$defaultName\s*=\s*['"]([^'"]+)['"]""")
                or _to_slug(php_file.stem)
            )
            name = _humanize(cmd_name.replace(":", " ").replace("-", " "))
            eps.append(EntryPoint(
                ep_type      = "cli",
                handler_file = _rel(php_file, root),
                name         = name,
                trigger      = f"php bin/console {cmd_name}",
                confidence   = _CONF_PATTERN,
            ))
    return eps


def _detect_symfony_handlers(root: Path) -> list[EntryPoint]:
    """Scan src/MessageHandler/, src/Handler/ for Symfony message handlers."""
    eps: list[EntryPoint] = []
    for handler_dir in [root / "src" / "MessageHandler",
                        root / "src" / "Handler",
                        root / "src" / "Message" / "Handler"]:
        if not handler_dir.is_dir():
            continue
        for php_file in handler_dir.glob("**/*.php"):
            src = _safe_read(php_file)
            if not src:
                continue
            if "MessageHandlerInterface" not in src and "__invoke" not in src:
                continue
            name = _humanize(php_file.stem.replace("Handler", "").strip("_"))
            eps.append(EntryPoint(
                ep_type      = "queue_worker",
                handler_file = _rel(php_file, root),
                name         = f"{name} Handler",
                trigger      = "Symfony Messenger bus dispatch",
                confidence   = _CONF_PATTERN,
            ))
    return eps


# ─── Generic Detectors (all frameworks) ────────────────────────────────────────

def _detect_generic_cli_scripts(root: Path) -> list[EntryPoint]:
    """
    Find CLI scripts by:
    1. Shebang: #!/usr/bin/env php
    2. php_sapi_name() / PHP_SAPI guards
    3. Location in bin/, scripts/, cli/, console/, commands/ directories
    """
    eps: list[EntryPoint] = []
    _CLI_DIRS  = {"bin", "scripts", "cli", "console", "commands", "tools", "utils"}
    _CLI_PATS  = [
        re.compile(r"php_sapi_name\s*\(\s*\)\s*===?\s*['\"]cli['\"]"),
        re.compile(r"PHP_SAPI\s*===?\s*['\"]cli['\"]"),
        re.compile(r"!isset\s*\(\s*\$_SERVER\s*\[\s*['\"]HTTP_HOST['\"]\s*\]\s*\)"),
    ]
    _SHEBANG = re.compile(r"^#!\s*/usr/bin/env\s+php")

    for php_file in root.rglob("*.php"):
        if any(part in _SKIP_DIRS for part in php_file.parts):
            continue
        # Directory-based detection
        in_cli_dir = any(
            part.lower() in _CLI_DIRS for part in php_file.relative_to(root).parts[:-1]
        )
        src = _safe_read(php_file)
        if not src:
            continue

        has_shebang = bool(_SHEBANG.match(src))
        has_cli_pat = any(p.search(src) for p in _CLI_PATS)

        if not (in_cli_dir or has_shebang or has_cli_pat):
            continue

        # Avoid duplicating files already covered by framework-specific detectors
        rel = _rel(php_file, root)
        if any(d in rel for d in ["Console/Commands", "src/Command", "src/Console"]):
            continue

        conf = _CONF_CONFIG if has_shebang else _CONF_PATTERN if has_cli_pat else _CONF_HEUR
        name = _humanize(php_file.stem)
        eps.append(EntryPoint(
            ep_type      = "cli",
            handler_file = rel,
            name         = name,
            trigger      = f"php {rel}",
            confidence   = conf,
        ))
    return eps


_WEBHOOK_CONTENT_PATS = [
    re.compile(r"HTTP_X_HUB_SIGNATURE", re.I),
    re.compile(r"HTTP_X_STRIPE_SIGNATURE", re.I),
    re.compile(r"HTTP_X_PAYPAL", re.I),
    re.compile(r"HTTP_X_WOOCOMMERCE", re.I),
    re.compile(r"HTTP_X_SHOPIFY", re.I),
    re.compile(r"HTTP_X_GITHUB_EVENT", re.I),
    re.compile(r"HTTP_X_TWILIO", re.I),
    re.compile(r"webhook.*secret|hmac.*sha", re.I),
]
_WEBHOOK_NAME_PATS = re.compile(
    r"(webhook|callback|ipn|notify|hook|postback|receive)", re.I
)


def _detect_webhook_files(root: Path) -> list[EntryPoint]:
    """Detect webhook receivers by content signatures and filename patterns."""
    eps: list[EntryPoint] = []
    for php_file in root.rglob("*.php"):
        if any(part in _SKIP_DIRS for part in php_file.parts):
            continue
        name_match = bool(_WEBHOOK_NAME_PATS.search(php_file.stem))
        if not name_match:
            continue  # fast path — only read files with webhook-y names

        src = _safe_read(php_file)
        if not src:
            continue

        content_match = any(p.search(src) for p in _WEBHOOK_CONTENT_PATS)
        if not (name_match and (content_match or "POST" in src)):
            continue

        conf = _CONF_CONFIG if content_match else _CONF_HEUR
        name = _humanize(php_file.stem.lower()
                         .replace("webhook", "")
                         .replace("callback", "")
                         .replace("notify", "notification")
                         .strip("_-") or php_file.stem)

        eps.append(EntryPoint(
            ep_type      = "webhook",
            handler_file = _rel(php_file, root),
            name         = f"{name} Webhook Receiver",
            trigger      = f"HTTP POST → /{php_file.stem}.php (external system)",
            confidence   = conf,
        ))
    return eps


_CRON_DIRS = {"cron", "crons", "jobs", "tasks", "scheduler", "scheduled", "batch"}


def _detect_cron_dirs(root: Path) -> list[EntryPoint]:
    """Detect PHP files living in directories conventionally used for cron jobs."""
    eps: list[EntryPoint] = []
    for php_file in root.rglob("*.php"):
        if any(part in _SKIP_DIRS for part in php_file.parts):
            continue
        parts = [p.lower() for p in php_file.relative_to(root).parts[:-1]]
        if not any(part in _CRON_DIRS for part in parts):
            continue
        name = _humanize(php_file.stem)
        eps.append(EntryPoint(
            ep_type      = "scheduled",
            handler_file = _rel(php_file, root),
            name         = f"Scheduled: {name}",
            schedule     = "see crontab",
            trigger      = f"php {_rel(php_file, root)}",
            confidence   = _CONF_HEUR,
        ))
    return eps


def _detect_cron_configs(root: Path) -> list[EntryPoint]:
    """
    Parse crontab / .cron / supervisor config files to extract PHP script
    invocations with their cron expressions.
    """
    eps: list[EntryPoint] = []
    _CRON_FILENAMES = {
        "crontab", ".crontab", "crontab.txt",
        "schedule", "schedule.txt",
    }
    _CRON_EXTENSIONS = {".cron", ".tab"}

    candidates: list[Path] = []
    for item in root.rglob("*"):
        if item.is_file() and (
            item.name.lower() in _CRON_FILENAMES
            or item.suffix.lower() in _CRON_EXTENSIONS
            or (item.parent.name.lower() in {"cron.d", "cron", "crontabs"})
        ):
            candidates.append(item)

    # Cron line: "*/5 * * * *  php /path/to/script.php"
    _CRON_LINE = re.compile(
        r"^(\S+\s+\S+\s+\S+\s+\S+\s+\S+)\s+.*?php\s+([^\s]+\.php)",
        re.MULTILINE,
    )

    for cfg_file in candidates:
        src = _safe_read(cfg_file)
        if not src:
            continue
        for m in _CRON_LINE.finditer(src):
            cron_expr, script_path = m.group(1), m.group(2)
            # Normalise script path to relative
            script_path = script_path.lstrip("/")
            name = _humanize(Path(script_path).stem)
            eps.append(EntryPoint(
                ep_type      = "scheduled",
                handler_file = script_path,
                name         = f"Cron: {name}",
                schedule     = cron_expr,
                trigger      = f"php {script_path}",
                confidence   = _CONF_CONFIG,
            ))
    return eps


_QUEUE_PATS = [
    re.compile(r"pheanstalk|Pheanstalk", re.I),
    re.compile(r"Resque::enqueue|Resque::pop", re.I),
    re.compile(r"\\Redis.*rpush|\\Redis.*lpop", re.I),
    re.compile(r"amqp_connect|AMQPConnection", re.I),
    re.compile(r"beanstalk", re.I),
]
_QUEUE_WORKER_DIRS = {"workers", "consumers", "processors", "handlers", "queues"}


def _detect_raw_queue_workers(root: Path) -> list[EntryPoint]:
    """
    Detect raw queue-consumer PHP scripts (non-framework) by content patterns
    or location in known queue-worker directories.
    """
    eps: list[EntryPoint] = []
    for php_file in root.rglob("*.php"):
        if any(part in _SKIP_DIRS for part in php_file.parts):
            continue
        parts = [p.lower() for p in php_file.relative_to(root).parts[:-1]]
        in_queue_dir = any(p in _QUEUE_WORKER_DIRS for p in parts)

        src = _safe_read(php_file)
        if not src:
            continue

        has_queue_pat = any(p.search(src) for p in _QUEUE_PATS)
        if not (in_queue_dir or has_queue_pat):
            continue

        # Skip Laravel/Symfony jobs already caught upstream
        rel = _rel(php_file, root)
        if any(d in rel for d in ["app/Jobs", "app/Listeners", "MessageHandler", "src/Handler"]):
            continue

        conf = _CONF_PATTERN if has_queue_pat else _CONF_HEUR
        name = _humanize(php_file.stem)
        eps.append(EntryPoint(
            ep_type      = "queue_worker",
            handler_file = rel,
            name         = f"{name} Worker",
            trigger      = "Queue consumer (long-running process)",
            confidence   = conf,
        ))
    return eps


# ─── Catalog Building ──────────────────────────────────────────────────────────

def _build_catalog(eps: list[EntryPoint]) -> EntryPointCatalog:
    by_type: dict[str, list[str]] = {}
    for ep in eps:
        by_type.setdefault(ep.ep_type, []).append(ep.ep_id)
    return EntryPointCatalog(
        entry_points = eps,
        by_type      = by_type,
        total        = len(eps),
    )


def _dedup(eps: list[EntryPoint]) -> list[EntryPoint]:
    """
    Deduplicate entry points.

    - HTTP entries use (handler_file, trigger) as the key so that multiple
      routes in the same file (e.g. GET + POST in one route.ts) are each
      preserved as distinct catalog entries.
    - Non-HTTP entries use handler_file alone (one entry per file).  When
      the same file is claimed by multiple detectors, keep the one with the
      highest confidence; break ties by specificity order:
      queue_worker > scheduled > cli > webhook > http.
    """
    _TYPE_PRIORITY = {"queue_worker": 4, "scheduled": 3, "cli": 2, "webhook": 1, "http": 0}

    seen: dict[str, EntryPoint] = {}

    for ep in eps:
        # Unique key: HTTP routes differ by (file, trigger); non-HTTP by file only
        if ep.ep_type == "http":
            key = f"http::{ep.handler_file}::{ep.trigger}"
        else:
            key = ep.handler_file

        if key not in seen:
            seen[key] = ep
        else:
            existing = seen[key]
            ep_pri   = _TYPE_PRIORITY.get(ep.ep_type, 0)
            ex_pri   = _TYPE_PRIORITY.get(existing.ep_type, 0)
            if ep.confidence > existing.confidence or (
                ep.confidence == existing.confidence and ep_pri > ex_pri
            ):
                seen[key] = ep

    return list(seen.values())


# ─── Serialisation ─────────────────────────────────────────────────────────────

def _save_catalog(catalog: EntryPointCatalog, output_path: str) -> None:
    import dataclasses
    data = dataclasses.asdict(catalog)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"  [stage13] Catalog saved → {output_path}")


def _load_catalog(output_path: str) -> EntryPointCatalog:
    from context import EntryPoint as EP, EntryPointCatalog as EPC  # noqa: PLC0415
    with open(output_path, encoding="utf-8") as fh:
        d = json.load(fh)
    return EPC(
        entry_points = [
            EP(
                ep_id        = ep.get("ep_id", ""),
                ep_type      = ep.get("ep_type", "http"),
                handler_file = ep.get("handler_file", ""),
                name         = ep.get("name", ""),
                schedule     = ep.get("schedule", ""),
                trigger      = ep.get("trigger", ""),
                confidence   = ep.get("confidence", 0.8),
            )
            for ep in d.get("entry_points", [])
        ],
        by_type      = d.get("by_type", {}),
        total        = d.get("total", 0),
        generated_at = d.get("generated_at", ""),
    )


# ─── Utilities ─────────────────────────────────────────────────────────────────

def _safe_read(path: Path) -> Optional[str]:
    """Read a file, returning None on any error."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _rel(path: Path, root: Path) -> str:
    """Return path relative to root as a forward-slash string."""
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _extract_regex(text: str, pattern: str) -> Optional[str]:
    """Return the first capture group of `pattern` in `text`, or None."""
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _humanize(name: str) -> str:
    """
    Convert a CamelCase or snake_case identifier into a readable title.

    Examples
    --------
    "SendNightlyDigest"    → "Send Nightly Digest"
    "process_payments"     → "Process Payments"
    "emails:send"          → "Emails Send"
    """
    # Split CamelCase
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    # Replace non-alphanumeric (except spaces) with space
    name = re.sub(r"[^a-zA-Z0-9 ]+", " ", name)
    # Collapse spaces and title-case
    return " ".join(w.capitalize() for w in name.split() if w)


def _to_slug(name: str) -> str:
    """Convert a file stem to a kebab-case artisan-style slug."""
    name = re.sub(r"([a-z])([A-Z])", r"\1-\2", name).lower()
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


# ─── Public helper: handler_file → ep_type lookup ──────────────────────────────

def build_ep_type_map(ctx: PipelineContext) -> dict[str, str]:
    """
    Build a dict mapping each handler_file to its ep_type.
    Used by Stage 4.5 to tag BusinessFlow.flow_type.
    Returns an empty dict if entry_point_catalog is not populated.
    """
    if not ctx.entry_point_catalog:
        return {}
    return {ep.handler_file: ep.ep_type
            for ep in ctx.entry_point_catalog.entry_points}
