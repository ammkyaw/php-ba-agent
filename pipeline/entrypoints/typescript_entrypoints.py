"""
pipeline/entrypoints/typescript_entrypoints.py — TypeScript / JavaScript Entry-Point Detector

Detects non-HTTP entry points in TypeScript / JavaScript projects:

  scheduled    — cron jobs (node-cron, cron, @nestjs/schedule, BullMQ schedulers)
  cli          — CLI scripts (commander, yargs, process.argv-guarded scripts,
                 bin/* entries in package.json)
  webhook      — files whose name / content suggests an incoming HTTP callback
  queue_worker — BullMQ / Bull / kue / agenda worker files

Detection is purely static (file scan + regex content patterns).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from context import EntryPoint, Framework

# ── Skip dirs ──────────────────────────────────────────────────────────────────
_SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", ".next", ".nuxt",
    "coverage", "tests", "test", "spec", "__tests__", "fixtures",
}

# ── Confidence levels ──────────────────────────────────────────────────────────
_CONF_CONFIG  = 1.0
_CONF_PATTERN = 0.8
_CONF_HEUR    = 0.6


# ─── Public interface ─────────────────────────────────────────────────────────

def detect(root: Path, framework: Framework) -> list[EntryPoint]:
    """Return all non-HTTP entry points found in a TS/JS project."""
    results: list[EntryPoint] = []
    results += _detect_package_bin(root)
    results += _detect_cron_jobs(root)
    results += _detect_cli_scripts(root)
    results += _detect_webhook_handlers(root)
    results += _detect_queue_workers(root)
    if framework == Framework.NESTJS:
        results += _detect_nestjs_tasks(root)
    return _dedup(results)


# ─── Detectors ────────────────────────────────────────────────────────────────

def _detect_package_bin(root: Path) -> list[EntryPoint]:
    """Detect CLI scripts declared in package.json → bin field."""
    eps: list[EntryPoint] = []
    pkg = root / "package.json"
    if not pkg.exists():
        return eps
    try:
        data = json.loads(pkg.read_text(encoding="utf-8"))
    except Exception:
        return eps
    bin_field = data.get("bin", {})
    if isinstance(bin_field, str):
        bin_field = {data.get("name", "cli"): bin_field}
    for cmd, script_path in bin_field.items():
        eps.append(EntryPoint(
            ep_type      = "cli",
            handler_file = script_path.lstrip("./"),
            name         = _humanize(cmd),
            trigger      = f"npx {cmd}  (or: node {script_path})",
            confidence   = _CONF_CONFIG,
        ))
    return eps


_CRON_PATTERNS = [
    re.compile(r"cron\.schedule\s*\(", re.I),
    re.compile(r"new\s+CronJob\s*\(", re.I),
    re.compile(r"@Cron\s*\(", re.I),          # NestJS @Cron
    re.compile(r"schedule\.scheduleJob\s*\(", re.I),
    re.compile(r"agenda\.define\s*\(", re.I),
    re.compile(r"new\s+Agenda\s*\(", re.I),
]
_CRON_EXPR = re.compile(
    r"""['"`]((?:\*|[\d,\-/]+)\s+(?:\*|[\d,\-/]+)\s+(?:\*|[\d,\-/]+)\s+(?:\*|[\d,\-/]+)\s+(?:\*|[\d,\-/]+))['"`]"""
)


def _detect_cron_jobs(root: Path) -> list[EntryPoint]:
    eps: list[EntryPoint] = []
    for src_file in _walk(root, {".ts", ".js", ".mjs"}):
        src = _safe_read(src_file)
        if not src:
            continue
        if not any(p.search(src) for p in _CRON_PATTERNS):
            continue
        # Try to extract cron expression
        cron_expr = ""
        m = _CRON_EXPR.search(src)
        if m:
            cron_expr = m.group(1)
        rel = _rel(src_file, root)
        name = _humanize(src_file.stem)
        eps.append(EntryPoint(
            ep_type      = "scheduled",
            handler_file = rel,
            name         = f"Scheduled: {name}",
            schedule     = cron_expr or "see cron definition",
            trigger      = f"Cron job in {src_file.name}",
            confidence   = _CONF_PATTERN,
        ))
    return eps


_CLI_PATTERNS = [
    re.compile(r"commander|\.command\s*\(", re.I),
    re.compile(r"yargs|\.argv\b"),
    re.compile(r"process\.argv\.slice"),
    re.compile(r"require\.main\s*===?\s*module"),
    re.compile(r"import\.meta\.url\s*===?\s*.*fileURLToPath"),
]
_CLI_DIRS = {"bin", "cli", "scripts", "commands", "cmd", "tools", "console"}


def _detect_cli_scripts(root: Path) -> list[EntryPoint]:
    eps: list[EntryPoint] = []
    for src_file in _walk(root, {".ts", ".js", ".mjs"}):
        rel_parts = [p.lower() for p in src_file.relative_to(root).parts[:-1]]
        in_cli_dir = any(p in _CLI_DIRS for p in rel_parts)
        src = _safe_read(src_file)
        if not src:
            continue
        has_cli_pat = any(p.search(src) for p in _CLI_PATTERNS)
        if not (in_cli_dir or has_cli_pat):
            continue
        conf = _CONF_PATTERN if has_cli_pat else _CONF_HEUR
        rel = _rel(src_file, root)
        eps.append(EntryPoint(
            ep_type      = "cli",
            handler_file = rel,
            name         = _humanize(src_file.stem),
            trigger      = f"node {rel}  (or: ts-node {rel})",
            confidence   = conf,
        ))
    return eps


_WEBHOOK_NAME = re.compile(r"(webhook|callback|notify|hook|ipn|receive)", re.I)
_WEBHOOK_CONTENT = [
    re.compile(r"x-hub-signature|stripe.*signature|paypal.*ipn", re.I),
    re.compile(r"hmac|crypto\.createHmac", re.I),
    re.compile(r"req\.headers\[.*signature", re.I),
]


def _detect_webhook_handlers(root: Path) -> list[EntryPoint]:
    eps: list[EntryPoint] = []
    for src_file in _walk(root, {".ts", ".js"}):
        if not _WEBHOOK_NAME.search(src_file.stem):
            continue
        src = _safe_read(src_file)
        if not src:
            continue
        has_content = any(p.search(src) for p in _WEBHOOK_CONTENT)
        rel = _rel(src_file, root)
        conf = _CONF_CONFIG if has_content else _CONF_HEUR
        eps.append(EntryPoint(
            ep_type      = "webhook",
            handler_file = rel,
            name         = f"{_humanize(src_file.stem)} Webhook",
            trigger      = f"POST /{src_file.stem} (external system)",
            confidence   = conf,
        ))
    return eps


_QUEUE_PATTERNS = [
    re.compile(r"new\s+Worker\s*\(", re.I),           # BullMQ
    re.compile(r"queue\.process\s*\(", re.I),          # Bull
    re.compile(r"kue\.createQueue|kue\.Job", re.I),
    re.compile(r"agenda\.define|agenda\.every", re.I),
    re.compile(r"amqplib.*createChannel|amqplib.*consume", re.I),
]
_QUEUE_DIRS = {"workers", "queues", "consumers", "jobs", "processors"}


def _detect_queue_workers(root: Path) -> list[EntryPoint]:
    eps: list[EntryPoint] = []
    for src_file in _walk(root, {".ts", ".js", ".mjs"}):
        rel_parts = [p.lower() for p in src_file.relative_to(root).parts[:-1]]
        in_queue_dir = any(p in _QUEUE_DIRS for p in rel_parts)
        src = _safe_read(src_file)
        if not src:
            continue
        has_queue_pat = any(p.search(src) for p in _QUEUE_PATTERNS)
        if not (in_queue_dir or has_queue_pat):
            continue
        rel = _rel(src_file, root)
        conf = _CONF_PATTERN if has_queue_pat else _CONF_HEUR
        eps.append(EntryPoint(
            ep_type      = "queue_worker",
            handler_file = rel,
            name         = f"{_humanize(src_file.stem)} Worker",
            trigger      = "Queue consumer (long-running process)",
            confidence   = conf,
        ))
    return eps


def _detect_nestjs_tasks(root: Path) -> list[EntryPoint]:
    """Detect NestJS @Cron() / @Interval() scheduled tasks."""
    eps: list[EntryPoint] = []
    for src_file in _walk(root, {".ts"}):
        src = _safe_read(src_file)
        if not src or ("@Cron" not in src and "@Interval" not in src):
            continue
        for m in re.finditer(r"@Cron\s*\(\s*['\"`]([^'\"` ]+)", src):
            rel = _rel(src_file, root)
            # Find method name after decorator
            remainder = src[m.end():m.end()+200]
            fn_m = re.search(r"\w+\s*\(", remainder)
            name = fn_m.group(0).rstrip("(").strip() if fn_m else src_file.stem
            eps.append(EntryPoint(
                ep_type      = "scheduled",
                handler_file = rel,
                name         = f"Scheduled: {_humanize(name)}",
                schedule     = m.group(1),
                trigger      = f"NestJS @Cron({m.group(1)})",
                confidence   = _CONF_CONFIG,
            ))
    return eps


# ─── Utilities ────────────────────────────────────────────────────────────────

def _walk(root: Path, extensions: set[str]):
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if any(p in _SKIP_DIRS for p in f.parts):
            continue
        if f.suffix in extensions:
            yield f


def _dedup(eps: list[EntryPoint]) -> list[EntryPoint]:
    seen: dict[str, EntryPoint] = {}
    for ep in eps:
        key = ep.handler_file
        if key not in seen or ep.confidence > seen[key].confidence:
            seen[key] = ep
    return list(seen.values())


def _safe_read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _humanize(name: str) -> str:
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"[^a-zA-Z0-9 ]+", " ", name)
    return " ".join(w.capitalize() for w in name.split() if w)
