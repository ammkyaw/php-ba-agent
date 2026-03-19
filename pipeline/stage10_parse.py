"""
pipeline/stage10_parse.py — PHP Code Parsing Stage

Invokes the nikic/php-parser PHP CLI bridge (parse_project.php), captures its
JSON output, and hydrates ctx.code_map with structured CodeMap data.

Responsibilities:
    - Locate and validate the PHP binary and parse_project.php script
    - Detect the project's PHP version from composer.json / .php-version
    - Run the parser subprocess with a configurable timeout
    - Convert the raw JSON payload into a typed CodeMap
    - Write a code_map.json snapshot to the output directory for later stages

Dependencies:
    - PHP >= 7.4 must be on PATH (or configured via PHPBA_PHP_BIN env var)
    - parse_project.php must be adjacent to this file (or set PHPBA_PARSER_SCRIPT)
    - nikic/php-parser must be installed via `composer require nikic/php-parser`
      in the same directory as parse_project.php

Resume behaviour:
    If ctx.stage("stage10_parse").status == COMPLETED and code_map.json already
    exists, the stage is skipped immediately.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from context import CodeMap, Framework, PipelineContext


# ─── Configuration ────────────────────────────────────────────────────────────

# Override with PHPBA_PHP_BIN=/usr/bin/php8.2 if needed
PHP_BIN: str = os.environ.get("PHPBA_PHP_BIN", "php")

# Override with PHPBA_PARSER_SCRIPT=/absolute/path/to/parse_project.php
_DEFAULT_SCRIPT = Path(__file__).parent.parent / "parsers" / "parse_project.php"
PARSER_SCRIPT: str = os.environ.get("PHPBA_PARSER_SCRIPT", str(_DEFAULT_SCRIPT))

# Subprocess timeout in seconds (large projects can take a while)
PARSE_TIMEOUT: int = int(os.environ.get("PHPBA_PARSE_TIMEOUT", "300"))

# Maximum stderr output preserved in the error message
MAX_STDERR_LEN: int = 2000

# Number of parallel PHP worker processes.
# Defaults to CPU count (capped at 8 to avoid overwhelming CI machines).
# Set PHPBA_WORKERS=1 to disable parallelism, PHPBA_WORKERS=0 for auto.
_cpu = min(os.cpu_count() or 1, 8)
PARSE_WORKERS: int = int(os.environ.get("PHPBA_WORKERS", str(_cpu)))

# File hash cache location.
# Stored in <project_root>/.php_ba_cache/file_hashes.json so that it
# persists across multiple runs against the same project.
# Set PHPBA_CACHE_FILE=/custom/path.json to override, or PHPBA_NO_CACHE=1 to disable.
CACHE_ENABLED: bool = os.environ.get("PHPBA_NO_CACHE", "").strip() not in ("1", "true", "yes")


# ─── Public Entry Point ───────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 1 entry point called by run_pipeline.py.

    Populates ctx.code_map and writes <output_dir>/code_map.json.
    Raises RuntimeError on unrecoverable failures.

    Args:
        ctx: Shared pipeline context; mutated in-place.
    """
    output_path = ctx.output_path("code_map.json")

    # ── Resume check ────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage10_parse") and Path(output_path).exists():
        print("  [stage1] Resuming — loading existing code_map.json")
        ctx.code_map = _load_code_map(output_path)
        return

    # ── Pre-flight ───────────────────────────────────────────────────────────
    _assert_php_available()
    _assert_parser_script_exists()
    _assert_project_path(ctx.php_project_path)

    # ── PHP version detection ────────────────────────────────────────────────
    php_version = _detect_project_php_version(ctx.php_project_path)
    print(f"  [stage1] Detected PHP version: {php_version}")
    print(f"  [stage1] Parsing project: {ctx.php_project_path}")
    print(f"  [stage1] Workers: {PARSE_WORKERS}, cache: {'on' if CACHE_ENABLED else 'off'}")

    # ── Resolve cache file path ───────────────────────────────────────────────
    cache_file: str | None = None
    if CACHE_ENABLED:
        cache_dir = Path(ctx.php_project_path) / ".php_ba_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Add .gitignore to the cache dir if it doesn't exist
        gi = cache_dir / ".gitignore"
        if not gi.exists():
            gi.write_text("*\n")
        cache_file = str(cache_dir / "file_hashes.json")

    # ── Invoke parser ────────────────────────────────────────────────────────
    raw = _run_parser(ctx.php_project_path, php_version,
                      workers=PARSE_WORKERS, cache_file=cache_file)

    # ── Parse JSON payload ───────────────────────────────────────────────────
    try:
        payload: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:500].replace("\n", " ")
        raise RuntimeError(
            f"[stage1] PHP parser produced invalid JSON. "
            f"Preview: {preview!r}"
        ) from exc

    # ── Log any parse errors encountered by the PHP side ────────────────────
    errors: list[dict] = payload.get("errors", [])
    if errors:
        print(f"  [stage1] Parser reported {len(errors)} file error(s):")
        for err in errors[:10]:
            print(f"           {err.get('file', '?')} — {err.get('message', '?')}")
        if len(errors) > 10:
            print(f"           ... and {len(errors) - 10} more (see code_map.json)")

    # ── Build CodeMap ────────────────────────────────────────────────────────
    code_map = _build_code_map(payload)
    ctx.code_map = code_map

    # ── Persist snapshot ─────────────────────────────────────────────────────
    _write_code_map(payload, output_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(
        f"  [stage1] Done — framework={code_map.framework.value}, "
        f"php={code_map.php_version}, "
        f"files={code_map.total_files}, "
        f"lines={code_map.total_lines}, "
        f"classes={len(code_map.classes)}, "
        f"controllers={len(code_map.controllers)}, "
        f"models={len(code_map.models)}, "
        f"routes={len(code_map.routes)}, "
        f"functions={len(code_map.functions)}, "
        f"sql_queries={len(code_map.sql_queries)}, "
        f"redirects={len(code_map.redirects)}, "
        f"call_graph={len(code_map.call_graph)} edges, "
        f"form_fields={len(code_map.form_fields)}, "
        f"service_deps={len(code_map.service_deps)}, "
        f"env_vars={len(code_map.env_vars)}, "
        f"auth_signals={len(code_map.auth_signals)}, "
        f"http_endpoints={len(code_map.http_endpoints)}, "
        f"table_columns={len(code_map.table_columns)}, "
        f"html_pages={len(code_map.html_pages)}, "
        f"superglobals={len(code_map.superglobals)}, "
        f"migrations={len(code_map.db_schema)}"
    )

    ctx.stage("stage10_parse").mark_completed(output_path)
    ctx.save()


# ─── Pre-flight Checks ────────────────────────────────────────────────────────

def _assert_php_available() -> None:
    """Verify that a PHP binary is reachable on PATH (or via PHPBA_PHP_BIN)."""
    if shutil.which(PHP_BIN) is None:
        raise RuntimeError(
            f"[stage1] PHP binary '{PHP_BIN}' not found on PATH. "
            "Install PHP or set the PHPBA_PHP_BIN environment variable."
        )


def _assert_parser_script_exists() -> None:
    """Ensure parse_project.php is present and the vendor autoloader exists."""
    script = Path(PARSER_SCRIPT)
    if not script.exists():
        raise RuntimeError(
            f"[stage1] Parser script not found: {script}. "
            "Expected parse_project.php alongside the pipeline directory."
        )

    vendor = script.parent / "vendor" / "autoload.php"
    if not vendor.exists():
        raise RuntimeError(
            f"[stage1] nikic/php-parser vendor autoloader not found at {vendor}. "
            f"Run: cd {script.parent} && composer require nikic/php-parser"
        )


def _assert_project_path(path: str) -> None:
    """Make sure the PHP project directory actually exists."""
    if not Path(path).is_dir():
        raise RuntimeError(
            f"[stage1] PHP project path is not a directory: {path}"
        )


# ─── PHP Version Detection ────────────────────────────────────────────────────

def _detect_project_php_version(project_path: str) -> str:
    """
    Attempt to determine the PHP version the project targets using several
    heuristics, falling back to '8.1' if none succeed.

    Detection order:
      1. .php-version file (phpenv / asdf convention)
      2. composer.json require.php constraint
      3. Running `php --version` against the system binary
    """
    root = Path(project_path)

    # 1. .php-version
    php_version_file = root / ".php-version"
    if php_version_file.exists():
        raw = php_version_file.read_text().strip()
        match = re.match(r"(\d+\.\d+)", raw)
        if match:
            return match.group(1)

    # 2. composer.json
    composer_file = root / "composer.json"
    if composer_file.exists():
        try:
            composer: dict = json.loads(composer_file.read_text(encoding="utf-8"))
            php_constraint: str = (
                composer.get("require", {}).get("php", "") or
                composer.get("require-dev", {}).get("php", "")
            )
            if php_constraint:
                # Extract first version-like token: ">=8.1" → "8.1"
                match = re.search(r"(\d+\.\d+)", php_constraint)
                if match:
                    return match.group(1)
        except (json.JSONDecodeError, OSError):
            pass

    # 3. System PHP binary
    try:
        out = subprocess.check_output(
            [PHP_BIN, "--version"], stderr=subprocess.DEVNULL, timeout=10, text=True
        )
        match = re.search(r"PHP (\d+\.\d+)", out)
        if match:
            return match.group(1)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return "8.1"  # safe default


# ─── Subprocess Invocation ────────────────────────────────────────────────────

def _run_parser(
    project_path: str,
    php_version: str,
    *,
    workers: int = 1,
    cache_file: str | None = None,
) -> str:
    """
    Execute parse_project.php as a subprocess and return its stdout (JSON).

    When workers > 1 the PHP script handles parallel dispatch internally
    via proc_open() worker sub-processes, so Python only needs to make a
    single call. The workers flag is forwarded as --workers=N.

    When cache_file is set the PHP script will:
      - Load the SHA1 hash cache from that path
      - Skip files whose hash hasn't changed
      - Save updated hashes after parsing

    Args:
        project_path: Absolute path to the PHP project root.
        php_version:  Target PHP version string (e.g. "8.1").
        workers:      Number of parallel PHP worker processes (default 1).
        cache_file:   Path to the JSON file hash cache (or None to disable).

    Returns:
        Raw stdout string (expected to be JSON).

    Raises:
        RuntimeError: If the process exits non-zero or times out.
    """
    cmd = [
        PHP_BIN,
        PARSER_SCRIPT,
        project_path,
        f"--php-version={php_version}",
    ]

    if workers > 1:
        cmd.append(f"--workers={workers}")

    if cache_file:
        cmd.append(f"--cache-file={cache_file}")

    print(f"  [stage1] Running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PARSE_TIMEOUT,
            encoding="utf-8",
            errors="replace",   # don't crash on bad encoding in source files
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"[stage1] PHP parser timed out after {PARSE_TIMEOUT}s. "
            "Try increasing PHPBA_PARSE_TIMEOUT or exclude large directories."
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"[stage1] Could not execute PHP binary '{PHP_BIN}'."
        )

    if proc.returncode != 0:
        stderr_preview = proc.stderr[:MAX_STDERR_LEN].strip()
        raise RuntimeError(
            f"[stage1] PHP parser exited with code {proc.returncode}.\n"
            f"STDERR: {stderr_preview}"
        )

    # Print informational stderr (cache/parallel stats, not errors)
    if proc.stderr.strip():
        for line in proc.stderr.strip().splitlines():
            # PHP errors go to stderr but so do our info messages; distinguish them
            tag = "  [stage1]" if line.startswith(("Cache:", "Parallel:")) else "  [stage1] PHP:"
            print(f"{tag} {line}", file=sys.stderr)

    return proc.stdout


# ─── CodeMap Construction ─────────────────────────────────────────────────────

def _build_code_map(payload: dict[str, Any]) -> CodeMap:
    """
    Convert the raw PHP-parser JSON payload into a typed CodeMap dataclass.

    Args:
        payload: Parsed JSON dict from parse_project.php stdout.

    Returns:
        Populated CodeMap instance.
    """
    framework_raw = payload.get("framework", "unknown")
    try:
        framework = Framework(framework_raw)
    except ValueError:
        framework = Framework.UNKNOWN

    return CodeMap(
        framework     = framework,
        php_version   = payload.get("php_version"),
        classes       = payload.get("classes", []),
        routes        = payload.get("routes", []),
        models        = payload.get("models", []),
        controllers   = payload.get("controllers", []),
        services      = payload.get("services", []),
        db_schema     = payload.get("db_schema", []),
        config_files  = payload.get("config_files", []),
        total_files   = int(payload.get("total_files", 0)),
        total_lines   = int(payload.get("total_lines", 0)),
        # Procedural extractions
        functions     = payload.get("functions", []),
        includes      = payload.get("includes", []),
        sql_queries   = payload.get("sql_queries",   []),
        redirects     = payload.get("redirects",     []),
        call_graph      = payload.get("call_graph",      []),
        form_fields     = payload.get("form_fields",     []),
        service_deps    = payload.get("service_deps",    []),
        env_vars        = payload.get("env_vars",        []),
        auth_signals    = payload.get("auth_signals",    []),
        http_endpoints  = payload.get("http_endpoints",  []),
        table_columns   = payload.get("table_columns",   []),
        globals       = payload.get("globals", []),
        html_pages    = payload.get("html_pages", []),
        superglobals  = payload.get("superglobals", []),
    )


# ─── Persistence Helpers ──────────────────────────────────────────────────────

def _write_code_map(payload: dict[str, Any], output_path: str) -> None:
    """
    Persist the full raw payload (including parser errors) to code_map.json.
    Later stages (graph, embed) can re-read this file directly if needed.

    Args:
        payload:     Raw parser JSON payload.
        output_path: Destination file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"  [stage1] Saved code_map.json → {output_path}")


def _load_code_map(output_path: str) -> CodeMap:
    """
    Re-hydrate a CodeMap from a previously saved code_map.json.
    Used when resuming a run.

    Args:
        output_path: Path to code_map.json.

    Returns:
        Populated CodeMap instance.

    Raises:
        RuntimeError: If the file cannot be read or parsed.
    """
    try:
        with open(output_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"[stage1] Failed to load existing code_map.json at {output_path}: {exc}"
        ) from exc

    return _build_code_map(payload)


# ─── Standalone Helper (used by tests / other stages) ────────────────────────

def get_classes_by_type(code_map: CodeMap, class_type: str) -> list[dict[str, Any]]:
    """
    Filter all class entries across every CodeMap bucket by their 'type' field.

    Args:
        code_map:   Populated CodeMap.
        class_type: One of 'controller', 'model', 'service', 'repository',
                    'middleware', 'event', 'listener', 'job', 'policy', etc.

    Returns:
        List of matching class dicts.
    """
    all_entries = (
        code_map.classes
        + code_map.controllers
        + code_map.models
        + code_map.services
    )
    return [c for c in all_entries if c.get("type") == class_type]


def get_routes_by_method(code_map: CodeMap, http_method: str) -> list[dict[str, Any]]:
    """
    Return all route entries matching a given HTTP method (case-insensitive).

    Args:
        code_map:    Populated CodeMap.
        http_method: HTTP verb, e.g. 'GET', 'POST'.

    Returns:
        Filtered list of route dicts.
    """
    method_upper = http_method.upper()
    return [r for r in code_map.routes if r.get("method", "").upper() == method_upper]


def summarise_code_map(code_map: CodeMap) -> dict[str, Any]:
    """
    Return a compact summary dict suitable for logging or preflight checks.

    Args:
        code_map: Populated CodeMap.

    Returns:
        Dict with counts and key metadata.
    """
    return {
        "framework":    code_map.framework.value,
        "php_version":  code_map.php_version,
        "total_files":  code_map.total_files,
        "total_lines":  code_map.total_lines,
        "classes":      len(code_map.classes),
        "controllers":  len(code_map.controllers),
        "models":       len(code_map.models),
        "services":     len(code_map.services),
        "routes":       len(code_map.routes),
        "migrations":   len(code_map.db_schema),
        "config_files": len(code_map.config_files),
        # Procedural
        "functions":    len(code_map.functions),
        "includes":     len(code_map.includes),
        "sql_queries":  len(code_map.sql_queries),
        "redirects":    len(code_map.redirects),
        "call_graph":     len(code_map.call_graph),
        "form_fields":    len(code_map.form_fields),
        "service_deps":   len(code_map.service_deps),
        "env_vars":       len(code_map.env_vars),
        "auth_signals":   len(code_map.auth_signals),
        "http_endpoints": len(code_map.http_endpoints),
        "table_columns":  len(code_map.table_columns),
        "globals":      len(code_map.globals),
        "html_pages":   len(code_map.html_pages),
        "superglobals": len(code_map.superglobals),
    }
