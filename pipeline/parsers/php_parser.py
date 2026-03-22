"""
pipeline/parsers/php_parser.py — PHP Language Adapter

Wraps the existing PHP parsing logic (nikic/php-parser CLI bridge) behind
the LanguageParser interface so stage10_parse.py can dispatch here
without knowing PHP-specific details.

All heavy logic lives here; stage10_parse.py becomes a thin dispatcher.

Environment variables (unchanged from previous stage10 behaviour):
    PHPBA_PHP_BIN       — PHP binary path (default: "php")
    PHPBA_PARSER_SCRIPT — path to parse_project.php
    PHPBA_PARSE_TIMEOUT — subprocess timeout in seconds (default: 300)
    PHPBA_WORKERS       — parallel PHP worker count (default: CPU count, max 8)
    PHPBA_NO_CACHE      — set to "1" to disable file-hash cache
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

from context import CodeMap, Framework, Language, PipelineContext
from pipeline.parsers.base import LanguageParser


# ─── Configuration ────────────────────────────────────────────────────────────

PHP_BIN: str       = os.environ.get("PHPBA_PHP_BIN", "php")
_DEFAULT_SCRIPT    = Path(__file__).parent.parent.parent / "parsers" / "parse_project.php"
PARSER_SCRIPT: str = os.environ.get("PHPBA_PARSER_SCRIPT", str(_DEFAULT_SCRIPT))
PARSE_TIMEOUT: int = int(os.environ.get("PHPBA_PARSE_TIMEOUT", "300"))
MAX_STDERR_LEN     = 2000

_cpu = min(os.cpu_count() or 1, 8)
PARSE_WORKERS: int = int(os.environ.get("PHPBA_WORKERS", str(_cpu)))
CACHE_ENABLED: bool = os.environ.get("PHPBA_NO_CACHE", "").strip() not in ("1", "true", "yes")


# ─── Adapter ──────────────────────────────────────────────────────────────────

class PHPParser(LanguageParser):
    """Language adapter for PHP projects (Laravel, Symfony, CodeIgniter, raw PHP)."""

    LANGUAGE = Language.PHP
    SUPPORTED_FRAMEWORKS = frozenset({
        Framework.LARAVEL,
        Framework.SYMFONY,
        Framework.CODEIGNITER,
        Framework.WORDPRESS,
        Framework.RAW_PHP,
        Framework.UNKNOWN,
    })

    @classmethod
    def detect(cls, project_path: str) -> bool:
        """Return True if PHP files or composer.json are present."""
        root = Path(project_path)
        if (root / "composer.json").exists():
            return True
        # Quick scan: any .php in top 2 directory levels
        for f in list(root.iterdir())[:50]:
            if f.suffix == ".php":
                return True
            if f.is_dir():
                for ff in list(f.iterdir())[:20]:
                    if ff.suffix == ".php":
                        return True
        return False

    def parse(self, project_path: str, ctx: PipelineContext) -> CodeMap:
        """
        Invoke the PHP CLI parser bridge and return a populated CodeMap.
        Writes code_map.json to ctx.output_path("code_map.json").
        """
        output_path = ctx.output_path("code_map.json")

        _assert_php_available()
        _assert_parser_script_exists()
        _assert_project_path(project_path)

        php_version = _detect_php_version(project_path)
        print(f"  [stage10/php] PHP version : {php_version}")
        print(f"  [stage10/php] Project     : {project_path}")
        print(f"  [stage10/php] Workers     : {PARSE_WORKERS}, cache: {'on' if CACHE_ENABLED else 'off'}")

        # Resolve cache file
        cache_file: str | None = None
        if CACHE_ENABLED:
            cache_dir = Path(project_path) / ".php_ba_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            gi = cache_dir / ".gitignore"
            if not gi.exists():
                gi.write_text("*\n")
            cache_file = str(cache_dir / "file_hashes.json")

        raw = _run_parser(project_path, php_version, workers=PARSE_WORKERS, cache_file=cache_file)

        try:
            payload: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            preview = raw[:500].replace("\n", " ")
            raise RuntimeError(f"[stage10/php] Invalid JSON from parser. Preview: {preview!r}") from exc

        errors: list[dict] = payload.get("errors", [])
        if errors:
            print(f"  [stage10/php] Parser reported {len(errors)} error(s):")
            for err in errors[:10]:
                print(f"           {err.get('file', '?')} — {err.get('message', '?')}")
            if len(errors) > 10:
                print(f"           … and {len(errors) - 10} more")

        code_map = _build_code_map(payload, php_version)

        # Persist raw payload
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"  [stage10/php] Saved code_map.json → {output_path}")

        return code_map


# ─── Pre-flight Checks ────────────────────────────────────────────────────────

def _assert_php_available() -> None:
    if shutil.which(PHP_BIN) is None:
        raise RuntimeError(
            f"[stage10/php] PHP binary '{PHP_BIN}' not found on PATH. "
            "Install PHP or set PHPBA_PHP_BIN."
        )


def _assert_parser_script_exists() -> None:
    script = Path(PARSER_SCRIPT)
    if not script.exists():
        raise RuntimeError(f"[stage10/php] Parser script not found: {script}.")
    vendor = script.parent / "vendor" / "autoload.php"
    if not vendor.exists():
        raise RuntimeError(
            f"[stage10/php] nikic/php-parser autoloader missing at {vendor}. "
            f"Run: cd {script.parent} && composer require nikic/php-parser"
        )


def _assert_project_path(path: str) -> None:
    if not Path(path).is_dir():
        raise RuntimeError(f"[stage10/php] Project path is not a directory: {path}")


# ─── PHP Version Detection ────────────────────────────────────────────────────

def _detect_php_version(project_path: str) -> str:
    root = Path(project_path)

    php_version_file = root / ".php-version"
    if php_version_file.exists():
        m = re.match(r"(\d+\.\d+)", php_version_file.read_text().strip())
        if m:
            return m.group(1)

    composer_file = root / "composer.json"
    if composer_file.exists():
        try:
            composer = json.loads(composer_file.read_text(encoding="utf-8"))
            constraint = (
                composer.get("require", {}).get("php", "") or
                composer.get("require-dev", {}).get("php", "")
            )
            if constraint:
                m = re.search(r"(\d+\.\d+)", constraint)
                if m:
                    return m.group(1)
        except (json.JSONDecodeError, OSError):
            pass

    try:
        out = subprocess.check_output(
            [PHP_BIN, "--version"], stderr=subprocess.DEVNULL, timeout=10, text=True
        )
        m = re.search(r"PHP (\d+\.\d+)", out)
        if m:
            return m.group(1)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return "8.1"


# ─── Subprocess Invocation ────────────────────────────────────────────────────

def _run_parser(
    project_path: str,
    php_version: str,
    *,
    workers: int = 1,
    cache_file: str | None = None,
) -> str:
    cmd = [PHP_BIN, PARSER_SCRIPT, project_path, f"--php-version={php_version}"]
    if workers > 1:
        cmd.append(f"--workers={workers}")
    if cache_file:
        cmd.append(f"--cache-file={cache_file}")

    print(f"  [stage10/php] Running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=PARSE_TIMEOUT,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"[stage10/php] Parser timed out after {PARSE_TIMEOUT}s. "
            "Increase PHPBA_PARSE_TIMEOUT or exclude large directories."
        )
    except FileNotFoundError:
        raise RuntimeError(f"[stage10/php] Could not execute PHP binary '{PHP_BIN}'.")

    if proc.returncode != 0:
        stderr_preview = proc.stderr[:MAX_STDERR_LEN].strip()
        raise RuntimeError(
            f"[stage10/php] Parser exited with code {proc.returncode}.\nSTDERR: {stderr_preview}"
        )

    if proc.stderr.strip():
        for line in proc.stderr.strip().splitlines():
            tag = "  [stage10/php]" if line.startswith(("Cache:", "Parallel:")) else "  [stage10/php] PHP:"
            print(f"{tag} {line}", file=sys.stderr)

    return proc.stdout


# ─── CodeMap Construction ─────────────────────────────────────────────────────

def _build_code_map(payload: dict[str, Any], php_version: str) -> CodeMap:
    framework_raw = payload.get("framework", "unknown")
    try:
        framework = Framework(framework_raw)
    except ValueError:
        framework = Framework.UNKNOWN

    return CodeMap(
        # Language identity
        language         = Language.PHP,
        language_version = php_version,
        # Core fields
        framework     = framework,
        classes       = payload.get("classes", []),
        routes        = payload.get("routes", []),
        models        = payload.get("models", []),
        controllers   = payload.get("controllers", []),
        services      = payload.get("services", []),
        db_schema     = payload.get("db_schema", []),
        config_files  = payload.get("config_files", []),
        total_files   = int(payload.get("total_files", 0)),
        total_lines   = int(payload.get("total_lines", 0)),
        # Language-neutral extended fields
        functions    = payload.get("functions", []),
        imports      = payload.get("includes", []),    # PHP includes → imports
        sql_queries  = payload.get("sql_queries", []),
        call_graph   = payload.get("call_graph", []),
        form_fields  = payload.get("form_fields", []),
        service_deps = payload.get("service_deps", []),
        env_vars     = payload.get("env_vars", []),
        auth_signals = payload.get("auth_signals", []),
        http_endpoints = payload.get("http_endpoints", []),
        table_columns  = payload.get("table_columns", []),
        globals        = payload.get("globals", []),
        input_params   = payload.get("superglobals", []),  # PHP superglobals → input_params
        # PHP-specific compat fields
        php_version  = php_version,
        html_pages   = payload.get("html_pages", []),
        includes     = payload.get("includes", []),
        redirects    = payload.get("redirects", []),
        superglobals = payload.get("superglobals", []),
    )


# ─── Resume loader (used by stage10 dispatcher) ───────────────────────────────

def load_code_map_from_file(output_path: str, php_version: str = "8.1") -> CodeMap:
    """Re-hydrate a CodeMap from a previously saved code_map.json."""
    try:
        with open(output_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"[stage10/php] Failed to load existing code_map.json at {output_path}: {exc}"
        ) from exc
    return _build_code_map(payload, payload.get("php_version", php_version))


# ─── Standalone helpers (used by other stages / tests) ───────────────────────

def get_classes_by_type(code_map: CodeMap, class_type: str) -> list[dict[str, Any]]:
    all_entries = code_map.classes + code_map.controllers + code_map.models + code_map.services
    return [c for c in all_entries if c.get("type") == class_type]


def get_routes_by_method(code_map: CodeMap, http_method: str) -> list[dict[str, Any]]:
    return [r for r in code_map.routes if r.get("method", "").upper() == http_method.upper()]


def summarise_code_map(code_map: CodeMap) -> dict[str, Any]:
    return {
        "language":       code_map.language.value,
        "language_version": code_map.language_version,
        "framework":      code_map.framework.value,
        "total_files":    code_map.total_files,
        "total_lines":    code_map.total_lines,
        "classes":        len(code_map.classes),
        "controllers":    len(code_map.controllers),
        "models":         len(code_map.models),
        "services":       len(code_map.services),
        "routes":         len(code_map.routes),
        "migrations":     len(code_map.db_schema),
        "functions":      len(code_map.functions),
        "imports":        len(code_map.imports),
        "sql_queries":    len(code_map.sql_queries),
        "call_graph":     len(code_map.call_graph),
        "form_fields":    len(code_map.form_fields),
        "service_deps":   len(code_map.service_deps),
        "env_vars":       len(code_map.env_vars),
        "auth_signals":   len(code_map.auth_signals),
        "http_endpoints": len(code_map.http_endpoints),
        "table_columns":  len(code_map.table_columns),
        "globals":        len(code_map.globals),
        "input_params":   len(code_map.input_params),
        # PHP compat
        "html_pages":     len(code_map.html_pages),
        "redirects":      len(code_map.redirects),
    }
