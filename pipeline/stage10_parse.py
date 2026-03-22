"""
pipeline/stage10_parse.py — Code Parsing Stage (Multi-Language Dispatcher)

Dispatches to the correct language adapter based on the language detected by
stage05_detect.  Falls back to PHP if stage05 was not run (backward compat).

Supported adapters
------------------
  PHP        → pipeline/parsers/php_parser.PHPParser
  TypeScript → pipeline/parsers/typescript_parser.TypeScriptParser
  JavaScript → pipeline/parsers/typescript_parser.TypeScriptParser  (same adapter)
  Java       → pipeline/parsers/java_parser.JavaParser
  Kotlin     → pipeline/parsers/java_parser.JavaParser               (same adapter)

Resume behaviour
----------------
If ctx.stage("stage10_parse").status == COMPLETED and code_map.json already
exists, the stage is skipped and the CodeMap is re-hydrated from disk.

PHP-only backward compat
------------------------
All public helpers that existed in the old stage10_parse.py are re-exported
from pipeline/parsers/php_parser.py so that any stage that did
``from pipeline.stage10_parse import get_classes_by_type`` still works.
"""

from __future__ import annotations

import json
from pathlib import Path

from context import CodeMap, Framework, Language, PipelineContext

# ── Re-export PHP helpers for backward compat ─────────────────────────────────
from pipeline.parsers.php_parser import (   # noqa: F401
    get_classes_by_type,
    get_routes_by_method,
    summarise_code_map,
    load_code_map_from_file as _php_load,
)


# ─── Public Entry Point ───────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 1 entry point called by run_pipeline.py.

    Selects the correct language adapter, runs the parse, populates
    ctx.code_map, and writes <output_dir>/code_map.json.

    Raises RuntimeError on unrecoverable failures.
    """
    output_path = ctx.output_path("code_map.json")

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage10_parse") and Path(output_path).exists():
        print("  [stage10] Resuming — loading existing code_map.json")
        ctx.code_map = _load_code_map(output_path)
        return

    # ── Pick language adapter ─────────────────────────────────────────────────
    adapter = _pick_adapter(ctx)
    print(f"  [stage10] Using adapter: {adapter.__class__.__name__}")

    # ── Run parse ─────────────────────────────────────────────────────────────
    code_map = adapter.parse(ctx.project_path, ctx)
    ctx.code_map = code_map

    # ── Summary ───────────────────────────────────────────────────────────────
    print(
        f"  [stage10] Done — language={code_map.language.value}, "
        f"framework={code_map.framework.value}, "
        f"files={code_map.total_files}, lines={code_map.total_lines}, "
        f"classes={len(code_map.classes)}, controllers={len(code_map.controllers)}, "
        f"models={len(code_map.models)}, routes={len(code_map.routes)}, "
        f"functions={len(code_map.functions)}, sql_queries={len(code_map.sql_queries)}, "
        f"table_columns={len(code_map.table_columns)}"
    )

    ctx.stage("stage10_parse").mark_completed(output_path)
    ctx.save()


# ─── Adapter Selection ────────────────────────────────────────────────────────

def _pick_adapter(ctx: PipelineContext):
    """
    Return the appropriate LanguageParser instance for this project.

    Resolution order:
      1. ctx.detected_language (set by stage05_detect)
      2. Probe-based auto-detection (if stage05 not run)
      3. Fall back to PHPParser (historical default)
    """
    from pipeline.stage05_detect import get_detected   # late import to avoid circularity
    lang, _ = get_detected(ctx)

    # Lazy import adapters only when needed
    if lang in (Language.TYPESCRIPT, Language.JAVASCRIPT):
        from pipeline.parsers.typescript_parser import TypeScriptParser
        return TypeScriptParser()

    if lang in (Language.JAVA, Language.KOTLIN):
        from pipeline.parsers.java_parser import JavaParser
        return JavaParser()

    # PHP (or UNKNOWN — fall back to PHP for backward compat)
    from pipeline.parsers.php_parser import PHPParser
    return PHPParser()


# ─── Resume loader ────────────────────────────────────────────────────────────

def _load_code_map(output_path: str) -> CodeMap:
    """Re-hydrate a CodeMap from a previously saved code_map.json."""
    try:
        with open(output_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"[stage10] Failed to load code_map.json: {exc}") from exc

    # Dispatch to the correct builder based on the stored _parser hint
    parser_hint = payload.get("_parser", "php")

    if parser_hint == "typescript":
        from pipeline.parsers.typescript_parser import TypeScriptParser, _code_map_to_payload
        # Reconstruct via context deserialization path
        return _reconstruct_from_payload(payload, Language.TYPESCRIPT)

    if parser_hint == "java":
        return _reconstruct_from_payload(payload, Language.JAVA)

    # Default: PHP
    from pipeline.parsers.php_parser import _build_code_map
    return _build_code_map(payload, payload.get("php_version", "8.1"))


def _reconstruct_from_payload(payload: dict, lang: Language) -> CodeMap:
    """Generic CodeMap reconstruction from a serialised payload dict."""
    try:
        fw = Framework(payload.get("framework", Framework.UNKNOWN.value))
    except ValueError:
        fw = Framework.UNKNOWN
    return CodeMap(
        language         = lang,
        language_version = payload.get("language_version") or payload.get("php_version"),
        framework        = fw,
        classes          = payload.get("classes", []),
        routes           = payload.get("routes", []),
        models           = payload.get("models", []),
        controllers      = payload.get("controllers", []),
        services         = payload.get("services", []),
        db_schema        = payload.get("db_schema", []),
        config_files     = payload.get("config_files", []),
        total_files      = payload.get("total_files", 0),
        total_lines      = payload.get("total_lines", 0),
        functions        = payload.get("functions", []),
        imports          = payload.get("imports", payload.get("includes", [])),
        sql_queries      = payload.get("sql_queries", []),
        call_graph       = payload.get("call_graph", []),
        form_fields      = payload.get("form_fields", []),
        service_deps     = payload.get("service_deps", []),
        env_vars         = payload.get("env_vars", []),
        auth_signals     = payload.get("auth_signals", []),
        http_endpoints   = payload.get("http_endpoints", []),
        table_columns    = payload.get("table_columns", []),
        globals          = payload.get("globals", []),
        execution_paths  = payload.get("execution_paths", []),
        components       = payload.get("components", []),
        input_params     = payload.get("input_params", payload.get("superglobals", [])),
        # PHP compat
        php_version      = payload.get("php_version"),
        html_pages       = payload.get("html_pages", []),
        includes         = payload.get("includes", []),
        redirects        = payload.get("redirects", []),
        superglobals     = payload.get("superglobals", []),
    )
