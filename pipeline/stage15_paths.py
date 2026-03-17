"""
pipeline/stage15_paths.py — Execution Path Simulator (Conditional Branch Extractor)

Performs a focused static analysis pass over each PHP entry-point file to extract:

    1. Entry conditions   — what session/POST/GET vars must exist for the page
                            to proceed vs. redirect away
    2. Branch map         — if/elseif/else conditions with their outcomes
                            (DB operation, redirect, session write, output)
    3. Data flow          — which POST/GET fields feed which SQL parameters
    4. Auth guard         — how session auth is checked (key, comparison, redirect)
    5. Happy path         — the "main success scenario" as an ordered step list

For LARAVEL projects this stage additionally extracts controller method flows:
    - Route→Controller→Method chains from routes/api.php and routes/web.php
    - Eloquent model operations (::create, ->save, ->update, ->delete, ->find)
    - Request validation rules ($request->validate(), Form Request classes)
    - Middleware auth guards (auth, can, role middleware)
    - Response patterns (return view(), return redirect(), return response()->json())

Architecture
------------
• _FileAnalyser        — regex-based analyser for raw PHP / legacy patterns
• _LaravelRouteAnalyser— parses routes/api.php and routes/web.php
• _LaravelMethodAnalyser— analyses controller methods via source-level regex
• _extract_laravel_controller_flows — bulk pass using already-structured
                                      code_map.controllers data

Framework detection drives which analysers run; all results are merged.

CHANGES vs previous version
----------------------------
• All exceptions inside the per-file loop are now LOGGED with full tracebacks —
  silent zero-output is no longer possible.
• Added _LaravelRouteAnalyser and _LaravelMethodAnalyser for framework=laravel;
  extracts controller method flows invisible to the raw-PHP regex pass.
• analyse_file() returns a result for Laravel controllers even when they have no
  $_POST / $_SESSION / header() patterns.
• _extract_laravel_controller_flows provides a bulk pass from code_map.controllers.
• Fixed duplicate `return steps` at end of _build_happy_path (was unreachable code).
• _collect_php_files: skip list expanded; tests/stubs/fixtures also skipped.
• Added execution_paths_errors.json written per-run for full error inspection.

Resume behaviour
----------------
If stage15_paths is COMPLETED and execution_paths.json exists, stage is skipped.
"""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any

from context import CodeMap, Framework, PipelineContext

OUTPUT_FILE = "execution_paths.json"
ERROR_FILE  = "execution_paths_errors.json"

# Directories to skip when collecting PHP files
_SKIP_DIRS = {
    "vendor", "node_modules", ".git", "cache", "logs", "storage",
    "tests", "test", "spec", "stubs", "fixtures",
}


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 1.5 entry point. Analyses each PHP file for conditional branches
    and execution paths, enriches ctx.code_map with the results.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If code_map is missing (Stage 1 not run).
    """
    output_path = ctx.output_path(OUTPUT_FILE)
    error_path  = ctx.output_path(ERROR_FILE)

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage15_paths") and Path(output_path).exists():
        paths = json.loads(Path(output_path).read_text(encoding="utf-8"))
        _attach_to_code_map(ctx.code_map, paths)
        print(f"  [stage15] Resuming — {len(paths)} file path(s) loaded.")
        return

    if ctx.code_map is None:
        raise RuntimeError("[stage15] ctx.code_map is None — run Stage 1 first.")

    framework = ctx.code_map.framework
    print(f"  [stage15] Analysing execution paths in "
          f"{ctx.code_map.total_files} file(s) ... (framework={framework.value})")

    # ── Collect PHP entry-point files ─────────────────────────────────────────
    php_files = _collect_php_files(ctx.php_project_path)
    print(f"  [stage15] Found {len(php_files)} PHP file(s) to analyse.")

    # ── Augment with Stage 1.3 entry-point catalog ────────────────────────────
    # Include handler files from the entry-point catalog that weren't picked up
    # by _collect_php_files (e.g. CLI-only queue workers, cron scripts deep in
    # subdirectories that the HTTP-focused collector skips).
    _ep_type_by_abs: dict[str, str] = {}   # abs_path → ep_type
    if ctx.entry_point_catalog and ctx.entry_point_catalog.entry_points:
        root = Path(ctx.php_project_path)
        php_files_set = set(php_files)
        added = 0
        for ep in ctx.entry_point_catalog.entry_points:
            abs_path = str((root / ep.handler_file).resolve())
            _ep_type_by_abs[abs_path] = ep.ep_type
            if abs_path not in php_files_set and Path(abs_path).exists():
                php_files.append(abs_path)
                php_files_set.add(abs_path)
                added += 1
        if added:
            print(f"  [stage15] Entry-point catalog: added {added} "
                  f"additional handler file(s) for analysis.")

    # ── Analyse each file ─────────────────────────────────────────────────────
    all_paths: list[dict[str, Any]] = []
    errors:    list[dict[str, str]] = []

    for php_file in sorted(php_files):
        try:
            result = analyse_file(php_file, ctx.php_project_path, framework)
            if result:
                # Tag ep_type from Stage 1.3 catalog when available
                ep_type = _ep_type_by_abs.get(str(Path(php_file).resolve()))
                if ep_type and ep_type != "http":
                    result["ep_type"] = ep_type
                all_paths.append(result)
        except Exception as exc:                              # noqa: BLE001
            # Capture full traceback — the old code swallowed these silently
            tb = traceback.format_exc()
            errors.append({
                "file":      str(Path(php_file).relative_to(ctx.php_project_path)),
                "error":     str(exc),
                "traceback": tb,
            })

    # ── Laravel controller enrichment pass ───────────────────────────────────
    # For Laravel projects, run a second pass using already-structured
    # code_map.controllers rather than re-parsing raw PHP — fills gaps for
    # controllers whose files had no $_POST / header() / $_SESSION signals.
    if framework == Framework.LARAVEL and ctx.code_map:
        laravel_paths = _extract_laravel_controller_flows(
            ctx.code_map, ctx.php_project_path
        )
        existing_files = {p["file"] for p in all_paths}
        for lp in laravel_paths:
            if lp["file"] not in existing_files:
                all_paths.append(lp)
            else:
                # Merge controller_flows and happy_path into existing record
                for existing in all_paths:
                    if existing["file"] == lp["file"]:
                        existing.setdefault("controller_flows", [])
                        existing["controller_flows"].extend(
                            lp.get("controller_flows", [])
                        )
                        if lp.get("happy_path"):
                            existing["happy_path"] = (
                                existing.get("happy_path", []) + lp["happy_path"]
                            )
                        break

        print(f"  [stage15] Laravel enrichment pass: "
              f"{len(laravel_paths)} controller record(s) added/merged.")

    # ── Error reporting ───────────────────────────────────────────────────────
    if errors:
        print(f"  [stage15] {len(errors)} file(s) raised exceptions during analysis:")
        for err in errors[:10]:
            print(f"    • {err['file']}: {err['error']}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more — see {error_path}")
        Path(error_path).write_text(
            json.dumps(errors, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        print("  [stage15] No analysis errors.")

    # ── Attach to CodeMap ─────────────────────────────────────────────────────
    _attach_to_code_map(ctx.code_map, all_paths)

    # ── Persist ───────────────────────────────────────────────────────────────
    Path(output_path).write_text(
        json.dumps(all_paths, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ctx.stage("stage15_paths").mark_completed(output_path)
    ctx.save()

    _print_summary(all_paths)
    print(f"  [stage15] Saved → {output_path}")


# ─── File Analyser (dispatch) ─────────────────────────────────────────────────

def analyse_file(
    php_file:     str,
    project_root: str,
    framework:    Framework = Framework.UNKNOWN,
) -> dict[str, Any] | None:
    """
    Analyse a single PHP file and return its execution path model.

    For Laravel projects:
      • Route files  → _LaravelRouteAnalyser
      • Controllers  → _LaravelMethodAnalyser
      • Other files  → raw PHP analyser fallback

    Args:
        php_file:     Absolute path to the .php file.
        project_root: Project root for computing relative paths.
        framework:    Detected framework; affects which analyser is used.

    Returns:
        Path model dict, or None if the file has no interesting content.
    """
    source = Path(php_file).read_text(encoding="utf-8", errors="replace")
    rel    = str(Path(php_file).relative_to(project_root))

    # ── Laravel-specific analysers ────────────────────────────────────────────
    if framework == Framework.LARAVEL:
        if _is_laravel_route_file(php_file):
            result = _LaravelRouteAnalyser(source, rel).analyse()
            if result:
                return result

        if _is_laravel_controller(php_file, source):
            result = _LaravelMethodAnalyser(source, rel).analyse()
            if result:
                return result

    # ── SuiteCRM / SugarCRM bean files ────────────────────────────────────────
    if _is_sugarcrm_bean(php_file, source):
        result = _SugarCRMBeanAnalyser(source, rel).analyse()
        if result:
            return result

    # ── Raw PHP / legacy fallback ─────────────────────────────────────────────
    return _FileAnalyser(source, rel).analyse()


# ─── Laravel Detection Helpers ────────────────────────────────────────────────

def _is_laravel_route_file(php_file: str) -> bool:
    """Return True if the file is a Laravel route definition file."""
    p = Path(php_file)
    return p.parent.name == "routes" and p.suffix == ".php"


def _is_laravel_controller(php_file: str, source: str) -> bool:
    """Return True if this PHP file looks like a Laravel controller."""
    p = Path(php_file)
    return (
        "Controller" in p.stem
        or "Controllers" in str(p)
        or re.search(r'extends\s+\w*Controller', source) is not None
    )


# ─── SuiteCRM / SugarCRM Bean Detection Helpers ───────────────────────────────

def _is_sugarcrm_bean(php_file: str, source: str) -> bool:
    """Return True if this file is a SuiteCRM/SugarCRM bean/model file.

    Detection heuristics (any one is sufficient):
    - File path matches modules/<ModuleName>/<ClassName>.php
    - Source contains 'extends SugarBean' / 'extends Basic' / etc.
    - Source contains '$this->db->query(' — characteristic SugarCRM DB call
    """
    p = Path(php_file)
    # Path heuristic: modules/<Module>/<File>.php
    parts = p.parts
    try:
        mod_idx = next(i for i, pt in enumerate(parts) if pt == "modules")
        if len(parts) - mod_idx >= 3:  # modules/<Mod>/<File>.php
            return True
    except StopIteration:
        pass
    # Source heuristics
    if re.search(
        r'\bextends\s+(?:SugarBean|Basic|Person|Company|SugarObject|'
        r'VardefManager|SugarLogger|SugarView)\b',
        source, re.IGNORECASE,
    ):
        return True
    if re.search(r'\$this\s*->\s*db\s*->\s*(?:query|insertRecord|updateRecord'
                 r'|deleteRecord|insert|update|delete)\s*\(', source, re.IGNORECASE):
        return True
    return False


class _SugarCRMBeanAnalyser:
    """
    Analyses SuiteCRM/SugarCRM bean files for SQL operations.

    Bean files (e.g. modules/Accounts/Account.php) extend SugarBean and
    interact with the database via ``$this->db->query()``,
    ``$this->db->insertRecord()``, ``DBManager::getInstance()->query()``,
    ``$this->save()``, etc.

    These files are never web entry points (no ``$_POST``, ``header()`` calls)
    so ``_FileAnalyser`` returns ``None`` for them.  This analyser emits a
    minimal path record so bean files appear in ``execution_paths`` and V-06
    can count their SQL write ops as covered by flows that reference the module.
    """

    _DB_CALL_PAT = re.compile(
        r'(?:'
        r'\$this\s*->\s*db\s*->\s*(?:query|limitQuery|prepare|'
        r'insertRecord|updateRecord|deleteRecord|insert|update|delete|fetchByAssoc)'
        r'|DBManager\s*::\s*getInstance\s*\(\s*\)\s*->\s*(?:query|insert|update|delete)'
        r'|\$(?:db|conn|pdo)\s*->\s*(?:query|insertRecord|updateRecord|deleteRecord|'
        r'insert|update|delete|prepare)'
        r')\s*\(',
        re.IGNORECASE,
    )
    _INLINE_SQL_PAT = re.compile(
        r'''['"]([ \t]*(?:SELECT|INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER|TRUNCATE)\b[^'"]{4,})['"]''',
        re.IGNORECASE,
    )
    _SAVE_PAT = re.compile(r'\$this\s*->\s*save\s*\(\s*\)', re.IGNORECASE)

    def __init__(self, source: str, filename: str) -> None:
        self.source   = source
        self.filename = filename

    def analyse(self) -> dict[str, Any] | None:
        data_flows: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()

        for m in self._DB_CALL_PAT.finditer(self.source):
            # Scan ahead up to 300 chars for an inline SQL string
            region = self.source[m.start(): m.start() + 300]
            sql_m  = self._INLINE_SQL_PAT.search(region)
            if sql_m:
                sql_frag = sql_m.group(1).strip()[:120]
                op       = _classify_sql_op(sql_frag)
                tbl      = _extract_table_name(sql_frag)
                key      = (op, tbl)
                if key not in seen_keys:
                    seen_keys.add(key)
                    data_flows.append({
                        "source":    "sugarcrm_db",
                        "sql":       sql_frag[:80],
                        "operation": op.upper(),
                        "table":     tbl,
                    })
            else:
                # No inline string — record the call site without a table name
                call_snippet = self.source[m.start(): m.end() + 30].strip()[:60]
                op_key = ("query", "")
                if op_key not in seen_keys:
                    seen_keys.add(op_key)
                    data_flows.append({
                        "source":    "sugarcrm_db",
                        "sql":       call_snippet,
                        "operation": "QUERY",
                        "table":     "",
                    })

        # $this->save() → SugarBean ORM INSERT/UPDATE
        save_count = len(self._SAVE_PAT.findall(self.source))
        if save_count:
            bean_table = Path(self.filename).stem.lower()
            data_flows.append({
                "source":    "sugarcrm_save",
                "sql":       f"$this->save() ×{save_count}",
                "operation": "INSERT/UPDATE",
                "table":     bean_table,
            })

        if not data_flows:
            return None

        happy_path = [
            (f"DB {df['operation']} on `{df['table']}`" if df["table"]
             else f"DB {df['operation']}")
            for df in data_flows[:6]
        ]

        return {
            "file":             self.filename,
            "type":             "sugarcrm_bean",
            "entry_conditions": [],
            "branches":         [],
            "data_flows":       data_flows,
            "auth_guard":       None,
            "happy_path":       happy_path,
        }


# ─── Laravel Route File Analyser ──────────────────────────────────────────────

class _LaravelRouteAnalyser:
    """
    Parses Laravel route files (routes/api.php, routes/web.php) to extract:
    - HTTP method + URI + controller@method or closure registrations
    - Middleware applied inline or via group
    - Auth requirements inferred from middleware names
    """

    _ROUTE_PAT = re.compile(
        r"Route\s*::\s*(get|post|put|patch|delete|any|match|resource|apiResource)"
        r"\s*\(\s*['\"]([^'\"]+)['\"]"
        r"(?:"
        r"\s*,\s*['\"]([^'\"@]+)@(\w+)['\"]"      # 'Ctrl@method'
        r"|\s*,\s*\[([^\]]*)\]"                     # array syntax
        r"|\s*,\s*\\?([A-Z][A-Za-z\\]+)::class"    # ::class invokable
        r")?",
        re.IGNORECASE,
    )
    _MIDDLEWARE_PAT       = re.compile(r"->middleware\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
    _INLINE_MIDDLEWARE_PAT= re.compile(r"['\"]middleware['\"]\s*=>\s*['\"]([^'\"]+)['\"]")

    def __init__(self, source: str, filename: str) -> None:
        self.source   = source
        self.filename = filename

    def analyse(self) -> dict[str, Any] | None:
        """Extract route registrations as entry conditions and happy paths."""
        routes: list[dict[str, Any]] = []

        for m in self._ROUTE_PAT.finditer(self.source):
            method     = m.group(1).upper()
            uri        = m.group(2)
            ctrl_class = m.group(3) or m.group(6) or ""
            ctrl_action= m.group(4) or "__invoke"
            array_body = m.group(5) or ""

            # Extract controller/method from array syntax
            if array_body and not ctrl_class:
                uses_m = re.search(
                    r"['\"]uses['\"]\s*=>\s*['\"]([^@'\"]+)@(\w+)['\"]",
                    array_body,
                )
                if uses_m:
                    ctrl_class  = uses_m.group(1)
                    ctrl_action = uses_m.group(2)

            # Collect middleware on this route's chained call context
            line_end  = self.source.find(";", m.start())
            line_ctx  = self.source[m.start(): line_end + 1 if line_end > 0 else m.end() + 200]
            middleware = (
                self._MIDDLEWARE_PAT.findall(line_ctx)
                + self._INLINE_MIDDLEWARE_PAT.findall(array_body)
            )

            auth_required = any(
                mw in ("auth", "auth:sanctum", "auth:api", "verified")
                for mw in middleware
            )

            routes.append({
                "http_method":   method,
                "uri":           uri,
                "controller":    ctrl_class.split("\\")[-1] if ctrl_class else "",
                "action":        ctrl_action,
                "middleware":    middleware,
                "auth_required": auth_required,
            })

        if not routes:
            return None

        happy_path = [
            f"{r['http_method']} {r['uri']}"
            + (f" → {r['controller']}@{r['action']}" if r["controller"] else "")
            + (" [auth]" if r["auth_required"] else "")
            for r in routes[:10]
        ]

        return {
            "file":             self.filename,
            "type":             "route_file",
            "entry_conditions": [{"type": "route_registration", "count": len(routes)}],
            "branches":         [],
            "data_flows":       [],
            "auth_guard":       None,
            "happy_path":       happy_path,
            "routes":           routes,
        }


# ─── Laravel Method-Level Controller Analyser ─────────────────────────────────

class _LaravelMethodAnalyser:
    """
    Analyses a Laravel controller at the method level to extract:
    - Per-public-method: HTTP inputs, Eloquent operations, return type
    - Constructor middleware for auth inference
    - Validation rules from $request->validate()
    """

    # Eloquent operations
    _ELOQUENT_PAT = re.compile(
        r'(\$\w+|[A-Z][A-Za-z]+)\s*(?:::|->)\s*'
        r'(create|save|update|delete|destroy|find|findOrFail|first|'
        r'firstOrCreate|firstOrFail|where|get|all|paginate|count|'
        r'insert|upsert|updateOrCreate|forceDelete|restore)',
        re.IGNORECASE,
    )
    # $request->validate([...])
    _VALIDATE_PAT = re.compile(
        r'(?:\$request|this)\s*->\s*validate\s*\(\s*\[([^\]]{3,})\]',
        re.DOTALL,
    )
    # Input field access
    _INPUT_PAT = re.compile(
        r'\$request\s*->\s*(?:input|get|post|query|only|except)\s*\(\s*[\'"](\w+)[\'"]'
        r'|\$request\s*->\s*(\w+)\b'
        r"|request\s*\(\s*['\"](\w+)['\"]",
    )
    _RETURN_VIEW_PAT  = re.compile(r'return\s+view\s*\(\s*[\'"]([^\'"]+)[\'"]')
    _RETURN_REDIR_PAT = re.compile(r'return\s+redirect\s*(?:\(\s*\)\s*->|)\s*(?:route|to|away)?\s*\(\s*[\'"]?([^\'\")\s,;]+)')
    _RETURN_JSON_PAT  = re.compile(r'return\s+(?:response\s*\(\s*\)\s*->)?\s*json\s*\(')
    _RETURN_BACK_PAT  = re.compile(r'return\s+(?:back\s*\(\s*\)|redirect\s*\(\s*\)\s*->\s*back\s*\()')
    _METHOD_PAT       = re.compile(r'\bpublic\s+function\s+(\w+)\s*\(([^)]*)\)', re.IGNORECASE)
    _CTOR_MIDDLEWARE  = re.compile(r'\$this\s*->\s*middleware\s*\(\s*[\'"]([^\'"]+)[\'"]', re.IGNORECASE)

    # Non-action method names to skip
    _SKIP_METHODS = frozenset({
        "boot", "register", "handle", "setUp", "tearDown", "authorize",
        "rules", "messages", "attributes", "prepareForValidation",
    })

    def __init__(self, source: str, filename: str) -> None:
        self.source   = source
        self.filename = filename

    def analyse(self) -> dict[str, Any] | None:
        """Extract per-method flows from a Laravel controller."""
        controller_flows: list[dict[str, Any]] = []
        all_happy_path:   list[str]            = []

        # Auth middleware declared in __construct
        global_middleware = self._CTOR_MIDDLEWARE.findall(self.source)
        global_auth = any(
            mw in ("auth", "auth:sanctum", "auth:api", "verified")
            for mw in global_middleware
        )

        for m in self._METHOD_PAT.finditer(self.source):
            method_name = m.group(1)
            if method_name.startswith("__") or method_name in self._SKIP_METHODS:
                continue

            # Extract method body using brace-balanced matching
            brace_pos = self.source.find("{", m.end())
            if brace_pos < 0:
                continue
            body_end = _find_matching(self.source, brace_pos, "{", "}")
            if body_end < 0:
                continue
            body = self.source[brace_pos + 1: body_end]

            # ── Inputs ───────────────────────────────────────────────────────
            inputs: list[str] = []
            _skip_fields = {"all", "user", "route", "file", "header", "ip",
                            "method", "url", "path", "has", "filled", "missing"}
            for im in self._INPUT_PAT.finditer(body):
                field = im.group(1) or im.group(2) or im.group(3) or ""
                if field and field not in _skip_fields:
                    inputs.append(field)
            inputs = list(dict.fromkeys(inputs))

            # ── Validation rules ──────────────────────────────────────────────
            validation_rules: list[str] = []
            for vm in self._VALIDATE_PAT.finditer(body):
                keys = re.findall(r"['\"](\w+)['\"]\s*=>", vm.group(1))
                validation_rules.extend(keys)

            # ── Eloquent ops ──────────────────────────────────────────────────
            db_ops: list[dict[str, str]] = []
            _skip_subjects = {"$request", "$response", "$this", "$validator",
                              "$job", "$event", "$listener", "$command"}
            for em in self._ELOQUENT_PAT.finditer(body):
                subj = em.group(1).lower()
                if subj in _skip_subjects:
                    continue
                db_ops.append({"model": em.group(1), "operation": em.group(2).lower()})

            # ── Return type ───────────────────────────────────────────────────
            return_type = "unknown"
            outputs:    list[str] = []
            if self._RETURN_JSON_PAT.search(body):
                return_type = "json"
                outputs.append("json_response")
            elif self._RETURN_BACK_PAT.search(body):
                return_type = "redirect_back"
                outputs.append("redirect_back")
            else:
                rv = self._RETURN_VIEW_PAT.search(body)
                if rv:
                    return_type = "view"
                    outputs.append(f"view:{rv.group(1)}")
                rr = self._RETURN_REDIR_PAT.search(body)
                if rr:
                    return_type = return_type if return_type != "unknown" else "redirect"
                    outputs.append(f"redirect:{rr.group(1)}")

            if not db_ops and not inputs and return_type == "unknown":
                continue

            flow: dict[str, Any] = {
                "method":           method_name,
                "inputs":           inputs,
                "validation_rules": validation_rules,
                "db_ops":           db_ops,
                "return_type":      return_type,
                "outputs":          outputs,
                "auth_required":    global_auth,
            }
            controller_flows.append(flow)

            # Build one happy_path entry per method
            parts: list[str] = []
            if inputs:
                parts.append(f"Accept: {', '.join(inputs[:4])}")
            if validation_rules:
                parts.append(f"Validate: {', '.join(validation_rules[:4])}")
            for op in db_ops[:3]:
                parts.append(f"Eloquent {op['operation']}: {op['model']}")
            if outputs:
                parts.append(f"Return {outputs[0]}")
            if parts:
                all_happy_path.append(f"[{method_name}] " + " → ".join(parts))

        if not controller_flows:
            return None

        data_flows = [
            {
                "source": f["inputs"],
                "sink":   "eloquent",
                "model":  f["db_ops"][0]["model"] if f["db_ops"] else "?",
                "method": f["method"],
            }
            for f in controller_flows if f["inputs"] and f["db_ops"]
        ]

        return {
            "file":             self.filename,
            "type":             "controller",
            "entry_conditions": [{"type": "auth_middleware", "required": True}]
                                  if global_auth else [],
            "branches":         [],
            "data_flows":       data_flows,
            "auth_guard":       {"type": "middleware", "middleware": global_middleware}
                                  if global_middleware else None,
            "happy_path":       all_happy_path,
            "controller_flows": controller_flows,
        }


# ─── Laravel Controller Bulk Pass ─────────────────────────────────────────────

def _extract_laravel_controller_flows(
    code_map: CodeMap,
    project_root: str,
) -> list[dict[str, Any]]:
    """
    Second-pass enrichment using structured data already in code_map.controllers
    rather than re-parsing raw PHP.  This fills gaps for controllers whose PHP
    source returned no raw-PHP signals (no $_POST, header(), or $_SESSION).

    Produces one execution-path record per controller that has registered routes.

    Args:
        code_map:     The parsed CodeMap from Stage 1.
        project_root: Project root path (for relative file names).

    Returns:
        List of execution-path records keyed by relative controller file path.
    """
    results: list[dict[str, Any]] = []

    # Index routes: controller class short name → list of route records
    route_index: dict[str, list[dict]] = {}
    for route in code_map.routes:
        # Support both 'controller' key and 'action' string like 'Ctrl@method'
        ctrl = route.get("controller") or ""
        if not ctrl:
            action_str = route.get("action", "")
            ctrl = action_str.split("@")[0] if "@" in action_str else ""
        if ctrl:
            # Normalise: strip namespace, keep short class name
            ctrl_key = ctrl.lstrip("\\").split("\\")[-1]
            route_index.setdefault(ctrl_key, []).append(route)

    for ctrl in code_map.controllers:
        ctrl_name = ctrl.get("name", "")
        ctrl_file = ctrl.get("file", "")
        if not ctrl_name or not ctrl_file:
            continue

        try:
            rel_file = str(Path(ctrl_file).relative_to(project_root))
        except ValueError:
            rel_file = ctrl_file

        routes_for_ctrl = route_index.get(ctrl_name, [])
        methods: list[dict[str, Any]] = ctrl.get("methods", [])

        if not methods and not routes_for_ctrl:
            continue

        happy_path:       list[str]            = []
        controller_flows: list[dict[str, Any]] = []

        for method in methods:
            m_name = method.get("name", "?")
            if m_name.startswith("__"):
                continue

            # Find matching route entry for this method
            route_match = next(
                (
                    r for r in routes_for_ctrl
                    if r.get("action", "").endswith(f"@{m_name}")
                    or r.get("method_name") == m_name
                ),
                None,
            )
            http_verb = route_match.get("http_method", "GET") if route_match else "GET"
            uri       = route_match.get("uri", "")           if route_match else ""

            flow: dict[str, Any] = {
                "method":        m_name,
                "http_verb":     http_verb,
                "uri":           uri,
                "inputs":        method.get("params", []),
                "db_ops":        method.get("db_ops", []),
                "return_type":   method.get("return_type", "unknown"),
                "auth_required": bool(method.get("middleware") or method.get("auth")),
            }
            controller_flows.append(flow)

            step = f"[{http_verb} {uri or m_name}] {ctrl_name}::{m_name}"
            if flow["db_ops"]:
                step += f" → {len(flow['db_ops'])} DB op(s)"
            happy_path.append(step)

        if not controller_flows:
            continue

        results.append({
            "file":             rel_file,
            "type":             "controller",
            "entry_conditions": [],
            "branches":         [],
            "data_flows":       [],
            "auth_guard":       None,
            "happy_path":       happy_path[:8],
            "controller_flows": controller_flows,
        })

    return results


# ─── Core Raw-PHP Analyser ─────────────────────────────────────────────────────

class _FileAnalyser:
    """
    Regex + lightweight AST-style analyser for a single PHP file.

    Handles raw PHP and non-Laravel frameworks.  Looks for:
    - $_SESSION / $_POST / $_GET based auth and input patterns
    - Direct mysqli/PDO SQL queries
    - header("Location:") redirects
    - include/require chains
    """

    def __init__(self, source: str, filename: str) -> None:
        self.source   = source
        self.filename = filename
        self.lines    = source.splitlines()

    def analyse(self) -> dict[str, Any] | None:
        entry_conditions = self._extract_entry_conditions()
        branches         = self._extract_branches()
        data_flows       = self._extract_data_flows()
        auth_guard       = self._extract_auth_guard()
        happy_path       = self._build_happy_path(
            entry_conditions, branches, data_flows, auth_guard
        )

        if (not entry_conditions and not branches
                and not data_flows and not auth_guard):
            return None

        return {
            "file":             self.filename,
            "type":             "raw_php",
            "entry_conditions": entry_conditions,
            "branches":         branches,
            "data_flows":       data_flows,
            "auth_guard":       auth_guard,
            "happy_path":       happy_path,
        }

    # ── Entry Conditions ──────────────────────────────────────────────────────

    def _extract_entry_conditions(self) -> list[dict[str, Any]]:
        conditions = []
        top = "\n".join(self.lines[:40])

        pat = re.compile(
            r"if\s*\(\s*(!?)isset\s*\(\s*\$_SESSION\[(['\"])(\w+)\2\]\s*\)\s*\)"
            r"[^{]*\{[^}]*header\s*\(\s*['\"]Location:\s*([^'\"]+)['\"]",
            re.IGNORECASE | re.DOTALL,
        )
        for m in pat.finditer(top):
            negated     = m.group(1) == "!"
            session_key = m.group(3)
            redirect_to = m.group(4).strip()
            conditions.append({
                "type":        "session_check",
                "key":         session_key,
                "op":          "not_isset" if negated else "isset",
                "on_fail":     "redirect",
                "redirect_to": redirect_to,
            })

        if re.search(r"session_start\s*\(\s*\)", top, re.IGNORECASE):
            conditions.append({"type": "session_start"})

        if re.search(r"\$_SERVER\[['\"]REQUEST_METHOD['\"]\]\s*==\s*['\"]POST['\"]",
                     top, re.IGNORECASE):
            conditions.append({"type": "method_check", "method": "POST"})

        return conditions

    # ── Branch Extraction ─────────────────────────────────────────────────────

    def _extract_branches(self) -> list[dict[str, Any]]:
        branches = []
        if_blocks = self._find_if_blocks(self.source)

        for block in if_blocks:
            condition = block["condition"]
            then_body = block["then_body"]
            else_body = block.get("else_body", "")

            cond_vars = _extract_vars_from_condition(condition)
            if not cond_vars and not _has_interesting_ops(then_body):
                continue

            then_actions = self._extract_actions(then_body)
            else_actions = self._extract_actions(else_body) if else_body else []

            if not then_actions and not else_actions:
                continue

            branches.append({
                "condition":      _clean_condition(condition),
                "condition_vars": cond_vars,
                "then":           then_actions,
                "else":           else_actions,
            })

        return branches[:10]

    def _find_if_blocks(self, source: str) -> list[dict[str, Any]]:
        blocks = []
        pat    = re.compile(r'\bif\s*\(', re.IGNORECASE)

        for m in pat.finditer(source):
            cond_start = m.end() - 1
            cond_end   = _find_matching(source, cond_start, "(", ")")
            if cond_end < 0:
                continue
            condition = source[cond_start + 1: cond_end].strip()

            brace_pos = source.find("{", cond_end)
            if brace_pos < 0:
                continue
            then_end = _find_matching(source, brace_pos, "{", "}")
            if then_end < 0:
                continue
            then_body = source[brace_pos + 1: then_end]

            else_body = ""
            after     = source[then_end + 1: then_end + 200].lstrip()
            else_m    = re.match(r'^else\s*\{', after, re.IGNORECASE)
            if else_m:
                else_start = then_end + 1 + (len(source[then_end + 1:]) - len(after))
                else_brace = source.find("{", else_start + else_m.start())
                if else_brace >= 0:
                    else_end = _find_matching(source, else_brace, "{", "}")
                    if else_end >= 0:
                        else_body = source[else_brace + 1: else_end]

            blocks.append({
                "condition": condition,
                "then_body": then_body,
                "else_body": else_body,
            })

        return blocks

    def _extract_actions(self, body: str) -> list[dict[str, Any]]:
        actions = []

        for m in re.finditer(
            r'(?:mysqli_query|mysql_query|\$(?:conn|db|pdo|mysqli)\s*->\s*query'
            r'|\$(?:conn|db|pdo|stmt)\s*->\s*prepare)\s*\(\s*["\']?\s*([^)]{5,})',
            body, re.IGNORECASE,
        ):
            sql_fragment = m.group(1)[:120].strip().strip("\"'")
            op  = _classify_sql_op(sql_fragment)
            tbl = _extract_table_name(sql_fragment)
            actions.append({"action": f"sql_{op}", "table": tbl, "detail": sql_fragment[:80]})

        for m in re.finditer(r'\$_SESSION\[([\'"])(\w+)\1\]\s*=\s*([^;]+);', body):
            actions.append({
                "action": "session_write",
                "key":    m.group(2),
                "detail": f"$_SESSION['{m.group(2)}'] = {m.group(3).strip()[:60]}",
            })

        for m in re.finditer(r'header\s*\(\s*[\'"]Location:\s*([^\'\"]+)[\'"]',
                              body, re.IGNORECASE):
            actions.append({"action": "redirect", "target": m.group(1).strip()})

        echo_count = len(re.findall(r'\becho\b|\bprint\b', body, re.IGNORECASE))
        if echo_count > 0:
            actions.append({"action": "output_html", "count": echo_count})

        if re.search(r'\bdie\b|\bexit\b', body, re.IGNORECASE):
            actions.append({"action": "terminate"})

        for m in re.finditer(r'\b(?:include|require)(?:_once)?\s*\(?[\'"]([^\'"]+)[\'"]',
                              body, re.IGNORECASE):
            actions.append({"action": "include", "file": m.group(1)})

        return actions

    # ── Data Flow ─────────────────────────────────────────────────────────────

    def _extract_data_flows(self) -> list[dict[str, Any]]:
        flows  = []
        source = self.source

        var_map: dict[str, str] = {}
        # Direct assignment: $var = $_POST['key']
        for m in re.finditer(
            r'\$(\w+)\s*=\s*\$_(POST|GET|REQUEST)\s*\[\s*[\'"](\w+)[\'"]\s*\]', source
        ):
            var_map[m.group(1)] = f"$_{m.group(2)}['{m.group(3)}']"

        # Wrapped/sanitized assignment: $var = anyFn($_POST['key'])
        # e.g. $var = intval($_POST['id']), $var = htmlspecialchars($_GET['q'])
        for m in re.finditer(
            r'\$(\w+)\s*=\s*\w+\s*\(\s*\$_(POST|GET|REQUEST)\s*\[\s*[\'"](\w+)[\'"]\s*\]',
            source,
        ):
            if m.group(1) not in var_map:   # don't overwrite direct assignments
                var_map[m.group(1)] = f"$_{m.group(2)}['{m.group(3)}']"

        # filter_input(INPUT_POST/GET/REQUEST, 'key') — standard PHP input filter
        for m in re.finditer(
            r'\$(\w+)\s*=\s*filter_input\s*\(\s*INPUT_(POST|GET|REQUEST)\s*,\s*[\'"](\w+)[\'"]',
            source, re.IGNORECASE,
        ):
            if m.group(1) not in var_map:
                var_map[m.group(1)] = f"$_{m.group(2)}['{m.group(3)}']"

        sql_var_map: dict[str, str] = {}
        for m in re.finditer(r'\$(\w+)\s*=\s*["\']([^"\']{10,})["\']', source):
            val = m.group(2).upper().strip()
            if any(val.startswith(kw) for kw in ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE")):
                sql_var_map[m.group(1)] = m.group(2)

        for m in re.finditer(
            r'\$(\w+)\s*=\s*["\']([^"\']{6,})["\']'
            r'(?:\s*\.\s*\$(\w+)(?:\s*\.\s*["\'][^"\']*["\'])?)+',
            source,
        ):
            val = m.group(2).upper().strip()
            if any(val.startswith(kw) for kw in ("SELECT", "INSERT", "UPDATE", "DELETE")):
                sql_var_map[m.group(1)] = source[m.start():m.end()]

        query_targets: list[str] = []

        for m in re.finditer(
            r'(?:mysqli_query|mysql_query|\$(?:conn|db|pdo|mysqli)\s*->\s*(?:query|prepare))'
            r'\s*\(\s*(?:\$\w+\s*,\s*)?["\']([^"\']{5,})["\']',
            source, re.IGNORECASE,
        ):
            query_targets.append(m.group(1))

        for m in re.finditer(
            r'(?:mysqli_query|mysql_query|\$(?:conn|db|pdo|mysqli)\s*->\s*(?:query|prepare))'
            r'\s*\(\s*\$\w+\s*,\s*\$(\w+)\s*\)',
            source, re.IGNORECASE,
        ):
            var_name = m.group(1)
            if var_name in sql_var_map:
                query_targets.append(sql_var_map[var_name])

        for sql in query_targets:
            table = _extract_table_name(sql)
            op    = _classify_sql_op(sql)

            field_mapping: dict[str, str] = {}
            for im in re.finditer(r'\$_(POST|GET|REQUEST)\[\'(\w+)\'\]', sql):
                field_mapping[im.group(2)] = f"$_{im.group(1)}['{im.group(2)}']"

            for var, origin in var_map.items():
                if f"${var}" in sql or f"'{var}'" in sql:
                    fname = re.search(r"\['(\w+)'\]", origin)
                    if fname:
                        field_mapping[fname.group(1)] = origin

            sql_pos = source.find(sql[:40])
            if sql_pos >= 0:
                context_window = source[max(0, sql_pos - 300): sql_pos + 50]
                for var, origin in var_map.items():
                    if f"${var}" in context_window:
                        fname = re.search(r"\['(\w+)'\]", origin)
                        if fname:
                            field_mapping[fname.group(1)] = origin

            if field_mapping or table:
                flows.append({
                    "source":        list(field_mapping.values()) or ["$_POST/$_GET"],
                    "sink":          f"sql_{op}",
                    "table":         table,
                    "field_mapping": field_mapping,
                })

        return flows

    # ── Auth Guard ────────────────────────────────────────────────────────────

    def _extract_auth_guard(self) -> dict[str, Any] | None:
        top = "\n".join(self.lines[:50])

        # Pattern 1: if (!isset($_SESSION['key'])) { header("Location: ...") }
        m = re.search(
            r'if\s*\(\s*(!?)isset\s*\(\s*\$_SESSION\[([\'"])(\w+)\2\]\s*\)\s*\)'
            r'[^{]*\{[^}]*header\s*\(\s*[\'"]Location:\s*([^\'\"]+)[\'"]',
            top, re.IGNORECASE | re.DOTALL,
        )
        if m:
            return {
                "key":      m.group(3),
                "check":    "not_isset" if m.group(1) == "!" else "isset",
                "redirect": m.group(4).strip(),
            }

        # Pattern 2: $_SESSION['key'] == 'value' comparison
        m = re.search(
            r'\$_SESSION\[([\'"])(\w+)\1\]\s*(==|!=|===)\s*[\'"](\w+)[\'"]',
            top,
        )
        if m:
            return {"key": m.group(2), "check": f"{m.group(3)} '{m.group(4)}'", "redirect": None}

        # Pattern 3: isset($_SESSION['key']) or die/exit  (short-circuit auth)
        # e.g.  isset($_SESSION['user']) or die('Not authorized');
        m = re.search(
            r'isset\s*\(\s*\$_SESSION\[([\'"])(\w+)\1\]\s*\)\s*or\s*(?:die|exit)\s*\(',
            top, re.IGNORECASE,
        )
        if m:
            return {"key": m.group(2), "check": "isset_or_die", "redirect": None}

        # Pattern 4: if (!defined('CONSTANT')) die/exit  (direct-access guard)
        # Generic pattern used by WordPress, Joomla, and many raw PHP apps.
        m = re.search(
            r'if\s*\(\s*!defined\s*\(\s*[\'"](\w+)[\'"]\s*\)\s*\)\s*(?:die|exit)\s*[(\s;]',
            top, re.IGNORECASE,
        )
        if m:
            return {"key": m.group(1), "check": "defined_guard", "redirect": None}

        # Pattern 5: require/include of a file with auth-related name
        # e.g. require_once('includes/auth.php'), include('session_check.php')
        # Generic — many PHP apps gate pages via an included auth file.
        m = re.search(
            r'(?:require|include)(?:_once)?\s*\(?\s*[\'"][^\'"]*'
            r'(?:auth|login|session_check|access_check|check_login|guard|security)[^\'"]*\.php[\'"]',
            top, re.IGNORECASE,
        )
        if m:
            return {"key": "auth_include", "check": "require_auth_file", "redirect": None}

        return None

    # ── Happy Path Builder ────────────────────────────────────────────────────

    def _build_happy_path(
        self,
        entry_conditions: list[dict],
        branches:         list[dict],
        data_flows:       list[dict],
        auth_guard:       dict | None,
    ) -> list[str]:
        steps: list[str] = []

        if any(c.get("type") == "session_start" for c in entry_conditions):
            steps.append("Session started (session_start)")

        if auth_guard:
            steps.append(
                f"Auth check: $_SESSION['{auth_guard['key']}'] "
                f"must be set (else → {auth_guard.get('redirect', 'redirect')})"
            )

        post_fields: set[str] = set()
        for flow in data_flows:
            for v in flow.get("source", []):
                m = re.search(r"\$_(POST|GET)\['(\w+)'\]", v)
                if m:
                    post_fields.add(f"$_{m.group(1)}['{m.group(2)}']")
        if post_fields:
            steps.append(f"Receive input: {', '.join(sorted(post_fields))}")

        auth_redirect = auth_guard.get("redirect") if auth_guard else None

        def _is_success_branch(b: dict) -> bool:
            then = b.get("then", [])
            return (
                any(a.get("action") == "session_write" for a in then) or
                any(
                    a.get("action") == "redirect"
                    and a.get("target", "") != auth_redirect
                    and "error" not in a.get("target", "").lower()
                    and "fail"  not in a.get("target", "").lower()
                    for a in then
                )
            )

        success_branches = [
            b for b in branches
            if _is_success_branch(b) and "_SESSION" not in b.get("condition", "")
        ]

        for branch in success_branches[:1]:
            for action in branch.get("then", []):
                act = action.get("action", "")
                tgt = action.get("target", "")
                if act == "redirect" and ("error" in tgt.lower() or tgt == auth_redirect):
                    continue
                if act.startswith("sql_"):
                    tbl = action.get("table", "?")
                    steps.append(
                        f"{act.replace('sql_','SQL ').upper()} on `{tbl}`: "
                        f"{action.get('detail','')[:60]}"
                    )
                elif act == "session_write":
                    steps.append(f"Write session: $_SESSION['{action.get('key')}']")
                elif act == "redirect":
                    steps.append(f"Redirect → {tgt}")
                elif act == "output_html":
                    steps.append("Render HTML output")
                elif act == "include":
                    steps.append(f"Include {action.get('file', '?')}")

        if not any("Redirect →" in s for s in steps):
            all_redirects = [
                m.group(1).strip()
                for m in re.finditer(
                    r'header\s*\(\s*[\'"]Location:\s*([^\'\"]+)[\'"]',
                    self.source, re.IGNORECASE,
                )
            ]
            success_redirect = next(
                (
                    r for r in reversed(all_redirects)
                    if r != auth_redirect
                    and "error" not in r.lower()
                    and "fail"  not in r.lower()
                ),
                None,
            )
            if success_redirect:
                steps.append(f"Redirect → {success_redirect}")

        return steps


# ─── Shared Helpers ───────────────────────────────────────────────────────────

def _find_matching(source: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    """Find the position of the matching closing character, skipping nested pairs."""
    depth  = 0
    i      = open_pos
    in_str: str | None = None

    while i < len(source):
        ch = source[i]
        if in_str:
            if ch == in_str and source[i - 1] != "\\":
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _extract_vars_from_condition(condition: str) -> list[str]:
    """Extract $_POST, $_GET, $_SESSION variable references from a condition."""
    return list(dict.fromkeys(
        re.findall(r'\$_(POST|GET|SESSION|REQUEST)\[[^\]]+\]', condition)
    ))


def _has_interesting_ops(body: str) -> bool:
    """Return True if body contains DB, redirect, or session operations."""
    return bool(re.search(
        r'mysqli_query|mysql_query|\$\w+->query|\bheader\s*\(|'
        r'\$_SESSION\s*\[|\bdie\b|\bexit\b',
        body, re.IGNORECASE,
    ))


def _clean_condition(condition: str) -> str:
    """Normalise a condition string for readability."""
    c = re.sub(r'\s+', ' ', condition).strip()
    return c[:120] + "..." if len(c) > 120 else c


def _classify_sql_op(sql: str) -> str:
    """Classify SQL string as read/write/ddl."""
    s = sql.strip().upper()
    if s.startswith("SELECT"):  return "read"
    if s.startswith("INSERT"):  return "write"
    if s.startswith("UPDATE"):  return "write"
    if s.startswith("DELETE"):  return "write"
    if s.startswith("CREATE"):  return "ddl"
    if s.startswith("DROP"):    return "ddl"
    if s.startswith("ALTER"):   return "ddl"
    return "query"


def _extract_table_name(sql: str) -> str:
    """Extract the primary table name from a SQL fragment."""
    sql_u = sql.upper()
    for pat in [
        r'FROM\s+[`\'"]?(\w+)',
        r'INTO\s+[`\'"]?(\w+)',
        r'UPDATE\s+[`\'"]?(\w+)',
        r'TABLE\s+[`\'"]?(\w+)',
    ]:
        m = re.search(pat, sql_u)
        if m:
            return sql[m.start(1): m.start(1) + len(m.group(1))].strip("`'\"").lower()
    return ""


def _collect_php_files(project_root: str) -> list[str]:
    """Collect all .php files from the project, excluding vendor/ and test dirs."""
    root  = Path(project_root)
    files = []
    for php in root.rglob("*.php"):
        if not any(part in _SKIP_DIRS for part in php.parts):
            files.append(str(php))
    return files


def _attach_to_code_map(code_map: CodeMap, paths: list[dict[str, Any]]) -> None:
    """Attach execution paths to the CodeMap, with compatibility shim."""
    if hasattr(code_map, "execution_paths"):
        code_map.execution_paths = paths
    else:
        code_map.globals = [
            g for g in (code_map.globals or [])
            if g.get("__type") != "execution_paths"
        ]
        code_map.globals.append({"__type": "execution_paths", "data": paths})


def _print_summary(paths: list[dict[str, Any]]) -> None:
    total_branches = sum(len(p.get("branches", []))        for p in paths)
    total_flows    = sum(len(p.get("data_flows", []))       for p in paths)
    guarded        = sum(1 for p in paths if p.get("auth_guard"))
    ctrl_methods   = sum(len(p.get("controller_flows", [])) for p in paths)
    by_type: dict[str, int] = {}
    for p in paths:
        t = p.get("type", "raw_php")
        by_type[t] = by_type.get(t, 0) + 1

    width = 54
    print(f"\n  {'=' * width}")
    print(f"  Stage 1.5 — Execution Path Analysis")
    print(f"  {'=' * width}")
    print(f"  Files analysed    : {len(paths)}")
    print(f"  By type           : "
          + ", ".join(f"{t}={c}" for t, c in sorted(by_type.items())))
    print(f"  Branches found    : {total_branches}")
    print(f"  Data flows        : {total_flows}")
    print(f"  Auth-guarded      : {guarded} file(s)")
    print(f"  Controller methods: {ctrl_methods}")
    print(f"  {'=' * width}")
    for p in paths[:10]:
        hp = p.get("happy_path", [])
        ag = p.get("auth_guard")
        print(f"  {p['file']} [{p.get('type', '?')}]")
        if ag:
            key = ag.get("key") or (ag.get("middleware") or ["?"])[0]
            print(f"    🔒 Auth: {key}")
        for step in hp[:3]:
            print(f"    → {step[:80]}")
    if len(paths) > 10:
        print(f"  ... and {len(paths) - 10} more")
    print(f"  {'=' * width}\n")
