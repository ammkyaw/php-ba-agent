"""
pipeline/stage29_invariants.py — Business Rule / Invariant Detection (Stage 2.9)

Extracts business rules from four fully-static sources — no LLM:

  Source 1 — DB schema (table_columns)
      NOT NULL columns      → VALIDATION  "X is required"
      Unique columns        → REFERENTIAL "X must be unique"

  Source 2 — Execution-path guard clauses (execution_paths.branches)
      Branch condition that terminates / redirects on violation
      → parse condition string with regex templates

  Source 3 — PHP source-file scanning
      Targeted regex patterns run over the actual PHP files identified
      by execution_paths.  Finds patterns that Stage 1.5 may have
      partially captured:  strlen, filter_var, preg_match, numeric
      thresholds, status comparisons, role/permission guards.

  Source 4 — SQL query constraints
      WHERE clauses referencing status / deleted / active columns
      give evidence for STATE_TRANSITION rules.

Rule categories
---------------
  VALIDATION       — format, length, required, enum
  AUTHORIZATION    — role, admin, permission, ownership
  STATE_TRANSITION — status checks, lifecycle ordering
  BUSINESS_LIMIT   — numeric thresholds / caps
  TEMPORAL         — date / time constraints
  REFERENTIAL      — uniqueness, FK existence

Output
------
  InvariantCollection → 2.9_invariants/rule_catalog.json
  ctx.invariants set on PipelineContext

Consumed by
-----------
  Stage 3  — embedded as chunk_type="business_rule" in ChromaDB
  Stage 4  — injected as GROUNDING in _build_user_prompt
  Stage 5  — BRD "Business Rules" section (future)
  Stage 6  — QA cross-check against AC conditions (future)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from context import BusinessRule, InvariantCollection, PipelineContext

# ── Configuration ─────────────────────────────────────────────────────────────

# DB columns that are NOT NULL for structural/ORM reasons, not business rules
_SKIP_COLUMNS = {
    "id", "created_at", "updated_at", "deleted_at", "deleted",
    "created_by", "modified_by", "modified_user_id", "created_user_id",
    "date_entered", "date_modified",
}

# Branch actions that indicate the if-block is a guard clause (reject path)
_GUARD_ACTIONS = {"terminate", "redirect", "die", "exit", "error"}

# Variables that are obviously loop/infrastructure/routing, not domain entities
_NOISE_VARS = re.compile(
    r'^(?:i|j|k|n|x|y|z|idx|count|len|size|num|index|offset|limit|page|'
    r'result|ret|err|error|errno|flag|tmp|temp|buf|buffer|row|rows|'
    r'key|val|item|elem|node|entry|obj|arr|data|line|char|byte|'
    r'view|action|module|subpanel|type|mode|format|method|op|task|'   # routing/controller vars
    r'return_module|return_action|return_id|current_module|'
    r'run|loop|step|cur|pos|ptr|ref|'
    r'_\w+)$'
)

# String values that indicate routing/controller context, not domain state
_ROUTING_VALUES = {
    "save", "edit", "create", "delete", "view", "detail", "list", "index",
    "search", "update", "show", "new", "get", "post", "put", "patch",
    "module", "default", "upload", "download", "export", "import",
    "ajax", "async", "popup", "inline", "banner", "widget",
}

# Noise condition patterns — definitely not business rules
_NOISE_CONDITION_RE = re.compile(
    r'^(?:'
    r'isset\s*\(|'                            # isset() — null checks
    r'empty\s*\(|'                            # empty()
    r'is_null\s*\(|'                          # is_null()
    r'!defined\s*\(|'                         # !defined('ENTRY_POINT')
    r'\$i\s*[<>]|'                            # loop counter
    r'\$j\s*[<>]|'
    r'count\s*\(\s*\$|'                       # count($arr) > 0
    r'sizeof\s*\(\s*\$|'
    r'php_?version|'                          # PHP version checks
    r'version_compare'
    r')',
    re.IGNORECASE,
)

# Directories to skip when reading PHP files
_SKIP_DIRS = {
    "vendor", "node_modules", ".git", "cache", "logs", "storage",
    "tests", "test", "spec", "stubs", "fixtures",
}

# Maximum rules to extract per source (avoid bloat on large codebases)
_MAX_SCHEMA_RULES  = 300
_MAX_BRANCH_RULES  = 400
_MAX_SOURCE_RULES  = 400


# ── Helpers ───────────────────────────────────────────────────────────────────

def _humanize(name: str) -> str:
    """Convert snake_case / camelCase to Title Case words."""
    # Split camelCase
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    # Replace _ with space
    name = name.replace("_", " ").replace("-", " ")
    return " ".join(w.capitalize() for w in name.split() if w)


def _extract_module_name(filepath: str) -> str:
    """Return the module directory name (SugarCRM style) or ''."""
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part.lower() == "modules" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def _assign_context(filepath: str, ac_file_map: dict[str, str]) -> str:
    """Resolve a file path to a bounded-context name."""
    ctx_name = ac_file_map.get(filepath, "")
    if not ctx_name:
        ctx_name = ac_file_map.get(Path(filepath).name.lower(), "")
    if not ctx_name:
        ctx_name = _extract_module_name(filepath)
    if not ctx_name:
        # Use first meaningful path segment after common prefixes
        parts = [p for p in Path(filepath).parts
                 if p.lower() not in (".", "..", "/", "modules", "include",
                                      "lib", "src", "app", "core")]
        ctx_name = parts[0] if parts else "General"
    return ctx_name


def _file_tables(filepath: str, file_to_tables: dict[str, set[str]]) -> list[str]:
    return sorted(file_to_tables.get(filepath, set()))


# ── Description templates (fully static) ─────────────────────────────────────

def _describe_condition(cond: str, entity: str, category: str) -> str | None:
    """
    Convert a raw PHP condition string into a human-readable business rule.
    Returns None if the condition can't be safely interpreted.
    """
    c = cond.strip()

    # ── VALIDATION: strlen($var) < N  →  "X must be at least N characters"
    m = re.search(r'strlen\s*\(\s*\$\w+\s*\)\s*(<)\s*(\d+)', c)
    if m:
        n = int(m.group(2))
        return f"{entity} must be at least {n} characters"

    m = re.search(r'strlen\s*\(\s*\$\w+\s*\)\s*(<=)\s*(\d+)', c)
    if m:
        n = int(m.group(2))
        return f"{entity} must be at least {n + 1} characters" if n > 0 else f"{entity} must not be empty"

    m = re.search(r'strlen\s*\(\s*\$\w+\s*\)\s*(>)\s*(\d+)', c)
    if m:
        n = int(m.group(2))
        if n == 0:
            return f"{entity} must not be empty"
        return f"{entity} must not exceed {n} characters"

    m = re.search(r'strlen\s*\(\s*\$\w+\s*\)\s*(>=)\s*(\d+)', c)
    if m:
        n = int(m.group(2))
        return f"{entity} must not exceed {n - 1} characters" if n > 1 else f"{entity} must not be empty"

    # ── VALIDATION: filter_var($var, FILTER_VALIDATE_*)
    m = re.search(r'filter_var\s*\([^,]+,\s*FILTER_VALIDATE_(\w+)', c, re.IGNORECASE)
    if m:
        ftype = m.group(1).lower().replace("_", " ")
        return f"{entity} must be a valid {ftype}"

    # ── VALIDATION: preg_match(pattern, $var)
    m = re.search(r'preg_match\s*\(\s*[\'"]([^\'"]{3,})[\'"]', c)
    if m:
        pattern = m.group(1)
        if re.search(r'^\^?[\^a-zA-Z\-]+\$?$', pattern.strip("/")):
            return f"{entity} must contain only letters"
        if re.search(r'\\d|\[0-9\]', pattern):
            return f"{entity} must be numeric"
        if re.search(r'@.*\\.', pattern):
            return f"{entity} must be a valid email address"
        return f"{entity} must match required format"

    # ── VALIDATION: in_array($var, [...]) — enum check
    m = re.search(r'in_array\s*\(\s*\$\w+\s*,\s*(?:array\s*\(|\[)([^\])]+)', c)
    if m:
        raw_vals = re.findall(r"['\"](\w+)['\"]", m.group(1))
        if raw_vals:
            vals = ", ".join(f"'{v}'" for v in raw_vals[:6])
            return f"{entity} must be one of: {vals}"

    # ── BUSINESS_LIMIT: $var > N / $var >= N  →  "X must not exceed N"
    m = re.search(r'\$(\w+)\s*(>)\s*(\d+(?:\.\d+)?)\b', c)
    if m and not _NOISE_VARS.match(m.group(1)):
        return f"{entity} must not exceed {m.group(3)}"

    m = re.search(r'\$(\w+)\s*(>=)\s*(\d+(?:\.\d+)?)\b', c)
    if m and not _NOISE_VARS.match(m.group(1)):
        n = float(m.group(3))
        return f"{entity} must not exceed {n - 1:.0f}" if n > 0 else f"{entity} must not be negative"

    # ── BUSINESS_LIMIT: $var < N / $var <= N  →  "X must be at least N"
    m = re.search(r'\$(\w+)\s*(<)\s*(\d+(?:\.\d+)?)\b', c)
    if m and not _NOISE_VARS.match(m.group(1)):
        n = float(m.group(3))
        return f"{entity} must be at least {n:.0f}"

    m = re.search(r'\$(\w+)\s*(<=)\s*(\d+(?:\.\d+)?)\b', c)
    if m and not _NOISE_VARS.match(m.group(1)):
        return f"{entity} must be at least {m.group(3)}"

    # ── STATE_TRANSITION: $obj->status == 'value' or $var == 'value'
    m = re.search(r'\$\w+(?:->\w+)?\s*===?\s*[\'"](\w+)[\'"]', c)
    if m:
        val = m.group(1)
        # Require value ≥ 3 chars, non-numeric, non-routing
        if (len(val) >= 3
                and not val.isdigit()
                and val not in ("", "null", "true", "false", "1", "0")
                and val.lower() not in _ROUTING_VALUES):
            return f"{entity} must be in '{val}' state"

    m = re.search(r'\$\w+(?:->\w+)?\s*!==?\s*[\'"](\w+)[\'"]', c)
    if m:
        val = m.group(1)
        if (len(val) >= 3
                and not val.isdigit()
                and val not in ("", "null", "true", "false", "deleted")
                and val.lower() not in _ROUTING_VALUES):
            return f"{entity} must not be in '{val}' state"

    # ── AUTHORIZATION: role / permission checks
    if re.search(r'\bis_admin\b', c, re.IGNORECASE):
        return "Only administrators can perform this action"
    if re.search(r'\bis_superuser\b', c, re.IGNORECASE):
        return "Only superusers can perform this action"
    if re.search(r'\bhas_permission\b|\bcheck_permission\b|\bACLController\b', c, re.IGNORECASE):
        return f"User must have permission to access {entity}"
    if re.search(r'\$\w+(?:->(?:is_admin|is_superuser|admin|type))\b', c):
        prop = re.search(r'->(\w+)', c)
        if prop:
            return f"Only users with {_humanize(prop.group(1))} can perform this action"

    # ── TEMPORAL: date comparisons
    if re.search(r'strtotime|date\s*\(|mktime|time\s*\(\s*\)', c, re.IGNORECASE):
        if re.search(r'[<>]', c):
            if ">" in c:
                return f"{entity} must be in the future"
            return f"{entity} must be in the past"

    # Could not interpret — return None to skip
    return None


def _extract_entity(cond: str, filepath: str,
                    post_keys: set[str],
                    file_to_tables: dict[str, set[str]]) -> str:
    """
    Best-effort extraction of the domain entity/field referenced by the condition.
    Priority: object property → POST key → variable name → file stem.
    """
    # Object property: $obj->property pattern
    # If obj is 'this'/'self', just use the property name directly
    m = re.search(r'\$(\w+)->(\w+)', cond)
    if m:
        obj_var = m.group(1).lower()
        prop    = m.group(2)
        if obj_var in ("this", "self", "that"):
            return _humanize(prop)
        obj = _humanize(m.group(1))
        return f"{obj}.{_humanize(prop)}"

    # Variable that maps to a POST key
    m = re.search(r'\$(\w+)', cond)
    if m:
        var = m.group(1)
        # Skip single-char or very short temp vars ($s, $vv, $x, $ok)
        if len(var) <= 2:
            return _humanize(Path(filepath).stem)
        if var.lower() in post_keys:
            return _humanize(var)
        if not _NOISE_VARS.match(var):
            return _humanize(var)

    # Fall back to file stem
    return _humanize(Path(filepath).stem)


def _category_from_cond(cond: str) -> str:
    """Heuristic category classification of a raw condition string."""
    c = cond.lower()
    if re.search(r'strlen|preg_match|filter_var|in_array', c):
        return "VALIDATION"
    if re.search(r'is_admin|is_superuser|has_permission|acl|role|permission', c):
        return "AUTHORIZATION"
    if re.search(r'->status\b|->state\b|->stage\b', c):
        return "STATE_TRANSITION"
    if re.search(r'[<>]=?\s*\d', c):
        return "BUSINESS_LIMIT"
    if re.search(r'strtotime|date\s*\(|mktime|time\s*\(\)', c):
        return "TEMPORAL"
    if re.search(r'===?\s*[\'"]', c):
        return "STATE_TRANSITION"
    return "VALIDATION"


# ── Phase 1: DB Schema Rules ──────────────────────────────────────────────────

def _extract_schema_rules(
    cm: Any,
    ac_file_map: dict[str, str],
    file_to_tables: dict[str, set[str]],
) -> list[dict]:
    """
    Two formats are supported:

    Format A — flat TableColumnEntry (Prisma / TypeScript / Java parsers):
        cm.table_columns = [{"table": "users", "column": "email", "nullable": False, ...}, ...]
        Each dict is ONE column.

    Format B — grouped DbSchemaEntry (PHP migration parsers):
        cm.db_schema = [{"table": "users", "columns": [{"name": "email", ...}], ...}, ...]
        Each dict is a table containing a list of column dicts.

    NOT NULL  →  VALIDATION  "X is required"
    UNIQUE    →  REFERENTIAL "X must be unique"
    Confidence: 1.0 (schema-enforced)
    """
    raw: list[dict] = []
    seen: set[tuple] = set()  # (table, col) dedup

    def _emit_col(table: str, col_name: str, fpath: str,
                  nullable: bool, unique: bool, primary: bool) -> None:
        ctx_name = _assign_context(fpath, ac_file_map) if fpath else (table or "General")
        key = (table.lower(), col_name.lower())
        if key in seen or col_name.lower() in _SKIP_COLUMNS:
            return
        seen.add(key)
        if not nullable:
            raw.append({
                "category":        "VALIDATION",
                "description":     f"{_humanize(col_name)} is required",
                "raw_expression":  f"{table}.{col_name} NOT NULL",
                "entity":          f"{_humanize(table)}.{_humanize(col_name)}",
                "bounded_context": ctx_name,
                "source_files":    [fpath] if fpath else [],
                "confidence":      1.0,
                "tables":          [table],
            })
        if unique and not primary:   # primary key uniqueness is an impl detail
            raw.append({
                "category":        "REFERENTIAL",
                "description":     f"{_humanize(col_name)} must be unique",
                "raw_expression":  f"{table}.{col_name} UNIQUE",
                "entity":          f"{_humanize(table)}.{_humanize(col_name)}",
                "bounded_context": ctx_name,
                "source_files":    [fpath] if fpath else [],
                "confidence":      1.0,
                "tables":          [table],
            })

    # ── Format A: flat TableColumnEntry (Prisma / TS / Java) ──────────────────
    # Each entry represents a single column: {table, column, type, nullable, ...}
    for tc in (cm.table_columns or []):
        col_name = tc.get("column", "")
        if not col_name:
            continue   # not a flat entry — skip (grouped entries handled below)
        _emit_col(
            table    = tc.get("table", ""),
            col_name = col_name,
            fpath    = tc.get("file", ""),
            nullable = tc.get("nullable", True),
            unique   = bool(tc.get("unique")),
            primary  = bool(tc.get("primary")),
        )

    # ── Format B: grouped DbSchemaEntry (PHP / SQL migrations) ────────────────
    # Each entry represents a table with a "columns" list: {table, columns: [...]}
    for entry in (cm.db_schema or []):
        table = entry.get("table", "")
        fpath = entry.get("file", "")
        for col in entry.get("columns", []):
            col_name = col.get("name", "") or col.get("column", "")
            if not col_name:
                continue
            _emit_col(
                table    = table,
                col_name = col_name,
                fpath    = fpath,
                nullable = col.get("nullable", True),
                unique   = bool(col.get("unique")),
                primary  = bool(col.get("primary")),
            )

    return raw[:_MAX_SCHEMA_RULES]


# ── Phase 2: Execution-Path Guard Clause Rules ────────────────────────────────

def _is_guard_branch(branch: dict) -> bool:
    """True if the 'then' actions indicate a rejection/termination."""
    for act in branch.get("then", []):
        if isinstance(act, dict):
            action_type = act.get("action", "").lower()
            if any(g in action_type for g in _GUARD_ACTIONS):
                return True
        elif isinstance(act, str):
            if any(g in act.lower() for g in _GUARD_ACTIONS):
                return True
    return False


def _extract_branch_rules(
    cm: Any,
    ac_file_map: dict[str, str],
    file_to_tables: dict[str, set[str]],
    post_keys: set[str],
) -> list[dict]:
    """
    Scan execution_paths[*].branches for guard clauses and convert to rules.
    Confidence: 0.8
    """
    raw: list[dict] = []
    seen_desc: set[str] = set()

    for ep in (cm.execution_paths or []):
        fpath = ep.get("file", "")
        if not fpath:
            continue
        ctx_name  = _assign_context(fpath, ac_file_map)
        tables    = _file_tables(fpath, file_to_tables)

        for branch in ep.get("branches", []):
            cond = branch.get("condition", "").strip()
            if not cond or len(cond) < 4:
                continue
            if _NOISE_CONDITION_RE.match(cond):
                continue
            if not _is_guard_branch(branch):
                continue

            # Negated guard: if (!BUSINESS_CONDITION) { terminate }
            # The actual rule is the positive form
            effective_cond = cond
            if cond.startswith("!") and not cond.startswith("!isset") and not cond.startswith("!empty"):
                effective_cond = cond[1:].strip().lstrip("(").rstrip(")")

            category = _category_from_cond(effective_cond)
            entity   = _extract_entity(effective_cond, fpath, post_keys, file_to_tables)
            desc     = _describe_condition(effective_cond, entity, category)
            if not desc:
                continue

            dedup_key = desc.lower()
            if dedup_key in seen_desc:
                # Find existing rule and add source file
                for r in raw:
                    if r["description"].lower() == dedup_key and fpath not in r["source_files"]:
                        r["source_files"].append(fpath)
                continue
            seen_desc.add(dedup_key)

            raw.append({
                "category":        category,
                "description":     desc,
                "raw_expression":  cond[:120],
                "entity":          entity,
                "bounded_context": ctx_name,
                "source_files":    [fpath],
                "confidence":      0.8,
                "tables":          tables,
            })

            if len(raw) >= _MAX_BRANCH_RULES:
                break
        if len(raw) >= _MAX_BRANCH_RULES:
            break

    return raw


# ── Phase 3: PHP Source-File Scanning ────────────────────────────────────────

# High-value patterns to scan for in raw PHP source.
# Each: (compiled_regex, category, description_fn)
# description_fn(match, filepath, post_keys, file_to_tables) → str | None
_SOURCE_PATTERNS: list[tuple] = [

    # strlen($var) comparisons (may appear outside guard context)
    (
        re.compile(r'strlen\s*\(\s*\$(\w+)\s*\)\s*([<>]=?)\s*(\d+)', re.IGNORECASE),
        "VALIDATION",
        lambda m, fp, pk, ft: (
            _describe_condition(m.group(0), _humanize(m.group(1)), "VALIDATION")
        ),
    ),

    # filter_var($var, FILTER_VALIDATE_*)
    (
        re.compile(r'filter_var\s*\(\s*\$(\w+)\s*,\s*(FILTER_VALIDATE_\w+)', re.IGNORECASE),
        "VALIDATION",
        lambda m, fp, pk, ft: (
            f"{_humanize(m.group(1))} must be a valid "
            f"{m.group(2).replace('FILTER_VALIDATE_', '').lower().replace('_', ' ')}"
        ),
    ),

    # preg_match('/pattern/', $var)
    (
        re.compile(r'preg_match\s*\(\s*([\'"])([^\'"]{4,})\1\s*,\s*\$(\w+)', re.IGNORECASE),
        "VALIDATION",
        lambda m, fp, pk, ft: (
            _describe_condition(
                f"preg_match('{m.group(2)}', ${m.group(3)})",
                _humanize(m.group(3)), "VALIDATION"
            )
        ),
    ),

    # in_array($var, [...]) — enum constraints
    (
        re.compile(
            r'in_array\s*\(\s*\$(\w+)\s*,\s*(?:array\s*\(|\[)([^\])]{4,})',
            re.IGNORECASE,
        ),
        "VALIDATION",
        lambda m, fp, pk, ft: (
            _describe_condition(m.group(0), _humanize(m.group(1)), "VALIDATION")
        ),
    ),

    # is_admin() / is_superuser() guards
    (
        re.compile(r'\b(is_admin|is_superuser)\s*\(\s*\$?\w*\s*\)', re.IGNORECASE),
        "AUTHORIZATION",
        lambda m, fp, pk, ft: (
            "Only administrators can perform this action"
            if "admin" in m.group(1).lower()
            else "Only superusers can perform this action"
        ),
    ),

    # ACLController::checkAccess  / ACL::access
    (
        re.compile(r'ACLController\s*::\s*\w+|ACL\s*::\s*access', re.IGNORECASE),
        "AUTHORIZATION",
        lambda m, fp, pk, ft: "Access control check required",
    ),

    # $obj->status == / === 'value'  (state transition evidence)
    (
        re.compile(r'\$(\w+)->(\w*status\w*|state|stage|phase)\s*===?\s*[\'"](\w+)[\'"]',
                   re.IGNORECASE),
        "STATE_TRANSITION",
        lambda m, fp, pk, ft: (
            None if m.group(3).lower() in _ROUTING_VALUES else (
                f"{_humanize(m.group(2))} must be '{m.group(3)}'"
                if m.group(1).lower() in ("this", "self", "that")
                else f"{_humanize(m.group(1))} {_humanize(m.group(2))} must be '{m.group(3)}'"
            )
        ),
    ),

    # Numeric guard: if ($var > N) { ... die/error }  — handled in source context
    # We look for the pattern adjacent to a rejection keyword
    (
        re.compile(
            r'if\s*\(\s*\$(\w+)\s*([<>]=?)\s*(\d+(?:\.\d+)?)\s*\)'
            r'[^{]{0,40}(?:die|exit|throw|error|return\s+false)',
            re.IGNORECASE | re.DOTALL,
        ),
        "BUSINESS_LIMIT",
        lambda m, fp, pk, ft: (
            None if _NOISE_VARS.match(m.group(1)) else
            _describe_condition(
                f"${m.group(1)} {m.group(2)} {m.group(3)}",
                _humanize(m.group(1)), "BUSINESS_LIMIT"
            )
        ),
    ),
]


def _should_skip_file(fpath: str) -> bool:
    """Return True for files that are unlikely to contain business logic."""
    parts = set(Path(fpath).parts)
    if parts & _SKIP_DIRS:
        return True
    name = Path(fpath).name.lower()
    # Skip pure view/template files and framework internals
    if re.search(r'(view|tpl|template|migration|install|upgrade|repair|'
                 r'vardefs?|metadata|language|lang)\.',
                 name, re.IGNORECASE):
        return True
    return False


def _extract_source_rules(
    cm: Any,
    php_project_path: str,
    ac_file_map: dict[str, str],
    file_to_tables: dict[str, set[str]],
    post_keys: set[str],
) -> list[dict]:
    """
    Scan PHP source files for targeted patterns.  Files are taken from
    execution_paths (already identified as business-logic files by Stage 1.5).
    Confidence: 0.6
    """
    raw: list[dict] = []
    seen_desc: set[str] = set()

    # Collect unique file paths from execution_paths
    file_paths: list[str] = list({
        ep.get("file", "") for ep in (cm.execution_paths or [])
        if ep.get("file")
    })

    project_root = Path(php_project_path)

    for rel_path in file_paths:
        if _should_skip_file(rel_path):
            continue
        full_path = project_root / rel_path
        if not full_path.is_file():
            continue

        try:
            source = full_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        ctx_name = _assign_context(rel_path, ac_file_map)
        tables   = _file_tables(rel_path, file_to_tables)

        for pattern, category, desc_fn in _SOURCE_PATTERNS:
            for m in pattern.finditer(source):
                try:
                    desc = desc_fn(m, rel_path, post_keys, file_to_tables)
                except Exception:
                    continue
                if not desc:
                    continue

                dedup_key = desc.lower()
                if dedup_key in seen_desc:
                    for r in raw:
                        if r["description"].lower() == dedup_key and rel_path not in r["source_files"]:
                            r["source_files"].append(rel_path)
                    continue
                seen_desc.add(dedup_key)

                raw.append({
                    "category":        category,
                    "description":     desc,
                    "raw_expression":  source[m.start(): m.start() + 80].split("\n")[0].strip(),
                    "entity":          _humanize(
                        (m.group(1) if m.lastindex and m.lastindex >= 1 else "")
                        or Path(rel_path).stem
                    ),
                    "bounded_context": ctx_name,
                    "source_files":    [rel_path],
                    "confidence":      0.6,
                    "tables":          tables,
                })

                if len(raw) >= _MAX_SOURCE_RULES:
                    break
            if len(raw) >= _MAX_SOURCE_RULES:
                break

    return raw


# ── Phase 4: SQL WHERE constraint hints ───────────────────────────────────────

def _extract_sql_rules(
    cm: Any,
    ac_file_map: dict[str, str],
) -> list[dict]:
    """
    Mine sql_queries WHERE clauses for status / state / active patterns.
    These give evidence of STATE_TRANSITION rules without requiring PHP parsing.
    Confidence: 0.6
    """
    raw: list[dict] = []
    seen_desc: set[str] = set()

    STATUS_RE = re.compile(
        r"(?:WHERE|AND|OR)\s+\w+\s*=\s*['\"](\w+)['\"]", re.IGNORECASE
    )
    DELETED_RE = re.compile(r'\bdeleted\s*=\s*0\b', re.IGNORECASE)
    ACTIVE_RE  = re.compile(r'\bactive\s*=\s*[\'"]?1[\'"]?\b', re.IGNORECASE)

    for q in (cm.sql_queries or []):
        sql   = q.get("raw_sql", "") or q.get("sql", "") or ""
        table = q.get("table", "")
        fpath = q.get("file", "")
        ctx   = _assign_context(fpath, ac_file_map)

        if DELETED_RE.search(sql):
            desc = f"{_humanize(table)} records must not be deleted"
            if desc.lower() not in seen_desc:
                seen_desc.add(desc.lower())
                raw.append({
                    "category":        "STATE_TRANSITION",
                    "description":     desc,
                    "raw_expression":  "deleted = 0",
                    "entity":          _humanize(table),
                    "bounded_context": ctx,
                    "source_files":    [fpath],
                    "confidence":      0.6,
                    "tables":          [table] if table else [],
                })

        if ACTIVE_RE.search(sql):
            desc = f"{_humanize(table)} must be active"
            if desc.lower() not in seen_desc:
                seen_desc.add(desc.lower())
                raw.append({
                    "category":        "STATE_TRANSITION",
                    "description":     desc,
                    "raw_expression":  "active = 1",
                    "entity":          _humanize(table),
                    "bounded_context": ctx,
                    "source_files":    [fpath],
                    "confidence":      0.6,
                    "tables":          [table] if table else [],
                })

    return raw


# ── TypeScript / JavaScript invariant extractors ─────────────────────────────

# Auth guard patterns in TypeScript execution paths and source files
_TS_AUTH_PATTERNS = re.compile(
    r'(?:^|[(\s!])(?:session|user|auth|currentUser|getServerSession|requireAuth'
    r'|isAuthenticated|isLoggedIn|verifyToken|validateSession)',
    re.IGNORECASE,
)

# Zod schema rules: z.string().min(N)  z.number().max(N)  z.string().email() etc.
_ZOD_MIN    = re.compile(r'\.min\s*\(\s*(\d+)\s*\)')
_ZOD_MAX    = re.compile(r'\.max\s*\(\s*(\d+)\s*\)')
_ZOD_EMAIL  = re.compile(r'\.email\s*\(\s*\)')
_ZOD_URL    = re.compile(r'\.url\s*\(\s*\)')
_ZOD_ENUM   = re.compile(r'z\.enum\s*\(\s*\[([^\]]+)\]')
_ZOD_FIELD  = re.compile(
    r'(\w+)\s*:\s*z\.'             # fieldName: z.
    r'(string|number|boolean|date|array|object|enum|union|literal)',
    re.IGNORECASE,
)

# Role / permission guards
_TS_ROLE_EXACT = re.compile(
    r"""(?:role|userRole|permission)\s*[!=]=+\s*['"](\w+)['"]""",
    re.IGNORECASE,
)
_TS_HAS_ROLE  = re.compile(
    r"""hasRole\s*\(\s*['"](\w+)['"]\s*\)""",
    re.IGNORECASE,
)

# State / status comparisons
_TS_STATUS = re.compile(
    r"""(?:status|state|phase|stage)\s*[!=]=+\s*['"](\w+)['"]""",
    re.IGNORECASE,
)

# Numeric threshold guards
_TS_NUMERIC = re.compile(r"""(\w+)\s*([<>]=?)\s*(\d+(?:\.\d+)?)""")

# Variables that are infra/loop noise in TypeScript
_TS_NOISE_VARS = re.compile(
    r'^(?:i|j|k|n|x|y|z|idx|count|len|size|num|index|offset|limit|page|'
    r'result|ret|err|error|tmp|temp|res|req|ctx|next|event|e|cb|fn|'
    r'key|val|item|elem|node|entry|row|col|data|line|char|byte)$',
    re.IGNORECASE,
)


def _describe_condition_ts(cond: str, entity: str) -> tuple[str | None, str]:
    """
    Convert a TypeScript guard condition into (description, category).
    Returns (None, '') if the condition is not recognisable as a business rule.
    """
    c = cond.strip()
    cl = c.lower()

    # ── AUTHORIZATION: session / auth guards ─────────────────────────────────
    if re.search(r'!session|session\s*===?\s*null|!user\b|user\s*===?\s*null'
                 r'|!auth\b|!isAuthenticated|!currentUser', cl):
        return "User must be authenticated to perform this action", "AUTHORIZATION"

    if re.search(r'!userId\b|userId\s*===?\s*null|!req\.user\b', cl):
        return "User ID is required", "AUTHORIZATION"

    # Role / permission
    m = _TS_ROLE_EXACT.search(c)
    if m:
        role_val = m.group(1)
        if role_val.lower() not in _ROUTING_VALUES and len(role_val) >= 3:
            op = "!==" if "!" in c[:c.index(role_val)] else "==="
            if "!" in c[:c.index(role_val)]:
                return f"User must have '{role_val}' role", "AUTHORIZATION"
            return f"Only users with '{role_val}' role can perform this action", "AUTHORIZATION"

    m = _TS_HAS_ROLE.search(c)
    if m:
        return f"User must have '{m.group(1)}' role", "AUTHORIZATION"

    # ── STATE_TRANSITION: status / state comparisons ─────────────────────────
    m = _TS_STATUS.search(c)
    if m:
        val = m.group(1)
        if len(val) >= 3 and not val.isdigit() and val.lower() not in _ROUTING_VALUES:
            if "!" in c or "!==" in c:
                return f"{entity} must not be in '{val}' state", "STATE_TRANSITION"
            return f"{entity} must be in '{val}' state", "STATE_TRANSITION"

    # ── VALIDATION: type checks ───────────────────────────────────────────────
    if re.search(r"typeof\s+\w+\s*!==?\s*['\"]string['\"]", c):
        return f"{entity} must be a string", "VALIDATION"
    if re.search(r"typeof\s+\w+\s*!==?\s*['\"]number['\"]", c):
        return f"{entity} must be a number", "VALIDATION"
    if re.search(r"!?\w+\.trim\(\)\.length|\.length\s*===?\s*0", c):
        return f"{entity} must not be empty", "VALIDATION"

    # ── BUSINESS_LIMIT: numeric thresholds ────────────────────────────────────
    m = _TS_NUMERIC.search(c)
    if m:
        var, op, num = m.group(1), m.group(2), m.group(3)
        if not _TS_NOISE_VARS.match(var):
            n = float(num)
            if op == ">":
                return f"{_humanize(var)} must not exceed {num}", "BUSINESS_LIMIT"
            if op == ">=":
                return f"{_humanize(var)} must not exceed {int(n)-1}", "BUSINESS_LIMIT"
            if op == "<":
                return f"{_humanize(var)} must be at least {num}", "BUSINESS_LIMIT"
            if op == "<=":
                return f"{_humanize(var)} must be at least {int(n)+1}", "BUSINESS_LIMIT"

    return None, ""


def _extract_branch_rules_ts(
    cm: Any,
    ac_file_map: dict[str, str],
    file_to_tables: dict[str, set[str]],
) -> list[dict]:
    """
    TypeScript execution-path guard clause rules.
    Confidence: 0.75
    """
    raw: list[dict] = []
    seen_desc: set[str] = set()

    for ep in (cm.execution_paths or []):
        fpath = ep.get("file", "")
        if not fpath:
            continue
        ctx_name = _assign_context(fpath, ac_file_map)
        tables   = _file_tables(fpath, file_to_tables)
        entity   = _humanize(Path(fpath).stem)

        for branch in ep.get("branches", []):
            cond = branch.get("condition", "").strip()
            if not cond or len(cond) < 4:
                continue
            if not _is_guard_branch(branch):
                continue

            # Strip outer negation for positive-form parsing
            effective = cond[1:].strip("()") if cond.startswith("!") else cond

            desc, category = _describe_condition_ts(effective, entity)
            if not desc:
                continue

            key = desc.lower()
            if key in seen_desc:
                for r in raw:
                    if r["description"].lower() == key and fpath not in r["source_files"]:
                        r["source_files"].append(fpath)
                continue
            seen_desc.add(key)

            raw.append({
                "category":        category,
                "description":     desc,
                "raw_expression":  cond[:120],
                "entity":          entity,
                "bounded_context": ctx_name,
                "source_files":    [fpath],
                "confidence":      0.75,
                "tables":          tables,
            })

            if len(raw) >= _MAX_BRANCH_RULES:
                break
        if len(raw) >= _MAX_BRANCH_RULES:
            break

    return raw


def _extract_source_rules_ts(
    cm: Any,
    project_path: str,
    ac_file_map: dict[str, str],
    file_to_tables: dict[str, set[str]],
) -> list[dict]:
    """
    Scan TypeScript/JavaScript source files for business-rule patterns:
      - Zod schema constraints (min/max/email/url/enum)
      - Auth guards (getServerSession, requireAuth, etc.)
      - Role checks
      - State comparisons
    Confidence: 0.65
    """
    raw: list[dict] = []
    seen_desc: set[str] = set()

    # Collect unique files from execution_paths
    file_paths: list[str] = list({
        ep.get("file", "") for ep in (cm.execution_paths or []) if ep.get("file")
    })

    project_root = Path(project_path)

    for rel_path in file_paths:
        if _should_skip_file(rel_path):
            continue
        full_path = project_root / rel_path
        if not full_path.is_file():
            continue
        try:
            source = full_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        ctx_name = _assign_context(rel_path, ac_file_map)
        tables   = _file_tables(rel_path, file_to_tables)

        def _add(desc: str, category: str, raw_expr: str, entity: str) -> None:
            key = desc.lower()
            if key in seen_desc:
                for r in raw:
                    if r["description"].lower() == key and rel_path not in r["source_files"]:
                        r["source_files"].append(rel_path)
                return
            seen_desc.add(key)
            raw.append({
                "category":        category,
                "description":     desc,
                "raw_expression":  raw_expr[:80],
                "entity":          entity,
                "bounded_context": ctx_name,
                "source_files":    [rel_path],
                "confidence":      0.65,
                "tables":          tables,
            })

        # ── Zod field rules ──────────────────────────────────────────────────
        for fm in _ZOD_FIELD.finditer(source):
            field_name = fm.group(1)
            entity     = _humanize(field_name)
            field_src  = source[fm.start():fm.start() + 200].split("\n")[0]

            # min / max
            for mm in _ZOD_MIN.finditer(field_src):
                _add(f"{entity} must be at least {mm.group(1)} characters",
                     "VALIDATION", field_src.strip(), entity)
            for mm in _ZOD_MAX.finditer(field_src):
                _add(f"{entity} must not exceed {mm.group(1)} characters",
                     "VALIDATION", field_src.strip(), entity)
            if _ZOD_EMAIL.search(field_src):
                _add(f"{entity} must be a valid email address",
                     "VALIDATION", field_src.strip(), entity)
            if _ZOD_URL.search(field_src):
                _add(f"{entity} must be a valid URL",
                     "VALIDATION", field_src.strip(), entity)

        # ── Zod enum (anywhere in file) ──────────────────────────────────────
        for em in _ZOD_ENUM.finditer(source):
            vals = re.findall(r"""['"]([\w\-]+)['"]""", em.group(1))
            if vals:
                parent_line = source[max(0, em.start()-80):em.start()].rsplit("\n", 1)[-1]
                field_m = re.search(r'(\w+)\s*:\s*$', parent_line.strip())
                entity  = _humanize(field_m.group(1)) if field_m else _humanize(Path(rel_path).stem)
                enum_vals = ", ".join(f"'{v}'" for v in vals[:6])
                _add(f"{entity} must be one of: {enum_vals}",
                     "VALIDATION", em.group(0)[:80], entity)

        # ── Auth guards ───────────────────────────────────────────────────────
        if re.search(r'getServerSession|requireAuth|withAuth|protectedRoute'
                     r'|auth\(\)\.protect|checkAuth', source, re.IGNORECASE):
            stem = _humanize(Path(rel_path).stem)
            _add(f"User must be authenticated to access {stem}",
                 "AUTHORIZATION",
                 "getServerSession / requireAuth / withAuth detected",
                 stem)

        # ── Role guards in source ─────────────────────────────────────────────
        for rm in _TS_ROLE_EXACT.finditer(source):
            role_val = rm.group(1)
            if len(role_val) >= 3 and role_val.lower() not in _ROUTING_VALUES:
                line = source[max(0, rm.start()-20):rm.end()+20].split("\n")[0].strip()
                entity = _humanize(Path(rel_path).stem)
                _add(f"User must have '{role_val}' role",
                     "AUTHORIZATION", line[:80], entity)

        # ── State comparisons in source ───────────────────────────────────────
        for sm in _TS_STATUS.finditer(source):
            val = sm.group(1)
            if len(val) >= 3 and not val.isdigit() and val.lower() not in _ROUTING_VALUES:
                context_src = source[max(0, sm.start()-10):sm.end()+10].split("\n")[0].strip()
                field_ctx   = re.search(r'(\w+)\.(?:status|state)', context_src)
                entity      = _humanize(field_ctx.group(1)) if field_ctx else _humanize(Path(rel_path).stem)
                _add(f"{entity} must be in '{val}' state",
                     "STATE_TRANSITION", context_src[:80], entity)

        if len(raw) >= _MAX_SOURCE_RULES:
            break

    return raw


# ── Assembly ──────────────────────────────────────────────────────────────────

def _merge_all(
    schema_rules: list[dict],
    branch_rules: list[dict],
    source_rules: list[dict],
    sql_rules:    list[dict],
) -> list[BusinessRule]:
    """Merge all raw dicts into BusinessRule objects, deduplicating by description."""
    seen: dict[str, BusinessRule] = {}   # lower(description) → rule
    seq = 1

    for raw in [schema_rules, branch_rules, source_rules, sql_rules]:
        for r in raw:
            key = r["description"].lower()
            if key in seen:
                existing = seen[key]
                for sf in r["source_files"]:
                    if sf not in existing.source_files:
                        existing.source_files.append(sf)
                # Keep highest confidence
                if r["confidence"] > existing.confidence:
                    existing.confidence = r["confidence"]
                continue

            rule = BusinessRule(
                rule_id         = f"rule_{seq:04d}",
                category        = r["category"],
                description     = r["description"],
                raw_expression  = r["raw_expression"],
                entity          = r["entity"],
                bounded_context = r["bounded_context"],
                source_files    = r["source_files"],
                confidence      = r["confidence"],
                tables          = r["tables"],
            )
            seen[key] = rule
            seq += 1

    return list(seen.values())


def _to_plain_english(rule: "BusinessRule") -> str:
    """
    Convert a BusinessRule into a BA-ready plain-English sentence.

    The `description` field is already human-readable but entity-centric
    (e.g. "password must be at least 8 characters").  This function produces
    a complete sentence that a business analyst can paste directly into a BRD:
      "User passwords must be at least 8 characters long."

    Rules:
      - Start with the entity/table name in Title Case
      - Use imperative mood for VALIDATION / REFERENTIAL / AUTHORIZATION
      - Use passive-present for STATE_TRANSITION ("Order status must progress
        from X to Y")
      - Cap at 120 characters; fall back to description if pattern fails.
    """
    desc  = rule.description.strip().rstrip(".")
    ent   = rule.entity or ""
    cat   = rule.category

    # Extract a human entity name from "Table.column" or "variable_name"
    if "." in ent:
        table, col = ent.split(".", 1)
        entity_label = f"{_humanize(table)} {_humanize(col)}"
    elif ent:
        entity_label = _humanize(ent)
    else:
        entity_label = ""

    # If description already starts with the entity, capitalise and punctuate.
    if desc and entity_label and desc.lower().startswith(entity_label.lower()):
        return desc[0].upper() + desc[1:] + "."

    # Prefix with entity label for VALIDATION / BUSINESS_LIMIT / REFERENTIAL
    if entity_label and cat in ("VALIDATION", "BUSINESS_LIMIT", "REFERENTIAL"):
        # Avoid double noun: "User Username must be unique" → skip if desc starts
        # with a word already present in entity_label
        first_word = desc.split()[0].lower() if desc else ""
        label_words = {w.lower() for w in entity_label.split()}
        if first_word not in label_words:
            sentence = f"{entity_label}: {desc[0].lower() + desc[1:]}."
        else:
            sentence = desc[0].upper() + desc[1:] + "."
        return sentence[:150]

    # AUTHORIZATION: add "Only authorised users …" framing if not already there
    if cat == "AUTHORIZATION" and "only" not in desc.lower():
        return f"Only authorised users may: {desc[0].lower() + desc[1:]}."[:150]

    # STATE_TRANSITION: "The {entity} status must …"
    if cat == "STATE_TRANSITION" and entity_label and "status" not in desc.lower():
        return f"The {entity_label} must satisfy: {desc[0].lower() + desc[1:]}."[:150]

    # Default: capitalise + punctuate
    return (desc[0].upper() + desc[1:] + ".") if desc else ""


def _build_collection(rules: list[BusinessRule]) -> InvariantCollection:
    by_category: dict[str, list[str]] = defaultdict(list)
    by_context:  dict[str, list[str]] = defaultdict(list)

    for r in rules:
        by_category[r.category].append(r.rule_id)
        ctx = r.bounded_context or "General"
        by_context[ctx].append(r.rule_id)

    return InvariantCollection(
        rules       = rules,
        total       = len(rules),
        by_category = dict(by_category),
        by_context  = dict(by_context),
    )


# ── Main entry point ───────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Stage 2.9 — Business Rule / Invariant Detection."""
    cm = getattr(ctx, "code_map", None)
    if cm is None:
        print("  [stage29] ⚠️  No code_map — skipping invariant detection.")
        ctx.invariants = InvariantCollection()
        return

    # ── Build helper indices ──────────────────────────────────────────────────
    # file → cluster name (from Stage 2.8)
    ac_file_map: dict[str, str] = {}
    ac = getattr(ctx, "action_clusters", None)
    if ac and ac.clusters:
        for cluster in ac.clusters:
            for fpath in cluster.files:
                ac_file_map[fpath] = cluster.name
                ac_file_map[Path(fpath).name.lower()] = cluster.name

    # file → set of tables (from sql_queries)
    file_to_tables: dict[str, set[str]] = defaultdict(set)
    for q in (cm.sql_queries or []):
        f = q.get("file", "")
        t = q.get("table", "")
        if f and t and t.upper() not in ("", "UNKNOWN"):
            file_to_tables[f].add(t.lower())

    # Detect language for dispatch
    lang = getattr(cm.language, "value", str(cm.language)).lower()
    is_ts  = lang in ("typescript", "javascript")
    is_php = lang == "php"

    # known POST / request field names (PHP: superglobals; TS: input_params)
    post_keys: set[str] = set()
    for s in (cm.superglobals or []):
        if s.get("var") in ("$_POST", "$_GET", "$_REQUEST") and s.get("key"):
            post_keys.add(s["key"].lower())
    for ip in (cm.input_params or []):
        name = ip.get("name", "") or ip.get("key", "")
        if name:
            post_keys.add(name.lower())

    # ── Phase 1: DB schema (language-neutral) ─────────────────────────────────
    print("  [stage29] Phase 1: Extracting schema constraints ...")
    schema_rules = _extract_schema_rules(cm, ac_file_map, file_to_tables)
    print(f"  [stage29]   → {len(schema_rules)} schema rule(s)")

    # ── Phase 2: Execution-path guard clauses (language-dispatched) ───────────
    print("  [stage29] Phase 2: Scanning execution-path guard clauses ...")
    if is_ts:
        branch_rules = _extract_branch_rules_ts(cm, ac_file_map, file_to_tables)
    else:
        branch_rules = _extract_branch_rules(cm, ac_file_map, file_to_tables, post_keys)
    print(f"  [stage29]   → {len(branch_rules)} guard-clause rule(s)")

    # ── Phase 3: Source-file scanning (language-dispatched) ───────────────────
    if is_ts:
        print("  [stage29] Phase 3: Scanning TypeScript source files ...")
        source_rules = _extract_source_rules_ts(
            cm, ctx.project_path, ac_file_map, file_to_tables
        )
    else:
        print("  [stage29] Phase 3: Scanning PHP source files ...")
        php_root = ctx.project_path
        source_rules = _extract_source_rules(
            cm, php_root, ac_file_map, file_to_tables, post_keys
        )
    print(f"  [stage29]   → {len(source_rules)} source-scan rule(s)")

    # ── Phase 4: SQL WHERE hints (language-neutral, field name fixed) ──────────
    print("  [stage29] Phase 4: Mining SQL WHERE constraints ...")
    sql_rules = _extract_sql_rules(cm, ac_file_map)
    print(f"  [stage29]   → {len(sql_rules)} SQL-derived rule(s)")

    # ── Merge and build collection ────────────────────────────────────────────
    rules      = _merge_all(schema_rules, branch_rules, source_rules, sql_rules)
    # Populate plain_english on every rule (BA-ready phrasing for BRD injection)
    for rule in rules:
        rule.plain_english = _to_plain_english(rule)
    collection = _build_collection(rules)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  [stage29] Invariant Detection complete — {collection.total} rules:")
    for cat, ids in sorted(collection.by_category.items()):
        print(f"    {cat:<20} {len(ids):>4} rules")
    print()

    # Spot-check: print a few rules per category
    shown: dict[str, int] = defaultdict(int)
    for r in rules:
        if shown[r.category] < 2:
            conf = f"{r.confidence:.1f}"
            print(f"    [{r.category}] {r.description}  (conf={conf})")
            shown[r.category] += 1
    print()

    # ── Persist ───────────────────────────────────────────────────────────────
    ctx.invariants = collection
    try:
        import dataclasses
        out_path = ctx.output_path("rule_catalog.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(collection), fh, indent=2, ensure_ascii=False)
        print(f"  [stage29] Saved → {out_path}")
    except Exception as exc:
        print(f"  [stage29] ⚠️  Could not save rule_catalog.json: {exc}")
