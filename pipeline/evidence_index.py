"""
pipeline/evidence_index.py — Per-Feature Evidence Index

Builds a lookup table that maps every domain-model feature name to the
concrete code artefacts that support it, drawn entirely from data already
collected by earlier pipeline stages (Stage 1, 1.5, 2).

Evidence types
--------------
  routes          HTTP route registrations (method + path + file)
  controllers     OOP controller class FQNs
  sql_queries     SQL operations (operation, table, file)
  form_fields     HTML <form> fields (file, field names)
  execution_paths Static-analysis entry paths (file, auth guard, key branch)

Join strategy
-------------
  SQL queries  → joined on feature['tables'] (most reliable — explicit table list)
  Everything   → fallback join on feature['pages'] basename set
  Both joins are case-insensitive.

Usage
-----
    from pipeline.evidence_index import build_evidence_index, format_evidence_block

    ev_idx = build_evidence_index(ctx, ctx.domain_model)
    block  = format_evidence_block(ev_idx.get("Campaign Management", {}))
    # block is a short Markdown string ready to embed in an LLM prompt

Token budget
------------
Each formatted block is ~6–12 lines (capped per category).
For a 25-feature project the total overhead per agent prompt is ~200 lines,
well within the 8 k-token budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ── Per-category caps — keep prompt overhead bounded ──────────────────────────
_MAX_ROUTES      = 5
_MAX_CONTROLLERS = 3
_MAX_SQL         = 6   # deduplicated by (operation, table)
_MAX_FORM_FILES  = 3
_MAX_FIELDS      = 8   # field names per form file
_MAX_EXEC_PATHS  = 3


# ─── Public API ───────────────────────────────────────────────────────────────

def build_evidence_index(ctx: Any, domain: Any) -> dict[str, dict]:
    """
    Build and return a dict mapping every feature name to its evidence items.

    Parameters
    ----------
    ctx    : PipelineContext — used for ctx.code_map
    domain : DomainModel    — used for domain.features (name, pages, tables)

    Returns
    -------
    dict[feature_name: str, evidence: dict]

    Each evidence dict has keys:
        routes, controllers, sql, form_fields, execution_paths
    All values are lists; empty list = no evidence found for that category.
    """
    cm = getattr(ctx, "code_map", None)
    if cm is None or domain is None:
        return {}

    routes_all     = cm.routes          or []
    controllers_all = cm.controllers    or []
    sql_all        = cm.sql_queries     or []
    forms_all      = cm.form_fields     or []
    eps_all        = cm.execution_paths or []

    result: dict[str, dict] = {}

    for feat in domain.features:
        feat_name = feat.get("name", "")

        # Page basenames (lower-case) for file-based joins
        pages_lower: set[str] = {
            Path(p).name.lower()
            for p in _to_strs(feat.get("pages", []))
            if p
        }

        # Table names (lower-case) for SQL join
        tables_lower: set[str] = {
            t.lower()
            for t in _to_strs(feat.get("tables", []))
            if t and t.upper() not in ("", "UNKNOWN")
        }

        result[feat_name] = {
            "routes":          _match_routes(routes_all,      pages_lower),
            "controllers":     _match_controllers(controllers_all, pages_lower),
            "sql":             _match_sql(sql_all, tables_lower, pages_lower),
            "form_fields":     _match_forms(forms_all,        pages_lower),
            "execution_paths": _match_eps(eps_all,            pages_lower),
        }

    return result


def format_evidence_block(ev: dict) -> str:
    """
    Format a single feature's evidence dict as a concise Markdown block
    suitable for embedding in an LLM prompt scaffold.

    Returns an empty string when no evidence was found for the feature.
    """
    lines: list[str] = []

    # ── Routes ────────────────────────────────────────────────────────────────
    for r in ev.get("routes", []):
        method = r.get("method", "?")
        path   = r.get("path",   "?")
        ffile  = Path(r.get("file", "?")).name
        lines.append(f"  - Route: `{method} {path}` [{ffile}]")

    # ── Controllers ───────────────────────────────────────────────────────────
    ctrls = ev.get("controllers", [])
    if ctrls:
        lines.append(f"  - Controller: {', '.join(f'`{c}`' for c in ctrls)}")

    # ── SQL queries ───────────────────────────────────────────────────────────
    for q in ev.get("sql", []):
        op    = q.get("operation", "?")
        table = q.get("table",     "?")
        qfile = Path(q.get("file", "?")).name
        lines.append(f"  - SQL: `{op} {table}` [{qfile}]")

    # ── Form fields ───────────────────────────────────────────────────────────
    for ff in ev.get("form_fields", []):
        fname  = Path(ff.get("file", "?")).name
        action = ff.get("action", "")
        method = ff.get("method", "")
        fields = [
            f.get("name", "")
            for f in ff.get("fields", [])
            if f.get("name") and f.get("type") not in ("hidden",)
        ][:_MAX_FIELDS]
        visible = [f for f in fields if f]
        # include hidden fields too if no visible ones found
        if not visible:
            visible = [
                f.get("name", "")
                for f in ff.get("fields", [])
                if f.get("name")
            ][:_MAX_FIELDS]
        if visible:
            action_str = f" → {action}" if action else ""
            lines.append(
                f"  - Form `{fname}` ({method}{action_str}): "
                f"{', '.join(f'`{n}`' for n in visible)}"
            )

    # ── Execution paths ───────────────────────────────────────────────────────
    for ep in ev.get("execution_paths", []):
        efile = Path(ep.get("file", "?")).name
        auth  = ep.get("auth_guard") or {}
        parts = [f"`{efile}`"]

        if auth and isinstance(auth, dict):
            key = auth.get("key", "")
            redir = auth.get("redirect", "")
            if key:
                parts.append(f"auth[`{key}`]")
            if redir:
                parts.append(f"→ redirect `{redir}`")

        # Show first branch condition if present
        branches = ep.get("branches") or []
        if branches and isinstance(branches[0], dict):
            cond = branches[0].get("condition", "")[:60]
            if cond:
                parts.append(f"if ({cond})")

        lines.append(f"  - ExecPath: {' | '.join(parts)}")

    if not lines:
        return ""

    return "**Evidence:**\n" + "\n".join(lines)


# ─── Internal matchers ────────────────────────────────────────────────────────

def _match_routes(routes: list[dict], pages: set[str]) -> list[dict]:
    """Filter routes whose source file matches a feature page, skip GROUP entries."""
    if not pages:
        return []
    out = []
    for r in routes:
        method = r.get("method", "")
        if method in ("GROUP",):
            continue
        if Path(r.get("file", "")).name.lower() in pages:
            out.append(r)
            if len(out) >= _MAX_ROUTES:
                break
    return out


def _match_controllers(controllers: list[dict], pages: set[str]) -> list[str]:
    """Return FQNs of controllers whose file matches a feature page."""
    if not pages:
        return []
    out = []
    for c in controllers:
        if Path(c.get("file", "")).name.lower() in pages:
            fqn = c.get("fqn") or c.get("name", "")
            if fqn:
                out.append(fqn)
            if len(out) >= _MAX_CONTROLLERS:
                break
    return out


def _match_sql(
    sql_all: list[dict],
    tables: set[str],
    pages: set[str],
) -> list[dict]:
    """
    Deduplicated SQL matches.

    Priority 1: join on table name (feature['tables'] — most reliable).
    Priority 2: join on file basename if no table matches found.
    """
    seen: set[tuple[str, str]] = set()
    out:  list[dict]           = []

    def _add(q: dict) -> bool:
        key = (q.get("operation", ""), q.get("table", ""))
        if key not in seen:
            seen.add(key)
            out.append(q)
        return len(out) >= _MAX_SQL

    # Pass 1 — table-based join
    if tables:
        for q in sql_all:
            tbl = q.get("table", "").lower()
            if tbl in tables and tbl not in ("", "unknown"):
                if _add(q):
                    return out

    # Pass 2 — file-based join (fallback)
    if not out and pages:
        for q in sql_all:
            if Path(q.get("file", "")).name.lower() in pages:
                if _add(q):
                    return out

    return out


def _match_forms(forms: list[dict], pages: set[str]) -> list[dict]:
    """Filter form_fields entries whose file matches a feature page."""
    if not pages:
        return []
    out = []
    for ff in forms:
        if Path(ff.get("file", "")).name.lower() in pages:
            out.append(ff)
            if len(out) >= _MAX_FORM_FILES:
                break
    return out


def _match_eps(eps: list[dict], pages: set[str]) -> list[dict]:
    """Filter execution_paths entries whose file matches a feature page."""
    if not pages:
        return []
    out = []
    for ep in eps:
        if Path(ep.get("file", "")).name.lower() in pages:
            out.append(ep)
            if len(out) >= _MAX_EXEC_PATHS:
                break
    return out


# ─── Utility ──────────────────────────────────────────────────────────────────

def _to_strs(items: list) -> list[str]:
    """Coerce a mixed list (str | dict) to list[str], same logic as stage5."""
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            val = next((v for v in item.values() if isinstance(v, str)), None)
            result.append(val if val is not None else str(item))
        else:
            result.append(str(item))
    return result
