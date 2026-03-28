"""
pipeline/evidence_index.py — Per-Feature Evidence Index + Confidence Scoring

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

Confidence scoring
------------------
Each evidence category contributes equally (+0.2) to a 0.0–1.0 score:
  route found          +0.2
  controller found     +0.2
  SQL found            +0.2
  form found           +0.2
  execution path found +0.2

Grade bands:
  🟢 HIGH    ≥ 0.8  (4–5 evidence types)
  🟡 MEDIUM  ≥ 0.4  (2–3 evidence types)
  🔴 LOW     < 0.4  (0–1 evidence type)

Usage
-----
    from pipeline.evidence_index import (
        build_evidence_index,
        format_evidence_block,
        compute_confidence,
        build_confidence_report,
    )

    ev_idx = build_evidence_index(ctx, ctx.domain_model)
    block  = format_evidence_block(ev_idx.get("Campaign Management", {}))
    report = build_confidence_report(ev_idx)
    # report is list[dict] sorted by score ascending (weakest first)

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

_EVIDENCE_PROFILES: dict[str, dict[str, int]] = {
    "default": {
        "routes": _MAX_ROUTES,
        "controllers": _MAX_CONTROLLERS,
        "sql": _MAX_SQL,
        "form_files": _MAX_FORM_FILES,
        "fields": _MAX_FIELDS,
        "exec_paths": _MAX_EXEC_PATHS,
    },
    # BA writers care more about concrete route/table/form evidence than class names.
    "brd": {
        "routes": 1,
        "controllers": 0,
        "sql": 2,
        "form_files": 1,
        "fields": 4,
        "exec_paths": 0,
    },
    "srs": {
        "routes": 2,
        "controllers": 0,
        "sql": 3,
        "form_files": 2,
        "fields": 6,
        "exec_paths": 1,
    },
    "ac": {
        "routes": 2,
        "controllers": 0,
        "sql": 2,
        "form_files": 2,
        "fields": 6,
        "exec_paths": 1,
    },
    "us": {
        "routes": 1,
        "controllers": 0,
        "sql": 1,
        "form_files": 1,
        "fields": 5,
        "exec_paths": 1,
    },
}

# ── Confidence thresholds ──────────────────────────────────────────────────────
_SCORE_PER_CATEGORY   = 0.2   # each of 5 evidence types = +0.2
_THRESHOLD_HIGH       = 0.8
_THRESHOLD_MEDIUM     = 0.4
_GRADE_HIGH           = "HIGH 🟢"
_GRADE_MEDIUM         = "MEDIUM 🟡"
_GRADE_LOW            = "LOW 🔴"

_EVIDENCE_KEYS = ("routes", "controllers", "sql", "form_fields", "execution_paths")


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


def compute_confidence(ev: dict) -> float:
    """
    Compute a 0.0–1.0 confidence score for a single feature's evidence dict.

    Each of the 5 evidence categories that has ≥1 item contributes +0.2.
    A score of 1.0 means all five evidence types are present.
    """
    score = 0.0
    for key in _EVIDENCE_KEYS:
        if ev.get(key):
            score += _SCORE_PER_CATEGORY
    return round(score, 2)


def _confidence_grade(score: float) -> str:
    if score >= _THRESHOLD_HIGH:
        return _GRADE_HIGH
    if score >= _THRESHOLD_MEDIUM:
        return _GRADE_MEDIUM
    return _GRADE_LOW


def build_confidence_report(ev_idx: dict[str, dict]) -> list[dict]:
    """
    Build a confidence report for every feature in the evidence index.

    Returns
    -------
    list[dict] sorted by score ascending (weakest features first), each entry:
        {
            "feature":  str,    feature name
            "score":    float,  0.0 – 1.0
            "grade":    str,    HIGH 🟢 / MEDIUM 🟡 / LOW 🔴
            "present":  list[str],  evidence categories that had data
            "missing":  list[str],  evidence categories with no data
        }
    """
    rows: list[dict] = []
    for feature, ev in ev_idx.items():
        score   = compute_confidence(ev)
        present = [k for k in _EVIDENCE_KEYS if ev.get(k)]
        missing = [k for k in _EVIDENCE_KEYS if not ev.get(k)]
        rows.append({
            "feature": feature,
            "score":   score,
            "grade":   _confidence_grade(score),
            "present": present,
            "missing": missing,
        })
    rows.sort(key=lambda r: (r["score"], r["feature"]))
    return rows


def format_evidence_block(ev: dict, profile: str = "default") -> str:
    """
    Format a single feature's evidence dict as a concise Markdown block
    suitable for embedding in an LLM prompt scaffold.

    The header line includes the confidence score and grade so the LLM
    agent knows how reliably the feature is backed by code evidence.

    Returns an empty string when no evidence was found for the feature.
    """
    lines: list[str] = []
    caps = _EVIDENCE_PROFILES.get(profile, _EVIDENCE_PROFILES["default"])

    # ── Routes ────────────────────────────────────────────────────────────────
    for r in ev.get("routes", [])[:caps["routes"]]:
        method = r.get("method", "?")
        path   = r.get("path",   "?")
        ffile  = Path(r.get("file", "?")).name
        lines.append(f"  - Route: `{method} {path}` [{ffile}]")

    # ── Controllers ───────────────────────────────────────────────────────────
    ctrls = ev.get("controllers", [])[:caps["controllers"]]
    if ctrls:
        lines.append(f"  - Controller: {', '.join(f'`{c}`' for c in ctrls)}")

    # ── SQL queries ───────────────────────────────────────────────────────────
    for q in ev.get("sql", [])[:caps["sql"]]:
        op    = q.get("operation", "?")
        table = q.get("table",     "?")
        qfile = Path(q.get("file", "?")).name
        lines.append(f"  - SQL: `{op} {table}` [{qfile}]")

    # ── Form fields ───────────────────────────────────────────────────────────
    for ff in ev.get("form_fields", [])[:caps["form_files"]]:
        fname  = Path(ff.get("file", "?")).name
        action = ff.get("action", "")
        method = ff.get("method", "")
        fields = [
            f.get("name", "")
            for f in ff.get("fields", [])
            if f.get("name") and f.get("type") not in ("hidden",)
        ][:caps["fields"]]
        visible = [f for f in fields if f]
        # include hidden fields too if no visible ones found
        if not visible:
            visible = [
                f.get("name", "")
                for f in ff.get("fields", [])
                if f.get("name")
            ][:caps["fields"]]
        if visible:
            action_str = f" → {action}" if action else ""
            lines.append(
                f"  - Form `{fname}` ({method}{action_str}): "
                f"{', '.join(f'`{n}`' for n in visible)}"
            )

    # ── Execution paths ───────────────────────────────────────────────────────
    for ep in ev.get("execution_paths", [])[:caps["exec_paths"]]:
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
        # No evidence at all — still emit a confidence label so the LLM
        # knows this feature has zero code backing.
        score = compute_confidence(ev)
        grade = _confidence_grade(score)
        return f"**Evidence (confidence: {score:.1f} {grade}):** _none found_"

    score  = compute_confidence(ev)
    grade  = _confidence_grade(score)
    header = f"**Evidence (confidence: {score:.1f} {grade}):**"
    return header + "\n" + "\n".join(lines)


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
