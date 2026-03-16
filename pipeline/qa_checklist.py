"""
pipeline/qa_checklist.py — Automated QA Checklist

Produces a structured pass/fail checklist across three dimensions:

  1. Completeness  — how much of the codebase is represented in business flows
     • Route Coverage       (flow_coverage.json → route.pct)
     • Execution-Path Coverage  (flow_coverage.json → exec_path.pct)
     • Table Coverage       (flow_coverage.json → table.pct)
     • Form Coverage        (flow_coverage.json → form.pct)

  2. Consistency  — cross-document traceability (derived from Pass D issues)
     • Domain Roles ↔ Flows        (C-01: actors in flows ∉ domain.user_roles)
     • Flows ↔ SRS Features        (C-04: features missing as headings in SRS)
     • Flows ↔ Acceptance Criteria (C-05: flows with no AC section)
     • Flows ↔ User Stories        (C-04: features missing as headings in US)

  3. Evidence  — every feature must have at least one behavioural code artifact
     • Behavioral evidence coverage  (route OR controller OR execution_path ≥ 1)

Status icons
------------
  ✅  PASS  — meets threshold
  ⚠️  WARN  — below best-practice threshold but not critical
  ❌  FAIL  — below minimum acceptable threshold

Output
------
  list[dict] — each dict:
    {
        "category":    "Completeness" | "Consistency" | "Evidence",
        "check":       str,   human-readable check name
        "status":      "pass" | "warn" | "fail",
        "icon":        "✅" | "⚠️" | "❌",
        "score":       float | None,   (coverage ratio, 0.0–1.0; None for count-based)
        "issue_count": int   | None,   (issue count; None for ratio-based)
        "detail":      str,  one-line description of the result
    }
"""

from __future__ import annotations

from typing import Any

# ── Completeness thresholds (same as flow_coverage.py) ────────────────────────
_COV_PASS = 0.85
_COV_WARN = 0.50

# ── Consistency thresholds (issue counts) ─────────────────────────────────────
_CON_PASS  = 0        # 0 issues → ✅
_CON_WARN  = 3        # 1–3 issues → ⚠️  (>3 → ❌)

# ── Evidence threshold (fraction of features with behavioral evidence) ─────────
_EV_PASS   = 1.00     # 100% → ✅
_EV_WARN   = 0.75     # ≥75% → ⚠️   (<75% → ❌)


# ─── Public API ────────────────────────────────────────────────────────────────

def build_checklist(
    flow_coverage_data: dict | None,
    pass_d_issues:      list[dict],
    ev_idx:             dict[str, dict] | None,
    dc_data:            dict | None = None,
) -> list[dict]:
    """
    Build and return the automated QA checklist.

    Parameters
    ----------
    flow_coverage_data : dict loaded from flow_coverage.json (may be None).
    pass_d_issues      : list of issue dicts from consistency_check.run_checks().
    ev_idx             : evidence index from evidence_index.build_evidence_index().
    dc_data            : dict from Stage 5.9 doc_coverage.json dimensions
                         (keyed by dimension name; may be None).

    Returns
    -------
    list[dict]  — checklist items, preserving category order.
    """
    items: list[dict] = []
    items.extend(_completeness_items(flow_coverage_data or {}))
    items.extend(_doc_coverage_items(dc_data or {}))   # Stage 5.9
    items.extend(_consistency_items(pass_d_issues))
    items.extend(_evidence_items(ev_idx or {}))
    return items


def format_checklist_md(items: list[dict]) -> str:
    """
    Render the checklist as a Markdown section suitable for embedding in
    the QA report.  Groups items by category into separate tables.
    """
    if not items:
        return ""

    lines: list[str] = ["## Automated QA Checklist", ""]

    # Group by category (preserve insertion order)
    cats: dict[str, list[dict]] = {}
    for item in items:
        cats.setdefault(item["category"], []).append(item)

    for cat, cat_items in cats.items():
        lines.append(f"### {cat}")
        lines.append("")
        lines.append(_table_header(cat))
        lines.append(_table_divider(cat))
        for item in cat_items:
            lines.append(_table_row(cat, item))
        lines.append("")

    # Overall summary line
    n_pass = sum(1 for i in items if i["status"] == "pass")
    n_warn = sum(1 for i in items if i["status"] == "warn")
    n_fail = sum(1 for i in items if i["status"] == "fail")
    lines += [
        f"**Checklist summary:** "
        f"{n_pass} ✅ passed · {n_warn} ⚠️ warnings · {n_fail} ❌ failed",
        "",
    ]
    return "\n".join(lines)


def checklist_overall_status(items: list[dict]) -> str:
    """Return 'pass', 'warn', or 'fail' for the entire checklist."""
    if any(i["status"] == "fail" for i in items):
        return "fail"
    if any(i["status"] == "warn" for i in items):
        return "warn"
    return "pass"


# ─── Completeness ──────────────────────────────────────────────────────────────

_COV_LABELS = {
    "exec_path": "Execution-Path Coverage",
    "route":     "Route Coverage",
    "table":     "Table Coverage",
    "form":      "Form Coverage",
}


def _completeness_items(fc: dict) -> list[dict]:
    items: list[dict] = []
    for key, label in _COV_LABELS.items():
        block = fc.get(key, {})
        if not isinstance(block, dict):
            continue
        pct     = float(block.get("pct",     0.0))
        covered = int(block.get("covered",  0))
        total   = int(block.get("total",    0))
        status, icon = _cov_status(pct)
        detail = (
            f"{covered}/{total} ({pct:.0%}) "
            + ("— all covered" if pct >= _COV_PASS else
               "— below 85% best-practice threshold" if pct >= _COV_WARN else
               "— critically low, most codebase not in any flow")
        )
        items.append(_item(
            category    = "Completeness",
            check       = label,
            status      = status,
            icon        = icon,
            score       = round(pct, 4),
            issue_count = None,
            detail      = detail,
        ))
    return items


def _cov_status(pct: float) -> tuple[str, str]:
    if pct >= _COV_PASS:
        return "pass", "✅"
    if pct >= _COV_WARN:
        return "warn", "⚠️"
    return "fail", "❌"


# ─── Document Coverage (Stage 5.9) ────────────────────────────────────────────

_DC_LABELS: dict[str, str] = {
    "entities":   "Entity Coverage → SRS",
    "flows":      "Flow Coverage → BRD",
    "spec_rules": "Business Rule Coverage → SRS+BRD",
    "states":     "State Machine Coverage → SRS",
    "relations":  "Relationship Coverage → SRS",
}


def _doc_coverage_items(dc: dict) -> list[dict]:
    """
    Build checklist items from Stage 5.9 document coverage dimensions.

    dc : {dimension_name: DimCoverage-as-dict} — may be empty when stage59
         has not run, in which case no items are emitted.
    """
    items: list[dict] = []
    for key, label in _DC_LABELS.items():
        dim = dc.get(key)
        if not isinstance(dim, dict):
            continue
        pct     = float(dim.get("pct", 0.0))
        covered = int(dim.get("covered", 0))
        total   = int(dim.get("total", 0))
        status, icon = _cov_status(pct)
        uncovered = dim.get("uncovered", [])
        sample    = uncovered[:3]
        suffix    = f" (+{len(uncovered) - 3} more)" if len(uncovered) > 3 else ""
        detail = (
            f"{covered}/{total} ({pct:.0%})"
            + (f" — missing: {', '.join(sample)}{suffix}" if uncovered else " — fully covered")
        )
        items.append(_item(
            category    = "Document Coverage",
            check       = label,
            status      = status,
            icon        = icon,
            score       = round(pct, 4),
            issue_count = None,
            detail      = detail,
        ))
    return items


# ─── Consistency ───────────────────────────────────────────────────────────────

def _consistency_items(pass_d_issues: list[dict]) -> list[dict]:
    """
    Derive 4 consistency checklist items from the Pass D issue list.

    Filter keys
    -----------
    C-01  category == "Domain Model vs Flows" and "actor" in description (case-insensitive)
    C-04/SRS  category == "Feature Heading Presence" and artefact == "SRS"
    C-05  category == "Flow → AC Coverage"
    C-04/US   category == "Feature Heading Presence" and artefact == "US"
    """
    # --- C-01: Domain Roles ↔ Flows ---
    role_issues = [
        i for i in pass_d_issues
        if i.get("category") == "Domain Model vs Flows"
        and "actor" in i.get("description", "").lower()
    ]
    # --- C-04/SRS: Flows ↔ SRS Features ---
    srs_issues = [
        i for i in pass_d_issues
        if i.get("category") == "Feature Heading Presence"
        and i.get("artefact") == "SRS"
    ]
    # --- C-05: Flows ↔ AC ---
    ac_issues = [
        i for i in pass_d_issues
        if i.get("category") == "Flow → AC Coverage"
    ]
    # --- C-04/US: Flows ↔ User Stories ---
    us_issues = [
        i for i in pass_d_issues
        if i.get("category") == "Feature Heading Presence"
        and i.get("artefact") == "US"
    ]

    specs = [
        ("Domain Roles ↔ Flows",        role_issues,
         "actors in flows not defined in domain model"),
        ("Flows ↔ SRS Features",         srs_issues,
         "features missing as headings in SRS"),
        ("Flows ↔ Acceptance Criteria",  ac_issues,
         "flows with no corresponding AC section"),
        ("Flows ↔ User Stories",          us_issues,
         "features missing as headings in User Stories"),
    ]

    items: list[dict] = []
    for check, grp, noun in specs:
        n       = len(grp)
        status, icon = _con_status(n)
        detail  = (
            f"No issues found — all {noun.split(' ')[0].lower()} consistent" if n == 0 else
            f"{n} issue{'s' if n > 1 else ''}: {noun}"
        )
        items.append(_item(
            category    = "Consistency",
            check       = check,
            status      = status,
            icon        = icon,
            score       = None,
            issue_count = n,
            detail      = detail,
        ))
    return items


def _con_status(n: int) -> tuple[str, str]:
    if n <= _CON_PASS:
        return "pass", "✅"
    if n <= _CON_WARN:
        return "warn", "⚠️"
    return "fail", "❌"


# ─── Evidence ──────────────────────────────────────────────────────────────────

def _evidence_items(ev_idx: dict[str, dict]) -> list[dict]:
    """
    Every feature must have at least one behavioural code artifact:
    route, controller, or execution_path.  Pure SQL/form evidence is
    insufficient — it shows data is touched but not how it is entered.
    """
    if not ev_idx:
        return []

    total        = len(ev_idx)
    no_behavioral = [
        feat for feat, ev in ev_idx.items()
        if not (ev.get("routes") or ev.get("controllers") or ev.get("execution_paths"))
    ]
    covered = total - len(no_behavioral)
    pct     = covered / total if total else 0.0

    if pct >= _EV_PASS:
        status, icon = "pass", "✅"
        detail = f"All {total} features have at least one route, controller, or execution path"
    elif pct >= _EV_WARN:
        status, icon = "warn", "⚠️"
        detail = (
            f"{covered}/{total} features have behavioral evidence. "
            f"No behavioral evidence: {', '.join(no_behavioral[:4])}"
            + (f" (+{len(no_behavioral)-4} more)" if len(no_behavioral) > 4 else "")
        )
    else:
        status, icon = "fail", "❌"
        detail = (
            f"Only {covered}/{total} features ({pct:.0%}) have a route, "
            f"controller, or execution path. "
            f"Missing: {', '.join(no_behavioral[:4])}"
            + (f" (+{len(no_behavioral)-4} more)" if len(no_behavioral) > 4 else "")
        )

    return [_item(
        category    = "Evidence",
        check       = "Behavioral Evidence Coverage",
        status      = status,
        icon        = icon,
        score       = round(pct, 4),
        issue_count = len(no_behavioral),
        detail      = detail,
    )]


# ─── Markdown rendering ────────────────────────────────────────────────────────

def _table_header(cat: str) -> str:
    if cat == "Completeness":
        return "| Check | Status | Score | Detail |"
    if cat == "Consistency":
        return "| Check | Status | Issues | Detail |"
    return "| Check | Status | Coverage | Detail |"


def _table_divider(cat: str) -> str:
    if cat == "Completeness":
        return "|-------|--------|-------|--------|"
    if cat == "Consistency":
        return "|-------|--------|--------|--------|"
    return "|-------|--------|----------|--------|"


def _table_row(cat: str, item: dict) -> str:
    icon   = item["icon"]
    check  = item["check"]
    detail = item["detail"]
    if cat == "Completeness":
        score = f"{item['score']:.0%}" if item["score"] is not None else "—"
        return f"| {check} | {icon} | {score} | {detail} |"
    if cat == "Consistency":
        count = str(item["issue_count"]) if item["issue_count"] is not None else "—"
        return f"| {check} | {icon} | {count} | {detail} |"
    # Evidence
    score = f"{item['score']:.0%}" if item["score"] is not None else "—"
    return f"| {check} | {icon} | {score} | {detail} |"


# ─── Internal factory ──────────────────────────────────────────────────────────

def _item(
    category:    str,
    check:       str,
    status:      str,
    icon:        str,
    score:       float | None,
    issue_count: int   | None,
    detail:      str,
) -> dict:
    return {
        "category":    category,
        "check":       check,
        "status":      status,
        "icon":        icon,
        "score":       score,
        "issue_count": issue_count,
        "detail":      detail,
    }
