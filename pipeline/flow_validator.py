"""
pipeline/flow_validator.py — Behavioral Validation (Stage 4.7)

Answers three critical questions about the generated business flows using
only deterministic code analysis.  No LLM is invoked.

  1. Are we missing business flows?
  2. Are generated flows actually valid (non-hallucinated)?
  3. Do flows match real execution paths?

Eight checks
------------
  V-01  Missing Business Flows
        Routes, forms, and execution paths not covered by any flow.

  V-02  Fake / Hallucinated Steps
        Flow steps that reference PHP files, route paths, or DB tables
        that do not exist anywhere in the parsed codebase.

  V-03  Broken Flows
        Flows that exist but cannot form a coherent chain:
        single-step flows, flows with zero evidence_files, or flows
        whose step pages are all unrecognised.

  V-04  Missing Branches
        Flows whose evidence files have ≥2 distinct redirect targets
        but the flow declares no branches (likely missing error /
        alternate path).

  V-05  Route Coverage Validation
        Percentage of HTTP routes (by path) covered by at least one flow.

  V-06  Database Operation Coverage
        Percentage of SQL write operations (INSERT/UPDATE/DELETE/REPLACE)
        whose source file appears in at least one flow step.

  V-07  Form Coverage
        Percentage of HTML forms covered by at least one flow step.

  V-08  Per-Flow Confidence Scoring
        For every flow, compute what fraction of its steps are backed by
        a real code artifact; grade as HIGH/MEDIUM/LOW.

Output schema
-------------
  The return value of ``run_validation()`` is a dict::

    {
        "generated_at": str,
        "total_flows":  int,
        "summary": {
            "overall_grade":  "PASS" | "WARN" | "FAIL",
            "overall_score":  float,     # 0.0 – 1.0 composite
            "checks_passed":  int,
            "checks_warned":  int,
            "checks_failed":  int,
        },
        "checks": {
            "V-01": { "name", "status", "icon",
                      "missing_count", "detail", "items": [...] },
            "V-02": { "name", "status", "icon",
                      "issue_count",   "detail", "items": [...] },
            "V-03": { "name", "status", "icon",
                      "broken_count",  "detail", "items": [...] },
            "V-04": { "name", "status", "icon",
                      "branch_count",  "detail", "items": [...] },
            "V-05": { "name", "status", "icon",
                      "covered", "total", "pct", "uncovered": [...] },
            "V-06": { "name", "status", "icon",
                      "covered", "total", "pct", "uncovered": [...] },
            "V-07": { "name", "status", "icon",
                      "covered", "total", "pct", "uncovered": [...] },
            "V-08": { "name",
                      "avg_confidence", "high_count", "medium_count",
                      "low_count", "flows": [...] },
        },
    }
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ── SQL write operations that represent meaningful state changes ───────────────
_WRITE_OPS: frozenset[str] = frozenset({"INSERT", "UPDATE", "DELETE", "REPLACE"})

# ── Tables too generic to be informative ──────────────────────────────────────
_SKIP_TABLES: frozenset[str] = frozenset({"", "unknown", "temp", "tmp", "dual"})

# ── Auth-related middleware names ─────────────────────────────────────────────
_AUTH_MIDDLEWARE: frozenset[str] = frozenset(
    {"auth", "auth:sanctum", "auth:api", "verified", "login"}
)

# ── Thresholds ─────────────────────────────────────────────────────────────────
_V01_WARN  = 5    # critical-severity missing items before WARN
_V01_FAIL  = 10   # > this many → FAIL

_V02_WARN  = 3    # hallucinated steps before WARN
_V02_FAIL  = 5    # > this many → FAIL

_V03_WARN  = 2    # broken flows before WARN
_V03_FAIL  = 4    # > this many → FAIL

_V04_WARN  = 5    # flows with missing branches before WARN
_V04_FAIL  = 10   # > this many → FAIL

_COV_PASS  = 0.85  # coverage PASS threshold
_COV_WARN  = 0.50  # coverage WARN threshold (below → FAIL)

_CONF_PASS = 0.70  # avg confidence PASS
_CONF_WARN = 0.40  # avg confidence WARN (below → FAIL)

# Max items stored in uncovered lists (avoids huge JSON)
_MAX_LIST  = 50


# ─── Public API ────────────────────────────────────────────────────────────────

def run_validation(ctx: Any) -> dict:
    """
    Run all 8 behavioral validation checks and return the full report dict.

    Parameters
    ----------
    ctx : PipelineContext  (needs code_map and business_flows)

    Returns
    -------
    dict — full validation report (see module docstring for schema)
    """
    cm    = getattr(ctx, "code_map",        None)
    flows = getattr(ctx, "business_flows",  None)

    flow_list = (flows.flows if flows else []) or []
    total_flows = len(flow_list)

    # ── Pre-compute shared indexes ─────────────────────────────────────────────
    known_files    = _build_known_files(cm)
    known_routes   = _build_known_route_paths(cm)
    known_tables   = _build_known_tables(cm)
    file_to_sql    = _build_file_sql_index(cm)
    file_to_redir  = _build_file_redir_index(cm)

    # flow_pages: file basenames that appear in any step.page
    # flow_paths: raw step.page values (for route path matching)
    # flow_evidence: basenames in evidence_files
    flow_pages, flow_paths, flow_evidence = _build_flow_pivot(flow_list)

    # ── Run checks ────────────────────────────────────────────────────────────
    v01 = _check_missing_flows(cm, flow_pages, flow_paths, flow_evidence, file_to_sql)
    v02 = _check_hallucinated_steps(flow_list, known_files, known_routes, known_tables)
    v03 = _check_broken_flows(flow_list, known_files, known_routes)
    v04 = _check_missing_branches(flow_list, file_to_redir)
    v05 = _check_route_coverage(cm, flow_pages, flow_paths)
    v06 = _check_db_write_coverage(cm, flow_pages)
    v07 = _check_form_coverage(cm, flow_pages)
    v08 = _check_flow_confidence(flow_list, known_files, known_routes)

    checks = {
        "V-01": v01, "V-02": v02, "V-03": v03, "V-04": v04,
        "V-05": v05, "V-06": v06, "V-07": v07, "V-08": v08,
    }

    # ── Summary ────────────────────────────────────────────────────────────────
    statuses = [c.get("status", "pass") for c in checks.values() if "status" in c]
    n_pass = statuses.count("pass")
    n_warn = statuses.count("warn")
    n_fail = statuses.count("fail")

    if n_fail > 0:
        overall_grade = "FAIL"
    elif n_warn > 0:
        overall_grade = "WARN"
    else:
        overall_grade = "PASS"

    # Composite score: average of normalized per-check scores
    scores: list[float] = []
    for chk in [v05, v06, v07]:          # coverage checks have a pct
        scores.append(float(chk.get("pct", 0.0)))
    scores.append(float(v08.get("avg_confidence", 0.0)))
    # Boolean checks: 1.0 if pass, 0.5 if warn, 0.0 if fail
    for chk in [v01, v02, v03, v04]:
        st = chk.get("status", "pass")
        scores.append(1.0 if st == "pass" else 0.5 if st == "warn" else 0.0)
    overall_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "total_flows":  total_flows,
        "summary": {
            "overall_grade": overall_grade,
            "overall_score": overall_score,
            "checks_passed": n_pass,
            "checks_warned": n_warn,
            "checks_failed": n_fail,
        },
        "checks": checks,
    }


def format_report_md(report: dict) -> str:
    """
    Render the validation report as a Markdown document for human review.

    Returns
    -------
    str — full Markdown content
    """
    lines: list[str] = []
    s     = report.get("summary", {})
    grade = s.get("overall_grade", "?")
    score = s.get("overall_score", 0.0)
    grade_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(grade, "?")

    lines += [
        "# Behavioral Validation Report",
        "",
        f"**Flows analysed:** {report.get('total_flows', 0)}  "
        f"│  **Overall grade:** {grade_icon} {grade}  "
        f"│  **Composite score:** {score:.0%}",
        f"**Generated:** {report.get('generated_at', '')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Check | Name | Status | Score / Count |",
        "|-------|------|--------|---------------|",
    ]

    chk_meta = [
        ("V-01", "Missing Business Flows"),
        ("V-02", "Fake / Hallucinated Steps"),
        ("V-03", "Broken Flows"),
        ("V-04", "Missing Branches"),
        ("V-05", "Route Coverage"),
        ("V-06", "Database Write Coverage"),
        ("V-07", "Form Coverage"),
        ("V-08", "Flow Confidence Scoring"),
    ]
    for cid, cname in chk_meta:
        chk = report["checks"].get(cid, {})
        icon = chk.get("icon", "")
        # Score column
        if "pct" in chk:
            score_str = f"{chk['pct']:.0%}"
        elif "avg_confidence" in chk:
            score_str = f"avg {chk['avg_confidence']:.2f}"
        elif "missing_count" in chk:
            score_str = f"{chk['missing_count']} missing"
        elif "issue_count" in chk:
            score_str = f"{chk['issue_count']} issues"
        elif "broken_count" in chk:
            score_str = f"{chk['broken_count']} broken"
        elif "branch_count" in chk:
            score_str = f"{chk['branch_count']} flows"
        else:
            score_str = "—"
        lines.append(f"| {cid} | {cname} | {icon} | {score_str} |")

    lines += [
        "",
        f"**Passed:** {s.get('checks_passed',0)} ✅  "
        f"**Warned:** {s.get('checks_warned',0)} ⚠️  "
        f"**Failed:** {s.get('checks_failed',0)} ❌",
        "",
        "---",
        "",
    ]

    # ── Per-check detail sections ──────────────────────────────────────────────
    lines += _section_v01(report["checks"].get("V-01", {}))
    lines += _section_v02(report["checks"].get("V-02", {}))
    lines += _section_v03(report["checks"].get("V-03", {}))
    lines += _section_v04(report["checks"].get("V-04", {}))
    lines += _section_cov("V-05", "Route Coverage Validation",
                          report["checks"].get("V-05", {}), "route", "method", "path")
    lines += _section_cov("V-06", "Database Write Operation Coverage",
                          report["checks"].get("V-06", {}), "db_write", "op", "table")
    lines += _section_cov("V-07", "Form Coverage",
                          report["checks"].get("V-07", {}), "form", "method", "action")
    lines += _section_v08(report["checks"].get("V-08", {}))

    return "\n".join(lines)


# ─── V-01: Missing Business Flows ─────────────────────────────────────────────

def _check_missing_flows(
    cm:             Any,
    flow_pages:     set[str],
    flow_paths:     set[str],
    flow_evidence:  set[str],
    file_to_sql:    dict[str, list[dict]],
) -> dict:
    """Routes, forms, execution paths with no corresponding flow step."""
    items: list[dict] = []

    # Routes
    for route in (getattr(cm, "routes", None) or []):
        method = (route.get("method") or "").upper()
        if method in ("GROUP", "MIDDLEWARE", "PREFIX", ""):
            continue
        path  = route.get("path") or route.get("uri") or ""
        rfile = route.get("file", "")
        rname = Path(rfile).name.lower() if rfile else ""

        if _route_in_flows(path, rname, flow_pages, flow_paths, flow_evidence):
            continue

        # Determine severity
        middleware = route.get("middleware", [])
        if isinstance(middleware, str):
            middleware = [middleware]
        has_auth = any(mw in _AUTH_MIDDLEWARE for mw in middleware)

        sql_writes = [
            q for q in file_to_sql.get(rfile, [])
            if (q.get("operation") or "").upper() in _WRITE_OPS
            and (q.get("table") or "").lower() not in _SKIP_TABLES
        ]
        if sql_writes:
            severity = "critical"
            reason   = (f"POST route with {len(sql_writes)} SQL write op(s) "
                        f"({', '.join(set(q['table'] for q in sql_writes[:3]))}) "
                        f"not captured in any flow")
        elif has_auth or method in ("POST", "PUT", "PATCH", "DELETE"):
            severity = "major"
            reason   = f"{method} route not captured in any flow"
        else:
            severity = "minor"
            reason   = f"{method} route not captured in any flow"

        items.append({
            "type":       "route",
            "identifier": f"{method} {path}",
            "file":       rfile,
            "severity":   severity,
            "reason":     reason,
        })

    # Forms (POST only — GET forms are usually search/filter, lower priority)
    for ff in (getattr(cm, "form_fields", None) or []):
        ffile = ff.get("file", "")
        fname = Path(ffile).name.lower() if ffile else ""
        fmethod = (ff.get("method") or "GET").upper()
        if fname and fname not in flow_pages and fname not in flow_evidence:
            sev = "major" if fmethod == "POST" else "minor"
            action = ff.get("action", "")
            items.append({
                "type":       "form",
                "identifier": f"{fmethod} form → {action or '(same page)'}",
                "file":       ffile,
                "severity":   sev,
                "reason":     f"{fmethod} form in {fname} not captured in any flow",
            })

    # Execution paths typed as 'controller' (highest-value entry points)
    for ep in (getattr(cm, "execution_paths", None) or []):
        if ep.get("type", "").lower() != "controller":
            continue
        efile = ep.get("file", "")
        ename = Path(efile).name.lower() if efile else ""
        if ename and ename not in flow_pages and ename not in flow_evidence:
            items.append({
                "type":       "execution_path",
                "identifier": ename,
                "file":       efile,
                "severity":   "minor",
                "reason":     "Controller execution path not captured in any flow",
            })

    # Deduplicate (routes and exec_paths can share files)
    seen: set[tuple] = set()
    unique: list[dict] = []
    for it in items:
        key = (it["type"], it["identifier"])
        if key not in seen:
            seen.add(key)
            unique.append(it)

    unique = unique[:_MAX_LIST]
    critical_count = sum(1 for i in unique if i["severity"] == "critical")
    major_count    = sum(1 for i in unique if i["severity"] == "major")

    if critical_count > _V01_FAIL or (critical_count + major_count) > _V01_FAIL * 2:
        status, icon = "fail", "❌"
    elif critical_count > _V01_WARN or (critical_count + major_count) > _V01_WARN * 2:
        status, icon = "warn", "⚠️"
    else:
        status, icon = "pass", "✅"

    detail = (
        "No uncovered routes or forms" if not unique else
        f"{len(unique)} uncovered — {critical_count} critical, {major_count} major"
    )

    return {
        "name":          "Missing Business Flows",
        "status":        status,
        "icon":          icon,
        "missing_count": len(unique),
        "detail":        detail,
        "items":         unique,
    }


# ─── V-02: Fake / Hallucinated Steps ──────────────────────────────────────────

def _check_hallucinated_steps(
    flow_list:    list,
    known_files:  set[str],
    known_routes: set[str],
    known_tables: set[str],
) -> dict:
    """Flow steps referencing files, routes, or tables that don't exist."""
    items: list[dict] = []

    for flow in flow_list:
        for step in (flow.steps or []):
            page = (step.page or "").strip()
            issues_for_step: list[str] = []

            # ── File check: step.page looks like a PHP file ────────────────────
            if page.endswith(".php"):
                fname = Path(page).name.lower()
                if fname not in known_files:
                    issues_for_step.append(
                        f"PHP file '{fname}' does not exist in the parsed codebase"
                    )

            # ── Route path check: step.page starts with '/' ───────────────────
            elif page.startswith("/") and not page.endswith(".php"):
                path_clean = page.split("?")[0].lower()
                if path_clean not in known_routes:
                    # Partial match: check if any known route starts with this prefix
                    if not any(r.startswith(path_clean) or path_clean.startswith(r)
                               for r in known_routes if r):
                        issues_for_step.append(
                            f"Route path '{page}' not found in registered routes"
                        )

            # ── DB ops check: tables mentioned in step.db_ops ─────────────────
            for db_op in (step.db_ops or []):
                # db_ops format: "INSERT users" or "SELECT sessions"
                parts = db_op.strip().split()
                if len(parts) >= 2:
                    table = parts[1].lower()
                    if (table not in _SKIP_TABLES
                            and table not in known_tables
                            and len(table) > 2):  # ignore very short tokens
                        issues_for_step.append(
                            f"DB op '{db_op}' references unknown table '{table}'"
                        )

            for reason in issues_for_step:
                items.append({
                    "flow_id":   flow.flow_id,
                    "flow_name": flow.name,
                    "step_num":  step.step_num,
                    "step_page": page,
                    "reason":    reason,
                })

    items = items[:_MAX_LIST]
    n = len(items)

    if n > _V02_FAIL:
        status, icon = "fail", "❌"
    elif n > _V02_WARN:
        status, icon = "warn", "⚠️"
    else:
        status, icon = "pass", "✅"

    return {
        "name":        "Fake / Hallucinated Steps",
        "status":      status,
        "icon":        icon,
        "issue_count": n,
        "detail":      ("No hallucinated references detected" if n == 0
                        else f"{n} step(s) reference non-existent code artifacts"),
        "items":       items,
    }


# ─── V-03: Broken Flows ───────────────────────────────────────────────────────

def _check_broken_flows(
    flow_list:    list,
    known_files:  set[str],
    known_routes: set[str],
) -> dict:
    """Flows that exist but cannot form a coherent, code-backed chain."""
    items: list[dict] = []

    for flow in flow_list:
        steps    = flow.steps or []
        ev_files = flow.evidence_files or []
        issues: list[str] = []
        severity = "minor"

        # Single-step (no chain at all)
        if len(steps) < 2:
            issues.append("Only 1 step — no route→controller→DB chain")
            severity = "major"

        # Zero evidence_files (LLM invented the flow with no code backing)
        if not ev_files:
            issues.append("No evidence_files — flow has no code backing")
            severity = "critical"
        else:
            # Evidence files all unrecognised in codebase
            backed = [
                f for f in ev_files
                if Path(f).name.lower() in known_files
            ]
            if ev_files and not backed:
                issues.append(
                    f"All {len(ev_files)} evidence file(s) are unrecognised "
                    f"in the parsed codebase"
                )
                severity = "major"

        # All step pages unrecognised (every page is an invented label)
        php_pages = [s.page for s in steps if (s.page or "").endswith(".php")]
        if php_pages:
            unrecognised = [
                p for p in php_pages
                if Path(p).name.lower() not in known_files
            ]
            if len(unrecognised) == len(php_pages):
                issues.append(
                    f"None of the {len(php_pages)} PHP step page(s) "
                    f"exist in the codebase"
                )
                severity = max(severity, "major",
                               key=lambda s: ["minor", "major", "critical"].index(s))

        if issues:
            items.append({
                "flow_id":   flow.flow_id,
                "flow_name": flow.name,
                "severity":  severity,
                "issues":    issues,
            })

    items = items[:_MAX_LIST]
    n = len(items)
    critical = sum(1 for i in items if i["severity"] == "critical")

    if n > _V03_FAIL or critical > 2:
        status, icon = "fail", "❌"
    elif n > _V03_WARN or critical > 0:
        status, icon = "warn", "⚠️"
    else:
        status, icon = "pass", "✅"

    return {
        "name":         "Broken Flows",
        "status":       status,
        "icon":         icon,
        "broken_count": n,
        "detail":       ("All flows have valid structure" if n == 0
                         else f"{n} flow(s) have structural issues; {critical} critical"),
        "items":        items,
    }


# ─── V-04: Missing Branches ───────────────────────────────────────────────────

def _check_missing_branches(
    flow_list:     list,
    file_to_redir: dict[str, list[dict]],
) -> dict:
    """
    Flows whose evidence files contain ≥2 distinct redirect targets but
    the flow has no branch steps (potential missing error / alternate path).
    """
    items: list[dict] = []

    for flow in flow_list:
        ev_files = flow.evidence_files or []
        branches = flow.branches or []

        # Collect all redirect targets from evidence files
        all_targets: set[str] = set()
        for fpath in ev_files:
            for r in file_to_redir.get(fpath, []):
                t = (r.get("target") or "").strip()
                if t:
                    all_targets.add(t)

        if len(all_targets) >= 2 and not branches:
            sorted_targets = sorted(all_targets)[:6]
            items.append({
                "flow_id":        flow.flow_id,
                "flow_name":      flow.name,
                "redirect_count": len(all_targets),
                "redirects":      sorted_targets,
                "issue":          (
                    f"Flow has {len(all_targets)} distinct redirect targets "
                    f"but no branches declared — possible missing error/alternate path"
                ),
            })

    items = items[:_MAX_LIST]
    n = len(items)

    if n > _V04_FAIL:
        status, icon = "fail", "❌"
    elif n > _V04_WARN:
        status, icon = "warn", "⚠️"
    elif n > 0:
        status, icon = "warn", "⚠️"
    else:
        status, icon = "pass", "✅"

    return {
        "name":         "Missing Branches",
        "status":       status,
        "icon":         icon,
        "branch_count": n,
        "detail":       ("All flows with multiple redirects declare branches" if n == 0
                         else f"{n} flow(s) may be missing error/alternate branches"),
        "items":        items,
    }


# ─── V-05: Route Coverage ─────────────────────────────────────────────────────

def _check_route_coverage(
    cm:         Any,
    flow_pages: set[str],
    flow_paths: set[str],
) -> dict:
    """% of HTTP routes (by path) covered by at least one flow step."""
    seen_paths: set[str] = set()
    covered:    list[dict] = []
    uncovered:  list[dict] = []

    for route in (getattr(cm, "routes", None) or []):
        method = (route.get("method") or "").upper()
        if method in ("GROUP", "MIDDLEWARE", "PREFIX", ""):
            continue
        path  = route.get("path") or route.get("uri") or ""
        rfile = route.get("file", "")
        rname = Path(rfile).name.lower() if rfile else ""

        # Deduplicate by (method, path)
        key = (method, path)
        if key in seen_paths:
            continue
        seen_paths.add(key)

        in_flow = _route_in_flows(path, rname, flow_pages, flow_paths, set())

        if in_flow:
            covered.append({"method": method, "path": path, "file": rfile})
        else:
            # Severity hint for uncovered
            middleware = route.get("middleware", [])
            if isinstance(middleware, str):
                middleware = [middleware]
            sev = "major" if method in ("POST", "PUT", "PATCH", "DELETE") else "minor"
            if any(mw in _AUTH_MIDDLEWARE for mw in middleware):
                sev = "major"
            uncovered.append({
                "method":   method,
                "path":     path,
                "file":     rfile,
                "severity": sev,
            })

    total = len(covered) + len(uncovered)
    pct   = len(covered) / total if total else 0.0
    status, icon = _cov_status(pct)

    return {
        "name":      "Route Coverage Validation",
        "status":    status,
        "icon":      icon,
        "covered":   len(covered),
        "total":     total,
        "pct":       round(pct, 4),
        "detail":    f"{len(covered)}/{total} ({pct:.0%}) routes covered by flows",
        "uncovered": uncovered[:_MAX_LIST],
    }


# ─── V-06: Database Write Operation Coverage ──────────────────────────────────

def _check_db_write_coverage(
    cm:         Any,
    flow_pages: set[str],
) -> dict:
    """% of SQL write ops (INSERT/UPDATE/DELETE/REPLACE) covered by flows."""
    seen: set[tuple[str, str]] = set()
    covered_count   = 0
    uncovered: list[dict] = []

    for q in (getattr(cm, "sql_queries", None) or []):
        op    = (q.get("operation") or "").upper()
        table = (q.get("table") or "").strip()
        qfile = q.get("file", "")
        qname = Path(qfile).name.lower() if qfile else ""

        if op not in _WRITE_OPS:
            continue
        if table.lower() in _SKIP_TABLES:
            continue

        key = (op, table.lower())
        if key in seen:
            continue
        seen.add(key)

        if qname and qname in flow_pages:
            covered_count += 1
        else:
            sev = "critical" if op == "DELETE" else "major"
            uncovered.append({
                "op":       op,
                "table":    table,
                "file":     qfile,
                "severity": sev,
            })

    total = covered_count + len(uncovered)
    pct   = covered_count / total if total else 0.0
    status, icon = _cov_status(pct)

    return {
        "name":      "Database Write Operation Coverage",
        "status":    status,
        "icon":      icon,
        "covered":   covered_count,
        "total":     total,
        "pct":       round(pct, 4),
        "detail":    (f"{covered_count}/{total} ({pct:.0%}) SQL write ops "
                      f"covered by flows"),
        "uncovered": uncovered[:_MAX_LIST],
    }


# ─── V-07: Form Coverage ──────────────────────────────────────────────────────

def _check_form_coverage(
    cm:         Any,
    flow_pages: set[str],
) -> dict:
    """% of HTML forms covered by at least one flow step."""
    seen:  set[str]  = set()
    covered_count    = 0
    uncovered: list[dict] = []

    for ff in (getattr(cm, "form_fields", None) or []):
        ffile  = ff.get("file", "")
        fname  = Path(ffile).name.lower() if ffile else ""
        if not fname or fname in seen:
            continue
        seen.add(fname)

        if fname in flow_pages:
            covered_count += 1
        else:
            method = (ff.get("method") or "GET").upper()
            action = ff.get("action", "")
            uncovered.append({
                "file":     ffile,
                "method":   method,
                "action":   action,
                "severity": "major" if method == "POST" else "minor",
            })

    total = covered_count + len(uncovered)
    pct   = covered_count / total if total else 0.0
    status, icon = _cov_status(pct)

    return {
        "name":      "Form Coverage",
        "status":    status,
        "icon":      icon,
        "covered":   covered_count,
        "total":     total,
        "pct":       round(pct, 4),
        "detail":    f"{covered_count}/{total} ({pct:.0%}) forms covered by flows",
        "uncovered": uncovered[:_MAX_LIST],
    }


# ─── V-08: Per-Flow Confidence Scoring ────────────────────────────────────────

def _check_flow_confidence(
    flow_list:    list,
    known_files:  set[str],
    known_routes: set[str],
) -> dict:
    """
    For each flow, compute what fraction of its steps are backed by a real
    code artifact (known file or known route path).

    Grading
    -------
    HIGH   🟢  backed_ratio ≥ 0.80
    MEDIUM 🟡  backed_ratio ≥ 0.50
    LOW    🔴  backed_ratio <  0.50
    """
    rows: list[dict] = []

    for flow in flow_list:
        steps    = flow.steps or []
        ev_files = flow.evidence_files or []

        # Step backing ratio
        backed_steps = 0
        for step in steps:
            page = (step.page or "").strip()
            fname = Path(page).name.lower()
            path_clean = page.split("?")[0].lower()
            if (fname in known_files
                    or path_clean in known_routes
                    or (fname and any(fname in f for f in known_files))):
                backed_steps += 1

        step_ratio = backed_steps / len(steps) if steps else 0.0

        # Evidence file backing ratio
        backed_ev = sum(
            1 for f in ev_files
            if Path(f).name.lower() in known_files
        )
        ev_ratio = backed_ev / len(ev_files) if ev_files else step_ratio

        # Combine: weight step_ratio 60%, ev_ratio 40%
        confidence = round(0.6 * step_ratio + 0.4 * ev_ratio, 2)

        if confidence >= 0.80:
            grade = "HIGH 🟢"
        elif confidence >= 0.50:
            grade = "MEDIUM 🟡"
        else:
            grade = "LOW 🔴"

        # Compare with stored confidence (flag LLM over-inflation)
        stored_conf = getattr(flow, "confidence", 1.0) or 1.0
        inflation_note = ""
        if stored_conf - confidence > 0.30:
            inflation_note = (
                f"stored confidence {stored_conf:.2f} vs computed {confidence:.2f} "
                f"— possible over-inflation"
            )

        row: dict = {
            "flow_id":      flow.flow_id,
            "flow_name":    flow.name,
            "confidence":   confidence,
            "grade":        grade,
            "backed_steps": backed_steps,
            "total_steps":  len(steps),
            "step_ratio":   round(step_ratio, 2),
            "ev_ratio":     round(ev_ratio, 2),
        }
        if inflation_note:
            row["note"] = inflation_note
        rows.append(row)

    # Sort: worst-first (most actionable at top)
    rows.sort(key=lambda r: (r["confidence"], r["flow_id"]))

    avg_conf = (
        round(sum(r["confidence"] for r in rows) / len(rows), 3)
        if rows else 0.0
    )
    high_count   = sum(1 for r in rows if r["confidence"] >= 0.80)
    medium_count = sum(1 for r in rows if 0.50 <= r["confidence"] < 0.80)
    low_count    = sum(1 for r in rows if r["confidence"] < 0.50)

    if avg_conf >= _CONF_PASS:
        icon = "🟢"
    elif avg_conf >= _CONF_WARN:
        icon = "🟡"
    else:
        icon = "🔴"

    return {
        "name":           "Per-Flow Confidence Scoring",
        "icon":           icon,
        "avg_confidence": avg_conf,
        "high_count":     high_count,
        "medium_count":   medium_count,
        "low_count":      low_count,
        "detail":         (f"avg {avg_conf:.2f} — "
                           f"{high_count} HIGH 🟢 · "
                           f"{medium_count} MEDIUM 🟡 · "
                           f"{low_count} LOW 🔴"),
        "flows":          rows,
    }


# ─── Markdown section renderers ───────────────────────────────────────────────

def _section_v01(chk: dict) -> list[str]:
    lines = [
        "## V-01: Missing Business Flows",
        "",
        f"**Status:** {chk.get('icon','')} {chk.get('status','').upper()}  "
        f"│  **Missing:** {chk.get('missing_count', 0)}",
        "",
        f"> {chk.get('detail', '')}",
        "",
    ]
    items = chk.get("items", [])
    if items:
        lines += [
            "| Type | Identifier | File | Severity | Reason |",
            "|------|------------|------|----------|--------|",
        ]
        for it in items:
            fname = Path(it.get("file", "")).name or "—"
            lines.append(
                f"| {it.get('type','')} | `{it.get('identifier','')}` "
                f"| {fname} | {it.get('severity','')} | {it.get('reason','')} |"
            )
        lines.append("")
    return lines


def _section_v02(chk: dict) -> list[str]:
    lines = [
        "## V-02: Fake / Hallucinated Steps",
        "",
        f"**Status:** {chk.get('icon','')} {chk.get('status','').upper()}  "
        f"│  **Issues:** {chk.get('issue_count', 0)}",
        "",
        f"> {chk.get('detail', '')}",
        "",
    ]
    items = chk.get("items", [])
    if items:
        lines += [
            "| Flow | Step | Page | Reason |",
            "|------|------|------|--------|",
        ]
        for it in items:
            lines.append(
                f"| {it.get('flow_id','')} {it.get('flow_name','')} "
                f"| Step {it.get('step_num','')} "
                f"| `{it.get('step_page','')}` "
                f"| {it.get('reason','')} |"
            )
        lines.append("")
    return lines


def _section_v03(chk: dict) -> list[str]:
    lines = [
        "## V-03: Broken Flows",
        "",
        f"**Status:** {chk.get('icon','')} {chk.get('status','').upper()}  "
        f"│  **Broken:** {chk.get('broken_count', 0)}",
        "",
        f"> {chk.get('detail', '')}",
        "",
    ]
    items = chk.get("items", [])
    if items:
        lines += [
            "| Flow | Severity | Issues |",
            "|------|----------|--------|",
        ]
        for it in items:
            issue_str = "; ".join(it.get("issues", []))
            lines.append(
                f"| {it.get('flow_id','')} {it.get('flow_name','')} "
                f"| {it.get('severity','')} "
                f"| {issue_str} |"
            )
        lines.append("")
    return lines


def _section_v04(chk: dict) -> list[str]:
    lines = [
        "## V-04: Missing Branches",
        "",
        f"**Status:** {chk.get('icon','')} {chk.get('status','').upper()}  "
        f"│  **Affected flows:** {chk.get('branch_count', 0)}",
        "",
        f"> {chk.get('detail', '')}",
        "",
    ]
    items = chk.get("items", [])
    if items:
        lines += [
            "| Flow | Redirect Count | Known Redirects |",
            "|------|----------------|-----------------|",
        ]
        for it in items:
            redirs = ", ".join(f"`{r}`" for r in it.get("redirects", []))
            lines.append(
                f"| {it.get('flow_id','')} {it.get('flow_name','')} "
                f"| {it.get('redirect_count', 0)} "
                f"| {redirs} |"
            )
        lines.append("")
    return lines


def _section_cov(
    vid:      str,
    title:    str,
    chk:      dict,
    _kind:    str,
    col1_key: str,
    col2_key: str,
) -> list[str]:
    """Generic coverage section renderer for V-05, V-06, V-07."""
    pct = chk.get("pct", 0.0)
    lines = [
        f"## {vid}: {title}",
        "",
        f"**Status:** {chk.get('icon','')} {chk.get('status','').upper()}  "
        f"│  **Coverage:** {chk.get('covered',0)}/{chk.get('total',0)} "
        f"({pct:.0%})",
        "",
        f"> {chk.get('detail', '')}",
        "",
    ]
    uncovered = chk.get("uncovered", [])
    if uncovered:
        col1_title = col1_key.upper()
        col2_title = col2_key.upper()
        lines += [
            f"| {col1_title} | {col2_title} | File | Severity |",
            "|--------|-------|------|----------|",
        ]
        for it in uncovered[:20]:
            fname = Path(it.get("file", "")).name or "—"
            lines.append(
                f"| {it.get(col1_key, '')} "
                f"| `{it.get(col2_key, '')}` "
                f"| {fname} "
                f"| {it.get('severity', '')} |"
            )
        if len(uncovered) > 20:
            lines.append(f"| … | _{len(uncovered)-20} more not shown_ | | |")
        lines.append("")
    return lines


def _section_v08(chk: dict) -> list[str]:
    lines = [
        "## V-08: Per-Flow Confidence Scoring",
        "",
        f"**Average confidence:** {chk.get('avg_confidence', 0.0):.2f}  "
        f"│  {chk.get('detail', '')}",
        "",
        "| Flow | Grade | Confidence | Backed Steps | Notes |",
        "|------|-------|------------|-------------|-------|",
    ]
    for row in chk.get("flows", []):
        backed = f"{row['backed_steps']}/{row['total_steps']}"
        note   = row.get("note", "")
        lines.append(
            f"| {row['flow_id']} {row['flow_name']} "
            f"| {row['grade']} "
            f"| {row['confidence']:.2f} "
            f"| {backed} "
            f"| {note} |"
        )
    lines.append("")
    return lines


# ─── Internal pivot builders ──────────────────────────────────────────────────

def _build_flow_pivot(
    flow_list: list,
) -> tuple[set[str], set[str], set[str]]:
    """
    Build three sets from the generated flows.

    Returns
    -------
    flow_pages    : lowercase file basenames from step.page
    flow_paths    : lowercase raw step.page values (for route path matching)
    flow_evidence : lowercase file basenames from flow.evidence_files
    """
    pages:    set[str] = set()
    paths:    set[str] = set()
    evidence: set[str] = set()

    for flow in flow_list:
        for step in (flow.steps or []):
            raw = (step.page or "").split("?")[0].strip()
            if raw:
                pages.add(Path(raw).name.lower())
                paths.add(raw.lower())

        for fpath in (flow.evidence_files or []):
            if fpath:
                evidence.add(Path(fpath).name.lower())

    return pages, paths, evidence


def _build_known_files(cm: Any) -> set[str]:
    """Set of all lowercase file basenames known to the code_map."""
    files: set[str] = set()
    if cm is None:
        return files
    for lst in [
        cm.routes or [],
        cm.controllers or [],
        cm.services or [],
        cm.form_fields or [],
        cm.sql_queries or [],
        cm.redirects or [],
        cm.execution_paths or [],
    ]:
        for item in lst:
            if isinstance(item, dict) and item.get("file"):
                files.add(Path(item["file"]).name.lower())
    for p in (cm.html_pages or []):
        if isinstance(p, str):
            files.add(Path(p).name.lower())
    return files


def _build_known_route_paths(cm: Any) -> set[str]:
    """Set of lowercase registered route paths."""
    if cm is None:
        return set()
    paths: set[str] = set()
    for r in (cm.routes or []):
        p = (r.get("path") or r.get("uri") or "").lower()
        if p:
            paths.add(p)
    return paths


def _build_known_tables(cm: Any) -> set[str]:
    """Set of lowercase table names known from SQL queries and db_schema."""
    tables: set[str] = set()
    if cm is None:
        return tables
    for q in (cm.sql_queries or []):
        t = (q.get("table") or "").strip().lower()
        if t and t not in _SKIP_TABLES:
            tables.add(t)
    for schema in (cm.db_schema or []):
        t = (schema.get("table_name") or schema.get("table", "")).strip().lower()
        if t and t not in _SKIP_TABLES:
            tables.add(t)
    return tables


def _build_file_sql_index(cm: Any) -> dict[str, list[dict]]:
    """file path → list of SQL query dicts."""
    idx: dict[str, list[dict]] = defaultdict(list)
    if cm is None:
        return idx
    for q in (cm.sql_queries or []):
        if q.get("file"):
            idx[q["file"]].append(q)
    return idx


def _build_file_redir_index(cm: Any) -> dict[str, list[dict]]:
    """file path → list of redirect dicts."""
    idx: dict[str, list[dict]] = defaultdict(list)
    if cm is None:
        return idx
    for r in (cm.redirects or []):
        if r.get("file"):
            idx[r["file"]].append(r)
    return idx


def _route_in_flows(
    path:          str,
    rname:         str,
    flow_pages:    set[str],
    flow_paths:    set[str],
    flow_evidence: set[str],
) -> bool:
    """Return True if a route is covered by any flow step or evidence file."""
    path_lower = path.lower()
    return (
        (rname and rname in flow_pages)
        or (rname and rname in flow_evidence)
        or (path_lower and path_lower in flow_paths)
    )


def _cov_status(pct: float) -> tuple[str, str]:
    if pct >= _COV_PASS:
        return "pass", "✅"
    if pct >= _COV_WARN:
        return "warn", "⚠️"
    return "fail", "❌"
