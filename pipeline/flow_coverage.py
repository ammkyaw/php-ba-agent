"""
pipeline/flow_coverage.py — Business Flow Coverage Metrics

Measures how much of the codebase's observable behaviour is represented
in the generated business flows (Stage 4.5).

Four metrics
------------
  1. Execution-Path Coverage
       pages_in_flows / pages_with_execution_paths
       How many PHP files that have a static-analysis path (Stage 1.5)
       appear in at least one flow step.

  2. Route Coverage
       route_files_in_flows / total_route_files
       How many files that register HTTP routes appear in flow steps.

  3. Database Table Coverage
       tables_touched_by_flow_pages / total_unique_tables
       How many DB tables (from sql_queries) are queried by files
       that appear in flow steps.

  4. Form Coverage
       form_files_in_flows / total_form_files
       How many HTML-form-hosting files appear in flow steps.

Join key
--------
All four metrics share the same pivot:
    flow_pages = { Path(step.page).name.lower()
                   for flow in generated_flows
                   for step in flow.steps }

This is a conservative, file-level match. A file "appears in flows"
if its basename (case-insensitive) appears as a step page somewhere in
the generated flow collection.

Thresholds
----------
  Flag as WARNING when metric < WARNING_THRESHOLD (default 85 %)
  Flag as CRITICAL when metric < CRITICAL_THRESHOLD (default 50 %)

Output
------
  Printed summary table to stdout.
  JSON saved to  <run_dir>/4.5_flows/flow_coverage.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _extract_module_name(filepath: str) -> str | None:
    """Return the module directory name from a SugarCRM/raw-PHP path, or None."""
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part.lower() == "modules" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _module_expand(exec_paths: list[dict], flow_pages: set[str]) -> set[str]:
    """
    Expand *flow_pages* to include all sibling files of any file whose module
    directory is already represented in *flow_pages*.

    Also applies directory-level expansion for non-module files: if any file
    in ``include/CalendarProvider/`` is referenced, all siblings in that
    directory are counted as covered.
    """
    module_files: dict[str, list[str]] = defaultdict(list)
    dir_files: dict[str, list[str]] = defaultdict(list)

    for ep in exec_paths:
        f = ep.get("file", "")
        if not f:
            continue
        mod = _extract_module_name(f)
        if mod:
            module_files[mod].append(Path(f).name.lower())
        else:
            parent = str(Path(f).parent)
            dir_files[parent].append(Path(f).name.lower())

    covered_modules: set[str] = set()
    for mod, files in module_files.items():
        if any(f in flow_pages for f in files):
            covered_modules.add(mod)

    covered_dirs: set[str] = set()
    for parent, files in dir_files.items():
        if any(f in flow_pages for f in files):
            covered_dirs.add(parent)

    expanded: set[str] = set(flow_pages)
    for mod in covered_modules:
        expanded.update(module_files[mod])
    for parent in covered_dirs:
        expanded.update(dir_files[parent])
    return expanded

# ── Thresholds ────────────────────────────────────────────────────────────────
WARNING_THRESHOLD  = 0.85   # below this → ⚠️  warning
CRITICAL_THRESHOLD = 0.50   # below this → 🔴 critical

# How many missing items to include in the JSON report (prevents huge files)
MAX_MISSING_IN_REPORT = 30

# ─── Public entry point ───────────────────────────────────────────────────────

def compute_and_save(ctx: Any) -> dict[str, Any]:
    """
    Compute all four coverage metrics, print a summary table, and save
    the result to ``4.5_flows/flow_coverage.json``.

    Parameters
    ----------
    ctx : PipelineContext

    Returns
    -------
    dict with keys: exec_path, route, table, form
    Each value is a dict: label, covered, total, pct, missing, status
    """
    report = _compute(ctx)
    _print_summary(report)
    _save(ctx, report)
    return report


# ─── Computation ─────────────────────────────────────────────────────────────

def _compute(ctx: Any) -> dict[str, Any]:
    cm    = getattr(ctx, "code_map",       None)
    flows = getattr(ctx, "business_flows", None)

    # ── Collect pages that appear in flow steps ───────────────────────────────
    flow_pages: set[str] = set()
    if flows and flows.flows:
        for flow in flows.flows:
            for step in flow.steps:
                if step.page:
                    # Normalise: strip query strings, lower-case
                    raw = step.page.split("?")[0].strip()
                    if raw:
                        flow_pages.add(Path(raw).name.lower())

    # ── Module-expanded flow_pages ────────────────────────────────────────────
    # If a flow references any file from a module dir, count all sibling files
    # in that module as covered (module-level granularity).
    exec_paths_list = list(cm.execution_paths or []) if cm else []
    module_flow_pages = _module_expand(exec_paths_list, flow_pages)

    # ── 1. Execution-path coverage ────────────────────────────────────────────
    ep_all: set[str] = set()
    if cm and cm.execution_paths:
        ep_all = {Path(ep["file"]).name.lower() for ep in cm.execution_paths
                  if ep.get("file")}
    ep_covered = ep_all & module_flow_pages
    ep_missing = sorted(ep_all - module_flow_pages)

    # ── 2. Route coverage ─────────────────────────────────────────────────────
    route_files: set[str] = set()
    _SKIP_METHODS = {"GROUP", "MIDDLEWARE", "PREFIX", "MIDDLEWARE_GROUP"}
    if cm and cm.routes:
        route_files = {
            Path(r["file"]).name.lower()
            for r in cm.routes
            if r.get("file")
            and (r.get("method") or "GROUP").upper() not in _SKIP_METHODS
        }
    rt_covered = route_files & module_flow_pages
    rt_missing = sorted(route_files - module_flow_pages)

    # ── 3. Database table coverage ────────────────────────────────────────────
    # All unique tables in the codebase
    all_tables: set[str] = set()
    if cm and cm.sql_queries:
        all_tables = {
            q["table"].lower()
            for q in cm.sql_queries
            if q.get("table") and q["table"].upper() not in ("", "UNKNOWN")
        }

    # Tables reachable by joining sql_query['file'] → module_flow_pages
    tables_in_flows: set[str] = set()
    if cm and cm.sql_queries and module_flow_pages:
        tables_in_flows = {
            q["table"].lower()
            for q in cm.sql_queries
            if (q.get("table") and q["table"].upper() not in ("", "UNKNOWN")
                and Path(q.get("file", "")).name.lower() in module_flow_pages)
        }
    tbl_covered = tables_in_flows
    tbl_missing = sorted(all_tables - tables_in_flows)

    # ── 4. Form coverage ──────────────────────────────────────────────────────
    form_files: set[str] = set()
    if cm and cm.form_fields:
        form_files = {
            Path(ff["file"]).name.lower()
            for ff in cm.form_fields
            if ff.get("file")
        }
    fm_covered = form_files & module_flow_pages
    fm_missing = sorted(form_files - module_flow_pages)

    # ── Assemble report ───────────────────────────────────────────────────────
    def _entry(label: str, covered: set, total: set, missing: list) -> dict:
        n_cov   = len(covered)
        n_total = len(total)
        pct     = n_cov / n_total if n_total else 0.0
        if pct >= WARNING_THRESHOLD:
            status = "ok"
        elif pct >= CRITICAL_THRESHOLD:
            status = "warning"
        else:
            status = "critical"
        return {
            "label":   label,
            "covered": n_cov,
            "total":   n_total,
            "pct":     round(pct, 4),
            "status":  status,
            "missing": missing[:MAX_MISSING_IN_REPORT],
        }

    return {
        "flow_pages_count": len(flow_pages),
        "exec_path": _entry(
            "Execution-Path Coverage",
            ep_covered, ep_all, ep_missing,
        ),
        "route": _entry(
            "Route Coverage",
            rt_covered, route_files, rt_missing,
        ),
        "table": _entry(
            "Database Table Coverage",
            tbl_covered, all_tables, tbl_missing,
        ),
        "form": _entry(
            "Form Coverage",
            fm_covered, form_files, fm_missing,
        ),
    }


# ─── Output helpers ──────────────────────────────────────────────────────────

def _status_icon(status: str) -> str:
    return {"ok": "✅", "warning": "⚠️ ", "critical": "🔴"}.get(status, "?")


def _print_summary(report: dict[str, Any]) -> None:
    """Print a concise coverage table to stdout."""
    n_flow_pages = report.get("flow_pages_count", 0)
    print(f"\n  [stage45] Flow Coverage Report "
          f"(flow step pages: {n_flow_pages})")
    print(f"  {'─' * 60}")
    print(f"  {'Metric':<32} {'Covered':>8} {'Total':>8} {'Coverage':>10}  ")
    print(f"  {'─' * 60}")

    for key in ("exec_path", "route", "table", "form"):
        m = report[key]
        icon = _status_icon(m["status"])
        pct_str = f"{m['pct']:.0%}"
        print(
            f"  {icon} {m['label']:<30} "
            f"{m['covered']:>8,} {m['total']:>8,} {pct_str:>10}"
        )

    print(f"  {'─' * 60}")

    # Flag warnings / critical
    flags = [
        (report[k]["label"], report[k]["status"])
        for k in ("exec_path", "route", "table", "form")
        if report[k]["status"] != "ok"
    ]
    if flags:
        print()
        for label, status in flags:
            icon = _status_icon(status)
            threshold = (
                f"< {WARNING_THRESHOLD:.0%}"
                if status == "warning"
                else f"< {CRITICAL_THRESHOLD:.0%}"
            )
            print(f"  [stage45] {icon}  {label} {threshold} — "
                  f"missing flows may exist")
    else:
        print(f"  [stage45] ✅  All coverage metrics above {WARNING_THRESHOLD:.0%}")
    print()


def _save(ctx: Any, report: dict[str, Any]) -> None:
    """Persist the coverage report to 4.5_flows/flow_coverage.json."""
    try:
        output_path = ctx.output_path("flow_coverage.json")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"  [stage45] ⚠️  Could not save flow_coverage.json: {exc}")
