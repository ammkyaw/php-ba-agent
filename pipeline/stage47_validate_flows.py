"""
pipeline/stage47_validate_flows.py — Behavioral Flow Validation (Stage 4.7)

Runs between Stage 4.5 (business flow extraction) and Stage 5 (BA documents)
so that any validation findings can be referenced in the final reports.

Answers three critical questions:
  1. Are we missing business flows?          (V-01)
  2. Are generated flows actually valid?     (V-02, V-03, V-04)
  3. Do flows match real execution paths?    (V-05, V-06, V-07, V-08)

No LLM is invoked — all eight checks are deterministic.

Outputs
-------
  flow_validation.json   →   4.7_validation/
  flow_validation.md     →   4.7_validation/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.flow_validator import format_report_md, run_validation

# Grade → console icon
_GRADE_ICON = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}


def run(ctx: Any) -> None:
    """Stage 4.7 entry point — called by run_pipeline.py."""
    flows = getattr(ctx, "business_flows", None)
    cm    = getattr(ctx, "code_map", None)

    if not flows or not flows.flows:
        print("  [stage47] ⚠️  No business flows found — skipping behavioral validation.")
        return

    n_flows    = len(flows.flows)
    n_routes   = len(getattr(cm, "routes",       None) or [])
    n_forms    = len(getattr(cm, "form_fields",   None) or [])
    n_sql      = len(getattr(cm, "sql_queries",   None) or [])
    n_eps      = len(getattr(cm, "execution_paths", None) or [])

    print(f"  [stage47] Behavioral validation:")
    print(f"            {n_flows} flows  ·  {n_routes} routes  ·  "
          f"{n_forms} forms  ·  {n_sql} SQL queries  ·  {n_eps} exec-paths")

    # ── Run validation ─────────────────────────────────────────────────────────
    report = run_validation(ctx)
    s      = report["summary"]
    grade  = s["overall_grade"]
    icon   = _GRADE_ICON.get(grade, "?")

    # ── Print check-by-check summary ──────────────────────────────────────────
    print(f"  [stage47] {icon} Overall: {grade}  "
          f"(score {s['overall_score']:.0%} — "
          f"{s['checks_passed']} ✅ passed · "
          f"{s['checks_warned']} ⚠️  warned · "
          f"{s['checks_failed']} ❌ failed)")

    checks = report["checks"]
    _print_check(checks, "V-01", "Missing Business Flows",    "missing_count",  "missing")
    _print_check(checks, "V-02", "Fake / Hallucinated Steps", "issue_count",    "issues")
    _print_check(checks, "V-03", "Broken Flows",              "broken_count",   "broken")
    _print_check(checks, "V-04", "Missing Branches",          "branch_count",   "flows")
    _print_cov  (checks, "V-05", "Route Coverage")
    _print_cov  (checks, "V-06", "DB Write Coverage")
    _print_cov  (checks, "V-07", "Form Coverage")
    _print_conf (checks, "V-08")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_path = ctx.output_path("flow_validation.json")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # ── Save Markdown ─────────────────────────────────────────────────────────
    md_path = ctx.output_path("flow_validation.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(format_report_md(report))

    print(f"  [stage47] Saved → {json_path}")
    print(f"  [stage47] Saved → {md_path}")

    # Attach summary to ctx for downstream stages (optional consumption)
    ctx.flow_validation = report


# ─── Console helpers ──────────────────────────────────────────────────────────

def _print_check(
    checks:    dict,
    vid:       str,
    label:     str,
    count_key: str,
    noun:      str,
) -> None:
    chk  = checks.get(vid, {})
    icon = chk.get("icon", "")
    n    = chk.get(count_key, 0)
    status = chk.get("status", "pass")
    if status == "pass":
        print(f"            {icon} {vid} {label}: {n} {noun}")
    else:
        detail = chk.get("detail", "")
        print(f"            {icon} {vid} {label}: {detail}")


def _print_cov(checks: dict, vid: str, label: str) -> None:
    chk  = checks.get(vid, {})
    icon = chk.get("icon", "")
    pct  = chk.get("pct", 0.0)
    cov  = chk.get("covered", 0)
    tot  = chk.get("total", 0)
    print(f"            {icon} {vid} {label}: {cov}/{tot} ({pct:.0%})")


def _print_conf(checks: dict, vid: str) -> None:
    chk  = checks.get(vid, {})
    icon = chk.get("icon", "")
    avg  = chk.get("avg_confidence", 0.0)
    hi   = chk.get("high_count", 0)
    med  = chk.get("medium_count", 0)
    lo   = chk.get("low_count", 0)
    print(f"            {icon} {vid} Flow Confidence: "
          f"avg {avg:.2f}  ·  {hi} HIGH 🟢 · {med} MEDIUM 🟡 · {lo} LOW 🔴")
