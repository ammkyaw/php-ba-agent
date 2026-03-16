"""
pipeline/stage6_qa.py — QA Review Agent

Reviews all four BA artefacts (BRD, SRS, AC, User Stories) for quality,
consistency, and coverage against the domain model and codebase evidence.

Two-pass strategy (CHANGED from single-pass)
--------------------------------------------
Problem: concatenating all 4 artefacts (~93k chars) + domain model into a
single prompt hit max_tokens on every run, causing truncated JSON and the
"Warning: JSON was truncated, recovered partial response" seen in the logs.

Solution: split into two focused calls, each well within the 4096-token
response budget:

  Pass A — Coverage pass (small prompt, ~4k chars):
    Input : domain model feature/entity/context lists + artefact *headings only*
    Output: coverage_score, features_covered, features_missing, summary

  Pass B — Consistency pass (medium prompt, ~15k chars):
    Input : first ARTEFACT_PREVIEW_CHARS of each artefact + domain model summary
    Output: consistency_score, issues list, strengths, recommendations

Results are merged into the same QAResult/report schema as before — no
downstream changes required.

QA dimensions checked
---------------------
    Coverage    — Does every domain feature appear in all docs?
    Consistency — Do page names, table names, field names match across docs?
    Completeness— Are required sections present in each document?
    Traceability— Can each AC/User Story be traced to a BRD requirement?
    Correctness — Are there contradictions or obvious factual errors?

Output
------
    qa_report.md        Human-readable Markdown report
    qa_result.json      Structured JSON for downstream processing
    ctx.qa_result       QAResult dataclass

Resume behaviour
----------------
If stage6_qa is COMPLETED and qa_report.md exists, stage is skipped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from context import PipelineContext, QAResult

QA_REPORT_FILE = "qa_report.md"
QA_JSON_FILE   = "qa_result.json"
MAX_TOKENS     = 8192

# Characters of each artefact sent to Pass B.
# 3000 × 4 = 12k + ~2k domain model = ~14k total — safe for a 4k response.
ARTEFACT_PREVIEW_CHARS = 3000

# Pass C — Reverse Audit
# Sample this many random PHP entry-point files and ask the LLM which domain
# model feature covers each (or "MISSING" if none does).
AUDIT_SAMPLE_SIZE  = 6
MAX_TOKENS_AUDIT   = 2_000


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 6 entry point. Reviews BA artefacts via two LLM passes and populates
    ctx.qa_result.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If prerequisites are missing.
    """
    report_path = ctx.output_path(QA_REPORT_FILE)

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage6_qa") and Path(report_path).exists():
        ctx.qa_result = _load_qa_result(ctx.output_path(QA_JSON_FILE))
        print(f"  [stage6] Already completed — "
              f"coverage={ctx.qa_result.coverage_score:.0%}, "
              f"consistency={ctx.qa_result.consistency_score:.0%}, "
              f"{len(ctx.qa_result.issues)} issue(s).")
        return

    _assert_prerequisites(ctx)

    print("  [stage6] Loading BA artefacts ...")
    artefacts = _load_artefacts(ctx)
    print(f"  [stage6] Loaded {len(artefacts)} artefact(s): "
          f"{', '.join(artefacts.keys())}")

    from pipeline.llm_client import call_llm
    from pipeline.consistency_check import run_checks as _run_consistency_checks
    from pipeline.consistency_check import format_summary as _consistency_summary
    from pipeline.cross_doc_check   import run_checks as _run_cross_doc_checks
    from pipeline.cross_doc_check   import format_summary as _cross_doc_summary
    from pipeline.evidence_index import build_evidence_index, build_confidence_report
    from pipeline.qa_checklist import (
        build_checklist as _build_checklist,
        checklist_overall_status as _checklist_status,
    )
    system_prompt = _build_system_prompt(ctx)

    # ── Confidence Scoring (no LLM) ───────────────────────────────────────────
    # Build once — used by Pass D (C-08), the checklist, and the QA report.
    conf_report: list[dict] = []
    ev_idx:      dict       = {}
    if ctx.domain_model and ctx.code_map:
        ev_idx      = build_evidence_index(ctx, ctx.domain_model)
        conf_report = build_confidence_report(ev_idx)
        n_low = sum(1 for r in conf_report if r["score"] < 0.4)
        n_hi  = sum(1 for r in conf_report if r["score"] >= 0.8)
        print(f"  [stage6] Confidence — {len(conf_report)} features scored "
              f"({n_hi} HIGH 🟢, "
              f"{len(conf_report)-n_hi-n_low} MEDIUM 🟡, "
              f"{n_low} LOW 🔴)")
        _save_confidence_json(ctx, conf_report)

    # ── Pass D: Deterministic Consistency Checks (no LLM) ─────────────────────
    # Runs BEFORE the LLM passes so findings appear at the top of the report
    # and can be referenced in the Pass B consistency prompt.
    print("  [stage6] Pass D — deterministic consistency checks ...")
    pass_d_issues = _run_consistency_checks(ctx, artefacts, conf_report)
    print(f"  [stage6] Pass D — {_consistency_summary(pass_d_issues)}")

    # ── Pass E: Cross-Document Consistency (no LLM) ───────────────────────────
    # Checks traceability between document pairs: orphan epics/AC sections,
    # undefined actors, and SRS→AC requirement gaps.  Issues are merged into
    # pass_d_issues so they flow through the same report path.
    print("  [stage6] Pass E — cross-document consistency checks ...")
    pass_e_issues = _run_cross_doc_checks(ctx, artefacts)
    print(f"  [stage6] Pass E — {_cross_doc_summary(pass_e_issues)}")
    pass_d_issues = pass_d_issues + pass_e_issues  # merge; D issues stay first

    # ── Automated QA Checklist (no LLM) ───────────────────────────────────────
    fc_data    = _load_flow_coverage(ctx)
    dc_data    = _load_doc_coverage(ctx)          # Stage 5.9 document coverage
    qa_checks  = _build_checklist(fc_data, pass_d_issues, ev_idx, dc_data=dc_data)
    cl_status  = _checklist_status(qa_checks)
    n_cl_pass  = sum(1 for c in qa_checks if c["status"] == "pass")
    n_cl_warn  = sum(1 for c in qa_checks if c["status"] == "warn")
    n_cl_fail  = sum(1 for c in qa_checks if c["status"] == "fail")
    print(f"  [stage6] Checklist — {len(qa_checks)} checks: "
          f"{n_cl_pass} ✅  {n_cl_warn} ⚠️  {n_cl_fail} ❌  "
          f"(overall: {cl_status.upper()})")
    _save_checklist_json(ctx, qa_checks)

    # ── Pass A: Coverage ──────────────────────────────────────────────────────
    pass_a_prompt = _build_coverage_prompt(ctx, artefacts)
    print(f"  [stage6] Pass A — coverage ({len(pass_a_prompt):,} chars) ...")
    raw_a     = call_llm(system_prompt, pass_a_prompt,
                         max_tokens=MAX_TOKENS, label="stage6_passA")
    pass_a_data = _parse_response(raw_a, pass_label="A")

    # ── Pass B: Consistency & Issues ─────────────────────────────────────────
    # pass_d_issues now includes Pass E cross-doc findings — injected here.
    pass_b_prompt = _build_consistency_prompt(ctx, artefacts, pass_d_issues)
    print(f"  [stage6] Pass B — consistency + issues ({len(pass_b_prompt):,} chars) ...")
    raw_b     = call_llm(system_prompt, pass_b_prompt,
                         max_tokens=MAX_TOKENS, label="stage6_passB")
    pass_b_data = _parse_response(raw_b, pass_label="B")

    # ── Pass C: Reverse Audit ─────────────────────────────────────────────────
    pass_c_data: dict[str, Any] = {}
    pass_c_prompt = _build_reverse_audit_prompt(ctx)
    if pass_c_prompt:
        print(f"  [stage6] Pass C — reverse audit ({len(pass_c_prompt):,} chars) ...")
        raw_c       = call_llm(system_prompt, pass_c_prompt,
                               max_tokens=MAX_TOKENS_AUDIT, label="stage6_passC")
        pass_c_data = _parse_response(raw_c, pass_label="C")
        n_missing   = sum(
            1 for e in pass_c_data.get("audit", [])
            if str(e.get("feature", "")).upper() == "MISSING"
        )
        print(f"  [stage6] Pass C — {len(pass_c_data.get('audit', []))} files audited, "
              f"{n_missing} MISSING from domain model.")
    else:
        print(f"  [stage6] Pass C — skipped (no entry-point pages found).")

    # ── Merge ─────────────────────────────────────────────────────────────────
    qa_data = _merge_passes(pass_a_data, pass_b_data, pass_c_data, pass_d_issues, ctx)

    # ── Build QAResult ────────────────────────────────────────────────────────
    coverage    = float(qa_data.get("coverage_score",    0.0))
    consistency = float(qa_data.get("consistency_score", 0.0))
    issues      = qa_data.get("issues", [])
    n_critical  = sum(1 for i in issues if i.get("severity") == "critical")
    n_major     = sum(1 for i in issues if i.get("severity") == "major")

    # Pass/fail determined here — never delegated to the LLM.
    passed = (n_critical == 0 and n_major == 0 and coverage >= 0.8)

    ctx.qa_result = QAResult(
        passed            = passed,
        issues            = issues,
        coverage_score    = coverage,
        consistency_score = consistency,
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    report_md = _build_report_md(qa_data, ctx, conf_report, qa_checks)
    Path(report_path).write_text(report_md, encoding="utf-8")

    with open(ctx.output_path(QA_JSON_FILE), "w", encoding="utf-8") as fh:
        json.dump(qa_data, fh, indent=2, ensure_ascii=False)

    ctx.stage("stage6_qa").mark_completed(report_path)
    ctx.save()

    _print_summary(ctx.qa_result)
    print(f"  [stage6] Report → {report_path}")


# ─── System Prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(ctx: PipelineContext | None = None) -> str:
    """Shared system prompt for both QA passes."""
    from pipeline.framework_hints import get_hints
    hints = get_hints(
        ctx.code_map.framework if ctx and ctx.code_map else "unknown"
    )
    return f"""You are a senior QA analyst reviewing Business Analysis documentation.
Your job is to check four BA artefacts (BRD, SRS, Acceptance Criteria, User Stories)
for quality, consistency, and coverage against a domain model.

Be objective and specific. Reference actual section names, feature names, table names,
and page names when raising issues. Do not invent problems that don't exist.

Framework-specific review focus:
{hints.qa_focus}

Output ONLY valid JSON — no markdown fences, no preamble, no trailing text."""


# ─── Pass A: Coverage Prompt ───────────────────────────────────────────────────

def _build_coverage_prompt(ctx: PipelineContext, artefacts: dict[str, str]) -> str:
    """
    Build the Pass A prompt.

    Sends only feature/entity/context lists from the domain model and
    markdown headings extracted from each artefact — intentionally small.
    """
    domain = ctx.domain_model
    parts  = ["=== DOMAIN MODEL ==="]
    parts.append(f"System: {domain.domain_name}")
    parts.append(f"Features ({len(domain.features)}): "
                 + ", ".join(f["name"] for f in domain.features))
    parts.append(f"Entities ({len(domain.key_entities)}): "
                 + ", ".join(domain.key_entities))
    parts.append(f"Bounded Contexts: " + ", ".join(domain.bounded_contexts))
    parts.append(f"User Roles: "       + ", ".join(r["role"] for r in domain.user_roles))

    parts.append("\n=== ARTEFACT HEADINGS ===")
    for name, content in artefacts.items():
        headings = _extract_headings(content)
        parts.append(f"\n{name} ({len(headings)} headings):")
        parts.append("\n".join(f"  • {h}" for h in headings[:30]))

    # ── Stage 5.9 gap injection ───────────────────────────────────────────────
    dc = getattr(ctx, "doc_coverage", None)
    if dc and dc.gap_summary:
        parts.append("\n=== STATIC COVERAGE GAPS — Stage 5.9 ===")
        parts.append(
            "These gaps were detected by deterministic analysis comparing "
            "pipeline signals (entities, flows, rules, states) to document text. "
            "Address each gap in your coverage assessment."
        )
        for gap in dc.gap_summary:
            parts.append(f"  • {gap}")
        parts.append(
            f"\nStatic coverage overall: {dc.overall_pct:.0%} "
            f"[{dc.overall_status.upper()}]"
        )

    parts.append("""
=== TASK ===
Check whether each feature/entity/bounded context from the domain model
appears as a heading or sub-heading in the artefacts.
Also explicitly reference any gaps from the STATIC COVERAGE GAPS section above
in your summary if they are significant.

Return ONLY this JSON (no other text):

{
  "coverage_score": 0.0-1.0,
  "summary": "2-3 sentence overall coverage assessment",
  "coverage": {
    "features_covered": ["feature names present in 2+ artefact headings"],
    "features_missing":  ["feature names absent from most artefacts"]
  }
}

Scoring: coverage_score = features_covered / total_features.
A feature is "covered" if its name (or a close synonym) appears in headings
in at least 2 of the 4 artefacts.  Trivial features (logout, static pages)
count if in even one artefact.""")

    return "\n".join(parts)


# ─── Pass B: Consistency + Issues Prompt ──────────────────────────────────────

def _build_consistency_prompt(
    ctx: PipelineContext,
    artefacts: dict[str, str],
    pass_d_issues: list[dict] | None = None,
) -> str:
    """
    Build the Pass B prompt.

    Sends the first ARTEFACT_PREVIEW_CHARS of each artefact — enough to spot
    naming mismatches, missing sections, and cross-doc contradictions without
    overflowing the token budget.

    pass_d_issues: deterministic issues already found by Pass D so the LLM
    can focus on novel issues rather than re-reporting the same ones.
    """
    domain = ctx.domain_model
    parts  = ["=== DOMAIN MODEL (ground truth) ==="]
    parts.append(f"System: {domain.domain_name}")
    parts.append(f"Features: " + ", ".join(f["name"] for f in domain.features))
    all_tables = ", ".join({t for f in domain.features for t in f.get("tables", [])})
    if all_tables:
        parts.append(f"Tables: {all_tables}")
    parts.append(f"Entities: " + ", ".join(domain.key_entities))
    parts.append(f"Roles: "    + ", ".join(r["role"] for r in domain.user_roles))

    # Inject already-known deterministic issues so the LLM doesn't duplicate them
    if pass_d_issues:
        parts.append(
            "\n=== ALREADY IDENTIFIED ISSUES (do NOT duplicate these) ==="
        )
        for iss in pass_d_issues:
            parts.append(
                f"  [{iss['severity'].upper()}] {iss['artefact']}: "
                f"{iss['description'][:120]}"
            )

    for name, content in artefacts.items():
        preview   = content[:ARTEFACT_PREVIEW_CHARS]
        truncated = len(content) > ARTEFACT_PREVIEW_CHARS
        parts.append(f"\n=== {name} (first {ARTEFACT_PREVIEW_CHARS:,} chars) ===")
        parts.append(preview)
        if truncated:
            parts.append(f"... [{len(content) - ARTEFACT_PREVIEW_CHARS:,} chars omitted]")

    parts.append(f"""
=== TASK ===
Check the four artefacts for:
  1. Naming inconsistencies — page names, table names, field names that differ
     between documents (e.g. "logout route" named differently in BRD vs AC)
  2. Missing required sections in any document
  3. Cross-document contradictions (e.g. an HTTP route specified differently)
  4. Entities from the domain model not mentioned in any artefact preview

Return ONLY this JSON (no other text):

{{
  "consistency_score": 0.0-1.0,
  "issues": [
    {{
      "severity": "critical | major | minor",
      "artefact": "BRD | SRS | AC | UserStories | cross-doc",
      "description": "Specific issue with section/feature/field name",
      "recommendation": "Concrete fix"
    }}
  ],
  "strengths": ["2-4 specific things done well"],
  "recommendations": ["2-4 high-impact overall improvements"]
}}

Scoring: consistency_score starts at 1.0.
  Deduct 0.20 per critical issue, 0.10 per major, 0.05 per minor.
  Minimum score is 0.0.""")

    return "\n".join(parts)


# ─── Pass C: Reverse Audit Prompt ─────────────────────────────────────────────

def _build_reverse_audit_prompt(ctx: PipelineContext) -> str | None:
    """
    Build the Pass C prompt.

    Samples AUDIT_SAMPLE_SIZE random PHP entry-point files and asks the LLM
    to name the domain model feature that covers each file, or "MISSING" if none
    does. Uses a deterministic random seed so reruns produce consistent samples.

    Returns None if no entry-point pages are available.
    """
    import random

    cm        = ctx.code_map
    all_pages = list(cm.html_pages or []) if cm else []
    if not all_pages:
        return None

    sample_size = min(AUDIT_SAMPLE_SIZE, len(all_pages))
    sampled     = random.Random(42).sample(all_pages, sample_size)

    domain         = ctx.domain_model
    feature_names  = [f["name"] for f in domain.features] if domain else []
    features_str   = "\n".join(f'  - "{n}"' for n in feature_names)
    files_str      = "\n".join(f'  - "{Path(p).name}"' for p in sampled)

    return f"""=== DOMAIN MODEL FEATURES ===
The domain model has {len(feature_names)} feature(s):
{features_str}

=== FILES TO AUDIT ===
The following {sample_size} PHP files were randomly sampled from the codebase:
{files_str}

=== TASK ===
For each file listed, identify which domain model feature (use the EXACT feature name
from the list above) best covers it. If no feature covers the file at all, answer
"MISSING".

Return ONLY this JSON:

{{
  "audit": [
    {{
      "file": "filename.php",
      "feature": "Exact Feature Name or MISSING",
      "confidence": "high | medium | low",
      "note": "optional 1-line explanation"
    }}
  ],
  "uncovered_count": 0,
  "audit_summary": "1-2 sentence summary of what the audit found"
}}"""


# ─── Response Parsing ──────────────────────────────────────────────────────────

def _parse_response(raw: str, pass_label: str = "") -> dict[str, Any]:
    """
    Parse LLM JSON response, stripping markdown fences.
    Falls back to partial JSON recovery on truncation.

    Args:
        raw:        Raw LLM output string.
        pass_label: "A" or "B" — used in log messages.

    Returns:
        Parsed dict, or empty dict on total failure (non-fatal — merging handles it).
    """
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        from pipeline.stage4_domain import _attempt_json_recovery
        recovered = _attempt_json_recovery(text)
        if recovered:
            print(f"  [stage6] Pass {pass_label}: truncated JSON — partial recovery OK.")
            return recovered
        print(f"  [stage6] Pass {pass_label}: JSON parse failed — using defaults.")
        return {}


def _extract_headings(content: str) -> list[str]:
    """Extract Markdown headings (## and ###) from a document for Pass A."""
    headings: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            heading = re.sub(r'^#+\s*', '', stripped).strip()
            if heading:
                headings.append(heading)
    return headings


# ─── Result Merging ────────────────────────────────────────────────────────────

def _merge_passes(
    pass_a:        dict[str, Any],
    pass_b:        dict[str, Any],
    pass_c:        dict[str, Any],
    pass_d_issues: list[dict],
    ctx:           PipelineContext,
) -> dict[str, Any]:
    """
    Merge Pass A (coverage) + Pass B (consistency+issues) + Pass C (reverse
    audit) + Pass D (deterministic consistency) into a single qa_data dict.

    Pass D issues are prepended so deterministic findings appear first.
    Defaults are provided for all keys so partial/failed passes produce a
    graceful fallback rather than a crash.
    """
    coverage_score    = float(pass_a.get("coverage_score",    0.5))
    consistency_score = float(pass_b.get("consistency_score", 0.9))
    # Prepend deterministic Pass D issues — they're always reliable
    issues            = list(pass_d_issues) + pass_b.get("issues", [])
    n_critical        = sum(1 for i in issues if i.get("severity") == "critical")
    n_major           = sum(1 for i in issues if i.get("severity") == "major")
    passed            = n_critical == 0 and n_major == 0 and coverage_score >= 0.8

    summary_a     = pass_a.get("summary", "")
    summary_b     = pass_b.get("summary", "")
    audit_summary = pass_c.get("audit_summary", "")
    combined      = " ".join(filter(None, [summary_a, summary_b, audit_summary])) or (
        f"Coverage: {coverage_score:.0%}. "
        f"Consistency: {consistency_score:.0%}. "
        f"{len(issues)} issue(s) found."
    )

    return {
        "passed":            passed,
        "coverage_score":    coverage_score,
        "consistency_score": consistency_score,
        "summary":           combined,
        "issues":            issues,
        "coverage":          pass_a.get("coverage", {
            "features_covered": [],
            "features_missing": [],
        }),
        "strengths":         pass_b.get("strengths", []),
        "recommendations":   pass_b.get("recommendations", []),
        # Pass C — Reverse Audit
        "audit":             pass_c.get("audit", []),
        "audit_summary":     audit_summary,
        "audit_uncovered":   int(pass_c.get("uncovered_count", 0)),
    }


# ─── Report Building ───────────────────────────────────────────────────────────

def _build_report_md(
    qa_data:     dict,
    ctx:         PipelineContext,
    conf_report: list[dict] | None = None,
    qa_checks:   list[dict] | None = None,
) -> str:
    """Render the QA result as a human-readable Markdown report."""
    domain        = ctx.domain_model
    passed        = qa_data.get("passed", False)
    cov           = qa_data.get("coverage_score", 0.0)
    con           = qa_data.get("consistency_score", 0.0)
    summary       = qa_data.get("summary", "")
    issues        = qa_data.get("issues", [])
    coverage      = qa_data.get("coverage", {})
    strengths     = qa_data.get("strengths", [])
    recs          = qa_data.get("recommendations", [])
    audit_entries = qa_data.get("audit", [])
    audit_summary = qa_data.get("audit_summary", "")

    critical = [i for i in issues if i.get("severity") == "critical"]
    major    = [i for i in issues if i.get("severity") == "major"]
    minor    = [i for i in issues if i.get("severity") == "minor"]

    # ── Per-dimension coverage metrics (from flow_coverage.json) ─────────────
    fc_data = _load_flow_coverage(ctx)
    _cov_labels = [
        ("exec_path", "Execution Path Coverage"),
        ("route",     "Route Coverage"),
        ("table",     "Database Coverage"),
        ("form",      "Form Coverage"),
    ]
    cov_rows: list[tuple[str, float]] = []   # (label, pct)
    for key, label in _cov_labels:
        block = fc_data.get(key, {})
        if isinstance(block, dict) and block.get("total", 0) > 0:
            cov_rows.append((label, float(block.get("pct", 0.0))))

    # ── Overall confidence = average of coverage metrics + avg feature score ──
    scores_for_overall: list[float] = [pct for _, pct in cov_rows]
    if conf_report:
        avg_feat = sum(r["score"] for r in conf_report) / len(conf_report)
        scores_for_overall.append(avg_feat)
    overall_conf = sum(scores_for_overall) / len(scores_for_overall) if scores_for_overall else cov

    # ── Status badge icons for coverage rows ─────────────────────────────────
    def _cov_icon(pct: float) -> str:
        return "✅" if pct >= 0.85 else "⚠️" if pct >= 0.50 else "❌"

    # ── Header block ──────────────────────────────────────────────────────────
    lines = [
        f"# QA Review Report — {domain.domain_name}",
        "",
        f"**Status:** {'✅ PASSED' if passed else '❌ FAILED'}  ",
        f"**Overall Confidence:** {overall_conf:.0%}  ",
        f"**Consistency Score:** {con:.0%}  ",
        f"**Issues:** {len(critical)} critical · {len(major)} major · {len(minor)} minor",
        "",
    ]

    # Per-dimension coverage table (only when flow_coverage data is available)
    if cov_rows:
        lines += ["## Coverage Breakdown", ""]
        col_w = max(len(lbl) for lbl, _ in cov_rows)
        for label, pct in cov_rows:
            bar = _coverage_bar(pct)
            icon = _cov_icon(pct)
            lines.append(
                f"| {label:<{col_w}} | {icon} {pct:>6.1%} | {bar} |"
            )
        lines.append("")
    else:
        # Fallback: single LLM coverage score when flow_coverage.json missing
        lines += [f"**Coverage Score:** {cov:.0%}  ", ""]

    lines += [
        "## Summary",
        "",
        summary,
        "",
    ]

    # ── Automated QA Checklist ────────────────────────────────────────────────
    if qa_checks:
        from pipeline.qa_checklist import format_checklist_md
        lines.append(format_checklist_md(qa_checks))

    lines += ["## Feature Coverage", ""]
    covered = coverage.get("features_covered", [])
    missing = coverage.get("features_missing", [])
    if covered:
        lines.append(f"**Covered ({len(covered)}):** "
                     + ", ".join(f"`{f}`" for f in covered))
    if missing:
        lines.append(f"\n**Missing ({len(missing)}):** "
                     + ", ".join(f"`{f}`" for f in missing))
    lines.append("")

    # ── Consistency Validation section (Pass D deterministic issues) ────────────
    det_issues = [i for i in issues if i.get("category")]
    if det_issues:
        lines += ["## Consistency Validation (Deterministic)", ""]
        # Group by category
        cats: dict[str, list] = {}
        for iss in det_issues:
            cats.setdefault(iss["category"], []).append(iss)
        for cat, cat_issues in cats.items():
            lines.append(f"### {cat}")
            for iss in cat_issues:
                sev_icon = {"critical": "🔴", "major": "🟠", "minor": "🟡"}.get(
                    iss.get("severity", ""), "•"
                )
                lines += [
                    "",
                    f"{sev_icon} **[{iss.get('severity','?').upper()}]** "
                    f"{iss.get('description','')}",
                    f"*Fix:* {iss.get('recommendation','')}",
                ]
            lines.append("")

    if issues:
        lines += ["## Issues", ""]
        for sev, group in [("🔴 Critical", critical),
                           ("🟠 Major",    major),
                           ("🟡 Minor",    minor)]:
            if group:
                lines.append(f"### {sev}")
                for i, issue in enumerate(group, 1):
                    lines += [
                        "",
                        f"**{i}. [{issue.get('artefact','?')}]** "
                        f"{issue.get('description','')}",
                        f"*Recommendation:* {issue.get('recommendation','')}",
                    ]
                lines.append("")

    if strengths:
        lines += ["## Strengths", ""]
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    if recs:
        lines += ["## Recommendations", ""]
        for r in recs:
            lines.append(f"- {r}")
        lines.append("")

    if audit_entries:
        lines += ["## Reverse Audit (Spot-Check)", ""]
        if audit_summary:
            lines.append(audit_summary)
            lines.append("")
        missing_files = [e for e in audit_entries
                         if str(e.get("feature", "")).upper() == "MISSING"]
        covered_files = [e for e in audit_entries
                         if str(e.get("feature", "")).upper() != "MISSING"]
        lines.append(
            f"**{len(covered_files)} of {len(audit_entries)} sampled file(s) covered** "
            f"· {len(missing_files)} MISSING from domain model"
        )
        lines.append("")
        lines.append("| File | Feature | Confidence | Note |")
        lines.append("|------|---------|-----------|------|")
        for entry in audit_entries:
            feat = entry.get("feature", "?")
            icon = "✅" if feat.upper() != "MISSING" else "❌"
            lines.append(
                f"| `{entry.get('file', '?')}` "
                f"| {icon} {feat} "
                f"| {entry.get('confidence', '?')} "
                f"| {entry.get('note', '')} |"
            )
        lines.append("")

    # ── Feature Confidence Scores ──────────────────────────────────────────────
    if conf_report:
        n_hi  = sum(1 for r in conf_report if r["score"] >= 0.8)
        n_med = sum(1 for r in conf_report if 0.4 <= r["score"] < 0.8)
        n_low = sum(1 for r in conf_report if r["score"] < 0.4)
        lines += [
            "## Feature Confidence Scores",
            "",
            f"Confidence measures how many of the 5 evidence types "
            f"(route, controller, SQL, form, execution path) back each feature.",
            f"",
            f"**{n_hi} HIGH 🟢** · **{n_med} MEDIUM 🟡** · **{n_low} LOW 🔴**",
            "",
            "| Feature | Score | Grade | Present | Missing |",
            "|---------|-------|-------|---------|---------|",
        ]
        for row in conf_report:
            present_str = ", ".join(row.get("present", [])) or "—"
            missing_str = ", ".join(row.get("missing", [])) or "—"
            lines.append(
                f"| {row['feature']} "
                f"| {row['score']:.1f} "
                f"| {row['grade']} "
                f"| {present_str} "
                f"| {missing_str} |"
            )
        lines.append("")

    if ctx.ba_artifacts:
        lines += ["## Reviewed Artefacts", ""]
        for label, path in [
            ("BRD",          ctx.ba_artifacts.brd_path),
            ("SRS",          ctx.ba_artifacts.srs_path),
            ("AC",           ctx.ba_artifacts.ac_path),
            ("User Stories", ctx.ba_artifacts.user_stories_path),
        ]:
            if path:
                lines.append(f"- **{label}:** `{Path(path).name}`")
        lines.append("")

    return "\n".join(lines)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _coverage_bar(pct: float, width: int = 20) -> str:
    """Render a compact ASCII progress bar for a coverage percentage."""
    filled = round(pct * width)
    empty  = width - filled
    return "█" * filled + "░" * empty


def _load_flow_coverage(ctx: PipelineContext) -> dict:
    """Load flow_coverage.json if it exists; return empty dict otherwise."""
    import os
    path = ctx.output_path("flow_coverage.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _load_doc_coverage(ctx: PipelineContext) -> dict:
    """
    Return Stage 5.9 doc coverage data from ctx.doc_coverage (preferred)
    or by loading doc_coverage.json from disk if the field is not hydrated.
    Returns empty dict when Stage 5.9 has not run.
    """
    import os
    dc = getattr(ctx, "doc_coverage", None)
    if dc is not None:
        # already hydrated — convert DimCoverage dicts to checklist-friendly shape
        return {d["dimension"]: d for d in (dc.dimensions or [])}
    path = ctx.output_path("doc_coverage.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return {d["dimension"]: d for d in data.get("dimensions", [])}
    return {}


def _save_checklist_json(ctx: PipelineContext, qa_checks: list[dict]) -> None:
    """Save the QA checklist as qa_checklist.json in the stage6 output dir."""
    path = ctx.output_path("qa_checklist.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(qa_checks, fh, indent=2, ensure_ascii=False)
    print(f"  [stage6] Checklist saved → {path}")


def _save_confidence_json(ctx: PipelineContext, conf_report: list[dict]) -> None:
    """Save the confidence report as confidence_scores.json in the stage6 output dir."""
    import json as _json
    path = ctx.output_path("confidence_scores.json")
    with open(path, "w", encoding="utf-8") as fh:
        _json.dump(conf_report, fh, indent=2, ensure_ascii=False)
    print(f"  [stage6] Confidence report saved → {path}")


def _load_artefacts(ctx: PipelineContext) -> dict[str, str]:
    """Load all four BA artefact files. Raises if any are missing."""
    ba      = ctx.ba_artifacts
    mapping = {
        "BRD":         ba.brd_path          if ba else None,
        "SRS":         ba.srs_path          if ba else None,
        "AC":          ba.ac_path           if ba else None,
        "UserStories": ba.user_stories_path if ba else None,
    }
    loaded  = {}
    missing = []
    for name, path in mapping.items():
        if path and Path(path).exists():
            loaded[name] = Path(path).read_text(encoding="utf-8")
        else:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"[stage6] Missing artefact files: {missing}. Run Stage 5 first."
        )
    return loaded


def _assert_prerequisites(ctx: PipelineContext) -> None:
    if ctx.domain_model is None:
        raise RuntimeError("[stage6] ctx.domain_model is None — run Stage 4 first.")
    if ctx.ba_artifacts is None:
        raise RuntimeError("[stage6] ctx.ba_artifacts is None — run Stage 5 first.")


def _load_qa_result(json_path: str) -> QAResult:
    """Restore QAResult from saved JSON."""
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return QAResult(
            passed            = data.get("passed", True),
            issues            = data.get("issues", []),
            coverage_score    = float(data.get("coverage_score", 0.0)),
            consistency_score = float(data.get("consistency_score", 0.0)),
        )
    except Exception:
        return QAResult()


def _print_summary(qa: QAResult) -> None:
    critical = sum(1 for i in qa.issues if i.get("severity") == "critical")
    major    = sum(1 for i in qa.issues if i.get("severity") == "major")
    minor    = sum(1 for i in qa.issues if i.get("severity") == "minor")
    status   = "✅ PASSED" if qa.passed else "❌ FAILED"

    print(f"\n  {'=' * 54}")
    print(f"  QA Review {status}")
    print(f"  {'=' * 54}")
    print(f"  Coverage    : {qa.coverage_score:.0%}")
    print(f"  Consistency : {qa.consistency_score:.0%}")
    print(f"  Issues      : {critical} critical · {major} major · {minor} minor")
    if qa.issues:
        for issue in qa.issues[:10]:
            sev  = issue.get("severity", "?").upper()
            art  = issue.get("artefact", "?")
            desc = issue.get("description", "")[:80]
            print(f"    [{sev}] {art}: {desc}")
        if len(qa.issues) > 10:
            print(f"    ... and {len(qa.issues) - 10} more in qa_report.md")
    print(f"  {'=' * 54}\n")
