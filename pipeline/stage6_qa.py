"""
pipeline/stage6_qa.py — QA Review Agent

Reviews all four BA artefacts (BRD, SRS, AC, User Stories) for quality,
consistency, and coverage against the domain model and codebase evidence.

What it does
------------
1. Loads all four artefact files
2. Calls the LLM with a QA reviewer system prompt
3. Parses the structured JSON review response
4. Populates ctx.qa_result with scores and issues
5. Writes a human-readable qa_report.md to the output directory

QA dimensions checked
---------------------
    Coverage    — Does every feature in the domain model appear in all docs?
    Consistency — Do page names, table names, field names match across docs?
    Completeness— Are required sections present in each document?
    Traceability— Can each AC/User Story be traced back to a BRD requirement?
    Correctness — Are there contradictions or obvious factual errors?

Output
------
    qa_report.md        Human-readable report with issues and recommendations
    ctx.qa_result       QAResult with scores and structured issue list

Scores (0.0–1.0)
    coverage_score    fraction of domain features covered across all docs
    consistency_score fraction of cross-doc references that are consistent

Resume behaviour
----------------
If stage6_qa is COMPLETED and qa_report.md exists, stage is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from context import PipelineContext, QAResult

QA_REPORT_FILE = "qa_report.md"
QA_JSON_FILE   = "qa_result.json"
MAX_TOKENS     = 4096

# How much of each artefact to send to the QA reviewer (chars)
# Sending full docs risks hitting token limits — send first N chars of each
ARTEFACT_PREVIEW_CHARS = 2500


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 6 entry point. Reviews all BA artefacts and populates ctx.qa_result.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If prerequisites are missing or LLM call fails.
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

    # ── Prerequisites ─────────────────────────────────────────────────────────
    _assert_prerequisites(ctx)

    print("  [stage6] Loading BA artefacts ...")
    artefacts = _load_artefacts(ctx)

    print(f"  [stage6] Loaded {len(artefacts)} artefact(s): "
          f"{', '.join(artefacts.keys())}")

    # ── Build prompts ─────────────────────────────────────────────────────────
    system_prompt = _build_system_prompt(ctx)
    user_prompt   = _build_user_prompt(ctx, artefacts)

    print(f"  [stage6] Calling LLM for QA review "
          f"({len(user_prompt):,} chars) ...")

    # ── Call LLM ──────────────────────────────────────────────────────────────
    from pipeline.llm_client import call_llm
    raw = call_llm(system_prompt, user_prompt,
                   max_tokens=MAX_TOKENS, label="stage6")

    # ── Parse response ────────────────────────────────────────────────────────
    qa_data = _parse_response(raw)

    # ── Build QAResult ────────────────────────────────────────────────────────
    coverage    = float(qa_data.get("coverage_score",    0.0))
    consistency = float(qa_data.get("consistency_score", 0.0))
    issues      = qa_data.get("issues", [])
    n_critical  = sum(1 for i in issues if i.get("severity") == "critical")
    n_major     = sum(1 for i in issues if i.get("severity") == "major")

    # Pass/fail decided in Python — never trust LLM for boolean verdicts.
    # Passes if: no critical issues, no major issues, coverage >= 0.8
    passed = (n_critical == 0 and n_major == 0 and coverage >= 0.8)

    ctx.qa_result = QAResult(
        passed            = passed,
        issues            = issues,
        coverage_score    = coverage,
        consistency_score = consistency,
    )

    # ── Write qa_report.md ────────────────────────────────────────────────────
    report_md = _build_report_md(qa_data, ctx)
    Path(report_path).write_text(report_md, encoding="utf-8")

    # ── Persist qa_result.json ────────────────────────────────────────────────
    with open(ctx.output_path(QA_JSON_FILE), "w", encoding="utf-8") as fh:
        json.dump(qa_data, fh, indent=2, ensure_ascii=False)

    ctx.stage("stage6_qa").mark_completed(report_path)
    ctx.save()

    _print_summary(ctx.qa_result)
    print(f"  [stage6] Report → {report_path}")


# ─── Prompt Building ───────────────────────────────────────────────────────────

def _build_system_prompt(ctx: PipelineContext | None = None) -> str:
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

Output ONLY valid JSON — no markdown fences, no preamble."""


def _build_user_prompt(ctx: PipelineContext, artefacts: dict[str, str]) -> str:
    domain = ctx.domain_model
    parts  = []

    # Domain model summary
    parts.append("=== DOMAIN MODEL (ground truth) ===")
    parts.append(f"System: {domain.domain_name}")
    parts.append(f"Features: {', '.join(f['name'] for f in domain.features)}")
    parts.append(f"Tables: {', '.join({t for f in domain.features for t in f.get('tables', [])})}")
    parts.append(f"Pages: {', '.join({p for f in domain.features for p in f.get('pages', [])})}")
    parts.append(f"Roles: {', '.join(r['role'] for r in domain.user_roles)}")
    parts.append(f"Entities: {', '.join(domain.key_entities)}")
    parts.append(f"Contexts: {', '.join(domain.bounded_contexts)}")

    # Artefact previews
    for name, content in artefacts.items():
        parts.append(f"\n=== {name.upper()} (first {ARTEFACT_PREVIEW_CHARS} chars) ===")
        parts.append(content[:ARTEFACT_PREVIEW_CHARS])
        if len(content) > ARTEFACT_PREVIEW_CHARS:
            parts.append(f"... [{len(content) - ARTEFACT_PREVIEW_CHARS} chars truncated]")

    parts.append("\n=== QA REVIEW TASK ===")
    parts.append(f"""Review the four artefacts above against the domain model and return this exact JSON:

{{
  "passed": true | false,
  "coverage_score": 0.0-1.0,
  "consistency_score": 0.0-1.0,
  "summary": "2-3 sentence overall assessment",
  "issues": [
    {{
      "severity": "critical | major | minor",
      "artefact": "BRD | SRS | AC | UserStories | cross-doc",
      "description": "Specific issue with reference to section/feature/field name",
      "recommendation": "Concrete fix"
    }}
  ],
  "coverage": {{
    "features_covered": ["list of feature names present in all docs"],
    "features_missing": ["list of feature names missing from any doc"]
  }},
  "strengths": ["list of 2-4 things done well"],
  "recommendations": ["list of 2-4 overall improvements"]
}}

Scoring guide:
  coverage_score    = features_covered / total_features
                      A feature counts as "covered" if it appears in AT LEAST
                      TWO of the four documents (BRD, SRS, AC, UserStories).
                      Trivial features (logout, static pages, no business logic)
                      count as covered if they appear in even ONE document with
                      at least 1 acceptance criterion or 1 user story.
                      Do NOT penalise trivial features for having only 1-2 AC
                      items — that is intentionally proportionate documentation.
  consistency_score = 1.0 if all page/table/field names match across docs,
                      reduce by 0.1 per inconsistency found
  passed            = true if no critical issues and coverage_score >= 0.8
                      (higher threshold now that trivial features are handled fairly)""")

    return "\n".join(parts)


# ─── Response Parsing ──────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict[str, Any]:
    """Parse LLM JSON response, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        )

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        # Attempt recovery
        from pipeline.stage4_domain import _attempt_json_recovery
        recovered = _attempt_json_recovery(text.strip())
        if recovered:
            print("  [stage6] Warning: JSON was truncated, recovered partial response.")
            return recovered
        raise RuntimeError(
            f"[stage6] Failed to parse QA response as JSON: {e}\n"
            f"Raw (first 300 chars): {raw[:300]}"
        )


# ─── Report Building ───────────────────────────────────────────────────────────

def _build_report_md(qa_data: dict, ctx: PipelineContext) -> str:
    """Render the QA result as a human-readable Markdown report."""
    domain   = ctx.domain_model
    passed   = qa_data.get("passed", False)
    cov      = qa_data.get("coverage_score", 0.0)
    con      = qa_data.get("consistency_score", 0.0)
    summary  = qa_data.get("summary", "")
    issues   = qa_data.get("issues", [])
    coverage = qa_data.get("coverage", {})
    strengths= qa_data.get("strengths", [])
    recs     = qa_data.get("recommendations", [])

    critical = [i for i in issues if i.get("severity") == "critical"]
    major    = [i for i in issues if i.get("severity") == "major"]
    minor    = [i for i in issues if i.get("severity") == "minor"]

    lines = [
        f"# QA Review Report — {domain.domain_name}",
        f"",
        f"**Status:** {'✅ PASSED' if passed else '❌ FAILED'}  ",
        f"**Coverage Score:** {cov:.0%}  ",
        f"**Consistency Score:** {con:.0%}  ",
        f"**Issues:** {len(critical)} critical · {len(major)} major · {len(minor)} minor",
        f"",
        f"## Summary",
        f"",
        summary,
        f"",
    ]

    # Coverage
    lines += [f"## Feature Coverage", ""]
    covered = coverage.get("features_covered", [])
    missing = coverage.get("features_missing", [])
    if covered:
        lines.append(f"**Covered ({len(covered)}):** "
                     + ", ".join(f"`{f}`" for f in covered))
    if missing:
        lines.append(f"\n**Missing ({len(missing)}):** "
                     + ", ".join(f"`{f}`" for f in missing))
    lines.append("")

    # Issues
    if issues:
        lines += ["## Issues", ""]
        for sev, group in [("🔴 Critical", critical),
                           ("🟠 Major",    major),
                           ("🟡 Minor",    minor)]:
            if group:
                lines.append(f"### {sev}")
                for i, issue in enumerate(group, 1):
                    lines += [
                        f"",
                        f"**{i}. [{issue.get('artefact','?')}]** {issue.get('description','')}",
                        f"*Recommendation:* {issue.get('recommendation','')}",
                    ]
                lines.append("")

    # Strengths
    if strengths:
        lines += ["## Strengths", ""]
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    # Recommendations
    if recs:
        lines += ["## Recommendations", ""]
        for r in recs:
            lines.append(f"- {r}")
        lines.append("")

    # Artefact paths
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

def _load_artefacts(ctx: PipelineContext) -> dict[str, str]:
    """Load all four BA artefact files. Raises if any are missing."""
    ba = ctx.ba_artifacts
    mapping = {
        "BRD":         ba.brd_path          if ba else None,
        "SRS":         ba.srs_path          if ba else None,
        "AC":          ba.ac_path           if ba else None,
        "UserStories": ba.user_stories_path if ba else None,
    }

    loaded = {}
    missing = []
    for name, path in mapping.items():
        if path and Path(path).exists():
            loaded[name] = Path(path).read_text(encoding="utf-8")
        else:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"[stage6] Missing artefact files: {missing}. "
            "Run Stage 5 first."
        )
    return loaded


def _assert_prerequisites(ctx: PipelineContext) -> None:
    if ctx.domain_model is None:
        raise RuntimeError(
            "[stage6] ctx.domain_model is None — run Stage 4 first."
        )
    if ctx.ba_artifacts is None:
        raise RuntimeError(
            "[stage6] ctx.ba_artifacts is None — run Stage 5 first."
        )


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

    status = "✅ PASSED" if qa.passed else "❌ FAILED"
    print(f"\n  {'=' * 54}")
    print(f"  QA Review {status}")
    print(f"  {'=' * 54}")
    print(f"  Coverage    : {qa.coverage_score:.0%}")
    print(f"  Consistency : {qa.consistency_score:.0%}")
    print(f"  Issues      : {critical} critical · {major} major · {minor} minor")
    if qa.issues:
        for issue in qa.issues:
            sev = issue.get("severity","?").upper()
            art = issue.get("artefact","?")
            desc = issue.get("description","")[:80]
            print(f"    [{sev}] {art}: {desc}")
    print(f"  {'=' * 54}\n")
