"""
pipeline/stage5_critic.py — Stage 5.0 Critic-Augmented Document Refinement

Provides the Critic loop that runs inline within Stage 5 document generation.
No new pipeline stage entry — this module is imported and called by stage5_workers.

Architecture
------------
For each generated BA document (BRD / SRS / AC / User Stories):

    Turn 1  ─── writer produces initial draft (existing stage5_workers logic)
                        │
                        ▼
    Critic call ─── CriticAgent.critique(doc_type, draft, ctx)
                    └─ structured JSON: {score, uncovered_rules,
                                         hallucinated_entities,
                                         structural_issues, rewrite_hints}
                        │
                    score ≥ CRITIC_THRESHOLD? ──────────────────► accept draft
                        │ no
                        ▼
    Turn 2  ─── Refiner call  (targeted surgical edit, NOT full regeneration)
                refine(draft, critic_pass) → corrected draft
                        │
                        ▼
                    accept (regardless of turn-2 score)

Critic design
-------------
The Critic is constrained: it checks against our existing static catalogs,
not open-ended quality. Three checks only:

  1. Rule coverage   — are STRONG/MODERATE spec rules addressed in the draft?
                       (keyword match on rule title + entity names — fast, cheap)
  2. Entity accuracy — does the draft mention entity names absent from catalog?
                       (exact-match against ctx.entities table names)
  3. Structure       — are the required section markers present?
                       (regex scan for known heading patterns per doc type)

The LLM Critic receives a compact catalog snapshot (≈600 tokens) + the draft
(truncated to 5 000 chars if needed) and returns JSON only.

The LLM Refiner receives the draft + critic hints and makes targeted edits.
It does NOT rebuild from scratch — only touches flagged sections.

Zero new pipeline-level stage.  Adds 0–2 LLM calls per document:
    • 0 extra calls  — draft passes on turn 1 (critic ≥ CRITIC_THRESHOLD)
    • 1 extra call   — critic LLM only (draft scored, no refinement needed)
    • 2 extra calls  — critic + refiner (draft failed, one targeted fix)

Downstream integration
----------------------
  ctx.ba_artifacts.critic_passes  — dict[doc_key, list[CriticPass]]
  consensus_check(ctx)             — deterministic post-Stage-5.5 gate
  critic_summary_block(ctx)        — Markdown for Stage 6 QA context injection
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Optional

from context import CriticPass, PipelineContext

# ── Constants ──────────────────────────────────────────────────────────────────

# Score threshold below which the refiner is triggered
CRITIC_THRESHOLD = 0.75

# Max chars of the draft fed to the critic (keeps cost down; critic needs structure,
# not verbatim content of every paragraph)
CRITIC_DRAFT_CHARS = 6_000

# Max tokens for critic and refiner calls
CRITIC_MAX_TOKENS  = 768
REFINER_MAX_TOKENS = 8_192

# Required structural markers per doc type (case-insensitive substring match)
_DOC_REQUIRED_MARKERS: dict[str, list[str]] = {
    "brd": ["Executive Summary", "Business Objectives", "Business Requirements"],
    "srs": ["Functional Requirements", "Non-Functional Requirements"],
    "ac":  ["Acceptance Criteria", "AC-"],
    "us":  ["Epic", "US-", "Story Points"],
}

# Spec rule categories relevant per doc type (None = all)
_DOC_RULE_CATEGORIES: dict[str, Optional[list[str]]] = {
    "brd": None,
    "srs": ["VALIDATION", "REFERENTIAL", "STATE", "STATE_TRANSITION", "BUSINESS_LIMIT"],
    "ac":  None,
    "us":  ["WORKFLOW", "AUTHORIZATION"],
}

# Doc labels for log messages
_DOC_LABELS = {"brd": "BRD", "srs": "SRS", "ac": "AC", "us": "User Stories"}


# ── Public API ─────────────────────────────────────────────────────────────────

def run_critic_loop(
    doc_type:  str,
    draft:     str,
    ctx:       PipelineContext,
    max_turns: int = 2,
) -> tuple[str, list[CriticPass]]:
    """
    Run the Critic loop on a single BA document draft.

    Parameters
    ----------
    doc_type  : "brd" | "srs" | "ac" | "us"
    draft     : Full text of the initial writer output.
    ctx       : Pipeline context (read-only during critic loop).
    max_turns : Maximum number of refinement turns (default 2 = 1 critic + 1 refine).

    Returns
    -------
    (final_document, list_of_CriticPass)
    The final_document is the (possibly refined) text ready for disk write.
    """
    passes: list[CriticPass] = []
    current = draft
    label   = _DOC_LABELS.get(doc_type, doc_type.upper())

    for turn in range(1, max_turns + 1):
        cp = _critique(doc_type, current, ctx, turn)
        passes.append(cp)

        print(
            f"  [stage5_critic] {label} turn {turn}: "
            f"score={cp.score:.2f} "
            f"({'PASS' if cp.passed else 'FAIL'})"
        )
        if cp.uncovered_rule_ids:
            print(f"    ↳ uncovered rules: {cp.uncovered_rule_ids[:5]}")
        if cp.hallucinated_entities:
            print(f"    ↳ hallucinated entities: {cp.hallucinated_entities[:5]}")
        if cp.structural_issues:
            print(f"    ↳ structural issues: {cp.structural_issues}")

        if cp.passed:
            break

        if turn == max_turns:
            # Exhausted turns — keep current draft as-is; log the residual issues
            print(
                f"  [stage5_critic] {label} max turns reached — "
                f"accepting draft with score {cp.score:.2f}"
            )
            break

        # Enrich hints with execution paths for uncovered rules
        _enrich_hints_with_exec_paths(cp, ctx)

        # Refine: targeted surgical edit
        print(f"  [stage5_critic] {label} → triggering refiner pass …")
        current = _refine(current, cp)

    return current, passes


def consensus_check(ctx: PipelineContext) -> dict:
    """
    Deterministic post-Stage-5.5 quality gate.

    Checks:
      - All STRONG-triangulated spec rules have document coverage
      - Average code links per requirement ≥ 1.0
      - Overall traceability coverage across BRD+SRS+AC+US ≥ 80%

    Returns a dict with keys: passed, reason, uncited_strong_ids,
    overall_coverage, avg_code_links.
    """
    if not ctx.traceability_meta:
        return {
            "passed": False,
            "reason": "Traceability matrix not built (Stage 5.5 not run).",
            "uncited_strong_ids": [],
            "overall_coverage": 0.0,
            "avg_code_links": 0.0,
        }

    tm = ctx.traceability_meta

    # Identify STRONG rules that are still uncited
    uncited_strong_ids: list[str] = []
    if ctx.triangulation:
        strong_ids = {
            r.rule_id
            for r in ctx.triangulation.rules
            if r.triangulation_status == "STRONG"
        }
        uncited_strong_ids = [
            rid for rid in tm.uncited_ids
            if rid.startswith("TR-") and rid[3:] in strong_ids
        ]

    overall_coverage = min(tm.covered_brd, tm.covered_srs, tm.covered_ac, tm.covered_us)
    passed = (
        len(uncited_strong_ids) == 0
        and tm.avg_code_links >= 1.0
        and overall_coverage >= 0.80
    )

    reasons: list[str] = []
    if uncited_strong_ids:
        reasons.append(
            f"{len(uncited_strong_ids)} STRONG rules uncited "
            f"({', '.join(uncited_strong_ids[:5])})"
        )
    if tm.avg_code_links < 1.0:
        reasons.append(f"avg_code_links={tm.avg_code_links:.2f} < 1.0")
    if overall_coverage < 0.80:
        reasons.append(f"overall_coverage={overall_coverage:.1%} < 80%")

    return {
        "passed":             passed,
        "reason":             "; ".join(reasons) if reasons else "All checks passed.",
        "uncited_strong_ids": uncited_strong_ids,
        "overall_coverage":   round(overall_coverage, 3),
        "avg_code_links":     tm.avg_code_links,
    }


def critic_summary_block(ctx: PipelineContext) -> str:
    """
    Return a Markdown block summarising all Critic passes.
    Injected into Stage 6 QA context so it knows what was already checked.
    Empty string if no critic passes were run.
    """
    if not ctx.ba_artifacts or not ctx.ba_artifacts.critic_passes:
        return ""

    lines: list[str] = ["## Critic-Loop Summary (Stage 5.0)\n"]
    lines.append(
        "_Documents were refined inline. Issues listed here were addressed "
        "before Stage 6 review._\n"
    )

    for doc_key, pass_list in ctx.ba_artifacts.critic_passes.items():
        label = _DOC_LABELS.get(doc_key, doc_key.upper())
        for p in pass_list:
            status = "✓ PASSED" if p["passed"] else f"✗ score={p['score']:.2f}"
            lines.append(f"### {label} — Turn {p['turn']} ({status})")
            if p.get("uncovered_rule_ids"):
                lines.append(
                    f"- Uncovered rules: {', '.join(p['uncovered_rule_ids'][:8])}"
                )
            if p.get("hallucinated_entities"):
                lines.append(
                    f"- Hallucinated entities: {', '.join(p['hallucinated_entities'])}"
                )
            if p.get("structural_issues"):
                for issue in p["structural_issues"]:
                    lines.append(f"- ⚠ {issue}")
            if p.get("rewrite_hints") and not p["passed"]:
                lines.append(f"- Rewrite hints applied: {len(p['rewrite_hints'])}")
            lines.append("")

    return "\n".join(lines)


def critic_context_block(ctx: PipelineContext, doc_type: str) -> str:
    """
    Build the compact catalog snapshot given to the Critic LLM.

    Kept under ~600 tokens.  Contains only the data the critic needs to
    check coverage — not the full domain model (writer already saw that).
    """
    lines: list[str] = ["=== CATALOG SNAPSHOT (for coverage audit) ===\n"]

    # ── Spec Rules ─────────────────────────────────────────────────────────
    if ctx.spec_rules and ctx.spec_rules.rules:
        cats = _DOC_RULE_CATEGORIES.get(doc_type)
        rules = ctx.spec_rules.rules
        if cats:
            rules = [r for r in rules if r.category in cats]

        tri_status: dict[str, str] = {}
        if ctx.triangulation:
            tri_status = {
                r.rule_id: r.triangulation_status
                for r in ctx.triangulation.rules
            }

        strong   = [r for r in rules if tri_status.get(r.rule_id) == "STRONG"]
        moderate = [r for r in rules if tri_status.get(r.rule_id) == "MODERATE"]
        other    = [r for r in rules if r.rule_id not in tri_status]

        lines.append(f"SPEC RULES [{len(rules)} total]:")
        if strong:
            lines.append("  STRONG — must be covered:")
            for r in strong[:15]:
                lines.append(f"    {r.rule_id} [{r.category}] {r.title}")
                if r.entities:
                    lines.append(f"      entities: {', '.join(r.entities)}")
        if moderate:
            lines.append("  MODERATE — should be covered:")
            for r in moderate[:10]:
                lines.append(f"    {r.rule_id} [{r.category}] {r.title}")
        if other:
            lines.append("  OTHER:")
            for r in other[:8]:
                lines.append(f"    {r.rule_id} [{r.category}] {r.title}")
        lines.append("")

    # ── Known Entities ──────────────────────────────────────────────────────
    if ctx.entities and ctx.entities.entities:
        entity_names = sorted({e.table.lower() for e in ctx.entities.entities})
        lines.append(
            f"KNOWN ENTITIES [{len(entity_names)} tables — "
            f"flag any other table/entity names as hallucinations]:"
        )
        lines.append("  " + ", ".join(entity_names))
        lines.append("")

    # ── Uncited requirements from traceability ──────────────────────────────
    if ctx.traceability_meta and ctx.traceability_meta.uncited_ids:
        doc_map = {"brd": "covered_brd", "srs": "covered_srs",
                   "ac": "covered_ac", "us": "covered_us"}
        coverage_attr = doc_map.get(doc_type)
        coverage_val  = getattr(ctx.traceability_meta, coverage_attr, 0.0) \
                        if coverage_attr else 0.0
        if coverage_val < 1.0:
            lines.append(
                f"UNCITED REQUIREMENTS (from Stage 5.5 — {doc_type.upper()} "
                f"coverage {coverage_val:.0%}):"
            )
            for uid in ctx.traceability_meta.uncited_ids[:12]:
                lines.append(f"  {uid}")
            lines.append("")

    return "\n".join(lines)


# ── Internal: Critic LLM call ──────────────────────────────────────────────────

_CRITIC_SYSTEM = """\
You are a requirements quality auditor. You receive:
  1. A CATALOG SNAPSHOT listing the spec rules, known entities, and uncited requirements.
  2. A draft BA document (may be truncated).

Your job is to identify:
  a) spec rules NOT covered at all (by title keywords or rule ID) in the draft
  b) entity/table names mentioned in the draft that do NOT appear in KNOWN ENTITIES
  c) required structural sections that are missing

Score formula (compute this yourself):
  covered_strong   = STRONG rules with title keyword OR rule_id found in draft
  total_strong     = total STRONG rules in catalog
  rule_score       = covered_strong / max(1, total_strong)
  entity_penalty   = 0.10 * len(hallucinated_entities)   # cap at 0.20
  structural_penalty = 0.10 * len(missing_sections)       # cap at 0.20
  score = max(0.0, round(rule_score - entity_penalty - structural_penalty, 2))

Respond with ONLY a valid JSON object, no prose, no markdown fences:
{
  "score": <float 0.0-1.0>,
  "uncovered_rule_ids": ["BR-001", ...],
  "hallucinated_entities": ["bad_table", ...],
  "structural_issues": ["Missing 'Non-Functional Requirements' section", ...],
  "rewrite_hints": [
    "Add coverage for BR-003 (password min length validation) in the validation section",
    ...
  ]
}

Keep rewrite_hints concise (≤ 10 items, each ≤ 20 words).
If nothing is wrong, return score=1.0 and empty arrays.\
"""


def _critique(
    doc_type: str,
    draft:    str,
    ctx:      PipelineContext,
    turn:     int,
) -> CriticPass:
    """Call the Critic LLM and return a CriticPass."""
    from pipeline.llm_client import call_llm

    catalog = critic_context_block(ctx, doc_type)
    draft_trunc = draft[:CRITIC_DRAFT_CHARS]
    if len(draft) > CRITIC_DRAFT_CHARS:
        draft_trunc += f"\n… [truncated — {len(draft) - CRITIC_DRAFT_CHARS} chars omitted]"

    # Required markers for this doc type
    req_markers = _DOC_REQUIRED_MARKERS.get(doc_type, [])
    markers_str = "\n".join(f"  - {m}" for m in req_markers)

    user = (
        f"DOC TYPE: {_DOC_LABELS.get(doc_type, doc_type.upper())}\n\n"
        f"{catalog}\n"
        f"REQUIRED SECTION MARKERS (check these appear in the draft):\n"
        f"{markers_str}\n\n"
        f"---\nDRAFT (turn {turn}):\n{draft_trunc}"
    )

    raw = call_llm(
        _CRITIC_SYSTEM,
        user,
        max_tokens   = CRITIC_MAX_TOKENS,
        temperature  = 0.1,
        label        = f"stage5_critic_{doc_type}",
        json_mode    = True,
    )

    return _parse_critic_response(raw, doc_type, turn)


def _parse_critic_response(raw: str, doc_type: str, turn: int) -> CriticPass:
    """Parse the critic LLM JSON response into a CriticPass, with fallback."""
    data: dict = {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract JSON object from response with regex
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    score                 = float(data.get("score", 0.5))
    uncovered_rule_ids    = list(data.get("uncovered_rule_ids", []))
    hallucinated_entities = list(data.get("hallucinated_entities", []))
    structural_issues     = list(data.get("structural_issues", []))
    rewrite_hints         = list(data.get("rewrite_hints", []))

    passed = score >= CRITIC_THRESHOLD

    return CriticPass(
        doc_type              = doc_type,
        turn                  = turn,
        score                 = round(score, 3),
        passed                = passed,
        uncovered_rule_ids    = uncovered_rule_ids,
        hallucinated_entities = hallucinated_entities,
        structural_issues     = structural_issues,
        rewrite_hints         = rewrite_hints,
    )


# ── Internal: Refiner LLM call ────────────────────────────────────────────────

_REFINER_SYSTEM = """\
You are a senior BA editor performing a targeted document revision.
You receive a draft document and a concise list of specific issues to fix.
Rules:
  - Make the MINIMUM changes necessary to address each issue.
  - Do NOT rewrite sections that are not mentioned in the issues.
  - Do NOT remove or rename any existing section headings.
  - Preserve all [BR-XXX] citation tags already present.
  - Output ONLY the corrected full document — no commentary, no preamble.\
"""


def _refine(draft: str, cp: CriticPass) -> str:
    """Call the Refiner LLM to surgically fix the issues flagged by the Critic."""
    from pipeline.llm_client import call_llm

    if not cp.rewrite_hints:
        return draft  # nothing actionable — return draft unchanged

    hints_block = "\n".join(f"{i+1}. {h}" for i, h in enumerate(cp.rewrite_hints))

    issues_block_parts: list[str] = [
        f"ISSUES TO FIX ({_DOC_LABELS.get(cp.doc_type, cp.doc_type.upper())}):",
        hints_block,
    ]
    if cp.uncovered_rule_ids:
        issues_block_parts.append(
            f"\nUncovered rule IDs (ensure each is cited or addressed): "
            + ", ".join(cp.uncovered_rule_ids)
        )
    if cp.hallucinated_entities:
        issues_block_parts.append(
            f"\nEntity names to REMOVE or CORRECT (not in catalog): "
            + ", ".join(cp.hallucinated_entities)
        )
    if cp.structural_issues:
        issues_block_parts.append(
            "\nStructural issues to fix:\n"
            + "\n".join(f"  - {s}" for s in cp.structural_issues)
        )

    issues_block = "\n".join(issues_block_parts)

    user = f"{issues_block}\n\n---\nDRAFT:\n{draft}"

    return call_llm(
        _REFINER_SYSTEM,
        user,
        max_tokens  = REFINER_MAX_TOKENS,
        temperature = 0.15,
        label       = f"stage5_refiner_{cp.doc_type}",
    )


# ── Internal: Execution-Path Hint Enrichment ───────────────────────────────────

# Max execution-path hints injected per failed critic pass (keeps refiner prompt
# within the token budget while still grounding the refiner in real code flow)
_MAX_EXEC_PATH_HINTS = 3


def _enrich_hints_with_exec_paths(cp: CriticPass, ctx: PipelineContext) -> None:
    """
    After a failed critic pass, inject relevant execution-path summaries into
    cp.rewrite_hints so the Refiner has concrete code-flow evidence to cite.

    For each uncovered rule ID, we look up which source file the rule came from
    (via spec_rule.file_path), then find Stage 1.5 execution paths whose entry
    point matches that file.  A concise summary of each path (up to
    _MAX_EXEC_PATH_HINTS paths) is appended to cp.rewrite_hints in-place.

    No-op when:
      - cp has no uncovered rules (nothing to enrich)
      - ctx.code_map.execution_paths is empty (Stage 1.5 not run)
      - ctx.spec_rules is empty (no rule catalogue)
    """
    if not cp.uncovered_rule_ids:
        return
    if not ctx.code_map or not ctx.code_map.execution_paths:
        return
    if not ctx.spec_rules or not ctx.spec_rules.rules:
        return

    # Build rule_id → basename of source file
    rule_to_file: dict[str, str] = {}
    for rule in ctx.spec_rules.rules:
        if rule.rule_id in cp.uncovered_rule_ids:
            fp = getattr(rule, "file_path", "") or ""
            if fp:
                rule_to_file[rule.rule_id] = Path(fp).name

    if not rule_to_file:
        return

    target_files: set[str] = set(rule_to_file.values())

    # Walk execution paths looking for entry points in target_files
    new_hints: list[str] = []
    seen_entries: set[str] = set()

    for ep in ctx.code_map.execution_paths:
        entry = Path(ep.get("entry_point", "")).name
        if entry not in target_files or entry in seen_entries:
            continue
        seen_entries.add(entry)

        # Summarise the happy path as "step1 → step2 → step3"
        steps = ep.get("happy_path") or []
        labels: list[str] = []
        for s in steps[:5]:
            label = (
                s.get("action")
                or s.get("page")
                or s.get("file", "")
            ).strip()
            if label:
                labels.append(label)

        path_desc = " → ".join(labels) if labels else "(no steps)"
        new_hints.append(
            f"Execution path for '{entry}' (covers uncovered rule "
            f"{', '.join(r for r, f in rule_to_file.items() if f == entry)[:60]}): "
            f"{path_desc}"
        )
        if len(new_hints) >= _MAX_EXEC_PATH_HINTS:
            break

    if new_hints:
        cp.rewrite_hints.extend(new_hints)
        print(
            f"  [stage5_critic] Enriched rewrite_hints with "
            f"{len(new_hints)} execution-path hint(s)."
        )
