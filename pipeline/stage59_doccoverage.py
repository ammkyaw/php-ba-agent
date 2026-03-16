"""
pipeline/stage59_doccoverage.py — Document Coverage Audit (Stage 5.9)

Fully static stage that runs after Stage 5 (documents generated) and before
Stage 6 (QA review).  It checks whether every signal discovered by the static
analysis pipeline actually made it into the generated BA documents.

Coverage dimensions
-------------------
  entities   → SRS        every core entity should have a section / mention
  flows      → BRD        every business flow should appear in process flows
  spec_rules → SRS + BRD  every BR-NNN rule should be referenced by ID or title
  states     → SRS        every state-machine entity + its states should appear
  relations  → SRS        high-confidence relationships should appear in data model

Matching strategy
-----------------
All matching is case-insensitive substring search against the document text.
An item is "covered" if ANY of its candidate search terms appears in the
target document(s).  This avoids false negatives from minor wording differences
(e.g. "email" matches "Email", "emails", "email_status", etc.).

Outputs
-------
  doc_coverage.json           — structured coverage per signal type (Stage 6 loads this)
  doc_coverage_summary.md     — human-readable gap report

Resume behaviour
----------------
If stage59_doccoverage is COMPLETED and doc_coverage.json exists, the stage is
skipped.  Use --force stage59_doccoverage to re-run.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from context import PipelineContext

# ── Thresholds ─────────────────────────────────────────────────────────────────
PASS_THRESHOLD = 0.80   # ≥ 80 % = pass (green)
WARN_THRESHOLD = 0.60   # ≥ 60 % = warn (yellow); < 60 % = fail (red)

# ── Output filenames ──────────────────────────────────────────────────────────
COV_FILE = "doc_coverage.json"
MD_FILE  = "doc_coverage_summary.md"


# ── Internal result dataclasses ───────────────────────────────────────────────

@dataclass
class DimCoverage:
    """Coverage for one signal dimension in one document."""
    dimension:    str        # "entities" | "flows" | "spec_rules" | "states" | "relations"
    document:     str        # "SRS" | "BRD" | "AC" | "UserStories"
    total:        int
    covered:      int
    uncovered:    list[str]  # names / IDs of uncovered items
    pct:          float      # 0.0 – 1.0
    status:       str        # "pass" | "warn" | "fail"


@dataclass
class DocCoverageResult:
    """Top-level result written to doc_coverage.json."""
    dimensions:   list[DimCoverage]  = field(default_factory=list)
    overall_pct:  float              = 0.0
    overall_status: str              = "fail"
    gap_summary:  list[str]          = field(default_factory=list)  # human sentences
    generated_at: str                = ""


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 5.9 entry point.  Checks document coverage for all static-analysis
    signals and saves doc_coverage.json + doc_coverage_summary.md.
    """
    cov_path = ctx.output_path(COV_FILE)

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage59_doccoverage") and Path(cov_path).exists():
        print("  [stage59] Already completed — skipping.")
        return

    # ── Load documents ────────────────────────────────────────────────────────
    docs = _load_documents(ctx)
    if not docs:
        print("  [stage59] ⚠️  No BA documents found — skipping coverage audit.")
        return

    print(f"  [stage59] Auditing {len(docs)} document(s) against static signals ...")

    dims: list[DimCoverage] = []

    # ── 1. Entities → SRS ─────────────────────────────────────────────────────
    if ctx.entities and ctx.entities.entities:
        dim = _check_entities(ctx, docs)
        dims.append(dim)
        print(f"  [stage59] Entities   → SRS : {dim.covered}/{dim.total} "
              f"({dim.pct:.0%})  [{dim.status.upper()}]")

    # ── 2. Flows → BRD ────────────────────────────────────────────────────────
    if ctx.business_flows and ctx.business_flows.flows:
        dim = _check_flows(ctx, docs)
        dims.append(dim)
        print(f"  [stage59] Flows      → BRD : {dim.covered}/{dim.total} "
              f"({dim.pct:.0%})  [{dim.status.upper()}]")

    # ── 3. Spec Rules → SRS + BRD ─────────────────────────────────────────────
    if ctx.spec_rules and ctx.spec_rules.rules:
        dim = _check_spec_rules(ctx, docs)
        dims.append(dim)
        print(f"  [stage59] SpecRules  → docs: {dim.covered}/{dim.total} "
              f"({dim.pct:.0%})  [{dim.status.upper()}]")

    # ── 4. State machines → SRS ───────────────────────────────────────────────
    if ctx.state_machines and ctx.state_machines.machines:
        dim = _check_state_machines(ctx, docs)
        dims.append(dim)
        print(f"  [stage59] States     → SRS : {dim.covered}/{dim.total} "
              f"({dim.pct:.0%})  [{dim.status.upper()}]")

    # ── 5. Relationships → SRS ────────────────────────────────────────────────
    if ctx.relationships and ctx.relationships.relationships:
        dim = _check_relationships(ctx, docs)
        dims.append(dim)
        print(f"  [stage59] Relations  → SRS : {dim.covered}/{dim.total} "
              f"({dim.pct:.0%})  [{dim.status.upper()}]")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if dims:
        total_items   = sum(d.total   for d in dims)
        covered_items = sum(d.covered for d in dims)
        overall_pct   = covered_items / total_items if total_items else 1.0
    else:
        overall_pct = 1.0

    overall_status = _status(overall_pct)
    gap_summary    = _build_gap_summary(dims)

    result = DocCoverageResult(
        dimensions      = dims,
        overall_pct     = round(overall_pct, 4),
        overall_status  = overall_status,
        gap_summary     = gap_summary,
        generated_at    = _now(),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_json(result, cov_path)
    md_path = ctx.output_path(MD_FILE)
    _save_markdown(result, ctx, md_path)

    ctx.doc_coverage = result
    ctx.stage("stage59_doccoverage").mark_completed(cov_path)
    ctx.save()

    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(overall_status, "?")
    print(f"  [stage59] Overall coverage: {overall_pct:.0%} {icon}  "
          f"({covered_items}/{total_items} items across {len(dims)} dimension(s))")
    print(f"  [stage59] Saved → {cov_path}")
    if gap_summary:
        print(f"  [stage59] Gaps:")
        for line in gap_summary[:6]:
            print(f"    • {line}")
        if len(gap_summary) > 6:
            print(f"    … and {len(gap_summary) - 6} more — see {md_path}")


# ─── Coverage Checkers ────────────────────────────────────────────────────────

def _check_entities(ctx: PipelineContext, docs: dict[str, str]) -> DimCoverage:
    """
    Core, non-system entities from Stage 4.1 → SRS document.

    An entity is covered if its name OR table name appears (case-insensitive)
    in the SRS text.
    """
    srs_text = _doc_text(docs, "SRS")
    candidates = [
        e for e in ctx.entities.entities
        if e.is_core and not e.is_system
    ]

    covered:   list[str] = []
    uncovered: list[str] = []
    for ent in candidates:
        terms = {ent.name.lower(), ent.table.lower()}
        if any(t in srs_text for t in terms if t):
            covered.append(ent.name)
        else:
            uncovered.append(ent.name)

    return _make_dim("entities", "SRS", covered, uncovered)


def _check_flows(ctx: PipelineContext, docs: dict[str, str]) -> DimCoverage:
    """
    Business flows from Stage 4.5 → BRD document.

    A flow is covered if its name (or any significant word from it ≥4 chars)
    appears in the BRD text.  This handles common reformulations like
    "Login Flow" → "Login".
    """
    brd_text = _doc_text(docs, "BRD")
    flows    = ctx.business_flows.flows

    covered:   list[str] = []
    uncovered: list[str] = []
    for flow in flows:
        terms = _name_terms(flow.name)
        if any(t in brd_text for t in terms):
            covered.append(flow.name)
        else:
            uncovered.append(flow.name)

    return _make_dim("flows", "BRD", covered, uncovered)


def _check_spec_rules(ctx: PipelineContext, docs: dict[str, str]) -> DimCoverage:
    """
    Spec rules from Stage 4.6 → SRS + BRD (combined search).

    A rule is covered if its rule_id (e.g. "BR-001") OR its title OR any of
    its tags appears in either SRS or BRD.
    """
    srs_text  = _doc_text(docs, "SRS")
    brd_text  = _doc_text(docs, "BRD")
    combined  = srs_text + " " + brd_text

    covered:   list[str] = []
    uncovered: list[str] = []
    for rule in ctx.spec_rules.rules:
        terms: set[str] = set()
        terms.add(rule.rule_id.lower())
        terms.update(_name_terms(rule.title))
        terms.update(t.lower() for t in (rule.tags or []) if len(t) >= 4)
        if any(t in combined for t in terms if t):
            covered.append(rule.rule_id)
        else:
            uncovered.append(f"{rule.rule_id} ({rule.title})")

    return _make_dim("spec_rules", "SRS+BRD", covered, uncovered)


def _check_state_machines(ctx: PipelineContext, docs: dict[str, str]) -> DimCoverage:
    """
    State machines from Stage 4.3 → SRS document.

    A machine is covered if the entity name appears in SRS AND at least one
    of its concrete state values also appears.  This ensures the lifecycle is
    described, not merely the entity is mentioned.
    """
    srs_text = _doc_text(docs, "SRS")

    covered:   list[str] = []
    uncovered: list[str] = []
    for sm in ctx.state_machines.machines:
        entity_terms = _name_terms(sm.entity)
        entity_found = any(t in srs_text for t in entity_terms if t)

        state_found = any(
            s.lower() in srs_text
            for s in sm.states
            if len(s) >= 3
        )

        label = f"{sm.entity}.{sm.field}"
        if entity_found and state_found:
            covered.append(label)
        else:
            reason = []
            if not entity_found: reason.append("entity missing")
            if not state_found:  reason.append("states missing")
            uncovered.append(f"{label} ({', '.join(reason)})")

    return _make_dim("states", "SRS", covered, uncovered)


def _check_relationships(ctx: PipelineContext, docs: dict[str, str]) -> DimCoverage:
    """
    High-confidence relationships from Stage 4.2 → SRS document.

    A relationship is covered if BOTH entity names appear in the SRS text.
    We only check conf ≥ 0.75 to avoid noise from weak signals.
    """
    srs_text = _doc_text(docs, "SRS")
    high_rels = [r for r in ctx.relationships.relationships if r.confidence >= 0.75]

    covered:   list[str] = []
    uncovered: list[str] = []
    for rel in high_rels:
        from_terms = _name_terms(rel.from_entity)
        to_terms   = _name_terms(rel.to_entity)
        from_found = any(t in srs_text for t in from_terms if t)
        to_found   = any(t in srs_text for t in to_terms   if t)
        label = f"{rel.from_entity} {rel.cardinality} {rel.to_entity}"
        if from_found and to_found:
            covered.append(label)
        else:
            missing = []
            if not from_found: missing.append(rel.from_entity)
            if not to_found:   missing.append(rel.to_entity)
            uncovered.append(f"{label} (entity not found: {', '.join(missing)})")

    return _make_dim("relations", "SRS", covered, uncovered)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_documents(ctx: PipelineContext) -> dict[str, str]:
    """
    Load all available BA documents.  Missing files are silently skipped so
    coverage is computed against whatever was generated.
    """
    ba = ctx.ba_artifacts
    if ba is None:
        return {}
    mapping = {
        "BRD":         ba.brd_path,
        "SRS":         ba.srs_path,
        "AC":          ba.ac_path,
        "UserStories": ba.user_stories_path,
    }
    docs: dict[str, str] = {}
    for name, path in mapping.items():
        if path and Path(path).exists():
            docs[name] = Path(path).read_text(encoding="utf-8").lower()
    return docs


def _doc_text(docs: dict[str, str], key: str) -> str:
    """Return lowercased document text, or empty string if not loaded."""
    return docs.get(key, "")


def _name_terms(name: str) -> set[str]:
    """
    Split a name like "Email Sending Flow" into search terms:
    the full lowercased name + individual words ≥ 4 chars.

    This handles reformulations — "Email Sending Flow" also matches
    a document that says "Email" or "Sending" independently.
    """
    low = name.lower()
    words = {w for w in re.split(r"[\s_\-/]+", low) if len(w) >= 4}
    words.add(low)
    return words


def _make_dim(
    dimension: str,
    document:  str,
    covered:   list[str],
    uncovered: list[str],
) -> DimCoverage:
    total  = len(covered) + len(uncovered)
    cnt    = len(covered)
    pct    = cnt / total if total else 1.0
    return DimCoverage(
        dimension = dimension,
        document  = document,
        total     = total,
        covered   = cnt,
        uncovered = uncovered,
        pct       = round(pct, 4),
        status    = _status(pct),
    )


def _status(pct: float) -> str:
    if pct >= PASS_THRESHOLD: return "pass"
    if pct >= WARN_THRESHOLD: return "warn"
    return "fail"


def _build_gap_summary(dims: list[DimCoverage]) -> list[str]:
    """Human-readable sentences for each failing/warning dimension."""
    lines: list[str] = []
    for d in dims:
        if d.status == "pass":
            continue
        sample = d.uncovered[:4]
        suffix = f" (+{len(d.uncovered) - 4} more)" if len(d.uncovered) > 4 else ""
        lines.append(
            f"{d.dimension.capitalize()} coverage in {d.document} is "
            f"{d.pct:.0%} ({d.covered}/{d.total}) — "
            f"missing: {', '.join(sample)}{suffix}"
        )
    return lines


# ─── Persistence ──────────────────────────────────────────────────────────────

def _save_json(result: DocCoverageResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, indent=2, ensure_ascii=False)


def _save_markdown(result: DocCoverageResult, ctx: PipelineContext, path: str) -> None:
    lines: list[str] = [
        "# Document Coverage Audit — Stage 5.9",
        "",
        f"**Overall**: {result.overall_pct:.0%}  "
        f"[{result.overall_status.upper()}]",
        "",
        "## Coverage by Dimension",
        "",
        "| Dimension | Target | Covered | Total | % | Status |",
        "|-----------|--------|---------|-------|---|--------|",
    ]
    icons = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
    for d in result.dimensions:
        lines.append(
            f"| {d.dimension} | {d.document} | {d.covered} "
            f"| {d.total} | {d.pct:.0%} | {icons.get(d.status,'?')} |"
        )
    lines.append("")

    for d in result.dimensions:
        if not d.uncovered:
            continue
        lines += [f"## Uncovered {d.dimension.capitalize()} → {d.document}", ""]
        for item in d.uncovered:
            lines.append(f"- `{item}`")
        lines.append("")

    if result.gap_summary:
        lines += ["## Gap Summary (for Stage 6 QA)", ""]
        for g in result.gap_summary:
            lines.append(f"- {g}")
        lines.append("")

    lines += [
        f"*Generated by Stage 5.9 at {result.generated_at}*",
    ]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _now() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()
