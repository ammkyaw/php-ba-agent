"""
pipeline/stage48_triangulate.py — Stage 4.8 Evidence Triangulation

Cross-references every SpecRule (Stage 4.6) against all independent
static-analysis evidence streams to produce a per-rule triangulation score.

Six evidence type labels
------------------------
  code_static   — Stage 2.9 invariants (guard clauses, schema constraints)
  state_model   — Stage 4.3 state machines (field transitions with guards)
  flow_model    — Stage 4.5 business flows (end-to-end steps with auth+DB ops)
  entity_model  — Stage 4.1/4.2 entities + relationships (columns, FKs)
  semantic_role — Stage 2.7 role tags (actor inference, AUTH/BUSINESS roles)
  behavior_graph— Stage 2.5 behavior graph (structural behavioral patterns)

Scoring
-------
  triangulation_score  = len(corroborating_types) / max_applicable_for_category
  STRONG   ≥ 0.60  (majority of applicable sources agree)
  MODERATE ≥ 0.33  (at least 1–2 sources agree)
  WEAK     < 0.33  (single-source or zero → ⚠ flag for LLM scrutiny)

Contradictions
--------------
  A rule whose source IDs reference artefacts that don't exist in the catalog
  (phantom invariants, missing flows, non-existent entities) is flagged with
  a contradicting_types entry and a human-readable contradiction_notes message.

Zero LLM calls.  Runs after Stage 4.6 (spec_rules), before Stage 5 (BRD/SRS).

Downstream usage
----------------
  Stage 5 BRD/SRS — injects ⚠ LOW-CONFIDENCE REQUIREMENTS section into prompts
  Stage 5.9       — weights document-coverage scoring by triangulation_status
  Stage 6 QA      — expands checklist with weak + contradicted requirements
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from context import (
    PipelineContext,
    TriangulatedRule,
    TriangulationReport,
)

# ── Evidence type labels ───────────────────────────────────────────────────────
_T_CODE_STATIC   = "code_static"     # Stage 2.9 invariants
_T_STATE_MODEL   = "state_model"     # Stage 4.3 state machines
_T_FLOW_MODEL    = "flow_model"      # Stage 4.5 business flows
_T_ENTITY_MODEL  = "entity_model"    # Stage 4.1/4.2 entities + relationships
_T_SEMANTIC_ROLE = "semantic_role"   # Stage 2.7 role tags
_T_BEHAVIOR      = "behavior_graph"  # Stage 2.5 behavior graph

# ── Max-applicable evidence types per rule category ───────────────────────────
# Only counts evidence types that CAN reasonably corroborate a given category.
# This is the denominator — category-specific, not a global constant.
_MAX_APPLICABLE: dict[str, list[str]] = {
    "VALIDATION":       [_T_CODE_STATIC, _T_FLOW_MODEL, _T_ENTITY_MODEL, _T_SEMANTIC_ROLE],
    "AUTHORIZATION":    [_T_CODE_STATIC, _T_FLOW_MODEL, _T_SEMANTIC_ROLE, _T_BEHAVIOR],
    "STATE":            [_T_STATE_MODEL, _T_FLOW_MODEL, _T_CODE_STATIC,   _T_BEHAVIOR],
    "STATE_TRANSITION": [_T_STATE_MODEL, _T_FLOW_MODEL, _T_CODE_STATIC,   _T_BEHAVIOR],
    "REFERENTIAL":      [_T_ENTITY_MODEL, _T_CODE_STATIC, _T_FLOW_MODEL],
    "BUSINESS_LIMIT":   [_T_CODE_STATIC, _T_FLOW_MODEL, _T_STATE_MODEL],
    "WORKFLOW":         [_T_FLOW_MODEL, _T_STATE_MODEL, _T_BEHAVIOR, _T_SEMANTIC_ROLE],
}
_MAX_APPLICABLE_DEFAULT = [_T_CODE_STATIC, _T_FLOW_MODEL, _T_ENTITY_MODEL, _T_SEMANTIC_ROLE]

# Semantic role sets that corroborate each rule category
_ROLE_CORROBORATE: dict[str, set[str]] = {
    "VALIDATION":    {"AUTH_ACTION", "BUSINESS_ACTION", "CRUD_ACTION"},
    "AUTHORIZATION": {"AUTH_ACTION", "BUSINESS_ACTION"},
    "WORKFLOW":      {"BUSINESS_ACTION", "INTEGRATION_ACTION"},
}
_ROLE_CORROBORATE_DEFAULT = {"BUSINESS_ACTION", "AUTH_ACTION", "CRUD_ACTION"}

# Scoring thresholds
_STRONG_THRESHOLD   = 0.60
_MODERATE_THRESHOLD = 0.33

INDEX_FILE = "evidence_triangulation.json"


# ── Public entry point ─────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Build the Evidence Triangulation report and attach it to ctx."""
    stage = ctx.stage("stage48_triangulate")
    stage.mark_running()

    if not ctx.spec_rules or not ctx.spec_rules.rules:
        print("  [stage48] No spec rules found — skipping triangulation.")
        stage.mark_skipped()
        return

    total_rules = len(ctx.spec_rules.rules)
    print(f"  [stage48] Triangulating {total_rules} spec rules …")

    # ── Pre-build lookup indexes ───────────────────────────────────────────────

    # Entities known to exist (table names, lower-case)
    known_tables: frozenset[str] = frozenset(
        e.table.lower() for e in (ctx.entities.entities if ctx.entities else [])
    )

    # Tables that have state machines
    machine_tables: frozenset[str] = frozenset(
        m.table.lower() for m in (ctx.state_machines.machines if ctx.state_machines else [])
    )
    known_machine_ids: frozenset[str] = frozenset(
        m.machine_id for m in (ctx.state_machines.machines if ctx.state_machines else [])
    )

    # Flow IDs known to exist
    known_flow_ids: frozenset[str] = frozenset(
        f.flow_id for f in (ctx.business_flows.flows if ctx.business_flows else [])
    )

    # Tables covered by relationship catalog
    rel_tables: frozenset[str] = frozenset()
    if ctx.relationships:
        rel_tables = frozenset(
            r.from_entity.lower() for r in ctx.relationships.relationships
        ) | frozenset(
            r.to_entity.lower() for r in ctx.relationships.relationships
        )

    # Semantic role: file → list[role]
    sr_file_roles: dict[str, list[str]] = {}
    if ctx.semantic_roles:
        for tag in ctx.semantic_roles.actions:
            sr_file_roles.setdefault(tag.file, []).append(tag.role)

    # Behavior graph bounded-context names (lower-case)
    behavior_contexts: frozenset[str] = frozenset()
    if ctx.behavior_graph:
        nodes = ctx.behavior_graph.get("nodes", [])
        behavior_contexts = frozenset(
            str(n.get("module", n.get("cluster", ""))).lower()
            for n in nodes
            if isinstance(n, dict) and (n.get("module") or n.get("cluster"))
        )

    # ── Per-rule triangulation ─────────────────────────────────────────────────
    triangulated: list[TriangulatedRule] = []

    for rule in ctx.spec_rules.rules:
        corroborating: list[str] = []
        contradicting: list[str] = []
        contradiction_notes: list[str] = []

        applicable = _MAX_APPLICABLE.get(rule.category, _MAX_APPLICABLE_DEFAULT)

        # ── 1. code_static ─────────────────────────────────────────────────────
        if _T_CODE_STATIC in applicable:
            if rule.source_invariants:
                corroborating.append(_T_CODE_STATIC)

        # ── 2. state_model ─────────────────────────────────────────────────────
        if _T_STATE_MODEL in applicable:
            if rule.source_machines:
                if ctx.state_machines:
                    valid = any(mid in known_machine_ids for mid in rule.source_machines)
                    if valid:
                        corroborating.append(_T_STATE_MODEL)
                    else:
                        contradicting.append(_T_STATE_MODEL)
                        contradiction_notes.append(
                            f"source_machines {rule.source_machines} not found in "
                            f"state machine catalog"
                        )
                # No state_machines built yet → neutral (omit from corroborating)
            elif rule.category in ("STATE", "STATE_TRANSITION") and ctx.state_machines:
                # A STATE rule with no machine references but machines exist for
                # its entities is suspicious (missing linkage, possible gap).
                rule_tables_lower = {e.lower() for e in rule.entities}
                if rule_tables_lower & machine_tables:
                    contradiction_notes.append(
                        f"STATE rule has no source_machines but state machines "
                        f"exist for {rule.entities}"
                    )

        # ── 3. flow_model ──────────────────────────────────────────────────────
        if _T_FLOW_MODEL in applicable:
            if rule.source_flows:
                if ctx.business_flows:
                    valid = any(fid in known_flow_ids for fid in rule.source_flows)
                    if valid:
                        corroborating.append(_T_FLOW_MODEL)
                    else:
                        contradicting.append(_T_FLOW_MODEL)
                        contradiction_notes.append(
                            f"source_flows {rule.source_flows} not found in "
                            f"business flow catalog"
                        )
                # No flows built yet → neutral

        # ── 4. entity_model ────────────────────────────────────────────────────
        if _T_ENTITY_MODEL in applicable:
            if rule.entities and ctx.entities:
                rule_tables_lower = {e.lower() for e in rule.entities}
                matched = rule_tables_lower & known_tables
                if matched:
                    corroborating.append(_T_ENTITY_MODEL)
                else:
                    # Entities mentioned in the rule don't appear in the catalog.
                    # Could be pre-4.1 name mismatch — flag but don't hard-contradict.
                    contradicting.append(_T_ENTITY_MODEL)
                    contradiction_notes.append(
                        f"entities {sorted(rule_tables_lower)} not found in entity "
                        f"catalog — possible hallucination or name mismatch"
                    )
            elif rule.category == "REFERENTIAL" and ctx.relationships and rule.entities:
                # Referential rules may be corroborated by relationship catalog alone
                rule_tables_lower = {e.lower() for e in rule.entities}
                if rule_tables_lower & rel_tables:
                    corroborating.append(_T_ENTITY_MODEL)

        # ── 5. semantic_role ───────────────────────────────────────────────────
        if _T_SEMANTIC_ROLE in applicable and ctx.semantic_roles and rule.source_files:
            relevant_roles = _ROLE_CORROBORATE.get(
                rule.category, _ROLE_CORROBORATE_DEFAULT
            )
            for fpath in rule.source_files:
                file_roles = sr_file_roles.get(fpath, [])
                if any(r in relevant_roles for r in file_roles):
                    corroborating.append(_T_SEMANTIC_ROLE)
                    break

        # ── 6. behavior_graph ──────────────────────────────────────────────────
        if _T_BEHAVIOR in applicable and ctx.behavior_graph and rule.bounded_context:
            ctx_lower = rule.bounded_context.lower()
            if any(ctx_lower in bc or bc in ctx_lower for bc in behavior_contexts):
                corroborating.append(_T_BEHAVIOR)

        # ── Score ───────────────────────────────────────────────────────────────
        max_n = len(applicable)
        score = len(corroborating) / max_n if max_n > 0 else 0.0

        if score >= _STRONG_THRESHOLD:
            status = "STRONG"
        elif score >= _MODERATE_THRESHOLD:
            status = "MODERATE"
        else:
            status = "WEAK"

        triangulated.append(TriangulatedRule(
            rule_id              = rule.rule_id,
            title                = rule.title,
            category             = rule.category,
            corroborating_types  = corroborating,
            contradicting_types  = contradicting,
            triangulation_score  = round(score, 3),
            triangulation_status = status,
            max_applicable       = max_n,
            contradiction_notes  = contradiction_notes,
        ))

    # ── Aggregate ─────────────────────────────────────────────────────────────
    strong_count        = sum(1 for r in triangulated if r.triangulation_status == "STRONG")
    moderate_count      = sum(1 for r in triangulated if r.triangulation_status == "MODERATE")
    weak_count          = sum(1 for r in triangulated if r.triangulation_status == "WEAK")
    contradiction_count = sum(1 for r in triangulated if r.contradicting_types)
    weak_ids            = [r.rule_id for r in triangulated if r.triangulation_status == "WEAK"]
    contradiction_ids   = [r.rule_id for r in triangulated if r.contradicting_types]

    report = TriangulationReport(
        rules               = triangulated,
        total               = len(triangulated),
        strong_count        = strong_count,
        moderate_count      = moderate_count,
        weak_count          = weak_count,
        contradiction_count = contradiction_count,
        weak_rule_ids       = weak_ids,
        contradiction_ids   = contradiction_ids,
    )
    ctx.triangulation = report

    # ── Persist ────────────────────────────────────────────────────────────────
    out_path = ctx.output_path(INDEX_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(report), f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"  [stage48] {len(triangulated)} rules triangulated")
    print(f"           ✓ STRONG={strong_count}  ~ MODERATE={moderate_count}  "
          f"⚠ WEAK={weak_count}")
    if contradiction_count:
        print(f"           ✗ {contradiction_count} rules have contradicting evidence")
        for rid in contradiction_ids:
            tr = next(r for r in triangulated if r.rule_id == rid)
            for note in tr.contradiction_notes:
                print(f"             {rid}: {note}")

    ctx.save()
    stage.mark_completed(output_path=out_path)


# ── Query helpers (consumed by Stage 5 / Stage 6) ─────────────────────────────

def weak_rules_block(ctx: PipelineContext, max_rules: int = 20) -> str:
    """
    Return a Markdown block listing WEAK and contradicted SpecRules for
    injection into Stage 5 BRD/SRS/AC prompts.

    Empty string if no triangulation data or no weak rules exist.
    """
    if not ctx.triangulation:
        return ""

    weak  = [r for r in ctx.triangulation.rules if r.triangulation_status == "WEAK"]
    contd = [r for r in ctx.triangulation.rules if r.contradicting_types]

    if not weak and not contd:
        return ""

    lines: list[str] = ["## ⚠ Low-Confidence Requirements (Evidence Triangulation)\n"]
    lines.append(
        "_These requirements have limited independent corroboration. "
        "Use hedging language ('the system appears to…') and flag for business confirmation._\n"
    )

    if weak:
        lines.append(f"\n### Single-Source Rules ({len(weak)} rules)\n")
        for r in weak[:max_rules]:
            sources = ", ".join(r.corroborating_types) if r.corroborating_types else "none"
            lines.append(f"- **{r.rule_id}** [{r.category}] {r.title}  "
                         f"_(score={r.triangulation_score:.2f}, source: {sources})_")

    if contd:
        lines.append(f"\n### Contradicted Rules ({len(contd)} rules)\n")
        for r in contd[:max_rules]:
            notes = "; ".join(r.contradiction_notes)
            lines.append(f"- **{r.rule_id}** [{r.category}] {r.title}  "
                         f"_⚠ {notes}_")

    return "\n".join(lines)
