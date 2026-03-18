"""
pipeline/stage5_workers.py — Stage 5.0 Critic-Augmented BA Document Generation

Runs four LLM agents concurrently using asyncio, each producing one
BA artefact from the shared DomainModel built in Stage 4.

Agents
------
    BRDAgent        Business Requirements Document  → brd.md
    SRSAgent        Software Requirements Spec      → srs.md
    ACAgent         Acceptance Criteria             → ac.md
    UserStoryAgent  User Stories (Gherkin-style)    → user_stories.md

Critic loop (Stage 5.0 enhancement)
------------------------------------
After each writer produces a draft, stage5_critic.run_critic_loop() is called:
    Turn 1  → writer draft
    Critic  → structured JSON: score + uncovered rules + hallucinated entities
    Turn 2  → refiner makes targeted edits (only if score < CRITIC_THRESHOLD)
              (at most 1 extra LLM call per document; 0 extra if draft passes)
CriticPass results are stored in ctx.ba_artifacts.critic_passes for Stage 6.

Set STAGE5_SKIP_CRITIC=1 in environment to disable the critic loop (fast mode).

Execution model
---------------
All four agents run concurrently via asyncio.gather(). Each agent:
    1. Checks if its output file already exists (resume support)
    2. Builds a document-specific system prompt
    3. Assembles a user prompt from ctx.domain_model
    4. Calls llm_client.call_llm() in a thread pool (blocking SDK → async)
    5. Runs the Critic loop on the draft (optional, see above)
    6. Writes the final Markdown output to the run directory
    7. Marks its stage as COMPLETED in ctx

Resume behaviour
----------------
Each agent independently checks if its output already exists and is non-empty.
Only the agents whose outputs are missing are re-run.

Output files
------------
    outputs/run_<id>/brd.md
    outputs/run_<id>/srs.md
    outputs/run_<id>/ac.md
    outputs/run_<id>/user_stories.md
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from pathlib import Path
from typing import Callable

from context import BAArtifacts, DomainModel, PipelineContext

# ── Token budget ───────────────────────────────────────────────────────────────
MAX_TOKENS = 8192

# ── Critic loop feature flag ───────────────────────────────────────────────────
# Set STAGE5_SKIP_CRITIC=1 to disable the critic loop (e.g. for fast local runs)
_CRITIC_ENABLED = os.environ.get("STAGE5_SKIP_CRITIC", "0").strip() != "1"

# Doc-type key mapping: stage_name → critic doc_type string
_STAGE_TO_DOC_TYPE = {
    "stage5_brd":         "brd",
    "stage5_srs":         "srs",
    "stage5_ac":          "ac",
    "stage5_userstories": "us",
}


# ─── Public Entry Point ────────────────────────────────────────────────────────

async def run(ctx: PipelineContext) -> None:
    """
    Stage 5 entry point. Runs all four BA document agents in parallel.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If domain_model is missing or any agent fails.
    """
    if ctx.domain_model is None:
        raise RuntimeError(
            "[stage5] ctx.domain_model is None — run Stage 4 first."
        )

    if ctx.ba_artifacts is None:
        ctx.ba_artifacts = BAArtifacts()

    # Define all four agents: (stage_name, output_filename, agent_fn)
    agents: list[tuple[str, str, Callable]] = [
        ("stage5_brd",         "brd.md",          _run_brd_agent),
        ("stage5_srs",         "srs.md",          _run_srs_agent),
        ("stage5_ac",          "ac.md",           _run_ac_agent),
        ("stage5_userstories", "user_stories.md", _run_userstories_agent),
    ]

    # Filter to only agents that need running
    pending = []
    for stage_name, filename, agent_fn in agents:
        output_path = ctx.output_path(filename)
        if ctx.is_stage_done(stage_name) and Path(output_path).exists():
            print(f"  [stage5] Skipping {stage_name} (already completed)")
            _set_artifact_path(ctx, stage_name, output_path)
        else:
            pending.append((stage_name, output_path, agent_fn))

    if not pending:
        print("  [stage5] All agents already completed.")
        return

    print(f"  [stage5] Running {len(pending)} agent(s) in parallel ...")

    # Run pending agents concurrently
    tasks = [
        _run_agent(ctx, stage_name, output_path, agent_fn)
        for stage_name, output_path, agent_fn in pending
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for failures
    failures = []
    for (stage_name, output_path, _), result in zip(pending, results):
        if isinstance(result, Exception):
            ctx.stage(stage_name).mark_failed(str(result))
            failures.append(f"{stage_name}: {result}")
        else:
            ctx.stage(stage_name).mark_completed(output_path)
            _set_artifact_path(ctx, stage_name, output_path)

    ctx.save()

    if failures:
        raise RuntimeError(
            f"[stage5] {len(failures)} agent(s) failed:\n" +
            "\n".join(f"  • {f}" for f in failures)
        )

    print(f"  [stage5] All agents completed.")


# ─── Agent Runner ──────────────────────────────────────────────────────────────

async def _run_agent(
    ctx:        PipelineContext,
    stage_name: str,
    output_path: str,
    agent_fn:   Callable,
) -> None:
    """Run a single agent in a thread pool (SDK calls are blocking)."""
    ctx.stage(stage_name).mark_running()
    ctx.save()
    print(f"  [stage5] Starting {stage_name} ...")

    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(
        None,
        lambda: agent_fn(ctx.domain_model, ctx)
    )

    # Gemini sometimes returns literal \n sequences instead of real newlines
    # (especially when the response is long). Unescape before writing.
    if "\\n" in content and content.count("\n") < 10:
        content = content.replace("\\n", "\n").replace("\\t", "\t")

    # ── Critic loop (Stage 5.0) ────────────────────────────────────────────
    if _CRITIC_ENABLED:
        doc_type = _STAGE_TO_DOC_TYPE.get(stage_name)
        if doc_type:
            from pipeline.stage5_critic import run_critic_loop
            content, passes = await loop.run_in_executor(
                None,
                lambda: run_critic_loop(doc_type, content, ctx)
            )
            # Persist CriticPass results on ctx for Stage 6 consumption
            if ctx.ba_artifacts is None:
                ctx.ba_artifacts = BAArtifacts()
            ctx.ba_artifacts.critic_passes[doc_type] = [
                dataclasses.asdict(p) for p in passes
            ]

    Path(output_path).write_text(content, encoding="utf-8")
    print(f"  [stage5] {stage_name} → {Path(output_path).name} "
          f"({len(content):,} chars)")


# ─── BRD Agent ────────────────────────────────────────────────────────────────

def _run_brd_agent(domain: DomainModel, ctx: PipelineContext) -> str:
    """Generate the Business Requirements Document."""
    from pipeline.llm_client import call_llm
    from pipeline.evidence_index import build_evidence_index, format_evidence_block

    # Compact domain summary — avoids sending the full JSON blob
    feature_lines = "\n".join(
        f"  - {f.get('name','?')}: {f.get('description','')} | pages={f.get('pages',[])} tables={f.get('tables',[])}"
        for f in domain.features
        if isinstance(f, dict)
    )
    role_lines = "\n".join(
        f"  - {r.get('role','?')}: {r.get('description','')}"
        for r in domain.user_roles
        if isinstance(r, dict)
    )
    rule_lines = "\n".join(
        f"  - {rule}"
        for f in domain.features
        for rule in _to_str_list(f.get("business_rules", []))
    ) or "  - None explicitly defined"

    from pipeline.framework_hints import get_hints
    hints = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")

    # Build evidence index once — maps feature_name → {routes, sql, ...}
    ev_idx = build_evidence_index(ctx, domain)

    system = f"""You are a senior Business Analyst writing a formal Business Requirements Document (BRD).
Write in professional business language using actual system names from the evidence.
Use proper Markdown with headers, bullet points, and tables.
When an Evidence block is provided under a feature, use those routes/SQL/fields to write
specific, grounded acceptance criteria and descriptions — not generic placeholders.

Quality rules:
- Number every business rule BR-01, BR-02, … so the critic can cross-reference them
- Use exact entity/table names from the domain model — do NOT paraphrase or abbreviate
- Never leave placeholder text like "TBD", "[Enter here]", or "as needed"
- Each feature section must state at least one measurable acceptance criterion
Output ONLY the document — no preamble, no commentary after.

Framework context: {hints.brd_note}{_preflight_system_note(ctx)}"""

    # Build explicit feature list for the BRD prompt so none get skipped
    all_feature_names = ", ".join(f['name'] for f in domain.features)
    n_features = len(domain.features)

    # Pre-build the BR scaffold — numbered ### headings + evidence block per feature.
    # Using ### (not bold) ensures _extract_headings() in stage6 finds them.
    def _br_entry(i: int, f: dict) -> str:
        ev_block = format_evidence_block(ev_idx.get(f["name"], {}))
        ev_str   = f"\n{ev_block}" if ev_block else ""
        return (
            f"### BR-{i:02d}: {f['name']}\n"
            f"- Description: {f.get('description', 'see domain model')}\n"
            f"- Priority: High/Medium/Low\n"
            f"- Acceptance: [write 1 specific criterion using evidence below]"
            f"{ev_str}"
        )

    br_scaffold = "\n\n".join(
        _br_entry(i, f) for i, f in enumerate(domain.features, 1)
    )

    user = f"""Write a BRD for: {domain.domain_name}

System description: {domain.description}

User Roles:
{role_lines}

Features:
{feature_lines}

Entities: {", ".join(_to_str_list(domain.key_entities))}
Bounded Contexts: {", ".join(_to_str_list(domain.bounded_contexts))}

Business Rules:
{rule_lines}
{_format_spec_rules_for_prompt(ctx)}
{_format_plain_english_rules(ctx)}
{_format_background_flows(ctx)}

CRITICAL: Section 5 MUST use EXACTLY these {n_features} subsection headings in order.
Do NOT rename, merge, skip, or reorder them. Fill in the Priority and Acceptance fields.
For trivial features write Priority: Low and a brief 1-line acceptance criterion.

Write these sections (keep each section concise — 3-8 bullet points or rows):

# Business Requirements Document — {domain.domain_name}

## 1. Executive Summary
2-3 sentences.

## 2. Business Objectives
Numbered list (max 6).

## 3. Scope
### In Scope
List every feature by name: {all_feature_names}
### Out of Scope

## 4. Stakeholders
Markdown table: Role | Description | Key Interests

## 5. Business Requirements

{br_scaffold}

## 6. Data Requirements
One paragraph per DB table describing purpose and key fields.

## 7. Business Rules
Numbered list of all rules from the features above.

## 8. Assumptions and Constraints

## 9. Glossary
Markdown table: Term | Definition"""

    # ── GraphRAG semantic context ─────────────────────────────────────────────
    # Inject graph-community-aware retrieval so the BRD is grounded in
    # cross-module entity relationships not visible from the domain model alone.
    if hasattr(ctx, "graph_query"):
        try:
            _gc_brd = ctx.graph_query(
                "user roles business requirements workflows features"
            )
        except Exception:
            _gc_brd = None
        if _gc_brd and _gc_brd.strip():
            user = (
                user
                + "\n\nGRAPH-AWARE SEMANTIC CONTEXT "
                + "(use to validate stakeholder roles and feature descriptions):\n"
                + _gc_brd.strip()
            )

    return call_llm(_append_traceability_hints(system), user, max_tokens=MAX_TOKENS,
                    temperature=0.5,  # BRD: natural professional prose
                    label="stage5_brd")


# ─── SRS Agent ────────────────────────────────────────────────────────────────

def _run_srs_agent(domain: DomainModel, ctx: PipelineContext) -> str:
    """Generate the Software Requirements Specification."""
    from pipeline.llm_client import call_llm
    from pipeline.evidence_index import build_evidence_index, format_evidence_block

    from pipeline.framework_hints import get_hints
    hints = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")

    ev_idx = build_evidence_index(ctx, domain)

    system = f"""You are a senior software engineer writing a formal Software Requirements Specification (SRS)
following IEEE 830 conventions. Be precise and technical.
When an Evidence block is provided under a feature, derive concrete Input/Processing/Output/Tables
values from it rather than using placeholders — reference actual route paths, table names, and
form field names. Output clean Markdown only.

Quality rules:
- Number every functional requirement FR-3.X.Y so the critic can cross-reference them
- Use exact table/column names from the domain model — no paraphrasing
- Each FR must include: Input, Processing, Output, Tables (use "none" if truly empty)
- List ALL validation rules per FR (required fields, format checks, length limits)
- Include at least one negative/error scenario per feature (invalid input, unauthorised access)
- Never leave placeholder text like "TBD", "[to be determined]", or "as required"

{hints.srs_note}{_preflight_system_note(ctx)}"""

    all_feature_names = ", ".join(f['name'] for f in domain.features)
    n_features = len(domain.features)

    # Pre-build the FR scaffold for Section 3 with evidence per feature.
    def _fr_entry(i: int, f: dict) -> str:
        ev_block = format_evidence_block(ev_idx.get(f["name"], {}))
        ev_str   = f"\n{ev_block}" if ev_block else ""
        return (
            f"### 3.{i} {f['name']}\n"
            f"FR-{i:03d}: {f.get('description', '[describe requirement]')}\n"
            f"Input: [form fields / query params]\n"
            f"Processing: [business logic]\n"
            f"Output: [result / redirect / data change]\n"
            f"Pages: {', '.join(_to_str_list(f.get('pages', []))) or 'see domain model'}\n"
            f"Tables: {', '.join(_to_str_list(f.get('tables', []))) or 'see domain model'}"
            f"{ev_str}"
        )

    fr_scaffold = "\n\n".join(
        _fr_entry(i, f) for i, f in enumerate(domain.features, 1)
    )

    spec_rules_section = _format_spec_rules_for_prompt(
        ctx, categories=["VALIDATION", "REFERENTIAL", "STATE", "BUSINESS_LIMIT"]
    )

    user = f"""Using the domain model below, write a complete SRS for the '{domain.domain_name}'.

DOMAIN MODEL:
{_format_domain_for_prompt(domain)}
{spec_rules_section}

CRITICAL: Section 3 MUST contain exactly {n_features} subsections — one per feature.
Do not skip ANY feature. For trivial features (e.g. logout, static pages) write a brief
FR with: Input=none, Processing=minimal, Output=redirect or display, Tables=none.
Fill in the bracketed placeholders using the Evidence blocks provided.

Write the SRS with these exact sections:

# Software Requirements Specification — {domain.domain_name}

## 1. Introduction
### 1.1 Purpose
### 1.2 System Overview
### 1.3 Definitions and Abbreviations

## 2. Overall Description
### 2.1 System Context
### 2.2 User Classes and Characteristics
For each user role: name, technical level, frequency of use, key tasks.
### 2.3 Operating Environment
### 2.4 Assumptions and Dependencies

## 3. Functional Requirements

{fr_scaffold}

## 4. Non-Functional Requirements
### 4.1 Security Requirements
### 4.2 Performance Requirements
### 4.3 Usability Requirements
### 4.4 Reliability Requirements

## 5. External Interface Requirements
### 5.1 User Interface
Describe the page flow and key UI interactions.
### 5.2 Database Interface
Describe each table, its purpose, and key fields inferred from the codebase.

## 6. System Constraints
Technical and business constraints on the implementation."""

    # ── GraphRAG semantic context ─────────────────────────────────────────────
    # Cross-module functional/interface facts from the stage38 graph index.
    if hasattr(ctx, "graph_query"):
        try:
            _gc_srs = ctx.graph_query(
                "functional requirements system interface deployment"
            )
        except Exception:
            _gc_srs = None
        if _gc_srs and _gc_srs.strip():
            user = (
                user
                + "\n\nGRAPH-AWARE SEMANTIC CONTEXT "
                + "(use for sections 3 and 5 — functional and interface requirements):\n"
                + _gc_srs.strip()
            )

    # ── Environment variables → Section 2.3 / Section 6 ─────────────────────
    # env_vars carry deployment/config facts (DB host, mail server, API keys,
    # feature flags) that belong in SRS Section 2.3 (Operating Environment)
    # and Section 6 (System Constraints).  Previously ignored by stage5.
    if ctx.code_map:
        _env_vars = getattr(ctx.code_map, "env_vars", None) or []
        if _env_vars:
            from collections import defaultdict as _dd
            _by_prefix: dict = _dd(list)
            for ev in _env_vars:
                key    = ev.get("key", "?")
                prefix = key.split("_")[0] if "_" in key else "OTHER"
                default = ev.get("default")
                entry  = key if default is None else f"{key}={default!r}"
                _by_prefix[prefix].append(entry)
            env_lines = "\n".join(
                f"  {pfx}_*: {', '.join(sorted(set(vs)))}"
                for pfx, vs in sorted(_by_prefix.items())
            )
            user = (
                user
                + "\n\nENVIRONMENT VARIABLES "
                + "(use for Section 2.3 Operating Environment and Section 6 System "
                + "Constraints — describe each group's purpose and deployment impact):\n"
                + env_lines
            )

    return call_llm(_append_traceability_hints(system), user, max_tokens=MAX_TOKENS,
                    temperature=0.4,  # SRS: technical writing with some variability
                    label="stage5_srs")


# ─── AC Agent ─────────────────────────────────────────────────────────────────

def _run_ac_agent(domain: DomainModel, ctx: PipelineContext) -> str:
    """Generate Acceptance Criteria for all features."""
    from pipeline.llm_client import call_llm
    from pipeline.evidence_index import build_evidence_index, format_evidence_block

    from pipeline.framework_hints import get_hints
    hints = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")

    ev_idx = build_evidence_index(ctx, domain)

    system = f"""You are a QA lead writing Acceptance Criteria for a software system.
Use Given/When/Then (Gherkin) format for all interactive scenarios (form submissions,
logins, data mutations). Use plain pass/fail statements only for static display rules.
When an Evidence block is provided, derive Given/When/Then values directly from it:
  - Given: reference the auth guard / session precondition from ExecPath
  - When: reference the actual route, form field names, and HTTP method
  - Then: reference the SQL operation and table, or the redirect target
Output clean Markdown only.

Quality rules:
- Every feature MUST include at least one negative scenario (invalid input, missing auth,
  duplicate submission, boundary value) in addition to the happy path
- Use specific, testable values — not "valid data" but "email format user@example.com"
- Reference actual spec rule IDs (BR-XX) where the rule backs an AC item
- Do NOT write vague criteria like "system works correctly" or "page loads"

{hints.ac_template}"""

    ep_section     = _format_execution_paths_for_prompt(ctx)
    flows_section  = _format_business_flows_for_prompt(ctx)

    # Pre-build the AC scaffold — exact feature-name ## headings + evidence blocks.
    # Headings are FIXED; model fills in Given/When/Then using the evidence.
    def _ac_entry(i: int, f: dict) -> str:
        ev_block = format_evidence_block(ev_idx.get(f["name"], {}))
        ev_str   = f"\n{ev_block}\n" if ev_block else ""
        return (
            f"## AC-{i:02d}: {f['name']}\n"
            f"**Feature:** {f.get('description', 'see domain model')}\n"
            f"**Pages:** {', '.join(_to_str_list(f.get('pages', []))) or 'see domain model'}\n"
            f"**Tables:** {', '.join(_to_str_list(f.get('tables', []))) or 'see domain model'}\n"
            f"{ev_str}\n"
            f"### Acceptance Criteria:\n\n"
            f"**AC-{i:02d}-01: Happy path**\n"
            f"- Given: [precondition — use auth/session from evidence]\n"
            f"- When: [action — use route/form fields from evidence]\n"
            f"- Then: [result — use SQL table/redirect from evidence]\n\n"
            f"**AC-{i:02d}-02: Validation failure**\n"
            f"- Given: [precondition]\n"
            f"- When: [invalid or edge action]\n"
            f"- Then: [expected error/rejection]"
        )

    ac_scaffold = "\n\n---\n\n".join(
        _ac_entry(i, f) for i, f in enumerate(domain.features, 1)
    )

    spec_rules_ac = _format_spec_rules_for_prompt(ctx)

    user = f"""Using the domain model below, write complete Acceptance Criteria for '{domain.domain_name}'.

DOMAIN MODEL:
{_format_domain_for_prompt(domain)}
{flows_section}
{ep_section}
{spec_rules_ac}

CRITICAL HEADING RULE: The section headings below are FIXED. You MUST use them verbatim —
do NOT rename, merge, reorder, or replace them. Each "## AC-XX: [Feature Name]" line
must appear EXACTLY as written. Only fill in the criteria content under each heading.
Omitting or renaming any heading causes QA coverage failures.

# Acceptance Criteria — {domain.domain_name}

## Overview
Brief description of how acceptance testing should be approached for this system.

---

{ac_scaffold}

---

## Test Data Requirements
What test data is needed to execute these criteria."""

    return call_llm(_append_traceability_hints(system), user, max_tokens=MAX_TOKENS,
                    temperature=0.35,  # AC: structured but prose criteria
                    label="stage5_ac")


# ─── User Story Agent ──────────────────────────────────────────────────────────

def _run_userstories_agent(domain: DomainModel, ctx: PipelineContext) -> str:
    """Generate User Stories with story points and priorities."""
    from pipeline.llm_client import call_llm

    from pipeline.framework_hints import get_hints
    from pipeline.evidence_index import build_evidence_index, format_evidence_block
    hints = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")

    ev_idx = build_evidence_index(ctx, domain)

    system = f"""You are an Agile product owner writing User Stories for a development team.
Write stories in standard format: 'As a [role], I want to [action], so that [benefit]'.
Include story points (Fibonacci: 1,2,3,5,8,13), priority (Must/Should/Could/Won't),
and detailed acceptance criteria for each story.
When an Evidence block is provided under an Epic, write the "I want to" and acceptance
criteria using the actual route paths, form field names, and table names from the evidence
rather than generic placeholders. Output clean Markdown only.

Quality rules:
- Prioritise by business value: authentication/core data flows = Must; reporting/export = Should
- Story points guide: 1=trivial display, 2=simple form, 3=form+validation, 5=multi-step flow,
  8=complex workflow with branching, 13=cross-cutting concern or integration
- Each story's acceptance criteria must include at least one failure/edge case scenario
- "So that" must state a concrete business benefit — not "the system works"
- Use exact field names, table names, and route paths from the Evidence block

{hints.story_note}"""

    ep_section     = _format_execution_paths_for_prompt(ctx)
    flows_section  = _format_business_flows_for_prompt(ctx)

    # Pre-build one Epic scaffold per feature — exact feature-name ## headings + evidence.
    # Model fills in the story content — must NOT rename the ## Epic: lines.
    us_counter = [1]  # mutable counter for story IDs
    def _epic_block(f: dict) -> str:
        feat_name = f['name']
        desc   = f.get('description', '')
        pages  = ', '.join(_to_str_list(f.get('pages', []))) or 'see domain model'
        tables = ', '.join(_to_str_list(f.get('tables', []))) or 'see domain model'
        idx    = us_counter[0]
        us_counter[0] += 1
        ev_block = format_evidence_block(ev_idx.get(feat_name, {}))
        ev_str   = f"\n{ev_block}\n" if ev_block else ""
        return (
            f"## Epic: {feat_name}\n\n"
            f"{ev_str}"
            f"### US-{idx:03d}: [Story title for {feat_name}]\n"
            f"**As a** [role]\n"
            f"**I want to** [action — derive from route/form fields in evidence]\n"
            f"**So that** [benefit]\n\n"
            f"**Priority:** Must Have / Should Have / Could Have / Won't Have\n"
            f"**Story Points:** [1 | 2 | 3 | 5 | 8 | 13]\n"
            f"**Pages:** {pages}\n"
            f"**Tables:** {tables}\n\n"
            f"**Acceptance Criteria:**\n"
            f"- [ ] [criterion 1 — reference actual field/table from evidence]\n"
            f"- [ ] [criterion 2 — negative/edge case]\n\n"
            f"**Notes:** {desc}"
        )

    epic_scaffold = "\n\n---\n\n".join(_epic_block(f) for f in domain.features)

    spec_rules_us = _format_spec_rules_for_prompt(
        ctx, categories=["WORKFLOW", "AUTHORIZATION"]
    )

    user = f"""Using the domain model below, write a complete User Story backlog for '{domain.domain_name}'.

DOMAIN MODEL:
{_format_domain_for_prompt(domain)}
{flows_section}
{ep_section}
{spec_rules_us}

CRITICAL HEADING RULE: The "## Epic: [Feature Name]" headings below are FIXED.
You MUST use them verbatim — do NOT rename, merge, reorder, or replace them.
Only fill in the story content (title, As a / I want / So that, criteria, notes).
Omitting or renaming any Epic heading causes QA coverage failures.

# User Story Backlog — {domain.domain_name}

## Epic Summary
One-line description per epic listed below.

---

{epic_scaffold}

---

## Backlog Summary Table
A Markdown table: Story ID | Title | Epic | Priority | Points | Status
Status for all stories should be "To Do"

## Total Story Points
Sum by priority band."""

    return call_llm(_append_traceability_hints(system), user, max_tokens=MAX_TOKENS,
                    temperature=0.5,  # user stories: narrative writing
                    label="stage5_userstories")


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _format_spec_rules_for_prompt(ctx: PipelineContext, categories: list[str] | None = None) -> str:
    """
    Format Stage 4.6 SpecRules as a structured prompt block.

    Parameters
    ----------
    categories : if provided, only include rules from these categories.
                 Useful for targeting: BRD → all, SRS → VALIDATION+REFERENTIAL,
                 AC → all, UserStories → WORKFLOW+AUTHORIZATION.
    """
    sr_col = getattr(ctx, "spec_rules", None)
    if not sr_col or not sr_col.rules:
        return ""

    rules = sr_col.rules
    if categories:
        rules = [r for r in rules if r.category in categories]
    if not rules:
        return ""

    lines = [
        f"\n=== MINED BUSINESS RULES — STAGE 4.6 ({len(rules)} rules) ===",
        "These rules were synthesised from DB constraints, guard clauses, state machines, "
        "and business flows. Use them DIRECTLY in the document — cite rule IDs.",
    ]

    # Group by category
    by_cat: dict[str, list] = {}
    for r in rules:
        by_cat.setdefault(r.category, []).append(r)

    for cat, cat_rules in sorted(by_cat.items()):
        lines.append(f"\n[{cat}]")
        for rule in cat_rules[:20]:   # cap per category to stay within token budget
            lines.append(f"  {rule.rule_id}  {rule.title}")
            lines.append(f"    Given: {rule.given}")
            lines.append(f"    When:  {rule.when}")
            lines.append(f"    Then:  {rule.then}")
            if rule.entities:
                lines.append(f"    Entities: {', '.join(rule.entities)}")
        if len(cat_rules) > 20:
            lines.append(f"  … and {len(cat_rules) - 20} more {cat} rules")

    return "\n".join(lines)


def _format_business_flows_for_prompt(ctx: PipelineContext) -> str:
    """
    Format Stage 4.5 BusinessFlows as a structured prompt block for AC and
    UserStory agents.  Returns empty string if Stage 4.5 was not run.

    Each flow is rendered as an ordered step list with auth gates, DB ops,
    inputs/outputs, and branch conditions — giving agents enough detail to
    write precise Given/When/Then criteria without guessing.
    """
    bfc = getattr(ctx, "business_flows", None)
    if not bfc or not bfc.flows:
        return ""

    lines = [
        "\n=== BUSINESS FLOWS (from graph traversal + static analysis) ===",
        "Use these to write one AC/Story section per flow, "
        "covering the happy path and each listed branch.",
    ]

    for f in bfc.flows:
        lines.append(
            f"\n[{f.flow_id}] {f.name}  "
            f"| Actor: {f.actor}  "
            f"| Context: {f.bounded_context}  "
            f"| Confidence: {f.confidence:.0%}"
        )
        lines.append(f"  Trigger    : {f.trigger}")
        lines.append(f"  Terminates : {f.termination}")
        lines.append(f"  Steps:")
        for s in f.steps:
            auth = " [AUTH REQUIRED]" if s.auth_required else ""
            meth = f" ({s.http_method})" if s.http_method else ""
            db   = f" → DB: {'; '.join(s.db_ops[:2])}" if s.db_ops else ""
            inp  = f" inputs=[{', '.join(s.inputs)}]" if s.inputs else ""
            out  = f" outputs=[{', '.join(s.outputs[:2])}]" if s.outputs else ""
            lines.append(
                f"    {s.step_num}. {s.page}{meth}{auth}{db}{inp}{out}"
            )
            lines.append(f"       {s.action}")
        if f.branches:
            lines.append(f"  Branches ({len(f.branches)}):")
            for b in f.branches:
                lines.append(
                    f"    • At {b['at_page']}: if ({b['condition']}) "
                    f"→ {', '.join(b['alternate'])}"
                )

    return "\n".join(lines)


def _format_execution_paths_for_prompt(ctx: PipelineContext) -> str:
    """
    Format stage15 execution paths as a concise prompt block.
    Returns empty string if stage15 was not run.
    """
    cm = ctx.code_map
    if cm is None:
        return ""
    exec_paths = getattr(cm, "execution_paths", None) or []
    if not exec_paths:
        return ""

    lines = [
        "\n=== REAL BRANCH CONDITIONS (from static code analysis) ===",
        "Use these to write precise Given/When/Then AC derived from actual code logic.",
    ]
    for ep in exec_paths:
        fname = ep.get("file", "?")
        lines.append(f"\n{fname}:")

        ag = ep.get("auth_guard")
        if ag:
            ag_key = ag.get("key", "?")
            ag_redir = ag.get("redirect", "redirect")
            lines.append(
                f"  \u2022 Auth required: session['{ag_key}'] "
                f"(else \u2192 {ag_redir})"
            )

        for flow in ep.get("data_flows", []):
            fields = list(flow.get("field_mapping", {}).keys())
            table  = flow.get("table", "?")
            op     = flow.get("sink", "sql")
            if fields:
                lines.append(
                    f"  • Input: [{', '.join(fields)}] → {op} on `{table}`"
                )

        for b in ep.get("branches", [])[:2]:
            cond = b.get("condition", "")[:80]
            then_acts = [
                f"{a.get('action','')} {a.get('target') or a.get('key') or a.get('table','')}"
                .strip()
                for a in b.get("then", [])
            ]
            else_acts = [
                f"{a.get('action','')} {a.get('target') or a.get('key') or a.get('table','')}"
                .strip()
                for a in b.get("else", [])
            ]
            if cond and (then_acts or else_acts):
                lines.append(f"  • Branch: if ({cond})")
                for t in then_acts:
                    lines.append(f"      ✓ {t}")
                for e in else_acts:
                    lines.append(f"      ✗ {e}")

    return "\n".join(lines)


def _format_plain_english_rules(ctx: PipelineContext, max_rules: int = 40) -> str:
    """
    Return a compact block of BA-ready plain-English invariant sentences
    (Stage 2.9 plain_english field) for injection into writer prompts.

    Only includes rules that didn't make it into Stage 4.6 SpecRules — these
    are the low-confidence invariants the LLM writer would otherwise miss.
    """
    inv = getattr(ctx, "invariants", None)
    if not inv or not inv.rules:
        return ""

    # IDs already covered by SpecRules (via source_invariants)
    covered: set[str] = set()
    sr = getattr(ctx, "spec_rules", None)
    if sr:
        for rule in sr.rules:
            covered.update(rule.source_invariants or [])

    extras = [
        r for r in inv.rules
        if r.rule_id not in covered and r.plain_english
    ]
    if not extras:
        return ""

    lines = [
        f"\n=== ADDITIONAL BUSINESS RULES — Stage 2.9 "
        f"({len(extras)} rules not yet in SpecRules) ===",
        "Use these to enrich the Business Rules section. Do not duplicate "
        "rules already listed in MINED BUSINESS RULES above.",
    ]
    by_cat: dict[str, list] = {}
    for r in extras[:max_rules]:
        by_cat.setdefault(r.category, []).append(r)
    for cat, rs in sorted(by_cat.items()):
        lines.append(f"\n[{cat}]")
        for r in rs:
            lines.append(f"  • {r.plain_english}")
    return "\n".join(lines)


def _format_background_flows(ctx: PipelineContext, max_flows: int = 15) -> str:
    """
    Return a structured block describing non-HTTP entry points (cron jobs, CLI
    commands, webhooks, queue workers) for injection into the BRD writer prompt.

    These flows are invisible to user-journey analysis but represent real system
    behaviour that belongs in the BRD's Background Processing / Integration
    Requirements sections.

    Sources (in priority order):
      1. BusinessFlows with flow_type != "http" (Stage 4.5 tagged from Stage 1.3)
      2. EntryPointCatalog directly (Stage 1.3, for types not yet in any flow)

    Returns empty string when no non-HTTP entry points are found.
    """
    # Collect typed flows from Stage 4.5
    bfc = getattr(ctx, "business_flows", None)
    bg_flows = [
        f for f in (bfc.flows if bfc else [])
        if getattr(f, "flow_type", "http") != "http"
    ]

    # Collect raw entry points from Stage 1.3 NOT already represented by a flow
    ep_cat  = getattr(ctx, "entry_point_catalog", None)
    flow_files: set[str] = set()
    for f in bg_flows:
        for ev in f.evidence_files:
            flow_files.add(ev)

    raw_eps = []
    if ep_cat:
        for ep in ep_cat.entry_points:
            if ep.ep_type != "http" and ep.handler_file not in flow_files:
                raw_eps.append(ep)

    if not bg_flows and not raw_eps:
        return ""

    _TYPE_LABELS = {
        "scheduled":    "Scheduled / Cron",
        "cli":          "CLI / Admin Command",
        "webhook":      "Webhook Receiver",
        "queue_worker": "Queue Worker",
    }

    lines = [
        "\n=== BACKGROUND OPERATIONS (Stage 1.3 + Stage 4.5) ===",
        "The system has the following non-HTTP entry points. Include them in:",
        "  • BRD Section 3 (Scope) — list each as an 'In Scope' integration",
        "  • BRD Section 5 (Business Requirements) — add a sub-section per type",
        "  • BRD Section 8 (Assumptions) — note scheduling and queue infrastructure",
    ]

    # Group: typed flows first
    if bg_flows:
        lines.append("\n[Flows with non-HTTP triggers]")
        for f in bg_flows[:max_flows]:
            label = _TYPE_LABELS.get(f.flow_type, f.flow_type.replace("_", " ").title())
            lines.append(
                f"  [{label}] {f.name}"
                f"  | trigger: {f.trigger}"
            )
            if getattr(f, "schedule", None) or (
                hasattr(ctx, "entry_point_catalog") and ctx.entry_point_catalog
            ):
                # Try to find schedule from catalog
                for ep in (ep_cat.entry_points if ep_cat else []):
                    if any(ep.handler_file in ev for ev in f.evidence_files) and ep.schedule:
                        lines.append(f"    schedule: {ep.schedule}")
                        break
            lines.append(f"    termination: {f.termination}")

    # Raw entry points not covered by any flow
    if raw_eps:
        lines.append("\n[Additional entry points (no flow extracted yet)]")
        for ep in raw_eps[:max(0, max_flows - len(bg_flows))]:
            label = _TYPE_LABELS.get(ep.ep_type, ep.ep_type)
            sched = f"  | schedule: {ep.schedule}" if ep.schedule else ""
            lines.append(f"  [{label}] {ep.name}  | {ep.trigger}{sched}")

    return "\n".join(lines)


def _preflight_system_note(ctx: PipelineContext) -> str:
    """
    Build a short system-prompt addendum from ctx.preflight warnings and
    signal flags.  Previously ctx.preflight was consumed only for quality_score;
    the richer warning/signal data was completely ignored by BA agents.

    Returns an empty string if no preflight data or no actionable signals.

    The note is injected into system prompts so the LLM knows up-front which
    document sections will have thin evidence and how to handle them gracefully
    (e.g. "no SQL found → data-model section will be sparse, use route evidence").
    """
    pf = getattr(ctx, "preflight", None)
    if not pf:
        return ""

    notes: list[str] = []
    signals: dict = getattr(pf, "signals", {}) or {}
    warnings: list[str] = getattr(pf, "warnings", []) or []

    # Signal-driven structural hints
    if not signals.get("has_sql"):
        notes.append(
            "⚠ No SQL queries detected — the data model section will be sparse. "
            "Infer entity structure from route names, form fields, and execution paths."
        )
    if not signals.get("has_forms"):
        notes.append(
            "⚠ No HTML forms detected — input/processing descriptions must be derived "
            "from route parameters and execution path entry conditions."
        )
    if not signals.get("has_auth"):
        notes.append(
            "⚠ No authentication guards detected — omit auth-related acceptance criteria "
            "or mark them as 'not applicable' rather than inventing session logic."
        )
    if not signals.get("has_graph"):
        notes.append(
            "⚠ Knowledge graph is absent — cross-module dependency reasoning will be limited."
        )

    # Surfaced preflight warnings (max 3 to keep prompt lean)
    for w in warnings[:3]:
        if w not in notes:
            notes.append(f"ℹ {w}")

    if not notes:
        return ""

    return (
        "\n\nCODEBASE SIGNAL WARNINGS (from static pre-flight analysis):\n"
        + "\n".join(f"  {n}" for n in notes)
        + "\n\nWrite around these gaps — do not fabricate evidence for sections "
        + "where signals are absent."
    )


def _append_traceability_hints(system_prompt: str) -> str:
    """
    Append [BR-XXX] citation instructions to a Stage 5 system prompt.

    When Stage 5.5 Traceability is enabled the writer is instructed to embed
    rule citations so that Pass B backward-linking can achieve exact matches
    instead of falling back to keyword guessing.
    """
    from pipeline.stage55_traceability import traceability_hints
    return system_prompt + "\n\n" + traceability_hints()


def _to_str_list(items: list) -> list[str]:
    """
    Coerce a list that may contain dicts (LLM over-structured output) to list[str].

    Stage 4's LLM sometimes returns lists of dicts instead of lists of plain
    strings — e.g. business_rules as [{"rule": "...", "severity": "high"}]
    instead of ["..."].  Every join() in this module must go through this
    helper to avoid TypeError: sequence item 0: expected str instance, dict found.

    Coercion strategy:
      str   → kept as-is
      dict  → use the first string value found, falling back to str(dict)
      other → str(item)
    """
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            # Pick the first string value; common keys: "rule", "step", "name", "text"
            val = next((v for v in item.values() if isinstance(v, str)), None)
            result.append(val if val is not None else str(item))
        else:
            result.append(str(item))
    return result


def _format_domain_for_prompt(domain: DomainModel) -> str:
    """
    Format DomainModel as structured prose for LLM consumption.

    Replaces json.dumps() to eliminate structural overhead (indentation,
    quoted keys, empty arrays, brackets) that wastes ~45% of tokens without
    adding semantic value.  ALL information is preserved — nothing is
    summarised or dropped.

    Token comparison (20-feature project):
        json.dumps indent=2  →  ~3,700 tokens
        this formatter       →  ~2,000 tokens  (-46%)

    Format design
    -------------
    - Pipe-separated inline lists for short enumerations (pages, tables, rules)
    - Numbered steps for ordered workflows
    - Explicit "none" / "n/a" to prevent hallucination of missing fields
    - Section headers in ALLCAPS so the LLM can scan structure at a glance
    - No JSON keys, no brackets, no indentation overhead
    """
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(f"DOMAIN: {domain.domain_name}")
    lines.append(f"DESCRIPTION: {domain.description or 'n/a'}")

    # ── User Roles ────────────────────────────────────────────────────────────
    lines.append("\nUSER ROLES:")
    if domain.user_roles:
        for r in domain.user_roles:
            lines.append(f"  • {r.get('role','?')}: {r.get('description','')}")
    else:
        lines.append("  • none defined")

    # ── Key Entities ──────────────────────────────────────────────────────────
    entities_str = ", ".join(_to_str_list(domain.key_entities)) if domain.key_entities else "none"
    lines.append(f"\nKEY ENTITIES: {entities_str}")

    # ── Bounded Contexts ──────────────────────────────────────────────────────
    contexts_str = ", ".join(_to_str_list(domain.bounded_contexts)) if domain.bounded_contexts else "none"
    lines.append(f"BOUNDED CONTEXTS: {contexts_str}")

    # ── Features ─────────────────────────────────────────────────────────────
    lines.append(f"\nFEATURES ({len(domain.features)} total):")
    for i, f in enumerate(domain.features, 1):
        name  = f.get("name", f"Feature {i}")
        desc  = f.get("description", "")
        pages = ", ".join(_to_str_list(f.get("pages", []))) or "none"
        tbls  = ", ".join(_to_str_list(f.get("tables", []))) or "none"
        rules = _to_str_list(f.get("business_rules", []))
        wfs   = f.get("workflows", [])

        lines.append(f"\n  [{i}] {name}")
        lines.append(f"       Desc   : {desc}")
        lines.append(f"       Pages  : {pages}")
        lines.append(f"       Tables : {tbls}")

        if rules:
            lines.append(f"       Rules  : {' | '.join(rules)}")

        for wf in wfs:
            wf_name  = wf.get("name", "workflow")
            wf_steps = _to_str_list(wf.get("steps", []))
            if wf_steps:
                lines.append(f"       Flow ({wf_name}): {' → '.join(wf_steps)}")

    # ── Domain-level Workflows ────────────────────────────────────────────────
    if domain.workflows:
        lines.append(f"\nWORKFLOWS ({len(domain.workflows)} total):")
        for wf in domain.workflows:
            wf_name  = wf.get("name", "?")
            wf_steps = _to_str_list(wf.get("steps", []))
            step_str = " → ".join(wf_steps) if wf_steps else "no steps"
            lines.append(f"  • {wf_name}: {step_str}")

    return "\n".join(lines)


def _set_artifact_path(ctx: PipelineContext, stage_name: str, path: str) -> None:
    """Map stage name → BAArtifacts field."""
    if ctx.ba_artifacts is None:
        ctx.ba_artifacts = BAArtifacts()
    mapping = {
        "stage5_brd":         "brd_path",
        "stage5_srs":         "srs_path",
        "stage5_ac":          "ac_path",
        "stage5_userstories": "user_stories_path",
    }
    field = mapping.get(stage_name)
    if field:
        setattr(ctx.ba_artifacts, field, path)