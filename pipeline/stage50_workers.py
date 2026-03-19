"""
pipeline/stage50_workers.py — Stage 5.0 Critic-Augmented BA Document Generation

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
After each writer produces a draft, stage50_critic.run_critic_loop() is called:
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
MAX_TOKENS = 8192  # kept for critic loop compatibility

# ── Section-by-section generation ──────────────────────────────────────────────
# Each doc is split into: front matter | requirement batches | tail sections.
# STAGE5_SECTION_BATCH  : features per LLM call (default 10)
# STAGE5_SECTION_TOKENS : max output tokens per section call (default 2500)
_SECTION_BATCH  = max(1,  int(os.environ.get("STAGE5_SECTION_BATCH",  "10") or "10"))
_SECTION_TOKENS = max(500, int(os.environ.get("STAGE5_SECTION_TOKENS", "2500") or "2500"))

# ── Critic loop feature flag ───────────────────────────────────────────────────
# Set STAGE5_SKIP_CRITIC=1 to disable the critic loop (e.g. for fast local runs)
_CRITIC_ENABLED = os.environ.get("STAGE5_SKIP_CRITIC", "0").strip() != "1"

# Doc-type key mapping: stage_name → critic doc_type string
_STAGE_TO_DOC_TYPE = {
    "stage50_brd":         "brd",
    "stage50_srs":         "srs",
    "stage50_ac":          "ac",
    "stage50_userstories": "us",
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
        ("stage50_brd",         "brd.md",          _run_brd_agent),
        ("stage50_srs",         "srs.md",          _run_srs_agent),
        ("stage50_ac",          "ac.md",           _run_ac_agent),
        ("stage50_userstories", "user_stories.md", _run_userstories_agent),
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
            from pipeline.stage50_critic import run_critic_loop
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
    """Generate the BRD section-by-section to avoid timeouts on large projects."""
    from pipeline.evidence_index import build_evidence_index, format_evidence_block
    from pipeline.framework_hints import get_hints

    hints  = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")
    ev_idx = build_evidence_index(ctx, domain)

    all_feature_names = ", ".join(f["name"] for f in domain.features)
    role_lines = "\n".join(
        f"  - {r.get('role','?')}: {r.get('description','')}"
        for r in domain.user_roles if isinstance(r, dict)
    )
    rule_lines = "\n".join(
        f"  - {rule}"
        for f in domain.features
        for rule in _to_str_list(f.get("business_rules", []))
    ) or "  - None explicitly defined"

    system = _append_traceability_hints(f"""You are a senior Business Analyst writing a formal Business Requirements Document (BRD).
Write in professional business language using actual system names from the evidence.
Use proper Markdown with headers, bullet points, and tables.
When an Evidence block is provided, use routes/SQL/fields to write specific grounded criteria.

Quality rules:
- Number every business rule BR-01, BR-02, … for critic cross-reference
- Use exact entity/table names — do NOT paraphrase or abbreviate
- Never leave placeholder text like "TBD", "[Enter here]", or "as needed"
- Each feature section must state at least one measurable acceptance criterion
- Output ONLY the requested section(s) — no preamble, no extra commentary

Framework context: {hints.brd_note}{_preflight_system_note(ctx)}""")

    domain_hdr = _compact_domain_header(domain)
    spec_block  = _format_spec_rules_for_prompt(ctx)
    plain_rules = _format_plain_english_rules(ctx)
    bg_flows    = _format_background_flows(ctx)

    gc_ctx = ""
    if hasattr(ctx, "graph_query"):
        try:
            _gc = ctx.graph_query("user roles business requirements workflows features")
            if _gc and _gc.strip():
                gc_ctx = "\n\nGRAPH-AWARE CONTEXT (for stakeholder/feature validation):\n" + _gc.strip()
        except Exception:
            pass

    parts: list[str] = []

    # ── Section A: front matter (1–4) ────────────────────────────────────────
    front_user = f"""Write ONLY sections 1–4 of the BRD. Do NOT write section 5 or beyond.

{domain_hdr}
User Roles:
{role_lines}
Features (for scope list only): {all_feature_names}
{spec_block}{plain_rules}{gc_ctx}

Output EXACTLY:

# Business Requirements Document — {domain.domain_name}

## 1. Executive Summary
2–3 sentences describing what the system does and its business value.

## 2. Business Objectives
Numbered list (max 6 measurable objectives).

## 3. Scope
### In Scope
List every feature: {all_feature_names}
{bg_flows}
### Out of Scope

## 4. Stakeholders
Markdown table: Role | Description | Key Interests"""

    parts.append(_call_section(system, front_user, "stage50_brd_front", 0.5, 1500))

    # ── Section B: business requirements batched by feature ───────────────────
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

    req_header_written = False
    for batch_start, batch in _feature_batches(domain.features, _SECTION_BATCH):
        br_from = batch_start + 1
        br_to   = batch_start + len(batch)
        scaffold = "\n\n".join(_br_entry(i, f) for i, f in enumerate(batch, br_from))
        section_hdr = "## 5. Business Requirements\n\n" if not req_header_written else ""
        req_header_written = True

        req_user = f"""Write ONLY Business Requirements BR-{br_from:02d} through BR-{br_to:02d}.
Do NOT write any other section. Do NOT repeat the document title or previous BRs.

{domain_hdr}
{spec_block}

{section_hdr}Fill in Priority and Acceptance for EACH entry below.
For trivial features write Priority: Low and a brief 1-line criterion.

{scaffold}"""
        parts.append(_call_section(system, req_user,
                                   f"stage50_brd_req_{batch_start // _SECTION_BATCH + 1}",
                                   0.4, _SECTION_TOKENS))

    # ── Section C: tail (6–9) ─────────────────────────────────────────────────
    entities_str = ", ".join(_to_str_list(domain.key_entities)) or "none"
    tail_user = f"""Write ONLY sections 6–9 of the BRD. Do NOT repeat earlier sections.

{domain_hdr}
Entities: {entities_str}
Business Rules:
{rule_lines}
{spec_block}

Output EXACTLY:

## 6. Data Requirements
One paragraph per key DB table describing its purpose and key fields.

## 7. Business Rules
Numbered list of all rules extracted from features and spec rules above.

## 8. Assumptions and Constraints

## 9. Glossary
Markdown table: Term | Definition"""

    parts.append(_call_section(system, tail_user, "stage50_brd_tail", 0.5, 1200))
    return "\n\n".join(parts)


# ─── SRS Agent ────────────────────────────────────────────────────────────────

def _run_srs_agent(domain: DomainModel, ctx: PipelineContext) -> str:
    """Generate the SRS section-by-section to avoid timeouts on large projects."""
    from pipeline.evidence_index import build_evidence_index, format_evidence_block
    from pipeline.framework_hints import get_hints

    hints  = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")
    ev_idx = build_evidence_index(ctx, domain)

    system = _append_traceability_hints(f"""You are a senior software engineer writing a formal SRS (IEEE 830).
Be precise and technical. Use Evidence blocks to derive concrete Input/Processing/Output/Tables values.

Quality rules:
- Number every FR as FR-3.X.Y for critic cross-reference
- Use exact table/column names — no paraphrasing
- Each FR must have: Input, Processing, Output, Tables (write "none" if truly absent)
- List ALL validation rules per FR
- Include at least one negative/error scenario per feature
- Never leave placeholder text like "TBD" or "as required"
- Output ONLY the requested section(s)

{hints.srs_note}{_preflight_system_note(ctx)}""")

    domain_hdr   = _compact_domain_header(domain)
    spec_block   = _format_spec_rules_for_prompt(
        ctx, categories=["VALIDATION", "REFERENTIAL", "STATE", "BUSINESS_LIMIT"])

    # Collect env vars for sections 2.3 / 6
    env_block = ""
    if ctx.code_map:
        _env_vars = getattr(ctx.code_map, "env_vars", None) or []
        if _env_vars:
            from collections import defaultdict as _dd
            _by_prefix: dict = _dd(list)
            for ev in _env_vars:
                key    = ev.get("key", "?")
                prefix = key.split("_")[0] if "_" in key else "OTHER"
                default = ev.get("default")
                _by_prefix[prefix].append(key if default is None else f"{key}={default!r}")
            env_block = "\nENVIRONMENT VARIABLES:\n" + "\n".join(
                f"  {p}_*: {', '.join(sorted(set(vs)))}"
                for p, vs in sorted(_by_prefix.items())
            )

    gc_ctx = ""
    if hasattr(ctx, "graph_query"):
        try:
            _gc = ctx.graph_query("functional requirements system interface deployment")
            if _gc and _gc.strip():
                gc_ctx = "\n\nGRAPH-AWARE CONTEXT (sections 3+5):\n" + _gc.strip()
        except Exception:
            pass

    parts: list[str] = []

    # ── Section A: front matter (1–2) ────────────────────────────────────────
    front_user = f"""Write ONLY sections 1 and 2 of the SRS. Do NOT write section 3 or beyond.

{domain_hdr}{env_block}{gc_ctx}

Output EXACTLY:

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
### 2.4 Assumptions and Dependencies"""

    parts.append(_call_section(system, front_user, "stage50_srs_front", 0.4, 1500))

    # ── Section B: FR batches (Section 3) ────────────────────────────────────
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

    fr_header_written = False
    for batch_start, batch in _feature_batches(domain.features, _SECTION_BATCH):
        fr_from = batch_start + 1
        fr_to   = batch_start + len(batch)
        scaffold = "\n\n".join(_fr_entry(i, f) for i, f in enumerate(batch, fr_from))
        section_hdr = "## 3. Functional Requirements\n\n" if not fr_header_written else ""
        fr_header_written = True

        fr_user = f"""Write ONLY functional requirements 3.{fr_from} through 3.{fr_to}.
Do NOT write any other section. Do NOT repeat the document title or previous FRs.
For trivial features write: Input=none, Processing=minimal, Output=redirect or display, Tables=none.

{domain_hdr}
{spec_block}

{section_hdr}Fill in all bracketed placeholders using the Evidence blocks provided.

{scaffold}"""
        parts.append(_call_section(system, fr_user,
                                   f"stage50_srs_fr_{batch_start // _SECTION_BATCH + 1}",
                                   0.4, _SECTION_TOKENS))

    # ── Section C: tail (4–6) ─────────────────────────────────────────────────
    tail_user = f"""Write ONLY sections 4, 5, and 6 of the SRS. Do NOT repeat earlier sections.

{domain_hdr}{env_block}

Output EXACTLY:

## 4. Non-Functional Requirements
### 4.1 Security Requirements
### 4.2 Performance Requirements
### 4.3 Usability Requirements
### 4.4 Reliability Requirements

## 5. External Interface Requirements
### 5.1 User Interface
Describe the page flow and key UI interactions.
### 5.2 Database Interface
Describe each table, its purpose, and key fields.

## 6. System Constraints
Technical and business constraints on the implementation."""

    parts.append(_call_section(system, tail_user, "stage50_srs_tail", 0.4, 1200))
    return "\n\n".join(parts)


# ─── AC Agent ─────────────────────────────────────────────────────────────────

def _run_ac_agent(domain: DomainModel, ctx: PipelineContext) -> str:
    """Generate Acceptance Criteria section-by-section."""
    from pipeline.evidence_index import build_evidence_index, format_evidence_block
    from pipeline.framework_hints import get_hints

    hints  = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")
    ev_idx = build_evidence_index(ctx, domain)

    system = _append_traceability_hints(f"""You are a QA lead writing Acceptance Criteria.
Use Given/When/Then (Gherkin) for interactive scenarios; plain pass/fail for display rules.
Derive Given/When/Then from Evidence blocks:
  - Given: auth guard / session precondition
  - When: actual route, form field names, HTTP method
  - Then: SQL operation + table, or redirect target

Quality rules:
- Every feature MUST include at least one negative scenario
- Use specific testable values (not "valid data" but "email format user@example.com")
- Reference spec rule IDs (BR-XX) where applicable
- Do NOT write vague criteria like "system works correctly"
- Output ONLY the requested section(s)

{hints.ac_template}""")

    ep_section    = _format_execution_paths_for_prompt(ctx)
    flows_section = _format_business_flows_for_prompt(ctx)
    spec_block    = _format_spec_rules_for_prompt(ctx)
    domain_hdr    = _compact_domain_header(domain)

    parts: list[str] = []

    # ── Section A: header + overview ─────────────────────────────────────────
    hdr_user = f"""Write ONLY the title and Overview section. Do NOT write any AC-XX sections yet.

{domain_hdr}

Output EXACTLY:

# Acceptance Criteria — {domain.domain_name}

## Overview
Brief description of how acceptance testing should be approached for this system."""

    parts.append(_call_section(system, hdr_user, "stage50_ac_header", 0.35, 400))

    # ── Section B: AC entries batched ────────────────────────────────────────
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

    for batch_start, batch in _feature_batches(domain.features, _SECTION_BATCH):
        ac_from = batch_start + 1
        ac_to   = batch_start + len(batch)
        scaffold = "\n\n---\n\n".join(_ac_entry(i, f) for i, f in enumerate(batch, ac_from))

        ac_user = f"""Write ONLY AC-{ac_from:02d} through AC-{ac_to:02d}.
CRITICAL: Use the EXACT headings "## AC-XX: [Feature Name]" as written — do NOT rename them.
Do NOT write any other section or repeat previous ACs.

{domain_hdr}
{flows_section}
{ep_section}
{spec_block}

---

{scaffold}"""
        parts.append(_call_section(system, ac_user,
                                   f"stage50_ac_batch_{batch_start // _SECTION_BATCH + 1}",
                                   0.35, _SECTION_TOKENS))

    # ── Section C: test data requirements ────────────────────────────────────
    tail_user = f"""Write ONLY the Test Data Requirements section. Do NOT repeat any AC sections.

{domain_hdr}

Output EXACTLY:

---

## Test Data Requirements
What test data is needed to execute these criteria (users, records, files, etc.)."""

    parts.append(_call_section(system, tail_user, "stage50_ac_tail", 0.35, 500))
    return "\n\n".join(parts)


# ─── User Story Agent ──────────────────────────────────────────────────────────

def _run_userstories_agent(domain: DomainModel, ctx: PipelineContext) -> str:
    """Generate User Stories section-by-section (one epic batch per call)."""
    from pipeline.evidence_index import build_evidence_index, format_evidence_block
    from pipeline.framework_hints import get_hints

    hints  = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")
    ev_idx = build_evidence_index(ctx, domain)

    system = _append_traceability_hints(f"""You are an Agile product owner writing User Stories.
Format: 'As a [role], I want to [action], so that [benefit]'.
Include story points (Fibonacci: 1,2,3,5,8,13) and priority (Must/Should/Could/Won't).
Derive "I want to" and AC from Evidence blocks — use actual routes, fields, and tables.

Quality rules:
- Auth/core data flows = Must Have; reporting/export = Should Have
- Points guide: 1=display, 2=simple form, 3=form+validation, 5=multi-step,
  8=complex branching, 13=cross-cutting/integration
- Each story AC must include at least one failure/edge case
- "So that" = concrete business benefit, not "the system works"
- Output ONLY the requested section(s)

{hints.story_note}""")

    ep_section    = _format_execution_paths_for_prompt(ctx)
    flows_section = _format_business_flows_for_prompt(ctx)
    spec_block    = _format_spec_rules_for_prompt(ctx, categories=["WORKFLOW", "AUTHORIZATION"])
    domain_hdr    = _compact_domain_header(domain)

    parts: list[str] = []

    # ── Section A: title + epic summary ──────────────────────────────────────
    all_epics = "\n".join(
        f"  - {f['name']}: {f.get('description', '')[:80]}"
        for f in domain.features
    )
    hdr_user = f"""Write ONLY the title and Epic Summary section. Do NOT write any epic detail yet.

{domain_hdr}

Output EXACTLY:

# User Story Backlog — {domain.domain_name}

## Epic Summary
{all_epics}"""

    parts.append(_call_section(system, hdr_user, "stage50_us_header", 0.5, 600))

    # ── Section B: epic batches ───────────────────────────────────────────────
    # story counter must be consistent across batches
    us_global_idx = [1]

    def _epic_block(f: dict) -> str:
        feat_name = f["name"]
        desc      = f.get("description", "")
        pages     = ", ".join(_to_str_list(f.get("pages", []))) or "see domain model"
        tables    = ", ".join(_to_str_list(f.get("tables", []))) or "see domain model"
        idx       = us_global_idx[0]
        us_global_idx[0] += 1
        ev_block  = format_evidence_block(ev_idx.get(feat_name, {}))
        ev_str    = f"\n{ev_block}\n" if ev_block else ""
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

    for batch_start, batch in _feature_batches(domain.features, _SECTION_BATCH):
        scaffold = "\n\n---\n\n".join(_epic_block(f) for f in batch)
        epic_names = ", ".join(f["name"] for f in batch)

        epic_user = f"""Write ONLY the epic detail for: {epic_names}.
CRITICAL: Use EXACT headings "## Epic: [Feature Name]" as written — do NOT rename them.
Do NOT write any other section or repeat previous epics.

{domain_hdr}
{flows_section}
{ep_section}
{spec_block}

---

{scaffold}"""
        parts.append(_call_section(system, epic_user,
                                   f"stage50_us_epics_{batch_start // _SECTION_BATCH + 1}",
                                   0.5, _SECTION_TOKENS))

    # ── Section C: summary table + totals ─────────────────────────────────────
    # Collect US IDs for the table hint
    us_ids_hint = ", ".join(
        f"US-{i:03d}" for i in range(1, len(domain.features) + 1)
    )
    tail_user = f"""Write ONLY the Backlog Summary Table and Total Story Points sections.
Do NOT repeat any epic detail.

Story IDs to include: {us_ids_hint}

Output EXACTLY:

---

## Backlog Summary Table
Markdown table: Story ID | Title | Epic | Priority | Points | Status
(Status = "To Do" for all)

## Total Story Points
Sum by priority band (Must Have / Should Have / Could Have / Won't Have)."""

    parts.append(_call_section(system, tail_user, "stage50_us_tail", 0.5, 800))
    return "\n\n".join(parts)


# ─── Section-by-section helpers ────────────────────────────────────────────────

def _call_section(
    system:      str,
    user:        str,
    label:       str,
    temperature: float,
    max_tokens:  int = 2500,
) -> str:
    """
    Make one focused LLM call for a single document section.

    Smaller max_tokens per call (vs 8192 for the whole doc) means:
    - No timeout risk — each call generates at most ~2500 tokens
    - Better model focus — full attention on just this section
    - Easy per-section retry on failure
    """
    from pipeline.llm_client import call_llm
    return call_llm(system, user, max_tokens=max_tokens,
                    temperature=temperature, label=label)


def _feature_batches(features: list, batch_size: int):
    """
    Yield (start_index, batch) pairs for a feature list.
    start_index is the 0-based offset into the full feature list,
    used to compute consistent BR-XX / FR-3.X / AC-XX numbering.
    """
    for i in range(0, len(features), batch_size):
        yield i, features[i : i + batch_size]


def _compact_domain_header(domain: DomainModel) -> str:
    """
    Minimal domain context block for section-level calls.
    Omits the full feature list (sent per-batch instead) to keep
    each call's input prompt small and focused.
    """
    role_lines = "\n".join(
        f"  • {r.get('role','?')}: {r.get('description','')}"
        for r in domain.user_roles if isinstance(r, dict)
    ) or "  • none defined"
    entities = ", ".join(_to_str_list(domain.key_entities)) or "none"
    contexts = ", ".join(_to_str_list(domain.bounded_contexts)) or "none"
    return (
        f"DOMAIN: {domain.domain_name}\n"
        f"DESCRIPTION: {domain.description or 'n/a'}\n"
        f"USER ROLES:\n{role_lines}\n"
        f"KEY ENTITIES: {entities}\n"
        f"BOUNDED CONTEXTS: {contexts}"
    )


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
        "stage50_brd":         "brd_path",
        "stage50_srs":         "srs_path",
        "stage50_ac":          "ac_path",
        "stage50_userstories": "user_stories_path",
    }
    field = mapping.get(stage_name)
    if field:
        setattr(ctx.ba_artifacts, field, path)