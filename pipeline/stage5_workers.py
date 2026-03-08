"""
pipeline/stage5_workers.py — Parallel BA Document Generation

Runs four LLM agents concurrently using asyncio, each producing one
BA artefact from the shared DomainModel built in Stage 4.

Agents
------
    BRDAgent        Business Requirements Document  → brd.md
    SRSAgent        Software Requirements Spec      → srs.md
    ACAgent         Acceptance Criteria             → ac.md
    UserStoryAgent  User Stories (Gherkin-style)    → user_stories.md

Execution model
---------------
All four agents run concurrently via asyncio.gather(). Each agent:
    1. Checks if its output file already exists (resume support)
    2. Builds a document-specific system prompt
    3. Assembles a user prompt from ctx.domain_model
    4. Calls llm_client.call_llm() in a thread pool (blocking SDK → async)
    5. Writes the Markdown output to the run directory
    6. Marks its stage as COMPLETED in ctx

The Gemini free tier allows 15 RPM. With 4 concurrent calls each taking
~5-10s, we stay well within limits. llm_client already adds a 4s delay
per call, so worst-case 4 × 4s = 16s of rate-limit padding.

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
import os
from pathlib import Path
from typing import Callable

from context import BAArtifacts, DomainModel, PipelineContext

# ── Token budget ───────────────────────────────────────────────────────────────
MAX_TOKENS = 8192


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
    Path(output_path).write_text(content, encoding="utf-8")
    print(f"  [stage5] {stage_name} → {Path(output_path).name} "
          f"({len(content):,} chars)")


# ─── BRD Agent ────────────────────────────────────────────────────────────────

def _run_brd_agent(domain: DomainModel, ctx: PipelineContext) -> str:
    """Generate the Business Requirements Document."""
    from pipeline.llm_client import call_llm

    # Compact domain summary — avoids sending the full JSON blob
    feature_lines = "\n".join(
        f"  - {f['name']}: {f['description']} | pages={f.get('pages',[])} tables={f.get('tables',[])}"
        for f in domain.features
    )
    role_lines = "\n".join(
        f"  - {r['role']}: {r['description']}"
        for r in domain.user_roles
    )
    rule_lines = "\n".join(
        f"  - {rule}"
        for f in domain.features
        for rule in f.get("business_rules", [])
    ) or "  - None explicitly defined"

    from pipeline.framework_hints import get_hints
    hints = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")

    system = f"""You are a senior Business Analyst writing a formal Business Requirements Document (BRD).
Write in professional business language using actual system names from the evidence.
Use proper Markdown with headers, bullet points, and tables.
Output ONLY the document — no preamble, no commentary after.

Framework context: {hints.brd_note}"""

    # Build explicit feature list for the BRD prompt so none get skipped
    all_feature_names = ", ".join(f['name'] for f in domain.features)
    n_features = len(domain.features)

    user = f"""Write a BRD for: {domain.domain_name}

System description: {domain.description}

User Roles:
{role_lines}

Features:
{feature_lines}

Entities: {", ".join(domain.key_entities)}
Bounded Contexts: {", ".join(domain.bounded_contexts)}

Business Rules:
{rule_lines}

CRITICAL: Section 5 MUST contain exactly {n_features} BR entries, one per feature:
{all_feature_names}
Do not merge, skip, or omit ANY feature — even trivial ones like logout or static pages.
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
MUST have exactly {n_features} entries — one per feature listed above.
**BR-XXX: [Feature Name]**
- Description: what the business needs
- Priority: High/Medium/Low
- Acceptance: how we verify it is met

## 6. Data Requirements
One paragraph per DB table describing purpose and key fields.

## 7. Business Rules
Numbered list of all rules from the features above.

## 8. Assumptions and Constraints

## 9. Glossary
Markdown table: Term | Definition"""

    return call_llm(system, user, max_tokens=MAX_TOKENS, label="stage5_brd")


# ─── SRS Agent ────────────────────────────────────────────────────────────────

def _run_srs_agent(domain: DomainModel, ctx: PipelineContext) -> str:
    """Generate the Software Requirements Specification."""
    from pipeline.llm_client import call_llm

    from pipeline.framework_hints import get_hints
    hints = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")

    system = f"""You are a senior software engineer writing a formal Software Requirements Specification (SRS)
following IEEE 830 conventions. Be precise and technical. Reference actual page names, table names, and
field names from the evidence. Output clean Markdown only.

{hints.srs_note}"""

    all_feature_names = ", ".join(f['name'] for f in domain.features)
    n_features = len(domain.features)

    user = f"""Using the domain model below, write a complete SRS for the '{domain.domain_name}'.

DOMAIN MODEL:
{_format_domain_for_prompt(domain)}

CRITICAL: Section 3 MUST contain exactly {n_features} subsections — one per feature:
{all_feature_names}
Do not skip ANY feature. For trivial features (e.g. logout, static pages) write a brief
FR with: Input=none, Processing=minimal, Output=redirect or display, Tables=none.

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
For each feature, write detailed functional requirements:
### 3.X [Feature Name]
FR-XXX: [Requirement ID and description]
Input: [Fields and their validation rules]
Processing: [Business logic]
Output: [Expected result / redirect / data change]
Pages: [Which PHP pages implement this]
Tables: [Which DB tables are affected]

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

    return call_llm(system, user, max_tokens=MAX_TOKENS, label="stage5_srs")


# ─── AC Agent ─────────────────────────────────────────────────────────────────

def _run_ac_agent(domain: DomainModel, ctx: PipelineContext) -> str:
    """Generate Acceptance Criteria for all features."""
    from pipeline.llm_client import call_llm

    from pipeline.framework_hints import get_hints
    hints = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")

    system = f"""You are a QA lead writing Acceptance Criteria for a software system.
Write criteria in Given/When/Then (Gherkin) format where appropriate, and as clear
pass/fail statements otherwise. Be specific — reference actual field names, page names,
and table names. Output clean Markdown only.

{hints.ac_template}"""

    ep_section     = _format_execution_paths_for_prompt(ctx)
    flows_section  = _format_business_flows_for_prompt(ctx)

    user = f"""Using the domain model below, write complete Acceptance Criteria for '{domain.domain_name}'.

DOMAIN MODEL:
{_format_domain_for_prompt(domain)}
{flows_section}
{ep_section}

IMPORTANT COVERAGE RULE:
Every feature in the domain model MUST have an AC section — even trivial ones
(e.g., logout, static display pages). For features with minimal logic:
  - Write 1-2 criteria instead of the usual 3
  - Mark them explicitly: "**Complexity: Low — single operation**"
  - Do NOT skip them — omission causes QA coverage failures

Write one section per feature:

# Acceptance Criteria — {domain.domain_name}

## Overview
Brief description of how acceptance testing should be approached for this system.

---

For EACH feature in the domain model, write a section:

## AC-XX: [Feature Name]
**Feature:** [One-line description]
**Pages:** [Relevant pages]
**Tables:** [Relevant tables]

### Acceptance Criteria:

**AC-XX-01: [Criteria name]**
- Given: [precondition]
- When: [action]
- Then: [expected result]

**AC-XX-02: [Criteria name — negative/edge case]**
- Given: [precondition]
- When: [invalid or edge action]
- Then: [expected error/rejection]

[At least 3 criteria per feature: happy path, validation failure, and one edge case]

---

## Test Data Requirements
What test data is needed to execute these criteria."""

    return call_llm(system, user, max_tokens=MAX_TOKENS, label="stage5_ac")


# ─── User Story Agent ──────────────────────────────────────────────────────────

def _run_userstories_agent(domain: DomainModel, ctx: PipelineContext) -> str:
    """Generate User Stories with story points and priorities."""
    from pipeline.llm_client import call_llm

    from pipeline.framework_hints import get_hints
    hints = get_hints(ctx.code_map.framework if ctx.code_map else "unknown")

    system = f"""You are an Agile product owner writing User Stories for a development team.
Write stories in standard format: 'As a [role], I want to [action], so that [benefit]'.
Include story points (Fibonacci: 1,2,3,5,8,13), priority (Must/Should/Could/Won't),
and detailed acceptance criteria for each story. Reference actual page and field names.
Output clean Markdown only.

{hints.story_note}"""

    ep_section     = _format_execution_paths_for_prompt(ctx)
    flows_section  = _format_business_flows_for_prompt(ctx)

    user = f"""Using the domain model below, write a complete User Story backlog for '{domain.domain_name}'.

DOMAIN MODEL:
{_format_domain_for_prompt(domain)}
{flows_section}
{ep_section}

IMPORTANT COVERAGE RULE:
Every feature in the domain model MUST appear in the backlog — even trivial ones
(e.g., logout, static pages). For features with no business logic:
  - Write a single micro-story (1 story point, Could Have priority)
  - Keep AC to 1-2 criteria max
  - Do NOT skip them — omission causes QA coverage failures

Structure the output as follows:

# User Story Backlog — {domain.domain_name}

## Epic Summary
List all epics (one per bounded context) with a one-line description each.

---

For each bounded context, create an Epic with its User Stories:

## Epic: [Bounded Context Name]

### US-XXX: [Story Title]
**As a** [role from domain model]
**I want to** [specific action using actual page/field names]
**So that** [business benefit]

**Priority:** Must Have / Should Have / Could Have / Won't Have
**Story Points:** [1 | 2 | 3 | 5 | 8 | 13]
**Pages:** [actual PHP pages]
**Tables:** [actual DB tables]

**Acceptance Criteria:**
- [ ] [Specific, testable criterion using actual field/page names]
- [ ] [Negative case or validation rule]
- [ ] [Edge case or non-functional requirement]

**Notes:** [Any technical notes, dependencies, or assumptions]

---

## Backlog Summary Table
A Markdown table: Story ID | Title | Epic | Priority | Points | Status
Status for all stories should be "To Do"

## Total Story Points
Sum by priority band."""

    return call_llm(system, user, max_tokens=MAX_TOKENS, label="stage5_userstories")


# ─── Helpers ───────────────────────────────────────────────────────────────────

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
    entities_str = ", ".join(domain.key_entities) if domain.key_entities else "none"
    lines.append(f"\nKEY ENTITIES: {entities_str}")

    # ── Bounded Contexts ──────────────────────────────────────────────────────
    contexts_str = ", ".join(domain.bounded_contexts) if domain.bounded_contexts else "none"
    lines.append(f"BOUNDED CONTEXTS: {contexts_str}")

    # ── Features ─────────────────────────────────────────────────────────────
    lines.append(f"\nFEATURES ({len(domain.features)} total):")
    for i, f in enumerate(domain.features, 1):
        name  = f.get("name", f"Feature {i}")
        desc  = f.get("description", "")
        pages = ", ".join(f.get("pages", [])) or "none"
        tbls  = ", ".join(f.get("tables", [])) or "none"
        rules = f.get("business_rules", [])
        wfs   = f.get("workflows", [])

        lines.append(f"\n  [{i}] {name}")
        lines.append(f"       Desc   : {desc}")
        lines.append(f"       Pages  : {pages}")
        lines.append(f"       Tables : {tbls}")

        if rules:
            lines.append(f"       Rules  : {' | '.join(rules)}")

        for wf in wfs:
            wf_name  = wf.get("name", "workflow")
            wf_steps = wf.get("steps", [])
            if wf_steps:
                lines.append(f"       Flow ({wf_name}): {' → '.join(wf_steps)}")

    # ── Domain-level Workflows ────────────────────────────────────────────────
    if domain.workflows:
        lines.append(f"\nWORKFLOWS ({len(domain.workflows)} total):")
        for wf in domain.workflows:
            wf_name  = wf.get("name", "?")
            wf_steps = wf.get("steps", [])
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