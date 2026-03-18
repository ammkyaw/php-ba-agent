"""
stage62_architecture.py — Pipeline Stage 6.2: Architecture Reconstruction

Synthesises the code_map, domain model, business flows, and graph metadata
into a structured system-architecture document.  Runs AFTER Stage 6 QA
(so the domain model and flows are validated) and BEFORE Stage 6.5 postprocess
and Stage 6.7 diagrams (both of which can reference the architecture JSON).

Placement in the pipeline:
    …stage6_qa → stage62_architecture → stage65_postprocess → stage67_diagrams → stage7_pdf…

Outputs written to ctx.output_dir:
    architecture.json   — structured architecture data (consumed by stage67)
    architecture.md     — human-readable Markdown document

ctx fields read:
    ctx.code_map         (CodeMap)                — from Stage 1
    ctx.graph_meta       (GraphMeta)              — from Stage 2
    ctx.domain_model     (DomainModel)            — from Stage 4   [required]
    ctx.business_flows   (BusinessFlowCollection) — from Stage 4.5 [required]
    ctx.qa_result        (QAResult)               — from Stage 6   [optional, informational]

ctx fields written:
    ctx.architecture_meta (ArchitectureMeta)      — lightweight summary for downstream stages

Stage key: "stage62_architecture"
"""

from __future__ import annotations

import dataclasses
import json
import re
import textwrap
from pathlib import Path
from typing import Any

from pipeline.llm_client import call_llm

from context import (
    ArchitectureMeta,
    BusinessFlowCollection,
    CodeMap,
    DomainModel,
    GraphMeta,
    PipelineContext,
    QAResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_NAME  = "stage62_architecture"
MAX_TOKENS  = 8192

# Hard cap on formatted context sent to the LLM to avoid blowing context window
_MAX_CONTEXT_CHARS = 24_000


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(ctx: PipelineContext) -> None:
    """
    Execute Stage 6.2 — Architecture Reconstruction.

    Called by run_pipeline.py via the STAGES registry.  Idempotent: skips
    if both output files already exist and the stage is marked completed.

    Args:
        ctx: Shared pipeline context.  Must have domain_model and
             business_flows populated; code_map and graph_meta are used
             when available for richer output.

    Raises:
        RuntimeError: If required upstream context is missing.
        anthropic.APIError: On LLM API failures (propagated to pipeline).
        ValueError: If the LLM response cannot be parsed as valid JSON.
    """
    json_path = Path(ctx.output_path("architecture.json"))
    md_path   = Path(ctx.output_path("architecture.md"))

    # ── Resumability check ──────────────────────────────────────────────────
    if (
        ctx.is_stage_done(STAGE_NAME)
        and json_path.exists()
        and md_path.exists()
    ):
        print(f"[{STAGE_NAME}] Already completed — skipping.")
        return

    # ── Dependency guard ────────────────────────────────────────────────────
    if ctx.domain_model is None:
        raise RuntimeError(
            f"[{STAGE_NAME}] requires ctx.domain_model "
            "(Stage 4 must complete first)."
        )
    if ctx.business_flows is None:
        raise RuntimeError(
            f"[{STAGE_NAME}] requires ctx.business_flows "
            "(Stage 4.5 must complete first)."
        )

    print(f"[{STAGE_NAME}] Running — Architecture Reconstruction…")

    # ── Build LLM context block ─────────────────────────────────────────────
    context_block = _build_context_block(ctx)

    # ── Call LLM ────────────────────────────────────────────────────────────
    raw_response = _call_llm(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_USER_PROMPT_TEMPLATE.format(context_block=context_block),
    )

    # ── Parse & validate response ───────────────────────────────────────────
    data = _parse_json(raw_response)
    _validate_schema(data)

    # ── Write outputs ───────────────────────────────────────────────────────
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{STAGE_NAME}] Wrote {json_path}")

    md_path.write_text(_build_markdown(data), encoding="utf-8")
    print(f"[{STAGE_NAME}] Wrote {md_path}")

    # ── Update ctx ──────────────────────────────────────────────────────────
    ctx.architecture_meta = ArchitectureMeta(
        json_path       = str(json_path),
        md_path         = str(md_path),
        component_count = len(data.get("components", [])),
        data_flow_count = len(data.get("data_flows", [])),
        sequence_count  = len(data.get("sequence_flows", [])),
        tech_stack      = data.get("technology_observations", {}).get("stack", []),
    )

    ctx.stage(STAGE_NAME).mark_completed(output_path=str(json_path))
    ctx.save()
    print(f"[{STAGE_NAME}] Completed.")


# ---------------------------------------------------------------------------
# Context formatting helpers
# ---------------------------------------------------------------------------

def _build_context_block(ctx: PipelineContext) -> str:
    """
    Assemble a compact, LLM-friendly context string from all available
    upstream artefacts.  Hard-truncated to _MAX_CONTEXT_CHARS so we never
    exceed the model's context window.
    """
    parts: list[str] = []

    if ctx.code_map is not None:
        parts.append(_format_code_map(ctx.code_map))

    if ctx.graph_meta is not None:
        parts.append(_format_graph_meta(ctx.graph_meta))

    parts.append(_format_domain_model(ctx.domain_model))    # type: ignore[arg-type]
    parts.append(_format_business_flows(ctx.business_flows)) # type: ignore[arg-type]

    if ctx.qa_result is not None:
        parts.append(_format_qa_result(ctx.qa_result))

    full = "\n\n".join(parts)

    # Truncate with a clear marker so the LLM knows it's intentional
    if len(full) > _MAX_CONTEXT_CHARS:
        full = full[:_MAX_CONTEXT_CHARS] + "\n\n[... context truncated for length ...]"

    return full


def _format_code_map(cm: CodeMap) -> str:
    lines = [
        "## Code Map (Stage 1 Parser Output)",
        f"- Framework     : {cm.framework.value}",
        f"- PHP Version   : {cm.php_version or 'unknown'}",
        f"- Total Files   : {cm.total_files}",
        f"- Total Lines   : {cm.total_lines}",
        f"- Controllers   : {len(cm.controllers)}",
        f"- Models        : {len(cm.models)}",
        f"- Services      : {len(cm.services)}",
        f"- Routes        : {len(cm.routes)}",
        f"- DB Tables     : {len(cm.db_schema)}",
        f"- HTML Pages    : {len(cm.html_pages)}",
        f"- SQL Queries   : {len(cm.sql_queries)}",
        f"- Auth Signals  : {len(cm.auth_signals)}",
    ]
    if cm.http_endpoints:
        sample = ", ".join(
            f"{e.get('method','?')} {e.get('path','?')}"
            for e in cm.http_endpoints[:10]
        )
        lines.append(
            f"- HTTP Endpoints: {sample}"
            + (" …" if len(cm.http_endpoints) > 10 else "")
        )
    if cm.db_schema:
        lines.append(
            "- DB Schema     : "
            + ", ".join(t.get("table", "?") for t in cm.db_schema[:15])
        )
    return "\n".join(lines)


def _format_graph_meta(gm: GraphMeta) -> str:
    return "\n".join([
        "## Knowledge Graph (Stage 2)",
        f"- Nodes      : {gm.node_count}",
        f"- Edges      : {gm.edge_count}",
        f"- Node types : {', '.join(gm.node_types) or 'n/a'}",
        f"- Edge types : {', '.join(gm.edge_types) or 'n/a'}",
    ])


def _format_domain_model(dm: DomainModel) -> str:
    lines = [
        "## Domain Model (Stage 4)",
        f"Domain      : {dm.domain_name}",
        f"Description : {dm.description}",
    ]
    if dm.user_roles:
        lines.append("User Roles  :")
        for role in dm.user_roles:
            lines.append(f"  - {role.get('name', '?')}: {role.get('description', '')}")
    if dm.bounded_contexts:
        lines.append("Bounded Contexts: " + ", ".join(dm.bounded_contexts))
    if dm.key_entities:
        lines.append("Key Entities    : " + ", ".join(dm.key_entities))
    if dm.features:
        lines.append("Features:")
        for feat in dm.features[:20]:
            lines.append(f"  - {feat.get('name', '?')}: {feat.get('description', '')}")
        if len(dm.features) > 20:
            lines.append(f"  … and {len(dm.features) - 20} more")
    return "\n".join(lines)


def _format_business_flows(bfc: BusinessFlowCollection) -> str:
    lines = [f"## Business Flows (Stage 4.5) — {bfc.total} flows"]
    for flow in bfc.flows:
        lines.append(f"\n### {flow.flow_id}: {flow.name}")
        lines.append(f"  Actor   : {flow.actor}")
        lines.append(f"  Context : {flow.bounded_context}")
        lines.append(f"  Trigger : {flow.trigger}")
        lines.append(f"  End     : {flow.termination}")
        lines.append("  Steps   :")
        for step in flow.steps:
            db_note = f" [DB: {', '.join(step.db_ops)}]" if step.db_ops else ""
            auth    = " [AUTH]" if step.auth_required else ""
            lines.append(
                f"    {step.step_num}. [{step.http_method or '?'}] "
                f"{step.page} — {step.action}{db_note}{auth}"
            )
        if flow.branches:
            lines.append(f"  Branches: {len(flow.branches)}")
    return "\n".join(lines)


def _format_qa_result(qa: QAResult) -> str:
    lines = [
        "## QA Review (Stage 6)",
        f"- Passed            : {qa.passed}",
        f"- Coverage Score    : {qa.coverage_score:.2f}",
        f"- Consistency Score : {qa.consistency_score:.2f}",
    ]
    if qa.issues:
        lines.append(f"- Open Issues ({len(qa.issues)}):")
        for issue in qa.issues[:5]:
            lines.append(
                f"  • [{issue.get('severity', '?')}] {issue.get('description', '')}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the configured LLM provider via llm_client and return the response text.

    Provider and model are resolved by llm_client (GEMINI_API_KEY / ANTHROPIC_API_KEY
    / LLM_PROVIDER env vars).  Retry and rate-limit handling is done in llm_client.
    """
    return call_llm(
        system_prompt = system_prompt,
        user_prompt   = user_prompt,
        max_tokens    = MAX_TOKENS,
        temperature   = 0.1,   # architecture: deterministic JSON structure
        label         = STAGE_NAME,
    )


# ---------------------------------------------------------------------------
# JSON parsing & schema validation
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict[str, Any]:
    """
    Robustly extract a JSON object from LLM output.

    Handles four patterns (tried in order):
      1. ```json … ``` fenced code block
      2. Raw JSON / prose preamble — extract first { … last }
      3. Truncated JSON recovery — Gemini hits max_tokens mid-object;
         we incrementally strip trailing incomplete entries until
         json.loads() succeeds, then patch any required top-level keys
         with empty defaults so _validate_schema still passes.
      4. Hard failure with a clear diagnostic message.
    """
    text = text.strip()

    # ── Pattern 1: fenced code block ─────────────────────────────────────────
    fence = re.search(r"```(?:json)?\s*(\{.*)", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
        # strip closing fence if present
        candidate = re.sub(r"```\s*$", "", candidate, flags=re.DOTALL).strip()
        text = candidate

    # ── Pattern 2: strip prose preamble ──────────────────────────────────────
    if not text.startswith("{"):
        start = text.find("{")
        if start == -1:
            raise ValueError(
                f"[{STAGE_NAME}] LLM response contains no JSON object.\n"
                f"Preview: {text[:300]!r}"
            )
        text = text[start:]

    # ── Try direct parse first (fast path) ───────────────────────────────────
    # Strip to outermost { … } in case there is trailing prose after the JSON.
    end = text.rfind("}")
    clean = text[: end + 1] if end != -1 else text
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass  # fall through to truncation recovery

    # ── Pattern 3: truncation recovery ───────────────────────────────────────
    # Gemini (and occasionally Claude) truncates mid-value when it hits the
    # max_tokens limit.  Strategy: walk backwards through the text removing
    # the last incomplete JSON entry on each attempt until we get a valid
    # object, then fill in any missing required top-level keys with safe
    # defaults so downstream validation can still succeed.
    recovered = _attempt_json_recovery(text)
    if recovered is not None:
        print(
            f"  [{STAGE_NAME}] ⚠️  JSON was truncated — recovered partial response. "
            f"Some fields may be incomplete."
        )
        return recovered

    # ── Pattern 4: hard failure ───────────────────────────────────────────────
    try:
        json.loads(text)   # re-run to get the clean error message
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"[{STAGE_NAME}] Failed to parse LLM JSON: {exc}\n"
            f"Offending text: {text[:400]!r}"
        ) from exc
    # Should never reach here
    raise ValueError(f"[{STAGE_NAME}] JSON parse failed for unknown reason.")


def _attempt_json_recovery(text: str) -> dict[str, Any] | None:
    """
    Try to salvage a truncated JSON object by progressively trimming the
    trailing incomplete content and closing any open arrays/objects.

    Algorithm:
      - Find the last position of a clearly-complete value (a closing ] or }
        or a quoted string end) and insert the necessary closing brackets to
        make the JSON valid.
      - Try up to MAX_RECOVERY_ATTEMPTS truncation points, stepping backward
        through the text on each failure.

    Returns parsed dict on success, None if all attempts fail.
    """
    MAX_RECOVERY_ATTEMPTS = 12

    # Characters that typically end a complete JSON value
    _GOOD_TRAIL = re.compile(r'[\]}"\'\d]')

    candidate = text.strip()

    for _ in range(MAX_RECOVERY_ATTEMPTS):
        # Strip the last character and any trailing whitespace/comma
        candidate = candidate.rstrip().rstrip(",").rstrip()
        if not candidate:
            break

        # Count unclosed braces/brackets to figure out what we need to close
        closed = _close_json(candidate)
        try:
            return json.loads(closed)
        except json.JSONDecodeError:
            # Remove the last "token" — walk back to the previous structural char
            last_good = max(
                candidate.rfind("}"),
                candidate.rfind("]"),
                candidate.rfind('"'),
            )
            if last_good <= 0:
                break
            candidate = candidate[: last_good + 1]

    return None


def _close_json(text: str) -> str:
    """
    Given a truncated JSON string, append the minimum closing brackets/braces
    needed to make it syntactically valid.

    Tracks the nesting stack while ignoring characters inside string literals.
    Returns the completed string.
    """
    stack: list[str] = []
    in_string = False
    escape    = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    # Close any open arrays first, then objects (LIFO order of the stack)
    return text + "".join(reversed(stack))


def _validate_schema(data: dict[str, Any]) -> None:
    """
    Validate the parsed architecture dict has the expected top-level keys
    and types.  Raises ValueError with a clear message on the first violation.
    """
    required: dict[str, type] = {
        "overview":                dict | str,
        "components":              list,
        "data_flows":              list,
        "sequence_flows":          list,
        "integration_points":      list,
        "technology_observations": dict,
    }
    for key, expected in required.items():
        if key not in data:
            raise ValueError(f"[{STAGE_NAME}] LLM JSON missing required key: '{key}'")
        if not isinstance(data[key], expected):
            raise ValueError(
                f"[{STAGE_NAME}] Key '{key}' expected {expected}, "
                f"got {type(data[key]).__name__}"
            )

    # Component shape
    for i, comp in enumerate(data["components"]):
        if not isinstance(comp, dict):
            raise ValueError(f"[{STAGE_NAME}] components[{i}] must be a dict")
        for f in ("name", "description", "type"):
            if f not in comp:
                raise ValueError(f"[{STAGE_NAME}] components[{i}] missing field '{f}'")

    # Data-flow shape
    for i, df in enumerate(data["data_flows"]):
        if not isinstance(df, dict):
            raise ValueError(f"[{STAGE_NAME}] data_flows[{i}] must be a dict")
        for f in ("name", "steps"):
            if f not in df:
                raise ValueError(f"[{STAGE_NAME}] data_flows[{i}] missing field '{f}'")
        if not isinstance(df["steps"], list):
            raise ValueError(f"[{STAGE_NAME}] data_flows[{i}]['steps'] must be a list")

    # Sequence-flow shape
    for i, sf in enumerate(data["sequence_flows"]):
        if not isinstance(sf, dict):
            raise ValueError(f"[{STAGE_NAME}] sequence_flows[{i}] must be a dict")
        for f in ("name", "actors", "steps"):
            if f not in sf:
                raise ValueError(f"[{STAGE_NAME}] sequence_flows[{i}] missing field '{f}'")


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _build_markdown(data: dict[str, Any]) -> str:
    """Render the validated architecture dict into a well-structured Markdown document."""
    md: list[str] = []

    def h1(t: str)  -> None: md.append(f"# {t}\n")
    def h2(t: str)  -> None: md.append(f"## {t}\n")
    def h3(t: str)  -> None: md.append(f"### {t}\n")
    def p(t: str)   -> None: md.append(f"{t}\n")
    def li(t: str)  -> None: md.append(f"- {t}")
    def sep()       -> None: md.append("")

    h1("System Architecture")

    # 1. Overview ─────────────────────────────────────────────────────────────
    h2("1. System Overview")
    overview = data.get("overview", "")
    p(overview if isinstance(overview, str) else json.dumps(overview, indent=2))

    # 2. Component Architecture ───────────────────────────────────────────────
    h2("2. Component Architecture")
    components = data.get("components", [])
    if components:
        # Group by component type for readability
        by_type: dict[str, list[dict]] = {}
        for comp in components:
            by_type.setdefault(comp.get("type", "Other"), []).append(comp)
        for comp_type, comps in sorted(by_type.items()):
            h3(comp_type)
            for comp in comps:
                li(f"**{comp['name']}** — {comp['description']}")
                for resp in comp.get("responsibilities", []):
                    md.append(f"  - {resp}")
            sep()
    else:
        p("_No components identified._")

    # 3. Data Flows ───────────────────────────────────────────────────────────
    h2("3. Data Flows")
    data_flows = data.get("data_flows", [])
    if data_flows:
        for df in data_flows:
            h3(df["name"])
            if df.get("description"):
                p(df["description"])
            for step in df["steps"]:
                li(step)
            sep()
    else:
        p("_No data flows identified._")

    # 4. Sequence Flows ───────────────────────────────────────────────────────
    h2("4. Key Sequence Flows")
    sequence_flows = data.get("sequence_flows", [])
    if sequence_flows:
        for sf in sequence_flows:
            h3(sf["name"])
            if sf.get("description"):
                p(sf["description"])
            actors = sf.get("actors", [])
            if actors:
                p(f"**Actors:** {', '.join(actors)}")
            for step in sf["steps"]:
                li(step)
            sep()
    else:
        p("_No sequence flows identified._")

    # 5. Integration Points ───────────────────────────────────────────────────
    h2("5. Integration Points")
    integration_points = data.get("integration_points", [])
    if integration_points:
        for ip in integration_points:
            if isinstance(ip, dict):
                li(
                    f"**{ip.get('name', '?')}** "
                    f"({ip.get('type', '?')}) — {ip.get('description', '')}"
                )
            else:
                li(str(ip))
        sep()
    else:
        p("_No integration points identified._")

    # 6. Technology Observations ──────────────────────────────────────────────
    h2("6. Technology Observations")
    tech = data.get("technology_observations", {})
    if tech:
        for section, label in (
            ("stack",           "Stack"),
            ("patterns",        "Patterns & Practices"),
            ("risks",           "Technical Risks"),
            ("recommendations", "Recommendations"),
        ):
            items = tech.get(section, [])
            if items:
                h3(label)
                for item in items:
                    li(item)
                sep()
    else:
        p("_No technology observations available._")

    return "\n".join(md)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior software architect specialising in reverse-engineering
    legacy PHP systems into clear, structured architecture documentation.

    Your task is to analyse the provided pipeline artefacts (code map, knowledge
    graph summary, domain model, and business flows) and reconstruct a complete
    system architecture picture.

    Rules:
    - Base ALL conclusions strictly on the provided data. Do not invent components
      or flows that are not evidenced by the inputs.
    - Use precise, technical language appropriate for a software architect audience.
    - Component types must be one of: "Frontend", "Backend", "Database", "Service",
      "Middleware", "External", "Configuration".
    - Return ONLY a valid JSON object — no preamble, no explanation, no markdown fences.
""")

_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    Analyse the following pipeline artefacts and return a single JSON object
    with exactly these keys:

    {{
      "overview": "<2-4 sentence system overview>",

      "components": [
        {{
          "name": "<component name>",
          "type": "<Frontend|Backend|Database|Service|Middleware|External|Configuration>",
          "description": "<what it does>",
          "responsibilities": ["<responsibility 1>", "..."]
        }}
      ],

      "data_flows": [
        {{
          "name": "<flow name>",
          "description": "<optional brief description>",
          "steps": ["<step 1 e.g. 'User submits form → login.php'>", "..."]
        }}
      ],

      "sequence_flows": [
        {{
          "name": "<flow name>",
          "description": "<optional brief description>",
          "actors": ["<Actor 1>", "..."],
          "steps": ["<step 1>", "..."]
        }}
      ],

      "integration_points": [
        {{
          "name": "<integration name>",
          "type": "<Database|Session|FileSystem|HTTP|External>",
          "description": "<what integrates with what and how>"
        }}
      ],

      "technology_observations": {{
        "stack":           ["<tech 1>", "..."],
        "patterns":        ["<pattern 1>", "..."],
        "risks":           ["<risk 1>", "..."],
        "recommendations": ["<rec 1>", "..."]
      }}
    }}

    --- PIPELINE ARTEFACTS ---

    {context_block}
""")
