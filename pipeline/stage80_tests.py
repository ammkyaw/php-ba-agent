"""
pipeline/stage80_tests.py — Test Case Generator (Stage 8)

Converts BA artefacts into executable test assets:

  tests.feature        — Gherkin BDD scenarios (one per AC criterion)
  playwright_tests.js  — Playwright JS end-to-end tests mirroring the Gherkin
  pytest_tests.py      — pytest unit/integration stubs for backend logic

Sources (in priority order, all combined into a single LLM context):
  ctx.ba_artifacts.ac_path        — Acceptance Criteria Markdown  [required]
  ctx.business_flows              — FlowStep sequences             [enriches scenarios]
  ctx.domain_model                — actors, entities, bounded ctxs [enriches test data]

Placement in the pipeline
--------------------------
  … stage70_pdf → stage80_tests

Stage 8 is a standalone QA deliverable that sits after the BA document bundle
is finalised.  It does not modify any earlier artefact.

LLM strategy
------------
The AC file may be large.  We split it into bounded-context chunks so each LLM
call stays under _MAX_AC_CHARS.  Results are merged into a single feature file
and test file at the end.

Resume behaviour
----------------
If stage80_tests is COMPLETED and all three output files exist the stage is
skipped.  Pass --force to regenerate.

Output layout
-------------
outputs/run_<id>/
  tests.feature          — Gherkin feature file
  playwright_tests.js    — Playwright spec
  pytest_tests.py        — pytest stubs
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

from context import (
    BusinessFlow,
    DomainModel,
    PipelineContext,
    TestSuiteArtifacts,
)
from pipeline.llm_client import call_llm

# ── Constants ──────────────────────────────────────────────────────────────────

STAGE_NAME = "stage80_tests"
MAX_TOKENS = 8192   # per-format call — one format per call avoids truncation

# AC text budget per LLM call — keeps prompt + response within model limits
_MAX_AC_CHARS    = 3_500  # halved — smaller input → smaller output → less truncation risk
# Maximum business flows to include in context (most relevant first)
_MAX_FLOWS       = 5
# Maximum steps per flow in the context block
_MAX_FLOW_STEPS  = 6

OUTPUT_FILES = {
    "gherkin":    "tests.feature",
    "playwright": "playwright_tests.js",
    "pytest":     "pytest_tests.py",
}


# ── Public Entry Point ─────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 8 entry point.  Generates Gherkin, Playwright, and pytest artefacts
    from the pipeline's BA outputs.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If required upstream stages have not been run.
    """
    gherkin_path    = ctx.output_path(OUTPUT_FILES["gherkin"])
    playwright_path = ctx.output_path(OUTPUT_FILES["playwright"])
    pytest_path     = ctx.output_path(OUTPUT_FILES["pytest"])

    # ── Resume check ──────────────────────────────────────────────────────────
    if ctx.is_stage_done(STAGE_NAME) and _all_outputs_exist(ctx):
        print(f"  [{STAGE_NAME}] Already completed — skipping.")
        return

    # ── Pre-flight ────────────────────────────────────────────────────────────
    _assert_prerequisites(ctx)

    print(f"  [{STAGE_NAME}] Loading AC artefact ...")
    ac_text = Path(ctx.ba_artifacts.ac_path).read_text(encoding="utf-8")

    # ── Build shared context block (flows + domain) ───────────────────────────
    context_block = _build_context_block(ctx)

    # ── Split AC into bounded-context chunks ──────────────────────────────────
    chunks = _chunk_ac(ac_text)
    print(f"  [{STAGE_NAME}] AC split into {len(chunks)} chunk(s)")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pipeline.llm_client import get_max_workers

    all_gherkin:    list[str] = ["" for _ in chunks]
    all_playwright: list[str] = ["" for _ in chunks]
    all_pytest:     list[str] = ["" for _ in chunks]

    def _single_task(idx_0: int, chunk: str, fmt: str) -> tuple[int, str, str]:
        idx = idx_0 + 1
        base_user = _USER_PROMPT_BASE.format(
            context_block = context_block,
            ac_chunk      = chunk,
            chunk_index   = idx,
            total_chunks  = len(chunks),
        )
        return idx_0, fmt, _call_single_format(base_user, fmt, idx)

    def _store(idx_0: int, fmt: str, res: str) -> None:
        """Write a completed task result back into the correct output slot."""
        if fmt == "gherkin":       all_gherkin[idx_0]    = res
        elif fmt == "playwright":  all_playwright[idx_0] = res
        elif fmt == "pytest":      all_pytest[idx_0]     = res

    tasks = []
    for idx_0, chunk in enumerate(chunks):
        for fmt in ["gherkin", "playwright", "pytest"]:
            tasks.append((idx_0, chunk, fmt))

    workers     = get_max_workers()
    concurrency = min(len(tasks), workers)
    print(
        f"  [{STAGE_NAME}] {len(tasks)} LLM calls total "
        f"({len(chunks)} chunk(s) × 3 formats) — "
        f"{concurrency} running concurrently ..."
    )

    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_task = {pool.submit(_single_task, *t): t for t in tasks}
            failed = 0
            for fut in as_completed(future_to_task):
                idx_0, chunk, fmt = future_to_task[fut]
                try:
                    idx_0, fmt, res = fut.result()
                    _store(idx_0, fmt, res)
                    print(f"  [{STAGE_NAME}] ✓ chunk {idx_0 + 1}/{len(chunks)} [{fmt}]")
                except Exception as exc:
                    failed += 1
                    print(
                        f"  [{STAGE_NAME}] ✗ chunk {idx_0 + 1}/{len(chunks)} [{fmt}] "
                        f"failed — {exc}"
                    )
            if failed:
                raise RuntimeError(
                    f"[{STAGE_NAME}] {failed}/{len(tasks)} task(s) failed. "
                    f"See log above for details."
                )
    else:
        for t in tasks:
            idx_0, chunk, fmt = t
            try:
                idx_0, fmt, res = _single_task(*t)
                _store(idx_0, fmt, res)
                print(f"  [{STAGE_NAME}] ✓ chunk {idx_0 + 1}/{len(chunks)} [{fmt}]")
            except Exception as exc:
                raise RuntimeError(
                    f"[{STAGE_NAME}] chunk {idx_0 + 1} [{fmt}] failed — {exc}"
                ) from exc

    # ── Merge chunks into final files ─────────────────────────────────────────
    domain_name = ctx.domain_model.domain_name if ctx.domain_model else "System"

    gherkin_merged    = _merge_gherkin(all_gherkin, domain_name)
    playwright_merged = _merge_playwright(all_playwright, domain_name)
    pytest_merged     = _merge_pytest(all_pytest, domain_name)

    # ── Write output files ────────────────────────────────────────────────────
    Path(gherkin_path).write_text(gherkin_merged, encoding="utf-8")
    Path(playwright_path).write_text(playwright_merged, encoding="utf-8")
    Path(pytest_path).write_text(pytest_merged, encoding="utf-8")

    print(f"  [{STAGE_NAME}] ✓ {OUTPUT_FILES['gherkin']}")
    print(f"  [{STAGE_NAME}] ✓ {OUTPUT_FILES['playwright']}")
    print(f"  [{STAGE_NAME}] ✓ {OUTPUT_FILES['pytest']}")

    # ── Update context ────────────────────────────────────────────────────────
    scenario_count = gherkin_merged.count("\n  Scenario")
    ctx.test_suite = TestSuiteArtifacts(
        gherkin_path    = gherkin_path,
        playwright_path = playwright_path,
        pytest_path     = pytest_path,
        scenario_count  = scenario_count,
    )
    ctx.stage(STAGE_NAME).mark_completed(ctx.output_dir)
    ctx.save()

    print(f"  [{STAGE_NAME}] Complete — {scenario_count} scenario(s) generated.")


# ── Pre-flight ─────────────────────────────────────────────────────────────────

def _assert_prerequisites(ctx: PipelineContext) -> None:
    """Raise RuntimeError with actionable message if upstream outputs are missing."""
    if ctx.ba_artifacts is None:
        raise RuntimeError(
            f"[{STAGE_NAME}] ctx.ba_artifacts is None — run Stage 5 first.\n"
            f"  python run_pipeline.py --resume <context.json> --until stage50_ac"
        )
    if not ctx.ba_artifacts.ac_path:
        raise RuntimeError(
            f"[{STAGE_NAME}] ctx.ba_artifacts.ac_path is empty — "
            f"Stage 5 AC agent may have failed."
        )
    ac_path = Path(ctx.ba_artifacts.ac_path)
    if not ac_path.exists():
        raise RuntimeError(
            f"[{STAGE_NAME}] AC file not found: {ac_path}\n"
            f"  Re-run Stage 5: python run_pipeline.py --resume <context.json> "
            f"--until stage50_ac --force stage50_ac"
        )


def _all_outputs_exist(ctx: PipelineContext) -> bool:
    """Return True if all three output files already exist."""
    return all(
        Path(ctx.output_path(fname)).exists()
        for fname in OUTPUT_FILES.values()
    )


# ── Context Block ──────────────────────────────────────────────────────────────

def _build_context_block(ctx: PipelineContext) -> str:
    """
    Build the supplementary context block sent to the LLM alongside each AC
    chunk.  Includes domain roles and the most-confident business flows
    (capped at _MAX_FLOWS) to help the LLM generate realistic test data and
    accurate actor names.
    """
    parts: list[str] = []

    # Domain context
    dm = ctx.domain_model
    if dm:
        parts.append(f"DOMAIN: {dm.domain_name}")
        if dm.user_roles:
            roles_str = ", ".join(r["role"] for r in dm.user_roles)
            parts.append(f"ACTORS: {roles_str}")
        if dm.key_entities:
            parts.append(f"KEY ENTITIES: {', '.join(dm.key_entities[:8])}")
        parts.append("")

    # Business flows — ordered by confidence, capped
    bfc = ctx.business_flows
    if bfc and bfc.flows:
        flows_sorted = sorted(bfc.flows, key=lambda f: f.confidence, reverse=True)
        parts.append(f"BUSINESS FLOWS ({min(len(flows_sorted), _MAX_FLOWS)} most confident):")
        for flow in flows_sorted[:_MAX_FLOWS]:
            parts.append(f"\n  [{flow.flow_id}] {flow.name}  (actor: {flow.actor})")
            parts.append(f"  Trigger: {flow.trigger}")
            for step in flow.steps[:_MAX_FLOW_STEPS]:
                method = f" [{step.http_method}]" if step.http_method else ""
                auth   = " [AUTH]" if step.auth_required else ""
                db     = f" [DB: {step.db_ops[0][:40]}]" if step.db_ops else ""
                parts.append(f"    Step {step.step_num}: {step.page}{method}{auth}{db}")
            parts.append(f"  Termination: {flow.termination}")
        parts.append("")

    return "\n".join(parts)


# ── AC Chunking ────────────────────────────────────────────────────────────────

def _chunk_ac(ac_text: str) -> list[str]:
    """
    Split the AC Markdown into bounded-context chunks small enough for one LLM
    call each.

    Strategy:
      1. Try to split on H2 headings (## Feature / ## Context).
      2. If any chunk is still over _MAX_AC_CHARS, split on H3 headings.
      3. Hard-truncate as a last resort.

    Returns at least one chunk (the full text if it fits).
    """
    if len(ac_text) <= _MAX_AC_CHARS:
        return [ac_text]

    # Split on H2 headings
    h2_sections = re.split(r"(?=^## )", ac_text, flags=re.MULTILINE)
    chunks: list[str] = []

    for section in h2_sections:
        if not section.strip():
            continue
        if len(section) <= _MAX_AC_CHARS:
            chunks.append(section)
        else:
            # Split on H3 headings within this section
            h3_sections = re.split(r"(?=^### )", section, flags=re.MULTILINE)
            current = ""
            for h3 in h3_sections:
                if len(current) + len(h3) <= _MAX_AC_CHARS:
                    current += h3
                else:
                    if current:
                        chunks.append(current)
                    # Hard-truncate oversized H3 sections
                    current = h3[:_MAX_AC_CHARS]
            if current:
                chunks.append(current)

    return chunks if chunks else [ac_text[:_MAX_AC_CHARS]]


# ── LLM Response Parsing ───────────────────────────────────────────────────────

# Delimiter tokens used in the structured output format.
# Using delimiters instead of JSON avoids Gemini's tendency to emit literal
# (unescaped) newlines inside JSON string values, which produces invalid JSON
# that no parser can recover.
_DELIMITERS: dict[str, tuple[str, str]] = {
    "gherkin":    ("---GHERKIN---",    "---END_GHERKIN---"),
    "playwright": ("---PLAYWRIGHT---", "---END_PLAYWRIGHT---"),
    "pytest":     ("---PYTEST---",     "---END_PYTEST---"),
}


def _parse_llm_response(text: str) -> dict[str, Any]:
    """
    Extract the three test artefacts from the LLM response using delimiter tags.

    Format expected from the LLM:
        ---GHERKIN---
        <gherkin content>
        ---END_GHERKIN---
        ---PLAYWRIGHT---
        <playwright content>
        ---END_PLAYWRIGHT---
        ---PYTEST---
        <pytest content>
        ---END_PYTEST---

    Falls back to JSON extraction if none of the delimiters are found
    (for future compatibility or alternative model output).
    """
    result: dict[str, Any] = {}

    for key, (start_tag, end_tag) in _DELIMITERS.items():
        pattern = re.escape(start_tag) + r"\s*(.*?)\s*" + re.escape(end_tag)
        m = re.search(pattern, text, re.DOTALL)
        if m:
            result[key] = m.group(1).strip()

    if result:
        return result

    # Fallback: try JSON extraction (handles models that ignore the delimiter format)
    return _parse_json_fallback(text)


def _parse_json_fallback(text: str) -> dict[str, Any]:
    """
    Last-resort JSON extraction for models that ignore the delimiter format.
    Handles fenced blocks and truncated JSON via bracket-completion recovery.
    """
    text = text.strip()

    fence = re.search(r"```(?:json)?\s*(\{.*)", text, re.DOTALL)
    if fence:
        text = re.sub(r"```\s*$", "", fence.group(1), flags=re.DOTALL).strip()
    elif not text.startswith("{"):
        start = text.find("{")
        if start == -1:
            return {}
        text = text[start:]

    # raw_decode stops at the first complete JSON object — handles "Extra data"
    # errors when the model emits a second object or trailing thinking tokens.
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    end = text.rfind("}")
    clean = text[: end + 1] if end != -1 else text
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    candidate = clean
    for _ in range(10):
        candidate = candidate.rstrip().rstrip(",").rstrip()
        try:
            return json.loads(_close_brackets(candidate))
        except json.JSONDecodeError:
            last = max(candidate.rfind("}"), candidate.rfind("]"), candidate.rfind('"'))
            if last <= 0:
                break
            candidate = candidate[:last + 1]

    return {}


def _close_brackets(text: str) -> str:
    """Append minimum closing brackets to make truncated JSON syntactically valid."""
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
        elif ch in ("}", "]") and stack and stack[-1] == ch:
            stack.pop()
    return text + "".join(reversed(stack))


def _validate_response(data: dict[str, Any], chunk_idx: int) -> None:
    """Warn (don't raise) if expected keys are missing — we merge what we have."""
    for key in ("gherkin", "playwright", "pytest"):
        if not data.get(key):
            print(f"  [{STAGE_NAME}] ⚠️  Chunk {chunk_idx}: '{key}' missing from LLM response.")


# ── Output Merging ─────────────────────────────────────────────────────────────

def _merge_gherkin(chunks: list[str], domain_name: str) -> str:
    """
    Merge per-chunk Gherkin into a single .feature file.

    The first Feature: declaration is kept; duplicate Feature: lines from
    subsequent chunks are stripped so we end up with one coherent feature file.
    """
    lines: list[str] = [
        f"# Generated by PHP-BA Agent — Stage 8",
        f"# Domain: {domain_name}",
        "",
        f"Feature: {domain_name} — Acceptance Test Suite",
        "",
    ]

    for chunk in chunks:
        for line in chunk.splitlines():
            # Skip duplicate Feature: headers from later chunks
            stripped = line.strip()
            if stripped.lower().startswith("feature:"):
                continue
            lines.append(line)
        lines.append("")   # blank line between chunks

    return "\n".join(lines)


def _merge_playwright(chunks: list[str], domain_name: str) -> str:
    """
    Merge per-chunk Playwright specs into one .js file.

    Strips duplicate import/require statements and wraps everything in a single
    top-level describe block if the LLM produced multiple describe blocks.
    """
    header = textwrap.dedent(f"""\
        // Generated by PHP-BA Agent — Stage 8
        // Domain: {domain_name}
        // Run with: npx playwright test playwright_tests.js

        const {{ test, expect }} = require('@playwright/test');

    """)

    # Collect all describe/test blocks, stripping duplicate imports
    body_lines: list[str] = []
    seen_imports: set[str] = set()

    for chunk in chunks:
        for line in chunk.splitlines():
            stripped = line.strip()
            # Deduplicate require/import lines
            if stripped.startswith(("const {", "import ", "const playwright",
                                    "require(", "const { test")):
                if stripped not in seen_imports:
                    seen_imports.add(stripped)
                # Always skip — header already has the canonical import
                continue
            body_lines.append(line)
        body_lines.append("")

    return header + "\n".join(body_lines)


def _merge_pytest(chunks: list[str], domain_name: str) -> str:
    """
    Merge per-chunk pytest stubs into one .py file.

    Strips duplicate import blocks and combines test functions.
    """
    slug = re.sub(r"[^a-z0-9]", "_", domain_name.lower()).strip("_")
    header = textwrap.dedent(f"""\
        # Generated by PHP-BA Agent — Stage 8
        # Domain: {domain_name}
        # Run with: pytest pytest_tests.py -v

        import pytest
        import requests

        BASE_URL = "http://localhost"   # update to match your dev server

    """)

    body_lines: list[str] = []
    seen_imports: set[str] = set()

    for chunk in chunks:
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "BASE_URL", "# Generated")):
                if stripped not in seen_imports:
                    seen_imports.add(stripped)
                continue
            body_lines.append(line)
        body_lines.append("")

    return header + "\n".join(body_lines)


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""    You are a senior QA automation engineer specialising in BDD and end-to-end
    testing for PHP web applications.

    You will receive:
    1. CONTEXT — domain information and business flows extracted from the codebase
    2. ACCEPTANCE CRITERIA — a section of the BA acceptance criteria document

    Your job is to convert each acceptance criterion into three test artefacts.

    GHERKIN (.feature):
    - One Scenario per acceptance criterion
    - Use Given/When/Then/And steps
    - Use realistic test data (valid emails, passwords, IDs)
    - Tag each scenario with its bounded context: @authentication @booking etc.
    - Do NOT emit a Feature: line — it will be added by the pipeline

    PLAYWRIGHT (JavaScript):
    - One test() block per Gherkin scenario
    - Use page.goto(), page.fill(), page.click(), expect(page)
    - Use await page.waitForURL() or await expect(page).toHaveURL() after navigation
    - Use the same realistic test data as Gherkin
    - Group tests in a describe() block named after the bounded context
    - Do NOT emit import/require lines — they will be added by the pipeline

    PYTEST (Python):
    - One test function per criterion, prefixed with test_
    - Use the requests library for HTTP calls
    - Write REAL assertions using assert response.status_code, assert "keyword" in response.text
    - Only add @pytest.mark.skip(reason="needs live server") if the test requires a running
      server to execute — do NOT skip tests that can be written as pure unit or HTTP assertions
    - Do NOT emit import lines — they will be added by the pipeline

    OUTPUT FORMAT — respond using EXACTLY these delimiter lines, no JSON:
    ---GHERKIN---
    <gherkin scenarios here>
    ---END_GHERKIN---
    ---PLAYWRIGHT---
    <playwright tests here>
    ---END_PLAYWRIGHT---
    ---PYTEST---
    <pytest stubs here>
    ---END_PYTEST---

    Produce nothing outside these delimiter blocks. No preamble. No explanation.
""")

# Base user prompt — format label and delimiter injected per call
_USER_PROMPT_BASE = textwrap.dedent("""    === CONTEXT ===
    {context_block}

    === ACCEPTANCE CRITERIA (chunk {chunk_index} of {total_chunks}) ===
    {ac_chunk}
""")

# Keep old name as alias so any external references still work
_USER_PROMPT_TEMPLATE = _USER_PROMPT_BASE

_FORMAT_INSTRUCTIONS: dict[str, str] = {
    "gherkin": textwrap.dedent("""\
        Generate ONLY Gherkin scenarios for every acceptance criterion above.
        Use Given/When/Then/And steps. Tag each scenario. No Feature: line.
        Wrap output EXACTLY like this — nothing outside the tags:
        ---GHERKIN---
        (scenarios here)
        ---END_GHERKIN---
    """),
    "playwright": textwrap.dedent("""\
        Generate ONLY Playwright (JavaScript) test blocks for every acceptance
        criterion above. One test() per criterion inside a describe() block.
        Use page.goto/fill/click/expect. No import lines.
        Wrap output EXACTLY like this — nothing outside the tags:
        ---PLAYWRIGHT---
        (tests here)
        ---END_PLAYWRIGHT---
    """),
    "pytest": textwrap.dedent("""\
        Generate ONLY pytest stubs for every acceptance criterion above.
        One test_ function per criterion. Use requests library. No import lines.
        Wrap output EXACTLY like this — nothing outside the tags:
        ---PYTEST---
        (stubs here)
        ---END_PYTEST---
    """),
}

_FORMAT_SYSTEM: dict[str, str] = {
    "gherkin": textwrap.dedent("""\
        You are a senior QA engineer specialising in BDD testing.
        Write Gherkin .feature scenarios from acceptance criteria.
        Produce ONLY the content between the delimiter tags requested.
        No preamble. No explanation. No JSON.
    """),
    "playwright": textwrap.dedent("""\
        You are a senior QA automation engineer specialising in Playwright E2E tests.
        Write Playwright JavaScript test blocks from acceptance criteria.
        Produce ONLY the content between the delimiter tags requested.
        No preamble. No explanation. No JSON. No import/require lines.
    """),
    "pytest": textwrap.dedent("""\
        You are a senior QA automation engineer specialising in pytest API tests.
        Write pytest stubs using the requests library from acceptance criteria.
        Produce ONLY the content between the delimiter tags requested.
        No preamble. No explanation. No JSON. No import lines.
    """),
}


def _extract_format_content(raw: str, fmt: str) -> str:
    """
    Extract format content from raw LLM response using multiple strategies:
    1. Delimiter tags  ---GHERKIN--- / ---END_GHERKIN---
    2. Markdown code fence  ```gherkin / ``` or ```javascript / ``` or ```python / ```
    3. Truncated delimiter (start tag present, end tag missing)
    4. Bare response (whole response treated as content)
    """
    start_tag, end_tag = _DELIMITERS[fmt]

    # Strategy 1: delimiters
    pattern = re.escape(start_tag) + r"\s*(.*?)\s*" + re.escape(end_tag)
    m = re.search(pattern, raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Strategy 2: markdown code fences
    _FENCE_LANGS: dict[str, list[str]] = {
        "gherkin":    ["gherkin", "feature", "cucumber"],
        "playwright": ["javascript", "js", "typescript", "ts"],
        "pytest":     ["python", "py"],
    }
    for lang in _FENCE_LANGS.get(fmt, []):
        fence_pat = rf"```{lang}\s*(.*?)\s*```"
        fm = re.search(fence_pat, raw, re.DOTALL | re.IGNORECASE)
        if fm:
            return fm.group(1).strip()
    # generic fence
    generic = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    if generic:
        return generic.group(1).strip()

    # Strategy 3: truncated delimiter
    if start_tag in raw:
        return raw[raw.index(start_tag) + len(start_tag):].strip()

    # Strategy 4: bare response — use as-is if non-empty
    stripped = raw.strip()
    if stripped:
        return stripped

    return ""


def _call_single_format(base_user: str, fmt: str, chunk_idx: int) -> str:
    """Call LLM for one format (gherkin/playwright/pytest) and return extracted content."""
    start_tag, _end_tag = _DELIMITERS[fmt]
    label = f"{STAGE_NAME}_chunk{chunk_idx}_{fmt}"

    # Prefill: end the user prompt with the start tag so the model continues from it
    # rather than needing to emit the tag itself — more reliable on local models.
    user_prompt = base_user + "\n" + _FORMAT_INSTRUCTIONS[fmt] + f"\n{start_tag}\n"

    raw = call_llm(
        system_prompt = _FORMAT_SYSTEM[fmt],
        user_prompt   = user_prompt,
        max_tokens    = MAX_TOKENS,
        label         = label,
    )

    content = _extract_format_content(raw, fmt)
    if content:
        return content

    # Last resort retry
    print(f"  [{STAGE_NAME}] ⚠️  Chunk {chunk_idx} {fmt}: empty — retrying ...")
    retry_raw = call_llm(
        system_prompt = _FORMAT_SYSTEM[fmt],
        user_prompt   = user_prompt,
        max_tokens    = MAX_TOKENS,
        label         = f"{label}_retry",
    )
    result = _extract_format_content(retry_raw, fmt)
    if not result:
        print(f"  [{STAGE_NAME}] ⚠️  Chunk {chunk_idx} {fmt}: retry also empty — skipping.")
    return result