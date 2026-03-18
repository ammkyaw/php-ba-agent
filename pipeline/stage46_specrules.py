"""
pipeline/stage46_specrules.py — Specification Mining (Stage 4.6)

Synthesises all static analysis signals into formal, BA-ready business rules
with Given/When/Then structure.  Runs after all static stages (4.1–4.3) and
after the LLM-generated business flows (4.5).

Passes
------
Pass 1 — Promote Stage 2.9 invariants
    Take InvariantCollection rules (conf ≥ 0.6) and convert to Given/When/Then.
    Uses ONE batched LLM call (≤30 rules per batch) for natural-language
    formalization.  Falls back to static templates if LLM is unavailable.

Pass 2 — Mine state machine rules (static)
    For each StateMachine: emit WORKFLOW rules for valid transitions,
    STATE rules for dead/terminal states.

Pass 3 — Mine business flow rules (static)
    For each BusinessFlow: emit AUTHORIZATION rules (auth_required steps),
    WORKFLOW rules (branch conditions as preconditions/guards).

Pass 4 — Mine referential integrity rules (static)
    For each EntityRelationship: emit REFERENTIAL rules from FK constraints.

Deduplication
    Group candidates by (bounded_context, category, semantic_fingerprint).
    Merge sources; keep highest confidence.

Output
------
    SpecRuleCollection  →  4.6_specrules/spec_rules.json
    Markdown summary    →  4.6_specrules/spec_rules_summary.md
    ctx.spec_rules set on context

Consumed by
-----------
    Stage 4.7  (behavioral validation — validate flows against rules)
    Stage 5    (BRD business rules, SRS validation rules, AC Given/When/Then)
    Stage 9    (knowledge graph rule nodes + typed edges)
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from context import PipelineContext, SpecRule, SpecRuleCollection

# ── Tunables ──────────────────────────────────────────────────────────────────
MIN_INVARIANT_CONF   = 0.60   # minimum Stage 2.9 confidence to promote
LLM_BATCH_SIZE       = 15     # rules per LLM call (token budget)
                               # 25 → consistent truncation at item 24 (model output
                               # limit hit); 15 × ~280 chars/rule ≈ 4 200 chars —
                               # comfortably under any local-model output budget.
                               # Auto-split still catches edge-case overruns.
LLM_MAX_TOKENS       = 8192   # response budget per batch
MERGE_CONF_THRESHOLD = 0.65   # minimum confidence to keep in final collection

# ── Static GWT templates for Pass 1 fallback ─────────────────────────────────

_CATEGORY_GIVEN: dict[str, str] = {
    "VALIDATION":    "Given a user submits a form or performs an operation",
    "AUTHORIZATION": "Given a user attempts to access a protected resource",
    "STATE_TRANSITION": "Given the system processes a state change",
    "BUSINESS_LIMIT":   "Given a user performs an operation with a numerical constraint",
    "TEMPORAL":         "Given a time-sensitive operation is performed",
    "REFERENTIAL":      "Given a record references another entity",
}
_CATEGORY_WHEN: dict[str, str] = {
    "VALIDATION":       "When the input data is validated",
    "AUTHORIZATION":    "When access control is checked",
    "STATE_TRANSITION": "When the entity status changes",
    "BUSINESS_LIMIT":   "When the system enforces a business limit",
    "TEMPORAL":         "When a date or time constraint is evaluated",
    "REFERENTIAL":      "When referential integrity is checked",
}
_CATEGORY_THEN_PREFIX: dict[str, str] = {
    "VALIDATION":       "Then the system enforces",
    "AUTHORIZATION":    "Then the system allows or denies access based on",
    "STATE_TRANSITION": "Then the system enforces the lifecycle rule",
    "BUSINESS_LIMIT":   "Then the system enforces the limit",
    "TEMPORAL":         "Then the system enforces the temporal constraint",
    "REFERENTIAL":      "Then the system ensures referential integrity for",
}


def _static_gwt(category: str, description: str, entity: str) -> tuple[str, str, str]:
    """Produce a rough static Given/When/Then triple from a Stage 2.9 rule."""
    given = _CATEGORY_GIVEN.get(category, "Given the relevant operation is initiated")
    if entity and entity.lower() not in given.lower():
        given = f"Given a {entity} operation is initiated"

    when = _CATEGORY_WHEN.get(category, "When the business constraint is evaluated")

    then_prefix = _CATEGORY_THEN_PREFIX.get(category, "Then the system enforces")
    then = f"{then_prefix}: {description}"

    return given, when, then


def _fingerprint(category: str, context: str, description: str) -> str:
    """Stable 8-char hash for deduplication."""
    text = f"{category}|{context}|{description.lower()}"
    return hashlib.md5(text.encode()).hexdigest()[:8]


def _humanize_table(table: str) -> str:
    return table.replace("_", " ").title()


# ── Pass 1 — Promote Stage 2.9 invariants ────────────────────────────────────

_LLM_SYSTEM = """\
You are a Business Analyst converting technical code constraints into formal business rules.
For EACH constraint in the input array produce a JSON object with these exact keys:
  "id"          : the original id value, unchanged
  "title"       : ≤10-word business-friendly rule name (no code symbols)
  "given"       : precondition phrase starting with "Given "
  "when"        : trigger phrase starting with "When "
  "then"        : expected behaviour phrase starting with "Then "
  "confidence"  : float 0.0–1.0 — how likely this is a genuine business rule vs framework noise
  "tags"        : array of 1–4 lowercase keyword strings

CONFIDENCE GUIDE:
  0.9–1.0  Real domain rule: password length, booking time window, max quantity, payment validation
  0.6–0.8  Likely real: email format check, required field, unique constraint, auth redirect
  0.3–0.5  Ambiguous: generic status flags, soft-delete (deleted=0/1), ordering/pagination defaults
  0.0–0.2  Boilerplate — set ≤ 0.2 for these patterns:
             • CSRF / _token validation
             • Internal Laravel routing (middleware group, prefix, namespace)
             • Framework lifecycle hooks (boot(), register(), handle())
             • Auto-increment primary keys, created_at/updated_at timestamps
             • Debug/logging guards (APP_DEBUG, log level)

OUTPUT RULES (strictly enforced):
- Respond with a JSON ARRAY only — your response must begin with [ and end with ]
- No prose, no markdown fences, no explanation, no reasoning trace
- Do NOT write any text before or after the JSON array
Keep technical jargon OUT of given/when/then; use plain business language.
given/when/then must each be a single sentence — no bullet points or sub-clauses.
"""


def _extract_json_array(raw: str) -> list | None:
    """
    Try every reasonable strategy to pull a JSON array out of `raw`.

    Strategies (in order):
      1. Direct parse — model followed instructions exactly
      2. Strip ``` code fences, then parse
      3. raw_decode scan — iterate every '[' position; stops at the end of the
         first balanced JSON value, so it handles "Thinking Process:" preambles,
         markdown bullets/links that contain '[', and trailing prose after array
      4. Truncation recovery — if the response was cut mid-array, close it and
         drop the last (incomplete) element before parsing
      5. Dict wrapper — {"rules": [...]} etc.
    Returns the parsed list, or None if all strategies fail.
    """
    raw = raw.strip()

    def _is_rule_list(lst: list) -> bool:
        """True if lst looks like the expected array of rule dicts (not a tags/string list)."""
        return bool(lst) and isinstance(lst[0], dict)

    # 1. Direct parse
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and _is_rule_list(parsed):
                return parsed
        except json.JSONDecodeError:
            pass  # may be truncated — fall through to recovery

    # 2. Strip code fences
    clean = raw
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z]*\s*", "", clean)
        clean = re.sub(r"\s*```\s*$", "", clean).strip()
        if clean.startswith("["):
            try:
                parsed = json.loads(clean)
                if isinstance(parsed, list) and _is_rule_list(parsed):
                    return parsed
            except json.JSONDecodeError:
                pass

    # 3. Find the first balanced JSON array of OBJECTS anywhere in the response.
    #    Uses raw_decode (stops at the end of the first valid JSON value) so it
    #    handles:
    #      - "Thinking Process: ... \n[{...}]" preambles
    #      - markdown that contains "[" (bullets, links, code refs)
    #      - trailing prose / code fences after the array
    #    Skips string/number arrays (e.g. "tags": ["kw1", "kw2"] inside objects)
    #    because _is_rule_list requires the first element to be a dict.
    _decoder = json.JSONDecoder()
    for _m in re.finditer(r"\[", raw):
        try:
            _parsed, _ = _decoder.raw_decode(raw, _m.start())
            if isinstance(_parsed, list) and _is_rule_list(_parsed):
                return _parsed
        except json.JSONDecodeError:
            continue

    # 4. Truncation recovery — two sub-strategies for MAX_TOKENS cut-off:
    candidate = clean if clean.startswith("[") else raw
    if candidate.startswith("["):
        # 4a. Last object IS complete but closing "]" is missing — just append it.
        #     The model generated all items correctly but omitted the final "]".
        #     Zero data loss — no warning needed.
        try:
            parsed = json.loads(candidate.rstrip() + "]")
            if isinstance(parsed, list) and _is_rule_list(parsed):
                return parsed
        except json.JSONDecodeError:
            pass

        # 4b. Array cut mid-object — the last element is incomplete; drop it.
        #     Handles: '[{...}, {...}, {...'  →  '[{...}, {...}]'
        #     Real truncation: one item is lost and retried via auto-split.
        brace_open = candidate.rfind("{")
        if brace_open > 0:
            truncated = candidate[:brace_open].rstrip().rstrip(",") + "]"
            try:
                parsed = json.loads(truncated)
                if isinstance(parsed, list) and _is_rule_list(parsed):
                    print(f"  [stage46]   ⚠️  Truncated response — recovered {len(parsed)} item(s).")
                    return parsed
            except json.JSONDecodeError:
                pass

    # 5. dict wrapper {"rules": [...]}
    try:
        parsed = json.loads(clean or raw)
        if isinstance(parsed, dict):
            for key in ("rules", "items", "results", "data"):
                val = parsed.get(key)
                if isinstance(val, list) and _is_rule_list(val):
                    return val
    except json.JSONDecodeError:
        pass

    return None


def _static_fallback_items(candidates: list[dict]) -> list[dict]:
    """Build minimal static GWT for a list of candidates (used as fallback)."""
    results = []
    for c in candidates:
        cat  = c.get("category", "VALIDATION")
        desc = c.get("description", "")
        ent  = c.get("entity", "")
        given, when, then = _static_gwt(cat, desc, ent)
        results.append({
            "id":         c["id"],
            "title":      desc[:60],
            "given":      given,
            "when":       when,
            "then":       then,
            "confidence": c.get("confidence", 0.7),
            "tags":       [cat.lower()],
        })
    return results


def _llm_formalize_batch(candidates: list[dict], batch_num: int) -> list[dict]:
    """Send a batch of rule candidates to the LLM and return structured results.

    If the model's response is truncated (some candidate IDs missing from the
    output), the missing items are split in half and retried recursively.
    This continues until every item is either LLM-enriched or falls back to
    static GWT (when a single-item batch still fails).
    """
    if not candidates:
        return []

    try:
        from pipeline.llm_client import call_llm
        user_payload = json.dumps(candidates, ensure_ascii=False)
        raw = call_llm(
            system_prompt = _LLM_SYSTEM,
            user_prompt   = user_payload,
            max_tokens    = LLM_MAX_TOKENS,
            temperature   = 0.1,   # extraction: deterministic rule classification
            label         = f"spec_rules_batch_{batch_num}",
            prefill       = "[",   # force model to start JSON array immediately,
                                   # preventing any "Thinking Process:" preamble
        )

        if not raw or not raw.strip():
            print(f"  [stage46] ⚠️  LLM batch {batch_num} returned empty response; using static fallback.")
        else:
            parsed = _extract_json_array(raw)
            if parsed is not None:
                # Check for truncation: find candidate IDs not in the LLM output
                returned_ids = {
                    item.get("id", "") for item in parsed if isinstance(item, dict)
                }
                missed = [c for c in candidates if c["id"] not in returned_ids]

                if not missed:
                    return parsed   # all items accounted for ✓

                # Some items were cut off — split and retry rather than
                # silently falling back for the whole missed set
                print(
                    f"  [stage46]   ⚠️  {len(missed)} item(s) missing from batch "
                    f"{batch_num} response; retrying in smaller sub-batches …"
                )
                results = list(parsed)
                if len(missed) == 1:
                    # Single missing item — recurse once; will fall back if needed
                    results.extend(_llm_formalize_batch(missed, batch_num))
                else:
                    mid = len(missed) // 2
                    results.extend(_llm_formalize_batch(missed[:mid], batch_num))
                    results.extend(_llm_formalize_batch(missed[mid:], batch_num))
                return results

            # Could not extract any JSON array at all
            snippet = raw.strip()[:400].replace("\n", " ")
            print(f"  [stage46] ⚠️  LLM batch {batch_num} — could not extract JSON array. "
                  f"Response preview: {snippet!r}")
            print(f"  [stage46]   Using static fallback.")

    except Exception as exc:
        print(f"  [stage46] ⚠️  LLM batch {batch_num} failed ({exc}); using static fallback.")

    return _static_fallback_items(candidates)


def _pass1_from_invariants(ctx: PipelineContext) -> list[dict]:
    """Promote Stage 2.9 invariants → raw SpecRule dicts."""
    if not ctx.invariants or not ctx.invariants.rules:
        return []

    eligible = [r for r in ctx.invariants.rules if r.confidence >= MIN_INVARIANT_CONF]
    print(f"  [stage46] Pass 1 — {len(eligible)} invariant(s) eligible for promotion.")

    # Prepare batches for LLM
    batches: list[list[dict]] = []
    for i in range(0, len(eligible), LLM_BATCH_SIZE):
        chunk = eligible[i:i + LLM_BATCH_SIZE]
        batches.append([{
            "id":          r.rule_id,
            "category":    r.category,
            "description": r.description,
            "raw":         r.raw_expression,
            "entity":      r.entity,
            "context":     r.bounded_context,
            "confidence":  r.confidence,
        } for r in chunk])

    # Build lookup from rule_id → original rule
    rule_by_id = {r.rule_id: r for r in eligible}

    results: list[dict] = []
    for batch_num, batch in enumerate(batches, 1):
        print(f"  [stage46]   batch {batch_num}/{len(batches)} ({len(batch)} rules) → LLM …")
        llm_out = _llm_formalize_batch(batch, batch_num)

        # Build id → llm result map — guard against non-dict elements
        # (_extract_json_array now enforces this, but keep the check as safety net)
        llm_map = {
            item.get("id", ""): item
            for item in llm_out
            if isinstance(item, dict)
        }

        for item in batch:
            orig    = rule_by_id[item["id"]]
            llm_res = llm_map.get(item["id"], {})

            title = llm_res.get("title") or orig.description[:60]
            given, when, then = (
                llm_res.get("given", ""),
                llm_res.get("when", ""),
                llm_res.get("then", ""),
            )
            if not given or not when or not then:
                given, when, then = _static_gwt(orig.category, orig.description, orig.entity)

            conf = float(llm_res.get("confidence", orig.confidence))
            tags = list(llm_res.get("tags", []) or [])
            if orig.category.lower() not in tags:
                tags.insert(0, orig.category.lower())

            results.append({
                "category":        orig.category,
                "title":           title,
                "description":     orig.description,
                "given":           given,
                "when":            when,
                "then":            then,
                "entities":        [orig.entity] if orig.entity else [],
                "bounded_context": orig.bounded_context,
                "source_invariants": [orig.rule_id],
                "source_machines":   [],
                "source_flows":      [],
                "source_files":      orig.source_files,
                "confidence":        conf,
                "tags":              tags,
                "pass_origin":       "pass1_invariant",
                "_fp": _fingerprint(orig.category, orig.bounded_context, orig.description),
            })

    return results


# ── Pass 2 — Mine state machine rules (static) ───────────────────────────────

def _pass2_from_statemachines(ctx: PipelineContext) -> list[dict]:
    if not ctx.state_machines or not ctx.state_machines.machines:
        return []

    results: list[dict] = []
    for sm in ctx.state_machines.machines:
        entity_name = sm.entity
        # Transition rules
        for t in sm.transitions:
            desc  = (f"{entity_name} can transition from '{t.from_state}' "
                     f"to '{t.to_state}' via {t.trigger or 'the system'}")
            guard_clause = f" only when {t.guard}" if t.guard else ""
            results.append({
                "category":        "STATE",
                "title":           f"{entity_name} {t.from_state} → {t.to_state}",
                "description":     desc,
                "given":           f"Given a {entity_name} is in '{t.from_state}' state",
                "when":            (f"When '{t.trigger}' is executed{guard_clause}"
                                    if t.trigger else f"When the transition is triggered{guard_clause}"),
                "then":            f"Then the {entity_name} transitions to '{t.to_state}' state",
                "entities":        [entity_name],
                "bounded_context": sm.bounded_context,
                "source_invariants": [],
                "source_machines":   [sm.machine_id],
                "source_flows":      [],
                "source_files":      t.source_files,
                "confidence":        t.confidence,
                "tags":              ["state-machine", sm.table, sm.field],
                "pass_origin":       "pass2_state",
                "_fp": _fingerprint("STATE", sm.bounded_context,
                                    f"{t.from_state}->{t.to_state}"),
            })

        # Dead state warnings
        for ds in sm.dead_states:
            desc = f"{entity_name} state '{ds}' is defined but unreachable from any initial state"
            results.append({
                "category":        "STATE",
                "title":           f"{entity_name} dead state: {ds}",
                "description":     desc,
                "given":           f"Given the {entity_name} lifecycle is defined",
                "when":            "When all possible transitions are evaluated",
                "then":            f"Then state '{ds}' should be unreachable — verify if intentional or missing transition",
                "entities":        [entity_name],
                "bounded_context": sm.bounded_context,
                "source_invariants": [],
                "source_machines":   [sm.machine_id],
                "source_flows":      [],
                "source_files":      [],
                "confidence":        0.70,
                "tags":              ["state-machine", "dead-state", sm.table],
                "pass_origin":       "pass2_state",
                "_fp": _fingerprint("STATE", sm.bounded_context, f"dead:{ds}"),
            })

        # Terminal state rules
        for ts in sm.terminal_states:
            if ts in sm.dead_states:
                continue
            desc = f"{entity_name} in '{ts}' state cannot transition further"
            results.append({
                "category":        "STATE",
                "title":           f"{entity_name} terminal state: {ts}",
                "description":     desc,
                "given":           f"Given a {entity_name} is in '{ts}' state",
                "when":            "When any state transition is attempted",
                "then":            f"Then the system must not allow further transitions from '{ts}'",
                "entities":        [entity_name],
                "bounded_context": sm.bounded_context,
                "source_invariants": [],
                "source_machines":   [sm.machine_id],
                "source_flows":      [],
                "source_files":      [],
                "confidence":        0.75,
                "tags":              ["state-machine", "terminal-state", sm.table],
                "pass_origin":       "pass2_state",
                "_fp": _fingerprint("STATE", sm.bounded_context, f"terminal:{ts}"),
            })

    return results


# ── Pass 3 — Mine business flow rules (static) ───────────────────────────────

def _pass3_from_flows(ctx: PipelineContext) -> list[dict]:
    if not ctx.business_flows or not ctx.business_flows.flows:
        return []

    results: list[dict] = []
    for flow in ctx.business_flows.flows:
        context = flow.bounded_context
        actor   = flow.actor or "User"

        for step in flow.steps:
            # Authorization rule: steps marked auth_required
            if step.auth_required:
                desc = f"Only authenticated {actor} can perform: {step.action} on {step.page}"
                results.append({
                    "category":        "AUTHORIZATION",
                    "title":           f"Auth required: {step.action[:40]}",
                    "description":     desc,
                    "given":           f"Given a user attempts to perform '{step.action}' on '{step.page}'",
                    "when":            "When the system checks authentication",
                    "then":            f"Then only authenticated {actor} should be allowed to proceed",
                    "entities":        [],
                    "bounded_context": context,
                    "source_invariants": [],
                    "source_machines":   [],
                    "source_flows":      [flow.flow_id],
                    "source_files":      flow.evidence_files[:2],
                    "confidence":        0.80,
                    "tags":              ["authorization", "authentication", actor.lower()],
                    "pass_origin":       "pass3_flow",
                    "_fp": _fingerprint("AUTHORIZATION", context,
                                        f"{step.action}:{step.page}"),
                })

            # Workflow rules: branch steps with inputs → guards
            if step.is_branch and step.inputs:
                for inp in step.inputs[:3]:   # cap to avoid explosion
                    desc = (f"In flow '{flow.name}': step {step.step_num} "
                            f"requires input '{inp}' before proceeding")
                    results.append({
                        "category":        "WORKFLOW",
                        "title":           f"Precondition: {inp[:40]}",
                        "description":     desc,
                        "given":           f"Given the {actor} is performing '{flow.name}'",
                        "when":            f"When step {step.step_num} '{step.action}' is reached",
                        "then":            f"Then '{inp}' must be provided before the flow continues",
                        "entities":        [],
                        "bounded_context": context,
                        "source_invariants": [],
                        "source_machines":   [],
                        "source_flows":      [flow.flow_id],
                        "source_files":      flow.evidence_files[:2],
                        "confidence":        0.72,
                        "tags":              ["workflow", "precondition", inp.lower()],
                        "pass_origin":       "pass3_flow",
                        "_fp": _fingerprint("WORKFLOW", context,
                                            f"{flow.flow_id}:step{step.step_num}:{inp}"),
                    })

    return results


# ── Pass 4 — Mine referential integrity rules (static) ───────────────────────

def _pass4_from_relationships(ctx: PipelineContext) -> list[dict]:
    if not ctx.relationships or not ctx.relationships.relationships:
        return []

    # Build entity display names
    ent_names: dict[str, str] = {}
    if ctx.entities:
        for e in ctx.entities.entities:
            ent_names[e.table] = e.name

    results: list[dict] = []
    for rel in ctx.relationships.relationships:
        if rel.confidence < 0.70:
            continue
        parent = ent_names.get(rel.from_entity, _humanize_table(rel.from_entity))
        child  = ent_names.get(rel.to_entity,   _humanize_table(rel.to_entity))
        col    = rel.via_column or rel.via_table or "foreign key"

        if rel.cardinality == "N:M":
            desc  = (f"A {parent} can be associated with many {child}s "
                     f"and vice versa via {rel.via_table or col}")
            given = f"Given a {parent} and a {child} exist independently"
            when  = f"When they are linked via '{col}'"
            then  = (f"Then the association is recorded and both {parent} "
                     f"and {child} must exist before the link is created")
        elif rel.cardinality == "1:1":
            desc  = f"Each {parent} has exactly one {child} via {col}"
            given = f"Given a {child} record is created"
            when  = f"When '{col}' is set"
            then  = f"Then it must reference a valid {parent} and no other {child} may reference the same {parent}"
        else:  # 1:N
            desc  = f"A {parent} can have many {child}s; each {child} must reference a valid {parent}"
            given = f"Given a {child} record is created or updated"
            when  = f"When '{col}' is set"
            then  = f"Then it must reference a valid {parent} record that exists in the system"

        results.append({
            "category":        "REFERENTIAL",
            "title":           f"{parent} → {child} integrity ({rel.cardinality})",
            "description":     desc,
            "given":           given,
            "when":            when,
            "then":            then,
            "entities":        [rel.from_entity, rel.to_entity],
            "bounded_context": ent_names.get(rel.from_entity, rel.from_entity),
            "source_invariants": [],
            "source_machines":   [],
            "source_flows":      [],
            "source_files":      rel.source_files[:2],
            "confidence":        rel.confidence,
            "tags":              ["referential-integrity", rel.from_entity, rel.to_entity],
            "pass_origin":       "pass4_referential",
            "_fp": _fingerprint("REFERENTIAL", rel.from_entity,
                                f"{rel.from_entity}:{rel.to_entity}"),
        })

    return results


# ── Pass 5 — Mine auth signals (static) ──────────────────────────────────────

def _pass5_from_auth_signals(ctx: PipelineContext) -> list[dict]:
    """
    Mine AUTHORIZATION rules directly from code_map.auth_signals.

    auth_signals are extracted by stage15 from every PHP file that has
    session checks, redirect guards, or role gates — they are the raw
    code-level authentication patterns, more granular than what pass3
    derives from flow steps.

    Produces one AUTHORIZATION rule per unique (type, pattern) pair with
    the source file attached so traceability links back to exact PHP guards.
    """
    cm = getattr(ctx, "code_map", None)
    if not cm:
        return []
    auth_signals = getattr(cm, "auth_signals", None) or []
    if not auth_signals:
        return []

    results: list[dict] = []
    seen_fps: set[str] = set()

    for sig in auth_signals:
        sig_type    = sig.get("type", "")
        pattern     = sig.get("pattern", "")
        detail      = sig.get("detail", "")
        fpath       = sig.get("file", "")
        context_key = Path(fpath).stem.replace("_", " ").title() if fpath else "System"

        if not pattern:
            continue

        # Map signal type to human-readable guard description
        if sig_type == "session_check":
            desc  = f"Access to '{context_key}' requires an authenticated session (checks {pattern})"
            given = f"Given a user attempts to access '{context_key}'"
            when  = f"When the system checks session variable '{pattern}'"
            then  = "Then the user must be authenticated; unauthenticated users are redirected"
            conf  = 0.85
        elif sig_type == "role_check":
            desc  = f"Access to '{context_key}' is restricted to role: {detail or pattern}"
            given = f"Given a user attempts to access '{context_key}'"
            when  = f"When the system checks the user's role ({pattern})"
            then  = f"Then only users with role '{detail or pattern}' are permitted"
            conf  = 0.88
        elif sig_type == "redirect_guard":
            desc  = f"'{context_key}' redirects unauthorised users to: {detail or pattern}"
            given = f"Given an unauthenticated user requests '{context_key}'"
            when  = f"When the authentication guard evaluates '{pattern}'"
            then  = f"Then the user is redirected to '{detail or pattern}'"
            conf  = 0.80
        else:
            desc  = f"'{context_key}' enforces auth guard: {pattern}"
            given = f"Given a user accesses '{context_key}'"
            when  = f"When the system evaluates guard '{pattern}'"
            then  = "Then the system enforces the authentication requirement"
            conf  = 0.72

        fp = _fingerprint("AUTHORIZATION", context_key, desc[:60])
        if fp in seen_fps:
            continue
        seen_fps.add(fp)

        results.append({
            "category":        "AUTHORIZATION",
            "title":           f"Auth guard: {context_key[:45]}",
            "description":     desc,
            "given":           given,
            "when":            when,
            "then":            then,
            "entities":        [],
            "bounded_context": context_key,
            "source_invariants": [],
            "source_machines":   [],
            "source_flows":      [],
            "source_files":      [fpath] if fpath else [],
            "confidence":        conf,
            "tags":              ["authorization", sig_type, "auth-guard"],
            "pass_origin":       "pass5_auth_signal",
            "_fp": fp,
        })

    return results


# ── Pass 6 — Mine integration rules from service_deps + external systems ──────

def _pass6_from_integrations(ctx: PipelineContext) -> list[dict]:
    """
    Mine INTEGRATION rules from:
      - ctx.semantic_roles.external_systems  (detected external services)
      - code_map.service_deps                (class-level dependency graph)

    Produces one INTEGRATION rule per detected external system describing
    the integration contract — previously these were completely ignored by
    stage46, leaving the BRD/SRS without any integration-boundary rules.
    """
    results: list[dict] = []
    seen_fps: set[str] = set()

    # ── External systems from semantic_roles ──────────────────────────────────
    sr = getattr(ctx, "semantic_roles", None)
    ext_systems = (sr.external_systems if sr else []) or []

    _CATEGORY_DESC: dict[str, str] = {
        "PAYMENT":    "payment processing",
        "EMAIL":      "email delivery",
        "STORAGE":    "file/object storage",
        "SMS_PUSH":   "SMS / push notifications",
        "AUTH_OAUTH": "OAuth / SSO authentication",
        "MONITORING": "monitoring / observability",
        "CRM":        "CRM / customer data",
        "ERP":        "ERP / enterprise data",
        "INFRA":      "infrastructure (cache / queue / search)",
    }

    for ext in ext_systems:
        name     = ext.name or "External Service"
        cat      = ext.category or "INTEGRATION"
        cat_desc = _CATEGORY_DESC.get(cat, "external service integration")
        env_note = (f"; configured via {', '.join(ext.env_keys[:3])}"
                    if ext.env_keys else "")
        class_note = (f" (class: {', '.join(ext.class_hints[:2])})"
                      if ext.class_hints else "")

        desc  = (f"The system integrates with {name} for {cat_desc}"
                 f"{env_note}{class_note}")
        given = f"Given the system needs to perform a {cat_desc} operation"
        when  = f"When the integration with {name} is invoked"
        then  = (f"Then the system must have valid {name} credentials configured"
                 f"{env_note} and handle integration failures gracefully")

        fp = _fingerprint("INTEGRATION", name, desc[:60])
        if fp in seen_fps:
            continue
        seen_fps.add(fp)

        results.append({
            "category":        "INTEGRATION",
            "title":           f"{name} integration ({cat})",
            "description":     desc,
            "given":           given,
            "when":            when,
            "then":            then,
            "entities":        [],
            "bounded_context": name,
            "source_invariants": [],
            "source_machines":   [],
            "source_flows":      [],
            "source_files":      [],
            "confidence":        0.82,
            "tags":              ["integration", cat.lower(), name.lower()],
            "pass_origin":       "pass6_integration",
            "_fp": fp,
        })

    return results


# ── Behavior-graph confidence boost ──────────────────────────────────────────

def _boost_by_behavior_graph(
    rules_raw: list[dict],
    behavior_graph: dict | None,
) -> list[dict]:
    """
    After dedup, boost rule confidence for rules whose source_files appear
    in high-confidence behavior_graph paths.

    A rule backed by high-confidence route→controller→SQL paths has stronger
    code evidence than one derived from a single low-confidence path.  Boost
    capped at +0.08 so no rule exceeds 0.97.
    """
    if not behavior_graph:
        return rules_raw

    # Build file-name → max path confidence index
    _file_conf: dict[str, float] = {}
    for bp in (behavior_graph.get("paths") or []):
        _conf = bp.get("confidence", 0.0)
        for _rf in (bp.get("reachable_files") or []):
            _fname = Path(_rf).name.lower()
            if _conf > _file_conf.get(_fname, 0.0):
                _file_conf[_fname] = _conf

    if not _file_conf:
        return rules_raw

    for rule in rules_raw:
        src_fnames = {Path(f).name.lower() for f in (rule.get("source_files") or [])}
        if not src_fnames:
            continue
        max_bg_conf = max((_file_conf.get(fn, 0.0) for fn in src_fnames), default=0.0)
        if max_bg_conf >= 0.75:
            rule["confidence"] = min(rule["confidence"] + 0.08, 0.97)
        elif max_bg_conf >= 0.50:
            rule["confidence"] = min(rule["confidence"] + 0.04, 0.97)

    return rules_raw


# ── Deduplication & collection building ──────────────────────────────────────

def _dedup(all_raw: list[dict]) -> list[dict]:
    """
    Group candidates by _fp (fingerprint).  Keep highest confidence per group;
    merge all source lists.
    """
    groups: dict[str, dict] = {}
    for r in all_raw:
        fp = r.get("_fp", _fingerprint(r["category"], r.get("bounded_context", ""), r["description"]))
        if fp not in groups:
            groups[fp] = dict(r)
            for key in ("source_invariants", "source_machines", "source_flows",
                        "source_files", "tags", "entities"):
                groups[fp][key] = list(r.get(key, []))
        else:
            existing = groups[fp]
            if r["confidence"] > existing["confidence"]:
                existing["confidence"] = r["confidence"]
                for field in ("title", "given", "when", "then"):
                    if r.get(field):
                        existing[field] = r[field]
            for key in ("source_invariants", "source_machines", "source_flows",
                        "source_files", "tags", "entities"):
                for v in r.get(key, []):
                    if v and v not in existing[key]:
                        existing[key].append(v)

    return list(groups.values())


def _build_collection(rules_raw: list[dict]) -> SpecRuleCollection:
    """Convert raw dicts into SpecRule objects and build index dictionaries."""
    # Filter below threshold and sort by confidence desc
    rules_raw = [r for r in rules_raw if r["confidence"] >= MERGE_CONF_THRESHOLD]
    rules_raw.sort(key=lambda r: -r["confidence"])

    rules: list[SpecRule] = []
    by_category: dict[str, list[str]] = defaultdict(list)
    by_entity:   dict[str, list[str]] = defaultdict(list)
    by_flow:     dict[str, list[str]] = defaultdict(list)
    by_context:  dict[str, list[str]] = defaultdict(list)

    for seq, r in enumerate(rules_raw, 1):
        rule_id = f"BR-{seq:03d}"
        sr = SpecRule(
            rule_id           = rule_id,
            category          = r["category"],
            title             = r.get("title", "")[:100],
            description       = r.get("description", ""),
            given             = r.get("given", ""),
            when              = r.get("when", ""),
            then              = r.get("then", ""),
            entities          = r.get("entities", []),
            bounded_context   = r.get("bounded_context", ""),
            source_invariants = r.get("source_invariants", []),
            source_machines   = r.get("source_machines", []),
            source_flows      = r.get("source_flows", []),
            source_files      = r.get("source_files", []),
            confidence        = r["confidence"],
            tags              = r.get("tags", []),
            pass_origin       = r.get("pass_origin", ""),
        )
        rules.append(sr)
        by_category[sr.category].append(rule_id)
        for ent in sr.entities:
            by_entity[ent].append(rule_id)
        for fid in sr.source_flows:
            by_flow[fid].append(rule_id)
        if sr.bounded_context:
            by_context[sr.bounded_context].append(rule_id)

    return SpecRuleCollection(
        rules       = rules,
        total       = len(rules),
        by_category = dict(by_category),
        by_entity   = dict(by_entity),
        by_flow     = dict(by_flow),
        by_context  = dict(by_context),
    )


# ── Markdown summary ─────────────────────────────────────────────────────────

def _write_markdown(collection: SpecRuleCollection, path: str) -> None:
    lines = [
        "# Specification Mining — Business Rules",
        "",
        f"**Generated:** {collection.generated_at}  ",
        f"**Total rules:** {collection.total}",
        "",
    ]

    # Summary by category
    lines += ["## Summary by Category", ""]
    for cat, ids in sorted(collection.by_category.items()):
        lines.append(f"| {cat} | {len(ids)} rules |")
    lines.append("")

    # Rules grouped by category
    for cat in sorted(collection.by_category):
        lines += [f"## {cat}", ""]
        cat_rules = [r for r in collection.rules if r.category == cat]
        for rule in cat_rules:
            lines += [
                f"### {rule.rule_id} — {rule.title}",
                "",
                f"**Context:** {rule.bounded_context}  ",
                f"**Confidence:** {rule.confidence:.2f}  ",
                f"**Pass:** {rule.pass_origin}  ",
                "",
                f"- **Given:** {rule.given}",
                f"- **When:** {rule.when}",
                f"- **Then:** {rule.then}",
                "",
            ]
            if rule.entities:
                lines.append(f"**Entities:** {', '.join(rule.entities)}  ")
            if rule.tags:
                lines.append(f"**Tags:** {', '.join(rule.tags)}  ")
            sources = []
            if rule.source_invariants:
                sources.append(f"invariants: {', '.join(rule.source_invariants)}")
            if rule.source_machines:
                sources.append(f"state machines: {', '.join(rule.source_machines)}")
            if rule.source_flows:
                sources.append(f"flows: {', '.join(rule.source_flows)}")
            if sources:
                lines.append(f"**Sources:** {'; '.join(sources)}  ")
            lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ── Main entry point ──────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    print("  [stage46] Specification Mining — synthesising business rules …")

    # ── Pass 1: Invariants → LLM formalization ────────────────────────────
    raw_p1 = _pass1_from_invariants(ctx)
    print(f"  [stage46] Pass 1 complete — {len(raw_p1)} rule(s) from invariants.")

    # ── Pass 2: State machines (static) ───────────────────────────────────
    raw_p2 = _pass2_from_statemachines(ctx)
    print(f"  [stage46] Pass 2 complete — {len(raw_p2)} rule(s) from state machines.")

    # ── Pass 3: Business flows (static) ───────────────────────────────────
    raw_p3 = _pass3_from_flows(ctx)
    print(f"  [stage46] Pass 3 complete — {len(raw_p3)} rule(s) from flows.")

    # ── Pass 4: Relationships (static) ────────────────────────────────────
    raw_p4 = _pass4_from_relationships(ctx)
    print(f"  [stage46] Pass 4 complete — {len(raw_p4)} rule(s) from relationships.")

    # ── Pass 5: Auth signals (static) ─────────────────────────────────────
    # Mine AUTHORIZATION rules from raw code_map.auth_signals — the actual
    # session checks, redirect guards, and role gates extracted by stage15.
    # Previously ignored; surfaces grounded auth rules that pass3 may miss.
    raw_p5 = _pass5_from_auth_signals(ctx)
    print(f"  [stage46] Pass 5 complete — {len(raw_p5)} rule(s) from auth signals.")

    # ── Pass 6: External integrations (static) ────────────────────────────
    # Mine INTEGRATION rules from external systems detected by stage27 and
    # class-level service dependencies from stage1.
    raw_p6 = _pass6_from_integrations(ctx)
    print(f"  [stage46] Pass 6 complete — {len(raw_p6)} rule(s) from integrations.")

    # ── Deduplication ──────────────────────────────────────────────────────
    all_raw = _dedup(raw_p1 + raw_p2 + raw_p3 + raw_p4 + raw_p5 + raw_p6)
    print(f"  [stage46] After deduplication — {len(all_raw)} unique rule(s).")

    # ── Behavior-graph confidence boost ───────────────────────────────────
    # Rules whose source_files appear in high-confidence behavior_graph paths
    # get a small confidence boost (≤ +0.08) — stronger code backing = higher
    # rule trust.  ctx.behavior_graph was previously ignored entirely here.
    all_raw = _boost_by_behavior_graph(all_raw, getattr(ctx, "behavior_graph", None))
    print(f"  [stage46] Behavior-graph confidence boost applied.")

    # ── Build collection ───────────────────────────────────────────────────
    collection = _build_collection(all_raw)
    ctx.spec_rules = collection

    # ── Print summary ──────────────────────────────────────────────────────
    total = collection.total
    print(f"\n  [stage46] Specification Mining complete — {total} business rules:")
    for cat, ids in sorted(collection.by_category.items()):
        print(f"    {cat:<20} {len(ids):>3} rules")
    print()

    # Sample: show first 5 rules across all categories
    for rule in collection.rules[:5]:
        print(f"    [{rule.rule_id}] [{rule.category}] {rule.title}")
        print(f"           conf={rule.confidence:.2f}  {rule.pass_origin}")
    if total > 5:
        print(f"    … and {total - 5} more rules")
    print()

    # ── Persist JSON ──────────────────────────────────────────────────────
    try:
        import dataclasses
        out_path = ctx.output_path("spec_rules.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(collection), fh, indent=2, ensure_ascii=False)
        print(f"  [stage46] Saved → {out_path}")
    except Exception as exc:
        print(f"  [stage46] ⚠️  Could not save spec_rules.json: {exc}")

    # ── Persist Markdown summary ───────────────────────────────────────────
    try:
        md_path = str(Path(ctx.output_path("spec_rules.json")).parent / "spec_rules_summary.md")
        _write_markdown(collection, md_path)
        print(f"  [stage46] Markdown → {md_path}")
    except Exception as exc:
        print(f"  [stage46] ⚠️  Could not write markdown summary: {exc}")
