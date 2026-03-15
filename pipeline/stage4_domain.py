"""
pipeline/stage4_domain.py — Domain Analyst Agent

Uses the Claude API (claude-sonnet-4-20250514) to analyse the embedded
codebase and produce a structured DomainModel that all Stage 5 agents
(BRD, SRS, AC, UserStories) will use as their shared foundation.

What it does
------------
1. Retrieves semantically relevant chunks from ChromaDB across 6 query angles
2. Assembles a focused context window (≤ 6000 tokens of codebase signal)
3. Calls Claude with a structured system prompt to extract:
      - domain_name       e.g. "Car Rental Management System"
      - description       2-3 sentence system overview
      - user_roles        e.g. [{"role": "Customer", "description": "..."}]
      - features          e.g. [{"name": "User Registration", "description": "...",
                                  "pages": [...], "tables": [...]}]
      - workflows         e.g. [{"name": "Rental Booking Flow", "steps": [...]}]
      - bounded_contexts  e.g. ["Authentication", "Booking", "Reporting"]
      - key_entities      e.g. ["User", "Car", "Booking", "Payment"]
4. Parses the JSON response and hydrates ctx.domain_model
5. Saves domain_model.json to the output directory

Retrieval strategy
------------------
Fires 6 targeted queries at ChromaDB to cover all angles:
    Q1  "user roles and authentication login registration"
    Q2  "database tables entities data model"
    Q3  "business features workflows user actions"
    Q4  "form inputs user interactions pages"
    Q5  "navigation flow page redirects session"
    Q6  "core functions business logic processing"

Each query returns top-5 chunks. After deduplication, the best
≤ 25 chunks are assembled into the prompt context.

Quality calibration
-------------------
Uses ctx.preflight.quality_score to adjust Claude's instruction tone:
    ≥ 70  Normal — "extract the domain model precisely from the evidence"
    < 70  Conservative — "note where evidence is thin; avoid speculation"

Resume behaviour
----------------
If stage4_domain is COMPLETED and domain_model.json exists, the stage
is skipped and ctx.domain_model is restored from the saved file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from context import DomainModel, PipelineContext

# ── Configuration ──────────────────────────────────────────────────────────────
# Three focused calls instead of one giant call — each produces a small, bounded
# JSON so local models (Qwen, Mistral, etc.) never hit output-token limits.
MAX_TOKENS_META     = 4_000   # Call A: domain_name, description, key_entities, bounded_contexts
MAX_TOKENS_FEATURES = 8_000   # Call B: features list (largest field)
MAX_TOKENS_ROLES_WF = 8_000   # Call C: user_roles + workflows
DOMAIN_FILE      = "domain_model.json"

# Retrieval: queries × top-k chunks each
RETRIEVAL_QUERIES = [
    ("user roles authentication login registration logout session",   "auth"),
    ("database tables entities data model SQL queries",               "data"),
    ("business features workflows user actions operations",           "features"),
    ("form inputs POST fields user interactions pages submissions",   "forms"),
    ("navigation flow redirect pages include session check",          "nav"),
    ("core functions business logic processing validation",           "logic"),
]
TOP_K_PER_QUERY  = 5
MAX_TOTAL_CHUNKS = 15     # hard cap on chunks sent to Claude
MAX_CONTEXT_CHARS = 5000  # soft cap on total context character length


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 4 entry point. Retrieves codebase context, calls Claude, and
    populates ctx.domain_model with a structured DomainModel.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If prerequisites are missing or Claude API fails.
    """
    output_path = ctx.output_path(DOMAIN_FILE)

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage4_domain") and Path(output_path).exists():
        ctx.domain_model = _load_domain_model(output_path)
        print(f"  [stage4] Already completed — "
              f"domain='{ctx.domain_model.domain_name}', "
              f"{len(ctx.domain_model.features)} feature(s).")
        return

    # ── Pre-flight ────────────────────────────────────────────────────────────
    _assert_prerequisites(ctx)

    print(f"  [stage4] Retrieving codebase context from ChromaDB ...")

    # ── Retrieve chunks ───────────────────────────────────────────────────────
    chunks = _retrieve_context(ctx)
    print(f"  [stage4] Retrieved {len(chunks)} unique chunks across "
          f"{len(RETRIEVAL_QUERIES)} query angles")

    # ── Build shared user prompt (same evidence for all 3 calls) ─────────────
    quality_score = _get_quality_score(ctx)
    user_prompt   = _build_user_prompt(ctx, chunks)

    from pipeline.llm_client import get_provider, get_model
    print(f"  [stage4] LLM: {get_provider()} / {get_model()} | "
          f"context={len(user_prompt)} chars | quality={quality_score}")

    debug_dir = str(Path(output_path).parent)

    # ── Call A: meta + entities + contexts (fast, ~4K tokens) ────────────────
    print(f"  [stage4] Call 1/3 — domain name, entities, contexts ...")
    raw_a = _call_part(_system_meta(quality_score), user_prompt,
                       MAX_TOKENS_META, "stage4-A")
    data_a = _parse_partial(raw_a, "A", debug_dir)

    # ── Call B: features (largest field, ~8K tokens) ──────────────────────────
    print(f"  [stage4] Call 2/3 — features ...")
    raw_b = _call_part(_system_features(quality_score), user_prompt,
                       MAX_TOKENS_FEATURES, "stage4-B")
    data_b = _parse_partial(raw_b, "B", debug_dir)

    # ── Call C: user roles + workflows (~8K tokens) ───────────────────────────
    print(f"  [stage4] Call 3/3 — user roles, workflows ...")
    raw_c = _call_part(_system_roles_workflows(quality_score), user_prompt,
                       MAX_TOKENS_ROLES_WF, "stage4-C")
    data_c = _parse_partial(raw_c, "C", debug_dir)

    # ── Merge all three dicts (A wins on overlaps, B/C fill in their fields) ──
    merged = {**data_a, **data_b, **data_c}

    # ── Hydrate DomainModel ───────────────────────────────────────────────────
    domain_model = _hydrate_domain_model(merged)

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_domain_model(domain_model, output_path)
    ctx.domain_model = domain_model
    ctx.stage("stage4_domain").mark_completed(output_path)
    ctx.save()

    print(f"  [stage4] Done — domain='{domain_model.domain_name}'")
    print(f"  [stage4]   Roles    : {[r['role'] for r in domain_model.user_roles]}")
    print(f"  [stage4]   Features : {[f['name'] for f in domain_model.features]}")
    print(f"  [stage4]   Entities : {domain_model.key_entities}")
    print(f"  [stage4]   Contexts : {domain_model.bounded_contexts}")
    print(f"  [stage4] Saved → {output_path}")


# ─── Retrieval ─────────────────────────────────────────────────────────────────

def _retrieve_context(ctx: PipelineContext) -> list[dict[str, Any]]:
    """
    Fire all retrieval queries against ChromaDB and return a deduplicated,
    ranked list of the most relevant chunks.
    """
    from pipeline.stage3_embed import query_collection

    seen_ids: set[str]      = set()
    scored:   list[tuple]   = []   # (score, chunk_dict)

    for query_text, angle in RETRIEVAL_QUERIES:
        try:
            results = query_collection(ctx, query_text, n_results=TOP_K_PER_QUERY)
        except Exception as e:
            print(f"  [stage4] Warning: query '{angle}' failed: {e}")
            continue

        for rank, result in enumerate(results):
            chunk_id = result["id"]
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            # Score: lower distance = more relevant; earlier rank = better
            # Combined score: 1 - distance (0→bad, 1→perfect) + rank bonus
            relevance = (1.0 - result["distance"]) + (TOP_K_PER_QUERY - rank) * 0.02
            scored.append((relevance, result))

    # Sort by relevance descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:MAX_TOTAL_CHUNKS]]

    # Trim to MAX_CONTEXT_CHARS total
    total_chars = 0
    trimmed = []
    for chunk in top_chunks:
        chunk_len = len(chunk["text"])
        if total_chars + chunk_len > MAX_CONTEXT_CHARS:
            break
        trimmed.append(chunk)
        total_chars += chunk_len

    return trimmed



def _modules_from_graph(ctx: "PipelineContext") -> dict[str, dict]:
    """
    Load the graph pickle and extract G.graph["modules"] from Stage 2 Pass 15.

    Returns a dict of {module_id: module_info} or {} if graph is unavailable
    or was built before Pass 15 was added.

    Used to pre-seed bounded_contexts so the LLM refines detected modules
    rather than guessing from scratch.
    """
    if ctx.graph_meta is None or not ctx.graph_meta.graph_path:
        return {}
    try:
        import pickle
        with open(ctx.graph_meta.graph_path, "rb") as fh:
            G = pickle.load(fh)
        return G.graph.get("modules", {})
    except Exception:
        return {}

# ─── Prompt Building ───────────────────────────────────────────────────────────

def _evidence_instruction(quality_score: int) -> str:
    """Return the evidence-strictness instruction based on quality score."""
    if quality_score >= 70:
        return (
            "Extract the domain model precisely and completely from the evidence provided. "
            "Be specific — use actual table names, field names, page names, and function names "
            "from the codebase evidence."
        )
    return (
        "Extract what you can from the evidence, but note where evidence is thin. "
        "Avoid speculation — if a feature or entity is not evidenced in the codebase "
        "context, do not invent it."
    )


_SYSTEM_PREAMBLE = """\
You are a senior Business Analyst and software architect specialising in
reverse-engineering legacy PHP applications into structured business domain models.

Your task is to analyse the provided PHP codebase evidence and extract specific
parts of the domain model in JSON format.

CRITICAL OUTPUT RULES — you MUST follow these exactly:
- Output ONLY the raw JSON object — no markdown fences (no ```json), no headings, no explanation
- Start your response with {{ and end with }} — nothing before or after
- Use EXACTLY the field names shown — do NOT rename, restructure, or add wrapper keys
- Use actual names from the codebase (table names, page names, field names)
"""


def _system_meta(quality_score: int) -> str:
    """Call A — domain name, description, key entities, bounded contexts."""
    return f"""{_SYSTEM_PREAMBLE}
{_evidence_instruction(quality_score)}

Output ONLY this JSON (no other fields):
{{
  "domain_name": "Short descriptive system name (e.g. 'Car Rental Management System')",
  "description": "2-3 sentence plain-English overview of what the system does and who uses it",
  "key_entities": ["Entity1", "Entity2"],
  "bounded_contexts": ["Context1", "Context2"]
}}
For bounded_contexts: use the STRUCTURALLY DETECTED MODULES as the primary source.
Rename for clarity, merge near-empty ones, but anchor to detected module names.

Now produce the JSON for domain_name, description, key_entities, and bounded_contexts:"""


def _system_features(quality_score: int) -> str:
    """Call B — features list only."""
    return f"""{_SYSTEM_PREAMBLE}
{_evidence_instruction(quality_score)}

Output ONLY this JSON (no other fields):
{{
  "features": [
    {{
      "name": "Feature name (e.g. User Registration)",
      "description": "What this feature does from a business perspective",
      "pages": ["registration.php"],
      "tables": ["users"],
      "inputs": ["field1", "field2"],
      "outputs": "What the user gets after completing this feature",
      "business_rules": ["Rule 1", "Rule 2"]
    }}
  ]
}}
Every feature must reference the pages and tables that implement it.
Be concrete and specific — use actual filenames and table names from the evidence.

Now produce the JSON for features only:"""


def _system_roles_workflows(quality_score: int) -> str:
    """Call C — user roles and workflows."""
    return f"""{_SYSTEM_PREAMBLE}
{_evidence_instruction(quality_score)}

Output ONLY this JSON (no other fields):
{{
  "user_roles": [
    {{
      "role": "Role name (e.g. Customer, Admin, Staff)",
      "description": "What this role can do in the system",
      "entry_points": ["login.php", "registration.php"]
    }}
  ],
  "workflows": [
    {{
      "name": "Workflow name (e.g. Rental Booking Flow)",
      "actor": "Who performs this workflow",
      "steps": [
        {{"step": 1, "action": "User navigates to login.php", "page": "login.php"}},
        {{"step": 2, "action": "User submits credentials", "page": "login.php"}}
      ],
      "preconditions": ["User must have an account"],
      "postconditions": ["Booking is saved in request table"]
    }}
  ]
}}

Now produce the JSON for user_roles and workflows only:"""


def _build_user_prompt(ctx: PipelineContext, chunks: list[dict]) -> str:
    """
    Assemble the user prompt from retrieved chunks, CodeMap metadata,
    and graph summary.
    """
    parts: list[str] = []

    # ── System metadata header ────────────────────────────────────────────────
    cm = ctx.code_map
    parts.append("=== CODEBASE METADATA ===")
    parts.append(f"Framework  : {cm.framework.value}")
    parts.append(f"PHP version: {cm.php_version}")
    parts.append(f"Files      : {cm.total_files}")
    parts.append(f"Lines      : {cm.total_lines}")

    if ctx.graph_meta:
        parts.append(f"Graph      : {ctx.graph_meta.node_count} nodes, "
                     f"{ctx.graph_meta.edge_count} edges")
        parts.append(f"Node types : {', '.join(ctx.graph_meta.node_types)}")
        parts.append(f"Edge types : {', '.join(ctx.graph_meta.edge_types)}")

    # ── DB tables summary ─────────────────────────────────────────────────────
    unique_tables = sorted({
        q.get("table", "") for q in cm.sql_queries
        if q.get("table") and q["table"] not in ("UNKNOWN", "")
    })
    if unique_tables:
        parts.append(f"\n=== DATABASE TABLES ===")
        parts.append(", ".join(unique_tables))

        # Per-table: which files read/write it
        for table in unique_tables:
            readers = sorted({q["file"] for q in cm.sql_queries
                              if q.get("table") == table and q.get("operation") == "SELECT"})
            writers = sorted({q["file"] for q in cm.sql_queries
                              if q.get("table") == table
                              and q.get("operation") in ("INSERT","UPDATE","DELETE","REPLACE")})
            ddl = sorted({q["file"] for q in cm.sql_queries
                          if q.get("table") == table
                          and q.get("operation") in ("CREATE","ALTER","DROP")})
            lines = [f"Table '{table}':"]
            if ddl:     lines.append(f"  Schema defined in: {', '.join(ddl)}")
            if writers: lines.append(f"  Written by: {', '.join(writers)}")
            if readers: lines.append(f"  Read by: {', '.join(readers)}")
            parts.append("\n".join(lines))

    # ── Form inputs summary ───────────────────────────────────────────────────
    if cm.superglobals:
        parts.append(f"\n=== FORM INPUTS (by page) ===")
        from collections import defaultdict
        sg_by_file: dict[str, list] = defaultdict(list)
        for sg in cm.superglobals:
            sg_by_file[sg["file"]].append(sg)
        for filepath, sgs in sorted(sg_by_file.items()):
            filename  = Path(filepath).name
            post_keys = sorted({s["key"] for s in sgs
                                if s["var"] == "$_POST" and s.get("key")})
            sess_keys = sorted({s["key"] for s in sgs
                                if s["var"] == "$_SESSION" and s.get("key")})
            if post_keys:
                parts.append(f"{filename} POST fields: {', '.join(post_keys)}")
            if sess_keys:
                parts.append(f"{filename} SESSION keys: {', '.join(sess_keys)}")

    # ── Redirect / navigation summary ────────────────────────────────────────
    if cm.redirects:
        parts.append(f"\n=== NAVIGATION / REDIRECTS ===")
        for r in cm.redirects:
            filename = Path(r["file"]).name
            parts.append(f"{filename} → {r['target']}")

    # ── HTML pages (entry points) ─────────────────────────────────────────────
    if cm.html_pages:
        parts.append(f"\n=== PAGE ENTRY POINTS ===")
        parts.append(", ".join(Path(p).name for p in sorted(cm.html_pages)))

    # ── Functions ─────────────────────────────────────────────────────────────
    if cm.functions:
        parts.append(f"\n=== USER-DEFINED FUNCTIONS ===")
        for fn in cm.functions:
            params = [p["name"] for p in fn.get("params", [])]
            doc    = fn.get("docblock") or ""
            line   = f"  {fn['name']}({', '.join(params)})"
            if doc:
                line += f" — {doc}"
            parts.append(line)

    # ── Call graph ────────────────────────────────────────────────────────────
    call_graph = getattr(cm, "call_graph", None) or []
    if call_graph:
        parts.append(f"\n=== FUNCTION CALL GRAPH ({len(call_graph)} edges) ===")
        # Group by file for readability
        from collections import defaultdict
        by_file: dict = defaultdict(list)
        for edge in call_graph:
            by_file[edge.get("file","?")].append(
                f"{edge.get('caller','?')} → {edge.get('callee','?')}"
            )
        for fpath, edges in sorted(by_file.items()):
            parts.append(f"{Path(fpath).name}: {', '.join(edges[:8])}"
                         + (" ..." if len(edges) > 8 else ""))

    # ── Form fields ───────────────────────────────────────────────────────────
    form_fields = getattr(cm, "form_fields", None) or []
    if form_fields:
        parts.append(f"\n=== HTML FORM FIELDS ===")
        for form in form_fields:
            fname  = Path(form.get("file","?")).name
            action = form.get("action","") or "(no action)"
            method = form.get("method","POST")
            fields = [f.get("name","?") for f in form.get("fields", []) if f.get("name")]
            parts.append(f"{fname} [{method} → {action}]: {', '.join(fields)}")

    # ── Service dependencies ──────────────────────────────────────────────────
    service_deps = getattr(cm, "service_deps", None) or []
    if service_deps:
        parts.append(f"\n=== SERVICE DEPENDENCIES (DI / Constructor Injection) ===")
        from collections import defaultdict
        deps_by_class: dict = defaultdict(list)
        for dep in service_deps:
            deps_by_class[dep.get("class","?")].append(dep.get("dep_class","?"))
        for cls, deps in sorted(deps_by_class.items()):
            parts.append(f"  {cls} depends on: {', '.join(deps)}")

    # ── Environment variables ─────────────────────────────────────────────────
    env_vars = getattr(cm, "env_vars", None) or []
    if env_vars:
        parts.append(f"\n=== ENVIRONMENT VARIABLES ===")
        # Group by category prefix (DB_, MAIL_, APP_, etc.)
        from collections import defaultdict
        by_prefix: dict = defaultdict(list)
        for ev in env_vars:
            key = ev.get("key","?")
            prefix = key.split("_")[0] if "_" in key else "OTHER"
            default = ev.get("default")
            entry = key if default is None else f"{key}={default!r}"
            by_prefix[prefix].append(entry)
        for prefix in sorted(by_prefix):
            parts.append(f"  {prefix}_*: {', '.join(sorted(set(by_prefix[prefix])))}")

    # ── Auth signals ──────────────────────────────────────────────────────────
    auth_signals = getattr(cm, "auth_signals", None) or []
    if auth_signals:
        from collections import Counter as _Counter
        parts.append(f"\n=== AUTHENTICATION & AUTHORISATION SIGNALS ===")
        by_type: _Counter = _Counter(s["type"] for s in auth_signals)
        parts.append("  Signal counts: " + ", ".join(
            f"{t}={n}" for t, n in sorted(by_type.items())))
        seen_pats: set = set()
        for sig in auth_signals:
            pat = sig.get("pattern", "?")
            if pat in seen_pats:
                continue
            seen_pats.add(pat)
            detail = f" [{sig['detail']}]" if sig.get("detail") else ""
            parts.append(f"  {sig['type']}: {pat}{detail}"
                         f" ({Path(sig.get('file','?')).name}:{sig.get('line','?')})")

    # ── HTTP entry points ─────────────────────────────────────────────────────
    http_endpoints = getattr(cm, "http_endpoints", None) or []
    if http_endpoints:
        parts.append(f"\n=== HTTP ENTRY POINTS ({len(http_endpoints)}) ===")
        for ep in http_endpoints:
            handler  = ep.get("handler") or Path(ep.get("file", "?")).name
            accepts  = "/".join(ep.get("accepts", []))
            produces = ep.get("produces", "?")
            ep_type  = ep.get("type", "page")
            parts.append(f"  [{ep_type}] {handler} — accepts {accepts}, produces {produces}")

    # ── Table/column definitions ──────────────────────────────────────────────
    table_columns = getattr(cm, "table_columns", None) or []
    if table_columns:
        parts.append(f"\n=== DATABASE TABLES & COLUMNS ({len(table_columns)} tables) ===")
        for tbl in table_columns:
            tname   = tbl.get("table", "?")
            source  = tbl.get("source", "?")
            cols    = tbl.get("columns", [])
            col_str = ", ".join(c.get("name") or "?" for c in cols[:12] if isinstance(c, dict))
            if len(cols) > 12:
                col_str += f" ... +{len(cols) - 12} more"
            parts.append(f"  {tname} ({source}): {col_str}")

    # ── Execution paths (stage15) ────────────────────────────────────────────
    exec_paths = getattr(cm, "execution_paths", None) or []
    if exec_paths:
        parts.append(f"\n=== EXECUTION PATHS & BRANCH ANALYSIS ===")
        parts.append(
            "Each entry below is derived from static analysis of a PHP file. "
            "Use this to identify workflows, auth requirements, and business rules."
        )
        for ep in exec_paths:
            fname = ep.get("file", "?")
            parts.append(f"\nFile: {fname}")

            # Auth guard
            ag = ep.get("auth_guard")
            if ag:
                parts.append(
                    f"  Auth guard: session['{ag['key']}'] required "
                    f"(else redirect → {ag.get('redirect','?')})"
                )

            # Entry conditions
            for ec in ep.get("entry_conditions", []):
                if ec.get("type") == "method_check":
                    parts.append(f"  Accepts: HTTP {ec.get('method','?')}")

            # Happy path (primary success scenario)
            hp = ep.get("happy_path", [])
            if hp:
                parts.append("  Happy path:")
                for step in hp:
                    parts.append(f"    → {step}")

            # Data flows
            for flow in ep.get("data_flows", []):
                fields = ", ".join(flow.get("field_mapping", {}).keys())
                table  = flow.get("table", "?")
                op     = flow.get("sink", "sql_query")
                if fields:
                    parts.append(
                        f"  Data flow: POST fields [{fields}] → {op} on `{table}`"
                    )

            # Branches summary (condition + outcome)
            branches = ep.get("branches", [])
            if branches:
                parts.append(f"  Branches ({len(branches)}):")
                for b in branches[:3]:          # cap at 3 per file
                    cond = b.get("condition","?")[:80]
                    then = [a.get("action","?") for a in b.get("then",[])]
                    els  = [a.get("action","?") for a in b.get("else",[])]
                    parts.append(f"    if ({cond})")
                    parts.append(f"      then: {', '.join(then) or 'none'}")
                    if els:
                        parts.append(f"      else: {', '.join(els)}")

    # ── Detected modules from Stage 2 Pass 15 ───────────────────────────────
    # Pre-seed the LLM with structurally-detected bounded contexts so it
    # refines evidence-based modules rather than inventing them from scratch.
    detected_modules = _modules_from_graph(ctx)
    if detected_modules:
        parts.append(f"\n=== STRUCTURALLY DETECTED MODULES ({len(detected_modules)}) ===")
        parts.append(
            "These modules were detected automatically from directory structure, "
            "namespaces, and call-graph community analysis. Use them as the basis "
            "for bounded_contexts — rename or merge if the code evidence warrants it, "
            "but do not drop modules that have significant node counts."
        )
        for mod_id, info in sorted(detected_modules.items()):
            method  = info.get("method", "unknown")
            q_score = info.get("q_score")
            count   = info.get("node_count", 0)
            ntypes  = info.get("node_types", {})
            type_summary = ", ".join(f"{t}:{n}" for t, n in sorted(ntypes.items()) if n > 0)
            q_str = f"  modularity_Q={q_score:.3f}" if q_score is not None else ""
            parts.append(
                f"  [{info['name']}]  id={mod_id}  nodes={count}"
                f"  detection={method}{q_str}"
                f"  ({type_summary})"
            )

    # ── Retrieved semantic chunks ─────────────────────────────────────────────
    parts.append(f"\n=== SEMANTIC CONTEXT ({len(chunks)} chunks) ===")
    for i, chunk in enumerate(chunks, 1):
        ctype = chunk["metadata"].get("chunk_type", "?")
        src   = chunk["metadata"].get("source_file", "?")
        parts.append(f"\n[{i}] {ctype} | {Path(src).name}")
        parts.append(chunk["text"])

    parts.append("\n=== END OF CODEBASE EVIDENCE ===")

    return "\n".join(parts)


# ─── LLM Call Helper ───────────────────────────────────────────────────────────

def _call_part(system_prompt: str, user_prompt: str,
               max_tokens: int, label: str) -> str:
    """Call the configured LLM and return the raw response string."""
    from pipeline.llm_client import call_llm
    return call_llm(
        system_prompt = system_prompt,
        user_prompt   = user_prompt,
        max_tokens    = max_tokens,
        label         = label,
    )


# ─── Response Parsing ──────────────────────────────────────────────────────────


def _attempt_json_recovery(text: str) -> dict | None:
    """
    Try to recover a partial JSON response that was truncated mid-stream.
    Attempts to close all open brackets/braces to make it valid JSON.
    Returns parsed dict on success, None on failure.
    """
    import re

    # Find the last complete top-level field by scanning for the deepest
    # valid truncation point — remove the last incomplete entry
    # Strategy: strip back to the last complete '},' or '}' at depth 1
    t = text.strip()

    # Remove trailing partial content after last complete array/object close
    # Walk backwards finding the last valid comma-terminated or closed block
    for cutpoint in range(len(t), 0, -1):
        candidate = t[:cutpoint].rstrip().rstrip(",")
        # Try closing all open structures
        open_braces   = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        if open_braces < 0 or open_brackets < 0:
            continue
        closed = candidate + ("]" * open_brackets) + ("}" * open_braces)
        try:
            return json.loads(closed)
        except json.JSONDecodeError:
            continue

    return None


def _parse_partial(raw: str, label: str, debug_dir: str | None = None) -> dict:
    """
    Parse one LLM response (for a single call A/B/C) into a plain dict.
    Handles markdown fences, prose wrappers, top-level arrays, and truncated JSON.
    Returns a (possibly partial) dict — missing keys are just absent.

    If debug_dir is provided, the raw response is saved to
    {debug_dir}/stage4_raw_{label}.txt before parsing so failures can be inspected.
    """
    # ── Save raw response for debugging ──────────────────────────────────────
    if debug_dir:
        try:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            Path(debug_dir, f"stage4_raw_{label}.txt").write_text(raw, encoding="utf-8")
        except Exception:
            pass  # never let debug I/O crash the pipeline

    text = raw.strip()

    # Strip markdown / prose — find the outermost { or [ block
    if not text.startswith("{") and not text.startswith("["):
        start_brace   = text.find("{")
        start_bracket = text.find("[")
        # Pick whichever comes first (and exists)
        if start_brace == -1:
            start = start_bracket
            end   = text.rfind("]")
        elif start_bracket == -1:
            start = start_brace
            end   = text.rfind("}")
        else:
            if start_brace < start_bracket:
                start = start_brace
                end   = text.rfind("}")
            else:
                start = start_bracket
                end   = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        else:
            text = "\n".join(
                line for line in text.splitlines()
                if not line.strip().startswith("```")
                and not line.strip().startswith("#")
            ).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        recovered = _attempt_json_recovery(text)
        if recovered:
            print(f"  [stage4-{label}] Warning: JSON truncated, recovered partial response.")
            data = recovered
        else:
            print(f"  [stage4-{label}] ⚠️  Could not parse JSON — call returned empty dict.")
            return {}

    # ── Handle top-level array (model returned [...] instead of {{...}}) ─────
    # This happens with some local models when asked for a list field.
    _LABEL_KEY = {"A": None, "B": "features", "C": "user_roles"}
    if isinstance(data, list):
        wrap_key = _LABEL_KEY.get(label)
        if wrap_key:
            print(f"  [stage4-{label}] Top-level array — wrapping as '{wrap_key}'.")
            data = {wrap_key: data}
        else:
            print(f"  [stage4-{label}] ⚠️  Unexpected top-level array — discarding.")
            return {}

    # ── Unwrap envelope keys (local models often wrap in {{"result": {{...}}}} etc.) ──
    if (isinstance(data, dict)
            and len(data) == 1
            and isinstance(list(data.values())[0], dict)):
        wrapper_key = list(data.keys())[0]
        print(f"  [stage4-{label}] Unwrapping envelope key '{wrapper_key}'.")
        data = list(data.values())[0]

    return data if isinstance(data, dict) else {}


def _hydrate_domain_model(data: dict) -> DomainModel:
    """
    Apply field remapping, synthesize missing critical fields, and build
    a DomainModel from the merged dict produced by the three partial calls.
    """
    # ── Field remapping ───────────────────────────────────────────────────────
    _ALIASES: dict[str, list[str]] = {
        "domain_name":      ["name", "system_name", "title", "domain", "project_name",
                             "application_name", "app_name", "project", "system"],
        "description":      ["summary", "overview", "desc", "purpose", "about",
                             "context", "background", "introduction"],
        "user_roles":       ["roles", "actors", "users", "user_types", "stakeholders",
                             "personas", "role_list", "role", "user_list",
                             "groups", "access_levels", "user_groups", "actor_list"],
        "key_entities":     ["entities", "models", "objects", "core_entities", "nodes",
                             "classes", "tables", "resources"],
        "bounded_contexts": ["modules", "contexts", "domains", "subsystems",
                             "components", "packages", "namespaces", "services"],
        "workflows":        ["flows", "processes", "user_flows",
                             "journeys", "scenarios", "interactions", "edges"],
        "features":         ["capabilities", "functionality", "functions",
                             "requirements", "stories", "tasks", "use_cases",
                             "feature_list", "feature", "business_features",
                             "modules_features", "epics", "operations"],
    }
    remapped: list[str] = []
    for target, aliases in _ALIASES.items():
        if target not in data:
            for alias in aliases:
                if alias in data:
                    val = data[alias]
                    if target == "key_entities" and val and isinstance(val[0], dict):
                        val = [e.get("name") or e.get("title") or str(e)
                               for e in val if isinstance(e, dict)]
                    data[target] = val
                    remapped.append(f"{alias}→{target}")
                    break
    if remapped:
        print(f"  [stage4] Field remapping applied: {', '.join(remapped)}")

    # ── Synthesize missing critical fields ───────────────────────────────────
    if "domain_name" not in data:
        guessed = (
            data.get("title") or data.get("name") or data.get("system") or
            data.get("project") or data.get("app") or "Unknown System"
        )
        data["domain_name"] = guessed if isinstance(guessed, str) and guessed else "Unknown System"
        print(f"  [stage4] ⚠️  domain_name missing — synthesized: '{data['domain_name']}'")

    if "description" not in data:
        entities    = data.get("key_entities", [])[:3]
        contexts    = data.get("bounded_contexts", [])[:3]
        entity_str  = ", ".join(str(e) for e in entities)  if entities  else ""
        context_str = ", ".join(str(c) for c in contexts)  if contexts  else ""
        parts: list[str] = []
        if entity_str:  parts.append(f"Core entities: {entity_str}")
        if context_str: parts.append(f"Bounded contexts: {context_str}")
        data["description"] = (
            ". ".join(parts) + "." if parts
            else "PHP web application (description not available from model response)."
        )
        print(f"  [stage4] ⚠️  description missing — synthesized from available fields.")

    # Warn on any still-missing non-critical fields
    required_warn = {"features", "user_roles", "key_entities", "bounded_contexts", "workflows"}
    missing_warn  = required_warn - set(data.keys())
    if missing_warn:
        print(f"  [stage4] ⚠️  Missing fields defaulting to []: {sorted(missing_warn)}")

    return DomainModel(
        domain_name      = data.get("domain_name", "Unknown System"),
        description      = data.get("description", ""),
        user_roles       = data.get("user_roles", []),
        features         = data.get("features", []),
        workflows        = data.get("workflows", []),
        bounded_contexts = data.get("bounded_contexts", []),
        key_entities     = data.get("key_entities", []),
    )


# ─── Persistence ───────────────────────────────────────────────────────────────

def _save_domain_model(model: DomainModel, path: str) -> None:
    """Serialise DomainModel to JSON."""
    import dataclasses
    data = dataclasses.asdict(model)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _load_domain_model(path: str) -> DomainModel:
    """Restore DomainModel from saved JSON."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return DomainModel(
        domain_name      = data.get("domain_name", ""),
        description      = data.get("description", ""),
        user_roles       = data.get("user_roles", []),
        features         = data.get("features", []),
        workflows        = data.get("workflows", []),
        bounded_contexts = data.get("bounded_contexts", []),
        key_entities     = data.get("key_entities", []),
    )


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _get_quality_score(ctx: PipelineContext) -> int:
    """Read quality score from preflight report if available."""
    report_path = ctx.output_path("preflight_report.json")
    try:
        with open(report_path, encoding="utf-8") as fh:
            return json.load(fh).get("quality_score", 70)
    except Exception:
        return 70  # safe default


def _assert_prerequisites(ctx: PipelineContext) -> None:
    """Raise RuntimeError if required upstream stages are missing."""
    if ctx.code_map is None:
        raise RuntimeError(
            "[stage4] ctx.code_map is None — run Stage 1 first."
        )
    if ctx.embedding_meta is None:
        raise RuntimeError(
            "[stage4] ctx.embedding_meta is None — run Stage 3 first."
        )
    # Check that at least one LLM provider is configured
    from pipeline.llm_client import get_provider
    try:
        get_provider()
    except RuntimeError as e:
        raise RuntimeError(f"[stage4] {e}") from e