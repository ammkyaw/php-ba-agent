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
CLAUDE_MODEL     = "claude-sonnet-4-20250514"
MAX_TOKENS       = 8192
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

    # ── Build prompt ──────────────────────────────────────────────────────────
    quality_score = _get_quality_score(ctx)
    system_prompt = _build_system_prompt(quality_score)
    user_prompt   = _build_user_prompt(ctx, chunks)

    print(f"  [stage4] Calling Claude ({CLAUDE_MODEL}) ...")
    print(f"  [stage4] Context: {len(user_prompt)} chars, "
          f"quality_score={quality_score}")

    # ── Call Claude ───────────────────────────────────────────────────────────
    raw_response = _call_claude(system_prompt, user_prompt)

    # ── Parse response ────────────────────────────────────────────────────────
    domain_model = _parse_response(raw_response)

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

def _build_system_prompt(quality_score: int) -> str:
    """
    Build the system prompt for the DomainAnalystAgent.
    Adjusts instruction strictness based on signal quality score.
    """
    if quality_score >= 70:
        evidence_instruction = (
            "Extract the domain model precisely and completely from the evidence provided. "
            "Be specific — use actual table names, field names, page names, and function names "
            "from the codebase evidence."
        )
    else:
        evidence_instruction = (
            "Extract what you can from the evidence, but note where evidence is thin. "
            "Avoid speculation — if a feature or entity is not evidenced in the codebase "
            "context, do not invent it. Mark uncertain items with a 'confidence': 'low' field."
        )

    return f"""You are a senior Business Analyst and software architect specialising in
reverse-engineering legacy PHP applications into structured business domain models.

Your task is to analyse the provided PHP codebase evidence and produce a complete,
structured domain model in JSON format.

{evidence_instruction}

Rules:
- Output ONLY valid JSON — no markdown fences, no preamble, no explanation
- Use the actual names from the codebase (table names, page names, field names)
- Every feature must reference the pages and tables that implement it
- Workflows must describe real user journeys evidenced in the code
- Be concrete and specific, not generic

Output this exact JSON structure:
{{
  "domain_name": "Short descriptive system name (e.g. 'Car Rental Management System')",
  "description": "2-3 sentence plain-English overview of what the system does and who uses it",
  "user_roles": [
    {{
      "role": "Role name (e.g. Customer, Admin, Staff)",
      "description": "What this role can do in the system",
      "entry_points": ["login.php", "registration.php"]
    }}
  ],
  "key_entities": ["Entity1", "Entity2"],
  "bounded_contexts": ["Context1", "Context2"],  // Use the STRUCTURALLY DETECTED MODULES as the primary source for these. Rename for clarity, merge near-empty ones, but anchor to detected module names.
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
}}"""


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
            col_str = ", ".join(c.get("name", "?") for c in cols[:12])
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
    parts.append("\nNow produce the domain model JSON:")

    return "\n".join(parts)


# ─── Claude API Call ───────────────────────────────────────────────────────────

def _call_claude(system_prompt: str, user_prompt: str) -> str:
    """
    Call the configured LLM provider (Claude or Gemini) and return the response.
    Provider is determined by llm_client.get_provider() — see pipeline/llm_client.py.
    """
    from pipeline.llm_client import call_llm
    return call_llm(
        system_prompt = system_prompt,
        user_prompt   = user_prompt,
        max_tokens    = MAX_TOKENS,
        label         = "stage4",
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


def _parse_response(raw: str) -> DomainModel:
    """
    Parse Claude's JSON response into a DomainModel.
    Handles markdown fences if Claude accidentally includes them.

    Raises:
        RuntimeError: If the response cannot be parsed as valid JSON
                      or is missing required fields.
    """
    # Strip any accidental markdown fences
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Attempt recovery: find the last complete top-level key and truncate
        recovered = _attempt_json_recovery(text)
        if recovered:
            print(f"  [stage4] Warning: JSON was truncated, recovered partial response.")
            data = recovered
        else:
            raise RuntimeError(
                f"[stage4] Failed to parse Claude's response as JSON: {e}\n"
                f"Raw response (first 500 chars):\n{raw[:500]}"
            )

    # Validate required top-level keys
    required = {"domain_name", "description", "features", "user_roles",
                "key_entities", "bounded_contexts", "workflows"}
    missing = required - set(data.keys())
    if missing:
        raise RuntimeError(
            f"[stage4] Claude response missing required fields: {missing}\n"
            f"Got keys: {list(data.keys())}"
        )

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
