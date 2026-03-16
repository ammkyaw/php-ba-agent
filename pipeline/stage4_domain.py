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
MAX_TOKENS_GAP_FILL = 6_000   # Call D: gap-fill features for uncovered pages
DOMAIN_FILE      = "domain_model.json"
COVERAGE_FILE    = "coverage_report.json"

# Gap-fill: max modules per module-grouped call (covers ~10× more files than pages)
GAP_FILL_MAX_MODULES = 30
# Gap-fill: max individual files per call (fallback when no module structure)
GAP_FILL_MAX_PAGES = 100
# Gap-fill: maximum number of loop rounds
MAX_GAP_ROUNDS = 20

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

    # ── Build grounding lists from code_map (anti-hallucination) ─────────────
    cm = ctx.code_map
    known_tables: list[str] = sorted({
        q.get("table", "") for q in (cm.sql_queries or [])
        if q.get("table") and q["table"] not in ("UNKNOWN", "")
    })

    # POST field names — used to ground 'inputs' arrays in features so the
    # LLM doesn't hallucinate form fields that don't exist in the codebase.
    known_fields: list[str] = sorted({
        s["key"] for s in (cm.superglobals or [])
        if s.get("var") == "$_POST" and s.get("key")
    })

    # Wide filter set: ALL file basenames across the entire code_map.
    # This prevents _filter_hallucinated_refs from incorrectly stripping
    # controller / service / execution-path references just because they
    # are not in cm.html_pages.
    known_files_all: set[str] = _build_all_known_filenames(cm)

    # Grounding list injected into the prompt: exec_paths first (most
    # informative for business logic), then html_pages, then controllers.
    # Capped to keep the prompt token budget manageable.
    known_pages: list[str] = _build_grounding_pages(cm, cap=300)

    # ── Call A: meta + entities + contexts (fast, ~4K tokens) ────────────────
    print(f"  [stage4] Call 1/3 — domain name, entities, contexts ...")
    raw_a = _call_part(_system_meta(quality_score), user_prompt,
                       MAX_TOKENS_META, "stage4-A")
    data_a = _parse_partial(raw_a, "A", debug_dir)

    # ── Call B: features (largest field, ~8K tokens) ──────────────────────────
    print(f"  [stage4] Call 2/3 — features ...")
    raw_b = _call_part(
        _system_features(quality_score,
                         known_tables=known_tables,
                         known_pages=known_pages,
                         known_fields=known_fields),
        user_prompt, MAX_TOKENS_FEATURES, "stage4-B",
    )
    data_b = _parse_partial(raw_b, "B", debug_dir)

    # Filter hallucinated page/table names.
    # Use known_files_all (not just known_pages) so that controller / service
    # file references are preserved — only truly invented filenames are dropped.
    data_b = _filter_hallucinated_refs(data_b,
                                       known_pages_lower=known_files_all,
                                       known_tables_lower={t.lower() for t in known_tables})

    # ── Call C: user roles + workflows (~8K tokens) ───────────────────────────
    print(f"  [stage4] Call 3/3 — user roles, workflows ...")
    raw_c = _call_part(_system_roles_workflows(quality_score), user_prompt,
                       MAX_TOKENS_ROLES_WF, "stage4-C")
    data_c = _parse_partial(raw_c, "C", debug_dir)

    # ── Merge all three dicts (A wins on overlaps, B/C fill in their fields) ──
    merged = {**data_a, **data_b, **data_c}

    # ── Hydrate DomainModel ───────────────────────────────────────────────────
    domain_model = _hydrate_domain_model(merged)

    # ── Coverage metrics ──────────────────────────────────────────────────────
    coverage_report = _compute_coverage(ctx, domain_model, debug_dir)

    # ── Gap-fill pass ─────────────────────────────────────────────────────────
    # Use the wide filter set for gap-fill too so new features can reference
    # any real file (not only html_pages).
    domain_model = _gap_fill_pass(
        ctx, domain_model, coverage_report, user_prompt, quality_score, debug_dir,
        known_tables=known_tables, known_pages_lower=known_files_all,
        known_fields=known_fields,
    )

    # ── Post-gap-fill coverage (shows improvement vs initial report) ───────────
    print(f"  [stage4] Coverage after gap-fill:")
    _compute_coverage(ctx, domain_model, debug_dir)

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


def _build_all_known_filenames(cm: Any) -> set[str]:
    """
    Build a comprehensive set of ALL lowercase file basenames known to the
    code_map.  Used as the wide filter for _filter_hallucinated_refs() so
    that real controller / service / execution-path filenames are NOT stripped
    just because they are absent from cm.html_pages.

    Draws from every list in cm that carries a "file" key, plus html_pages.
    """
    files: set[str] = set()
    for p in (cm.html_pages or []):
        if isinstance(p, str):
            files.add(Path(p).name.lower())
    for lst in [
        cm.routes or [],
        cm.controllers or [],
        cm.services or [],
        cm.form_fields or [],
        cm.sql_queries or [],
        cm.execution_paths or [],
        cm.redirects or [],
    ]:
        for item in lst:
            if isinstance(item, dict) and item.get("file"):
                files.add(Path(item["file"]).name.lower())
    return files


def _build_grounding_pages(cm: Any, cap: int = 150) -> list[str]:
    """
    Build a **prioritised** list of PHP filenames to inject into the features
    prompt as grounding.

    Priority order (highest value for feature mapping first):
      1. Execution-path files (stage15 detected entry points — carry most context)
      2. HTML pages (traditional view/entry-point files)
      3. Controller files
      4. Service files
    Capped at *cap* entries to avoid blowing the token budget.

    Returns unique basenames (insertion-order deduplicated).
    """
    seen:  set[str]  = set()
    pages: list[str] = []

    def _add(fname: str) -> None:
        key = fname.lower()
        if key not in seen:
            seen.add(key)
            pages.append(fname)

    # 1 — Execution paths (most important: business logic entry points)
    for ep in (cm.execution_paths or []):
        f = ep.get("file", "")
        if f:
            _add(Path(f).name)

    # 2 — HTML pages (view / traditional entry points)
    for p in sorted(cm.html_pages or []):
        _add(Path(p).name)

    # 3 — Controllers
    for c in (cm.controllers or []):
        f = c.get("file", "")
        if f:
            _add(Path(f).name)

    # 4 — Services (lower priority — implementation detail, not entry point)
    for s in (cm.services or []):
        f = s.get("file", "")
        if f:
            _add(Path(f).name)

    return pages[:cap]


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


def _system_features(
    quality_score: int,
    known_tables: list[str] | None = None,
    known_pages:  list[str] | None = None,
    known_fields: list[str] | None = None,
) -> str:
    """Call B — features list only.

    known_tables / known_pages / known_fields: when provided, injected as
    mandatory grounding so the model cannot hallucinate names that don't exist
    in the actual codebase.
    """
    # ── Build grounding block ────────────────────────────────────────────────
    grounding_parts: list[str] = []
    if known_tables:
        tables_str = ", ".join(known_tables[:100])   # cap to avoid token overflow
        grounding_parts.append(
            "MANDATORY GROUNDING — ACTUAL DATABASE TABLES IN THIS CODEBASE:\n"
            f"{tables_str}\n"
            "You MUST use ONLY table names from the list above in 'tables' arrays.\n"
            "Do NOT invent table names like 'products', 'orders', 'cart_items' "
            "unless they actually appear in this list."
        )
    if known_pages:
        pages_str = ", ".join(known_pages[:250])     # cap to avoid token overflow
        grounding_parts.append(
            "MANDATORY GROUNDING — ACTUAL PHP PAGE FILES IN THIS CODEBASE:\n"
            f"{pages_str}\n"
            "You MUST use ONLY filenames from the list above in 'pages' arrays.\n"
            "Do NOT invent filenames like 'registration.php', 'catalog.php', "
            "'cart.php', 'checkout.php' unless they actually appear in this list."
        )
    if known_fields:
        fields_str = ", ".join(known_fields[:200])   # cap to avoid token overflow
        grounding_parts.append(
            "MANDATORY GROUNDING — ACTUAL POST FORM FIELDS IN THIS CODEBASE:\n"
            f"{fields_str}\n"
            "You MUST populate each feature's 'inputs' array ONLY with field names "
            "from the list above that are relevant to that feature.\n"
            "Do NOT invent field names. If a feature has no matching fields, use []."
        )
    grounding = ("\n\n" + "\n\n".join(grounding_parts)) if grounding_parts else ""

    return f"""{_SYSTEM_PREAMBLE}
{_evidence_instruction(quality_score)}{grounding}

Output ONLY this JSON (no other fields):
{{
  "features": [
    {{
      "name": "Feature name",
      "description": "What this feature does from a business perspective",
      "pages": ["actual_page.php"],
      "tables": ["actual_table"],
      "inputs": ["field1", "field2"],
      "outputs": "What the user gets after completing this feature",
      "business_rules": ["Rule 1", "Rule 2"]
    }}
  ]
}}
Every feature MUST reference pages and tables that ACTUALLY EXIST in the codebase
(from the grounding lists above). Never use placeholder or invented names.

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


def _group_uncovered_by_module(
    uncovered_files: list[str],
    exec_paths: list[dict],
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Group uncovered file basenames by their module directory.

    Returns
    -------
    module_groups : {module_name: [file1.php, file2.php, ...]}
        Only modules that have at least one uncovered file.
    flat_remainder : [file.php, ...]
        Files that are NOT inside a modules/ directory (no module grouping possible).
    """
    from collections import defaultdict

    # Build file-basename → module name map from exec_paths
    basename_to_module: dict[str, str] = {}
    for ep in exec_paths:
        f = ep.get("file", "")
        if not f:
            continue
        mod = _extract_module_name(f)
        if mod:
            basename_to_module[Path(f).name.lower()] = mod

    module_groups: dict[str, list[str]] = defaultdict(list)
    flat_remainder: list[str] = []

    for fname in uncovered_files:
        mod = basename_to_module.get(fname.lower())
        if mod:
            module_groups[mod].append(fname)
        else:
            flat_remainder.append(fname)

    return dict(module_groups), flat_remainder


def _system_gap_fill(quality_score: int, uncovered_pages: list[str],
                     existing_feature_names: list[str],
                     known_tables: list[str] | None = None,
                     known_fields: list[str] | None = None,
                     module_groups: dict[str, list[str]] | None = None) -> str:
    """Call D — gap-fill features for pages not yet covered by known features.

    When *module_groups* is supplied the prompt presents modules (with sample
    filenames) instead of bare filenames — one feature per module covers all
    sibling files through module-expansion and is far more token-efficient.
    """
    existing_str = ", ".join(f'"{n}"' for n in existing_feature_names[:20])
    grounding_parts: list[str] = []
    if known_tables:
        t_str = ", ".join(known_tables[:100])
        grounding_parts.append(
            f"MANDATORY GROUNDING — use ONLY these actual table names in 'tables' arrays:\n{t_str}"
        )
    if known_fields:
        f_str = ", ".join(known_fields[:200])
        grounding_parts.append(
            f"MANDATORY GROUNDING — populate 'inputs' ONLY from these actual POST field names:\n{f_str}\n"
            "Do NOT invent field names. If a feature has no matching fields, use []."
        )
    tables_grounding = ("\n" + "\n\n".join(grounding_parts) + "\n") if grounding_parts else ""

    if module_groups:
        # Module-grouped format: one feature per module
        module_lines = []
        for mod, files in module_groups.items():
            sample = ", ".join(files[:5])
            extra  = f" (+{len(files)-5} more)" if len(files) > 5 else ""
            module_lines.append(f'  - {mod}: {sample}{extra}')
        modules_str = "\n".join(module_lines)

        coverage_instruction = f"""The following application MODULES are NOT yet covered by any existing feature.
Each module is a group of PHP files under modules/<ModuleName>/:

{modules_str}

Existing features already extracted (do NOT duplicate): {existing_str}

Create ONE feature per module (or merge closely related small modules into one feature).
Each feature's "pages" array MUST include the representative PHP filenames listed above."""
    else:
        pages_str = ", ".join(f'"{p}"' for p in uncovered_pages)
        coverage_instruction = f"""The following PHP pages are NOT yet covered by any existing feature:
{pages_str}

Existing features already extracted (do NOT duplicate): {existing_str}

For each uncovered page, determine what business feature it represents.
Group multiple uncovered pages under one feature if they implement the same business function."""

    return f"""{_SYSTEM_PREAMBLE}
{_evidence_instruction(quality_score)}{tables_grounding}

{coverage_instruction}

Output ONLY this JSON (no other fields):
{{
  "features": [
    {{
      "name": "Feature name",
      "description": "What this feature does from a business perspective",
      "pages": ["representative_file.php"],
      "tables": ["relevant_table"],
      "inputs": ["field1", "field2"],
      "outputs": "What the user gets after completing this feature",
      "business_rules": []
    }}
  ]
}}

Now produce the JSON for new features covering the uncovered modules/pages:"""


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

    # ── Action Clusters from Stage 2.8 ───────────────────────────────────────
    # Provide the similarity-based cluster map as an alternative bounded-context
    # seed.  This is especially useful when the Stage 2 graph was built without
    # community detection (no detected_modules) or the graph is absent.
    ac = getattr(ctx, "action_clusters", None)
    if ac and ac.clusters and not detected_modules:
        parts.append(f"\n=== ACTION CLUSTERS — STAGE 2.8 ({len(ac.clusters)}) ===")
        parts.append(
            "These clusters were computed from shared DB tables, route prefixes, "
            "and module directory structure.  Use them as the basis for "
            "bounded_contexts — merge near-empty ones, rename for clarity, "
            "but anchor to these cluster names."
        )
        for c in ac.clusters[:40]:   # cap to avoid prompt bloat
            tbl_preview = ", ".join(c.tables[:5])
            if len(c.tables) > 5:
                tbl_preview += f" (+{len(c.tables)-5} more)"
            parts.append(
                f"  [{c.name}]  files={c.file_count}"
                + (f"  tables=[{tbl_preview}]" if tbl_preview else "")
                + (f"  route={c.route_prefix}" if c.route_prefix else "")
            )

    # ── Detected Business Rules from Stage 2.9 ───────────────────────────────
    # High-confidence rules (schema-enforced ≥ 0.9, guard-clause ≥ 0.75) are
    # injected as hard constraints.  The LLM must surface these in features,
    # workflows, and key_entities rather than inventing its own constraints.
    inv = getattr(ctx, "invariants", None)
    if inv and inv.rules:
        high_conf = [r for r in inv.rules if r.confidence >= 0.75]
        if high_conf:
            parts.append(
                f"\n=== DETECTED BUSINESS RULES — STAGE 2.9 "
                f"({len(high_conf)} high-confidence) ==="
            )
            parts.append(
                "These rules were extracted statically from DB schema and guard "
                "clauses.  When generating features and workflows, reference these "
                "exact constraints.  Do NOT contradict them."
            )
            # Group by category for readability
            by_cat: dict[str, list] = {}
            for r in high_conf:
                by_cat.setdefault(r.category, []).append(r)
            for cat, cat_rules in sorted(by_cat.items()):
                parts.append(f"\n  [{cat}]")
                for r in cat_rules[:15]:      # cap per category
                    parts.append(f"    • {r.description}  [{r.entity}]")

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
    """Call the configured LLM and return the raw response string.

    json_mode=True forces local models (Ollama etc.) to output valid JSON
    via response_format, preventing Qwen3 / DeepSeek-R1 from writing prose.
    """
    from pipeline.llm_client import call_llm
    return call_llm(
        system_prompt = system_prompt,
        user_prompt   = user_prompt,
        max_tokens    = max_tokens,
        label         = label,
        json_mode     = True,
    )


# ─── Hallucination Filter ──────────────────────────────────────────────────────


def _filter_hallucinated_refs(
    data: dict,
    known_pages_lower:  set[str],
    known_tables_lower: set[str],
) -> dict:
    """
    Strip page and table references from features that don't exist in the
    actual codebase (i.e., the model hallucinated them).

    For each feature:
      - "pages"  → keep only names that appear (case-insensitive) in known_pages_lower
      - "tables" → keep only names that appear (case-insensitive) in known_tables_lower

    Empty grounding sets mean the code_map had nothing to check against, so
    we skip filtering for that dimension to avoid falsely removing valid refs.
    """
    features = [f for f in data.get("features", []) if isinstance(f, dict)]
    if not features:
        return data
    data["features"] = features

    hallucinated_pages  = 0
    hallucinated_tables = 0

    for feat in features:
        if not isinstance(feat, dict):
            continue

        if known_pages_lower:
            orig_pages = feat.get("pages", [])
            real_pages = [p for p in orig_pages
                          if isinstance(p, str) and Path(p).name.lower() in known_pages_lower]
            hallucinated_pages += len(orig_pages) - len(real_pages)
            feat["pages"] = real_pages

        if known_tables_lower:
            orig_tables = feat.get("tables", [])
            real_tables = [t for t in orig_tables
                           if isinstance(t, str) and t.lower() in known_tables_lower]
            hallucinated_tables += len(orig_tables) - len(real_tables)
            feat["tables"] = real_tables

    if hallucinated_pages or hallucinated_tables:
        print(
            f"  [stage4-B] Removed {hallucinated_pages} hallucinated page ref(s) "
            f"and {hallucinated_tables} hallucinated table ref(s) from features."
        )

    data["features"] = features
    return data


# ─── Coverage Metrics ──────────────────────────────────────────────────────────


def _extract_module_name(filepath: str) -> str | None:
    """
    Extract the module directory name from a SugarCRM / raw-PHP style path.

    Examples
    --------
    modules/ACL/ACLController.php        → "ACL"
    custom/modules/ACL/ACLController.php → "ACL"
    include/MassUpdate.php               → None  (not in a module dir)
    """
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part.lower() == "modules" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _build_module_expansion(
    exec_paths: list[dict],
    covered_pages: set[str],
) -> set[str]:
    """
    If any file in a module directory is already covered, expand coverage to
    ALL files in that module directory (module-level coverage).

    Also applies directory-level expansion for non-module files: if any file
    in a directory like include/CalendarProvider/ is covered, all siblings
    in that directory are counted as covered.

    Returns an expanded set of covered page basenames (lowercase).
    """
    from collections import defaultdict

    module_files: dict[str, list[str]] = defaultdict(list)
    dir_files: dict[str, list[str]] = defaultdict(list)  # parent_dir → [basename, ...]

    for ep in exec_paths:
        f = ep.get("file", "")
        if not f:
            continue
        mod = _extract_module_name(f)
        if mod:
            module_files[mod].append(Path(f).name.lower())
        else:
            # Group non-module files by their immediate parent directory
            parent = str(Path(f).parent)
            dir_files[parent].append(Path(f).name.lower())

    # Which modules have at least one covered file?
    covered_modules: set[str] = set()
    for mod, files in module_files.items():
        if any(f in covered_pages for f in files):
            covered_modules.add(mod)

    # Which non-module directories have at least one covered file?
    covered_dirs: set[str] = set()
    for parent, files in dir_files.items():
        if any(f in covered_pages for f in files):
            covered_dirs.add(parent)

    # Expand: all files in covered modules/dirs are considered covered
    expanded: set[str] = set(covered_pages)
    for mod in covered_modules:
        expanded.update(module_files[mod])
    for parent in covered_dirs:
        expanded.update(dir_files[parent])
    return expanded


def _compute_coverage(
    ctx: PipelineContext,
    domain_model: "DomainModel",
    debug_dir: str | None = None,
) -> dict:
    """
    Compute four coverage metrics after domain model extraction.

      exec_coverage   — fraction of execution-path entry points in any feature's 'pages'
                        (primary metric for API / controller-heavy codebases)
      page_coverage   — fraction of cm.html_pages in any feature's 'pages'
                        (traditional metric for view-based codebases)
      table_coverage  — fraction of unique SQL tables in any feature's 'tables'
      field_coverage  — fraction of POST fields in any feature's 'inputs'

    ``pages_uncovered`` in the returned report is drawn from execution-path
    files first (highest value for gap-fill), then from html_pages not already
    included.  The gap-fill pass uses this list to request new features.

    Saves coverage_report.json to debug_dir and prints a summary line.
    """
    cm = ctx.code_map

    # ── Collect everything the features claim to cover ────────────────────────
    covered_pages  : set[str] = set()
    covered_tables : set[str] = set()
    covered_inputs : set[str] = set()
    for feat in domain_model.features:
        for p in feat.get("pages", []):
            covered_pages.add(Path(p).name.lower())
        for t in feat.get("tables", []):
            covered_tables.add(t.lower())
        for inp in feat.get("inputs", []):
            covered_inputs.add(inp.lower())

    # ── Module-expanded coverage set ──────────────────────────────────────────
    # If any file in a module dir is explicitly covered, all sibling files in
    # that module are counted as covered (module-level granularity).
    module_covered_pages = _build_module_expansion(
        cm.execution_paths or [], covered_pages
    )

    # ── Execution-path coverage (primary for API codebases) ───────────────────
    # execution_paths are the stage15 static-analysis entry points — they map
    # 1-to-1 to controller/handler files and represent actual business logic.
    ep_all: list[str] = sorted({
        Path(ep["file"]).name
        for ep in (cm.execution_paths or [])
        if ep.get("file")
    })
    ep_covered   = [p for p in ep_all if p.lower() in module_covered_pages]
    ep_uncovered = [p for p in ep_all if p.lower() not in module_covered_pages]
    exec_cov     = len(ep_covered) / len(ep_all) if ep_all else 1.0

    # ── HTML-page coverage (secondary — view / entry-point files) ────────────
    html_all       = [Path(p).name for p in (cm.html_pages or [])]
    html_covered   = [p for p in html_all if p.lower() in module_covered_pages]
    html_uncovered = [p for p in html_all if p.lower() not in module_covered_pages]
    page_cov       = len(html_covered) / len(html_all) if html_all else 1.0

    # ── Gap-fill priority list: exec paths first, then remaining html_pages ───
    seen_unc: set[str] = set()
    pages_uncovered: list[str] = []
    for p in ep_uncovered:
        key = p.lower()
        if key not in seen_unc:
            seen_unc.add(key)
            pages_uncovered.append(p)
    for p in html_uncovered:
        key = p.lower()
        if key not in seen_unc:
            seen_unc.add(key)
            pages_uncovered.append(p)

    pages_covered = sorted(set(ep_covered + html_covered))

    # ── Table coverage ────────────────────────────────────────────────────────
    all_tables = sorted({
        q.get("table", "") for q in (cm.sql_queries or [])
        if q.get("table") and q["table"] not in ("UNKNOWN", "")
    })
    tables_covered   = [t for t in all_tables if t.lower() in covered_tables]
    tables_uncovered = [t for t in all_tables if t.lower() not in covered_tables]
    table_cov        = len(tables_covered) / len(all_tables) if all_tables else 1.0

    # ── Field coverage ────────────────────────────────────────────────────────
    all_fields = sorted({
        s["key"] for s in (cm.superglobals or [])
        if s.get("var") == "$_POST" and s.get("key")
    })
    fields_covered   = [f for f in all_fields if f.lower() in covered_inputs]
    fields_uncovered = [f for f in all_fields if f.lower() not in covered_inputs]
    field_cov        = len(fields_covered) / len(all_fields) if all_fields else 1.0

    report = {
        # Primary metric: execution-path coverage
        "exec_coverage":     round(exec_cov,  3),
        "exec_covered":      ep_covered,
        "exec_uncovered":    ep_uncovered,
        # Secondary metric: html-page coverage
        "page_coverage":     round(page_cov,  3),
        "pages_covered":     pages_covered,
        "pages_uncovered":   pages_uncovered,   # used by gap-fill pass
        # Table + field
        "table_coverage":    round(table_cov, 3),
        "field_coverage":    round(field_cov, 3),
        "tables_covered":    tables_covered,
        "tables_uncovered":  tables_uncovered,
        "fields_covered":    fields_covered,
        "fields_uncovered":  fields_uncovered,
    }

    # ── Persist ───────────────────────────────────────────────────────────────
    if debug_dir:
        try:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            cov_path = Path(debug_dir, COVERAGE_FILE)
            with open(cov_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [stage4] Warning: could not save {COVERAGE_FILE}: {e}")

    # ── Print summary ─────────────────────────────────────────────────────────
    # Show exec-path coverage as the headline number; html-page as secondary.
    print(
        f"  [stage4] Coverage — "
        f"exec-paths: {exec_cov:.0%} ({len(ep_covered)}/{len(ep_all)}), "
        f"pages: {page_cov:.0%} ({len(html_covered)}/{len(html_all)}), "
        f"tables: {table_cov:.0%} ({len(tables_covered)}/{len(all_tables)}), "
        f"fields: {field_cov:.0%} ({len(fields_covered)}/{len(all_fields)})"
    )
    if ep_uncovered:
        sample = ep_uncovered[:8]
        suffix = f" ... +{len(ep_uncovered) - 8} more" if len(ep_uncovered) > 8 else ""
        print(f"  [stage4]   Uncovered exec-paths: {sample}{suffix}")
    elif html_uncovered:
        sample = html_uncovered[:8]
        suffix = f" ... +{len(html_uncovered) - 8} more" if len(html_uncovered) > 8 else ""
        print(f"  [stage4]   Uncovered pages     : {sample}{suffix}")
    if tables_uncovered:
        sample = tables_uncovered[:8]
        suffix = f" ... +{len(tables_uncovered) - 8} more" if len(tables_uncovered) > 8 else ""
        print(f"  [stage4]   Uncovered tables    : {sample}{suffix}")

    return report


def _gap_fill_pass(
    ctx: PipelineContext,
    domain_model: "DomainModel",
    coverage_report: dict,
    user_prompt: str,
    quality_score: int,
    debug_dir: str | None = None,
    known_tables: list[str] | None = None,
    known_pages_lower: set[str] | None = None,
    known_fields: list[str] | None = None,
) -> "DomainModel":
    """
    If page_coverage < 1.0, loop up to MAX_GAP_ROUNDS additional LLM calls
    (Call D, D2, D3 …) to extract features for uncovered entry points and
    merge them into the domain model.  Each round re-computes uncovered pages
    against the latest features so new features discovered in round N are
    accounted for in round N+1.

    Returns the (possibly enriched) DomainModel.
    """
    cm = ctx.code_map

    # Track which modules have already been sent to the LLM so we don't
    # spin indefinitely on modules the model refuses to list page refs for.
    attempted_modules: set[str] = set()
    # Track which non-module files have already been sent to avoid re-sending.
    attempted_flat: set[str] = set()
    # Allow up to this many consecutive empty/failed LLM rounds before giving up.
    MAX_CONSECUTIVE_EMPTY = 3
    consecutive_empty = 0

    for gap_round in range(1, MAX_GAP_ROUNDS + 1):
        # Re-compute which pages are still uncovered after previous rounds
        covered_pages: set[str] = set()
        for feat in domain_model.features:
            for p in feat.get("pages", []):
                covered_pages.add(Path(p).name.lower())

        ep_all: list[str] = sorted({
            Path(ep["file"]).name
            for ep in (cm.execution_paths or [])
            if ep.get("file")
        })
        html_all: list[str] = [Path(p).name for p in (cm.html_pages or [])]

        # Use module-expansion so module-covered siblings don't count as uncovered
        module_covered = _build_module_expansion(
            cm.execution_paths or [], covered_pages
        )

        seen_unc: set[str] = set()
        uncovered: list[str] = []
        for p in ep_all + html_all:
            if p.lower() not in module_covered and p.lower() not in seen_unc:
                seen_unc.add(p.lower())
                uncovered.append(p)

        if not uncovered:
            print(f"  [stage4] All pages covered after round {gap_round - 1} — done.")
            break

        # ── Group uncovered files by module and by flat remainder ─────────────
        module_groups, flat_files = _group_uncovered_by_module(
            uncovered, cm.execution_paths or []
        )

        # Modules not yet attempted this pipeline run
        fresh_module_groups = {
            m: files for m, files in module_groups.items()
            if m not in attempted_modules
        }

        call_label = f"stage4-D{gap_round}"
        existing_names = [f["name"] for f in domain_model.features]

        if fresh_module_groups:
            # ── Module-grouped mode (covers ~10× more files per call) ──────
            batch_modules = dict(list(fresh_module_groups.items())[:GAP_FILL_MAX_MODULES])
            attempted_modules.update(batch_modules.keys())
            n_files = sum(len(v) for v in batch_modules.values())
            print(
                f"  [stage4] Call D{gap_round} — gap-fill "
                f"{len(batch_modules)} module(s) (~{n_files} files) "
                f"of {len(fresh_module_groups)} fresh uncovered modules"
                + (f" ({len(flat_files)} flat files pending)" if flat_files else "")
            )
            gap_system = _system_gap_fill(
                quality_score, [], existing_names,
                known_tables=known_tables, known_fields=known_fields,
                module_groups=batch_modules,
            )
        elif flat_files:
            # ── Flat fallback: non-module files (include/, lib/, etc.) ──────
            fresh_flat = [f for f in flat_files if f.lower() not in attempted_flat]
            gap_pages = fresh_flat[:GAP_FILL_MAX_PAGES]
            if not gap_pages:
                print(f"  [stage4] All flat files sent in previous rounds — done.")
                break
            attempted_flat.update(f.lower() for f in gap_pages)
            print(
                f"  [stage4] Call D{gap_round} — gap-fill "
                f"{len(gap_pages)}/{len(flat_files)} non-module file(s): "
                f"{gap_pages[:5]}{'...' if len(gap_pages) > 5 else ''}"
            )
            gap_system = _system_gap_fill(
                quality_score, gap_pages, existing_names,
                known_tables=known_tables, known_fields=known_fields,
            )
        else:
            print(f"  [stage4] No fresh modules or flat files remain — done.")
            break
        raw_d          = _call_part(gap_system, user_prompt, MAX_TOKENS_GAP_FILL,
                                    call_label)
        data_d         = _parse_partial(raw_d, f"D{gap_round}", debug_dir)

        # Filter hallucinated refs from gap-fill response
        if known_pages_lower or known_tables:
            data_d = _filter_hallucinated_refs(
                data_d,
                known_pages_lower  = known_pages_lower  or set(),
                known_tables_lower = {t.lower() for t in (known_tables or [])},
            )

        # Normalise: some models return a bare list
        new_features: list = data_d.get("features", [])
        if not new_features and isinstance(data_d, list):
            new_features = data_d

        if not new_features:
            consecutive_empty += 1
            print(
                f"  [stage4] Gap-fill round {gap_round} returned no new features "
                f"({consecutive_empty}/{MAX_CONSECUTIVE_EMPTY} consecutive empty)."
            )
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                print(f"  [stage4] {MAX_CONSECUTIVE_EMPTY} consecutive empty rounds — stopping.")
                break
            continue  # try next batch of files

        consecutive_empty = 0  # reset on any successful response

        # Merge — skip duplicates by name (case-insensitive)
        existing_lower = {f["name"].lower() for f in domain_model.features}
        added: list[str] = []
        for feat in new_features:
            if isinstance(feat, dict) and feat.get("name"):
                if feat["name"].lower() not in existing_lower:
                    domain_model.features.append(feat)
                    existing_lower.add(feat["name"].lower())
                    added.append(feat["name"])

        if added:
            consecutive_empty = 0
            suffix = f" ... +{len(added) - 5} more" if len(added) > 5 else ""
            print(f"  [stage4] Gap-fill round {gap_round} added "
                  f"{len(added)} feature(s): {added[:5]}{suffix}")
        else:
            consecutive_empty += 1
            print(
                f"  [stage4] Gap-fill round {gap_round}: all returned features already present "
                f"({consecutive_empty}/{MAX_CONSECUTIVE_EMPTY} consecutive empty)."
            )
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                print(f"  [stage4] {MAX_CONSECUTIVE_EMPTY} consecutive empty rounds — stopping.")
                break

    return domain_model


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