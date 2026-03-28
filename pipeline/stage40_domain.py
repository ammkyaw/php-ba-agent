"""
pipeline/stage40_domain.py — Domain Analyst Agent

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
If stage40_domain is COMPLETED and domain_model.json exists, the stage
is skipped and ctx.domain_model is restored from the saved file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, NamedTuple

from context import DomainModel, PipelineContext


class CallSpec(NamedTuple):
    """Specification for a single stage-4 LLM call."""
    group:          str        # "A" | "B" | "C"
    system_prompt:  str
    user_prompt:    str
    max_tokens:     int
    llm_label:      str        # telemetry / logging label
    parse_label:    str        # key used when parsing the response
    model_override: str | None

# ── Configuration ──────────────────────────────────────────────────────────────
# Three focused calls instead of one giant call — each produces a small, bounded
# JSON so local models (Qwen, Mistral, etc.) never hit output-token limits.
MAX_TOKENS_META     = 4_000   # Call A: domain_name, description, key_entities, bounded_contexts
MAX_TOKENS_FEATURES = 8_000   # Call B: features list (largest field)
MAX_TOKENS_ROLES_WF = 8_000   # Call C: user_roles + workflows
MAX_TOKENS_GAP_FILL = int(os.environ.get("STAGE4_MAX_TOKENS_GAP_FILL", "4000") or "4000")
# ↑ Kept deliberately lower than B/C calls: shorter responses loop less often on
#   quantised models (qwen3-coder int8), and multiple smaller gap-fill rounds
#   (MAX_GAP_ROUNDS=20) cover more ground than one large looping call.
DOMAIN_FILE      = "domain_model.json"
COVERAGE_FILE    = "coverage_report.json"

# Gap-fill: max modules per module-grouped call (covers ~10× more files than pages)
GAP_FILL_MAX_MODULES = 30
# Gap-fill: max individual files per call (fallback when no module structure)
GAP_FILL_MAX_PAGES = 100
# Gap-fill: maximum number of loop rounds.
# Override with STAGE4_MAX_GAP_ROUNDS=N (e.g. 5 for speed, 20 for thoroughness).
# Set STAGE4_SKIP_GAP_FILL=1 to bypass all D-calls (fastest, lower page coverage).
MAX_GAP_ROUNDS = int(os.environ.get("STAGE4_MAX_GAP_ROUNDS", "20") or "20")
_SKIP_GAP_FILL = os.environ.get("STAGE4_SKIP_GAP_FILL", "0").strip() == "1"

# Retrieval: queries × top-k chunks each
RETRIEVAL_QUERIES = [
    ("user roles authentication login registration logout session",   "auth"),
    ("database tables entities data model SQL queries",               "data"),
    ("business features workflows user actions operations",           "features"),
    ("form inputs POST fields user interactions pages submissions",   "forms"),
    ("navigation flow redirect pages include session check",          "nav"),
    ("core functions business logic processing validation",           "logic"),
]
TOP_K_PER_QUERY   = 5
MAX_TOTAL_CHUNKS  = 25    # hard cap on chunks sent to the LLM
MAX_CONTEXT_CHARS = 20_000  # soft cap on total context character length


def _env_int(name: str, default: int) -> int:
    """Read integer env var with a sane fallback."""
    return int(os.environ.get(name, str(default)) or str(default))


# Stage 4 prompt profiles: each sub-call now gets only the evidence family it
# actually needs instead of sharing one giant all-sections prompt.
_RETRIEVAL_PROFILE_QUERIES: dict[str, list[tuple[str, str]]] = {
    "A": [
        RETRIEVAL_QUERIES[0],  # auth
        RETRIEVAL_QUERIES[1],  # data
        RETRIEVAL_QUERIES[2],  # features
        RETRIEVAL_QUERIES[5],  # logic
    ],
    "B_CORE": [
        RETRIEVAL_QUERIES[1],  # data
        RETRIEVAL_QUERIES[2],  # features
        RETRIEVAL_QUERIES[3],  # forms
        RETRIEVAL_QUERIES[5],  # logic
    ],
    "B_UI": [
        RETRIEVAL_QUERIES[2],  # features
        RETRIEVAL_QUERIES[3],  # forms
        RETRIEVAL_QUERIES[4],  # nav
        RETRIEVAL_QUERIES[5],  # logic
    ],
    "C_ROLES": [
        RETRIEVAL_QUERIES[0],  # auth
        RETRIEVAL_QUERIES[2],  # features
        RETRIEVAL_QUERIES[4],  # nav
    ],
    "C_WORKFLOWS": [
        RETRIEVAL_QUERIES[0],  # auth
        RETRIEVAL_QUERIES[2],  # features
        RETRIEVAL_QUERIES[4],  # nav
        RETRIEVAL_QUERIES[5],  # logic
    ],
    "GAP": [
        RETRIEVAL_QUERIES[2],  # features
        RETRIEVAL_QUERIES[3],  # forms
        RETRIEVAL_QUERIES[4],  # nav
        RETRIEVAL_QUERIES[5],  # logic
    ],
}

_PROFILE_MAX_TOTAL_CHUNKS: dict[str, int] = {
    "A":           _env_int("STAGE4_A_MAX_TOTAL_CHUNKS", 12),
    "B_CORE":      _env_int("STAGE4_B_CORE_MAX_TOTAL_CHUNKS", 14),
    "B_UI":        _env_int("STAGE4_B_UI_MAX_TOTAL_CHUNKS", 14),
    "C_ROLES":     _env_int("STAGE4_C_ROLES_MAX_TOTAL_CHUNKS", 10),
    "C_WORKFLOWS": _env_int("STAGE4_C_WORKFLOWS_MAX_TOTAL_CHUNKS", 12),
    "GAP":         _env_int("STAGE4_GAP_MAX_TOTAL_CHUNKS", 12),
}

_PROFILE_MAX_CONTEXT_CHARS: dict[str, int] = {
    "A":           _env_int("STAGE4_A_CONTEXT_CHARS", 8_000),
    "B_CORE":      _env_int("STAGE4_B_CORE_CONTEXT_CHARS", 10_000),
    "B_UI":        _env_int("STAGE4_B_UI_CONTEXT_CHARS", 10_000),
    "C_ROLES":     _env_int("STAGE4_C_ROLES_CONTEXT_CHARS", 6_000),
    "C_WORKFLOWS": _env_int("STAGE4_C_WORKFLOWS_CONTEXT_CHARS", 8_000),
    "GAP":         _env_int("STAGE4_GAP_CONTEXT_CHARS", 8_000),
}

# ── Per-section caps inside _build_user_prompt ────────────────────────────────
# Large codebases (SuiteCRM, etc.) produce thousands of entries per section.
# These caps keep the prompt within a manageable token budget while still
# giving the LLM enough signal to identify patterns.
# Override any cap with env vars (set to 0 to disable a section entirely).
_CAP_EXEC_PATHS    = int(os.environ.get("STAGE4_CAP_EXEC_PATHS",    "40")  or "40")
_CAP_HTTP_EPS      = int(os.environ.get("STAGE4_CAP_HTTP_EPS",      "80")  or "80")
_CAP_TABLE_COLS    = int(os.environ.get("STAGE4_CAP_TABLE_COLS",     "60")  or "60")
_CAP_DB_TABLES     = int(os.environ.get("STAGE4_CAP_DB_TABLES",      "60")  or "60")
_CAP_TS_TYPES      = int(os.environ.get("STAGE4_CAP_TS_TYPES",       "60")  or "60")
_CAP_FUNCTIONS     = int(os.environ.get("STAGE4_CAP_FUNCTIONS",      "80")  or "80")
_CAP_CALL_GRAPH    = int(os.environ.get("STAGE4_CAP_CALL_GRAPH",     "40")  or "40")
_CAP_FORM_FILES    = int(os.environ.get("STAGE4_CAP_FORM_FILES",     "60")  or "60")
_CAP_REDIRECTS     = int(os.environ.get("STAGE4_CAP_REDIRECTS",      "60")  or "60")
_CAP_AUTH_SIGNALS  = int(os.environ.get("STAGE4_CAP_AUTH_SIGNALS",   "40")  or "40")
_CAP_COMPONENTS    = int(os.environ.get("STAGE4_CAP_COMPONENTS",     "60")  or "60")



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
    if ctx.is_stage_done("stage40_domain") and Path(output_path).exists():
        ctx.domain_model = _load_domain_model(output_path)
        print(f"  [stage4] Already completed — "
              f"domain='{ctx.domain_model.domain_name}', "
              f"{len(ctx.domain_model.features)} feature(s).")
        return

    # ── Language-aware preamble ───────────────────────────────────────────────
    global _ACTIVE_PREAMBLE
    _ACTIVE_PREAMBLE = _build_system_preamble(ctx)

    # ── Pre-flight ────────────────────────────────────────────────────────────
    _assert_prerequisites(ctx)

    print(f"  [stage4] Retrieving targeted codebase context from ChromaDB ...")
    quality_score = _get_quality_score(ctx)
    debug_dir = str(Path(output_path).parent)

    # ── Build grounding lists from code_map (anti-hallucination) ─────────────
    cm = ctx.code_map
    known_tables: list[str] = sorted({
        q.get("table", "") for q in (cm.sql_queries or [])
        if q.get("table") and q["table"] not in ("UNKNOWN", "")
    })

    # For TypeScript / serverless projects without SQL, treat type definition
    # names as the entity vocabulary — equivalent role to known_tables.
    known_type_names: list[str] = sorted({
        td.get("name", "") for td in (getattr(cm, "type_definitions", None) or [])
        if td.get("name")
    })
    # Merge into known_tables so _format_schema_grounding and anti-hallucination
    # filters accept both SQL table names and TS type names as valid entities.
    if known_type_names and not known_tables:
        known_tables = known_type_names

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
    # STAGE4_PAGES_CAP / STAGE4_TABLES_CAP: reduce for large codebases to avoid
    # filling the context window with grounding lists before the LLM can output.
    # SuiteCRM has 300+ pages — injecting all of them leaves almost no output budget.
    # Recommended: 100-150 for 32K context, 200+ for 64K+ context.
    _pages_cap  = int(os.environ.get("STAGE4_PAGES_CAP",  "150") or "150")
    _tables_cap = int(os.environ.get("STAGE4_TABLES_CAP", "40")  or "40")
    known_pages: list[str] = _build_grounding_pages(cm, cap=_pages_cap)

    prompt_profiles = ["A", "B_CORE", "B_UI", "C_ROLES", "C_WORKFLOWS", "GAP"]
    prompt_chunks: dict[str, list[dict[str, Any]]] = {}

    # ── Parallel retrieval — all 6 profiles are independent ChromaDB reads ────
    from pipeline.llm_client import get_max_workers as _max_w_early
    _ret_workers = _max_w_early()

    def _retrieve_profile(profile: str) -> tuple[str, list]:
        queries = _RETRIEVAL_PROFILE_QUERIES[profile]
        chunks  = _retrieve_context(
            ctx,
            queries          = queries,
            max_total_chunks = _PROFILE_MAX_TOTAL_CHUNKS[profile],
            max_context_chars= _PROFILE_MAX_CONTEXT_CHARS[profile],
        )
        return profile, chunks

    if _ret_workers > 1 and len(prompt_profiles) > 1:
        from concurrent.futures import ThreadPoolExecutor as _RetTPE
        with _RetTPE(max_workers=min(len(prompt_profiles), _ret_workers)) as _ret_pool:
            _ret_futs = [_ret_pool.submit(_retrieve_profile, p) for p in prompt_profiles]
            for fut in _ret_futs:
                _profile, _chunks = fut.result()
                prompt_chunks[_profile] = _chunks
    else:
        for _p in prompt_profiles:
            _, _chunks = _retrieve_profile(_p)
            prompt_chunks[_p] = _chunks

    for profile in prompt_profiles:
        _queries = _RETRIEVAL_PROFILE_QUERIES[profile]
        print(
            f"  [stage4]   {profile:<11} {len(prompt_chunks[profile])} chunk(s) "
            f"across {len(_queries)} query angles"
        )

    prompt_a = _build_user_prompt(ctx, prompt_chunks["A"], profile="A")
    prompt_b_core = _build_user_prompt(ctx, prompt_chunks["B_CORE"], profile="B_CORE")
    prompt_b_ui = _build_user_prompt(ctx, prompt_chunks["B_UI"], profile="B_UI")
    prompt_c_roles = _build_user_prompt(ctx, prompt_chunks["C_ROLES"], profile="C_ROLES")
    prompt_c_workflows = _build_user_prompt(ctx, prompt_chunks["C_WORKFLOWS"], profile="C_WORKFLOWS")
    gap_prompt = _build_user_prompt(ctx, prompt_chunks["GAP"], profile="GAP")

    # ── RAG augmentation ──────────────────────────────────────────────────────
    # Keep the RAG snippets targeted too; duplicating them onto every prompt
    # would push us back toward the same max-model-len problem.
    from pipeline.rag import CodeChunkIndex, is_enabled as _rag_enabled, get_top_k as _rag_top_k
    if _rag_enabled():
        _rag_idx = CodeChunkIndex(str(Path(output_path).parent))
        _rag_idx.build(ctx)
        _rag_ctx_a = _rag_idx.format_context(
            "domain entities data models user roles business objects database tables",
            top_k=_rag_top_k(),
            header="## RAG: Top relevant code chunks (entities / tables)",
        )
        _rag_ctx_features = _rag_idx.format_context(
            "business features use cases forms pages workflows user actions",
            top_k=_rag_top_k(),
            header="## RAG: Top relevant code chunks (features)",
        )
        _rag_ctx_workflows = _rag_idx.format_context(
            "authentication navigation redirects entry points workflows business operations",
            top_k=_rag_top_k(),
            header="## RAG: Top relevant code chunks (workflows / auth)",
        )
        _rag_idx.close()
        if _rag_ctx_a:
            prompt_a += "\n\n" + _rag_ctx_a
        if _rag_ctx_features:
            prompt_b_core += "\n\n" + _rag_ctx_features
            gap_prompt += "\n\n" + _rag_ctx_features
        if _rag_ctx_workflows:
            prompt_c_workflows += "\n\n" + _rag_ctx_workflows
        print(
            f"  [stage4] RAG augmentation: "
            f"+{len(_rag_ctx_a)} chars (A), "
            f"+{len(_rag_ctx_features)} chars (features), "
            f"+{len(_rag_ctx_workflows)} chars (workflows)"
        )

    for _name, _prompt in [
        ("A", prompt_a),
        ("B_CORE", prompt_b_core),
        ("B_UI", prompt_b_ui),
        ("C_ROLES", prompt_c_roles),
        ("C_WORKFLOWS", prompt_c_workflows),
        ("GAP", gap_prompt),
    ]:
        print(f"  [stage4] Prompt {_name:<11} {len(_prompt):,} chars "
              f"(~{len(_prompt) // 4:,} tokens)")

    from pipeline.llm_client import get_provider, get_model, get_max_workers as _max_w
    print(f"  [stage4] LLM: {get_provider()} / {get_model()} | quality={quality_score}")
    _workers = _max_w()

    # ── Self-consistency voting ───────────────────────────────────────────────
    # STAGE4_CONSISTENCY_RUNS=N runs Calls A and B N times independently and
    # merges results by taking the union of entities/features/contexts.
    # Default is 1 (off) to preserve speed; set to 3 for higher accuracy.
    _consistency_runs = max(1, int(os.environ.get("STAGE4_CONSISTENCY_RUNS", "1") or "1"))

    # ── Multi-model ensemble ──────────────────────────────────────────────────
    # LLM_ENSEMBLE_MODELS=model1,model2 runs each call once per model and
    # merges by union.  Combines orthogonal model strengths: one model may
    # catch entities the other misses.  Works with local (Ollama) models only.
    _raw_ensemble = os.environ.get("LLM_ENSEMBLE_MODELS", "").strip()
    _ensemble_models: list[str | None] = (
        [m.strip() for m in _raw_ensemble.split(",") if m.strip()]
        if _raw_ensemble else [None]
    )
    if len(_ensemble_models) > 1:
        print(f"  [stage4] Ensemble mode: {len(_ensemble_models)} models")

    # ── Calls A + B + C (narrow prompts per sub-task) ───────────────────────
    _sys_b = _system_features(quality_score,
                               known_tables=known_tables,
                               known_pages=known_pages,
                               known_fields=known_fields,
                               table_columns=getattr(cm, "table_columns", None) or [])
    _sys_a = _system_meta(quality_score)
    _sys_c_roles = _system_roles_only(quality_score)
    _sys_c_workflows = _system_workflows_only(quality_score)

    def _make_suffix(mdl_idx: int, run: int, total: int, letter: str) -> str:
        if total == 1:
            return letter
        if len(_ensemble_models) > 1:
            return f"{letter}{mdl_idx+1}.{run+1}"
        return f"{letter}{run+1}"

    _call_specs: list[CallSpec] = []
    _total_a = _consistency_runs * len(_ensemble_models)
    _total_b = _consistency_runs * len(_ensemble_models)
    for _mi, _mdl in enumerate(_ensemble_models):
        for _run in range(_consistency_runs):
            _suffix_a = _make_suffix(_mi, _run, _total_a, "A")
            _suffix_b = _make_suffix(_mi, _run, _total_b, "B")

            _call_specs.append(CallSpec(
                group="A", system_prompt=_sys_a, user_prompt=prompt_a,
                max_tokens=MAX_TOKENS_META,
                llm_label=f"stage4-{_suffix_a}", parse_label=_suffix_a,
                model_override=_mdl,
            ))
            _call_specs.append(CallSpec(
                group="B", system_prompt=_sys_b, user_prompt=prompt_b_core,
                max_tokens=MAX_TOKENS_FEATURES,
                llm_label=f"stage4-{_suffix_b}-core", parse_label=f"{_suffix_b}-core",
                model_override=_mdl,
            ))
            _call_specs.append(CallSpec(
                group="B", system_prompt=_sys_b, user_prompt=prompt_b_ui,
                max_tokens=MAX_TOKENS_FEATURES,
                llm_label=f"stage4-{_suffix_b}-ui", parse_label=f"{_suffix_b}-ui",
                model_override=_mdl,
            ))

    _call_specs.append(CallSpec(
        group="C", system_prompt=_sys_c_roles, user_prompt=prompt_c_roles,
        max_tokens=MAX_TOKENS_ROLES_WF,
        llm_label="stage4-C1", parse_label="C1", model_override=None,
    ))
    _call_specs.append(CallSpec(
        group="C", system_prompt=_sys_c_workflows, user_prompt=prompt_c_workflows,
        max_tokens=MAX_TOKENS_ROLES_WF,
        llm_label="stage4-C2", parse_label="C2", model_override=None,
    ))

    def _run_stage4_call(args: CallSpec) -> tuple[str, str, dict]:
        _group, _system, _user, _max_tokens, _llm_label, _parse_label, _mdl = args
        raw = _call_part(
            _system,
            _user,
            _max_tokens,
            _llm_label,
            model_override=_mdl,
        )
        return _group, _parse_label, _parse_partial(raw, _parse_label, debug_dir)

    if _workers > 1 and len(_call_specs) > 1:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        print(f"  [stage4] Running {len(_call_specs)} independent LLM call(s) "
              f"in parallel (workers={min(len(_call_specs), _workers)}) ...")
        with _TPE(max_workers=min(len(_call_specs), _workers)) as _pool:
            # Submit in _call_specs order so results are always collected in the
            # same order — _merge_consistency_b dedupes by "first seen wins", so
            # nondeterministic collection via as_completed() would make feature
            # content vary across runs.
            _futs = [_pool.submit(_run_stage4_call, spec) for spec in _call_specs]

            # Collect in submission order.  Gather ALL failures before raising so
            # the user sees every broken call at once rather than first-fail-fast.
            # Never substitute {} — a silent empty result silently under-populates
            # the domain model and cascades into degraded BRD/SRS/AC/User Stories.
            _call_results: list[tuple[str, str, dict]] = []
            _failures: list[str] = []
            for _fut, _spec in zip(_futs, _call_specs):
                _grp, _lbl = _spec.group, _spec.parse_label
                try:
                    _call_results.append(_fut.result())
                except Exception as _exc:
                    print(f"  [stage4] ✗ Call {_lbl} failed: {_exc}")
                    _failures.append(f"{_lbl}: {_exc}")

            if _failures:
                raise RuntimeError(
                    f"[stage4] {len(_failures)} LLM sub-call(s) failed — "
                    f"Stage 4 cannot produce a reliable domain model:\n"
                    + "\n".join(f"  • {f}" for f in _failures)
                )
    else:
        _call_results = [_run_stage4_call(spec) for spec in _call_specs]

    _a_results: list[dict] = []
    _b_results: list[dict] = []
    _c_results: dict[str, dict] = {}
    for _group, _parse_label, _parsed in _call_results:
        if _group == "A":
            _a_results.append(_parsed)
        elif _group == "B":
            _b_results.append(_parsed)
        else:
            _c_results[_parse_label] = _parsed

    data_a = _merge_consistency_a(_a_results)
    data_b = _merge_consistency_b(_b_results)
    data_b = _filter_hallucinated_refs(data_b,
                                       known_pages_lower=known_files_all,
                                       known_tables_lower={t.lower() for t in known_tables})
    data_c = {
        **_c_results.get("C1", {}),
        **_c_results.get("C2", {}),
    }

    # ── Merge all three dicts (A wins on overlaps, B/C fill in their fields) ──
    merged = {**data_a, **data_b, **data_c}

    # ── Hydrate DomainModel ───────────────────────────────────────────────────
    domain_model = _hydrate_domain_model(merged)

    # ── Coverage metrics ──────────────────────────────────────────────────────
    coverage_report = _compute_coverage(ctx, domain_model, debug_dir)

    # ── Gap-fill pass ─────────────────────────────────────────────────────────
    # Use the wide filter set for gap-fill too so new features can reference
    # any real file (not only html_pages).
    # Skip entirely when STAGE4_SKIP_GAP_FILL=1 (saves 15-30 min on large repos).
    if _SKIP_GAP_FILL:
        print(f"  [stage4] Gap-fill skipped (STAGE4_SKIP_GAP_FILL=1)")
    else:
        if MAX_GAP_ROUNDS < 20:
            print(f"  [stage4] Gap-fill capped at {MAX_GAP_ROUNDS} rounds "
                  f"(STAGE4_MAX_GAP_ROUNDS={MAX_GAP_ROUNDS})")
        domain_model = _gap_fill_pass(
            ctx, domain_model, coverage_report, gap_prompt, quality_score, debug_dir,
            known_tables=known_tables, known_pages_lower=known_files_all,
            known_fields=known_fields,
        )

    # ── Static table + field enrichment (no LLM) ──────────────────────────────
    # Derive table and field membership purely from the code_map data already
    # collected by Stage 1 parsers.  For every file a feature references in
    # its 'pages' array, look up which SQL tables / POST fields that file uses
    # and inject them into the feature's tables/inputs arrays.
    # This is free (no LLM calls) and typically adds 20-40 pp of table/field
    # coverage that the LLM missed because it only saw ~25 evidence chunks.
    _t_added, _f_added = _static_enrich_tables_and_fields(domain_model, cm)
    if _t_added or _f_added:
        print(f"  [stage4] Static enrichment: +{_t_added} table ref(s), "
              f"+{_f_added} field ref(s) derived from code_map")

    # ── Post-gap-fill coverage (shows improvement vs initial report) ───────────
    # Store result on ctx so stage58/stage59 can reference final coverage data
    # without re-running the computation.  Also writes coverage_report.json.
    print(f"  [stage4] Coverage after gap-fill + static enrichment:")
    ctx.domain_coverage = _compute_coverage(ctx, domain_model, debug_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_domain_model(domain_model, output_path)
    ctx.domain_model = domain_model
    ctx.stage("stage40_domain").mark_completed(output_path)
    ctx.save()

    print(f"  [stage4] Done — domain='{domain_model.domain_name}'")
    print(f"  [stage4]   Roles    : {[r['role'] for r in domain_model.user_roles]}")
    print(f"  [stage4]   Features : {[f['name'] for f in domain_model.features]}")
    print(f"  [stage4]   Entities : {domain_model.key_entities}")
    print(f"  [stage4]   Contexts : {domain_model.bounded_contexts}")
    print(f"  [stage4] Saved → {output_path}")


# ─── Self-Consistency Merge Helpers ───────────────────────────────────────────

def _merge_consistency_a(results: list[dict]) -> dict:
    """
    Merge multiple Call-A results (domain_name, key_entities, bounded_contexts).

    Strategy:
      - domain_name / description: take from the run that produced the most entities
      - key_entities: union across all runs, deduplicated case-insensitively
      - bounded_contexts: union across all runs, deduplicated case-insensitively
    """
    if len(results) == 1:
        return results[0]

    # Pick the base result (most entities → likely highest quality run)
    base = max(results, key=lambda r: len(r.get("key_entities") or []))
    merged = dict(base)

    # Union of entities
    seen_ent: set[str] = set()
    entities: list[str] = []
    for r in results:
        for e in (r.get("key_entities") or []):
            key = str(e).strip().lower()
            if key and key not in seen_ent:
                seen_ent.add(key)
                entities.append(e)
    merged["key_entities"] = entities

    # Union of bounded contexts
    seen_ctx: set[str] = set()
    contexts: list[str] = []
    for r in results:
        for c in (r.get("bounded_contexts") or []):
            key = str(c).strip().lower()
            if key and key not in seen_ctx:
                seen_ctx.add(key)
                contexts.append(c)
    merged["bounded_contexts"] = contexts

    if len(results) > 1:
        print(f"  [stage4] Consistency merge A: "
              f"{len(entities)} entities, {len(contexts)} contexts "
              f"(from {len(results)} runs)")
    return merged


def _merge_consistency_b(results: list[dict]) -> dict:
    """
    Merge multiple Call-B results (features list).

    Strategy: union of all features across runs, deduplicated by name
    (case-insensitive).  Features from later runs fill gaps not covered by
    earlier runs.
    """
    if len(results) == 1:
        return results[0]

    seen_names: set[str] = set()
    merged_features: list[dict] = []
    for r in results:
        for f in (r.get("features") or []):
            key = (f.get("name") or "").strip().lower()
            if key and key not in seen_names:
                seen_names.add(key)
                merged_features.append(f)

    print(f"  [stage4] Consistency merge B: "
          f"{len(merged_features)} unique features (from {len(results)} runs)")
    return {"features": merged_features}


# ─── Retrieval ─────────────────────────────────────────────────────────────────

def _retrieve_context(
    ctx: PipelineContext,
    queries: list[tuple[str, str]] | None = None,
    max_total_chunks: int = MAX_TOTAL_CHUNKS,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> list[dict[str, Any]]:
    """
    Fire all retrieval queries against ChromaDB and return a deduplicated,
    ranked list of the most relevant chunks.
    """
    from pipeline.stage30_embed import query_collection

    active_queries = queries or RETRIEVAL_QUERIES
    seen_ids: set[str]      = set()
    scored:   list[tuple]   = []   # (score, chunk_dict)

    for query_text, angle in active_queries:
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
    top_chunks = [chunk for _, chunk in scored[:max_total_chunks]]

    # Trim to the per-profile character budget
    total_chars = 0
    trimmed = []
    for chunk in top_chunks:
        chunk_len = len(chunk["text"])
        if total_chars + chunk_len > max_context_chars:
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


def _build_system_preamble(ctx) -> str:
    """Return a language-aware system preamble for the domain analyst LLM."""
    from context import Language  # avoid circular at module level  # noqa: PLC0415
    lang = Language.PHP
    fw   = "unknown"
    if ctx.code_map:
        lang = ctx.code_map.language
        fw   = ctx.code_map.framework.value

    _lang_desc = {
        Language.PHP:        f"legacy PHP ({fw}) applications",
        Language.TYPESCRIPT: f"TypeScript ({fw}) applications",
        Language.JAVASCRIPT: f"JavaScript ({fw}) applications",
        Language.JAVA:       f"Java ({fw}) applications",
        Language.KOTLIN:     f"Kotlin ({fw}) applications",
        Language.UNKNOWN:    "software applications",
    }
    desc = _lang_desc.get(lang, "software applications")
    return (
        f"You are a senior Business Analyst and software architect specialising in\n"
        f"reverse-engineering {desc} into structured business domain models.\n\n"
        f"Your task is to analyse the provided codebase evidence and extract specific\n"
        f"parts of the domain model in JSON format."
    )


# Set by run() to inject a language-aware preamble before calling prompt builders.
_ACTIVE_PREAMBLE: str = ""

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


def _get_preamble() -> str:
    """Return language-aware preamble if set, else the PHP default."""
    return _ACTIVE_PREAMBLE if _ACTIVE_PREAMBLE else _SYSTEM_PREAMBLE


def _system_meta(quality_score: int) -> str:
    """Call A — domain name, description, key entities, bounded contexts."""
    return f"""{_get_preamble()}
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


def _format_schema_grounding(
    known_tables: list[str],
    table_columns: list[dict],
) -> str:
    """
    Build a rich DB schema grounding block.

    When table_columns is available, emit full column info (name + type +
    nullable + FK) for each table.  Falls back to the flat table-name list
    when column data is absent.
    """
    # Index column data by table name (lower-cased for case-insensitive match).
    # Two formats:
    #   Format A — flat TableColumnEntry (Prisma/TS/Java): each entry IS one column
    #              {"table":"users", "column":"email", "nullable":False, ...}
    #   Format B — grouped DbSchemaEntry (PHP migrations): entry has a columns list
    #              {"table":"users", "columns":[{"name":"email", ...}], ...}
    col_index: dict[str, list[dict]] = {}
    for entry in (table_columns or []):
        tname = (entry.get("table") or "").strip().lower()
        if not tname:
            continue
        if entry.get("column"):
            # Format A: flat — synthesise a column dict matching Format B shape
            col_index.setdefault(tname, []).append({
                "name":     entry["column"],
                "type":     entry.get("type", ""),
                "nullable": entry.get("nullable", True),
                "default":  entry.get("default"),
            })
        else:
            # Format B: grouped — entry carries a columns list
            col_index.setdefault(tname, []).extend(entry.get("columns") or [])

    lines: list[str] = [
        "MANDATORY GROUNDING — DATABASE SCHEMA (actual tables and columns):"
    ]
    _tables_cap_grnd = int(os.environ.get("STAGE4_TABLES_CAP", "40") or "40")
    for table in known_tables[:_tables_cap_grnd]:
        cols = col_index.get(table.lower(), [])
        if cols:
            col_parts: list[str] = []
            for c in cols[:20]:              # cap to 20 cols per table
                cname = c.get("name", "?")
                ctype = c.get("type", "")
                nullable = " NULL" if c.get("nullable") else " NOT NULL"
                default  = f" DEFAULT {c['default']}" if c.get("default") not in (None, "") else ""
                col_parts.append(f"{cname} ({ctype}{nullable}{default})")
            lines.append(f"  {table}: {', '.join(col_parts)}")
        else:
            lines.append(f"  {table}")

    lines += [
        "You MUST use ONLY table names from the schema above in 'tables' arrays.",
        "Do NOT invent table names. Use the exact column names shown when writing business_rules.",
    ]
    return "\n".join(lines)


def _system_features(
    quality_score:  int,
    known_tables:   list[str] | None = None,
    known_pages:    list[str] | None = None,
    known_fields:   list[str] | None = None,
    table_columns:  list[dict] | None = None,
) -> str:
    """Call B — features list only.

    known_tables / known_pages / known_fields / table_columns: when provided,
    injected as mandatory grounding so the model cannot hallucinate names that
    don't exist in the actual codebase.  table_columns enriches table grounding
    with column names and types for more accurate business_rules generation.
    """
    # ── Build grounding block ────────────────────────────────────────────────
    grounding_parts: list[str] = []
    if known_tables:
        grounding_parts.append(
            _format_schema_grounding(known_tables, table_columns or [])
        )
    if known_pages:
        pages_str = ", ".join(known_pages[:250])     # cap to avoid token overflow
        grounding_parts.append(
            "MANDATORY GROUNDING — ACTUAL SOURCE FILES IN THIS CODEBASE:\n"
            f"{pages_str}\n"
            "You MUST use ONLY filenames from the list above in 'pages' arrays.\n"
            "Do NOT invent filenames that do not appear in this list."
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

    return f"""{_get_preamble()}
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
    return f"""{_get_preamble()}
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


def _system_roles_only(quality_score: int) -> str:
    """Call C1 — user roles only."""
    return f"""{_get_preamble()}
{_evidence_instruction(quality_score)}

Output ONLY this JSON (no other fields):
{{
  "user_roles": [
    {{
      "role": "Role name (e.g. Customer, Admin, Staff)",
      "description": "What this role can do in the system",
      "entry_points": ["login.php", "registration.php"]
    }}
  ]
}}

Now produce the JSON for user_roles only:"""


def _system_workflows_only(quality_score: int) -> str:
    """Call C2 — workflows only."""
    return f"""{_get_preamble()}
{_evidence_instruction(quality_score)}

Output ONLY this JSON (no other fields):
{{
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

Now produce the JSON for workflows only:"""


def _group_uncovered_by_module(
    uncovered_files: list[str],
    exec_paths: list[dict],
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Group uncovered file basenames by their logical module.

    Supports both SugarCRM-style modules/ directories (PHP) and
    Next.js / TypeScript route-group / feature-directory structure.

    Collision handling: when two files share the same basename (e.g. page.tsx
    in different route segments), both candidate module names are recorded and
    the basename is placed in the FIRST module seen — the collision avoidance
    logic in _build_module_expansion handles cross-module expansion separately.

    Returns
    -------
    module_groups : {module_name: [file1, file2, ...]}
        Only modules that have at least one uncovered file.
    flat_remainder : [file, ...]
        Files for which no module could be determined.
    """
    from collections import defaultdict

    # Build file-basename → module name map from exec_paths.
    # For collisions (same basename in multiple modules), keep the first module
    # assigned — gap-fill will retry remaining files in subsequent rounds.
    basename_to_module: dict[str, str] = {}
    for ep in exec_paths:
        f = ep.get("file", "")
        if not f:
            continue
        mod = _extract_module_name(f) or _extract_module_name_ts(f)
        if mod:
            basename = Path(f).name.lower()
            if basename not in basename_to_module:
                basename_to_module[basename] = mod

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
                     module_groups: dict[str, list[str]] | None = None,
                     language: str = "php") -> str:
    """Call D — gap-fill features for pages not yet covered by known features.

    When *module_groups* is supplied the prompt presents modules (with sample
    filenames) instead of bare filenames — one feature per module covers all
    sibling files through module-expansion and is far more token-efficient.
    """
    existing_str = ", ".join(f'"{n}"' for n in existing_feature_names)
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

    is_php  = language == "php"
    file_term  = "PHP files" if is_php else "source files"
    file_ext   = ".php"     if is_php else ""
    module_dir = "modules/<ModuleName>/" if is_php else "the application"

    if module_groups:
        # Module-grouped format: one feature per module
        module_lines = []
        for mod, files in module_groups.items():
            sample = ", ".join(files[:5])
            extra  = f" (+{len(files)-5} more)" if len(files) > 5 else ""
            module_lines.append(f'  - {mod}: {sample}{extra}')
        modules_str = "\n".join(module_lines)

        coverage_instruction = f"""The following application MODULES are NOT yet covered by any existing feature.
Each module is a group of {file_term} within {module_dir}:

{modules_str}

Existing features already extracted (do NOT duplicate): {existing_str}

Create ONE feature per module (or merge closely related small modules into one feature).
Each feature's "pages" array MUST include the representative filenames listed above."""
    else:
        pages_str = ", ".join(f'"{p}"' for p in uncovered_pages)
        coverage_instruction = f"""The following {file_term} are NOT yet covered by any existing feature:
{pages_str}

Existing features already extracted (do NOT duplicate): {existing_str}

For each uncovered file, determine what business feature it represents.
Group multiple uncovered files under one feature if they implement the same business function."""

    example_file = f"representative_file{file_ext}"
    return f"""{_get_preamble()}
{_evidence_instruction(quality_score)}{tables_grounding}

{coverage_instruction}

Output ONLY this JSON (no other fields):
{{
  "features": [
    {{
      "name": "Feature name",
      "description": "What this feature does from a business perspective",
      "pages": ["{example_file}"],
      "tables": ["relevant_table"],
      "inputs": ["field1", "field2"],
      "outputs": "What the user gets after completing this feature",
      "business_rules": []
    }}
  ]
}}

Now produce the JSON for new features covering the uncovered modules/pages:"""


def _rag_relevant_stems(chunks: list[dict]) -> set[str]:
    """
    Return the set of lowercase file stem names that appear in the top RAG chunks.
    Used to RAG-sort static sections so the most relevant files survive the cap.
    """
    stems: set[str] = set()
    for chunk in chunks:
        src = chunk.get("metadata", {}).get("source_file", "")
        if src:
            stems.add(Path(src).stem.lower())
    return stems


def _rag_sort(items: list, key_fn, relevant_stems: set[str]) -> list:
    """
    Stable-sort *items* so entries whose key_fn(item) stem appears in
    relevant_stems come first.  Order within each group is preserved.
    """
    return sorted(items, key=lambda x: (0 if Path(key_fn(x)).stem.lower() in relevant_stems else 1))


def _build_user_prompt(
    ctx: PipelineContext,
    chunks: list[dict],
    profile: str = "full",
) -> str:
    """
    Assemble the user prompt from retrieved chunks, CodeMap metadata,
    and graph summary.
    """
    profile = (profile or "full").upper()
    _is_full = profile in {"FULL", "ALL"}

    _include_db_tables     = _is_full or profile in {"B_CORE", "GAP"}
    _include_ts_types      = _is_full or profile in {"A", "B_CORE", "GAP"}
    _include_components    = _is_full or profile in {"B_UI", "GAP"}
    _include_form_inputs   = _is_full or profile in {"B_UI", "GAP", "C_WORKFLOWS"}
    _include_redirects     = _is_full or profile in {"B_UI", "GAP", "C_WORKFLOWS"}
    _include_pages         = _is_full or profile in {"B_UI", "GAP", "C_WORKFLOWS"}
    _include_functions     = _is_full
    _include_call_graph    = _is_full
    _include_form_fields   = _is_full or profile in {"B_UI", "GAP", "C_WORKFLOWS"}
    _include_service_deps  = _is_full
    _include_env_vars      = _is_full
    _include_sem_roles     = _is_full or profile in {"A", "C_ROLES", "C_WORKFLOWS"}
    _include_ext_systems   = _is_full or profile in {"A"}
    _include_db_schema     = _is_full or profile in {"B_CORE", "GAP"}
    _include_auth_signals  = _is_full or profile in {"C_ROLES", "C_WORKFLOWS", "GAP"}
    _include_http_eps      = _is_full or profile in {"B_UI", "GAP", "C_ROLES", "C_WORKFLOWS"}
    _include_table_cols    = _is_full or profile in {"A", "B_CORE", "GAP"}
    _include_exec_paths    = _is_full or profile in {"B_UI", "GAP", "C_WORKFLOWS"}
    _include_modules       = _is_full or profile in {"A"}
    _include_entities      = _is_full or profile in {"A", "B_CORE", "GAP"}
    _include_relationships = _is_full or profile in {"A", "B_CORE", "GAP"}
    _include_rules         = _is_full or profile in {"B_CORE", "GAP", "C_WORKFLOWS"}
    _include_state         = _is_full or profile in {"B_CORE", "GAP", "C_WORKFLOWS"}
    _include_graphrag      = _is_full or profile in {"B_CORE", "B_UI", "GAP", "C_ROLES", "C_WORKFLOWS"}

    if profile == "A":
        _cap_db_tables = min(_CAP_DB_TABLES, 12)
        _cap_ts_types = min(_CAP_TS_TYPES, 20)
        _cap_table_cols = min(_CAP_TABLE_COLS, 20)
        _cap_modules = 25
        _cap_entities = 20
        _cap_relationships = 20
        _cap_type_fields = 10
        _cap_graphrag_chars = 1_200
        _cap_table_refs = 4
    elif profile == "B_CORE":
        _cap_db_tables = min(_CAP_DB_TABLES, 20)
        _cap_ts_types = min(_CAP_TS_TYPES, 30)
        _cap_table_cols = min(_CAP_TABLE_COLS, 30)
        _cap_modules = 20
        _cap_entities = 25
        _cap_relationships = 25
        _cap_type_fields = 12
        _cap_graphrag_chars = 1_800
        _cap_table_refs = 5
    else:
        _cap_db_tables = _CAP_DB_TABLES if _is_full else min(_CAP_DB_TABLES, 40)
        _cap_ts_types = _CAP_TS_TYPES if _is_full else min(_CAP_TS_TYPES, 40)
        _cap_table_cols = _CAP_TABLE_COLS if _is_full else min(_CAP_TABLE_COLS, 40)
        _cap_modules = 40 if not _is_full else 999_999
        _cap_entities = 30 if not _is_full else 999_999
        _cap_relationships = 40 if not _is_full else 999_999
        _cap_type_fields = 12 if not _is_full else 999_999
        _cap_graphrag_chars = 2_000 if not _is_full else 10_000
        _cap_table_refs = 6 if not _is_full else 999_999

    _cap_components = _CAP_COMPONENTS if _is_full else min(_CAP_COMPONENTS, 40)
    _cap_form_files = _CAP_FORM_FILES if _is_full else min(_CAP_FORM_FILES, 40)
    _cap_redirects = _CAP_REDIRECTS if _is_full else min(_CAP_REDIRECTS, 40)
    _cap_functions = _CAP_FUNCTIONS if _is_full else min(_CAP_FUNCTIONS, 40)
    _cap_call_graph = _CAP_CALL_GRAPH if _is_full else min(_CAP_CALL_GRAPH, 25)
    _cap_auth_signals = _CAP_AUTH_SIGNALS if _is_full else min(_CAP_AUTH_SIGNALS, 30)
    _cap_http_eps = _CAP_HTTP_EPS if _is_full else min(_CAP_HTTP_EPS, 40)
    _cap_exec_paths = _CAP_EXEC_PATHS if _is_full else min(_CAP_EXEC_PATHS, 20)

    parts: list[str] = []
    _rel_stems = _rag_relevant_stems(chunks)   # files surfaced by RAG — prioritised in caps

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
    if _include_db_tables and unique_tables:
        _shown_tables = unique_tables[:_cap_db_tables] if _cap_db_tables > 0 else unique_tables
        _omitted_tbl  = len(unique_tables) - len(_shown_tables)
        parts.append(f"\n=== DATABASE TABLES ({len(unique_tables)} total"
                     + (f", showing {len(_shown_tables)})" if _omitted_tbl else ")"))
        parts.append(", ".join(_shown_tables)
                     + (f"  … +{_omitted_tbl} more omitted" if _omitted_tbl else ""))

        # Per-table: which files read/write it
        for table in _shown_tables:
            readers = sorted({q["file"] for q in cm.sql_queries
                              if q.get("table") == table and q.get("operation") == "SELECT"})
            writers = sorted({q["file"] for q in cm.sql_queries
                              if q.get("table") == table
                              and q.get("operation") in ("INSERT","UPDATE","DELETE","REPLACE")})
            ddl = sorted({q["file"] for q in cm.sql_queries
                          if q.get("table") == table
                          and q.get("operation") in ("CREATE","ALTER","DROP")})
            lines = [f"Table '{table}':"]
            if ddl:
                _ddl = ddl[:_cap_table_refs]
                _suffix = f" ... +{len(ddl) - len(_ddl)} more" if len(ddl) > len(_ddl) else ""
                lines.append(f"  Schema defined in: {', '.join(_ddl)}{_suffix}")
            if writers:
                _writers = writers[:_cap_table_refs]
                _suffix = f" ... +{len(writers) - len(_writers)} more" if len(writers) > len(_writers) else ""
                lines.append(f"  Written by: {', '.join(_writers)}{_suffix}")
            if readers:
                _readers = readers[:_cap_table_refs]
                _suffix = f" ... +{len(readers) - len(_readers)} more" if len(readers) > len(_readers) else ""
                lines.append(f"  Read by: {', '.join(_readers)}{_suffix}")
            parts.append("\n".join(lines))

    # ── TypeScript type definitions (interfaces + type aliases) ─────────────
    # This is the primary schema source of truth for serverless/NoSQL projects
    # that have no SQL tables or ORM models.  Always shown when present so the
    # LLM can ground entity names, field names, and relationships.
    _ts_types = getattr(cm, "type_definitions", None) or []
    if _include_ts_types and _ts_types:
        _shown_types = _ts_types[:_cap_ts_types] if _cap_ts_types > 0 else _ts_types
        _omitted_ty  = len(_ts_types) - len(_shown_types)
        parts.append(
            f"\n=== TYPESCRIPT TYPE DEFINITIONS ({len(_ts_types)} total"
            + (f", showing {len(_shown_types)})" if _omitted_ty else ")")
            + " — treat as the data schema ===" )
        for td in _shown_types:
            kind   = td.get("kind", "interface")
            name   = td.get("name", "?")
            src_f  = Path(td.get("file", "")).name
            fields = td.get("fields", [])
            if fields:
                field_strs = [
                    f"{f['name']}{'?' if f.get('optional') else ''}: {f.get('type','any')}"
                    for f in fields[:_cap_type_fields]
                ]
                if len(fields) > _cap_type_fields:
                    field_strs.append(f"... +{len(fields) - _cap_type_fields} more")
                parts.append(f"  {kind} {name}  [{src_f}]")
                parts.append(f"    fields: {', '.join(field_strs)}")
            else:
                parts.append(f"  {kind} {name}  [{src_f}]  (no fields extracted)")
        if _omitted_ty:
            parts.append(f"  … +{_omitted_ty} more type definitions omitted")

    # ── Component hierarchy (stage22) ────────────────────────────────────────
    # Feed page-level component trees to the LLM so it sees distinct features
    # (e.g. SprintPlanningTab, RiskRegisterTab) as separate documented features
    # rather than one undifferentiated "Dashboard" page.
    components = getattr(cm, "components", None) or []
    if _include_components and components:
        pages    = [c for c in components if c.get("is_page")]
        non_pages = [c for c in components if not c.get("is_page")]

        # Build a name→file lookup for quick child resolution
        _name_to_file: dict[str, str] = {
            c["name"]: c["file"] for c in components if c.get("name") and c.get("file")
        }

        _shown_comps = components[:_cap_components] if _cap_components > 0 else components
        _omit_comp   = len(components) - len(_shown_comps)

        parts.append(
            f"\n=== COMPONENT HIERARCHY ({len(components)} total"
            + (f", showing {len(_shown_comps)}" if _omit_comp else "")
            + f" | {len(pages)} page(s), {len(non_pages)} shared component(s)) ==="
        )
        parts.append(
            "These are the frontend React/Vue components extracted from Stage 2.2. "
            "Page-level components are labelled with their route. "
            "Each component's children list reveals sub-features visible on that page. "
            "Use this hierarchy to identify distinct functional areas and user workflows."
        )

        # Show pages first (they are the feature entry points)
        _page_shown = 0
        for comp in _shown_comps:
            if not comp.get("is_page"):
                continue
            _page_shown += 1
            name   = comp.get("name", "?")
            file_  = comp.get("file", "?")
            route  = comp.get("route", "")
            hooks  = comp.get("hooks", [])
            ch     = comp.get("children", [])
            props  = comp.get("props", [])

            route_str = f"  route={route}" if route else ""
            parts.append(f"\n  [PAGE] {name}  ({file_}){route_str}")
            if props:
                parts.append(f"    props: {', '.join(props[:8])}")
            if hooks:
                parts.append(f"    hooks: {', '.join(hooks[:6])}")
            if ch:
                # Prefer pre-resolved children_files dict (stage22 post-process);
                # fall back to runtime lookup for old component_graph.json files.
                # children_files values are list[str] — len>1 means a name collision.
                _cf = comp.get("children_files") or {}
                ch_resolved = []
                for child_name in ch[:12]:
                    paths = _cf.get(child_name)
                    if paths is None:
                        # old graph (str value) or runtime fallback
                        legacy = _name_to_file.get(child_name)
                        paths = [legacy] if legacy else None
                    elif isinstance(paths, str):
                        # old graph serialised as plain string — normalise to list
                        paths = [paths]
                    if paths:
                        if len(paths) == 1:
                            ch_resolved.append(f"{child_name} ({paths[0]})")
                        else:
                            # Ambiguous: two+ components share this name
                            joined = ", ".join(paths[:3])
                            ch_resolved.append(
                                f"{child_name} ({len(paths)} matches: {joined})"
                            )
                    else:
                        ch_resolved.append(child_name)
                parts.append(f"    children: {', '.join(ch_resolved)}")

        # Then shared/reusable components (condensed — one line each)
        _shared_lines = []
        for comp in _shown_comps:
            if comp.get("is_page"):
                continue
            name   = comp.get("name", "?")
            file_  = comp.get("file", "?")
            hooks  = comp.get("hooks", [])
            ch     = comp.get("children", [])
            line   = f"  [COMPONENT] {name}  ({file_})"
            if hooks:
                line += f"  hooks=[{', '.join(hooks[:4])}]"
            if ch:
                line += f"  children=[{', '.join(ch[:6])}]"
            _shared_lines.append(line)

        if _shared_lines:
            parts.append("\n  Shared / reusable components:")
            parts.extend(_shared_lines[:30])   # cap to avoid excessive prompt bloat
            if len(_shared_lines) > 30:
                parts.append(f"  … +{len(_shared_lines) - 30} more shared components omitted")

        if _omit_comp:
            parts.append(f"  … +{_omit_comp} more components omitted (raise STAGE4_CAP_COMPONENTS to include)")

    # ── Form inputs summary ───────────────────────────────────────────────────
    if _include_form_inputs and cm.superglobals:
        parts.append(f"\n=== FORM INPUTS (by page) ===")
        from collections import defaultdict
        sg_by_file: dict[str, list] = defaultdict(list)
        for sg in cm.superglobals:
            sg_by_file[sg["file"]].append(sg)
        _form_files = _rag_sort(list(sg_by_file.items()), lambda x: x[0], _rel_stems)
        if _cap_form_files > 0 and len(_form_files) > _cap_form_files:
            parts.append(f"  (showing {_cap_form_files}/{len(_form_files)} files — RAG-prioritised)")
            _form_files = _form_files[:_cap_form_files]
        for filepath, sgs in _form_files:
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
    if _include_redirects and cm.redirects:
        _redirects = _rag_sort(list(cm.redirects), lambda r: r.get("file", ""), _rel_stems)
        _omit_r = 0
        if _cap_redirects > 0 and len(_redirects) > _cap_redirects:
            _omit_r = len(_redirects) - _cap_redirects
            _redirects = _redirects[:_cap_redirects]
        parts.append(f"\n=== NAVIGATION / REDIRECTS"
                     + (f" ({len(cm.redirects)} total, showing {len(_redirects)})" if _omit_r else "") + " ===")
        for r in _redirects:
            filename = Path(r["file"]).name
            parts.append(f"{filename} → {r['target']}")
        if _omit_r:
            parts.append(f"  … +{_omit_r} more redirects omitted")

    # ── HTML pages (entry points) ─────────────────────────────────────────────
    if _include_pages and cm.html_pages:
        parts.append(f"\n=== PAGE ENTRY POINTS ===")
        parts.append(", ".join(Path(p).name for p in sorted(cm.html_pages)))

    # ── Functions ─────────────────────────────────────────────────────────────
    if _include_functions and cm.functions:
        _fns = cm.functions
        _omit_fn = 0
        if _cap_functions > 0 and len(_fns) > _cap_functions:
            _omit_fn = len(_fns) - _cap_functions
            _fns = _fns[:_cap_functions]
        parts.append(f"\n=== USER-DEFINED FUNCTIONS ({len(cm.functions)} total"
                     + (f", showing {len(_fns)})" if _omit_fn else ")"))
        for fn in _fns:
            params = [p["name"] for p in fn.get("params", [])]
            doc    = fn.get("docblock") or ""
            line   = f"  {fn['name']}({', '.join(params)})"
            if doc:
                line += f" — {doc}"
            parts.append(line)
        if _omit_fn:
            parts.append(f"  … +{_omit_fn} more functions omitted")

    # ── Call graph ────────────────────────────────────────────────────────────
    call_graph = getattr(cm, "call_graph", None) or []
    if _include_call_graph and call_graph:
        parts.append(f"\n=== FUNCTION CALL GRAPH ({len(call_graph)} edges) ===")
        # Group by file for readability
        from collections import defaultdict
        by_file: dict = defaultdict(list)
        for edge in call_graph:
            by_file[edge.get("file","?")].append(
                f"{edge.get('caller','?')} → {edge.get('callee','?')}"
            )
        _cg_files = _rag_sort(list(by_file.items()), lambda x: x[0], _rel_stems)
        _omit_cg = 0
        if _cap_call_graph > 0 and len(_cg_files) > _cap_call_graph:
            _omit_cg = len(_cg_files) - _cap_call_graph
            _cg_files = _cg_files[:_cap_call_graph]
        for fpath, edges in _cg_files:
            parts.append(f"{Path(fpath).name}: {', '.join(edges[:8])}"
                         + (" ..." if len(edges) > 8 else ""))
        if _omit_cg:
            parts.append(f"  … +{_omit_cg} more files omitted")

    # ── Form fields ───────────────────────────────────────────────────────────
    form_fields = getattr(cm, "form_fields", None) or []
    if _include_form_fields and form_fields:
        parts.append(f"\n=== HTML FORM FIELDS ===")
        for form in form_fields:
            fname  = Path(form.get("file","?")).name
            action = form.get("action","") or "(no action)"
            method = form.get("method","POST")
            fields = [f.get("name","?") for f in form.get("fields", []) if f.get("name")]
            parts.append(f"{fname} [{method} → {action}]: {', '.join(fields)}")

    # ── Service dependencies ──────────────────────────────────────────────────
    service_deps = getattr(cm, "service_deps", None) or []
    if _include_service_deps and service_deps:
        parts.append(f"\n=== SERVICE DEPENDENCIES (DI / Constructor Injection) ===")
        from collections import defaultdict
        deps_by_class: dict = defaultdict(list)
        for dep in service_deps:
            deps_by_class[dep.get("class","?")].append(dep.get("dep_class","?"))
        for cls, deps in sorted(deps_by_class.items()):
            parts.append(f"  {cls} depends on: {', '.join(deps)}")

    # ── Environment variables ─────────────────────────────────────────────────
    env_vars = getattr(cm, "env_vars", None) or []
    if _include_env_vars and env_vars:
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

    # ── Semantic actor roles (stage27) ───────────────────────────────────────
    # Pre-computed actor/role tags from SemanticRoleIndex — gives the LLM
    # grounded user-role names instead of having to guess from file names.
    # Also exposes detected external integrations (Stripe, email, S3, …).
    sr = getattr(ctx, "semantic_roles", None)
    if _include_sem_roles and sr:
        biz_actions = [t for t in (sr.actions or [])
                       if t.role == "BUSINESS_ACTION" and t.confidence >= 0.70]
        if biz_actions:
            parts.append(f"\n=== SEMANTIC ACTOR ROLES — STAGE 2.7 ===")
            parts.append(
                "These actors were pre-computed from route/controller semantic analysis. "
                "Use them to name user roles in bounded_contexts accurately."
            )
            from collections import defaultdict as _dd27
            _by_actor: dict = _dd27(list)
            for _t in biz_actions:
                _actor = _t.actor or "User"
                _by_actor[_actor].append(_t.symbol)
            for _actor, _syms in sorted(_by_actor.items()):
                parts.append(f"  Actor '{_actor}': {', '.join(_syms[:6])}"
                             + (" …" if len(_syms) > 6 else ""))
        if _include_ext_systems and sr.external_systems:
            parts.append(f"\n=== EXTERNAL SYSTEM INTEGRATIONS — STAGE 2.7 ===")
            parts.append(
                "These external systems were detected from env-var keys, class names, "
                "and service dependencies. Document their integration contracts."
            )
            for _ext in sr.external_systems:
                _env = f"  env_keys=[{', '.join(_ext.env_keys[:3])}]" if _ext.env_keys else ""
                parts.append(f"  {_ext.name} [{_ext.category}]{_env}")

    # ── DB schema / migrations (stage1) ──────────────────────────────────────
    # Migration history gives schema-evolution context: which tables were
    # created/altered and with what columns — useful for entity lifecycle.
    db_schema = getattr(cm, "db_schema", None) or []
    if _include_db_schema and db_schema:
        parts.append(f"\n=== DB SCHEMA / MIGRATIONS ({len(db_schema)} migration(s)) ===")
        parts.append("Use for entity lifecycle, column-level data model, and schema evolution.")
        _seen_mig: set = set()
        for _mig in db_schema[:30]:
            _op    = _mig.get("operation", "?")
            _tbl   = _mig.get("table", "?")
            _cols  = [c.get("name") or c if isinstance(c, str) else str(c)
                      for c in (_mig.get("columns") or [])[:8]]
            _key   = f"{_op}:{_tbl}"
            if _key in _seen_mig:
                continue
            _seen_mig.add(_key)
            _col_str = f"  cols=[{', '.join(_cols)}]" if _cols else ""
            parts.append(f"  {_op} {_tbl}{_col_str}")

    # ── Auth signals ──────────────────────────────────────────────────────────
    auth_signals = getattr(cm, "auth_signals", None) or []
    if _include_auth_signals and auth_signals:
        from collections import Counter as _Counter
        parts.append(f"\n=== AUTHENTICATION & AUTHORISATION SIGNALS ===")
        # PHP parser uses "type"; TypeScript parser uses "kind" — accept both
        def _sig_type(s: dict) -> str:
            return s.get("type") or s.get("kind") or "unknown"
        by_type: _Counter = _Counter(_sig_type(s) for s in auth_signals)
        parts.append("  Signal counts: " + ", ".join(
            f"{t}={n}" for t, n in sorted(by_type.items())))
        seen_pats: set = set()
        _auth_shown = 0
        for sig in auth_signals:
            if _cap_auth_signals > 0 and _auth_shown >= _cap_auth_signals:
                parts.append(f"  … +{len(auth_signals) - _auth_shown} more signals omitted")
                break
            pat = sig.get("pattern", "?")
            if pat in seen_pats:
                continue
            seen_pats.add(pat)
            _auth_shown += 1
            detail = f" [{sig['detail']}]" if sig.get("detail") else ""
            parts.append(f"  {_sig_type(sig)}: {pat}{detail}"
                         f" ({Path(sig.get('file','?')).name}:{sig.get('line','?')})")

    # ── HTTP entry points ─────────────────────────────────────────────────────
    http_endpoints = getattr(cm, "http_endpoints", None) or []
    if _include_http_eps and http_endpoints:
        _eps = http_endpoints
        _omit_ep = 0
        if _cap_http_eps > 0 and len(_eps) > _cap_http_eps:
            _omit_ep = len(_eps) - _cap_http_eps
            _eps = _eps[:_cap_http_eps]
        parts.append(f"\n=== HTTP ENTRY POINTS ({len(http_endpoints)} total"
                     + (f", showing {len(_eps)})" if _omit_ep else ")"))
        for ep in _eps:
            handler  = ep.get("handler") or Path(ep.get("file", "?")).name
            accepts  = "/".join(ep.get("accepts", []))
            produces = ep.get("produces", "?")
            ep_type  = ep.get("type", "page")
            parts.append(f"  [{ep_type}] {handler} — accepts {accepts}, produces {produces}")
        if _omit_ep:
            parts.append(f"  … +{_omit_ep} more endpoints omitted")

    # ── Table/column definitions ──────────────────────────────────────────────
    table_columns = getattr(cm, "table_columns", None) or []
    if _include_table_cols and table_columns:
        _tcols = table_columns
        _omit_tc = 0
        if _cap_table_cols > 0 and len(_tcols) > _cap_table_cols:
            _omit_tc = len(_tcols) - _cap_table_cols
            _tcols = _tcols[:_cap_table_cols]
        parts.append(f"\n=== DATABASE TABLES & COLUMNS ({len(table_columns)} tables"
                     + (f", showing {len(_tcols)})" if _omit_tc else ")"))
        for tbl in _tcols:
            tname   = tbl.get("table", "?")
            source  = tbl.get("source", "?")
            cols    = tbl.get("columns", [])
            col_str = ", ".join(c.get("name") or "?" for c in cols[:12] if isinstance(c, dict))
            if len(cols) > 12:
                col_str += f" ... +{len(cols) - 12} more"
            parts.append(f"  {tname} ({source}): {col_str}")
        if _omit_tc:
            parts.append(f"  … +{_omit_tc} more tables omitted")

    # ── Execution paths (stage15) ────────────────────────────────────────────
    exec_paths = getattr(cm, "execution_paths", None) or []
    if _include_exec_paths and exec_paths:
        _exps = _rag_sort(list(exec_paths), lambda ep: ep.get("file", ""), _rel_stems)
        _omit_exp = 0
        if _cap_exec_paths > 0 and len(_exps) > _cap_exec_paths:
            _omit_exp = len(_exps) - _cap_exec_paths
            _exps = _exps[:_cap_exec_paths]
        parts.append(f"\n=== EXECUTION PATHS & BRANCH ANALYSIS ({len(exec_paths)} files"
                     + (f", showing {len(_exps)})" if _omit_exp else ")"))
        parts.append(
            "Each entry below is derived from static analysis of a PHP file. "
            "Use this to identify workflows, auth requirements, and business rules."
        )
        for ep in _exps:
            fname = ep.get("file", "?")
            parts.append(f"\nFile: {fname}")

            # Auth guard
            ag = ep.get("auth_guard")
            if ag and ag.get("present", True):
                key = ag.get("key") or ag.get("guard") or "auth"
                parts.append(
                    f"  Auth guard: {key!r} required "
                    f"(else redirect → {ag.get('redirect','?')})"
                )

            # Entry conditions (PHP: dicts with "type"; TS: plain strings)
            for ec in ep.get("entry_conditions", []):
                if isinstance(ec, str):
                    parts.append(f"  Condition: {ec}")
                elif isinstance(ec, dict) and ec.get("type") == "method_check":
                    parts.append(f"  Accepts: HTTP {ec.get('method','?')}")

            # Happy path (primary success scenario)
            hp = ep.get("happy_path", [])
            if hp:
                parts.append("  Happy path:")
                for step in hp:
                    parts.append(f"    → {step}")

            # Data flows (PHP: "data_flows" list of dicts; TS: "data_flow")
            for flow in ep.get("data_flows", ep.get("data_flow", [])):
                if not isinstance(flow, dict):
                    continue
                fields = ", ".join(flow.get("field_mapping", {}).keys())
                table  = flow.get("table", "?")
                op     = flow.get("sink", "sql_query")
                if fields:
                    parts.append(
                        f"  Data flow: POST fields [{fields}] → {op} on `{table}`"
                    )

            # Branches summary (PHP: "branches" with then/else; TS: "branch_map" with outcome)
            branches = ep.get("branches", ep.get("branch_map", []))
            if branches:
                parts.append(f"  Branches ({len(branches)}):")
                for b in branches[:3]:          # cap at 3 per file
                    if not isinstance(b, dict):
                        continue
                    cond = b.get("condition","?")[:80]
                    # PHP format: then/else lists; TS format: outcome string
                    if "outcome" in b:
                        parts.append(f"    if ({cond}) → {b['outcome']}")
                    else:
                        then = [a.get("action","?") for a in b.get("then",[])]
                        els  = [a.get("action","?") for a in b.get("else",[])]
                        parts.append(f"    if ({cond})")
                        parts.append(f"      then: {', '.join(then) or 'none'}")
                        if els:
                            parts.append(f"      else: {', '.join(els)}")
        if _omit_exp:
            parts.append(f"\n  … +{_omit_exp} more execution-path files omitted")

    # ── Detected modules from Stage 2 Pass 15 ───────────────────────────────
    # Pre-seed the LLM with structurally-detected bounded contexts so it
    # refines evidence-based modules rather than inventing them from scratch.
    detected_modules = _modules_from_graph(ctx)
    if _include_modules and detected_modules:
        parts.append(f"\n=== STRUCTURALLY DETECTED MODULES ({len(detected_modules)}) ===")
        parts.append(
            "These modules were detected automatically from directory structure, "
            "namespaces, and call-graph community analysis. Use them as the basis "
            "for bounded_contexts — rename or merge if the code evidence warrants it, "
            "but do not drop modules that have significant node counts."
        )
        _mods = list(sorted(detected_modules.items()))[:_cap_modules]
        for mod_id, info in _mods:
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
        if len(detected_modules) > len(_mods):
            parts.append(f"  … +{len(detected_modules) - len(_mods)} more modules omitted")

    # ── Action Clusters from Stage 2.8 ───────────────────────────────────────
    # Provide the similarity-based cluster map as an alternative bounded-context
    # seed.  This is especially useful when the Stage 2 graph was built without
    # community detection (no detected_modules) or the graph is absent.
    ac = getattr(ctx, "action_clusters", None)
    if _include_modules and ac and ac.clusters and not detected_modules:
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

    # ── Entities from Stage 4.1 ──────────────────────────────────────────────
    ent_col = getattr(ctx, "entities", None)
    if _include_entities and ent_col and ent_col.entities:
        core_ents = [e for e in ent_col.entities if e.is_core and not e.is_system]
        if core_ents:
            parts.append(
                f"\n=== DETECTED ENTITIES — STAGE 4.1 ({len(core_ents)} core) ==="
            )
            parts.append(
                "These entities were extracted statically from the DB schema. "
                "Use them as your key_entities and as the basis for bounded_contexts."
            )
            _ents = core_ents[:_cap_entities]
            for ent in _ents:
                col_names = [c.name for c in ent.columns[:6]]
                pk_note   = f"  pk={ent.primary_key}" if ent.primary_key else ""
                pivot_note = "  [pivot]" if ent.is_pivot else ""
                parts.append(
                    f"  • {ent.name:<30} table={ent.table:<25} "
                    f"context={ent.bounded_context}{pk_note}{pivot_note}"
                    + (f"  cols=[{', '.join(col_names)}{'…' if len(ent.columns) > 6 else ''}]"
                       if col_names else "")
                )
            if len(core_ents) > len(_ents):
                parts.append(f"  … and {len(core_ents) - len(_ents)} more core entities")

    # ── Relationships from Stage 4.2 ─────────────────────────────────────────
    rel_col = getattr(ctx, "relationships", None)
    if _include_relationships and rel_col and rel_col.relationships:
        high_rel = [r for r in rel_col.relationships if r.confidence >= 0.75]
        if high_rel:
            parts.append(
                f"\n=== DETECTED RELATIONSHIPS — STAGE 4.2 "
                f"({len(high_rel)} high-confidence) ==="
            )
            parts.append(
                "These entity relationships were reconstructed from FK constraints, "
                "SQL JOINs, ORM declarations, and column naming patterns. "
                "Use them to define accurate cardinality in bounded_contexts and "
                "data models.  Do NOT invent relationships that contradict these."
            )
            _rels = high_rel[:_cap_relationships]
            for rel in _rels:
                parts.append(
                    f"  • {rel.from_entity:<25} {rel.cardinality:<5} {rel.to_entity:<25}"
                    f"  via={rel.via_column or rel.via_table or '?':<20}"
                    f"  [{','.join(rel.signals)}]"
                )
            if len(high_rel) > len(_rels):
                parts.append(f"  … and {len(high_rel) - len(_rels)} more")

    # ── Detected Business Rules from Stage 2.9 ───────────────────────────────
    # High-confidence rules (schema-enforced ≥ 0.9, guard-clause ≥ 0.75) are
    # injected as hard constraints.  The LLM must surface these in features,
    # workflows, and key_entities rather than inventing its own constraints.
    inv = getattr(ctx, "invariants", None)
    if _include_rules and inv and inv.rules:
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

    # ── State machines (Stage 4.3) ───────────────────────────────────────────
    sm_col = getattr(ctx, "state_machines", None)
    if _include_state and sm_col and sm_col.machines:
        parts.append(
            f"\n=== DETECTED STATE MACHINES — STAGE 4.3 "
            f"({sm_col.total} machine(s)) ==="
        )
        parts.append(
            "These entity lifecycles were reconstructed statically from DB and PHP. "
            "Use them to define entity states in bounded_contexts and workflows."
        )
        for sm in sm_col.machines[:10]:   # cap at 10 to keep prompt size sane
            states_str = " → ".join(sm.initial_states + [
                s for s in sm.states
                if s not in sm.initial_states and s not in sm.terminal_states
            ] + sm.terminal_states) or ", ".join(sm.states)
            parts.append(
                f"  • {sm.entity}.{sm.field} [{sm.bounded_context}]: {states_str}"
            )
            if sm.dead_states:
                parts.append(f"    (dead/unreachable: {', '.join(sm.dead_states)})")

    # ── GraphRAG semantic context ─────────────────────────────────────────────
    # ctx.graph_query() uses the community-graph-aware index built by stage38.
    # Inject it so the LLM can validate entity relationships and cross-module
    # operations that are not obvious from directory structure alone.
    # Previously computed but never used in LLM prompts — wiring it here
    # prevents the stage38 computation from being wasted.
    if _include_graphrag and hasattr(ctx, "graph_query"):
        _grag_topics = [
            "user roles permissions authentication",
            "business features workflows operations",
        ]
        _grag_hits: list[str] = []
        for _topic in _grag_topics:
            try:
                _gc = ctx.graph_query(_topic)
            except Exception:
                _gc = None
            if _gc and _gc.strip():
                _gc_txt = _gc.strip()
                if len(_gc_txt) > _cap_graphrag_chars:
                    _gc_txt = _gc_txt[:_cap_graphrag_chars].rstrip() + " ..."
                _grag_hits.append(_gc_txt)
        if _grag_hits:
            parts.append(f"\n=== GRAPH-AWARE SEMANTIC CONTEXT ===")
            parts.append(
                "The following was retrieved via graph-community-aware semantic search "
                "(GraphRAG). Use it to validate entity relationships and identify "
                "cross-module business operations not apparent from file structure alone."
            )
            parts.extend(_grag_hits)

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
               max_tokens: int, label: str,
               model_override: str | None = None) -> str:
    """Call the configured LLM and return the raw response string.

    json_mode=True forces local models (Ollama etc.) to output valid JSON:
      - OpenAI-compat path:   response_format={"type":"json_object"}
      - Ollama native path:   format="json"  (top-level payload field)

    prefill="{" is a secondary safety net for thinking models (Qwen3,
    DeepSeek-R1): the assistant turn starts with "{" so the model cannot
    emit any preamble or markdown fence before the JSON object.
    On Claude the prefill is sent as a partial assistant message; on local
    models it is appended to the messages array the same way.
    """
    from pipeline.llm_client import call_llm, get_provider
    # Use prefill only for local models — Claude handles structured output
    # differently and prefill interferes with its extended-thinking mode.
    use_prefill = "{" if get_provider() == "local" else ""
    raw = call_llm(
        system_prompt  = system_prompt,
        user_prompt    = user_prompt,
        max_tokens     = max_tokens,
        temperature    = 0.2,   # extraction: recall > precision, keep generous default
        label          = label,
        json_mode      = True,
        prefill        = use_prefill,
        model_override = model_override,
    )
    # vLLM's OpenAI-compat endpoint returns only the model's *continuation*,
    # not the prefill text itself.  Re-attach "{" so _parse_partial always
    # receives a complete JSON document ({"features": [...]} not "features": [...]).
    # Ollama native echoes the prefill, so we guard with startswith to avoid
    # double-prepending.
    if use_prefill and raw and not raw.lstrip().startswith(use_prefill):
        raw = use_prefill + raw
    return raw


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


# Directories that are pure infrastructure in TypeScript projects
_TS_INFRA_DIRS: frozenset[str] = frozenset({
    "src", "app", "pages", "api", "lib", "utils", "helpers", "types",
    "hooks", "styles", "public", "config", ".", "", "node_modules",
    "dist", "build", ".next",
})
# Generic component sub-folders that don't represent a domain concept
_TS_GENERIC_COMPONENTS: frozenset[str] = frozenset({
    "common", "shared", "base", "core", "ui", "layout", "primitives",
})


def _extract_module_name_ts(filepath: str) -> str | None:
    """
    Extract a logical module name from a TypeScript / Next.js path.

    Priority
    --------
    1. Route groups  : app/(auth)/login/page.tsx     → "auth"
    2. API segment   : app/api/users/route.ts        → "users_api"
    3. App segment   : app/dashboard/page.tsx        → "dashboard"
    4. Feature dir   : src/features/payments/svc.ts  → "payments"
    5. Component grp : components/UserProfile/Card   → "UserProfile"

    Returns None for infra files (lib/, utils/, hooks/, config/ …).
    """
    parts = [p for p in Path(filepath).parts if p not in (".", "..")]

    # Pass 1 — route groups take highest priority
    for part in parts:
        m = re.match(r'^\((\w+)\)$', part)
        if m:
            return m.group(1)

    # Pass 2 — structural keywords
    for i, part in enumerate(parts):
        part_l = part.lower()

        # app/api/<module>/  (avoid dynamic segments [id])
        if part_l == "api" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if not nxt.startswith("[") and nxt.lower() not in _TS_INFRA_DIRS:
                return f"{nxt}_api"

        # app/<segment>/  – first meaningful segment directly under app/
        if part_l == "app" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if (not re.match(r'^[\[\(]', nxt)
                    and nxt.lower() not in _TS_INFRA_DIRS
                    and nxt.lower() != "api"):
                return nxt

        # src/features/<module>/
        if part_l == "features" and i + 1 < len(parts):
            return parts[i + 1]

        # components/<Group>/ or ui/<Group>/
        if part_l in ("components", "ui") and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt.lower() not in _TS_GENERIC_COMPONENTS | _TS_INFRA_DIRS:
                return nxt

    return None


def _static_enrich_tables_and_fields(
    domain_model: "DomainModel",
    cm: Any,
) -> tuple[int, int]:
    """
    Static (no-LLM) enrichment pass: derive table and field coverage directly
    from the code_map data that was already collected by Stage 1 parsers.

    Logic
    -----
    Tables  — for every file that a feature already references in its 'pages'
              array, look up which SQL tables that file queries (from
              cm.sql_queries) and add any missing tables to the feature.
              Also uses cm.table_columns to add Prisma/ORM model names for
              TypeScript projects.

    Fields  — same idea: look up which $_POST / input_params fields each
              covered file uses (from cm.superglobals + cm.input_params) and
              add them to the feature's 'inputs' array.

    Returns (tables_added, fields_added) counts for logging.
    """
    from collections import defaultdict

    # ── Build file-basename → tables index ────────────────────────────────────
    file_to_tables: dict[str, set[str]] = defaultdict(set)
    for q in (cm.sql_queries or []):
        f = q.get("file", "")
        t = q.get("table", "")
        if f and t and t.upper() not in ("", "UNKNOWN"):
            file_to_tables[Path(f).name.lower()].add(t.lower())
    # Also index table_columns (flat Prisma/TS format)
    for tc in (cm.table_columns or []):
        f = tc.get("file", "")
        t = tc.get("table", "")
        if f and t:
            file_to_tables[Path(f).name.lower()].add(t.lower())

    # ── Build file-basename → fields index ────────────────────────────────────
    _cm_lang = getattr(cm.language, "value", str(cm.language)).lower()
    _is_ts_cm = _cm_lang in ("typescript", "javascript")

    file_to_fields: dict[str, set[str]] = defaultdict(set)
    # PHP superglobals — $_POST only (domain inputs); $_GET is usually pagination/filter
    for s in (cm.superglobals or []):
        f = s.get("file", "")
        k = s.get("key", "")
        if f and k and s.get("var") in ("$_POST", "$_REQUEST"):
            file_to_fields[Path(f).name.lower()].add(k.lower())
    # TypeScript / Java input_params — body only; query/route params are infrastructure.
    # PHP already has its fields from the superglobals loop above; running input_params
    # for PHP would double-add those keys AND add $_SERVER/$_SESSION/$_GET keys as fields.
    if _is_ts_cm:
        for ip in (cm.input_params or []):
            f = ip.get("file", "")
            k = ip.get("name", "") or ip.get("key", "")
            if not f or not k:
                continue
            if ip.get("source", "body") != "body":
                continue   # skip req.query.* and req.params.*
            file_to_fields[Path(f).name.lower()].add(k.lower())

    # ── Build module → file-basenames index (used for module-expanded enrichment) ──
    # When a feature references one file in a module, enrich it with data from ALL
    # sibling files in that module.  This recovers fields/tables that the LLM omitted
    # because its evidence window only contained one or two representative files.
    _module_to_basenames: dict[str, list[str]] = defaultdict(list)
    _basename_to_module: dict[str, str] = {}
    for ep in (cm.execution_paths or []):
        _f = ep.get("file", "")
        if not _f:
            continue
        _mod = _extract_module_name(_f) or _extract_module_name_ts(_f)
        if _mod:
            _bn = Path(_f).name.lower()
            _module_to_basenames[_mod].append(_bn)
            if _bn not in _basename_to_module:
                _basename_to_module[_bn] = _mod

    tables_added = 0
    fields_added = 0

    for feat in domain_model.features:
        existing_tables = {t.lower() for t in feat.get("tables", [])}
        existing_fields = {inp.lower() for inp in feat.get("inputs", [])}

        # Collect explicit page basenames, then expand to module siblings
        explicit_basenames: set[str] = {
            Path(page).name.lower() for page in feat.get("pages", [])
        }
        expanded_basenames: set[str] = set(explicit_basenames)
        for bn in explicit_basenames:
            mod = _basename_to_module.get(bn)
            if mod:
                expanded_basenames.update(_module_to_basenames[mod])

        for basename in expanded_basenames:
            for tbl in file_to_tables.get(basename, set()):
                if tbl not in existing_tables:
                    feat.setdefault("tables", []).append(tbl)
                    existing_tables.add(tbl)
                    tables_added += 1

            for fld in file_to_fields.get(basename, set()):
                if fld not in existing_fields:
                    feat.setdefault("inputs", []).append(fld)
                    existing_fields.add(fld)
                    fields_added += 1

    return tables_added, fields_added


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
        mod = _extract_module_name(f) or _extract_module_name_ts(f)
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

    # ── Table coverage (language-aware) ───────────────────────────────────────
    _lang = getattr(cm.language, "value", str(cm.language)).lower()
    _is_ts = _lang in ("typescript", "javascript")

    # SQL tables from queries (all languages)
    _sql_tables: set[str] = {
        q.get("table", "") for q in (cm.sql_queries or [])
        if q.get("table") and q["table"] not in ("UNKNOWN", "")
    }
    if _is_ts and not _sql_tables:
        # TypeScript (Prisma/Firebase) — use type_definition model names as entities
        _sql_tables = {
            td.get("name", "") for td in (getattr(cm, "type_definitions", None) or [])
            if td.get("name")
        }
    all_tables = sorted(_sql_tables)
    tables_covered   = [t for t in all_tables if t.lower() in covered_tables]
    tables_uncovered = [t for t in all_tables if t.lower() not in covered_tables]
    table_cov        = len(tables_covered) / len(all_tables) if all_tables else 1.0

    # ── Field coverage (language-aware) ──────────────────────────────────────
    # PHP: $_POST fields are domain inputs; $_GET is mostly pagination/filter
    _php_fields: set[str] = {
        s["key"] for s in (cm.superglobals or [])
        if s.get("var") in ("$_POST", "$_REQUEST") and s.get("key")
    }
    # TypeScript: only req.body.* fields are domain inputs.
    # req.query.* (page, limit, cursor, sort, filter) and req.params.* (id, slug)
    # are infrastructure — they inflate the denominator without being LLM-coverable.
    # Guard with _is_ts: PHP populates cm.input_params with all superglobal data
    # (no "source" field), so the default "body" would pull in every $_SERVER,
    # $_SESSION, and $_GET key — inflating the PHP denominator by ~25%.
    _ts_fields: set[str] = (
        {
            (ip.get("name", "") or ip.get("key", ""))
            for ip in (cm.input_params or [])
            if (ip.get("name") or ip.get("key"))
            and ip.get("source", "body") == "body"
        }
        if _is_ts else set()
    )
    # Merge both so a mixed PHP+TS project gets full coverage
    all_fields = sorted(_php_fields | _ts_fields)
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
    _lang = getattr(cm.language, "value", str(cm.language)).lower()

    # Track which modules have already been sent to the LLM so we don't
    # spin indefinitely on modules the model refuses to list page refs for.
    attempted_modules: set[str] = set()
    # Track which non-module files have already been sent to avoid re-sending.
    attempted_flat: set[str] = set()
    # Allow up to this many consecutive empty/failed LLM rounds before giving up.
    # Set higher than 3 so repetition-loop failures on quantised models
    # (which produce empty dicts) don't abort gap-fill prematurely.
    MAX_CONSECUTIVE_EMPTY = int(os.environ.get("STAGE4_MAX_CONSECUTIVE_EMPTY", "5") or "5")
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
                module_groups=batch_modules, language=_lang,
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
                language=_lang,
            )
        else:
            print(f"  [stage4] No fresh modules or flat files remain — done.")
            break
        # ── Per-round targeted evidence ────────────────────────────────────────
        # Build a fresh user-prompt by querying ChromaDB with module/file names
        # as search terms.  The static gap_prompt (built before the loop) only
        # contains the 12 highest-overall chunks; by round 3+ those are rarely
        # relevant to the specific modules being asked about.
        if fresh_module_groups:
            _gap_q = " ".join(list(batch_modules.keys())[:6]) + " business logic features"
        else:
            _gap_q = " ".join((gap_pages or [])[:6]) + " business logic features"
        try:
            _fresh_chunks = _retrieve_context(
                ctx,
                queries           = [(_gap_q, "gap_round")],
                max_total_chunks  = _PROFILE_MAX_TOTAL_CHUNKS["GAP"],
                max_context_chars = _PROFILE_MAX_CONTEXT_CHARS["GAP"],
            )
            round_user_prompt = (
                _build_user_prompt(ctx, _fresh_chunks, profile="GAP")
                if _fresh_chunks else user_prompt
            )
        except Exception:
            round_user_prompt = user_prompt   # fall back to static prompt on error

        raw_d  = _call_part(gap_system, round_user_prompt, MAX_TOKENS_GAP_FILL, call_label)
        data_d = _parse_partial(raw_d, f"D{gap_round}", debug_dir)

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
    Recover a partial/truncated JSON response.

    Strategy: collect all positions of '}' that are outside string literals,
    then — starting from the rightmost — try to close the JSON from that point.
    The first (rightmost) candidate that produces valid JSON is returned.

    This is O(N + K·N) where K ≤ _MAX_CANDIDATES = 30, effectively O(N).
    String-aware scanning skips { } [ ] characters inside quoted values,
    which the old naive brace-count approach confused with structural tokens.
    """
    _MAX_CANDIDATES = 30   # try at most this many } positions from the right

    t = text.strip()
    if not t:
        return None

    # ── Collect all out-of-string '}' positions (forward scan, O(N)) ─────────
    close_positions: list[int] = []
    in_string   = False
    escape_next = False

    for i, ch in enumerate(t):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "}":
            close_positions.append(i)

    if not close_positions:
        return None

    # ── Try candidates from rightmost backwards ───────────────────────────────
    def _count_open(s: str) -> tuple[int, int]:
        """String-aware count of unclosed { [ in s. Returns (braces, brackets)."""
        ob = obr = 0
        in_string = escape_next = False
        for c in s:
            if escape_next:        escape_next = False; continue
            if c == "\\" and in_string: escape_next = True; continue
            if c == '"':           in_string = not in_string; continue
            if in_string:          continue
            if   c == "{": ob  += 1
            elif c == "}": ob  -= 1
            elif c == "[": obr += 1
            elif c == "]": obr -= 1
        return ob, obr

    for pos in reversed(close_positions[-_MAX_CANDIDATES:]):
        candidate = t[: pos + 1].rstrip().rstrip(",")
        ob, obr = _count_open(candidate)
        if ob < 0 or obr < 0:
            continue
        closed = candidate + ("]" * obr) + ("}" * ob)
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

    # raw_decode stops at the first complete JSON object — handles "Extra data"
    # errors when the model emits a second object or trailing thinking tokens.
    try:
        data, _ = json.JSONDecoder().raw_decode(text)
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
    _label_upper = label.upper()
    wrap_key = None
    if _label_upper.startswith("B") or _label_upper.startswith("D"):
        wrap_key = "features"
    elif _label_upper in {"C", "C1", "C_ROLES"}:
        wrap_key = "user_roles"
    elif _label_upper in {"C2", "C_WORKFLOWS"}:
        wrap_key = "workflows"
    if isinstance(data, list):
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
