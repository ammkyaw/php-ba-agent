"""
pipeline/stage39_preflight.py — Context Pre-flight Checks

Runs a battery of fast, deterministic checks on the pipeline context
(CodeMap + GraphMeta + EmbeddingMeta + static analysis outputs from
Stages 3.5–3.8) BEFORE the expensive Claude API calls in Stage 4+.
Acts as a comprehensive pre-LLM safety gate to catch bad inputs early.

Two outcomes:
    PASS  — ctx.preflight.passed = True,  pipeline continues to Stage 4
    BLOCK — ctx.preflight.passed = False, pipeline halts with clear diagnosis

Check categories
----------------
HARD BLOCKERS (preflight.passed = False — halt pipeline):
    B1. code_map is None or has zero PHP files
    B2. No SQL queries found AND no DB tables in graph
        → would produce a BA doc with no data model
    B3. embedding_meta is None or ChromaDB collection is empty
    B4. ChromaDB collection returns zero results for a basic sanity query
    B5. graph_meta is None or graph has zero nodes

WARNINGS (preflight.passed stays True — pipeline continues with caveats):
    W1. Fewer than 3 unique DB tables found
        → BA docs will have a thin data model section
    W2. No form inputs ($POST/$GET) found anywhere
        → SRS functional requirements will be sparse
    W3. No redirect patterns found
        → Navigation flow section will be empty
    W4. No user-defined functions found (fully inline procedural)
        → Function-level BA coverage will be absent
    W5. Only one chunk type present in the collection
        → Retrieval diversity will be low
    W6. CodeMap has parse errors recorded
        → Some files may have been partially parsed
    W7. PHP version < 7.0 detected
        → Modern PHP patterns may not apply
    W8. graph_rag_meta is None or graphrag index is empty (Stage 3.8)
        → Stage 4 retrieval will fall back to raw ChromaDB only
    W9. ctx.entities is None or zero entities extracted (Stage 3.5)
        → Domain model will lack structured entity grounding
    W10. ctx.relationships is None or zero relationships mapped (Stage 3.6)
        → ER diagram and relationship context will be absent from Stage 4
    W11. ctx.state_machines is None or zero machines found (Stage 3.7)
        → State lifecycle section will be absent (may be expected for simple apps)

Signal quality score
--------------------
After all checks, a 0–100 score is computed and stored in the report.
Stage 4 uses this to calibrate how much context to retrieve and
how confidently to phrase the generated BA documents.

    90–100  Excellent — full SQL, forms, auth, graph all present
    70–89   Good      — most signals present, minor gaps
    50–69   Fair      — usable but some doc sections will be thin
    < 50    Poor      — BA docs will be generic; consider re-parsing

Resume behaviour
----------------
If stage39_preflight is already COMPLETED and preflight_report.json exists,
the stage is skipped and ctx.preflight is restored from the report.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from context import PipelineContext, PreflightResult


# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_TABLES_WARN          = 3      # warn if fewer unique DB tables
MIN_CHUNKS               = 5      # block if ChromaDB has fewer chunks than this
SANITY_QUERY_MAX_DIST    = 0.95   # block if best match distance > this (0=identical, 1=unrelated)
SCORE_WEIGHTS = {
    "has_sql":          20,   # SQL queries found
    "has_tables":       15,   # DB table nodes in graph
    "has_forms":        10,   # $_POST/$_GET reads
    "has_auth":         10,   # session/redirect patterns
    "has_functions":    10,   # user-defined functions
    "has_graph":         5,   # knowledge graph built
    "has_embeddings":    5,   # ChromaDB populated
    "has_graphrag":     10,   # graph-aware context index built (Stage 3.8)
    "has_entities":      5,   # entities extracted (Stage 3.5)
    "has_relationships": 5,   # relationships mapped (Stage 3.6)
    "has_statemachines": 5,   # state machines found (Stage 3.7)
}


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 3.9 entry point. Runs all preflight checks and populates
    ctx.preflight. Raises PreflightBlocker if any hard blocker is found.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        PreflightBlocker: If a hard blocker is found. Message includes
                          which check failed and how to fix it.
    """
    report_path = ctx.output_path("preflight_report.json")

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage39_preflight") and Path(report_path).exists():
        ctx.preflight = _restore_preflight(report_path)
        print(f"  [stage39] Already completed — "
              f"{'PASSED' if ctx.preflight.passed else 'FAILED'}, "
              f"{len(ctx.preflight.warnings)} warning(s).")
        return

    print("  [stage39] Running preflight checks ...")

    blockers: list[str] = []
    warnings: list[str] = []
    signals:  dict[str, bool] = {k: False for k in SCORE_WEIGHTS}
    details:  dict[str, Any]  = {}

    # ── Hard Blockers ─────────────────────────────────────────────────────────

    # B1: CodeMap present and non-empty
    if ctx.code_map is None:
        blockers.append(
            "B1: ctx.code_map is None — Stage 1 (parse) must complete before preflight."
        )
    else:
        if ctx.code_map.total_files == 0:
            blockers.append(
                "B1: CodeMap reports 0 source files parsed. "
                "Re-run Stage 1 or check the project path."
            )
        details["total_files"]    = ctx.code_map.total_files
        details["total_lines"]    = ctx.code_map.total_lines
        details["framework"]      = ctx.code_map.framework.value
        details["php_version"]    = ctx.code_map.php_version
        details["sql_query_count"]= len(ctx.code_map.sql_queries)
        details["function_count"] = len(ctx.code_map.functions)
        details["include_count"]  = len(ctx.code_map.includes)
        details["redirect_count"] = len(ctx.code_map.redirects)
        details["superglobal_count"] = len(ctx.code_map.superglobals)
        details["parse_errors"]   = getattr(ctx.code_map, "errors", [])

    # B2: Some DB signal must exist
    # Laravel uses Eloquent ORM — raw SQL is rare. Collect table names from
    # four sources and use the union set for accurate table_count.
    sql_count = len(ctx.code_map.sql_queries) if ctx.code_map else 0
    unique_tables: set[str] = set()

    if ctx.code_map:
        # Source 1: raw SQL query strings (works for procedural PHP)
        for q in ctx.code_map.sql_queries:
            t = q.get("table", "")
            if t and t != "UNKNOWN":
                unique_tables.add(t)

        # Source 2: table_columns extracted by Stage 1 (works for Laravel)
        # code_map.table_columns is a list[dict] with a "table" key.
        for col in (ctx.code_map.table_columns or []):
            t = col.get("table", "") if isinstance(col, dict) else str(col)
            if t and t != "UNKNOWN":
                unique_tables.add(t)

        # Source 3: migration files — parse create_table / Schema::create calls
        # Stage 1 stores migrations as list[str] (file paths) or list[dict].
        import re as _re
        _CREATE_PAT = _re.compile(
            r'''Schema::create\s*\(\s*['"]([\w]+)['"]|'''
            r'''createTable\s*\(\s*['"]([\w]+)['"]|'''
            r'''\$table\s*=\s*['"]([\w]+)['"]''',
            _re.IGNORECASE,
        )
        for mig in (getattr(ctx.code_map, "migrations", None) or []):
            # migrations may be file-path strings or dicts with "table" key
            if isinstance(mig, dict):
                t = mig.get("table", "") or mig.get("name", "")
                if t and t != "UNKNOWN":
                    unique_tables.add(t)
            elif isinstance(mig, str):
                # Derive table name from migration filename as a cheap heuristic:
                # "2023_01_create_invoices_table.php" → "invoices"
                fname = _re.sub(r'^\d{4}_\d{2}_\d{2}_\d{6}_', '', mig)
                fname = _re.sub(r'_(table)?\.php$', '', fname)
                fname = _re.sub(r'^create_|^add_\w+_to_|^drop_', '', fname)
                if fname and len(fname) > 2:
                    unique_tables.add(fname)

    # Source 4: graph db_table nodes (Stage 2)
    graph_table_count = 0
    if ctx.graph_meta:
        graph_table_count = (
            ctx.graph_meta.node_types.count("db_table")
            if ctx.graph_meta.node_types else 0
        )

    table_count = max(len(unique_tables), graph_table_count)
    details["unique_tables"] = sorted(unique_tables)

    if sql_count == 0 and table_count == 0:
        _b2_lang = ctx.code_map.language.value if ctx.code_map else "unknown"
        blockers.append(
            "B2: No SQL queries or DB tables found anywhere in the codebase. "
            "BA documents would have no data model section. "
            f"Check that the Stage 1 parser is extracting DB queries or ORM models "
            f"from your {_b2_lang} project "
            "(run: python run_stage1.py <project> then inspect code_map.json)."
        )

    # B3: Embedding meta present and collection non-empty
    if ctx.embedding_meta is None:
        blockers.append(
            "B3: embedding_meta is None — Stage 3 (embed) must complete before preflight."
        )
    elif ctx.embedding_meta.total_chunks < MIN_CHUNKS:
        blockers.append(
            f"B3: ChromaDB collection has only {ctx.embedding_meta.total_chunks} chunks "
            f"(minimum {MIN_CHUNKS}). Re-run Stage 3."
        )
    else:
        details["total_chunks"]      = ctx.embedding_meta.total_chunks
        details["embedding_model"]   = ctx.embedding_meta.embedding_model
        details["collection_name"]   = ctx.embedding_meta.collection_name
        signals["has_embeddings"]    = True

    # B4: ChromaDB sanity query — collection must return sensible results
    if ctx.embedding_meta and ctx.embedding_meta.total_chunks >= MIN_CHUNKS:
        sanity_ok, sanity_msg, sanity_dist = _run_sanity_query(ctx)
        details["sanity_query_distance"] = sanity_dist
        details["sanity_query_result"]   = sanity_msg
        if not sanity_ok:
            blockers.append(
                f"B4: ChromaDB sanity query failed — {sanity_msg}. "
                "The collection may be corrupt. Delete the chromadb/ folder and re-run Stage 3."
            )

    # B5: Graph meta present and non-empty
    if ctx.graph_meta is None:
        blockers.append(
            "B5: graph_meta is None — Stage 2 (graph) must complete before preflight."
        )
    elif ctx.graph_meta.node_count == 0:
        blockers.append(
            "B5: Knowledge graph has 0 nodes. Re-run Stage 2."
        )
    else:
        details["graph_nodes"]   = ctx.graph_meta.node_count
        details["graph_edges"]   = ctx.graph_meta.edge_count
        details["graph_node_types"] = ctx.graph_meta.node_types
        details["graph_edge_types"] = ctx.graph_meta.edge_types
        signals["has_graph"] = True

    # ── Warnings ──────────────────────────────────────────────────────────────
    if not blockers and ctx.code_map:

        # W1: Thin data model
        if 0 < table_count < MIN_TABLES_WARN:
            warnings.append(
                f"W1: Only {table_count} unique DB table(s) found. "
                "The data model section of BA documents will be thin. "
                "Verify SQL queries are being extracted correctly."
            )
        if table_count >= 1:
            signals["has_sql"]    = sql_count > 0
            signals["has_tables"] = table_count > 0

        # W2: No form inputs — language-aware
        from context import Framework as _FwW2, Language as _LangW2
        _lang_w2 = ctx.code_map.language

        # PHP: superglobals ($_POST, $_GET, $_REQUEST)
        post_reads = [s for s in ctx.code_map.superglobals
                      if s.get("var") in ("$_POST", "$_GET", "$_REQUEST")]

        # Laravel: controller inputs + POST endpoints
        _laravel_inputs = 0
        if ctx.code_map.framework == _FwW2.LARAVEL:
            for _ep in (getattr(ctx.code_map, "execution_paths", []) or []):
                for _cf in _ep.get("controller_flows", []):
                    _laravel_inputs += len(_cf.get("inputs", []))
            _laravel_inputs += sum(
                1 for e in ctx.code_map.http_endpoints
                if e.get("method", "").upper() in ("POST", "PUT", "PATCH", "DELETE")
            )

        # TypeScript/JavaScript: React-controlled form_fields (RHF, Shadcn, onSubmit)
        # and input_params (req.body, req.query, req.params from API routes)
        _ts_form_count = len(getattr(ctx.code_map, "form_fields", None) or [])
        _ts_input_count = len([
            p for p in (getattr(ctx.code_map, "input_params", None) or [])
            if p.get("source") in ("body", "query", "params")
        ])
        # Also count POST/PUT/PATCH HTTP endpoints as evidence of form submissions
        _ts_post_eps = sum(
            1 for e in (ctx.code_map.http_endpoints or [])
            if e.get("method", "").upper() in ("POST", "PUT", "PATCH", "DELETE")
        )
        _ts_forms = _ts_form_count + _ts_input_count + _ts_post_eps

        _total_form_signals = len(post_reads) + _laravel_inputs + _ts_forms
        if _total_form_signals > 0:
            signals["has_forms"] = True
            details["form_field_count"] = (
                len({s.get("key") for s in post_reads if s.get("key")})
                + _laravel_inputs
                + _ts_forms
            )
        else:
            details["form_field_count"] = 0
            if ctx.code_map.framework == _FwW2.LARAVEL:
                warnings.append(
                    "W2 (INFO): No $_POST/$_GET reads found (expected for Laravel — "
                    "uses $request->input()). Re-run Stage 1.5 if execution_paths is empty."
                )
            elif _lang_w2 in (_LangW2.TYPESCRIPT, _LangW2.JAVASCRIPT):
                warnings.append(
                    "W2: No React form fields or API route inputs detected. "
                    "SRS functional requirements sections will lack form field details. "
                    "Check that TypeScript parser extracted form_fields and input_params."
                )
            else:
                warnings.append(
                    "W2: No $_POST or $_GET reads found. "
                    "SRS functional requirements sections will lack form field details."
                )

        # W3: No navigation / auth signals — language-aware
        from context import Framework as _FwW3, Language as _LangW3
        _lang_w3 = ctx.code_map.language

        # PHP: header() redirects
        _REDIRECT_TYPES = {"redirect", "redirect_back"}
        _laravel_return_count = 0
        _laravel_redirects = 0
        if ctx.code_map.framework == _FwW3.LARAVEL:
            for _ep in (getattr(ctx.code_map, "execution_paths", []) or []):
                for _cf in _ep.get("controller_flows", []):
                    rt = _cf.get("return_type") or ""
                    if rt:
                        _laravel_return_count += 1
                    if rt in _REDIRECT_TYPES:
                        _laravel_redirects += 1

        # TypeScript/JavaScript: auth signals (middleware, guards, useRouter, redirect())
        # and Next.js Server Action / API route navigation evidence
        _TS_AUTH_KINDS = {
            "auth_guard", "nextjs_redirect", "useRouter", "clerk_auth",
            "next_auth", "firebase_auth", "middleware", "protected_route",
        }
        _ts_auth_signals = [
            s for s in (getattr(ctx.code_map, "auth_signals", None) or [])
            if (s.get("kind") or s.get("type") or "") in _TS_AUTH_KINDS
        ]
        # Also count execution_paths with an auth_guard as navigation evidence
        _ts_exec_guards = sum(
            1 for ep in (getattr(ctx.code_map, "execution_paths", None) or [])
            if ep.get("auth_guard") and ep["auth_guard"].get("present", False)
        )
        _ts_nav = len(_ts_auth_signals) + _ts_exec_guards

        if ctx.code_map.redirects:
            signals["has_auth"] = any(
                r.get("target", "").endswith("login.php") or
                r.get("target", "").endswith("session.php")
                for r in ctx.code_map.redirects
            )
        elif _laravel_return_count > 0:
            signals["has_auth"] = _laravel_redirects > 0
        elif _ts_nav > 0:
            signals["has_auth"] = True
        else:
            if ctx.code_map.framework == _FwW3.LARAVEL:
                warnings.append(
                    "W3 (INFO): No header() redirects found (expected for Laravel). "
                    "Re-run Stage 1.5 to populate controller return types."
                )
            elif _lang_w3 in (_LangW3.TYPESCRIPT, _LangW3.JAVASCRIPT):
                warnings.append(
                    "W3: No auth guards or navigation signals detected. "
                    "Navigation flow diagrams will be absent from BA docs. "
                    "Check that TypeScript parser extracted auth_signals and execution_paths."
                )
            else:
                warnings.append(
                    "W3: No header() redirect patterns found. "
                    "Navigation flow diagrams will be absent from BA docs."
                )

        # W4: No user-defined functions
        if not ctx.code_map.functions:
            from context import Language as _LangW4
            if ctx.code_map.language == _LangW4.PHP:
                _w4_detail = "The codebase appears to be entirely inline procedural PHP."
            else:
                _w4_detail = (
                    f"No functions were extracted from the "
                    f"{ctx.code_map.language.value} codebase."
                )
            warnings.append(
                f"W4: No user-defined functions found. {_w4_detail} "
                "Function-level BA coverage will be absent."
            )
        else:
            signals["has_functions"] = True

        # W5: Chunk type diversity
        if ctx.embedding_meta:
            chunk_types = _get_chunk_types(ctx)
            details["chunk_types"] = chunk_types
            if len(chunk_types) <= 1:
                warnings.append(
                    f"W5: Only 1 chunk type in ChromaDB ({chunk_types}). "
                    "Retrieval diversity will be low. Re-run Stage 3."
                )

        # W6: Parse errors
        parse_errors = getattr(ctx.code_map, "errors", [])
        if parse_errors:
            warnings.append(
                f"W6: {len(parse_errors)} file(s) had parse errors during Stage 1. "
                f"Affected: {', '.join(e.get('file','?') for e in parse_errors[:3])}. "
                "Some features may be missing from BA docs."
            )

        # W7: Old PHP version — only relevant for PHP projects
        from context import Language as _LangW7
        if ctx.code_map.language == _LangW7.PHP:
            php_ver = ctx.code_map.php_version or ""
            try:
                major = int(php_ver.split(".")[0])
                if major < 7:
                    warnings.append(
                        f"W7: PHP {php_ver} detected. Modern PHP patterns (typed properties, "
                        "null coalescing, etc.) will not apply. BA docs will reflect legacy PHP idioms."
                    )
            except (ValueError, IndexError):
                pass

    # ── W8–W11: Static analysis output checks (Stages 3.5–3.8) ───────────────

    # W8: GraphRAG index (Stage 3.8)
    if ctx.graph_rag_meta is None:
        warnings.append(
            "W8: graph_rag_meta is None — Stage 3.8 (graphrag) did not complete. "
            "Stage 4 retrieval will fall back to raw ChromaDB only; "
            "context quality will be lower."
        )
    else:
        node_count = getattr(ctx.graph_rag_meta, "node_count", None)
        if node_count is not None and node_count == 0:
            warnings.append(
                "W8: GraphRAG index was built but contains 0 nodes. "
                "Re-run Stage 3.8 (graphrag) to rebuild the index."
            )
        else:
            signals["has_graphrag"] = True
            details["graphrag_nodes"] = node_count

    # W9: Entity catalog (Stage 3.5)
    entity_count = len(ctx.entities.entities) if ctx.entities else 0
    if ctx.entities is None or entity_count == 0:
        warnings.append(
            "W9: No entities extracted (Stage 3.5). "
            "Domain model will lack structured entity grounding. "
            "Check that Stage 3.5 completed and code_map has DB tables."
        )
    else:
        signals["has_entities"] = True
        details["entity_count"] = entity_count

    # W10: Relationship catalog (Stage 3.6)
    rel_count = len(ctx.relationships.relationships) if ctx.relationships else 0
    if ctx.relationships is None or rel_count == 0:
        warnings.append(
            "W10: No relationships mapped (Stage 3.6). "
            "ER diagram and relationship context will be absent from Stage 4. "
            "This is expected if the codebase has only one entity."
        )
    else:
        signals["has_relationships"] = True
        details["relationship_count"] = rel_count

    # W11: State machine catalog (Stage 3.7)
    sm_count = len(ctx.state_machines.machines) if ctx.state_machines else 0
    if ctx.state_machines is None or sm_count == 0:
        warnings.append(
            "W11: No state machines found (Stage 3.7). "
            "State lifecycle sections will be absent — this is expected for "
            "simple CRUD apps with no explicit status/state fields."
        )
    else:
        signals["has_statemachines"] = True
        details["state_machine_count"] = sm_count

    # ── Compute signal quality score ──────────────────────────────────────────
    score = sum(SCORE_WEIGHTS[k] for k, v in signals.items() if v)
    details["signals"]       = signals
    details["quality_score"] = score
    details["quality_band"]  = _score_band(score)

    # ── Assemble result ───────────────────────────────────────────────────────
    passed = len(blockers) == 0
    ctx.preflight = PreflightResult(
        passed   = passed,
        warnings = warnings,
        blockers = blockers,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(passed, blockers, warnings, score, details)

    # ── Save report ───────────────────────────────────────────────────────────
    _save_report(report_path, passed, blockers, warnings, score, details)

    if passed:
        ctx.stage("stage39_preflight").mark_completed(report_path)
        ctx.save()
    else:
        ctx.stage("stage39_preflight").mark_failed(
            f"{len(blockers)} blocker(s): " + " | ".join(blockers)
        )
        ctx.save()
        raise PreflightBlocker(
            f"[stage39] Preflight FAILED — {len(blockers)} blocker(s).\n"
            + "\n".join(f"  • {b}" for b in blockers)
            + f"\n\nFull report: {report_path}"
        )


# ─── Checks ────────────────────────────────────────────────────────────────────

_SANITY_QUERIES: dict[str, str] = {
    "typescript":  "React component API route database query",
    "javascript":  "JavaScript module function database query",
    "java":        "Spring service repository database query",
    "kotlin":      "Kotlin coroutine service repository database",
    "php":         "PHP file database query SQL",
}

def _run_sanity_query(ctx: PipelineContext) -> tuple[bool, str, float]:
    """
    Fire a basic semantic query at ChromaDB to confirm it's returning
    sensible results. Returns (ok, message, best_distance).

    The query string is language-adaptive so it retrieves chunks that are
    actually relevant to the project's tech stack rather than always
    searching for PHP-specific terminology.
    """
    lang = ctx.code_map.language.value if ctx.code_map else "php"
    query = _SANITY_QUERIES.get(lang, f"{lang} source file function class database")
    try:
        from pipeline.stage30_embed import query_collection
        results = query_collection(ctx, query, n_results=1)
        if not results:
            return False, "Query returned zero results", 1.0
        dist = results[0]["distance"]
        if dist > SANITY_QUERY_MAX_DIST:
            return False, f"Best match distance {dist:.3f} > threshold {SANITY_QUERY_MAX_DIST}", dist
        return True, f"OK (best distance={dist:.3f})", dist
    except Exception as e:
        return False, f"Exception during query: {e}", 1.0


# Candidate metadata keys tried in priority order.
# Stage 3 versions differ on which key they write.
_CHUNK_TYPE_KEYS = ("chunk_type", "type", "source_type", "doc_type", "category")


def _get_chunk_types(ctx: PipelineContext) -> list[str]:
    """
    Return the distinct chunk types present in this run.

    Strategy (two-tier):
      1. Read chunks_manifest.json written by Stage 3 — it always carries a
         "chunk_type" field and is fast to parse. This is the reliable path
         because Stage 3 does NOT write chunk_type into ChromaDB metadata.
      2. Fall back to querying ChromaDB metadata if the manifest is missing,
         trying multiple candidate key names.
    """
    import json as _json
    from pathlib import Path as _Path

    # ── Tier 1: manifest file (preferred) ────────────────────────────────────
    # chunks_manifest.json lives next to the chromadb/ folder
    manifest_path = _Path(ctx.embedding_meta.chroma_path).parent / "chunks_manifest.json"
    if manifest_path.exists():
        try:
            manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
            types = sorted({
                c.get("chunk_type", "")
                for c in manifest
                if c.get("chunk_type", "") not in ("", None)
            })
            if types:
                return types
        except Exception:
            pass  # fall through to ChromaDB

    # ── Tier 2: ChromaDB metadata (fallback) ─────────────────────────────────
    try:
        from pipeline.stage30_embed import _try_load_collection, COLLECTION_NAME
        collection = _try_load_collection(
            ctx.embedding_meta.chroma_path,
            COLLECTION_NAME,
        )
        if collection is None:
            return []

        sample_size = min(500, ctx.embedding_meta.total_chunks)
        results = collection.get(limit=sample_size, include=["metadatas"])
        metadatas: list[dict] = results.get("metadatas") or []
        if not metadatas:
            return []

        # Try each candidate key; keep the result with the most distinct values
        best: list[str] = []
        for key in _CHUNK_TYPE_KEYS:
            found = sorted({
                m.get(key, "")
                for m in metadatas
                if m.get(key, "") not in ("", None)
            })
            if len(found) > len(best):
                best = found

        if not best:
            all_keys: set[str] = set()
            for m in metadatas[:20]:
                all_keys.update(m.keys())
            print(f"  [stage39] W5 debug: no type key found in ChromaDB. "
                  f"Available metadata keys: {sorted(all_keys)}")

        return best
    except Exception as exc:
        print(f"  [stage39] W5: chunk type lookup failed — {exc}")
        return []


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _score_band(score: int) -> str:
    if score >= 90: return "excellent"
    if score >= 70: return "good"
    if score >= 50: return "fair"
    return "poor"


def _print_summary(
    passed:   bool,
    blockers: list[str],
    warnings: list[str],
    score:    int,
    details:  dict,
) -> None:
    band  = _score_band(score)
    bar   = "█" * (score // 5) + "░" * (20 - score // 5)

    print(f"\n  {'='*54}")
    print(f"  Preflight {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"  {'='*54}")
    print(f"  Signal quality : [{bar}] {score}/100 ({band})")
    print(f"  Files parsed   : {details.get('total_files', '?')}")
    print(f"  SQL queries    : {details.get('sql_query_count', '?')}")
    print(f"  Unique tables  : {len(details.get('unique_tables', []))}")
    print(f"  Form fields    : {details.get('form_field_count', '?')}")
    print(f"  Chunks         : {details.get('total_chunks', '?')}")
    print(f"  Chunk types    : {', '.join(details.get('chunk_types', []))}")
    print(f"  Graph nodes    : {details.get('graph_nodes', '?')}")
    print(f"  GraphRAG nodes : {details.get('graphrag_nodes', '?')}")
    print(f"  Entities       : {details.get('entity_count', '?')}")
    print(f"  Relationships  : {details.get('relationship_count', '?')}")
    print(f"  State machines : {details.get('state_machine_count', '?')}")

    if details.get("unique_tables"):
        print(f"  Tables found   : {', '.join(details['unique_tables'])}")

    signals = details.get("signals", {})
    sig_str = "  ".join(
        f"{'✓' if v else '✗'} {k.replace('has_', '')}"
        for k, v in signals.items()
    )
    print(f"  Signals        : {sig_str}")

    if blockers:
        print(f"\n  BLOCKERS ({len(blockers)}):")
        for b in blockers:
            print(f"    ✗ {b}")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    ⚠  {w}")

    print(f"  {'='*54}\n")


def _save_report(
    path:     str,
    passed:   bool,
    blockers: list[str],
    warnings: list[str],
    score:    int,
    details:  dict,
) -> None:
    report = {
        "passed":        passed,
        "quality_score": score,
        "quality_band":  _score_band(score),
        "blockers":      blockers,
        "warnings":      warnings,
        "details":       details,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


def _restore_preflight(report_path: str) -> PreflightResult:
    """Rebuild PreflightResult from a saved JSON report."""
    with open(report_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return PreflightResult(
        passed   = data.get("passed", False),
        warnings = data.get("warnings", []),
        blockers = data.get("blockers", []),
    )


# ─── Exception ────────────────────────────────────────────────────────────────

class PreflightBlocker(RuntimeError):
    """
    Raised when preflight finds a hard blocker.
    Caught by run_pipeline.py as a PipelineBlocker.
    """
    pass
