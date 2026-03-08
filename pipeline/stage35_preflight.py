"""
pipeline/stage35_preflight.py — Context Pre-flight Checks

Runs a battery of fast, deterministic checks on the pipeline context
(CodeMap + GraphMeta + EmbeddingMeta) BEFORE the expensive Claude API
calls in Stage 4+. Acts as a safety gate to catch bad inputs early.

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
If stage35_preflight is already COMPLETED and preflight_report.json exists,
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
    "has_sql":        25,   # SQL queries found
    "has_tables":     20,   # DB table nodes in graph
    "has_forms":      15,   # $_POST/$_GET reads
    "has_auth":       10,   # session/redirect patterns
    "has_functions":  10,   # user-defined functions
    "has_graph":      10,   # knowledge graph built
    "has_embeddings": 10,   # ChromaDB populated
}


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 3.5 entry point. Runs all preflight checks and populates
    ctx.preflight. Raises PreflightBlocker if any hard blocker is found.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        PreflightBlocker: If a hard blocker is found. Message includes
                          which check failed and how to fix it.
    """
    report_path = ctx.output_path("preflight_report.json")

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage35_preflight") and Path(report_path).exists():
        ctx.preflight = _restore_preflight(report_path)
        print(f"  [stage35] Already completed — "
              f"{'PASSED' if ctx.preflight.passed else 'FAILED'}, "
              f"{len(ctx.preflight.warnings)} warning(s).")
        return

    print("  [stage35] Running preflight checks ...")

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
                "B1: CodeMap reports 0 PHP files parsed. "
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
    sql_count  = len(ctx.code_map.sql_queries) if ctx.code_map else 0
    table_count = 0
    if ctx.graph_meta:
        table_count = ctx.graph_meta.node_types.count("db_table") if ctx.graph_meta.node_types else 0
    # Also count from code_map directly
    if ctx.code_map:
        unique_tables = {q.get("table", "") for q in ctx.code_map.sql_queries
                         if q.get("table") and q["table"] != "UNKNOWN"}
        table_count = max(table_count, len(unique_tables))
        details["unique_tables"] = sorted(unique_tables)

    if sql_count == 0 and table_count == 0:
        blockers.append(
            "B2: No SQL queries or DB tables found anywhere in the codebase. "
            "BA documents would have no data model section. "
            "Check that parse_project.php is extracting SQL from your project "
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

        # W2: No form inputs
        post_reads = [s for s in ctx.code_map.superglobals
                      if s.get("var") in ("$_POST", "$_GET", "$_REQUEST")]
        if not post_reads:
            warnings.append(
                "W2: No $_POST or $_GET reads found. "
                "SRS functional requirements sections will lack form field details."
            )
        else:
            signals["has_forms"] = True
            details["form_field_count"] = len({s.get("key") for s in post_reads if s.get("key")})

        # W3: No redirects
        if not ctx.code_map.redirects:
            warnings.append(
                "W3: No header() redirect patterns found. "
                "Navigation flow diagrams will be absent from BA docs."
            )
        else:
            signals["has_auth"] = any(
                r.get("target", "").endswith("login.php") or
                r.get("target", "").endswith("session.php")
                for r in ctx.code_map.redirects
            )

        # W4: No user-defined functions
        if not ctx.code_map.functions:
            warnings.append(
                "W4: No user-defined functions found. "
                "The codebase appears to be entirely inline procedural PHP. "
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

        # W7: Old PHP version
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
        ctx.stage("stage35_preflight").mark_completed(report_path)
        ctx.save()
    else:
        ctx.stage("stage35_preflight").mark_failed(
            f"{len(blockers)} blocker(s): " + " | ".join(blockers)
        )
        ctx.save()
        raise PreflightBlocker(
            f"[stage35] Preflight FAILED — {len(blockers)} blocker(s).\n"
            + "\n".join(f"  • {b}" for b in blockers)
            + f"\n\nFull report: {report_path}"
        )


# ─── Checks ────────────────────────────────────────────────────────────────────

def _run_sanity_query(ctx: PipelineContext) -> tuple[bool, str, float]:
    """
    Fire a basic semantic query at ChromaDB to confirm it's returning
    sensible results. Returns (ok, message, best_distance).
    """
    try:
        from pipeline.stage3_embed import query_collection
        results = query_collection(ctx, "PHP file database query", n_results=1)
        if not results:
            return False, "Query returned zero results", 1.0
        dist = results[0]["distance"]
        if dist > SANITY_QUERY_MAX_DIST:
            return False, f"Best match distance {dist:.3f} > threshold {SANITY_QUERY_MAX_DIST}", dist
        return True, f"OK (best distance={dist:.3f})", dist
    except Exception as e:
        return False, f"Exception during query: {e}", 1.0


def _get_chunk_types(ctx: PipelineContext) -> list[str]:
    """Query ChromaDB for the set of chunk types present in the collection."""
    try:
        from pipeline.stage3_embed import _try_load_collection, COLLECTION_NAME
        collection = _try_load_collection(
            ctx.embedding_meta.chroma_path,
            COLLECTION_NAME,
        )
        if collection is None:
            return []
        # Fetch a sample of metadata to see what chunk types are present
        results = collection.get(limit=200, include=["metadatas"])
        types = {m.get("chunk_type", "unknown") for m in results["metadatas"]}
        return sorted(types)
    except Exception:
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
