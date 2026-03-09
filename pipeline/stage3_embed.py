"""
pipeline/stage3_embed.py — Embedding & Vector Index Stage

Converts the CodeMap (Stage 1) and Knowledge Graph (Stage 2) into semantically
rich text chunks, embeds them with a local sentence-transformers model (all-MiniLM-L6-v2), and stores
them in a persistent ChromaDB collection.

This gives Stage 4 (DomainAnalystAgent) and Stage 5 agents the ability to do
semantic search over the codebase — e.g.:
    "find all pages that handle user authentication"
    "what tables does the registration flow touch?"
    "which functions validate user input?"

Architecture
------------
Chunking strategy — one chunk per meaningful unit:

    CHUNK TYPE          CONTENT EXAMPLE
    ─────────────────   ──────────────────────────────────────────────────────
    file_summary        "login.php is a page-level entry point that reads
                         $_POST[uname], $_POST[password], queries table
                         'registerhere' (SELECT), and redirects to
                         carrental.php on success."

    function_def        "Function validate_phone_number in registration.php:
                         params=[$phone], validates phone number format using
                         filter_var and strlen."

    sql_operation       "registration.php performs INSERT on table
                         'registerhere' with fields: fname, lname, phone,
                         gender, dob, email, uname, pwd, city, country, state."

    navigation_flow     "checksession.php redirects unauthenticated users to
                         login.php. Included by: rent.php, myorder.php,
                         request.php."

    form_inputs         "rent.php reads form fields: no, cc, dc, doi, dor,
                         distance, carid, dln from $_POST."

    graph_neighbourhood "TABLE:registerhere is connected to: registration.php
                         (sql_write), checksession.php (sql_read),
                         rent.php (sql_read), myorder.php (sql_read)."

    db_schema           "Migration creates table 'users' with columns: id
                         (bigIncrements), name (string), email (string,
                         unique), password (string)."

    class_summary       "Class UserController (controller) in
                         app/Http/Controllers/UserController.php extends
                         Controller, implements: []. Methods: index, store,
                         show, update, destroy."

Metadata stored per chunk (used for filtering in Stage 4):
    chunk_type, source_file, node_id, framework, entity_type, table_name,
    http_method, involves_auth, involves_db, involves_redirect

Resume behaviour
----------------
If stage3_embed is COMPLETED and the ChromaDB collection already contains
documents, the stage is skipped.

Dependencies
------------
    pip install chromadb sentence-transformers
    No API key required — model runs locally.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from context import CodeMap, EmbeddingMeta, Framework, PipelineContext

# ── Configuration ──────────────────────────────────────────────────────────────

COLLECTION_NAME   = "phpba_codebase"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # local sentence-transformers model
CHROMA_SUBDIR     = "chromadb"
CHUNK_BATCH_SIZE  = 64       # documents per ChromaDB upsert call
MAX_CHUNK_CHARS   = 1000     # hard cap — all-MiniLM-L6-v2 truncates at ~256 tokens (~1000 chars)

# Semantic splitting constants
COLS_PER_SCHEMA_PART   = 12   # db_schema: max columns per sub-chunk
EDGES_SPLIT_THRESHOLD  = 14   # graph_neighbourhood: split when degree exceeds this

# Superglobals that carry user-supplied input (used for auth/input tagging)
INPUT_SUPERGLOBALS  = {"$_POST", "$_GET", "$_REQUEST", "$_FILES"}
SESSION_SUPERGLOBALS = {"$_SESSION", "$_COOKIE"}

# SQL operations that modify data
WRITE_OPERATIONS = {"INSERT", "UPDATE", "DELETE", "REPLACE"}
READ_OPERATIONS  = {"SELECT"}
DDL_OPERATIONS   = {"CREATE", "DROP", "ALTER", "TRUNCATE"}


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 3 entry point.

    Builds text chunks from CodeMap + graph, embeds them locally via sentence-transformers,
    and persists to ChromaDB. Populates ctx.embedding_meta.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If code_map is missing or ChromaDB fails.
    """
    chroma_path = ctx.output_path(CHROMA_SUBDIR)

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage3_embed"):
        existing = _try_load_collection(chroma_path, COLLECTION_NAME)
        if existing and existing.count() > 0:
            print(f"  [stage3] Resuming — collection '{COLLECTION_NAME}' "
                  f"already has {existing.count()} chunks.")
            ctx.embedding_meta = EmbeddingMeta(
                collection_name = COLLECTION_NAME,
                chroma_path     = chroma_path,
                total_chunks    = existing.count(),
                embedding_model = EMBEDDING_MODEL,
            )
            return

    # ── Deferred heavy imports (skipped entirely on resume) ─────────────────
    # chromadb triggers sentence-transformers model load on import — only do
    # this when we actually need to embed, not on every pipeline boot.
    import chromadb  # noqa: F401 (used in helper functions below)

    # ── Pre-flight ────────────────────────────────────────────────────────────
    if ctx.code_map is None:
        raise RuntimeError("[stage3] ctx.code_map is None — run Stage 1 first.")

    # ── Load graph (optional — enriches chunks but not required) ─────────────
    graph = None
    if ctx.graph_meta and ctx.graph_meta.graph_path:
        graph = _load_graph_safe(ctx.graph_meta.graph_path)

    print(f"  [stage3] Building chunks from CodeMap "
          f"(framework={ctx.code_map.framework.value}) ...")

    # ── Build all chunks ──────────────────────────────────────────────────────
    chunks = _build_chunks(ctx.code_map, graph)
    print(f"  [stage3] Built {len(chunks)} chunks across "
          f"{len({c['metadata']['chunk_type'] for c in chunks})} types")

    # ── Initialise ChromaDB + OpenAI embedding function ───────────────────────
    client     = _make_chroma_client(chroma_path)
    collection = _get_or_create_collection(client, COLLECTION_NAME)

    # ── Embed + upsert in batches ─────────────────────────────────────────────
    total = _upsert_chunks(collection, chunks)

    print(f"  [stage3] Embedded and stored {total} chunks → {chroma_path}")

    # ── Persist metadata ──────────────────────────────────────────────────────
    ctx.embedding_meta = EmbeddingMeta(
        collection_name = COLLECTION_NAME,
        chroma_path     = chroma_path,
        total_chunks    = total,
        embedding_model = EMBEDDING_MODEL,
    )

    # Save chunk manifest for debugging / Stage 4 inspection
    manifest_path = ctx.output_path("chunks_manifest.json")
    _save_manifest(chunks, manifest_path)

    ctx.stage("stage3_embed").mark_completed(chroma_path)
    ctx.save()

    print(f"  [stage3] Done — {total} chunks in collection '{COLLECTION_NAME}'")
    print(f"  [stage3] Chunk manifest → {manifest_path}")


# ─── Chunk Builders ────────────────────────────────────────────────────────────

def _build_chunks(code_map: CodeMap, graph: Any | None) -> list[dict]:
    """
    Orchestrate all chunking strategies and return a flat list of chunk dicts.
    Each dict has keys: id, text, metadata.
    """
    chunks: list[dict] = []

    # Per-file context lookup tables (built once, reused by all builders)
    ctx_tables = _build_context_tables(code_map)

    # Build file→module_id lookup from graph node attributes (Pass 15).
    # Falls back to empty dict gracefully when graph is absent.
    file_module: dict[str, str] = {}
    if graph is not None:
        for _nid, _d in graph.nodes(data=True):
            _fp = _d.get("file", "")
            _mi = _d.get("module_id", "")
            if _fp and _mi and _fp not in file_module:
                file_module[_fp] = _mi

    chunks += _chunks_file_summaries(code_map, ctx_tables, file_module)
    chunks += _chunks_function_defs(code_map, file_module)
    chunks += _chunks_sql_operations(code_map, ctx_tables, file_module)
    chunks += _chunks_navigation_flows(code_map, ctx_tables, file_module)
    chunks += _chunks_form_inputs(code_map, file_module)
    chunks += _chunks_class_summaries(code_map, file_module)
    chunks += _chunks_db_schema(code_map, file_module)

    if graph is not None:
        chunks += _chunks_graph_neighbourhoods(graph, code_map)

    # Deduplicate by text (identical summaries can arise for small projects)
    seen_texts: set[str] = set()
    unique: list[dict] = []
    for chunk in chunks:
        if chunk["text"] not in seen_texts:
            seen_texts.add(chunk["text"])
            unique.append(chunk)

    return unique


# ── Context Tables ─────────────────────────────────────────────────────────────

def _build_context_tables(cm: CodeMap) -> dict[str, Any]:
    """
    Pre-compute per-file lookup dicts so individual chunk builders
    don't have to iterate the full lists repeatedly.
    """
    # sql queries grouped by file
    sql_by_file: dict[str, list] = defaultdict(list)
    for q in cm.sql_queries:
        sql_by_file[q["file"]].append(q)

    # redirects grouped by file
    redirects_by_file: dict[str, list] = defaultdict(list)
    for r in cm.redirects:
        redirects_by_file[r["file"]].append(r)

    # includes grouped by file (what each file pulls in)
    includes_by_file: dict[str, list] = defaultdict(list)
    for inc in cm.includes:
        includes_by_file[inc["file"]].append(inc)

    # reverse includes: what files include THIS file
    included_by: dict[str, list] = defaultdict(list)
    for inc in cm.includes:
        included_by[inc["target"]].append(inc["file"])

    # superglobals grouped by file
    sg_by_file: dict[str, list] = defaultdict(list)
    for sg in cm.superglobals:
        sg_by_file[sg["file"]].append(sg)

    # functions grouped by file
    fns_by_file: dict[str, list] = defaultdict(list)
    for fn in cm.functions:
        fns_by_file[fn["file"]].append(fn)

    # all source files (union of everything we know about)
    all_files: set[str] = set()
    for bucket in [cm.sql_queries, cm.redirects, cm.includes,
                   cm.superglobals, cm.functions]:
        for item in bucket:
            all_files.add(item["file"])
    for f in cm.html_pages:
        all_files.add(f)

    return {
        "sql_by_file":      dict(sql_by_file),
        "redirects_by_file": dict(redirects_by_file),
        "includes_by_file": dict(includes_by_file),
        "included_by":      dict(included_by),
        "sg_by_file":       dict(sg_by_file),
        "fns_by_file":      dict(fns_by_file),
        "all_files":        all_files,
        "html_pages":       set(cm.html_pages),
    }



def _module_id_for_file(filepath: str, file_module: dict[str, str]) -> str:
    """
    Return the module_id for a given file path from the graph node lookup.
    Falls back to "unknown" when the graph was not loaded or the file was
    not reached by Stage 2 Pass 15.
    """
    if not filepath or not file_module:
        return "unknown"
    # Exact match first; then try basename (handles relative vs absolute mismatches)
    if filepath in file_module:
        return file_module[filepath]
    base = Path(filepath).name
    for fp, mid in file_module.items():
        if Path(fp).name == base:
            return mid
    return "unknown"


# ─── Semantic Anchors & Splitting Helpers ─────────────────────────────────────

def _make_anchor(chunk_type: str, metadata: dict[str, str]) -> str:
    """
    Build a one-line structured header that is prepended to every chunk
    before embedding.

    Purpose
    -------
    sentence-transformers (all-MiniLM-L6-v2) learns token importance
    from position and frequency.  A consistent structured prefix encodes
    the four most retrieval-relevant facts at the very start of every chunk:

        CHUNK_TYPE | primary_subject | module:name | role/context

    This lets the model distinguish "UserController methods" from
    "users table schema" even when the body text is similar.

    Format per chunk type
    ---------------------
    FILE_SUMMARY        FILE_SUMMARY | login.php | module:auth | page entry-point
    FUNCTION_DEF        FUNCTION_DEF | validate_phone | login.php | module:auth
    SQL_OPERATION       SQL_OPERATION | INSERT | table:users | registration.php
    NAVIGATION_FLOW     NAVIGATION_FLOW | checksession.php | auth guard | module:auth
    FORM_INPUTS         FORM_INPUTS | registration.php | POST fields | module:auth
    CLASS_SUMMARY       CLASS_SUMMARY | UserController | controller | module:auth
    DB_SCHEMA           DB_SCHEMA | CREATE TABLE users | columns 1-12 | module:unknown
    GRAPH_NEIGHBOURHOOD GRAPH_NEIGHBOURHOOD | TABLE:users | db_table | degree:24
    """
    t = chunk_type.upper()

    if t == "FILE_SUMMARY":
        fname  = metadata.get("filename", "?")
        role   = metadata.get("file_role", "script")
        module = metadata.get("module_id", "unknown")
        return f"FILE_SUMMARY | {fname} | module:{module} | {role} entry-point"

    if t == "FUNCTION_DEF":
        name   = metadata.get("function_name", "?")
        fname  = metadata.get("filename", "?")
        module = metadata.get("module_id", "unknown")
        return f"FUNCTION_DEF | {name} | {fname} | module:{module}"

    if t == "SQL_OPERATION":
        op     = metadata.get("sql_operation", "?")
        table  = metadata.get("table_name", "?")
        fname  = metadata.get("filename", "?")
        module = metadata.get("module_id", "unknown")
        return f"SQL_OPERATION | {op} | table:{table} | {fname} | module:{module}"

    if t == "NAVIGATION_FLOW":
        fname  = metadata.get("filename", "?")
        is_auth = metadata.get("involves_auth", "false") == "true"
        module = metadata.get("module_id", "unknown")
        role   = "auth guard" if is_auth else "navigation"
        return f"NAVIGATION_FLOW | {fname} | {role} | module:{module}"

    if t == "FORM_INPUTS":
        fname  = metadata.get("filename", "?")
        fields = metadata.get("post_fields", "")
        module = metadata.get("module_id", "unknown")
        preview = (fields[:40] + "...") if len(fields) > 40 else fields
        return f"FORM_INPUTS | {fname} | POST:{preview} | module:{module}"

    if t == "CLASS_SUMMARY":
        name   = metadata.get("class_name", "?")
        etype  = metadata.get("entity_type", "class")
        module = metadata.get("module_id", "unknown")
        return f"CLASS_SUMMARY | {name} | {etype} | module:{module}"

    if t == "DB_SCHEMA":
        table  = metadata.get("table_name", "?")
        op     = metadata.get("sql_operation", "CREATE")
        part   = metadata.get("part_label", "")
        module = metadata.get("module_id", "unknown")
        part_str = f" | {part}" if part else ""
        return f"DB_SCHEMA | {op} TABLE {table}{part_str} | module:{module}"

    if t == "GRAPH_NEIGHBOURHOOD":
        nid    = metadata.get("node_id", "?")
        ntype  = metadata.get("node_type", "?")
        degree = metadata.get("degree", "?")
        part   = metadata.get("part_label", "")
        part_str = f" | {part}" if part else ""
        return f"GRAPH_NEIGHBOURHOOD | {nid} | {ntype} | degree:{degree}{part_str}"

    # Fallback for any future chunk type
    return f"{t} | {metadata.get('filename', metadata.get('source_file', '?'))}"


def _split_db_schema(
    table:     str,
    operation: str,
    filename:  str,
    columns:   list[dict],
    base_meta: dict[str, str],
    n_per_part: int = COLS_PER_SCHEMA_PART,
) -> list[dict]:
    """
    Split a wide table schema into sub-chunks of at most n_per_part columns.

    Each sub-chunk repeats the table header so it is self-contained:

        DB_SCHEMA | CREATE TABLE wp_posts | columns 13-24 of 40 | module:cms
        Database Schema: CREATE TABLE 'wp_posts' (columns 13-24 of 40)
        Defined in: 2024_create_wp_posts.php
        Columns:
          col_13: varchar(255) [nullable]
          ...

    Called only when len(columns) > n_per_part.  When the table fits in one
    chunk it is emitted by _chunks_db_schema directly (no overhead).
    """
    total   = len(columns)
    batches = [columns[i:i + n_per_part] for i in range(0, total, n_per_part)]
    chunks  = []

    for idx, batch in enumerate(batches):
        start    = idx * n_per_part + 1
        end      = start + len(batch) - 1
        part_lbl = f"columns {start}-{end} of {total}"

        col_lines = []
        for col in batch:
            col_name = col.get("name", "?")
            col_type = col.get("type", "?")
            mods     = col.get("modifiers", [])
            mod_str  = f" [{', '.join(mods)}]" if mods else ""
            col_lines.append(f"  {col_name}: {col_type}{mod_str}")

        body = (
            f"Database Schema: {operation.upper()} TABLE '{table}' ({part_lbl})\n"
            f"Defined in: {filename}\n"
            f"Columns:\n" + "\n".join(col_lines)
        )

        meta = {**base_meta, "part_label": part_lbl, "part_num": str(idx + 1),
                "total_parts": str(len(batches))}
        anchor = _make_anchor("db_schema", meta)
        text   = anchor + "\n" + body

        chunks.append(_make_chunk(
            text        = text,
            chunk_type  = "db_schema",
            source_file = base_meta.get("source_file", ""),
            metadata    = meta,
        ))

    return chunks


def _split_graph_neighbourhood(
    node_id:   str,
    node_name: str,
    node_type: str,
    degree:    int,
    source_file: str,
    in_edges:  list[str],
    out_edges: list[str],
    db_summary: list[str],
    base_meta: dict[str, str],
) -> list[dict]:
    """
    Split a high-degree graph node into at most three focused sub-chunks:

        Part 1 — "connections summary"  (header + compact read/write/ddl lines)
        Part 2 — "incoming edges"       (all ← edges in batches)
        Part 3 — "outgoing edges"       (all → edges in batches)

    Part 1 always exists.  Parts 2 and 3 are only created when there are
    enough edges to warrant it (threshold: EDGES_SPLIT_THRESHOLD).

    Each sub-chunk repeats the node identity line so it is self-contained.
    """
    chunks = []
    header = f"Graph node: {node_name} (type={node_type}, degree={degree})"
    if source_file:
        header += f"\nDefined in: {source_file}"

    # ── Part 1: summary (db_summary lines + compact edge counts) ─────────────
    summary_lines = [header]
    summary_lines.append(
        f"Connections: {len(in_edges)} incoming, {len(out_edges)} outgoing"
    )
    summary_lines.extend(db_summary)   # Read by / Written by / Schema defined in
    # Include a sample of edges so the summary is useful on its own
    SAMPLE = 4
    if in_edges:
        summary_lines.append("Sample incoming:\n" + "\n".join(in_edges[:SAMPLE]))
    if out_edges:
        summary_lines.append("Sample outgoing:\n" + "\n".join(out_edges[:SAMPLE]))

    meta1 = {**base_meta, "part_label": "summary"}
    anchor1 = _make_anchor("graph_neighbourhood", meta1)
    chunks.append(_make_chunk(
        text        = anchor1 + "\n" + "\n".join(summary_lines),
        chunk_type  = "graph_neighbourhood",
        source_file = source_file,
        metadata    = meta1,
    ))

    # ── Part 2: full incoming edges ───────────────────────────────────────────
    if in_edges:
        body2 = header + "\nIncoming connections (all):\n" + "\n".join(in_edges)
        meta2 = {**base_meta, "part_label": "incoming"}
        anchor2 = _make_anchor("graph_neighbourhood", meta2)
        chunks.append(_make_chunk(
            text        = anchor2 + "\n" + body2,
            chunk_type  = "graph_neighbourhood",
            source_file = source_file,
            metadata    = meta2,
        ))

    # ── Part 3: full outgoing edges ───────────────────────────────────────────
    if out_edges:
        body3 = header + "\nOutgoing connections (all):\n" + "\n".join(out_edges)
        meta3 = {**base_meta, "part_label": "outgoing"}
        anchor3 = _make_anchor("graph_neighbourhood", meta3)
        chunks.append(_make_chunk(
            text        = anchor3 + "\n" + body3,
            chunk_type  = "graph_neighbourhood",
            source_file = source_file,
            metadata    = meta3,
        ))

    return chunks


# ── Chunk Type 1: Per-file summaries

def _chunks_file_summaries(cm: CodeMap, ctx: dict, file_module: dict[str, str] | None = None) -> list[dict]:
    """
    One rich summary chunk per PHP source file.
    Combines: file role, SQL ops, redirects, form inputs, includes, functions.
    This is the primary chunk a BA agent will retrieve for page-level questions.
    """
    chunks = []

    for filepath in sorted(ctx["all_files"]):
        filename   = Path(filepath).name
        is_page    = filepath in ctx["html_pages"]
        role       = "page" if is_page else "script"

        parts: list[str] = [
            f"File: {filename} ({role})",
            f"Path: {filepath}",
        ]

        # SQL operations
        sqls = ctx["sql_by_file"].get(filepath, [])
        if sqls:
            sql_lines = []
            for q in sqls:
                op    = q.get("operation", "?")
                table = q.get("table", "?")
                caller = q.get("caller", "GLOBAL_SCRIPT")
                scope  = f" (in function {caller})" if caller != "GLOBAL_SCRIPT" else ""
                sql_lines.append(f"{op} on table '{table}'{scope}")
            parts.append("Database operations:\n  " + "\n  ".join(sql_lines))

        # Form inputs
        sgs = ctx["sg_by_file"].get(filepath, [])
        if sgs:
            post_keys    = sorted({s["key"] for s in sgs
                                   if s["var"] == "$_POST" and s["key"]})
            session_keys = sorted({s["key"] for s in sgs
                                   if s["var"] == "$_SESSION" and s["key"]})
            get_keys     = sorted({s["key"] for s in sgs
                                   if s["var"] == "$_GET" and s["key"]})
            if post_keys:
                parts.append(f"Reads POST fields: {', '.join(post_keys)}")
            if session_keys:
                parts.append(f"Reads SESSION keys: {', '.join(session_keys)}")
            if get_keys:
                parts.append(f"Reads GET fields: {', '.join(get_keys)}")

        # Redirects
        redirs = ctx["redirects_by_file"].get(filepath, [])
        if redirs:
            targets = sorted({r["target"] for r in redirs})
            parts.append(f"Redirects to: {', '.join(targets)}")

        # Includes
        incs = ctx["includes_by_file"].get(filepath, [])
        if incs:
            targets = sorted({inc["target"] for inc in incs})
            parts.append(f"Includes: {', '.join(targets)}")

        # Included by
        inc_by = ctx["included_by"].get(Path(filepath).name, [])
        if inc_by:
            parts.append(f"Included by: {', '.join(sorted(set(inc_by)))}")

        # Functions defined
        fns = ctx["fns_by_file"].get(filepath, [])
        if fns:
            fn_names = [f["name"] for f in fns]
            parts.append(f"Defines functions: {', '.join(fn_names)}")

        text = "\n".join(parts)

        # Trim oversized file summaries: execution_path branches are the most
        # likely culprit for God-class files.  Keep the header + SQL + redirects
        # and truncate branch details, re-joining to stay under MAX_CHUNK_CHARS.
        if len(text) > MAX_CHUNK_CHARS:
            # Rebuild, capping any "Branches" line group to 2 entries
            trimmed_parts = []
            in_branch_block = False
            branch_count = 0
            for p in parts:
                if p.startswith("Branches (") or p.startswith("  if ("):
                    in_branch_block = True
                if in_branch_block:
                    branch_count += 1
                    if branch_count <= 2:
                        trimmed_parts.append(p)
                    elif branch_count == 3:
                        trimmed_parts.append("  ... (truncated for embedding)")
                    # skip rest
                else:
                    trimmed_parts.append(p)
            text = "\n".join(trimmed_parts)
            # Hard fallback: if still too long, trim from the end
            if len(text) > MAX_CHUNK_CHARS:
                text = text[:MAX_CHUNK_CHARS - 30] + "\n  ... (truncated)"

        # Metadata flags used for filtered retrieval
        involves_auth = any(
            r["target"] in ("login.php", "session.php", "logout.php")
            for r in redirs
        ) or any(
            s["var"] == "$_SESSION" for s in sgs
        ) or "session" in filepath.lower() or "login" in filepath.lower()

        involves_db     = bool(sqls)
        involves_redirect = bool(redirs)
        involves_input  = bool(post_keys if sgs else [])

        tables = list({q.get("table", "") for q in sqls if q.get("table")})

        chunks.append(_make_chunk(
            text        = text,
            chunk_type  = "file_summary",
            source_file = filepath,
            metadata    = {
                "filename":          filename,
                "file_role":         role,
                "involves_auth":     str(involves_auth).lower(),
                "involves_db":       str(involves_db).lower(),
                "involves_redirect": str(involves_redirect).lower(),
                "involves_input":    str(involves_input).lower(),
                "tables":            ",".join(tables),
                "framework":         cm.framework.value,
                "module_id":         _module_id_for_file(filepath, file_module or {}),
            },
        ))

    return chunks


# ── Chunk Type 2: Function Definitions ────────────────────────────────────────

def _chunks_function_defs(cm: CodeMap, file_module: dict[str, str] | None = None) -> list[dict]:
    """
    One chunk per defined function, describing its signature,
    purpose (from docblock), params, and what it does (SQL/redirects
    within its body if caller matches).
    """
    chunks = []

    for fn in cm.functions:
        name      = fn["name"]
        filepath  = fn["file"]
        filename  = Path(filepath).name
        params    = [p["name"] for p in fn.get("params", [])]
        ret       = fn.get("return_type") or "void"
        docblock  = fn.get("docblock") or ""
        line      = fn.get("line", "?")

        parts = [
            f"Function: {name}",
            f"File: {filename} (line {line})",
            f"Parameters: {', '.join(params) if params else 'none'}",
            f"Returns: {ret}",
        ]
        if docblock:
            parts.append(f"Description: {docblock}")

        # Pull SQL ops attributed to this function
        fn_sqls = [q for q in cm.sql_queries if q.get("caller") == name]
        if fn_sqls:
            ops = [f"{q['operation']} on '{q['table']}'" for q in fn_sqls]
            parts.append(f"Database operations: {', '.join(ops)}")

        # Pull redirects attributed to this function
        fn_redirs = [r for r in cm.redirects if r.get("caller") == name]
        if fn_redirs:
            targets = [r["target"] for r in fn_redirs]
            parts.append(f"Redirects to: {', '.join(targets)}")

        text = "\n".join(parts)

        chunks.append(_make_chunk(
            text        = text,
            chunk_type  = "function_def",
            source_file = filepath,
            metadata    = {
                "function_name": name,
                "filename":      filename,
                "framework":     cm.framework.value,
                "involves_db":   str(bool(fn_sqls)).lower(),
                "module_id":     _module_id_for_file(filepath, file_module or {}),
            },
        ))

    return chunks


# ── Chunk Type 3: SQL Operations ──────────────────────────────────────────────

def _chunks_sql_operations(cm: CodeMap, ctx: dict, file_module: dict[str, str] | None = None) -> list[dict]:
    """
    One chunk per unique (file, table, operation) combination.
    These are the most important chunks for data-flow questions.
    Includes the actual SQL preview so the agent can see field names.
    """
    chunks = []
    seen: set[tuple] = set()

    for q in cm.sql_queries:
        table     = q.get("table", "UNKNOWN")
        operation = q.get("operation", "SELECT").upper()
        filepath  = q["file"]
        filename  = Path(filepath).name
        sql_text  = q.get("sql", "")
        caller    = q.get("caller", "GLOBAL_SCRIPT")
        key       = (filepath, table, operation)

        if key in seen:
            continue
        seen.add(key)

        op_desc = {
            "SELECT":   "reads from",
            "INSERT":   "inserts into",
            "UPDATE":   "updates",
            "DELETE":   "deletes from",
            "REPLACE":  "replaces in",
            "CREATE":   "creates table",
            "DROP":     "drops table",
            "ALTER":    "alters table",
            "TRUNCATE": "truncates",
        }.get(operation, "queries")

        scope = f"Function '{caller}'" if caller != "GLOBAL_SCRIPT" else f"File {filename}"

        parts = [
            f"SQL Operation: {operation} on table '{table}'",
            f"Source: {scope} in {filename}",
            f"Description: {scope} {op_desc} '{table}'",
        ]

        # Extract field names from INSERT/SELECT SQL if available
        if sql_text:
            fields = _extract_sql_fields(sql_text, operation)
            if fields:
                parts.append(f"Fields involved: {', '.join(fields)}")
            # Trim the SQL preview for the chunk
            sql_preview = sql_text[:300].replace("\n", " ").strip()
            parts.append(f"SQL: {sql_preview}")

        text = "\n".join(parts)

        chunks.append(_make_chunk(
            text        = text,
            chunk_type  = "sql_operation",
            source_file = filepath,
            metadata    = {
                "table_name":    table,
                "sql_operation": operation,
                "filename":      filename,
                "involves_db":   "true",
                "framework":     cm.framework.value,
                "is_write":      str(operation in WRITE_OPERATIONS).lower(),
                "is_read":       str(operation in READ_OPERATIONS).lower(),
                "is_ddl":        str(operation in DDL_OPERATIONS).lower(),
                "module_id":     _module_id_for_file(filepath, file_module or {}),
            },
        ))

    return chunks


# ── Chunk Type 4: Navigation Flows ────────────────────────────────────────────

def _chunks_navigation_flows(cm: CodeMap, ctx: dict, file_module: dict[str, str] | None = None) -> list[dict]:
    """
    Chunks describing page-to-page navigation and authentication gates.
    One chunk per file that has redirects or is a significant include hub.
    """
    chunks = []

    # Group redirects by target to describe "who redirects to login.php"
    redirects_to_target: dict[str, list[str]] = defaultdict(list)
    for r in cm.redirects:
        redirects_to_target[r["target"]].append(r["file"])

    # Per-file navigation chunk
    for filepath in sorted(ctx["all_files"]):
        filename  = Path(filepath).name
        redirs    = ctx["redirects_by_file"].get(filepath, [])
        incs      = ctx["includes_by_file"].get(filepath, [])
        inc_by    = ctx["included_by"].get(filename, [])

        if not redirs and not incs and not inc_by:
            continue  # no navigation info for this file

        parts = [f"Navigation flow for: {filename}"]

        if incs:
            inc_targets = [inc["target"] for inc in incs]
            parts.append(f"Includes these files on load: {', '.join(inc_targets)}")

        if inc_by:
            parts.append(f"This file is included by: {', '.join(sorted(set(inc_by)))}")

        if redirs:
            for r in redirs:
                caller = r.get("caller", "GLOBAL_SCRIPT")
                scope  = f"function '{caller}'" if caller != "GLOBAL_SCRIPT" else "the script"
                parts.append(f"Redirects to '{r['target']}' from {scope}")

        # Auth gate description
        is_auth_gate = any(
            r["target"] == "login.php" for r in redirs
        ) or "checksession" in filename.lower()
        if is_auth_gate:
            parts.append(
                "Authentication gate: this file enforces login — "
                "unauthenticated users are redirected to login.php"
            )

        text = "\n".join(parts)

        chunks.append(_make_chunk(
            text        = text,
            chunk_type  = "navigation_flow",
            source_file = filepath,
            metadata    = {
                "filename":      filename,
                "involves_auth": str(is_auth_gate).lower(),
                "involves_redirect": "true",
                "framework":     cm.framework.value,
                "module_id":     _module_id_for_file(filepath, file_module or {}),
            },
        ))

    # Cross-file navigation chunk: "multiple pages redirect to login.php"
    for target, sources in redirects_to_target.items():
        if len(sources) > 1:
            text = (
                f"Authentication enforcement: {len(sources)} files redirect to "
                f"'{target}': {', '.join(sorted(set(sources)))}. "
                f"This indicates '{target}' is the central authentication entry point."
            )
            chunks.append(_make_chunk(
                text       = text,
                chunk_type = "navigation_flow",
                source_file = target,
                metadata   = {
                    "filename":      target,
                    "involves_auth": "true",
                    "involves_redirect": "true",
                    "framework":     cm.framework.value,
                },
            ))

    return chunks


# ── Chunk Type 5: Form Inputs ──────────────────────────────────────────────────

def _chunks_form_inputs(cm: CodeMap, file_module: dict[str, str] | None = None) -> list[dict]:
    """
    One chunk per file that reads form/session data.
    Groups all $_POST/$_GET/$_SESSION keys per file into a single
    descriptive chunk — critical for SRS functional requirements.
    """
    chunks = []

    # Group by file
    sg_by_file: dict[str, list] = defaultdict(list)
    for sg in cm.superglobals:
        sg_by_file[sg["file"]].append(sg)

    for filepath, sgs in sg_by_file.items():
        filename = Path(filepath).name

        post_keys    = sorted({s["key"] for s in sgs
                               if s["var"] == "$_POST"    and s["key"]})
        get_keys     = sorted({s["key"] for s in sgs
                               if s["var"] == "$_GET"     and s["key"]})
        session_keys = sorted({s["key"] for s in sgs
                               if s["var"] == "$_SESSION" and s["key"]})
        cookie_keys  = sorted({s["key"] for s in sgs
                               if s["var"] == "$_COOKIE"  and s["key"]})

        if not any([post_keys, get_keys, session_keys, cookie_keys]):
            continue

        parts = [f"Form and session data for: {filename}"]

        if post_keys:
            parts.append(
                f"Accepts POST form fields: {', '.join(post_keys)}\n"
                f"  → This page processes a form submission with "
                f"{len(post_keys)} field(s)."
            )
        if get_keys:
            parts.append(f"Reads GET parameters: {', '.join(get_keys)}")
        if session_keys:
            parts.append(
                f"Uses session variables: {', '.join(session_keys)}\n"
                f"  → Requires an active user session."
            )
        if cookie_keys:
            parts.append(f"Reads cookies: {', '.join(cookie_keys)}")

        # Infer purpose from field names
        purpose = _infer_form_purpose(post_keys + get_keys)
        if purpose:
            parts.append(f"Inferred form purpose: {purpose}")

        text = "\n".join(parts)

        chunks.append(_make_chunk(
            text        = text,
            chunk_type  = "form_inputs",
            source_file = filepath,
            metadata    = {
                "filename":       filename,
                "post_fields":    ",".join(post_keys),
                "session_fields": ",".join(session_keys),
                "involves_input": "true",
                "involves_auth":  str(bool(session_keys)).lower(),
                "framework":      cm.framework.value,
                "module_id":      _module_id_for_file(filepath, file_module or {}),
            },
        ))

    return chunks


# ── Chunk Type 6: Class Summaries (OOP projects) ──────────────────────────────

def _chunks_class_summaries(cm: CodeMap, file_module: dict[str, str] | None = None) -> list[dict]:
    """
    One chunk per class/controller/model/service.
    For raw_php projects this may be empty.
    """
    chunks = []

    all_classes = (
        [(c, "controller") for c in cm.controllers] +
        [(c, "model")      for c in cm.models]      +
        [(c, "service")    for c in cm.services]     +
        [(c, "class")      for c in cm.classes]
    )

    for cls, entity_type in all_classes:
        name      = cls.get("name", "?")
        fqn       = cls.get("fqn", name)
        filepath  = cls.get("file", "")
        filename  = Path(filepath).name
        extends   = cls.get("extends")
        implements = cls.get("implements", [])
        methods   = [m["name"] for m in cls.get("methods", [])]
        docblock  = (cls.get("docblock") or {}).get("summary", "")

        parts = [
            f"Class: {name} ({entity_type})",
            f"File: {filename}",
            f"Fully qualified name: {fqn}",
        ]
        if extends:
            parts.append(f"Extends: {extends}")
        if implements:
            parts.append(f"Implements: {', '.join(implements)}")
        if methods:
            parts.append(f"Methods: {', '.join(methods)}")
        if docblock:
            parts.append(f"Description: {docblock}")

        text = "\n".join(parts)

        chunks.append(_make_chunk(
            text        = text,
            chunk_type  = "class_summary",
            source_file = filepath,
            metadata    = {
                "class_name":  name,
                "entity_type": entity_type,
                "filename":    filename,
                "framework":   cm.framework.value,
                "module_id":   _module_id_for_file(filepath, file_module or {}),
            },
        ))

    return chunks


# ── Chunk Type 7: DB Schema (migrations) ──────────────────────────────────────

def _chunks_db_schema(cm: CodeMap, file_module: dict[str, str] | None = None) -> list[dict]:
    """
    One chunk per migration/schema operation.
    Describes table structure with column names and types.
    """
    chunks = []

    for schema in cm.db_schema:
        operation = schema.get("operation", "create")
        table     = schema.get("table", "unknown")
        columns   = schema.get("columns", [])
        filepath  = schema.get("file", "")
        filename  = Path(filepath).name

        base_meta = {
            "table_name":    table,
            "sql_operation": operation.upper(),
            "involves_db":   "true",
            "filename":      filename,
            "framework":     cm.framework.value,
            "module_id":     _module_id_for_file(filepath, file_module or {}),
            "source_file":   filepath,
        }

        if columns and len(columns) > COLS_PER_SCHEMA_PART:
            # Wide table — semantic split into batches of COLS_PER_SCHEMA_PART
            chunks.extend(_split_db_schema(
                table      = table,
                operation  = operation,
                filename   = filename,
                columns    = columns,
                base_meta  = base_meta,
            ))
        else:
            # Narrow table — single chunk (normal path)
            col_lines = []
            for col in (columns or []):
                col_name = col.get("name", "?")
                col_type = col.get("type", "?")
                mods     = col.get("modifiers", [])
                mod_str  = f" [{', '.join(mods)}]" if mods else ""
                col_lines.append(f"  {col_name}: {col_type}{mod_str}")

            parts = [
                f"Database Schema: {operation.upper()} TABLE '{table}'",
                f"Defined in: {filename}",
            ]
            if col_lines:
                parts.append("Columns:\n" + "\n".join(col_lines))

            chunks.append(_make_chunk(
                text        = "\n".join(parts),
                chunk_type  = "db_schema",
                source_file = filepath,
                metadata    = base_meta,
            ))

    return chunks


# ── Chunk Type 8: Graph Neighbourhoods ────────────────────────────────────────

def _chunks_graph_neighbourhoods(graph: Any, cm: CodeMap) -> list[dict]:
    """
    One chunk per high-degree node in the knowledge graph.
    Describes the node's connections — useful for "what touches table X?"
    queries in Stage 4.

    Uses G.graph["index"] (built by Pass 14 in Stage 2) when available to
    avoid an O(N) full node scan.  Falls back to graph.nodes(data=True) for
    graphs loaded from older runs that pre-date the index.
    """
    import networkx as nx
    chunks = []

    # Prefer the pre-built index for O(1) membership checks; fall back gracefully
    _index = graph.graph.get("index", {})

    # Collect candidate node IDs: prefer index-keyed nodes, else full scan
    if _index:
        # All typed nodes from the index (excludes noise types automatically)
        _candidate_ids: set[str] = set()
        for _key in ("functions", "pages", "scripts", "controllers", "models",
                     "services", "routes", "http_endpoints", "db_tables", "forms"):
            _candidate_ids.update(_index.get(_key, []))
        node_iter = [(nid, graph.nodes[nid]) for nid in _candidate_ids
                     if graph.has_node(nid)]
    else:
        node_iter = list(graph.nodes(data=True))

    # Only generate neighbourhood chunks for nodes with degree >= 2
    for node_id, data in node_iter:
        degree = graph.degree(node_id)
        if degree < 2:
            continue

        node_type = data.get("type", "unknown")
        node_name = data.get("name", node_id)

        # Describe incoming and outgoing connections
        in_edges:  list[str] = []
        out_edges: list[str] = []

        for src, _, edata in graph.in_edges(node_id, data=True):
            etype = edata.get("edge_type", "?")
            src_label = _short_node_label(src)
            in_edges.append(f"  ← {src_label} ({etype})")

        for _, dst, edata in graph.out_edges(node_id, data=True):
            etype = edata.get("edge_type", "?")
            dst_label = _short_node_label(dst)
            out_edges.append(f"  → {dst_label} ({etype})")

        parts = [
            f"Graph node: {node_name} (type={node_type}, degree={degree})",
        ]
        if data.get("file"):
            parts.append(f"Defined in: {data['file']}")
        if in_edges:
            parts.append("Incoming connections:\n" + "\n".join(in_edges))
        if out_edges:
            parts.append("Outgoing connections:\n" + "\n".join(out_edges))

        # For DB tables: generate a special "all files touching this table" summary
        if node_type == "db_table":
            readers = [_short_node_label(s) for s, _, d in graph.in_edges(node_id, data=True)
                       if d.get("edge_type") == "sql_read"]
            writers = [_short_node_label(s) for s, _, d in graph.in_edges(node_id, data=True)
                       if d.get("edge_type") == "sql_write"]
            ddl     = [_short_node_label(s) for s, _, d in graph.in_edges(node_id, data=True)
                       if d.get("edge_type") == "sql_ddl"]
            if readers:
                parts.append(f"Read by: {', '.join(readers)}")
            if writers:
                parts.append(f"Written by: {', '.join(writers)}")
            if ddl:
                parts.append(f"Schema defined in: {', '.join(ddl)}")

        source_file = data.get("file", "")
        base_meta_g = {
            "node_id":    node_id,
            "node_type":  node_type,
            "degree":     str(degree),
            "table_name": node_name if node_type == "db_table" else "",
            "involves_db": str(node_type == "db_table").lower(),
            "framework":  cm.framework.value,
            "module_id":  data.get("module_id", "unknown"),
        }

        if degree > EDGES_SPLIT_THRESHOLD:
            # High-degree node — semantic split into summary + edges sub-chunks
            chunks.extend(_split_graph_neighbourhood(
                node_id     = node_id,
                node_name   = node_name,
                node_type   = node_type,
                degree      = degree,
                source_file = source_file,
                in_edges    = in_edges,
                out_edges   = out_edges,
                db_summary  = [p for p in parts
                               if p.startswith(("Read by:", "Written by:", "Schema defined in:"))],
                base_meta   = base_meta_g,
            ))
        else:
            # Normal node — single chunk
            text = "\n".join(parts)
            chunks.append(_make_chunk(
                text        = text,
                chunk_type  = "graph_neighbourhood",
                source_file = source_file,
                metadata    = base_meta_g,
            ))

    return chunks


# ─── ChromaDB Helpers ──────────────────────────────────────────────────────────

def _make_chroma_client(chroma_path: str):
    """Create a persistent ChromaDB client at the given path."""
    import chromadb
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_path)


def _get_or_create_collection(
    client,
    name: str,
):
    """
    Get existing collection or create a new one with a local
    sentence-transformers embedding function (all-MiniLM-L6-v2).
    Uses cosine distance for semantic similarity.
    No API key required — model weights are downloaded once and cached locally.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    return client.get_or_create_collection(
        name               = name,
        embedding_function = ef,
        metadata           = {"hnsw:space": "cosine"},
    )


def _upsert_chunks(
    collection,
    chunks: list[dict],
) -> int:
    """
    Upsert all chunks into ChromaDB in batches.
    Returns total number of documents stored.

    Deduplicates by ID before batching — ChromaDB raises DuplicateIDError if
    any batch contains two entries with the same ID, even on upsert.  Dupes
    can arise when multiple code-map entries produce identical text for the
    same source file and chunk type.  Last-writer-wins (we keep the last seen
    chunk for each ID so no content is silently dropped).
    """
    # ── Deduplicate by ID (last-writer-wins) ──────────────────────────────────
    seen: dict[str, dict] = {}
    for chunk in chunks:
        seen[chunk["id"]] = chunk
    unique_chunks = list(seen.values())

    n_dupes = len(chunks) - len(unique_chunks)
    if n_dupes:
        print(f"  [stage3] ⚠️  Deduplicated {n_dupes} chunk(s) with colliding IDs "
              f"({len(unique_chunks)} unique chunks remain)")

    total = 0
    for i in range(0, len(unique_chunks), CHUNK_BATCH_SIZE):
        batch = unique_chunks[i : i + CHUNK_BATCH_SIZE]
        collection.upsert(
            ids        = [c["id"]       for c in batch],
            documents  = [c["text"]     for c in batch],
            metadatas  = [c["metadata"] for c in batch],
        )
        total += len(batch)
        print(f"  [stage3] Upserted {total}/{len(unique_chunks)} chunks ...", end="\r")

    print()  # newline after progress
    return total


def _try_load_collection(
    chroma_path: str,
    name: str,
    for_query: bool = False,
):
    """
    Try to load an existing ChromaDB collection.

    Args:
        for_query: If True, loads with the real SentenceTransformer EF so
                   .query() calls work correctly (used by query_collection
                   and the preflight sanity check). If False (default), uses
                   a no-op EF safe for .count()/.get() only — avoids loading
                   the model on pipeline resume.
    """
    try:
        import chromadb as _chroma
        client = _chroma.PersistentClient(path=chroma_path)

        if for_query:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        else:
            class _NoOpEF:
                def __call__(self, input):  # noqa: A002
                    return [[0.0] * 384] * len(input)
            ef = _NoOpEF()

        return client.get_collection(name=name, embedding_function=ef)
    except Exception:
        return None


# ─── Chunk Factory ─────────────────────────────────────────────────────────────

def _make_chunk(
    text:        str,
    chunk_type:  str,
    source_file: str,
    metadata:    dict[str, str],
) -> dict:
    """
    Build a chunk dict with a stable deterministic ID.

    ID is derived from chunk_type + source_file + first 80 chars of text,
    so re-running the pipeline produces the same IDs (enabling upsert idempotency).
    """
    # Stable ID: hash type + file + full text.
    # Using text[:80] caused collisions when multiple chunks of the same type
    # came from the same file with identical openings (e.g. repeated SQL ops,
    # form fields, or class summaries that all start with the same boilerplate).
    # Hashing the full text guarantees uniqueness as long as content differs.
    id_seed  = f"{chunk_type}::{source_file}::{text}"
    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_seed))

    # Ensure all metadata values are strings (ChromaDB requirement)
    safe_meta = {
        "chunk_type":  chunk_type,
        "source_file": source_file,
        **{k: str(v) for k, v in metadata.items()},
    }

    # Prepend semantic anchor unless the text already starts with one
    # (split helpers pre-build text with anchor included to control part labels)
    anchor_prefix = chunk_type.upper() + " |"
    if not text.lstrip().startswith(anchor_prefix):
        anchor = _make_anchor(chunk_type, safe_meta)
        text   = anchor + "\n" + text.strip()
    else:
        text = text.strip()

    return {
        "id":       chunk_id,
        "text":     text,
        "metadata": safe_meta,
    }


# ─── Query Helper (used by Stage 4+) ──────────────────────────────────────────

def query_collection(
    ctx: PipelineContext,
    query: str,
    n_results: int = 8,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search over the embedded codebase.

    Args:
        ctx:       Pipeline context with embedding_meta populated.
        query:     Natural language query string.
        n_results: Number of results to return.
        where:     Optional ChromaDB metadata filter dict.
                   e.g. {"involves_db": "true"}
                   e.g. {"chunk_type": "sql_operation"}

    Returns:
        List of result dicts with keys: text, metadata, distance, id.

    Example:
        results = query_collection(ctx, "pages that handle user registration")
        results = query_collection(ctx, "tables modified during checkout",
                                   where={"is_write": "true"})
    """
    if ctx.embedding_meta is None:
        raise RuntimeError("[stage3] embedding_meta is None — run Stage 3 first.")

    collection = _try_load_collection(
        ctx.embedding_meta.chroma_path,
        ctx.embedding_meta.collection_name,
        for_query=True,
    )
    if collection is None:
        raise RuntimeError(
            f"[stage3] ChromaDB collection '{ctx.embedding_meta.collection_name}' "
            f"not found at {ctx.embedding_meta.chroma_path}"
        )

    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results":   min(n_results, collection.count()),
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # Flatten ChromaDB's nested list response into a clean list of dicts
    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "text":     doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "id":       results["ids"][0][i],
        })

    return output


# ─── Misc Helpers ──────────────────────────────────────────────────────────────

def _assert_openai_key() -> None:
    """No-op — using local sentence-transformers, no API key needed."""
    pass


def _load_graph_safe(gpickle_path: str) -> Any | None:
    """Load graph from pickle, returning None on any error."""
    try:
        with open(gpickle_path, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        print(f"  [stage3] Warning: could not load graph ({e}) — skipping graph chunks")
        return None


def _extract_sql_fields(sql: str, operation: str) -> list[str]:
    """
    Extract column names from SQL string for INSERT and SELECT.
    Returns empty list if fields can't be parsed.
    """
    fields: list[str] = []

    if operation == "INSERT":
        # INSERT INTO table(col1, col2, ...) VALUES(...)
        match = re.search(r'\(\s*([^)]+)\s*\)\s*values', sql, re.IGNORECASE)
        if match:
            raw = match.group(1)
            fields = [f.strip().strip('`\'"') for f in raw.split(',')]

    elif operation == "SELECT":
        # SELECT col1, col2 FROM ...
        match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if match:
            cols = match.group(1).strip()
            if cols != '*':
                fields = [f.strip().strip('`\'"') for f in cols.split(',')]

    return [f for f in fields if f and len(f) < 64]  # filter noise


def _infer_form_purpose(field_names: list[str]) -> str:
    """
    Heuristically infer the purpose of a form from its field names.
    Used to add plain-language descriptions to form_input chunks.
    """
    fields_lower = {f.lower() for f in field_names}

    if fields_lower & {"uname", "username", "email", "password", "pwd", "passwd"}:
        if fields_lower & {"fname", "lname", "phone", "dob", "gender"}:
            return "User registration form — collects personal details and creates an account"
        return "User login form — authenticates an existing user"

    if fields_lower & {"cc", "dc", "doi", "dor", "carid", "distance"}:
        return "Car rental booking form — collects booking dates, car selection, and charges"

    if fields_lower & {"card", "cvv", "expiry", "payment", "amount"}:
        return "Payment form — processes financial transaction"

    if fields_lower & {"search", "query", "keyword", "q"}:
        return "Search form"

    if fields_lower & {"address", "city", "state", "country", "zip", "postal"}:
        return "Address / shipping details form"

    return ""


def _short_node_label(node_id: str) -> str:
    """Shorten a graph node ID for use in chunk text."""
    if node_id.startswith("TABLE:"):    return f"table '{node_id[6:]}'"
    if node_id.startswith("REDIRECT:"): return f"→{node_id[9:]}"
    if node_id.startswith("ROUTE:"):    return node_id[6:]
    if "::" in node_id:
        file_part, fn_part = node_id.rsplit("::", 1)
        return f"{Path(file_part).name}::{fn_part}"
    return Path(node_id).name or node_id


def _save_manifest(chunks: list[dict], path: str) -> None:
    """
    Save a lightweight manifest of all chunks (without embeddings)
    for debugging and inspection.
    """
    manifest = [
        {
            "id":         c["id"],
            "chunk_type": c["metadata"]["chunk_type"],
            "source":     c["metadata"]["source_file"],
            "text_len":   len(c["text"]),
            "preview":    c["text"][:120].replace("\n", " "),
        }
        for c in chunks
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


# ─── Update context._from_dict (called externally to restore embedding_meta) ──

def _load_embedding_meta(chroma_path: str) -> EmbeddingMeta:
    """Reconstruct EmbeddingMeta from an existing ChromaDB path."""
    collection = _try_load_collection(chroma_path, COLLECTION_NAME)
    count = collection.count() if collection else 0
    return EmbeddingMeta(
        collection_name = COLLECTION_NAME,
        chroma_path     = chroma_path,
        total_chunks    = count,
        embedding_model = EMBEDDING_MODEL,
    )