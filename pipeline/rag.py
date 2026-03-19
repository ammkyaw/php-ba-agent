"""
pipeline/rag.py — BM25-style Code Chunk Retriever

For large codebases where the full static analysis exceeds the LLM context
window, this module provides targeted retrieval so each stage only receives
the most relevant code chunks for its specific questions.

Implementation: SQLite FTS5 with unicode61 tokenizer (BM25 ranking built-in).
No external embedding model required — faster and deterministic.

Usage:
    from pipeline.rag import CodeChunkIndex

    # Build the index once per run
    idx = CodeChunkIndex(output_dir="/path/to/outputs/run_xxx")
    idx.build(ctx)          # indexes classes, functions, routes, entities, tables

    # Query per stage
    chunks = idx.query("user authentication login password", top_k=8)
    # Returns list of dicts: {type, name, path, snippet}

    # Convenience: format as prompt context block
    context_block = idx.format_context("authentication and session management", top_k=6)

Environment variables:
    RAG_ENABLED   "1"|"true"|"yes" to enable (default: disabled for speed)
                  Enable when codebase is large (>200 PHP files) or when
                  prompts exceed the model's context window.
    RAG_TOP_K     Number of chunks to retrieve per query (default: 8)
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any


_ENABLED_VALUES = ("1", "true", "yes")
_DEFAULT_TOP_K  = 8


def is_enabled() -> bool:
    return os.environ.get("RAG_ENABLED", "").strip().lower() in _ENABLED_VALUES


def get_top_k() -> int:
    return int(os.environ.get("RAG_TOP_K", str(_DEFAULT_TOP_K)) or str(_DEFAULT_TOP_K))


class CodeChunkIndex:
    """
    SQLite FTS5 index of PHP code chunks.

    Chunk types indexed:
      class      — PHP class with method list and SQL ops
      function   — top-level function with call list
      route      — HTTP route → controller mapping
      entity     — domain entity with attribute list
      table      — DB table with column list
      invariant  — business rule / invariant from stage29
    """

    def __init__(self, output_dir: str):
        self._db_path = str(Path(output_dir) / "rag_index.db")
        self._conn: sqlite3.Connection | None = None

    # ── Public API ──────────────────────────────────────────────────────────────

    def build(self, ctx: Any) -> int:
        """
        Index all code chunks from the pipeline context.
        Returns the total number of chunks indexed.
        """
        conn = self._connect()
        conn.execute("DELETE FROM code_chunks")  # rebuild from scratch each run

        chunks = list(self._extract_chunks(ctx))
        conn.executemany(
            "INSERT INTO code_chunks (id, chunk_type, name, path, content) VALUES (?,?,?,?,?)",
            chunks,
        )
        conn.commit()
        print(f"  [rag] Indexed {len(chunks)} code chunk(s) into FTS5")
        return len(chunks)

    def query(self, text: str, top_k: int | None = None) -> list[dict]:
        """
        Return the top-K most relevant chunks for *text*.
        Each result: {type, name, path, content}.
        Returns empty list if index is empty or query is blank.
        """
        if not text or not text.strip():
            return []
        k = top_k if top_k is not None else get_top_k()
        conn = self._connect()
        fts_q = self._fts_query(text)
        if not fts_q:
            return []
        try:
            rows = conn.execute(
                """SELECT chunk_type, name, path, content
                   FROM code_chunks
                   WHERE code_chunks MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_q, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # empty table or syntax error
        return [
            {"type": r[0], "name": r[1], "path": r[2], "content": r[3]}
            for r in rows
        ]

    def format_context(self, query: str, top_k: int | None = None, header: str = "") -> str:
        """
        Query and format retrieved chunks as a prompt-ready context block.
        Returns empty string if nothing retrieved.
        """
        chunks = self.query(query, top_k)
        if not chunks:
            return ""
        lines = [header or f"## Relevant code context for: {query[:60]}"]
        for c in chunks:
            label = f"[{c['type'].upper()}] {c['name']}"
            if c["path"]:
                label += f"  ({c['path']})"
            lines.append(f"\n{label}")
            lines.append(c["content"])
        return "\n".join(lines)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Chunk extraction ────────────────────────────────────────────────────────

    def _extract_chunks(self, ctx: Any):
        """Yield (id, type, name, path, content) tuples from context."""
        seen: set[str] = set()

        yield from self._chunks_from_code_map(ctx, seen)
        yield from self._chunks_from_routes(ctx, seen)
        yield from self._chunks_from_entities(ctx, seen)
        yield from self._chunks_from_tables(ctx, seen)
        yield from self._chunks_from_invariants(ctx, seen)

    def _chunks_from_code_map(self, ctx: Any, seen: set):
        # CodeMap is a dataclass with flat list fields, not a path-keyed dict.
        # Each list item is a dict with fields like 'name', 'file', 'methods', etc.
        cm = getattr(ctx, "code_map", None)
        if cm is None:
            return

        # Classes, controllers, models, services all share the same dict shape
        all_class_lists = [
            getattr(cm, "classes",     None) or [],
            getattr(cm, "controllers", None) or [],
            getattr(cm, "models",      None) or [],
            getattr(cm, "services",    None) or [],
        ]
        for cls_list in all_class_lists:
            for cls_info in cls_list:
                if not isinstance(cls_info, dict):
                    continue
                cls_name = cls_info.get("name") or cls_info.get("class_name", "")
                path     = cls_info.get("file") or cls_info.get("path", "")
                if not cls_name:
                    continue
                cid = f"class:{path}:{cls_name}"
                if cid in seen:
                    continue
                seen.add(cid)
                # methods may be a list of strings or a dict keyed by method name
                raw_methods = cls_info.get("methods") or []
                if isinstance(raw_methods, dict):
                    methods = list(raw_methods.keys())[:15]
                else:
                    methods = [m if isinstance(m, str) else m.get("name", "") for m in raw_methods[:15]]
                extends    = cls_info.get("extends", "") or cls_info.get("parent", "")
                implements = ", ".join(cls_info.get("implements") or [])
                content = (
                    f"class {cls_name}"
                    + (f" extends {extends}" if extends else "")
                    + (f" implements {implements}" if implements else "")
                    + (f". Methods: {', '.join(str(m) for m in methods if m)}." if methods else ".")
                )
                yield (cid, "class", cls_name, path, content)

        # Standalone / global functions
        for fn_info in (getattr(cm, "functions", None) or []):
            if not isinstance(fn_info, dict):
                continue
            fn_name = fn_info.get("name") or fn_info.get("function_name", "")
            path    = fn_info.get("file") or fn_info.get("path", "")
            if not fn_name:
                continue
            cid = f"fn:{path}:{fn_name}"
            if cid in seen:
                continue
            seen.add(cid)
            calls = [str(c) for c in (fn_info.get("calls") or [])[:6]]
            content = (
                f"function {fn_name}" + (f" in {path}" if path else "")
                + (f". Calls: {', '.join(calls)}." if calls else ".")
            )
            yield (cid, "function", fn_name, path, content)

    def _chunks_from_routes(self, ctx: Any, seen: set):
        cm = getattr(ctx, "code_map", ctx)  # ctx may be CodeMap or PipelineContext
        for route in (getattr(cm, "routes", None) or []):
            if not isinstance(route, dict):
                continue
            uri  = route.get("uri") or route.get("path", "")
            verb = (route.get("method") or route.get("verb", "GET")).upper()
            ctrl = route.get("controller", "")
            act  = route.get("action") or route.get("method_name", "")
            if not uri:
                continue
            cid = f"route:{verb}:{uri}"
            if cid in seen:
                continue
            seen.add(cid)
            content = f"Route {verb} {uri}"
            if ctrl:
                content += f" handled by {ctrl}::{act}"
            yield (cid, "route", uri, "", content)

    def _chunks_from_entities(self, ctx: Any, seen: set):
        domain = getattr(ctx, "domain_model", None)
        if domain is None:
            return
        for e in (getattr(domain, "entities", None) or []):
            name  = (e.get("name") if isinstance(e, dict) else getattr(e, "name", "")) or ""
            attrs = (e.get("attributes") if isinstance(e, dict) else getattr(e, "attributes", [])) or []
            if not name:
                continue
            cid = f"entity:{name}"
            if cid in seen:
                continue
            seen.add(cid)
            attr_names = [
                (a.get("name") if isinstance(a, dict) else str(a)) for a in attrs[:10]
            ]
            content = f"Domain entity {name}. Attributes: {', '.join(str(a) for a in attr_names)}."
            yield (cid, "entity", name, "", content)

    def _chunks_from_tables(self, ctx: Any, seen: set):
        cm   = getattr(ctx, "code_map", ctx)
        cols = getattr(cm, "table_columns", None) or []
        for tbl in cols:
            if not isinstance(tbl, dict):
                continue
            tname = tbl.get("table", "")
            if not tname:
                continue
            cid = f"table:{tname}"
            if cid in seen:
                continue
            seen.add(cid)
            col_names = [c.get("column", "") for c in (tbl.get("columns") or [])[:15]]
            content = f"Database table {tname}. Columns: {', '.join(col_names)}."
            yield (cid, "table", tname, "", content)

    def _chunks_from_invariants(self, ctx: Any, seen: set):
        cm = getattr(ctx, "code_map", ctx)
        for inv in (getattr(cm, "invariants", None) or getattr(ctx, "invariants", None) or []):
            if not isinstance(inv, dict):
                continue
            iid  = inv.get("id", "")
            desc = inv.get("description", inv.get("rule", ""))[:200]
            if not desc:
                continue
            cid = f"inv:{iid or desc[:40]}"
            if cid in seen:
                continue
            seen.add(cid)
            content = f"Business invariant {iid}: {desc}"
            yield (cid, "invariant", iid or "rule", "", content)

    # ── SQLite helpers ──────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS code_chunks USING fts5(
                    id       UNINDEXED,
                    chunk_type UNINDEXED,
                    name,
                    path,
                    content,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
            """)
            self._conn.commit()
        return self._conn

    @staticmethod
    def _fts_query(text: str) -> str:
        """
        Convert free text to an FTS5 query.
        Extracts alphanumeric tokens, deduplicates, joins with OR.
        Strips camelCase into separate tokens (loginUser → login user).
        """
        # Split camelCase
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        tokens = list(dict.fromkeys(re.findall(r"[a-zA-Z0-9_]{2,}", text)))
        # Escape FTS5 special chars inside tokens
        safe = [t.replace('"', '""') for t in tokens[:30]]
        return " OR ".join(f'"{t}"' for t in safe) if safe else ""
