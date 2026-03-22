"""
SQLite-backed exact-match LLM response cache.

Cache key is sha256(model + system_prompt + user_prompt + temperature +
json_mode + prefill) — any change to the inputs produces a different key,
so cached responses are always semantically equivalent to a fresh call.

Environment variables:
    LLM_CACHE_ENABLED  "1"|"true"|"yes" to enable  (default: disabled)
    LLM_CACHE_PATH     Path to the SQLite DB file
                       Default: .llm_cache.db in the current working directory

Quick-start:
    export LLM_CACHE_ENABLED=1          # enable
    export LLM_CACHE_PATH=/tmp/llm.db   # optional custom location
    python run_pipeline.py ...

The cache is thread-safe (WAL mode + per-operation connection lock).
It stores responses indefinitely — there is no TTL because an identical
input to a deterministic LLM call will always produce an equivalent
(if not byte-for-byte identical) response.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from datetime import datetime

# ── Constants ──────────────────────────────────────────────────────────────────

_ENABLED_VALUES = ("1", "true", "yes")
_DEFAULT_DB     = ".llm_cache.db"

# One lock guards all SQLite access.  The lock is per-process; concurrent
# processes each open the same WAL-mode DB so reads overlap safely.
_lock = threading.Lock()


# ── Public helpers ─────────────────────────────────────────────────────────────

def is_enabled() -> bool:
    """Return True when LLM_CACHE_ENABLED is set to a truthy value."""
    return os.environ.get("LLM_CACHE_ENABLED", "").strip().lower() in _ENABLED_VALUES


def cache_path() -> str:
    """Return the resolved path to the SQLite cache file."""
    return os.environ.get("LLM_CACHE_PATH", _DEFAULT_DB)


def make_key(
    model:         str,
    system_prompt: str,
    user_prompt:   str,
    temperature:   float,
    json_mode:     bool = False,
    prefill:       str  = "",
) -> str:
    """
    Compute a deterministic cache key for the given call parameters.

    All parameters that affect the response are included so that any
    change to the inputs produces a distinct key and never returns a
    stale hit.
    """
    payload = "\x00".join([
        model,
        system_prompt,
        user_prompt,
        str(temperature),
        str(json_mode),
        prefill,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(key: str, label: str = "") -> str | None:
    """
    Look up *key* in the cache.

    Returns the cached response string on a hit, or None on a miss.
    Increments hit_count and prints a notice when returning a cached value.
    No-op (returns None) when the cache is disabled.
    """
    if not is_enabled():
        return None

    with _lock:
        conn = _connect()
        row  = conn.execute(
            "SELECT response FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()

        if row is None:
            conn.close()
            return None

        conn.execute(
            "UPDATE llm_cache SET hit_count = hit_count + 1 WHERE key = ?",
            (key,),
        )
        conn.commit()
        conn.close()

    tag = f"[{label}] " if label else ""
    print(f"  {tag}LLM cache HIT — returning cached response")
    return row[0]


def put(key: str, response: str, label: str = "") -> None:
    """
    Store *response* under *key*.

    Uses INSERT OR REPLACE so re-running the pipeline on the same inputs
    refreshes the entry rather than failing on a duplicate key.
    No-op when the cache is disabled.
    """
    if not is_enabled():
        return

    with _lock:
        conn = _connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache
                (key, response, stage, created_at, hit_count)
            VALUES (?, ?, ?, ?, 0)
            """,
            (key, response, label, datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()


def stats() -> dict:
    """
    Return a summary dict: {enabled, entries, total_hits, db_path}.
    Useful for end-of-run reporting.
    """
    if not is_enabled():
        return {"enabled": False}

    with _lock:
        conn    = _connect()
        entries = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        hits    = conn.execute(
            "SELECT COALESCE(SUM(hit_count), 0) FROM llm_cache"
        ).fetchone()[0]
        conn.close()

    return {
        "enabled":    True,
        "entries":    entries,
        "total_hits": hits,
        "db_path":    cache_path(),
    }


def clear() -> int:
    """Delete all cached entries.  Returns the number of rows deleted."""
    if not is_enabled():
        return 0
    with _lock:
        conn    = _connect()
        deleted = conn.execute("DELETE FROM llm_cache").rowcount
        conn.commit()
        conn.close()
    return deleted


# ── Private ────────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """
    Open (and initialise if needed) the SQLite cache database.

    Called inside _lock so the CREATE TABLE IF NOT EXISTS is race-free.
    WAL mode allows concurrent readers without blocking each other.
    """
    conn = sqlite3.connect(cache_path(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            key        TEXT    PRIMARY KEY,
            response   TEXT    NOT NULL,
            stage      TEXT    NOT NULL DEFAULT '',
            created_at TEXT    NOT NULL,
            hit_count  INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn
