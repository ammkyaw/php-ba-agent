"""
pipeline/stage35_entities.py — Entity Extraction (Stage 3.5)

Builds a catalogue of every business entity (DB table / ORM model) found
in the codebase.  Fully static — no LLM.

Sources
-------
1. code_map.table_columns — strongest: actual schema with column metadata
2. code_map.db_schema     — schema entries that may not have column detail
3. code_map.sql_queries   — discovers tables not in schema (dynamic DDL, etc.)
4. PHP source scan        — ``$table = 'name'`` / ``protected $table`` ORM
                            property → ties PHP class to a table name

Flags
-----
is_system  Tables matching known infrastructure patterns (log_*, config*,
           upgrade_*, audit_*, sugar_*, jobqueue*, tracker*, cache*).

is_pivot   Tables with exactly 2 columns that are both FK-pattern (*_id)
           and no other meaningful columns — junction tables for N:M.

is_core    Populated as False initially; Stage 3.6 sets it to True on any
           entity that gets at least one relationship. Stage 3.7 also marks
           entities that have a state machine.  Stage 3.5 pre-marks entities
           that appear in action clusters.

Output
------
    EntityCollection   (written to 4.1_entities/entity_catalog.json)
    ctx.entities set on context

Consumed by
-----------
    Stage 3.6  (relationship reconstruction — needs entity list)
    Stage 4    (domain model grounding)
    Stage 3.7  (state machine entity labelling)
    Stage 4.5  (flow validation — entity existence checks)
    Stage 6.7  (ER diagram node set)
    Stage 9    (knowledge graph entity nodes)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from context import Entity, EntityCollection, EntityColumn, PipelineContext

# ── Tunables ──────────────────────────────────────────────────────────────────
MIN_COLUMN_COUNT_FOR_PIVOT = 2   # pivot tables have at most this many non-id cols

# ── System / infrastructure table patterns ───────────────────────────────────
_SYSTEM_TABLE_RE = re.compile(
    r"^("
    r"log_|audit_|sugar_|cache_|tracker_|jobqueue|upgrade_|repair_|inbound_email_autoreply"
    r"|acl_roles|acl_fields|acl_actions"   # ACL tables are infrastructure in SugarCRM
    r"|config|version|metadata|history_|activities|"
    r"securitygroup|sugar_portal|sugar_feed|sugar_user_theme"
    r")",
    re.IGNORECASE,
)

# ── PHP ORM $table property patterns ─────────────────────────────────────────
# protected $table = 'orders';   OR   public static $table = 'orders';
_PHP_TABLE_PROP_RE = re.compile(
    r"""(?:protected|public|private)?\s*(?:static\s+)?\$table\s*=\s*['"](\w+)['"]""",
)
# class Foo extends Model  or  class Foo extends SugarBean  → entity name hint
_PHP_CLASS_RE = re.compile(
    r"""class\s+(\w+)\s+extends\s+\w+""",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _humanize(table: str) -> str:
    """
    Convert snake_case table name to a human-readable entity name.
    'aos_quotes_items' → 'Aos Quotes Items'   (later LLM cleans this up)
    'accounts'         → 'Account'
    """
    name = table.replace("_", " ").title()
    # simple singularisation of last word (very rough)
    words = name.split()
    if words:
        last = words[-1]
        if last.endswith("ies"):
            words[-1] = last[:-3] + "y"
        elif last.endswith("sses") or last.endswith("xes") or last.endswith("ches"):
            words[-1] = last[:-2]
        elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
            words[-1] = last[:-1]
    return " ".join(words)


def _is_system_table(table: str) -> bool:
    return bool(_SYSTEM_TABLE_RE.match(table))


def _looks_like_fk_column(col_name: str) -> bool:
    return col_name.endswith("_id") or col_name.endswith("_key")


def _is_pivot_table(columns: list[EntityColumn]) -> bool:
    """
    A pivot/junction table has exactly 2 FK-pattern columns and ≤ 1 other
    meaningful column (e.g. a sort_order or created_at).
    """
    fk_cols  = [c for c in columns if _looks_like_fk_column(c.name) or c.is_foreign_key]
    non_meta = [
        c for c in columns
        if c.name not in ("id", "date_entered", "date_modified", "created_by",
                          "modified_user_id", "deleted", "sort_order", "created_at",
                          "updated_at")
        and not _looks_like_fk_column(c.name)
        and not c.is_primary_key
    ]
    return len(fk_cols) >= 2 and len(non_meta) <= 1


def _assign_context(table: str, ac_table_map: dict[str, str]) -> str:
    return ac_table_map.get(table.lower(), "")


# ── Column extraction from table_columns ─────────────────────────────────────

def _columns_from_table_columns(
    table_columns: list[dict],
    target_table: str,
) -> list[EntityColumn]:
    """Extract EntityColumn list for one table from code_map.table_columns."""
    cols: list[EntityColumn] = []
    for entry in table_columns:
        tbl = (entry.get("table") or "").lower()
        if tbl != target_table.lower():
            continue
        col_name = entry.get("column") or entry.get("name") or ""
        if not col_name:
            continue
        col_type = entry.get("type") or entry.get("data_type") or ""
        nullable  = entry.get("nullable", True)
        # FK detection from explicit REFERENCES or convention
        fk_ref   = entry.get("references") or entry.get("foreign_key") or ""
        ref_tbl  = entry.get("references_table") or ""
        ref_col  = entry.get("references_column") or ""
        if fk_ref and not ref_tbl:
            # parse "table(column)" format if present
            m = re.match(r"(\w+)\((\w+)\)", fk_ref)
            if m:
                ref_tbl, ref_col = m.group(1), m.group(2)
        is_fk = bool(fk_ref or ref_tbl or _looks_like_fk_column(col_name))
        is_pk = bool(entry.get("primary_key") or entry.get("is_primary") or col_name == "id")
        cols.append(EntityColumn(
            name              = col_name,
            data_type         = str(col_type),
            nullable          = bool(nullable),
            is_primary_key    = is_pk,
            is_foreign_key    = is_fk,
            references_table  = ref_tbl,
            references_column = ref_col,
        ))
    return cols


# ── PHP source scan for ORM table bindings ────────────────────────────────────

def _scan_php_for_table_bindings(php_project_path: str) -> dict[str, str]:
    """
    Scan PHP source for ``$table = 'name'`` ORM property.
    Returns {table_name: source_file_path}.
    """
    bindings: dict[str, str] = {}
    try:
        for php_file in Path(php_project_path).rglob("*.php"):
            try:
                src = php_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in _PHP_TABLE_PROP_RE.finditer(src):
                tbl = m.group(1).lower()
                if tbl and tbl not in bindings:
                    bindings[tbl] = str(php_file)
    except Exception:
        pass
    return bindings


# ── Main entry point ──────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    cm = getattr(ctx, "code_map", None)
    if cm is None:
        print("  [stage35] ⚠️  No code_map — skipping entity extraction.")
        ctx.entities = EntityCollection()
        return

    # Build action-cluster table → context lookup
    ac_table_map: dict[str, str] = {}   # table_lower → cluster_name
    ac_cluster_tables: set[str]  = set()
    if ctx.action_clusters:
        for cluster in ctx.action_clusters.clusters:
            for tbl in cluster.tables:
                key = tbl.lower()
                ac_table_map[key] = cluster.name
                ac_cluster_tables.add(key)

    # ── Collect all tables ─────────────────────────────────────────────────
    all_tables: set[str] = set()

    # From table_columns
    for col_entry in (cm.table_columns or []):
        tbl = (col_entry.get("table") or "").lower()
        if tbl:
            all_tables.add(tbl)

    # From db_schema
    for schema_entry in (cm.db_schema or []):
        tbl = (schema_entry.get("table") or schema_entry.get("name") or "").lower()
        if tbl:
            all_tables.add(tbl)

    # From sql_queries (FROM / table field)
    for q in (cm.sql_queries or []):
        tbl = (q.get("table") or "").lower()
        if tbl and tbl not in ("unknown", ""):
            all_tables.add(tbl)

    # PHP ORM scan
    orm_bindings = _scan_php_for_table_bindings(ctx.php_project_path)
    all_tables.update(orm_bindings.keys())

    print(f"  [stage35] {len(all_tables)} unique table(s) discovered.")

    # ── Build Entity objects ───────────────────────────────────────────────
    entities: list[Entity] = []
    seq = 1

    for table in sorted(all_tables):
        if not table or len(table) < 2:
            continue

        # Columns
        columns = _columns_from_table_columns(cm.table_columns or [], table)

        # Primary key heuristic: explicit > 'id' column > first column
        pk = ""
        for c in columns:
            if c.is_primary_key:
                pk = c.name
                break
        if not pk and any(c.name == "id" for c in columns):
            pk = "id"
        elif not pk and columns:
            pk = columns[0].name

        is_system  = _is_system_table(table)
        is_pivot   = _is_pivot_table(columns) if columns else False
        in_cluster = table in ac_cluster_tables

        # is_core: pre-mark by cluster membership (stage36 will update further)
        is_core = in_cluster and not is_system

        bounded_context = _assign_context(table, ac_table_map)
        if not bounded_context:
            # Fallback: strip common prefix words for a readable context
            parts = table.split("_")
            bounded_context = parts[0].title() if parts else table.title()

        # Source file: ORM binding if known
        source_files = [orm_bindings[table]] if table in orm_bindings else []

        entities.append(Entity(
            entity_id       = f"ent_{seq:03d}",
            name            = _humanize(table),
            table           = table,
            bounded_context = bounded_context,
            columns         = columns,
            primary_key     = pk,
            is_pivot        = is_pivot,
            is_core         = is_core,
            is_system       = is_system,
            source_files    = source_files,
            confidence      = 1.0,
        ))
        seq += 1

    core_count = sum(1 for e in entities if e.is_core)
    total      = len(entities)

    collection = EntityCollection(
        entities   = entities,
        total      = total,
        core_count = core_count,
    )

    # ── Print summary ─────────────────────────────────────────────────────
    system_count = sum(1 for e in entities if e.is_system)
    pivot_count  = sum(1 for e in entities if e.is_pivot)
    print(f"  [stage35] Entity Extraction complete — {total} entities "
          f"(core={core_count}, system={system_count}, pivot={pivot_count}).")
    for e in [x for x in entities if x.is_core][:20]:
        print(f"    {e.entity_id}  {e.name:<30} table={e.table}  cols={len(e.columns)}")
    if core_count > 20:
        print(f"    ... and {core_count - 20} more core entities")
    print()

    ctx.entities = collection

    # ── Persist ───────────────────────────────────────────────────────────
    try:
        import dataclasses
        out_path = ctx.output_path("entity_catalog.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(collection), fh, indent=2, ensure_ascii=False)
        print(f"  [stage35] Saved → {out_path}")
    except Exception as exc:
        print(f"  [stage35] ⚠️  Could not save entity_catalog.json: {exc}")
