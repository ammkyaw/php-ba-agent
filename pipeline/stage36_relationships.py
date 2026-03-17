"""
pipeline/stage36_relationships.py — Entity Relationship Reconstruction (Stage 3.6)

Detects relationships between entities and their cardinality.
Fully static — no LLM.

Signal sources (highest → lowest confidence)
--------------------------------------------
A  Explicit FK annotation in table_columns  REFERENCES clause     conf 0.95
B  Pivot table detection (2 FK cols, ≤1 non-meta col)             conf 0.90
C  ORM relationship methods in PHP source                          conf 0.85
   hasMany / has_many / HasMany
   hasOne  / has_one  / HasOne
   belongsTo / belongs_to / BelongsTo
   belongsToMany / hasManyThrough
   SugarCRM: $this->load_relationship / VarDef 'relate' / 'link'
D  SQL JOIN analysis                                               conf 0.80
E  Column name pattern  — orders.customer_id → customers.id        conf 0.75

Cardinality rules
-----------------
  FK in child table (no UNIQUE on FK col)  → parent 1:N child
  UNIQUE constraint on FK col              → parent 1:1 child
  ORM hasMany / hasOne  → follows declaration semantics
  Pivot table           → N:M

Deduplication
-------------
Same (from, to) pair collapses into one entry.
All detected signals are merged into the `signals` list.
Highest-confidence signal wins for confidence / cardinality / rel_type.

Output
------
    EntityRelationshipCollection  → 4.2_relationships/relationship_catalog.json
    Mermaid erDiagram             → 4.2_relationships/er_diagram.mmd
    ctx.relationships set on context
    ctx.entities updated — is_core=True for any entity with ≥1 relationship

Consumed by
-----------
    Stage 4    (domain model grounding — entity list + relationships)
    Stage 3.7  (state machine entity name enrichment)
    Stage 4.5  (flow validation)
    Stage 6.7  (ER diagram)
    Stage 9    (knowledge graph edges)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from context import (
    EntityRelationship,
    EntityRelationshipCollection,
    PipelineContext,
)

# ── Tunables ──────────────────────────────────────────────────────────────────
CONF_FK_EXPLICIT   = 0.95
CONF_PIVOT         = 0.90
CONF_ORM           = 0.85
CONF_SQL_JOIN      = 0.80
CONF_COL_PATTERN   = 0.75

# ── SQL JOIN parser ───────────────────────────────────────────────────────────
# JOIN orders o ON o.customer_id = customers.id
# JOIN orders ON orders.customer_id = c.id
_JOIN_RE = re.compile(
    r"JOIN\s+[`'\"]?(\w+)[`'\"]?\s+(?:\w+\s+)?ON\s+"
    r"[`'\"]?\w+[`'\"]?\.([`'\"]?(\w+)[`'\"]?)\s*=\s*[`'\"]?\w+[`'\"]?\.([`'\"]?(\w+)[`'\"]?)",
    re.IGNORECASE,
)
# Simpler: JOIN orders USING (customer_id)
_JOIN_USING_RE = re.compile(
    r"JOIN\s+[`'\"]?(\w+)[`'\"]?\s+USING\s*\(\s*[`'\"]?(\w+)[`'\"]?\s*\)",
    re.IGNORECASE,
)
# FROM clause — capture primary table
_FROM_RE = re.compile(r"FROM\s+[`'\"]?(\w+)[`'\"]?", re.IGNORECASE)

# ── ORM relationship patterns ─────────────────────────────────────────────────
# Laravel-style: $this->hasMany(Order::class)  or  hasMany('orders')
_ORM_HAS_MANY_RE    = re.compile(
    r"""(?:hasMany|has_many|HasMany)\s*\(\s*['"]?(\w+)['"]?""",
    re.IGNORECASE,
)
_ORM_HAS_ONE_RE     = re.compile(
    r"""(?:hasOne|has_one|HasOne)\s*\(\s*['"]?(\w+)['"]?""",
    re.IGNORECASE,
)
_ORM_BELONGS_TO_RE  = re.compile(
    r"""(?:belongsTo|belongs_to|BelongsTo)\s*\(\s*['"]?(\w+)['"]?""",
    re.IGNORECASE,
)
_ORM_BTM_RE         = re.compile(
    r"""(?:belongsToMany|manyToMany|HasManyThrough|hasManyThrough)\s*\(\s*['"]?(\w+)['"]?""",
    re.IGNORECASE,
)
# SugarCRM VarDef 'type' => 'relate' or 'link', 'module' => 'Accounts'
_SUGAR_RELATE_RE    = re.compile(
    r"""'type'\s*=>\s*'(?:relate|link)'\s*(?:.*?)'module'\s*=>\s*'(\w+)'""",
    re.IGNORECASE | re.DOTALL,
)
# $this->load_relationship('account_id')  or  load_relationship('contacts')
_SUGAR_LOAD_REL_RE  = re.compile(
    r"""load_relationship\s*\(\s*['"](\w+)['"]""",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _table_from_class_or_name(raw: str, known_tables: set[str]) -> str:
    """
    Try to map a class name / model name to a table name.
    'Order' → 'orders', 'AOS_Quotes' → 'aos_quotes'
    """
    # Try direct lower
    candidate = raw.lower()
    if candidate in known_tables:
        return candidate
    # Try pluralised
    for suffix in ("s", "es"):
        if (candidate + suffix) in known_tables:
            return candidate + suffix
    # Try snake_case (CamelCase → snake_case)
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", raw).lower()
    if snake in known_tables:
        return snake
    for suffix in ("s", "es"):
        if (snake + suffix) in known_tables:
            return snake + suffix
    return ""


def _fk_col_to_table(col_name: str, known_tables: set[str]) -> str:
    """
    'customer_id' → 'customers', 'account_id' → 'accounts'
    Returns "" if no match found.
    """
    if not col_name.endswith("_id"):
        return ""
    base = col_name[:-3]   # strip '_id'
    if base in known_tables:
        return base
    if (base + "s") in known_tables:
        return base + "s"
    if (base + "es") in known_tables:
        return base + "es"
    return ""


def _has_unique_on(col_name: str, table: str, table_columns: list[dict]) -> bool:
    """Return True if column has a UNIQUE constraint declared in table_columns."""
    for entry in table_columns:
        if (entry.get("table") or "").lower() != table.lower():
            continue
        if (entry.get("column") or entry.get("name") or "").lower() != col_name.lower():
            continue
        unique = entry.get("unique") or entry.get("is_unique") or False
        if unique:
            return True
    return False


def _is_pivot(entity_id_or_table: str, entities_by_table: dict) -> bool:
    ent = entities_by_table.get(entity_id_or_table)
    return ent.is_pivot if ent else False


def _merge_rels(raw_list: list[dict]) -> list[dict]:
    """
    Deduplicate by (from_entity, to_entity).
    Keep highest-confidence entry; merge signals & source_files.
    """
    key_map: dict[tuple[str, str], dict] = {}
    for r in raw_list:
        key = (r["from_entity"], r["to_entity"])
        if key not in key_map:
            key_map[key] = dict(r)
            key_map[key]["signals"]      = list(r["signals"])
            key_map[key]["source_files"] = list(r["source_files"])
        else:
            existing = key_map[key]
            if r["confidence"] > existing["confidence"]:
                existing["confidence"] = r["confidence"]
                existing["cardinality"] = r["cardinality"]
                existing["rel_type"]    = r["rel_type"]
                existing["via_column"]  = r.get("via_column") or existing["via_column"]
                existing["via_table"]   = r.get("via_table")  or existing["via_table"]
            for sig in r["signals"]:
                if sig not in existing["signals"]:
                    existing["signals"].append(sig)
            for f in r["source_files"]:
                if f and f not in existing["source_files"]:
                    existing["source_files"].append(f)
    return list(key_map.values())


# ── Signal A: Explicit FK in table_columns ────────────────────────────────────

def _detect_fk_explicit(
    table_columns: list[dict],
    known_tables: set[str],
) -> list[dict]:
    results = []
    for entry in (table_columns or []):
        tbl = (entry.get("table") or "").lower()
        col = (entry.get("column") or entry.get("name") or "").lower()
        ref_tbl = (entry.get("references_table") or "").lower()
        ref_col = (entry.get("references_column") or "").lower()
        fk_raw  = (entry.get("references") or entry.get("foreign_key") or "")
        if fk_raw and not ref_tbl:
            m = re.match(r"(\w+)\((\w+)\)", fk_raw)
            if m:
                ref_tbl, ref_col = m.group(1).lower(), m.group(2).lower()
        if not ref_tbl:
            continue
        if tbl not in known_tables or ref_tbl not in known_tables:
            continue
        results.append({
            "from_entity":  ref_tbl,   # parent (referenced)
            "to_entity":    tbl,        # child (FK owner)
            "cardinality":  "1:1" if _has_unique_on(col, tbl, table_columns) else "1:N",
            "rel_type":     "has_one"   if _has_unique_on(col, tbl, table_columns) else "has_many",
            "via_column":   col,
            "via_table":    "",
            "confidence":   CONF_FK_EXPLICIT,
            "signals":      ["foreign_key"],
            "source_files": [],
        })
    return results


# ── Signal B: Pivot table detection ──────────────────────────────────────────

def _detect_pivot(entities_by_table: dict, known_tables: set[str]) -> list[dict]:
    results = []
    for table, ent in entities_by_table.items():
        if not ent.is_pivot:
            continue
        fk_cols = [c for c in ent.columns if c.name.endswith("_id") or c.is_foreign_key]
        if len(fk_cols) < 2:
            continue
        for col in fk_cols:
            ref_tbl = col.references_table.lower() if col.references_table else _fk_col_to_table(col.name, known_tables)
            if not ref_tbl:
                continue
            # The two FK tables are in N:M via this pivot
            other_fks = [c for c in fk_cols if c != col]
            for other in other_fks:
                other_ref = other.references_table.lower() if other.references_table else _fk_col_to_table(other.name, known_tables)
                if not other_ref or other_ref == ref_tbl:
                    continue
                results.append({
                    "from_entity":  ref_tbl,
                    "to_entity":    other_ref,
                    "cardinality":  "N:M",
                    "rel_type":     "many_to_many",
                    "via_column":   col.name,
                    "via_table":    table,
                    "confidence":   CONF_PIVOT,
                    "signals":      ["pivot_table"],
                    "source_files": [],
                })
    return results


# ── Signal C: ORM relationship scanning ──────────────────────────────────────

def _detect_orm(
    php_project_path: str,
    known_tables: set[str],
    entities_by_table: dict,
) -> list[dict]:
    results = []
    # Build class→table map from entities source files
    class_to_table: dict[str, str] = {}
    for tbl, ent in entities_by_table.items():
        # Assume class name ≈ entity humanized or table CamelCase
        class_to_table[ent.name.replace(" ", "")] = tbl
        class_to_table[tbl.title().replace("_", "")] = tbl

    try:
        for php_file in Path(php_project_path).rglob("*.php"):
            try:
                src = php_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            # Which table does this file's class own?
            own_table = ""
            prop_m = re.search(r"""\$table\s*=\s*['"](\w+)['"]""", src)
            if prop_m:
                own_table = prop_m.group(1).lower()
            if not own_table:
                # Guess from file path: modules/Accounts/Account.php → accounts
                stem = php_file.stem.lower()
                if stem in known_tables:
                    own_table = stem
                elif (stem + "s") in known_tables:
                    own_table = stem + "s"

            if not own_table:
                continue

            def _rel(pattern, rel_t, card, f_path=str(php_file)):
                for m in pattern.finditer(src):
                    raw = m.group(1)
                    related = _table_from_class_or_name(raw, known_tables)
                    if not related or related == own_table:
                        return
                    results.append({
                        "from_entity":  own_table,
                        "to_entity":    related,
                        "cardinality":  card,
                        "rel_type":     rel_t,
                        "via_column":   "",
                        "via_table":    "",
                        "confidence":   CONF_ORM,
                        "signals":      ["orm"],
                        "source_files": [f_path],
                    })

            _rel(_ORM_HAS_MANY_RE,   "has_many",     "1:N")
            _rel(_ORM_HAS_ONE_RE,    "has_one",      "1:1")
            _rel(_ORM_BELONGS_TO_RE, "belongs_to",   "1:N")
            _rel(_ORM_BTM_RE,        "many_to_many", "N:M")

            # SugarCRM VarDef relate/link
            for m in _SUGAR_RELATE_RE.finditer(src):
                raw = m.group(1)
                related = _table_from_class_or_name(raw, known_tables)
                if related and related != own_table:
                    results.append({
                        "from_entity":  own_table,
                        "to_entity":    related,
                        "cardinality":  "1:N",
                        "rel_type":     "has_many",
                        "via_column":   "",
                        "via_table":    "",
                        "confidence":   CONF_ORM,
                        "signals":      ["orm_vardef"],
                        "source_files": [str(php_file)],
                    })
    except Exception:
        pass

    return results


# ── Signal D: SQL JOIN analysis ───────────────────────────────────────────────

def _detect_sql_joins(
    sql_queries: list[dict],
    known_tables: set[str],
) -> list[dict]:
    results = []
    for q in (sql_queries or []):
        sql  = q.get("query") or q.get("sql") or ""
        fpath = q.get("file", "")
        if not sql:
            continue

        # Primary (FROM) table
        from_m = _FROM_RE.search(sql)
        base_table = from_m.group(1).lower() if from_m else ""

        for m in _JOIN_RE.finditer(sql):
            joined_table = m.group(1).lower()
            left_col     = m.group(3).lower()
            right_col    = m.group(5).lower()
            if joined_table not in known_tables:
                continue
            # Determine FK direction by which column ends in _id
            if left_col.endswith("_id"):
                parent = _fk_col_to_table(left_col, known_tables) or base_table
                child  = joined_table
                col    = left_col
            elif right_col.endswith("_id"):
                parent = _fk_col_to_table(right_col, known_tables) or joined_table
                child  = base_table
                col    = right_col
            else:
                parent = base_table
                child  = joined_table
                col    = left_col

            if not parent or not child or parent == child:
                continue
            if parent not in known_tables or child not in known_tables:
                continue

            results.append({
                "from_entity":  parent,
                "to_entity":    child,
                "cardinality":  "1:N",
                "rel_type":     "has_many",
                "via_column":   col,
                "via_table":    "",
                "confidence":   CONF_SQL_JOIN,
                "signals":      ["sql_join"],
                "source_files": [fpath] if fpath else [],
            })

        for m in _JOIN_USING_RE.finditer(sql):
            joined_table = m.group(1).lower()
            col          = m.group(2).lower()
            if joined_table not in known_tables or not base_table:
                continue
            parent = _fk_col_to_table(col, known_tables) or base_table
            child  = joined_table if joined_table != parent else base_table
            if parent and child and parent != child and parent in known_tables and child in known_tables:
                results.append({
                    "from_entity":  parent,
                    "to_entity":    child,
                    "cardinality":  "1:N",
                    "rel_type":     "has_many",
                    "via_column":   col,
                    "via_table":    "",
                    "confidence":   CONF_SQL_JOIN,
                    "signals":      ["sql_join_using"],
                    "source_files": [fpath] if fpath else [],
                })
    return results


# ── Signal E: Column name pattern ─────────────────────────────────────────────

def _detect_col_patterns(
    table_columns: list[dict],
    known_tables: set[str],
) -> list[dict]:
    results = []
    # Group columns by table
    by_table: dict[str, list[str]] = defaultdict(list)
    for entry in (table_columns or []):
        tbl = (entry.get("table") or "").lower()
        col = (entry.get("column") or entry.get("name") or "").lower()
        if tbl and col:
            by_table[tbl].append(col)

    for tbl, cols in by_table.items():
        if tbl not in known_tables:
            continue
        for col in cols:
            ref_tbl = _fk_col_to_table(col, known_tables)
            if not ref_tbl or ref_tbl == tbl:
                continue
            # Skip if already captured by FK explicit signal
            results.append({
                "from_entity":  ref_tbl,
                "to_entity":    tbl,
                "cardinality":  "1:N",
                "rel_type":     "has_many",
                "via_column":   col,
                "via_table":    "",
                "confidence":   CONF_COL_PATTERN,
                "signals":      ["column_pattern"],
                "source_files": [],
            })
    return results


# ── Mermaid erDiagram generation ──────────────────────────────────────────────

def _build_mermaid(
    relationships: list[EntityRelationship],
    entities_by_table: dict,
    core_only: bool = True,
) -> str:
    _CARDINALITY_MAP = {
        "1:1": ("||", "||"),
        "1:N": ("||", "o{"),
        "N:M": ("}o", "o{"),
    }
    lines = ["erDiagram"]
    seen_entities: set[str] = set()

    for rel in relationships:
        ent_from = entities_by_table.get(rel.from_entity)
        ent_to   = entities_by_table.get(rel.to_entity)
        if core_only:
            if (ent_from and ent_from.is_system) or (ent_to and ent_to.is_system):
                continue
        left_m, right_m = _CARDINALITY_MAP.get(rel.cardinality, ("||", "o{"))
        label = rel.via_table if rel.cardinality == "N:M" else rel.via_column
        lines.append(
            f"    {rel.from_entity.upper()} {left_m}--{right_m} "
            f"{rel.to_entity.upper()} : \"{label}\""
        )
        seen_entities.add(rel.from_entity)
        seen_entities.add(rel.to_entity)

    # Add column definitions for core entities
    for tbl in sorted(seen_entities):
        ent = entities_by_table.get(tbl)
        if not ent or not ent.columns:
            continue
        if core_only and ent.is_system:
            continue
        col_lines = []
        for c in ent.columns[:8]:   # cap to keep diagram readable
            pk_tag = " PK" if c.is_primary_key else (" FK" if c.is_foreign_key else "")
            dt = c.data_type[:15] if c.data_type else "varchar"
            col_lines.append(f"        {dt} {c.name}{pk_tag}")
        if col_lines:
            lines.append(f"    {tbl.upper()} {{")
            lines.extend(col_lines)
            lines.append("    }")

    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    cm = getattr(ctx, "code_map", None)
    if cm is None:
        print("  [stage36] ⚠️  No code_map — skipping relationship reconstruction.")
        ctx.relationships = EntityRelationshipCollection()
        return

    if ctx.entities is None:
        print("  [stage36] ⚠️  No entity catalog (stage35 not run) — skipping.")
        ctx.relationships = EntityRelationshipCollection()
        return

    entities_by_table = {e.table: e for e in ctx.entities.entities}
    known_tables: set[str] = set(entities_by_table.keys())

    table_columns = cm.table_columns or []
    sql_queries   = cm.sql_queries   or []

    print(f"  [stage36] Scanning {len(known_tables)} entities for relationships …")

    # ── Run all signal detectors ────────────────────────────────────────────
    raw_A = _detect_fk_explicit(table_columns, known_tables)
    raw_B = _detect_pivot(entities_by_table, known_tables)
    raw_C = _detect_orm(ctx.php_project_path, known_tables, entities_by_table)
    raw_D = _detect_sql_joins(sql_queries, known_tables)
    raw_E = _detect_col_patterns(table_columns, known_tables)

    all_raw = raw_A + raw_B + raw_C + raw_D + raw_E
    merged  = _merge_rels(all_raw)

    print(f"  [stage36] Signal counts: FK={len(raw_A)} pivot={len(raw_B)} "
          f"ORM={len(raw_C)} JOIN={len(raw_D)} col_pattern={len(raw_E)} "
          f"→ {len(merged)} unique relationship(s).")

    # ── Build EntityRelationship objects ────────────────────────────────────
    relationships: list[EntityRelationship] = []
    participating_tables: set[str] = set()
    seq = 1

    for r in sorted(merged, key=lambda x: (-x["confidence"], x["from_entity"])):
        relationships.append(EntityRelationship(
            rel_id       = f"rel_{seq:03d}",
            from_entity  = r["from_entity"],
            to_entity    = r["to_entity"],
            cardinality  = r["cardinality"],
            rel_type     = r["rel_type"],
            via_column   = r.get("via_column", ""),
            via_table    = r.get("via_table", ""),
            confidence   = r["confidence"],
            signals      = r["signals"],
            source_files = r["source_files"],
        ))
        participating_tables.add(r["from_entity"])
        participating_tables.add(r["to_entity"])
        seq += 1

    # ── Update entity is_core flags ─────────────────────────────────────────
    for tbl in participating_tables:
        ent = entities_by_table.get(tbl)
        if ent and not ent.is_system:
            ent.is_core = True

    # Refresh EntityCollection core_count
    if ctx.entities:
        ctx.entities.core_count = sum(1 for e in ctx.entities.entities if e.is_core)

    # ── Print summary ────────────────────────────────────────────────────────
    total = len(relationships)
    card_counts: dict[str, int] = defaultdict(int)
    for rel in relationships:
        card_counts[rel.cardinality] += 1

    print(f"\n  [stage36] Relationship Reconstruction complete — {total} relationship(s) "
          f"(1:1={card_counts['1:1']} 1:N={card_counts['1:N']} N:M={card_counts['N:M']}):")
    for rel in relationships[:25]:
        print(f"    {rel.rel_id}  {rel.from_entity:<25} {rel.cardinality:<5} "
              f"{rel.to_entity:<25}  via={rel.via_column or rel.via_table or '?'}  "
              f"conf={rel.confidence:.2f}  [{','.join(rel.signals)}]")
    if total > 25:
        print(f"    … and {total - 25} more")
    print()

    entity_names = sorted(participating_tables)
    collection = EntityRelationshipCollection(
        relationships = relationships,
        entity_names  = entity_names,
        total         = total,
    )
    ctx.relationships = collection

    # ── Persist catalog ──────────────────────────────────────────────────────
    try:
        import dataclasses
        out_path = ctx.output_path("relationship_catalog.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(collection), fh, indent=2, ensure_ascii=False)
        print(f"  [stage36] Saved → {out_path}")
    except Exception as exc:
        print(f"  [stage36] ⚠️  Could not save relationship_catalog.json: {exc}")

    # ── Persist Mermaid ER diagram ───────────────────────────────────────────
    try:
        mermaid_src = _build_mermaid(relationships, entities_by_table, core_only=True)
        er_path     = str(Path(ctx.output_path("relationship_catalog.json")).parent / "er_diagram.mmd")
        Path(er_path).write_text(mermaid_src, encoding="utf-8")
        ctx.relationships.mermaid_path = er_path
        print(f"  [stage36] ER diagram  → {er_path}")
    except Exception as exc:
        print(f"  [stage36] ⚠️  Could not write er_diagram.mmd: {exc}")

    # Re-save entity catalog so updated is_core flags are persisted
    if ctx.entities:
        try:
            import dataclasses
            ent_path = ctx.output_path("entity_catalog.json")
            with open(ent_path, "w", encoding="utf-8") as fh:
                json.dump(dataclasses.asdict(ctx.entities), fh, indent=2, ensure_ascii=False)
        except Exception:
            pass
