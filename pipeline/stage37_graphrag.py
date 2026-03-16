"""
pipeline/stage37_graphrag.py — Graph-Aware Context Index (Stage 3.7)

Runs after Stage 3.5 preflight (all static extracts confirmed healthy).
Merges every Stage 2.x structured output onto a single in-memory traversal
index — zero LLM calls, zero new library dependencies.

Index structure
---------------
  file_nodes    — PHP file  → cluster / role / actor / entities / routes / rules
  entity_nodes  — DB table  → columns / relationships / states / rules / cluster
  cluster_nodes — context   → file_count / entities / rules / route_prefix
  rule_texts    — rule_id   → human-readable description (for context blocks)

Query API
---------
  load_index(path)                         → dict
  query_graph(index, topic, depth, ...)    → Markdown context string

Downstream stages access this via ``ctx.graph_query(topic)`` which lazily loads
the index from ctx.graph_rag_meta.index_path and caches it on ctx.

Used by
-------
  Stage 4    domain model     — global bounded-context overview
  Stage 4.5  flow extraction  — entity + rule context per flow seed
  Stage 5    BRD/SRS/AC       — per-section deep context retrieval
  Stage 6.2  architecture     — component + integration context

Output
------
  3.7_graphrag/graph_context_index.json
  ctx.graph_rag_meta  set on PipelineContext
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from context import GraphRAGMeta, PipelineContext

# ── Constants ─────────────────────────────────────────────────────────────────
INDEX_FILE       = "graph_context_index.json"
_MAX_RULES_NODE  = 10   # business rules stored per node
_MAX_CALLS       = 8    # call-graph edges stored per file node
_MAX_COLS        = 8    # entity columns stored per entity node
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "that", "this", "with", "are", "from",
    "how", "what", "does", "can", "should", "will", "all", "its",
    "get", "set", "use", "let", "via", "per", "any", "one", "two",
    "has", "had", "not", "but", "been", "each", "into", "was",
})


# ── Public entry point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Build the graph-aware context index, persist it, and update ctx."""
    print("  [stage37] Building Graph-Aware Context Index…")

    index = _build_index(ctx)

    out_path = ctx.output_path(INDEX_FILE)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)

    meta = GraphRAGMeta(
        index_path    = out_path,
        file_count    = len(index.get("file_nodes",    {})),
        entity_count  = len(index.get("entity_nodes",  {})),
        cluster_count = len(index.get("cluster_nodes", {})),
        generated_at  = datetime.utcnow().isoformat(),
    )
    ctx.graph_rag_meta = meta
    ctx.stage("stage37_graphrag").mark_completed(out_path)
    ctx.save()

    print(
        f"  [stage37] Index built — "
        f"{meta.file_count} file nodes · "
        f"{meta.entity_count} entity nodes · "
        f"{meta.cluster_count} cluster nodes"
    )
    print(f"  [stage37] Saved → {out_path}")


def load_index(index_path: str) -> dict:
    """Load the pre-built index from disk (called lazily by ctx.graph_query)."""
    with open(index_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── Query API ─────────────────────────────────────────────────────────────────

def query_graph(
    index:     dict,
    topic:     str,
    depth:     int = 2,
    max_chars: int = 3000,
) -> str:
    """
    Graph-aware context retrieval for a free-form topic string.

    Scores all nodes by keyword overlap, takes the top seeds, expands up to
    ``depth`` hops across file→entity→cluster→rule connections, and assembles
    a structured Markdown context block ready for LLM prompt injection.

    Parameters
    ----------
    index     : dict returned by load_index()
    topic     : e.g. "payment processing flow", "user authentication rules"
    depth     : hop expansion radius (1 = immediate neighbours, 2 = broader)
    max_chars : hard cap on the returned string length

    Returns
    -------
    str — Markdown block; empty string when the index is empty or no match.
    """
    if not index:
        return ""

    keywords = _extract_keywords(topic)
    if not keywords:
        return ""

    file_nodes    = index.get("file_nodes",    {})
    entity_nodes  = index.get("entity_nodes",  {})
    cluster_nodes = index.get("cluster_nodes", {})
    rule_texts    = index.get("rule_texts",    {})

    # ── Score nodes ───────────────────────────────────────────────────────────
    top_files    = sorted(_score_file_nodes(file_nodes,       keywords), key=lambda x: -x[1])[:4]
    top_entities = sorted(_score_entity_nodes(entity_nodes,   keywords), key=lambda x: -x[1])[:4]
    top_clusters = sorted(_score_cluster_nodes(cluster_nodes, keywords), key=lambda x: -x[1])[:2]

    # ── 1-hop expansion ───────────────────────────────────────────────────────
    expansion_clusters: set[str] = set()
    expansion_entities: set[str] = set()
    expansion_rules:    set[str] = set()

    for fname, _ in top_files:
        fn = file_nodes.get(fname, {})
        if fn.get("cluster"):
            expansion_clusters.add(fn["cluster"])
        expansion_entities.update(fn.get("entities", []))
        expansion_rules.update(fn.get("rules", []))

    for ename, _ in top_entities:
        en = entity_nodes.get(ename, {})
        if en.get("cluster"):
            expansion_clusters.add(en["cluster"])
        expansion_rules.update(en.get("rules", []))

    for cname, _ in top_clusters:
        cn = cluster_nodes.get(cname, {})
        expansion_entities.update(cn.get("entities", []))
        expansion_rules.update(cn.get("rules", []))

    # ── 2-hop expansion ───────────────────────────────────────────────────────
    if depth >= 2:
        for cname in list(expansion_clusters):
            cn = cluster_nodes.get(cname, {})
            expansion_rules.update(cn.get("rules", []))
            expansion_entities.update(cn.get("entities", []))

    # ── Assemble Markdown block ───────────────────────────────────────────────
    lines: list[str] = []

    # Bounded contexts
    all_clusters = {c for c, _ in top_clusters} | expansion_clusters
    if all_clusters:
        lines.append("### Bounded Contexts")
        for cname in sorted(all_clusters):
            cn = cluster_nodes.get(cname)
            if not cn:
                continue
            ents = ", ".join(cn.get("entities", [])[:6]) or "—"
            nf   = cn.get("file_count", 0)
            nr   = len(cn.get("rules", []))
            pfx  = cn.get("route_prefix", "")
            line = f"- **{cname}** ({nf} files · entities: {ents}"
            if pfx:
                line += f" · prefix: {pfx}"
            if nr:
                line += f" · {nr} rules"
            line += ")"
            lines.append(line)

    # Entities
    all_entities = {e for e, _ in top_entities} | expansion_entities
    if all_entities:
        lines.append("\n### Entities")
        for ename in sorted(all_entities)[:8]:
            en = entity_nodes.get(ename)
            if not en:
                continue
            cols   = ", ".join(en.get("key_columns",   [])[:5]) or "—"
            rels   = "; ".join(en.get("relationships", [])[:3])
            states = ", ".join(en.get("states",        [])[:6])
            line   = f"- **{ename}** [{en.get('cluster', '')}]"
            line  += f"\n  columns: {cols}"
            if rels:
                line += f"\n  relationships: {rels}"
            if states:
                line += f"\n  states: {states}"
            lines.append(line)

    # Endpoints / actions
    if top_files:
        lines.append("\n### Endpoints / Actions")
        for fname, _ in top_files:
            fn      = file_nodes.get(fname, {})
            bname   = Path(fname).name
            role    = fn.get("role",  "")
            actor   = fn.get("actor", "")
            routes  = fn.get("routes", [])
            rstr    = " | ".join(
                f"{r.get('method','')} {r.get('path','')}" for r in routes[:2]
            )
            line = f"- `{bname}`"
            if role:
                line += f" [{role}]"
            if actor:
                line += f" actor={actor}"
            if rstr:
                line += f" — {rstr}"
            lines.append(line)

    # Business rules
    all_rule_ids = expansion_rules | {
        rid
        for fname, _ in top_files
        for rid in file_nodes.get(fname, {}).get("rules", [])
    }
    if all_rule_ids and rule_texts:
        lines.append("\n### Business Rules")
        for rid in sorted(all_rule_ids)[:6]:
            if rid in rule_texts:
                lines.append(f"- [{rid}] {rule_texts[rid]}")

    if not lines:
        return ""

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n… (truncated)"
    return result


# ── Index builder ─────────────────────────────────────────────────────────────

def _build_index(ctx: PipelineContext) -> dict:
    """Merge all Stage 2.x outputs into the traversal index dict."""
    cm = ctx.code_map
    if not cm:
        print("  [stage37] ⚠️  No code_map — index will be empty.")
        return {}

    # ── Stage 2.8: cluster lookups ────────────────────────────────────────────
    file_cluster:   dict[str, str]      = {}
    cluster_tables: dict[str, set[str]] = defaultdict(set)

    if ctx.action_clusters:
        for c in ctx.action_clusters.clusters:
            for f in c.files:
                file_cluster[f] = c.name
            for t in c.tables:
                cluster_tables[c.name].add(t)

    # ── Stage 2.7: role/actor/entity lookups ──────────────────────────────────
    file_role: dict[str, dict] = {}
    if ctx.semantic_roles:
        for tag in ctx.semantic_roles.actions:
            file_role[tag.file] = {
                "role":        tag.role,
                "actor":       tag.actor,
                "entities":    tag.entities,
                "http_method": tag.http_method,
                "route_path":  tag.route_path,
            }

    # ── Routes ────────────────────────────────────────────────────────────────
    file_routes: dict[str, list[dict]] = defaultdict(list)
    for r in (cm.routes or []):
        f = r.get("file", "")
        if f:
            file_routes[f].append({
                "method": r.get("method", ""),
                "path":   r.get("path") or r.get("uri", ""),
            })

    # ── SQL queries → file→tables ─────────────────────────────────────────────
    file_tables: dict[str, set[str]] = defaultdict(set)
    for q in (cm.sql_queries or []):
        f = q.get("file", "")
        t = q.get("table", "")
        if f and t and t.upper() not in ("", "UNKNOWN"):
            file_tables[f].add(t)

    # ── Stage 2.9: invariants → file/entity/rule lookups ─────────────────────
    file_rules:   dict[str, list[str]] = defaultdict(list)
    entity_rules: dict[str, list[str]] = defaultdict(list)
    rule_texts:   dict[str, str]       = {}

    if ctx.invariants:
        for rule in ctx.invariants.rules:
            rule_texts[rule.rule_id] = rule.description[:120]
            for sf in rule.source_files:
                file_rules[sf].append(rule.rule_id)
            if rule.entity:
                # "User.password" → "users" (approx table name)
                tbl = re.sub(r"\.[^.]+$", "", rule.entity).lower().rstrip("s") + "s"
                entity_rules[tbl].append(rule.rule_id)
                entity_rules[rule.entity.split(".")[0].lower()].append(rule.rule_id)

    # ── Call graph ────────────────────────────────────────────────────────────
    file_calls: dict[str, list[str]] = defaultdict(list)
    for edge in (cm.call_graph or []):
        caller = edge.get("caller", "")
        callee = edge.get("callee", "")
        if caller and callee:
            file_calls[caller].append(callee)

    # ── Build file_nodes ──────────────────────────────────────────────────────
    all_files: set[str] = set()
    for ep in (cm.execution_paths or []):
        if ep.get("file"):
            all_files.add(ep["file"])
    for r in (cm.routes or []):
        if r.get("file"):
            all_files.add(r["file"])

    file_nodes: dict[str, dict] = {}
    for fpath in all_files:
        ri       = file_role.get(fpath, {})
        entities = ri.get("entities") or sorted(file_tables.get(fpath, set()))
        file_nodes[fpath] = {
            "cluster":  file_cluster.get(fpath, ""),
            "role":     ri.get("role",  ""),
            "actor":    ri.get("actor", ""),
            "entities": entities[:8],
            "routes":   file_routes.get(fpath, [])[:4],
            "rules":    file_rules.get(fpath, [])[:_MAX_RULES_NODE],
            "calls":    file_calls.get(fpath, [])[:_MAX_CALLS],
        }

    # ── Build entity_nodes ────────────────────────────────────────────────────
    entity_nodes: dict[str, dict] = {}

    # Relationships
    rel_by_entity: dict[str, list[str]] = defaultdict(list)
    if ctx.relationships:
        for rel in ctx.relationships.relationships:
            rel_by_entity[rel.from_entity].append(
                f"{rel.to_entity} ({rel.rel_type}, {rel.cardinality})"
            )
            rel_by_entity[rel.to_entity].append(
                f"{rel.from_entity} ({_invert_rel(rel.rel_type)}, {rel.cardinality})"
            )

    # State machines
    states_by_table: dict[str, list[str]] = defaultdict(list)
    if ctx.state_machines:
        for sm in ctx.state_machines.machines:
            states_by_table[sm.table].extend(sm.states)

    if ctx.entities:
        for ent in ctx.entities.entities:
            key_cols = [
                c.name for c in ent.columns
                if c.is_primary_key or c.is_foreign_key
                or c.name in ("name", "status", "type", "email", "created_at", "title")
            ][:_MAX_COLS]
            # deduplicate rule IDs
            r_ids = list(dict.fromkeys(
                entity_rules.get(ent.table, []) +
                entity_rules.get(ent.name.lower(), [])
            ))[:_MAX_RULES_NODE]

            cluster = ent.bounded_context or (
                file_cluster.get(ent.source_files[0], "") if ent.source_files else ""
            )
            entity_nodes[ent.table] = {
                "name":          ent.name,
                "cluster":       cluster,
                "key_columns":   key_cols,
                "relationships": rel_by_entity.get(ent.table, [])[:6],
                "states":        sorted(set(states_by_table.get(ent.table, [])))[:10],
                "rules":         r_ids,
                "is_core":       ent.is_core,
                "source_files":  ent.source_files[:3],
            }

    # ── Build cluster_nodes ───────────────────────────────────────────────────
    cluster_nodes: dict[str, dict] = {}
    if ctx.action_clusters:
        for c in ctx.action_clusters.clusters:
            c_rules: set[str] = set()
            for f in c.files:
                c_rules.update(file_rules.get(f, []))
            cluster_nodes[c.name] = {
                "file_count":   c.file_count,
                "files":        c.files[:10],
                "entities":     sorted(cluster_tables.get(c.name, set()))[:10],
                "rules":        sorted(c_rules)[:_MAX_RULES_NODE],
                "route_prefix": c.route_prefix,
            }

    return {
        "file_nodes":    file_nodes,
        "entity_nodes":  entity_nodes,
        "cluster_nodes": cluster_nodes,
        "rule_texts":    rule_texts,
    }


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _extract_keywords(topic: str) -> set[str]:
    """Extract significant lowercase tokens (≥3 chars) from a topic string."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", topic.lower())
    return {w for w in words if w not in _STOPWORDS}


def _score_file_nodes(nodes: dict, keywords: set[str]) -> list[tuple[str, float]]:
    results = []
    for fpath, node in nodes.items():
        text = " ".join([
            Path(fpath).stem.lower(),
            node.get("cluster", "").lower(),
            node.get("role",    "").lower(),
            " ".join(node.get("entities", [])).lower(),
            " ".join(r.get("path", "") for r in node.get("routes", [])).lower(),
        ])
        score = sum(1.0 for kw in keywords if kw in text)
        if score > 0:
            results.append((fpath, score))
    return results


def _score_entity_nodes(nodes: dict, keywords: set[str]) -> list[tuple[str, float]]:
    results = []
    for ename, node in nodes.items():
        text = " ".join([
            ename.lower(),
            node.get("name",    "").lower(),
            node.get("cluster", "").lower(),
            " ".join(node.get("states", [])).lower(),
        ])
        score = sum(1.0 for kw in keywords if kw in text)
        if score > 0:
            results.append((ename, score))
    return results


def _score_cluster_nodes(nodes: dict, keywords: set[str]) -> list[tuple[str, float]]:
    results = []
    for cname, node in nodes.items():
        text = " ".join([
            cname.lower(),
            " ".join(node.get("entities", [])).lower(),
            node.get("route_prefix", "").lower(),
        ])
        # Cluster-level matches are higher value (broader coverage)
        score = sum(1.5 for kw in keywords if kw in text)
        if score > 0:
            results.append((cname, score))
    return results


# ── Misc ──────────────────────────────────────────────────────────────────────

def _invert_rel(rel_type: str) -> str:
    return {
        "has_many":     "belongs_to",
        "has_one":      "belongs_to",
        "belongs_to":   "has_many",
        "many_to_many": "many_to_many",
    }.get(rel_type, rel_type)
