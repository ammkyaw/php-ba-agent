"""
pipeline/stage9_knowledge_graph.py — System Knowledge Graph Builder (Stage 9)

Produces a *business-level* knowledge graph that connects domain concepts
across all prior pipeline outputs into a single navigable JSON structure.

This is distinct from the Stage 2 *code* graph (code_graph.gpickle), which
models PHP source structure (files, classes, functions, call edges).  Stage 9
models the *system* in terms that a BA, QA engineer, or downstream tool can
reason about:

    actor  → flow  → page  → table
    feature → flow → bounded_context
    entity → table

Node types (7)
--------------
  feature         domain_model.features
  entity          domain_model.key_entities
  page            code_map.html_pages  +  flow evidence_files
  table           code_map.db_schema / table_columns
  flow            business_flows.flows
  actor           domain_model.user_roles
  bounded_context domain_model.bounded_contexts

Edge types (10)
---------------
  actor_performs      actor       → flow
  flow_in_context     flow        → bounded_context
  flow_uses           flow        → page
  flow_reads          flow        → table  (SELECT)
  flow_writes         flow        → table  (INSERT / UPDATE / DELETE)
  feature_includes    feature     → flow
  feature_depends_on  feature     → entity
  entity_stored_in    entity      → table
  page_queries        page        → table  (from code_map.sql_queries)
  context_contains    bounded_context → feature

Output
------
  knowledge_graph.json   — nodes + edges + summary statistics
  ctx.knowledge_graph_meta — KnowledgeGraphMeta dataclass

Sources required
----------------
  ctx.domain_model      — Stage 4  (required)
  ctx.business_flows    — Stage 4.5 (required)
  ctx.code_map          — Stage 1   (optional — enriches pages + tables)

Resume behaviour
----------------
If stage9_knowledge_graph is COMPLETED and knowledge_graph.json exists, the
stage is skipped and ctx.knowledge_graph_meta is restored from the file header.

Placement in the pipeline
-------------------------
  … stage8_tests → stage9_knowledge_graph

Stage 9 runs last — it consumes *all* prior structured outputs to produce a
complete cross-domain graph.  It does not feed back into any earlier stage.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from context import KnowledgeGraphMeta, PipelineContext

# ── Constants ──────────────────────────────────────────────────────────────────

STAGE_NAME   = "stage9_knowledge_graph"
OUTPUT_FILE  = "knowledge_graph.json"

# SQL operation keywords → edge type
_READ_KEYWORDS  = ("select",)
_WRITE_KEYWORDS = ("insert", "update", "delete", "replace", "truncate")

# Minimum name length to consider a token as a meaningful entity/table match
_MIN_MATCH_LEN = 3


# ── Public Entry Point ─────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 9 entry point.  Builds the system knowledge graph and persists it.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If required upstream stages (4 and 4.5) are absent.
    """
    output_path = ctx.output_path(OUTPUT_FILE)

    # ── Resume check ──────────────────────────────────────────────────────────
    if ctx.is_stage_done(STAGE_NAME) and Path(output_path).exists():
        print(f"  [{STAGE_NAME}] Already completed — loading from {output_path}")
        ctx.knowledge_graph_meta = _load_meta(output_path)
        return

    _assert_prerequisites(ctx)

    print(f"  [{STAGE_NAME}] Building system knowledge graph ...")

    # ── Graph accumulators ────────────────────────────────────────────────────
    # nodes: id → {id, type, label, **attrs}
    # edges: list of {source, target, type, **attrs}
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]]      = []

    # ── Pass 1: Add nodes from every source ───────────────────────────────────
    print(f"  [{STAGE_NAME}]   Pass 1: collecting nodes ...")
    _add_actor_nodes(nodes, ctx)
    _add_bounded_context_nodes(nodes, ctx)
    _add_feature_nodes(nodes, ctx)
    _add_entity_nodes(nodes, ctx)
    _add_flow_nodes(nodes, ctx)
    _add_page_nodes(nodes, ctx)
    _add_table_nodes(nodes, ctx)

    n_nodes = len(nodes)
    print(f"  [{STAGE_NAME}]   {n_nodes} nodes: "
          f"{_count_by_type(nodes, 'actor')} actors, "
          f"{_count_by_type(nodes, 'bounded_context')} contexts, "
          f"{_count_by_type(nodes, 'feature')} features, "
          f"{_count_by_type(nodes, 'entity')} entities, "
          f"{_count_by_type(nodes, 'flow')} flows, "
          f"{_count_by_type(nodes, 'page')} pages, "
          f"{_count_by_type(nodes, 'table')} tables")

    # ── Pass 2: Add edges ──────────────────────────────────────────────────────
    print(f"  [{STAGE_NAME}]   Pass 2: resolving edges ...")
    _add_context_feature_edges(edges, nodes, ctx)
    _add_actor_flow_edges(edges, nodes, ctx)
    _add_flow_context_edges(edges, nodes, ctx)
    _add_flow_page_edges(edges, nodes, ctx)
    _add_flow_table_edges(edges, nodes, ctx)
    _add_feature_flow_edges(edges, nodes, ctx)
    _add_feature_entity_edges(edges, nodes, ctx)
    _add_entity_table_edges(edges, nodes, ctx)
    _add_page_table_edges(edges, nodes, ctx)

    # Deduplicate edges (same source+target+type counts as one)
    edges = _dedup_edges(edges)

    n_edges = len(edges)
    edge_type_counts = defaultdict(int)
    for e in edges:
        edge_type_counts[e["type"]] += 1
    print(f"  [{STAGE_NAME}]   {n_edges} edges: "
          + ", ".join(f"{t}={c}" for t, c in sorted(edge_type_counts.items())))

    # ── Build output payload ──────────────────────────────────────────────────
    domain_name = ctx.domain_model.domain_name if ctx.domain_model else "Unknown"
    payload = {
        "meta": {
            "domain":       domain_name,
            "node_count":   n_nodes,
            "edge_count":   n_edges,
            "node_types":   sorted({n["type"] for n in nodes.values()}),
            "edge_types":   sorted(edge_type_counts.keys()),
            "edge_type_counts": dict(sorted(edge_type_counts.items())),
            "sources": {
                "domain_model":   ctx.domain_model is not None,
                "business_flows": ctx.business_flows is not None,
                "code_map":       ctx.code_map is not None,
            },
        },
        "nodes": list(nodes.values()),
        "edges": edges,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"  [{STAGE_NAME}]   Saved → {output_path}")

    # ── Update context ────────────────────────────────────────────────────────
    ctx.knowledge_graph_meta = KnowledgeGraphMeta(
        json_path         = output_path,
        node_count        = n_nodes,
        edge_count        = n_edges,
        node_types        = sorted({n["type"] for n in nodes.values()}),
        edge_type_counts  = dict(edge_type_counts),
    )
    ctx.stage(STAGE_NAME).mark_completed(output_path)
    ctx.save()
    print(f"  [{STAGE_NAME}] Complete.")


# ── Pre-flight ─────────────────────────────────────────────────────────────────

def _assert_prerequisites(ctx: PipelineContext) -> None:
    if ctx.domain_model is None:
        raise RuntimeError(
            f"[{STAGE_NAME}] ctx.domain_model is None — run Stage 4 first."
        )
    if ctx.business_flows is None or not ctx.business_flows.flows:
        raise RuntimeError(
            f"[{STAGE_NAME}] ctx.business_flows is empty — run Stage 4.5 first."
        )


# ── Pass 1: Node Builders ──────────────────────────────────────────────────────

def _node_id(node_type: str, name: str) -> str:
    """Produce a stable, collision-safe node ID: 'type:normalised_name'."""
    slug = re.sub(r"[^a-z0-9_]", "_", name.lower().strip()).strip("_")
    return f"{node_type}:{slug}"


def _add_node(
    nodes: dict[str, dict],
    node_type: str,
    name: str,
    **attrs: Any,
) -> str:
    """
    Add a node if not already present; return its ID.
    Subsequent calls with the same ID are no-ops (first writer wins).
    """
    nid = _node_id(node_type, name)
    if nid not in nodes:
        nodes[nid] = {"id": nid, "type": node_type, "label": name, **attrs}
    return nid


def _add_actor_nodes(nodes: dict, ctx: PipelineContext) -> None:
    """One node per user role in domain_model."""
    for role in ctx.domain_model.user_roles:
        role_name = role.get("role") or role.get("name", "Unknown")
        _add_node(nodes, "actor", role_name,
                  description=role.get("description", ""))


def _add_bounded_context_nodes(nodes: dict, ctx: PipelineContext) -> None:
    for context_name in ctx.domain_model.bounded_contexts:
        _add_node(nodes, "bounded_context", context_name)


def _add_feature_nodes(nodes: dict, ctx: PipelineContext) -> None:
    for feat in ctx.domain_model.features:
        name = feat.get("name") or feat.get("title", "Unknown Feature")
        _add_node(nodes, "feature", name,
                  description=feat.get("description", ""),
                  bounded_context=feat.get("bounded_context", ""))


def _add_entity_nodes(nodes: dict, ctx: PipelineContext) -> None:
    for entity_name in ctx.domain_model.key_entities:
        _add_node(nodes, "entity", entity_name)


def _add_flow_nodes(nodes: dict, ctx: PipelineContext) -> None:
    for flow in ctx.business_flows.flows:
        _add_node(nodes, "flow", flow.name,
                  flow_id        = flow.flow_id,
                  actor          = flow.actor,
                  bounded_context= flow.bounded_context,
                  trigger        = flow.trigger,
                  termination    = flow.termination,
                  confidence     = flow.confidence,
                  step_count     = len(flow.steps))


def _add_page_nodes(nodes: dict, ctx: PipelineContext) -> None:
    """
    Collect pages from two sources:
      1. code_map.html_pages — files detected as HTML-mixing entry points
      2. business_flows evidence_files — files that evidence a flow
    """
    page_paths: set[str] = set()

    if ctx.code_map:
        for p in ctx.code_map.html_pages:
            page_paths.add(p)

    for flow in ctx.business_flows.flows:
        for p in flow.evidence_files:
            page_paths.add(p)
        for step in flow.steps:
            if step.page:
                page_paths.add(step.page)

    for page_path in sorted(page_paths):
        page_name = Path(page_path).name
        _add_node(nodes, "page", page_name,
                  file_path=page_path)


def _add_table_nodes(nodes: dict, ctx: PipelineContext) -> None:
    """
    Collect tables from two sources:
      1. code_map.db_schema — migration / CREATE TABLE entries
      2. code_map.table_columns — per-table column definitions
    """
    if not ctx.code_map:
        return

    seen: set[str] = set()

    for schema_entry in ctx.code_map.db_schema:
        tname = schema_entry.get("table") or schema_entry.get("name", "")
        if tname and tname not in seen:
            seen.add(tname)
            cols = schema_entry.get("columns", [])
            col_names = [c.get("name", c) if isinstance(c, dict) else str(c)
                         for c in cols]
            _add_node(nodes, "table", tname,
                      column_count=len(col_names),
                      columns=col_names[:20])   # cap at 20 for readability

    for col_entry in ctx.code_map.table_columns:
        tname = col_entry.get("table", "")
        if tname and tname not in seen:
            seen.add(tname)
            _add_node(nodes, "table", tname)


# ── Pass 2: Edge Builders ──────────────────────────────────────────────────────

def _add_edge(
    edges: list,
    source: str,
    target: str,
    edge_type: str,
    **attrs: Any,
) -> None:
    """Append an edge dict; deduplication happens later in _dedup_edges."""
    edges.append({"source": source, "target": target, "type": edge_type, **attrs})


def _add_context_feature_edges(edges, nodes, ctx):
    """bounded_context → feature  (context_contains)"""
    for feat in ctx.domain_model.features:
        fname = feat.get("name") or feat.get("title", "")
        bc    = feat.get("bounded_context", "")
        if fname and bc:
            src = _node_id("bounded_context", bc)
            dst = _node_id("feature", fname)
            if src in nodes and dst in nodes:
                _add_edge(edges, src, dst, "context_contains")


def _add_actor_flow_edges(edges, nodes, ctx):
    """actor → flow  (actor_performs)"""
    for flow in ctx.business_flows.flows:
        src = _node_id("actor", flow.actor)
        dst = _node_id("flow", flow.name)
        if src in nodes and dst in nodes:
            _add_edge(edges, src, dst, "actor_performs",
                      flow_id=flow.flow_id)


def _add_flow_context_edges(edges, nodes, ctx):
    """flow → bounded_context  (flow_in_context)"""
    for flow in ctx.business_flows.flows:
        src = _node_id("flow", flow.name)
        dst = _node_id("bounded_context", flow.bounded_context)
        if src in nodes and dst in nodes:
            _add_edge(edges, src, dst, "flow_in_context")


def _add_flow_page_edges(edges, nodes, ctx):
    """flow → page  (flow_uses) — derived from FlowStep.page"""
    for flow in ctx.business_flows.flows:
        flow_node = _node_id("flow", flow.name)
        if flow_node not in nodes:
            continue
        for step in flow.steps:
            if not step.page:
                continue
            page_name = Path(step.page).name
            page_node = _node_id("page", page_name)
            if page_node in nodes:
                _add_edge(edges, flow_node, page_node, "flow_uses",
                          step=step.step_num,
                          http_method=step.http_method or "")


def _add_flow_table_edges(edges, nodes, ctx):
    """
    flow → table  (flow_reads / flow_writes)

    Inferred from FlowStep.db_ops — each op string is matched against known
    table names.  Read ops (SELECT) → flow_reads; write ops → flow_writes.
    """
    table_names = {
        data["label"].lower(): nid
        for nid, data in nodes.items()
        if data["type"] == "table"
    }
    if not table_names:
        return

    for flow in ctx.business_flows.flows:
        flow_node = _node_id("flow", flow.name)
        if flow_node not in nodes:
            continue
        for step in flow.steps:
            for db_op in step.db_ops:
                op_lower = db_op.lower()
                edge_type = (
                    "flow_reads"  if any(k in op_lower for k in _READ_KEYWORDS)
                    else "flow_writes"
                )
                # Match table name tokens in the db_op string
                for tname, tnode in table_names.items():
                    if len(tname) >= _MIN_MATCH_LEN and tname in op_lower:
                        _add_edge(edges, flow_node, tnode, edge_type,
                                  via_step=step.step_num,
                                  op_fragment=db_op[:60])


def _add_feature_flow_edges(edges, nodes, ctx):
    """
    feature → flow  (feature_includes)

    Match by bounded_context (flow.bounded_context == feature.bounded_context)
    and keyword overlap between feature name and flow name.
    """
    features = ctx.domain_model.features
    if not features:
        return

    for flow in ctx.business_flows.flows:
        flow_node = _node_id("flow", flow.name)
        if flow_node not in nodes:
            continue
        flow_words = set(_tokenise(flow.name))

        for feat in features:
            fname = feat.get("name") or feat.get("title", "")
            if not fname:
                continue
            feat_node = _node_id("feature", fname)
            if feat_node not in nodes:
                continue

            # Match if: same bounded_context, OR ≥1 shared keyword
            same_context = (feat.get("bounded_context", "") == flow.bounded_context)
            shared_words = flow_words & set(_tokenise(fname))
            if same_context or shared_words:
                _add_edge(edges, feat_node, flow_node, "feature_includes",
                          match_reason=(
                              "same_context" if same_context else
                              f"shared_keywords:{','.join(sorted(shared_words)[:3])}"
                          ))


def _add_feature_entity_edges(edges, nodes, ctx):
    """
    feature → entity  (feature_depends_on)

    Match when an entity name token appears in the feature name or description.
    """
    entities = ctx.domain_model.key_entities
    if not entities:
        return

    for feat in ctx.domain_model.features:
        fname = feat.get("name") or feat.get("title", "")
        fdesc = feat.get("description", "")
        if not fname:
            continue
        feat_node = _node_id("feature", fname)
        if feat_node not in nodes:
            continue

        feat_text = (fname + " " + fdesc).lower()
        for entity_name in entities:
            entity_node = _node_id("entity", entity_name)
            if entity_node not in nodes:
                continue
            if (len(entity_name) >= _MIN_MATCH_LEN
                    and entity_name.lower() in feat_text):
                _add_edge(edges, feat_node, entity_node, "feature_depends_on")


def _add_entity_table_edges(edges, nodes, ctx):
    """
    entity → table  (entity_stored_in)

    Match when the entity name is a substring of the table name or vice-versa.
    Common convention: entity 'User' → table 'users'.
    """
    entities = ctx.domain_model.key_entities
    if not entities:
        return

    table_nodes = {
        data["label"].lower(): nid
        for nid, data in nodes.items()
        if data["type"] == "table"
    }

    for entity_name in entities:
        entity_node = _node_id("entity", entity_name)
        if entity_node not in nodes:
            continue
        ename_lower = entity_name.lower()

        for tname, tnode in table_nodes.items():
            # Match: 'user' in 'users', 'users' contains 'user'
            if (len(ename_lower) >= _MIN_MATCH_LEN
                    and (ename_lower in tname or tname.rstrip("s") == ename_lower)):
                _add_edge(edges, entity_node, tnode, "entity_stored_in")


def _add_page_table_edges(edges, nodes, ctx):
    """
    page → table  (page_queries)

    Sourced from code_map.sql_queries which carries {caller, table, operation}.
    """
    if not ctx.code_map:
        return

    for sql_entry in ctx.code_map.sql_queries:
        caller    = sql_entry.get("caller", "")
        table     = sql_entry.get("table",  "")
        operation = sql_entry.get("operation", "").lower()
        if not (caller and table):
            continue

        page_name = Path(caller).name
        page_node = _node_id("page", page_name)
        table_node = _node_id("table", table)

        if page_node in nodes and table_node in nodes:
            _add_edge(edges, page_node, table_node, "page_queries",
                      operation=operation)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """
    Split a CamelCase or snake_case or spaced name into lowercase tokens,
    filtering out short stop-words.
    """
    # Insert space before uppercase letters (CamelCase → Camel Case)
    spaced = re.sub(r"([A-Z])", r" \1", text)
    tokens = re.split(r"[\s_\-]+", spaced.lower())
    return [t for t in tokens if len(t) >= _MIN_MATCH_LEN]


def _count_by_type(nodes: dict, node_type: str) -> int:
    return sum(1 for n in nodes.values() if n["type"] == node_type)


def _dedup_edges(edges: list[dict]) -> list[dict]:
    """Remove duplicate edges with the same (source, target, type) triple."""
    seen: set[tuple] = set()
    result: list[dict] = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge["type"])
        if key not in seen:
            seen.add(key)
            result.append(edge)
    return result


# ── Persistence ────────────────────────────────────────────────────────────────

def _load_meta(path: str) -> "KnowledgeGraphMeta":
    """Restore KnowledgeGraphMeta from the meta block in an existing JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    meta = data.get("meta", {})
    return KnowledgeGraphMeta(
        json_path        = path,
        node_count       = meta.get("node_count", 0),
        edge_count       = meta.get("edge_count", 0),
        node_types       = meta.get("node_types", []),
        edge_type_counts = meta.get("edge_type_counts", {}),
    )
