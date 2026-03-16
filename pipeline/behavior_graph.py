"""
pipeline/behavior_graph.py — Behavior Graph Extraction (Stage 2.5)

Builds a formal, persisted behavior graph from code_map data produced by
Stage 1 (parse) and Stage 1.5 (execution paths).  No LLM is invoked — this
stage is entirely deterministic.

Graph structure
---------------
Nodes
  route        HTTP route entry (method + path)
  controller   OOP controller class (handler of a route)
  service      Service / repository class reached via constructor injection
  sql          SQL operation (op + table)
  redirect     HTTP redirect target

Edges
  handles      route → controller
  delegates_to controller → service  OR  service → service
  queries      (controller | service) → sql
  redirects_to (controller | service) → redirect

Behavior Path
  One complete traversal:
      Route → Controller → [Services…] → [SQL ops…] → [Redirect]

Output
------
behavior_graph.json  (saved to 2.5_behavior/ via ctx.output_path())

Schema::

    {
        "nodes": [
            {"id": str, "type": str, "label": str, "file": str, "metadata": {...}},
            ...
        ],
        "edges": [
            {"source": str, "target": str, "type": str},
            ...
        ],
        "paths": [
            {
                "path_id":        str,          # e.g. "path_001"
                "entry_node":     str,          # node id of the route
                "node_ids":       [str],        # ordered ids in traversal order
                "confidence":     float,        # 0.25 – 0.85
                "route_method":   str,
                "route_path":     str,
                "controller":     str,          # short class name
                "controller_fqn": str,
                "action_method":  str,
                "services":       [str],        # short dep class names
                "sql_tables":     [str],
                "sql_ops":        [{"op": str, "table": str, "file": str}],
                "redirect":       str,          # first redirect target or ""
                "has_auth":       bool,
                "middleware":     [str],
                "reachable_files":[str],        # controller + service files
            },
            ...
        ],
        "summary": {
            "total_paths":         int,
            "total_nodes":         int,
            "total_edges":         int,
            "node_type_counts":    {"route": n, "controller": n, ...},
            "edge_type_counts":    {"handles": n, ...},
            "paths_with_sql":      int,
            "paths_with_redirect": int,
            "paths_with_auth":     int,
            "avg_confidence":      float,
        }
    }
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

# Tables that add noise without differentiating features
_SKIP_TABLES: frozenset[str] = frozenset({"", "unknown", "temp", "tmp", "dual"})

_MAX_SQL_PER_PATH   = 6    # cap SQL nodes per behavior path
_MAX_SERVICE_DEPTH  = 3    # BFS depth over service_deps
_MAX_PATHS          = 500  # safety cap on total paths extracted


# ─── Public API ────────────────────────────────────────────────────────────────

def extract(ctx: Any) -> dict:
    """
    Extract a behavior graph from *ctx.code_map*.

    Parameters
    ----------
    ctx : PipelineContext

    Returns
    -------
    dict — the full behavior_graph document (nodes, edges, paths, summary)
    """
    cm = getattr(ctx, "code_map", None)
    if cm is None:
        return _empty_graph("No code_map in context")

    # ── Index build ────────────────────────────────────────────────────────────
    fqn_to_file, class_to_file = _build_class_indexes(cm)
    direct_deps                = _build_dep_index(cm)
    file_to_sql                = _build_file_sql_index(cm)
    file_to_redir              = _build_file_redir_index(cm)

    # ── Per-path extraction ────────────────────────────────────────────────────
    nodes:     dict[str, dict] = {}   # id → node dict  (deduplication)
    edges:     dict[tuple, dict] = {} # (src, tgt, type) → edge dict
    paths:     list[dict]      = []

    for route in (cm.routes or []):
        method  = (route.get("method") or "").upper()
        path    = route.get("path") or route.get("uri") or ""
        handler = route.get("handler") or ""

        # Skip non-HTTP and anonymous entries
        if method in ("GROUP", "MIDDLEWARE", "PREFIX", ""):
            continue
        if not handler or handler in ("(closure)", "(group)", ""):
            continue
        if not path:
            continue

        # ── Parse handler → controller FQN + action method ────────────────────
        if "@" in handler:
            ctrl_fqn, action_method = handler.rsplit("@", 1)
        elif "::" in handler:
            ctrl_fqn, action_method = handler.rsplit("::", 1)
        elif ":" in handler:
            ctrl_fqn, action_method = handler.rsplit(":", 1)
        else:
            ctrl_fqn, action_method = handler, "index"

        ctrl_short = ctrl_fqn.split("\\")[-1].split("/")[-1]
        if not ctrl_short:
            continue

        ctrl_file = fqn_to_file.get(ctrl_fqn) or class_to_file.get(ctrl_short, "")

        # ── Transitive BFS over service_deps ──────────────────────────────────
        visited: set[str] = {ctrl_short}
        queue: deque[tuple[str, int, str]] = deque([(ctrl_short, 0, "")])
        # List of (dep_short, parent_short) in discovery order (for edge creation)
        service_chain: list[tuple[str, str]] = []
        reachable_files: list[str] = []
        if ctrl_file:
            reachable_files.append(ctrl_file)

        while queue:
            cls, depth, parent = queue.popleft()
            if depth >= _MAX_SERVICE_DEPTH:
                continue
            for dep in direct_deps.get(cls, set()):
                if dep in visited:
                    continue
                visited.add(dep)
                queue.append((dep, depth + 1, cls))
                service_chain.append((dep, cls))
                dep_file = class_to_file.get(dep, "")
                if dep_file and dep_file not in reachable_files:
                    reachable_files.append(dep_file)

        # ── Collect SQL ────────────────────────────────────────────────────────
        sql_seen: set[tuple[str, str]] = set()
        sql_ops:  list[dict]           = []
        for fpath in reachable_files:
            for q in file_to_sql.get(fpath, []):
                op    = (q.get("operation") or "").upper()
                table = (q.get("table") or "").strip()
                if not op or table.lower() in _SKIP_TABLES:
                    continue
                key = (op, table.lower())
                if key not in sql_seen:
                    sql_seen.add(key)
                    sql_ops.append({"op": op, "table": table, "file": fpath})
                if len(sql_ops) >= _MAX_SQL_PER_PATH:
                    break
            if len(sql_ops) >= _MAX_SQL_PER_PATH:
                break

        # ── Collect first redirect ─────────────────────────────────────────────
        redirect_target = ""
        for fpath in ([ctrl_file] if ctrl_file else []) + reachable_files:
            redirs = file_to_redir.get(fpath, [])
            if redirs:
                redirect_target = redirs[0].get("target", "")
                break

        # ── Auth from middleware ───────────────────────────────────────────────
        middleware = route.get("middleware", [])
        if isinstance(middleware, str):
            middleware = [middleware]
        has_auth = any(
            mw in ("auth", "auth:sanctum", "auth:api", "verified", "login")
            for mw in middleware
        )

        # ── Confidence ────────────────────────────────────────────────────────
        confidence = _path_confidence(
            has_ctrl_file = bool(ctrl_file),
            has_services  = bool(service_chain),
            has_sql       = bool(sql_ops),
            n_sql_tables  = len({s["table"].lower() for s in sql_ops}),
            has_redirect  = bool(redirect_target),
            has_auth      = has_auth,
        )

        # Skip bare route entries (no controller resolved)
        if not ctrl_file and not sql_ops:
            continue

        # ── Build/merge graph nodes ────────────────────────────────────────────
        path_id = f"path_{len(paths) + 1:03d}"

        # Route node
        route_node_id = _node_id("route", f"{method}:{path}")
        _upsert_node(nodes, route_node_id, "route", f"{method} {path}", "", {
            "method": method,
            "path":   path,
            "middleware": middleware,
        })

        # Controller node
        ctrl_node_id = _node_id("controller", ctrl_fqn or ctrl_short)
        _upsert_node(nodes, ctrl_node_id, "controller",
                     f"{ctrl_short}::{action_method}",
                     ctrl_file, {
                         "fqn":    ctrl_fqn,
                         "short":  ctrl_short,
                         "method": action_method,
                     })

        # handles edge: route → controller
        _upsert_edge(edges, route_node_id, ctrl_node_id, "handles")

        # Service nodes + delegates_to edges
        svc_node_map: dict[str, str] = {}  # short_name → node_id
        for dep_short, parent_short in service_chain:
            dep_file   = class_to_file.get(dep_short, "")
            svc_nid    = _node_id("service", dep_short)
            _upsert_node(nodes, svc_nid, "service", dep_short, dep_file, {
                "short": dep_short,
            })
            svc_node_map[dep_short] = svc_nid

            # Parent is either the controller or another service
            if parent_short == ctrl_short:
                parent_nid = ctrl_node_id
            else:
                parent_nid = svc_node_map.get(parent_short, ctrl_node_id)
            _upsert_edge(edges, parent_nid, svc_nid, "delegates_to")

        # SQL nodes + queries edges
        sql_node_ids: list[str] = []
        for sql_item in sql_ops:
            sql_label = f"{sql_item['op']} {sql_item['table']}"
            sql_nid   = _node_id("sql", sql_label)
            _upsert_node(nodes, sql_nid, "sql", sql_label, sql_item["file"], {
                "operation": sql_item["op"],
                "table":     sql_item["table"],
            })
            sql_node_ids.append(sql_nid)

            # queries edge: the file that owns the SQL points to its class
            # If the SQL file matches a service, link from that service; else ctrl
            sql_fname = Path(sql_item["file"]).name.lower()
            querier_nid = ctrl_node_id  # default
            for dep_short, svc_nid in svc_node_map.items():
                dep_file = class_to_file.get(dep_short, "")
                if dep_file and Path(dep_file).name.lower() == sql_fname:
                    querier_nid = svc_nid
                    break
            _upsert_edge(edges, querier_nid, sql_nid, "queries")

        # Redirect node + redirects_to edge
        redir_node_id = ""
        if redirect_target:
            redir_label   = Path(redirect_target.split("?")[0]).name or redirect_target
            redir_node_id = _node_id("redirect", redirect_target[:120])
            _upsert_node(nodes, redir_node_id, "redirect", redir_label[:80],
                         "", {"target": redirect_target})
            _upsert_edge(edges, ctrl_node_id, redir_node_id, "redirects_to")

        # ── Build ordered node_ids for this path ──────────────────────────────
        path_node_ids: list[str] = [route_node_id, ctrl_node_id]
        for _, svc_nid in [(d, svc_node_map[d]) for d in svc_node_map]:
            if svc_nid not in path_node_ids:
                path_node_ids.append(svc_nid)
        path_node_ids.extend(nid for nid in sql_node_ids if nid not in path_node_ids)
        if redir_node_id and redir_node_id not in path_node_ids:
            path_node_ids.append(redir_node_id)

        paths.append({
            "path_id":         path_id,
            "entry_node":      route_node_id,
            "node_ids":        path_node_ids,
            "confidence":      confidence,
            "route_method":    method,
            "route_path":      path,
            "controller":      ctrl_short,
            "controller_fqn":  ctrl_fqn,
            "action_method":   action_method,
            "services":        list(svc_node_map.keys()),
            "sql_ops":         sql_ops,
            "sql_tables":      [s["table"] for s in sql_ops],
            "redirect":        redirect_target,
            "has_auth":        has_auth,
            "middleware":      middleware,
            "reachable_files": reachable_files,
        })

        if len(paths) >= _MAX_PATHS:
            break

    # ── Summary ────────────────────────────────────────────────────────────────
    nodes_list = list(nodes.values())
    edges_list = list(edges.values())

    node_type_counts: dict[str, int] = defaultdict(int)
    for n in nodes_list:
        node_type_counts[n["type"]] += 1

    edge_type_counts: dict[str, int] = defaultdict(int)
    for e in edges_list:
        edge_type_counts[e["type"]] += 1

    avg_conf = (
        round(sum(p["confidence"] for p in paths) / len(paths), 3)
        if paths else 0.0
    )

    summary = {
        "total_paths":         len(paths),
        "total_nodes":         len(nodes_list),
        "total_edges":         len(edges_list),
        "node_type_counts":    dict(node_type_counts),
        "edge_type_counts":    dict(edge_type_counts),
        "paths_with_sql":      sum(1 for p in paths if p["sql_ops"]),
        "paths_with_redirect": sum(1 for p in paths if p["redirect"]),
        "paths_with_auth":     sum(1 for p in paths if p["has_auth"]),
        "avg_confidence":      avg_conf,
    }

    return {
        "nodes":   nodes_list,
        "edges":   edges_list,
        "paths":   paths,
        "summary": summary,
    }


def save(graph: dict, ctx: Any) -> str:
    """
    Persist *graph* to behavior_graph.json inside the run output directory.

    Returns
    -------
    str  — absolute path of the written file.
    """
    out_path = ctx.output_path("behavior_graph.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=2)
    return out_path


def load(ctx: Any) -> dict | None:
    """
    Load a previously saved behavior_graph.json from the run output directory.

    Returns None if the file does not exist yet.
    """
    out_path = ctx.output_path("behavior_graph.json")
    if not Path(out_path).exists():
        return None
    with open(out_path, encoding="utf-8") as fh:
        return json.load(fh)


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _node_id(node_type: str, key: str) -> str:
    """Stable, filesystem-safe node identifier."""
    safe = key.replace("\\", "_").replace("/", "_").replace(" ", "_")
    # Truncate to keep IDs readable
    if len(safe) > 80:
        safe = safe[:80]
    return f"{node_type}:{safe}"


def _upsert_node(
    nodes:    dict[str, dict],
    node_id:  str,
    ntype:    str,
    label:    str,
    file:     str,
    metadata: dict,
) -> None:
    """Add node if not already present; existing nodes are not overwritten."""
    if node_id not in nodes:
        nodes[node_id] = {
            "id":       node_id,
            "type":     ntype,
            "label":    label,
            "file":     file,
            "metadata": metadata,
        }


def _upsert_edge(
    edges:  dict[tuple, dict],
    source: str,
    target: str,
    etype:  str,
) -> None:
    """Add edge if not already present."""
    key = (source, target, etype)
    if key not in edges:
        edges[key] = {"source": source, "target": target, "type": etype}


def _build_class_indexes(
    cm: Any,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Two-tier class lookup.

    Returns
    -------
    fqn_to_file   : full FQN → file path (exact match, most reliable)
    class_to_file : short name → file path (first-seen wins)
    """
    fqn_to_file:   dict[str, str] = {}
    class_to_file: dict[str, str] = {}

    for c in list(cm.controllers or []) + list(cm.services or []):
        fqn   = c.get("fqn", "")
        short = c.get("name") or (fqn.split("\\")[-1].split("/")[-1] if fqn else "")
        fpath = c.get("file", "")
        if fqn and fpath:
            fqn_to_file[fqn] = fpath
        if short and fpath and short not in class_to_file:
            class_to_file[short] = fpath

    return fqn_to_file, class_to_file


def _build_dep_index(cm: Any) -> dict[str, set[str]]:
    """class short-name → set of injected dep class names (one-hop)."""
    deps: dict[str, set[str]] = defaultdict(set)
    for d in (cm.service_deps or []):
        cls = d.get("class", "")
        dep = d.get("dep_class", "")
        if cls and dep:
            deps[cls].add(dep)
    return deps


def _build_file_sql_index(cm: Any) -> dict[str, list[dict]]:
    """file path → list of SQL query dicts."""
    idx: dict[str, list[dict]] = defaultdict(list)
    for q in (cm.sql_queries or []):
        if q.get("file"):
            idx[q["file"]].append(q)
    return idx


def _build_file_redir_index(cm: Any) -> dict[str, list[dict]]:
    """file path → list of redirect dicts."""
    idx: dict[str, list[dict]] = defaultdict(list)
    for r in (cm.redirects or []):
        if r.get("file"):
            idx[r["file"]].append(r)
    return idx


def _path_confidence(
    has_ctrl_file: bool,
    has_services:  bool,
    has_sql:       bool,
    n_sql_tables:  int,
    has_redirect:  bool,
    has_auth:      bool,
) -> float:
    """Score a behavior path 0.25–0.85 based on how much chain evidence exists."""
    score = 0.25                          # base: route found
    if has_ctrl_file:  score += 0.15     # controller resolved
    if has_services:   score += 0.15     # at least one injected service
    if has_sql:        score += 0.15     # SQL operations found
    if n_sql_tables >= 2: score += 0.05  # multiple tables → richer operation
    if has_redirect:   score += 0.05     # output path found
    if has_auth:       score += 0.05     # auth boundary documented
    return round(min(0.85, score), 2)


def _empty_graph(reason: str = "") -> dict:
    return {
        "nodes":   [],
        "edges":   [],
        "paths":   [],
        "summary": {
            "total_paths":         0,
            "total_nodes":         0,
            "total_edges":         0,
            "node_type_counts":    {},
            "edge_type_counts":    {},
            "paths_with_sql":      0,
            "paths_with_redirect": 0,
            "paths_with_auth":     0,
            "avg_confidence":      0.0,
            "note":                reason,
        },
    }
