"""
pipeline/stage25_behavior.py — Behavior Graph Extraction Stage

Runs between Stage 2 (knowledge graph) and Stage 3 (vector embeddings).

Deterministically traces every route through its handler chain:

    Route → Controller → [Services via DI] → [SQL ops] → [Redirect]

No LLM is involved.  The output (behavior_graph.json) is consumed by:
  • Stage 4.5 (business flow extraction) — can skip re-traversal
  • Stage 6    (QA review)              — richer coverage evidence

Outputs
-------
  behavior_graph.json   →   2.5_behavior/
"""

from __future__ import annotations

from typing import Any

from pipeline.behavior_graph import extract, save


def run(ctx: Any) -> None:
    """Stage 2.5 entry point — called by run_pipeline.py."""
    cm = getattr(ctx, "code_map", None)
    if cm is None:
        print("  [stage25] ⚠️  No code_map available — skipping behavior graph extraction.")
        return

    n_routes     = len(cm.routes or [])
    n_controllers = len(cm.controllers or [])
    n_services   = len(cm.services or [])
    n_sql        = len(cm.sql_queries or [])

    print(f"  [stage25] Extracting behavior graph from:")
    print(f"            {n_routes} routes · {n_controllers} controllers · "
          f"{n_services} services · {n_sql} SQL queries")

    graph = extract(ctx)
    s     = graph["summary"]

    out_path = save(graph, ctx)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"  [stage25] ✅ Behavior graph built:")
    print(f"            {s['total_paths']} paths · "
          f"{s['total_nodes']} nodes · "
          f"{s['total_edges']} edges")
    print(f"            avg confidence: {s['avg_confidence']:.2f} | "
          f"with SQL: {s['paths_with_sql']} | "
          f"with redirect: {s['paths_with_redirect']} | "
          f"with auth: {s['paths_with_auth']}")

    # Node type breakdown
    ntc = s.get("node_type_counts", {})
    if ntc:
        parts = "  ·  ".join(f"{t}: {n}" for t, n in sorted(ntc.items()))
        print(f"            Nodes — {parts}")

    print(f"  [stage25] Saved → {out_path}")

    # Attach the loaded graph to ctx so Stage 4.5 / Stage 6 can consume it
    # without re-reading from disk
    ctx.behavior_graph = graph
