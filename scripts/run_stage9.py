"""
run_stage9.py — Standalone runner for Stage 9 (System Knowledge Graph Builder)

Builds a cross-domain knowledge graph that connects actors, bounded contexts,
features, entities, business flows, pages, and database tables into a single
navigable JSON file (knowledge_graph.json).

This graph is semantically different from Stage 2's code_graph.gpickle:

  code_graph       — PHP source structure (files, classes, calls, redirects)
  knowledge_graph  — Business system (actors, features, flows, entities, tables)

Required upstream stages:
  ctx.domain_model    — Stage 4
  ctx.business_flows  — Stage 4.5

Optional (enriches page and table nodes):
  ctx.code_map        — Stage 1

Usage
-----
    python run_stage9.py --resume outputs/run_20240101_120000/context.json

    # Force re-run:
    python run_stage9.py --resume outputs/run_.../context.json --force

    # Inspect inputs without building the graph:
    python run_stage9.py --resume outputs/run_.../context.json --inspect

Exit codes
----------
0  completed (or already done)
1  stage raised an exception
2  bad arguments or missing dependencies
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # locate project root

from context import PipelineContext, StageStatus
from pipeline.stage90_knowledge_graph import run as stage9_run, OUTPUT_FILE


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog            = "run_stage9.py",
        description     = "Run Stage 9 (System Knowledge Graph Builder) standalone.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = __doc__,
    )
    parser.add_argument(
        "--resume",
        metavar  = "CONTEXT_FILE",
        required = True,
        help     = "Path to context.json from a pipeline run.",
    )
    parser.add_argument(
        "--force",
        action  = "store_true",
        default = False,
        help    = "Re-run even if stage90_knowledge_graph is already COMPLETED.",
    )
    parser.add_argument(
        "--inspect",
        action  = "store_true",
        default = False,
        help    = "Print a summary of available inputs, then exit without building.",
    )
    return parser.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _die(msg: str, code: int = 2) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)


def _load_context(args: argparse.Namespace) -> PipelineContext:
    path = Path(args.resume)
    if not path.exists():
        _die(f"Context file not found: {path}")

    ctx = PipelineContext.load(str(path))
    print(f"Loaded run  : {ctx.run_id}")
    print(f"Output dir  : {ctx.output_dir}")

    if args.force:
        ctx.stages["stage90_knowledge_graph"].status = StageStatus.PENDING
        ctx.stages["stage90_knowledge_graph"].error  = None
        print("--force     : stage90_knowledge_graph reset to PENDING")

    return ctx


def _check_dependencies(ctx: PipelineContext, resume_path: str) -> None:
    errors: list[str] = []
    if ctx.domain_model is None:
        errors.append(
            "ctx.domain_model is None — run Stage 4 first:\n"
            f"       python run_pipeline.py --resume {resume_path} --until stage40_domain"
        )
    if ctx.business_flows is None or not ctx.business_flows.flows:
        errors.append(
            "ctx.business_flows is empty — run Stage 4.5 first:\n"
            f"       python run_pipeline.py --resume {resume_path} --until stage45_flows"
        )
    if errors:
        for msg in errors:
            print(f"Error: {msg}", file=sys.stderr)
        sys.exit(2)


# ─── --inspect Mode ───────────────────────────────────────────────────────────

def _print_inspect(ctx: PipelineContext) -> None:
    print("\n── Stage 9 Input Inspection ────────────────────────────────────")

    dm = ctx.domain_model
    print(f"\n[domain_model]  {'✅ present' if dm else '❌ MISSING (required)'}")
    if dm:
        print(f"  Domain       : {dm.domain_name}")
        print(f"  Actors       : {len(dm.user_roles)}")
        print(f"  Contexts     : {len(dm.bounded_contexts)}  → {dm.bounded_contexts}")
        print(f"  Features     : {len(dm.features)}")
        print(f"  Key entities : {len(dm.key_entities)}  → {dm.key_entities[:8]}")

    bfc = ctx.business_flows
    print(f"\n[business_flows]  {'✅ present' if bfc else '❌ MISSING (required)'}")
    if bfc:
        print(f"  Flows        : {bfc.total}")
        actors = sorted({f.actor for f in bfc.flows})
        print(f"  Unique actors: {actors}")

    cm = ctx.code_map
    print(f"\n[code_map]  {'✅ present' if cm else '⚠️  absent (optional)'}")
    if cm:
        print(f"  HTML pages   : {len(cm.html_pages)}")
        print(f"  DB tables    : {len(cm.db_schema)}")
        print(f"  SQL queries  : {len(cm.sql_queries)}")

    # Existing output
    out = ctx.output_path(OUTPUT_FILE)
    exists = Path(out).exists()
    print(f"\n[output]  {'✅ exists' if exists else '— not yet built'}: {out}")
    stage_result = ctx.stages.get("stage90_knowledge_graph")
    if stage_result:
        print(f"[stage status]  {stage_result.status.value}")

    print("\n────────────────────────────────────────────────────────────────")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    ctx  = _load_context(args)
    _check_dependencies(ctx, args.resume)

    # Input summary header
    dm  = ctx.domain_model
    bfc = ctx.business_flows
    cm  = ctx.code_map
    print(f"domain_model  : {dm.domain_name if dm else 'absent'}"
          + (f"  ({len(dm.features)} features, {len(dm.key_entities)} entities)" if dm else ""))
    print(f"business_flows: {bfc.total if bfc else 0} flow(s)")
    print(f"code_map      : {'present' if cm else 'absent'}"
          + (f"  ({len(cm.html_pages)} pages, {len(cm.db_schema)} tables)" if cm else ""))
    print(f"stage status  : {ctx.stages['stage90_knowledge_graph'].status.value}")
    print()

    if args.inspect:
        _print_inspect(ctx)
        sys.exit(0)

    # Run the stage
    try:
        stage9_run(ctx)
    except RuntimeError as exc:
        _die(str(exc), code=2)
    except Exception as exc:
        print(f"\nStage 9 raised an exception: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Output summary
    kg = ctx.knowledge_graph_meta
    if kg:
        print(f"\nOutput  → {kg.json_path}")
        print(f"Nodes   : {kg.node_count}  ({', '.join(kg.node_types)})")
        print(f"Edges   : {kg.edge_count}")
        for etype, count in sorted(kg.edge_type_counts.items()):
            print(f"  {etype:<25s} {count}")

    if ctx.stages["stage90_knowledge_graph"].status.value == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
