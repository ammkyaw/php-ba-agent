"""
run_stage62.py — Standalone runner for Stage 6.2 (Architecture Reconstruction)

Runs the Architecture Reconstruction agent against an existing pipeline run
without re-running the full pipeline.  Useful when:
  • Stage 6.2 was skipped or failed and you want to retry it alone
  • You edited domain_model or business_flows and want a fresh architecture doc
  • You are developing / tuning the Stage 6.2 prompt and want fast iteration

Stage 6.2 requires upstream context produced by earlier stages:
  ctx.domain_model    — required (Stage 4)
  ctx.business_flows  — required (Stage 4.5)
  ctx.code_map        — optional (Stage 1)   richer output when present
  ctx.graph_meta      — optional (Stage 2)   richer output when present
  ctx.qa_result       — optional (Stage 6)   QA notes included when present

Usage
-----
    python run_stage62.py --resume outputs/run_20240101_120000/context.json

    # Force re-run even if already completed:
    python run_stage62.py --resume outputs/run_.../context.json --force

    # Dry-run: call the LLM and print result without writing files:
    python run_stage62.py --resume outputs/run_.../context.json --dry-run

    # Inspect ctx inputs without calling the LLM:
    python run_stage62.py --resume outputs/run_.../context.json --inspect

Exit codes
----------
0  stage completed (or already done and --force not given)
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
from pipeline.stage62_architecture import (
    _build_context_block,
    _call_llm,
    _parse_json,
    _validate_schema,
    _SYSTEM_PROMPT,
    _USER_PROMPT_TEMPLATE,
    run as stage62_run,
)


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog            = "run_stage62.py",
        description     = "Run Stage 6.2 (Architecture Reconstruction) standalone.",
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
        help    = "Re-run even if stage62_architecture is already marked COMPLETED.",
    )
    parser.add_argument(
        "--dry-run",
        dest    = "dry_run",
        action  = "store_true",
        default = False,
        help    = "Call the LLM and print the raw response without writing any files "
                  "or updating context.json.",
    )
    parser.add_argument(
        "--inspect",
        action  = "store_true",
        default = False,
        help    = "Print a summary of ctx fields Stage 6.2 will read, then exit "
                  "without calling the LLM.",
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
    print(f"Loaded run : {ctx.run_id}")
    print(f"Output dir : {ctx.output_dir}")

    if args.force:
        ctx.stages["stage62_architecture"].status = StageStatus.PENDING
        ctx.stages["stage62_architecture"].error  = None
        print("--force    : stage62_architecture reset to PENDING")

    return ctx


def _check_dependencies(ctx: PipelineContext, args: argparse.Namespace) -> None:
    """Abort early if required upstream stages have not produced their output."""
    errors: list[str] = []
    if ctx.domain_model is None:
        errors.append(
            "ctx.domain_model is None — run Stage 4 first:\n"
            f"       python run_pipeline.py --resume {args.resume} --until stage4_domain"
        )
    if ctx.business_flows is None:
        errors.append(
            "ctx.business_flows is None — run Stage 4.5 first:\n"
            f"       python run_pipeline.py --resume {args.resume} --until stage45_flows"
        )
    if errors:
        for msg in errors:
            print(f"Error: {msg}", file=sys.stderr)
        sys.exit(2)


# ─── --inspect Mode ───────────────────────────────────────────────────────────

def _print_inspect(ctx: PipelineContext) -> None:
    """Print a human-readable summary of what Stage 6.2 will consume."""
    dm  = ctx.domain_model
    bfc = ctx.business_flows
    cm  = ctx.code_map
    gm  = ctx.graph_meta
    qa  = ctx.qa_result

    print("\n── Stage 6.2 Input Inspection ──────────────────────────────────")

    print(f"\n[domain_model]  {'✅ present' if dm else '❌ MISSING (required)'}")
    if dm:
        print(f"  Domain          : {dm.domain_name}")
        print(f"  Bounded contexts: {', '.join(dm.bounded_contexts) or '(none)'}")
        print(f"  Key entities    : {', '.join(dm.key_entities) or '(none)'}")
        print(f"  Features        : {len(dm.features)}")
        print(f"  User roles      : {len(dm.user_roles)}")

    print(f"\n[business_flows]  {'✅ present' if bfc else '❌ MISSING (required)'}")
    if bfc:
        print(f"  Total flows  : {bfc.total}")
        for flow in bfc.flows[:3]:
            print(f"  • {flow.flow_id}: {flow.name} ({len(flow.steps)} steps)")
        if len(bfc.flows) > 3:
            print(f"    … and {len(bfc.flows) - 3} more")

    print(f"\n[code_map]  {'✅ present' if cm else '⚠️  absent (optional — less detailed output)'}")
    if cm:
        print(f"  Framework : {cm.framework.value}")
        print(f"  Files     : {cm.total_files}   Lines: {cm.total_lines}")
        print(f"  DB tables : {len(cm.db_schema)}")

    print(f"\n[graph_meta]  {'✅ present' if gm else '⚠️  absent (optional)'}")
    if gm:
        print(f"  Nodes: {gm.node_count}   Edges: {gm.edge_count}")

    print(f"\n[qa_result]  {'✅ present' if qa else '⚠️  absent (optional)'}")
    if qa:
        print(f"  Passed: {qa.passed}   Coverage: {qa.coverage_score:.2f}   "
              f"Consistency: {qa.consistency_score:.2f}")

    stage_result = ctx.stages.get("stage62_architecture")
    if stage_result:
        print(f"\n[stage62_architecture status]  {stage_result.status.value}")
        if stage_result.output_path:
            print(f"  Previous output: {stage_result.output_path}")

    print("\n────────────────────────────────────────────────────────────────")


# ─── --dry-run Mode ───────────────────────────────────────────────────────────

def _run_dry(ctx: PipelineContext) -> None:
    """Call the LLM and print response + parse result — no files written."""
    print("\n── Dry Run: building context block ────────────────────────────")
    context_block = _build_context_block(ctx)
    print(f"Context block : {len(context_block)} chars")
    print(f"Preview (first 400 chars):\n{context_block[:400]}\n")

    print("── Calling LLM ─────────────────────────────────────────────────")
    raw = _call_llm(_SYSTEM_PROMPT,
                    _USER_PROMPT_TEMPLATE.format(context_block=context_block))

    print("\n── Raw LLM Response ────────────────────────────────────────────")
    print(raw)

    print("\n── Parsing & Validating JSON ───────────────────────────────────")
    try:
        data = _parse_json(raw)
        _validate_schema(data)
        print("✅ Valid JSON — schema OK")
        print(f"  Components      : {len(data.get('components', []))}")
        print(f"  Data flows      : {len(data.get('data_flows', []))}")
        print(f"  Sequence flows  : {len(data.get('sequence_flows', []))}")
        print(f"  Int. points     : {len(data.get('integration_points', []))}")
        stack = data.get("technology_observations", {}).get("stack", [])
        print(f"  Tech stack      : {', '.join(stack) or '(none)'}")
    except ValueError as exc:
        print(f"❌ Parse/validation error: {exc}")
        sys.exit(1)

    print("\n── Dry run complete — no files written, ctx not updated. ───────")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Mutual-exclusion guard
    if args.inspect and args.dry_run:
        _die("--inspect and --dry-run are mutually exclusive.")

    # Load & validate context
    ctx = _load_context(args)
    _check_dependencies(ctx, args)

    # Print a consistent header
    flows_count = len(ctx.business_flows.flows) if ctx.business_flows else 0
    print(f"Domain         : {ctx.domain_model.domain_name}")
    print(f"Business flows : {flows_count}")
    print(f"code_map       : {'present' if ctx.code_map   else 'absent'}")
    print(f"graph_meta     : {'present' if ctx.graph_meta else 'absent'}")
    print(f"qa_result      : {'present' if ctx.qa_result  else 'absent'}")
    print(f"Stage status   : {ctx.stages['stage62_architecture'].status.value}")
    print()

    # --inspect: summarise inputs and exit
    if args.inspect:
        _print_inspect(ctx)
        sys.exit(0)

    # --dry-run: call LLM, print result, no file I/O
    if args.dry_run:
        try:
            _run_dry(ctx)
        except Exception as exc:
            print(f"\nDry run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)

    # Normal run
    try:
        stage62_run(ctx)
    except RuntimeError as exc:
        _die(str(exc), code=2)
    except Exception as exc:
        print(f"\nStage 6.2 raised an exception: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Print output summary
    if ctx.architecture_meta:
        print(f"\narchitecture.json → {ctx.architecture_meta.json_path}")
        print(f"architecture.md   → {ctx.architecture_meta.md_path}")
        print(f"Components        : {ctx.architecture_meta.component_count}")
        print(f"Data flows        : {ctx.architecture_meta.data_flow_count}")
        print(f"Sequence flows    : {ctx.architecture_meta.sequence_count}")
        if ctx.architecture_meta.tech_stack:
            print(f"Tech stack        : {', '.join(ctx.architecture_meta.tech_stack)}")

    print(f"\nNext steps:")
    print(f"  python run_stage65.py --resume {ctx.context_file}")
    print(f"  python run_stage67.py --resume {ctx.context_file}")

    if ctx.stages["stage62_architecture"].status.value == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
