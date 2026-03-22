"""
run_stage8.py — Standalone runner for Stage 8 (Test Case Generator)

Generates Gherkin BDD scenarios, Playwright JS end-to-end tests, and pytest
stubs from the pipeline's Acceptance Criteria artefact.

Stage 8 requires upstream outputs:
  ctx.ba_artifacts.ac_path  — required (Stage 5 AC agent)
  ctx.business_flows        — optional (Stage 4.5) — richer scenarios when present
  ctx.domain_model          — optional (Stage 4)   — actor / entity context

Usage
-----
    python run_stage8.py --resume outputs/run_20240101_120000/context.json

    # Force re-run even if already completed:
    python run_stage8.py --resume outputs/run_.../context.json --force

    # Inspect inputs without calling the LLM:
    python run_stage8.py --resume outputs/run_.../context.json --inspect

    # Show how the AC will be chunked without calling the LLM:
    python run_stage8.py --resume outputs/run_.../context.json --dry-run

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
from pipeline.stage80_tests import (
    _chunk_ac,
    _build_context_block,
    OUTPUT_FILES,
    run as stage8_run,
)


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog            = "run_stage8.py",
        description     = "Run Stage 8 (Test Case Generator) standalone.",
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
        help    = "Re-run even if stage80_tests is already marked COMPLETED.",
    )
    parser.add_argument(
        "--inspect",
        action  = "store_true",
        default = False,
        help    = "Print a summary of what Stage 8 will consume, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        dest    = "dry_run",
        action  = "store_true",
        default = False,
        help    = "Show how the AC file will be chunked without calling the LLM.",
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
        ctx.stages["stage80_tests"].status = StageStatus.PENDING
        ctx.stages["stage80_tests"].error  = None
        print("--force     : stage80_tests reset to PENDING")

    return ctx


def _check_dependencies(ctx: PipelineContext, resume_path: str) -> None:
    errors: list[str] = []
    if ctx.ba_artifacts is None:
        errors.append(
            "ctx.ba_artifacts is None — run Stage 5 first:\n"
            f"       python run_pipeline.py --resume {resume_path} --until stage50_ac"
        )
    elif not ctx.ba_artifacts.ac_path:
        errors.append("ctx.ba_artifacts.ac_path is empty — Stage 5 AC agent may have failed.")
    elif not Path(ctx.ba_artifacts.ac_path).exists():
        errors.append(f"AC file not found: {ctx.ba_artifacts.ac_path}")

    if errors:
        for msg in errors:
            print(f"Error: {msg}", file=sys.stderr)
        sys.exit(2)


# ─── --inspect Mode ───────────────────────────────────────────────────────────

def _print_inspect(ctx: PipelineContext) -> None:
    print("\n── Stage 8 Input Inspection ────────────────────────────────────")

    # BA artefacts
    ba = ctx.ba_artifacts
    print(f"\n[ba_artifacts]  {'✅ present' if ba else '❌ MISSING (required)'}")
    if ba:
        for label, path in [("ac_path", ba.ac_path), ("brd_path", ba.brd_path),
                             ("srs_path", ba.srs_path), ("user_stories_path", ba.user_stories_path)]:
            exists = Path(path).exists() if path else False
            tick   = "✅" if exists else ("⚠️ " if path else "—")
            print(f"  {tick} {label}: {path or '(not set)'}")
        if ba.ac_path and Path(ba.ac_path).exists():
            size = Path(ba.ac_path).stat().st_size
            print(f"  AC file size: {size:,} bytes")

    # Business flows
    bfc = ctx.business_flows
    print(f"\n[business_flows]  {'✅ present' if bfc else '⚠️  absent (optional — fewer scenarios)'}")
    if bfc:
        print(f"  Flows: {bfc.total}")

    # Domain model
    dm = ctx.domain_model
    print(f"\n[domain_model]  {'✅ present' if dm else '⚠️  absent (optional)'}")
    if dm:
        print(f"  Domain : {dm.domain_name}")
        print(f"  Actors : {', '.join(r['role'] for r in dm.user_roles) or '(none)'}")

    # Existing outputs
    print("\n[current outputs]")
    for key, fname in OUTPUT_FILES.items():
        path = ctx.output_path(fname)
        exists = Path(path).exists()
        print(f"  {'✅' if exists else '—'} {fname}")

    stage_result = ctx.stages.get("stage80_tests")
    if stage_result:
        print(f"\n[stage80_tests status]  {stage_result.status.value}")

    print("\n────────────────────────────────────────────────────────────────")


# ─── --dry-run Mode ───────────────────────────────────────────────────────────

def _run_dry(ctx: PipelineContext) -> None:
    """Show chunking plan and context block without calling the LLM."""
    print("\n── Dry Run ─────────────────────────────────────────────────────")

    ac_text = Path(ctx.ba_artifacts.ac_path).read_text(encoding="utf-8")
    print(f"AC file     : {ctx.ba_artifacts.ac_path}")
    print(f"AC size     : {len(ac_text):,} chars")

    chunks = _chunk_ac(ac_text)
    print(f"Chunks      : {len(chunks)} (each ≤ 6,000 chars)")
    for i, chunk in enumerate(chunks, 1):
        # Show the first heading in each chunk as a label
        first_heading = next(
            (line.strip() for line in chunk.splitlines() if line.startswith("#")),
            "(no heading)"
        )
        print(f"  Chunk {i}: {len(chunk):,} chars — {first_heading}")

    print(f"\nContext block preview (first 300 chars):")
    ctx_block = _build_context_block(ctx)
    print(ctx_block[:300])

    print(f"\nLLM calls needed : {len(chunks)}")
    print("── Dry run complete — no LLM called, no files written. ─────────")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.inspect and args.dry_run:
        _die("--inspect and --dry-run are mutually exclusive.")

    ctx = _load_context(args)
    _check_dependencies(ctx, args.resume)

    # Header
    ac_size = Path(ctx.ba_artifacts.ac_path).stat().st_size
    flows   = len(ctx.business_flows.flows) if ctx.business_flows else 0
    print(f"AC file     : {ctx.ba_artifacts.ac_path} ({ac_size:,} bytes)")
    print(f"Flows       : {flows}")
    print(f"domain_model: {'present' if ctx.domain_model else 'absent'}")
    print(f"Stage status: {ctx.stages['stage80_tests'].status.value}")
    print()

    if args.inspect:
        _print_inspect(ctx)
        sys.exit(0)

    if args.dry_run:
        _run_dry(ctx)
        sys.exit(0)

    # Normal run
    try:
        stage8_run(ctx)
    except RuntimeError as exc:
        _die(str(exc), code=2)
    except Exception as exc:
        print(f"\nStage 8 raised an exception: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Output summary
    if ctx.test_suite:
        print(f"\nGherkin     → {ctx.test_suite.gherkin_path}")
        print(f"Playwright  → {ctx.test_suite.playwright_path}")
        print(f"pytest      → {ctx.test_suite.pytest_path}")
        print(f"Scenarios   : {ctx.test_suite.scenario_count}")

    if ctx.stages["stage80_tests"].status.value == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
