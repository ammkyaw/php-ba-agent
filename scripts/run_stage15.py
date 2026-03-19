"""
run_stage15.py — Standalone runner for Stage 1.5: Execution Path Simulator

Usage:
    # Run against an existing pipeline context
    python run_stage15.py outputs/run_20240101_120000/context.json

    # Force re-run even if already completed
    python run_stage15.py outputs/run_.../context.json --force

    # Run directly against a PHP project (creates a minimal context)
    python run_stage15.py --project /path/to/php-project
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # locate project root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1.5 — Execution Path Simulator (standalone)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("context_file", nargs="?",
                        help="Path to an existing context.json")
    source.add_argument("--project", metavar="PATH",
                        help="Path to PHP project (runs stages 0+1+1.5)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if already completed")
    args = parser.parse_args()

    from context import PipelineContext, StageStatus

    if args.project:
        # Quick path: parse project on the fly then run stage15
        project_path = Path(args.project)
        if not project_path.exists():
            print(f"Project path not found: {project_path}")
            sys.exit(1)

        ctx = PipelineContext.create(php_project_path=str(project_path))
        print(f"New run: {ctx.run_id}")

        from pipeline.stage00_validate import run as stage0
        from pipeline.stage10_parse    import run as stage1

        print("Running stage00_validate ...")
        stage0(ctx)
        ctx.stage("stage00_validate").mark_completed()
        ctx.save()

        print("Running stage10_parse ...")
        stage1(ctx)
        ctx.stage("stage10_parse").mark_completed()
        ctx.save()

    else:
        ctx_file = Path(args.context_file)
        if not ctx_file.exists():
            print(f"Context file not found: {ctx_file}")
            sys.exit(1)
        ctx = PipelineContext.load(str(ctx_file))
        print(f"Loaded run: {ctx.run_id}")

    if ctx.code_map is None:
        print("ERROR: code_map is None — run Stage 1 first.")
        sys.exit(1)

    # Force reset if requested
    if args.force:
        ctx.stage("stage15_paths").status = StageStatus.PENDING
        import os
        out = ctx.output_path("execution_paths.json")
        if os.path.exists(out):
            os.remove(out)
            print(f"Removed existing {out}")
        ctx.save()

    print("Running stage15_paths ...")
    ctx.stage("stage15_paths").mark_running()
    ctx.save()

    from pipeline.stage15_paths import run as stage15
    stage15(ctx)

    print(f"\nstage15_paths completed.")
    print(f"Output: {ctx.output_path('execution_paths.json')}")
    print(f"\nRun Stage 2 next:")
    print(f"  python run_stage2.py {ctx.context_file}")


if __name__ == "__main__":
    main()
