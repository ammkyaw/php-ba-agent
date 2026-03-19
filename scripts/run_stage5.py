"""
run_stage5.py — Standalone runner for Stage 5 (BA Document Generation)

Usage:
    python run_stage5.py outputs/run_<timestamp>/context.json
    python run_stage5.py /path/to/php-project

Individual agents can be forced to re-run even if already completed:
    python run_stage5.py outputs/run_.../context.json --force brd
    python run_stage5.py outputs/run_.../context.json --force srs
    python run_stage5.py outputs/run_.../context.json --force ac
    python run_stage5.py outputs/run_.../context.json --force userstories
    python run_stage5.py outputs/run_.../context.json --force all
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # locate project root


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 — BA Document Generation")
    parser.add_argument("context", help="context.json path or PHP project path")
    parser.add_argument(
        "--force",
        choices=["brd", "srs", "ac", "userstories", "all"],
        help="Force re-run a specific agent even if already completed",
    )
    args = parser.parse_args()

    from context import PipelineContext, StageStatus
    from pipeline.stage50_workers import run as stage5_run

    ctx = _load_or_create(args.context)

    # Apply force flag — reset the relevant stage(s) to PENDING
    if args.force:
        force_map = {
            "brd":         ["stage50_brd"],
            "srs":         ["stage50_srs"],
            "ac":          ["stage50_ac"],
            "userstories": ["stage50_userstories"],
            "all":         ["stage50_brd", "stage50_srs", "stage50_ac", "stage50_userstories"],
        }
        for stage_name in force_map[args.force]:
            ctx.stage(stage_name).status = StageStatus.PENDING
            print(f"  Forcing re-run: {stage_name}")

    print(f"Run ID : {ctx.run_id}")
    print(f"Project: {ctx.php_project_path}")

    asyncio.run(stage5_run(ctx))

    _print_summary(ctx)

    print(f"\nRun Stage 6 next:")
    print(f"  python run_stage6.py {ctx.context_file}")


def _load_or_create(arg: str):
    from context import PipelineContext

    path = Path(arg)
    if path.suffix == ".json" and path.exists():
        ctx = PipelineContext.load(str(path))
        print(f"Resuming run: {ctx.run_id}")
        return ctx
    if path.is_dir():
        ctx = PipelineContext.create(php_project_path=str(path))
        print(f"New run: {ctx.run_id}")
        return ctx
    print(f"ERROR: Path not found: {path}")
    sys.exit(1)


def _print_summary(ctx) -> None:
    if not ctx.ba_artifacts:
        return

    width = 58
    print(f"\n  {'=' * width}")
    print(f"  Stage 5 — BA Artefacts")
    print(f"  {'=' * width}")

    artefacts = [
        ("BRD",          ctx.ba_artifacts.brd_path),
        ("SRS",          ctx.ba_artifacts.srs_path),
        ("AC",           ctx.ba_artifacts.ac_path),
        ("User Stories", ctx.ba_artifacts.user_stories_path),
    ]

    for name, path in artefacts:
        if path and Path(path).exists():
            size = Path(path).stat().st_size
            lines = Path(path).read_text(encoding="utf-8").count("\n")
            print(f"  ✓ {name:<14} {Path(path).name}  ({lines} lines, {size:,} bytes)")
        else:
            print(f"  ✗ {name:<14} not generated")

    print(f"  {'=' * width}")


if __name__ == "__main__":
    main()
