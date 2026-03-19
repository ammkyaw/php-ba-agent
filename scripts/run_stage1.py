"""
run_stage1.py — Standalone runner for Stage 1 (PHP Parsing)

Usage:
    # Start a new run from a project path
    python run_stage1.py /path/to/php-project

    # Resume an existing run (created by run_stage0.py or a previous stage)
    python run_stage1.py outputs/run_20260308_104553/context.json
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # locate project root


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_stage1.py <project-path | context.json>")
        sys.exit(1)

    from context import PipelineContext
    from pipeline.stage10_parse import run, summarise_code_map

    ctx = _load_or_create(sys.argv[1])

    print(f"Run ID: {ctx.run_id}")
    run(ctx)
    print(summarise_code_map(ctx.code_map))
    print(f"\nRun Stage 1.5 next:")
    print(f"  python run_stage15.py {ctx.context_file}")


def _load_or_create(arg: str):
    from context import PipelineContext

    path = Path(arg)

    # If it's an existing .json file → resume
    if path.suffix == ".json" and path.exists():
        ctx = PipelineContext.load(str(path))
        print(f"Resuming run: {ctx.run_id}")
        return ctx

    # Otherwise treat as project path → new run
    if not path.exists():
        print(f"ERROR: Path not found: {path}")
        import sys; sys.exit(1)

    ctx = PipelineContext.create(php_project_path=str(path))
    print(f"New run: {ctx.run_id}")
    return ctx


if __name__ == "__main__":
    main()
