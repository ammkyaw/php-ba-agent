"""
run_stage6.py — Standalone runner for Stage 6 (QA Review)

Usage:
    python run_stage6.py outputs/run_<timestamp>/context.json
    python run_stage6.py /path/to/php-project
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # locate project root


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_stage6.py <context.json | project-path>")
        sys.exit(1)

    from context import PipelineContext
    from pipeline.stage60_qa import run as stage6_run

    ctx = _load_or_create(sys.argv[1])

    print(f"Run ID : {ctx.run_id}")
    print(f"Project: {ctx.php_project_path}")

    stage6_run(ctx)

    print(f"\nRun Stage 6.5 next:")
    print(f"  python run_stage65.py {ctx.context_file}")


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


if __name__ == "__main__":
    main()
