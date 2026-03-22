"""
run_stage39.py — Standalone runner for Stage 3.9 (Preflight)

Usage:
    python run_stage39.py outputs/run_<timestamp>/context.json
    python run_stage39.py /path/to/php-project
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # locate project root


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_stage39.py <context.json | project-path>")
        sys.exit(1)

    from context import PipelineContext
    from pipeline.stage39_preflight import run as stage39_run, PreflightBlocker

    path = Path(sys.argv[1])
    if path.suffix == ".json" and path.exists():
        ctx = PipelineContext.load(str(path))
        print(f"Resuming run: {ctx.run_id}")
    elif path.is_dir():
        ctx = PipelineContext.create(php_project_path=str(path))
        print(f"New run: {ctx.run_id}")
    else:
        print(f"ERROR: Path not found: {path}")
        sys.exit(1)

    try:
        stage39_run(ctx)
        print(f"Preflight passed — ready for Stage 4.")
        print(f"\nRun Stage 4 next:")
        print(f"  python run_stage4.py {ctx.context_file}")
    except PreflightBlocker as e:
        print(f"\n{e}")
        print(f"\nFix the issues above then re-run:")
        print(f"  python run_stage39.py {ctx.context_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
