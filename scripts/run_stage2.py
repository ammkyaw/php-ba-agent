"""
run_stage2.py — Standalone runner for Stage 2 (Knowledge Graph)

Usage:
    # Start a new run from a project path (runs Stage 1 first if needed)
    python run_stage2.py /path/to/php-project

    # Resume an existing run
    python run_stage2.py outputs/run_20260308_104553/context.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # locate project root


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_stage2.py <project-path | context.json>")
        sys.exit(1)

    from context import PipelineContext
    from pipeline.stage2_graph import run, load_graph, summarise_graph

    ctx = _load_or_create(sys.argv[1])

    print(f"Run ID: {ctx.run_id}")
    run(ctx)

    G = load_graph(ctx.graph_meta.graph_path)
    print(json.dumps(summarise_graph(G), indent=2))
    print(f"\nRun Stage 3 next:")
    print(f"  python run_stage3.py {ctx.context_file}")


def _load_or_create(arg: str):
    from context import PipelineContext

    path = Path(arg)

    if path.suffix == ".json" and path.exists():
        ctx = PipelineContext.load(str(path))
        print(f"Resuming run: {ctx.run_id}")
        return ctx

    if not path.exists():
        print(f"ERROR: Path not found: {path}")
        sys.exit(1)

    # New run — need to run Stage 1 first
    from pipeline.stage1_parse import run as stage1_run
    ctx = PipelineContext.create(php_project_path=str(path))
    print(f"New run: {ctx.run_id} — running Stage 1 first...")
    stage1_run(ctx)
    return ctx


if __name__ == "__main__":
    main()
