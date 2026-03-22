"""
run_stage4.py — Standalone runner for Stage 4 (Domain Analyst Agent)

Usage:
    python run_stage4.py outputs/run_<timestamp>/context.json
    python run_stage4.py /path/to/php-project
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # locate project root


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_stage4.py <context.json | project-path>")
        sys.exit(1)

    from context import PipelineContext
    from pipeline.stage40_domain import run as stage4_run

    ctx = _load_or_create(sys.argv[1])

    print(f"Run ID : {ctx.run_id}")
    print(f"Project: {ctx.php_project_path}")

    stage4_run(ctx)

    if ctx.domain_model:
        _print_domain_model(ctx.domain_model)

    print(f"\nRun Stage 5 next:")
    print(f"  python run_stage5.py {ctx.context_file}")


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


def _print_domain_model(dm) -> None:
    width = 58
    print(f"\n  {'=' * width}")
    print(f"  Domain Model")
    print(f"  {'=' * width}")
    print(f"  Name       : {dm.domain_name}")
    print(f"  Description: {dm.description}")
    print(f"  Entities   : {', '.join(dm.key_entities)}")
    print(f"  Contexts   : {', '.join(dm.bounded_contexts)}")

    if dm.user_roles:
        print(f"\n  User Roles ({len(dm.user_roles)}):")
        for r in dm.user_roles:
            print(f"    • {r['role']}: {r.get('description', '')}")

    if dm.features:
        print(f"\n  Features ({len(dm.features)}):")
        for f in dm.features:
            tables = f.get("tables", [])
            pages  = f.get("pages", [])
            print(f"    • {f['name']}")
            if tables: print(f"        tables : {', '.join(tables)}")
            if pages:  print(f"        pages  : {', '.join(pages)}")

    if dm.workflows:
        print(f"\n  Workflows ({len(dm.workflows)}):")
        for w in dm.workflows:
            steps = w.get("steps", [])
            print(f"    • {w['name']} ({len(steps)} steps) — actor: {w.get('actor','?')}")

    print(f"  {'=' * width}")


if __name__ == "__main__":
    main()
