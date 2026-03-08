"""
run_stage0.py — Standalone runner for Stage 0 (Validate)

Usage:
    python run_stage0.py /path/to/php-project
    python run_stage0.py /path/to/php-project --output-dir my_outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PHP-BA Agent — Stage 0: Validate PHP project"
    )
    p.add_argument("project", help="Path to the PHP project directory")
    p.add_argument("--output-dir", default="outputs",
                   help="Base output directory (default: outputs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_path = Path(args.project)
    if not project_path.exists():
        print(f"ERROR: Project path not found: {project_path}")
        sys.exit(1)

    from context import PipelineContext
    from pipeline.stage0_validate import run as stage0_run, PipelineError

    ctx = PipelineContext.create(
        php_project_path=str(project_path),
        output_base=args.output_dir,
    )
    print(f"New run: {ctx.run_id}")

    try:
        stage0_run(ctx)
    except PipelineError as e:
        print(f"\n✗ BLOCKED: {e}")
        sys.exit(1)

    # Print report summary
    report_path = Path(ctx.output_path("validation_report.json"))
    if report_path.exists():
        report = json.loads(report_path.read_text())
        print(f"\n{'='*55}")
        print(f"  Validation Report")
        print(f"{'='*55}")
        print(f"  Project   : {report['project_path']}")
        print(f"  PHP       : {report['php_version']} ({report['php_binary']})")
        print(f"  Framework : {report['framework_hint']}")
        print(f"  PHP files : {report['php_file_count']}")
        print(f"  Est. lines: {report['estimated_lines']:,}")
        print(f"  Warnings  : {len(report['warnings'])}")
        for w in report["warnings"]:
            print(f"    ⚠  {w}")
        print(f"{'='*55}")
        print(f"  Context   : {ctx.context_file}")
        print(f"\nRun Stage 1 next:")
        print(f"  python run_stage1.py {ctx.context_file}")


if __name__ == "__main__":
    main()
