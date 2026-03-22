"""
run_stage7.py — Standalone runner for Stage 7 (PDF Conversion & Delivery Bundle)

Usage:
    python run_stage7.py outputs/run_<timestamp>/context.json

Requires one of:
    LibreOffice: brew install --cask libreoffice  (macOS)
                 sudo apt install libreoffice      (Linux)
    docx2pdf:    pip install docx2pdf              (requires MS Word on macOS/Windows)
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # locate project root


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_stage7.py <context.json>")
        sys.exit(1)

    from context import PipelineContext
    from pipeline.stage70_pdf import run as stage7_run

    ctx = _load(sys.argv[1])
    print(f"Run ID : {ctx.run_id}")
    print(f"Project: {ctx.php_project_path}")

    stage7_run(ctx)

    delivery = Path(ctx.output_dir) / "delivery"
    print(f"\n📦 Delivery bundle ready:")
    if delivery.exists():
        for f in sorted(delivery.iterdir()):
            size = f.stat().st_size
            print(f"   {f.name:<50} {size:>10,} bytes")


def _load(arg: str) -> "PipelineContext":
    from context import PipelineContext
    path = Path(arg)
    if path.suffix == ".json" and path.exists():
        ctx = PipelineContext.load(str(path))
        print(f"Resuming run: {ctx.run_id}")
        return ctx
    print(f"ERROR: context.json not found: {path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
