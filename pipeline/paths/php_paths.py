"""
pipeline/paths/php_paths.py — PHP Execution Path Adapter

Thin wrapper that re-exports the full stage15_paths logic behind a
language-agnostic interface.  stage15_paths.py dispatches here for PHP.
"""

from __future__ import annotations

# Re-export the original module's internals unchanged.
# stage15_paths.py already contains all PHP-specific path logic.
from pipeline.stage15_paths import (   # noqa: F401
    analyse_file,
    _collect_php_files,
    _FileAnalyser,
)

from context import CodeMap, PipelineContext
from pathlib import Path


def enrich(ctx: PipelineContext) -> list[dict]:
    """
    Run the PHP static execution-path analysis and return the results.
    Mutates ctx.code_map.execution_paths in-place.

    Returns the list of execution path dicts (same contract as stage15).
    """
    # Delegate to the existing stage15 run() logic by calling it directly.
    # stage15_paths.run() already writes execution_paths.json and mutates ctx.
    from pipeline.stage15_paths import run as _stage15_run  # noqa: PLC0415
    _stage15_run(ctx)
    return ctx.code_map.execution_paths if ctx.code_map else []
