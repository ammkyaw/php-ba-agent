"""
pipeline/entrypoints/php_entrypoints.py — PHP Entry-Point Detector

Thin re-export of the original stage13_entrypoints logic, now behind a
language-agnostic interface.  stage13_entrypoints.py dispatches here for
PHP / Language.UNKNOWN projects.
"""

from __future__ import annotations

# Re-export every public symbol from the original module so that
# stage13_entrypoints.py can import from here without duplication.
from pipeline.stage13_entrypoints import (   # noqa: F401
    run as _original_run,
    _detect_all,
    _build_catalog,
    _dedup,
    _save_catalog,
    _load_catalog,
    build_ep_type_map,
    # Utility helpers
    _safe_read,
    _rel,
    _extract_regex,
    _humanize,
    _to_slug,
)

from context import EntryPoint, EntryPointCatalog, Framework, PipelineContext
from pathlib import Path


def detect(root: Path, framework: Framework) -> list[EntryPoint]:
    """
    Detect PHP entry points (cron, CLI, webhooks, queue workers).
    Delegates to the original implementation unchanged.
    """
    return _detect_all(root, framework)
