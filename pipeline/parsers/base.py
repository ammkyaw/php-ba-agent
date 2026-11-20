"""
pipeline/parsers/base.py — Abstract Language Parser Interface

Every language adapter (PHP, TypeScript, Java, …) must implement
``LanguageParser`` so that stage10_parse.py can dispatch to the
correct backend without knowing language-specific details.

Contract
--------
• ``detect(project_path)``   — returns True if this adapter can handle the project
• ``parse(project_path, ctx)`` — populates ctx.code_map in-place; returns CodeMap
• ``LANGUAGE``                — class-level Language enum value for this adapter
• ``SUPPORTED_FRAMEWORKS``    — class-level frozenset of Framework values
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from context import CodeMap, Framework, Language, PipelineContext


class LanguageParser(ABC):
    """
    Abstract base class for all language-specific code parsers.

    Subclasses implement three things:
      1. LANGUAGE / SUPPORTED_FRAMEWORKS class attributes (for registry lookup)
      2. detect()  — cheap probe to decide if this parser owns the project
      3. parse()   — full parse returning a populated CodeMap
    """

    #: Language enum value this adapter handles (must override)
    LANGUAGE: ClassVar[Language] = Language.UNKNOWN

    #: Set of Framework values this adapter can detect (must override)
    SUPPORTED_FRAMEWORKS: ClassVar[frozenset[Framework]] = frozenset()

    # ── Abstract interface ────────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def detect(cls, project_path: str) -> bool:
        """
        Return True if this parser can handle the given project.

        Must be cheap — no subprocess calls, at most a few file stat() checks.
        Called by stage05_detect before committing to a parser.

        Args:
            project_path: Absolute path to the project root.

        Returns:
            True if the project appears to be written in this parser's language.
        """

    @abstractmethod
    def parse(self, project_path: str, ctx: PipelineContext) -> CodeMap:
        """
        Parse the project at ``project_path`` and return a populated CodeMap.

        Implementations must:
          • Set ``code_map.language`` to their LANGUAGE value
          • Set ``code_map.language_version`` if detectable
          • Set ``code_map.framework`` to the best matching Framework value
          • Populate all language-neutral CodeMap fields where applicable
          • Write code_map.json to ctx.output_path("code_map.json")

        Args:
            project_path: Absolute path to the project root.
            ctx:          Pipeline context (for output_path() and logging).

        Returns:
            Populated CodeMap instance.

        Raises:
            RuntimeError: On unrecoverable parse failure.
        """

    # ── Shared helpers (available to all subclasses) ──────────────────────────

    @staticmethod
    def iter_source_files(
        root: Path,
        extensions: set[str],
        skip_dirs: set[str] | None = None,
    ):
        """
        Yield all source files under ``root`` with the given extensions,
        skipping common non-source directories.

        Args:
            root:       Project root Path.
            extensions: File extensions to include, e.g. {".ts", ".tsx"}.
            skip_dirs:  Directory names to prune (defaults to common skip set).

        Yields:
            Path objects for each matching source file.
        """
        _default_skip = {
            "vendor", "node_modules", ".git", "cache", "logs", "storage",
            "dist", "build", ".next", ".nuxt", "target", "out", "__pycache__",
            "tests", "test", "spec", "stubs", "fixtures", ".idea", ".vscode",
            ".claude",  # Claude Code worktrees / project files
        }
        skip = skip_dirs if skip_dirs is not None else _default_skip
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            if any(part in skip for part in f.parts):
                continue
            if f.suffix in extensions:
                yield f

    @staticmethod
    def safe_read(path: Path) -> str | None:
        """Read a source file, returning None on any error."""
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

    @staticmethod
    def rel(path: Path, root: Path) -> str:
        """Return path relative to root as a forward-slash POSIX string."""
        try:
            return str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")
