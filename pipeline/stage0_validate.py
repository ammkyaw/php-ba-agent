"""
pipeline/stage0_validate.py — Input Validation Stage

The first gate in the pipeline. Validates that the PHP project path is
usable before any expensive work (parsing, graph building, embedding) begins.

Checks performed
----------------
HARD BLOCKERS (raise PipelineError — pipeline cannot continue):
    1. Project path exists and is a directory
    2. At least one .php file is present
    3. PHP binary is available on PATH (or PHPBA_PHP_BIN env override)
    4. PHP version is >= 5.6 (minimum for nikic/php-parser v4+)
    5. nikic/php-parser is installed (composer vendor dir or phar present)
    6. parse_project.php script is locatable

WARNINGS (stored in ctx — pipeline continues):
    W1. No composer.json found (framework detection may be limited)
    W2. PHP version < 7.4 (some AST features limited, BA quality may be lower)
    W3. Project has > 500 PHP files (large project — Stage 1 may be slow)
    W4. Project has > 50,000 lines of PHP (ditto)
    W5. No .env or config file found (DB credentials/config won't be mapped)

Resume behaviour
----------------
If stage0_validate is already COMPLETED, the stage is skipped on resume
unless --force stage0_validate is passed.

Output written to context
-------------------------
    ctx.stages["stage0_validate"]  — COMPLETED or FAILED
    ctx.code_map.php_version       — detected PHP version string (e.g. "8.1.0")
    ctx.code_map.framework         — preliminary framework hint (refined in Stage 1)
    ctx.code_map.total_files       — PHP file count (exact count done in Stage 1)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from context import CodeMap, Framework, PipelineContext


# ── Environment overrides ─────────────────────────────────────────────────────
PHP_BIN_ENV    = "PHPBA_PHP_BIN"          # override PHP binary path
PARSER_ENV     = "PHPBA_PARSER_SCRIPT"    # override parse_project.php path
MIN_PHP_VER    = (5, 6)                   # hard minimum
WARN_PHP_VER   = (7, 4)                   # warn if below this
WARN_FILE_COUNT = 500                     # warn if > N php files
WARN_LINE_COUNT = 50_000                  # warn if > N total lines


class PipelineError(RuntimeError):
    """Hard blocker — pipeline cannot continue past Stage 0."""
    pass


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 0 entry point. Validates the PHP project and environment.

    Populates ctx.code_map with preliminary metadata (php_version, framework,
    rough file count). Records warnings in the stage result for downstream
    stages to inspect.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        PipelineError: On any hard blocker. Message includes remediation advice.
    """
    if ctx.is_stage_done("stage0_validate"):
        print("  [stage0] Already completed — skipping.")
        return

    project_path = Path(ctx.php_project_path)
    warnings:  list[str] = []
    blockers:  list[str] = []

    print(f"  [stage0] Validating project: {project_path}")

    # ── BLOCKER 1: path exists ────────────────────────────────────────────────
    if not project_path.exists():
        raise PipelineError(
            f"[stage0] Project path does not exist: {project_path}"
        )
    if not project_path.is_dir():
        raise PipelineError(
            f"[stage0] Project path is not a directory: {project_path}"
        )

    # ── BLOCKER 2: has PHP files ──────────────────────────────────────────────
    php_files = _count_php_files(project_path)
    if php_files == 0:
        raise PipelineError(
            f"[stage0] No .php files found under {project_path}. "
            "Check the path points to the project root."
        )
    print(f"  [stage0] Found {php_files} PHP file(s)")

    # ── BLOCKER 3: PHP binary available ──────────────────────────────────────
    php_bin = _find_php_binary()
    if php_bin is None:
        raise PipelineError(
            "[stage0] PHP binary not found on PATH.\n"
            "Install PHP (brew install php / apt install php-cli) or set "
            f"the {PHP_BIN_ENV} environment variable to the full path."
        )
    print(f"  [stage0] PHP binary: {php_bin}")

    # ── BLOCKER 4: PHP version >= 5.6 ────────────────────────────────────────
    php_version_str = _get_php_version(php_bin)
    if php_version_str is None:
        raise PipelineError(
            f"[stage0] Could not determine PHP version from: {php_bin}"
        )

    php_ver_tuple = _parse_version_tuple(php_version_str)
    if php_ver_tuple < MIN_PHP_VER:
        raise PipelineError(
            f"[stage0] PHP {php_version_str} is too old. "
            f"Minimum required: {'.'.join(str(v) for v in MIN_PHP_VER)}. "
            "Please upgrade PHP."
        )
    print(f"  [stage0] PHP version: {php_version_str}")

    # ── BLOCKER 5: nikic/php-parser installed ─────────────────────────────────
    parser_ok, parser_msg = _check_parser_installed(project_path)
    if not parser_ok:
        agent_root = Path(__file__).parent.parent
        raise PipelineError(
            f"[stage0] nikic/php-parser not found. {parser_msg}\n"
            f"Run from the BA agent directory:\n"
            f"  cd {agent_root} && composer require nikic/php-parser"
        )

    # ── BLOCKER 6: parse_project.php locatable ────────────────────────────────
    parser_script = _find_parser_script(project_path)
    if parser_script is None:
        raise PipelineError(
            "[stage0] parse_project.php not found.\n"
            "Expected locations:\n"
            "  • Same directory as run_pipeline.py\n"
            "  • Project root\n"
            f"  • Path set via {PARSER_ENV} environment variable"
        )
    print(f"  [stage0] Parser script: {parser_script}")

    # ── WARNING W1: no composer.json ─────────────────────────────────────────
    if not (project_path / "composer.json").exists():
        warnings.append(
            "No composer.json found — framework detection will rely on "
            "directory structure heuristics only."
        )

    # ── WARNING W2: PHP version below 7.4 ────────────────────────────────────
    if php_ver_tuple < WARN_PHP_VER:
        warnings.append(
            f"PHP {php_version_str} is below 7.4. Some modern PHP AST features "
            "will be unavailable; BA document quality may be reduced."
        )

    # ── WARNING W3/W4: large project ─────────────────────────────────────────
    if php_files > WARN_FILE_COUNT:
        warnings.append(
            f"Large project: {php_files} PHP files found. "
            "Stage 1 parsing may take several minutes."
        )

    total_lines = _estimate_line_count(project_path)
    if total_lines > WARN_LINE_COUNT:
        warnings.append(
            f"Large codebase: ~{total_lines:,} lines of PHP. "
            "Consider using --until stage3_embed to validate parsing first."
        )

    # ── WARNING W5: no config/env file ───────────────────────────────────────
    config_hints = [".env", ".env.example", "config.php", "database.php",
                    "wp-config.php", "application/config/database.php"]
    has_config = any(
        (project_path / hint).exists() or
        any(project_path.rglob(hint))
        for hint in config_hints
    )
    if not has_config:
        warnings.append(
            "No .env or database config file found. "
            "DB connection details and table names may not be fully mapped."
        )

    # ── Preliminary framework detection ──────────────────────────────────────
    framework = _detect_framework_hint(project_path)
    print(f"  [stage0] Framework hint: {framework.value}")

    # ── Emit warnings ─────────────────────────────────────────────────────────
    if warnings:
        print(f"  [stage0] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    ⚠  {w}")

    # ── Populate CodeMap with preliminary data ────────────────────────────────
    if ctx.code_map is None:
        ctx.code_map = CodeMap()

    ctx.code_map.php_version = php_version_str
    ctx.code_map.framework   = framework
    ctx.code_map.total_files = php_files   # rough count; Stage 1 will be exact

    # ── Save validation report ────────────────────────────────────────────────
    report_path = ctx.output_path("validation_report.json")
    _save_report(
        path         = report_path,
        project_path = str(project_path),
        php_bin      = php_bin,
        php_version  = php_version_str,
        parser_script= parser_script,
        php_files    = php_files,
        total_lines  = total_lines,
        framework    = framework.value,
        warnings     = warnings,
    )

    ctx.stage("stage0_validate").mark_completed(report_path)
    ctx.save()

    status = "✓ PASSED" if not warnings else f"✓ PASSED with {len(warnings)} warning(s)"
    print(f"  [stage0] Validation {status} → {report_path}")


# ─── Validation Helpers ────────────────────────────────────────────────────────

def _count_php_files(project_path: Path) -> int:
    """Count all .php files under the project root, excluding vendor/."""
    count = 0
    for f in project_path.rglob("*.php"):
        # Skip vendor and node_modules
        parts = f.parts
        if "vendor" in parts or "node_modules" in parts:
            continue
        count += 1
    return count


def _estimate_line_count(project_path: Path) -> int:
    """
    Estimate total PHP line count by sampling up to 200 files.
    Fast approximation — Stage 1 gets the exact count.
    """
    total = 0
    files = [
        f for f in project_path.rglob("*.php")
        if "vendor" not in f.parts and "node_modules" not in f.parts
    ]
    # Sample at most 200 files to keep validation fast
    sample = files[:200]
    for f in sample:
        try:
            total += sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))
        except OSError:
            pass
    # Extrapolate if we sampled fewer than all files
    if len(sample) < len(files) and len(sample) > 0:
        total = int(total * len(files) / len(sample))
    return total


def _find_php_binary() -> Optional[str]:
    """
    Locate the PHP CLI binary.
    Checks PHPBA_PHP_BIN env var first, then common locations, then PATH.
    """
    # Environment override
    env_bin = os.environ.get(PHP_BIN_ENV)
    if env_bin and Path(env_bin).is_file():
        return env_bin

    # Common explicit paths (macOS Homebrew, Linux distros)
    candidates = [
        "/usr/bin/php",
        "/usr/local/bin/php",
        "/opt/homebrew/bin/php",
        "/usr/bin/php8.1",
        "/usr/bin/php8.0",
        "/usr/bin/php7.4",
    ]
    for c in candidates:
        if Path(c).is_file():
            return c

    # Fall back to PATH lookup via `which`
    try:
        result = subprocess.run(
            ["which", "php"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            found = result.stdout.strip()
            if found:
                return found
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Windows: `where php`
    try:
        result = subprocess.run(
            ["where", "php"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if lines:
                return lines[0].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def _get_php_version(php_bin: str) -> Optional[str]:
    """
    Run `php --version` and extract the version string (e.g. "8.1.12").
    Returns None if the command fails.
    """
    try:
        result = subprocess.run(
            [php_bin, "--version"],
            capture_output=True, text=True, timeout=10
        )
        # Output: "PHP 8.1.12 (cli) ..."
        match = re.search(r"PHP\s+(\d+\.\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """Convert "8.1.12" → (8, 1, 12). Returns (0,) on parse failure."""
    try:
        return tuple(int(x) for x in version_str.split(".")[:3])
    except ValueError:
        return (0,)


def _check_parser_installed(project_path: Path) -> tuple[bool, str]:
    """
    Check that nikic/php-parser is available. Accepts:
      - vendor/nikic/php-parser/  in the BA agent's own directory (primary)
      - vendor/nikic/php-parser/  in the analyzed PHP project (legacy)
      - phpparser.phar             (standalone phar anywhere on search path)
    """
    # ── Primary: BA agent's own vendor/ (installed via composer in agent root) ──
    agent_root = Path(__file__).parent.parent
    agent_vendor = agent_root / "vendor" / "nikic" / "php-parser"
    if agent_vendor.is_dir():
        return True, f"Found at {agent_vendor}"

    # ── Fallback: vendor/ inside the analyzed PHP project ─────────────────────
    project_vendor = project_path / "vendor" / "nikic" / "php-parser"
    if project_vendor.is_dir():
        return True, f"Found at {project_vendor}"

    # ── Phar: project root, pipeline dir, agent root, or cwd ─────────────────
    for search_dir in [project_path, Path(__file__).parent, agent_root, Path.cwd()]:
        phar = search_dir / "phpparser.phar"
        if phar.is_file():
            return True, f"Found phar at {phar}"

    return False, (
        f"nikic/php-parser not found in {agent_vendor} "
        "or as phpparser.phar.\n"
        f"Fix: cd {agent_root} && composer require nikic/php-parser"
    )


def _find_parser_script(project_path: Path) -> Optional[str]:
    """
    Locate parse_project.php. Checks (in order):
      1. PHPBA_PARSER_SCRIPT env var
      2. Same directory as this Python file (pipeline/)
      3. Parent of pipeline/ (project root)
      4. Project root itself
    """
    env_path = os.environ.get(PARSER_ENV)
    if env_path and Path(env_path).is_file():
        return env_path

    search_dirs = [
        Path(__file__).parent,                       # pipeline/
        Path(__file__).parent.parent / "parsers",    # project root/parsers/
        Path(__file__).parent.parent,                # project root
        project_path,                                # the PHP project itself
        Path.cwd() / "parsers",                      # cwd/parsers/
        Path.cwd(),                                  # wherever we're running from
    ]
    for d in search_dirs:
        candidate = d / "parse_project.php"
        if candidate.is_file():
            return str(candidate)

    return None


def _detect_framework_hint(project_path: Path) -> Framework:
    """
    Quick heuristic framework detection based on directory structure and files.
    Stage 1 refines this with full composer.json + file-content analysis.
    """
    # Laravel
    if (project_path / "artisan").exists():
        return Framework.LARAVEL
    if (project_path / "app" / "Http" / "Controllers").is_dir():
        return Framework.LARAVEL

    # Symfony
    if (project_path / "bin" / "console").exists():
        return Framework.SYMFONY
    if (project_path / "config" / "bundles.php").exists():
        return Framework.SYMFONY

    # WordPress
    if (project_path / "wp-config.php").exists():
        return Framework.WORDPRESS
    if (project_path / "wp-includes").is_dir():
        return Framework.WORDPRESS

    # CodeIgniter
    if (project_path / "application" / "config" / "config.php").exists():
        return Framework.CODEIGNITER
    if (project_path / "system" / "core" / "CodeIgniter.php").exists():
        return Framework.CODEIGNITER

    # composer.json check
    composer_json = project_path / "composer.json"
    if composer_json.exists():
        try:
            import json
            data = json.loads(composer_json.read_text(encoding="utf-8"))
            require = {**data.get("require", {}), **data.get("require-dev", {})}
            if "laravel/framework" in require:
                return Framework.LARAVEL
            if "symfony/framework-bundle" in require:
                return Framework.SYMFONY
            if any("codeigniter" in k.lower() for k in require):
                return Framework.CODEIGNITER
            if "roots/wordpress" in require or "johnpbloch/wordpress" in require:
                return Framework.WORDPRESS
            # Has composer.json but no known framework — likely custom PHP
            return Framework.RAW_PHP
        except Exception:
            pass

    return Framework.RAW_PHP


def _save_report(
    path: str,
    project_path: str,
    php_bin: str,
    php_version: str,
    parser_script: str,
    php_files: int,
    total_lines: int,
    framework: str,
    warnings: list[str],
) -> None:
    """Persist a JSON validation report for audit and debugging."""
    import json
    report = {
        "project_path":   project_path,
        "php_binary":     php_bin,
        "php_version":    php_version,
        "parser_script":  parser_script,
        "php_file_count": php_files,
        "estimated_lines": total_lines,
        "framework_hint": framework,
        "warnings":       warnings,
        "blockers":       [],
        "passed":         True,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
