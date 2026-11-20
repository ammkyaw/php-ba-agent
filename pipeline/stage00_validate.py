"""
pipeline/stage00_validate.py — Input Validation Stage

The first gate in the pipeline. Validates that the project path is usable
before any expensive work (parsing, graph building, embedding) begins.
Supports PHP, TypeScript, JavaScript, and Java projects.

Checks performed
----------------
HARD BLOCKERS (raise PipelineError — pipeline cannot continue):
    1. Project path exists and is a directory
    2. At least one supported source file is present
       (.php / .ts / .tsx / .js / .jsx / .java)

PHP-only blockers (skipped for non-PHP projects):
    3. PHP binary is available on PATH (or PHPBA_PHP_BIN env override)
    4. PHP version is >= 5.6
    5. nikic/php-parser is installed
    6. parse_project.php script is locatable

WARNINGS (stored in ctx — pipeline continues):
    W1. No manifest file found (composer.json / package.json / pom.xml)
    W2. PHP version < 7.4 (PHP only)
    W3. Project has > 500 source files (large project warning)
    W4. Project has > 50,000 lines of code
    W5. No .env or config file found

Resume behaviour
----------------
If stage00_validate is already COMPLETED, the stage is skipped on resume
unless --force stage00_validate is passed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from context import CodeMap, Framework, PipelineContext


# ── Environment overrides ─────────────────────────────────────────────────────
PHP_BIN_ENV     = "PHPBA_PHP_BIN"
PARSER_ENV      = "PHPBA_PARSER_SCRIPT"
MIN_PHP_VER     = (5, 6)
WARN_PHP_VER    = (7, 4)
WARN_FILE_COUNT = 500
WARN_LINE_COUNT = 50_000

# Supported source extensions per language
_LANG_EXTENSIONS: dict[str, list[str]] = {
    "php":        [".php"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx", ".mjs", ".cjs"],
    "java":       [".java"],
}
_ALL_EXTENSIONS = [ext for exts in _LANG_EXTENSIONS.values() for ext in exts]
_SKIP_DIRS      = {"vendor", "node_modules", ".git", "__pycache__", "dist",
                   "build", ".next", "out", "target", ".gradle"}


class PipelineError(RuntimeError):
    """Hard blocker — pipeline cannot continue past Stage 0."""
    pass


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    if ctx.is_stage_done("stage00_validate"):
        print("  [stage0] Already completed — skipping.")
        return

    project_path = Path(ctx.php_project_path)
    warnings: list[str] = []

    print(f"  [stage0] Validating project: {project_path}")

    # ── BLOCKER 1: path exists ────────────────────────────────────────────────
    if not project_path.exists():
        raise PipelineError(f"[stage0] Project path does not exist: {project_path}")
    if not project_path.is_dir():
        raise PipelineError(f"[stage0] Project path is not a directory: {project_path}")

    # ── BLOCKER 2: has supported source files ─────────────────────────────────
    file_counts = _count_source_files(project_path)
    total_source = sum(file_counts.values())
    if total_source == 0:
        raise PipelineError(
            f"[stage0] No supported source files found under {project_path}.\n"
            f"Supported: PHP (.php), TypeScript (.ts/.tsx), "
            f"JavaScript (.js/.jsx), Java (.java).\n"
            "Check the path points to the project root."
        )

    primary_lang = max(file_counts, key=lambda k: file_counts[k])
    print(f"  [stage0] Source files: " +
          ", ".join(f"{v} {k}" for k, v in file_counts.items() if v > 0))
    print(f"  [stage0] Primary language detected: {primary_lang}")

    # ── PHP-specific blockers ─────────────────────────────────────────────────
    php_bin        = None
    php_version_str = None
    parser_script  = None

    if file_counts.get("php", 0) > 0:
        # BLOCKER 3: PHP binary
        php_bin = _find_php_binary()
        if php_bin is None:
            raise PipelineError(
                "[stage0] PHP binary not found on PATH.\n"
                "Install PHP (brew install php / apt install php-cli) or set "
                f"the {PHP_BIN_ENV} environment variable to the full path."
            )
        print(f"  [stage0] PHP binary: {php_bin}")

        # BLOCKER 4: PHP version
        php_version_str = _get_php_version(php_bin)
        if php_version_str is None:
            raise PipelineError(
                f"[stage0] Could not determine PHP version from: {php_bin}"
            )
        php_ver_tuple = _parse_version_tuple(php_version_str)
        if php_ver_tuple < MIN_PHP_VER:
            raise PipelineError(
                f"[stage0] PHP {php_version_str} is too old. "
                f"Minimum required: {'.'.join(str(v) for v in MIN_PHP_VER)}."
            )
        print(f"  [stage0] PHP version: {php_version_str}")

        # BLOCKER 5: nikic/php-parser
        parser_ok, parser_msg = _check_parser_installed(project_path)
        if not parser_ok:
            agent_root = Path(__file__).parent.parent
            raise PipelineError(
                f"[stage0] nikic/php-parser not found. {parser_msg}\n"
                f"Run: cd {agent_root} && composer require nikic/php-parser"
            )

        # BLOCKER 6: parse_project.php
        parser_script = _find_parser_script(project_path)
        if parser_script is None:
            raise PipelineError(
                "[stage0] parse_project.php not found.\n"
                "Expected locations: pipeline/, parsers/, or project root.\n"
                f"Override with {PARSER_ENV} environment variable."
            )
        print(f"  [stage0] Parser script: {parser_script}")

        if _parse_version_tuple(php_version_str) < WARN_PHP_VER:
            warnings.append(
                f"PHP {php_version_str} is below 7.4. Some AST features "
                "unavailable; BA quality may be reduced."
            )

    # ── WARNING W1: no manifest file ──────────────────────────────────────────
    manifests = {
        "php":        "composer.json",
        "typescript": "package.json",
        "javascript": "package.json",
        "java":       "pom.xml",
    }
    manifest = manifests.get(primary_lang)
    if manifest and not (project_path / manifest).exists():
        warnings.append(
            f"No {manifest} found — framework detection will rely on "
            "directory structure heuristics only."
        )

    # ── WARNING W3/W4: large project ──────────────────────────────────────────
    if total_source > WARN_FILE_COUNT:
        warnings.append(
            f"Large project: {total_source} source files found. "
            "Stage 1 parsing may take several minutes."
        )
    total_lines = _estimate_line_count(project_path, primary_lang)
    if total_lines > WARN_LINE_COUNT:
        warnings.append(
            f"Large codebase: ~{total_lines:,} lines. "
            "Consider --until stage30_embed to validate parsing first."
        )

    # ── WARNING W5: no config/env file ────────────────────────────────────────
    config_hints = [".env", ".env.example", ".env.local", "config.php",
                    "database.php", "wp-config.php", "application.properties",
                    "application.yml", "appsettings.json"]
    if not any(
        (project_path / h).exists() or any(project_path.rglob(h))
        for h in config_hints
    ):
        warnings.append(
            "No .env or config file found. "
            "DB/API connection details may not be fully mapped."
        )

    # ── Framework detection ───────────────────────────────────────────────────
    framework = _detect_framework_hint(project_path, primary_lang)
    print(f"  [stage0] Framework hint: {framework.value}")

    # ── Emit warnings ─────────────────────────────────────────────────────────
    if warnings:
        print(f"  [stage0] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    ⚠  {w}")

    # ── Populate CodeMap ──────────────────────────────────────────────────────
    if ctx.code_map is None:
        ctx.code_map = CodeMap()

    ctx.code_map.php_version = php_version_str
    ctx.code_map.framework   = framework
    ctx.code_map.total_files = total_source

    # ── Save validation report ────────────────────────────────────────────────
    report_path = ctx.output_path("validation_report.json")
    report = {
        "project_path":    str(project_path),
        "primary_language": primary_lang,
        "file_counts":     file_counts,
        "php_binary":      php_bin,
        "php_version":     php_version_str,
        "parser_script":   parser_script,
        "estimated_lines": total_lines,
        "framework_hint":  framework.value,
        "warnings":        warnings,
        "passed":          True,
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    ctx.stage("stage00_validate").mark_completed(report_path)
    ctx.save()

    status = "✓ PASSED" if not warnings else f"✓ PASSED with {len(warnings)} warning(s)"
    print(f"  [stage0] Validation {status} → {report_path}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _count_source_files(project_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {lang: 0 for lang in _LANG_EXTENSIONS}
    for f in project_path.rglob("*"):
        if any(part in _SKIP_DIRS for part in f.parts):
            continue
        ext = f.suffix.lower()
        for lang, exts in _LANG_EXTENSIONS.items():
            if ext in exts:
                counts[lang] += 1
    return counts


def _estimate_line_count(project_path: Path, primary_lang: str) -> int:
    exts = _LANG_EXTENSIONS.get(primary_lang, [])
    files = [
        f for f in project_path.rglob("*")
        if f.suffix.lower() in exts
        and not any(p in _SKIP_DIRS for p in f.parts)
    ]
    sample = files[:200]
    total = 0
    for f in sample:
        try:
            total += sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))
        except OSError:
            pass
    if len(sample) < len(files) and sample:
        total = int(total * len(files) / len(sample))
    return total


def _find_php_binary() -> Optional[str]:
    env_bin = os.environ.get(PHP_BIN_ENV)
    if env_bin and Path(env_bin).is_file():
        return env_bin
    candidates = ["/usr/bin/php", "/usr/local/bin/php", "/opt/homebrew/bin/php",
                  "/usr/bin/php8.1", "/usr/bin/php8.0", "/usr/bin/php7.4"]
    for c in candidates:
        if Path(c).is_file():
            return c
    for cmd in (["which", "php"], ["where", "php"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                lines = r.stdout.strip().splitlines()
                if lines:
                    return lines[0].strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return None


def _get_php_version(php_bin: str) -> Optional[str]:
    try:
        r = subprocess.run([php_bin, "--version"],
                           capture_output=True, text=True, timeout=10)
        m = re.search(r"PHP\s+(\d+\.\d+\.\d+)", r.stdout)
        if m:
            return m.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in version_str.split(".")[:3])
    except ValueError:
        return (0,)


def _check_parser_installed(project_path: Path) -> tuple[bool, str]:
    agent_root = Path(__file__).parent.parent
    agent_vendor = agent_root / "vendor" / "nikic" / "php-parser"
    if agent_vendor.is_dir():
        return True, f"Found at {agent_vendor}"
    project_vendor = project_path / "vendor" / "nikic" / "php-parser"
    if project_vendor.is_dir():
        return True, f"Found at {project_vendor}"
    for d in [project_path, Path(__file__).parent, agent_root, Path.cwd()]:
        if (d / "phpparser.phar").is_file():
            return True, f"Found phar at {d / 'phpparser.phar'}"
    return False, (
        f"nikic/php-parser not found in {agent_vendor}.\n"
        f"Fix: cd {agent_root} && composer require nikic/php-parser"
    )


def _find_parser_script(project_path: Path) -> Optional[str]:
    env_path = os.environ.get(PARSER_ENV)
    if env_path and Path(env_path).is_file():
        return env_path
    for d in [Path(__file__).parent,
              Path(__file__).parent.parent / "parsers",
              Path(__file__).parent.parent,
              project_path,
              Path.cwd() / "parsers",
              Path.cwd()]:
        c = d / "parse_project.php"
        if c.is_file():
            return str(c)
    return None


def _detect_framework_hint(project_path: Path, primary_lang: str) -> Framework:
    # ── Next.js ───────────────────────────────────────────────────────────────
    if (project_path / "next.config.js").exists() or \
       (project_path / "next.config.ts").exists() or \
       (project_path / "next.config.mjs").exists():
        return Framework.NEXTJS

    # ── React (without Next.js) ───────────────────────────────────────────────
    pkg = project_path / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            if "next" in deps:
                return Framework.NEXTJS
            if "react" in deps and "@angular/core" not in deps:
                return Framework.REACT
            if "@angular/core" in deps:
                return Framework.ANGULAR
            if "vue" in deps:
                return Framework.VUE
            if "express" in deps or "fastify" in deps or "koa" in deps:
                return Framework.EXPRESS
        except Exception:
            pass

    # ── Spring Boot (Java) ────────────────────────────────────────────────────
    pom = project_path / "pom.xml"
    if pom.exists():
        try:
            content = pom.read_text(encoding="utf-8", errors="ignore")
            if "spring-boot" in content:
                return Framework.SPRING_BOOT
        except Exception:
            pass
    if (project_path / "src" / "main" / "java").is_dir():
        return Framework.SPRING_BOOT

    # ── PHP frameworks ────────────────────────────────────────────────────────
    if (project_path / "artisan").exists() or \
       (project_path / "app" / "Http" / "Controllers").is_dir():
        return Framework.LARAVEL
    if (project_path / "bin" / "console").exists() or \
       (project_path / "config" / "bundles.php").exists():
        return Framework.SYMFONY
    if (project_path / "wp-config.php").exists() or \
       (project_path / "wp-includes").is_dir():
        return Framework.WORDPRESS
    if (project_path / "application" / "config" / "config.php").exists():
        return Framework.CODEIGNITER

    composer = project_path / "composer.json"
    if composer.exists():
        try:
            data = json.loads(composer.read_text(encoding="utf-8"))
            req = {**data.get("require", {}), **data.get("require-dev", {})}
            if "laravel/framework" in req:
                return Framework.LARAVEL
            if "symfony/framework-bundle" in req:
                return Framework.SYMFONY
            if any("codeigniter" in k.lower() for k in req):
                return Framework.CODEIGNITER
        except Exception:
            pass
        return Framework.RAW_PHP

    return Framework.RAW_PHP
