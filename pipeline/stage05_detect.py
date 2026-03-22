"""
pipeline/stage05_detect.py — Language & Framework Detection (Stage 0.5)

Runs immediately after Stage 0 (validation) and before Stage 1 (parse).
Probes the project root to determine:

  • Primary programming language  → ctx.code_map.language (Language enum)
  • Detected framework            → ctx.detected_framework (Framework enum)

Detection order
---------------
  1. Presence of manifest / config files
       composer.json / *.php   → PHP
       package.json + tsconfig.json / *.ts files → TypeScript
       package.json (no tsconfig, no .ts)         → JavaScript
       pom.xml / build.gradle + *.java            → Java
       build.gradle.kts / *.kt                    → Kotlin

  2. File extension census  (top N source files by extension)
       If manifest clues are ambiguous, count file extensions and pick
       the dominant one.

  3. Framework sub-detection  (runs after language is confirmed)
       PHP  → laravel / symfony / codeigniter / wordpress / raw_php
       TS   → nextjs / nuxtjs / vue / nestjs / express / fastify / react
       Java → spring_boot / quarkus / micronaut

Outputs
-------
  • Writes ``language_detect.json`` to the 0.5_detect subdirectory
  • Sets ``ctx.detected_language``   (Language enum)
  • Sets ``ctx.detected_framework``  (Framework enum)
  • Both are read by stage10_parse to pick the correct language adapter.

Resume behaviour
----------------
If stage05_detect is COMPLETED and language_detect.json exists the stage
is skipped and the stored values are restored onto ctx.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from context import Framework, Language, PipelineContext

OUTPUT_FILE = "language_detect.json"

# ─── File-based detection signals ─────────────────────────────────────────────

_PHP_SIGNALS = {
    "composer.json",
    "artisan",          # Laravel
    "bin/console",      # Symfony
    "wp-config.php",    # WordPress
    "system/core/CodeIgniter.php",  # CodeIgniter 3
}

_TS_SIGNALS = {
    "tsconfig.json",
    "tsconfig.base.json",
    "tsconfig.app.json",
}

_JAVA_SIGNALS = {
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "gradlew",
}

# Framework-within-language signals (file presence → Framework enum)
_PHP_FRAMEWORK_SIGNALS: list[tuple[str, Framework]] = [
    ("artisan",                              Framework.LARAVEL),
    ("composer.json:laravel/framework",      Framework.LARAVEL),     # content check
    ("bin/console",                          Framework.SYMFONY),
    ("composer.json:symfony/framework-bundle", Framework.SYMFONY),   # content check
    ("wp-config.php",                        Framework.WORDPRESS),
    ("wp-config-sample.php",                 Framework.WORDPRESS),
    ("system/core/CodeIgniter.php",          Framework.CODEIGNITER),
    ("application/config/config.php",        Framework.CODEIGNITER),
]

_TS_FRAMEWORK_SIGNALS: list[tuple[str, Framework]] = [
    ("next.config.js",    Framework.NEXTJS),
    ("next.config.mjs",   Framework.NEXTJS),
    ("next.config.ts",    Framework.NEXTJS),
    ("nuxt.config.ts",    Framework.NUXTJS),
    ("nuxt.config.js",    Framework.NUXTJS),
    ("vite.config.ts",    Framework.VUE),    # Vite is often used with Vue (also React/Svelte)
    ("vue.config.js",     Framework.VUE),
    ("package.json:nuxt", Framework.NUXTJS),   # content check
    ("package.json:next",  Framework.NEXTJS),   # content check
    ("package.json:@nestjs/core", Framework.NESTJS),
    ("package.json:fastify",      Framework.FASTIFY),
    ("package.json:express",      Framework.EXPRESS),
    ("package.json:vue",          Framework.VUE),
    ("package.json:react",        Framework.REACT),
]

_JAVA_FRAMEWORK_SIGNALS: list[tuple[str, Framework]] = [
    ("pom.xml:spring-boot",         Framework.SPRING_BOOT),
    ("build.gradle:spring-boot",    Framework.SPRING_BOOT),
    ("pom.xml:quarkus",             Framework.QUARKUS),
    ("build.gradle:quarkus",        Framework.QUARKUS),
    ("pom.xml:micronaut",           Framework.MICRONAUT),
    ("build.gradle:micronaut",      Framework.MICRONAUT),
]


# ─── Public Entry Point ───────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 0.5 entry point.  Detects language + framework and stores results
    on ctx as ``detected_language`` and ``detected_framework``.

    Args:
        ctx: Shared pipeline context; mutated in-place.
    """
    output_path = ctx.output_path(OUTPUT_FILE)

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage05_detect") and Path(output_path).exists():
        _restore_from_file(ctx, output_path)
        return

    root = Path(ctx.project_path)
    print(f"  [stage05] Detecting language & framework for: {root.name}")

    language  = _detect_language(root)
    framework = _detect_framework(root, language)

    print(f"  [stage05] Language  → {language.value}")
    print(f"  [stage05] Framework → {framework.value}")

    # Store on context (transient attributes; also written to JSON)
    ctx.detected_language  = language
    ctx.detected_framework = framework

    result = {
        "language":  language.value,
        "framework": framework.value,
        "project":   root.name,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    ctx.stage("stage05_detect").mark_completed(output_path)
    ctx.save()


# ─── Language Detection ───────────────────────────────────────────────────────

def _detect_language(root: Path) -> Language:
    """
    Probe the project root and return the most likely primary language.
    Uses a signal-scoring approach so mixed repos score correctly.
    """
    scores: dict[Language, int] = {
        Language.PHP:        0,
        Language.TYPESCRIPT: 0,
        Language.JAVASCRIPT: 0,
        Language.JAVA:       0,
        Language.KOTLIN:     0,
    }

    # ── Manifest / config file signals ───────────────────────────────────────
    for sig in _PHP_SIGNALS:
        if (root / sig).exists():
            scores[Language.PHP] += 3

    for sig in _TS_SIGNALS:
        if (root / sig).exists():
            scores[Language.TYPESCRIPT] += 3

    for sig in _JAVA_SIGNALS:
        if (root / sig).exists():
            scores[Language.JAVA] += 3

    # package.json presence: JS baseline, TS takes over if tsconfig found
    if (root / "package.json").exists():
        scores[Language.JAVASCRIPT] += 2
        # If tsconfig already scored, TS wins over JS
        if scores[Language.TYPESCRIPT] > 0:
            scores[Language.JAVASCRIPT] = 0

    # ── File extension census (sample top 500 files) ──────────────────────────
    ext_count: dict[str, int] = {}
    for f in list(root.rglob("*"))[:2000]:
        if not f.is_file():
            continue
        parts = set(f.parts)
        if parts & {"vendor", "node_modules", ".git", "dist", "build", "target"}:
            continue
        ext_count[f.suffix] = ext_count.get(f.suffix, 0) + 1

    scores[Language.PHP]        += min(ext_count.get(".php", 0), 20)
    scores[Language.TYPESCRIPT] += min(ext_count.get(".ts", 0) + ext_count.get(".tsx", 0), 20)
    scores[Language.JAVASCRIPT] += min(ext_count.get(".js", 0) + ext_count.get(".jsx", 0), 10)
    scores[Language.JAVA]       += min(ext_count.get(".java", 0), 20)
    scores[Language.KOTLIN]     += min(ext_count.get(".kt", 0), 20)

    # Vue SFCs → TypeScript (or JS) project
    vue_count = ext_count.get(".vue", 0)
    if vue_count > 0:
        if scores[Language.TYPESCRIPT] >= scores[Language.JAVASCRIPT]:
            scores[Language.TYPESCRIPT] += min(vue_count, 10)
        else:
            scores[Language.JAVASCRIPT] += min(vue_count, 10)

    # ── Pick the highest scorer ────────────────────────────────────────────────
    if not any(scores.values()):
        return Language.UNKNOWN

    best = max(scores, key=lambda l: scores[l])
    # Tie between Java and Kotlin → prefer Java (broader ecosystem)
    if best == Language.KOTLIN and scores[Language.JAVA] == scores[Language.KOTLIN]:
        best = Language.JAVA

    return best


# ─── Framework Detection ──────────────────────────────────────────────────────

def _detect_framework(root: Path, language: Language) -> Framework:
    """
    Given the detected language, probe for the specific framework.
    Returns Framework.UNKNOWN if no clear match is found.
    """
    if language == Language.PHP:
        signals = _PHP_FRAMEWORK_SIGNALS
    elif language in (Language.TYPESCRIPT, Language.JAVASCRIPT):
        signals = _TS_FRAMEWORK_SIGNALS
    elif language in (Language.JAVA, Language.KOTLIN):
        signals = _JAVA_FRAMEWORK_SIGNALS
    else:
        return Framework.UNKNOWN

    # Read candidate files once to avoid redundant I/O
    _file_cache: dict[str, str] = {}

    def _read(rel: str) -> str:
        if rel not in _file_cache:
            p = root / rel
            _file_cache[rel] = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
        return _file_cache[rel]

    for sig, fw in signals:
        if ":" in sig:
            # Content check: "filename:substring"
            filename, substring = sig.split(":", 1)
            if substring.lower() in _read(filename).lower():
                return fw
        else:
            # File-existence check
            if (root / sig).exists():
                return fw

    # Language-specific fallbacks
    if language == Language.PHP:
        return Framework.RAW_PHP
    return Framework.UNKNOWN


# ─── Restore from saved file ──────────────────────────────────────────────────

def _restore_from_file(ctx: PipelineContext, output_path: str) -> None:
    try:
        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)
        lang_val = data.get("language", Language.UNKNOWN.value)
        fw_val   = data.get("framework", Framework.UNKNOWN.value)
        try:
            ctx.detected_language  = Language(lang_val)
        except ValueError:
            ctx.detected_language  = Language.UNKNOWN
        try:
            ctx.detected_framework = Framework(fw_val)
        except ValueError:
            ctx.detected_framework = Framework.UNKNOWN
        print(
            f"  [stage05] Resuming — language={ctx.detected_language.value}, "
            f"framework={ctx.detected_framework.value}"
        )
    except Exception as exc:
        print(f"  [stage05] Warning: could not restore from {output_path}: {exc}")
        ctx.detected_language  = Language.UNKNOWN
        ctx.detected_framework = Framework.UNKNOWN


# ─── Convenience accessor (used by stage10/13/15 dispatchers) ─────────────────

def get_detected(ctx: PipelineContext) -> tuple[Language, Framework]:
    """
    Return the (Language, Framework) pair stored on ctx by stage05_detect.
    Falls back to inferring from ctx.code_map if stage05 was not run
    (e.g. old runs loaded from context.json without stage05 output).
    """
    lang = getattr(ctx, "detected_language", None)
    fw   = getattr(ctx, "detected_framework", None)
    if lang is None or fw is None:
        # Infer from code_map if available (backward compat)
        if ctx.code_map is not None:
            lang = ctx.code_map.language
            fw   = ctx.code_map.framework
        else:
            lang = Language.PHP       # historic default
            fw   = Framework.UNKNOWN
    return lang, fw
