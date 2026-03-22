"""
pipeline/stage22_components.py — Frontend Component Graph (Stage 2.2)

Runs after Stage 2.0 (knowledge graph) for TypeScript/JavaScript projects.
Skipped automatically for PHP and Java projects.

Extracts the frontend component hierarchy and writes component_graph.json:

  • Vue SFC (.vue files)   — script, template, props, emits, components
  • React components (.tsx/.jsx) — props interface, hooks used, child imports
  • Next.js pages / App Router  — page components tagged with route path

Output
------
  component_graph.json  — array of component descriptors:
    {
      "name":       "UserProfile",
      "file":       "src/components/UserProfile.vue",
      "type":       "vue_sfc" | "react_fc" | "nextjs_page",
      "props":      ["userId", "showAvatar"],
      "emits":      ["update:user"],
      "children":   ["Avatar", "EditForm"],
      "route":      "/profile/:id",    // Next.js only
      "hooks":      ["useState", "useEffect"],
      "is_page":    false
    }

Also populates ctx.code_map.components for downstream stages.

Resume behaviour
----------------
If stage22_components is COMPLETED and component_graph.json exists, the
stage is skipped and ctx.code_map.components is restored from disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from context import Framework, Language, PipelineContext

OUTPUT_FILE = "component_graph.json"

_SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", ".next", ".nuxt",
    "coverage", "tests", "test", "spec", "__tests__",
}


# ─── Public Entry Point ───────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Stage 2.2 entry point."""
    output_path = ctx.output_path(OUTPUT_FILE)

    # ── Skip for non-frontend languages ──────────────────────────────────────
    if ctx.code_map is None:
        print("  [stage22] Skipped — code_map not available")
        ctx.stage("stage22_components").mark_skipped()
        ctx.save()
        return

    language = ctx.code_map.language
    if language not in (Language.TYPESCRIPT, Language.JAVASCRIPT):
        print(f"  [stage22] Skipped — not a frontend project (language={language.value})")
        ctx.stage("stage22_components").mark_skipped()
        ctx.save()
        return

    framework = ctx.code_map.framework
    if framework not in (Framework.VUE, Framework.NUXTJS, Framework.NEXTJS,
                         Framework.REACT, Framework.NESTJS, Framework.UNKNOWN):
        # Express / Fastify are pure backend — no component graph needed
        if framework in (Framework.EXPRESS, Framework.FASTIFY):
            print(f"  [stage22] Skipped — backend-only framework ({framework.value})")
            ctx.stage("stage22_components").mark_skipped()
            ctx.save()
            return

    # ── Resume check ──────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage22_components") and Path(output_path).exists():
        components = json.loads(Path(output_path).read_text(encoding="utf-8"))
        ctx.code_map.components = components
        print(f"  [stage22] Resuming — {len(components)} component(s) loaded")
        return

    root = Path(ctx.project_path)
    print(f"  [stage22] Extracting component graph (framework={framework.value}) …")

    components: list[dict[str, Any]] = []

    if framework in (Framework.VUE, Framework.NUXTJS):
        components += _extract_vue_components(root)
    if framework in (Framework.REACT, Framework.NEXTJS):
        components += _extract_react_components(root)
    if framework == Framework.NEXTJS:
        _tag_nextjs_routes(components, root)
    # For unknown/nestjs, try both Vue and React
    if framework in (Framework.UNKNOWN, Framework.NESTJS):
        components += _extract_vue_components(root)
        components += _extract_react_components(root)

    # Deduplicate by file
    seen: set[str] = set()
    deduped = []
    for c in components:
        if c["file"] not in seen:
            seen.add(c["file"])
            deduped.append(c)
    components = deduped

    # Persist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(components, fh, indent=2, ensure_ascii=False)

    ctx.code_map.components = components
    ctx.stage("stage22_components").mark_completed(output_path)
    ctx.save()

    print(f"  [stage22] Done — {len(components)} component(s) extracted")
    _print_summary(components)


# ─── Vue SFC Extractor ────────────────────────────────────────────────────────

def _extract_vue_components(root: Path) -> list[dict]:
    components = []
    for f in _walk(root, {".vue"}):
        src = _safe_read(f)
        if not src:
            continue
        rel = _rel(f, root)
        comp = {
            "name":     _component_name_from_file(f),
            "file":     rel,
            "type":     "vue_sfc",
            "props":    _extract_vue_props(src),
            "emits":    _extract_vue_emits(src),
            "children": _extract_vue_children(src),
            "hooks":    [],
            "is_page":  _is_vue_page(f, root),
            "route":    "",
        }
        components.append(comp)
    return components


def _extract_vue_props(src: str) -> list[str]:
    props: list[str] = []
    # defineProps<{ name: string; age: number }>()
    m = re.search(r"defineProps\s*[<(]\s*\{([^}]+)\}", src)
    if m:
        for pm in re.finditer(r"(\w+)\s*[?:]", m.group(1)):
            props.append(pm.group(1))
    # props: ['name', 'age']
    m = re.search(r"props\s*:\s*\[([^\]]+)\]", src)
    if m:
        for pm in re.finditer(r"['\"](\w+)['\"]", m.group(1)):
            props.append(pm.group(1))
    # props: { name: { type: String } }
    m = re.search(r"props\s*:\s*\{([^}]+)\}", src)
    if m:
        for pm in re.finditer(r"(\w+)\s*:", m.group(1)):
            props.append(pm.group(1))
    return list(dict.fromkeys(props))[:15]


def _extract_vue_emits(src: str) -> list[str]:
    emits: list[str] = []
    m = re.search(r"defineEmits\s*\(\s*\[([^\]]+)\]", src)
    if m:
        for em in re.finditer(r"['\"]([^'\"]+)['\"]", m.group(1)):
            emits.append(em.group(1))
    m = re.search(r"emits\s*:\s*\[([^\]]+)\]", src)
    if m:
        for em in re.finditer(r"['\"]([^'\"]+)['\"]", m.group(1)):
            emits.append(em.group(1))
    return emits[:10]


def _extract_vue_children(src: str) -> list[str]:
    """Component tags used in template."""
    children: list[str] = []
    template_m = re.search(r"<template>(.*?)</template>", src, re.DOTALL)
    if template_m:
        for m in re.finditer(r"<([A-Z][A-Za-z0-9]+)", template_m.group(1)):
            children.append(m.group(1))
    return list(dict.fromkeys(children))[:20]


def _is_vue_page(f: Path, root: Path) -> bool:
    parts = list(f.relative_to(root).parts)
    return any(p in ("pages", "views") for p in parts[:-1])


# ─── React Component Extractor ────────────────────────────────────────────────

def _extract_react_components(root: Path) -> list[dict]:
    components = []
    for f in _walk(root, {".tsx", ".jsx"}):
        src = _safe_read(f)
        if not src:
            continue
        rel = _rel(f, root)
        # Must export a React component (function returning JSX)
        if not re.search(r"export\s+(?:default\s+)?(?:function|const)\s+\w+", src):
            continue
        if "<" not in src:
            continue
        comp = {
            "name":     _component_name_from_file(f),
            "file":     rel,
            "type":     "react_fc",
            "props":    _extract_react_props(src),
            "emits":    [],
            "children": _extract_react_children(src),
            "hooks":    _extract_react_hooks(src),
            "is_page":  _is_react_page(f, root),
            "route":    "",
        }
        components.append(comp)
    return components


def _extract_react_props(src: str) -> list[str]:
    props: list[str] = []
    # interface FooProps { name: string; age?: number }
    for m in re.finditer(r"interface\s+\w+Props\s*\{([^}]+)\}", src):
        for pm in re.finditer(r"(\w+)\s*[?:]", m.group(1)):
            props.append(pm.group(1))
    # type FooProps = { name: string }
    for m in re.finditer(r"type\s+\w+Props\s*=\s*\{([^}]+)\}", src):
        for pm in re.finditer(r"(\w+)\s*[?:]", m.group(1)):
            props.append(pm.group(1))
    # Destructured params: ({ name, age }: FooProps)
    for m in re.finditer(r"function\s+\w+\s*\(\s*\{([^}]+)\}", src):
        for pm in re.finditer(r"(\w+)(?:\s*[,}]|\s*:)", m.group(1)):
            props.append(pm.group(1))
    return list(dict.fromkeys(props))[:15]


def _extract_react_children(src: str) -> list[str]:
    """JSX component tags used (PascalCase only)."""
    children = []
    for m in re.finditer(r"<([A-Z][A-Za-z0-9]+)[\s/>]", src):
        children.append(m.group(1))
    return list(dict.fromkeys(children))[:20]


def _extract_react_hooks(src: str) -> list[str]:
    hooks = []
    for m in re.finditer(r"\b(use[A-Z]\w+)\s*\(", src):
        hooks.append(m.group(1))
    return list(dict.fromkeys(hooks))[:10]


def _is_react_page(f: Path, root: Path) -> bool:
    parts = list(f.relative_to(root).parts)
    return any(p in ("pages", "app") for p in parts[:-1])


# ─── Next.js route tagging ────────────────────────────────────────────────────

def _tag_nextjs_routes(components: list[dict], root: Path) -> None:
    for comp in components:
        if not comp.get("is_page"):
            continue
        f = root / comp["file"]
        # pages/foo/bar.tsx → /foo/bar
        try:
            parts = list(f.relative_to(root).parts)
        except ValueError:
            continue
        if "pages" in parts:
            idx = parts.index("pages")
            route_parts = parts[idx+1:]
            route = "/" + "/".join(route_parts)
            route = re.sub(r"\.(tsx?|jsx?)$", "", route)
            route = re.sub(r"/index$", "", route) or "/"
            route = re.sub(r"\[(\w+)\]", r":\1", route)
            comp["route"] = route
        elif "app" in parts:
            idx = parts.index("app")
            route_parts = [p for p in parts[idx+1:] if not p.startswith("(")]
            if route_parts and route_parts[-1] in ("page.tsx", "page.ts", "page.js"):
                route_parts = route_parts[:-1]
            route = "/" + "/".join(route_parts)
            route = re.sub(r"\[(\w+)\]", r":\1", route)
            comp["route"] = route or "/"


# ─── Utilities ────────────────────────────────────────────────────────────────

def _component_name_from_file(f: Path) -> str:
    stem = f.stem
    # PascalCase from kebab-case
    return "".join(w.capitalize() for w in re.split(r"[-_.]", stem))


def _walk(root: Path, extensions: set[str]):
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if any(p in _SKIP_DIRS for p in f.parts):
            continue
        if f.suffix in extensions:
            yield f


def _safe_read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _print_summary(components: list[dict]) -> None:
    by_type: dict[str, int] = {}
    pages = 0
    for c in components:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
        if c.get("is_page"):
            pages += 1
    for kind, count in sorted(by_type.items()):
        print(f"  [stage22]   {kind}: {count}")
    if pages:
        print(f"  [stage22]   page components: {pages}")
