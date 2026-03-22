"""
pipeline/paths/typescript_paths.py — TypeScript / JavaScript Execution Path Analyser

Static analysis of TypeScript / JavaScript source files to extract:

  1. Entry conditions  — which request parameters / headers must exist
  2. Branch map        — if/else/switch branches with their outcomes
  3. Data flow         — which req.body fields feed which downstream calls
  4. Auth guard        — JWT / session / passport guard patterns per handler
  5. Happy path        — ordered step list for the main success scenario
  6. Async chain       — Promise / async-await chain depth (approximate)

For Next.js and NestJS, controller method flows are also extracted.

Output is written into ctx.code_map.execution_paths (list of dicts),
mirroring the schema produced by stage15_paths.py for PHP.
"""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any

from context import CodeMap, Framework, PipelineContext

OUTPUT_FILE  = "execution_paths.json"
ERROR_FILE   = "execution_paths_errors.json"

_SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", ".next", ".nuxt",
    "coverage", "tests", "test", "spec", "__tests__",
}
_TS_EXTS = {".ts", ".tsx", ".js", ".jsx", ".mjs"}


# ─── Public interface ─────────────────────────────────────────────────────────

def enrich(ctx: PipelineContext) -> list[dict]:
    """
    Analyse TypeScript / JS source files for execution paths.
    Writes execution_paths.json, mutates ctx.code_map.execution_paths.

    Returns the list of execution path dicts.
    """
    output_path = ctx.output_path(OUTPUT_FILE)
    error_path  = ctx.output_path(ERROR_FILE)
    root = Path(ctx.project_path)
    code_map = ctx.code_map

    if code_map is None:
        raise RuntimeError("[stage15/ts] ctx.code_map is None — run Stage 1 first.")

    framework = code_map.framework
    all_paths: list[dict] = []
    errors:    list[dict] = []

    # Build a set of handler files from routes for targeted analysis
    handler_files: set[str] = {r.get("file", "") for r in code_map.routes}

    source_files = _collect_files(root)
    print(f"  [stage15/ts] Analysing {len(source_files)} source files …")

    for src_file in source_files:
        rel = _rel(src_file, root)
        try:
            src = src_file.read_text(encoding="utf-8", errors="ignore")
            paths = _analyse_file(src, rel, framework, is_handler=(rel in handler_files))
            all_paths.extend(paths)
        except Exception as exc:
            errors.append({"file": rel, "error": str(exc), "trace": traceback.format_exc()[-500:]})

    # Also extract controller flows from NestJS controllers
    if framework == Framework.NESTJS:
        for cls in (code_map.controllers or []):
            f = cls.get("file", "")
            src_file = root / f
            if src_file.exists():
                try:
                    src = src_file.read_text(encoding="utf-8", errors="ignore")
                    all_paths.extend(_extract_nestjs_flows(src, f))
                except Exception as exc:
                    errors.append({"file": f, "error": str(exc)})

    print(f"  [stage15/ts] Extracted {len(all_paths)} execution paths, {len(errors)} error(s)")

    # Persist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(all_paths, fh, indent=2, ensure_ascii=False)
    with open(error_path, "w", encoding="utf-8") as fh:
        json.dump(errors, fh, indent=2, ensure_ascii=False)

    if code_map:
        code_map.execution_paths = all_paths

    return all_paths


# ─── File collection ──────────────────────────────────────────────────────────

def _collect_files(root: Path) -> list[Path]:
    result = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if any(p in _SKIP_DIRS for p in f.parts):
            continue
        if f.suffix in _TS_EXTS:
            result.append(f)
    return result


# ─── Per-file analysis ────────────────────────────────────────────────────────

def _analyse_file(src: str, rel: str, framework: Framework, is_handler: bool = False) -> list[dict]:
    """
    Analyse one TypeScript / JS source file.
    Returns a list of execution path dicts (one per handler function found).
    """
    results = []

    # Find handler functions: Express (req, res) / NestJS decorated methods
    handlers = _find_handlers(src, rel)
    for handler in handlers:
        path_dict = {
            "file":             rel,
            "handler":          handler["name"],
            "http_method":      handler.get("http_method", ""),
            "route":            handler.get("route", ""),
            "entry_conditions": _extract_entry_conditions(handler["body"]),
            "branch_map":       _extract_branches(handler["body"]),
            "data_flow":        _extract_data_flow(handler["body"]),
            "auth_guard":       _extract_auth_guard(handler["body"]),
            "happy_path":       _build_happy_path(handler["body"], handler["name"]),
            "async_chain":      _detect_async_depth(handler["body"]),
        }
        results.append(path_dict)

    return results


def _find_handlers(src: str, rel: str) -> list[dict]:
    """Find all route handler functions in the file."""
    handlers = []

    # Express: app.get('/path', async (req, res) => { ... }) or named function
    for m in re.finditer(
        r"""(?:app|router)\.(get|post|put|patch|delete)\s*\(\s*['"`]([^'"`]+)['"`]\s*,\s*"""
        r"""(?:async\s+)?(?:function\s+(\w+)\s*)?\(""",
        src, re.IGNORECASE
    ):
        method, route, fn_name = m.group(1).upper(), m.group(2), m.group(3) or ""
        body = _extract_function_body(src, m.end() - 1)
        handlers.append({"name": fn_name or f"{method}_{route}", "http_method": method, "route": route, "body": body})

    # NestJS: @Get/@Post decorated method
    for m in re.finditer(
        r"@(Get|Post|Put|Patch|Delete)\s*\([^)]*\)\s*"
        r"(?:async\s+)?(\w+)\s*\(",
        src
    ):
        method, fn_name = m.group(1).upper(), m.group(2)
        body = _extract_function_body(src, src.index("(", m.end()) if "(" in src[m.end():m.end()+50] else m.end())
        handlers.append({"name": fn_name, "http_method": method, "route": "", "body": body})

    # Exported async functions (Next.js page handlers, generic)
    for m in re.finditer(r"export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(", src):
        fn_name = m.group(1)
        if fn_name.upper() in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            body = _extract_function_body(src, m.end() - 1)
            handlers.append({"name": fn_name, "http_method": fn_name.upper(), "route": "", "body": body})

    return handlers


def _extract_function_body(src: str, pos: int) -> str:
    """Extract up to 2000 chars of function body after opening brace."""
    start = src.find("{", pos)
    if start == -1:
        return ""
    depth = 0
    for i in range(start, min(start + 3000, len(src))):
        if src[i] == "{": depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i+1]
    return src[start:start+2000]


def _extract_entry_conditions(body: str) -> list[str]:
    """Extract guard conditions near the start of the function."""
    conditions = []
    # if (!req.body.field) return res.status(400)
    for m in re.finditer(r"if\s*\(!?req\.(body|query|params)\.(\w+)\)", body[:500]):
        conditions.append(f"req.{m.group(1)}.{m.group(2)} required")
    # Missing field checks
    for m in re.finditer(r"if\s*\(.*?===?\s*undefined|if\s*\(!.*?\)", body[:500]):
        conditions.append(m.group(0)[:80])
    return conditions[:5]


def _extract_branches(body: str) -> list[dict]:
    """Extract if/else/switch branches."""
    branches = []
    for m in re.finditer(r"if\s*\(([^)]{0,100})\)\s*\{", body):
        condition = m.group(1).strip()
        # Get a snippet of what happens in the branch
        branch_body = _extract_function_body(body, m.end() - 1)
        outcome = _summarise_outcome(branch_body[:200])
        branches.append({"condition": condition[:80], "outcome": outcome})
    return branches[:10]


def _extract_data_flow(body: str) -> list[dict]:
    """Map req.body / req.query fields to what they feed."""
    flows = []
    for m in re.finditer(r"(?:const|let|var)\s+(\w+)\s*=\s*req\.(body|query|params)\.(\w+)", body):
        var_name, source, field = m.group(1), m.group(2), m.group(3)
        flows.append({"input": f"req.{source}.{field}", "local_var": var_name})
    # Destructured: const { email } = req.body
    for m in re.finditer(r"const\s*\{([^}]+)\}\s*=\s*req\.(body|query|params)", body):
        for field in re.findall(r"\b(\w+)\b", m.group(1)):
            flows.append({"input": f"req.{m.group(2)}.{field}", "local_var": field})
    return flows[:10]


def _extract_auth_guard(body: str) -> dict:
    """Detect auth guard pattern at top of handler."""
    guard: dict[str, Any] = {"present": False}
    if re.search(r"passport\.authenticate|verifyToken|checkAuth|jwt\.verify", body[:400]):
        guard = {"present": True, "kind": "middleware_or_jwt"}
    if re.search(r"req\.user\b", body[:400]):
        guard = {"present": True, "kind": "req.user check"}
    if re.search(r"if\s*\(!req\.user\)", body[:400]):
        guard = {"present": True, "kind": "explicit user check", "redirect": "401/403"}
    return guard


def _build_happy_path(body: str, fn_name: str) -> list[str]:
    """Build an ordered list of happy-path steps from the handler body."""
    steps = []
    # Input extraction
    if re.search(r"req\.body|req\.query|req\.params", body):
        steps.append("Extract request inputs")
    # Validation
    if re.search(r"\.validate\(|zod|yup|class-validator|if.*!.*req\.", body):
        steps.append("Validate inputs")
    # DB ops
    if re.search(r"prisma\.|repository\.|findOne|findMany|save\(|\.create\(", body):
        steps.append("Query / mutate database")
    # External calls
    if re.search(r"axios\.|fetch\(|http\.request|https\.request", body):
        steps.append("Call external service")
    # Response
    if re.search(r"res\.json\(|res\.send\(|res\.status\(|return.*Response", body):
        steps.append("Send response")
    return steps if steps else [f"Execute {fn_name}"]


def _detect_async_depth(body: str) -> int:
    """Count the depth of await chains as a proxy for async complexity."""
    return min(len(re.findall(r"\bawait\b", body)), 10)


def _summarise_outcome(snippet: str) -> str:
    if re.search(r"res\.status\s*\(\s*4\d\d", snippet):
        return "error response (4xx)"
    if re.search(r"res\.status\s*\(\s*5\d\d", snippet):
        return "error response (5xx)"
    if re.search(r"res\.json\(|res\.send\(", snippet):
        return "success response"
    if re.search(r"throw\s+new", snippet):
        return "throws exception"
    if re.search(r"return\b", snippet):
        return "early return"
    return "branch body"


# ─── NestJS controller flow extraction ───────────────────────────────────────

def _extract_nestjs_flows(src: str, rel: str) -> list[dict]:
    flows = []
    for m in re.finditer(
        r"@(Get|Post|Put|Patch|Delete)\s*\(([^)]*)\)\s*(?:async\s+)?(\w+)\s*\(",
        src
    ):
        method, route_arg, fn_name = m.group(1).upper(), m.group(2).strip("'\" "), m.group(3)
        body = _extract_function_body(src, src.find("(", m.end()))
        flows.append({
            "file":             rel,
            "handler":          fn_name,
            "http_method":      method,
            "route":            route_arg,
            "entry_conditions": _extract_entry_conditions(body),
            "branch_map":       _extract_branches(body),
            "data_flow":        _extract_data_flow(body),
            "auth_guard":       _extract_auth_guard(body),
            "happy_path":       _build_happy_path(body, fn_name),
            "async_chain":      _detect_async_depth(body),
            "kind":             "nestjs_controller",
        })
    return flows


# ─── Utilities ────────────────────────────────────────────────────────────────

def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
