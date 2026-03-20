"""
pipeline/paths/java_paths.py — Java / Kotlin Execution Path Analyser

Static analysis of Java / Kotlin source files to extract execution paths:

  1. Entry conditions  — @NotNull / @Valid / manual null-checks at method start
  2. Branch map        — if/else/switch/try-catch with outcomes
  3. Data flow         — @RequestBody / @RequestParam → service calls
  4. Auth guard        — @PreAuthorize / SecurityContextHolder checks
  5. Happy path        — ordered step list for main success scenario
  6. Exception flows   — throws / try-catch blocks

Output mirrors the schema from stage15_paths.py so downstream stages
(stage20_graph, stage45_flows, etc.) work without modification.
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
    "target", "build", ".git", "generated", "generated-sources",
    "test", "tests", "it", "integration-test",
}
_JAVA_EXTS = {".java", ".kt"}


# ─── Public interface ─────────────────────────────────────────────────────────

def enrich(ctx: PipelineContext) -> list[dict]:
    """
    Analyse Java / Kotlin source files for execution paths.
    Writes execution_paths.json, mutates ctx.code_map.execution_paths.
    """
    output_path = ctx.output_path(OUTPUT_FILE)
    error_path  = ctx.output_path(ERROR_FILE)
    root = Path(ctx.project_path)
    code_map = ctx.code_map

    if code_map is None:
        raise RuntimeError("[stage15/java] ctx.code_map is None — run Stage 1 first.")

    framework = code_map.framework
    all_paths: list[dict] = []
    errors:    list[dict] = []

    source_files = _collect_files(root)
    print(f"  [stage15/java] Analysing {len(source_files)} source files …")

    for src_file in source_files:
        rel = _rel(src_file, root)
        try:
            src = src_file.read_text(encoding="utf-8", errors="ignore")
            paths = _analyse_file(src, rel, framework)
            all_paths.extend(paths)
        except Exception as exc:
            errors.append({"file": rel, "error": str(exc), "trace": traceback.format_exc()[-500:]})

    print(f"  [stage15/java] Extracted {len(all_paths)} execution paths, {len(errors)} error(s)")

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
        if f.suffix in _JAVA_EXTS:
            result.append(f)
    return result


# ─── Per-file analysis ────────────────────────────────────────────────────────

def _analyse_file(src: str, rel: str, framework: Framework) -> list[dict]:
    """Analyse one Java/Kotlin source file for controller method flows."""
    results = []

    # Find Spring controller methods decorated with @GetMapping/@PostMapping etc.
    for m in re.finditer(
        r"@(Get|Post|Put|Patch|Delete|Request)Mapping\s*(?:\([^)]*\))?\s*"
        r"(?:@[^\n]+\n\s*)*"        # any other annotations
        r"(?:public|protected)?\s+\S+\s+(\w+)\s*\(",
        src, re.DOTALL
    ):
        verb = m.group(1)
        method = verb.upper() if verb != "Request" else "ANY"
        fn_name = m.group(2)
        body = _extract_method_body(src, m.end() - 1)

        results.append({
            "file":             rel,
            "handler":          fn_name,
            "http_method":      method,
            "route":            "",
            "entry_conditions": _extract_entry_conditions(body),
            "branch_map":       _extract_branches(body),
            "data_flow":        _extract_data_flow(body),
            "auth_guard":       _extract_auth_guard(body),
            "happy_path":       _build_happy_path(body, fn_name),
            "exception_flows":  _extract_exception_flows(body),
        })

    # CommandLineRunner / ApplicationRunner run() methods
    if "implements CommandLineRunner" in src or "implements ApplicationRunner" in src:
        for m in re.finditer(r"(?:public\s+)?void\s+run\s*\(", src):
            body = _extract_method_body(src, m.end() - 1)
            results.append({
                "file":         rel,
                "handler":      "run",
                "http_method":  "CLI",
                "route":        "",
                "entry_conditions": [],
                "branch_map":   _extract_branches(body),
                "data_flow":    [],
                "auth_guard":   {},
                "happy_path":   _build_happy_path(body, "run"),
                "exception_flows": _extract_exception_flows(body),
            })

    return results


def _extract_method_body(src: str, pos: int) -> str:
    """Extract up to 2500 chars of method body after opening brace."""
    start = src.find("{", pos)
    if start == -1:
        return ""
    depth = 0
    for i in range(start, min(start + 4000, len(src))):
        if src[i] == "{": depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i+1]
    return src[start:start+2500]


def _extract_entry_conditions(body: str) -> list[str]:
    conditions = []
    # if (param == null) throw / return
    for m in re.finditer(r"if\s*\(\s*(\w+)\s*==\s*null\)", body[:600]):
        conditions.append(f"{m.group(1)} must not be null")
    # Objects.requireNonNull
    for m in re.finditer(r"Objects\.requireNonNull\s*\(\s*(\w+)", body[:600]):
        conditions.append(f"{m.group(1)} required (requireNonNull)")
    # @Valid / @NotNull annotations on params (already in signature, note them)
    if "@Valid" in body[:200] or "@NotNull" in body[:200]:
        conditions.append("Input validated via @Valid/@NotNull")
    return conditions[:5]


def _extract_branches(body: str) -> list[dict]:
    branches = []
    for m in re.finditer(r"if\s*\(([^)]{0,100})\)\s*\{", body):
        condition = m.group(1).strip()
        branch_body = _extract_method_body(body, m.end() - 1)
        outcome = _summarise_outcome(branch_body[:200])
        branches.append({"condition": condition[:80], "outcome": outcome})
    # try-catch
    for m in re.finditer(r"catch\s*\(\s*([\w.]+)\s+(\w+)\s*\)", body):
        branches.append({
            "condition": f"catch {m.group(1)}",
            "outcome": "exception handler",
        })
    return branches[:10]


def _extract_data_flow(body: str) -> list[dict]:
    flows = []
    # Service calls with variables: service.save(entity) / repo.findById(id)
    for m in re.finditer(r"(\w+(?:Service|Repository|Repo|Dao))\.(\w+)\s*\(([^)]{0,80})\)", body):
        flows.append({
            "service": m.group(1),
            "method":  m.group(2),
            "args":    m.group(3).strip()[:60],
        })
    return flows[:10]


def _extract_auth_guard(body: str) -> dict:
    guard: dict[str, Any] = {"present": False}
    if re.search(r"SecurityContextHolder|getAuthentication\(\)", body[:500]):
        guard = {"present": True, "kind": "SecurityContextHolder"}
    if re.search(r"@PreAuthorize|@Secured", body[:200]):
        guard = {"present": True, "kind": "annotation_based"}
    if re.search(r"principal|getPrincipal\(\)", body[:400]):
        guard = {"present": True, "kind": "principal_check"}
    return guard


def _build_happy_path(body: str, fn_name: str) -> list[str]:
    steps = []
    if re.search(r"@RequestBody|@RequestParam|@PathVariable|getParameter", body[:300]):
        steps.append("Extract request inputs")
    if re.search(r"@Valid|validate\(|Validator|BindingResult", body):
        steps.append("Validate inputs")
    if re.search(r"service\.|repository\.|repo\.|\.save\(|\.findBy|\.delete\(", body):
        steps.append("Query / mutate via service/repository")
    if re.search(r"ResponseEntity|return\s+new\b|\.ok\(\)|\.created\(|\.noContent\(", body):
        steps.append("Build and return response")
    return steps if steps else [f"Execute {fn_name}"]


def _extract_exception_flows(body: str) -> list[dict]:
    flows = []
    for m in re.finditer(r"throw\s+new\s+([\w.]+)\s*\(([^)]{0,80})\)", body):
        flows.append({"type": m.group(1), "message": m.group(2).strip()[:60]})
    return flows[:5]


def _summarise_outcome(snippet: str) -> str:
    if re.search(r"throw\s+new", snippet):       return "throws exception"
    if re.search(r"\.badRequest\(\)", snippet):   return "400 Bad Request"
    if re.search(r"\.notFound\(\)", snippet):      return "404 Not Found"
    if re.search(r"\.ok\(\)", snippet):            return "200 OK response"
    if re.search(r"return\b", snippet):            return "early return"
    return "branch body"


# ─── Utilities ────────────────────────────────────────────────────────────────

def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
