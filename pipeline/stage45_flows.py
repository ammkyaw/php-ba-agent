"""
pipeline/stage45_flows.py — Business Flow Extractor

Sits between Stage 4 (DomainAnalystAgent) and Stage 5 (parallel BA workers).
Extracts concrete, multi-step business flows by combining three evidence sources:

    Source A  Graph traversal  — follows redirect/handles/calls edges from each
                                  http_endpoint node (from Stage 2) up to MAX_DEPTH
                                  hops to discover multi-file journeys as node paths.

    Source B  Happy-path stitch — for each graph path, merges the per-file
                                   happy_path + branch arrays from Stage 1.5
                                   execution_paths into a single ordered FlowStep list.

    Source C  LLM enrichment   — one Claude call per bounded context (not per flow)
                                  to assign names, actors, triggers, and termination
                                  states to the graph-derived path skeletons.

Output
------
    ctx.business_flows  — BusinessFlowCollection dataclass
    business_flows.json — persisted to the run output directory

    ctx.domain_model.workflows is also replaced with richer flow summaries so
    Stage 5 agents automatically receive better data without code changes.

Resume behaviour
----------------
If stage45_flows is COMPLETED and business_flows.json exists, the stage is
skipped and ctx.business_flows is restored from the saved file.

Architecture notes
------------------
Pass A  Find all http_endpoint + entry-point nodes in G.graph["index"]
Pass B  BFS from each entry-point following redirect/handles/calls/entry_point edges
Pass C  Deduplicate paths (same file sequence → keep longest)
Pass D  Stitch execution_paths happy_path arrays along each path
Pass E  Build FlowStep objects with db_ops, auth, inputs/outputs from execution_paths
Pass F  Group candidate flows by bounded_context from domain_model
Pass G  One Claude call per bounded context to name + describe flow skeletons
Pass H  Merge LLM names/descriptions back onto FlowStep-backed BusinessFlow objects
Pass I  Replace domain_model.workflows with flow summaries
"""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

from context import (
    BusinessFlow,
    BusinessFlowCollection,
    FlowStep,
    PipelineContext,
)

# ── Configuration ───────────────────────────────────────────────────────────────
CLAUDE_MODEL   = "claude-sonnet-4-20250514"
MAX_TOKENS     = 8192
FLOWS_FILE     = "business_flows.json"

# Graph traversal limits
MAX_DEPTH      = 8    # max hops from an entry-point node
MAX_PATHS      = 40   # cap on candidate paths before dedup
MAX_PATH_LEN   = 12   # max steps in a single flow

# BFS edge types to follow — must match edge_type values written by stage2_graph.py
# stage2 writes: "redirects_to", "submits_to", "handles", "entry_point",
#                "calls", "defines", "includes"
# NOTE: only navigation edges belong here. "includes" and "defines" are code-
# structure edges (require/include, class→method) — following them pulls in
# helper files like session.php, db.php that are not user-facing pages and
# would appear as phantom participants in flow/sequence diagrams.
FOLLOW_EDGES   = {"redirects_to", "submits_to", "handles", "entry_point",
                  "calls",
                  # Legacy alias kept for backward compat with older graph pickles
                  "redirect"}
# Edge types that represent a meaningful "user moves to next page" transition
PAGE_EDGES     = {"redirects_to", "submits_to", "entry_point", "redirect"}

# Minimum path length (in page-type nodes) to be worth reporting
MIN_PAGE_STEPS = 2


# ─── Flow Pattern Definitions ─────────────────────────────────────────────────
#
# Each FlowPattern is a named template that Stage 4.5 Pass F.5 matches against
# a flow skeleton algorithmically — no LLM call needed.  The match result is
# passed to the LLM in Pass G so it can name, describe, and set the actor /
# termination more accurately.
#
# Detection signals used (all available on skeleton + FlowStep objects):
#   files         — list of PHP filenames/stems in the flow
#   inputs        — all form field names across all steps
#   db_ops        — all SQL fragments across all steps
#   http_methods  — set of HTTP verbs used
#   auth_steps    — count of steps with auth_required=True
#   step_count    — total number of steps
#   outputs       — all session keys + redirect targets
#
# Scoring: each matched rule scores +1.  A pattern is "matched" when its score
# reaches or exceeds its threshold.  The highest-scoring pattern wins.
# Ties are broken by pattern priority (lower index in _PATTERNS wins).

from dataclasses import dataclass as _dc, field as _field
from typing import Callable as _Callable

@_dc
class FlowPattern:
    """
    Named business flow template used for algorithmic pattern matching.

    Attributes:
        name        Human-readable pattern name, e.g. "Login Flow"
        description Short phrase describing the pattern for the LLM prompt
        rules       List of (signal_name, callable) — each callable receives
                    the pre-computed signal dict and returns True/False
        threshold   Minimum number of rules that must match to classify
        actor_hint  Suggested actor override (None = keep LLM inference)
        term_hint   Suggested termination phrase template (None = keep LLM)
    """
    name:        str
    description: str
    rules:       list[tuple[str, _Callable[[dict], bool]]]
    threshold:   int
    actor_hint:  str | None = None
    term_hint:   str | None = None


# Helper predicates — keep lambdas readable
def _any_file(*kws):
    """Return True if any file stem contains any of the given keywords."""
    return lambda s: any(
        kw in stem
        for stem in s["file_stems"]
        for kw in kws
    )

def _any_input(*kws):
    """Return True if any input field name contains any keyword."""
    return lambda s: any(kw in inp.lower() for inp in s["inputs"] for kw in kws)

def _any_db(*kws):
    """Return True if any db_op fragment contains any keyword."""
    return lambda s: any(kw in op.lower() for op in s["db_ops"] for kw in kws)

def _has_method(method: str):
    return lambda s: method.upper() in s["http_methods"]

def _step_count_gte(n: int):
    return lambda s: s["step_count"] >= n

def _auth_steps_gte(n: int):
    return lambda s: s["auth_steps"] >= n

def _any_output(*kws):
    return lambda s: any(kw in out.lower() for out in s["outputs"] for kw in kws)

def _any_session_write(*kws):
    return lambda s: any(
        kw in out.lower()
        for out in s["outputs"]
        if out.startswith("$_SESSION")
        for kw in kws
    )


# Registry — ordered by priority (first match wins on equal score)
_PATTERNS: list[FlowPattern] = [

    FlowPattern(
        name        = "Login Flow",
        description = "User authentication: credential entry, validation, session creation",
        rules       = [
            ("login/signin filename",    _any_file("login", "signin", "sign_in", "auth")),
            ("password input field",     _any_input("password", "passwd")),
            ("username/email input",     _any_input("username", "email", "user")),
            ("POST method used",         _has_method("POST")),
            ("session write on success", _any_session_write("user", "id", "role", "logged")),
            ("redirect after login",     _any_output("→ ")),
        ],
        threshold   = 3,
        actor_hint  = None,   # LLM infers (could be Guest or any role)
        term_hint   = "User is authenticated and redirected to the application",
    ),

    FlowPattern(
        name        = "Registration Flow",
        description = "New user account creation: form submission, validation, account persistence",
        rules       = [
            ("register/signup filename", _any_file("register", "signup", "sign_up",
                                                    "create_account", "newuser", "new_user")),
            ("email input field",        _any_input("email")),
            ("password input field",     _any_input("password", "passwd")),
            ("INSERT db operation",      _any_db("insert")),
            ("POST method used",         _has_method("POST")),
        ],
        threshold   = 3,
        actor_hint  = "New User",
        term_hint   = "New user account is created and user is redirected",
    ),

    FlowPattern(
        name        = "Password Reset Flow",
        description = "Forgotten password recovery via email token or security question",
        rules       = [
            ("reset/forgot filename",    _any_file("reset", "forgot", "recover",
                                                    "change_password", "changepass")),
            ("email or token input",     _any_input("email", "token", "code")),
            ("password input field",     _any_input("password", "passwd", "newpass")),
            ("UPDATE db operation",      _any_db("update")),
        ],
        threshold   = 2,
        term_hint   = "User password is updated and user can log in with new credentials",
    ),

    FlowPattern(
        name        = "Checkout Flow",
        description = "E-commerce or booking purchase: cart review, payment entry, order creation",
        rules       = [
            ("cart/checkout filename",   _any_file("cart", "checkout", "payment",
                                                    "order", "purchase", "pay")),
            ("multi-step (3+ pages)",    _step_count_gte(3)),
            ("INSERT order db op",       _any_db("insert")),
            ("payment/total input",      _any_input("total", "amount", "price",
                                                     "payment", "card")),
            ("POST method",              _has_method("POST")),
        ],
        threshold   = 3,
        term_hint   = "Order is confirmed and user receives confirmation",
    ),

    FlowPattern(
        name        = "Approval Flow",
        description = "Request review and decision: submission, manager review, approve/reject",
        rules       = [
            ("approve/review filename",  _any_file("approv", "review", "reject",
                                                    "confirm", "authoris", "authorize")),
            ("status input or db op",    _any_input("status", "decision", "action")),
            ("UPDATE status db op",      _any_db("update")),
            ("auth required throughout", _auth_steps_gte(2)),
            ("POST method",              _has_method("POST")),
        ],
        threshold   = 3,
        term_hint   = "Request is approved or rejected and requester is notified",
    ),

    FlowPattern(
        name        = "CRUD Flow",
        description = "Standard create/read/update/delete data management for an entity",
        rules       = [
            ("list/add/edit/delete filename", _any_file("list", "add", "edit", "delete",
                                                         "manage", "index", "view")),
            ("write db operation",       _any_db("insert", "update", "delete")),
            ("read db operation",        _any_db("select")),
            ("form inputs present",      lambda s: len(s["inputs"]) >= 2),
            ("GET and POST both used",   lambda s: {"GET", "POST"}.issubset(s["http_methods"])),
        ],
        threshold   = 3,
        term_hint   = "Data record is created, updated, or deleted successfully",
    ),

    FlowPattern(
        name        = "Multi-step Wizard",
        description = "Sequential multi-page form where each step collects different data",
        rules       = [
            ("step/wizard/form filename", _any_file("step", "wizard", "form",
                                                     "page", "stage")),
            ("4+ steps in flow",          _step_count_gte(4)),
            ("inputs on multiple steps",  lambda s: s["steps_with_inputs"] >= 3),
            ("POST method",               _has_method("POST")),
            ("session or redirect chain", lambda s: len(s["outputs"]) >= 3),
        ],
        threshold   = 3,
        term_hint   = "All wizard steps are completed and final submission is processed",
    ),

    FlowPattern(
        name        = "Search / Filter Flow",
        description = "Query-driven list retrieval: search form, filter params, results display",
        rules       = [
            ("search/filter/list filename", _any_file("search", "filter", "find",
                                                       "list", "browse", "results")),
            ("SELECT db operation",          _any_db("select")),
            ("GET method used",              _has_method("GET")),
            ("search/query input",           _any_input("search", "query", "q",
                                                         "keyword", "filter")),
            ("no INSERT/UPDATE/DELETE",      lambda s: not any(
                kw in op.lower() for op in s["db_ops"]
                for kw in ("insert", "update", "delete")
            )),
        ],
        threshold   = 3,
        term_hint   = "Matching records are displayed to the user",
    ),

    FlowPattern(
        name        = "File Upload Flow",
        description = "Binary file or document upload: selection, validation, storage",
        rules       = [
            ("upload/import filename",  _any_file("upload", "import", "attach",
                                                   "file", "document")),
            ("file input field",        _any_input("file", "attachment", "upload",
                                                    "document", "image")),
            ("POST method",             _has_method("POST")),
            ("INSERT db operation",     _any_db("insert")),
        ],
        threshold   = 2,
        term_hint   = "File is uploaded and stored; metadata is persisted",
    ),

    FlowPattern(
        name        = "Report / Export Flow",
        description = "Data aggregation and export: parameter selection, query, download",
        rules       = [
            ("report/export filename",  _any_file("report", "export", "download",
                                                   "print", "generate", "summary")),
            ("SELECT db operation",     _any_db("select")),
            ("GET method",              _has_method("GET")),
            ("no form writes",          lambda s: not any(
                kw in op.lower() for op in s["db_ops"]
                for kw in ("insert", "update", "delete")
            )),
        ],
        threshold   = 2,
        term_hint   = "Report data is retrieved and presented or downloaded",
    ),

    FlowPattern(
        name        = "Logout Flow",
        description = "Session termination: session destruction and redirect to public page",
        rules       = [
            ("logout/signout filename", _any_file("logout", "signout", "sign_out",
                                                   "logoff", "log_out")),
            ("session destroy output",  lambda s: any(
                "session" in o.lower() or "destroy" in o.lower()
                for o in s["outputs"]
            )),
        ],
        threshold   = 1,
        term_hint   = "User session is destroyed and user is redirected to the login page",
    ),
]

# Pattern name → FlowPattern for fast lookup
_PATTERN_BY_NAME: dict[str, FlowPattern] = {p.name: p for p in _PATTERNS}


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 4.5 entry point. Extracts BusinessFlows and populates ctx.business_flows.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If Stage 4 (domain_model) or Stage 2 (graph) are missing.
    """
    output_path = ctx.output_path(FLOWS_FILE)

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage45_flows") and Path(output_path).exists():
        ctx.business_flows = _load_flows(output_path)
        n = ctx.business_flows.total
        print(f"  [stage45] Already completed — {n} flow(s) loaded.")
        _patch_domain_workflows(ctx)
        return

    _assert_prerequisites(ctx)

    print("  [stage45] Loading knowledge graph ...")
    G = _load_graph(ctx)

    # ── Tier-0: Laravel route-based flows ────────────────────────────────────────
    # For Laravel projects, build flows directly from route→controller data
    # BEFORE the generic graph BFS — produces multi-step flows with real
    # inputs/db_ops/confidence rather than 2-step conf=0.20 stubs.
    from context import Framework as _Framework
    laravel_skeletons: list[dict] = []
    if ctx.code_map and ctx.code_map.framework == _Framework.LARAVEL:
        laravel_skeletons = _traverse_laravel_routes(G, ctx)
        if laravel_skeletons:
            print(f"  [stage45]   Tier-0 Laravel: "
                  f"{len(laravel_skeletons)} route-based skeleton(s)")

    print("  [stage45] Pass A–C: Graph traversal + path deduplication ...")
    raw_paths = _traverse_graph(G, ctx)
    print(f"  [stage45]   {len(raw_paths)} candidate path(s) found")

    # Laravel skeletons first (higher quality); graph paths fill the rest
    all_raw: list = laravel_skeletons + raw_paths

    print("  [stage45] Pass D–E: Stitching execution_paths onto graph paths ...")
    flow_skeletons = _stitch_execution_paths(all_raw, ctx, G)
    print(f"  [stage45]   {len(flow_skeletons)} flow skeleton(s) built")

    print("  [stage45] Pass F: Grouping by bounded context ...")
    context_groups = _group_by_context(flow_skeletons, ctx)

    print(f"  [stage45] Pass F.5: Flow pattern mining ...")
    patterns_found = _classify_patterns(flow_skeletons)
    pattern_counts: dict[str, int] = {}
    for p in patterns_found:
        if p:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
    if pattern_counts:
        for pname, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"  [stage45]   {pname}: {cnt} flow(s)")
    else:
        print(f"  [stage45]   No patterns matched (will rely on LLM naming)")

    print(f"  [stage45] Pass G–H: LLM enrichment "
          f"({len(context_groups)} context group(s)) ...")
    flows = _enrich_with_llm(flow_skeletons, context_groups, ctx)
    print(f"  [stage45]   {len(flows)} named flow(s) produced")

    print("  [stage45] Pass I: Updating domain_model.workflows ...")
    collection = _build_collection(flows)
    _patch_domain_workflows(ctx, flows)

    _save_flows(collection, output_path)
    ctx.business_flows = collection
    ctx.stage("stage45_flows").mark_completed(output_path)
    ctx.save()

    print(f"  [stage45] Done → {output_path}")
    for f in flows:
        print(f"  [stage45]   [{f.flow_id}] {f.name} "
              f"({len(f.steps)} steps, actor={f.actor}, "
              f"ctx={f.bounded_context}, conf={f.confidence:.2f})")


# ─── Prerequisites ─────────────────────────────────────────────────────────────

def _assert_prerequisites(ctx: PipelineContext) -> None:
    """Raise RuntimeError if required upstream outputs are missing."""
    if ctx.domain_model is None:
        raise RuntimeError(
            "[stage45] ctx.domain_model is None — run Stage 4 first."
        )
    if ctx.graph_meta is None or not ctx.graph_meta.graph_path:
        raise RuntimeError(
            "[stage45] ctx.graph_meta is missing — run Stage 2 first."
        )
    gpickle = ctx.graph_meta.graph_path
    if not Path(gpickle).exists():
        raise RuntimeError(
            f"[stage45] Graph file not found: {gpickle} — run Stage 2 first."
        )



# --- Tier-0: Laravel Route-Based Flow Traversal ---

def _traverse_laravel_routes(G, ctx) -> list:
    """
    Build multi-step flow skeletons from Laravel route->controller data.
    Called as Tier-0 BEFORE the generic graph BFS in _traverse_graph().
    Produces 3-step flows: route_entry -> controller_action -> model_operation.
    Results are pre-built dicts passed through _stitch_execution_paths unchanged.
    """
    from collections import defaultdict
    from context import FlowStep

    if not ctx.code_map or not ctx.code_map.routes:
        return []

    # Index Stage 1.5 execution_paths by file and by controller stem name
    ep_by_ctrl_name: dict = {}
    exec_paths = getattr(ctx.code_map, "execution_paths", None) or []
    for ep in exec_paths:
        fname = ep.get("file", "")
        if ep.get("type") == "controller" and fname:
            stem = Path(fname).stem
            ep_by_ctrl_name[stem] = ep

    # Group routes by controller short name.
    # Actual route schema: {method, path, handler, file, line, source}
    # handler examples:
    #   "App\\Http\\Controllers\\V1\\Admin\\InvoicesController@index"
    #   "(closure)"  <- skip these
    ctrl_routes: dict = defaultdict(list)
    for route in ctx.code_map.routes:
        handler = route.get("handler", "") or ""
        # Skip closures, group markers, and empty handlers
        if not handler or handler in ("(closure)", "(group)", ""):
            continue
        # Parse "Namespace\\ClassName@method" -> short class name
        if "@" in handler:
            ctrl_fqn = handler.split("@")[0]
        elif "::" in handler:
            ctrl_fqn = handler.split("::")[0]
        else:
            ctrl_fqn = handler
        ctrl_short = ctrl_fqn.split("\\")[-1].split("/")[-1]
        if ctrl_short and ctrl_short != "(closure)":
            ctrl_routes[ctrl_short].append(route)

    skeletons = []

    for ctrl_name, routes in ctrl_routes.items():
        ctrl_ep = ep_by_ctrl_name.get(ctrl_name, {})
        ctrl_flows_by_method: dict = {}
        for cf in ctrl_ep.get("controller_flows", []):
            ctrl_flows_by_method[cf.get("method", "")] = cf

        for route in routes:
            # Use actual Stage 1 field names: method / path / handler
            http_verb   = (route.get("method") or "GET").upper()
            # Skip GROUP/middleware-only entries
            if http_verb in ("GROUP", "MIDDLEWARE", "PREFIX"):
                continue
            uri         = route.get("path") or route.get("uri") or "?"
            handler_str = route.get("handler", "")
            method_name = (
                handler_str.split("@")[-1] if "@" in handler_str
                else handler_str.split("::")[-1] if "::" in handler_str
                else "index"
            )
            middleware = route.get("middleware", [])
            if isinstance(middleware, str):
                middleware = [middleware]
            auth_required = any(
                mw in ("auth", "auth:sanctum", "auth:api", "verified")
                for mw in middleware
            )

            cf               = ctrl_flows_by_method.get(method_name, {})
            inputs           = cf.get("inputs", [])
            validation_rules = cf.get("validation_rules", [])
            db_ops           = cf.get("db_ops", [])
            return_type      = cf.get("return_type", "unknown")
            outputs          = cf.get("outputs", [])
            auth_required    = auth_required or cf.get("auth_required", False)

            db_op_strs = [
                f"{op.get('operation','?')}:{op.get('model','?')}"
                for op in (db_ops if isinstance(db_ops, list) else [])
                if isinstance(op, dict)
            ]

            step1 = FlowStep(
                step_num=1, page=uri, action=f"{http_verb} {uri}",
                http_method=http_verb, auth_required=auth_required,
                db_ops=[], inputs=[], outputs=[],
            )
            step2 = FlowStep(
                step_num=2, page=f"{ctrl_name}::{method_name}",
                action=(
                    f"Validate [{', '.join(str(v) for v in validation_rules[:3])}]"
                    if validation_rules else f"Process {method_name}"
                ),
                http_method=http_verb, auth_required=auth_required,
                db_ops=db_op_strs,
                inputs=[str(i) for i in inputs[:8]],
                outputs=[str(o) for o in outputs[:4]],
            )
            steps = [step1, step2]

            # Build step 3 from db_ops — handles both string and dict formats.
            # String format (Stage 1.5 output): "Eloquent create: Customer"
            # Dict format (hypothetical):       {"operation": "create", "model": "Customer"}
            _step3_model = ""
            _step3_action = ""
            if db_ops and isinstance(db_ops, list):
                _first = db_ops[0]
                if isinstance(_first, dict):
                    _step3_model  = _first.get("model", "?")
                    _step3_action = f"Eloquent {_first.get('operation', '?')}"  
                elif isinstance(_first, str) and _first.strip():
                    # Parse "Eloquent create: Customer" or "Eloquent where: Invoice"
                    _parts = _first.replace("Eloquent", "").strip().split(":")
                    _step3_action = f"Eloquent {_parts[0].strip()}" if _parts else _first
                    _step3_model  = _parts[1].strip() if len(_parts) > 1 else _first
            if _step3_model or _step3_action:
                step3 = FlowStep(
                    step_num=3, page=_step3_model or "Model",
                    action=_step3_action or f"Eloquent op on {_step3_model}",
                    http_method=None, auth_required=False,
                    db_ops=db_op_strs, inputs=[],
                    outputs=[str(o) for o in outputs[:4]],
                )
                steps.append(step3)

            confidence = _laravel_route_confidence(
                has_controller_ep=bool(ctrl_ep),
                has_inputs=bool(inputs),
                has_db_ops=bool(db_ops),
                has_validation=bool(validation_rules),
                has_middleware=bool(middleware),
                step_count=len(steps),
            )

            skeletons.append({
                "path_nodes":     [],
                "files":          [ctrl_ep.get("file", f"{ctrl_name}.php")],
                "steps":          steps,
                "branches":       [],
                "evidence_files": [ctrl_ep.get("file", "")] if ctrl_ep else [],
                "raw_confidence": confidence,
                "_laravel_meta": {
                    "http_verb": http_verb, "uri": uri,
                    "controller": ctrl_name, "method": method_name,
                    "middleware": middleware, "inputs": inputs,
                    "validation_rules": validation_rules,
                    "db_ops": db_op_strs, "return_type": return_type,
                },
            })

    return skeletons


def _laravel_route_confidence(
    has_controller_ep, has_inputs, has_db_ops,
    has_validation, has_middleware, step_count,
) -> float:
    """Score a Laravel route skeleton 0.20-0.80 based on evidence quality."""
    score = 0.20
    if has_controller_ep: score += 0.15
    if has_inputs:        score += 0.10
    if has_db_ops:        score += 0.15
    if has_validation:    score += 0.10
    if has_middleware:    score += 0.05
    if step_count >= 3:   score += 0.05
    return round(min(0.80, score), 2)


def _build_laravel_flow_skeleton(sk: dict) -> str:
    """Render a Tier-0 skeleton as a description for LLM enrichment prompts."""
    meta = sk.get("_laravel_meta", {})
    if not meta:
        return ""
    lines = [f"HTTP {meta['http_verb']} {meta['uri']}",
             f"  Controller: {meta['controller']}::{meta['method']}"]
    if meta.get("middleware"):
        lines.append(f"  Middleware: {', '.join(meta['middleware'])}")
    if meta.get("inputs"):
        lines.append(f"  Inputs: {', '.join(str(i) for i in meta['inputs'][:6])}")
    if meta.get("validation_rules"):
        lines.append(f"  Validates: {', '.join(str(v) for v in meta['validation_rules'][:6])}")
    if meta.get("db_ops"):
        lines.append(f"  DB ops: {', '.join(meta['db_ops'][:4])}")
    if meta.get("return_type") and meta["return_type"] != "unknown":
        lines.append(f"  Returns: {meta['return_type']}")
    return "\n".join(lines)


# ─── Pass A–C: Graph Traversal ────────────────────────────────────────────────

def _load_graph(ctx: PipelineContext):
    """Load the NetworkX DiGraph from the .gpickle produced by Stage 2."""
    with open(ctx.graph_meta.graph_path, "rb") as fh:
        return pickle.load(fh)


def _traverse_graph(G, ctx: PipelineContext) -> list[list[str]]:
    """
    Pass A: Identify all entry-point nodes (http_endpoint + is_entry_point tagged).
    Pass B: BFS from each, following FOLLOW_EDGES up to MAX_DEPTH hops.
    Pass C: Deduplicate — keep the longest path for each unique file sequence.

    Returns a list of node-ID paths (each path is a list[str]).

    Entry-point discovery (three tiers, most-specific first):
      Tier 1 — index: http_endpoints + entry_points + routes  (framework apps)
      Tier 2 — node attr: any node with is_entry_point=True or type=http_endpoint
      Tier 3 — raw PHP fallback: page/script nodes that have at least one
               outgoing edge in FOLLOW_EDGES (redirects_to, submits_to, handles).
               These are interactive files that redirect/submit to other pages,
               which is the defining characteristic of a flow entry-point in a
               raw PHP app with no framework routing.
    """
    import networkx as nx

    # ── Tier 1: use the index built by Stage 2 Pass 14 ────────────────────────
    idx = G.graph.get("index", {})
    entry_nodes: list[str] = list({
        *idx.get("http_endpoints", []),
        *idx.get("entry_points", []),
        *idx.get("routes", []),
    })

    # ── Tier 2: node attribute scan ───────────────────────────────────────────
    if not entry_nodes:
        entry_nodes = [n for n, d in G.nodes(data=True)
                       if d.get("is_entry_point") or d.get("type") == "http_endpoint"]

    # ── Tier 3: raw PHP fallback — page/script nodes with outbound navigation edges
    #
    # Runs when:
    #   (a) no entry nodes found at all (original condition), OR
    #   (b) Tier-1/2 found nodes but NONE of them have real navigation edges
    #       (redirects_to / submits_to / handles / redirect).
    #
    # Case (b) covers projects like SugarCRM / raw PHP apps where stage2 indexes
    # controller-method EP: nodes and ROUTE: nodes that have zero outgoing edges,
    # causing BFS to produce 0 valid paths even though 300+ page nodes with
    # redirects_to edges exist in the graph.
    #
    # When Tier-3 triggers in case (b), its nav-page nodes are MERGED with the
    # existing entry_nodes so framework-level EP: paths are still explored.
    _NAV_EDGES = {"redirects_to", "submits_to", "handles", "redirect"}
    _tier_has_nav = entry_nodes and any(
        any(G.edges[n, dst].get("edge_type", "") in _NAV_EDGES
            for dst in G.successors(n))
        for n in entry_nodes
        if n in G
    )
    if not entry_nodes or not _tier_has_nav:
        _page_types = {"page", "script"}
        _nav_pages = [
            n for n, d in G.nodes(data=True)
            if d.get("type") in _page_types
            and any(
                G.edges[n, dst].get("edge_type", "") in _NAV_EDGES
                for dst in G.successors(n)
            )
        ]
        if _nav_pages:
            _prev = len(entry_nodes)
            entry_nodes = list({*entry_nodes, *_nav_pages})
            added = len(entry_nodes) - _prev
            label = "as only entry-points" if _prev == 0 else f"+{added} to existing {_prev}"
            print(f"  [stage45]   Tier-3: {len(_nav_pages)} page/script nav nodes "
                  f"({label}) — raw PHP project fallback")

    # ── REDIRECT: node → target page resolution map ───────────────────────────
    # Stage 2 creates  page_node --[redirects_to]--> REDIRECT:target.php
    # but never adds a second edge from REDIRECT:target.php back to the actual
    # target page node.  We build this lookup once so the BFS can hop through
    # REDIRECT: nodes to reach their target page file node.
    #
    # Strategy: match REDIRECT:<name> against every page/script node whose
    # filename matches the redirect target basename.
    _redirect_target: dict[str, str] = {}   # REDIRECT:x → page_node_id
    _page_nodes_by_file: dict[str, str] = {}  # filename (lower, no ext) → node_id
    for _nid, _d in G.nodes(data=True):
        if _d.get("type") in ("page", "script"):
            _stem = Path(_d.get("file", _nid)).name.lower()
            _page_nodes_by_file[_stem] = _nid
            # also index without extension for loose matching
            _page_nodes_by_file[Path(_stem).stem] = _nid

    for _nid, _d in G.nodes(data=True):
        if _d.get("type") == "redirect":
            _target_url = _d.get("url", _d.get("name", ""))
            # Extract filename from URL: "login.php", "dashboard.php", "../login.php"
            _target_file = Path(_target_url).name.lower()
            _target_stem = Path(_target_file).stem
            _resolved = (
                _page_nodes_by_file.get(_target_file) or
                _page_nodes_by_file.get(_target_stem)
            )
            if _resolved:
                _redirect_target[_nid] = _resolved

    candidate_paths: list[list[str]] = []

    for start in entry_nodes:
        # BFS — each frontier element is the current path so far
        queue: deque[list[str]] = deque([[start]])
        visited_from_start: set[str] = {start}

        while queue and len(candidate_paths) < MAX_PATHS:
            path = queue.popleft()
            current = path[-1]

            if len(path) >= MAX_PATH_LEN:
                candidate_paths.append(path)
                continue

            extended = False
            for _, dst, edata in G.out_edges(current, data=True):
                etype = edata.get("edge_type", "")
                if etype not in FOLLOW_EDGES:
                    continue
                if dst in visited_from_start:
                    continue
                # Skip EXTCALL nodes — they're not pages
                if str(dst).startswith("EXTCALL:"):
                    continue

                # ── REDIRECT: pass-through ────────────────────────────────────
                # If dst is a REDIRECT: node, replace it with the actual target
                # page node so the BFS continues through to real PHP files.
                # Include the REDIRECT: node in the path so _file_fingerprint
                # can still see it (it will skip it because it has no "file").
                if str(dst).startswith("REDIRECT:"):
                    visited_from_start.add(dst)
                    resolved_page = _redirect_target.get(dst)
                    if resolved_page and resolved_page not in visited_from_start:
                        visited_from_start.add(resolved_page)
                        queue.append(path + [dst, resolved_page])
                        extended = True
                    # If unresolvable, still record path so far (dead-end branch)
                    continue

                visited_from_start.add(dst)
                queue.append(path + [dst])
                extended = True

            if not extended and len(path) >= MIN_PAGE_STEPS:
                candidate_paths.append(path)

    # Pass C: deduplicate by file sequence fingerprint — keep longest per fingerprint
    def _file_fingerprint(path: list[str]) -> tuple[str, ...]:
        """Collapse a node-ID path to the sequence of distinct PHP files it touches.
        REDIRECT: intermediate nodes are skipped — they carry no file attribute."""
        seen: list[str] = []
        for n in path:
            if str(n).startswith("REDIRECT:"):
                continue   # pass-through node — no file to record
            ndata = G.nodes[n]
            f = ndata.get("file", "")
            if f and (not seen or seen[-1] != f):
                seen.append(f)
        return tuple(seen)

    best: dict[tuple, list[str]] = {}
    for path in candidate_paths:
        fp = _file_fingerprint(path)
        if len(fp) >= MIN_PAGE_STEPS:
            if fp not in best or len(path) > len(best[fp]):
                best[fp] = path

    return list(best.values())


# ─── Pass D–E: Happy-Path Stitching ───────────────────────────────────────────

def _stitch_execution_paths(
    raw_paths: list[list[str]],
    ctx: PipelineContext,
    G,
) -> list[dict[str, Any]]:
    """
    Pass D: For each graph path, collect the PHP files it visits in order.
    Pass E: Merge Stage 1.5 execution_paths data for those files into FlowStep
            objects, carrying db_ops, auth guards, inputs, and outputs.

    Returns a list of "flow skeleton" dicts — structured data before LLM naming.
    Each dict has:
        path_nodes    : list[str]   — raw graph node IDs
        files         : list[str]   — ordered PHP files
        steps         : list[FlowStep]
        branches      : list[dict]
        evidence_files: list[str]
        raw_confidence: float
    """
    # Index execution_paths by file for O(1) lookup
    ep_by_file: dict[str, dict] = {}
    exec_paths = getattr(ctx.code_map, "execution_paths", None) or []
    for ep in exec_paths:
        fname = ep.get("file", "")
        if fname:
            ep_by_file[fname] = ep

    skeletons: list[dict[str, Any]] = []

    for path_nodes in raw_paths:
        # ── Tier-0 pass-through ────────────────────────────────────────────────
        # Pre-built skeletons from _traverse_laravel_routes() arrive as dicts
        # (not lists of node IDs). Pass them through unchanged — they already
        # have steps, files, raw_confidence, etc. fully populated.
        if isinstance(path_nodes, dict) and "steps" in path_nodes:
            skeletons.append(path_nodes)
            continue

        # Ordered unique files this path visits
        files: list[str] = []
        file_set: set[str] = set()
        for n in path_nodes:
            f = G.nodes[n].get("file", "")
            if f and f not in file_set:
                files.append(f)
                file_set.add(f)

        if len(files) < MIN_PAGE_STEPS:
            continue

        steps: list[FlowStep] = []
        branches: list[dict]  = []
        step_num = 1

        for file_path in files:
            ep = ep_by_file.get(file_path, {})
            page_name = Path(file_path).name

            # Determine HTTP method from entry_conditions
            http_method: Optional[str] = None
            for ec in ep.get("entry_conditions", []):
                if ec.get("type") == "method_check":
                    http_method = ec.get("method")

            # Auth guard
            auth_required = bool(ep.get("auth_guard"))

            # Inputs: from form_fields / POST entry_conditions
            inputs: list[str] = list({
                ec.get("key", "")
                for ec in ep.get("entry_conditions", [])
                if ec.get("key")
            })

            # Outputs: session keys written + redirect targets
            outputs: list[str] = []
            for b in ep.get("branches", []):
                for action in b.get("then", []) + b.get("else", []):
                    if action.get("action") == "session_write":
                        outputs.append(f"$_SESSION['{action.get('key','?')}']")
                    elif action.get("action") == "redirect":
                        outputs.append(f"→ {action.get('target','?')}")

            # DB operations from happy path and branches
            db_ops: list[str] = []
            for step_text in ep.get("happy_path", []):
                lower = step_text.lower()
                for kw in ("select", "insert", "update", "delete", "query"):
                    if kw in lower:
                        db_ops.append(step_text[:80])
                        break

            # Build the happy-path action description
            happy_path = ep.get("happy_path", [])
            if happy_path:
                action_desc = happy_path[0]  # first step is the trigger action
            else:
                # Synthesise from node name in graph
                node_data = next(
                    (G.nodes[n] for n in path_nodes if G.nodes[n].get("file") == file_path),
                    {}
                )
                action_desc = node_data.get("name", page_name)

            steps.append(FlowStep(
                step_num     = step_num,
                page         = page_name,
                action       = action_desc,
                http_method  = http_method,
                db_ops       = db_ops,
                auth_required= auth_required,
                inputs       = inputs,
                outputs      = outputs[:4],   # cap at 4 outputs per step
            ))
            step_num += 1

            # Collect branch info (alternate paths)
            for b in ep.get("branches", []):
                cond = b.get("condition", "")
                else_actions = b.get("else", [])
                if else_actions and cond:
                    branches.append({
                        "at_page":   page_name,
                        "condition": cond[:100],
                        "alternate": [a.get("action","?") for a in else_actions[:3]],
                    })

        # Confidence heuristic: how many files have execution_path evidence
        covered = sum(1 for f in files if f in ep_by_file)
        confidence = covered / len(files) if files else 0.5

        skeletons.append({
            "path_nodes":    path_nodes,
            "files":         files,
            "steps":         steps,
            "branches":      branches[:6],     # cap branches per flow
            "evidence_files": files,
            "raw_confidence": round(confidence, 2),
        })

    return skeletons


# ─── Pass F: Group by Bounded Context ─────────────────────────────────────────

def _group_by_context(
    skeletons: list[dict[str, Any]],
    ctx: PipelineContext,
) -> dict[str, list[int]]:
    """
    Assign each skeleton to a bounded context.

    Strategy (two-phase with fallback):

    Phase 1 — Graph module_id lookup (preferred):
        If the graph was built with Stage 2 Pass 15, every code node carries a
        module_id attribute.  For each skeleton, collect the module_ids of all
        files in its evidence set by querying graph nodes, then assign the
        skeleton to the plurality module.  The module's display name
        (G.graph["modules"][id]["name"]) is used as the context key so it
        matches domain_model.bounded_contexts populated in Stage 4.

    Phase 2 — Keyword fallback (legacy / no graph):
        If the graph is unavailable or pre-dates Pass 15, fall back to the
        original keyword-matching approach against domain_model.bounded_contexts.

    Returns: {bounded_context_name: [skeleton_indices]}
    """
    import pickle
    from collections import Counter as _Counter

    # ── Phase 1: graph module_id lookup ──────────────────────────────────────
    G = None
    modules_summary: dict[str, dict] = {}
    file_to_module: dict[str, str]   = {}   # filepath → module display name

    if ctx.graph_meta and ctx.graph_meta.graph_path:
        try:
            with open(ctx.graph_meta.graph_path, "rb") as _fh:
                G = pickle.load(_fh)
            modules_summary = G.graph.get("modules", {})
            # Build file→module_name map from node attributes
            for _nid, _d in G.nodes(data=True):
                _fp  = _d.get("file", "")
                _mid = _d.get("module_id", "")
                if _fp and _mid and _fp not in file_to_module:
                    _mname = modules_summary.get(_mid, {}).get("name", _mid.title())
                    file_to_module[_fp] = _mname
        except Exception:
            G = None

    if file_to_module:
        # Assign each skeleton to its plurality module
        groups: dict[str, list[int]] = defaultdict(list)
        _raw_contexts = ctx.domain_model.bounded_contexts or []
        _domain_contexts = set(
            c["name"] if isinstance(c, dict) else c for c in _raw_contexts
        )

        for i, sk in enumerate(skeletons):
            mod_counts: _Counter = _Counter()
            for fpath in sk.get("files", []):
                # Exact match first, then basename fallback
                mname = file_to_module.get(fpath, "")
                if not mname:
                    base = Path(fpath).name
                    mname = next(
                        (m for fp, m in file_to_module.items()
                         if Path(fp).name == base),
                        ""
                    )
                if mname:
                    mod_counts[mname] += 1

            if mod_counts:
                best = mod_counts.most_common(1)[0][0]
            else:
                # No file matched any module — use the first domain context
                _first = (_raw_contexts or [{"name": "General"}])[0]
                best = _first["name"] if isinstance(_first, dict) else _first

            # If domain_model already has a matching context name, preserve it
            # (handles case where Stage 4 renamed a module slightly)
            if best not in _domain_contexts and _domain_contexts:
                # Find the closest match by shared words
                best_words = set(best.lower().split())
                scored = [
                    (len(best_words & set(c.lower().split())), c)
                    for c in _domain_contexts
                ]
                scored.sort(reverse=True)
                if scored and scored[0][0] > 0:
                    best = scored[0][1]

            groups[best].append(i)

        return dict(groups)

    # ── Phase 2: keyword fallback (no module_id in graph) ────────────────────
    contexts = ctx.domain_model.bounded_contexts or ["General"]

    context_keywords: dict[str, set[str]] = {}
    for c in contexts:
        words = re.sub(r"([A-Z])", r" \1", c).lower().split()
        context_keywords[c] = set(words)

    groups_kw: dict[str, list[int]] = defaultdict(list)

    for i, sk in enumerate(skeletons):
        files_lower = " ".join(Path(f).stem.lower() for f in sk["files"])
        best_ctx    = contexts[0]
        best_score  = 0

        for c, kws in context_keywords.items():
            score = sum(1 for kw in kws if kw in files_lower)
            if score > best_score:
                best_score = score
                best_ctx   = c

        groups_kw[best_ctx].append(i)

    return dict(groups_kw)


# ─── Pass F.5: Flow Pattern Mining ───────────────────────────────────────────

def _classify_patterns(
    skeletons: list[dict[str, Any]],
) -> list[str | None]:
    """
    Pass F.5 — Algorithmic pattern mining.  No LLM required.

    For each skeleton, extract a signal dict from its files, steps, inputs,
    db_ops and HTTP methods, then score it against every entry in _PATTERNS.
    Returns a list (parallel to skeletons) of matched pattern names or None.

    The result is attached to each skeleton as skeleton["pattern"] and is
    passed to the LLM in Pass G so it can use the pattern as a naming hint.

    Signal dict schema
    ------------------
    file_stems        : list[str]   — lowercased stem of each PHP file
    inputs            : list[str]   — all input field names across all steps
    db_ops            : list[str]   — all db_op strings across all steps
    http_methods      : set[str]    — distinct HTTP verbs seen
    auth_steps        : int         — number of steps with auth_required=True
    step_count        : int         — total number of steps
    outputs           : list[str]   — all output strings across all steps
    steps_with_inputs : int         — number of steps that have at least one input
    """
    results: list[str | None] = []

    for sk in skeletons:
        signals = _extract_signals(sk)
        best_name:  str | None = None
        best_score: int        = 0

        for pattern in _PATTERNS:
            score = sum(
                1 for _rule_name, rule_fn in pattern.rules
                if rule_fn(signals)
            )
            if score >= pattern.threshold and score > best_score:
                best_score = score
                best_name  = pattern.name

        sk["pattern"]       = best_name
        sk["pattern_score"] = best_score
        results.append(best_name)

    return results


def _extract_signals(sk: dict[str, Any]) -> dict[str, Any]:
    """Build the signal dict used by FlowPattern rule functions."""
    steps: list[FlowStep] = sk.get("steps", [])

    all_inputs:  list[str] = []
    all_db_ops:  list[str] = []
    all_outputs: list[str] = []
    http_methods: set[str] = set()
    auth_steps  = 0
    steps_with_inputs = 0

    for step in steps:
        all_inputs.extend(step.inputs or [])
        all_db_ops.extend(step.db_ops or [])
        all_outputs.extend(step.outputs or [])
        if step.http_method:
            http_methods.add(step.http_method.upper())
        if step.auth_required:
            auth_steps += 1
        if step.inputs:
            steps_with_inputs += 1

    file_stems = [
        Path(f).stem.lower().replace("-", "_")
        for f in sk.get("files", [])
    ]

    return {
        "file_stems":        file_stems,
        "inputs":            [i.lower() for i in all_inputs],
        "db_ops":            all_db_ops,
        "http_methods":      http_methods,
        "auth_steps":        auth_steps,
        "step_count":        len(steps),
        "outputs":           all_outputs,
        "steps_with_inputs": steps_with_inputs,
    }


# ─── Pass G–H: LLM Enrichment ─────────────────────────────────────────────────

def _enrich_with_llm(
    skeletons: list[dict[str, Any]],
    context_groups: dict[str, list[int]],
    ctx: PipelineContext,
) -> list[BusinessFlow]:
    """
    Pass G: For each bounded context, build one Claude prompt containing all
            flow skeletons in that context and request structured naming.
    Pass H: Merge LLM-returned names/actors/descriptions back onto FlowStep
            skeletons to produce final BusinessFlow objects.

    One API call per bounded context keeps token usage bounded.
    """
    from pipeline.llm_client import call_llm

    flows: list[BusinessFlow] = []
    flow_counter = 1
    domain = ctx.domain_model

    # Cap: send at most MAX_LLM_BATCH skeletons per LLM call to avoid
    # max_tokens truncation on large context groups (e.g. Routes with 80+).
    MAX_LLM_BATCH = 20

    for context_name, indices in context_groups.items():
        group_skeletons = [skeletons[i] for i in indices]

        # Split into sub-batches if needed
        enrichments: list[dict] = []
        system = _build_llm_system_prompt(domain)
        for batch_start in range(0, len(group_skeletons), MAX_LLM_BATCH):
            batch = group_skeletons[batch_start:batch_start + MAX_LLM_BATCH]
            batch_label = f"stage45_{context_name[:15]}_{batch_start//MAX_LLM_BATCH+1}"
            user = _build_llm_user_prompt(context_name, batch, domain)
            try:
                raw = call_llm(system, user, max_tokens=MAX_TOKENS,
                               label=batch_label)
                batch_enr = _parse_llm_response(raw, len(batch))
            except Exception as e:
                print(f"  [stage45] Warning: LLM call failed for '{context_name}' "
                      f"batch {batch_start//MAX_LLM_BATCH+1}: {e}")
                batch_enr = [{}] * len(batch)
            enrichments.extend(batch_enr)

        # Merge enrichments onto skeletons
        for sk, enr in zip(group_skeletons, enrichments):
            flow_id = f"flow_{flow_counter:03d}"
            flow_counter += 1

            # Pattern hint: if LLM didn't supply a termination, fall back to
            # the pattern's term_hint before the generic _infer_termination()
            pattern      = sk.get("pattern")
            pattern_obj  = _PATTERN_BY_NAME.get(pattern) if pattern else None
            termination  = (
                enr.get("termination")
                or (pattern_obj.term_hint if pattern_obj else None)
                or _infer_termination(sk)
            )
            # Actor: pattern actor_hint overrides inferred actor only when LLM
            # didn't supply one (preserves LLM judgement when available)
            actor = (
                enr.get("actor")
                or (pattern_obj.actor_hint if pattern_obj else None)
                or _infer_actor(sk, domain)
            )
            # Pattern match adds a small confidence bonus (already partially
            # rewarded by LLM success via _final_confidence)
            pattern_bonus = 0.05 if (pattern and not enr) else 0.0

            flows.append(BusinessFlow(
                flow_id          = flow_id,
                name             = enr.get("name") or _fallback_name(sk),
                actor            = actor,
                bounded_context  = context_name,
                trigger          = enr.get("trigger") or _infer_trigger(sk),
                steps            = sk["steps"],
                branches         = sk["branches"],
                termination      = termination,
                evidence_files   = sk["evidence_files"],
                confidence       = _final_confidence(
                                       sk["raw_confidence"] + pattern_bonus,
                                       bool(enr),
                                   ),
                replaces_workflow= enr.get("replaces_workflow"),
            ))

    return flows


def _build_llm_system_prompt(domain) -> str:
    return f"""You are a senior Business Analyst extracting business flows from PHP codebase evidence.
You will receive skeletons of user journeys (sequences of PHP pages with actions).
Your job is to assign each skeleton a meaningful business name, actor, trigger, and termination.

Domain: {domain.domain_name}
Known actors: {", ".join(r["role"] for r in domain.user_roles) or "Unknown"}
Bounded contexts: {", ".join(domain.bounded_contexts) or "General"}

Rules:
- Output ONLY valid JSON — no markdown fences, no explanation
- Use business language, not technical PHP file names
- name: short verb-noun phrase e.g. "Customer Login", "Booking Submission"
- actor: the user role performing the flow
- trigger: one sentence — what causes the flow to begin
- termination: one sentence — the successful end state
- replaces_workflow: name of an existing workflow this replaces, or null
- Respond with a JSON array of objects, one per skeleton, in the same order"""


def _build_llm_user_prompt(
    context_name: str,
    skeletons: list[dict[str, Any]],
    domain,
) -> str:
    parts = [f"Bounded context: {context_name}",
             f"Existing workflows: {[w.get('name','?') for w in domain.workflows]}",
             "",
             "Flow skeletons to enrich (in order):"]

    for i, sk in enumerate(skeletons):
        parts.append(f"\nSkeleton {i+1}:")
        # Include pattern hint if mining found a match — helps LLM name accurately
        pattern = sk.get("pattern")
        if pattern:
            pattern_obj = _PATTERN_BY_NAME.get(pattern)
            desc = pattern_obj.description if pattern_obj else ""
            parts.append(f"  Pattern match: {pattern}" + (f" — {desc}" if desc else ""))
            if pattern_obj and pattern_obj.term_hint:
                parts.append(f"  Suggested termination: {pattern_obj.term_hint}")
        parts.append(f"  Files visited: {' → '.join(Path(f).name for f in sk['files'])}")
        for step in sk["steps"]:
            auth_tag = " [AUTH]" if step.auth_required else ""
            db_tag   = f" [DB: {', '.join(step.db_ops[:2])}]" if step.db_ops else ""
            inputs_tag = f" inputs={step.inputs}" if step.inputs else ""
            parts.append(
                f"  Step {step.step_num}: {step.page} — {step.action}"
                f"{auth_tag}{db_tag}{inputs_tag}"
            )
        if sk["branches"]:
            parts.append(f"  Branches: {len(sk['branches'])} alternate path(s)")

    parts.append(f"""
Return a JSON array with exactly {len(skeletons)} objects:
[
  {{
    "name": "Business Flow Name",
    "actor": "Role Name",
    "trigger": "What initiates this flow",
    "termination": "Successful end state",
    "replaces_workflow": "existing workflow name or null"
  }}
]""")

    return "\n".join(parts)


def _parse_llm_response(raw: str, expected: int) -> list[dict[str, Any]]:
    """
    Parse Claude's JSON array response.  Returns a list of enrichment dicts,
    padded with empty dicts if the count doesn't match expected.
    """
    text = raw.strip()
    # Strip accidental markdown fences
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            # Pad or trim to expected length
            data = (data + [{}] * expected)[:expected]
            return data
    except json.JSONDecodeError:
        # Try to extract array with regex
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, list):
                    return (data + [{}] * expected)[:expected]
            except json.JSONDecodeError:
                pass

    return [{}] * expected


# ─── Fallback Helpers ──────────────────────────────────────────────────────────

def _fallback_name(sk: dict[str, Any]) -> str:
    """Generate a name from file stems when LLM fails."""
    names = [Path(f).stem.replace("_", " ").replace("-", " ").title()
             for f in sk["files"][:3]]
    return " → ".join(names) if names else "Unknown Flow"


def _infer_actor(sk: dict[str, Any], domain) -> str:
    """Infer actor from auth guard presence and domain roles."""
    has_auth = any(step.auth_required for step in sk["steps"])
    if domain.user_roles:
        if has_auth:
            # Second role is usually the authenticated one (admin/customer)
            return domain.user_roles[min(1, len(domain.user_roles)-1)]["role"]
        return domain.user_roles[0]["role"]
    return "Authenticated User" if has_auth else "Anonymous User"


def _infer_trigger(sk: dict[str, Any]) -> str:
    """Infer trigger from the first step."""
    if sk["steps"]:
        first = sk["steps"][0]
        method = first.http_method or "GET"
        return f"User {method}s {first.page}"
    return "User navigates to entry point"


def _infer_termination(sk: dict[str, Any]) -> str:
    """Infer termination from the last step's outputs."""
    if sk["steps"]:
        last = sk["steps"][-1]
        redirects = [o for o in last.outputs if o.startswith("→")]
        if redirects:
            return f"User is redirected to {redirects[-1][2:].strip()}"
        if last.db_ops:
            return f"Data persisted via {last.page}"
    return "Flow completes successfully"


def _final_confidence(raw: float, llm_succeeded: bool) -> float:
    """
    Combine graph-traversal confidence with LLM enrichment success.
    Graph-only = raw score, LLM-confirmed = raw + 0.2 bonus (capped at 1.0).
    """
    bonus = 0.2 if llm_succeeded else 0.0
    return round(min(1.0, raw + bonus), 2)


# ─── Pass I: Patch domain_model.workflows ─────────────────────────────────────

def _patch_domain_workflows(
    ctx: PipelineContext,
    flows: Optional[list[BusinessFlow]] = None,
) -> None:
    """
    Replace ctx.domain_model.workflows with richer flow summaries derived
    from BusinessFlow objects.  Stage 5 agents will automatically use these.

    Called both on fresh run (flows provided) and on resume (loaded from JSON).
    """
    if ctx.domain_model is None:
        return

    source = flows or (
        ctx.business_flows.flows if ctx.business_flows else []
    )
    if not source:
        return

    ctx.domain_model.workflows = [
        {
            "name":           f.name,
            "actor":          f.actor,
            "bounded_context": f.bounded_context,
            "trigger":        f.trigger,
            "steps": [
                {
                    "step":        s.step_num,
                    "page":        s.page,
                    "action":      s.action,
                    "http_method": s.http_method,
                    "auth":        s.auth_required,
                    "db_ops":      s.db_ops,
                    "inputs":      s.inputs,
                    "outputs":     s.outputs,
                }
                for s in f.steps
            ],
            "branches":       f.branches,
            "termination":    f.termination,
            "confidence":     f.confidence,
        }
        for f in source
    ]


# ─── Build Collection ──────────────────────────────────────────────────────────

def _build_collection(flows: list[BusinessFlow]) -> BusinessFlowCollection:
    """Assemble a BusinessFlowCollection with cross-reference indexes."""
    by_context: dict[str, list[str]] = defaultdict(list)
    by_actor:   dict[str, list[str]] = defaultdict(list)

    for f in flows:
        by_context[f.bounded_context].append(f.flow_id)
        by_actor[f.actor].append(f.flow_id)

    return BusinessFlowCollection(
        flows      = flows,
        total      = len(flows),
        by_context = dict(by_context),
        by_actor   = dict(by_actor),
    )


# ─── Persistence ──────────────────────────────────────────────────────────────

def _save_flows(collection: BusinessFlowCollection, path: str) -> None:
    """Serialise BusinessFlowCollection to JSON."""
    import dataclasses

    def _ser(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _ser(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):  return [_ser(i) for i in obj]
        if isinstance(obj, dict):  return {k: _ser(v) for k, v in obj.items()}
        return obj

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_ser(collection), fh, indent=2, ensure_ascii=False)
    print(f"  [stage45] Saved → {path}")


def _load_flows(path: str) -> BusinessFlowCollection:
    """Restore BusinessFlowCollection from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    flows = [
        BusinessFlow(
            flow_id          = f["flow_id"],
            name             = f["name"],
            actor            = f["actor"],
            bounded_context  = f["bounded_context"],
            trigger          = f["trigger"],
            steps            = [FlowStep(**s) for s in f["steps"]],
            branches         = f["branches"],
            termination      = f["termination"],
            evidence_files   = f["evidence_files"],
            confidence       = f["confidence"],
            replaces_workflow= f.get("replaces_workflow"),
        )
        for f in data.get("flows", [])
    ]

    return BusinessFlowCollection(
        flows        = flows,
        total        = data.get("total", len(flows)),
        by_context   = data.get("by_context", {}),
        by_actor     = data.get("by_actor", {}),
        generated_at = data.get("generated_at", ""),
    )