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
MAX_TOKENS     = 4096
FLOWS_FILE     = "business_flows.json"

# Graph traversal limits
MAX_DEPTH      = 8    # max hops from an entry-point node
MAX_PATHS      = 40   # cap on candidate paths before dedup
MAX_PATH_LEN   = 12   # max steps in a single flow

# BFS edge types to follow (in priority order)
FOLLOW_EDGES   = {"redirect", "handles", "entry_point", "calls", "defines"}
# Edge types that represent a meaningful "user moves to next page" transition
PAGE_EDGES     = {"redirect", "entry_point"}

# Minimum path length (in page-type nodes) to be worth reporting
MIN_PAGE_STEPS = 2


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

    print("  [stage45] Pass A–C: Graph traversal + path deduplication ...")
    raw_paths = _traverse_graph(G, ctx)
    print(f"  [stage45]   {len(raw_paths)} candidate path(s) found")

    print("  [stage45] Pass D–E: Stitching execution_paths onto graph paths ...")
    flow_skeletons = _stitch_execution_paths(raw_paths, ctx, G)
    print(f"  [stage45]   {len(flow_skeletons)} flow skeleton(s) built")

    print("  [stage45] Pass F: Grouping by bounded context ...")
    context_groups = _group_by_context(flow_skeletons, ctx)

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
    """
    import networkx as nx

    # Use the graph index built by Stage 2 Pass 14 when available
    idx = G.graph.get("index", {})
    entry_nodes: list[str] = list({
        *idx.get("http_endpoints", []),
        *idx.get("entry_points", []),
        # Also include route nodes so framework-routed apps are covered
        *idx.get("routes", []),
    })

    if not entry_nodes:
        # Fallback: any node tagged is_entry_point
        entry_nodes = [n for n, d in G.nodes(data=True)
                       if d.get("is_entry_point") or d.get("type") == "http_endpoint"]

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
                visited_from_start.add(dst)
                queue.append(path + [dst])
                extended = True

            if not extended and len(path) >= MIN_PAGE_STEPS:
                candidate_paths.append(path)

    # Pass C: deduplicate by file sequence fingerprint — keep longest per fingerprint
    def _file_fingerprint(path: list[str]) -> tuple[str, ...]:
        """Collapse a node-ID path to the sequence of distinct PHP files it touches."""
        seen: list[str] = []
        for n in path:
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
        _domain_contexts = set(ctx.domain_model.bounded_contexts or [])

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
                best = (ctx.domain_model.bounded_contexts or ["General"])[0]

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

    for context_name, indices in context_groups.items():
        group_skeletons = [skeletons[i] for i in indices]

        system = _build_llm_system_prompt(domain)
        user   = _build_llm_user_prompt(context_name, group_skeletons, domain)

        try:
            raw = call_llm(system, user, max_tokens=MAX_TOKENS,
                           label=f"stage45_{context_name[:20]}")
            enrichments = _parse_llm_response(raw, len(group_skeletons))
        except Exception as e:
            print(f"  [stage45] Warning: LLM call failed for '{context_name}': {e}")
            enrichments = [{}] * len(group_skeletons)

        # Merge enrichments onto skeletons
        for sk, enr in zip(group_skeletons, enrichments):
            flow_id = f"flow_{flow_counter:03d}"
            flow_counter += 1

            flows.append(BusinessFlow(
                flow_id          = flow_id,
                name             = enr.get("name") or _fallback_name(sk),
                actor            = enr.get("actor") or _infer_actor(sk, domain),
                bounded_context  = context_name,
                trigger          = enr.get("trigger") or _infer_trigger(sk),
                steps            = sk["steps"],
                branches         = sk["branches"],
                termination      = enr.get("termination") or _infer_termination(sk),
                evidence_files   = sk["evidence_files"],
                confidence       = _final_confidence(sk["raw_confidence"], bool(enr)),
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
