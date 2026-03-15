"""
pipeline/stage67_diagrams.py — Automatic Diagram Generation

Generates three families of Mermaid diagrams from the structured data already
produced by earlier pipeline stages.  All generation is algorithmic — no LLM
calls, no external services, zero additional cost.

Diagram families
----------------
1. Sequence diagrams        (1 per BusinessFlow)
   sequenceDiagram — shows Actor ↔ Page ↔ Database message flow.
   Source: ctx.business_flows (FlowStep objects).
   Output: diagrams/seq_<flow_id>.mmd

2. Process flow diagrams    (1 per bounded context)
   flowchart LR — shows the happy-path steps and branch conditions for every
   flow within a bounded context, grouped into subgraphs.
   Source: ctx.business_flows grouped by bounded_context.
   Output: diagrams/flow_<context_slug>.mmd

3. Architecture diagram     (1 per run)
   graph TD — shows components as labelled boxes grouped by type, with
   integration-point edges between them.

   Source priority:
     [1] ctx.architecture_meta.json_path  (architecture.json from Stage 6.2)
         Uses components[], integration_points[], and technology_observations.
         This is the richest, LLM-validated source and is preferred.
     [2] G.graph["modules"] from the knowledge-graph pickle  (Stage 2)
         Used when Stage 6.2 was skipped or its output is absent.
     [3] ctx.domain_model.bounded_contexts + features  (Stage 4)
         Last-resort fallback when both above are unavailable.

   Output: diagrams/architecture.mmd

After generating .mmd files the stage injects ```mermaid fenced blocks into
the existing Markdown artefacts so they appear in the rendered documents:
  • architecture.mmd  → top of srs.md  (System Architecture section)
  • flow_*.mmd        → top of brd.md  (Process Flows section)
  • seq_*.mmd         → embedded in ac.md under each feature section

Resume behaviour
----------------
If stage67_diagrams is COMPLETED and diagrams/ directory is non-empty the
stage is skipped.

Output layout
-------------
outputs/run_<id>/
└── diagrams/
    ├── architecture.mmd
    ├── flow_authentication.mmd
    ├── flow_booking.mmd
    ├── seq_flow_001.mmd
    └── seq_flow_002.mmd

Mermaid rendering
-----------------
The .mmd files render natively in:
  GitHub (README / wiki), GitLab, Notion, Obsidian, VS Code (Mermaid extension),
  Confluence (Mermaid macro), MkDocs (mermaid plugin).

To render to PNG/SVG for standalone DOCX embedding install mermaid-cli:
  npm install -g @mermaid-js/mermaid-cli
  mmdc -i diagrams/architecture.mmd -o diagrams/architecture.png

Dependencies
------------
  Pure Python standard library — no third-party packages required.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from context import (
    BusinessFlow,
    BusinessFlowCollection,
    DomainModel,
    FlowStep,
    PipelineContext,
)

# ── Constants ──────────────────────────────────────────────────────────────────

DIAGRAMS_SUBDIR = "diagrams"

# Sequence diagram limits — diagrams with more participants become unreadable
SEQ_MAX_PARTICIPANTS = 6    # Actor + pages + DB; merge extras into "Other Pages"
SEQ_MAX_STEPS        = 20   # Long flows get truncated with a note

# Process flow limits
FLOW_MAX_NODES = 30         # Per bounded-context subgraph

# Architecture diagram limits
ARCH_MAX_NODES_PER_MODULE = 8   # Show top-N nodes per module to avoid clutter

# Mermaid label safe length — longer labels get truncated
LABEL_MAX_CHARS = 45

# Component type → Mermaid CSS class (for architecture diagram from Stage 6.2)
_COMP_TYPE_CLASS: dict[str, str] = {
    "Frontend":      "frontendNode",
    "Backend":       "backendNode",
    "Database":      "dbNode",
    "Service":       "serviceNode",
    "Middleware":    "middlewareNode",
    "External":      "externalNode",
    "Configuration": "configNode",
}


# ── Public Entry Point ─────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 6.7 entry point.  Generates all three diagram families and injects
    Mermaid blocks into the existing Markdown artefacts.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If required upstream stages have not been run.
    """
    diagrams_dir = ctx.output_path(DIAGRAMS_SUBDIR)

    # ── Resume check ──────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage67_diagrams"):
        existing = list(Path(diagrams_dir).glob("*.mmd")) if Path(diagrams_dir).exists() else []
        if existing:
            print(f"  [stage67] Already completed — {len(existing)} diagrams present.")
            return

    # ── Pre-flight ────────────────────────────────────────────────────────────
    if ctx.domain_model is None:
        raise RuntimeError("[stage67] ctx.domain_model is None — run Stage 4 first.")

    Path(diagrams_dir).mkdir(parents=True, exist_ok=True)

    generated: list[str] = []   # all .mmd paths created

    # ── 1. Architecture diagram ───────────────────────────────────────────────
    print("  [stage67] Generating architecture diagram ...")
    try:
        arch_mmd  = _build_architecture_diagram(ctx)
        arch_path = os.path.join(diagrams_dir, "architecture.mmd")
        Path(arch_path).write_text(arch_mmd, encoding="utf-8")
        generated.append(arch_path)

        # Report which source was used so the operator knows
        source = _architecture_source_label(ctx)
        print(f"  [stage67] ✓ architecture.mmd  (source: {source})")
    except Exception as e:
        print(f"  [stage67] ✗ architecture.mmd failed: {e}")

    # ── 2. Process flow diagrams (per bounded context) ────────────────────────
    flows = _get_flows(ctx)
    if flows:
        print(f"  [stage67] Generating process flow diagrams ({len(flows)} flows) ...")
        flow_mmds = _build_process_flow_diagrams(flows, ctx.domain_model)
        for slug, mmd_text in flow_mmds.items():
            path = os.path.join(diagrams_dir, f"flow_{slug}.mmd")
            Path(path).write_text(mmd_text, encoding="utf-8")
            generated.append(path)
            print(f"  [stage67] ✓ flow_{slug}.mmd")
    else:
        print("  [stage67] No business flows found — skipping process flow diagrams.")

    # ── 3. Sequence diagrams (per flow) ───────────────────────────────────────
    if flows:
        print("  [stage67] Generating sequence diagrams ...")
        seq_count = 0
        for flow in flows:
            mmd_text = _build_sequence_diagram(flow)
            path     = os.path.join(diagrams_dir, f"seq_{flow.flow_id}.mmd")
            Path(path).write_text(mmd_text, encoding="utf-8")
            generated.append(path)
            seq_count += 1
        print(f"  [stage67] ✓ {seq_count} sequence diagram(s)")

    # ── 4. Inject into existing Markdown artefacts ────────────────────────────
    print("  [stage67] Injecting diagrams into Markdown artefacts ...")
    _inject_into_markdown(ctx, diagrams_dir, flows or [])

    # ── Mark stage ────────────────────────────────────────────────────────────
    ctx.stage("stage67_diagrams").mark_completed(diagrams_dir)
    ctx.save()

    print(f"  [stage67] Complete — {len(generated)} diagram(s) written to {diagrams_dir}/")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_flows(ctx: PipelineContext) -> list[BusinessFlow]:
    """Return the list of BusinessFlow objects if stage 4.5 was run."""
    bfc = getattr(ctx, "business_flows", None)
    if bfc and bfc.flows:
        return bfc.flows
    return []


def _safe_label(text: str, max_chars: int = LABEL_MAX_CHARS) -> str:
    """
    Truncate and sanitise a string for use as a Mermaid node label.

    Mermaid chokes on double-quotes, angle brackets, and very long labels.
    We strip those and truncate to max_chars with an ellipsis.
    """
    text = re.sub(r'["\\[\\]<>{}|]', "", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars - 1] + "…"
    return text


def _node_id(text: str) -> str:
    """
    Convert arbitrary text to a valid Mermaid node ID (alphanumeric + underscore).
    """
    slug = re.sub(r"[^a-zA-Z0-9]", "_", text).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug or "node"


def _load_graph(ctx: PipelineContext) -> Any | None:
    """Load the knowledge graph pickle if it exists, else return None."""
    if ctx.graph_meta and ctx.graph_meta.graph_path:
        gpickle = ctx.graph_meta.graph_path
        if Path(gpickle).exists():
            try:
                with open(gpickle, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                pass
    # Fallback: search for gpickle in the graph stage subdir
    graph_dir = Path(ctx.output_path("code_graph.gpickle")).parent
    for candidate in graph_dir.glob("*.gpickle"):
        try:
            with open(candidate, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    return None


def _load_architecture_json(ctx: PipelineContext) -> dict[str, Any] | None:
    """
    Load the architecture.json produced by Stage 6.2.

    Returns the parsed dict if available and valid, otherwise None.
    Checks ctx.architecture_meta.json_path first, then falls back to
    looking for architecture.json directly in ctx.output_dir.
    """
    # Primary: use the path stored on ctx by Stage 6.2
    candidates: list[Path] = []

    if ctx.architecture_meta and ctx.architecture_meta.json_path:
        candidates.append(Path(ctx.architecture_meta.json_path))

    # Fallback: well-known location (routed via output_path)
    candidates.append(Path(ctx.output_path("architecture.json")))

    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # Sanity-check: must have at least components
                if isinstance(data.get("components"), list):
                    return data
            except (json.JSONDecodeError, OSError):
                continue

    return None


def _architecture_source_label(ctx: PipelineContext) -> str:
    """Return a human-readable label for which architecture source was used."""
    if _load_architecture_json(ctx) is not None:
        return "Stage 6.2 architecture.json"
    if _load_graph(ctx) is not None:
        return "Stage 2 knowledge graph"
    return "Stage 4 domain model (fallback)"


# ── 1. Architecture Diagram ────────────────────────────────────────────────────

def _build_architecture_diagram(ctx: PipelineContext) -> str:
    """
    Build a Mermaid graph TD architecture diagram.

    Source priority:
      1. architecture.json  (Stage 6.2) — richest, LLM-validated
      2. G.graph["modules"] (Stage 2)   — graph-derived
      3. domain_model       (Stage 4)   — last resort

    When using Stage 6.2 data, components are grouped by their `type` field
    into Mermaid subgraphs, with integration_point edges drawn between them.
    """
    arch_data = _load_architecture_json(ctx)

    if arch_data is not None:
        return _build_arch_from_stage62(arch_data)

    # Fallback to graph / domain model
    G = _load_graph(ctx)
    return _build_arch_from_graph_or_domain(ctx, G)


def _build_arch_from_stage62(arch_data: dict[str, Any]) -> str:
    """
    Build a Mermaid graph TD from Stage 6.2 architecture.json.

    Layout:
      • One subgraph per component type  (Frontend, Backend, Database, …)
      • Each component is a node inside its type subgraph
      • integration_points become labelled edges between component nodes

    Example output:
      graph TD
          subgraph Frontend["Frontend"]
              login_php["login.php — Entry point"]
          end
          subgraph Database["Database"]
              DB_mysql["MySQL — Relational data store"]
          end
          login_php -->|"reads"| DB_mysql
    """
    lines = ["graph TD"]

    components: list[dict[str, Any]] = arch_data.get("components", [])

    # ── Build component node map: name → node_id ───────────────────────────
    # Used later for edge lookup
    comp_node_map: dict[str, str] = {}   # component name → mermaid node id

    # Group components by type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for comp in components:
        comp_type = comp.get("type", "Other")
        by_type[comp_type].append(comp)

    # ── Emit one subgraph per type ─────────────────────────────────────────
    for comp_type, comps in sorted(by_type.items()):
        type_id = _node_id(comp_type)
        lines.append(f'    subgraph {type_id}["{comp_type}"]')

        for comp in comps:
            name  = comp.get("name", "?")
            desc  = comp.get("description", "")
            nid   = _node_id(name)
            label = _safe_label(f"{name} — {desc}" if desc else name, max_chars=50)

            # Use cylinder shape for Database nodes, box for everything else
            if comp_type == "Database":
                lines.append(f'        {nid}[("{label}")]')
            else:
                lines.append(f'        {nid}["{label}"]')

            comp_node_map[name] = nid

        lines.append("    end")
        lines.append("")

    # ── Emit integration point edges ───────────────────────────────────────
    integration_points: list[dict[str, Any]] = arch_data.get("integration_points", [])
    seen_edges: set[tuple[str, str]] = set()

    for ip in integration_points:
        ip_name = ip.get("name", "")
        ip_desc = ip.get("description", "")
        ip_type = ip.get("type", "")

        # Try to resolve source → target from the description text.
        # Stage 6.2 descriptions tend to follow patterns like:
        #   "login.php reads from MySQL"
        #   "rent.php writes to request table via dbcreation.php"
        src_id, dst_id = _resolve_integration_edge(ip_desc, ip_name, comp_node_map)

        if src_id and dst_id:
            edge_key = (src_id, dst_id)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edge_lbl = _safe_label(ip_type or ip_name, 30)
                lines.append(f'    {src_id} -->|"{edge_lbl}"| {dst_id}')

    lines.append("")

    # ── Tech stack note (if available) ────────────────────────────────────
    tech = arch_data.get("technology_observations", {})
    stack = tech.get("stack", [])
    if stack:
        stack_str = _safe_label(", ".join(stack[:5]), 60)
        lines.append(f'    techNote[/"Stack: {stack_str}"/]')
        lines.append("")

    # ── Styles ─────────────────────────────────────────────────────────────
    _append_arch_styles(lines, comp_node_map, by_type)

    return "\n".join(lines)


def _resolve_integration_edge(
    description: str,
    name: str,
    comp_node_map: dict[str, str],
) -> tuple[str | None, str | None]:
    """
    Attempt to extract a (source_node_id, dest_node_id) pair from an
    integration point description or name by searching for known component
    names within the text.

    Strategy: find the first two component names that appear in the text
    (in order).  The first is treated as source, second as destination.
    Returns (None, None) if fewer than two components can be resolved.
    """
    text = f"{name} {description}".lower()

    found: list[str] = []
    for comp_name, node_id in comp_node_map.items():
        if comp_name.lower() in text and node_id not in found:
            found.append(node_id)
        if len(found) == 2:
            break

    if len(found) >= 2:
        return found[0], found[1]
    return None, None


def _append_arch_styles(
    lines: list[str],
    comp_node_map: dict[str, str],
    by_type: dict[str, list[dict]],
) -> None:
    """Append classDef declarations and class assignments to the diagram lines."""
    # Define all component-type styles
    style_defs = {
        "frontendNode":    "fill:#dcfce7,stroke:#16a34a,color:#14532d",
        "backendNode":     "fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f",
        "dbNode":          "fill:#ede9fe,stroke:#7c3aed,color:#2e1065",
        "serviceNode":     "fill:#fef9c3,stroke:#ca8a04,color:#713f12",
        "middlewareNode":  "fill:#ffedd5,stroke:#ea580c,color:#431407",
        "externalNode":    "fill:#f1f5f9,stroke:#94a3b8,color:#1e293b",
        "configNode":      "fill:#fce7f3,stroke:#db2777,color:#500724",
    }
    for cls, style in style_defs.items():
        lines.append(f"    classDef {cls} {style}")

    # Assign each node to its class
    for comp_type, comps in by_type.items():
        css_class = _COMP_TYPE_CLASS.get(comp_type, "externalNode")
        nids = [_node_id(c.get("name", "?")) for c in comps]
        if nids:
            lines.append(f'    class {",".join(nids)} {css_class}')


# ── Legacy architecture builder (fallback when Stage 6.2 not available) ────────

def _build_arch_from_graph_or_domain(ctx: PipelineContext, G: Any | None) -> str:
    """
    Original architecture diagram builder using Stage 2 graph or Stage 4
    domain model.  Used only when architecture.json is not available.
    """
    dm    = ctx.domain_model
    lines = ["graph TD"]

    _ARCH_NODE_TYPES = frozenset({
        "page", "script", "db_table",
        "controller", "service", "model",
        "route", "http_endpoint",
    })

    def _is_noise_module(key: str, name: str) -> bool:
        lk = key.lower()
        ln = name.lower()
        return (
            lk.startswith("post_") or lk.startswith("get_") or
            lk.startswith("session_") or lk.startswith("cookie_") or
            lk.startswith("extcall") or lk.startswith("$") or
            ln.startswith("$ ") or ln.startswith("extcall:") or
            "extcall" in lk
        )

    modules: dict[str, dict[str, list[str]]] = {}

    if G is not None and G.graph.get("modules"):
        for mod_key, mod_info in G.graph["modules"].items():
            mod_name = mod_info.get("name", mod_key.title())
            if _is_noise_module(mod_key, mod_name):
                continue

            pages:       list[str] = []
            tables:      list[str] = []
            controllers: list[str] = []

            for nid in mod_info.get("nodes", []):
                if not G.has_node(nid):
                    continue
                ndata = G.nodes[nid]
                ntype = ndata.get("type", "")
                name  = ndata.get("name", nid)

                if ntype not in _ARCH_NODE_TYPES:
                    continue

                if ntype in ("page", "script", "route", "http_endpoint"):
                    pages.append(name)
                elif ntype == "db_table":
                    tables.append(name)
                elif ntype in ("controller", "service", "model"):
                    controllers.append(name)

            if not (pages or tables or controllers):
                continue

            modules[mod_key] = {
                "name":        mod_name,
                "pages":       sorted(pages)[:ARCH_MAX_NODES_PER_MODULE],
                "tables":      sorted(tables)[:ARCH_MAX_NODES_PER_MODULE],
                "controllers": sorted(controllers)[:ARCH_MAX_NODES_PER_MODULE],
            }
    else:
        # Domain model fallback
        clean_contexts = [
            c for c in (dm.bounded_contexts or [])
            if not _is_noise_module(_node_id(c.lower()), c)
            and not c.startswith("$")
            and not c.upper() == c
        ]

        for ctx_name in clean_contexts:
            slug   = _node_id(ctx_name.lower())
            pages:  list[str] = []
            tables: list[str] = []
            for feat in dm.features:
                if ctx_name.lower() in feat.get("name", "").lower() or \
                   any(ctx_name.lower() in p.lower() for p in feat.get("pages", [])):
                    pages.extend(feat.get("pages", []))
                    tables.extend(feat.get("tables", []))
            modules[slug] = {
                "name":        ctx_name,
                "pages":       sorted(set(pages))[:ARCH_MAX_NODES_PER_MODULE],
                "tables":      sorted(set(tables))[:ARCH_MAX_NODES_PER_MODULE],
                "controllers": [],
            }

        if not modules:
            all_pages:  list[str] = []
            all_tables: list[str] = []
            for feat in dm.features:
                all_pages.extend(
                    p for p in feat.get("pages", [])
                    if not p.startswith("$") and not p.startswith("EXTCALL:")
                )
                all_tables.extend(
                    t for t in feat.get("tables", [])
                    if not t.startswith("$") and t.upper() != t
                )
            modules["system"] = {
                "name":        dm.domain_name or "System",
                "pages":       sorted(set(all_pages))[:ARCH_MAX_NODES_PER_MODULE],
                "tables":      sorted(set(all_tables))[:ARCH_MAX_NODES_PER_MODULE],
                "controllers": [],
            }

    # ── Emit subgraphs ─────────────────────────────────────────────────────
    node_map: dict[tuple[str, str], str] = {}

    for mod_key, mod_info in modules.items():
        safe_mod = _node_id(mod_key)
        mod_name = _safe_label(mod_info["name"])
        lines.append(f'    subgraph {safe_mod}["{mod_name}"]')

        for page in mod_info["pages"]:
            nid = f"{safe_mod}_{_node_id(page)}"
            lbl = _safe_label(Path(page).name)
            lines.append(f'        {nid}["{lbl}"]')
            node_map[(mod_key, page)] = nid

        for ctrl in mod_info["controllers"]:
            nid = f"{safe_mod}_ctrl_{_node_id(ctrl)}"
            lbl = _safe_label(ctrl)
            lines.append(f'        {nid}["{lbl}"]')
            node_map[(mod_key, ctrl)] = nid

        for table in mod_info["tables"]:
            if not table or table.upper() in ("UNKNOWN", "NULL", "?"):
                continue
            nid = f"{safe_mod}_tbl_{_node_id(table)}"
            lbl = _safe_label(table)
            lines.append(f'        {nid}[("{lbl}")]')
            node_map[(mod_key, table)] = nid

        lines.append("    end")
        lines.append("")

    # ── Cross-module edges from graph ──────────────────────────────────────
    if G is not None:
        CROSS_EDGE_TYPES = {"calls", "sql_read", "sql_write", "redirects_to", "includes"}
        seen_edges: set[tuple[str, str, str]] = set()

        for src, dst, edata in G.edges(data=True):
            etype   = edata.get("edge_type", "")
            if etype not in CROSS_EDGE_TYPES:
                continue
            src_mod = G.nodes[src].get("module_id", "")
            dst_mod = G.nodes[dst].get("module_id", "")
            if not src_mod or not dst_mod or src_mod == dst_mod:
                continue

            src_name = G.nodes[src].get("name", src)
            dst_name = G.nodes[dst].get("name", dst)
            src_nid  = node_map.get((src_mod, src_name))
            dst_nid  = node_map.get((dst_mod, dst_name))

            if not src_nid or not dst_nid:
                continue

            edge_key = (src_nid, dst_nid, etype)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            lines.append(f'    {src_nid} -->|"{_edge_label(etype)}"| {dst_nid}')

    # ── Styles (legacy) ────────────────────────────────────────────────────
    lines.append("")
    lines.append("    classDef dbNode fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f")
    lines.append("    classDef pageNode fill:#dcfce7,stroke:#16a34a,color:#14532d")
    lines.append("    classDef ctrlNode fill:#fef9c3,stroke:#ca8a04,color:#713f12")

    db_nids   = [nid for (_, _n), nid in node_map.items() if "tbl_" in nid]
    page_nids = [nid for (_, _n), nid in node_map.items() if "tbl_" not in nid and "ctrl_" not in nid]
    ctrl_nids = [nid for (_, _n), nid in node_map.items() if "ctrl_" in nid]

    if db_nids:
        lines.append(f'    class {",".join(db_nids)} dbNode')
    if page_nids:
        lines.append(f'    class {",".join(page_nids)} pageNode')
    if ctrl_nids:
        lines.append(f'    class {",".join(ctrl_nids)} ctrlNode')

    return "\n".join(lines)


def _edge_label(edge_type: str) -> str:
    """Human-readable Mermaid edge label from internal edge_type string."""
    return {
        "sql_read":     "reads",
        "sql_write":    "writes",
        "calls":        "calls",
        "redirects_to": "redirects",
        "includes":     "includes",
    }.get(edge_type, edge_type)


# ── 2. Process Flow Diagrams ───────────────────────────────────────────────────

def _build_process_flow_diagrams(
    flows:  list[BusinessFlow],
    domain: DomainModel,
) -> dict[str, str]:
    """
    Build one Mermaid flowchart LR per bounded context.

    Each flow within the context becomes a subgraph.  Steps are connected
    in sequence; branches become diamond decision nodes.

    Returns dict of {context_slug: mermaid_text}.
    """
    by_context: dict[str, list[BusinessFlow]] = defaultdict(list)
    for flow in flows:
        ctx_slug = _node_id(flow.bounded_context.lower()) or "general"
        by_context[ctx_slug].append(flow)

    result: dict[str, str] = {}

    for ctx_slug, ctx_flows in by_context.items():
        lines   = ["flowchart LR"]
        node_ct = 0

        for flow in ctx_flows:
            f_slug = _node_id(flow.flow_id)
            f_name = _safe_label(flow.name)

            lines.append(f'    subgraph {f_slug}["{f_name}"]')
            lines.append(f'        direction LR')

            steps = flow.steps[:SEQ_MAX_STEPS]

            prev_id: str | None = None

            for step in steps:
                s_id    = f"{f_slug}_s{step.step_num}"
                auth_badge = "🔒 " if step.auth_required else ""
                db_badge   = "💾 " if step.db_ops        else ""
                label_txt  = _safe_label(f"{auth_badge}{db_badge}{step.step_num}. {step.action}")
                node_ct += 1

                lines.append(f'        {s_id}["{label_txt}"]')

                if prev_id:
                    method_lbl = f'|"{step.http_method}"|' if step.http_method else ""
                    lines.append(f'        {prev_id} -->{method_lbl} {s_id}')

                prev_id = s_id

            # Emit branch decision nodes
            for b_idx, branch in enumerate(flow.branches):
                condition = _safe_label(branch.get("condition", "?"), max_chars=35)
                alternate = branch.get("alternate", [])
                alt_label = _safe_label(", ".join(alternate) if alternate else "error")

                d_id  = f"{f_slug}_d{b_idx}"
                a_id  = f"{f_slug}_a{b_idx}"
                node_ct += 2

                lines.append(f'        {d_id}{{"{condition}?"}}')
                lines.append(f'        {a_id}["{alt_label}"]')
                if prev_id:
                    lines.append(f'        {prev_id} --> {d_id}')
                lines.append(f'        {d_id} -->|"No"| {a_id}')

            # Termination node
            term_id  = f"{f_slug}_end"
            term_lbl = _safe_label(flow.termination)
            lines.append(f'        {term_id}(("{term_lbl}"))')
            if prev_id:
                lines.append(f'        {prev_id} --> {term_id}')

            lines.append("    end")
            lines.append("")

            if node_ct >= FLOW_MAX_NODES:
                if len(ctx_flows) > ctx_flows.index(flow) + 1:
                    remaining = len(ctx_flows) - ctx_flows.index(flow) - 1
                    lines.append(
                        f'    note["... {remaining} more flow(s) omitted for readability"]'
                    )
                break

        lines.append("    classDef termNode fill:#bbf7d0,stroke:#16a34a,stroke-width:2px")
        lines.append("    classDef branchNode fill:#fef9c3,stroke:#ca8a04")
        lines.append("    classDef stepNode fill:#f1f5f9,stroke:#94a3b8")

        result[ctx_slug] = "\n".join(lines)

    return result


# ── 3. Sequence Diagrams ───────────────────────────────────────────────────────

def _build_sequence_diagram(flow: BusinessFlow) -> str:
    """
    Build a Mermaid sequenceDiagram for one BusinessFlow.

    Participants: Actor, each unique page (capped at SEQ_MAX_PARTICIPANTS-2),
    and "Database" (if any DB ops exist).  Extra pages are merged into
    "Other Pages" to keep the diagram readable.

    Each FlowStep becomes:
      Actor ->> Page: action (METHOD)
      Page  ->> DB:   op on table   (if db_ops)
      DB   -->> Page: result
      Page -->> Actor: output        (if outputs)

    Branch conditions appear as alt/else blocks.
    """
    lines: list[str] = ["sequenceDiagram"]

    actor_name = _safe_label(flow.actor, 30)
    has_db     = any(s.db_ops for s in flow.steps)

    seen_pages:     list[str] = []
    page_set:       set[str]  = set()
    overflow_pages: set[str]  = set()

    page_cap = SEQ_MAX_PARTICIPANTS - 1 - (1 if has_db else 0) - 1

    for step in flow.steps:
        page_short = Path(step.page).name
        if page_short not in page_set:
            if len(seen_pages) < page_cap:
                seen_pages.append(page_short)
                page_set.add(page_short)
            else:
                overflow_pages.add(page_short)

    use_overflow  = bool(overflow_pages)
    overflow_name = "OtherPages"

    # ── Declare participants ──────────────────────────────────────────────
    lines.append(f'    participant {_node_id(actor_name)} as {actor_name}')
    for page in seen_pages:
        lines.append(f'    participant {_node_id(page)} as {page}')
    if use_overflow:
        lines.append(f'    participant {overflow_name} as Other Pages')
    if has_db:
        lines.append(f'    participant DB as Database')
    lines.append("")

    # ── Emit steps ────────────────────────────────────────────────────────
    actor_id = _node_id(actor_name)
    steps    = flow.steps[:SEQ_MAX_STEPS]

    for step in steps:
        page_short = Path(step.page).name
        page_id    = _node_id(page_short) if page_short in page_set else overflow_name

        method_part = f" [{step.http_method}]" if step.http_method else ""
        action_lbl  = _safe_label(f"{step.action}{method_part}", 50)

        if step.auth_required and step.step_num == 1:
            lines.append(f'    Note over {actor_id},{page_id}: 🔒 Auth required')

        lines.append(f'    {actor_id} ->> {page_id}: {action_lbl}')

        if step.inputs:
            inputs_str = _safe_label(", ".join(step.inputs[:6]), 50)
            lines.append(f'    Note right of {page_id}: inputs: {inputs_str}')

        for db_op in step.db_ops[:2]:
            db_lbl = _safe_label(db_op, 45)
            lines.append(f'    {page_id} ->> DB: {db_lbl}')
            lines.append(f'    DB -->> {page_id}: result')

        if step.outputs:
            out_lbl = _safe_label(", ".join(step.outputs[:3]), 45)
            lines.append(f'    {page_id} -->> {actor_id}: {out_lbl}')

        lines.append("")

    # ── Branch conditions as alt blocks ───────────────────────────────────
    for branch in flow.branches[:3]:
        condition  = _safe_label(branch.get("condition", "condition fails"), 50)
        at_page    = Path(branch.get("at_page", "page")).name
        at_page_id = _node_id(at_page) if at_page in page_set else overflow_name
        alternates = branch.get("alternate", [])
        alt_lbl    = _safe_label(", ".join(alternates), 45) if alternates else "error handler"

        lines.append(f'    alt {condition}')
        lines.append(f'        {at_page_id} -->> {actor_id}: {alt_lbl}')
        lines.append(f'    else success')
        lines.append(f'        {at_page_id} -->> {actor_id}: continue')
        lines.append(f'    end')
        lines.append("")

    if len(flow.steps) > SEQ_MAX_STEPS:
        extra = len(flow.steps) - SEQ_MAX_STEPS
        lines.append(f'    Note over {actor_id}: ... {extra} more step(s) omitted')

    return "\n".join(lines)


# ── 4. Markdown Injection ──────────────────────────────────────────────────────

def _inject_into_markdown(
    ctx:          PipelineContext,
    diagrams_dir: str,
    flows:        list[BusinessFlow],
) -> None:
    """
    Inject Mermaid fenced blocks into the existing Markdown artefacts.

    Injection points:
      srs.md          → architecture diagram inserted after the first H1
      brd.md          → process flow diagrams inserted after first H1
      ac.md           → sequence diagrams inserted above each matching feature section
      user_stories.md → sequence diagrams inserted after each Epic heading

    If an artefact is missing the injection is silently skipped.
    """
    arch_path = os.path.join(diagrams_dir, "architecture.mmd")
    arch_mmd  = Path(arch_path).read_text(encoding="utf-8") if Path(arch_path).exists() else None

    # ── SRS: architecture diagram ──────────────────────────────────────────
    srs_path = ctx.output_path("srs.md")
    if Path(srs_path).exists() and arch_mmd:
        _prepend_diagram_to_section(
            md_path    = srs_path,
            section_h1 = True,
            diagram    = arch_mmd,
            caption    = "System Architecture — components and integration points",
        )

    # ── BRD: process flow diagrams ─────────────────────────────────────────
    brd_path = ctx.output_path("brd.md")
    if Path(brd_path).exists() and flows:
        flow_mmds = _collect_flow_mmds(diagrams_dir)
        if flow_mmds:
            _prepend_diagram_to_section(
                md_path    = brd_path,
                section_h1 = True,
                diagram    = "\n\n".join(
                    f"```mermaid\n{mmd}\n```" for mmd in flow_mmds.values()
                ),
                caption    = "Business Process Flows",
                raw_block  = True,
            )

    # ── AC: sequence diagrams per flow ────────────────────────────────────
    ac_path = ctx.output_path("ac.md")
    if Path(ac_path).exists() and flows:
        md = Path(ac_path).read_text(encoding="utf-8")
        for flow in flows:
            seq_path = os.path.join(diagrams_dir, f"seq_{flow.flow_id}.mmd")
            if not Path(seq_path).exists():
                continue
            seq_mmd   = Path(seq_path).read_text(encoding="utf-8")
            seq_block = _mermaid_block(seq_mmd, f"Sequence: {flow.name}")
            md = _insert_before_section(md, seq_block, flow.name, flow.bounded_context)
        Path(ac_path).write_text(md, encoding="utf-8")

    # ── User Stories: sequence diagrams per epic ───────────────────────────
    us_path = ctx.output_path("user_stories.md")
    if Path(us_path).exists() and flows:
        md = Path(us_path).read_text(encoding="utf-8")
        for flow in flows:
            seq_path = os.path.join(diagrams_dir, f"seq_{flow.flow_id}.mmd")
            if not Path(seq_path).exists():
                continue
            seq_mmd   = Path(seq_path).read_text(encoding="utf-8")
            seq_block = _mermaid_block(seq_mmd, f"Sequence: {flow.name}")
            md = _insert_before_section(md, seq_block, flow.bounded_context, flow.name)
        Path(us_path).write_text(md, encoding="utf-8")


def _mermaid_block(mmd_text: str, caption: str = "") -> str:
    """Wrap Mermaid text in a fenced code block with an optional caption."""
    cap_line = f"\n*{caption}*\n" if caption else ""
    return f"\n{cap_line}\n```mermaid\n{mmd_text}\n```\n"


def _prepend_diagram_to_section(
    md_path:    str,
    diagram:    str,
    caption:    str  = "",
    section_h1: bool = False,
    raw_block:  bool = False,
) -> None:
    """
    Insert a diagram block into an existing Markdown file.

    If section_h1 is True: insert immediately after the first H1 heading.
    Otherwise: insert at the very top of the file.
    raw_block: if True, `diagram` is already a full fenced block string.
    """
    md = Path(md_path).read_text(encoding="utf-8")

    if raw_block:
        block = f"\n## Diagrams\n\n*{caption}*\n\n{diagram}\n\n---\n"
    else:
        block = _mermaid_block(diagram, caption)
        block = f"\n## Diagrams\n{block}\n---\n"

    if section_h1:
        match = re.search(r'^#\s+.+$', md, re.MULTILINE)
        if match:
            insert_at = match.end()
            md = md[:insert_at] + "\n" + block + md[insert_at:]
        else:
            md = block + md
    else:
        md = block + md

    Path(md_path).write_text(md, encoding="utf-8")


def _insert_before_section(
    md:       str,
    block:    str,
    *keywords: str,
) -> str:
    """
    Insert `block` immediately before the first Markdown H2/H3 heading that
    contains any of the given keywords (case-insensitive).

    If no matching heading is found the block is appended at the end.
    """
    pattern   = re.compile(r'^(#{2,3}\s+.+)$', re.MULTILINE)
    kws_lower = [k.lower() for k in keywords if k]

    for match in pattern.finditer(md):
        heading_lower = match.group(1).lower()
        if any(kw in heading_lower for kw in kws_lower):
            pos = match.start()
            return md[:pos] + block + "\n" + md[pos:]

    return md + "\n" + block


def _collect_flow_mmds(diagrams_dir: str) -> dict[str, str]:
    """Read all flow_*.mmd files and return {slug: content}."""
    result = {}
    for path in sorted(Path(diagrams_dir).glob("flow_*.mmd")):
        slug = path.stem.removeprefix("flow_")
        result[slug] = path.read_text(encoding="utf-8")
    return result
