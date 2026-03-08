"""
pipeline/stage2_graph.py — Knowledge Graph Builder

Converts the CodeMap produced by Stage 1 into a directed NetworkX graph,
then saves three artefacts to the run output directory:

    code_graph.gpickle   — NetworkX DiGraph for later pipeline stages
    code_graph.json      — Human-readable nodes + edges (mirrors old format)
    code_graph.png       — Matplotlib visualisation with colour-coded node types

Node types
----------
    page            — PHP file that mixes HTML and logic (entry point)
    function        — Defined PHP function or class method
    db_table        — Database table touched by SQL
    include_file    — File pulled in via include/require
    redirect        — header("Location: ...") target URL
    script          — Non-page PHP file (pure logic, no HTML)
    route           — Registered HTTP route (GET /path, POST /api/...)
    controller      — OOP controller class
    model           — OOP model / entity class
    service         — Service, repository, or other OOP class
    superglobal     — $_POST/$_GET/$_SESSION key access
    form            — HTML <form> element (file + action)
    http_endpoint   — Detected HTTP entry point (page/API/form handler)
    class_dep       — External class dependency (injected service / Facade)
    external_function — PHP built-in or extension function called from user code

Edge types
----------
    calls           — function_or_page  → function          (function call)
    includes        — file              → include_file       (include/require)
    sql_read        — function_or_page  → db_table           (SELECT)
    sql_write       — function_or_page  → db_table           (INSERT/UPDATE/DELETE/REPLACE)
    sql_ddl         — function_or_page  → db_table           (CREATE/DROP/ALTER/TRUNCATE)
    redirects_to    — function_or_page  → redirect           (header Location)
    defines         — script/page/class → function/class     (where it is defined)
    handles         — route             → function/method    (route → controller action)
    has_form        — page              → form               (page contains form)
    submits_to      — form              → page/endpoint      (form action target)
    reads_input     — function_or_page  → superglobal        ($_POST read)
    form_field_of   — superglobal       → form               (links $_POST[x] to <input name=x>)
    depends_on      — class             → class_dep          (constructor DI / Facade usage)
    method_of       — class             → function           (method belongs to class)
    inherits        — class             → class              (extends)
    implements      — class             → interface          (implements)
    entry_point     — http_endpoint     → page/function      (entry point owns handler)

Resume behaviour
----------------
If stage2_graph is already COMPLETED and all three output files exist, the
stage is skipped and graph_meta is reloaded from code_graph.json.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from context import CodeMap, GraphMeta, PipelineContext

# ── Optional visualisation deps (gracefully absent) ───────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — safe for servers
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Node-type colour palette (used in PNG) ────────────────────────────────────
NODE_COLORS: dict[str, str] = {
    "page":              "#4A90D9",   # blue
    "function":          "#7ED321",   # green
    "db_table":          "#F5A623",   # orange
    "include_file":      "#9B9B9B",   # grey
    "redirect":          "#D0021B",   # red
    "script":            "#B8E0F7",   # light blue
    "superglobal":       "#BD10E0",   # purple
    "route":             "#E91E8C",   # pink
    "controller":        "#26C6DA",   # cyan
    "model":             "#66BB6A",   # mid-green
    "service":           "#FFA726",   # amber
    "form":              "#AB47BC",   # violet
    "http_endpoint":     "#EF5350",   # bright red
    "class_dep":         "#78909C",   # blue-grey
    "external_function": "#546E7A",   # dark blue-grey (visually distinct but muted)
    "unknown":           "#CCCCCC",   # fallback
}

# ── PHP built-ins that are too common/noisy to add as graph nodes ─────────────
# Security-relevant and I/O functions ARE kept (password_verify, curl_exec, etc.)
# Pure data-manipulation functions (strlen, array_map, etc.) are skipped.
_PHP_NOISE_BUILTINS: frozenset[str] = frozenset({
    # String manipulation
    "strlen","strpos","strrpos","substr","str_replace","str_contains",
    "str_starts_with","str_ends_with","str_split","strtolower","strtoupper",
    "trim","ltrim","rtrim","explode","implode","sprintf","printf","number_format",
    "nl2br","strip_tags","htmlspecialchars","htmlspecialchars_decode","addslashes",
    "stripslashes","ucfirst","ucwords","str_pad","str_repeat","wordwrap","chunk_split",
    "md5","sha1","crc32","ord","chr",
    # Array
    "count","array_map","array_filter","array_merge","array_push","array_pop",
    "array_shift","array_unshift","array_keys","array_values","array_unique",
    "array_flip","array_reverse","array_slice","array_splice","array_search",
    "array_combine","array_diff","array_intersect","array_chunk","array_column",
    "in_array","isset","empty","unset","sort","rsort","asort","arsort","ksort",
    "krsort","usort","uasort","uksort","compact","extract","list","range",
    # Math / type
    "intval","floatval","strval","boolval","settype","gettype","is_array",
    "is_string","is_int","is_float","is_bool","is_null","is_numeric","is_object",
    "is_callable","abs","ceil","floor","round","max","min","pow","sqrt","rand",
    "mt_rand","pi",
    # Output / control
    "echo","print","var_dump","print_r","var_export","ob_start","ob_end_clean",
    "ob_get_clean","exit","die",
    # Date (common, low signal)
    "date","time","mktime","strtotime","microtime",
    # Misc low-signal
    "defined","define","constant","class_exists","method_exists","function_exists",
    "property_exists","interface_exists","get_class","get_object_vars",
    "call_user_func","call_user_func_array","func_get_args","compact","extract",
})

# ── SQL operation → edge type mapping ────────────────────────────────────────
SQL_EDGE_TYPE: dict[str, str] = {
    "SELECT":   "sql_read",
    "INSERT":   "sql_write",
    "UPDATE":   "sql_write",
    "DELETE":   "sql_write",
    "REPLACE":  "sql_write",
    "CREATE":   "sql_ddl",
    "DROP":     "sql_ddl",
    "ALTER":    "sql_ddl",
    "TRUNCATE": "sql_ddl",
}


# ─── Public Entry Point ───────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 2 entry point called by run_pipeline.py.

    Reads ctx.code_map, builds a DiGraph, writes .gpickle / .json / .png,
    and populates ctx.graph_meta.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If code_map is missing or graph construction fails.
    """
    gpickle_path = ctx.output_path("code_graph.gpickle")
    json_path    = ctx.output_path("code_graph.json")
    png_path     = ctx.output_path("code_graph.png")

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage2_graph") and Path(json_path).exists():
        print("  [stage2] Resuming — loading existing code_graph.json")
        ctx.graph_meta = _load_graph_meta(json_path)
        return

    if ctx.code_map is None:
        raise RuntimeError(
            "[stage2] ctx.code_map is None — Stage 1 must run before Stage 2."
        )

    print(f"  [stage2] Building knowledge graph from CodeMap …")

    # ── Build graph ───────────────────────────────────────────────────────────
    G = _build_graph(ctx.code_map)

    print(
        f"  [stage2] Graph built — "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    # ── Save .gpickle ─────────────────────────────────────────────────────────
    _save_gpickle(G, gpickle_path)

    # ── Save .json ────────────────────────────────────────────────────────────
    _save_json(G, json_path)

    # ── Save .png ─────────────────────────────────────────────────────────────
    if HAS_MATPLOTLIB:
        _save_png(G, png_path)
        print(f"  [stage2] Visualisation saved → {png_path}")
    else:
        print("  [stage2] matplotlib not installed — skipping PNG (pip install matplotlib)")

    # ── Populate GraphMeta ────────────────────────────────────────────────────
    node_types = list({d["type"]      for _, d    in G.nodes(data=True) if "type"      in d})
    edge_types = list({d["edge_type"] for _, _, d in G.edges(data=True) if "edge_type" in d})

    ctx.graph_meta = GraphMeta(
        graph_path  = gpickle_path,
        node_count  = G.number_of_nodes(),
        edge_count  = G.number_of_edges(),
        node_types  = sorted(node_types),
        edge_types  = sorted(edge_types),
    )

    ctx.stage("stage2_graph").mark_completed(json_path)
    ctx.save()

    print(
        f"  [stage2] Done — node types: {sorted(node_types)}, "
        f"edge types: {sorted(edge_types)}"
    )


# ─── Graph Construction ───────────────────────────────────────────────────────

def _build_graph(code_map: CodeMap) -> nx.DiGraph:
    """
    Build and return the full directed knowledge graph from a CodeMap.

    Passes:
        1.  Register all PHP source files as page/script nodes
        2.  Register all defined functions as function nodes + defines edges
        3.  OOP class/controller/model nodes + defines/inherits/implements edges
        4.  Include/require edges
        5.  SQL edges (read / write / ddl)
        6.  Redirect edges
        7.  Route nodes + route→handler (handles) edges         [NEW: connects routes]
        8.  Superglobal reads_input edges
        9.  Call graph edges from code_map.call_graph           [NEW: fn→fn calls]
        10. Form nodes + has_form / submits_to edges            [NEW: form linkage]
        11. Superglobal → form form_field_of edges              [NEW: $_POST ↔ <input>]
        12. Service dependency edges (DI / Facades)             [NEW: class→dep]
        13. HTTP endpoint nodes + entry_point edges             [NEW: entry-point tagging]
        14. API endpoint enrichment — multi-signal is_api inference [NEW: path/mw/file/callee]
        15. Module clustering — dir+namespace+Louvain hybrid     [NEW: module_id on every node]

    Args:
        code_map: Populated CodeMap from Stage 1.

    Returns:
        nx.DiGraph with typed nodes and edges.
    """
    G = nx.DiGraph()

    html_page_set = set(code_map.html_pages)

    # ── Pass 1: source file nodes ─────────────────────────────────────────────
    all_source_files: set[str] = set()
    for fn in code_map.functions:
        all_source_files.add(fn["file"])
    for inc in code_map.includes:
        all_source_files.add(inc["file"])
    for sql in code_map.sql_queries:
        all_source_files.add(sql["file"])
    for redir in code_map.redirects:
        all_source_files.add(redir["file"])
    for cls in code_map.classes + code_map.controllers + code_map.models + code_map.services:
        all_source_files.add(cls["file"])

    for filepath in all_source_files:
        node_type = "page" if filepath in html_page_set else "script"
        _add_node(G, filepath, node_type=node_type, file=filepath)

    for filepath in html_page_set:
        _add_node(G, filepath, node_type="page", file=filepath)

    # ── Pass 2: function definition nodes + defines edges ─────────────────────
    defined_functions: dict[str, str] = {}   # fn_name → file
    # Also track class→method for call resolution in Pass 9
    class_methods: dict[str, str] = {}       # "ClassName::method" → node_id
    # Reverse index: bare method name → list of node_ids (many classes can share a name)
    # Replaces the O(N) endswith scan in Pass 9 with an O(1) dict lookup.
    method_name_index: dict[str, list[str]] = {}

    for fn in code_map.functions:
        fn_id   = _function_node_id(fn["name"], fn["file"])
        fn_name = fn["name"]
        defined_functions[fn_name] = fn["file"]

        _add_node(
            G, fn_id,
            node_type   = "function",
            name        = fn_name,
            file        = fn["file"],
            line        = fn.get("line"),
            params      = [p["name"] for p in fn.get("params", [])],
            return_type = fn.get("return_type"),
            docblock    = fn.get("docblock"),
        )
        _add_edge(G, src=fn["file"], dst=fn_id, edge_type="defines",
                  file=fn["file"], line=fn.get("line"))

    # ── Pass 3: OOP class nodes ───────────────────────────────────────────────
    for bucket, cls_type in [
        (code_map.controllers, "controller"),
        (code_map.models,      "model"),
        (code_map.services,    "service"),
        (code_map.classes,     "class"),
    ]:
        for cls in bucket:
            cls_id = cls.get("fqn") or cls["name"]
            _add_node(
                G, cls_id,
                node_type  = cls_type,
                name       = cls["name"],
                file       = cls["file"],
                line       = cls.get("line"),
                extends    = cls.get("extends"),
                implements = cls.get("implements", []),
            )
            _add_edge(G, src=cls["file"], dst=cls_id, edge_type="defines",
                      file=cls["file"])

            # inherits edge
            if cls.get("extends"):
                parent_id = cls["extends"]
                _add_node(G, parent_id, node_type="class", name=parent_id)
                _add_edge(G, src=cls_id, dst=parent_id, edge_type="inherits",
                          file=cls["file"])

            # implements edges
            for iface in cls.get("implements", []):
                _add_node(G, iface, node_type="class", name=iface)
                _add_edge(G, src=cls_id, dst=iface, edge_type="implements",
                          file=cls["file"])

            # Method nodes
            for method in cls.get("methods", []):
                method_id = f"{cls_id}::{method['name']}"
                _add_node(
                    G, method_id,
                    node_type   = "function",
                    name        = method["name"],
                    file        = cls["file"],
                    line        = method.get("line"),
                    visibility  = method.get("visibility"),
                    return_type = method.get("return_type"),
                    parent      = cls_id,
                )
                _add_edge(G, src=cls_id, dst=method_id, edge_type="method_of",
                          file=cls["file"])

                # Register for call resolution: "ClassName::methodName" → node_id
                class_methods[f"{cls['name']}::{method['name']}"] = method_id
                # Reverse index: method name → [node_ids] for O(1) Pass 9 lookup
                method_name_index.setdefault(method["name"], []).append(method_id)
                # Also register without class for same-class $this-> calls
                defined_functions[method["name"]] = cls["file"]

    # ── Pass 4: include / require edges ───────────────────────────────────────
    for inc in code_map.includes:
        source_file  = inc["file"]
        target_clean = inc["target"].lstrip("./").lstrip("/")
        target_node_type = "page" if target_clean in html_page_set else "include_file"
        _add_node(G, target_clean, node_type=target_node_type, file=target_clean)

        caller   = inc.get("caller", "GLOBAL_SCRIPT")
        edge_src = edge_src = _resolve_caller(caller, source_file, defined_functions, G)
        _add_edge(G, src=edge_src, dst=target_clean, edge_type="includes",
                  inc_type=inc["type"], file=source_file, line=inc.get("line"))

    # ── Pass 5: SQL edges ─────────────────────────────────────────────────────
    sql_seen: set[tuple] = set()
    for sql in code_map.sql_queries:
        table     = _table_node_id(sql.get("table", "UNKNOWN"))
        operation = sql.get("operation", "SELECT").upper()
        caller    = sql.get("caller", "GLOBAL_SCRIPT")
        src_file  = sql["file"]
        edge_type = SQL_EDGE_TYPE.get(operation, "sql_read")

        _add_node(G, table, node_type="db_table", name=sql.get("table", "UNKNOWN"))
        edge_src  = _resolve_caller(caller, src_file, defined_functions, G)
        dedup_key = (edge_src, table, edge_type)
        if dedup_key not in sql_seen:
            sql_seen.add(dedup_key)
            _add_edge(G, src=edge_src, dst=table, edge_type=edge_type,
                      operation=operation, file=src_file, line=sql.get("line"))

    # ── Pass 6: redirect edges ────────────────────────────────────────────────
    for redir in code_map.redirects:
        target        = redir["target"]
        caller        = redir.get("caller", "GLOBAL_SCRIPT")
        src_file      = redir["file"]
        redir_node_id = f"REDIRECT:{target}"
        _add_node(G, redir_node_id, node_type="redirect", name=target, url=target)
        edge_src = _resolve_caller(caller, src_file, defined_functions, G)
        _add_edge(G, src=edge_src, dst=redir_node_id, edge_type="redirects_to",
                  file=src_file, line=redir.get("line"))

    # ── Pass 7: route nodes + route→handler edges ─────────────────────────────
    # NEW: connects each route to its concrete controller method node
    for route in code_map.routes:
        handler  = route.get("handler")
        src_file = route.get("file", "routes")
        path     = route.get("path", "")
        method   = route.get("method", "ANY")
        prefix   = route.get("prefix", "")
        mw       = route.get("middleware")

        full_path     = f"{prefix}{path}" if prefix else path
        route_node_id = f"ROUTE:{method}:{full_path}"

        _add_node(
            G, route_node_id,
            node_type  = "route",
            name       = f"{method} {full_path}",
            method     = method,
            path       = full_path,
            handler    = handler,
            middleware = mw,
            file       = src_file,
            line       = route.get("line"),
        )
        _add_node(G, src_file, node_type="script", file=src_file)
        _add_edge(G, src=src_file, dst=route_node_id, edge_type="defines",
                  file=src_file, line=route.get("line"))

        # Connect route → handler method/function
        if handler and handler != "(closure)":
            handler_node = _resolve_route_handler(
                handler, class_methods, method_name_index, defined_functions, G
            )
            if handler_node:
                _add_edge(G, src=route_node_id, dst=handler_node,
                          edge_type="handles", file=src_file, line=route.get("line"))

    # ── Pass 8: superglobal reads ──────────────────────────────────────────────
    sg_seen: set[tuple] = set()
    for sg in code_map.superglobals:
        var      = sg.get("var", "$_POST")
        key      = sg.get("key")
        caller   = sg.get("caller", "GLOBAL_SCRIPT")
        src_file = sg["file"]

        sg_node_id = f"{var}[{key}]" if key else var
        _add_node(G, sg_node_id, node_type="superglobal", name=sg_node_id,
                  var=var, key=key)
        edge_src  = _resolve_caller(caller, src_file, defined_functions, G)
        dedup_key = (edge_src, sg_node_id)
        if dedup_key not in sg_seen:
            sg_seen.add(dedup_key)
            _add_edge(G, src=edge_src, dst=sg_node_id, edge_type="reads_input",
                      file=src_file, line=sg.get("line"))

    # ── Pass 9: function call graph edges ─────────────────────────────────────
    # Wires code_map.call_graph edges.  Three resolution tiers:
    #   1. User-defined function/method in the graph         → calls edge
    #   2. Unknown callee not in noise list                  → EXTCALL: node + calls edge
    #   3. Noise built-in (strlen, array_map, etc.)          → silently skipped
    call_seen: set[tuple] = set()
    for edge in getattr(code_map, "call_graph", []):
        caller_name = edge.get("caller", "GLOBAL_SCRIPT")
        callee_name = edge.get("callee", "")
        src_file    = edge.get("file", "")
        if not callee_name:
            continue

        caller_node = _resolve_caller(caller_name, src_file, defined_functions, G)

        # ── Tier 1: resolve to an existing user-defined node ─────────────────
        callee_node = None
        callee_same = f"{src_file}::{callee_name}"
        if G.has_node(callee_same):
            callee_node = callee_same
        elif callee_name in defined_functions:
            callee_node = f"{defined_functions[callee_name]}::{callee_name}"
        elif callee_name in method_name_index:
            # O(1) reverse lookup replaces the O(N) endswith scan over class_methods.
            # If multiple classes define the same method name, prefer the one whose
            # file matches the call site; otherwise take the first registered.
            candidates = method_name_index[callee_name]
            same_file  = [n for n in candidates if G.nodes[n].get("file") == src_file]
            callee_node = same_file[0] if same_file else candidates[0]

        # ── Tier 2: external / built-in function ─────────────────────────────
        if callee_node is None:
            if callee_name in _PHP_NOISE_BUILTINS:
                continue  # skip noisy low-signal built-ins

            # Security/IO/interesting external call — add as EXTCALL node
            callee_node = f"EXTCALL:{callee_name}"
            _add_node(
                G, callee_node,
                node_type = "external_function",
                name      = callee_name,
            )

        dedup_key = (caller_node, callee_node)
        if dedup_key not in call_seen:
            call_seen.add(dedup_key)
            _add_edge(G, src=caller_node, dst=callee_node,
                      edge_type="calls", file=src_file, line=edge.get("line"))

    # ── Pass 10: form nodes + has_form / submits_to edges ─────────────────────
    # NEW: creates form nodes and links pages → forms → target pages
    for form in getattr(code_map, "form_fields", []):
        src_file = form.get("file", "")
        action   = form.get("action", "")
        method   = form.get("method", "POST")
        fields   = [f.get("name") for f in form.get("fields", []) if f.get("name")]

        # Form node ID: file + action (distinguishes multiple forms per file)
        form_node_id = f"FORM:{src_file}:{action or '(self)'}"
        _add_node(
            G, form_node_id,
            node_type = "form",
            name      = f"Form → {action or '(self)'}",
            file      = src_file,
            action    = action,
            method    = method,
            fields    = fields,
        )

        # page → form
        page_node = src_file
        _add_node(G, page_node, node_type="page", file=src_file)
        _add_edge(G, src=page_node, dst=form_node_id, edge_type="has_form",
                  file=src_file)

        # form → action target (submits_to)
        if action:
            target_clean = action.lstrip("./").lstrip("/")
            if target_clean:
                target_type = "page" if target_clean in html_page_set else "script"
                _add_node(G, target_clean, node_type=target_type, file=target_clean)
                _add_edge(G, src=form_node_id, dst=target_clean,
                          edge_type="submits_to", method=method, file=src_file)

    # ── Pass 11: superglobal → form (form_field_of) ───────────────────────────
    # NEW: links $_POST[email] to the form that has <input name="email">
    # Build a lookup: file → {field_name → form_node_id}
    file_form_fields: dict[str, dict[str, str]] = defaultdict(dict)
    for form in getattr(code_map, "form_fields", []):
        src_file     = form.get("file", "")
        action       = form.get("action", "")
        form_node_id = f"FORM:{src_file}:{action or '(self)'}"
        for field in form.get("fields", []):
            fname = field.get("name")
            if fname:
                file_form_fields[src_file][fname] = form_node_id
        # Also index by action target file (POST handler reads $_POST)
        if action:
            target_clean = action.lstrip("./").lstrip("/")
            for field in form.get("fields", []):
                fname = field.get("name")
                if fname:
                    file_form_fields[target_clean][fname] = form_node_id

    for sg in code_map.superglobals:
        var      = sg.get("var", "")
        key      = sg.get("key")
        src_file = sg["file"]
        if var not in ("$_POST", "$_GET", "$_REQUEST") or not key:
            continue

        sg_node_id = f"{var}[{key}]"
        # Find matching form
        form_node_id = (
            file_form_fields.get(src_file, {}).get(key)
        )
        if form_node_id and G.has_node(form_node_id) and G.has_node(sg_node_id):
            _add_edge(G, src=sg_node_id, dst=form_node_id,
                      edge_type="form_field_of", file=src_file)

    # ── Pass 12: service dependency edges ────────────────────────────────────
    # NEW: class→dep edges from constructor injection + Facades
    dep_seen: set[tuple] = set()
    for dep in getattr(code_map, "service_deps", []):
        cls_name   = dep.get("class", "")
        dep_class  = dep.get("dep_class", "")
        dep_var    = dep.get("dep_var")
        src_file   = dep.get("file", "")
        if not cls_name or not dep_class:
            continue

        # Find the class node (try FQN then simple name)
        cls_node = cls_name
        if not G.has_node(cls_node):
            # Try to find by searching existing nodes
            matches = [n for n in G.nodes() if G.nodes[n].get("name") == cls_name]
            cls_node = matches[0] if matches else cls_name
            _add_node(G, cls_node, node_type="class", name=cls_name, file=src_file)

        # Dependency node
        dep_node_id = f"DEP:{dep_class}"
        _add_node(G, dep_node_id, node_type="class_dep", name=dep_class)

        dedup_key = (cls_node, dep_node_id)
        if dedup_key not in dep_seen:
            dep_seen.add(dedup_key)
            _add_edge(G, src=cls_node, dst=dep_node_id, edge_type="depends_on",
                      dep_var=dep_var, file=src_file)

    # ── Pass 13: HTTP entry-point nodes ───────────────────────────────────────
    # NEW: creates http_endpoint nodes and links them to the page/method they tag
    for ep in getattr(code_map, "http_endpoints", []):
        src_file = ep.get("file", "")
        handler  = ep.get("handler")      # e.g. "AuthController::login" or None
        ep_type  = ep.get("type", "page") # page / api / form_handler / form_processor
        accepts  = ep.get("accepts", [])
        produces = ep.get("produces", "html")

        ep_node_id = f"EP:{handler or src_file}"
        _add_node(
            G, ep_node_id,
            node_type    = "http_endpoint",
            name         = handler or Path(src_file).name,
            ep_type      = ep_type,
            accepts      = accepts,
            produces     = produces,
            file         = src_file,
            line         = ep.get("line"),
            is_api       = (produces == "json"),
            is_entry     = True,
        )

        # Link endpoint → handler node or file
        target_node = None
        if handler and "::" in handler:
            # Try class_methods lookup first, then construct expected ID
            target_node = class_methods.get(handler)
            if not target_node:
                # Handler is "ClassName::method" — try to find the method node
                parts = handler.split("::", 1)
                possible = [n for n in G.nodes()
                            if G.nodes[n].get("name") == parts[1]
                            and G.nodes[n].get("parent","").endswith(parts[0])]
                target_node = possible[0] if possible else None
        if not target_node:
            target_node = src_file
            _add_node(G, src_file, node_type="page" if src_file in html_page_set else "script",
                      file=src_file)

        _add_edge(G, src=ep_node_id, dst=target_node, edge_type="entry_point",
                  ep_type=ep_type, file=src_file)

        # Tag the target node with entry-point metadata
        if G.has_node(target_node):
            G.nodes[target_node]["is_entry_point"] = True
            G.nodes[target_node]["ep_type"]        = ep_type
            G.nodes[target_node]["accepts"]        = accepts
            G.nodes[target_node]["produces"]       = produces

    # ── Pass 14: API endpoint enrichment (multi-signal inference) ─────────────
    # Runs after the full graph is built so it can inspect reachable EXTCALL nodes.
    _enrich_api_endpoints(G, code_map)

    # ── Build graph-level index for O(1) type lookups by downstream stages ────
    # Replaces repeated G.nodes(data=True) scans in Stage 3, 4, and 5.
    # Access via: G.graph["index"]["routes"], G.graph["index"]["api_endpoints"], etc.
    _build_index(G)

    # ── Pass 15: Module clustering (three-signal hybrid) ──────────────────────
    # Annotates every node with module_id / module_name / module_confidence.
    # Adds G.graph["modules"] summary and G.graph["index"]["by_module"].
    _cluster_modules(G)

    return G



# ─── Module Clustering (Pass 15) ─────────────────────────────────────────────

# Node types that represent "code" — used for the Louvain subgraph.
# Noise types (superglobal, redirect, external_function, class_dep) are excluded.
_CODE_NODE_TYPES: frozenset[str] = frozenset({
    "function", "controller", "model", "service",
    "page", "script", "route", "form", "http_endpoint",
})

# Directory segments that carry no module meaning and should be skipped
_DIR_SKIP: frozenset[str] = frozenset({
    "app", "src", "lib", "php", "public", "www", "htdocs",
    "Http", "http", "pipeline", "vendor", "node_modules",
})

# Louvain modularity threshold:
#   Q < LOW  → graph is flat, use directory signal only
#   Q < MID  → weak structure, annotate with low confidence
#   Q ≥ MID  → meaningful communities, merge with directory signal
_Q_LOW = 0.10
_Q_MID = 0.30


def _cluster_modules(G: nx.DiGraph) -> None:
    """
    Pass 15 — three-signal hybrid module clustering.

    Combines directory path, namespace prefix, and Louvain graph communities
    to assign every node a logical module (bounded context).

    Algorithm
    ---------
    Step A  Extract directory-based groups from each node's file path.
            Uses depth-adaptive scanning to find the first meaningful
            directory segment not in _DIR_SKIP.

    Step B  Extract namespace groups from node FQN where available.
            Namespace prefix (first non-vendor segment) refines Step A.

    Step C  Build an undirected subgraph of CODE_NODE_TYPES only (excludes
            superglobals, redirects, EXTCALL nodes — they are noise for
            community detection purposes).

    Step D  Run Louvain on the code subgraph.  If Q < _Q_LOW (flat app with
            no modular structure) skip Louvain entirely and use Step A/B only.

    Step E  For each Louvain community, find its plurality directory group
            and merge.  This fixes Louvain's tendency to over-split when
            cross-controller edges are sparse.

    Step F  Legacy flat-file fallback: if all nodes land in a single "root"
            group (no meaningful dirs), apply keyword clustering on file stems
            using domain-neutral signal words (auth, booking, payment, …).

    Step G  Annotate every node in G with:
                module_id         slug, e.g. "auth"
                module_name       title-cased, e.g. "Authentication"
                module_confidence float 0.0–1.0

    Step H  Populate G.graph["modules"] summary dict and extend
            G.graph["index"]["by_module"].

    Downstream consumers
    --------------------
    Stage 3  Embedding chunks include module_id → better semantic retrieval
    Stage 4  domain_model.bounded_contexts pre-seeded from detected modules
    Stage 4.5 _group_by_context uses module_id instead of filename keywords
    Stage 5  BRD/SRS sections map to real modules with node-count evidence
    """
    # ── Step A: Directory groups ──────────────────────────────────────────────
    dir_groups: dict[str, list[str]] = defaultdict(list)   # dir_key → [node_ids]

    def _dir_key(node_id: str, data: dict) -> str:
        """
        Derive a module key from the node's file path.
        Walks path parts from deepest to shallowest, returning the first
        segment that is not in _DIR_SKIP and is not the filename itself.
        Falls back to "root" for top-level flat files.
        """
        file_path = data.get("file", "") or node_id
        parts = [p for p in Path(file_path).parts
                 if p not in (".", "..", "/") and not p.endswith(".php")]
        # Walk from deepest directory inward, skip noise segments
        for part in reversed(parts):
            if part not in _DIR_SKIP and len(part) > 2:
                return part.lower()
        return "root"

    for node_id, data in G.nodes(data=True):
        key = _dir_key(node_id, data)
        dir_groups[key].append(node_id)

    # ── Step B: Namespace refinement ─────────────────────────────────────────
    # If a node has a FQN like "App\Http\Controllers\Auth\LoginController",
    # override its dir_key with the first non-trivial namespace segment.
    ns_override: dict[str, str] = {}  # node_id → refined key
    _NS_SKIP = frozenset({"App", "app", "Http", "Controllers", "Models",
                           "Services", "Providers", "Console", "Exceptions"})
    for node_id, data in G.nodes(data=True):
        fqn = data.get("fqn", "") or data.get("parent", "") or ""
        if not fqn or "\\" not in fqn:
            continue
        segments = [s for s in fqn.split("\\") if s and s not in _NS_SKIP]
        if segments:
            ns_override[node_id] = segments[0].lower()

    # Apply namespace overrides to dir_groups
    if ns_override:
        # Remove nodes from their dir_groups and reassign
        for key in list(dir_groups.keys()):
            dir_groups[key] = [n for n in dir_groups[key] if n not in ns_override]
        for node_id, ns_key in ns_override.items():
            dir_groups[ns_key].append(node_id)
        # Remove empty groups
        dir_groups = defaultdict(list, {k: v for k, v in dir_groups.items() if v})

    # ── Step C: Build code-only undirected subgraph ───────────────────────────
    code_nodes = [n for n, d in G.nodes(data=True)
                  if d.get("type") in _CODE_NODE_TYPES]
    UG = G.to_undirected().subgraph(code_nodes).copy()
    # Remove isolates — nodes with no edges cannot be community-assigned
    isolates = set(nx.isolates(UG))
    UG.remove_nodes_from(isolates)

    # ── Step D: Louvain community detection ───────────────────────────────────
    louvain_map: dict[str, int] = {}   # node_id → community index
    q_score = 0.0

    if UG.number_of_nodes() >= 4 and UG.number_of_edges() >= 2:
        try:
            communities = nx.algorithms.community.louvain_communities(UG, seed=42)
            if len(communities) > 1:
                q_score = nx.algorithms.community.modularity(UG, communities)
            for i, comm in enumerate(communities):
                for n in comm:
                    louvain_map[n] = i
        except Exception:
            pass   # Louvain can fail on degenerate graphs — graceful skip

    use_louvain = q_score >= _Q_LOW and len(louvain_map) > 0

    # ── Step E: Merge Louvain communities → directory groups ──────────────────
    # For each Louvain community, find the plurality dir_group and assign.
    # This fixes over-splitting: 15 per-controller communities → 3 modules.
    final_assignment: dict[str, str] = {}   # node_id → module_key

    if use_louvain:
        # Invert dir_groups to get node → dir_key lookup
        node_to_dir: dict[str, str] = {
            n: k for k, nodes in dir_groups.items() for n in nodes
        }

        # Group Louvain community indices by their plurality dir_key
        comm_to_dir: dict[int, str] = {}
        community_members: dict[int, list[str]] = defaultdict(list)
        for n, ci in louvain_map.items():
            community_members[ci].append(n)

        for ci, members in community_members.items():
            dir_counts: dict[str, int] = defaultdict(int)
            for n in members:
                dk = node_to_dir.get(n, "root")
                dir_counts[dk] += 1
            comm_to_dir[ci] = max(dir_counts, key=dir_counts.__getitem__)

        # Assign based on merged community → dir mapping
        for node_id in G.nodes():
            if node_id in louvain_map:
                final_assignment[node_id] = comm_to_dir[louvain_map[node_id]]
            else:
                # Isolates and non-code nodes fall back to dir_key
                for dk, members in dir_groups.items():
                    if node_id in members:
                        final_assignment[node_id] = dk
                        break
                else:
                    final_assignment[node_id] = "root"
    else:
        # Q too low or no Louvain result — use directory signal only
        for dk, members in dir_groups.items():
            for node_id in members:
                final_assignment[node_id] = dk

    # ── Step F: Legacy flat-file fallback ─────────────────────────────────────
    # If everything landed in "root" (no meaningful directory structure),
    # apply keyword matching on file stems and route paths.
    unique_modules = set(final_assignment.values())
    if unique_modules == {"root"} or (len(unique_modules) == 1):
        _KEYWORD_MODULES: list[tuple[str, tuple[str, ...]]] = [
            ("auth",     ("login", "logout", "register", "auth", "password",
                          "session", "user", "profile", "account", "signin")),
            ("booking",  ("book", "rent", "reserv", "order", "schedul",
                          "appointment", "slot", "availab", "calendar")),
            ("payment",  ("pay", "invoice", "billing", "charge", "transact",
                          "checkout", "cart", "refund", "receipt")),
            ("admin",    ("admin", "dashboard", "manage", "report",
                          "statistic", "analytic", "panel", "staff")),
            ("catalog",  ("product", "item", "catalog", "categor",
                          "inventory", "stock", "listing", "vehicle", "car")),
            ("notify",   ("notif", "email", "sms", "message", "alert",
                          "remind", "mail")),
        ]

        def _keyword_module(node_id: str, data: dict) -> str:
            text = " ".join([
                Path(data.get("file", "")).stem.lower(),
                data.get("name", "").lower(),
                data.get("path", "").lower(),
            ])
            for mod_key, keywords in _KEYWORD_MODULES:
                if any(kw in text for kw in keywords):
                    return mod_key
            return "core"

        for node_id, data in G.nodes(data=True):
            final_assignment[node_id] = _keyword_module(node_id, data)

    # ── Step G: Annotate every node ───────────────────────────────────────────
    def _to_name(key: str) -> str:
        """Convert a slug like "auth" or "Http" to a title-cased display name."""
        _CANONICAL_NAMES = {
            "auth": "Authentication", "booking": "Booking",
            "payment": "Payment",     "admin": "Administration",
            "catalog": "Catalog",     "notify": "Notifications",
            "core": "Core",           "root": "Application",
        }
        if key in _CANONICAL_NAMES:
            return _CANONICAL_NAMES[key]
        return key.replace("_", " ").replace("-", " ").title()

    for node_id, data in G.nodes(data=True):
        mod_key = final_assignment.get(node_id, "root")
        # Confidence: high if Louvain was used and Q is strong,
        #             medium if directory-only, low if keyword fallback
        if use_louvain and q_score >= _Q_MID and node_id in louvain_map:
            conf = round(min(1.0, 0.6 + q_score * 0.4), 2)
        elif mod_key not in ("root", "core"):
            conf = 0.70
        else:
            conf = 0.40
        data["module_id"]         = mod_key
        data["module_name"]       = _to_name(mod_key)
        data["module_confidence"] = conf

    # ── Step H: G.graph["modules"] summary + index extension ─────────────────
    module_nodes: dict[str, list[str]] = defaultdict(list)
    for node_id, data in G.nodes(data=True):
        module_nodes[data["module_id"]].append(node_id)

    modules_summary: dict[str, dict] = {}
    for mod_key, members in sorted(module_nodes.items()):
        type_counts: dict[str, int] = defaultdict(int)
        for n in members:
            t = G.nodes[n].get("type", "unknown")
            type_counts[t] += 1
        modules_summary[mod_key] = {
            "name":        _to_name(mod_key),
            "node_count":  len(members),
            "node_types":  dict(sorted(type_counts.items())),
            "q_score":     round(q_score, 4) if use_louvain else None,
            "method":      "louvain+directory" if use_louvain else "directory",
            "nodes":       members,
        }

    G.graph["modules"] = modules_summary

    # Extend the existing index with a by_module lookup
    if "index" in G.graph:
        G.graph["index"]["by_module"] = {
            mod_key: info["nodes"] for mod_key, info in modules_summary.items()
        }



def _build_index(G: nx.DiGraph) -> None:
    """
    Populate G.graph["index"] — a dict of node-type → list[node_id].

    Built once after all passes complete.  Downstream stages (3, 4, 5) should
    use this instead of scanning G.nodes(data=True) on every query.

    Index keys
    ----------
    functions        — all function / method nodes
    pages            — HTML-mixed PHP files
    scripts          — pure-logic PHP files
    routes           — registered HTTP routes
    controllers      — OOP controller classes
    models           — OOP model / entity classes
    services         — service / repository classes
    db_tables        — database tables
    redirects        — redirect target URLs
    superglobals     — $_POST/$_GET/$_SESSION key reads
    forms            — HTML <form> elements
    http_endpoints   — detected HTTP entry points
    api_endpoints    — subset of routes/endpoints with is_api=True
    external_fns     — EXTCALL:* nodes (PHP built-ins / extensions)
    class_deps       — injected dependencies (DEP:ClassName)
    entry_points     — nodes tagged is_entry_point=True
    """
    idx: dict[str, list[str]] = {
        "functions":     [],
        "pages":         [],
        "scripts":       [],
        "routes":        [],
        "controllers":   [],
        "models":        [],
        "services":      [],
        "db_tables":     [],
        "redirects":     [],
        "superglobals":  [],
        "forms":         [],
        "http_endpoints":[],
        "api_endpoints": [],
        "external_fns":  [],
        "class_deps":    [],
        "entry_points":  [],
    }

    _TYPE_MAP = {
        "function":          "functions",
        "page":              "pages",
        "script":            "scripts",
        "route":             "routes",
        "controller":        "controllers",
        "model":             "models",
        "service":           "services",
        "db_table":          "db_tables",
        "redirect":          "redirects",
        "superglobal":       "superglobals",
        "form":              "forms",
        "http_endpoint":     "http_endpoints",
        "external_function": "external_fns",
        "class_dep":         "class_deps",
    }

    for node_id, data in G.nodes(data=True):
        ntype = data.get("type", "")
        key   = _TYPE_MAP.get(ntype)
        if key:
            idx[key].append(node_id)

        # Cross-cutting flags
        if data.get("is_api"):
            idx["api_endpoints"].append(node_id)
        if data.get("is_entry_point"):
            idx["entry_points"].append(node_id)

    G.graph["index"] = idx


# ─── API Endpoint Enrichment ──────────────────────────────────────────────────

# Route path prefixes that reliably indicate an API endpoint
_API_PATH_PREFIXES: tuple[str, ...] = (
    "/api/", "/api", "/v1/", "/v2/", "/v3/",
    "/rest/", "/graphql", "/webhook", "/webhooks",
)

# Middleware names that indicate an API context
_API_MIDDLEWARE: frozenset[str] = frozenset({
    "api", "auth:api", "auth:sanctum", "throttle",
    "jwt", "jwt.auth", "passport",
})

# Route file names that register API routes (Laravel convention)
_API_ROUTE_FILES: frozenset[str] = frozenset({
    "api.php", "api_routes.php", "routes_api.php",
})

# Method name patterns that suggest an API handler
_API_METHOD_PATTERNS: tuple[str, ...] = (
    "api", "json", "Ajax", "Xhr", "store", "destroy",
    "index", "show", "create", "update",  # REST verbs
)

# EXTCALL names that confirm JSON output
_JSON_OUTPUT_CALLS: frozenset[str] = frozenset({
    "json_encode", "json_decode",
})


def _enrich_api_endpoints(G: nx.DiGraph, code_map: "CodeMap") -> None:
    """
    Pass 14 — multi-signal API endpoint inference.

    Runs after the full graph is built.  For every route node and every
    http_endpoint node, combine five independent signals to determine whether
    the endpoint is an API endpoint and what it produces:

    Signal 1 — Path prefix   : /api/, /v1/, /graphql, /webhook, …
    Signal 2 — Middleware     : "api", "auth:api", "throttle", "jwt", …
    Signal 3 — Route file     : routes/api.php (Laravel convention)
    Signal 4 — Handler name   : method name contains "json", "Ajax", etc.
    Signal 5 — Callee graph   : reachable EXTCALL:json_encode from handler

    Any single strong signal (path/middleware/file) is sufficient.
    Weak signals (handler name, callee) require at least one other weak signal.

    Updates node attributes in-place:
        is_api   : bool
        produces : "json" | "html" | "redirect" | "mixed"
        ep_type  : "api" | "page" | "form_handler" | "form_processor"
        api_signals : list[str]  — which signals fired (for debugging / BA docs)
    """
    # Pre-build a fast lookup: file_basename → all route-file route node ids
    # so signal 3 can be checked without re-scanning routes
    api_route_files: set[str] = set()
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "route":
            fname = Path(data.get("file", "")).name
            if fname in _API_ROUTE_FILES:
                api_route_files.add(node_id)

    # Pre-build a set of all EXTCALL:json_encode / json_decode node IDs
    json_extcall_nodes: set[str] = {
        n for n in G.nodes()
        if n.startswith("EXTCALL:") and G.nodes[n].get("name") in _JSON_OUTPUT_CALLS
    }

    def _reachable_json_call(start_node: str, depth: int = 4) -> bool:
        """
        BFS from start_node — returns True if any json_encode/json_decode
        EXTCALL node is reachable within `depth` hops via `calls` edges.
        Bounded to avoid full-graph traversal on large codebases.
        """
        if not json_extcall_nodes:
            return False
        visited: set[str] = set()
        frontier = {start_node}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for _, dst, edata in G.out_edges(node, data=True):
                    if edata.get("edge_type") == "calls" and dst not in visited:
                        if dst in json_extcall_nodes:
                            return True
                        next_frontier.add(dst)
                        visited.add(dst)
            frontier = next_frontier
            if not frontier:
                break
        return False

    def _classify_node(node_id: str, data: dict) -> None:
        """Evaluate all signals and update the node in-place."""
        signals: list[str] = []

        path       = data.get("path", "")
        middleware = data.get("middleware", "") or ""
        file_name  = Path(data.get("file", "")).name
        handler    = data.get("handler", "") or ""
        produces   = data.get("produces", "html")
        ep_type    = data.get("ep_type",  "page")

        # Already confirmed as JSON by the PHP parser — count as a strong signal
        if produces == "json" or data.get("is_api"):
            signals.append("php_parser_detected")

        # ── Signal 1: path prefix ─────────────────────────────────────────────
        path_lower = path.lower()
        if any(path_lower.startswith(pfx) or f"/{pfx.strip('/')}" in path_lower
               for pfx in _API_PATH_PREFIXES):
            signals.append("path_prefix")

        # ── Signal 2: middleware ──────────────────────────────────────────────
        if isinstance(middleware, list):
            mw_set = {m.lower() for m in middleware}
        else:
            mw_set = {m.strip().lower() for m in str(middleware).split(",")}
        if mw_set & _API_MIDDLEWARE:
            signals.append(f"middleware:{mw_set & _API_MIDDLEWARE}")

        # ── Signal 3: route file ──────────────────────────────────────────────
        if file_name in _API_ROUTE_FILES or node_id in api_route_files:
            signals.append("api_route_file")

        # ── Signal 4: handler method name ─────────────────────────────────────
        method_name = handler.split("::")[-1].split("@")[-1] if handler else ""
        if any(pat.lower() in method_name.lower() for pat in _API_METHOD_PATTERNS):
            signals.append(f"handler_name:{method_name}")

        # ── Signal 5: callee graph — reaches json_encode ──────────────────────
        # Follow handles edges from the route/endpoint node to find the handler,
        # then BFS via calls edges. Also check the file-qualified twin node
        # (file::method) that Pass 9 creates for call graph edges.
        handler_start_nodes: list[str] = []

        # 5a. Find handler via handles edge
        for _, dst, edata in G.out_edges(node_id, data=True):
            if edata.get("edge_type") == "handles":
                handler_start_nodes.append(dst)
                # 5b. Also try the file-qualified node used by call graph edges
                method_name = G.nodes[dst].get("name", "")
                file_       = G.nodes[dst].get("file", "")
                if method_name and file_:
                    twin = f"{file_}::{method_name}"
                    if G.has_node(twin):
                        handler_start_nodes.append(twin)

        # 5c. Fallback: try node itself (for http_endpoint nodes)
        if not handler_start_nodes:
            handler_start_nodes = [node_id]

        for start in handler_start_nodes:
            if _reachable_json_call(start):
                signals.append("reachable_json_encode")
                break

        # ── Decision: is this an API endpoint? ───────────────────────────────
        # Strong signals (each sufficient on its own):
        #   path_prefix, middleware, api_route_file, php_parser_detected,
        #   reachable_json_encode  — if the handler calls json_encode, it IS an API
        # Weak signals need at least 2 to confirm (handler name alone isn't enough)
        strong = {
            "path_prefix", "api_route_file",
            "php_parser_detected", "reachable_json_encode",
        }
        has_strong = any(any(s.startswith(st) for st in strong) for s in signals)
        # middleware is strong but stored with a prefix like "middleware:{'auth:api'}"
        if not has_strong:
            has_strong = any(s.startswith("middleware:") for s in signals)

        weak_count = sum(1 for s in signals if not any(s.startswith(st) for st in strong)
                         and not s.startswith("middleware:"))
        is_api = has_strong or weak_count >= 2

        if is_api and signals:
            G.nodes[node_id]["is_api"]      = True
            G.nodes[node_id]["produces"]    = "json"
            G.nodes[node_id]["ep_type"]     = "api"
            G.nodes[node_id]["api_signals"] = signals

            # Propagate to the handler node too
            for _, dst, edata in G.out_edges(node_id, data=True):
                if edata.get("edge_type") == "handles" and G.has_node(dst):
                    G.nodes[dst]["is_api"]      = True
                    G.nodes[dst]["produces"]    = "json"
                    G.nodes[dst]["api_signals"] = signals

    # ── Apply to all route nodes ──────────────────────────────────────────────
    for node_id, data in list(G.nodes(data=True)):
        if data.get("type") in ("route", "http_endpoint"):
            _classify_node(node_id, data)


# ─── Node / Edge Helpers ──────────────────────────────────────────────────────

def _add_node(G: nx.DiGraph, node_id: str, **attrs: Any) -> None:
    """
    Add node only if not already present; never overwrite existing data.
    Normalises the 'node_type' kwarg to the 'type' key used everywhere else.
    """
    if node_id not in G:
        # Rename node_type → type for consistent lookup
        if "node_type" in attrs:
            attrs["type"] = attrs.pop("node_type")
        G.add_node(node_id, **attrs)


def _add_edge(G: nx.DiGraph, src: str, dst: str, **attrs: Any) -> None:
    """
    Add a directed edge src → dst.  If the same (src, dst, type) already
    exists, merge file/line info rather than creating a duplicate.
    """
    if not G.has_node(src):
        _add_node(G, src, node_type="unknown")
    if not G.has_node(dst):
        _add_node(G, dst, node_type="unknown")

    edge_type = attrs.get("edge_type", "unknown")

    # Check if an edge of the same type already exists
    if G.has_edge(src, dst):
        existing = G[src][dst]
        if existing.get("edge_type") == edge_type:
            return  # exact duplicate — skip
    G.add_edge(src, dst, **attrs)


def _function_node_id(fn_name: str, file: str) -> str:
    """
    Create a stable node ID for a function.
    Includes file to handle same-named functions in different files.
    """
    return f"{file}::{fn_name}"


def _table_node_id(table_name: str) -> str:
    return f"TABLE:{table_name}"


def _resolve_caller(
    caller: str,
    src_file: str,
    defined_functions: dict[str, str],
    G: nx.DiGraph,
) -> str:
    """
    Resolve a caller string to its graph node ID.

    - "GLOBAL_SCRIPT" → the source file node
    - A known function name → its function node (file::name)
    - An unknown name → fall back to source file
    """
    if caller == "GLOBAL_SCRIPT":
        return src_file

    # Prefer same-file function first, then any file
    fn_id_same_file = f"{src_file}::{caller}"
    if G.has_node(fn_id_same_file):
        return fn_id_same_file

    if caller in defined_functions:
        return f"{defined_functions[caller]}::{caller}"

    # Unknown caller — fall back to source file
    return src_file


def _resolve_route_handler(
    handler: str,
    class_methods: dict[str, str],
    method_name_index: dict[str, list[str]],
    defined_functions: dict[str, str],
    G: nx.DiGraph,
) -> str | None:
    """
    Resolve a route handler string to a graph node ID.

    Handles these Laravel handler formats:
        "UserController@index"            — classic string
        "App\\Http\\Controllers\\UserController@index"  — fully qualified
        "UserController::index"           — static-style (from resource expansion)
        "(closure)@index"                 — anonymous closure (skip)

    Resolution order (all O(1)):
        1. Exact match in class_methods dict  "ClassName::method"
        2. method_name_index keyed by method name, filtered by class suffix
        3. Conventional node ID by construction (may not exist yet)

    Returns the node ID string, or None if no matching node can be found.
    """
    if not handler or handler.startswith("(closure)"):
        return None

    # Normalise separator: @ or ::
    sep = "@" if "@" in handler else ("::" if "::" in handler else None)
    if not sep:
        return None

    cls_part, method_part = handler.rsplit(sep, 1)

    # Strip namespace prefix to get bare class name
    bare_class = cls_part.split("\\")[-1]

    # 1. Exact match in class_methods — O(1)
    exact_key = f"{bare_class}::{method_part}"
    if exact_key in class_methods:
        return class_methods[exact_key]

    # 2. method_name_index: find candidates whose parent ends with bare_class — O(candidates)
    #    Replaces the previous O(N_nodes) full graph scan.
    candidates = method_name_index.get(method_part, [])
    for node_id in candidates:
        parent = str(G.nodes[node_id].get("parent", ""))
        if parent == bare_class or parent.endswith(f"\\{bare_class}"):
            return node_id

    # 3. Fall back: conventional node ID (may not exist — _add_edge will create as unknown)
    candidate = f"{bare_class}::{method_part}"
    if G.has_node(candidate):
        return candidate

    return None


# ─── Persistence ─────────────────────────────────────────────────────────────

def _save_gpickle(G: nx.DiGraph, path: str) -> None:
    """Serialise the graph using pickle (fast, lossless, used by Stage 3+)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [stage2] Graph saved → {path}")


def _save_json(G: nx.DiGraph, path: str) -> None:
    """
    Write a human-readable JSON file with separate nodes and edges arrays.
    Matches the shape of the old code_graph.json for compatibility.
    """
    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({"id": node_id, **data})

    edges = []
    for src, dst, data in G.edges(data=True):
        edges.append({"from": src, "to": dst, **data})

    payload = {
        "meta": {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "node_types": sorted({d.get("type",      "unknown") for _, d    in G.nodes(data=True)}),
            "edge_types": sorted({d.get("edge_type", "unknown") for _, _, d in G.edges(data=True)}),
        },
        "nodes": nodes,
        "edges": edges,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"  [stage2] JSON export saved → {path}")


def _load_graph_meta(json_path: str) -> GraphMeta:
    """Re-hydrate GraphMeta from an existing code_graph.json (resume path)."""
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    meta = data.get("meta", {})
    gpickle_path = str(Path(json_path).with_suffix(".gpickle"))
    return GraphMeta(
        graph_path  = gpickle_path,
        node_count  = meta.get("node_count", 0),
        edge_count  = meta.get("edge_count", 0),
        node_types  = meta.get("node_types", []),
        edge_types  = meta.get("edge_types", []),
    )


# ─── Visualisation ────────────────────────────────────────────────────────────

def _save_png(G: nx.DiGraph, path: str) -> None:
    """
    Render the graph to a PNG using matplotlib.

    Layout strategy:
      - Small graphs  (≤ 40 nodes): spring layout — good for readability
      - Medium graphs (≤ 120 nodes): kamada-kawai layout — cleaner clusters
      - Large graphs  (> 120 nodes): shell layout with type-based rings

    Nodes are colour-coded by type; edges are styled by type.
    """
    n = G.number_of_nodes()

    if n == 0:
        print("  [stage2] Empty graph — skipping PNG")
        return

    # ── Layout ────────────────────────────────────────────────────────────────
    if n <= 40:
        pos = nx.spring_layout(G, seed=42, k=2.5 / (n ** 0.5))
    elif n <= 120:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42)
    else:
        pos = _shell_layout_by_type(G)

    # ── Figure sizing — scale with node count ─────────────────────────────────
    fig_size = max(14, min(36, n * 0.45))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.75))
    ax.set_facecolor("#1A1A2E")
    fig.patch.set_facecolor("#1A1A2E")

    # ── Node colours ──────────────────────────────────────────────────────────
    node_color_list = [
        NODE_COLORS.get(G.nodes[n].get("type", "unknown"), NODE_COLORS["unknown"])
        for n in G.nodes()
    ]
    node_sizes = [_node_size(G, n) for n in G.nodes()]

    # ── Edge colours by type ──────────────────────────────────────────────────
    edge_color_map = {
        "calls":         "#AAAAAA",
        "includes":      "#FFD700",
        "sql_read":      "#00BFFF",
        "sql_write":     "#FF6347",
        "sql_ddl":       "#FF1493",
        "redirects_to":  "#FF8C00",
        "defines":       "#7CFC00",
        "reads_input":   "#BD10E0",
        "handles":       "#E91E8C",   # route → handler
        "has_form":      "#AB47BC",   # page → form
        "submits_to":    "#CE93D8",   # form → target
        "form_field_of": "#7B1FA2",   # superglobal ↔ form field
        "depends_on":    "#78909C",   # class → dependency
        "method_of":     "#80CBC4",   # class → method
        "inherits":      "#A5D6A7",   # class → parent
        "implements":    "#C5E1A5",   # class → interface
        "entry_point":   "#EF5350",   # http_endpoint → handler
        "unknown":       "#555555",
    }

    # Draw edge groups separately for consistent styling
    edge_groups: dict[str, list[tuple]] = defaultdict(list)
    for src, dst, data in G.edges(data=True):
        etype = data.get("edge_type", "unknown")
        edge_groups[etype].append((src, dst))

    for etype, edge_list in edge_groups.items():
        color = edge_color_map.get(etype, "#555555")
        nx.draw_networkx_edges(
            G, pos,
            edgelist   = edge_list,
            edge_color = color,
            alpha      = 0.65,
            arrows     = True,
            arrowsize  = 14,
            width      = 1.4,
            ax         = ax,
            connectionstyle = "arc3,rad=0.08",
        )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    nx.draw_networkx_nodes(
        G, pos,
        node_color = node_color_list,
        node_size  = node_sizes,
        alpha      = 0.92,
        ax         = ax,
    )

    # ── Labels — shorten long node IDs ───────────────────────────────────────
    labels = {n: _short_label(n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels      = labels,
        font_size   = max(5, min(9, 180 // n)),
        font_color  = "#FFFFFF",
        font_weight = "bold",
        ax          = ax,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    node_legend = [
        mpatches.Patch(color=color, label=ntype)
        for ntype, color in NODE_COLORS.items()
        if ntype != "unknown"
    ]
    edge_legend = [
        mpatches.Patch(color=color, label=etype)
        for etype, color in edge_color_map.items()
        if etype != "unknown"
    ]
    legend = ax.legend(
        handles   = node_legend + edge_legend,
        loc       = "upper left",
        fontsize  = 8,
        framealpha= 0.3,
        facecolor = "#2A2A4A",
        labelcolor= "white",
        ncol      = 2,
    )

    ax.set_title(
        "PHP Project Knowledge Graph",
        color="white", fontsize=14, fontweight="bold", pad=12,
    )
    ax.axis("off")
    plt.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _shell_layout_by_type(G: nx.DiGraph) -> dict:
    """
    Arrange nodes in concentric rings grouped by type.
    Order (inner → outer): db_table, redirect, function, include_file, page/script
    """
    type_order = ["db_table", "redirect", "superglobal", "external_function",
                  "function", "include_file", "page", "script",
                  "route", "controller", "model", "service", "form",
                  "http_endpoint", "class_dep", "unknown"]
    shells: dict[str, list] = defaultdict(list)
    for node in G.nodes():
        ntype = G.nodes[node].get("type", "unknown")
        shells[ntype].append(node)

    shell_list = [shells[t] for t in type_order if shells.get(t)]
    # Fall back if only one shell
    if len(shell_list) < 2:
        shell_list = [list(G.nodes())]

    return nx.shell_layout(G, nlist=shell_list)


def _node_size(G: nx.DiGraph, node: str) -> int:
    """Scale node size by degree — hubs are larger."""
    degree = G.degree(node)
    if degree == 0:
        return 300
    return min(300 + degree * 120, 2500)


def _short_label(node_id: str) -> str:
    """
    Shorten a node ID for display:
      'app/Http/Controllers/UserController.php::index' → 'UserController::index'
      'TABLE:users'  → 'users'
      'REDIRECT:login.php' → '→login.php'
    """
    if node_id.startswith("TABLE:"):
        return node_id[6:]
    if node_id.startswith("REDIRECT:"):
        return "→" + node_id[9:]
    if node_id.startswith("ROUTE:"):
        parts = node_id[6:].split(":", 1)
        return f"[{parts[0]}] {parts[1]}" if len(parts) == 2 else node_id[6:]
    if node_id.startswith("INCLUDE:"):
        return node_id[8:]

    # file::function  →  file_basename::function
    if "::" in node_id:
        file_part, fn_part = node_id.rsplit("::", 1)
        base = Path(file_part).name
        return f"{base}::{fn_part}"

    # plain file path → basename
    return Path(node_id).name or node_id


# ─── Standalone Helpers (used by Stage 3 / tests) ────────────────────────────

def load_graph(gpickle_path: str) -> nx.DiGraph:
    """
    Load a previously saved graph from a .gpickle file.

    Args:
        gpickle_path: Path to the .gpickle file written by Stage 2.

    Returns:
        nx.DiGraph instance.

    Raises:
        RuntimeError: If the file cannot be read.
    """
    try:
        with open(gpickle_path, "rb") as fh:
            return pickle.load(fh)
    except (OSError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            f"[stage2] Failed to load graph from {gpickle_path}: {exc}"
        ) from exc


def get_nodes_by_type(G: nx.DiGraph, node_type: str) -> list[dict[str, Any]]:
    """
    Return all nodes of a given type as dicts.

    Args:
        G:         The knowledge graph.
        node_type: One of page, function, db_table, include_file, redirect, script.

    Returns:
        List of {'id': ..., ...attrs} dicts.
    """
    return [
        {"id": n, **data}
        for n, data in G.nodes(data=True)
        if data.get("type") == node_type
    ]


def get_edges_by_type(G: nx.DiGraph, edge_type: str) -> list[dict[str, Any]]:
    """
    Return all edges of a given type as dicts.

    Args:
        G:         The knowledge graph.
        edge_type: One of calls, includes, sql_read, sql_write, redirects_to, defines.

    Returns:
        List of {'from': ..., 'to': ..., ...attrs} dicts.
    """
    return [
        {"from": src, "to": dst, **data}
        for src, dst, data in G.edges(data=True)
        if data.get("edge_type") == edge_type
    ]


def summarise_graph(G: nx.DiGraph) -> dict[str, Any]:
    """
    Return a compact summary dict of the graph, useful for logging and
    Stage 3.5 preflight checks.

    Args:
        G: The knowledge graph.

    Returns:
        Dict with counts per node type and edge type.
    """
    node_type_counts: dict[str, int] = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_type_counts[data.get("type", "unknown")] += 1

    edge_type_counts: dict[str, int] = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_type_counts[data.get("edge_type", "unknown")] += 1

    # Most connected nodes (top 5 by total degree)
    top_hubs = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:5]
    hub_info = [
        {"id": n, "degree": G.degree(n), "type": G.nodes[n].get("type")}
        for n in top_hubs
    ]

    return {
        "total_nodes":       G.number_of_nodes(),
        "total_edges":       G.number_of_edges(),
        "nodes_by_type":     dict(node_type_counts),
        "edges_by_type":     dict(edge_type_counts),
        "top_hubs":          hub_info,
        "is_dag":            nx.is_directed_acyclic_graph(G),
        "weakly_connected_components": nx.number_weakly_connected_components(G),
    }