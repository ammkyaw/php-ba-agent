"""
pipeline/stage37_statemachines.py — State Machine Reconstruction (Stage 3.7)

Detects entity lifecycle state machines from static signals only — no LLM.

Algorithm
---------
Phase 1 — Detect state fields
    • table_columns with names matching *status*, *state*, *phase*, *stage*,
      *lifecycle*, *step*, *condition* (case-insensitive).
    • sql_queries: UPDATE/INSERT columns with status-like names.
    • InvariantCollection (Stage 2.9) STATE_TRANSITION rules — each already
      identified a field that carries state.

Phase 2 — Extract state values
    • SQL literals: ``SET status = 'X'`` and ``WHERE status = 'X'``.
    • Execution-path branch conditions: ``$obj->status === 'X'``.
    • PHP source scan: ``$obj->status = 'X';`` and ``$arr['status'] = 'X';``.
    • Values are normalised to Title-Case; obvious non-states (0,1,'','NULL',
      routing tokens, UUID-like strings) are discarded.

Phase 3 — Detect transitions
    A — SQL SET+WHERE: ``UPDATE t SET status='B' WHERE … status='A'``  → conf 0.90
    B — Branch guard + sibling UPDATE in same file: branch with ``status='A'``
        followed by SQL UPDATE touching same table                       → conf 0.75
    C — Source proximity: consecutive assignments to same status field   → conf 0.50

Phase 4 — Build directed graph
    • Compute initial states  (no incoming edges).
    • Compute terminal states (no outgoing edges).
    • Detect dead states      (unreachable from any initial state via BFS).
    • Generate Mermaid stateDiagram-v2 source.

Output
------
    StateMachineCollection  (written to 4.3_statemachines/state_machine_catalog.json)
    ctx.state_machines set on context

Consumed by
-----------
    Stage 4.5  (business flow validation — are observed transitions legal?)
    Stage 6.7  (Mermaid state diagrams)
    Stage 8    (test case generator — boundary transitions)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from context import (
    PipelineContext,
    StateMachine,
    StateMachineCollection,
    StateTransition,
)

# ── Tunables ──────────────────────────────────────────────────────────────────
MIN_STATES_FOR_MACHINE  = 2    # ignore (table, field) pairs with < 2 distinct states
MIN_TRANSITIONS         = 0    # emit machines even without transitions (state inventory is useful)
CONF_SQL_SET_WHERE      = 0.90
CONF_BRANCH_UPDATE      = 0.75
CONF_PROXIMITY          = 0.50

# ── State field name patterns ─────────────────────────────────────────────────
_STATE_FIELD_RE = re.compile(
    r"(status|state|phase|stage|lifecycle|step|condition)$",
    re.IGNORECASE,
)

# ── Noise filters for state values ───────────────────────────────────────────
_NUMERIC_RE    = re.compile(r"^\d+$")
_UUID_RE       = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_SHORT_RE      = re.compile(r"^.{1,2}$")        # single/double-char literals → skip

_NOISE_VALUES  = {
    "", "null", "none", "true", "false", "yes", "no", "n/a",
    # SQL tokens
    "asc", "desc", "limit", "offset", "where", "set", "and", "or",
    # routing values reused from invariant detection
    "save", "edit", "create", "view", "list", "search", "delete",
    "detail", "update", "index", "show", "new", "get", "post",
    "put", "patch", "head", "options", "upload", "download",
    "module", "action", "subpanel", "return", "export", "import",
    # PHP booleans as strings
    "1", "0",
}

# ── SQL parsing helpers ───────────────────────────────────────────────────────
_SET_STATUS_RE  = re.compile(
    r"SET\s+[`'\"]?(\w+)[`'\"]?\s*=\s*['\"]([^'\"]{2,50})['\"]",
    re.IGNORECASE,
)
_WHERE_STATUS_RE = re.compile(
    r"WHERE\s.*?[`'\"]?(\w+)[`'\"]?\s*=\s*['\"]([^'\"]{2,50})['\"]",
    re.IGNORECASE,
)
_UPDATE_TABLE_RE = re.compile(
    r"UPDATE\s+[`'\"]?(\w+)[`'\"]?",
    re.IGNORECASE,
)

# ── PHP source patterns ───────────────────────────────────────────────────────
# $obj->status = 'X' or $obj->status='X'
_OBJ_ASSIGN_RE  = re.compile(
    r"""\$(\w+)\s*->\s*(\w+)\s*=\s*['"]([^'"]{2,50})['"]""",
)
# $arr['status'] = 'X'
_ARR_ASSIGN_RE  = re.compile(
    r"""\$\w+\s*\[\s*['"](\w+)['"]\s*\]\s*=\s*['"]([^'"]{2,50})['"]""",
)
# $obj->status === 'X' or == 'X'
_OBJ_CMP_RE     = re.compile(
    r"""\$(\w+)\s*->\s*(\w+)\s*={2,3}\s*['"]([^'"]{2,50})['"]""",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_state_field(col_name: str) -> bool:
    return bool(_STATE_FIELD_RE.search(col_name))


# Values that look like XML/WSDL artefacts, class types, or non-lifecycle tokens
_XML_TYPE_VALUES = {
    "binding", "body", "envelope", "header", "message", "porttype", "schema",
    "service", "operation", "fault", "part", "type", "element", "attribute",
    "extension", "restriction", "sequence", "choice", "complextype",
    "simpletype", "annotation", "documentation", "import",
}


def _is_noise_value(val: str) -> bool:
    v = val.strip().lower()
    if not v:
        return True
    if v in _NOISE_VALUES:
        return True
    if v in _XML_TYPE_VALUES:
        return True
    if _NUMERIC_RE.match(v):
        return True
    if _UUID_RE.match(v):
        return True
    if _SHORT_RE.match(v):
        return True
    return False


def _obj_matches_table(obj_var: str, table: str) -> bool:
    """
    Return True if the PHP variable name looks like it refers to the given table.
    e.g., $email → emails, $emailObj → emails, $case → cases
    Very permissive — only reject obvious mismatches.
    """
    obj = obj_var.lower().rstrip("s")     # singularise
    tbl = table.lower().rstrip("s")       # singularise table too
    if not obj or not tbl:
        return True   # can't tell — allow
    # Exact or substring match in either direction
    if obj == tbl:
        return True
    if obj in tbl or tbl in obj:
        return True
    # Allow generic object names that give no information
    generic = {"obj", "object", "bean", "row", "record", "item", "entity",
                "model", "data", "result", "this", "self", "that", "arr", "array"}
    if obj in generic:
        return True
    return False


def _norm_state(val: str) -> str:
    """Normalise state value to Title-Case."""
    val = val.strip()
    # already multi-word like 'in_progress' → 'In Progress'
    val = val.replace("_", " ").replace("-", " ")
    return val.title()


def _extract_module_name(filepath: str) -> str | None:
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part.lower() == "modules" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _assign_context(filepath: str, ac_file_map: dict[str, str]) -> str:
    """Resolve a file path to a bounded-context name."""
    if filepath in ac_file_map:
        return ac_file_map[filepath]
    mod = _extract_module_name(filepath)
    if mod:
        return mod
    parts = Path(filepath).parts
    for part in reversed(parts[:-1]):
        if part not in (".", "..", "/", ""):
            return part
    return "Unknown"


def _build_mermaid(entity: str, field: str, states: list[str],
                   transitions: list[StateTransition],
                   initial: list[str], terminal: list[str]) -> str:
    lines = ["stateDiagram-v2"]
    lines.append(f"    %% {entity}.{field} lifecycle")
    for s in states:
        safe = s.replace(" ", "_")
        lines.append(f"    {safe} : {s}")
    for s in initial:
        safe = s.replace(" ", "_")
        lines.append(f"    [*] --> {safe}")
    seen_transitions: set[tuple[str, str]] = set()
    for t in transitions:
        key = (t.from_state, t.to_state)
        if key in seen_transitions:
            continue
        seen_transitions.add(key)
        fs = t.from_state.replace(" ", "_")
        ts = t.to_state.replace(" ", "_")
        label = t.trigger if t.trigger else ""
        if t.guard:
            label = f"{label} [{t.guard}]" if label else f"[{t.guard}]"
        if label:
            lines.append(f"    {fs} --> {ts} : {label}")
        else:
            lines.append(f"    {fs} --> {ts}")
    for s in terminal:
        safe = s.replace(" ", "_")
        lines.append(f"    {safe} --> [*]")
    return "\n".join(lines)


# ── Phase 1: Detect state fields ─────────────────────────────────────────────

def _detect_state_fields(
    cm: Any,
    invariant_rules: list[Any],
) -> set[tuple[str, str]]:   # set of (table, field)
    """Return (table, field) pairs that look like state-carrying columns."""
    state_fields: set[tuple[str, str]] = set()

    # From table_columns
    for col_entry in (cm.table_columns or []):
        table = col_entry.get("table", "").lower()
        col   = col_entry.get("column", "")
        if not table or not col:
            continue
        if _is_state_field(col):
            state_fields.add((table, col.lower()))

    # From sql_queries — scan SET clauses
    for q in (cm.sql_queries or []):
        sql   = q.get("query", "") or q.get("sql", "")
        table = q.get("table", "").lower()
        if not sql or not table:
            continue
        for m in _SET_STATUS_RE.finditer(sql):
            col = m.group(1)
            if _is_state_field(col):
                state_fields.add((table, col.lower()))

    # From Stage 2.9 STATE_TRANSITION rules — they reference a table.field
    for rule in invariant_rules:
        if getattr(rule, "category", "") == "STATE_TRANSITION":
            entity = getattr(rule, "entity", "")
            for tbl_name in getattr(rule, "tables", []):
                if tbl_name:
                    # Use 'status' as the default field if entity doesn't specify
                    field = "status"
                    if "." in entity:
                        field = entity.split(".")[-1].lower()
                    if _is_state_field(field):
                        state_fields.add((tbl_name.lower(), field))

    return state_fields


# ── Phase 2: Extract state values ────────────────────────────────────────────

def _extract_state_values(
    cm: Any,
    state_fields: set[tuple[str, str]],
    php_project_path: str,
) -> dict[tuple[str, str], set[str]]:
    """
    Return {(table, field): set_of_state_values}.
    """
    values: dict[tuple[str, str], set[str]] = defaultdict(set)

    # ── SQL literal scan ──────────────────────────────────────────────────────
    for q in (cm.sql_queries or []):
        sql   = q.get("query", "") or q.get("sql", "")
        table = q.get("table", "").lower()
        if not sql:
            continue

        for m in _SET_STATUS_RE.finditer(sql):
            col, val = m.group(1).lower(), m.group(2)
            if (table, col) in state_fields and not _is_noise_value(val):
                values[(table, col)].add(_norm_state(val))

        for m in _WHERE_STATUS_RE.finditer(sql):
            col, val = m.group(1).lower(), m.group(2)
            if (table, col) in state_fields and not _is_noise_value(val):
                values[(table, col)].add(_norm_state(val))

        # For queries without a known table, try to match via UPDATE regex
        if not table:
            tm = _UPDATE_TABLE_RE.search(sql)
            if tm:
                table = tm.group(1).lower()
            for m in _SET_STATUS_RE.finditer(sql):
                col, val = m.group(1).lower(), m.group(2)
                if table and (table, col) in state_fields and not _is_noise_value(val):
                    values[(table, col)].add(_norm_state(val))

    # ── Execution-path branch conditions ─────────────────────────────────────
    for ep in (cm.execution_paths or []):
        for branch in ep.get("branches", []):
            cond = branch.get("condition", "")
            if not cond:
                continue
            for m in _OBJ_CMP_RE.finditer(cond):
                col, val = m.group(2).lower(), m.group(3)
                if _is_state_field(col) and not _is_noise_value(val):
                    # Match against any (*, col) pair
                    for tf in state_fields:
                        if tf[1] == col:
                            values[tf].add(_norm_state(val))

    # ── PHP source scan ───────────────────────────────────────────────────────
    try:
        project = Path(php_project_path)
        for php_file in project.rglob("*.php"):
            try:
                src = php_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for m in _OBJ_ASSIGN_RE.finditer(src):
                obj_var, col, val = m.group(1), m.group(2).lower(), m.group(3)
                if _is_state_field(col) and not _is_noise_value(val):
                    for tf in state_fields:
                        if tf[1] == col and _obj_matches_table(obj_var, tf[0]):
                            values[tf].add(_norm_state(val))

            for m in _ARR_ASSIGN_RE.finditer(src):
                col, val = m.group(1).lower(), m.group(2)
                if _is_state_field(col) and not _is_noise_value(val):
                    for tf in state_fields:
                        if tf[1] == col:
                            values[tf].add(_norm_state(val))

            for m in _OBJ_CMP_RE.finditer(src):
                obj_var, col, val = m.group(1), m.group(2).lower(), m.group(3)
                if _is_state_field(col) and not _is_noise_value(val):
                    for tf in state_fields:
                        if tf[1] == col and _obj_matches_table(obj_var, tf[0]):
                            values[tf].add(_norm_state(val))

    except Exception:
        pass  # non-fatal; SQL + execution_paths alone may be sufficient

    return dict(values)


# ── Phase 3: Detect transitions ───────────────────────────────────────────────

def _detect_transitions_sql(
    cm: Any,
    state_fields: set[tuple[str, str]],
) -> list[dict]:
    """
    Method A — SQL UPDATE with both SET field='to' and WHERE field='from'.
    conf = 0.90
    """
    results = []
    for q in (cm.sql_queries or []):
        sql   = q.get("query", "") or q.get("sql", "")
        table = q.get("table", "").lower()
        fpath = q.get("file", "")
        if not sql or "UPDATE" not in sql.upper():
            continue

        # Infer table from query if missing
        if not table:
            tm = _UPDATE_TABLE_RE.search(sql)
            if tm:
                table = tm.group(1).lower()

        for m_set in _SET_STATUS_RE.finditer(sql):
            col_set = m_set.group(1).lower()
            val_to  = m_set.group(2)
            if (table, col_set) not in state_fields:
                continue
            if _is_noise_value(val_to):
                continue

            for m_where in _WHERE_STATUS_RE.finditer(sql):
                col_wh  = m_where.group(1).lower()
                val_from = m_where.group(2)
                if col_wh != col_set:
                    continue
                if _is_noise_value(val_from):
                    continue
                if val_from == val_to:
                    continue

                trigger = Path(fpath).stem if fpath else "sql_update"
                results.append({
                    "table":       table,
                    "field":       col_set,
                    "from_state":  _norm_state(val_from),
                    "to_state":    _norm_state(val_to),
                    "trigger":     trigger,
                    "guard":       "",
                    "source_files": [fpath] if fpath else [],
                    "confidence":  CONF_SQL_SET_WHERE,
                })

    return results


def _detect_transitions_branch(
    cm: Any,
    state_fields: set[tuple[str, str]],
) -> list[dict]:
    """
    Method B — execution_path branch whose condition checks field='from'
    and whose then-actions include an SQL UPDATE on the same table.
    conf = 0.75
    """
    # Build: file → list of sql_queries (UPDATE) for quick lookup
    file_updates: dict[str, list[dict]] = defaultdict(list)
    for q in (cm.sql_queries or []):
        sql  = q.get("query", "") or q.get("sql", "")
        if sql and "UPDATE" in sql.upper():
            fpath = q.get("file", "")
            if fpath:
                file_updates[fpath].append(q)

    results = []
    for ep in (cm.execution_paths or []):
        ep_file = ep.get("file", "")
        for branch in ep.get("branches", []):
            cond  = branch.get("condition", "")
            if not cond:
                continue

            # Extract (field, from_state) from branch condition
            for m in _OBJ_CMP_RE.finditer(cond):
                col      = m.group(2).lower()
                val_from = m.group(3)
                if not _is_state_field(col) or _is_noise_value(val_from):
                    continue

                # Find a matching UPDATE in the same file
                for q in file_updates.get(ep_file, []):
                    sql   = q.get("query", "") or q.get("sql", "")
                    table = q.get("table", "").lower()
                    if not table:
                        tm = _UPDATE_TABLE_RE.search(sql)
                        if tm:
                            table = tm.group(1).lower()

                    if (table, col) not in state_fields:
                        continue

                    for m_set in _SET_STATUS_RE.finditer(sql):
                        if m_set.group(1).lower() != col:
                            continue
                        val_to = m_set.group(2)
                        if _is_noise_value(val_to) or val_to == val_from:
                            continue

                        results.append({
                            "table":       table,
                            "field":       col,
                            "from_state":  _norm_state(val_from),
                            "to_state":    _norm_state(val_to),
                            "trigger":     Path(ep_file).stem if ep_file else "branch_action",
                            "guard":       cond[:120],
                            "source_files": [ep_file],
                            "confidence":  CONF_BRANCH_UPDATE,
                        })

    return results


def _detect_transitions_proximity(
    cm: Any,
    state_fields: set[tuple[str, str]],
    php_project_path: str,
) -> list[dict]:
    """
    Method C — consecutive PHP source assignments to the same status field
    within 40 lines (from_state = earlier value → to_state = later value).
    conf = 0.50
    """
    results = []
    try:
        project = Path(php_project_path)
        for php_file in project.rglob("*.php"):
            try:
                lines = php_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue

            # Collect (lineno, col, obj_var, val) for all status assignments
            assignments: list[tuple[int, str, str, str]] = []
            for i, line in enumerate(lines):
                for m in _OBJ_ASSIGN_RE.finditer(line):
                    obj_var, col, val = m.group(1), m.group(2).lower(), m.group(3)
                    if _is_state_field(col) and not _is_noise_value(val):
                        assignments.append((i, col, obj_var, val))
                for m in _ARR_ASSIGN_RE.finditer(line):
                    col, val = m.group(1).lower(), m.group(2)
                    if _is_state_field(col) and not _is_noise_value(val):
                        assignments.append((i, col, "", val))

            # Pair consecutive assignments to the same col within 40 lines
            for idx in range(len(assignments) - 1):
                lno_a, col_a, obj_a, val_a = assignments[idx]
                lno_b, col_b, obj_b, val_b = assignments[idx + 1]
                if col_a != col_b:
                    continue
                if lno_b - lno_a > 40:
                    continue
                if val_a == val_b:
                    continue

                # Match col to a known (table, field), filtered by obj var
                for table, field in state_fields:
                    if field != col_a:
                        continue
                    # Both assignments should agree on the object variable
                    if obj_a and not _obj_matches_table(obj_a, table):
                        continue
                    if obj_b and not _obj_matches_table(obj_b, table):
                        continue
                    results.append({
                        "table":       table,
                        "field":       col_a,
                        "from_state":  _norm_state(val_a),
                        "to_state":    _norm_state(val_b),
                        "trigger":     php_file.stem,
                        "guard":       "",
                        "source_files": [str(php_file)],
                        "confidence":  CONF_PROXIMITY,
                    })
    except Exception:
        pass

    return results


def _merge_transitions(
    all_raw: list[dict],
) -> list[dict]:
    """
    Deduplicate by (table, field, from_state, to_state).
    Keep the entry with highest confidence; merge source_files.
    """
    key_map: dict[tuple, dict] = {}
    for t in all_raw:
        key = (t["table"], t["field"], t["from_state"], t["to_state"])
        if key not in key_map:
            key_map[key] = dict(t)
        else:
            existing = key_map[key]
            if t["confidence"] > existing["confidence"]:
                existing["confidence"] = t["confidence"]
                if t["trigger"]:
                    existing["trigger"] = t["trigger"]
                if t["guard"] and not existing["guard"]:
                    existing["guard"] = t["guard"]
            # merge source_files
            existing_files = set(existing["source_files"])
            for f in t["source_files"]:
                if f and f not in existing_files:
                    existing["source_files"].append(f)
                    existing_files.add(f)

    return list(key_map.values())


# ── Phase 4: Build directed graph + Mermaid ───────────────────────────────────

def _build_machines(
    transitions_by_tf: dict[tuple[str, str], list[dict]],
    values_by_tf: dict[tuple[str, str], set[str]],
    ac_file_map: dict[str, str],
) -> list[StateMachine]:
    machines = []
    seq = 1

    for (table, field), raw_transitions in sorted(transitions_by_tf.items()):
        if len(raw_transitions) < MIN_TRANSITIONS:
            continue

        # All distinct states
        state_set: set[str] = set()
        for t in raw_transitions:
            state_set.add(t["from_state"])
            state_set.add(t["to_state"])
        # Enrich from Phase-2 value extraction
        state_set |= (values_by_tf.get((table, field), set()))
        states = sorted(state_set)

        if len(states) < MIN_STATES_FOR_MACHINE:
            continue

        # Build adjacency: outgoing and incoming
        outgoing: dict[str, set[str]] = defaultdict(set)
        incoming: dict[str, set[str]] = defaultdict(set)
        trans_objs: list[StateTransition] = []

        for t in raw_transitions:
            outgoing[t["from_state"]].add(t["to_state"])
            incoming[t["to_state"]].add(t["from_state"])
            trans_objs.append(StateTransition(
                from_state   = t["from_state"],
                to_state     = t["to_state"],
                trigger      = t["trigger"],
                guard        = t["guard"],
                source_files = t["source_files"],
                confidence   = t["confidence"],
            ))

        initial_states  = sorted(s for s in states if s not in incoming)
        terminal_states = sorted(s for s in states if s not in outgoing)

        # BFS from initial to find reachable states
        reachable: set[str] = set()
        queue = deque(initial_states)
        while queue:
            node = queue.popleft()
            if node in reachable:
                continue
            reachable.add(node)
            for nxt in outgoing.get(node, set()):
                if nxt not in reachable:
                    queue.append(nxt)
        dead_states = sorted(s for s in states if s not in reachable)

        # Infer entity name: Title-Case of the table (drop trailing 's' singularise)
        entity = table.replace("_", " ").title()
        if entity.endswith("s") and len(entity) > 3:
            entity = entity[:-1]

        # Bounded context: use first source file of highest-confidence transition
        context_file = ""
        for t in sorted(raw_transitions, key=lambda x: -x["confidence"]):
            if t["source_files"]:
                context_file = t["source_files"][0]
                break
        bounded_context = _assign_context(context_file, ac_file_map) if context_file else table.title()

        mermaid = _build_mermaid(entity, field, states, trans_objs, initial_states, terminal_states)

        machines.append(StateMachine(
            machine_id      = f"sm_{seq:03d}",
            entity          = entity,
            table           = table,
            field           = field,
            bounded_context = bounded_context,
            states          = states,
            initial_states  = initial_states,
            terminal_states = terminal_states,
            dead_states     = dead_states,
            transitions     = trans_objs,
            mermaid         = mermaid,
        ))
        seq += 1

    return machines


# ── Main entry point ──────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    cm = getattr(ctx, "code_map", None)
    if cm is None:
        print("  [stage37] ⚠️  No code_map — skipping state machine reconstruction.")
        ctx.state_machines = StateMachineCollection()
        return

    # Build file → context lookup from Stage 2.8 action clusters
    ac_file_map: dict[str, str] = {}
    if ctx.action_clusters:
        for cluster in ctx.action_clusters.clusters:
            for f in cluster.files:
                ac_file_map[f] = cluster.name

    invariant_rules = list((ctx.invariants.rules if ctx.invariants else []))

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    state_fields = _detect_state_fields(cm, invariant_rules)
    print(f"  [stage37] Phase 1 — {len(state_fields)} state field(s) detected.")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    values_by_tf = _extract_state_values(cm, state_fields, ctx.php_project_path)
    total_vals = sum(len(v) for v in values_by_tf.values())
    print(f"  [stage37] Phase 2 — {total_vals} state value(s) extracted "
          f"across {len(values_by_tf)} field(s).")

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    raw_A = _detect_transitions_sql(cm, state_fields)
    raw_B = _detect_transitions_branch(cm, state_fields)
    raw_C = _detect_transitions_proximity(cm, state_fields, ctx.php_project_path)

    all_raw = _merge_transitions(raw_A + raw_B + raw_C)
    print(f"  [stage37] Phase 3 — {len(all_raw)} unique transition(s) "
          f"(A={len(raw_A)}, B={len(raw_B)}, C={len(raw_C)}).")

    # Group transitions by (table, field)
    transitions_by_tf: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in all_raw:
        transitions_by_tf[(t["table"], t["field"])].append(t)

    # Ensure every state field from Phase 2 has an entry (even if no transitions)
    for tf, vals in values_by_tf.items():
        if tf not in transitions_by_tf:
            transitions_by_tf[tf]  # creates empty list — filtered out in Phase 4

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    machines = _build_machines(transitions_by_tf, values_by_tf, ac_file_map)
    total = len(machines)

    collection = StateMachineCollection(machines=machines, total=total)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n  [stage37] State Machine Reconstruction complete — {total} machine(s):")
    for sm in machines:
        print(f"    [{sm.machine_id}] {sm.entity}.{sm.field:<20} "
              f"{len(sm.states):>2} states  "
              f"{len(sm.transitions):>2} transitions  "
              f"dead={len(sm.dead_states)}")
    print()

    ctx.state_machines = collection

    # ── Persist ───────────────────────────────────────────────────────────────
    try:
        import dataclasses
        out_path = ctx.output_path("state_machine_catalog.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(collection), fh, indent=2, ensure_ascii=False)
        print(f"  [stage37] Saved → {out_path}")

        # Also write individual Mermaid files for Stage 6.7
        diag_dir = Path(out_path).parent / "mermaid"
        diag_dir.mkdir(parents=True, exist_ok=True)
        for sm in machines:
            fname = f"{sm.table}_{sm.field}.mmd"
            (diag_dir / fname).write_text(sm.mermaid, encoding="utf-8")
        if machines:
            print(f"  [stage37] Mermaid files → {diag_dir}/")
    except Exception as exc:
        print(f"  [stage37] ⚠️  Could not save state_machine_catalog.json: {exc}")
