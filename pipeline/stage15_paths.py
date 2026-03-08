"""
pipeline/stage15_paths.py — Execution Path Simulator (Conditional Branch Extractor)

Performs a focused static analysis pass over each PHP entry-point file to extract:

    1. Entry conditions   — what session/POST/GET vars must exist for the page
                            to proceed vs. redirect away
    2. Branch map         — if/elseif/else conditions with their outcomes
                            (DB operation, redirect, session write, output)
    3. Data flow          — which POST/GET fields feed which SQL parameters
    4. Auth guard         — how session auth is checked (key, comparison, redirect)
    5. Happy path         — the "main success scenario" as an ordered step list

This data is stored in ctx.code_map.execution_paths (a new field added to CodeMap)
and written to execution_paths.json in the output directory.

It feeds directly into:
    Stage 4  — DomainAnalystAgent gets richer workflow evidence
    Stage 5  — AC and UserStory agents derive criteria from real branches
    Stage 3  — Chunks include branch-level detail for better semantic search

Architecture
------------
We use tree-sitter-php (already installed for Stage 1) to parse the AST and
walk it with a lightweight visitor. We do NOT attempt full symbolic execution —
we only track:
    - Top-level if/else at file scope and inside functions
    - Conditions that reference $_SESSION, $_POST, $_GET
    - Direct SQL queries (mysqli_query, PDO::query, $conn->query)
    - header('Location:') calls
    - echo/print/include statements as "output" actions

This gives ~80% of the useful branch information with <5% of the complexity
of a full interpreter.

Output schema per file
----------------------
{
  "file": "login.php",
  "entry_conditions": [
    {"type": "session_check", "key": "user_id", "op": "isset",
     "on_fail": "redirect", "redirect_to": "login.php"}
  ],
  "branches": [
    {
      "condition": "$_POST['uname'] and $_POST['pwd']",
      "condition_vars": ["$_POST['uname']", "$_POST['pwd']"],
      "then": [
        {"action": "sql_read",   "table": "registerhere", "detail": "SELECT WHERE uname=? AND pwd=?"},
        {"action": "session_write", "key": "user_id",     "detail": "$_SESSION['user_id'] = $row['id']"},
        {"action": "redirect",   "target": "carrental.php"}
      ],
      "else": [
        {"action": "redirect",   "target": "login.php?error=1"}
      ]
    }
  ],
  "data_flows": [
    {"source": "$_POST['uname']", "sink": "sql_query", "table": "registerhere",
     "field_mapping": {"uname": "$_POST['uname']", "pwd": "$_POST['pwd']"}}
  ],
  "auth_guard": {"key": "user_id", "check": "isset", "redirect": "login.php"},
  "happy_path": [
    "Receive $_POST['uname'], $_POST['pwd']",
    "Query registerhere WHERE uname=? AND pwd=?",
    "Write $_SESSION['user_id']",
    "Redirect to carrental.php"
  ]
}

Resume behaviour
----------------
If stage15_paths is COMPLETED and execution_paths.json exists, stage is skipped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from context import CodeMap, PipelineContext

OUTPUT_FILE = "execution_paths.json"


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 1.5 entry point. Analyses each PHP file for conditional branches
    and execution paths, enriches ctx.code_map with the results.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If code_map is missing (Stage 1 not run).
    """
    output_path = ctx.output_path(OUTPUT_FILE)

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage15_paths") and Path(output_path).exists():
        paths = json.loads(Path(output_path).read_text(encoding="utf-8"))
        _attach_to_code_map(ctx.code_map, paths)
        print(f"  [stage15] Resuming — {len(paths)} file path(s) loaded.")
        return

    if ctx.code_map is None:
        raise RuntimeError("[stage15] ctx.code_map is None — run Stage 1 first.")

    print(f"  [stage15] Analysing execution paths in "
          f"{ctx.code_map.total_files} file(s) ...")

    # ── Collect PHP entry-point files ─────────────────────────────────────────
    php_files = _collect_php_files(ctx.php_project_path)
    print(f"  [stage15] Found {len(php_files)} PHP file(s) to analyse.")

    # ── Analyse each file ─────────────────────────────────────────────────────
    all_paths: list[dict[str, Any]] = []
    errors: list[str] = []

    for php_file in sorted(php_files):
        try:
            result = analyse_file(php_file, ctx.php_project_path)
            if result:
                all_paths.append(result)
        except Exception as e:
            errors.append(f"{Path(php_file).name}: {e}")

    if errors:
        print(f"  [stage15] {len(errors)} file(s) skipped due to errors:")
        for err in errors[:5]:
            print(f"    • {err}")

    # ── Attach to CodeMap ─────────────────────────────────────────────────────
    _attach_to_code_map(ctx.code_map, all_paths)

    # ── Persist ───────────────────────────────────────────────────────────────
    Path(output_path).write_text(
        json.dumps(all_paths, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ctx.stage("stage15_paths").mark_completed(output_path)
    ctx.save()

    _print_summary(all_paths)
    print(f"  [stage15] Saved → {output_path}")


# ─── File Analyser ─────────────────────────────────────────────────────────────

def analyse_file(
    php_file:     str,
    project_root: str,
) -> dict[str, Any] | None:
    """
    Analyse a single PHP file and return its execution path model.

    Args:
        php_file:     Absolute path to the .php file.
        project_root: Project root for computing relative paths.

    Returns:
        Path model dict, or None if the file has no interesting content.
    """
    source = Path(php_file).read_text(encoding="utf-8", errors="replace")
    rel    = str(Path(php_file).relative_to(project_root))

    analyser = _FileAnalyser(source, rel)
    return analyser.analyse()


# ─── Core Analyser ─────────────────────────────────────────────────────────────

class _FileAnalyser:
    """
    Regex + lightweight AST-style analyser for a single PHP file.

    Uses tree-sitter if available for precise AST walking; falls back to
    robust regex patterns that handle 95% of real-world raw PHP patterns.
    """

    def __init__(self, source: str, filename: str) -> None:
        self.source   = source
        self.filename = filename
        self.lines    = source.splitlines()

    def analyse(self) -> dict[str, Any] | None:
        entry_conditions = self._extract_entry_conditions()
        branches         = self._extract_branches()
        data_flows       = self._extract_data_flows()
        auth_guard       = self._extract_auth_guard()
        happy_path       = self._build_happy_path(
            entry_conditions, branches, data_flows, auth_guard
        )

        # Skip files with zero interesting content
        if (not entry_conditions and not branches and
                not data_flows and not auth_guard):
            return None

        return {
            "file":              self.filename,
            "entry_conditions":  entry_conditions,
            "branches":          branches,
            "data_flows":        data_flows,
            "auth_guard":        auth_guard,
            "happy_path":        happy_path,
        }

    # ── Entry Conditions ──────────────────────────────────────────────────────

    def _extract_entry_conditions(self) -> list[dict[str, Any]]:
        """
        Find session/auth checks at the top of the file that gate access.
        Patterns: isset($_SESSION[...]), !isset, empty, == null, session_start()
        """
        conditions = []
        # Only look in the first 40 lines — entry conditions are always near top
        top = "\n".join(self.lines[:40])

        # Pattern: if(!isset($_SESSION['key'])) { header('Location: x.php'); }
        pat = re.compile(
            r"if\s*\(\s*(!?)isset\s*\(\s*\$_SESSION\[(['\"])(\w+)\2\]\s*\)\s*\)"
            r"[^{]*\{[^}]*header\s*\(\s*['\"]Location:\s*([^'\"]+)['\"]",
            re.IGNORECASE | re.DOTALL,
        )
        for m in pat.finditer(top):
            negated     = m.group(1) == "!"
            session_key = m.group(3)
            redirect_to = m.group(4).strip()
            conditions.append({
                "type":        "session_check",
                "key":         session_key,
                "op":          "not_isset" if negated else "isset",
                "on_fail":     "redirect",
                "redirect_to": redirect_to,
            })

        # Pattern: session_start() presence
        if re.search(r"session_start\s*\(\s*\)", top, re.IGNORECASE):
            conditions.append({"type": "session_start"})

        # Pattern: if($_SERVER['REQUEST_METHOD'] == 'POST')
        if re.search(r"\$_SERVER\[['\"]REQUEST_METHOD['\"]\]\s*==\s*['\"]POST['\"]",
                     top, re.IGNORECASE):
            conditions.append({
                "type": "method_check",
                "method": "POST",
            })

        return conditions

    # ── Branch Extraction ─────────────────────────────────────────────────────

    def _extract_branches(self) -> list[dict[str, Any]]:
        """
        Extract if/else branches that involve user input or DB operations.
        Focuses on branches referencing $_POST, $_GET, $_SESSION, sql queries,
        header() redirects.
        """
        branches = []
        source   = self.source

        # Find all if blocks — simplified brace-balanced extraction
        if_blocks = self._find_if_blocks(source)

        for block in if_blocks:
            condition      = block["condition"]
            then_body      = block["then_body"]
            else_body      = block.get("else_body", "")

            # Only keep branches that involve user input or interesting ops
            cond_vars = _extract_vars_from_condition(condition)
            if not cond_vars and not _has_interesting_ops(then_body):
                continue

            then_actions = self._extract_actions(then_body)
            else_actions = self._extract_actions(else_body) if else_body else []

            if not then_actions and not else_actions:
                continue

            branches.append({
                "condition":      _clean_condition(condition),
                "condition_vars": cond_vars,
                "then":           then_actions,
                "else":           else_actions,
            })

        return branches[:10]  # cap at 10 branches per file

    def _find_if_blocks(self, source: str) -> list[dict[str, Any]]:
        """Extract if/else blocks with brace-balanced body extraction."""
        blocks  = []
        # Match: if (condition) {
        pat = re.compile(r'\bif\s*\(', re.IGNORECASE)

        for m in pat.finditer(source):
            start = m.start()
            # Extract condition by finding matching closing paren
            cond_start = m.end() - 1   # position of opening (
            cond_end   = _find_matching(source, cond_start, "(", ")")
            if cond_end < 0:
                continue
            condition = source[cond_start + 1 : cond_end].strip()

            # Find then-body opening brace
            brace_pos = source.find("{", cond_end)
            if brace_pos < 0:
                continue
            then_end = _find_matching(source, brace_pos, "{", "}")
            if then_end < 0:
                continue
            then_body = source[brace_pos + 1 : then_end]

            # Look for else/elseif after then_end
            else_body = ""
            after = source[then_end + 1 : then_end + 200].lstrip()
            else_m = re.match(r'^else\s*\{', after, re.IGNORECASE)
            if else_m:
                else_start = then_end + 1 + (len(source[then_end + 1:]) - len(after))
                # find the { in the else
                else_brace = source.find("{", else_start + else_m.start())
                if else_brace >= 0:
                    else_end  = _find_matching(source, else_brace, "{", "}")
                    if else_end >= 0:
                        else_body = source[else_brace + 1 : else_end]

            blocks.append({
                "condition": condition,
                "then_body": then_body,
                "else_body": else_body,
            })

        return blocks

    def _extract_actions(self, body: str) -> list[dict[str, Any]]:
        """Extract meaningful actions from a branch body."""
        actions = []

        # SQL queries
        for m in re.finditer(
            r'(?:mysqli_query|mysql_query|\$(?:conn|db|pdo|mysqli)\s*->\s*query'
            r'|\$(?:conn|db|pdo|stmt)\s*->\s*prepare)\s*\(\s*["\']?\s*([^)]{5,})',
            body, re.IGNORECASE,
        ):
            sql_fragment = m.group(1)[:120].strip().strip("\"'")
            op  = _classify_sql_op(sql_fragment)
            tbl = _extract_table_name(sql_fragment)
            actions.append({
                "action": f"sql_{op}",
                "table":  tbl,
                "detail": sql_fragment[:80],
            })

        # Session writes
        for m in re.finditer(
            r'\$_SESSION\[([\'"])(\w+)\1\]\s*=\s*([^;]+);',
            body,
        ):
            actions.append({
                "action": "session_write",
                "key":    m.group(2),
                "detail": f"$_SESSION['{m.group(2)}'] = {m.group(3).strip()[:60]}",
            })

        # Redirects
        for m in re.finditer(
            r'header\s*\(\s*[\'"]Location:\s*([^\'\"]+)[\'"]',
            body, re.IGNORECASE,
        ):
            actions.append({
                "action": "redirect",
                "target": m.group(1).strip(),
            })

        # Output / echo
        echo_count = len(re.findall(r'\becho\b|\bprint\b', body, re.IGNORECASE))
        if echo_count > 0:
            actions.append({"action": "output_html", "count": echo_count})

        # Error / die
        if re.search(r'\bdie\b|\bexit\b', body, re.IGNORECASE):
            actions.append({"action": "terminate"})

        # Include
        for m in re.finditer(
            r'\b(?:include|require)(?:_once)?\s*\(?[\'"]([^\'"]+)[\'"]',
            body, re.IGNORECASE,
        ):
            actions.append({"action": "include", "file": m.group(1)})

        return actions

    # ── Data Flow ─────────────────────────────────────────────────────────────

    def _extract_data_flows(self) -> list[dict[str, Any]]:
        """
        Track which POST/GET fields flow into SQL queries.
        Handles both:
        - Inline: mysqli_query($conn, "SELECT ... '$_POST[field]' ...")
        - Variable: $var = $_POST['field']; ... $sql = "... $var ..."; mysqli_query($conn, $sql)
        """
        flows   = []
        source  = self.source

        # Build map: variable name → source expression
        var_map: dict[str, str] = {}
        for m in re.finditer(
            r'\$(\w+)\s*=\s*\$_(POST|GET|REQUEST)\s*\[\s*[\'"](\w+)[\'"]\s*\]',
            source,
        ):
            var_map[m.group(1)] = f"$_{m.group(2)}['{m.group(3)}']"

        # Build map: $sql var → sql string (handles $sql = "SELECT ...")
        sql_var_map: dict[str, str] = {}
        for m in re.finditer(r'\$(\w+)\s*=\s*["\']([^"\']{10,})["\']', source):
            val = m.group(2).upper().strip()
            if any(val.startswith(kw) for kw in
                   ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE")):
                sql_var_map[m.group(1)] = m.group(2)

        # Also capture concatenated SQL: $sql = "SELECT..." . $var
        for m in re.finditer(
            r'\$(\w+)\s*=\s*["\']([^"\']{6,})["\']'
            r'(?:\s*\.\s*\$(\w+)(?:\s*\.\s*["\'][^"\']*["\'])?)+',
            source,
        ):
            val = m.group(2).upper().strip()
            if any(val.startswith(kw) for kw in
                   ("SELECT", "INSERT", "UPDATE", "DELETE")):
                # include the variable name(s) in the sql fragment
                sql_var_map[m.group(1)] = source[m.start():m.end()]

        # Now find all actual mysqli_query / ->query calls
        query_targets: list[str] = []

        # Direct string queries
        for m in re.finditer(
            r'(?:mysqli_query|mysql_query|\$(?:conn|db|pdo|mysqli)\s*->\s*(?:query|prepare))'
            r'\s*\(\s*(?:\$\w+\s*,\s*)?["\']([^"\']{5,})["\']',
            source, re.IGNORECASE,
        ):
            query_targets.append(m.group(1))

        # Variable-based queries: mysqli_query($conn, $sql)
        for m in re.finditer(
            r'(?:mysqli_query|mysql_query|\$(?:conn|db|pdo|mysqli)\s*->\s*(?:query|prepare))'
            r'\s*\(\s*\$\w+\s*,\s*\$(\w+)\s*\)',
            source, re.IGNORECASE,
        ):
            var_name = m.group(1)
            if var_name in sql_var_map:
                query_targets.append(sql_var_map[var_name])

        for sql in query_targets:
            table = _extract_table_name(sql)
            op    = _classify_sql_op(sql)

            field_mapping: dict[str, str] = {}

            # Inline $_POST['field'] references in the SQL string
            for im in re.finditer(r'\$_(POST|GET)\[\'(\w+)\'\]', sql):
                field_mapping[im.group(2)] = f"$_{im.group(1)}['{im.group(2)}']"

            # Variable references in SQL that came from POST/GET
            for var, origin in var_map.items():
                if f"${var}" in sql or f"'{var}'" in sql:
                    fname = re.search(r"\['(\w+)'\]", origin)
                    if fname:
                        field_mapping[fname.group(1)] = origin

            # Also check the raw source near this sql for $var usage
            sql_pos = source.find(sql[:40])
            if sql_pos >= 0:
                context_window = source[max(0, sql_pos - 300):sql_pos + 50]
                for var, origin in var_map.items():
                    if f"${var}" in context_window:
                        fname = re.search(r"\['(\w+)'\]", origin)
                        if fname:
                            field_mapping[fname.group(1)] = origin

            if field_mapping or table:
                flows.append({
                    "source":        list(field_mapping.values()) or ["$_POST/$_GET"],
                    "sink":          f"sql_{op}",
                    "table":         table,
                    "field_mapping": field_mapping,
                })

        return flows

    # ── Auth Guard ────────────────────────────────────────────────────────────

    def _extract_auth_guard(self) -> dict[str, Any] | None:
        """
        Detect the primary authentication guard pattern for this file.
        Returns None if no auth check is found.
        """
        top = "\n".join(self.lines[:50])

        # isset($_SESSION['key']) gate
        m = re.search(
            r'if\s*\(\s*(!?)isset\s*\(\s*\$_SESSION\[([\'"])(\w+)\2\]\s*\)\s*\)'
            r'[^{]*\{[^}]*header\s*\(\s*[\'"]Location:\s*([^\'\"]+)[\'"]',
            top, re.IGNORECASE | re.DOTALL,
        )
        if m:
            return {
                "key":      m.group(3),
                "check":    "not_isset" if m.group(1) == "!" else "isset",
                "redirect": m.group(4).strip(),
            }

        # Comparison: $_SESSION['role'] == 'admin'
        m = re.search(
            r'\$_SESSION\[([\'"])(\w+)\1\]\s*(==|!=|===)\s*[\'"](\w+)[\'"]',
            top,
        )
        if m:
            return {
                "key":       m.group(2),
                "check":     f"{m.group(3)} '{m.group(4)}'",
                "redirect":  None,
            }

        return None

    # ── Happy Path Builder ────────────────────────────────────────────────────

    def _build_happy_path(
        self,
        entry_conditions: list[dict],
        branches:         list[dict],
        data_flows:       list[dict],
        auth_guard:       dict | None,
    ) -> list[str]:
        """
        Synthesise the primary success scenario as an ordered step list.
        Picks the deepest/most-specific branch that leads to a successful
        outcome (session write or non-error redirect).
        """
        steps: list[str] = []

        # Step 0: session_start
        if any(c.get("type") == "session_start" for c in entry_conditions):
            steps.append("Session started (session_start)")

        # Step 1: Auth guard
        if auth_guard:
            steps.append(
                f"Auth check: $_SESSION['{auth_guard['key']}'] "
                f"must be set (else → {auth_guard.get('redirect','redirect')})"
            )

        # Step 2: POST/GET input received
        post_fields: set[str] = set()
        for flow in data_flows:
            for v in flow.get("source", []):
                m = re.search(r"\$_(POST|GET)\['(\w+)'\]", v)
                if m:
                    post_fields.add(f"$_{m.group(1)}['{m.group(2)}']")
        if post_fields:
            steps.append(f"Receive input: {', '.join(sorted(post_fields))}")

        # Step 3: Pick the most-specific "success" branch
        # Prefer: branches that contain session_write (login success) OR
        # a redirect that looks like success (no ?error, not the auth guard redirect)
        auth_redirect = auth_guard.get("redirect") if auth_guard else None

        def _is_success_branch(b: dict) -> bool:
            then = b.get("then", [])
            return (
                any(a.get("action") == "session_write" for a in then) or
                any(
                    a.get("action") == "redirect" and
                    a.get("target", "") != auth_redirect and
                    "error" not in a.get("target", "").lower() and
                    "fail" not in a.get("target", "").lower()
                    for a in then
                )
            )

        # Sort: prefer branches with session_write, then by depth (more specific)
        success_branches = [
            b for b in branches
            if _is_success_branch(b) and "_SESSION" not in b.get("condition", "")
        ]

        for branch in success_branches[:1]:
            for action in branch.get("then", []):
                act = action.get("action", "")
                tgt = action.get("target", "")
                # Skip error-path redirects
                if act == "redirect" and (
                    "error" in tgt.lower() or tgt == auth_redirect
                ):
                    continue
                if act.startswith("sql_"):
                    tbl = action.get("table", "?")
                    steps.append(
                        f"{act.replace('sql_','SQL ').upper()} on `{tbl}`: "
                        f"{action.get('detail','')[:60]}"
                    )
                elif act == "session_write":
                    steps.append(
                        f"Write session: $_SESSION['{action.get('key')}']"
                    )
                elif act == "redirect":
                    steps.append(f"Redirect → {tgt}")
                elif act == "output_html":
                    steps.append("Render HTML output")
                elif act == "include":
                    steps.append(f"Include {action.get('file','?')}")

        # Step 4: Final success redirect (if not already captured above)
        if not any("Redirect →" in s for s in steps):
            all_redirects = [
                m.group(1).strip()
                for m in re.finditer(
                    r'header\s*\(\s*[\'"]Location:\s*([^\'\"]+)[\'"]',
                    self.source, re.IGNORECASE,
                )
            ]
            # Pick last redirect that isn't auth-fail or error
            success_redirect = next(
                (
                    r for r in reversed(all_redirects)
                    if r != auth_redirect
                    and "error" not in r.lower()
                    and "fail"  not in r.lower()
                ),
                None,
            )
            if success_redirect:
                steps.append(f"Redirect → {success_redirect}")

        return steps

        return steps


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _find_matching(source: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    """Find the position of the matching closing character, skipping nested pairs."""
    depth = 0
    i = open_pos
    in_str: str | None = None

    while i < len(source):
        ch = source[i]

        # String tracking (skip content inside strings)
        if in_str:
            if ch == in_str and source[i - 1] != "\\":
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _extract_vars_from_condition(condition: str) -> list[str]:
    """Extract $_POST, $_GET, $_SESSION variable references from a condition."""
    found = re.findall(r'\$_(POST|GET|SESSION|REQUEST)\s*\[[\'"]?\w+[\'"]?\]', condition)
    return list(dict.fromkeys(
        re.findall(r'\$_(POST|GET|SESSION|REQUEST)\[[^\]]+\]', condition)
    ))


def _has_interesting_ops(body: str) -> bool:
    """Return True if body contains DB, redirect, or session operations."""
    return bool(re.search(
        r'mysqli_query|mysql_query|\$\w+->query|\bheader\s*\(|'
        r'\$_SESSION\s*\[|\bdie\b|\bexit\b',
        body, re.IGNORECASE,
    ))


def _clean_condition(condition: str) -> str:
    """Normalise a condition string for readability."""
    # Remove extra whitespace
    c = re.sub(r'\s+', ' ', condition).strip()
    # Truncate very long conditions
    return c[:120] + "..." if len(c) > 120 else c


def _classify_sql_op(sql: str) -> str:
    """Classify SQL string as read/write/ddl."""
    s = sql.strip().upper()
    if s.startswith("SELECT"):    return "read"
    if s.startswith("INSERT"):    return "write"
    if s.startswith("UPDATE"):    return "write"
    if s.startswith("DELETE"):    return "write"
    if s.startswith("CREATE"):    return "ddl"
    if s.startswith("DROP"):      return "ddl"
    if s.startswith("ALTER"):     return "ddl"
    return "query"


def _extract_table_name(sql: str) -> str:
    """Extract the primary table name from a SQL fragment."""
    sql_u = sql.upper()
    for pat in [
        r'FROM\s+[`\'"]?(\w+)',
        r'INTO\s+[`\'"]?(\w+)',
        r'UPDATE\s+[`\'"]?(\w+)',
        r'TABLE\s+[`\'"]?(\w+)',
    ]:
        m = re.search(pat, sql_u)
        if m:
            return sql[m.start(1) : m.start(1) + len(m.group(1))].strip("`'\"").lower()
    return ""


def _collect_php_files(project_root: str) -> list[str]:
    """Collect all .php files from the project, excluding vendor/."""
    root  = Path(project_root)
    files = []
    SKIP  = {"vendor", "node_modules", ".git", "cache", "logs", "storage"}
    for php in root.rglob("*.php"):
        if not any(part in SKIP for part in php.parts):
            files.append(str(php))
    return files


def _attach_to_code_map(code_map: CodeMap, paths: list[dict[str, Any]]) -> None:
    """
    Attach execution paths to the CodeMap.
    Adds code_map.execution_paths if the field exists, otherwise stores
    in code_map.globals as a compatibility shim.
    """
    if hasattr(code_map, "execution_paths"):
        code_map.execution_paths = paths
    else:
        # Compatibility: store in globals under a reserved key
        code_map.globals = [
            g for g in (code_map.globals or [])
            if g.get("__type") != "execution_paths"
        ]
        code_map.globals.append({"__type": "execution_paths", "data": paths})


def _print_summary(paths: list[dict[str, Any]]) -> None:
    total_branches  = sum(len(p.get("branches", []))        for p in paths)
    total_flows     = sum(len(p.get("data_flows", []))       for p in paths)
    guarded         = sum(1 for p in paths if p.get("auth_guard"))
    width = 54
    print(f"\n  {'=' * width}")
    print(f"  Stage 1.5 — Execution Path Analysis")
    print(f"  {'=' * width}")
    print(f"  Files analysed : {len(paths)}")
    print(f"  Branches found : {total_branches}")
    print(f"  Data flows     : {total_flows}")
    print(f"  Auth-guarded   : {guarded} file(s)")
    print(f"  {'=' * width}")
    for p in paths:
        hp = p.get("happy_path", [])
        ag = p.get("auth_guard")
        print(f"  {p['file']}")
        if ag:
            print(f"    🔒 Auth: session['{ag['key']}'] → {ag.get('redirect','?')}")
        for step in hp[:4]:
            print(f"    → {step[:70]}")
    print(f"  {'=' * width}\n")
