"""
run_query.py — Interactive Natural Language Query CLI for the System Knowledge Graph

Translates plain-English questions into graph queries and returns human-readable
answers.  Works with or without Neo4j:

  With NEO4J_URI set    → NL → Cypher → Neo4j → formatted answer
  Without NEO4J_URI     → NL → JSON filter → in-memory traversal → formatted answer

Setup
-----
    # Required: complete pipeline through Stage 9
    python run_pipeline.py --project /path/to/php-project

    # Optional: start Neo4j (Docker)
    docker run -d --name neo4j \\
      -p 7474:7474 -p 7687:7687 \\
      -e NEO4J_AUTH=neo4j/your-password \\
      neo4j:5
    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_PASSWORD=your-password

Usage
-----
    # Interactive session (REPL)
    python run_query.py --resume outputs/run_20240101_120000/context.json

    # Single question (non-interactive)
    python run_query.py --resume outputs/run_.../context.json \\
                        --ask "Which features depend on login?"

    # Show schema and exit
    python run_query.py --resume outputs/run_.../context.json --schema

    # Show the generated query without executing it
    python run_query.py --resume outputs/run_.../context.json \\
                        --ask "What tables does checkout write?" --dry-run

Example questions
-----------------
    Which features depend on login?
    What tables does the checkout flow write?
    Who can perform the booking flow?
    What pages query the users table?
    What is in the Authentication context?
    Which flows does the admin actor perform?
    What entities are stored in the orders table?

Exit codes
----------
0  success
1  stage/query error
2  bad arguments or missing files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog            = "run_query.py",
        description     = "Natural language query interface for the system knowledge graph.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = __doc__,
    )
    parser.add_argument(
        "--resume",
        metavar  = "CONTEXT_FILE",
        required = True,
        help     = "Path to context.json from a completed pipeline run.",
    )
    parser.add_argument(
        "--ask",
        metavar  = "QUESTION",
        default  = None,
        help     = "Ask a single question and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--schema",
        action  = "store_true",
        default = False,
        help    = "Print the graph schema and exit.",
    )
    parser.add_argument(
        "--dry-run",
        dest    = "dry_run",
        action  = "store_true",
        default = False,
        help    = "Show generated query without executing it.",
    )
    parser.add_argument(
        "--show-query",
        dest    = "show_query",
        action  = "store_true",
        default = False,
        help    = "Always print the generated Cypher / filter alongside the answer.",
    )
    return parser.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _die(msg: str, code: int = 2) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)


def _print_result(result, show_query: bool = False) -> None:
    """Pretty-print a QueryResult to stdout."""
    if result.error:
        print(f"\n⚠️  Error: {result.error}")

    if show_query and result.query_used:
        mode_label = "Cypher" if result.mode == "neo4j" else "JSON filter"
        print(f"\n── Generated {mode_label} ──────────────────────────────────")
        print(result.query_used)
        print("───────────────────────────────────────────────────────────")

    if result.answer:
        print(f"\n{result.answer}")

    if result.raw_rows:
        print(f"\n  ({len(result.raw_rows)} result(s)  mode={result.mode})")


def _dry_run_result(result, engine) -> None:
    """Print the generated query without executing, then exit."""
    mode_label = "Cypher" if engine._neo4j_cfg else "JSON filter"
    print(f"\n── Generated {mode_label} (dry run — not executed) ────────")
    print(result.query_used or "(no query generated)")
    print("─────────────────────────────────────────────────────────────")


# ─── Single-question Mode ─────────────────────────────────────────────────────

def _run_single(engine, question: str, args: argparse.Namespace) -> int:
    """Run one question, print the result, return exit code."""
    if args.dry_run:
        # Generate query only — no execution, no formatting
        if engine._neo4j_cfg:
            query = engine._generate_cypher(question)
        else:
            spec  = engine._generate_json_filter(question)
            import json
            query = json.dumps(spec, indent=2)

        from pipeline.graph_query import QueryResult
        result = QueryResult(question=question, query_used=query,
                             mode="neo4j" if engine._neo4j_cfg else "json")
        _dry_run_result(result, engine)
        return 0

    result = engine.query(question)
    _print_result(result, show_query=args.show_query)
    return 1 if result.error else 0


# ─── Interactive REPL ─────────────────────────────────────────────────────────

_HELP_TEXT = """
Commands
--------
  <question>   Ask any natural-language question about the system graph
  /schema      Show graph schema (node types, edge types, examples)
  /mode        Show current query mode (neo4j or json)
  /query       Toggle showing generated queries alongside answers
  /help        Show this help
  /quit        Exit
  Ctrl-C       Exit

Example questions
-----------------
  Which features depend on login?
  What tables does the checkout flow write?
  Who can perform the booking flow?
  What pages query the users table?
  What is in the Authentication context?
"""

def _run_repl(engine, args: argparse.Namespace) -> int:
    """Run an interactive REPL until the user quits."""
    show_query = args.show_query

    mode_str = (
        f"Neo4j ({engine._neo4j_cfg[0]})"
        if engine._neo4j_cfg
        else "JSON (offline — set NEO4J_URI to enable Cypher)"
    )
    print(f"\n╔══════════════════════════════════════════════════════╗")
    print(f"║   System Knowledge Graph — Natural Language Query    ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print(f"  Domain : {engine._schema.domain_name}")
    print(f"  Mode   : {mode_str}")
    print(f"  Nodes  : {len(engine._graph['nodes'])}")
    print(f"  Edges  : {len(engine._graph['edges'])}")
    print(f"\nType a question or /help.  Ctrl-C to exit.\n")

    while True:
        try:
            user_input = input("❓ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return 0

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye.")
            return 0

        if user_input.lower() == "/help":
            print(_HELP_TEXT)
            continue

        if user_input.lower() == "/schema":
            print(f"\n{engine.schema_summary()}\n")
            continue

        if user_input.lower() == "/mode":
            print(f"  Mode: {mode_str}\n")
            continue

        if user_input.lower() == "/query":
            show_query = not show_query
            state = "ON" if show_query else "OFF"
            print(f"  Show queries: {state}\n")
            continue

        # It's a question
        result = engine.query(user_input)
        _print_result(result, show_query=show_query)
        print()

    return 0


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    context_file = Path(args.resume)
    if not context_file.exists():
        _die(f"Context file not found: {context_file}")

    # Import here so startup errors surface cleanly
    try:
        from pipeline.graph_query import GraphQueryEngine
    except ImportError as exc:
        _die(f"Could not import graph_query module: {exc}")

    try:
        engine = GraphQueryEngine(str(context_file))
    except RuntimeError as exc:
        _die(str(exc))
    except FileNotFoundError as exc:
        _die(str(exc))

    # --schema
    if args.schema:
        print(engine.schema_summary())
        sys.exit(0)

    # --ask (single question, non-interactive)
    if args.ask:
        code = _run_single(engine, args.ask, args)
        sys.exit(code)

    # Interactive REPL
    code = _run_repl(engine, args)
    sys.exit(code)


if __name__ == "__main__":
    main()
