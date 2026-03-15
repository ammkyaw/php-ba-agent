"""
run_stage45.py — Standalone runner for Stage 4.5 (Business Flow Extraction)

Extracts BusinessFlow objects from the knowledge graph and domain model
without re-running the full pipeline.  Useful when:
  • Stage 4.5 produced 0 flows and you want to re-run after fixing the graph
  • You want to re-extract flows after editing domain_model or the graph
  • You want to inspect traversal results without LLM enrichment (--no-llm)
  • Stage 4.5 failed or was skipped and you want to run it alone

Prerequisites (must be completed in the pipeline run being resumed):
  Stage 2  — graph pickle (ctx.graph_meta.graph_path must exist)
  Stage 4  — domain model (ctx.domain_model must not be None)

Stage 1.5 (execution_paths) is optional but improves flow quality when present.

Usage
-----
# Run from a pipeline run that has completed stages 2 and 4:
    python run_stage45.py --resume outputs/run_20240101_120000/context.json

# Force re-run even if stage45 was already marked completed:
    python run_stage45.py --resume outputs/run_.../context.json --force

# Skip LLM enrichment (graph traversal only — no API calls, instant):
    python run_stage45.py --resume outputs/run_.../context.json --no-llm

# Write business_flows.json to a different location:
    python run_stage45.py --resume outputs/run_.../context.json --output /path/to/dir

Exit codes
----------
0  flows extracted and saved successfully (or stage already done and skipped)
1  stage raised an exception during execution
2  bad arguments or missing prerequisites (graph not found, domain_model None, etc.)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog            = "run_stage45.py",
        description     = "Run Stage 4.5 (business flow extraction) standalone.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = __doc__,
    )

    parser.add_argument(
        "--resume",
        metavar  = "CONTEXT_FILE",
        required = True,
        help     = "Path to context.json from a pipeline run that has completed "
                   "Stage 2 (graph) and Stage 4 (domain model).",
    )
    parser.add_argument(
        "--output", "-o",
        metavar = "OUTPUT_DIR",
        default = None,
        help    = "Directory where business_flows.json is written.  "
                  "Defaults to the run's output_dir.",
    )
    parser.add_argument(
        "--force",
        action  = "store_true",
        default = False,
        help    = "Re-run even if stage45_flows is already marked COMPLETED.",
    )
    parser.add_argument(
        "--no-llm",
        dest    = "no_llm",
        action  = "store_true",
        default = False,
        help    = "Skip LLM enrichment (Pass G–H).  Flows are produced using only "
                  "graph traversal and heuristic naming — no API calls, runs instantly.  "
                  "Useful for debugging graph structure without spending API quota.",
    )

    return parser.parse_args()


# ─── Input validation (runs before any pipeline imports) ──────────────────────

def _validate_args(args: argparse.Namespace) -> None:
    """Check filesystem pre-conditions before importing any pipeline code."""
    p = Path(args.resume)
    if not p.exists():
        _die(f"Context file not found: {p}")
    if not p.is_file():
        _die(f"Not a file: {p}")


def _die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(2)


# ─── sys.path setup ───────────────────────────────────────────────────────────

def _setup_path() -> None:
    """Prepend the project root to sys.path so pipeline imports resolve."""
    project_root = str(Path(__file__).parent.resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# ─── Module import with flat-layout fallback ──────────────────────────────────

def _import_stage45() -> object:
    """
    Import stage45_flows and return the module.

    Tries the package layout (pipeline/stage45_flows.py) first, then falls
    back to a flat layout (stage45_flows.py in the same directory).
    """
    try:
        import pipeline.stage45_flows as mod
        return mod
    except ImportError:
        pass
    try:
        import stage45_flows as mod  # flat layout (dev / test)
        return mod
    except ImportError:
        _die(
            "Could not import stage45_flows from pipeline/ or current directory.\n"
            "       Make sure the file exists and context.py is importable."
        )


# ─── Context loading ──────────────────────────────────────────────────────────

def _load_context(args: argparse.Namespace) -> "PipelineContext":
    """Load context.json and apply --force / --output overrides."""
    from context import PipelineContext, StageStatus

    ctx = PipelineContext.load(str(Path(args.resume)))
    print(f"Loaded run : {ctx.run_id}")
    print(f"Output dir : {ctx.output_dir}")

    if args.output:
        # Override where business_flows.json lands — redirect output_dir so
        # ctx.output_path("business_flows.json") resolves to the new dir.
        # We store it as a private attribute rather than mutating output_dir
        # globally, to avoid affecting context.json save location.
        ctx._output_dir_override = str(Path(args.output).resolve())
        Path(ctx._output_dir_override).mkdir(parents=True, exist_ok=True)
        print(f"Output dir overridden → {ctx._output_dir_override}")

    if args.force:
        ctx.stages["stage45_flows"].status = StageStatus.PENDING
        ctx.stages["stage45_flows"].error  = None
        print("--force : stage45_flows reset to PENDING")

    return ctx


# ─── Prerequisite checks (with helpful fix hints) ────────────────────────────

def _check_prerequisites(ctx: "PipelineContext") -> None:
    """
    Validate that the upstream stages stage45 depends on are complete.
    Exits with code 2 and a helpful message if anything is missing.
    """
    resume_hint = f"python run_pipeline.py --resume {ctx.output_dir}/context.json"

    if ctx.domain_model is None:
        _die(
            "ctx.domain_model is None — Stage 4 (domain analysis) must be completed first.\n"
            f"       Run: {resume_hint} --until stage4_domain"
        )

    if ctx.graph_meta is None or not ctx.graph_meta.graph_path:
        _die(
            "ctx.graph_meta is missing — Stage 2 (knowledge graph) must be completed first.\n"
            f"       Run: {resume_hint} --until stage2_graph"
        )

    gpickle = Path(ctx.graph_meta.graph_path)
    if not gpickle.exists():
        _die(
            f"Graph file not found: {gpickle}\n"
            f"       Stage 2 may have been run in a different location, or the file was deleted.\n"
            f"       Run: {resume_hint} --force stage2_graph"
        )

    # Warn (don't block) if stage15 was skipped — flows will still work,
    # but execution_path detail (auth guards, db ops, inputs) will be absent.
    if ctx.code_map and not ctx.code_map.execution_paths:
        print(
            "  [run_stage45] Note: ctx.code_map.execution_paths is empty — "
            "Stage 1.5 may not have run.\n"
            "                Flow steps will be generated from graph structure only "
            "(no auth/db/input detail)."
        )


# ─── --no-llm: monkey-patch _enrich_with_llm ─────────────────────────────────

def _patch_no_llm(s45_mod: object) -> None:
    """
    Replace stage45's _enrich_with_llm with a heuristic-only version that
    produces BusinessFlow objects from the skeletons without any API calls.

    The patched version calls the same fallback helpers (_fallback_name,
    _infer_actor, _infer_trigger, _infer_termination, _final_confidence) that
    the real enricher uses when the LLM call fails, so the output structure is
    identical — just without LLM-assigned names and actors.
    """
    from context import BusinessFlow

    def _no_llm_enrich(skeletons, context_groups, ctx):
        flows = []
        flow_counter = 1
        domain = ctx.domain_model

        for context_name, indices in context_groups.items():
            for i in indices:
                sk = skeletons[i]
                flow_id = f"flow_{flow_counter:03d}"
                flow_counter += 1
                flows.append(BusinessFlow(
                    flow_id          = flow_id,
                    name             = s45_mod._fallback_name(sk),
                    actor            = s45_mod._infer_actor(sk, domain),
                    bounded_context  = context_name,
                    trigger          = s45_mod._infer_trigger(sk),
                    steps            = sk["steps"],
                    branches         = sk["branches"],
                    termination      = s45_mod._infer_termination(sk),
                    evidence_files   = sk["evidence_files"],
                    confidence       = s45_mod._final_confidence(
                                           sk["raw_confidence"], False),
                    replaces_workflow= None,
                ))

        return flows

    s45_mod._enrich_with_llm = _no_llm_enrich
    print("--no-llm : LLM enrichment disabled — using heuristic naming only")


# ─── --output redirect ────────────────────────────────────────────────────────

def _patch_output_dir(s45_mod: object, ctx: "PipelineContext") -> None:
    """
    If --output was given, patch ctx.output_path() for this stage only so
    business_flows.json lands in the override directory.

    We monkey-patch output_path on the instance (not the class) so other
    methods that call ctx.output_path() are unaffected after stage45 returns.
    """
    override = getattr(ctx, "_output_dir_override", None)
    if not override:
        return

    import os
    original_output_path = ctx.output_path  # bound method

    def _patched_output_path(filename: str) -> str:
        if filename == s45_mod.FLOWS_FILE:
            return os.path.join(override, filename)
        return original_output_path(filename)

    ctx.output_path = _patched_output_path


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # 1. Validate filesystem paths before any imports
    _validate_args(args)

    # 2. sys.path
    _setup_path()

    # 3. Import stage45
    s45_mod = _import_stage45()

    # 4. Load context
    ctx = _load_context(args)

    # 5. Prerequisites
    _check_prerequisites(ctx)

    # 6. Apply --no-llm patch
    if args.no_llm:
        _patch_no_llm(s45_mod)

    # 7. Apply --output redirect
    _patch_output_dir(s45_mod, ctx)

    # 8. Summary line
    llm_label  = "disabled (--no-llm)" if args.no_llm else "enabled"
    graph_node_count = ctx.graph_meta.node_count if ctx.graph_meta else "?"
    print(f"Graph nodes    : {graph_node_count}")
    print(f"Domain model   : {ctx.domain_model.domain_name!r}")
    print(f"LLM enrichment : {llm_label}")
    print()

    # 9. Run
    try:
        s45_mod.run(ctx)
    except RuntimeError as exc:
        # RuntimeError from stage45 prerequisites — hard failure
        _die(str(exc))
    except Exception as exc:
        print(f"\nStage 4.5 raised an exception: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # 10. Non-zero exit if stage recorded a failure
    if ctx.stages["stage45_flows"].status.value == "failed":
        sys.exit(1)

    # 11. Summary
    if ctx.business_flows:
        n = ctx.business_flows.total
        print(f"\nExtracted {n} flow(s).  "
              f"Resume the pipeline with:\n"
              f"  python run_pipeline.py --resume "
              f"{Path(args.resume)}")


if __name__ == "__main__":
    main()
