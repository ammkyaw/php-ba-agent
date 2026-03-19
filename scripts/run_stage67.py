"""
run_stage67.py — Standalone runner for Stage 6.7 (Diagram Generation)

Generates Mermaid diagram files from a completed (or partial) pipeline run
without re-running the full pipeline.  Useful when:
  • You want to regenerate diagrams after editing domain_model or business_flows
  • Stage 6.7 was skipped or failed and you want to run it alone
  • You only want a subset of diagram types (architecture, flows, or sequences)
  • You want diagrams without injecting them into the Markdown artefacts

Stage 6.7 requires upstream context produced by earlier stages:
  ctx.domain_model    — required (Stage 4)
  ctx.business_flows  — optional (Stage 4.5); sequences and flow diagrams are
                        skipped silently when absent

Usage
-----
# Run stage67 from a completed or partial pipeline run:
    python run_stage67.py --resume outputs/run_20240101_120000/context.json

# Force re-run even if stage67 was already marked completed:
    python run_stage67.py --resume outputs/run_.../context.json --force

# Generate only specific diagram types:
    python run_stage67.py --resume outputs/run_.../context.json --only arch
    python run_stage67.py --resume outputs/run_.../context.json --only seq
    python run_stage67.py --resume outputs/run_.../context.json --only flow
    python run_stage67.py --resume outputs/run_.../context.json --only arch --only flow

# Skip injection of Mermaid blocks into Markdown artefacts:
    python run_stage67.py --resume outputs/run_.../context.json --no-inject

# Write .mmd files to a different directory:
    python run_stage67.py --resume outputs/run_.../context.json --output /path/to/diagrams

Exit codes
----------
0  all requested diagrams generated successfully
1  one or more generators raised an exception
2  bad arguments or environment error (missing file, domain_model absent, etc.)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

# ── Valid diagram type tokens ──────────────────────────────────────────────────
DIAGRAM_TYPES = ("arch", "flow", "seq")


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog            = "run_stage67.py",
        description     = "Run Stage 6.7 (diagram generation) standalone.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = __doc__,
    )

    parser.add_argument(
        "--resume",
        metavar  = "CONTEXT_FILE",
        required = True,
        help     = "Path to context.json from a pipeline run.",
    )
    parser.add_argument(
        "--output", "-o",
        metavar = "OUTPUT_DIR",
        default = None,
        help    = "Directory where .mmd files are written.  "
                  "Defaults to <run_output_dir>/diagrams/.",
    )
    parser.add_argument(
        "--force",
        action  = "store_true",
        default = False,
        help    = "Re-run even if stage67_diagrams is already marked COMPLETED.",
    )
    parser.add_argument(
        "--only",
        metavar  = "TYPE",
        action   = "append",
        default  = None,
        choices  = DIAGRAM_TYPES,
        help     = f"Generate only this diagram type: {DIAGRAM_TYPES}.  "
                   "Repeat the flag to select multiple types.  Default: all three.",
    )
    parser.add_argument(
        "--no-inject",
        dest    = "no_inject",
        action  = "store_true",
        default = False,
        help    = "Skip injecting Mermaid blocks into the Markdown artefacts.",
    )

    return parser.parse_args()


# ─── Input validation ─────────────────────────────────────────────────────────

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

def _import_stage67() -> object:
    """
    Import stage67_diagrams and return the module.

    Tries the package layout (pipeline/stage67_diagrams.py) first, then falls
    back to a flat layout (stage67_diagrams.py in the same directory).
    """
    try:
        import pipeline.stage67_diagrams as mod
        return mod
    except ImportError:
        pass
    try:
        import stage67_diagrams as mod  # flat layout (dev / test)
        return mod
    except ImportError:
        _die(
            "Could not import stage67_diagrams from pipeline/ or current directory.\n"
            "       Make sure the file exists and context.py is importable."
        )


# ─── Context loading ──────────────────────────────────────────────────────────

def _load_context(args: argparse.Namespace) -> "PipelineContext":
    """Load context.json, apply --force and --output overrides."""
    from context import PipelineContext, StageStatus

    ctx = PipelineContext.load(str(Path(args.resume)))
    print(f"Loaded run : {ctx.run_id}")
    print(f"Output dir : {ctx.output_dir}")

    # Redirect diagram output dir if --output given
    if args.output:
        # We don't change ctx.output_dir (that affects all stage paths).
        # Instead we stash the override so _patch_output_dir() can apply it.
        ctx._diagrams_override = str(Path(args.output).resolve())
        Path(ctx._diagrams_override).mkdir(parents=True, exist_ok=True)
        print(f"Diagrams dir overridden → {ctx._diagrams_override}")

    if args.force:
        ctx.stages["stage67_diagrams"].status = StageStatus.PENDING
        ctx.stages["stage67_diagrams"].error  = None
        print("--force : stage67_diagrams reset to PENDING")

    return ctx


# ─── --only filter: monkey-patch stage67's run() ─────────────────────────────

def _patch_stage67(s67_mod: object, args: argparse.Namespace) -> None:
    """
    When --only and/or --no-inject are given, replace the module's ``run``
    function with a wrapper that skips the unwanted diagram families and/or
    the Markdown injection step.

    Patching the module keeps stage67's internal helpers (all the _build_*
    functions) completely unchanged — we only intercept the top-level
    orchestration inside run().
    """
    only_set   = set(args.only) if args.only else set(DIAGRAM_TYPES)
    no_inject  = args.no_inject

    # Nothing to patch if generating everything and injecting
    if only_set == set(DIAGRAM_TYPES) and not no_inject:
        return

    original_run = s67_mod.run

    def _patched_run(ctx: "PipelineContext") -> None:
        import os
        from pathlib import Path as _Path

        # ── Resolve diagrams dir (honoring --output override) ─────────────────
        diagrams_dir = getattr(ctx, "_diagrams_override", None) or \
                       ctx.output_path(s67_mod.DIAGRAMS_SUBDIR)

        # ── Resume check (mirrors the original) ───────────────────────────────
        if ctx.is_stage_done("stage67_diagrams"):
            existing = list(_Path(diagrams_dir).glob("*.mmd")) \
                       if _Path(diagrams_dir).exists() else []
            if existing:
                print(f"  [stage67] Already completed — {len(existing)} diagrams present.")
                return

        if ctx.domain_model is None:
            raise RuntimeError("[stage67] ctx.domain_model is None — run Stage 4 first.")

        _Path(diagrams_dir).mkdir(parents=True, exist_ok=True)
        generated: list[str] = []
        flows = s67_mod._get_flows(ctx)

        # ── Architecture ──────────────────────────────────────────────────────
        if "arch" in only_set:
            print("  [stage67] Generating architecture diagram ...")
            try:
                mmd  = s67_mod._build_architecture_diagram(ctx)
                path = os.path.join(diagrams_dir, "architecture.mmd")
                _Path(path).write_text(mmd, encoding="utf-8")
                generated.append(path)
                print("  [stage67] ✓ architecture.mmd")
            except Exception as e:
                print(f"  [stage67] ✗ architecture.mmd failed: {e}")
        else:
            print("  [stage67] Skipping architecture diagram (--only filter)")

        # ── Process flow diagrams ─────────────────────────────────────────────
        if "flow" in only_set:
            if flows:
                print(f"  [stage67] Generating process flow diagrams ({len(flows)} flows) ...")
                flow_mmds = s67_mod._build_process_flow_diagrams(flows, ctx.domain_model)
                for slug, mmd_text in flow_mmds.items():
                    path = os.path.join(diagrams_dir, f"flow_{slug}.mmd")
                    _Path(path).write_text(mmd_text, encoding="utf-8")
                    generated.append(path)
                    print(f"  [stage67] ✓ flow_{slug}.mmd")
            else:
                print("  [stage67] No business flows found — skipping process flow diagrams.")
        else:
            print("  [stage67] Skipping process flow diagrams (--only filter)")

        # ── Sequence diagrams ─────────────────────────────────────────────────
        if "seq" in only_set:
            if flows:
                print("  [stage67] Generating sequence diagrams ...")
                for flow in flows:
                    mmd  = s67_mod._build_sequence_diagram(flow)
                    path = os.path.join(diagrams_dir, f"seq_{flow.flow_id}.mmd")
                    _Path(path).write_text(mmd, encoding="utf-8")
                    generated.append(path)
                print(f"  [stage67] ✓ {len(flows)} sequence diagram(s)")
            else:
                print("  [stage67] No business flows found — skipping sequence diagrams.")
        else:
            print("  [stage67] Skipping sequence diagrams (--only filter)")

        # ── Markdown injection ────────────────────────────────────────────────
        if no_inject:
            print("  [stage67] Skipping Markdown injection (--no-inject)")
        else:
            print("  [stage67] Injecting diagrams into Markdown artefacts ...")
            s67_mod._inject_into_markdown(ctx, diagrams_dir, flows or [])

        # ── Mark stage ────────────────────────────────────────────────────────
        ctx.stage("stage67_diagrams").mark_completed(diagrams_dir)
        ctx.save()
        print(f"  [stage67] Complete — {len(generated)} diagram(s) written to {diagrams_dir}/")

    s67_mod.run = _patched_run


# ─── --output: redirect diagrams_dir inside ctx ───────────────────────────────

def _patch_output_dir(s67_mod: object, ctx: "PipelineContext") -> None:
    """
    If --output was given, make ctx.output_path(DIAGRAMS_SUBDIR) return the
    override path by temporarily replacing ctx.output_dir for this one stage.

    We use a lightweight override attribute set in _load_context() instead of
    mutating output_dir globally (which would affect context.json on save).
    """
    # Already stashed on ctx by _load_context — the patched run() reads it directly.
    pass  # nothing to do here; handled inside _patched_run via ctx._diagrams_override


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # 1. Validate before any imports
    _validate_args(args)

    # 2. sys.path
    _setup_path()

    # 3. Import stage67
    s67_mod = _import_stage67()

    # 4. Load context
    ctx = _load_context(args)

    # 5. Pre-flight: domain_model must exist
    if ctx.domain_model is None:
        _die(
            "ctx.domain_model is None in this run.\n"
            "       Stage 4 (domain analysis) must be completed before running stage67.\n"
            f"       Run: python run_pipeline.py --resume {args.resume} --until stage40_domain"
        )

    # 6. Patch module for --only / --no-inject
    _patch_stage67(s67_mod, args)

    # 7. Report what we're about to do
    only_label = ", ".join(sorted(args.only)) if args.only else "all (arch, flow, seq)"
    inject_label = "no" if args.no_inject else "yes"
    flows_available = bool(getattr(ctx, "business_flows", None) and
                           ctx.business_flows and ctx.business_flows.flows)
    print(f"Diagram types  : {only_label}")
    print(f"Inject into MD : {inject_label}")
    print(f"Business flows : {'%d found' % len(ctx.business_flows.flows) if flows_available else 'none (seq+flow diagrams will be skipped)'}")
    print()

    # 8. Run
    try:
        s67_mod.run(ctx)
    except RuntimeError as exc:
        # RuntimeError from stage67 means a hard pre-condition failed
        _die(str(exc))
    except Exception as exc:
        print(f"\nStage 6.7 raised an exception: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # 9. Exit 1 if stage recorded a failure
    if ctx.stages["stage67_diagrams"].status.value == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
