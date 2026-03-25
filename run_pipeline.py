"""
run_pipeline.py — BA Agent Pipeline Entry Point  (multi-language)

Stages (in execution order):
    Stage 0    validate       — input validation
    Stage 0.5  detect         — language + framework auto-detection (NEW)
    Stage 1    parse          — code parsing (PHP / TypeScript / Java dispatcher)
    Stage 1.3  entrypoints    — system entry-point catalog (cron/CLI/webhook/queue, static)
    Stage 1.5  paths          — execution-path / branch extraction
    Stage 2    graph          — knowledge graph (NetworkX)
    Stage 2.2  components     — frontend component graph (Vue/React, skipped for PHP/Java)
    Stage 2.5  behavior       — behavior graph extraction (Route→Ctrl→Service→SQL→Redirect)
    Stage 2.7  semanticroles — semantic role tagging (action type + external systems)
    Stage 3    embed          — vector index (ChromaDB)
    Stage 3.5  entities       — Entity Extraction (static)
    Stage 3.6  relationships  — Entity Relationship Reconstruction (static)
    Stage 3.7  statemachines  — State Machine Reconstruction (static)
    Stage 3.8  graphrag       — graph-aware context index (enriched by 3.5-3.7, static)
    Stage 3.9  preflight      — pre-LLM gate: validates all static context
    Stage 4    domain         — DomainAnalystAgent (LLM)
    Stage 4.5  flows          — BusinessFlowExtractor (LLM)
    Stage 4.6  specrules      — Specification Mining (static + 1 LLM batch)
    Stage 4.7  validate       — Behavioral Flow Validation (deterministic)
    Stage 4.8  triangulate    — Evidence Triangulation (static)
    Stage 5    brd/srs/ac/us  — parallel BA document agents (LLM)
    Stage 5.5  traceability   — Automated Traceability Matrix (static)
    Stage 5.8  doccoverage    — Document Coverage Audit (static)
    Stage 5.9  accuracy       — BA Document Accuracy Report (static)
    Stage 6    qa             — QAReviewAgent (LLM)
    Stage 6.2  architecture   — ArchitectureReconstructionAgent (LLM)
    Stage 6.5  postprocess    — DOCX formatting
    Stage 6.7  diagrams       — Mermaid diagram generation (algorithmic)
    Stage 7    pdf            — DOCX → PDF delivery bundle
    Stage 8    tests          — Test Case Generator (Gherkin + Playwright + pytest)
    Stage 9    knowledge_graph — System Knowledge Graph (cross-domain JSON)

Usage:
    # Full run
    python run_pipeline.py --project /path/to/php-project

    # Resume from a specific stage (skips completed stages automatically)
    python run_pipeline.py --resume outputs/run_20240101_120000/context.json

    # Run only up to a specific stage (useful for testing)
    python run_pipeline.py --project /path/to/php-project --until stage40_domain

    # Force re-run a specific stage even if already completed
    python run_pipeline.py --resume outputs/run_.../context.json --force stage50_brd
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from pathlib import Path
from typing import Optional

from context import PipelineContext, StageStatus
from pipeline.llm_client import set_telemetry_path, write_telemetry_summary

# Stage imports (each is a self-contained module)
from pipeline.stage00_validate     import run as stage0
from pipeline.stage05_detect       import run as stage05
from pipeline.stage10_parse        import run as stage1
from pipeline.stage22_components   import run as stage22
from pipeline.stage13_entrypoints  import run as stage13
from pipeline.stage15_paths        import run as stage15
from pipeline.stage20_graph        import run as stage2
from pipeline.stage25_behavior     import run as stage25
from pipeline.stage27_semanticroles import run as stage27
from pipeline.stage28_clusters     import run as stage28
from pipeline.stage29_invariants   import run as stage29
from pipeline.stage30_embed        import run as stage3
from pipeline.stage35_entities      import run as stage35
from pipeline.stage36_relationships import run as stage36
from pipeline.stage37_statemachines import run as stage37
from pipeline.stage38_graphrag      import run as stage38
from pipeline.stage39_preflight     import run as stage39
from pipeline.stage40_domain       import run as stage4
from pipeline.stage45_flows        import run as stage45
from pipeline.stage46_specrules    import run as stage46
from pipeline.stage47_validate_flows import run as stage47
from pipeline.stage48_triangulate  import run as stage48
from pipeline.stage50_workers      import run as stage5
from pipeline.stage55_traceability import run as stage55
from pipeline.stage58_doccoverage  import run as stage58
from pipeline.stage59_accuracy_report import run as stage59
from pipeline.stage60_qa          import run as stage6
from pipeline.stage62_architecture import run as stage62
from pipeline.stage65_postprocess  import run as stage65
from pipeline.stage67_diagrams     import run as stage67
from pipeline.stage70_pdf         import run as stage7
from pipeline.stage80_tests        import run as stage8
from pipeline.stage90_knowledge_graph import run as stage9


# Stage registry — defines execution order
STAGES: list[tuple[str, any]] = [
    ("stage00_validate",     stage0),
    ("stage05_detect",       stage05),   # language + framework auto-detection
    ("stage10_parse",        stage1),
    ("stage13_entrypoints",  stage13),  # entry-point catalog: cron/cli/webhook/queue (feeds stage15 + stage45 + stage5)
    ("stage15_paths",        stage15),  # execution-path extraction (feeds stage45 + stage5)
    ("stage20_graph",        stage2),
    ("stage22_components",   stage22),   # frontend component graph (skipped for PHP/Java)
    ("stage25_behavior",     stage25),  # behavior graph extraction (feeds stage45 + stage6)
    ("stage27_semanticroles", stage27), # semantic role tagging (feeds stage28 + stage45 + stage9)
    ("stage28_clusters",     stage28),  # action clustering (feeds stage4 + stage45 bounded contexts)
    ("stage29_invariants",   stage29),  # business rule extraction (feeds stage3 chunks + stage4 grounding)
    ("stage30_embed",        stage3),
    ("stage35_entities",      stage35),  # entity extraction (feeds stage36 + stage38 enrichment + stage4 grounding)
    ("stage36_relationships", stage36),  # relationship reconstruction (feeds stage38 enrichment + stage4 + stage67 ER)
    ("stage37_statemachines", stage37),  # state machine reconstruction (feeds stage38 enrichment + stage4 + stage45 + stage67)
    ("stage38_graphrag",      stage38),  # graph-aware context index — enriched by stage35/36/37 (feeds stage4 + stage45 + stage5)
    ("stage39_preflight",     stage39),  # pre-LLM gate — validates all static context before Stage 4
    ("stage40_domain",       stage4),
    ("stage45_flows",        stage45),  # business flow extraction (feeds stage62 + stage67)
    ("stage46_specrules",    stage46),  # specification mining — formal business rules (feeds stage5 + stage9)
    ("stage47_validate",     stage47),  # behavioral validation (are flows missing / valid / real?)
    ("stage48_triangulate",  stage48),  # evidence triangulation — cross-source confidence scoring (feeds stage5 + stage6)
    ("stage50_brd",          None),     # ← parallel group handled below
    ("stage50_srs",          None),
    ("stage50_ac",           None),
    ("stage50_userstories",  None),
    ("stage55_traceability", stage55),  # automated traceability matrix (static)
    ("stage58_doccoverage",     stage58),  # document coverage audit — signals vs documents (static)
    ("stage59_accuracy_report", stage59), # single-file HTML accuracy report (static)
    ("stage60_qa",           stage6),
    ("stage62_architecture", stage62),  # architecture reconstruction (feeds stage65 + stage67)
    ("stage65_postprocess",  stage65),
    ("stage67_diagrams",     stage67),  # diagram generation (needs stage45 flows + stage62 arch)
    ("stage70_pdf",          stage7),   # DOCX → PDF delivery bundle
    ("stage80_tests",        stage8),   # test case generator (Gherkin + Playwright + pytest)
    ("stage90_knowledge_graph", stage9), # system knowledge graph (cross-domain JSON)
]

STAGE5_NAMES = {"stage50_brd", "stage50_srs", "stage50_ac", "stage50_userstories"}


class PipelineBlocker(Exception):
    """Raised when a stage finds a condition that should halt the pipeline."""
    pass


async def run_pipeline(ctx: PipelineContext, until: Optional[str] = None, force: Optional[str] = None) -> None:
    print("\n" + "=" * 60)
    lang = getattr(getattr(ctx, "code_map", None), "language", None)
    lang_label = lang.value.upper() if lang and lang.value != "unknown" else "Auto-Detect"
    print(f"  BA Agent Pipeline  [{lang_label}]")
    print("=" * 60)

    stage5_triggered = False

    for stage_name, stage_fn in STAGES:

        # Handle stage5 parallel group
        if stage_name in STAGE5_NAMES:
            if stage5_triggered:
                continue
            stage5_triggered = True
            group_name = "stage50_[brd|srs|ac|userstories]"
            all_done = all(ctx.is_stage_done(n) for n in STAGE5_NAMES)
            if all_done and force not in STAGE5_NAMES:
                print(f"Skipping {group_name} (all completed)")
                continue
            print(f"Running {group_name} (parallel)...")
            try:
                await stage5(ctx)
                print(f"{group_name} completed\n")
            except Exception as e:
                _handle_stage_error(ctx, list(STAGE5_NAMES), e)
                return
            if until in STAGE5_NAMES:
                break
            continue

        # Skip if already completed (unless forced)
        if ctx.is_stage_done(stage_name) and stage_name != force:
            print(f"Skipping {stage_name} (already completed)")
            continue

        print(f"Running {stage_name}...")
        ctx.stage(stage_name).mark_running()
        ctx.save()

        try:
            if asyncio.iscoroutinefunction(stage_fn):
                await stage_fn(ctx)
            else:
                stage_fn(ctx)
            ctx.stage(stage_name).mark_completed()
            ctx.save()
            print(f"{stage_name} completed\n")

        except PipelineBlocker as e:
            ctx.stage(stage_name).mark_failed(str(e))
            ctx.save()
            print(f"Pipeline blocked at {stage_name}: {e}")
            print(f"Resume with: python run_pipeline.py --resume {ctx.context_file}")
            return

        except Exception as e:
            _handle_stage_error(ctx, [stage_name], e)
            return

        if until and stage_name == until:
            print(f"Stopping at requested stage: {until}")
            break

    # Generate confidence-based human review report after all stages complete
    try:
        from pipeline.review_report import generate as _gen_review
        _gen_review(ctx)
    except Exception as _rev_exc:
        print(f"  [review_report] Warning: could not generate review report: {_rev_exc}")

    # Write LLM call telemetry summary grouped by stage
    try:
        write_telemetry_summary(ctx.output_path("llm_summary.json"))
        print(f"  [telemetry] LLM call summary written → llm_summary.json")
    except Exception as _tel_exc:
        print(f"  [telemetry] Warning: could not write telemetry summary: {_tel_exc}")

    print("\nPipeline complete! Outputs:", ctx.output_dir)


def _handle_stage_error(ctx: PipelineContext, stage_names: list[str], exc: Exception) -> None:
    error_msg = f"{type(exc).__name__}: {exc}"
    for name in stage_names:
        ctx.stage(name).mark_failed(error_msg)
    ctx.save()
    print(f"Stage failed: {error_msg}")
    traceback.print_exc()
    print(f"Resume with: python run_pipeline.py --resume {ctx.context_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHP-BA Agent — generate BA docs from a PHP project.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--project", metavar="PATH", help="Path to the PHP project directory.")
    source.add_argument("--resume", metavar="CONTEXT_FILE", help="Resume from a saved context.json.")
    parser.add_argument("--until", metavar="STAGE_NAME", help="Stop after this stage (inclusive).")
    parser.add_argument("--force", metavar="STAGE_NAME", help="Force re-run this stage.")
    parser.add_argument("--output-dir", metavar="DIR", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid_names = {name for name, _ in STAGES}

    if args.project:
        project_path = Path(args.project)
        if not project_path.exists():
            print(f"Project path not found: {project_path}")
            sys.exit(1)
        ctx = PipelineContext.create(php_project_path=str(project_path), output_base=args.output_dir)
        print(f"New run created: {ctx.run_id}")
    else:
        context_file = Path(args.resume)
        if not context_file.exists():
            print(f"Context file not found: {context_file}")
            sys.exit(1)
        ctx = PipelineContext.load(str(context_file))
        print(f"Resuming run: {ctx.run_id}")

    set_telemetry_path(ctx.output_path("llm_calls.jsonl"))

    for arg_name, arg_val in [("--until", args.until), ("--force", args.force)]:
        if arg_val and arg_val not in valid_names:
            print(f"Unknown stage for {arg_name}: '{arg_val}'. Valid: {sorted(valid_names)}")
            sys.exit(1)

    asyncio.run(run_pipeline(ctx, until=args.until, force=args.force))


if __name__ == "__main__":
    main()
