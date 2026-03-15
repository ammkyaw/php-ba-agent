"""
run_stage3.py — Standalone runner for Stage 3 (Embed)

Usage:
    # Normal run — embed and store chunks
    python run_stage3.py outputs/run_<timestamp>/context.json

    # Dry run — build and print chunks WITHOUT calling OpenAI or ChromaDB
    python run_stage3.py outputs/run_<timestamp>/context.json --dry-run

    # Query the collection after embedding (smoke-test retrieval)
    python run_stage3.py outputs/run_<timestamp>/context.json --query "pages that handle login"

Requires:
    pip install chromadb sentence-transformers
    No API key needed — model runs locally.
"""

from __future__ import annotations

import os
# Silence HuggingFace Hub unauthenticated-request warnings and tokenizer
# parallelism warnings — these are cosmetic and not errors.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Set empty HF_TOKEN to suppress "unauthenticated requests" notice
if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = ""


import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # locate project root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PHP-BA Agent — Stage 3: Embed codebase into ChromaDB")
    p.add_argument("context_file", help="Path to context.json from a completed Stage 1+2 run")
    p.add_argument("--dry-run", action="store_true",
                   help="Build chunks and print manifest without calling OpenAI or ChromaDB")
    p.add_argument("--query", metavar="TEXT",
                   help="After embedding, run a test semantic query and print top results")
    p.add_argument("--n-results", type=int, default=5,
                   help="Number of results to return for --query (default: 5)")
    p.add_argument("--filter-type", metavar="CHUNK_TYPE",
                   help="Filter --query results by chunk_type "
                        "(file_summary|sql_operation|form_inputs|navigation_flow|function_def|graph_neighbourhood)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from context import PipelineContext

    ctx_path = Path(args.context_file)

    # Accept either a context.json (resume) or a project path (new run)
    if ctx_path.suffix == ".json" and ctx_path.exists():
        ctx = PipelineContext.load(str(ctx_path))
        print(f"Resuming run: {ctx.run_id}")
    elif ctx_path.is_dir():
        ctx = PipelineContext.create(php_project_path=str(ctx_path))
        print(f"New run: {ctx.run_id}")
    else:
        print(f"ERROR: Path not found or not a context.json / project dir: {ctx_path}")
        sys.exit(1)

    if ctx.code_map is None:
        print("ERROR: code_map is None — run Stage 1 first.")
        sys.exit(1)

    # ── Dry run: just build and print chunks ─────────────────────────────────
    if args.dry_run:
        import pickle
        from pipeline.stage3_embed import _build_chunks, _build_context_tables

        graph = None
        if ctx.graph_meta and ctx.graph_meta.graph_path:
            try:
                with open(ctx.graph_meta.graph_path, "rb") as fh:
                    graph = pickle.load(fh)
                print(f"Graph loaded: {ctx.graph_meta.node_count} nodes")
            except Exception as e:
                print(f"Warning: could not load graph ({e})")

        chunks = _build_chunks(ctx.code_map, graph)

        from collections import Counter
        type_counts = Counter(c["metadata"]["chunk_type"] for c in chunks)

        print(f"\n{'='*60}")
        print(f"  DRY RUN — {len(chunks)} chunks (no embedding called)")
        print(f"{'='*60}")
        for ctype, count in sorted(type_counts.items()):
            print(f"  {ctype:<35} {count:>3}")

        print(f"\n  All IDs unique: {len({c['id'] for c in chunks}) == len(chunks)}")
        print(f"  Avg text length: {sum(len(c['text']) for c in chunks)//len(chunks)} chars")
        print(f"  Max text length: {max(len(c['text']) for c in chunks)} chars")

        print(f"\n{'='*60}")
        print("  Chunk previews (first of each type):")
        print(f"{'='*60}")
        seen_types: set[str] = set()
        for c in chunks:
            ctype = c["metadata"]["chunk_type"]
            if ctype not in seen_types:
                seen_types.add(ctype)
                preview = c["text"][:300].replace("\n", " | ")
                src     = c["metadata"]["source_file"]
                print(f"\n[{ctype}] {src}")
                print(f"  {preview}")

        # Save manifest to output dir
        manifest_path = ctx.output_path("chunks_manifest_dryrun.json")
        manifest = [
            {
                "chunk_type": c["metadata"]["chunk_type"],
                "source":     c["metadata"]["source_file"],
                "text_len":   len(c["text"]),
                "preview":    c["text"][:150].replace("\n", " "),
                "metadata":   c["metadata"],
            }
            for c in chunks
        ]
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        print(f"\n  Manifest saved → {manifest_path}")
        return

    # ── Normal run ────────────────────────────────────────────────────────────
    from pipeline.stage3_embed import run as stage3_run
    stage3_run(ctx)

    print(f"\nEmbedding complete!")
    print(f"  Collection : {ctx.embedding_meta.collection_name}")
    print(f"  Chunks     : {ctx.embedding_meta.total_chunks}")
    print(f"  ChromaDB   : {ctx.embedding_meta.chroma_path}")
    print(f"  Model      : {ctx.embedding_meta.embedding_model}")

    if not args.query:
        print(f"\nRun Stage 3.5 next:")
        print(f"  python run_stage35.py {ctx.context_file}")

    # ── Optional semantic query test ──────────────────────────────────────────
    if args.query:
        print(f"\n{'='*60}")
        print(f"  Query: \"{args.query}\"")
        if args.filter_type:
            print(f"  Filter: chunk_type = {args.filter_type}")
        print(f"{'='*60}")

        from pipeline.stage3_embed import query_collection
        where = {"chunk_type": args.filter_type} if args.filter_type else None

        results = query_collection(ctx, args.query, n_results=args.n_results, where=where)

        for i, r in enumerate(results, 1):
            ctype  = r["metadata"]["chunk_type"]
            src    = r["metadata"]["source_file"]
            dist   = r["distance"]
            print(f"\n  [{i}] {ctype} | {src} | similarity={1-dist:.3f}")
            print(f"  {'-'*54}")
            # Print first 400 chars of result text
            for line in r["text"][:400].split("\n"):
                print(f"    {line}")
            if len(r["text"]) > 400:
                print(f"    ... [{len(r['text'])-400} chars truncated]")


if __name__ == "__main__":
    main()
