"""
pipeline/stage28_clusters.py — Action Clustering (Stage 2.8)

Groups PHP files into bounded-context clusters using deterministic similarity
signals.  No ML / embedding dependencies.

Algorithm
---------
Phase 1 — Module seeds
    Every file under ``modules/<Name>/`` anchors a cluster named after the module.

Phase 2 — Inverted indices
    Build three lookups from CodeMap data:
      • file → {tables}      (from sql_queries)
      • file → route_prefix  (from routes — first path segment, e.g. "/accounts")
      • file → {redirect_targets} (from redirects)

Phase 3 — Score non-module files against module clusters
    For each non-module file f and each module cluster c:
        score = Σ weights:
            +2  for each shared DB table
            +1  if route prefix matches
            +0.5 for each shared redirect target
    Assign f to the cluster with highest score ≥ ASSIGN_THRESHOLD.

Phase 4 — Directory fallback
    Unclustered files group with siblings in the same parent directory.

Phase 5 — Name & emit
    Module clusters → name = module name (already set).
    Directory clusters → name = last two path segments, e.g. "CalendarProvider".
    Singleton directory clusters are merged into a catch-all "Misc" cluster.

Output
------
    ActionClusterCollection  (written to 2.8_clusters/action_clusters.json)
    ctx.action_clusters set on context

Consumed by
-----------
    Stage 4   (domain model bounded_contexts grounding)
    Stage 4.5 (flow group-by-context primary grouping)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from context import ActionCluster, ActionClusterCollection, PipelineContext

# ── Tunables ──────────────────────────────────────────────────────────────────
ASSIGN_THRESHOLD   = 1.5   # min score to assign a flat file to a module cluster
TABLE_WEIGHT       = 2.0   # score per shared table
ROUTE_PREFIX_WEIGHT= 1.0   # score if route prefix matches
REDIRECT_WEIGHT    = 0.5   # score per shared redirect target
SINGLETON_MIN      = 2     # directory clusters with fewer files → "Misc"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_module_name(filepath: str) -> str | None:
    """Return the module directory name from a SugarCRM/raw-PHP path, or None."""
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part.lower() == "modules" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _route_prefix(route_path: str) -> str:
    """
    Extract the first non-empty path segment as a normalised prefix.
    '/accounts/edit' → '/accounts'
    'accounts'       → '/accounts'
    ''               → ''
    """
    rp = route_path.strip("/")
    segment = rp.split("/")[0] if rp else ""
    return f"/{segment}" if segment else ""


# ── Main entry point ──────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    cm = getattr(ctx, "code_map", None)
    if cm is None:
        print("  [stage28] ⚠️  No code_map available — skipping clustering.")
        ctx.action_clusters = ActionClusterCollection()
        return

    exec_paths = list(cm.execution_paths or [])

    # ── Phase 1: Module seed clusters ─────────────────────────────────────────
    # module_name → list of file paths
    module_to_files: dict[str, list[str]] = defaultdict(list)
    flat_files: list[str] = []   # files not under any modules/<Name>/ dir

    # Collect all unique file paths from execution_paths (Stage 1.5 output)
    all_files: list[str] = sorted({
        ep["file"] for ep in exec_paths if ep.get("file")
    })

    for fpath in all_files:
        mod = _extract_module_name(fpath)
        if mod:
            module_to_files[mod].append(fpath)
        else:
            flat_files.append(fpath)

    print(f"  [stage28] {len(module_to_files)} module clusters seeded "
          f"({len(all_files)} total files, {len(flat_files)} flat).")

    # ── Phase 2: Inverted indices ──────────────────────────────────────────────
    # file → set of tables
    file_tables: dict[str, set[str]] = defaultdict(set)
    for q in (cm.sql_queries or []):
        f = q.get("file", "")
        t = q.get("table", "")
        if f and t and t.upper() not in ("", "UNKNOWN"):
            file_tables[f].add(t.lower())

    # file → route prefix
    file_route_prefix: dict[str, str] = {}
    for r in (cm.routes or []):
        f = r.get("file", "")
        path = r.get("path") or r.get("uri") or ""
        if f and path:
            file_route_prefix[f] = _route_prefix(str(path))

    # file → set of redirect targets (basename only to normalise)
    file_redirects: dict[str, set[str]] = defaultdict(set)
    for rd in (cm.redirects or []):
        src = rd.get("source", "")
        tgt = rd.get("target", "")
        if src and tgt:
            file_redirects[src].add(Path(tgt.split("?")[0]).name.lower())

    # ── Pre-compute cluster-level sets for scoring ────────────────────────────
    # cluster tables = union of tables touched by all files in the module
    cluster_tables: dict[str, set[str]] = {}
    for mod, files in module_to_files.items():
        cluster_tables[mod] = set().union(*(file_tables.get(f, set()) for f in files))

    # cluster route prefixes = most common prefix among module files
    cluster_prefix: dict[str, str] = {}
    for mod, files in module_to_files.items():
        prefix_counts: dict[str, int] = defaultdict(int)
        for f in files:
            pfx = file_route_prefix.get(f, "")
            if pfx:
                prefix_counts[pfx] += 1
        if prefix_counts:
            cluster_prefix[mod] = max(prefix_counts, key=lambda k: prefix_counts[k])
        else:
            cluster_prefix[mod] = f"/{mod.lower()}"

    # cluster redirect basenames = union across module files
    cluster_redirects: dict[str, set[str]] = {}
    for mod, files in module_to_files.items():
        cluster_redirects[mod] = set().union(*(file_redirects.get(f, set()) for f in files))

    # ── Phase 3: Score flat files → module clusters ───────────────────────────
    assigned: dict[str, str] = {}   # file → module name
    for fpath in flat_files:
        f_tables    = file_tables.get(fpath, set())
        f_prefix    = file_route_prefix.get(fpath, "")
        f_redirects = file_redirects.get(fpath, set())

        best_mod:   str   = ""
        best_score: float = 0.0

        for mod in module_to_files:
            score = 0.0
            # shared tables (×2)
            score += TABLE_WEIGHT * len(f_tables & cluster_tables[mod])
            # route prefix match
            if f_prefix and f_prefix == cluster_prefix.get(mod):
                score += ROUTE_PREFIX_WEIGHT
            # shared redirect basenames
            score += REDIRECT_WEIGHT * len(f_redirects & cluster_redirects.get(mod, set()))

            if score > best_score:
                best_score = score
                best_mod   = mod

        if best_score >= ASSIGN_THRESHOLD:
            assigned[fpath] = best_mod

    # ── Phase 4: Directory fallback for unassigned flat files ─────────────────
    unassigned_by_dir: dict[str, list[str]] = defaultdict(list)
    for fpath in flat_files:
        if fpath not in assigned:
            parent = str(Path(fpath).parent)
            unassigned_by_dir[parent].append(fpath)

    # ── Phase 5: Assemble clusters ────────────────────────────────────────────
    clusters: list[ActionCluster] = []
    cluster_seq = 1

    # Module clusters
    for mod, files in sorted(module_to_files.items()):
        # Gather extra files assigned from flat pool
        extra = [f for f, m in assigned.items() if m == mod]
        all_cluster_files = sorted(set(files) | set(extra))
        tables = sorted(cluster_tables.get(mod, set()))
        clusters.append(ActionCluster(
            cluster_id   = f"cluster_{cluster_seq:03d}",
            name         = mod,
            files        = all_cluster_files,
            tables       = tables,
            route_prefix = cluster_prefix.get(mod, ""),
            module       = mod,
            file_count   = len(all_cluster_files),
        ))
        cluster_seq += 1

    # Directory clusters (≥ SINGLETON_MIN files)
    misc_files: list[str] = []
    for parent, files in sorted(unassigned_by_dir.items()):
        if len(files) < SINGLETON_MIN:
            misc_files.extend(files)
            continue
        dir_tables = sorted(set().union(*(file_tables.get(f, set()) for f in files)))
        # Derive a readable name from the last 1-2 path segments
        parts = Path(parent).parts
        name_parts = [p for p in parts[-2:] if p not in (".", "..", "/")]
        name = "/".join(name_parts) if name_parts else parent
        clusters.append(ActionCluster(
            cluster_id   = f"cluster_{cluster_seq:03d}",
            name         = name,
            files        = sorted(files),
            tables       = dir_tables,
            route_prefix = "",
            module       = "",
            file_count   = len(files),
        ))
        cluster_seq += 1

    # Misc catch-all (singletons + truly unassigned)
    if misc_files:
        misc_tables = sorted(set().union(*(file_tables.get(f, set()) for f in misc_files)))
        clusters.append(ActionCluster(
            cluster_id   = f"cluster_{cluster_seq:03d}",
            name         = "Misc",
            files        = sorted(misc_files),
            tables       = misc_tables,
            route_prefix = "",
            module       = "",
            file_count   = len(misc_files),
        ))

    total = len(clusters)
    collection = ActionClusterCollection(
        clusters = clusters,
        total    = total,
    )

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n  [stage28] Action Clustering complete — {total} clusters:")
    for c in clusters[:20]:
        print(f"    [{c.cluster_id}] {c.name:<30} {c.file_count:>4} files  "
              f"tables: {len(c.tables)}")
    if total > 20:
        print(f"    ... and {total - 20} more clusters")
    print()

    # ── Persist ───────────────────────────────────────────────────────────────
    ctx.action_clusters = collection
    try:
        out_path = ctx.output_path("action_clusters.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            import dataclasses
            json.dump(dataclasses.asdict(collection), fh, indent=2, ensure_ascii=False)
        print(f"  [stage28] Saved → {out_path}")
    except Exception as exc:
        print(f"  [stage28] ⚠️  Could not save action_clusters.json: {exc}")
