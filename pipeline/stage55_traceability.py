"""
pipeline/stage55_traceability.py — Stage 5.5 Automated Traceability Matrix

Builds a bidirectional traceability matrix that links every formal requirement
(SpecRule, BusinessFlow) to its exact code evidence and to every section of
the generated BA documents that cites it.

Two passes
----------
Pass A — Forward links  (code → requirement)
    Populated purely from the structured catalogs already on ctx.  For every
    SpecRule the pass records:
      • source_files          → code_artifact type="file"
      • source_invariants     → code_artifact type="invariant"
      • source_flows          → code_artifact type="flow"
      • source_machines       → code_artifact type="state_machine"
      • entities              → code_artifact type="entity"
    For every BusinessFlow (not already covered by a SpecRule) the pass
    records its evidence_files as code artifacts.

Pass B — Backward links  (document → requirement)
    Scans the generated BRD / SRS / AC / User-Story Markdown files for:
      • Exact citation pattern: [BR-XXX] or [TR-BR-XXX]  — high-confidence
      • Keyword overlap: title words ≥ 2 overlap with a paragraph           — low-confidence
    Both match types are stored; only exact matches increment covered_in_*
    flags for Stage 5.9 and Stage 6 consumption.

Outputs
-------
  traceability_matrix.json  — full machine-readable matrix
  traceability_report.md    — human-readable coverage report

Downstream helpers
------------------
  uncited_rules_block(ctx)  — Markdown injection for Stage 6 QA prompt
  traceability_hints()      — Short hint block injected into Stage 5 prompts
                              to instruct LLM agents to embed [BR-XXX] citations.

Zero LLM calls.  Runs after Stage 5 document generation, before Stage 5.9.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Optional

from context import (
    PipelineContext,
    TraceableRequirement,
    TraceabilityMeta,
)

# ── Constants ──────────────────────────────────────────────────────────────────

MATRIX_FILE = "traceability_matrix.json"
REPORT_FILE = "traceability_report.md"

# Regex that matches [BR-001], [BR-123], [TR-BR-001], etc.
_CITATION_RE = re.compile(r"\[(?:TR-)?(?:BR|FR|AR|SR|RL|BL|WF)-\d+\]", re.IGNORECASE)

# Minimum word-overlap for a keyword match (words ≥ 4 chars)
_KW_OVERLAP_MIN = 2
_KW_MIN_WORD_LEN = 4

# Doc names used in coverage flags and reports
_DOC_KEYS = ("brd", "srs", "ac", "us")
_DOC_LABELS = {
    "brd": "BRD",
    "srs": "SRS",
    "ac":  "Acceptance Criteria",
    "us":  "User Stories",
}


# ── Public entry point ─────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Build the Automated Traceability Matrix and attach it to ctx."""
    stage = ctx.stage("stage55_traceability")
    stage.mark_running()

    has_rules = bool(ctx.spec_rules and ctx.spec_rules.rules)
    has_flows = bool(ctx.business_flows and ctx.business_flows.flows)

    if not has_rules and not has_flows:
        print("  [stage55] No spec rules or business flows — skipping traceability.")
        stage.mark_skipped()
        return

    # ── Pass A — Forward links ────────────────────────────────────────────────
    requirements: list[TraceableRequirement] = []
    covered_rule_ids: set[str] = set()

    if has_rules:
        tri_index: dict[str, str] = {}
        if ctx.triangulation:
            tri_index = {
                r.rule_id: r.triangulation_status
                for r in ctx.triangulation.rules
            }

        for rule in ctx.spec_rules.rules:
            tr = TraceableRequirement(
                tr_id           = f"TR-{rule.rule_id}",
                source_type     = "spec_rule",
                source_id       = rule.rule_id,
                title           = rule.title,
                category        = rule.category,
                bounded_context = rule.bounded_context,
                triangulation_status = tri_index.get(rule.rule_id, ""),
            )

            artifacts: list[dict] = []

            # Source files
            for fpath in (rule.source_files or []):
                artifacts.append({"type": "file", "path": fpath})

            # Invariants (Stage 2.9)
            for inv_id in (rule.source_invariants or []):
                artifacts.append({"type": "invariant", "id": inv_id})

            # Business flows (Stage 4.5)
            for fid in (rule.source_flows or []):
                artifacts.append({"type": "flow", "id": fid})
                # Record covered flows so Pass A doesn't double-add them
                covered_rule_ids.add(fid)

            # State machines (Stage 4.3)
            for mid in (rule.source_machines or []):
                artifacts.append({"type": "state_machine", "id": mid})

            # Entities (Stage 4.1)
            for ent in (rule.entities or []):
                artifacts.append({"type": "entity", "table": ent})

            tr.code_artifacts = artifacts
            tr.code_link_count = len(artifacts)
            requirements.append(tr)

    # ── Pass A — Flows not already linked from a SpecRule ─────────────────────
    if has_flows:
        for flow in ctx.business_flows.flows:
            if flow.flow_id in covered_rule_ids:
                continue  # already reachable from a SpecRule
            tr = TraceableRequirement(
                tr_id           = f"TR-F-{flow.flow_id}",
                source_type     = "business_flow",
                source_id       = flow.flow_id,
                title           = flow.name,
                category        = "WORKFLOW",
                bounded_context = flow.bounded_context,
            )
            artifacts = [
                {"type": "file", "path": fpath}
                for fpath in (flow.evidence_files or [])
            ]
            tr.code_artifacts  = artifacts
            tr.code_link_count = len(artifacts)
            requirements.append(tr)

    print(f"  [stage55] Pass A: {len(requirements)} traceable requirements built.")

    # ── Pass B — Backward links from generated documents ─────────────────────
    doc_paths: dict[str, Optional[str]] = {
        "brd": None,
        "srs": None,
        "ac":  None,
        "us":  None,
    }
    if ctx.ba_artifacts:
        doc_paths["brd"] = ctx.ba_artifacts.brd_path
        doc_paths["srs"] = ctx.ba_artifacts.srs_path
        doc_paths["ac"]  = ctx.ba_artifacts.ac_path
        doc_paths["us"]  = ctx.ba_artifacts.user_stories_path

    # Build a citation → TR requirement index (both plain and TR- prefixed)
    tr_citation_index: dict[str, TraceableRequirement] = {}
    for req in requirements:
        # "[BR-001]" style  → TR-BR-001 source_id = "BR-001"
        if req.source_type == "spec_rule":
            tr_citation_index[f"[{req.source_id}]"] = req
            tr_citation_index[f"[TR-{req.source_id}]"] = req

    # Pre-compute keyword sets for keyword-overlap fallback
    kw_index: list[tuple[set[str], TraceableRequirement]] = []
    for req in requirements:
        words = {
            w.lower() for w in re.split(r"\W+", req.title)
            if len(w) >= _KW_MIN_WORD_LEN
        }
        if words:
            kw_index.append((words, req))

    citation_hits = 0
    keyword_hits  = 0

    for doc_key, doc_path in doc_paths.items():
        if not doc_path:
            continue
        p = Path(doc_path)
        if not p.exists():
            continue

        text = p.read_text(encoding="utf-8", errors="replace")
        paragraphs = _split_paragraphs(text)

        for para_idx, para in enumerate(paragraphs):
            para_upper = para.upper()

            # ── Exact citation match ──────────────────────────────────────
            for m in _CITATION_RE.finditer(para):
                raw = m.group(0)
                key_upper = raw.upper()
                # Normalise: [TR-BR-001] → [BR-001]
                norm_key = re.sub(r"^\[TR-", "[", key_upper)
                req = tr_citation_index.get(norm_key) or tr_citation_index.get(key_upper)
                if req is None:
                    continue
                snippet = _snippet(para, m.start())
                req.document_citations.append({
                    "doc":        _DOC_LABELS[doc_key],
                    "match_type": "exact_citation",
                    "snippet":    snippet,
                    "para_index": para_idx,
                })
                _set_covered(req, doc_key)
                citation_hits += 1

            # ── Keyword-overlap fallback ──────────────────────────────────
            para_words = {
                w.lower() for w in re.split(r"\W+", para)
                if len(w) >= _KW_MIN_WORD_LEN
            }
            for kw_set, req in kw_index:
                if len(kw_set & para_words) >= _KW_OVERLAP_MIN:
                    # Avoid adding duplicate citation for same doc/para
                    already = any(
                        c["doc"] == _DOC_LABELS[doc_key] and c["para_index"] == para_idx
                        for c in req.document_citations
                    )
                    if already:
                        continue
                    snippet = para[:120].replace("\n", " ").strip()
                    req.document_citations.append({
                        "doc":        _DOC_LABELS[doc_key],
                        "match_type": "keyword_overlap",
                        "snippet":    snippet,
                        "para_index": para_idx,
                    })
                    keyword_hits += 1
                    # keyword match does NOT set covered_in_* — only exact citations do

    print(f"  [stage55] Pass B: {citation_hits} exact citations, "
          f"{keyword_hits} keyword matches.")

    # ── Compute per-requirement coverage scores ────────────────────────────────
    for req in requirements:
        n_covered = sum([
            req.covered_in_brd,
            req.covered_in_srs,
            req.covered_in_ac,
            req.covered_in_us,
        ])
        req.doc_coverage_score = round(n_covered / 4, 3)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total = len(requirements)
    if total == 0:
        print("  [stage55] No requirements to trace — skipping output.")
        stage.mark_skipped()
        return

    n_brd = sum(1 for r in requirements if r.covered_in_brd)
    n_srs = sum(1 for r in requirements if r.covered_in_srs)
    n_ac  = sum(1 for r in requirements if r.covered_in_ac)
    n_us  = sum(1 for r in requirements if r.covered_in_us)
    total_code_links = sum(r.code_link_count for r in requirements)
    avg_code_links   = round(total_code_links / total, 2)
    uncited = [r for r in requirements if not r.document_citations]
    uncited_ids = [r.tr_id for r in uncited]

    meta = TraceabilityMeta(
        total_requirements = total,
        covered_brd        = round(n_brd / total, 3),
        covered_srs        = round(n_srs / total, 3),
        covered_ac         = round(n_ac  / total, 3),
        covered_us         = round(n_us  / total, 3),
        avg_code_links     = avg_code_links,
        uncited_count      = len(uncited),
        uncited_ids        = uncited_ids,
    )

    # ── Persist matrix JSON ───────────────────────────────────────────────────
    matrix_path = ctx.output_path(MATRIX_FILE)
    matrix_data: dict = {
        "meta":         dataclasses.asdict(meta),
        "requirements": [dataclasses.asdict(r) for r in requirements],
    }

    # ── Append domain-model coverage stats (from stage4 gap-fill) ────────────
    # domain_coverage tracks how many execution paths / tables / pages the
    # domain model actually covers after LLM extraction + gap-fill.  Surfacing
    # it here lets the traceability matrix flag unmapped code artifacts alongside
    # uncited requirements, giving QA engineers a single place to look.
    domain_cov = getattr(ctx, "domain_coverage", None) or {}
    if domain_cov:
        matrix_data["domain_coverage"] = {
            "exec_coverage":    domain_cov.get("exec_coverage",  None),
            "page_coverage":    domain_cov.get("page_coverage",  None),
            "table_coverage":   domain_cov.get("table_coverage", None),
            "field_coverage":   domain_cov.get("field_coverage", None),
            "exec_uncovered":   domain_cov.get("exec_uncovered",   [])[:50],
            "tables_uncovered": domain_cov.get("tables_uncovered", [])[:50],
            "pages_uncovered":  domain_cov.get("pages_uncovered",  [])[:50],
        }

    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix_data, f, indent=2)

    # ── Persist Markdown report ───────────────────────────────────────────────
    report_path = ctx.output_path(REPORT_FILE)
    _write_report(report_path, requirements, meta, domain_cov=domain_cov)

    # ── Attach to context ─────────────────────────────────────────────────────
    meta.matrix_path = matrix_path
    meta.report_path = report_path
    ctx.traceability_meta = meta

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"  [stage55] {total} requirements traced")
    print(f"           BRD={meta.covered_brd:.0%}  SRS={meta.covered_srs:.0%}  "
          f"AC={meta.covered_ac:.0%}  US={meta.covered_us:.0%}")
    print(f"           avg code links={avg_code_links:.1f}  "
          f"uncited={meta.uncited_count}")
    if uncited_ids:
        print(f"           ⚠ Uncited: {', '.join(uncited_ids[:10])}"
              + (" …" if len(uncited_ids) > 10 else ""))
    if domain_cov:
        def _pct(v: object) -> str:
            return f"{v:.0%}" if isinstance(v, float) else "?"
        print(f"           domain coverage — "
              f"exec: {_pct(domain_cov.get('exec_coverage'))}  "
              f"tables: {_pct(domain_cov.get('table_coverage'))}  "
              f"pages: {_pct(domain_cov.get('page_coverage'))}")

    ctx.save()
    stage.mark_completed(output_path=matrix_path)


# ── Public helpers for downstream stages ──────────────────────────────────────

def uncited_rules_block(ctx: PipelineContext, max_rules: int = 30) -> str:
    """
    Return a Markdown block listing requirements with no document coverage.
    Intended for injection into Stage 6 QA prompts as auto-fail targets.
    Empty string if no traceability data or all requirements are cited.
    """
    if not ctx.traceability_meta or not ctx.traceability_meta.uncited_ids:
        return ""

    if not hasattr(ctx, "_traceability_requirements"):
        return ""   # matrix not loaded into memory

    lines: list[str] = ["## ⚠ Uncited Requirements (Traceability Matrix)\n"]
    lines.append(
        "_The following requirements have no citations in any generated BA document. "
        "They must be covered or explicitly deferred._\n"
    )
    for tr_id in ctx.traceability_meta.uncited_ids[:max_rules]:
        req = next(
            (r for r in ctx._traceability_requirements if r.tr_id == tr_id), None
        )
        if req:
            lines.append(
                f"- **{req.tr_id}** [{req.category}] {req.title}  "
                f"_(code links: {req.code_link_count}, "
                f"context: {req.bounded_context})_"
            )
    return "\n".join(lines)


def traceability_hints() -> str:
    """
    Short system-prompt snippet injected into Stage 5 BRD/SRS/AC/US agents.

    Instructs the LLM to embed [BR-XXX] citation tags so Stage 5.5 Pass B
    can build exact backward links without keyword guessing.
    """
    return (
        "## Traceability Instructions\n"
        "After each requirement or acceptance criterion sentence, append the\n"
        "applicable rule ID in square brackets, e.g. `[BR-012]` or `[BR-012, BR-015]`.\n"
        "Use IDs exactly as they appear in the spec rules catalog (BR-XXX format).\n"
        "If multiple rules apply, list all of them separated by commas inside a single pair\n"
        "of brackets: `[BR-001, BR-007]`.\n"
        "Do not fabricate rule IDs — only cite IDs that appear in the provided rule list.\n"
    )


def coverage_summary_block(ctx: PipelineContext) -> str:
    """
    Return a one-paragraph Markdown summary of traceability coverage.
    Used by Stage 5.9 doc-coverage and Stage 6 QA intro sections.
    """
    if not ctx.traceability_meta:
        return ""
    m = ctx.traceability_meta
    return (
        f"**Traceability Coverage** ({m.total_requirements} requirements traced): "
        f"BRD {m.covered_brd:.0%} · SRS {m.covered_srs:.0%} · "
        f"AC {m.covered_ac:.0%} · US {m.covered_us:.0%}. "
        f"Average code links per requirement: {m.avg_code_links:.1f}. "
        f"Uncited requirements: {m.uncited_count}."
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _set_covered(req: TraceableRequirement, doc_key: str) -> None:
    if doc_key == "brd":
        req.covered_in_brd = True
    elif doc_key == "srs":
        req.covered_in_srs = True
    elif doc_key == "ac":
        req.covered_in_ac = True
    elif doc_key == "us":
        req.covered_in_us = True


def _split_paragraphs(text: str) -> list[str]:
    """Split document text into paragraph-like chunks for citation scanning."""
    # Split on blank lines; also treat heading lines as own paragraphs.
    raw = re.split(r"\n{2,}", text)
    result: list[str] = []
    for chunk in raw:
        chunk = chunk.strip()
        if chunk:
            result.append(chunk)
    return result


def _snippet(para: str, match_start: int, radius: int = 80) -> str:
    """Return a short context snippet around the citation match."""
    start = max(0, match_start - radius)
    end   = min(len(para), match_start + radius)
    snip  = para[start:end].replace("\n", " ").strip()
    return snip


def _write_report(
    report_path: str,
    requirements: list[TraceableRequirement],
    meta: TraceabilityMeta,
    domain_cov: dict | None = None,
) -> None:
    """Write the human-readable Markdown traceability report."""
    lines: list[str] = []

    lines.append("# Automated Traceability Matrix\n")
    lines.append(f"_Generated: {meta.generated_at}_\n")

    # ── Coverage summary ────────────────────────────────────────────────────
    lines.append("## Coverage Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total requirements | {meta.total_requirements} |")
    lines.append(f"| BRD coverage       | {meta.covered_brd:.1%} |")
    lines.append(f"| SRS coverage       | {meta.covered_srs:.1%} |")
    lines.append(f"| AC coverage        | {meta.covered_ac:.1%} |")
    lines.append(f"| US coverage        | {meta.covered_us:.1%} |")
    lines.append(f"| Avg code links     | {meta.avg_code_links:.1f} |")
    lines.append(f"| Uncited count      | {meta.uncited_count} |")
    lines.append("")

    # ── Domain-model coverage (from stage4 gap-fill) ────────────────────────
    if domain_cov:
        lines.append("## Domain-Model Coverage (Stage 4)\n")
        lines.append("_How much of the parsed codebase is reflected in the domain model._\n")
        lines.append(f"| Metric | Coverage | Uncovered count |")
        lines.append(f"|--------|----------|-----------------|")
        ec = domain_cov.get("exec_coverage")
        pc = domain_cov.get("page_coverage")
        tc = domain_cov.get("table_coverage")
        fc = domain_cov.get("field_coverage")
        eu = len(domain_cov.get("exec_uncovered",   []))
        pu = len(domain_cov.get("pages_uncovered",  []))
        tu = len(domain_cov.get("tables_uncovered", []))
        if ec is not None:
            lines.append(f"| Execution paths | {ec:.1%} | {eu} |")
        if pc is not None:
            lines.append(f"| HTML pages      | {pc:.1%} | {pu} |")
        if tc is not None:
            lines.append(f"| DB tables       | {tc:.1%} | {tu} |")
        if fc is not None:
            lines.append(f"| POST fields     | {fc:.1%} | — |")
        lines.append("")

        # List uncovered exec-paths (most actionable for gap-fill follow-up)
        exec_unc = domain_cov.get("exec_uncovered", [])
        if exec_unc:
            lines.append("### Unmapped Execution Paths\n")
            lines.append(
                "_These controller/handler files have no corresponding domain feature._\n"
            )
            for fname in exec_unc[:30]:
                lines.append(f"- `{fname}`")
            if len(exec_unc) > 30:
                lines.append(f"- _… {len(exec_unc) - 30} more not shown_")
            lines.append("")

        tables_unc = domain_cov.get("tables_uncovered", [])
        if tables_unc:
            lines.append("### Unmapped DB Tables\n")
            lines.append(
                "_These tables appear in SQL queries but are not referenced in any feature._\n"
            )
            for tname in tables_unc[:30]:
                lines.append(f"- `{tname}`")
            if len(tables_unc) > 30:
                lines.append(f"- _… {len(tables_unc) - 30} more not shown_")
            lines.append("")

    # ── Full matrix table ───────────────────────────────────────────────────
    lines.append("## Full Traceability Matrix\n")
    lines.append(
        "| TR-ID | Category | Title | Context | Code Links | BRD | SRS | AC | US | Score |"
    )
    lines.append(
        "|-------|----------|-------|---------|-----------|-----|-----|----|----|-------|"
    )

    for req in requirements:
        brd = "✓" if req.covered_in_brd else "—"
        srs = "✓" if req.covered_in_srs else "—"
        ac  = "✓" if req.covered_in_ac  else "—"
        us  = "✓" if req.covered_in_us  else "—"
        title_trunc = req.title[:50] + ("…" if len(req.title) > 50 else "")
        tri = f" `{req.triangulation_status}`" if req.triangulation_status else ""
        lines.append(
            f"| {req.tr_id} | {req.category} | {title_trunc}{tri} | "
            f"{req.bounded_context} | {req.code_link_count} | "
            f"{brd} | {srs} | {ac} | {us} | {req.doc_coverage_score:.2f} |"
        )

    lines.append("")

    # ── Uncited requirements ────────────────────────────────────────────────
    uncited = [r for r in requirements if not r.document_citations]
    if uncited:
        lines.append(f"## ⚠ Uncited Requirements ({len(uncited)})\n")
        lines.append(
            "_These requirements have zero citations in any generated document. "
            "Review and add coverage or explicitly defer._\n"
        )
        for req in uncited:
            code_list = ", ".join(
                a.get("path") or a.get("id") or a.get("table") or "?"
                for a in req.code_artifacts[:3]
            )
            tri = f"  _(triangulation: {req.triangulation_status})_" if req.triangulation_status else ""
            lines.append(
                f"- **{req.tr_id}** [{req.category}] {req.title}{tri}  \n"
                f"  _Code evidence: {code_list or 'none'}_"
            )
        lines.append("")

    # ── Per-requirement detail (code artifacts + citations) ──────────────────
    lines.append("## Requirement Detail\n")
    for req in requirements:
        lines.append(f"### {req.tr_id} — {req.title}\n")
        lines.append(f"- **Category:** {req.category}")
        lines.append(f"- **Bounded context:** {req.bounded_context}")
        lines.append(f"- **Source type:** {req.source_type} (`{req.source_id}`)")
        if req.triangulation_status:
            lines.append(f"- **Triangulation:** {req.triangulation_status}")
        lines.append(f"- **Doc coverage:** {req.doc_coverage_score:.0%}"
                     f"  (BRD={'✓' if req.covered_in_brd else '✗'}"
                     f" SRS={'✓' if req.covered_in_srs else '✗'}"
                     f" AC={'✓' if req.covered_in_ac else '✗'}"
                     f" US={'✓' if req.covered_in_us else '✗'})")

        if req.code_artifacts:
            lines.append("\n**Code artifacts:**")
            for art in req.code_artifacts:
                art_type = art.get("type", "")
                art_val  = art.get("path") or art.get("id") or art.get("table") or "?"
                lines.append(f"  - `{art_type}`: {art_val}")

        if req.document_citations:
            lines.append("\n**Document citations:**")
            for cit in req.document_citations:
                match_icon = "🔵" if cit["match_type"] == "exact_citation" else "🟡"
                lines.append(
                    f"  - {match_icon} **{cit['doc']}** "
                    f"({cit['match_type']}): _{cit['snippet'][:80]}_"
                )

        lines.append("")

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
