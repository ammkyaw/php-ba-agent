"""
pipeline/stage59_accuracy_report.py — BA Document Accuracy Report  (Stage 5.9)

Reads every upstream accuracy artifact and produces ONE self-contained HTML
file that answers the single question:

    "How accurately do the generated BA documents reflect the source code?"

Four evidence layers
--------------------
  stage47  flow_validation.json     — V-01–V-08 behavioral checks (is each
                                       flow real, covered, non-hallucinated?)
  stage55  traceability_matrix.json — which BA document sections cite which
                                       code artefact (forward + backward links)
  stage58  doc_coverage.json        — are entities / flows / rules / states /
                                       relations mentioned in the BA docs?
  ctx      business_flows + code_map — raw flows, routes, SQL queries, forms

Report sections
---------------
  1. Accuracy Dashboard     overall score, per-validator status cards
  2. Module Coverage Matrix routes / DB writes / forms / flows per module
  3. Documented Flows       each BA flow with its code evidence + citations
  4. Coverage Gaps          what is in the code but missing from the BA
  5. Hallucination Risks    BA claims with no code backing

Output: accuracy_report.html  (fully self-contained — no external resources)

Zero LLM calls.  Runs after Stage 5.8 (doc coverage).
"""

from __future__ import annotations

import html as _html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from context import PipelineContext
except ImportError:
    from pipeline.context import PipelineContext  # type: ignore

OUTPUT_FILE = "accuracy_report.html"

# ─── Artifact filenames produced by upstream stages ───────────────────────────
_V47_FILE   = "flow_validation.json"
_TR_FILE    = "traceability_matrix.json"
_COV_FILE   = "doc_coverage.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(ctx: PipelineContext) -> None:
    """Generate the single-file BA accuracy report."""

    # ── Load upstream artifacts (gracefully handle missing files) ─────────────
    v47   = _load_json(ctx, _V47_FILE)
    trace = _load_json(ctx, _TR_FILE)
    cov   = _load_json(ctx, _COV_FILE)

    flows = (ctx.business_flows.flows if ctx.business_flows else [])
    cm    = ctx.code_map

    # ctx.domain_coverage is set by stage4 after gap-fill (previously discarded).
    # It carries exec-path / page / table / field coverage from the domain model
    # extraction phase — used here to enrich the Coverage Gaps section.
    domain_cov = getattr(ctx, "domain_coverage", None) or {}

    # ── Derive report data ────────────────────────────────────────────────────
    score        = _composite_score(v47, cov)
    checks       = _extract_checks(v47)
    doc_dims     = _extract_doc_dims(cov)
    module_rows  = _build_module_matrix(cm, flows, v47)
    flow_cards   = _build_flow_cards(flows, trace)
    gaps         = _build_gaps(v47, cov, domain_cov)
    risks        = _build_risks(v47)

    stats = {
        "flows":   len(flows),
        "routes":  len(getattr(cm, "routes", None) or []),
        "writes":  sum(
            1 for q in (getattr(cm, "sql_queries", None) or [])
            if (q.get("operation") or "").upper() in {"INSERT", "UPDATE", "DELETE", "REPLACE"}
        ),
        "forms":   len(getattr(cm, "form_fields", None) or []),
    }

    html = _render_html(
        score, checks, doc_dims, module_rows, flow_cards, gaps, risks, stats, ctx
    )

    out_path = ctx.output_path(OUTPUT_FILE)
    Path(out_path).write_text(html, encoding="utf-8")

    ctx.stage("stage59_accuracy_report").mark_completed(out_path)
    ctx.save()

    print(f"  [stage59] Accuracy report saved → {out_path}  (score {score:.0%})")


# ═══════════════════════════════════════════════════════════════════════════════
# Data extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_json(ctx: PipelineContext, filename: str) -> dict:
    path = ctx.output_path(filename)
    if Path(path).exists():
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _composite_score(v47: dict, cov: dict) -> float:
    """Weighted average of behavioral-validation score and doc-coverage score."""
    v47_score  = v47.get("summary", {}).get("overall_score", 0.0)
    cov_score  = cov.get("overall_pct", 0.0) if cov else 0.0
    if v47 and cov:
        return round(v47_score * 0.6 + cov_score * 0.4, 4)
    if v47:
        return round(v47_score, 4)
    return round(cov_score, 4)


def _extract_checks(v47: dict) -> list[dict]:
    """Return list of {id, name, status, icon, detail, items} dicts."""
    checks = v47.get("checks", {})
    result = []
    for vid in ("V-01", "V-02", "V-03", "V-04", "V-05", "V-06", "V-07", "V-08"):
        c = checks.get(vid, {})
        if not c:
            continue
        result.append({
            "id":     vid,
            "name":   c.get("name", vid),
            "status": c.get("status", "pass"),
            "icon":   c.get("icon", "✅"),
            "detail": c.get("detail", ""),
            "items":  c.get("items") or c.get("uncovered") or c.get("flows") or [],
        })
    return result


def _extract_doc_dims(cov: dict) -> list[dict]:
    """Extract doc-coverage dimensions (entities, flows, spec_rules …)."""
    if not cov:
        return []
    dims = []
    for d in (cov.get("dimensions") or []):
        dims.append({
            "dimension": d.get("dimension", ""),
            "document":  d.get("document", ""),
            "covered":   d.get("covered", 0),
            "total":     d.get("total", 0),
            "pct":       d.get("pct", 0.0),
            "status":    d.get("status", "pass"),
            "uncovered": d.get("uncovered", []),
        })
    return dims


def _module_from_path(fpath: str) -> str:
    """Best-effort: extract a human-readable module name from a file path."""
    parts = Path(fpath).parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == "modules")
        return parts[idx + 1] if len(parts) > idx + 1 else "modules"
    except StopIteration:
        pass
    # Laravel: app/Http/Controllers/FooController.php → Foo
    for kw in ("Controllers", "Http"):
        try:
            idx = next(i for i, p in enumerate(parts) if p == kw)
            stem = Path(fpath).stem
            return stem.replace("Controller", "").replace("Resource", "") or stem
        except StopIteration:
            pass
    # Fallback: top-level dir
    return parts[1] if len(parts) > 1 else Path(fpath).stem


def _build_module_matrix(
    cm: Any, flows: list, v47: dict
) -> list[dict]:
    """Per-module coverage row: routes, DB writes, forms, flows, score."""
    if not cm:
        return []

    routes_all   = getattr(cm, "routes", None)     or []
    queries_all  = getattr(cm, "sql_queries", None) or []
    forms_all    = getattr(cm, "form_fields", None) or []

    # uncovered file sets from V-05/V-06/V-07
    checks = v47.get("checks", {})
    uncov_routes = {
        Path(i.get("file", "")).name.lower()
        for i in (checks.get("V-05", {}).get("uncovered") or [])
    }
    uncov_writes = {
        (i.get("op","").upper(), (i.get("table","") or "").lower())
        for i in (checks.get("V-06", {}).get("uncovered") or [])
    }
    uncov_forms = {
        Path(i.get("file", "")).name.lower()
        for i in (checks.get("V-07", {}).get("uncovered") or [])
    }

    # group routes by module
    route_by_mod: dict[str, list] = {}
    for r in routes_all:
        m = _module_from_path(r.get("file", ""))
        route_by_mod.setdefault(m, []).append(r)

    # group SQL writes by module
    write_by_mod: dict[str, list] = {}
    for q in queries_all:
        if (q.get("operation") or "").upper() not in {"INSERT","UPDATE","DELETE","REPLACE"}:
            continue
        m = _module_from_path(q.get("file", ""))
        write_by_mod.setdefault(m, []).append(q)

    # group forms by module
    form_by_mod: dict[str, list] = {}
    for f in forms_all:
        m = _module_from_path(f.get("file", ""))
        form_by_mod.setdefault(m, []).append(f)

    # group flows by bounded_context
    flow_by_ctx: dict[str, int] = {}
    for fl in flows:
        ctx_name = fl.bounded_context or "Unknown"
        flow_by_ctx[ctx_name] = flow_by_ctx.get(ctx_name, 0) + 1

    all_modules = (
        set(route_by_mod)
        | set(write_by_mod)
        | set(form_by_mod)
        | set(flow_by_ctx)
    )

    rows = []
    for mod in sorted(all_modules):
        mod_routes = route_by_mod.get(mod, [])
        mod_writes = write_by_mod.get(mod, [])
        mod_forms  = form_by_mod.get(mod, [])

        # Route coverage
        r_total   = len(mod_routes)
        r_uncov   = sum(
            1 for r in mod_routes
            if Path(r.get("file","")).name.lower() in uncov_routes
        )
        r_covered = r_total - r_uncov

        # Write coverage
        w_total   = len(mod_writes)
        w_uncov   = sum(
            1 for q in mod_writes
            if ((q.get("operation","").upper(), (q.get("table","") or "").lower()))
               in uncov_writes
        )
        w_covered = w_total - w_uncov

        # Form coverage
        f_total   = len(mod_forms)
        f_uncov   = sum(
            1 for ff in mod_forms
            if Path(ff.get("file","")).name.lower() in uncov_forms
        )
        f_covered = f_total - f_uncov

        fl_count = flow_by_ctx.get(mod, 0)

        # Module score = average of available coverage %s
        pcts = []
        if r_total: pcts.append(r_covered / r_total)
        if w_total: pcts.append(w_covered / w_total)
        if f_total: pcts.append(f_covered / f_total)
        mod_score = sum(pcts) / len(pcts) if pcts else (1.0 if fl_count else 0.0)

        rows.append({
            "module":    mod,
            "r_covered": r_covered, "r_total": r_total,
            "w_covered": w_covered, "w_total": w_total,
            "f_covered": f_covered, "f_total": f_total,
            "flows":     fl_count,
            "score":     round(mod_score, 4),
        })

    # sort worst first
    rows.sort(key=lambda r: r["score"])
    return rows


def _build_flow_cards(flows: list, trace: dict) -> list[dict]:
    """Build rich flow card data including BA doc citations from traceability."""
    # Build lookup: flow_id → list of cited-in BA doc references
    citations: dict[str, list[str]] = {}
    for item in (trace.get("items") or []):
        fid  = item.get("id") or item.get("flow_id") or ""
        docs = []
        for link in (item.get("backward_links") or []):
            doc  = link.get("document", "")
            sect = link.get("section", "")
            if doc:
                docs.append(f"{doc} § {sect}" if sect else doc)
        if fid and docs:
            citations[fid] = docs

    cards = []
    for fl in flows:
        fid  = fl.flow_id or ""
        conf = getattr(fl, "confidence", 1.0) or 1.0

        step_rows = []
        for s in (fl.steps or []):
            step_rows.append({
                "num":      s.step_num,
                "page":     s.page or "",
                "action":   s.action or "",
                "db_ops":   s.db_ops or [],
                "auth":     s.auth_required,
                "inputs":   s.inputs or [],
                "outputs":  s.outputs or [],
            })

        if conf >= 0.8:
            conf_label, conf_cls = "High", "pass"
        elif conf >= 0.5:
            conf_label, conf_cls = "Medium", "warn"
        else:
            conf_label, conf_cls = "Low", "fail"

        cards.append({
            "id":       fid,
            "name":     fl.name or fid,
            "actor":    fl.actor or "",
            "trigger":  fl.trigger or "",
            "context":  fl.bounded_context or "",
            "steps":    step_rows,
            "evidence": fl.evidence_files or [],
            "conf":     conf,
            "conf_lbl": conf_label,
            "conf_cls": conf_cls,
            "cited_in": citations.get(fid, []),
        })

    # sort: high-confidence first
    cards.sort(key=lambda c: -c["conf"])
    return cards


def _build_gaps(v47: dict, cov: dict, domain_cov: dict | None = None) -> dict:
    """Collect all gap lists from V-05/V-06/V-07, doc coverage dimensions,
    and the domain-model exec-path / table / field coverage report."""
    checks = v47.get("checks", {})

    uncov_routes = checks.get("V-05", {}).get("uncovered") or []
    uncov_writes = checks.get("V-06", {}).get("uncovered") or []
    uncov_forms  = checks.get("V-07", {}).get("uncovered") or []

    # Doc coverage gaps (entities, flows, spec_rules, states, relations missing from docs)
    doc_gaps: list[dict] = []
    for d in (cov.get("dimensions") or []) if cov else []:
        for name in (d.get("uncovered") or []):
            doc_gaps.append({
                "type": d.get("dimension", ""),
                "doc":  d.get("document", ""),
                "name": name,
            })

    # Domain-model coverage gaps — pages / tables / fields the domain model
    # did not map to any feature (from ctx.domain_coverage set by stage4).
    # Previously this data was computed and then discarded; now surfaces here.
    uncov_pages  = (domain_cov or {}).get("pages_uncovered", []) if domain_cov else []
    uncov_tables = (domain_cov or {}).get("tables_uncovered", []) if domain_cov else []
    uncov_fields = (domain_cov or {}).get("fields_uncovered", []) if domain_cov else []
    domain_cov_summary = {}
    if domain_cov:
        domain_cov_summary = {
            "exec_coverage":  domain_cov.get("exec_coverage", 0.0),
            "page_coverage":  domain_cov.get("page_coverage", 0.0),
            "table_coverage": domain_cov.get("table_coverage", 0.0),
            "field_coverage": domain_cov.get("field_coverage", 0.0),
        }

    return {
        "routes":       uncov_routes,
        "writes":       uncov_writes,
        "forms":        uncov_forms,
        "docs":         doc_gaps,
        "pages":        uncov_pages,
        "tables":       uncov_tables,
        "fields":       uncov_fields,
        "domain_stats": domain_cov_summary,
    }


def _build_risks(v47: dict) -> dict:
    """Collect hallucination and structural-issue items from V-02/V-03/V-04/V-08."""
    checks = v47.get("checks", {})
    return {
        "hallucinated": checks.get("V-02", {}).get("items") or [],
        "broken":       checks.get("V-03", {}).get("items") or [],
        "no_branches":  checks.get("V-04", {}).get("items") or [],
        "low_conf":     [
            f for f in (checks.get("V-08", {}).get("flows") or [])
            if (f.get("confidence") or 0) < 0.5
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HTML renderer
# ═══════════════════════════════════════════════════════════════════════════════

def _e(s: Any) -> str:
    """HTML-escape a value."""
    return _html.escape(str(s))


def _pct_bar(covered: int, total: int) -> str:
    if not total:
        return '<span class="na">N/A</span>'
    pct = covered / total
    cls = "pass" if pct >= 0.8 else ("warn" if pct >= 0.5 else "fail")
    return (
        f'<div class="bar-wrap" title="{covered}/{total}">'
        f'<div class="bar {cls}" style="width:{pct*100:.0f}%"></div>'
        f'</div><span class="bar-lbl">{pct:.0%}</span>'
    )


def _status_cls(status: str) -> str:
    return {"pass": "pass", "warn": "warn", "fail": "fail"}.get(status, "pass")


def _render_check_cards(checks: list[dict]) -> str:
    parts = []
    for c in checks:
        cls = _status_cls(c["status"])
        parts.append(
            f'<div class="check-card {cls}">'
            f'<span class="check-id">{_e(c["id"])}</span>'
            f'<span class="check-icon">{_e(c["icon"])}</span>'
            f'<div class="check-name">{_e(c["name"])}</div>'
            f'<div class="check-detail">{_e(c["detail"])}</div>'
            f'</div>'
        )
    return "\n".join(parts)


def _render_doc_dim_pills(dims: list[dict]) -> str:
    parts = []
    for d in dims:
        cls = _status_cls(d["status"])
        pct = d["pct"]
        parts.append(
            f'<span class="dim-pill {cls}" '
            f'title="{_e(d["dimension"])} in {_e(d["document"])}: '
            f'{d["covered"]}/{d["total"]}">'
            f'{_e(d["dimension"].capitalize())} → {_e(d["document"])} '
            f'<strong>{pct:.0%}</strong>'
            f'</span>'
        )
    return "\n".join(parts)


def _render_module_matrix(rows: list[dict]) -> str:
    if not rows:
        return '<p class="empty">No module data available.</p>'

    thead = (
        "<tr>"
        "<th>Module</th>"
        "<th>Routes</th>"
        "<th>DB Writes</th>"
        "<th>Forms</th>"
        "<th>Flows</th>"
        "<th>Score</th>"
        "</tr>"
    )
    tbodies = []
    for r in rows:
        cls = "pass" if r["score"] >= 0.8 else ("warn" if r["score"] >= 0.5 else "fail")
        tbodies.append(
            f"<tr>"
            f"<td class='mod-name'>{_e(r['module'])}</td>"
            f"<td>{_pct_bar(r['r_covered'], r['r_total'])}</td>"
            f"<td>{_pct_bar(r['w_covered'], r['w_total'])}</td>"
            f"<td>{_pct_bar(r['f_covered'], r['f_total'])}</td>"
            f"<td class='center'>{r['flows']}</td>"
            f"<td><span class='score-badge-sm {cls}'>{r['score']:.0%}</span></td>"
            f"</tr>"
        )
    return f"<table class='matrix'><thead>{thead}</thead><tbody>{''.join(tbodies)}</tbody></table>"


def _render_flow_cards(cards: list[dict]) -> str:
    if not cards:
        return '<p class="empty">No business flows found.</p>'

    parts = []
    for c in cards:
        # Steps table
        step_rows_html = ""
        for s in c["steps"]:
            auth_badge = ' <span class="auth-badge">🔒 Auth</span>' if s["auth"] else ""
            db_ops_str = ", ".join(s["db_ops"]) if s["db_ops"] else "—"
            inputs_str = ", ".join(s["inputs"]) if s["inputs"] else "—"
            step_rows_html += (
                f"<tr>"
                f"<td class='step-num'>{s['num']}</td>"
                f"<td>{_e(s['page'])}{auth_badge}</td>"
                f"<td>{_e(s['action'])}</td>"
                f"<td class='mono small'>{_e(db_ops_str)}</td>"
                f"<td class='mono small'>{_e(inputs_str)}</td>"
                f"</tr>"
            )

        # Evidence files
        ev_html = " ".join(
            f'<span class="ev-tag">{_e(Path(ef).name)}</span>'
            for ef in c["evidence"]
        ) or '<span class="na">none</span>'

        # BA doc citations
        cite_html = ""
        if c["cited_in"]:
            cite_html = (
                '<div class="citations"><strong>📄 Cited in:</strong> '
                + " &bull; ".join(_e(ci) for ci in c["cited_in"])
                + "</div>"
            )
        else:
            cite_html = '<div class="citations warn-txt">⚠ Not yet cited in any BA document</div>'

        parts.append(f"""
<details class="flow-card {c['conf_cls']}">
  <summary>
    <span class="flow-id">{_e(c['id'])}</span>
    <span class="flow-name">{_e(c['name'])}</span>
    <span class="flow-meta">{_e(c['actor'])} · {_e(c['context'])}</span>
    <span class="conf-badge {c['conf_cls']}">{c['conf_lbl']} ({c['conf']:.0%})</span>
  </summary>
  <div class="flow-body">
    <div class="flow-header-row">
      <div><strong>Trigger:</strong> {_e(c['trigger'])}</div>
      <div><strong>Bounded context:</strong> {_e(c['context'])}</div>
    </div>
    <table class="steps-table">
      <thead><tr>
        <th>#</th><th>Page / Route</th><th>Action</th>
        <th>DB Operations</th><th>Inputs</th>
      </tr></thead>
      <tbody>{step_rows_html}</tbody>
    </table>
    <div class="evidence-row"><strong>📎 Code evidence:</strong> {ev_html}</div>
    {cite_html}
  </div>
</details>""")

    return "\n".join(parts)


def _render_gaps(gaps: dict) -> str:
    sections = []

    # Routes
    if gaps["routes"]:
        rows = "".join(
            f"<tr><td>{_e(i.get('method','?'))}</td>"
            f"<td class='mono'>{_e(i.get('path',''))}</td>"
            f"<td class='small'>{_e(i.get('file',''))}</td></tr>"
            for i in gaps["routes"]
        )
        sections.append(
            f'<h3>🔗 Routes with no BA flow ({len(gaps["routes"])})</h3>'
            f'<table class="gap-table"><thead>'
            f'<tr><th>Method</th><th>Path</th><th>File</th></tr>'
            f'</thead><tbody>{rows}</tbody></table>'
        )

    # DB Writes
    if gaps["writes"]:
        rows = "".join(
            f"<tr>"
            f'<td><span class="sev-badge {_e(i.get("severity","major"))}">'
            f'{_e(i.get("severity","major").upper())}</span></td>'
            f"<td>{_e(i.get('op','?'))}</td>"
            f"<td class='mono'>{_e(i.get('table','?'))}</td>"
            f"<td class='small'>{_e(i.get('file',''))}</td>"
            f"</tr>"
            for i in gaps["writes"]
        )
        sections.append(
            f'<h3>🗄 Database writes with no BA flow ({len(gaps["writes"])})</h3>'
            f'<table class="gap-table"><thead>'
            f'<tr><th>Severity</th><th>Op</th><th>Table</th><th>File</th></tr>'
            f'</thead><tbody>{rows}</tbody></table>'
        )

    # Forms
    if gaps["forms"]:
        rows = "".join(
            f"<tr><td class='mono'>{_e(i.get('file',''))}</td>"
            f"<td>{_e(i.get('action',''))}</td>"
            f"<td>{_e(i.get('method',''))}</td></tr>"
            for i in gaps["forms"]
        )
        sections.append(
            f'<h3>📋 Forms with no BA flow ({len(gaps["forms"])})</h3>'
            f'<table class="gap-table"><thead>'
            f'<tr><th>File</th><th>Action</th><th>Method</th></tr>'
            f'</thead><tbody>{rows}</tbody></table>'
        )

    # Doc gaps
    if gaps["docs"]:
        by_type: dict[str, list[str]] = {}
        for g in gaps["docs"]:
            key = f'{g["type"]} → {g["doc"]}'
            by_type.setdefault(key, []).append(g["name"])
        grp_html = ""
        for label, names in by_type.items():
            tags = " ".join(f'<span class="ev-tag warn">{_e(n)}</span>' for n in names)
            grp_html += f'<div class="doc-gap-grp"><strong>{_e(label)}:</strong> {tags}</div>'
        sections.append(
            f'<h3>📄 Items missing from BA documents ({len(gaps["docs"])})</h3>'
            f'<div class="doc-gaps">{grp_html}</div>'
        )

    # Domain-model coverage gaps (from ctx.domain_coverage — stage4 final pass)
    # Surfaces exec-paths, tables, and POST fields not mapped to any feature.
    ds = gaps.get("domain_stats", {})
    if ds:
        stat_html = (
            f'<div class="dom-cov-stats">'
            f'<span title="Exec-path coverage">Exec-paths: <strong>{ds.get("exec_coverage",0):.0%}</strong></span> &nbsp;|&nbsp; '
            f'<span title="HTML page coverage">Pages: <strong>{ds.get("page_coverage",0):.0%}</strong></span> &nbsp;|&nbsp; '
            f'<span title="Table coverage">Tables: <strong>{ds.get("table_coverage",0):.0%}</strong></span> &nbsp;|&nbsp; '
            f'<span title="POST field coverage">Fields: <strong>{ds.get("field_coverage",0):.0%}</strong></span>'
            f'</div>'
        )
        uncov_pages  = gaps.get("pages", [])
        uncov_tables = gaps.get("tables", [])
        uncov_fields = gaps.get("fields", [])
        detail_html = ""
        if uncov_pages:
            tags = " ".join(f'<span class="ev-tag warn">{_e(p)}</span>' for p in uncov_pages[:30])
            more = f" <em>+{len(uncov_pages)-30} more</em>" if len(uncov_pages) > 30 else ""
            detail_html += f'<p><strong>Uncovered exec-paths / pages:</strong> {tags}{more}</p>'
        if uncov_tables:
            tags = " ".join(f'<span class="ev-tag warn">{_e(t)}</span>' for t in uncov_tables[:30])
            more = f" <em>+{len(uncov_tables)-30} more</em>" if len(uncov_tables) > 30 else ""
            detail_html += f'<p><strong>Uncovered tables:</strong> {tags}{more}</p>'
        if uncov_fields:
            tags = " ".join(f'<span class="ev-tag warn">{_e(f)}</span>' for f in uncov_fields[:30])
            more = f" <em>+{len(uncov_fields)-30} more</em>" if len(uncov_fields) > 30 else ""
            detail_html += f'<p><strong>Uncovered POST fields:</strong> {tags}{more}</p>'
        if stat_html or detail_html:
            sections.append(
                f'<h3>📊 Domain-model coverage after gap-fill</h3>'
                f'<p class="hint">Coverage of codebase artefacts by the domain model '
                f'features generated in Stage 4 (post gap-fill pass).</p>'
                f'{stat_html}{detail_html}'
            )

    if not sections:
        return '<p class="empty pass-txt">✅ No coverage gaps detected.</p>'
    return "\n".join(sections)


def _render_risks(risks: dict) -> str:
    parts = []

    if risks["hallucinated"]:
        rows = "".join(
            f"<tr><td>{_e(i.get('flow_id',''))}</td>"
            f"<td>{_e(i.get('flow_name',''))}</td>"
            f"<td>{_e(i.get('step',''))}</td>"
            f"<td class='small'>{_e(i.get('issue',''))}</td></tr>"
            for i in risks["hallucinated"]
        )
        parts.append(
            f'<h3>🚨 Hallucinated steps ({len(risks["hallucinated"])})</h3>'
            '<p class="hint">These BA steps reference tables, routes, or entities '
            'not found in the parsed source code.</p>'
            '<table class="gap-table"><thead>'
            '<tr><th>Flow ID</th><th>Flow Name</th><th>Step</th><th>Issue</th></tr>'
            f'</thead><tbody>{rows}</tbody></table>'
        )

    if risks["broken"]:
        rows = "".join(
            f"<tr><td>{_e(i.get('flow_id',''))}</td>"
            f"<td>{_e(i.get('flow_name',''))}</td>"
            f"<td>{_e(i.get('severity',''))}</td>"
            f"<td class='small'>{_e(i.get('issue',''))}</td></tr>"
            for i in risks["broken"]
        )
        parts.append(
            f'<h3>⛓ Structurally broken flows ({len(risks["broken"])})</h3>'
            '<p class="hint">Flows with only one step or no route→controller→DB chain.</p>'
            '<table class="gap-table"><thead>'
            '<tr><th>Flow ID</th><th>Flow Name</th><th>Severity</th><th>Issue</th></tr>'
            f'</thead><tbody>{rows}</tbody></table>'
        )

    if risks["low_conf"]:
        rows = "".join(
            f"<tr><td>{_e(f.get('flow_id',''))}</td>"
            f"<td>{_e(f.get('name',''))}</td>"
            f"<td>{f.get('confidence',0):.0%}</td>"
            f"<td class='small'>{_e(f.get('note',''))}</td></tr>"
            for f in risks["low_conf"]
        )
        parts.append(
            f'<h3>⚠ Low-confidence flows ({len(risks["low_conf"])})</h3>'
            '<p class="hint">These flows were generated with &lt;50% confidence '
            '— fewer backing files or steps than expected.  Review before including '
            'in final BA documents.</p>'
            '<table class="gap-table"><thead>'
            '<tr><th>Flow ID</th><th>Name</th><th>Confidence</th><th>Note</th></tr>'
            f'</thead><tbody>{rows}</tbody></table>'
        )

    if not parts:
        return '<p class="empty pass-txt">✅ No hallucination risks detected.</p>'
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Main HTML template
# ═══════════════════════════════════════════════════════════════════════════════

def _render_html(
    score:       float,
    checks:      list[dict],
    doc_dims:    list[dict],
    module_rows: list[dict],
    flow_cards:  list[dict],
    gaps:        dict,
    risks:       dict,
    stats:       dict,
    ctx:         PipelineContext,
) -> str:

    score_cls = "pass" if score >= 0.8 else ("warn" if score >= 0.5 else "fail")
    score_pct = f"{score:.0%}"
    ts        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    total_gaps = (
        len(gaps["routes"]) + len(gaps["writes"]) +
        len(gaps["forms"]) + len(gaps["docs"]) +
        len(gaps.get("pages", [])) + len(gaps.get("tables", []))
    )
    total_risks = (
        len(risks["hallucinated"]) + len(risks["broken"]) + len(risks["low_conf"])
    )

    check_cards_html = _render_check_cards(checks)
    doc_dim_html     = _render_doc_dim_pills(doc_dims)
    matrix_html      = _render_module_matrix(module_rows)
    flows_html       = _render_flow_cards(flow_cards)
    gaps_html        = _render_gaps(gaps)
    risks_html       = _render_risks(risks)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BA Accuracy Report</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       font-size: 14px; background: #f4f5f7; color: #1a1a2e; line-height: 1.5; }}

/* ── Top bar ─────────────────────────────────────────────────────────────── */
.topbar {{ background: #1a1a2e; color: #fff; padding: 0 2rem;
           display: flex; align-items: center; gap: 1.5rem;
           position: sticky; top: 0; z-index: 100; height: 56px; }}
.topbar h1 {{ font-size: 1.1rem; font-weight: 600; flex: 1; }}
.topbar .meta {{ font-size: .75rem; opacity: .65; }}
.score-hero {{ font-size: 1.4rem; font-weight: 700; padding: .2rem .8rem;
               border-radius: 6px; }}
.score-hero.pass {{ background:#1a6b3c; color:#fff; }}
.score-hero.warn {{ background:#9a6200; color:#fff; }}
.score-hero.fail {{ background:#8b1a1a; color:#fff; }}

/* ── Nav tabs ────────────────────────────────────────────────────────────── */
.nav {{ background:#2d2d4e; padding: 0 2rem; display:flex; gap: .25rem; }}
.nav a {{ color:#aab; font-size:.8rem; padding:.5rem .9rem;
          text-decoration:none; border-bottom:2px solid transparent;
          transition: all .15s; }}
.nav a:hover {{ color:#fff; border-bottom-color:#7b8cde; }}

/* ── Layout ──────────────────────────────────────────────────────────────── */
.container {{ max-width: 1400px; margin: 0 auto; padding: 1.5rem 2rem; }}
section {{ margin-bottom: 2.5rem; }}
h2 {{ font-size: 1rem; font-weight: 700; color: #1a1a2e;
      border-left: 4px solid #7b8cde; padding-left: .6rem;
      margin-bottom: 1rem; }}
h3 {{ font-size: .9rem; font-weight: 600; margin: 1.2rem 0 .5rem; color: #333; }}
.hint {{ font-size: .8rem; color: #666; margin-bottom: .5rem; }}

/* ── Stat strip ──────────────────────────────────────────────────────────── */
.stat-strip {{ display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.5rem; }}
.stat-card {{ background:#fff; border-radius:8px; padding:.8rem 1.2rem;
              flex:1; min-width:120px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
.stat-card .val {{ font-size:1.8rem; font-weight:700; color:#1a1a2e; }}
.stat-card .lbl {{ font-size:.75rem; color:#666; text-transform:uppercase; }}

/* ── Check cards ─────────────────────────────────────────────────────────── */
.check-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
               gap:.75rem; }}
.check-card {{ border-radius:8px; padding:.9rem 1rem; border-left: 4px solid #ccc;
               background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.07); }}
.check-card.pass {{ border-left-color: #22c55e; }}
.check-card.warn {{ border-left-color: #f59e0b; }}
.check-card.fail {{ border-left-color: #ef4444; }}
.check-id {{ font-size:.75rem; font-weight:700; color:#888; }}
.check-icon {{ float:right; font-size:1.1rem; }}
.check-name {{ font-weight:600; font-size:.85rem; margin:.2rem 0; }}
.check-detail {{ font-size:.78rem; color:#555; }}

/* ── Doc dim pills ───────────────────────────────────────────────────────── */
.dim-pill {{ display:inline-block; padding:.25rem .6rem; border-radius:12px;
             font-size:.78rem; margin:.2rem; }}
.dim-pill.pass {{ background:#dcfce7; color:#166534; }}
.dim-pill.warn {{ background:#fef9c3; color:#854d0e; }}
.dim-pill.fail {{ background:#fee2e2; color:#991b1b; }}

/* ── Module matrix ───────────────────────────────────────────────────────── */
.matrix {{ width:100%; border-collapse:collapse; background:#fff;
           border-radius:8px; overflow:hidden;
           box-shadow:0 1px 4px rgba(0,0,0,.08); }}
.matrix th {{ background:#2d2d4e; color:#fff; padding:.5rem .8rem;
              font-size:.8rem; text-align:left; }}
.matrix td {{ padding:.45rem .8rem; border-bottom:1px solid #f0f0f0;
              font-size:.82rem; }}
.matrix tr:last-child td {{ border-bottom: none; }}
.matrix tr:hover td {{ background: #f9f9ff; }}
.mod-name {{ font-weight:600; white-space:nowrap; }}
.center {{ text-align:center; }}

/* ── Progress bars ───────────────────────────────────────────────────────── */
.bar-wrap {{ display:inline-block; width:80px; height:8px;
             background:#e5e7eb; border-radius:4px; vertical-align:middle;
             margin-right:.35rem; }}
.bar {{ height:100%; border-radius:4px; }}
.bar.pass {{ background:#22c55e; }}
.bar.warn {{ background:#f59e0b; }}
.bar.fail {{ background:#ef4444; }}
.bar-lbl {{ font-size:.78rem; color:#444; }}
.na {{ color:#aaa; font-size:.78rem; }}

/* ── Score badges ────────────────────────────────────────────────────────── */
.score-badge-sm {{ display:inline-block; padding:.15rem .5rem; border-radius:4px;
                   font-size:.78rem; font-weight:700; }}
.score-badge-sm.pass {{ background:#dcfce7; color:#166534; }}
.score-badge-sm.warn {{ background:#fef9c3; color:#854d0e; }}
.score-badge-sm.fail {{ background:#fee2e2; color:#991b1b; }}

/* ── Flow cards ──────────────────────────────────────────────────────────── */
details.flow-card {{ background:#fff; border-radius:8px; margin-bottom:.6rem;
                     border-left:4px solid #ccc;
                     box-shadow:0 1px 3px rgba(0,0,0,.07); }}
details.flow-card.pass {{ border-left-color: #22c55e; }}
details.flow-card.warn {{ border-left-color: #f59e0b; }}
details.flow-card.fail {{ border-left-color: #ef4444; }}
details.flow-card summary {{ padding:.7rem 1rem; cursor:pointer;
                              display:flex; align-items:center; gap:.6rem;
                              list-style:none; }}
details.flow-card summary::-webkit-details-marker {{ display:none; }}
details[open].flow-card summary {{ border-bottom:1px solid #f0f0f0; }}
.flow-id {{ font-size:.72rem; color:#888; font-weight:600; min-width:60px; }}
.flow-name {{ font-weight:600; font-size:.88rem; flex:1; }}
.flow-meta {{ font-size:.75rem; color:#777; }}
.conf-badge {{ font-size:.72rem; font-weight:700; padding:.15rem .5rem;
               border-radius:4px; white-space:nowrap; }}
.conf-badge.pass {{ background:#dcfce7; color:#166534; }}
.conf-badge.warn {{ background:#fef9c3; color:#854d0e; }}
.conf-badge.fail {{ background:#fee2e2; color:#991b1b; }}
.flow-body {{ padding:.9rem 1rem 1rem; }}
.flow-header-row {{ display:flex; gap:2rem; font-size:.8rem; color:#555;
                    margin-bottom:.75rem; flex-wrap:wrap; }}

/* ── Steps table ─────────────────────────────────────────────────────────── */
.steps-table {{ width:100%; border-collapse:collapse; margin-bottom:.75rem;
                font-size:.8rem; }}
.steps-table th {{ background:#f0f1fa; padding:.35rem .6rem;
                   text-align:left; font-weight:600; color:#333; }}
.steps-table td {{ padding:.3rem .6rem; border-bottom:1px solid #f5f5f5; }}
.step-num {{ font-weight:700; color:#7b8cde; text-align:center; width:30px; }}
.auth-badge {{ font-size:.7rem; background:#fef9c3; color:#854d0e;
               padding:.1rem .35rem; border-radius:3px; margin-left:.3rem; }}

/* ── Evidence / citations ────────────────────────────────────────────────── */
.evidence-row {{ font-size:.8rem; margin-bottom:.4rem; }}
.ev-tag {{ display:inline-block; background:#eff6ff; color:#1d4ed8;
           padding:.1rem .45rem; border-radius:4px; font-size:.75rem;
           margin:.1rem; font-family:monospace; }}
.ev-tag.warn {{ background:#fef9c3; color:#854d0e; }}
.citations {{ font-size:.78rem; color:#555; }}
.warn-txt {{ color:#b45309; }}
.pass-txt {{ color:#166534; }}

/* ── Gap / risk tables ───────────────────────────────────────────────────── */
.gap-table {{ width:100%; border-collapse:collapse; font-size:.8rem;
              background:#fff; border-radius:6px; overflow:hidden;
              box-shadow:0 1px 3px rgba(0,0,0,.06); margin-bottom:1rem; }}
.gap-table th {{ background:#2d2d4e; color:#fff; padding:.4rem .7rem;
                 text-align:left; font-size:.78rem; }}
.gap-table td {{ padding:.35rem .7rem; border-bottom:1px solid #f5f5f5; }}
.sev-badge {{ display:inline-block; padding:.1rem .4rem; border-radius:3px;
              font-size:.72rem; font-weight:700; }}
.sev-badge.critical {{ background:#fee2e2; color:#991b1b; }}
.sev-badge.major    {{ background:#fef3c7; color:#92400e; }}
.sev-badge.minor    {{ background:#ecfdf5; color:#065f46; }}
.doc-gap-grp {{ margin:.3rem 0; font-size:.8rem; }}
.doc-gaps {{ padding:.5rem 0; }}

/* ── Misc ────────────────────────────────────────────────────────────────── */
.mono  {{ font-family: "SFMono-Regular", Consolas, monospace; }}
.small {{ font-size:.75rem; color:#666; }}
.empty {{ font-size:.85rem; color:#888; padding: .5rem 0; }}
</style>
</head>
<body>

<!-- ── Top bar ──────────────────────────────────────────────────────────── -->
<div class="topbar">
  <h1>BA Document Accuracy Report</h1>
  <span class="score-hero {score_cls}">{score_pct}</span>
  <span class="meta">Generated {_e(ts)}</span>
</div>

<!-- ── Nav ──────────────────────────────────────────────────────────────── -->
<nav class="nav">
  <a href="#dashboard">Dashboard</a>
  <a href="#matrix">Module Matrix</a>
  <a href="#flows">Documented Flows</a>
  <a href="#gaps">Coverage Gaps</a>
  <a href="#risks">Hallucination Risks</a>
</nav>

<div class="container">

<!-- ── Section 1: Dashboard ──────────────────────────────────────────────── -->
<section id="dashboard">
  <h2>Accuracy Dashboard</h2>

  <div class="stat-strip">
    <div class="stat-card">
      <div class="val">{stats['flows']}</div>
      <div class="lbl">BA Flows</div>
    </div>
    <div class="stat-card">
      <div class="val">{stats['routes']}</div>
      <div class="lbl">Routes in Code</div>
    </div>
    <div class="stat-card">
      <div class="val">{stats['writes']}</div>
      <div class="lbl">DB Writes in Code</div>
    </div>
    <div class="stat-card">
      <div class="val">{stats['forms']}</div>
      <div class="lbl">Forms in Code</div>
    </div>
    <div class="stat-card">
      <div class="val">{total_gaps}</div>
      <div class="lbl">Coverage Gaps</div>
    </div>
    <div class="stat-card">
      <div class="val">{total_risks}</div>
      <div class="lbl">Hallucination Risks</div>
    </div>
  </div>

  <h3>Behavioral Validation (Stage 4.7)</h3>
  <div class="check-grid">
    {check_cards_html}
  </div>

  {f'<h3>Document Coverage (Stage 5.8)</h3><div>{doc_dim_html}</div>' if doc_dims else ''}
</section>

<!-- ── Section 2: Module Coverage Matrix ─────────────────────────────────── -->
<section id="matrix">
  <h2>Module Coverage Matrix</h2>
  <p class="hint">
    Each row is a module or bounded context.  Percentages show what fraction
    of routes / DB write operations / HTML forms in that module are referenced
    by at least one business flow.  Sorted worst-first.
  </p>
  {matrix_html}
</section>

<!-- ── Section 3: Documented Flows ──────────────────────────────────────── -->
<section id="flows">
  <h2>Documented Flows with Code Evidence ({len(flow_cards)})</h2>
  <p class="hint">
    Click a flow to expand it.  Each flow shows the ordered steps, the backing
    PHP files, the DB operations, and — where available — which BA document
    section cites it (from Stage 5.5 traceability).
    Confidence colour: <span class="conf-badge pass">High</span>
    <span class="conf-badge warn">Medium</span>
    <span class="conf-badge fail">Low</span>
  </p>
  {flows_html}
</section>

<!-- ── Section 4: Coverage Gaps ──────────────────────────────────────────── -->
<section id="gaps">
  <h2>Coverage Gaps — In Code, Missing from BA</h2>
  <p class="hint">
    These code artefacts were found by static analysis but are not referenced
    by any business flow or BA document section.  Each item here represents
    a gap in the BA documentation.
  </p>
  {gaps_html}
</section>

<!-- ── Section 5: Hallucination Risks ────────────────────────────────────── -->
<section id="risks">
  <h2>Hallucination Risks — In BA, Unverified in Code</h2>
  <p class="hint">
    These BA claims could not be corroborated by static analysis.
    Review them against the source code before treating them as authoritative.
  </p>
  {risks_html}
</section>

</div><!-- /container -->
</body>
</html>"""
