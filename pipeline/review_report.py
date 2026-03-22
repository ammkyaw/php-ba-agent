"""
pipeline/review_report.py — Confidence-Based Human Review Report  (Post-pipeline)

Collects low-confidence and flagged items from upstream stages and produces
a prioritised review queue for domain expert validation.  One 30-minute
review session using this report is worth more than endless prompt tuning.

Output files (in the run output directory):
    review_required.json   — machine-readable review queue
    review_required.html   — human-friendly checklist (self-contained)

Review categories
-----------------
  SPEC_RULE    — stage46 rules with confidence < LOW_CONF_THRESHOLD
  FLOW         — stage45 flows marked is_draft=True or with no branches
  ENTITY       — stage35 entities with evidence_count < MIN_EVIDENCE
  ROUTE_GAP    — routes in cm.routes with no matching business flow
  VALIDATION   — stage47 checks that FAILED (not just warned)

Usage (called at end of run_pipeline.py):
    from pipeline.review_report import generate
    generate(ctx)

Zero LLM calls.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from context import PipelineContext

# Thresholds
LOW_CONF_THRESHOLD = 0.4    # spec rules below this get flagged
MIN_EVIDENCE       = 2      # entity evidence count floor


# ── Public API ─────────────────────────────────────────────────────────────────

def generate(ctx: PipelineContext) -> str | None:
    """
    Generate review_required.json + review_required.html in ctx.output_dir.
    Returns the path to the JSON file, or None if nothing to review.
    """
    items: list[dict] = []

    items.extend(_collect_spec_rules(ctx))
    items.extend(_collect_flows(ctx))
    items.extend(_collect_entities(ctx))
    items.extend(_collect_route_gaps(ctx))
    items.extend(_collect_validation_failures(ctx))

    if not items:
        print("  [review_report] No items require human review.")
        return None

    # Sort: HIGH priority first, then by category
    _PRIORITY = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    items.sort(key=lambda x: (_PRIORITY.get(x.get("priority", "LOW"), 2), x.get("category", "")))

    out_dir  = Path(ctx.output_dir)
    json_path = out_dir / "review_required.json"
    html_path = out_dir / "review_required.html"

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "run_id":       ctx.run_id,
        "total_items":  len(items),
        "summary": _summarise(items),
        "items":   items,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    html_path.write_text(_render_html(payload))

    print(f"  [review_report] {len(items)} item(s) require review → {json_path.name}")
    return str(json_path)


# ── Collectors ─────────────────────────────────────────────────────────────────

def _collect_spec_rules(ctx: PipelineContext) -> list[dict]:
    """Flag spec rules with confidence below LOW_CONF_THRESHOLD."""
    items: list[dict] = []
    rules_file = Path(ctx.output_dir) / "spec_rules.json"
    if not rules_file.exists():
        return items
    try:
        rules = json.loads(rules_file.read_text())
        if isinstance(rules, dict):
            rules = rules.get("rules", [])
        for r in (rules or []):
            conf = float(r.get("confidence", 1.0))
            if conf < LOW_CONF_THRESHOLD:
                items.append({
                    "category":    "SPEC_RULE",
                    "id":          r.get("id", "?"),
                    "priority":    "HIGH" if conf < 0.2 else "MEDIUM",
                    "confidence":  conf,
                    "title":       r.get("title", r.get("description", "")[:80]),
                    "issue":       f"Confidence {conf:.2f} — may be framework noise or hallucination",
                    "action":      "Verify this rule exists as an explicit business requirement",
                    "context":     {
                        "given": r.get("given", ""),
                        "when":  r.get("when", ""),
                        "then":  r.get("then", ""),
                        "tags":  r.get("tags", []),
                    },
                })
    except Exception as exc:
        print(f"  [review_report] Warning: could not read spec_rules.json: {exc}")
    return items


def _collect_flows(ctx: PipelineContext) -> list[dict]:
    """Flag draft flows or flows with no branches and no known termination."""
    items: list[dict] = []
    flows_file = Path(ctx.output_dir) / "business_flows.json"
    if not flows_file.exists():
        return items
    try:
        data = json.loads(flows_file.read_text())
        flows = data.get("flows", []) if isinstance(data, dict) else (data or [])
        for f in flows:
            issues = []
            if f.get("is_draft"):
                issues.append("marked as draft — LLM enrichment failed")
            if not f.get("branches") and f.get("source", "") not in ("laravel_route",):
                issues.append("no alternate branches — may be missing error paths")
            if not f.get("termination") or f.get("termination", "").startswith("Flow ends"):
                issues.append("generic termination — needs a specific success outcome")
            if issues:
                items.append({
                    "category":   "FLOW",
                    "id":         f.get("flow_id", "?"),
                    "priority":   "HIGH" if f.get("is_draft") else "LOW",
                    "confidence": f.get("confidence", 0.5),
                    "title":      f.get("name", f.get("flow_id", "?")),
                    "issue":      "; ".join(issues),
                    "action":     "Review flow steps and verify/add missing branches and outcomes",
                    "context": {
                        "actor":       f.get("actor", ""),
                        "trigger":     f.get("trigger", ""),
                        "termination": f.get("termination", ""),
                        "step_count":  len(f.get("steps", [])),
                    },
                })
    except Exception as exc:
        print(f"  [review_report] Warning: could not read business_flows.json: {exc}")
    return items


def _collect_entities(ctx: PipelineContext) -> list[dict]:
    """Flag entities with very low evidence counts."""
    items: list[dict] = []
    # Entities are stored on the domain model
    domain = getattr(ctx, "domain_model", None)
    if domain is None:
        return items
    entities = getattr(domain, "entities", None) or []
    for e in entities:
        if isinstance(e, dict):
            ev = e.get("evidence_count", e.get("evidence", 99))
        else:
            ev = getattr(e, "evidence_count", getattr(e, "evidence", 99))
        try:
            ev = int(ev)
        except (TypeError, ValueError):
            continue
        if ev < MIN_EVIDENCE:
            name = (e.get("name") if isinstance(e, dict) else getattr(e, "name", "?")) or "?"
            items.append({
                "category":   "ENTITY",
                "id":         name,
                "priority":   "MEDIUM",
                "confidence": ev / 5.0,  # normalise loosely
                "title":      name,
                "issue":      f"Entity has only {ev} evidence source(s) — possible hallucination",
                "action":     "Confirm this entity corresponds to a real database table or domain object",
                "context":    {"evidence_count": ev},
            })
    return items


def _collect_route_gaps(ctx: PipelineContext) -> list[dict]:
    """Flag routes that have no corresponding business flow."""
    items: list[dict] = []
    cm = ctx
    routes = getattr(cm, "routes", None) or []
    flows_file = Path(ctx.output_dir) / "business_flows.json"
    if not flows_file.exists() or not routes:
        return items
    try:
        data = json.loads(flows_file.read_text())
        flows = data.get("flows", []) if isinstance(data, dict) else (data or [])
        # Build set of URIs covered by flows
        covered_uris: set[str] = set()
        for f in flows:
            for step in (f.get("steps") or []):
                page = step.get("page", "") if isinstance(step, dict) else getattr(step, "page", "")
                if page:
                    covered_uris.add(page.split("?")[0].lower().rstrip("/") or "/")
        for route in routes:
            if not isinstance(route, dict):
                continue
            uri   = route.get("uri") or route.get("path", "")
            verb  = route.get("method") or route.get("verb", "GET")
            ctrl  = route.get("controller", "")
            if not uri:
                continue
            uri_clean = uri.lower().rstrip("/") or "/"
            if uri_clean not in covered_uris:
                items.append({
                    "category":   "ROUTE_GAP",
                    "id":         f"{verb} {uri}",
                    "priority":   "MEDIUM",
                    "confidence": 0.0,
                    "title":      f"{verb} {uri}",
                    "issue":      "Route has no corresponding business flow",
                    "action":     f"Add a flow for this route (controller: {ctrl})",
                    "context":    {"verb": verb, "uri": uri, "controller": ctrl},
                })
    except Exception as exc:
        print(f"  [review_report] Warning: route gap check failed: {exc}")
    return items


def _collect_validation_failures(ctx: PipelineContext) -> list[dict]:
    """Flag stage47 validation checks that FAILED."""
    items: list[dict] = []
    val_file = Path(ctx.output_dir) / "flow_validation.json"
    if not val_file.exists():
        return items
    try:
        data = json.loads(val_file.read_text())
        for check in (data.get("checks") or []):
            if check.get("status") == "FAIL":
                items.append({
                    "category":   "VALIDATION",
                    "id":         check.get("id", "?"),
                    "priority":   "HIGH",
                    "confidence": 0.0,
                    "title":      check.get("name", check.get("id", "?")),
                    "issue":      check.get("message", "Validation check failed"),
                    "action":     "Fix the underlying flow/route/branch issue and re-run stage47",
                    "context":    {
                        "score":   check.get("score"),
                        "details": check.get("details", "")[:200],
                    },
                })
    except Exception as exc:
        print(f"  [review_report] Warning: could not read flow_validation.json: {exc}")
    return items


# ── Helpers ────────────────────────────────────────────────────────────────────

def _summarise(items: list[dict]) -> dict:
    from collections import Counter
    by_cat  = Counter(i["category"] for i in items)
    by_prio = Counter(i["priority"] for i in items)
    return {"by_category": dict(by_cat), "by_priority": dict(by_prio)}


def _render_html(payload: dict) -> str:
    items   = payload["items"]
    summary = payload["summary"]
    run_id  = payload.get("run_id", "")
    ts      = payload.get("generated_at", "")

    _PRIO_COLOR = {"HIGH": "#d32f2f", "MEDIUM": "#f57c00", "LOW": "#388e3c"}
    _CAT_ICON   = {
        "SPEC_RULE": "📋", "FLOW": "🔄", "ENTITY": "🗂",
        "ROUTE_GAP": "🛣", "VALIDATION": "❌",
    }

    rows = ""
    for item in items:
        prio  = item.get("priority", "LOW")
        cat   = item.get("category", "")
        color = _PRIO_COLOR.get(prio, "#333")
        icon  = _CAT_ICON.get(cat, "•")
        ctx_rows = "".join(
            f"<tr><td style='color:#888;padding:2px 8px'>{k}</td>"
            f"<td style='padding:2px 8px'>{v}</td></tr>"
            for k, v in (item.get("context") or {}).items()
        )
        rows += f"""
<tr>
  <td style='padding:8px;white-space:nowrap'><span style='color:{color};font-weight:bold'>{prio}</span></td>
  <td style='padding:8px'>{icon} {cat}</td>
  <td style='padding:8px'><strong>{item.get('title','')}</strong><br>
    <span style='color:#555;font-size:0.9em'>{item.get('issue','')}</span></td>
  <td style='padding:8px;color:#1565c0'>{item.get('action','')}</td>
  <td style='padding:8px;font-size:0.85em'>
    <table>{ctx_rows}</table>
  </td>
</tr>"""

    summary_html = " | ".join(
        f"<b>{v}</b> {k}" for k, v in sorted(summary.get("by_category", {}).items())
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>Review Required — {run_id}</title>
<style>
  body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}
  h1{{color:#333}}
  table{{border-collapse:collapse;width:100%}}
  thead th{{background:#37474f;color:#fff;padding:8px;text-align:left}}
  tr:nth-child(even){{background:#f5f5f5}}
  .summary{{background:#e3f2fd;padding:12px;border-radius:4px;margin-bottom:16px}}
</style>
</head><body>
<h1>🔍 Human Review Required</h1>
<div class='summary'>
  <b>{len(items)} item(s)</b> require domain expert review &nbsp;|&nbsp;
  Run: <code>{run_id}</code> &nbsp;|&nbsp; Generated: {ts}<br>
  {summary_html}
</div>
<table>
<thead><tr>
  <th>Priority</th><th>Category</th><th>Item</th><th>Suggested Action</th><th>Context</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
</body></html>"""
