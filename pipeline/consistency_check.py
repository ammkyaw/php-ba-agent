"""
pipeline/consistency_check.py — Deterministic Consistency Validation

Runs rule-based cross-document consistency checks without any LLM calls.
Results are added to the stage6 issue list before the LLM passes, so
they appear in the QA report and influence the final pass/fail decision.

Checks implemented
------------------
  C-01  Actor/Role Consistency
        Every flow actor must be defined as a role in the domain model.
        Actors that appear in flows but not in domain.user_roles are
        undocumented actors — a traceability gap.

  C-02  Bounded-Context Consistency
        Every flow's bounded_context must appear in domain.bounded_contexts.
        Mismatches indicate the flow extractor discovered a sub-domain the
        domain model analyst missed.

  C-03  Flow → Feature Traceability
        Each flow should be traceable to a domain model feature via its
        step pages or bounded_context.  Untraced flows represent business
        behaviour not captured in the domain model.

  C-04  Feature Heading Presence (post-generation)
        After stage5 generates the BA documents, verify that every domain
        feature appears as a heading in each artefact.  If the LLM ignored
        the scaffold and renamed/dropped a section this check catches it.
        AC:  ## AC-XX: {feature_name}
        US:  ## Epic: {feature_name}
        SRS: ### 3.X {feature_name}   (or any ### containing the name)

  C-05  Flow → AC Coverage
        The domain feature associated with each flow must have at least one
        Acceptance Criteria section in the AC artefact.  Missing AC for a
        flow-backed feature is a direct gap in QA coverage.

Output
------
  Returns list[dict] — each dict matches the stage6 issue schema:
    { severity, artefact, description, recommendation, category }

  "category" is an extra key (not in the LLM issue schema) used only for
  grouping in the QA report — downstream code ignores unknown keys.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# ─── Severity constants ───────────────────────────────────────────────────────
_CRITICAL = "critical"
_MAJOR    = "major"
_MINOR    = "minor"

# Fuzzy match: a feature name is "present" in a heading if this fraction
# of its words appear in the heading text (case-insensitive).
_FUZZY_WORD_RATIO = 0.75


# ─── Public API ───────────────────────────────────────────────────────────────

def run_checks(ctx: Any, artefacts: dict[str, str] | None = None) -> list[dict]:
    """
    Run all consistency checks and return a combined list of issues.

    Parameters
    ----------
    ctx       : PipelineContext
    artefacts : optional dict {name: content} from stage6 _load_artefacts().
                If omitted, heading-presence checks (C-04, C-05) are skipped.

    Returns
    -------
    list[dict]  — zero or more issue dicts compatible with stage6 schema.
    """
    issues: list[dict] = []
    issues.extend(_check_actor_roles(ctx))
    issues.extend(_check_bounded_contexts(ctx))
    issues.extend(_check_flow_feature_trace(ctx))
    if artefacts:
        issues.extend(_check_feature_heading_presence(ctx, artefacts))
        issues.extend(_check_flow_ac_coverage(ctx, artefacts))
    return issues


def format_summary(issues: list[dict]) -> str:
    """One-line summary string for the console."""
    c = sum(1 for i in issues if i.get("severity") == _CRITICAL)
    m = sum(1 for i in issues if i.get("severity") == _MAJOR)
    n = sum(1 for i in issues if i.get("severity") == _MINOR)
    if not issues:
        return "✅  No consistency issues found."
    return (
        f"{len(issues)} issue(s) — "
        f"{c} critical · {m} major · {n} minor"
    )


# ─── C-01: Actor / Role Consistency ──────────────────────────────────────────

def _check_actor_roles(ctx: Any) -> list[dict]:
    """Every flow actor must exist in domain.user_roles."""
    domain = getattr(ctx, "domain_model", None)
    flows  = getattr(ctx, "business_flows", None)
    if not domain or not flows or not flows.flows:
        return []

    known_roles: set[str] = {
        r["role"].strip().lower()
        for r in (domain.user_roles or [])
        if isinstance(r, dict) and r.get("role")
    }
    all_roles_str = ", ".join(r["role"] for r in (domain.user_roles or []) if isinstance(r, dict))

    issues: list[dict] = []
    reported: set[str] = set()

    for flow in flows.flows:
        actor = (flow.actor or "").strip()
        if not actor:
            continue
        if actor.lower() not in known_roles and actor not in reported:
            reported.add(actor)
            issues.append(_issue(
                severity = _MAJOR,
                artefact = "cross-doc",
                category = "Domain Model vs Flows",
                description = (
                    f"Flow actor '{actor}' (seen in {flow.flow_id}: {flow.name}) "
                    f"is not defined in domain model user_roles "
                    f"[{all_roles_str}]."
                ),
                recommendation = (
                    f"Add '{actor}' to domain model user_roles, or correct "
                    f"the flow actor to one of: {all_roles_str}."
                ),
            ))

    return issues


# ─── C-02: Bounded-Context Consistency ───────────────────────────────────────

def _check_bounded_contexts(ctx: Any) -> list[dict]:
    """Every flow bounded_context must exist in domain.bounded_contexts."""
    domain = getattr(ctx, "domain_model", None)
    flows  = getattr(ctx, "business_flows", None)
    if not domain or not flows or not flows.flows:
        return []

    known_bcs: set[str] = {
        bc.strip().lower()
        for bc in (domain.bounded_contexts or [])
        if isinstance(bc, str)
    }
    all_bcs_str = ", ".join(domain.bounded_contexts or [])

    issues: list[dict] = []
    reported: set[str] = set()

    for flow in flows.flows:
        bc = (flow.bounded_context or "").strip()
        if not bc:
            continue
        if bc.lower() not in known_bcs and bc not in reported:
            reported.add(bc)
            issues.append(_issue(
                severity = _MAJOR,
                artefact = "cross-doc",
                category = "Domain Model vs Flows",
                description = (
                    f"Flow bounded_context '{bc}' (seen in {flow.flow_id}: {flow.name}) "
                    f"is not defined in domain model bounded_contexts "
                    f"[{all_bcs_str}]."
                ),
                recommendation = (
                    f"Add '{bc}' to domain model bounded_contexts, or "
                    f"reassign the flow to one of: {all_bcs_str}."
                ),
            ))

    return issues


# ─── C-03: Flow → Feature Traceability ───────────────────────────────────────

def _check_flow_feature_trace(ctx: Any) -> list[dict]:
    """
    Each flow should map to at least one domain model feature via either:
      a) its step pages overlapping the feature's pages list, or
      b) its bounded_context fuzzy-matching a feature name.
    Flows with no matching feature are 'untraced'.
    """
    domain = getattr(ctx, "domain_model", None)
    flows  = getattr(ctx, "business_flows", None)
    if not domain or not flows or not flows.flows:
        return []

    # Build feature page sets (basename, lower-case)
    feature_pages: dict[str, set[str]] = {}
    for feat in domain.features:
        pages = {
            Path(p.split("?")[0]).name.lower()
            for p in _to_strs(feat.get("pages", []))
            if p
        }
        feature_pages[feat["name"]] = pages

    feature_names_lower = [f["name"].lower() for f in domain.features]

    issues: list[dict] = []

    for flow in flows.flows:
        # Collect this flow's step pages (basename, lower-case)
        flow_step_pages: set[str] = {
            Path(s.page.split("?")[0]).name.lower()
            for s in flow.steps
            if s.page
        }
        bc = (flow.bounded_context or "").lower()

        # Try page-based match
        matched_by_page = any(
            flow_step_pages & fp
            for fp in feature_pages.values()
            if fp
        )

        # Try bounded_context → feature name fuzzy match
        matched_by_bc = (
            bc and any(
                _fuzzy_contains(feat_name_lower, bc)
                or _fuzzy_contains(bc, feat_name_lower)
                for feat_name_lower in feature_names_lower
            )
        )

        if not matched_by_page and not matched_by_bc:
            issues.append(_issue(
                severity = _MINOR,
                artefact = "cross-doc",
                category = "Flow → Feature Traceability",
                description = (
                    f"Flow '{flow.flow_id}: {flow.name}' "
                    f"(actor={flow.actor}, bc={flow.bounded_context}) "
                    f"cannot be traced to any domain model feature. "
                    f"Step pages: {sorted(flow_step_pages) or '(none)'}."
                ),
                recommendation = (
                    "Add a domain model feature whose 'pages' list includes "
                    f"the flow's step pages, or align bounded_context "
                    f"'{flow.bounded_context}' with an existing feature."
                ),
            ))

    return issues


# ─── C-04: Feature Heading Presence ──────────────────────────────────────────

def _check_feature_heading_presence(
    ctx: Any,
    artefacts: dict[str, str],
) -> list[dict]:
    """
    Verify every domain feature name appears as a heading in each BA artefact.
    Catches LLM scaffold non-compliance (renamed / dropped sections).

    Expected heading patterns (from the scaffolds in stage5_workers.py):
      AC  → ## AC-XX: {feature_name}
      US  → ## Epic: {feature_name}
      SRS → ### 3.X {feature_name}   (or any ### or ## containing the name)
      BRD → ### BR-XX: {feature_name}
    """
    domain = getattr(ctx, "domain_model", None)
    if not domain:
        return []

    # Pre-extract headings per artefact
    headings: dict[str, list[str]] = {
        name: _extract_headings(content)
        for name, content in artefacts.items()
    }

    issues: list[dict] = []

    for feat in domain.features:
        name = feat.get("name", "")
        if not name:
            continue
        name_lower = name.lower()

        for artefact_name, hdrs in headings.items():
            # A feature is "present" if any heading contains the feature name
            # (or ≥75 % of its words appear in the heading — handles minor
            # LLM paraphrasing like "Management" → "Mgmt")
            present = any(
                _fuzzy_contains(h.lower(), name_lower)
                for h in hdrs
            )
            if not present:
                issues.append(_issue(
                    severity = _MAJOR,
                    artefact = artefact_name,
                    category = "Feature Heading Presence",
                    description = (
                        f"{artefact_name}: feature '{name}' has no matching "
                        f"heading. The LLM may have renamed or omitted its "
                        f"section despite the scaffold instruction."
                    ),
                    recommendation = (
                        f"Manually add or correct the '{name}' section in "
                        f"{artefact_name}, or re-run stage5 to regenerate."
                    ),
                ))

    return issues


# ─── C-05: Flow → AC Coverage ─────────────────────────────────────────────────

def _check_flow_ac_coverage(
    ctx: Any,
    artefacts: dict[str, str],
) -> list[dict]:
    """
    For each business flow, identify its associated domain feature (via page
    or bc match), then verify that feature has an AC section in the AC artefact.
    """
    domain = getattr(ctx, "domain_model", None)
    flows  = getattr(ctx, "business_flows", None)
    if not domain or not flows or not flows.flows:
        return []

    ac_content = artefacts.get("AC", "")
    ac_headings_lower = [h.lower() for h in _extract_headings(ac_content)]

    # Build feature page sets
    feature_pages: dict[str, set[str]] = {
        feat["name"]: {
            Path(p.split("?")[0]).name.lower()
            for p in _to_strs(feat.get("pages", []))
            if p
        }
        for feat in domain.features
    }

    issues: list[dict] = []

    for flow in flows.flows:
        flow_step_pages: set[str] = {
            Path(s.page.split("?")[0]).name.lower()
            for s in flow.steps
            if s.page
        }
        bc_lower = (flow.bounded_context or "").lower()

        # Find the best-matching feature
        matched_feature: str | None = None

        # Page-based match (strongest signal)
        for feat_name, feat_pages in feature_pages.items():
            if feat_pages and flow_step_pages & feat_pages:
                matched_feature = feat_name
                break

        # Fallback: bc fuzzy match
        if not matched_feature and bc_lower:
            for feat in domain.features:
                if _fuzzy_contains(feat["name"].lower(), bc_lower) or \
                   _fuzzy_contains(bc_lower, feat["name"].lower()):
                    matched_feature = feat["name"]
                    break

        if not matched_feature:
            continue  # C-03 already reported this as untraced

        # Check AC has a heading for this feature
        feat_lower = matched_feature.lower()
        ac_present = any(
            _fuzzy_contains(h, feat_lower) for h in ac_headings_lower
        )
        if not ac_present:
            issues.append(_issue(
                severity = _MAJOR,
                artefact = "AC",
                category = "Flow → AC Coverage",
                description = (
                    f"Flow '{flow.flow_id}: {flow.name}' traces to feature "
                    f"'{matched_feature}', but no Acceptance Criteria section "
                    f"for that feature was found in the AC artefact."
                ),
                recommendation = (
                    f"Add '## AC-XX: {matched_feature}' section to the AC "
                    f"document with at least 2 Given/When/Then criteria."
                ),
            ))

    return issues


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _issue(
    severity: str,
    artefact: str,
    description: str,
    recommendation: str,
    category: str = "",
) -> dict:
    return {
        "severity":       severity,
        "artefact":       artefact,
        "description":    description,
        "recommendation": recommendation,
        "category":       category,
    }


def _extract_headings(content: str) -> list[str]:
    """Extract ## and ### Markdown headings from document content."""
    headings: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            heading = re.sub(r"^#+\s*", "", stripped).strip()
            if heading:
                headings.append(heading)
    return headings


def _fuzzy_contains(haystack: str, needle: str) -> bool:
    """
    True if needle (or ≥ FUZZY_WORD_RATIO of its words) appear in haystack.
    Both strings should already be lower-cased by the caller.
    """
    if needle in haystack:
        return True
    words = [w for w in re.split(r"\W+", needle) if len(w) > 2]
    if not words:
        return needle in haystack
    matches = sum(1 for w in words if w in haystack)
    return matches / len(words) >= _FUZZY_WORD_RATIO


def _to_strs(items: list) -> list[str]:
    """Coerce mixed list (str | dict) to list[str]."""
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            val = next((v for v in item.values() if isinstance(v, str)), None)
            result.append(val if val is not None else str(item))
        else:
            result.append(str(item))
    return result
