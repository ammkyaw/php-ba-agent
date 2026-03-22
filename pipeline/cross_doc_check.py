"""
pipeline/cross_doc_check.py — Cross-Document Consistency Check (Pass E)

Fully static (no LLM).  Runs in Stage 6 alongside Pass D, before the LLM
passes, so findings are injected into Pass B's prompt and appear in the QA
report and checklist.

Checks implemented
------------------
  E-01  Orphan Epic Headings
        Every "## Epic: X" in UserStories must correspond to a domain feature.
        An epic with no matching feature is a hallucinated section not grounded
        in the domain model — the LLM invented it.

  E-02  Orphan AC Sections
        Every "## AC-XX: X" in the AC artefact must correspond to a domain
        feature.  Missing domain-model backing indicates the LLM added an
        acceptance-criteria section for something that was never agreed as a
        feature — a traceability gap.

  E-03  Undefined Actors in UserStories
        "As a [X]" story patterns extract actor names.  Each actor X must
        match a defined role in domain.user_roles.  Actors that appear in
        stories but not in the domain model are undocumented actors — either
        a genuine role the domain analyst missed or an LLM hallucination.

  E-04  SRS Feature → AC Test Gap
        Every "### 3.N Feature" section in the SRS represents a stated
        functional requirement.  Each requirement MUST have a corresponding
        "## AC-XX: Feature" section in the AC artefact.  SRS requirements
        without AC coverage represent untested business behaviour.

Issue schema (compatible with consistency_check.py / stage6 issue list)
-----------------------------------------------------------------------
  {
    "severity":       "critical" | "major" | "minor",
    "artefact":       document name or pair, e.g. "UserStories" or "SRS→AC",
    "description":    human-readable problem statement,
    "recommendation": actionable fix,
    "category":       "Cross-Document Consistency",
  }
"""

from __future__ import annotations

import re
from typing import Any


# ─── Severity constants ───────────────────────────────────────────────────────
_CRITICAL = "critical"
_MAJOR    = "major"
_MINOR    = "minor"

# Fuzzy match: accept heading as matching feature if ≥ this fraction of
# significant words appear in the feature name (or vice versa).
_FUZZY_WORD_RATIO = 0.70


# ─── Public API ───────────────────────────────────────────────────────────────

def run_checks(ctx: Any, artefacts: dict[str, str]) -> list[dict]:
    """
    Run all cross-document consistency checks and return a combined issue list.

    Parameters
    ----------
    ctx       : PipelineContext
    artefacts : {name: content} loaded by stage6 _load_artefacts().
                Keys: "BRD", "SRS", "AC", "UserStories"

    Returns
    -------
    list[dict]  — zero or more issue dicts compatible with stage6 schema.
    """
    issues: list[dict] = []
    issues.extend(_check_orphan_epics(ctx, artefacts))
    issues.extend(_check_orphan_ac_sections(ctx, artefacts))
    issues.extend(_check_undefined_actors(ctx, artefacts))
    issues.extend(_check_srs_feature_ac_gap(ctx, artefacts))
    return issues


def format_summary(issues: list[dict]) -> str:
    """One-line console summary."""
    c = sum(1 for i in issues if i.get("severity") == _CRITICAL)
    m = sum(1 for i in issues if i.get("severity") == _MAJOR)
    n = sum(1 for i in issues if i.get("severity") == _MINOR)
    if not issues:
        return "✅  No cross-document issues found."
    return (
        f"{len(issues)} cross-doc issue(s) — "
        f"{c} critical · {m} major · {n} minor"
    )


# ─── E-01: Orphan Epic Headings ───────────────────────────────────────────────

def _check_orphan_epics(ctx: Any, artefacts: dict[str, str]) -> list[dict]:
    """
    E-01: Every '## Epic: X' in UserStories should match a domain feature.
    """
    us_content = artefacts.get("UserStories", "")
    if not us_content:
        return []

    feature_names = _feature_names_lower(ctx)
    if not feature_names:
        return []

    epic_headings = re.findall(
        r'^##\s+Epic:\s+(.+)$', us_content, re.MULTILINE | re.IGNORECASE
    )

    issues: list[dict] = []
    for heading in epic_headings:
        heading_clean = heading.strip()
        if not _fuzzy_match_any(heading_clean.lower(), feature_names):
            issues.append(_issue(
                severity       = _MINOR,
                artefact       = "UserStories",
                description    = (
                    f"Orphan epic heading '## Epic: {heading_clean}' has no "
                    "matching feature in the domain model."
                ),
                recommendation = (
                    f"Either add '{heading_clean}' to the domain model features "
                    "or rename this epic to match an existing feature."
                ),
                category       = "Cross-Document Consistency",
            ))

    return issues


# ─── E-02: Orphan AC Sections ────────────────────────────────────────────────

def _check_orphan_ac_sections(ctx: Any, artefacts: dict[str, str]) -> list[dict]:
    """
    E-02: Every '## AC-XX: X' in AC should match a domain feature.
    """
    ac_content = artefacts.get("AC", "")
    if not ac_content:
        return []

    feature_names = _feature_names_lower(ctx)
    if not feature_names:
        return []

    # Match "## AC-01: Feature Name" or "## AC-1: Feature Name"
    ac_headings = re.findall(
        r'^##\s+AC-\d+:\s+(.+)$', ac_content, re.MULTILINE | re.IGNORECASE
    )

    issues: list[dict] = []
    for heading in ac_headings:
        heading_clean = heading.strip()
        if not _fuzzy_match_any(heading_clean.lower(), feature_names):
            issues.append(_issue(
                severity       = _MINOR,
                artefact       = "AC",
                description    = (
                    f"AC section '## AC-XX: {heading_clean}' has no matching "
                    "feature in the domain model."
                ),
                recommendation = (
                    f"Either add '{heading_clean}' to the domain model or "
                    "align this AC section to an existing feature."
                ),
                category       = "Cross-Document Consistency",
            ))

    return issues


# ─── E-03: Undefined Actors in UserStories ────────────────────────────────────

def _check_undefined_actors(ctx: Any, artefacts: dict[str, str]) -> list[dict]:
    """
    E-03: Every 'As a [X]' actor in UserStories should match a defined role.
    """
    us_content = artefacts.get("UserStories", "")
    if not us_content:
        return []

    dm = getattr(ctx, "domain_model", None)
    if not dm or not dm.user_roles:
        return []

    defined_roles_lower = {
        r["role"].lower()
        for r in dm.user_roles
        if isinstance(r, dict) and r.get("role")
    }
    if not defined_roles_lower:
        return []

    # Extract actors from "As a [X]" / "As an [X]" patterns (case-insensitive)
    raw_actors = re.findall(
        r'[Aa]s (?:a|an)\s+([A-Za-z][A-Za-z0-9 _\-]{1,40}?)(?:[,\.]|\s+I\s+want)',
        us_content
    )

    # Normalize and deduplicate
    seen: set[str] = set()
    actor_set: list[str] = []
    for actor in raw_actors:
        key = actor.strip().lower()
        if key and key not in seen:
            seen.add(key)
            actor_set.append(actor.strip())

    issues: list[dict] = []
    for actor in actor_set:
        actor_low = actor.lower()
        # Accept if it matches any defined role (fuzzy)
        if not _fuzzy_match_any(actor_low, defined_roles_lower):
            issues.append(_issue(
                severity       = _MINOR,
                artefact       = "UserStories",
                description    = (
                    f"Actor '{actor}' appears in user stories but is not "
                    "defined in the domain model user roles."
                ),
                recommendation = (
                    f"Add '{actor}' to the domain model user_roles or "
                    "rename this actor to match an existing role."
                ),
                category       = "Cross-Document Consistency",
            ))

    return issues


# ─── E-04: SRS Feature → AC Test Gap ─────────────────────────────────────────

def _check_srs_feature_ac_gap(ctx: Any, artefacts: dict[str, str]) -> list[dict]:
    """
    E-04: Every '### 3.N Feature' section in SRS must have an AC section.

    This checks SRS→AC traceability using the *actual* generated SRS headings
    (not the domain model), catching cases where SRS added extra sections that
    AC did not cover.
    """
    srs_content = artefacts.get("SRS", "")
    ac_content  = artefacts.get("AC",  "")
    if not srs_content or not ac_content:
        return []

    # Extract SRS functional requirement headings: "### 3.N FeatureName"
    # Also accept "## 3. FeatureName" or "### FeatureName" inside a 3.x section
    srs_features = re.findall(
        r'^###\s+3\.\d+\s+(.+)$', srs_content, re.MULTILINE
    )
    if not srs_features:
        # Broader fallback: any ### heading that looks like a feature section
        srs_features = re.findall(
            r'^###\s+(?:Feature[:\s]+|FR\d+[:\s]+)?(.+)$',
            srs_content, re.MULTILINE
        )

    if not srs_features:
        return []

    # Build set of AC section names (lowercased) from "## AC-XX: Name"
    ac_sections_lower: set[str] = {
        m.strip().lower()
        for m in re.findall(
            r'^##\s+AC-\d+:\s+(.+)$', ac_content, re.MULTILINE | re.IGNORECASE
        )
    }

    # Also accept plain "## Feature Name" in AC as coverage
    ac_headings_lower: set[str] = {
        re.sub(r'^#+\s*', '', line.strip()).lower()
        for line in ac_content.splitlines()
        if line.strip().startswith('##')
    }
    all_ac_lower = ac_sections_lower | ac_headings_lower

    issues: list[dict] = []
    for feat in srs_features:
        feat_clean = feat.strip()
        if not feat_clean or len(feat_clean) < 3:
            continue
        if not _fuzzy_match_any(feat_clean.lower(), all_ac_lower):
            issues.append(_issue(
                severity       = _MAJOR,
                artefact       = "SRS→AC",
                description    = (
                    f"SRS functional requirement '### 3.X {feat_clean}' has no "
                    "corresponding section in the AC artefact."
                ),
                recommendation = (
                    f"Add '## AC-XX: {feat_clean}' to the AC artefact with "
                    "acceptance criteria for this requirement."
                ),
                category       = "Cross-Document Consistency",
            ))

    return issues


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _feature_names_lower(ctx: Any) -> set[str]:
    """Return lowercased feature names from the domain model."""
    dm = getattr(ctx, "domain_model", None)
    if not dm or not dm.features:
        return set()
    return {
        f["name"].lower()
        for f in dm.features
        if isinstance(f, dict) and f.get("name")
    }


def _fuzzy_match_any(needle: str, haystack: set[str]) -> bool:
    """
    Return True if needle fuzzy-matches any string in haystack.

    Fuzzy: needle is in haystack_item (or vice versa), OR ≥ FUZZY_WORD_RATIO
    of needle's significant words (len ≥ 3) appear in the haystack_item.
    """
    for candidate in haystack:
        if needle in candidate or candidate in needle:
            return True
        if _fuzzy_word_overlap(needle, candidate):
            return True
    return False


def _fuzzy_word_overlap(a: str, b: str) -> bool:
    """True if ≥ FUZZY_WORD_RATIO of significant words in `a` appear in `b`."""
    words = [w for w in re.split(r"\W+", a) if len(w) >= 3]
    if not words:
        return False
    matches = sum(1 for w in words if w in b)
    return matches / len(words) >= _FUZZY_WORD_RATIO


def _issue(
    severity: str,
    artefact: str,
    description: str,
    recommendation: str,
    category: str = "Cross-Document Consistency",
) -> dict:
    return {
        "severity":       severity,
        "artefact":       artefact,
        "description":    description,
        "recommendation": recommendation,
        "category":       category,
    }
