import os
import glob
import re
from difflib import SequenceMatcher
from collections import defaultdict

DATASET_DIR = "test-projects"
ARTIFACTS = ["brd.md", "srs.md", "ac.md"]

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# AI-assisted extraction from .md
def extract_sections(md_text):
    """
    Extracts actors, entities, rules, workflows from md text.
    Simple heuristic based on headings and keywords.
    """
    actors, entities, rules, workflows = set(), set(), set(), set()
    lines = [l.strip() for l in md_text.splitlines() if l.strip()]

    workflow_keywords = ["status", "workflow", "lifecycle", "->"]
    actor_keywords = ["actor", "roles"]
    entity_keywords = ["entity", "entities"]
    rule_keywords = ["rule", "business rule", "BR"]

    current_section = ""
    for line in lines:
        l_lower = line.lower()
        if any(k in l_lower for k in workflow_keywords):
            current_section = "workflow"
            continue
        elif any(k in l_lower for k in actor_keywords):
            current_section = "actors"
            continue
        elif any(k in l_lower for k in entity_keywords):
            current_section = "entities"
            continue
        elif any(k in l_lower for k in rule_keywords):
            current_section = "rules"
            continue

        # extraction
        if current_section == "workflow":
            # detect status transitions like A -> B
            if "->" in line:
                transitions = [s.strip() for s in line.split("->")]
                for t in transitions:
                    workflows.add(t)
        elif current_section == "actors":
            actors.add(line)
        elif current_section == "entities":
            entities.add(line)
        elif current_section == "rules":
            rules.add(line)
        else:
            # heuristic: lines containing "must", "cannot", "require" likely rules
            if any(k in li
