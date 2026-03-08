"""
pipeline/stage65_postprocess.py — Post-Processing & DOCX Generation

Converts all four Markdown BA artefacts (BRD, SRS, AC, User Stories)
and the QA report into formatted DOCX files using the docx npm package.

What it does
------------
1. Parses each Markdown file into a document tree
2. Generates a Node.js script that uses the docx package
3. Runs the script to produce .docx files
4. Validates the output files exist and are non-empty
5. Writes a final summary to the output directory

Output files
------------
    outputs/run_<id>/brd.docx
    outputs/run_<id>/srs.docx
    outputs/run_<id>/ac.docx
    outputs/run_<id>/user_stories.docx
    outputs/run_<id>/qa_report.docx
    outputs/run_<id>/pipeline_summary.md    (final run summary)

DOCX formatting
---------------
    - Arial font throughout
    - US Letter page size (8.5" × 11"), 1" margins
    - Blue heading style (H1: 16pt bold, H2: 13pt bold, H3: 11pt bold)
    - Proper bullet/numbered lists (no unicode bullets)
    - Tables with header row shading
    - Page numbers in footer
    - Document title + run ID in header

Resume behaviour
----------------
If stage65_postprocess is COMPLETED and all .docx files exist, stage is skipped.

Dependencies
------------
    Node.js >= 14  (checked at runtime)
    docx npm package  (installed automatically if missing)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from context import PipelineContext

DOCX_NODE_PKG   = "docx"
SUMMARY_FILE    = "pipeline_summary.md"

# Documents to convert: (stage_key, md_filename, docx_filename, title)
DOCUMENTS = [
    ("stage5_brd",         "brd.md",          "brd.docx",          "Business Requirements Document"),
    ("stage5_srs",         "srs.md",          "srs.docx",          "Software Requirements Specification"),
    ("stage5_ac",          "ac.md",           "ac.docx",           "Acceptance Criteria"),
    ("stage5_userstories", "user_stories.md", "user_stories.docx", "User Story Backlog"),
    ("stage6_qa",          "qa_report.md",    "qa_report.docx",    "QA Review Report"),
]


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 6.5 entry point. Converts all Markdown artefacts to DOCX.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If Node.js is missing or docx conversion fails.
    """
    summary_path = ctx.output_path(SUMMARY_FILE)

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage65_postprocess"):
        docx_files = [ctx.output_path(d[2]) for d in DOCUMENTS]
        if all(Path(p).exists() for p in docx_files):
            print(f"  [stage65] Already completed — "
                  f"{len(docx_files)} DOCX files present.")
            return

    print("  [stage65] Starting post-processing ...")

    # ── Check Node.js ─────────────────────────────────────────────────────────
    node_bin = _find_node()
    npm_bin  = _find_npm()
    print(f"  [stage65] Node.js: {node_bin}")

    # ── Ensure docx package is available ─────────────────────────────────────
    _ensure_docx_package(npm_bin, ctx.output_dir)

    # ── Convert each document ─────────────────────────────────────────────────
    converted = []
    failed    = []

    for stage_key, md_file, docx_file, title in DOCUMENTS:
        md_path   = ctx.output_path(md_file)
        docx_path = ctx.output_path(docx_file)

        if not Path(md_path).exists():
            print(f"  [stage65] Skipping {md_file} (not found)")
            continue

        print(f"  [stage65] Converting {md_file} → {docx_file} ...")
        try:
            _convert_md_to_docx(
                md_path    = md_path,
                docx_path  = docx_path,
                title      = title,
                run_id     = ctx.run_id,
                node_bin   = node_bin,
                output_dir = ctx.output_dir,
            )
            size = Path(docx_path).stat().st_size
            print(f"  [stage65] ✓ {docx_file} ({size:,} bytes)")
            converted.append(docx_path)
        except Exception as e:
            print(f"  [stage65] ✗ {docx_file} failed: {e}")
            failed.append((docx_file, str(e)))

    # ── Write pipeline summary ────────────────────────────────────────────────
    _write_summary(ctx, converted, failed, summary_path)

    # ── Mark stage ────────────────────────────────────────────────────────────
    if failed:
        ctx.stage("stage65_postprocess").mark_failed(
            f"{len(failed)} DOCX conversion(s) failed: "
            + ", ".join(f for f, _ in failed)
        )
    else:
        ctx.stage("stage65_postprocess").mark_completed(summary_path)

    ctx.save()
    _print_summary(ctx, converted, failed)


# ─── Markdown Parser ───────────────────────────────────────────────────────────

def _parse_markdown(md_text: str) -> list[dict[str, Any]]:
    """
    Parse Markdown into a list of block elements.
    Each block is a dict with a 'type' key.

    Supported types:
        heading     level (1-6), text
        paragraph   text (may contain inline bold/code)
        bullet      text, level (0-based)
        numbered    text, level (0-based)
        table       headers (list[str]), rows (list[list[str]])
        hr          (horizontal rule)
        code        text (fenced code block)
    """
    blocks: list[dict] = []
    lines  = md_text.splitlines()
    i      = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Blank line
        if not stripped:
            i += 1
            continue

        # Fenced code block
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            blocks.append({"type": "code", "text": "\n".join(code_lines)})
            i += 1
            continue

        # Heading
        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if m:
            blocks.append({"type": "heading",
                           "level": len(m.group(1)),
                           "text": m.group(2).strip()})
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}$", stripped):
            blocks.append({"type": "hr"})
            i += 1
            continue

        # Table
        if "|" in stripped and i + 1 < len(lines) and re.match(r"^\|[-| :]+\|", lines[i + 1].strip()):
            headers = [c.strip() for c in stripped.strip("|").split("|")]
            rows    = []
            i += 2  # skip header + separator
            while i < len(lines) and "|" in lines[i]:
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(row)
                i += 1
            blocks.append({"type": "table", "headers": headers, "rows": rows})
            continue

        # Bullet list
        m = re.match(r"^(\s*)([-*+])\s+(.*)", line)
        if m:
            indent = len(m.group(1)) // 2
            blocks.append({"type": "bullet",
                           "text": m.group(3).strip(),
                           "level": indent})
            i += 1
            continue

        # Numbered list
        m = re.match(r"^(\s*)(\d+)[.)]\s+(.*)", line)
        if m:
            indent = len(m.group(1)) // 2
            blocks.append({"type": "numbered",
                           "text": m.group(3).strip(),
                           "level": indent})
            i += 1
            continue

        # Paragraph (default) — stop at bullets, numbers, headings, tables
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nl = lines[i]
            ns = nl.strip()
            if not ns:
                break
            if (re.match(r"^#{1,6}\s", nl) or
                ns.startswith("|") or
                ns.startswith("```") or
                re.match(r"^\s*[-*+]\s+", nl) or
                re.match(r"^\s*\d+[.)]\s+", nl) or
                re.match(r"^[-*_]{3,}$", ns)):
                break
            para_lines.append(ns)
            i += 1
        blocks.append({"type": "paragraph",
                       "text": " ".join(para_lines)})

    return blocks


# ─── DOCX Generation ───────────────────────────────────────────────────────────

def _convert_md_to_docx(
    md_path:    str,
    docx_path:  str,
    title:      str,
    run_id:     str,
    node_bin:   str,
    output_dir: str,
) -> None:
    """Parse Markdown and generate a DOCX file via a Node.js script."""
    # Resolve all paths to absolute BEFORE building the script or subprocess call
    abs_output_dir = str(Path(output_dir).resolve())
    abs_docx_path  = str(Path(docx_path).resolve())

    md_text = Path(md_path).read_text(encoding="utf-8")
    blocks  = _parse_markdown(md_text)
    script  = _generate_node_script(blocks, abs_docx_path, title, run_id)

    # Write the script into the output_dir so that `require('docx')` resolves
    # against the node_modules installed there. Use .js (CommonJS), not .mjs.
    script_path = Path(abs_output_dir) / f"_gen_{Path(docx_path).stem}.js"
    script_path.write_text(script, encoding="utf-8")

    try:
        result = subprocess.run(
            [node_bin, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=abs_output_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Node.js script failed (exit {result.returncode}):\n"
                f"{result.stderr[:500]}"
            )
        if not Path(abs_docx_path).exists():
            raise RuntimeError(f"Script ran but {abs_docx_path} was not created.")
    finally:
        if script_path.exists():
            script_path.unlink()


def _generate_node_script(
    blocks:    list[dict],
    out_path:  str,
    title:     str,
    run_id:    str,
) -> str:
    """Generate a self-contained Node.js ESM script that produces the DOCX."""

    # Escape string for JS
    def js(s: str) -> str:
        return json.dumps(str(s))

    # Build children array entries
    child_lines = []

    # Track numbering state
    in_bullet   = False
    in_numbered = False

    for block in blocks:
        btype = block["type"]

        if btype == "heading":
            level = block["level"]
            text  = block["text"]
            # Strip markdown bold from headings
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
            heading_map = {1: "HEADING_1", 2: "HEADING_2", 3: "HEADING_3",
                           4: "HEADING_4", 5: "HEADING_5", 6: "HEADING_6"}
            hl = heading_map.get(level, "HEADING_3")
            child_lines.append(
                f"    new Paragraph({{ heading: HeadingLevel.{hl}, "
                f"children: [new TextRun({{ text: {js(text)}, bold: true }})] }}),"
            )
            in_bullet = in_numbered = False

        elif btype == "paragraph":
            text = block["text"]
            if not text.strip():
                continue
            runs = _parse_inline(text)
            run_code = ", ".join(
                f"new TextRun({{ text: {js(r['text'])}, bold: {str(r['bold']).lower()}, "
                f"italics: {str(r['italic']).lower()}, "
                f"font: 'Arial', size: 24 }})"
                for r in runs
            )
            child_lines.append(
                f"    new Paragraph({{ spacing: {{ after: 120 }}, "
                f"children: [{run_code}] }}),"
            )

        elif btype == "bullet":
            text  = block["text"]
            level = min(block.get("level", 0), 2)
            runs  = _parse_inline(text)
            run_code = ", ".join(
                f"new TextRun({{ text: {js(r['text'])}, bold: {str(r['bold']).lower()}, "
                f"font: 'Arial', size: 24 }})"
                for r in runs
            )
            child_lines.append(
                f"    new Paragraph({{ numbering: {{ reference: 'bullets', level: {level} }}, "
                f"children: [{run_code}] }}),"
            )

        elif btype == "numbered":
            text  = block["text"]
            level = min(block.get("level", 0), 2)
            runs  = _parse_inline(text)
            run_code = ", ".join(
                f"new TextRun({{ text: {js(r['text'])}, bold: {str(r['bold']).lower()}, "
                f"font: 'Arial', size: 24 }})"
                for r in runs
            )
            child_lines.append(
                f"    new Paragraph({{ numbering: {{ reference: 'numbers', level: {level} }}, "
                f"children: [{run_code}] }}),"
            )

        elif btype == "table":
            headers = block["headers"]
            rows    = block["rows"]
            child_lines.append(_generate_table_code(headers, rows))

        elif btype == "hr":
            child_lines.append(
                "    new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, "
                "size: 6, color: '2E75B6', space: 1 } }, children: [] }),"
            )

        elif btype == "code":
            text = block["text"]
            child_lines.append(
                f"    new Paragraph({{ children: [new TextRun({{ text: {js(text)}, "
                f"font: 'Courier New', size: 20 }})] }}),"
            )

    children_str = "\n".join(child_lines)
    out_path_js  = out_path.replace("\\", "/")
    title_js     = js(title)
    run_id_js    = js(run_id)

    return f"""const {{ Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
     HeadingLevel, AlignmentType, LevelFormat, BorderStyle,
     WidthType, ShadingType, VerticalAlign, PageNumber,
     Header, Footer }} = require('docx');
const fs = require('fs');

const doc = new Document({{
  numbering: {{
    config: [
      {{ reference: 'bullets',
         levels: [
           {{ level: 0, format: LevelFormat.BULLET, text: '\\u2022', alignment: AlignmentType.LEFT,
              style: {{ paragraph: {{ indent: {{ left: 720, hanging: 360 }} }} }} }},
           {{ level: 1, format: LevelFormat.BULLET, text: '\\u25E6', alignment: AlignmentType.LEFT,
              style: {{ paragraph: {{ indent: {{ left: 1080, hanging: 360 }} }} }} }},
           {{ level: 2, format: LevelFormat.BULLET, text: '\\u25AA', alignment: AlignmentType.LEFT,
              style: {{ paragraph: {{ indent: {{ left: 1440, hanging: 360 }} }} }} }},
         ]
      }},
      {{ reference: 'numbers',
         levels: [
           {{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
              style: {{ paragraph: {{ indent: {{ left: 720, hanging: 360 }} }} }} }},
           {{ level: 1, format: LevelFormat.DECIMAL, text: '%2.', alignment: AlignmentType.LEFT,
              style: {{ paragraph: {{ indent: {{ left: 1080, hanging: 360 }} }} }} }},
         ]
      }},
    ]
  }},
  styles: {{
    default: {{ document: {{ run: {{ font: 'Arial', size: 24 }} }} }},
    paragraphStyles: [
      {{ id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
         run: {{ size: 32, bold: true, font: 'Arial', color: '1F3864' }},
         paragraph: {{ spacing: {{ before: 320, after: 160 }}, outlineLevel: 0 }} }},
      {{ id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
         run: {{ size: 28, bold: true, font: 'Arial', color: '2E75B6' }},
         paragraph: {{ spacing: {{ before: 240, after: 120 }}, outlineLevel: 1 }} }},
      {{ id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
         run: {{ size: 24, bold: true, font: 'Arial', color: '2E75B6' }},
         paragraph: {{ spacing: {{ before: 200, after: 80 }}, outlineLevel: 2 }} }},
    ]
  }},
  sections: [{{
    properties: {{
      page: {{
        size: {{ width: 12240, height: 15840 }},
        margin: {{ top: 1440, right: 1440, bottom: 1440, left: 1440 }}
      }}
    }},
    headers: {{
      default: new Header({{
        children: [new Paragraph({{
          alignment: AlignmentType.RIGHT,
          children: [
            new TextRun({{ text: {title_js}, bold: true, font: 'Arial', size: 20, color: '666666' }}),
            new TextRun({{ text: '  |  Run: ' + {run_id_js}, font: 'Arial', size: 18, color: '999999' }}),
          ]
        }})]
      }})
    }},
    footers: {{
      default: new Footer({{
        children: [new Paragraph({{
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({{ text: 'Page ', font: 'Arial', size: 18, color: '999999' }}),
            new TextRun({{ children: [PageNumber.CURRENT], font: 'Arial', size: 18, color: '999999' }}),
            new TextRun({{ text: ' of ', font: 'Arial', size: 18, color: '999999' }}),
            new TextRun({{ children: [PageNumber.TOTAL_PAGES], font: 'Arial', size: 18, color: '999999' }}),
          ]
        }})]
      }})
    }},
    children: [
{children_str}
    ]
  }}]
}});

Packer.toBuffer(doc).then(buf => {{
  fs.writeFileSync({js(out_path_js)}, buf);
  console.log('Written:', {js(out_path_js)});
}}).catch(err => {{ console.error(err); process.exit(1); }});
"""


def _generate_table_code(headers: list[str], rows: list[list[str]]) -> str:
    """Generate docx Table JavaScript code."""
    n_cols     = max(len(headers), max((len(r) for r in rows), default=1))
    table_w    = 9360
    col_w      = table_w // n_cols if n_cols > 0 else table_w
    col_widths = [col_w] * n_cols

    def cell(text: str, is_header: bool = False) -> str:
        t = json.dumps(str(text))
        shade = ', shading: { fill: "D5E8F0", type: ShadingType.CLEAR }' if is_header else ''
        return (
            f"new TableCell({{ width: {{ size: {col_w}, type: WidthType.DXA }}, "
            f"margins: {{ top: 80, bottom: 80, left: 120, right: 120 }}{shade}, "
            f"children: [new Paragraph({{ children: [new TextRun("
            f"{{ text: {t}, bold: {str(is_header).lower()}, font: 'Arial', size: 22 }})] }})] }})"
        )

    header_row = (
        "new TableRow({ tableHeader: true, children: ["
        + ", ".join(cell(h, True) for h in headers)
        + "] })"
    )
    data_rows = []
    for row in rows:
        cells = [cell(row[j] if j < len(row) else "") for j in range(n_cols)]
        data_rows.append("new TableRow({ children: [" + ", ".join(cells) + "] })")

    all_rows = ", ".join([header_row] + data_rows)
    widths   = json.dumps(col_widths)

    return (
        f"    new Table({{ width: {{ size: {table_w}, type: WidthType.DXA }}, "
        f"columnWidths: {widths}, rows: [{all_rows}] }}),"
    )


def _parse_inline(text: str) -> list[dict[str, Any]]:
    """
    Parse inline Markdown formatting into runs.
    Handles: **bold**, *italic*, `code`, and plain text.
    Returns list of {text, bold, italic, code} dicts.
    """
    runs   = []
    # Pattern: **bold**, *italic*, `code`
    pattern = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`")
    last    = 0

    for m in pattern.finditer(text):
        if m.start() > last:
            runs.append({"text": text[last:m.start()], "bold": False, "italic": False})
        if m.group(1) is not None:
            runs.append({"text": m.group(1), "bold": True,  "italic": False})
        elif m.group(2) is not None:
            runs.append({"text": m.group(2), "bold": False, "italic": True})
        elif m.group(3) is not None:
            runs.append({"text": m.group(3), "bold": False, "italic": False})
        last = m.end()

    if last < len(text):
        runs.append({"text": text[last:], "bold": False, "italic": False})

    return runs or [{"text": text, "bold": False, "italic": False}]


# ─── Node.js / npm Helpers ─────────────────────────────────────────────────────

def _find_node() -> str:
    """Find the Node.js binary. Raises if not found."""
    import shutil
    node = shutil.which("node") or shutil.which("node.js")
    if node:
        return node
    raise RuntimeError(
        "Node.js not found. Install it from https://nodejs.org/\n"
        "Or: brew install node  (macOS)"
    )


def _find_npm() -> str:
    """Find the npm binary."""
    import shutil
    npm = shutil.which("npm")
    if npm:
        return npm
    raise RuntimeError("npm not found. Install Node.js from https://nodejs.org/")


def _ensure_docx_package(npm_bin: str, work_dir: str) -> None:
    """Install docx npm package in work_dir if not already present."""
    node_modules = Path(work_dir) / "node_modules" / "docx"
    if node_modules.exists():
        return

    print(f"  [stage65] Installing docx npm package in {work_dir} ...")
    result = subprocess.run(
        [npm_bin, "install", "--save", DOCX_NODE_PKG],
        capture_output=True,
        text=True,
        cwd=work_dir,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to install docx package:\n{result.stderr[:400]}\n"
            "Run manually: npm install docx  (in your output directory)"
        )
    print(f"  [stage65] docx package installed.")


# ─── Summary ───────────────────────────────────────────────────────────────────

def _write_summary(
    ctx:          PipelineContext,
    converted:    list[str],
    failed:       list[tuple],
    summary_path: str,
) -> None:
    """Write a final pipeline summary Markdown file."""
    from datetime import datetime
    qa = ctx.qa_result

    lines = [
        f"# Pipeline Run Summary",
        f"",
        f"**Run ID:** `{ctx.run_id}`  ",
        f"**Project:** `{ctx.php_project_path}`  ",
        f"**Completed:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"",
        f"## Domain",
        f"",
        f"**System:** {ctx.domain_model.domain_name if ctx.domain_model else 'Unknown'}  ",
    ]

    if ctx.domain_model:
        lines += [
            f"**Description:** {ctx.domain_model.description}  ",
            f"**Features:** {', '.join(f['name'] for f in ctx.domain_model.features)}  ",
            f"**Entities:** {', '.join(ctx.domain_model.key_entities)}  ",
            f"",
        ]

    if qa:
        status = "✅ PASSED" if qa.passed else "❌ FAILED"
        lines += [
            f"## QA Results",
            f"",
            f"**Status:** {status}  ",
            f"**Coverage:** {qa.coverage_score:.0%}  ",
            f"**Consistency:** {qa.consistency_score:.0%}  ",
            f"**Issues:** {len(qa.issues)} total  ",
            f"",
        ]

    lines += [f"## Output Files", f""]
    for p in converted:
        size = Path(p).stat().st_size
        lines.append(f"- ✅ `{Path(p).name}` ({size:,} bytes)")
    for fname, err in failed:
        lines.append(f"- ❌ `{fname}` — {err[:80]}")

    Path(summary_path).write_text("\n".join(lines), encoding="utf-8")


def _print_summary(
    ctx:       PipelineContext,
    converted: list[str],
    failed:    list[tuple],
) -> None:
    width = 58
    print(f"\n  {'=' * width}")
    print(f"  Post-Processing Complete")
    print(f"  {'=' * width}")
    for p in converted:
        size = Path(p).stat().st_size
        print(f"  ✓ {Path(p).name:<30} {size:>10,} bytes")
    for fname, err in failed:
        print(f"  ✗ {fname:<30} FAILED: {err[:40]}")
    print(f"  {'=' * width}")
    print(f"  Output dir: {ctx.output_dir}")
    print(f"  {'=' * width}\n")