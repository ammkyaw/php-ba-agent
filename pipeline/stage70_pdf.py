"""
pipeline/stage70_pdf.py — DOCX → PDF Conversion for Final Delivery

Converts all five DOCX artefacts from Stage 6.5 into PDF files using
LibreOffice (headless mode). Falls back to a pure-Python approach via
docx2pdf if LibreOffice is unavailable on the host machine.

Output files
------------
    outputs/run_<id>/brd.pdf
    outputs/run_<id>/srs.pdf
    outputs/run_<id>/ac.pdf
    outputs/run_<id>/user_stories.pdf
    outputs/run_<id>/qa_report.pdf
    outputs/run_<id>/delivery/          ← final delivery bundle
        Car_Rental_System_BRD.pdf
        Car_Rental_System_SRS.pdf
        Car_Rental_System_AC.pdf
        Car_Rental_System_UserStories.pdf
        Car_Rental_System_QAReport.pdf
        README.txt

Conversion strategy
-------------------
1. LibreOffice headless (soffice --headless --convert-to pdf)
   - Best fidelity, handles tables/fonts/headers perfectly
   - Available on macOS (brew install --cask libreoffice), Linux (apt/yum)
2. docx2pdf Python package
   - Uses Microsoft Word on macOS/Windows if available
   - Fallback when LibreOffice is absent
3. If both fail: copies DOCX files to delivery/ and notes conversion failed

Resume behaviour
----------------
If stage70_pdf is COMPLETED and all PDF files exist, stage is skipped.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from context import PipelineContext

# Documents to convert: (docx_filename, pdf_filename, delivery_label)
DOCUMENTS = [
    ("brd.docx",          "brd.pdf",          "BRD"),
    ("srs.docx",          "srs.pdf",          "SRS"),
    ("ac.docx",           "ac.pdf",           "AC"),
    ("user_stories.docx", "user_stories.pdf", "UserStories"),
    ("qa_report.docx",    "qa_report.pdf",    "QAReport"),
]

DELIVERY_DIR = "delivery"


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """
    Stage 7 entry point. Converts DOCX artefacts to PDF and bundles them.

    Args:
        ctx: Shared pipeline context; mutated in-place.

    Raises:
        RuntimeError: If no conversion method is available.
    """
    delivery_dir = Path(ctx.output_path(DELIVERY_DIR))

    # ── Resume check ─────────────────────────────────────────────────────────
    if ctx.is_stage_done("stage70_pdf"):
        pdf_files = [ctx.output_path(d[1]) for d in DOCUMENTS]
        if all(Path(p).exists() for p in pdf_files):
            print(f"  [stage7] Already completed — "
                  f"{len(pdf_files)} PDF files present.")
            return

    print("  [stage7] Starting PDF conversion ...")

    # ── Detect conversion method ──────────────────────────────────────────────
    converter = _detect_converter()
    print(f"  [stage7] Converter: {converter}")

    # ── Convert each DOCX ─────────────────────────────────────────────────────
    converted = []
    failed    = []

    for docx_file, pdf_file, label in DOCUMENTS:
        docx_path = ctx.output_path(docx_file)
        pdf_path  = ctx.output_path(pdf_file)

        if not Path(docx_path).exists():
            print(f"  [stage7] Skipping {docx_file} (not found)")
            failed.append((pdf_file, "source DOCX not found"))
            continue

        print(f"  [stage7] Converting {docx_file} → {pdf_file} ...")
        try:
            _convert(docx_path, pdf_path, converter, ctx.output_dir)
            size = Path(pdf_path).stat().st_size
            print(f"  [stage7] ✓ {pdf_file} ({size:,} bytes)")
            converted.append((pdf_file, pdf_path, label))
        except Exception as e:
            print(f"  [stage7] ✗ {pdf_file} failed: {e}")
            failed.append((pdf_file, str(e)))

    # ── Build delivery bundle ─────────────────────────────────────────────────
    delivery_dir.mkdir(exist_ok=True)
    _build_delivery_bundle(ctx, converted, failed, delivery_dir)

    # ── Mark stage ────────────────────────────────────────────────────────────
    if len(converted) == len(DOCUMENTS):
        ctx.stage("stage70_pdf").mark_completed(str(delivery_dir))
    elif converted:
        # Partial success — mark completed with a warning note
        ctx.stage("stage70_pdf").mark_completed(str(delivery_dir))
        print(f"  [stage7] Warning: {len(failed)} conversion(s) failed, "
              f"{len(converted)} succeeded.")
    else:
        ctx.stage("stage70_pdf").mark_failed(
            f"All {len(DOCUMENTS)} conversions failed. "
            f"Check that LibreOffice or docx2pdf is installed."
        )

    ctx.save()
    _print_summary(converted, failed, delivery_dir)


# ─── Converter Detection ───────────────────────────────────────────────────────

def _detect_converter() -> str:
    """
    Detect the best available DOCX→PDF converter.

    Returns:
        'libreoffice' | 'docx2pdf' | 'none'
    """
    # 1. LibreOffice (best quality)
    soffice = _find_soffice()
    if soffice:
        return f"libreoffice:{soffice}"

    # 2. docx2pdf Python package (uses MS Word on macOS/Windows)
    try:
        import docx2pdf  # noqa: F401
        return "docx2pdf"
    except ImportError:
        pass

    return "none"


def _find_soffice() -> Optional[str]:
    """Find the LibreOffice soffice binary."""
    candidates = [
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",  # macOS
        "/usr/bin/soffice",
        "/usr/lib/libreoffice/program/soffice",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


# ─── Conversion Implementations ───────────────────────────────────────────────

def _convert(
    docx_path:  str,
    pdf_path:   str,
    converter:  str,
    output_dir: str,
) -> None:
    """Dispatch to the appropriate converter."""
    if converter.startswith("libreoffice:"):
        soffice = converter.split(":", 1)[1]
        _convert_libreoffice(docx_path, pdf_path, soffice, output_dir)
    elif converter == "docx2pdf":
        _convert_docx2pdf(docx_path, pdf_path)
    else:
        raise RuntimeError(
            "No PDF converter found. Install one of:\n"
            "  • LibreOffice: brew install --cask libreoffice  (macOS)\n"
            "                 sudo apt install libreoffice      (Linux)\n"
            "  • docx2pdf:    pip install docx2pdf              (requires MS Word on macOS/Windows)"
        )


def _convert_libreoffice(
    docx_path:  str,
    pdf_path:   str,
    soffice:    str,
    output_dir: str,
) -> None:
    """Convert using LibreOffice headless mode."""
    abs_docx  = str(Path(docx_path).resolve())
    abs_outdir = str(Path(output_dir).resolve())

    result = subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf",
         "--outdir", abs_outdir, abs_docx],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice failed (exit {result.returncode}):\n"
            f"{result.stderr[:400]}"
        )

    # LibreOffice outputs <input_stem>.pdf in the outdir
    lo_output = Path(abs_outdir) / (Path(docx_path).stem + ".pdf")
    if not lo_output.exists():
        raise RuntimeError(
            f"LibreOffice ran but output not found at {lo_output}"
        )

    # Move to the target pdf_path if different
    abs_pdf = str(Path(pdf_path).resolve())
    if str(lo_output) != abs_pdf:
        shutil.move(str(lo_output), abs_pdf)


def _convert_docx2pdf(docx_path: str, pdf_path: str) -> None:
    """Convert using the docx2pdf package (requires MS Word)."""
    from docx2pdf import convert
    convert(docx_path, pdf_path)
    if not Path(pdf_path).exists():
        raise RuntimeError(f"docx2pdf ran but {pdf_path} was not created.")


# ─── Delivery Bundle ───────────────────────────────────────────────────────────

def _build_delivery_bundle(
    ctx:         PipelineContext,
    converted:   list[tuple],
    failed:      list[tuple],
    delivery_dir: Path,
) -> None:
    """
    Copy PDFs into a clean delivery/ folder with human-readable names.
    Includes a README.txt describing the contents.
    """
    domain_name = ctx.domain_model.domain_name if ctx.domain_model else "System"
    # Sanitise for use in filenames
    safe_name = re.sub(r"[^\w\s-]", "", domain_name).strip().replace(" ", "_")

    # Map label → delivery filename
    label_map = {
        "BRD":        f"{safe_name}_BRD.pdf",
        "SRS":        f"{safe_name}_SRS.pdf",
        "AC":         f"{safe_name}_AcceptanceCriteria.pdf",
        "UserStories":f"{safe_name}_UserStories.pdf",
        "QAReport":   f"{safe_name}_QAReport.pdf",
    }

    readme_lines = [
        f"# {domain_name} — BA Documentation Package",
        f"",
        f"Generated by PHP-BA Agent  |  Run: {ctx.run_id}",
        f"",
        f"## Contents",
        f"",
    ]

    for pdf_file, pdf_path, label in converted:
        dest_name = label_map.get(label, pdf_file)
        dest_path = delivery_dir / dest_name
        shutil.copy2(pdf_path, dest_path)
        size = dest_path.stat().st_size
        readme_lines.append(f"  {dest_name:<45} {size:>10,} bytes")

    for pdf_file, err in failed:
        readme_lines.append(f"  {pdf_file:<45} FAILED: {err[:50]}")

    readme_lines += [
        f"",
        f"## Document Descriptions",
        f"",
        f"  BRD              Business Requirements Document — stakeholder needs & business rules",
        f"  SRS              Software Requirements Specification — functional & non-functional requirements",
        f"  AcceptanceCriteria  Given/When/Then test criteria for all features",
        f"  UserStories      Agile backlog with story points and priorities",
        f"  QAReport         Quality review: coverage score, consistency score, issues found",
    ]

    if ctx.qa_result:
        readme_lines += [
            f"",
            f"## QA Summary",
            f"",
            f"  Coverage    : {ctx.qa_result.coverage_score:.0%}",
            f"  Consistency : {ctx.qa_result.consistency_score:.0%}",
            f"  Status      : {'PASSED' if ctx.qa_result.passed else 'FAILED'}",
            f"  Issues      : {len(ctx.qa_result.issues)} total",
        ]

    (delivery_dir / "README.txt").write_text(
        "\n".join(readme_lines), encoding="utf-8"
    )


# ─── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(
    converted:   list[tuple],
    failed:      list[tuple],
    delivery_dir: Path,
) -> None:
    width = 58
    print(f"\n  {'=' * width}")
    print(f"  Stage 7 — PDF Delivery Bundle")
    print(f"  {'=' * width}")
    for pdf_file, pdf_path, label in converted:
        size = Path(pdf_path).stat().st_size
        print(f"  ✓ {pdf_file:<30} {size:>10,} bytes")
    for pdf_file, err in failed:
        print(f"  ✗ {pdf_file:<30} FAILED")
    print(f"  {'─' * width}")
    print(f"  Delivery bundle: {delivery_dir}/")
    print(f"  {'=' * width}\n")
