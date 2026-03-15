"""
fix_newlines.py — One-time repair for BA artefacts with literal \\n sequences

Run this against any run that was produced before the newline fix:
    python fix_newlines.py outputs/run_20260308_104910/

It will repair brd.md, srs.md, ac.md, user_stories.md in-place if they
contain literal \\n sequences instead of real newlines.
"""
import sys
from pathlib import Path


def fix_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    real_newlines = text.count("\n")
    literal_newlines = text.count("\\n")

    if literal_newlines > 0 and real_newlines < 10:
        fixed = text.replace("\\n", "\n").replace("\\t", "\t")
        path.write_text(fixed, encoding="utf-8")
        lines_after = fixed.count("\n")
        print(f"  ✓ Fixed {path.name}: {real_newlines} → {lines_after} lines "
              f"({len(fixed):,} bytes)")
    else:
        print(f"  — {path.name}: OK ({real_newlines} lines, no fix needed)")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python fix_newlines.py outputs/run_<timestamp>/")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"ERROR: Not a directory: {run_dir}")
        sys.exit(1)

    targets = ["brd.md", "srs.md", "ac.md", "user_stories.md"]
    print(f"Checking {run_dir} ...")
    for filename in targets:
        path = run_dir / filename
        if path.exists():
            fix_file(path)
        else:
            print(f"  — {filename}: not found, skipping")


if __name__ == "__main__":
    main()
