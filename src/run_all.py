"""
Usage:  python src/run_all.py --samples 1 2 3 4 5 6 8 10 11 12 13 14
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
PROC = SRC.parent / "data" / "processed"


def run(script, *a):
    cmd = [sys.executable, str(SRC / script), *a]
    print("\n>", " ".join(cmd[1:]))
    subprocess.run(cmd, check=False)


ap = argparse.ArgumentParser()
ap.add_argument("--samples", nargs="+", required=True)
ap.add_argument("--reverse", action="store_true")
args = ap.parse_args()

for s in args.samples:
    for scan in ("T1", "T2"):
        (PROC / s / f"{scan}_attachments.csv").unlink(missing_ok=True)   # no stale files
        run("detect_attachments.py", "--sample", s, "--scan", scan, "--show", "none")

run("match_attachments.py", "--samples", *args.samples, *(["--reverse"] if args.reverse else []))
run("export_xlsx.py", "--samples", *args.samples)