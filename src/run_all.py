"""
Run attachment detection, volume measurement, and Excel export across samples.

Usage:
    python src/run_all.py --all
    python src/run_all.py --samples 1 2 4 5 6 8 10 11 12 13 14
"""
import argparse
import subprocess
import sys
from config import DATA_DIR, PROCESSED_DIR, SRC_DIR

# excluded directories dict
EXCLUDED = {}


def discover_samples():
    """Return sorted sample folder names under DATA_DIR (excluding EXCLUDED dict)."""
    found = [
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and d != PROCESSED_DIR and d.name not in EXCLUDED
        and (d / "T1.stl").exists() and (d / "T2.stl").exists()
    ]
    return sorted(found, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))


def run(script, *a):
    cmd = [sys.executable, str(SRC_DIR / script), *a]
    print("\n>", " ".join(cmd[1:]))
    subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--samples", nargs="+", help="Sample folder names")
    group.add_argument("--all", action="store_true",
                       help="Auto-discover all active sample folders in data/")
    ap.add_argument("--reverse", action="store_true")
    args = ap.parse_args()

    samples = discover_samples() if args.all else args.samples
    if not samples:
        raise SystemExit(f"No sample folders found in {DATA_DIR}")
    if args.all:
        print(f"Discovered samples: {' '.join(samples)}")

    detect_flags = ["--reverse"] if args.reverse else []

    for s in samples:
        for scan in ("T1", "T2"):
            # no stale files
            (PROCESSED_DIR / s /
             f"{scan}_attachments.csv").unlink(missing_ok=True)
            run("detect_attachments.py", "--sample",
                s, "--scan", scan, "--show", "none", *detect_flags)
        run("measure_volume.py", "--sample", s)

    run("export_xlsx.py", "--samples", *samples)


if __name__ == "__main__":
    main()
