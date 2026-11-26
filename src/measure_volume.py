# src/measure_volume.py
"""
Phase 1 - Step 3a: volume of the detected attachments, for every detected scan
(T1, T2, T1_aligned) found in data/processed/<sample>/.

This script does NOT fit any surface. The base surface of every attachment was fitted once
by detect_attachments.py, which saved the height h of each attachment vertex above it.
Here only the volume is integrated from those heights.

Per attachment, per scan:
  volume = sum over mesh triangles whose three vertices are all in the attachment mask of
           (area of the triangle PROJECTED on the plane normal to n_ref)
           x (mean h of its 3 vertices)
  n_ref (nx, ny, nz) comes from <scan>_attachments.csv. Projected area (not surface area)
  means steep walls do not add extra volume.

Known bias: triangles that touch the mask border are not counted, and the low foot of the
attachment (below the detector's height cut) is outside the mask, so absolute volumes are
slightly low. The same code is used for every scan.

Consistency check: h95 of the heights file must equal h95_mm in the CSV, otherwise the
files come from different detection runs (yellow warning).

Output: data/processed/<sample>/volumes.csv with columns id, <scan>_volume_mm3 for every
scan that has an attachments CSV (e.g. T1_volume_mm3, T2_volume_mm3, T1_aligned_volume_mm3).

Input (per scan): <scan>_attachments.csv, <scan>_attachment_labels.npy,
        <scan>_attachment_height.npy in data/processed/<sample>/, and the mesh
        (data/<sample>/<scan>.stl, T1_aligned: data/processed/<sample>/T1_aligned.stl)

Usage:
    python src/measure_volume.py --sample 2
"""

import argparse
import csv

import numpy as np

from config import PROCESSED_DIR, PROJECT_ROOT
from detect_attachments import SCANS, load_mesh, scan_path

H95_TOL = 1e-3  # mm


def warn(msg):
    print(f"\033[33m[WARNING] {msg}\033[0m")


def measure_attachment(points, faces, labels, height, row):
    """Returns a dict of results or None if the mask is empty or has no triangles."""
    k = int(row["id"])
    in_reg = labels == k
    if not in_reg.any():
        return None
    h_mask = height[in_reg]
    if not np.isfinite(h_mask).all():
        return None
    n_ref = np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])])
    n_ref /= np.linalg.norm(n_ref)

    tri = faces[in_reg[faces].all(axis=1)]
    if len(tri) == 0:
        return None
    a, b, c = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
    a_proj = 0.5 * np.abs(np.cross(b - a, c - a) @ n_ref)
    h_tri = height[tri].mean(axis=1)
    vol = float((a_proj * h_tri).sum())
    area_proj = float(a_proj.sum())
    return dict(area_proj=area_proj, mean_h=vol / area_proj if area_proj > 0 else 0.0,
                volume=vol, n_triangles=int(len(tri)), n_mask_vertices=int(in_reg.sum()),
                h95=float(np.percentile(h_mask, 95)))


def measure_scan(sample, scan):
    """
    Returns {id: volume in mm3 or None} for one scan, or None if the scan cannot be measured
    (a yellow warning is printed in that case). A failed attachment keeps its id with None.
    """
    out_dir = PROCESSED_DIR / sample
    paths = {"mesh": scan_path(sample, scan),
             "labels": out_dir / f"{scan}_attachment_labels.npy",
             "height": out_dir / f"{scan}_attachment_height.npy",
             "csv": out_dir / f"{scan}_attachments.csv"}
    missing = [str(p.relative_to(PROJECT_ROOT))
               for p in paths.values() if not p.exists()]
    if missing:
        warn(f"{scan}: missing {', '.join(missing)} -> column left empty")
        return None

    mesh = load_mesh(paths["mesh"])
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    labels = np.load(paths["labels"])
    height = np.load(paths["height"]).astype(np.float64)
    if len(labels) != len(points) or len(height) != len(points):
        warn(f"{scan}: labels ({len(labels)}) / heights ({len(height)}) do not match "
             f"mesh vertices ({len(points)}) -> column left empty")
        return None
    with open(paths["csv"], newline="") as f:
        att = list(csv.DictReader(f))
    print(f"\n[{scan}] attachments in CSV: {len(att)}")

    result = {}
    print(" id  qc     area_proj  mean_h  volume   h95    (terminal only)")
    for row in att:
        k = int(row["id"])
        res = measure_attachment(points, faces, labels, height, row)
        if res is None:
            warn(f"{scan} #{k}: empty mask or no heights -> volume left empty")
            result[k] = None
            continue
        if res["n_mask_vertices"] != int(row["n_vertices"]):
            warn(f"{scan} #{k}: mask has {res['n_mask_vertices']} vertices, "
                 f"CSV says {row['n_vertices']} (files from different runs?)")
        if abs(res["h95"] - float(row["h95_mm"])) > H95_TOL:
            warn(f"{scan} #{k}: h95 from heights {res['h95']:.4f} != CSV "
                 f"{float(row['h95_mm']):.4f} (files from different runs?)")
        print(f"{k:3d}  {row['qc_flag']:5s}  {res['area_proj']:8.2f}  {res['mean_h']:6.3f}  "
              f"{res['volume']:6.2f}  {res['h95']:5.2f}")
        result[k] = res["volume"]

    vols = np.array([v for v in result.values() if v is not None])
    if len(vols):
        print(f"[{scan}] volume mm3: mean {vols.mean():.2f} | min {vols.min():.2f} | "
              f"max {vols.max():.2f} (n = {len(vols)})")
    return result


def main(args):
    out_dir = PROCESSED_DIR / args.sample
    found = [s for s in SCANS if (out_dir / f"{s}_attachments.csv").exists()]
    if not found:
        raise SystemExit(f"\033[31m[ERROR] No <scan>_attachments.csv in {out_dir}. "
                         "volumes.csv was NOT written. "
                         "Run detect_attachments.py first.\033[0m")
    print(f"Scans found: {', '.join(found)}")

    results = {scan: measure_scan(args.sample, scan) for scan in found}
    available = {s: r for s, r in results.items() if r is not None}
    if not available:
        raise SystemExit("\033[31m[ERROR] No scan could be measured "
                         "(see warnings above). volumes.csv was NOT written.\033[0m")
    ids = sorted(set().union(*[r.keys() for r in available.values()]))
    if not ids:
        raise SystemExit("\033[31m[ERROR] No attachments found in any scan. "
                         "volumes.csv was NOT written.\033[0m")
    counts = {s: len(r) for s, r in available.items()}
    if len(set(counts.values())) > 1:
        warn(
            f"different number of attachments ({counts}); rows are joined by id only")

    columns = ["id"] + [f"{s}_volume_mm3" for s in found]
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "volumes.csv"
    with open(dst, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=columns)
        wr.writeheader()
        for k in ids:
            row = {"id": k}
            for s in found:
                v = results[s].get(k) if results[s] is not None else None
                row[f"{s}_volume_mm3"] = "" if v is None else f"{v:.4f}"
            wr.writerow(row)
    print(f"\nSaved {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Volume of detected attachments for every detected scan of a sample.")
    ap.add_argument("--sample", required=True,
                    help="Sample folder name, e.g. 2")
    main(ap.parse_args())
