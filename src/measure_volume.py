"""
Phase 1 - Step 3a: volume of the detected attachments, for T1 and T2 (no alignment needed).

Reads the output of detect_attachments.py for each scan and computes, for every attachment,
the volume between the scan surface and the tooth surface that would be there without the
attachment. Both scans are written to ONE csv.

Method (per attachment, per scan):
  1. Mask = vertices with label k in <scan>_attachment_labels.npy.
  2. Base surface = quadric fitted to a thin ring just outside the mask (RING_IN..RING_OUT,
     steep wall removed by the normal filter, vertices of other attachments excluded).
     The same ring parameters and fit function as the detector are used.
  3. h = height of each mask vertex above the base surface along the attachment normal
     (negative values are clipped to 0).
  4. volume = sum over mesh triangles whose three vertices are in the mask of
        (area of the triangle PROJECTED on the base plane) x (mean h of its 3 vertices).

Known bias: triangles that touch the mask border are not counted, and the low foot of the
attachment (below the detector's height cut) is outside the mask, so absolute volumes are
slightly low. The same code is used for T1 and T2.

Rows are joined by attachment id only. T1 and T2 are in different frames here, so the ids
of two independent detections are NOT guaranteed to be the same physical attachment.

The mesh is loaded and cleaned exactly like in detect_attachments.py, so vertex indices
match the labels.

Input (per scan = T1, T2):
        data/<sample>/<scan>.stl
        data/processed/<sample>/<scan>_attachment_labels.npy
        data/processed/<sample>/<scan>_attachments.csv
Output: data/processed/<sample>/volumes.csv

Usage:
    python src/measure_volume.py --sample 2
"""

import argparse
import csv

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from detect_attachments import (DATA_DIR, PATCH_RADIUS, PROCESSED_DIR, PROJECT_ROOT, RING_IN,
                                RING_NORMAL_DOT, RING_OUT, design_matrix, fit_quadric_robust,
                                tangent_frame)

MIN_RING_POINTS = 50
SCANS = ("T1", "T2")
FIELDS = ["qc_flag", "cx", "cy", "cz", "area_proj_mm2", "mean_h_mm", "volume_mm3",
          "ring_rms_mm", "n_ring", "n_triangles"]


def warn(msg):
    print(f"\033[33m[WARNING] {msg}\033[0m")


def load_mesh(path):
    """Same preprocessing as detect_attachments.py (vertex order must match the labels)."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return (np.asarray(mesh.vertices), np.asarray(mesh.triangles),
            np.asarray(mesh.vertex_normals).copy())


def measure_attachment(points, faces, normals, vtree, labels, row):
    """Returns a dict of results or None if no clean ring could be fitted."""
    k = int(row["id"])
    in_reg = labels == k
    reg = np.where(in_reg)[0]
    if len(reg) == 0:
        return None
    n_ref = np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])])
    n_ref /= np.linalg.norm(n_ref)
    centre = points[reg].mean(axis=0)

    patch = np.asarray(vtree.query_ball_point(
        centre, r=PATCH_RADIUS), dtype=np.int64)
    P, N = points[patch], normals[patch]
    # make vertex normals point outward
    if normals[reg].mean(axis=0) @ n_ref < 0:
        N = -N
    e1, e2 = tangent_frame(n_ref)
    rel = P - centre
    u, v, w = rel @ e1, rel @ e2, rel @ n_ref

    d = cKDTree(points[reg]).query(P)[0]
    ring = (d >= RING_IN) & (d <= RING_OUT) & (
        N @ n_ref >= RING_NORMAL_DOT) & (labels[patch] < 0)
    if ring.sum() < MIN_RING_POINTS:
        return None
    fit = fit_quadric_robust(u[ring], v[ring], w[ring])
    if fit is None:
        return None
    coef, ring_rms, _ = fit
    h = w - design_matrix(u, v) @ coef

    U = np.zeros(len(points))
    V = np.zeros(len(points))
    H = np.zeros(len(points))
    U[patch], V[patch], H[patch] = u, v, h

    tri = faces[in_reg[faces].all(axis=1)]
    if len(tri) == 0:
        return None
    du1, dv1 = U[tri[:, 1]] - U[tri[:, 0]], V[tri[:, 1]] - V[tri[:, 0]]
    du2, dv2 = U[tri[:, 2]] - U[tri[:, 0]], V[tri[:, 2]] - V[tri[:, 0]]
    a_proj = 0.5 * np.abs(du1 * dv2 - du2 * dv1)
    h_tri = np.maximum(H[tri], 0).mean(axis=1)
    vol = float((a_proj * h_tri).sum())
    area_proj = float(a_proj.sum())
    return dict(area_proj_mm2=area_proj, mean_h_mm=vol / area_proj if area_proj > 0 else 0.0,
                volume_mm3=vol, ring_rms_mm=ring_rms, n_ring=int(ring.sum()),
                n_triangles=int(len(tri)), n_mask_vertices=int(len(reg)))


def measure_scan(sample, scan):
    """
    Returns {id: row dict with FIELDS} for one scan, or None if the scan cannot be measured
    (a yellow warning is printed in that case). A failed attachment keeps its id with empty values.
    """
    out_dir = PROCESSED_DIR / sample
    paths = {"mesh": DATA_DIR / sample / f"{scan}.stl",
             "labels": out_dir / f"{scan}_attachment_labels.npy",
             "csv": out_dir / f"{scan}_attachments.csv"}
    missing = [str(p.relative_to(PROJECT_ROOT))
               for p in paths.values() if not p.exists()]
    if missing:
        warn(f"{scan}: missing {', '.join(missing)} -> {scan} columns left empty")
        return None

    points, faces, normals = load_mesh(paths["mesh"])
    labels = np.load(paths["labels"])
    if len(labels) != len(points):
        warn(f"{scan}: labels ({len(labels)}) do not match mesh vertices ({len(points)}) "
             f"-> {scan} columns left empty")
        return None
    with open(paths["csv"], newline="") as f:
        att = list(csv.DictReader(f))
    print(f"\n[{scan}] attachments in CSV: {len(att)}")

    vtree = cKDTree(points)
    result = {}
    print(" id  qc     area_proj  mean_h  volume   ring_rms  n_ring")
    for row in att:
        k = int(row["id"])
        res = measure_attachment(points, faces, normals, vtree, labels, row)
        base = dict(qc_flag=row["qc_flag"],
                    cx=row["cx"], cy=row["cy"], cz=row["cz"])
        if res is None:
            warn(f"{scan} #{k}: no clean ring or empty mask -> measurements left empty")
            result[k] = base
            continue
        if res["n_mask_vertices"] != int(row["n_vertices"]):
            warn(f"{scan} #{k}: mask has {res['n_mask_vertices']} vertices, "
                 f"CSV says {row['n_vertices']}")
        print(f"{k:3d}  {row['qc_flag']:5s}  {res['area_proj_mm2']:8.2f}  "
              f"{res['mean_h_mm']:6.3f}  {res['volume_mm3']:6.2f}  {res['ring_rms_mm']:8.3f}  "
              f"{res['n_ring']:6d}")
        result[k] = {**base, **{key: res[key] for key in FIELDS if key in res}}

    vols = np.array([float(r["volume_mm3"])
                    for r in result.values() if r.get("volume_mm3", "") != ""])
    if len(vols):
        print(f"[{scan}] volume mm3: mean {vols.mean():.2f} | min {vols.min():.2f} | "
              f"max {vols.max():.2f} (n = {len(vols)})")
    return result


def fmt(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    return x


def main(args):
    results = {scan: measure_scan(args.sample, scan) for scan in SCANS}

    available = {s: r for s, r in results.items() if r is not None}
    if not available:
        raise SystemExit("\033[31m[ERROR] Neither T1 nor T2 could be measured "
                         "(see warnings above). volumes.csv was NOT written. "
                         "Run detect_attachments.py first.\033[0m")
    ids = sorted(set().union(*[r.keys() for r in available.values()]))
    if not ids:
        raise SystemExit("\033[31m[ERROR] No attachments found in any scan. "
                         "volumes.csv was NOT written.\033[0m")
    counts = {s: len(r) for s, r in available.items()}
    if len(counts) == 2 and len(set(counts.values())) > 1:
        warn(
            f"different number of attachments ({counts}); rows are joined by id only")

    columns = ["id"] + [f"{s}_{f}" for s in SCANS for f in FIELDS]
    out_dir = PROCESSED_DIR / args.sample
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "volumes.csv"
    with open(dst, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=columns)
        wr.writeheader()
        for k in ids:
            row = {"id": k}
            for s in SCANS:
                r = results[s].get(k) if results[s] is not None else None
                for fld in FIELDS:
                    row[f"{s}_{fld}"] = "" if r is None else fmt(
                        r.get(fld, ""))
            wr.writerow(row)
    print(f"\nSaved {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Volume of detected attachments (T1 and T2).")
    ap.add_argument("--sample", required=True,
                    help="Sample folder name, e.g. 2")
    main(ap.parse_args())
