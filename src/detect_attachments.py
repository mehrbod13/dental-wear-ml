"""
Phase 1 - Step 2 (simple version 2): detect attachments on T1.

Idea: an attachment is a bump above the tooth surface.
  1. Downsample the vertices to a uniform ~0.3 mm grid (same density for fine/coarse scans).
  2. For every point, fit a local quadric surface to its neighbours (radius ~4 mm),
     repeatedly discarding points that lie above the fit (the bump itself).
     h = height of the point above that base surface (mm).
  3. Candidates: h > threshold, buccal-facing, away from scan borders.
  4. DBSCAN groups candidates; clusters with a plausible area (mm^2) are attachments.
  5. Labels are mapped back to the original mesh vertices.

Input : data/processed/<sample>/T1_aligned.stl (or data/<sample>/T1.stl if missing)
Output: data/processed/<sample>/
          T1_segmented.ply         gray mesh, attachments red
          T1_detection_debug.ply   candidates orange, attachments red
          T1_height_map.ply        h as colour: red = above base surface, blue = below
          T1_attachment_labels.npy int32 per vertex, -1 = background

Usage:
    python src/detect_attachments.py --sample 2
    python src/detect_attachments.py --sample 2 --height-threshold 0.1
    python src/detect_attachments.py --sample 2 --radius 5
    python src/detect_attachments.py --sample 2 --flip-normals
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# All lengths in millimetres
VOXEL = 0.3               # spacing of the working point set
FIT_RADIUS = 4.0          # neighbourhood for the local base-surface fit
TRIM = 0.06               # points more than this above the fit are excluded from the refit
HEIGHT_THRESHOLD = 0.15   # candidate if this far above the base surface
BORDER_MARGIN = 1.0       # ignore points closer than this to the scan border
BUCCAL_DOT = 0.35         # min. alignment of normal with the outward arch direction
DBSCAN_EPS = 0.7          # mm (at least 2.2 x voxel)
DBSCAN_MIN_SAMPLES = 6
MIN_AREA = 1.5            # mm^2
MAX_AREA = 35.0           # mm^2

GRAY = [0.72, 0.72, 0.72]
RED = [1.0, 0.0, 0.0]
ORANGE = [1.0, 0.55, 0.0]


def edges_of(faces):
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])


def vertex_areas(points, faces):
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    face_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return np.bincount(faces.ravel(), weights=np.repeat(face_area / 3, 3),
                       minlength=len(points))


def border_vertices(faces, n):
    """Vertices on an open border of the mesh (edges used by only one face)."""
    e = np.sort(edges_of(faces), axis=1)
    keys = e[:, 0].astype(np.int64) * n + e[:, 1]
    uniq, counts = np.unique(keys, return_counts=True)
    single = uniq[counts == 1]
    mask = np.zeros(n, dtype=bool)
    mask[single // n] = True
    mask[single % n] = True
    return mask


def buccal_mask(points, normals, min_dot):
    """Normals pointing away from the arch centre (perpendicular to the occlusal axis)."""
    centre = points.mean(axis=0)
    cov = (points - centre).T @ (points - centre) / len(points)
    _, vecs = np.linalg.eigh(cov)
    occlusal = vecs[:, 0]                      # smallest-variance axis
    mid = np.median(points, axis=0)

    def flatten(v):
        return v - np.outer(v @ occlusal, occlusal)

    radial = flatten(points - mid)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9
    return np.einsum("ij,ij->i", flatten(normals), radial) >= min_dot


def height_above_quadric(pts, nrm, radius, trim):
    """
    Per point: fit w = a u^2 + b uv + c v^2 + d u + e v + f in the local tangent frame
    (point at the origin), excluding points above the fit. Returns h = -f.
    """
    tree = cKDTree(pts)
    neigh = tree.query_ball_point(pts, r=radius, workers=-1)
    h = np.zeros(len(pts))
    for i, idx in enumerate(neigh):
        if len(idx) < 20:
            continue
        n = nrm[i]
        helper = np.array([1.0, 0, 0]) if abs(
            n[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(n, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        rel = pts[idx] - pts[i]
        u, v, w = rel @ e1, rel @ e2, rel @ n
        A = np.stack([u * u, u * v, v * v, u, v, np.ones_like(u)], axis=1)
        keep = np.ones(len(idx), dtype=bool)
        coef = None
        for _ in range(3):
            if keep.sum() < 12:
                break
            coef = np.linalg.lstsq(A[keep], w[keep], rcond=None)[0]
            keep = (w - A @ coef) < trim
        if coef is not None:
            h[i] = -coef[5]
    return h


def colored(mesh, colors):
    m = o3d.geometry.TriangleMesh(mesh)
    m.vertex_colors = o3d.utility.Vector3dVector(colors)
    return m


def main(args):
    aligned = PROCESSED_DIR / args.sample / "T1_aligned.stl"
    raw = DATA_DIR / args.sample / "T1.stl"
    in_path = aligned if aligned.exists() else raw
    print(f"Input: {in_path}")

    mesh = o3d.io.read_triangle_mesh(str(in_path))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()

    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    if args.flip_normals:
        normals = -normals
    n = len(points)
    print(f"Vertices: {n}")

    # 1. Uniform working point set
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    ds = pcd.voxel_down_sample(args.voxel)
    dpts = np.asarray(ds.points)
    dnrm = np.asarray(ds.normals)
    dnrm = dnrm / (np.linalg.norm(dnrm, axis=1, keepdims=True) + 1e-12)
    print(f"Working points: {len(dpts)} (voxel {args.voxel} mm)")

    # 2. Height above the local base surface
    print(
        f"Fitting local quadrics (radius {args.radius} mm) ... this takes a little while")
    h = height_above_quadric(dpts, dnrm, args.radius, TRIM)

    # 3. Candidates
    border = border_vertices(faces, n)
    if border.any():
        d_border, _ = cKDTree(points[border]).query(dpts)
        far = d_border > BORDER_MARGIN
    else:
        far = np.ones(len(dpts), dtype=bool)
    valid = far & buccal_mask(dpts, dnrm, args.buccal_dot)

    p = np.percentile(h[valid], [0.1, 1, 50, 99, 99.9])
    print(f"h percentiles (mm): p0.1 {p[0]:.3f} | p1 {p[1]:.3f} | p50 {p[2]:.3f} | "
          f"p99 {p[3]:.3f} | p99.9 {p[4]:.3f} | max {h[valid].max():.3f}")
    print("  If the negative tail is much larger than the positive one, try --flip-normals")

    cand = np.where(valid & (h >= args.height_threshold))[0]
    print(f"Candidates (h >= {args.height_threshold} mm): {len(cand)} points")

    # 4. DBSCAN on working points
    lab_ds = np.full(len(dpts), -1, dtype=np.int32)
    if len(cand) >= DBSCAN_MIN_SAMPLES:
        eps = max(args.eps, 2.2 * args.voxel)
        lab_ds[cand] = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES,
                              n_jobs=-1).fit_predict(dpts[cand])
        print(f"DBSCAN eps = {eps:.2f} mm")
    else:
        print("[!] Too few candidates. Lower --height-threshold.")

    # 5. Map back to original vertices, filter clusters by area
    dist, nn = cKDTree(dpts).query(points)
    near = dist <= 1.5 * args.voxel
    lab_orig = np.where(near, lab_ds[nn], -1)
    h_orig = h[nn]
    areas = vertex_areas(points, faces)

    labels_out = np.full(n, -1, dtype=np.int32)
    kept = 0
    print("\nClusters:")
    for c in sorted(set(lab_orig) - {-1}):
        m = lab_orig == c
        area = float(areas[m].sum())
        ok = args.min_area <= area <= args.max_area
        print(f"  [{'KEEP' if ok else 'skip'}] vertices={m.sum():6d} area={area:6.2f} mm^2 "
              f"h_max={h_orig[m].max():.3f} centroid={np.round(points[m].mean(0), 1)}")
        if ok:
            labels_out[m] = kept
            kept += 1
    attach = labels_out >= 0
    print(f"\nAttachments kept: {kept} (expected up to ~14)")

    # Save
    out_dir = PROCESSED_DIR / args.sample
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = np.tile(GRAY, (n, 1))
    colors[attach] = RED
    seg = colored(mesh, colors)
    o3d.io.write_triangle_mesh(str(out_dir / "T1_segmented.ply"), seg)

    dbg = np.tile(GRAY, (n, 1))
    dbg[near & (lab_ds[nn] >= 0)] = ORANGE
    dbg[attach] = RED
    o3d.io.write_triangle_mesh(
        str(out_dir / "T1_detection_debug.ply"), colored(mesh, dbg))

    x = np.clip(h_orig / 0.4, -1, 1)
    pos, neg = np.clip(x, 0, 1), np.clip(-x, 0, 1)
    heat = np.stack([1 - neg, 1 - pos - neg, 1 - pos], axis=1)
    o3d.io.write_triangle_mesh(
        str(out_dir / "T1_height_map.ply"), colored(mesh, heat))

    np.save(out_dir / "T1_attachment_labels.npy", labels_out)
    print(f"Saved to {out_dir}")

    if args.show:
        o3d.visualization.draw_geometries(
            [colored(mesh, heat)], window_name="h map (red = above base surface, +-0.4 mm)",
            width=1024, height=768)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Detect attachments on T1 (simple version 2).")
    ap.add_argument("--sample", required=True,
                    help="Sample folder name, e.g. 2")
    ap.add_argument("--height-threshold", type=float, default=HEIGHT_THRESHOLD)
    ap.add_argument("--radius", type=float, default=FIT_RADIUS)
    ap.add_argument("--voxel", type=float, default=VOXEL)
    ap.add_argument("--buccal-dot", type=float, default=BUCCAL_DOT,
                    help="set to -1 to disable the buccal filter")
    ap.add_argument("--eps", type=float, default=DBSCAN_EPS)
    ap.add_argument("--min-area", type=float, default=MIN_AREA)
    ap.add_argument("--max-area", type=float, default=MAX_AREA)
    ap.add_argument("--flip-normals", action="store_true")
    ap.add_argument(
        "--show", action=argparse.BooleanOptionalAction, default=True)
    main(ap.parse_args())
