"""
Phase 1 - Step 2 : automatic attachment detection on one scan (T1 or T2).

Idea: an attachment is a bump above the tooth surface. The tooth surface under it is
estimated from a RING AROUND the point (never from the attachment itself).

Stage 1 (seeds, coarse, uniform 0.3 mm working points):
  for every buccal point, fit a quadric to the annulus r_in..r_out (default 2.0..3.5 mm)
  using only annulus points with a similar normal. h = height of the point above the fit.
  The fit is rejected if the annulus is not a clean single surface (too few points,
  angular coverage too small, residual rms too large): gingival margins, grooves and
  crown edges are therefore "invalid", not "bumps".
  Seeds = valid points with h >= SEED_HEIGHT; DBSCAN groups them (no manual picks).
  Seed clusters with a small area (< MIN_SEED_AREA) are edge effects and are dropped.
Stage 2 (boundary, native mesh resolution):
  per cluster (largest seed area first), fit the base surface to a thin ring just outside
  the current region (wall points removed by a normal filter), grow the region to vertices
  with h >= EDGE_HEIGHT, repeat a few times. The LAST ring fit of this loop is THE base
  surface of the attachment: every number written by this project (h95, h_top, volume)
  uses the h of this one fit. Nothing is re-fitted later.
  Cleanup (selection only, the base surface is not touched): drop vertices below
  EDGE_REL x h95 (low bulges of neighbouring teeth) and keep only the largest connected
  piece on the mesh (by surface area). Attachment = what remains.
  Two attachments closer than MIN_SEPARATION are treated as duplicates (one per tooth).
  Attachments whose ring fit is poor or whose region is unusually large get
  qc_flag = "check" (base surface probably biased; measure them separately).

This script does NOT compute volumes. src/measure_volume.py reads the labels, the heights
and the CSV written here.

Input : data/<sample>/<scan>.stl   (scan = T1 or T2; detection does not need alignment)
Output: data/processed/<sample>/
          <scan>_segmented.ply         gray mesh, attachments red
          <scan>_detection_debug.ply   gray = not evaluated, light green = evaluated,
                                       orange = seeds, red = attachments
          <scan>_height_map.ply        stage-1 h (smooth), red = above base, blue = below
          <scan>_attachment_labels.npy int32 per vertex of the CLEANED mesh, -1 = background
          <scan>_attachment_height.npy float32 per vertex: h above the base surface (mm)
                                       for attachment vertices, NaN elsewhere
          <scan>_attachments.csv       one row per attachment (coordinates in the scan frame)

Usage:
    python src/detect_attachments.py --sample 1
    python src/detect_attachments.py --sample 2 --scan T2
    python src/detect_attachments.py --sample 2 --seed-height 0.4
    python src/detect_attachments.py --sample 2 --max-rms 0.2
    python src/detect_attachments.py --sample 2 --min-seed-area 2.0
    python src/detect_attachments.py --sample 2 --show heat
"""

import argparse
import csv

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from config import DATA_DIR, PROCESSED_DIR


# ---- All lengths in millimetres ------------------------------------------------------
# spacing of the working point set (same for fine/coarse scans)
VOXEL = 0.3

# Stage 1: annulus fit
ANN_IN = 2.0              # annulus inner radius (outside the attachment wall)
ANN_OUT = 3.5             # annulus outer radius
# annulus points must have a normal this similar to the centre normal
ANN_NORMAL_DOT = 0.65
ANN_MIN_POINTS = 60
# of 8 angular sectors that must contain points (no extrapolation)
ANN_MIN_SECTORS = 5
ANN_MAX_RMS = 0.15        # residual rms of the annulus fit; larger = not a clean surface
SEED_HEIGHT = 0.5         # seed if this far above the annulus surface
BORDER_MARGIN = 1.0       # ignore points closer than this to the scan border
BUCCAL_DOT = 0.35         # min. alignment of normal with the outward arch direction
DBSCAN_EPS = 0.7          # (at least 2.2 x voxel)
DBSCAN_MIN_SAMPLES = 5
SEED_MIN_AREA = 0.5       # mm^2, clusters below this are ignored silently
# mm^2. true: case 1 >= 3.35, case 2 >= 2.68, case 12 >= 2.48; false: case 1 <= 1.40, case 2 <= 2.05
MIN_SEED_AREA = 2.3
# mm. smallest true spacing seen in cases 1, 2, 12 is about 7.0; false duplicate in case 12 was 5.0
MIN_SEPARATION = 5.5

# Stage 2: ring fit and region growing (native mesh vertices)
PATCH_RADIUS = 6.0        # vertices around the cluster centre that are considered
SEED_RADIUS = 0.25        # initial region = vertices this close to seed points
RING_IN = 0.4             # thin ring just outside the region
RING_OUT = 1.6
RING_NORMAL_DOT = 0.8     # removes the steep wall from the ring
EDGE_HEIGHT = 0.25        # region = vertices at least this far above the base surface
EDGE_REL = 0.3            # cleanup: also at least this fraction of the region's h95
GROW_STEP = 1.0           # region can grow by at most this per iteration
MAX_EXTENT = 3.0          # region stays within this distance of the cluster centre
REFINE_ITERS = 4
MIN_REGION_VERTICES = 20

# Final acceptance
MIN_AREA = 2.0            # mm^2 (footprint, real surface area)
MAX_AREA = 25.0
MIN_TOP_H = 0.5           # mm, 95th percentile of h inside the region
# true: case 1 0.77-1.05, case 2 1.07-1.77, case 12 1.26-2.23; wrong base: >= 2.7
MAX_TOP_H = 2.5

# QC flag (not a rejection): base surface probably biased
QC_MAX_RING_RMS = 0.10    # mm; kept attachments in cases 1, 2 were 0.025-0.088
QC_MAX_AREA = 16.0        # mm^2; kept attachments in case 2 were 11-15.5

SCANS = ("T1", "T2")

GRAY = [0.72, 0.72, 0.72]
GREEN = [0.80, 0.93, 0.80]
RED = [1.0, 0.0, 0.0]
ORANGE = [1.0, 0.55, 0.0]


# ---- helpers -------------------------------------------------------------------------
def load_mesh(path):
    """
    The one place where a scan is read and cleaned. Vertex order of the result is the order
    of the label / height files, so every script must load meshes through this function.
    """
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


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


def largest_component(idx, edges, areas, n):
    """
    Boolean mask over idx: the mesh-connected piece (through mesh edges) with the largest
    surface area. Removes separate blobs that were grown onto neighbouring teeth.
    """
    local = np.full(n, -1, dtype=np.int64)
    local[idx] = np.arange(len(idx))
    sel = (local[edges[:, 0]] >= 0) & (local[edges[:, 1]] >= 0)
    e = local[edges[sel]]
    m = len(idx)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(m, m))
    n_comp, lab = connected_components(g, directed=False)
    if n_comp == 1:
        return np.ones(m, dtype=bool)
    size = np.bincount(lab, weights=areas[idx], minlength=n_comp)
    return lab == int(size.argmax())


def arch_geometry(points):
    """Occlusal axis (smallest-variance PCA axis) and arch centre (median)."""
    centre = points.mean(axis=0)
    cov = (points - centre).T @ (points - centre) / len(points)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, 0], np.median(points, axis=0)


def radial_directions(points, occlusal, mid):
    """Unit vectors from the arch centre, perpendicular to the occlusal axis."""
    v = points - mid
    v = v - np.outer(v @ occlusal, occlusal)
    r = np.linalg.norm(v, axis=1)
    return v / (r[:, None] + 1e-9), r


def flat_dot(normals, radial, occlusal):
    flat = normals - np.outer(normals @ occlusal, occlusal)
    return np.einsum("ij,ij->i", flat, radial)


def tangent_frame(n):
    helper = np.array([1.0, 0, 0]) if abs(
        n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def design_matrix(u, v):
    return np.stack([u * u, u * v, v * v, u, v, np.ones_like(u)], axis=1)


def fit_quadric_robust(u, v, w, passes=3, k=3.0, floor=0.06):
    """
    Fit w = a u^2 + b uv + c v^2 + d u + e v + f with symmetric trimming (|res| < max(k sigma, floor)).
    Symmetric on purpose: one-sided trimming at the noise level sinks the surface (failure v3).
    Returns (coef, rms of kept residuals, kept fraction) or None.
    """
    A = design_matrix(u, v)
    keep = np.ones(len(u), dtype=bool)
    coef = None
    res = None
    for _ in range(passes):
        if keep.sum() < 12:
            return None
        coef = np.linalg.lstsq(A[keep], w[keep], rcond=None)[0]
        res = w - A @ coef
        med = np.median(res[keep])
        sigma = 1.4826 * np.median(np.abs(res[keep] - med))
        keep = np.abs(res) < max(k * sigma, floor)
    if keep.sum() < 12 or keep.mean() < 0.5:
        return None
    return coef, float(np.sqrt(np.mean(res[keep] ** 2))), float(keep.mean())


# ---- stage 1 -------------------------------------------------------------------------
def annulus_height(pts, nrm, query_idx, a):
    """
    h[i] = height of point i above a quadric fitted to its annulus (a.ann_in..a.ann_out).
    Returns h, rms, ok (fit accepted), base_normal.
    """
    tree = cKDTree(pts)
    neigh = tree.query_ball_point(pts[query_idx], r=a.ann_out, workers=-1)
    h = np.zeros(len(pts))
    rms = np.full(len(pts), np.nan)
    ok = np.zeros(len(pts), dtype=bool)
    bn = nrm.copy()
    for i, idx in zip(query_idx, neigh):
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) < a.ann_min_points:
            continue
        n = nrm[i]
        rel = pts[idx] - pts[i]
        dist = np.linalg.norm(rel, axis=1)
        sel = (dist >= a.ann_in) & (nrm[idx] @ n >= a.ann_normal_dot)
        if sel.sum() < a.ann_min_points:
            continue
        e1, e2 = tangent_frame(n)
        rel = rel[sel]
        u, v, w = rel @ e1, rel @ e2, rel @ n
        sectors = ((np.arctan2(v, u) + np.pi) / (np.pi / 4)).astype(int) % 8
        if len(np.unique(sectors)) < a.ann_min_sectors:
            continue
        fit = fit_quadric_robust(u, v, w)
        if fit is None:
            continue
        coef, r, _ = fit
        rms[i] = r
        if r > a.max_rms:
            continue
        h[i] = -coef[5]
        # normal of the base surface at the point
        b = n - coef[3] * e1 - coef[4] * e2
        bn[i] = b / np.linalg.norm(b)
        ok[i] = True
    return h, rms, ok, bn


# ---- stage 2 -------------------------------------------------------------------------
def refine_cluster(points, normals, vtree, centre, n_ref, seed_pts):
    """
    Ring-based base surface + region growing on native vertices. Returns dict or None.
    The returned h is the height above the LAST ring fit (the base surface of this attachment).
    """
    patch = np.asarray(vtree.query_ball_point(
        centre, r=PATCH_RADIUS), dtype=np.int64)
    if len(patch) < 100:
        return None
    P, N = points[patch], normals[patch]
    e1, e2 = tangent_frame(n_ref)
    rel = P - centre
    u, v, w = rel @ e1, rel @ e2, rel @ n_ref
    A = design_matrix(u, v)
    near_c = np.linalg.norm(rel, axis=1) <= MAX_EXTENT
    normal_ok = N @ n_ref >= RING_NORMAL_DOT
    region = (cKDTree(seed_pts).query(P)[0] <= SEED_RADIUS) & near_c

    def fit_ring(reg):
        d = cKDTree(P[reg]).query(P)[0]
        ring = (d >= RING_IN) & (d <= RING_OUT) & normal_ok
        if ring.sum() < 50:
            return None
        fit = fit_quadric_robust(u[ring], v[ring], w[ring])
        if fit is None:
            return None
        return d, w - A @ fit[0], fit[1], int(ring.sum())

    for _ in range(REFINE_ITERS):
        if not region.any():
            return None
        res = fit_ring(region)
        if res is None:
            break
        d, h, _, _ = res
        new = (d <= GROW_STEP) & near_c & (h >= EDGE_HEIGHT)
        if not new.any():
            break
        done = np.array_equal(new, region)
        region = new
        if done:
            break
    if not region.any():
        return None
    res = fit_ring(region)
    if res is None:
        return None
    _, h, rms, n_ring = res
    return dict(idx=patch[region], h=h[region], rms=rms, n_ring=n_ring)


# ---- validation against the old manual picks (diagnostic only, T1) -------------------
def check_picks(out_dir, points, labels, centroids, wtree, h, rms, ok, valid):
    f = out_dir / "picks_T1.npy"
    if not f.exists():
        return
    p = np.load(f)
    if p.ndim != 2 or p.shape[1] != 3:
        print(
            f"\n[picks] {f.name} is not a (k,3) coordinate array (shape {p.shape}); skipped.")
        return
    cands = {"as saved": p}
    tf = out_dir / "T1_to_T2.npy"
    if tf.exists():
        M = np.linalg.inv(np.load(tf))
        cands["other frame"] = p @ M[:3, :3].T + M[:3, 3]
    att = labels >= 0
    if not att.any():
        print("\n[picks] no attachments detected, nothing to compare.")
        return
    atree = cKDTree(points[att])
    name = min(cands, key=lambda k: atree.query(cands[k])[0].sum())
    q = cands[name]
    print(
        f"\n[picks] comparing {len(q)} manual picks (frame: {name}); diagnostic only")
    found = 0
    for k, pt in enumerate(q):
        d_reg = atree.query(pt)[0]
        wi = wtree.query(pt)[1]
        hit = d_reg <= 1.0
        found += hit
        print(f"  pick {k}: dist to nearest attachment region {d_reg:5.2f} mm "
              f"-> {'FOUND' if hit else 'MISSED'} | stage-1 at pick: h={h[wi]:.2f} "
              f"rms={rms[wi]:.3f} valid_region={bool(valid[wi])} fit_ok={bool(ok[wi])}")
    print(
        f"  picks found: {found}/{len(q)}; detected attachments: {len(centroids)}")


# ---- main ----------------------------------------------------------------------------
def colored(mesh, colors):
    m = o3d.geometry.TriangleMesh(mesh)
    m.vertex_colors = o3d.utility.Vector3dVector(colors)
    return m


def main(args):
    scan = args.scan
    in_path = DATA_DIR / args.sample / f"{scan}.stl"
    print(f"Input: {in_path}")
    if not in_path.exists():
        raise SystemExit(f"Missing file: {in_path}")

    mesh = load_mesh(in_path)
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals).copy()
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

    # 2. Orient normals outward (outermost 15 % of the arch must face away from the centre)
    occ, mid = arch_geometry(dpts)
    radial, r = radial_directions(dpts, occ, mid)
    outer = r >= np.percentile(r, 85)
    frac = float((flat_dot(dnrm[outer], radial[outer], occ) > 0).mean())
    flip = (frac < 0.5) != args.flip_normals
    print(f"Outward-facing fraction on the outer arch: {frac:.2f} -> "
          f"{'flipping' if flip else 'keeping'} normals")
    if flip:
        dnrm, normals = -dnrm, -normals
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    wtree = cKDTree(dpts)
    dist4, idx4 = wtree.query(points, k=4, workers=-1)
    nn = idx4[:, 0]
    areas = vertex_areas(points, faces)
    wp_area = np.bincount(nn, weights=areas, minlength=len(dpts))
    mesh_edges = edges_of(faces)

    # 3. Which working points are evaluated
    border = border_vertices(faces, n)
    if border.any():
        far = cKDTree(points[border]).query(dpts)[0] > BORDER_MARGIN
    else:
        far = np.ones(len(dpts), dtype=bool)
    valid_region = far & (flat_dot(dnrm, radial, occ) >= args.buccal_dot)
    query_idx = np.where(valid_region)[0]
    print(f"Evaluated (buccal, away from border): {len(query_idx)} points")

    # 4. Stage 1
    print(
        f"Stage 1: annulus fits ({args.ann_in}-{args.ann_out} mm) ... takes a little while")
    h, rms, ok, bn = annulus_height(dpts, dnrm, query_idx, args)
    ok &= valid_region
    print(f"Fits accepted: {ok.sum()} of {len(query_idx)} "
          f"({100 * ok.sum() / max(len(query_idx), 1):.0f} %)")
    if ok.any():
        p = np.percentile(h[ok], [1, 50, 99, 99.9])
        pr = np.nanpercentile(rms[query_idx], [10, 50, 90])
        print(f"h (accepted) mm: p1 {p[0]:.3f} | p50 {p[1]:.3f} | p99 {p[2]:.3f} | "
              f"p99.9 {p[3]:.3f} | max {h[ok].max():.3f}")
        print(
            f"annulus rms (all evaluated) mm: p10 {pr[0]:.3f} | p50 {pr[1]:.3f} | p90 {pr[2]:.3f}")

    seeds = np.where(ok & (h >= args.seed_height))[0]
    print(f"Seed points (h >= {args.seed_height} mm): {len(seeds)}")

    # 5. Cluster seeds automatically
    lab_seed = np.full(len(dpts), -1, dtype=np.int32)
    if len(seeds) >= DBSCAN_MIN_SAMPLES:
        eps = max(args.eps, 2.2 * args.voxel)
        lab_seed[seeds] = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES,
                                 n_jobs=-1).fit_predict(dpts[seeds])
        print(f"DBSCAN eps = {eps:.2f} mm")
    else:
        print("[!] Too few seeds. Try --seed-height 0.4 or --max-rms 0.2")

    # 6. Stage 2 per cluster, largest seed area first
    vtree = cKDTree(points)
    clusters = []
    for c in sorted(set(lab_seed) - {-1}):
        mem = np.where(lab_seed == c)[0]
        area = float(wp_area[mem].sum())
        if area < SEED_MIN_AREA:
            continue
        clusters.append((area, c, mem, h[mem].max()))
    clusters.sort(key=lambda t: -t[0])
    print(f"\nSeed clusters: {len(clusters)}")

    labels_out = np.full(n, -1, dtype=np.int32)
    height_out = np.full(n, np.nan, dtype=np.float32)
    rows = []
    print("Clusters (largest seed area first, refined on native vertices):")
    for seed_area, c, mem, hmax_seed in clusters:
        centre = dpts[mem].mean(axis=0)
        tag = f"seed_area={seed_area:5.2f} h_seed_max={hmax_seed:.2f} at {np.round(centre, 1)}"
        if seed_area < args.min_seed_area:
            print(
                f"  [skip] {tag} | seed area < {args.min_seed_area} (edge effect)")
            continue
        n_ref = bn[mem].mean(axis=0)
        n_ref /= np.linalg.norm(n_ref)
        res = refine_cluster(points, normals, vtree, centre, n_ref, dpts[mem])
        if res is None:
            print(f"  [skip] {tag}: refinement failed (no clean ring)")
            continue
        idx, hh = res["idx"], res["h"]

        # cleanup (selection only): relative height cut, then the largest connected piece
        n_before = len(idx)
        cut = hh >= max(EDGE_HEIGHT, EDGE_REL * float(np.percentile(hh, 95)))
        idx, hh = idx[cut], hh[cut]
        if len(idx) < MIN_REGION_VERTICES:
            print(f"  [skip] {tag}: region vanished after cleanup")
            continue
        comp = largest_component(idx, mesh_edges, areas, n)
        idx, hh = idx[comp], hh[comp]
        if len(idx) < MIN_REGION_VERTICES:
            print(f"  [skip] {tag}: region too small after cleanup")
            continue
        trim = 100.0 * (1 - len(idx) / n_before)

        area = float(areas[idx].sum())
        h95 = float(np.percentile(hh, 95))
        h_top = float(hh[hh >= 0.8 * h95].mean())
        overlap = (labels_out[idx] >= 0).mean()
        cen = points[idx].mean(axis=0)
        reasons = []
        if overlap > 0.3:
            reasons.append("duplicate of a kept attachment")
        if rows:
            kept_c = np.array([[r_["cx"], r_["cy"], r_["cz"]] for r_ in rows])
            dmin = np.linalg.norm(kept_c - cen, axis=1)
            j = int(dmin.argmin())
            if dmin[j] < args.min_separation:
                reasons.append(
                    f"{dmin[j]:.1f} mm from kept #{j} (< {args.min_separation})")
        if not (args.min_area <= area <= args.max_area):
            reasons.append(
                f"area {area:.1f} outside {args.min_area}-{args.max_area}")
        if not (MIN_TOP_H <= h95 <= MAX_TOP_H):
            reasons.append(f"h95 {h95:.2f} outside {MIN_TOP_H}-{MAX_TOP_H}")
        info = (f"area={area:5.2f} mm2 h95={h95:.2f} h_top={h_top:.2f} "
                f"ring_rms={res['rms']:.3f} n_ring={res['n_ring']} trim={trim:.0f}%")
        if reasons:
            print(f"  [skip] {tag} | {info} | {'; '.join(reasons)}")
            continue
        qc = "check" if (
            res["rms"] > QC_MAX_RING_RMS or area > QC_MAX_AREA) else "ok"
        k = len(rows)
        labels_out[idx] = k
        height_out[idx] = hh
        rows.append(dict(id=k, cx=cen[0], cy=cen[1], cz=cen[2], area_mm2=area,
                         h95_mm=h95, h_top_mm=h_top, n_vertices=len(idx),
                         ring_rms_mm=res["rms"], nx=n_ref[0], ny=n_ref[1], nz=n_ref[2],
                         qc_flag=qc))
        print(
            f"  [KEEP #{k}{' QC=check' if qc == 'check' else ''}] {tag} | {info}")
    attach = labels_out >= 0
    n_check = sum(r_["qc_flag"] == "check" for r_ in rows)
    print(
        f"\nAttachments kept: {len(rows)} (expected up to ~14), flagged 'check': {n_check}")

    # 7. Save
    out_dir = PROCESSED_DIR / args.sample
    out_dir.mkdir(parents=True, exist_ok=True)

    # smooth stage-1 height map on the native mesh (inverse-distance weights, 4 neighbours)
    wgt = 1.0 / (dist4 + 1e-6)
    wgt /= wgt.sum(axis=1, keepdims=True)
    h_orig = (np.where(ok, h, 0.0)[idx4] * wgt).sum(axis=1)
    x = np.clip(h_orig / 0.4, -1, 1)
    pos, neg = np.clip(x, 0, 1), np.clip(-x, 0, 1)
    heat = np.stack([1 - neg, 1 - pos - neg, 1 - pos], axis=1)
    heat_mesh = colored(mesh, heat)
    o3d.io.write_triangle_mesh(
        str(out_dir / f"{scan}_height_map.ply"), heat_mesh)

    colors = np.tile(GRAY, (n, 1))
    colors[attach] = RED
    seg = colored(mesh, colors)
    o3d.io.write_triangle_mesh(str(out_dir / f"{scan}_segmented.ply"), seg)

    dbg = np.tile(GRAY, (n, 1))
    dbg[ok[nn] & (dist4[:, 0] <= 1.5 * args.voxel)] = GREEN
    dbg[(lab_seed[nn] >= 0) & (dist4[:, 0] <= 1.5 * args.voxel)] = ORANGE
    dbg[attach] = RED
    o3d.io.write_triangle_mesh(
        str(out_dir / f"{scan}_detection_debug.ply"), colored(mesh, dbg))

    np.save(out_dir / f"{scan}_attachment_labels.npy", labels_out)
    np.save(out_dir / f"{scan}_attachment_height.npy", height_out)
    if rows:
        with open(out_dir / f"{scan}_attachments.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    print(f"Saved to {out_dir}")

    if scan == "T1":
        check_picks(out_dir, points, labels_out, rows,
                    wtree, h, rms, ok, valid_region)

    if args.show != "none":
        o3d.visualization.draw_geometries(
            [seg if args.show == "seg" else heat_mesh],
            window_name=("attachments (red)" if args.show == "seg"
                         else "h map (red = above base, +-0.4 mm)"),
            width=1024, height=768)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Detect attachments on one scan automatically (v4.5).")
    ap.add_argument("--sample", required=True,
                    help="Sample folder name, e.g. 2")
    ap.add_argument("--scan", choices=SCANS, default="T1",
                    help="which scan to process")
    ap.add_argument("--voxel", type=float, default=VOXEL)
    ap.add_argument("--ann-in", type=float, default=ANN_IN)
    ap.add_argument("--ann-out", type=float, default=ANN_OUT)
    ap.add_argument("--ann-normal-dot", type=float, default=ANN_NORMAL_DOT)
    ap.add_argument("--ann-min-points", type=int, default=ANN_MIN_POINTS)
    ap.add_argument("--ann-min-sectors", type=int, default=ANN_MIN_SECTORS)
    ap.add_argument("--max-rms", type=float, default=ANN_MAX_RMS)
    ap.add_argument("--seed-height", type=float, default=SEED_HEIGHT)
    ap.add_argument("--min-seed-area", type=float, default=MIN_SEED_AREA,
                    help="drop seed clusters smaller than this (mm^2)")
    ap.add_argument("--min-separation", type=float, default=MIN_SEPARATION,
                    help="min. distance between two attachments (mm)")
    ap.add_argument("--buccal-dot", type=float, default=BUCCAL_DOT,
                    help="set to -1 to disable the buccal filter")
    ap.add_argument("--eps", type=float, default=DBSCAN_EPS)
    ap.add_argument("--min-area", type=float, default=MIN_AREA)
    ap.add_argument("--max-area", type=float, default=MAX_AREA)
    ap.add_argument("--flip-normals", action="store_true",
                    help="invert the automatic normal orientation decision")
    ap.add_argument("--show", choices=["seg", "heat", "none"], default="seg")
    main(ap.parse_args())
