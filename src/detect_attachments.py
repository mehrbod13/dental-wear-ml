"""
Automatic attachment detection on one scan (T1, T2 or T1_aligned).

Attachments are bumps above the tooth surface. Each one is found in two steps:
1. Seeds: for each buccal point, a quadric is fitted to a ring around it. Points that
   sit clearly above that surface are seeds, grouped with DBSCAN.
2. Region growing: for each seed cluster, the base surface is fitted to a thin ring just
   outside the region, and the region grows to the vertices above that base. The base
   from the last ring fit is used for the height and the area.

Clusters with a large seed area are accepted directly. Smaller ones only fill free slots
up to EXPECTED_COUNT, and only with a clean ring fit. Candidates are rejected for overlap,
spacing, area or height. qc_flag = "check" marks a poor ring fit or a large area; it does
not reject the attachment.

Numbering runs along the arch, starting from the end with the smaller lateral coordinate.
--reverse flips the order; --show-ids draws the ids.

T1_aligned is T1 moved into the T2 frame by align.py. Its label and height files index
T1_aligned.stl, not T1.stl. Volumes are computed by measure_volume.py.

Input : data/<sample>/<scan>.stl   (T1_aligned: data/processed/<sample>/T1_aligned.stl)
Output: data/processed/<sample>/
          <scan>_segmented.ply           attachments in red
          <scan>_attachment_labels.npy   int32 per vertex, -1 = background
          <scan>_attachment_height.npy   float32 height in mm, NaN elsewhere
          <scan>_attachments.csv         one row per attachment
        with --debug also:
          <scan>_detection_debug.ply     gray = not evaluated, green = evaluated,
                                         orange = seeds, red = attachments
          <scan>_height_map.ply          red = above base, blue = below

Usage:
    python src/detect_attachments.py --sample 1
    python src/detect_attachments.py --sample 2 --scan T2 --show none
    python src/detect_attachments.py --sample 1 --show-ids
    python src/detect_attachments.py --sample 1 --debug --show none
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
# mm^2. Clusters at or above this are accepted on their own.
# true: case 1 >= 3.35, case 2 T1 >= 2.68, case 12 >= 2.48; case 2 T2 has one true
# attachment at 0.78 (fragmented seeds). false: 1.38 (T1), 1.63 (T2). Seed area cannot
# separate them, so smaller clusters only fill free slots (see EXPECTED_COUNT).
MIN_SEED_AREA = 2.3
# attachments per scan (reference Excel); 0 = no slot filling
EXPECTED_COUNT = 14
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
# true: case 1 0.77-1.05, case 2 1.02-2.01, case 12 1.26-2.23; wrong base: >= 2.7
MAX_TOP_H = 2.5

# QC flag (not a rejection): base surface probably biased.
# Also the limit for attachments accepted from weak seed clusters.
QC_MAX_RING_RMS = 0.10    # mm; kept attachments in case 2 were 0.038-0.160
QC_MAX_AREA = 16.0        # mm^2; kept attachments in case 2 were 9-16.3

# --show-ids: length of the leader line drawn outward from each attachment
LEADER_LENGTH = 6.0

SCANS = ("T1", "T2", "T1_aligned")

GRAY = [0.72, 0.72, 0.72]
GREEN = [0.80, 0.93, 0.80]
RED = [1.0, 0.0, 0.0]
ORANGE = [1.0, 0.55, 0.0]


# ---- helpers -------------------------------------------------------------------------
def scan_path(sample, scan):
    """Raw scans are in data/<sample>; T1_aligned.stl is written by align.py."""
    if scan == "T1_aligned":
        return PROCESSED_DIR / sample / "T1_aligned.stl"
    return DATA_DIR / sample / f"{scan}.stl"


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


def colored(mesh, colors):
    m = o3d.geometry.TriangleMesh(mesh)
    m.vertex_colors = o3d.utility.Vector3dVector(colors)
    return m


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
    Fit w = a u^2 + b uv + c v^2 + d u + e v + f with symmetric trimming
    (|res| < max(k sigma, floor)). Symmetric on purpose: one-sided trimming at the noise
    level sinks the surface.
    Returns (coef, rms of kept residuals) or None.
    """
    A = design_matrix(u, v)
    keep = np.ones(len(u), dtype=bool)
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
    return coef, float(np.sqrt(np.mean(res[keep] ** 2)))


def arch_order(centres, occlusal, reverse=False):
    """
    Indices that sort attachment centres along the arch.
    Angles are measured around the centroid in the plane perpendicular to the occlusal axis;
    the sequence starts after the largest angular gap (the arch opening). Direction: the
    first attachment is at the end with the smaller coordinate along the lateral axis (the
    in-plane axis with the larger spread, sign chosen so its largest component is positive).
    """
    n = len(centres)
    if n < 3:
        return np.arange(n)[::-1] if reverse else np.arange(n)
    e1, e2 = tangent_frame(occlusal)
    X = centres - centres.mean(axis=0)
    p = np.stack([X @ e1, X @ e2], axis=1)
    ang = np.arctan2(p[:, 1], p[:, 0])
    order = np.argsort(ang)
    s = ang[order]
    gaps = np.diff(np.r_[s, s[0] + 2 * np.pi])
    order = np.roll(order, -((int(gaps.argmax()) + 1) % n))
    lat2 = np.linalg.eigh(p.T @ p)[1][:, 1]
    lat = lat2[0] * e1 + lat2[1] * e2
    if lat[np.abs(lat).argmax()] < 0:
        lat = -lat
    if centres[order[0]] @ lat > centres[order[-1]] @ lat:
        order = order[::-1]
    return order[::-1] if reverse else order


def height_map_mesh(mesh, h, ok, dist4, idx4):
    """Stage-1 h interpolated to the native vertices: red = above base, blue = below."""
    wgt = 1.0 / (dist4 + 1e-6)
    wgt /= wgt.sum(axis=1, keepdims=True)
    x = np.clip((np.where(ok, h, 0.0)[idx4] * wgt).sum(axis=1) / 0.4, -1, 1)
    pos, neg = np.clip(x, 0, 1), np.clip(-x, 0, 1)
    return colored(mesh, np.stack([1 - neg, 1 - pos - neg, 1 - pos], axis=1))


# ---- stage 1 -------------------------------------------------------------------------
def annulus_height(pts, nrm, query_idx, a):
    """
    h[i] = height of point i above a quadric fitted to its annulus (a.ann_in..a.ann_out).
    Returns h, ok (fit accepted), base_normal.
    """
    tree = cKDTree(pts)
    neigh = tree.query_ball_point(pts[query_idx], r=a.ann_out, workers=-1)
    h = np.zeros(len(pts))
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
        coef, r = fit
        if r > a.max_rms:
            continue
        h[i] = -coef[5]
        # normal of the base surface at the point
        b = n - coef[3] * e1 - coef[4] * e2
        bn[i] = b / np.linalg.norm(b)
        ok[i] = True
    return h, ok, bn


# ---- stage 2 -------------------------------------------------------------------------
def refine_cluster(points, normals, vtree, centre, n_ref, seed_pts,
                   edge_height=EDGE_HEIGHT, ring_in=RING_IN):
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
    if not region.any():
        return None

    def fit_ring(reg):
        d = cKDTree(P[reg]).query(P)[0]
        ring = (d >= ring_in) & (d <= RING_OUT) & normal_ok
        if ring.sum() < 50:
            return None
        fit = fit_quadric_robust(u[ring], v[ring], w[ring])
        if fit is None:
            return None
        return d, w - A @ fit[0], fit[1], int(ring.sum())

    for _ in range(REFINE_ITERS):
        res = fit_ring(region)
        if res is None:
            break
        d, h, _, _ = res
        new = (d <= GROW_STEP) & near_c & (h >= edge_height)
        if not new.any():
            break
        done = np.array_equal(new, region)
        region = new
        if done:
            break
    res = fit_ring(region)
    if res is None:
        return None
    _, h, rms, n_ring = res
    return dict(idx=patch[region], h=h[region], rms=rms, n_ring=n_ring)


def grow_candidate(mem, dpts, bn, ctx, args):
    """
    Stage 2 + cleanup for one seed cluster (member indices into the working points).
    Returns (candidate dict, "") or (None, reason).
    """
    points, normals, vtree, areas, edges = ctx
    centre = dpts[mem].mean(axis=0)
    n_ref = bn[mem].mean(axis=0)
    n_ref /= np.linalg.norm(n_ref)
    res = refine_cluster(points, normals, vtree, centre, n_ref, dpts[mem],
                         edge_height=args.edge_height, ring_in=args.ring_in)
    if res is None:
        return None, "refinement failed (no clean ring)"
    idx, hh = res["idx"], res["h"]

    # cleanup (selection only): relative height cut, then the largest connected piece
    n_before = len(idx)
    cut = hh >= max(args.edge_height, args.edge_rel *
                    float(np.percentile(hh, 95)))
    idx, hh = idx[cut], hh[cut]
    if len(idx) < MIN_REGION_VERTICES:
        return None, "region vanished after cleanup"
    comp = largest_component(idx, edges, areas, len(points))
    idx, hh = idx[comp], hh[comp]
    if len(idx) < MIN_REGION_VERTICES:
        return None, "region too small after cleanup"

    area = float(areas[idx].sum())
    h95 = float(np.percentile(hh, 95))
    cand = dict(
        idx=idx, h=hh, cen=points[idx].mean(axis=0), area=area, h95=h95,
        h_top=float(hh[hh >= 0.8 * h95].mean()), rms=res["rms"], n_ring=res["n_ring"],
        trim=100.0 * (1 - len(idx) / n_before), n_ref=n_ref,
        qc="check" if (res["rms"] > QC_MAX_RING_RMS or area > QC_MAX_AREA) else "ok")
    return cand, ""


def gate_reasons(c, kept, taken, args):
    """Reasons to reject a candidate given the attachments kept so far (empty = accept)."""
    reasons = []
    if taken[c["idx"]].mean() > 0.3:
        reasons.append("duplicate of a kept attachment")
    if kept:
        dmin = np.linalg.norm(np.array([k["cen"]
                                        for k in kept]) - c["cen"], axis=1).min()
        if dmin < args.min_separation:
            reasons.append(
                f"{dmin:.1f} mm from a kept attachment (< {args.min_separation})")
    if not (args.min_area <= c["area"] <= args.max_area):
        reasons.append(
            f"area {c['area']:.1f} outside {args.min_area}-{args.max_area}")
    if not (MIN_TOP_H <= c["h95"] <= MAX_TOP_H):
        reasons.append(f"h95 {c['h95']:.2f} outside {MIN_TOP_H}-{MAX_TOP_H}")
    return reasons


def describe(c):
    return (f"area={c['area']:5.2f} mm2 h95={c['h95']:.2f} h_top={c['h_top']:.2f} "
            f"ring_rms={c['rms']:.3f} n_ring={c['n_ring']} trim={c['trim']:.0f}%")


def evaluate_cluster(seed_area, mem, dpts, bn, h, ctx, kept, taken, args):
    """Returns (tag, candidate or None, list of rejection reasons)."""
    centre = dpts[mem].mean(axis=0)
    tag = (f"seed_area={seed_area:5.2f} h_seed_max={h[mem].max():.2f} "
           f"at {np.round(centre, 1)}")
    c, why = grow_candidate(mem, dpts, bn, ctx, args)
    if c is None:
        return tag, None, [why]
    c["tag"] = tag
    return tag, c, gate_reasons(c, kept, taken, args)


def report_skip(tag, c, reasons):
    info = f" | {describe(c)}" if c is not None else ""
    print(f"  [skip] {tag}{info} | {'; '.join(reasons)}")


# ---- viewer --------------------------------------------------------------------------
def show_ids(seg, rows):
    """
    Segmented mesh + for every attachment a black leader line (outward) with its id at the
    tip. A blue -> red line joins the tips in id order (blue = first).
    """
    cen = np.array([[r["cx"], r["cy"], r["cz"]] for r in rows])
    nrm = np.array([[r["nx"], r["ny"], r["nz"]] for r in rows])
    tips = cen + LEADER_LENGTH * nrm
    m = len(rows)
    lines = [[i, m + i]
             for i in range(m)] + [[m + i, m + i + 1] for i in range(m - 1)]
    colors = [[0, 0, 0]] * m + [[f, 0, 1 - f]
                                for f in np.linspace(0, 1, m - 1)]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.vstack([cen, tips])),
                              o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(colors)
    title = "attachment ids (blue -> red = id order)"
    try:
        gui = o3d.visualization.gui
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer(title, 1280, 800)
        vis.show_settings = False
        vis.add_geometry("mesh", seg)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 3.0
        vis.add_geometry("order", ls, mat)
        for i, p in enumerate(tips):
            vis.add_3d_label(p.astype(np.float32), str(i))
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()
    except (AttributeError, TypeError) as e:
        print(
            f"[!] Text labels not available in this Open3D ({e}); showing lines only")
        o3d.visualization.draw_geometries([seg, ls], window_name=title,
                                          width=1024, height=768)


# ---- main ----------------------------------------------------------------------------
def main(args):
    scan = args.scan
    in_path = scan_path(args.sample, scan)
    print(f"Input: {in_path}")
    if not in_path.exists():
        hint = " (run align.py first)" if scan == "T1_aligned" else ""
        raise SystemExit(f"Missing file: {in_path}{hint}")

    mesh = load_mesh(in_path)
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals).copy()
    n = len(points)
    print(f"Vertices: {n}")
    print(f"Stage 2 settings: edge_height={args.edge_height} edge_rel={args.edge_rel} "
          f"ring_in={args.ring_in}")

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
    h, ok, bn = annulus_height(dpts, dnrm, query_idx, args)
    ok &= valid_region
    print(f"Fits accepted: {ok.sum()} of {len(query_idx)} "
          f"({100 * ok.sum() / max(len(query_idx), 1):.0f} %)")

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

    clusters = []
    for c in sorted(set(lab_seed) - {-1}):
        mem = np.where(lab_seed == c)[0]
        area = float(wp_area[mem].sum())
        if area >= SEED_MIN_AREA:
            clusters.append((area, mem))
    clusters.sort(key=lambda t: -t[0])
    strong = [t for t in clusters if t[0] >= args.min_seed_area]
    weak = [t for t in clusters if t[0] < args.min_seed_area]
    print(f"\nSeed clusters: {len(clusters)} "
          f"({len(strong)} with seed area >= {args.min_seed_area}, {len(weak)} smaller)")

    # 6. Stage 2. Strong clusters first (largest seed area first; this order only decides
    #    which of two overlapping candidates survives). Final ids are assigned afterwards.
    ctx = (points, normals, cKDTree(points), areas, mesh_edges)
    taken = np.zeros(n, dtype=bool)
    kept = []

    for seed_area, mem in strong:
        tag, c, reasons = evaluate_cluster(seed_area, mem, dpts, bn, h, ctx,
                                           kept, taken, args)
        if reasons:
            report_skip(tag, c, reasons)
            continue
        taken[c["idx"]] = True
        kept.append(c)

    # Smaller clusters only fill free slots, best (cleanest) ring first
    slots = max(args.expected_count - len(kept), 0)
    if weak and slots == 0:
        print(f"  {len(weak)} smaller seed clusters ignored "
              f"(no free slot, expected count {args.expected_count})")
    elif weak:
        print(f"  {slots} free slot(s): checking {len(weak)} smaller seed clusters "
              f"(ring_rms <= {QC_MAX_RING_RMS} required)")
        pool = []
        for seed_area, mem in weak:
            tag, c, reasons = evaluate_cluster(seed_area, mem, dpts, bn, h, ctx,
                                               kept, taken, args)
            if c is not None and c["rms"] > QC_MAX_RING_RMS:
                reasons.append(f"ring_rms {c['rms']:.3f} > {QC_MAX_RING_RMS}")
            if reasons:
                report_skip(tag, c, reasons)
            else:
                pool.append(c)
        pool.sort(key=lambda c: c["rms"])
        for c in pool:
            reasons = gate_reasons(c, kept, taken, args) if len(kept) < args.expected_count \
                else ["no free slot"]
            if reasons:
                report_skip(c["tag"], c, reasons)
                continue
            taken[c["idx"]] = True
            kept.append(c)

    # 7. Number along the arch
    labels_out = np.full(n, -1, dtype=np.int32)
    height_out = np.full(n, np.nan, dtype=np.float32)
    rows = []
    order = (arch_order(np.array([k["cen"] for k in kept]), occ, args.reverse)
             if kept else [])
    print("\nAttachments in arch order (#0 = first):")
    for new_id, j in enumerate(order):
        k = kept[j]
        labels_out[k["idx"]] = new_id
        height_out[k["idx"]] = k["h"]
        rows.append(dict(id=new_id, cx=k["cen"][0], cy=k["cen"][1], cz=k["cen"][2],
                         area_mm2=k["area"], h95_mm=k["h95"], h_top_mm=k["h_top"],
                         n_vertices=len(k["idx"]), ring_rms_mm=k["rms"],
                         nx=k["n_ref"][0], ny=k["n_ref"][1], nz=k["n_ref"][2],
                         qc_flag=k["qc"]))
        print(f"  [KEEP #{new_id}{' QC=check' if k['qc'] == 'check' else ''}] "
              f"{k['tag']} | {describe(k)}")
    attach = labels_out >= 0
    n_check = sum(r_["qc_flag"] == "check" for r_ in rows)
    print(f"\nAttachments kept: {len(rows)} (expected {args.expected_count}), "
          f"flagged 'check': {n_check}")
    if args.expected_count and len(rows) != args.expected_count:
        print(f"[!] Count differs from {args.expected_count}: a missed or false detection "
              f"shifts the ids after it. Check the [skip] lines and the viewer.")

    # 8. Save
    out_dir = PROCESSED_DIR / args.sample
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = np.tile(GRAY, (n, 1))
    colors[attach] = RED
    seg = colored(mesh, colors)
    o3d.io.write_triangle_mesh(str(out_dir / f"{scan}_segmented.ply"), seg)

    np.save(out_dir / f"{scan}_attachment_labels.npy", labels_out)
    np.save(out_dir / f"{scan}_attachment_height.npy", height_out)
    csv_path = out_dir / f"{scan}_attachments.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    elif csv_path.exists():
        # no stale CSV next to all-background label/height files
        csv_path.unlink()
        print(f"[!] No attachments kept: removed stale {csv_path.name}")

    heat_mesh = None
    if args.debug or args.show == "heat":
        heat_mesh = height_map_mesh(mesh, h, ok, dist4, idx4)
    if args.debug:
        o3d.io.write_triangle_mesh(
            str(out_dir / f"{scan}_height_map.ply"), heat_mesh)
        dbg = np.tile(GRAY, (n, 1))
        near = dist4[:, 0] <= 1.5 * args.voxel
        dbg[ok[nn] & near] = GREEN
        dbg[(lab_seed[nn] >= 0) & near] = ORANGE
        dbg[attach] = RED
        o3d.io.write_triangle_mesh(str(out_dir / f"{scan}_detection_debug.ply"),
                                   colored(mesh, dbg))
    print(f"Saved to {out_dir}")

    # 9. View
    if args.show_ids and rows:
        show_ids(seg, rows)
    elif args.show != "none":
        seg_view = args.show == "seg"
        o3d.visualization.draw_geometries(
            [seg if seg_view else heat_mesh],
            window_name=("attachments (red)" if seg_view
                         else "h map (red = above base, +-0.4 mm)"),
            width=1024, height=768)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Detect attachments on one scan automatically.")
    ap.add_argument("--sample", required=True,
                    help="Sample folder name, e.g. 2")
    ap.add_argument("--scan", choices=SCANS, default="T1",
                    help="which scan to process (T1_aligned needs align.py first)")
    ap.add_argument("--voxel", type=float, default=VOXEL)
    ap.add_argument("--ann-in", type=float, default=ANN_IN)
    ap.add_argument("--ann-out", type=float, default=ANN_OUT)
    ap.add_argument("--ann-normal-dot", type=float, default=ANN_NORMAL_DOT)
    ap.add_argument("--ann-min-points", type=int, default=ANN_MIN_POINTS)
    ap.add_argument("--ann-min-sectors", type=int, default=ANN_MIN_SECTORS)
    ap.add_argument("--max-rms", type=float, default=ANN_MAX_RMS)
    ap.add_argument("--seed-height", type=float, default=SEED_HEIGHT)
    ap.add_argument("--min-seed-area", type=float, default=MIN_SEED_AREA,
                    help="seed clusters at least this big (mm^2) are accepted on their own; "
                         "smaller ones only fill free slots up to --expected-count")
    ap.add_argument("--expected-count", type=int, default=EXPECTED_COUNT,
                    help="attachments per scan; 0 = never accept smaller seed clusters")
    ap.add_argument("--min-separation", type=float, default=MIN_SEPARATION,
                    help="min. distance between two attachments (mm)")
    ap.add_argument("--buccal-dot", type=float, default=BUCCAL_DOT,
                    help="set to -1 to disable the buccal filter")
    ap.add_argument("--eps", type=float, default=DBSCAN_EPS)
    ap.add_argument("--min-area", type=float, default=MIN_AREA)
    ap.add_argument("--max-area", type=float, default=MAX_AREA)
    ap.add_argument("--edge-height", type=float, default=EDGE_HEIGHT,
                    help="region grows to vertices at least this far above the base (mm)")
    ap.add_argument("--edge-rel", type=float, default=EDGE_REL,
                    help="cleanup: also at least this fraction of h95 (0 disables)")
    ap.add_argument("--ring-in", type=float, default=RING_IN,
                    help="inner radius of the base-surface ring, distance from the region (mm)")
    ap.add_argument("--flip-normals", action="store_true",
                    help="invert the automatic normal orientation decision")
    ap.add_argument("--reverse", action="store_true",
                    help="number the attachments from the other end of the arch")
    ap.add_argument("--debug", action="store_true",
                    help="also write <scan>_height_map.ply and <scan>_detection_debug.ply")
    ap.add_argument("--show", choices=["seg", "heat", "none"], default="seg")
    ap.add_argument("--show-ids", action="store_true",
                    help="open the viewer with every id drawn on a leader line "
                         "(overrides --show)")
    main(ap.parse_args())
