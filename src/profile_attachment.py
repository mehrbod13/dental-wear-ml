"""
Phase 1 - Step 2a (diagnostic): look at the real shape of attachments.

You pick the centre of an attachment (Shift + Left Click). For every picked point
the script builds a local frame and saves a figure with:
  1. vertical cross-section   (along the occlusal axis)
  2. horizontal cross-section (along the arch)
  3. height map seen from the buccal side (outermost surface, contours every 0.25 mm)
Heights are measured from the picked point, so the tooth surface next to the
attachment shows up as a negative number whose size is the attachment height.

Input : data/<sample>/T1.stl  (raw; alignment is irrelevant here)
Output: data/processed/<sample>/profile_<k>.png

Usage:
    python src/profile_attachment.py --sample 2
    python src/profile_attachment.py --sample 2 --view-radius 6
"""

from scipy.spatial import cKDTree
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# All lengths in millimetres
VIEW_RADIUS = 8.0      # half-size of the window around the picked point
SLAB_HALF = 0.12       # half-thickness of the cross-section slab
NORMAL_RADIUS = 2.5    # neighbourhood used to estimate the local normal
GRID = 0.15            # cell size of the height map


def pick_points(mesh):
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals      # needed for shaded rendering
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick the CENTRE of attachments "
                      "(Shift+Left Click, Q to finish)", width=1024, height=768)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0
    print("\nShift + Left Click : add a point (centre of an attachment)")
    print("Shift + Right Click: remove the last point")
    print("Q / Esc            : finish")
    vis.run()
    idx = vis.get_picked_points()
    vis.destroy_window()
    return np.asarray(mesh.vertices)[idx]


def occlusal_axis(points):
    c = points - points.mean(axis=0)
    _, vecs = np.linalg.eigh(c.T @ c / len(points))
    return vecs[:, 0]      # smallest-variance axis


def local_frame(p, points, tree, occ, normal_radius):
    """Return outward normal n and in-surface axes e_h (arch), e_v (occlusal)."""
    idx = tree.query_ball_point(p, normal_radius)
    nb = points[idx] - points[idx].mean(axis=0)
    _, vecs = np.linalg.eigh(nb.T @ nb)
    n = vecs[:, 0]

    # outward = away from the arch centre, perpendicular to the occlusal axis
    mid = np.median(points, axis=0)
    radial = p - mid
    radial -= (radial @ occ) * occ
    radial /= np.linalg.norm(radial) + 1e-12
    if n @ radial < 0:
        n = -n
    tilt = np.degrees(np.arccos(np.clip(n @ radial, -1, 1)))

    e_v = occ - (occ @ n) * n
    e_v /= np.linalg.norm(e_v)
    # assumption: +y points toward gingiva/palate (Q7, unverified)
    if e_v[1] < 0:
        e_v = -e_v
    e_h = np.cross(n, e_v)
    return n, e_h, e_v, tilt, len(idx)


def make_figure(p, k, sample, rel_a, rel_b, rel_c, R, slab, grid, out_path, tilt):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    # 1. vertical cross-section: slab around a = 0, plot (b, c)
    m = np.abs(rel_a) < slab
    ax = axes[0]
    ax.scatter(rel_b[m], rel_c[m], s=2, c="k")
    ax.set_title(f"Vertical section ({m.sum()} points)\n"
                 "x: along occlusal axis (+ = toward +y), y: height along n")
    ax.set_xlabel("mm")
    ax.set_ylabel("height from picked point, mm")

    # 2. horizontal cross-section: slab around b = 0, plot (a, c)
    m = np.abs(rel_b) < slab
    ax = axes[1]
    ax.scatter(rel_a[m], rel_c[m], s=2, c="k")
    ax.set_title(f"Horizontal section ({m.sum()} points)\n"
                 "x: along the arch, y: height along n")
    ax.set_xlabel("mm")

    for ax in axes[:2]:
        ax.set_xlim(-R, R)
        ax.set_ylim(-4.0, 2.0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(-R, R + 0.1, 1.0))
        ax.set_yticks(np.arange(-4.0, 2.1, 0.5))
        ax.grid(alpha=0.35)
        ax.axhline(0, color="r", lw=0.8)

    # 3. height map: outermost surface per cell
    nbin = int(2 * R / grid)
    inside = (np.abs(rel_a) < R) & (np.abs(rel_b) < R)
    ia = np.floor((rel_a[inside] + R) / grid).astype(int).clip(0, nbin - 1)
    ib = np.floor((rel_b[inside] + R) / grid).astype(int).clip(0, nbin - 1)
    H = np.full((nbin, nbin), -np.inf)
    np.maximum.at(H, (ib, ia), rel_c[inside])
    H[np.isinf(H)] = np.nan
    Hm = np.ma.masked_invalid(H)
    ax = axes[2]
    im = ax.imshow(Hm, origin="lower", extent=[-R, R, -R, R], cmap="turbo",
                   vmin=-2.5, vmax=0.5)
    xs = np.linspace(-R + grid / 2, R - grid / 2, nbin)
    ax.contour(xs, xs, Hm, levels=np.arange(-2.5, 0.51, 0.25),
               colors="k", linewidths=0.4)
    ax.set_title("Height map from the buccal side (contours every 0.25 mm)\n"
                 "x: along the arch, y: along occlusal axis")
    ax.set_xlabel("mm")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="height from picked point, mm")

    fig.suptitle(f"sample {sample}, pick #{k}, point {np.round(p, 2)}, "
                 f"normal tilt vs radial {tilt:.0f} deg")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(args):
    path = DATA_DIR / args.sample / "T1.stl"
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    points = np.asarray(mesh.vertices)
    print(f"Loaded {path} ({len(points)} vertices)")

    picks = pick_points(mesh)
    if len(picks) == 0:
        print("Nothing picked.")
        return

    tree = cKDTree(points)
    occ = occlusal_axis(points)
    out_dir = PROCESSED_DIR / args.sample
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, p in enumerate(picks, 1):
        n, e_h, e_v, tilt, n_nb = local_frame(
            p, points, tree, occ, args.normal_radius)
        idx = tree.query_ball_point(p, args.view_radius * 1.6)
        rel = points[idx] - p
        out = out_dir / f"profile_{k}.png"
        make_figure(p, k, args.sample, rel @ e_h, rel @ e_v, rel @ n,
                    args.view_radius, args.slab, args.grid, out, tilt)
        print(f"pick {k}: point {np.round(p, 2)}, normal from {n_nb} vertices, "
              f"tilt vs radial {tilt:.0f} deg -> {out}")
        if tilt > 60:
            print("   [!] Normal is far from the outward direction. "
                  "Pick closer to the attachment centre or change --normal-radius.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Cross-sections and height map around picked attachments.")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--view-radius", type=float, default=VIEW_RADIUS)
    ap.add_argument("--slab", type=float, default=SLAB_HALF,
                    help="half-thickness of cross-section slab (use 0.2 on coarse meshes)")
    ap.add_argument("--normal-radius", type=float, default=NORMAL_RADIUS)
    ap.add_argument("--grid", type=float, default=GRID)
    main(ap.parse_args())
