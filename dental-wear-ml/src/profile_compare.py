"""
Phase 1 - Step 2b (diagnostic): overlay aligned T1 on T2 around each attachment.

Both scans are expressed in ONE local frame built from T2 only, and heights are
NOT re-referenced to the picked point, so a change of height is a real change.
Output figure per pick:
  1. vertical section   (T1 blue, T2 red)
  2. horizontal section (T1 blue, T2 red)
  3. height difference map  D = T2 - T1   (negative = T2 is lower = wear)
The printed numbers give the mean D on the attachment top and in a control ring.

Input : data/<sample>/T2.stl
        data/processed/<sample>/T1_aligned.stl, T1_to_T2.npy, picks_<tag>.npy
Output: data/processed/<sample>/compare_<k>.png

Usage:
    python src/profile_attachment.py --sample 1 --file T1.stl     # saves picks_T1.npy
    python src/profile_compare.py --sample 1
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from profile_attachment import (DATA_DIR, PROCESSED_DIR,  # noqa: E402
                                occlusal_axis, local_frame)

VIEW_RADIUS = 6.0
SLAB_HALF = 0.12
GRID = 0.15
NORMAL_RADIUS = 2.5
TOP_RADIUS = 1.0           # "attachment top" = cells closer than this to the pick
RING = (2.5, 4.0)          # control ring (tooth surface around the attachment)


def load_vertices(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.remove_duplicated_vertices()
    return np.asarray(mesh.vertices)


def height_grid(a, b, c, R, grid):
    """Outermost height per cell (NaN where there is no vertex)."""
    nbin = int(2 * R / grid)
    m = (np.abs(a) < R) & (np.abs(b) < R)
    ia = np.floor((a[m] + R) / grid).astype(int).clip(0, nbin - 1)
    ib = np.floor((b[m] + R) / grid).astype(int).clip(0, nbin - 1)
    H = np.full((nbin, nbin), -np.inf)
    np.maximum.at(H, (ib, ia), c[m])
    H[np.isinf(H)] = np.nan
    return H


def main(args):
    sdir = PROCESSED_DIR / args.sample
    p1 = load_vertices(sdir / "T1_aligned.stl")
    p2 = load_vertices(DATA_DIR / args.sample / "T2.stl")
    t1, t2 = cKDTree(p1), cKDTree(p2)
    occ = occlusal_axis(p2)

    picks = np.load(sdir / f"picks_{args.picks}.npy")
    T = np.load(sdir / "T1_to_T2.npy")
    picks = picks @ T[:3, :3].T + T[:3, 3]      # raw T1 frame -> T2 frame
    dist, j = t2.query(picks)
    picks = p2[j]                               # snap to a T2 vertex
    R, grid = args.view_radius, args.grid
    xs = np.linspace(-R + grid / 2, R - grid / 2, int(2 * R / grid))
    XX, YY = np.meshgrid(xs, xs)
    rr = np.hypot(XX, YY)

    for k, (p, d) in enumerate(zip(picks, dist), 1):
        n, e_h, e_v, tilt, _ = local_frame(p, p2, t2, occ, args.normal_radius)
        rel = []
        for pts, tree in ((p1, t1), (p2, t2)):
            idx = tree.query_ball_point(p, R * 1.6)
            q = pts[idx] - p
            rel.append((q @ e_h, q @ e_v, q @ n))
        (a1, b1, c1), (a2, b2, c2) = rel

        H1 = height_grid(a1, b1, c1, R, grid)
        H2 = height_grid(a2, b2, c2, R, grid)
        D = H2 - H1
        ok = np.isfinite(D)
        top = ok & (rr < TOP_RADIUS)
        ring = ok & (rr > RING[0]) & (rr < RING[1])
        print(f"pick {k}: snap {d:.2f} mm | top: mean D = {np.nanmean(D[top]):+.3f} mm "
              f"({top.sum()} cells) | ring: median D = {np.nanmedian(D[ring]):+.3f} mm "
              f"({ring.sum()} cells)")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
        for ax, (u1, u2, s1, s2, name) in zip(
                axes[:2],
                [(b1, b2, np.abs(a1) < args.slab, np.abs(a2) < args.slab, "Vertical"),
                 (a1, a2, np.abs(b1) < args.slab, np.abs(b2) < args.slab, "Horizontal")]):
            ax.scatter(u1[s1], c1[s1], s=3, c="tab:blue", label="T1 aligned")
            ax.scatter(u2[s2], c2[s2], s=3, c="tab:red", label="T2", alpha=0.7)
            ax.set_title(f"{name} section (heights NOT re-referenced)")
            ax.set_xlim(-R, R)
            ax.set_ylim(-3.0, 1.0)
            ax.set_aspect("equal")
            ax.grid(alpha=0.35)
            ax.legend(loc="lower right")
        im = axes[2].imshow(np.ma.masked_invalid(D), origin="lower",
                            extent=[-R, R, -R, R], cmap="coolwarm",
                            vmin=-0.3, vmax=0.3)
        axes[2].set_title("D = T2 - T1, mm  (blue = T2 lower = wear)")
        fig.colorbar(im, ax=axes[2], shrink=0.8)
        fig.suptitle(f"sample {args.sample}, pick #{k}, tilt {tilt:.0f} deg")
        fig.tight_layout()
        out = sdir / f"compare_{k}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Overlay aligned T1 on T2 per attachment.")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--picks", default="T1",
                    help="tag of picks_<tag>.npy (raw T1 frame)")
    ap.add_argument("--view-radius", type=float, default=VIEW_RADIUS)
    ap.add_argument("--slab", type=float, default=SLAB_HALF)
    ap.add_argument("--grid", type=float, default=GRID)
    ap.add_argument("--normal-radius", type=float, default=NORMAL_RADIUS)
    main(ap.parse_args())
