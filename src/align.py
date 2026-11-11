"""
Phase 1 — Step 1: Align T1 to T2 using point-to-plane ICP.

Usage:
    python src/align.py --sample 1
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def pick_points(mesh, title):
    """Pick points on mesh vertices. Shift+Left click to add, Shift+Right click to remove, Q to finish."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1024, height=768)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2.0

    print(f"\n[{title}]")
    print("  Shift + Left Click  : add a point")
    print("  Shift + Right Click : remove the last point")
    print("  Q / Esc             : finish picking")
    print("  (Click the window first to focus it, then hold Shift and click)")

    vis.run()
    picked_indices = vis.get_picked_points()  # list of int indices into pcd.points
    vis.destroy_window()

    if picked_indices:
        all_points = np.asarray(pcd.points)
        return all_points[picked_indices].astype(float)

    # Fallback: type coordinates manually if nothing was picked in the window.
    print("[!] No points picked. Enter coordinates as 'x y z', one per line, empty line to finish:")
    rows = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        rows.append([float(v) for v in line.replace(",", " ").split()])
    if not rows:
        raise RuntimeError(f"No points provided for {title}")
    return np.asarray(rows, dtype=float)


def kabsch(src, dst):
    """Rigid 4x4 transform mapping src[i] -> dst[i] (least squares, no scale)."""
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = dc - R @ sc
    return T


def main(sample_id):
    sample_dir = DATA_DIR / sample_id
    t1 = o3d.io.read_triangle_mesh(str(sample_dir / "T1.stl"))
    t2 = o3d.io.read_triangle_mesh(str(sample_dir / "T2.stl"))
    if not t1.has_vertices() or not t2.has_vertices():
        raise ValueError("Could not load T1 or T2 mesh.")
    t1.compute_vertex_normals()
    t2.compute_vertex_normals()

    # 1. Pick corresponding points
    p1 = pick_points(t1, "T1 - pick points WITHOUT attachments")
    p2 = pick_points(t2, "T2 - pick the SAME points WITHOUT attachments")
    if len(p1) != len(p2) or len(p1) < 3:
        raise RuntimeError(
            f"Need the same number of points (>=3) on both meshes. Got {len(p1)} and {len(p2)}.")

    # 2. Initial guess from picked pairs
    T_init = kabsch(p1, p2)

    # 3. Point-to-plane ICP
    scale = np.linalg.norm(t2.get_axis_aligned_bounding_box().get_extent())
    src = t1.sample_points_uniformly(number_of_points=50_000)
    dst = t2.sample_points_uniformly(number_of_points=50_000)
    dst.estimate_normals()
    src.estimate_normals()

    result = o3d.pipelines.registration.registration_icp(
        src, dst,
        max_correspondence_distance=0.1 * scale,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100),
    )

    print(f"\nfitness     : {result.fitness:.4f}")
    print(
        f"inlier_rmse : {result.inlier_rmse:.6f}  ({result.inlier_rmse / scale * 100:.3f}% of diagonal)")

    # 4. Save
    out_dir = PROCESSED_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "T1_to_T2.npy", result.transformation)

    t1_aligned = o3d.geometry.TriangleMesh(t1)
    t1_aligned.transform(result.transformation)
    o3d.io.write_triangle_mesh(str(out_dir / "T1_aligned.stl"), t1_aligned)
    print(f"Saved to {out_dir}")

    # 5. Preview
    t1_aligned.paint_uniform_color([1.0, 0.706, 0.0])
    t2.paint_uniform_color([0.0, 0.651, 0.929])
    o3d.visualization.draw_geometries(
        [t1_aligned, t2], window_name="T1 (orange) vs T2 (blue)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align T1 to T2 with point-to-plane ICP.")
    parser.add_argument("--sample", required=True,
                        help="Sample folder name, e.g. 1")
    args = parser.parse_args()
    main(args.sample)
