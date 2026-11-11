"""
Phase 1 - Step 1: Align T1 to T2 with point-to-plane ICP.

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

MIN_PAIRS = 4
SAMPLE_POINTS = 100_000
# coarse ICP max correspondence distance (x model diagonal)
COARSE_FACTOR = 0.05
# fine ICP max correspondence distance (x model diagonal)
FINE_FACTOR = 0.02
PICK_RMSE_WARN = 0.02  # warn if picked-point RMSE exceeds this (x diagonal)
FITNESS_WARN = 0.3     # warn if final fitness is below this


def diagonal(mesh):
    return float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices():
        raise ValueError(f"Could not load mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def show_picks(mesh, points, title):
    """Show the mesh in gray with picked points as red spheres, for visual check."""
    radius = max(diagonal(mesh) * 0.012, 1e-6)
    mesh_gray = o3d.geometry.TriangleMesh(mesh)
    mesh_gray.paint_uniform_color([0.7, 0.7, 0.7])
    geometries = [mesh_gray]
    for p in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(p)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        sphere.compute_vertex_normals()
        geometries.append(sphere)
    o3d.visualization.draw_geometries(
        geometries, window_name=f"{title} - check picks", width=1024, height=768)


def pick_points(mesh, title):
    """
    Pick points on mesh vertices. Returns an (N, 3) array.

    Controls:
        Shift + Left Click  : add a point
        Shift + Right Click : remove the last point
        Q / Esc             : finish picking
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    vertices = np.asarray(pcd.points)

    while True:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=title, width=1024, height=768)
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 2.0

        print(f"\n[{title}]")
        print("  Shift + Left Click  : add a point")
        print("  Shift + Right Click : remove the last point")
        print("  Q / Esc             : finish picking")
        print(f"  Pick at least {MIN_PAIRS} points, spread across the arch.")

        vis.run()
        indices = vis.get_picked_points()
        vis.destroy_window()

        if len(indices) >= MIN_PAIRS:
            points = vertices[indices].astype(float)
            print(f"[+] {len(points)} points picked:")
            for i, p in enumerate(points, 1):
                print(f"    {i}: {np.round(p, 3)}")
            return points

        print(
            f"[!] Only {len(indices)} point(s) picked, need at least {MIN_PAIRS}. Starting over.")


def kabsch(src, dst):
    """Rigid 4x4 transform mapping src[i] -> dst[i] (least squares, no scale)."""
    if np.linalg.matrix_rank(src - src.mean(axis=0)) < 3:
        raise ValueError(
            "Picked points are collinear or coplanar. Spread them across the arch.")

    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = dc - R @ sc
    return T


def sample_with_normals(mesh, scale):
    """
    Sample points from the mesh and compute normals from the local point
    neighbourhood, then orient them consistently and outward from the centre.
    Mesh triangle winding in STL files is not reliable, so it is not used.
    """
    pcd = mesh.sample_points_uniformly(number_of_points=SAMPLE_POINTS)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01 * scale, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    center = pts.mean(axis=0)
    if np.mean(np.sum((pts - center) * nrm, axis=1)) < 0:
        pcd.normals = o3d.utility.Vector3dVector(-nrm)
    return pcd


def main(sample_id):
    sample_dir = DATA_DIR / sample_id
    t1 = load_mesh(sample_dir / "T1.stl")
    t2 = load_mesh(sample_dir / "T2.stl")
    scale = max(diagonal(t1), diagonal(t2))
    print(f"Model diagonal (scale): {scale:.4f}")

    # 1. Pick corresponding points, in the same order on both meshes
    p1 = pick_points(t1, "T1 - pick points WITHOUT attachments")
    p2 = pick_points(t2, "T2 - pick the SAME points in the SAME order")
    if len(p1) != len(p2):
        raise RuntimeError(
            f"Point counts differ: T1={len(p1)}, T2={len(p2)}. Run again.")

    # 2. Initial alignment from picked pairs
    T_init = kabsch(p1, p2)
    residual = np.linalg.norm(
        (p1 @ T_init[:3, :3].T + T_init[:3, 3]) - p2, axis=1)
    pick_rmse = float(np.sqrt(np.mean(residual ** 2)))
    print(
        f"\nInitial alignment RMSE on picks: {pick_rmse:.4f} ({pick_rmse / scale * 100:.2f}% of diagonal)")
    if pick_rmse / scale > PICK_RMSE_WARN:
        print(
            "[!] Picks disagree more than expected. The same physical points may not match.")

    # 3. Point-to-plane ICP in two stages: coarse, then fine
    src = sample_with_normals(t1, scale)
    dst = sample_with_normals(t2, scale)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
    )
    p2plane = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    print("\n[1/2] Coarse ICP ...")
    coarse = o3d.pipelines.registration.registration_icp(
        src, dst, COARSE_FACTOR * scale, T_init, p2plane, criteria
    )

    print("[2/2] Fine ICP ...")
    fine = o3d.pipelines.registration.registration_icp(
        src, dst, FINE_FACTOR * scale, coarse.transformation, p2plane, criteria
    )

    fitness = fine.fitness
    rmse = fine.inlier_rmse
    print("\n" + "=" * 50)
    print(
        f"fitness     : {fitness:.4f}   (fraction of points within {FINE_FACTOR} x diagonal)")
    print(f"inlier_rmse : {rmse:.6f}  ({rmse / scale * 100:.3f}% of diagonal)")
    if fitness < FITNESS_WARN:
        print("[!] Low fitness. The alignment may be wrong. Re-check the picks.")
    print("=" * 50)

    # 4. Save
    out_dir = PROCESSED_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "T1_to_T2.npy", fine.transformation)

    t1_aligned = o3d.geometry.TriangleMesh(t1)
    t1_aligned.transform(fine.transformation)
    o3d.io.write_triangle_mesh(str(out_dir / "T1_aligned.stl"), t1_aligned)
    print(f"Saved to {out_dir}")

    # 5. Preview: T1 orange, T2 blue
    preview_t1 = o3d.geometry.TriangleMesh(t1_aligned)
    preview_t2 = o3d.geometry.TriangleMesh(t2)
    preview_t1.paint_uniform_color([1.0, 0.706, 0.0])
    preview_t2.paint_uniform_color([0.0, 0.651, 0.929])
    o3d.visualization.draw_geometries(
        [preview_t1, preview_t2], window_name="T1 (orange) vs T2 (blue)", width=1024, height=768
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align T1 to T2 with point-to-plane ICP.")
    parser.add_argument("--sample", required=True,
                        help="Sample folder name, e.g. 1")
    args = parser.parse_args()
    main(args.sample)
