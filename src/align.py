"""
Align T1 to T2 with robust point-to-plane ICP, starting from picked points.

Usage:
    python src/align.py --sample 1            # pick points, run ICP, save
    python src/align.py --sample 1 --check    # only report the saved alignment

Output: data/processed/<sample>/T1_to_T2.npy,
        data/processed/<sample>/T1_aligned.stl
"""

import argparse

import numpy as np
import open3d as o3d

from config import PROCESSED_DIR, DATA_DIR
from detect_attachments import load_mesh, GRAY

MIN_PAIRS = 4
SAMPLE_POINTS = 100_000
NORMAL_RADIUS = 0.75  # mm
# ICP stages: (max correspondence distance, Tukey cut-off), both in mm
STAGES = [(3.0, 1.0), (1.0, 0.3), (0.3, 0.1)]
PICK_RMSE_WARN = 1.0  # mm
FITNESS_WARN = 0.5
OVERLAP_MM = 1.0  # T1 vertices farther than this from T2 are not shared geometry
# Picking view: camera on the +z side looking toward -z, +y up (buccal view, gingiva on top).
# Same frame for all active cases; check it when a new case is added.
VIEW_FRONT = (0.0, 0.0, 1.0)
VIEW_UP = (0.0, 1.0, 0.0)
PICK_POINT_SIZE = 3.0


def pick_points(mesh, title):
    """
    Pick mesh vertices. Returns an (N, 3) array.
    Open3D picks points only from a PointCloud, so the cloud is made of the mesh vertices
    (picked index = vertex index).

    Controls:
        Shift + Left Click  : add a point
        Shift + Right Click : remove the last point
        Q / Esc             : finish picking
    """
    vertices = np.asarray(mesh.vertices)
    front = np.asarray(VIEW_FRONT, dtype=float)

    # normals only for shading: turn them toward the camera (STL orientation is inconsistent)
    normals = np.asarray(mesh.vertex_normals)
    normals = normals * np.where(normals @ front < 0, -1.0, 1.0)[:, None]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.paint_uniform_color(GRAY)
    center = vertices.mean(axis=0)

    while True:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=title, width=1024, height=768)
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = PICK_POINT_SIZE
        view = vis.get_view_control()
        view.set_lookat(center)
        view.set_front(list(VIEW_FRONT))
        view.set_up(list(VIEW_UP))
        view.set_zoom(0.6)

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

        print(f"[!] Only {len(indices)} point(s) picked, need at least "
              f"{MIN_PAIRS}. Starting over.")


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


def sample_cloud(mesh, with_normals):
    """Uniform surface samples; normals (unoriented) only where needed."""
    pcd = mesh.sample_points_uniformly(number_of_points=SAMPLE_POINTS)
    if with_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=NORMAL_RADIUS, max_nn=30))
    return pcd


def surface_distance(points, mesh):
    """Exact distance (mm) from each point to the triangle surface of mesh."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    return scene.compute_distance(query).numpy()


def report_quality(t1, t2, T):
    """Distance of every T1 vertex to the T2 surface, shared geometry only."""
    moved = np.asarray(t1.vertices) @ T[:3, :3].T + T[:3, 3]
    dist = surface_distance(moved, t2)
    shared = dist[dist < OVERLAP_MM]
    print("\n" + "=" * 50)
    print(f"T1 vertices within {OVERLAP_MM} mm of T2 : "
          f"{len(shared) / len(dist) * 100:.1f} %")
    print(f"distance median / p90 : {np.median(shared):.3f} / "
          f"{np.percentile(shared, 90):.3f} mm")
    print(f"share below 0.05 / 0.10 mm : "
          f"{np.mean(shared < 0.05) * 100:.1f} / "
          f"{np.mean(shared < 0.10) * 100:.1f} %")
    print("=" * 50)


def main(sample_id, check_only):
    sample_dir = DATA_DIR / sample_id
    out_dir = PROCESSED_DIR / sample_id
    t1 = load_mesh(sample_dir / "T1.stl")
    t2 = load_mesh(sample_dir / "T2.stl")

    if check_only:
        report_quality(t1, t2, np.load(out_dir / "T1_to_T2.npy"))
        return

    # 1. Pick corresponding points, in the same order on both meshes
    p1 = pick_points(t1, "T1 - pick points WITHOUT attachments")
    p2 = pick_points(t2, "T2 - pick the SAME points in the SAME order")
    if len(p1) != len(p2):
        raise RuntimeError(
            f"Point counts differ: T1={len(p1)}, T2={len(p2)}. Run again.")

    # 2. Initial alignment from picked pairs
    T = kabsch(p1, p2)
    residual = np.linalg.norm((p1 @ T[:3, :3].T + T[:3, 3]) - p2, axis=1)
    pick_rmse = float(np.sqrt(np.mean(residual ** 2)))
    print(f"\nInitial alignment RMSE on picks: {pick_rmse:.3f} mm")
    if pick_rmse > PICK_RMSE_WARN:
        print("[!] Picks disagree more than expected. "
              "The same physical points may not match.")

    # 3. Robust point-to-plane ICP, coarse to fine
    src = sample_cloud(t1, with_normals=False)
    dst = sample_cloud(t2, with_normals=True)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100)

    for i, (max_dist, k) in enumerate(STAGES, 1):
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k))
        result = o3d.pipelines.registration.registration_icp(
            src, dst, max_dist, T, estimation, criteria)
        T = result.transformation
        print(f"[{i}/{len(STAGES)}] max dist {max_dist} mm, Tukey {k} mm: "
              f"fitness {result.fitness:.4f}, inlier_rmse {result.inlier_rmse:.4f} mm")
        if result.fitness < FITNESS_WARN:
            print("[!] Low fitness. The alignment may be wrong. Re-check the picks.")

    # 4. Quality
    report_quality(t1, t2, T)

    # 5. Save
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "T1_to_T2.npy", T)

    t1_aligned = o3d.geometry.TriangleMesh(t1)
    t1_aligned.transform(T)
    t1_aligned.compute_triangle_normals()
    o3d.io.write_triangle_mesh(str(out_dir / "T1_aligned.stl"), t1_aligned)
    print(f"Saved to {out_dir}")

    # 6. Preview: T1 orange, T2 blue
    preview_t1 = o3d.geometry.TriangleMesh(t1_aligned)
    preview_t2 = o3d.geometry.TriangleMesh(t2)
    preview_t1.compute_vertex_normals()
    preview_t2.compute_vertex_normals()
    preview_t1.paint_uniform_color([1.0, 0.706, 0.0])
    preview_t2.paint_uniform_color([0.0, 0.651, 0.929])
    o3d.visualization.draw_geometries(
        [preview_t1, preview_t2], window_name="T1 (orange) vs T2 (blue)",
        width=1024, height=768)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align T1 to T2 with robust point-to-plane ICP.")
    parser.add_argument("--sample", required=True,
                        help="Sample folder name, e.g. 1")
    parser.add_argument("--check", action="store_true",
                        help="Only report the quality of the saved alignment")
    args = parser.parse_args()
    main(args.sample, args.check)
