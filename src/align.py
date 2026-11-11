"""
Phase 1 — Step 1: Align T1 (source) to T2 (target) using point-to-plane ICP.

Pipeline:
1. Load T1 and T2 meshes (STL).
2. Manually pick >= 3 corresponding points on both meshes (regions without attachments).
3. Estimate an initial transformation from the picked point pairs.
4. Run point-to-plane ICP to refine the alignment.
5. Report fitness and inlier_rmse.
6. Save the 4x4 transformation matrix to data/processed/<sample_id>/T1_to_T2.npy.
7. Save the aligned T1 mesh to data/processed/<sample_id>/T1_aligned.stl.

Usage:
    python align.py --sample 1
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def find_project_root() -> Path:
    """Walk upward from the script location until a directory containing `data/` is found."""
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "data").is_dir():
            return candidate
    return here  # fallback


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Load an STL mesh and compute vertex normals."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices():
        raise ValueError(f"Mesh is empty or could not be loaded: {path}")
    mesh.compute_vertex_normals()
    return mesh


def save_transformation(matrix: np.ndarray, out_path: Path) -> None:
    """Save a 4x4 transformation matrix as a .npy file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, matrix)
    print(f"[✓] Transformation matrix saved to: {out_path}")


def save_mesh(mesh: o3d.geometry.TriangleMesh, out_path: Path) -> None:
    """Save a mesh as STL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print(f"[✓] Aligned mesh saved to: {out_path}")


# --------------------------------------------------------------------------- #
# Manual point picking
# --------------------------------------------------------------------------- #
def pick_points(mesh: o3d.geometry.TriangleMesh, window_title: str) -> np.ndarray:
    """
    Open an interactive window to pick points on the mesh.

    Controls:
        Shift + Left Click : add a point
        Shift + Right Click: remove the last point
        Q or Esc           : close the window and return

    Returns:
        (N, 3) numpy array of picked vertex coordinates.
    """
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name=window_title, width=1024, height=768)
    vis.add_geometry(mesh)

    print(f"\n[{window_title}]")
    print("  Shift + Left Click  : add a point")
    print("  Shift + Right Click : remove the last point")
    print("  Q / Esc             : finish picking")
    print("  → Pick at least 3 points on regions WITHOUT attachments.\n")

    vis.run()

    picked = vis.get_picked_points()
    vis.destroy_window()

    if len(picked) == 0:
        raise RuntimeError(f"No points were picked for: {window_title}")

    indices = np.asarray([p.index for p in picked], dtype=int)
    points = np.asarray(mesh.vertices)[indices]
    print(f"[✓] {len(points)} point(s) picked for {window_title}")
    return points


def estimate_initial_transformation(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """
    Estimate a rigid transformation (4x4) that maps source_points → target_points.

    Uses Open3D's point-to-point ICP with the picked points as a tiny point cloud.
    A large correspondence distance is used because the points are already paired.
    """
    if len(source_points) != len(target_points):
        raise ValueError("Number of source and target points must match.")
    if len(source_points) < 3:
        raise ValueError(
            "At least 3 point pairs are required for a stable initial guess.")

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    # Distance threshold large enough to accept all picked pairs as correspondences.
    threshold = 1e6  # effectively "accept everything"

    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1),
    )

    print("[✓] Initial transformation estimated from picked point pairs.")
    return result.transformation


# --------------------------------------------------------------------------- #
# ICP refinement
# --------------------------------------------------------------------------- #
def mesh_to_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int = 50_000,
) -> o3d.geometry.PointCloud:
    """Densely sample a mesh to obtain a point cloud for ICP."""
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.5, max_nn=30)
    )
    return pcd


def run_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    trans_init: np.ndarray,
    max_correspondence_distance: float = 0.1,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Run point-to-plane ICP.

    Args:
        source_pcd: Point cloud of T1 (source).
        target_pcd: Point cloud of T2 (target).
        trans_init: Initial 4x4 transformation.
        max_correspondence_distance: Distance threshold for correspondences.
            Should be tuned to the scale of the model (e.g. 0.05–0.2 for dental scans).

    Returns:
        Open3D RegistrationResult with `.transformation`, `.fitness`, `.inlier_rmse`.
    """
    print("\n[→] Running point-to-plane ICP ...")
    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_correspondence_distance,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=100,
        ),
    )
    return result


def report_quality(result: o3d.pipelines.registration.RegistrationResult) -> None:
    """Print fitness and inlier_rmse with a simple interpretation."""
    fitness = result.fitness
    rmse = result.inlier_rmse

    print("\n" + "=" * 50)
    print("ICP quality report")
    print("=" * 50)
    print(
        f"  fitness      : {fitness:.4f}   (higher is better, 1.0 = perfect overlap)")
    print(f"  inlier_rmse  : {rmse:.6f}     (lower is better, in model units)")

    if fitness < 0.5:
        print(
            "  [⚠] fitness is low (< 0.5). Check the initial point picks or increase")
        print("      max_correspondence_distance.")
    elif rmse > 0.05:
        print("  [⚠] inlier_rmse is relatively high. The alignment may be loose.")
    else:
        print("  [✓] Alignment looks reasonable.")
    print("=" * 50 + "\n")


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def visualize_alignment(
    source_mesh: o3d.geometry.TriangleMesh,
    target_mesh: o3d.geometry.TriangleMesh,
    transformation: np.ndarray,
) -> None:
    """Show T1 (orange, transformed) and T2 (blue, original) together."""
    source_temp = copy.deepcopy(source_mesh)
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([1.0, 0.706, 0.0])  # orange
    target_temp = copy.deepcopy(target_mesh)
    target_temp.paint_uniform_color([0.0, 0.651, 0.929])  # blue

    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        window_name="Alignment result — T1 (orange) vs T2 (blue)",
        width=1024,
        height=768,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(sample_id: str) -> None:
    sample_dir = DATA_DIR / sample_id
    t1_path = sample_dir / "T1.stl"
    t2_path = sample_dir / "T2.stl"

    if not t1_path.exists():
        raise FileNotFoundError(f"T1 mesh not found: {t1_path}")
    if not t2_path.exists():
        raise FileNotFoundError(f"T2 mesh not found: {t2_path}")

    print(f"\n=== Aligning sample {sample_id} ===")

    # 1. Load meshes
    print("\n[1/5] Loading meshes ...")
    t1_mesh = load_mesh(t1_path)
    t2_mesh = load_mesh(t2_path)
    print(
        f"  T1: {len(t1_mesh.vertices)} vertices, {len(t1_mesh.triangles)} triangles")
    print(
        f"  T2: {len(t2_mesh.vertices)} vertices, {len(t2_mesh.triangles)} triangles")

    # 2. Manual point picking (regions without attachments)
    print("\n[2/5] Manual point picking ...")
    t1_points = pick_points(t1_mesh, "T1 — pick points WITHOUT attachments")
    t2_points = pick_points(
        t2_mesh, "T2 — pick the SAME points WITHOUT attachments")

    if len(t1_points) != len(t2_points):
        raise RuntimeError(
            f"Number of picked points differs: T1={len(t1_points)}, T2={len(t2_points)}. "
            "Please re-run and pick the same number of points on both meshes."
        )

    # 3. Initial transformation from picked points
    print("\n[3/5] Estimating initial transformation ...")
    trans_init = estimate_initial_transformation(t1_points, t2_points)

    # 4. Point-to-plane ICP refinement
    print("\n[4/5] Refining alignment with point-to-plane ICP ...")
    t1_pcd = mesh_to_point_cloud(t1_mesh)
    t2_pcd = mesh_to_point_cloud(t2_mesh)

    # NOTE: tune this threshold to the scale of your scans.
    # 0.1 is a reasonable starting point for dental STL models in mm.
    icp_result = run_icp(t1_pcd, t2_pcd, trans_init,
                         max_correspondence_distance=0.1)
    report_quality(icp_result)

    # 5. Save outputs
    print("\n[5/5] Saving outputs ...")
    out_dir = PROCESSED_DIR / sample_id
    save_transformation(icp_result.transformation, out_dir / "T1_to_T2.npy")

    t1_aligned = copy.deepcopy(t1_mesh)
    t1_aligned.transform(icp_result.transformation)
    save_mesh(t1_aligned, out_dir / "T1_aligned.stl")

    # 6. Visual check
    print("\n[✓] Done. Showing alignment preview (close the window to exit) ...")
    visualize_alignment(t1_mesh, t2_mesh, icp_result.transformation)

    print(f"\n=== Sample {sample_id} aligned successfully ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 — Step 1: Align T1 to T2 using point-to-plane ICP."
    )
    parser.add_argument(
        "--sample",
        type=str,
        required=True,
        help="Sample folder name (e.g. 1, 2, 8, 14).",
    )
    args = parser.parse_args()
    main(args.sample)
