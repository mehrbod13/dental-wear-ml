"""
Phase 1 - Step 2: Detect attachments on aligned T1 (curvature + DBSCAN, seeded).

Steps:
  1. Pick one point on the center of each attachment.
  2. Compute protrusion (distance from a Taubin-smoothed surface) per vertex.
  3. Keep high-protrusion vertices as candidates.
  4. Cluster candidates with DBSCAN.
  5. Keep only clusters that contain a vertex near one of the seeds.

Input : data/processed/<sample>/T1_aligned.stl
Output: data/processed/<sample>/T1_segmented.ply  (attachments red, rest gray)

Usage:
    python src/detect_attachments.py --sample 1
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SMOOTH_ITERATIONS = 50       # Taubin iterations for the base surface
CURVATURE_PERCENTILE = 90    # vertices above this protrusion percentile are candidates
DBSCAN_EPS_FACTOR = 0.01     # DBSCAN eps as a fraction of the model diagonal
DBSCAN_MIN_SAMPLES = 15      # minimum neighbours for a DBSCAN core point
# cluster must touch this radius around a seed (fraction of diagonal)
SEED_RADIUS_FACTOR = 0.05
MIN_CLUSTER_POINTS = 100
MAX_CLUSTER_POINTS = 30000


def diagonal(mesh):
    return float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))


def pick_seeds(mesh):
    """Pick one point on each attachment center. Returns vertex indices."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    while True:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Pick attachment centers",
                          width=1024, height=768)
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 2.0

        print("\n[Pick attachment centers]")
        print("  Shift + Left Click  : add a seed (one per attachment)")
        print("  Shift + Right Click : remove the last seed")
        print("  Q / Esc             : finish")

        vis.run()
        indices = vis.get_picked_points()
        vis.destroy_window()

        if indices:
            print(f"[+] {len(indices)} seed(s) picked.")
            return np.asarray(indices, dtype=int)
        print("[!] No seeds picked. Starting over.")


def protrusion(mesh):
    """Distance of each vertex from a Taubin-smoothed copy of the surface."""
    smooth = o3d.geometry.TriangleMesh(mesh).filter_smooth_taubin(
        number_of_iterations=SMOOTH_ITERATIONS
    )
    return np.linalg.norm(np.asarray(mesh.vertices) - np.asarray(smooth.vertices), axis=1)


def main(sample_id):
    in_path = PROCESSED_DIR / sample_id / "T1_aligned.stl"
    if not in_path.exists():
        raise FileNotFoundError(f"Run align.py first. Missing: {in_path}")

    mesh = o3d.io.read_triangle_mesh(str(in_path))
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()

    V = np.asarray(mesh.vertices)
    scale = diagonal(mesh)
    print(f"Vertices: {len(V)}, diagonal: {scale:.4f}")

    # 1. Seeds
    seeds = pick_seeds(mesh)

    # 2. Protrusion and candidates
    disp = protrusion(mesh)
    threshold = np.percentile(disp, CURVATURE_PERCENTILE)
    candidate_idx = np.where(disp >= threshold)[0]
    print(
        f"Candidates above {CURVATURE_PERCENTILE}th percentile: {len(candidate_idx)}")

    # 3. DBSCAN on candidate positions
    labels = DBSCAN(
        eps=DBSCAN_EPS_FACTOR * scale,
        min_samples=DBSCAN_MIN_SAMPLES,
    ).fit_predict(V[candidate_idx])

    # 4. Keep clusters that touch a seed
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(V)
    tree = o3d.geometry.KDTreeFlann(pcd)
    radius = SEED_RADIUS_FACTOR * scale

    seed_near = np.zeros(len(V), dtype=bool)
    for s in seeds:
        _, idx, _ = tree.search_radius_vector_3d(V[s], radius)
        seed_near[np.asarray(idx, dtype=int)] = True

    attach = np.zeros(len(V), dtype=bool)
    print("\nClusters:")
    for label in sorted(set(labels) - {-1}):
        members = candidate_idx[labels == label]
        size = len(members)
        touches_seed = seed_near[members].any()
        ok = touches_seed and MIN_CLUSTER_POINTS <= size <= MAX_CLUSTER_POINTS
        status = "KEEP" if ok else "skip"
        centroid = np.round(V[members].mean(axis=0), 2)
        print(
            f"  [{status}] cluster {label}: {size} vertices, seed nearby: {touches_seed}, centroid {centroid}")
        if ok:
            attach[members] = True

    # Seeds themselves are always part of their attachment
    attach[seeds] = True
    print(
        f"\nAttachment vertices: {attach.sum()} ({attach.sum() / len(V) * 100:.2f}% of mesh)")

    # 5. Color and save
    colors = np.tile([0.7, 0.7, 0.7], (len(V), 1))
    colors[attach] = [1.0, 0.0, 0.0]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    out_dir = PROCESSED_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "T1_segmented.ply"
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print(f"Saved to {out_path}")

    o3d.visualization.draw_geometries(
        [mesh], window_name="T1 segmented (red = attachments)", width=1024, height=768
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect attachments on aligned T1.")
    parser.add_argument("--sample", required=True,
                        help="Sample folder name, e.g. 1")
    args = parser.parse_args()
    main(args.sample)
