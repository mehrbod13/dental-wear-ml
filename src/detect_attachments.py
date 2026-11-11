"""
Phase 1 - Step 2: Detect attachments on aligned T1 using curvature + DBSCAN.

Input : data/processed/<sample>/T1_aligned.stl
Output: data/processed/<sample>/T1_segmented.ply
        (attachment vertices in red, the rest of the mesh in gray)

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

K_NEIGHBORS = 30            # neighbourhood size for curvature estimation
# vertices above this surface-variation percentile are candidates
CURVATURE_PERCENTILE = 92
DBSCAN_EPS_FACTOR = 0.01    # DBSCAN eps as a fraction of the model diagonal
DBSCAN_MIN_SAMPLES = 20     # minimum neighbours for a DBSCAN core point
MIN_CLUSTER_POINTS = 150    # clusters smaller than this are discarded
MAX_CLUSTER_POINTS = 20000  # clusters larger than this are discarded


def diagonal(mesh):
    return float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))


def surface_variation(points):
    """
    Per-vertex surface variation: smallest eigenvalue of the local covariance
    divided by the sum of eigenvalues. Near 0 on flat surfaces, higher on edges
    and protrusions.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(pcd)

    sv = np.zeros(len(points))
    for i, p in enumerate(points):
        _, idx, _ = tree.search_knn_vector_3d(p, K_NEIGHBORS)
        nb = points[idx] - points[idx].mean(axis=0)
        eig = np.linalg.eigvalsh(nb.T @ nb)  # ascending order
        total = eig.sum()
        sv[i] = eig[0] / total if total > 0 else 0.0
    return sv


def cluster_candidates(points, candidate_idx, scale):
    """Cluster candidate vertices with DBSCAN and keep clusters of plausible size."""
    labels = DBSCAN(
        eps=DBSCAN_EPS_FACTOR * scale,
        min_samples=DBSCAN_MIN_SAMPLES,
    ).fit_predict(points[candidate_idx])

    kept = []
    print("\nClusters found:")
    for label in sorted(set(labels) - {-1}):
        members = candidate_idx[labels == label]
        size = len(members)
        centroid = points[members].mean(axis=0)
        ok = MIN_CLUSTER_POINTS <= size <= MAX_CLUSTER_POINTS
        status = "KEEP" if ok else "skip"
        print(
            f"  [{status}] cluster {label}: {size} vertices, centroid {np.round(centroid, 2)}")
        if ok:
            kept.append(members)

    if not kept:
        return np.array([], dtype=int)
    return np.concatenate(kept)


def show_result(mesh):
    o3d.visualization.draw_geometries(
        [mesh], window_name="T1 segmented (red = attachments)", width=1024, height=768
    )


def main(sample_id):
    in_path = PROCESSED_DIR / sample_id / "T1_aligned.stl"
    if not in_path.exists():
        raise FileNotFoundError(f"Run align.py first. Missing: {in_path}")

    mesh = o3d.io.read_triangle_mesh(str(in_path))
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()

    points = np.asarray(mesh.vertices)
    scale = diagonal(mesh)
    print(f"Vertices: {len(points)}, diagonal: {scale:.4f}")

    # 1. Curvature
    print("\n[1/3] Estimating surface variation ...")
    sv = surface_variation(points)

    # 2. Candidates
    threshold = np.percentile(sv, CURVATURE_PERCENTILE)
    candidate_idx = np.where(sv >= threshold)[0]
    print(
        f"[2/3] Candidates above {CURVATURE_PERCENTILE}th percentile: {len(candidate_idx)}")

    # 3. Cluster
    print("[3/3] Clustering with DBSCAN ...")
    attach_idx = cluster_candidates(points, candidate_idx, scale)
    print(
        f"\nAttachment vertices: {len(attach_idx)} ({len(attach_idx) / len(points) * 100:.2f}% of mesh)")

    # Color and save
    mask = np.zeros(len(points), dtype=bool)
    mask[attach_idx] = True
    colors = np.tile([0.7, 0.7, 0.7], (len(points), 1))
    colors[mask] = [1.0, 0.0, 0.0]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    out_dir = PROCESSED_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "T1_segmented.ply"
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print(f"Saved to {out_path}")

    show_result(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect attachments on aligned T1.")
    parser.add_argument("--sample", required=True,
                        help="Sample folder name, e.g. 1")
    args = parser.parse_args()
    main(args.sample)
