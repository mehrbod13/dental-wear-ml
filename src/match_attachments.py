"""
Pair T1/T2 attachments by position, number them in arch order, write
data/processed/<sample>/volumes_matched.csv

Usage:
    python src/match_attachments.py --samples 1 2 3
    python src/match_attachments.py --samples 2 --reverse     # tooth 1 at the other end
"""
import argparse
import csv

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from detect_attachments import PROCESSED_DIR
from measure_volume import measure_scan


def apply(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def read_att(sample, scan):
    f = PROCESSED_DIR / sample / f"{scan}_attachments.csv"
    if not f.exists():
        return []
    with open(f, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [dict(id=int(r["id"]), qc=r["qc_flag"],
                 xyz=np.array([float(r["cx"]), float(r["cy"]), float(r["cz"])]))
            for r in rows]


def arch_order(pts):
    """Sort points around the arch. Start after the largest angular gap (the arch opening)."""
    X = pts - pts.mean(axis=0)
    _, vecs = np.linalg.eigh(X.T @ X)
    ang = np.arctan2(X @ vecs[:, 1], X @ vecs[:, 2])   # two largest-variance axes
    order = np.argsort(ang)
    s = ang[order]
    gaps = np.diff(np.r_[s, s[0] + 2 * np.pi])
    return np.roll(order, -((int(gaps.argmax()) + 1) % len(order)))


def match_sample(sample, args):
    out = PROCESSED_DIR / sample
    a1, a2 = read_att(sample, "T1"), read_att(sample, "T2")
    if not a1 or not a2:
        print(f"\n[sample {sample}] T1 or T2 has no attachments -> skipped")
        return
    c1 = np.array([a["xyz"] for a in a1])
    c2 = np.array([a["xyz"] for a in a2])

    tf = out / "T1_to_T2.npy"
    if tf.exists():
        M = np.load(tf)
        cands = {"as saved": M, "inverse": np.linalg.inv(M)}
        t2tree = cKDTree(c2)
        name = min(cands, key=lambda k: t2tree.query(apply(cands[k], c1))[0].mean())
        T = cands[name]
        print(f"\n[sample {sample}] T1_to_T2.npy used as: {name}")
    elif args.no_transform:
        T = np.eye(4)
    else:
        T, err = estimate_transform(c1, c2)
        if T is None:
            print(f"\n[sample {sample}] no {tf.name} and T1 has {len(a1)} / T2 has "
                  f"{len(a2)} attachments -> cannot estimate alignment. "
                  f"Make both scans find the same number first.")
            return
        print(f"\n[sample {sample}] alignment estimated from attachment positions, "
              f"fit error {err:.2f} mm")
        if err > 2.0:
            print("  [WARNING] fit error > 2 mm: pairing is doubtful")

    D = np.linalg.norm(apply(T, c1)[:, None, :] - c2[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(D)
    ents, used1, used2 = [], set(), set()
    for i, j in zip(ri, ci):
        if D[i, j] <= args.max_dist:
            ents.append(dict(i1=i, i2=j, xyz=c1[i], dist=D[i, j]))
            used1.add(i)
            used2.add(j)
    c2_t1 = apply(np.linalg.inv(T), c2)
    ents += [dict(i1=i, i2=None, xyz=c1[i], dist=None) for i in range(len(a1)) if i not in used1]
    ents += [dict(i1=None, i2=j, xyz=c2_t1[j], dist=None) for j in range(len(a2)) if j not in used2]

    n_pair = len(used1)
    md = np.mean([e["dist"] for e in ents if e["dist"] is not None]) if n_pair else float("nan")
    print(f"  T1: {len(a1)}  T2: {len(a2)}  paired: {n_pair}  mean pair distance {md:.2f} mm  "
          f"total teeth: {len(ents)}")
    if md > 2.0:
        print("  [WARNING] mean pair distance > 2 mm: alignment or pairing is doubtful")
    if len(ents) != args.teeth and not args.force:
        print(f"  [WARNING] {len(ents)} teeth instead of {args.teeth}: numbering would be "
              f"shifted -> NOT written. Fix detection (see [skip] lines) or use --force.")
        return

    order = arch_order(np.array([e["xyz"] for e in ents]))
    if args.reverse:
        order = order[::-1]

    v1 = measure_scan(sample, "T1", "pre") or {}
    v2 = measure_scan(sample, "T2", "post") or {}
    cols = ["tooth", "pre_id", "post_id", "pre_volume_mm3", "post_volume_mm3",
            "pair_dist_mm", "qc_pre", "qc_post", "x_T1", "y_T1", "z_T1"]
    with open(out / "volumes_matched.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for t, k in enumerate(order, start=1):
            e = ents[k]
            p1 = a1[e["i1"]] if e["i1"] is not None else None
            p2 = a2[e["i2"]] if e["i2"] is not None else None
            pre = v1.get(p1["id"]) if p1 else None
            post = v2.get(p2["id"]) if p2 else None
            wr.writerow(dict(
                tooth=t, pre_id="" if not p1 else p1["id"], post_id="" if not p2 else p2["id"],
                pre_volume_mm3="" if pre is None else f"{pre:.4f}",
                post_volume_mm3="" if post is None else f"{post:.4f}",
                pair_dist_mm="" if e["dist"] is None else f"{e['dist']:.2f}",
                qc_pre="" if not p1 else p1["qc"], qc_post="" if not p2 else p2["qc"],
                x_T1=f"{e['xyz'][0]:.1f}", y_T1=f"{e['xyz'][1]:.1f}", z_T1=f"{e['xyz'][2]:.1f}"))
    print(f"  Saved {out / 'volumes_matched.csv'}")


def kabsch(P, Q):
    """Rigid 4x4 transform T with apply(T, P) ~= Q (rows are points)."""
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = qc - R @ pc
    return T


def estimate_transform(c1, c2):
    """T1 -> T2 transform from attachment centroids. Needs len(c1) == len(c2)."""
    if len(c1) != len(c2):
        return None, None
    n = len(c1)
    o1, o2 = arch_order(c1), arch_order(c2)
    best_err, best_T = np.inf, None
    for seq2 in (o2, o2[::-1]):
        for s in range(n):
            perm = np.roll(seq2, s)
            T = kabsch(c1[o1], c2[perm])
            err = np.linalg.norm(apply(T, c1[o1]) - c2[perm], axis=1).mean()
            if err < best_err:
                best_err, best_T = err, T
    for _ in range(3):                                  # refine with nearest pairing
        D = np.linalg.norm(apply(best_T, c1)[:, None] - c2[None], axis=2)
        ri, ci = linear_sum_assignment(D)
        best_T = kabsch(c1[ri], c2[ci])
    D = np.linalg.norm(apply(best_T, c1)[:, None] - c2[None], axis=2)
    ri, ci = linear_sum_assignment(D)
    return best_T, float(D[ri, ci].mean())



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--teeth", type=int, default=14)
    ap.add_argument("--max-dist", type=float, default=4.0,
                    help="max T1-T2 centroid distance for a pair (mm)")
    ap.add_argument("--reverse", action="store_true", help="number teeth from the other end")
    ap.add_argument("--no-transform", action="store_true",
                    help="T1 and T2 are already in the same frame")
    ap.add_argument("--force", action="store_true", help="write even if teeth != 14")
    args = ap.parse_args()
    for s in args.samples:
        match_sample(s, args)