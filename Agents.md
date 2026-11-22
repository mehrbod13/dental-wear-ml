# AGENTS.md — dental-wear-ml

Guide for AI coding agents and human contributors. **Read it fully before changing code.**
It records what the project is, what the data looks like, what has been measured, what failed and why, what is implemented, and what to do next.

Status: **Phase 1 in progress.** Detection and volume measurement exist and were run on **case 2 T1 only**. Pairing exists but was **never run**. Wear % is **not implemented**. Last update: after the supervisor's Excel (group 2) became available, after detection / `measure_volume.py` / `match_attachments.py` arrived, and after paths moved to `config.py`.

## TL;DR for contributors 

- **Task:** measure wear (%) of orthodontic *attachments* (small composite bumps glued on teeth), not tooth wear, from two 3D scans of one upper jaw: T1 (before, "pre") and T2 (after, "post").
- **Expected wear (corrected):** the old claim "at most ~15 %, rare" is **contradicted by the supervisor's Excel** (group 2): wear 0.4–48 %, average ~22 %, nine of 14 attachments above 15 % (section 1). Do not design around "wear is small".
- **Data:** `data/<id>/T1.stl, T2.stl`, active cases 1, 2, 4, 5, 6, 8, 10–14 (3 excluded; 7 and 9 do not exist). Millimetres. Agents usually do **not** have the STL files; a maintainer runs the code and sends back terminal output and screenshots.
- **State:**
  - `detect_attachments.py` (ring-based, automatic) works on case 2 T1 (14 attachments). Not validated on other cases or on T2.
  - `measure_volume.py` gives volumes from saved heights. Case 2 T1 run: mean 5.86 mm³ vs Excel ~7.0, so volumes are probably biased low (section 3).
  - `match_attachments.py` (pairs T1 and T2 detections, numbers teeth) is written by a contributor, **not run**.
  - Wear %, noise baseline, tables: **not implemented**. Alignment (`align.py`) was run for cases 1, 2 only and its output is currently deleted.
- **Biggest facts:** (1) one scan of 6/11 cases has 4x fewer vertices; (2) global alignment error ~0.1 mm is not negligible; (3) attachment height is ~1–1.8 mm and a 4 mm-radius neighbourhood is not a height field; (4) measure each scan against its own thin-ring base surface; (5) the foot of an attachment is cut off by the detector, so volumes are biased low and wear % is biased high (section 3, Q12).
- **Next step:** section 7 ("Immediate next steps").
- **Before asking anything,** check section 11 (FAQ).

Legend: **[measured]** = seen in data/plots/terminal output. **[assumed]** = hypothesis, not verified. **[failed]** = tried and did not work. Untagged = a decision or a fact from the maintainers.

---

## 0. Working rules

- **Language:** the team works in Persian. Talk to Persian-speaking members in Persian. Code, comments, docstrings, docs and commit messages are in English. Study/exam material is not part of this repo.
- **Be concise.** Short, direct, low-commentary answers; no over-complicated solutions when a simpler documented one exists. Reply format that works: short text, numbers in mm, at most one table, then exactly what to run and what to send back; say what you saw, what you are unsure about, ask at most one question.
- **Never modify `data/<id>/T1.stl` or `T2.stl`.** Raw inputs. Derived files go to `data/processed/<id>/`.
- **Do not paste or print full meshes.** Each STL is 10–65 MB (100k–650k vertices). Use summary statistics.
- **Everything is in millimetres.** Every threshold, radius and tolerance is in mm, never in vertex counts or percentiles of vertices (mesh density differs ~4x between files).
- **Ask before running GUI scripts** (`align.py`, `profile_attachment.py`, any visualisation): they open Open3D windows and need a human.
- **Do not guess; look at the data first.** Three detection approaches written from assumptions failed (section 5). Before a new algorithm, produce a plot or number from real data that supports its assumptions.
- **Say when code is untested.** Agents have no STL access. State which code was not run on real data, and give the exact command and what output to send back.
- **Script conventions:** one script per step in `src/`; `argparse` with `--sample <id>`; constants at the top; paths come from `src/config.py` (`DATA_DIR`, `PROCESSED_DIR`; older scripts may still define `PROJECT_ROOT = Path(__file__).resolve().parent.parent`); results in `data/processed/<id>/`.
- **No version numbers in scripts.** Scripts, docstrings, argparse descriptions and file names carry no version labels; history lives in git and in section 5.
- **Give full files, not patches,** when the maintainer will paste code (whole files are pasted). Ask for the current file before patching; do not patch blind. Say explicitly which files a command overwrites.
- **Be honest about uncertainty.** A wrong wear number that looks valid is worse than none. Report quality metrics and flag suspicious cases instead of forcing a result.
- **When a maintainer says the output is wrong,** assume they are right and look for the false assumption (section 5) instead of tuning thresholds.
- **One method everywhere.** Do not use different methods in different steps that can conflict. The base surface is fitted **once, in detection**; later steps reuse its result.
- **Volume is a separate module.** `detect_attachments.py` must not compute or output any volume.
- **Do not use `trimesh` `mesh.volume`** (meshes are not watertight). Rigid ICP only, no scale.
- **Do not include case 3 in batch runs.**
- **Commit messages are short** (one subject line, optional short body).
- **Literature references are from memory and unverified;** check title and year before citing.

---

## 1. Project goal

Machine-learning dental project. **Final goal: compute the wear percentage of orthodontic attachments (not tooth wear)** by comparing two 3D scans of the same upper jaw: **T1 (before, intact)** and **T2 (after)**.

**Phase 1 (current): geometric mesh processing, no deep learning.** Tools: Python, `open3d`, `trimesh`, `scikit-learn`, `scipy`. A later phase may use deep learning (e.g. MeshSegNet, Teeth3DS) once Phase 1 gives labelled data and a baseline.

Original Phase 1 plan (`docs/Phase_1.md` / `Phase_1.png`, handwritten notes), with current status:

1. Align T1 and T2: manual picks on regions without attachments, then ICP. Done for cases 1, 2 only (output currently deleted). *Note:* the notes said "two points", but a rigid transform needs at least 3 non-collinear points; `align.py` requires 4 (`MIN_PAIRS = 4`).
2. Isolate attachments: the original "curvature + DBSCAN" idea **does not work** (section 5). Replaced by ring-based detection (section 3).
3. Wear volume: `trimesh` volume is not usable; replaced by `measure_volume.py` (section 3).
4. Tables: unify and compare with `scikit-learn`. Notes were illegible (Q1). `build_tables.py` is planned.

### Expected wear magnitude — supervisor's Excel (group 2) [measured from the sheet]

The old statement "wear at most ~15 %, rare" is **wrong for this group**. The sheet (label "پکبل سارمکو / مستطیل افقی با بول", i.e. a rectangular attachment; read from an image, translation uncertain) has columns: attachment 2.1–2.14, defect, excess, RMS, pre volume, post volume, wear loss, wear percent, SBS, ARI, failure mode. Defect/excess/RMS: meaning unknown. SBS/ARI/failure mode: not relevant. It is **[assumed]** that group 2 corresponds to case 2 (14 attachments); not confirmed.

| attachment | pre mm³ | post mm³ | wear % |
|---|---|---|---|
| 2.1 | 7.548 | 7.1065 | 5.85 |
| 2.2 | 7.3974 | 6.4658 | 12.59 |
| 2.3 | 6.4947 | 6.2038 | 4.48 |
| 2.4 | 5.5584 | 3.9237 | 29.41 |
| 2.5 | 5.2353 | 3.1452 | 39.92 |
| 2.6 | 6.9859 | 6.5704 | 5.95 |
| 2.7 | 8.2243 | 5.8233 | 29.19 |
| 2.8 | 7.9928 | 6.4647 | 19.12 |
| 2.9 | 6.6205 | 4.3051 | 34.97 |
| 2.10 (written "2.1") | 10.1573 | 5.2385 | 48.43 |
| 2.11 | 6.9146 | 3.7109 | 46.33 |
| 2.12 | 7.3965 | 5.1495 | 30.38 |
| 2.13 | 5.9218 | 5.8998 | 0.37 |
| 2.14 | 6.7856 | 5.7229 | 15.66 |

- Sheet average row: pre 7.01, post 5.42, wear loss 1.6, wear 21.86 %. Recomputed from the rows: pre 7.09, post 5.41, mean wear % 23.0 (small unexplained discrepancy; unchecked).
- **Order of rows 2.1–2.14 on the arch is unknown** (Q10). Without it, ids cannot be matched to Excel rows.
- Consequences: (a) wear of ~50 % means h95 falls from ~1.2 mm to ~0.6 mm, close to the detector gates (`SEED_HEIGHT`, `MIN_TOP_H` = 0.5), so heavily worn attachments may be **missed on T2**; (b) the noise baseline is still needed, but the dominant risk is now systematic bias (foot cut, base surface, detection differences between scans), not noise; (c) the Excel is the only ground truth so far, usable only after a one-to-one match.

### Sensitivity budget

- Attachment volume: **~7 mm³ for rectangular** (Excel; 5.2–10.2), **~3–4 mm³ for elliptical (case 1 only)**. Height ~1–1.8 mm; top area ~5–9 mm².
- Scan noise ~0.03 mm [measured on flat tops: ±0.05 mm], mesh edge 0.1–0.2 mm. Only integrated quantities (mean height over the top, or volume) are meaningful. Wear below ~5 % is probably not detectable. Case 1 shows no visible wear (section 4).

---

## 2. Data

### Layout

```
data/
  <id>/T1.stl, T2.stl            raw scans, ids: 1 2 3 4 5 6 8 10 11 12 13 14  (7 and 9 do not exist)
  processed/<id>/                derived outputs
      T1_aligned.stl, T1_to_T2.npy     align.py (cases 1, 2; currently deleted for case 2)
      <scan>_segmented.ply             detection result, coloured (scan = T1 or T2)
      <scan>_detection_debug.ply       detection debug mesh
      <scan>_height_map.ply            h-map for viewing
      <scan>_attachment_labels.npy     int32 per vertex, -1 = background
      <scan>_attachment_height.npy     float32 h above base surface for mask vertices, NaN elsewhere
      <scan>_attachments.csv           one row per attachment
      volumes.csv                      id, pre_volume_mm3, post_volume_mm3          (measure_volume.py)
      volumes_matched.csv              paired table                                  (match_attachments.py)
      picks_T1.npy / picks_T2.npy, profile_<T1|T2>_<k>.png, compare_<k>.png   diagnostic (picking scripts)
      T1_segmented.ply (case 1)        OUTDATED result of the first (failed) approach, do not trust
  mesh_report.txt                  output of src/test.py
docs/Phase_1.md, Phase_1.png       original notes
src/                               see section 3
```

Repo tree: `data/`, `docs/`, `Readme.md`, `requirements.txt`, `src/` (`config.py`, `align.py`, `detect_attachments.py`, `measure_volume.py`, `match_attachments.py`, `test.py`, `profile_attachment.py`, `profile_compare.py`). `processed/` content grows as cases are run.

- One folder = one patient/jaw. Every file is an **upper jaw with 14 teeth**, plus palate and gingiva. STL files carry **no tooth labels**.
- **Active cases: 11** (1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14).
- Processing status: case 1 has alignment + an invalid old segmentation; case 2 has run detection (T1) and volume (T1); case 2 alignment output was deleted; the rest are unprocessed. Cases 8 and 12 were never run with the current detector (predictions unconfirmed).
- Profile figures were made by hand for cases 1, 2, 4, 5, 8, 13.

### Case 3 is excluded for now

Its T1 is abnormal: bbox x from -85 to -29 (others about -34 to +34), different orientation (smallest extent on z, ~17 mm), surface area 3266 (others 4100–5300; T2 of the same case 6637), 59 mm centre offset to T2, and only ~70 % of the area and half the height (14 vs 27 mm) even after rotation. Probably a different frame plus incomplete/different data. **Skip it in all batch runs** (also remove it from example commands such as `--samples 1 2 3`). Revisit at the very end (Q5): try a global initial alignment (PCA / FPFH+RANSAC) and check that all 14 teeth and attachments exist before forcing any alignment.

### Measured scan properties (`src/test.py`, T1/T2 of all active cases)

- All files are **millimetres**. PCA extents ~64–70 x 52–66 x 20–28 mm. Case 14 is a smaller jaw (diagonal 85 mm vs 91–96 mm). Parameters must not depend on a fixed jaw size.
- Frame: x ~-34..+34, z ~-31..+31, y ~7..33. The arch lies in the x–z plane; **y is the occlusal–gingival axis, +y toward gingiva/palate** [measured, vertical sections of case 2]. Verify in each new case before hard-coding axes.
- Every mesh is a **single connected component and NOT watertight.**
- STL triangle order and normal orientation are not guaranteed consistent. Orient normals explicitly before using their sign.
- **T2 always has a larger surface area than T1** (more gingiva/palate, different crop border), e.g. case 12: 5308 vs 6697. Raw centroid differences of 1–4 mm come from the crop, not real motion. Whole-mesh ICP is biased by the trimmed borders.
- Raw T1/T2 bounding boxes are close (centre offset 0.2–3 mm), but saved rotations are **14.0° (case 1) and 9.8° (case 2)** and are real.

### Mesh density (most important data finding)

Mean edge length ~**0.10–0.12 mm** for "fine" scans (~500–650k vertices), ~**0.20–0.23 mm** for "coarse" scans (~120–165k vertices).

| case | T1 | T2 |
|---|---|---|
| 1, 5, 6, 10, 12 | fine | fine |
| 2, 8 | fine | **coarse** |
| 4, 11, 13, 14 | **coarse** | fine |

In 6 of 11 cases one scan has ~4x fewer vertices. Per-case numbers (vertices; mean edge mm; surface area; diagonal):

| case | T1 | T2 | bbox-centre dist | note |
|---|---|---|---|---|
| 1 | 503,682; 0.107; diag 93.7 | fine | – | rotation 14° |
| 2 | 473,408; 0.104; area 4590; diag 92.2 | coarse, area 6243 | – | rotation 9.8° |
| 5 | area 4697 | area 6353 | – | |
| 8 | 474,949; 0.116; 5614 | 145,477; 0.233; 6512 | 0.29 | |
| 10 | 520,825; 0.106; 5088 | 579,746; 0.109; 6048 | 2.97 | |
| 11 | 124,663; 0.214; 4639 | 576,582; 0.111; 6311 | 1.14 | T2/T1 vertices 4.6 |
| 12 | 541,907; 0.105; 5308 | 649,425; 0.107; 6697 | 0.36 | |
| 13 | 120,519; 0.212; 4438 | 530,528; 0.111; 5849 | 0.75 | |
| 14 | 133,995; 0.205; 4555 | 548,615; 0.110; 5750 | 0.55 | small jaw (diag 85) |

Risks and rules:

- Sharp edges of rectangular attachments round off on coarse meshes; this changes height and volume. A T1-vs-T2 difference can be a **resolution artefact, not wear**.
- Edge **position** on a coarse mesh is unreliable (~±0.1–0.2 mm). With perimeter 8–9 mm and area 5–6 mm², a 0.15 mm edge error is ~15–20 % area error, as large as the wear. **Independently thresholded edges in mixed-density cases (2, 8, 4, 11, 13, 14) are a known risk** (see Q11 for how the current pipeline handles it).
- Attachment **top height** is reliable even on the coarse mesh [measured, case 4].
- Do not upsample coarse meshes. Do not blindly decimate fine meshes to 0.2 mm. Detection works on a common 0.3 mm voxel grid for stage 1 and native vertices for measurement.
- Report mixed-density cases separately.

### Alignment quality of the existing global alignment [measured]

Nearest-point distance from aligned T1 to T2 (every 20th vertex), mm:

| case | mean | median | p90 | max |
|---|---|---|---|---|
| 1 | 0.051 | 0.026 | 0.076 | 4.57 |
| 2 | 0.097 | 0.050 | 0.186 | 4.25 |

Median is at noise level; max comes from regions present in one scan only. Case 2 is worse (coarse T2, or real tooth movement; cannot tell). These numbers mostly describe palate/gingiva and are optimistic (ICP minimises almost this metric); case 2 p90 0.19 mm is comparable to the wear being measured.

**Local registration error is larger and position-dependent** [measured, case 1]: posterior teeth offset ~+0.08 to +0.10 mm buccally, anterior ~0. **Global alignment alone is not accurate enough for wear measurement**; this is why each scan is measured against its own base surface.

---

## 3. Code inventory

| file | status |
|---|---|
| `src/config.py` | shared paths (`DATA_DIR`, `PROCESSED_DIR`), imported by `detect_attachments.py`; content not reviewed by the agent (assumed `DATA_DIR = <root>/data`, `PROCESSED_DIR = <root>/data/processed`) |
| `src/align.py` | step 1, manual picking + ICP, works, to be improved; output currently deleted |
| `src/detect_attachments.py` | step 2, automatic; run on case 2 T1 only |
| `src/measure_volume.py` | step 3, written, run on case 2 T1 (T2 missing) |
| `src/match_attachments.py` | step 4, written by a contributor, **not run**; see known issues |
| `src/test.py` | inspection script, working (needs `rtree`) |
| `src/profile_attachment.py` | diagnostic: picked centre -> sections + height map; works |
| `src/profile_compare.py` | diagnostic: aligned T1 vs T2 in one frame; works |
| `src/dist_heatmap.py` | proposed (distance of aligned T1 to T2, red at 0.3 mm); not confirmed to exist |
| `requirements.txt` | should contain `numpy open3d trimesh scikit-learn scipy rtree matplotlib`; check it matches |

### `src/detect_attachments.py`

CLI: `python src/detect_attachments.py --sample <id> [--scan T1|T2] [--show seg|heat|none] [other flags]`. `--scan` defaults to T1; there is no `--aligned` option (detection does not need alignment). Other flags: `--voxel`, `--ann-in`, `--ann-out`, `--ann-normal-dot`, `--ann-min-points`, `--ann-min-sectors`, `--max-rms`, `--seed-height`, `--min-seed-area`, `--min-separation`, `--buccal-dot` (-1 disables the buccal filter), `--eps`, `--min-area`, `--max-area`, `--flip-normals`. Input `DATA_DIR/<sample>/<scan>.stl` via the shared `load_mesh(path)` (open3d mesh after `remove_duplicated_vertices`, `remove_degenerate_triangles`, `compute_vertex_normals`); **every script must load meshes through this function** because the vertex order of the label/height files is the vertex order of its result. Outputs: see layout (section 2). **Overwrites** the `<scan>_*` files of that sample.

Method:
1. **Stage 1:** normals are oriented outward automatically (outermost 15 % of the arch must face away from the centre; `--flip-normals` inverts the decision). Per working point (0.3 mm voxel cloud, buccal and ≥ 1 mm from the scan border) a quadric is fitted to the annulus `ANN_IN..ANN_OUT` using only annulus points with a similar normal; the fit is rejected if points, angular sectors or residual rms are insufficient (gingival margins, grooves, crown edges become "invalid", not bumps). h = height above that fit. Seeds = valid points with h ≥ `SEED_HEIGHT`; DBSCAN on seeds; clusters under `SEED_MIN_AREA` are ignored silently, under `MIN_SEED_AREA` dropped as edge effects.
2. **Stage 2, per cluster (largest seed area first):** ring fit on native vertices (thin ring just outside the region, wall removed by a normal filter); the region grows where h ≥ `EDGE_HEIGHT` (up to `REFINE_ITERS` iterations). **The LAST ring fit is THE base surface** and defines the saved h.
3. **Cleanup (selection only, never refits):** keep vertices with `h >= max(EDGE_HEIGHT, EDGE_REL × h95)`, then the largest mesh-connected piece **by surface area**.
4. **Gates:** duplicate/overlap (> 30 % of vertices already labelled), distance to a kept attachment < `MIN_SEPARATION`, area outside `MIN_AREA..MAX_AREA`, h95 outside `MIN_TOP_H..MAX_TOP_H`. `qc_flag = "check"` if ring_rms > `QC_MAX_RING_RMS` or area > `QC_MAX_AREA`, otherwise "ok" (a warning, not a rejection).
5. `check_picks` (T1 only, diagnostic): if `picks_T1.npy` exists, reports whether the old manual picks fall within 1 mm of a detected region.

CSV columns: `id, cx, cy, cz, area_mm2, h95_mm, h_top_mm, n_vertices, ring_rms_mm, nx, ny, nz, qc_flag` (coordinates in the scan frame). No volume column.

Constants (mm unless stated): `VOXEL 0.3`, `ANN_IN 2.0`, `ANN_OUT 3.5`, `ANN_NORMAL_DOT 0.65`, `ANN_MIN_POINTS 60`, `ANN_MIN_SECTORS 5`, `ANN_MAX_RMS 0.15`, `SEED_HEIGHT 0.5`, `BORDER_MARGIN 1.0`, `BUCCAL_DOT 0.35`, `DBSCAN_EPS 0.7`, `DBSCAN_MIN_SAMPLES 5`, `SEED_MIN_AREA 0.5`, `MIN_SEED_AREA 2.3`, `MIN_SEPARATION 5.5`, `PATCH_RADIUS 6.0`, `SEED_RADIUS 0.25`, `RING_IN 0.4`, `RING_OUT 1.6`, `RING_NORMAL_DOT 0.8`, `EDGE_HEIGHT 0.25`, `EDGE_REL 0.3`, `GROW_STEP 1.0`, `MAX_EXTENT 3.0`, `REFINE_ITERS 4`, `MIN_REGION_VERTICES 20`, `MIN_AREA 2.0`, `MAX_AREA 25.0`, `MIN_TOP_H 0.5`, `MAX_TOP_H 2.5`, `QC_MAX_RING_RMS 0.10`, `QC_MAX_AREA 16.0`.

Known fragile margins (thresholds were set from few cases):
- `MIN_SEED_AREA`: case 1 true ≥ 3.35, false ≤ 1.40; case 2 true ≥ 2.68, false ≤ 2.05; case 12 true ≥ 2.48. Thin.
- `MAX_TOP_H`: true h95 values seen up to 2.23 (case 12); a wrong base gives ≥ 2.7; 2.5 leaves little room.
- `MIN_SEPARATION 5.5`: smallest true spacing seen ~7.0 mm; a false duplicate in case 12 was 5.0 mm away.
- Heavily worn T2 attachments may fall below `SEED_HEIGHT`/`MIN_TOP_H` and be missed (not observed yet).
- Case 12: whether the relative cut (`EDGE_REL`) removed the red patches on neighbours' proximal surfaces is **unconfirmed**; the effect of choosing the largest component by area is also unchecked.
- If **no** attachment is kept, `<scan>_attachments.csv` is **not written** (a stale CSV from an earlier run stays on disk), while the label and height files are still written (all background / NaN). Downstream scripts can read a stale CSV; delete it or check counts when a run finds nothing.

### `src/measure_volume.py`

Fits **no** surface. For each attachment: volume = Σ over triangles with all three vertices in the mask of (triangle area projected onto the plane normal to `n_ref`, i.e. `0.5·|cross(b−a, c−a)·n_ref|`) × (mean of the three saved h values). `n_ref` comes from CSV `nx, ny, nz`.

- Needs per scan (`SCANS = (("T1","pre"),("T2","post"))`): `data/<id>/<scan>.stl`, `<scan>_attachment_labels.npy`, `<scan>_attachment_height.npy`, `<scan>_attachments.csv`. No `--aligned` option.
- If one scan is missing: its column is empty and a yellow warning is printed. If both are missing: error, no output.
- Consistency warnings: mask vertex count vs CSV `n_vertices`; h95 from the height file vs `h95_mm` (tolerance 1e-3 mm).
- Output `volumes.csv` has only `id, pre_volume_mm3, post_volume_mm3` (maintainers dropped qc_flag, area, mean_h, n_ring, n_triangles; cx/cy/cz/ring_rms can be restored on request). Rows are joined **by id only**; ids from independent T1/T2 detections are **not** guaranteed to be the same attachment (that is the job of `match_attachments.py`).
- `measure_scan(sample, scan, label)` returns `{id: volume or None}`, or `None` if the scan is missing.
- **Known bias:** border triangles and the low foot (below the detector cut) are not counted, so volumes are low.

**Case 2 T1 result [measured]** (columns: id, qc, projected area mm², mean h, volume mm³, h95, centre x,y,z):

| id | qc | area_proj | mean_h | volume | h95 | centre (x, y, z) |
|---|---|---|---|---|---|---|
| 0 | ok | 6.32 | 0.760 | 4.80 | 1.07 | (-22.4, 12.6, 2.2) |
| 1 | ok | 6.04 | 0.760 | 4.59 | 1.08 | (-7.6, 15.6, 22.4) |
| 2 | ok | 6.83 | 1.023 | 6.98 | 1.55 | (1.2, 15.9, 23.6) |
| 3 | check | 7.96 | 1.072 | 8.53 | 1.66 | (23.1, 13.1, 0.6) |
| 4 | ok | 6.53 | 0.738 | 4.82 | 1.12 | (24.3, 14.0, -12.8) |
| 5 | check | 9.30 | 1.131 | 10.52 | 1.78 | (-19.8, 13.8, 9.4) |
| 6 | ok | 7.14 | 0.955 | 6.82 | 1.44 | (21.9, 13.5, 7.1) |
| 7 | ok | 5.95 | 0.836 | 4.98 | 1.25 | (-24.6, 12.5, -4.8) |
| 8 | ok | 5.73 | 0.796 | 4.56 | 1.22 | (9.1, 14.9, 23.0) |
| 9 | ok | 6.30 | 0.818 | 5.16 | 1.22 | (-15.4, 14.9, 17.5) |
| 10 | ok | 5.90 | 0.866 | 5.11 | 1.28 | (18.2, 14.6, 16.9) |
| 11 | ok | 5.60 | 0.837 | 4.69 | 1.23 | (25.5, 13.9, -20.5) |
| 12 | ok | 5.77 | 0.819 | 4.72 | 1.23 | (-26.6, 12.2, -13.3) |
| 13 | ok | 6.71 | 0.851 | 5.71 | 1.32 | (25.2, 13.5, -5.2) |

ring_rms: #3 0.123 and #5 0.124 (flagged); the others 0.042–0.088. Volume mm³: mean 5.86, min 4.56, max 10.52 (n = 14). The T2 warning appeared (T2 not detected yet).

Assessment against the Excel:
- Order of magnitude is right (the old surface-area volumes of 11–18 are now 4.6–10.5).
- Volumes look **low**: mean 5.86 vs ~7.0 (~17 % below), median ~5.0 vs ~7.0 (~29 % below). The missing foot may be worth ~1 mm³ [assumed, not measured].
- Ten attachments cluster at 4.6–5.7 while the Excel has only one below 5.2. Four have larger h95 (#2, #3, #5, #6): either truly larger or base surface biased; #3 and #5 are flagged, #2 and #6 are suspicious but unflagged. #5 (10.52) is slightly above the Excel max 10.16.
- Only a one-to-one match against the Excel can decide which cases are wrong (Q10).

### `src/match_attachments.py` (contributor's file, **not run, not reviewed by running**)

Docstring: pairs T1 and T2 detections and numbers teeth. Reads `T1_attachments.csv`, `T2_attachments.csv` (id, qc_flag, cx, cy, cz). A sample is skipped if either has no attachments.

- **Transform T1 → T2:** if `T1_to_T2.npy` exists, tries the matrix and its inverse and keeps the one that brings T1 centres closer to T2; with `--no-transform` uses identity; otherwise `estimate_transform` estimates it from centres alone (arch order, all cyclic shifts, both directions, Kabsch, 3 nearest-pair refinements). The centre-only estimate works only if T1 and T2 counts are equal.
- **Pairing:** Hungarian algorithm (`linear_sum_assignment`) on centre distances; accepted if distance ≤ `--max-dist` (default 4.0 mm). The rest become T1-only or T2-only rows.
- **Warnings:** mean pair distance or fit error > 2 mm. If the total tooth count ≠ `--teeth` (default 14), **nothing is written** unless `--force`.
- **Numbering:** `arch_order` sorts centres around the arch, starting after the largest angular gap; `--reverse` flips it. Direction is arbitrary (eigenvector signs).
- **Volumes:** calls `measure_scan` for T1 ("pre") and T2 ("post"); writes `volumes_matched.csv` with `tooth, pre_id, post_id, pre_volume_mm3, post_volume_mm3, pair_dist_mm, qc_pre, qc_post, x_T1, y_T1, z_T1`. **No wear %.**
- CLI: `--samples`, `--teeth`, `--max-dist`, `--reverse`, `--no-transform`, `--force`.

Known issues / risks:
- The file header says `# src/measure_scan.py` while the docstring and usage say `match_attachments.py`: make them consistent (if the file is really named `measure_scan.py`, `python src/match_attachments.py` fails). The docstring example has `--samples 1 2 3`: case 3 must not be there.
- Numbering direction is a guess until the Excel row order is known (Q10, Q13). A rough manual check: sorting the T1 volumes in arch order does not put the maximum (10.5) on Excel row 2.10 (10.16) in either direction.
- The centre-only transform is fragile: equal counts with one false and one missed detection give a wrong transform; only the 2 mm warning catches it.
- The default `--teeth 14` blocks case 12 and other cases that give 13 (use `--teeth` or `--force`).
- T2 is detected independently (own mask and base surface), unlike the T1-mask decision (Q11).
- A strongly worn attachment missed in T2 becomes a T1-only row with empty `post`.
- Terminal output is busy because `measure_scan` prints both tables.

### `src/align.py`
- Loads T1, T2 (`open3d`), asks for **≥ 4 corresponding points** (Shift+Click, same order, non-attachment areas) in `VisualizerWithEditing`, shows a check view ("pick points WITHOUT attachments").
- `kabsch()` gives the initial rigid transform (rejects collinear/coplanar picks).
- Samples 100k points per mesh, normals (radius 0.01 × diagonal) oriented outward, **two-stage point-to-plane ICP** (max correspondence 0.05 × diagonal, then 0.02 × diagonal, ~4.8 and ~1.9 mm).
- Warns if pick RMSE > 0.02 × diagonal or fitness < 0.3. Saves `T1_to_T2.npy` and `T1_aligned.stl`; orange (T1) vs blue (T2) preview. T2 is the fixed reference.
- Weaknesses: manual picking (not reproducible); ICP over the whole mesh including gingiva/palate and crop borders (no trimming); distances scaled by diagonal; 1.9 mm fine threshold too loose for sub-0.1 mm goals; no tooth-level alignment; quality metrics printed but not saved.
- It sets `pcd.normals = mesh.vertex_normals`, which is why its view is shaded; a bare `PointCloud` without normals renders flat in Open3D.
- Currently the detection/volume/pairing chain does not need alignment (only `match_attachments.py` optionally reads `T1_to_T2.npy`).

### `src/profile_attachment.py` (diagnostic)
- `python src/profile_attachment.py --sample <id> --file T1.stl` (or `T2.stl`); `--file` sets the mesh and output prefix (`profile_T1_<k>.png`, `profile_T2_<k>.png`); picks saved to `picks_<tag>.npy`. `--load-picks T1 --transform` places the same picks on the other scan. `--slab` (slice thickness, e.g. 0.2 for coarse) and `--grid` (height-map cell; 0.3 coarse, 0.15 fine).
- Click the centre of an attachment from the buccal side. Builds a local frame (outward normal, occlusal-gingival axis, arch axis); saves vertical section, horizontal section, height map (contours every 0.25 mm). Heights are relative to the picked point.
- Output names overwrite earlier files with the same prefix; copy the folder first.
- Observation only; picking does **not** belong in the final pipeline.

### `src/profile_compare.py` (diagnostic)
- `python src/profile_compare.py --sample <id>`. Needs `T1_aligned.stl`, `T1_to_T2.npy`, `picks_T1.npy`. One local frame from T2; T1 (blue) and T2 (red) in the same axes plus `D = T2 − T1` (negative = wear). Prints per pick `snap`, `top: mean D` (r < 1 mm) and `ring: median D` (local registration error).
- Case 1 was run (~43 s); case 5 has no alignment yet.

### `src/test.py`
Prints size, unit guess, PCA extents, density, watertightness, T1/T2 consistency, saved-alignment angle/translation, distance after alignment. Writes `data/mesh_report.txt`. Requires `rtree`. Usage: `python src/test.py [--samples 1 2 8]`.

---

## 4. What the data says about the attachments [measured]

Two shapes occur, sometimes in different cases:

- **Elliptical:** smooth dome, no plateau, no steep wall. Cases 1 and 5.
- **Rectangular (rounded corners):** flat top, near-vertical walls. Cases 2 and 8 (Excel group 2 is rectangular); flat tops also in case 4 (coarse).

The palate/gingiva is the large flat region at high y; teeth hang toward low y. Attachments are on the buccal side, about one per tooth (at most ~14), centres of neighbouring teeth ~7 mm apart. Do not assume the shape from the case id.

### Geometry (±0.2 mm, from sections and contour maps)

| quantity | rectangular (cases 8, 2) | elliptical (case 5; case 1 similar) |
|---|---|---|
| height above tooth surface | ~1–1.3 mm by hand; detector h95 in case 2: 1.07–1.78 mm | ~1.1–1.5 mm |
| top size | ~2.5–2.7 mm along arch × 2 mm occlusal-gingival (case 2: ~3 × 2) | dome ~2 × 3.5 mm at half height; width ~2.4 mm at −1 mm |
| top area | ~5–8 mm² (case 2 projected area 5.6–9.3) | – |
| volume | ~7 mm³ (Excel 5.2–10.2) | ~3–4 mm³ (case 1) |
| top flatness (along arch) | ±0.05 mm | not flat |
| walls | ~1 mm drop within 0.2–0.3 mm, near vertical, both arch sides and gingival side | gentle: ~1 mm drop over ~1–1.5 mm |
| occlusal side | **no wall**; top ramps up gently over ~2 mm (wedge/bevel) | – |

- **Top is tilted:** in the vertical section the top rises ~**0.5–0.7 mm** from occlusal to gingival side, nearly the same for picks with local normals tilted 9°, 37°, 57°, so it is real design (bevel), not a frame error [probable]. Case 4 shows less bevel but a coarse mesh cannot tell design from sampling loss.
- **Do not define the frame normal from the attachment top** (it differs ~15° from the tooth surface). Heights are measured relative to a **tooth base surface**.
- Size/orientation differ between cases (case 4 looked ~1.5–2 mm along arch and ~3 mm occlusal-gingival). Use size **ranges** (~1.5–4 mm).

### Surroundings (what limits the base surface)

- Gingival margin ~3.5–5 mm on the +y side; incisal/occlusal edge ~3–4 mm on the other side. Interdental groove 2.5–3.5 mm deep at ~±4 mm along the arch. Some attachments (case 10) show an undercut at the gingival margin (curve folds back at ~5 mm).
- **Clean tooth surface exists only in a narrow strip:** ~0.3–1.5 mm outside the wall base (~2 mm toward occlusal/gingival, 0.5–1 mm along the arch). Along the arch the surface falls at 35–45° toward the groove for 3–3.5 mm. A 4 mm-radius neighbourhood contains several surface layers and is not a height graph.

### T1 vs T2 results so far (hand-picked profiles, before the automatic pipeline)

**Case 1** (4 elliptical attachments, both fine), heights in mm:

| pick | position | normal tilt | ring (registration error) | top | top − ring |
|---|---|---|---|---|---|
| 1 | posterior (x=+27, z=-13) | 40° | +0.102 | +0.144 | +0.042 |
| 2 | anterior (x=11, z=23) | 18° | -0.005 | +0.024 | +0.029 |
| 3 | anterior (x=-16, z=23) | 18° | -0.013 | +0.030 | +0.043 |
| 4 | posterior (x=-25, z=-14) | 41° | +0.077 | +0.114 | +0.037 |

- No visible wear: sections overlap within ~0.05 mm. Wear above ~0.05 mm (~4 % of height) is not present in this case.
- Posterior teeth shifted ~+0.08..+0.10 mm buccally on both sides, anterior ~0: arch widening (real movement) or a small scale difference; cannot tell (Q2, Q9). Either way a global fit leaves ~100 µm local error.
- `top − ring` is stable (spread ~0.007 mm), so referencing to the neighbouring surface cancels registration error. A constant bias of ~**+0.04 mm** remains (not real growth; probably ring slopes/grooves or sampling differences); it must be measured on a control patch and subtracted.
- Wall points in the difference map are noisy (±0.3 mm); use only the top (r < 1 mm).

**Case 5** (4 elliptical, #14, #10, #5, #3): domes look the same; differences 0.05–0.25 mm have the sign opposite to wear and match pick/frame error. That comparison was **weak** (height reference was the clicked point, picks differ by tenths of a mm, frame tilt differed). Only "large wear (several tenths mm) not seen". Case 5 has no alignment yet.

**Cases 2, 4, 8, 13:** T1 profiles only; no wear conclusions from profiles. Case 2 T1 automatic detection: section 3.

---

## 5. What failed, and why (do not repeat)

### Detection

1. **Surface variation + DBSCAN [failed].** Per-vertex surface variation (Pauly 2003; k = 30), top 8 %, DBSCAN (eps 0.01 × diagonal, min_samples 20). Result: red = tooth–gingiva border and interdental grooves, no attachments. Reasons: surface variation finds creases, not protrusions; a percentile always selects 8 %; k-NN scale depends on density; whole mesh searched; slow Python loop.
2. **ROI + residual to a Taubin-smoothed surface + seed/grow + DBSCAN [failed].** Taubin smoothing is a low-pass filter that preserves shapes larger than ~14 × edge length (~1.4 mm on a 0.1 mm mesh) independent of iterations; a 2–3 mm attachment survived, residual ≈ 0 (p99.9 = 0.02 mm). Not caused by thresholds or flipped normals.
3. **Quadratic fit on all neighbours within 4 mm, iterative rejection [failed].** `h` median +0.45, p99 3.4, max 4.8 mm (expected ~0 / 0.3–1 / ~1). Reasons: a 4 mm radius covers several layers; the first fit includes the attachment; the 0.06 mm rejection equals noise so the surface sinks; attachment width ≈ fit radius so the fit sits on top.
4. **Ring-based approach (current, works on case 2 T1). Threshold history:**
   - `MAX_TOP_H = 1.6` came from elliptical case 1 and rejected true rectangular attachments in case 2 (h95 1.64, 1.77). Raised to 2.2, then to 2.5 (case 12 had a true one at 2.23).
   - `MIN_SEED_AREA = 2.5` dropped a true case-12 attachment (seed_area 2.48) and kept a false anterior blob at (-3.3, 13.2, 22.1) (h_seed_max 0.80, ring_rms 0.131, 5.0 mm from kept neighbours). Now `MIN_SEED_AREA = 2.3`, `MIN_SEPARATION = 5.5`.
   - Case 12 also showed red patches on proximal surfaces of neighbours; the cleanup cut `h >= max(EDGE_HEIGHT, 0.3 × h95)` and "largest component only" were added (effect unconfirmed).
   - A rendering bug (blocky h-map) was fixed by 4-neighbour inverse-distance interpolation.

General lesson: **any local smoother either swallows the attachment (large scale) or turns crown curvature into a false bump (small scale).** The surface under an attachment must be built from the narrow ring just outside its wall.

### Volume

- An earlier detector volume (Σ per-vertex surface area × h) **overestimated** because surface area counts the near-vertical walls: four values above the Excel max (#2 11.94, #3 13.48, #5 17.73, #6 11.63). It was removed from detection; volume moved to `measure_volume.py` using projected area.
- **Two base surfaces for one attachment (found and fixed):** detection fitted the ring around the region before cleanup, while an earlier `measure_volume` re-fitted a ring around the final (smaller) mask. Cleanup removes gentle foot vertices, which then entered the second ring and raised the base. Fix: single fit; detection saves h, `measure_volume` fits nothing.
- **Foot bias does not cancel in wear %** (an earlier claim that it "almost cancels" was wrong). If the missing foot volume is F, the computed wear is ΔV/(V1−F) instead of ΔV/V1, roughly 1.2× too large [my estimate, not measured]. With different cuts (0.3 × h95) in T1 and T2 the foot is not even the same in both scans.
- `largest_component` chooses by surface area, not by a hidden volume (area × h).

### Other lessons
- Picking points in a plain `PointCloud` shows nothing useful; give normals or use the mesh with `VisualizerWithEditing`.
- Comparing T1 and T2 with heights relative to a clicked point hides wear. Use one common frame and one common base reference.

---

## 6. Key technical decisions

1. **Rigid ICP only, no scale.** Scale is consistent (mm); diagonal ratios 0.97–1.07 come from crops. (A small scale difference remains one explanation for the case 1 buccal offset.)
2. **Trimmed ICP** on a stable overlapping region.
3. **Two-level alignment** (goal sub-0.1 mm): global on stable anatomy (palatal rugae), then local per attachment on a 6–10 mm crown patch excluding the attachment. Manual picking may stay as fallback/QC. **Not needed by the current detection/volume chain;** needed for any T1-mask approach and for the local-ICP cross-check.
4. **Attachment = protrusion above the underlying tooth surface.** Measure height above a base surface, not curvature.
5. **Measure each scan against its own base surface** (thin ring in that scan), then compare. Registration error and tooth motion largely cancel (`top − ring` stable to 0.007 mm in case 1). Keep local ICP as a cross-check.
6. **One base fit, in detection.** The last ring fit is saved as per-vertex h; nothing downstream refits (decided after the two-base-surface bug).
7. **Volume = Σ projected triangle area × mean h** over fully-masked triangles (not `mesh.volume`, not per-vertex surface area). Ray casting / fixed-grid integration remain fallbacks.
8. **Primary wear metric = mean height of the top above the base** (resolution-insensitive); volume and wear % second.
9. **Wear % = (V1 − V2) / V1 × 100** over the same base and mask; also report mean and max height loss in mm. See Q11 and Q12 for the unresolved mask/foot question. No wear % column exists yet.
10. **Noise baseline:** measure the same T1-vs-T2 difference on an attachment-free patch of the same tooth and subtract the constant bias (~+0.04 mm in case 1). Wear that does not clearly exceed it is "not measurable". Not implemented.
11. **Mixed-density cases** (2, 8, 4, 11, 13, 14) are reported separately.
12. Parameters in mm, relative to attachment size, never to jaw diagonal or vertex counts. Do not fix size/orientation (ranges only).
13. **T1 vs T2 detection: OPEN CONFLICT (Q11).** Earlier decision: detect on T1 only; compute V2 on the **T1 mask with the T2 base surface**, negative h clipped to 0, so edge wear is not lost and mixed-density edge noise does not enter. Current pipeline: T2 is detected independently (own mask, own base). Pick one before implementing wear %.
14. `volumes.csv` columns are fixed: `id, pre_volume_mm3, post_volume_mm3` (T1 = pre, T2 = post).
15. **Ids from independent detections are not comparable;** only `match_attachments.py` (or an equivalent) defines correspondence.

---

## 7. Pipeline status and next work

Values are starting points, to be tuned on cases 1, 2, 8 (and 4/13 for coarse). All in mm.

| step | script | status |
|---|---|---|
| 1 Preprocess | `load_mesh` in detection | done (native resolution; 0.3 mm voxel cloud for stage 1) |
| 2 Search region | in detection (`BUCCAL_DOT`, `BORDER_MARGIN`) | done |
| 3 Detection + base surface | `detect_attachments.py` | case 2 T1 only; T2, other cases unrun |
| 4 Volumes | `measure_volume.py` | case 2 T1 only |
| 5 Pairing / numbering | `match_attachments.py` | written, not run |
| 6 Alignment (global + local) | `align.py`; local ICP not written | output deleted; optional |
| 7 Wear %, noise baseline, mean-height metric | – | **not implemented** |
| 8 Tables | `build_tables.py` | planned (Q1 unknown) |
| 9 Validation | overlays, distance heatmap, Excel comparison | Excel needs row order (Q10) |

Proposed table columns (confirm with Q1): `case, tooth/attachment id, shape, area_mm2, h95_T1, h95_T2, V1_mm3, V2_mm3, wear_pct, noise_baseline, T1_density, T2_density, qc_flag`.

**Immediate next steps, in order:**
1. Run T2 detection and pairing on case 2:
   ```
   python src/detect_attachments.py --sample 2 --scan T2 --show none
   python src/match_attachments.py --samples 2
   ```
   (`detect_attachments.py` overwrites `T2_*` files of case 2; `match_attachments.py` overwrites `volumes_matched.csv`.) Send back the terminal output, especially the T2 attachment count (T2 of case 2 is coarse; ≠ 14 is possible), `paired`, `mean pair distance` and warnings.
2. Obtain the Excel row order on the arch (Q10) to match ids/teeth to rows and decide `--reverse`.
3. Decide Q11 (T1 mask vs independent T2) and Q12 (foot correction), then implement wear %.
4. Compare matched volumes with the Excel; investigate the low volumes (foot, base) and suspicious #2, #3, #5, #6.
5. Run detection on cases 8 and 12 (never run with the current detector), then fine/fine cases; add the control-patch noise baseline.
6. Update this file after each step.

---

## 8. Open questions

- **Q1:** What do the Phase 1 "tables" contain? Notes unreadable.
- **Q2:** Do teeth really move between T1 and T2? Case 1 shows a position-dependent offset (~0.1 mm posterior, ~0 anterior). Needs the distance heatmap on more cases.
- **Q3:** Exact attachment size by type; refine from detections (section 4 numbers are ±0.2 mm from plots; detector h95 in case 2: 1.07–1.78 mm).
- **Q4:** Are FDI tooth labels needed, or is an attachment id enough? STLs have no labels.
- **Q5:** Case 3: incomplete, different patient, or recoverable?
- **Q6 (partly answered):** Ground truth = the supervisor's Excel (group 2 only, 14 rows, wear 0.4–48 %). Other groups/cases unknown. Which cases have worn attachments?
- **Q7:** Closed for cases seen: y is occlusal–gingival, +y toward gingiva/palate. Re-verify on new cases.
- **Q8:** Shape per case: known 1, 5 elliptical; 2, 8 rectangular; 4 flat-top (coarse); unknown for 6, 10–14.
- **Q9:** Is the case 1 buccal offset real tooth movement or a small scale difference?
- **Q10:** In which order do Excel rows 2.1–2.14 sit on the arch (e.g. 2.1 = tooth 7 on the right, 2.14 = tooth 7 on the left)? Is group 2 really case 2? Needed for matching and for `--reverse`.
- **Q11:** Wear volume on the T1 mask with the T2 base (clip negatives), or independent T2 detection (current)? Affects edge wear, mixed-density cases, and foot bias.
- **Q12:** Foot-bias correction for wear %: missing foot F gives ΔV/(V1−F) (~1.2× too large, estimate). Options: same mask and cut in both scans, or an estimated foot correction. Needs a decision and a measurement.
- **Q13:** Numbering direction of `arch_order` (arbitrary eigenvector sign) vs the Excel order; depends on Q10.

---

## 9. Glossary and conventions

- **T1 / T2:** scan before / after ("pre" / "post" in volume tables). **T2 is the fixed reference frame** where alignment is used; `T1_to_T2.npy` is a 4x4 matrix from raw T1 into the T2 frame; `T1_aligned.stl` is T1 already transformed.
- **Buccal:** outer (cheek) side of a tooth, where attachments sit. **Occlusal / incisal:** biting edge side. **Gingival:** gum side. In the data, +y = gingiva/palate, −y = biting edge.
- **Attachment top:** flat (rectangular) or crest (elliptical) surface. **Wall / foot:** the steep side and its base.
- **Ring:** narrow band of tooth surface just outside the foot, used as the base reference. **Annulus:** the wider band (2.0–3.5 mm) used by stage 1 only.
- **Base surface:** estimated tooth surface under the attachment (the last ring fit in detection).
- **h / d / residual:** height above the base surface (mm); positive = above tooth. **h95:** 95th percentile of h over the mask. **h_top:** mean h of mask vertices with h ≥ 0.8 × h95, reported by detection.
- **Seed / region / mask:** seed = working points with h ≥ 0.5 mm; region = grown set (h ≥ 0.25); mask = final cleaned vertex set used for volume.
- **qc_flag:** "check" if ring_rms > 0.10 mm or area > 16 mm², else "ok"; a warning, not a rejection.
- **D = T2 − T1** height difference in one frame; negative = wear. **top / ring / top − ring** (`profile_compare.py`): mean D on the top, median D on the ring (= local registration error), and their difference.
- **snap:** distance between the clicked point and the nearest vertex. **tilt:** angle between the local normal and the radial direction of the arch.
- **fine / coarse:** mean edge ~0.1 mm / ~0.2 mm. **Foot:** low gentle skirt of an attachment cut off by the detector (volume bias).
- **Pick:** vertex clicked in an Open3D window (Shift+Left add, Shift+Right remove, Q/Esc finish).
- **FDI:** two-digit tooth numbering; not available in the files.
- **Wear %:** (V1 − V2) / V1 × 100 over the same base surface and mask.
- File naming: `<scan>_attachments.csv`, `<scan>_attachment_labels.npy`, `<scan>_attachment_height.npy`, `volumes.csv`, `volumes_matched.csv`; `profile_<T1|T2>_<k>.png`, `picks_<T1|T2>.npy`, `compare_<k>.png`.

---

## 10. Environment and workflow

- Run everything from the repo root (`python src/<script>.py --sample <id>`), git branch `main`. Python venv with `numpy open3d trimesh scikit-learn scipy rtree matplotlib`. Scripts import `config` (`from config import DATA_DIR, PROCESSED_DIR`), which works because `src/` is the script directory.
- Harmless Open3D messages: "Couldn't open plugin directory", "No plugins found, falling back on no decorations".
- `rtree` must be installed or `trimesh.proximity` fails (it already caused a missing result once).
- Typical workflow: a maintainer runs the command given by the agent and pastes back the **terminal output and screenshots**; whole files are pasted when asked (pasted code may lose indentation; treat that as a formatting artefact, not a bug). Outputs are sometimes renamed by hand to avoid overwriting, so say explicitly which files a command overwrites.
- A single run on a 500k-vertex mesh can take 30–60 s or more; say so, and keep point counts small (voxel 0.3 mm) when possible.
- Screenshots usually show Open3D windows: titles like "pick points WITHOUT attachments", "Pick the CENTRE of attachments (Shift+Left Click, Q to finish)", "attachments (red)", "h map (red = above base, +-0.4 mm)"; orange = T1, blue = T2, red = detected.
- Commit messages are short. Detection and `measure_volume.py` were committed earlier; later edits (paths via `config.py`, removal of version labels) and `match_attachments.py` are not confirmed as committed.

---

## 11. FAQ — questions a future contributor tends to ask (already answered)

- **Which scan is the reference?** T2 (where alignment is used). T1 is moved onto T2.
- **Are units mm?** Yes, everywhere.
- **Are there tooth labels or masks?** No. Only raw STL. Ground truth: the Excel (group 2) only.
- **Can I use `mesh.volume`?** No (open meshes). Use `measure_volume.py`.
- **Can thresholds be in vertices/percentiles?** No; fine and coarse meshes differ 4x. Use mm.
- **Which axis is up?** y; +y toward gingiva/palate (verified on cases profiled). Check on new cases.
- **Why not curvature/surface variation, or smoothing and subtracting?** Tested and failed (section 5).
- **Why four alignment picks, not two?** Two points leave a rotation free; the code requires ≥ 4.
- **Why is case 3 skipped?** Section 2. Never include it in batch runs.
- **Can the pipeline ask the user to click centres?** Only diagnostic/validation scripts. The final detector must be automatic.
- **How large is an attachment?** Height ~1–1.8 mm, top ~2.5–3 × 2 mm, area ~5–9 mm², volume ~7 mm³ (rectangular, Excel) or ~3–4 mm³ (elliptical, case 1).
- **How big is the wear?** Excel group 2: 0.4–48 %, average ~22 % (not "≤ 15 %"). Case 1 showed none. Wear below ~5 % is probably not detectable.
- **Is real tooth movement expected?** Possibly; unresolved (Q2, Q9).
- **Should T2 be detected separately or on the T1 mask?** Open (Q11). The current pipeline detects T2 independently; the earlier plan was the T1 mask with the T2 base.
- **Why does `volumes.csv` have only three columns?** Maintainer's decision (section 6, item 14); extra columns can be restored on request.
- **Are `volumes.csv` ids the same attachment in T1 and T2?** Not guaranteed; use `match_attachments.py`.
- **Does `detect_attachments.py` output volumes?** No, never.
- **Should scripts carry version numbers?** No (section 0).
- **Where do paths come from?** `src/config.py` (`DATA_DIR`, `PROCESSED_DIR`).
- **What language for code and docs?** English; talk to Persian-speaking members in Persian.
- **May I run GUI scripts?** Ask first.
- **Which cases to start with?** Case 2 (running now), then 8, 12 (never run with the current detector), then 1; coarse-T1 cases 4, 11, 13, 14 after that.
- **What to do if a result is suspicious?** Report it with quality metrics and flag it; never force a number.

---

## 12. Things to collect from the maintainers when needed

- Excel row order on the arch and which case group 2 is (Q10); Excel of other groups, if any; meaning of "defect / excess / RMS".
- The Phase 1 "tables" definition (Q1).
- Which cases have worn attachments (Q6) and shape per case (Q8).
- The current content of `src/config.py` and of any script before patching (whole files are pasted).
- Terminal output of every new run (attachment counts per scan, paired, mean pair distance, warnings).
- Output of `python src/test.py --samples <ids>` after new cases are added.

---

## 13. Reference list

- Pauly, Keiser, Gross. *Multi-scale Feature Extraction on Point-Sampled Surfaces.* Eurographics 2003 (surface variation).
- Lee, Varshney, Jacobs. *Mesh Saliency.* SIGGRAPH 2005.
- Besl & McKay 1992; Rusinkiewicz & Levoy 2001; Chetverikov et al. 2002 (ICP and Trimmed ICP).
- Rusu et al. 2009 (FPFH).
- Choi et al. *Accuracy and reliability of palatal superimposition of 3D digital models.* Angle Orthodontist, 2010.
- Tooth segmentation by curvature: Kondo 2004; Zhao 2006; Kronfeld 2010.
- Future deep learning: MeshSegNet (Lian et al. 2020), Teeth3DS dataset.