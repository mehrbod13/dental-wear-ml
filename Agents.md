# AGENTS.md — dental-wear-ml

Guide for AI agents and contributors. Read it fully before changing code.

Status: Phase 1 in progress. Detection and volume measurement were run on **case 2 T1 only**, with an earlier detector version. The current detector (arch-order ids) has not been validated; a rerun of case 2 is pending. Matching exists but was **never run**. Wear % is **not implemented**.

Tags: **[measured]** = seen in data, plots or terminal output. **[assumed]** = hypothesis, not verified. **[failed]** = tried, did not work. Untagged = a decision or a fact from the maintainers.

---

## 1. Task

- Measure the **wear of orthodontic attachments** (small composite bumps glued on teeth; this is not tooth wear) from two STL scans of one upper jaw: T1 (before, "pre") and T2 (after, "post").
- Output per attachment: V1 (volume at T1), V2 (volume at T2), wear % = (V1 − V2) / V1 × 100.
- Phase 1 = geometric mesh processing only (`open3d`, `trimesh`, `scikit-learn`, `scipy`). No learning. A later phase may use deep learning once Phase 1 gives labelled data.
- Active cases: 1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14. Case 3 is excluded. Cases 7 and 9 do not exist.
- Every case has 14 attachments per scan (per the reference Excel).
- Agents usually have **no STL access**. A maintainer runs the code and sends back terminal output and screenshots.

Original plan (`docs/Phase_1.md`, handwritten notes), with status:
1. Align T1 and T2 (manual picks on attachment-free regions, then ICP): done for cases 1 and 2 only; output currently deleted.
2. Isolate attachments: "curvature + DBSCAN" does not work; replaced by ring-based detection.
3. Wear volume: `trimesh` volume is unusable; replaced by `measure_volume.py`.
4. Tables: content unknown (Q1); `build_tables.py` is planned.

---

## 2. Working rules

**Language.** The team speaks Persian: talk to Persian speakers in Persian. Code, comments, docstrings, docs and commit messages are in English.

**Communication.**
- Be concise. Numbers in mm. At most one table per reply.
- Say what was seen and what is uncertain. Ask at most one question.
- End with exactly what to run and what to send back.
- Do not guess; look at the data first. Several approaches written from assumptions failed (see "What failed"). Before a new algorithm, produce a plot or number from real data that supports its assumptions.
- Say when code is untested; give the exact command and the output to send back.
- Be honest about uncertainty. A wrong wear number that looks valid is worse than none. Report quality metrics and flag suspicious cases instead of forcing a result.
- When a maintainer says an output is wrong, assume they are right and look for the false assumption instead of tuning thresholds.

**Safety.**
- Never modify `data/<id>/T1.stl` or `T2.stl`. Derived files go to `data/processed/<id>/`.
- Do not paste or print full meshes (10–65 MB each). Use summary statistics.
- Ask before running GUI scripts (`align.py`, `profile_attachment.py`, any `--show` viewer). Use `--show none` in batch.
- Say which files a command overwrites.
- Give full files, not patches. Ask for the current file before patching. Pasted code that lost its indentation is a formatting artefact.
- Case 3 never appears in batch runs or example commands.

**Code conventions.**
- One script per step in `src/`. `argparse` with `--sample <id>`. Constants at the top.
- Paths come from `src/config.py` (`DATA_DIR`, `PROCESSED_DIR`). Results go to `data/processed/<id>/`.
- No version labels in files, docstrings or argparse text. History lives in git.
- Everything in millimetres. Thresholds are never in vertex counts or vertex percentiles (mesh density differs ~4x between files).
- One method everywhere: the base surface is fitted **once, in detection**; later steps reuse its result. Arch numbering lives **once**, in `detect_attachments.arch_order`; other scripts import it.
- `detect_attachments.py` never computes or outputs volumes.
- Do not use `trimesh` `mesh.volume` (meshes are not watertight). Rigid ICP only, no scale.
- Every script loads meshes through the shared `load_mesh(path)` of `detect_attachments.py`, because the vertex order of the label and height files is the vertex order of its result.
- Commit messages are short (one subject line, optional short body).

---

## 3. Data

**Layout**
```
data/
  <id>/T1.stl, T2.stl          raw scans (read-only); ids 1 2 3 4 5 6 8 10 11 12 13 14
  processed/<id>/              derived files
    <scan>_segmented.ply       coloured detection result (scan = T1 or T2)
    <scan>_detection_debug.ply debug mesh, written only with --debug
    <scan>_height_map.ply      h-map for viewing
    <scan>_attachment_labels.npy  int32 per vertex, -1 = background
    <scan>_attachment_height.npy  float32 h above base for mask vertices, NaN elsewhere
    <scan>_attachments.csv     one row per attachment
    volumes.csv                id, pre_volume_mm3, post_volume_mm3
    volumes_matched.csv        paired table
    T1_aligned.stl, T1_to_T2.npy   from align.py
    picks_*.npy, profile_*.png, compare_*.png   diagnostics
  mesh_report.txt              output of src/test.py
docs/Phase_1.md, Phase_1.png   original notes
src/                           scripts (see "Code")
```
- One folder = one jaw: 14 teeth plus palate and gingiva. STL files carry **no tooth labels**.
- `T1_segmented.ply` of case 1 is an outdated result of a failed approach. Do not trust it.
- Processing status: case 2 has T1 detection and volume (earlier detector version); case 1 has alignment only; all others are unprocessed. Cases 8 and 12 were never run with the current detector.

**Scan properties [measured]** (`src/test.py`, all active cases)
- Units are millimetres. PCA extents ~64–70 × 52–66 × 20–28 mm. Case 14 is a smaller jaw (diagonal 85 mm vs 91–96 mm); parameters must not depend on jaw size.
- Frame: x ≈ −34..+34, z ≈ −31..+31, y ≈ 7..33. The arch lies in the x–z plane. **y = occlusal–gingival axis, +y toward gingiva/palate** (vertical sections of case 2). Verify on each new case before hard-coding axes.
- Every mesh is one connected component and **not watertight**.
- STL triangle order and normal orientation are not consistent. Orient normals explicitly before using their sign.
- T2 always has a larger surface area than T1 (more gingiva/palate, different crop). Raw centroid differences of 1–4 mm come from the crop, not motion. Whole-mesh ICP is biased by the trimmed borders.
- Saved rotations are real: 14.0° (case 1), 9.8° (case 2).

**Mesh density** (most important data finding). Fine: mean edge ≈ 0.10–0.12 mm, 470–650k vertices. Coarse: ≈ 0.20–0.23 mm, 120–165k vertices.

| T1 / T2 | Cases |
|---|---|
| fine / fine | 1, 5, 6, 10, 12 |
| fine / coarse | 2, 8 |
| coarse / fine | 4, 11, 13, 14 |

- Sharp edges of rectangular attachments round off on coarse meshes. A T1-vs-T2 difference can be a **resolution artefact, not wear**.
- Edge position on a coarse mesh is uncertain by ~±0.1–0.2 mm. With perimeter 8–9 mm and area 5–6 mm², a 0.15 mm edge error is a 15–20 % area error, as large as the wear. Independently thresholded edges in mixed-density cases are a known risk (Q11).
- Attachment top height is reliable even on the coarse mesh [measured, case 4].
- Do not upsample coarse meshes. Do not decimate fine meshes to 0.2 mm. Detection stage 1 uses a common 0.3 mm voxel cloud; measurement uses native vertices.
- Report mixed-density cases separately.

**Case 3 is excluded.** Its T1 is abnormal: bbox x −85..−29 (others −34..+34), different orientation, surface area 3266 (others 4100–5300), 59 mm centre offset to T2, only ~70 % of the area and half the height after rotation. Probably a different frame plus incomplete data. Revisit at the very end (Q5): global initial alignment (PCA / FPFH+RANSAC), and check that all 14 teeth and attachments exist.

**Alignment quality [measured]** (nearest-point distance, aligned T1 to T2, every 20th vertex, mm):

| case | mean | median | p90 | max |
|---|---|---|---|---|
| 1 | 0.051 | 0.026 | 0.076 | 4.57 |
| 2 | 0.097 | 0.050 | 0.186 | 4.25 |

These numbers mostly describe palate and gingiva and are optimistic (ICP minimises almost this metric). The max comes from regions present in one scan only. **Local error is larger** (case 1: posterior teeth offset ≈ +0.08..+0.10 mm buccally, anterior ≈ 0). Global alignment alone is not accurate enough for wear measurement; this is why each scan is measured against its own base surface.

---

## 4. Attachments

Two shapes occur (the Excel gives the shape per case, see "Reference Excel"):
- **Elliptical:** smooth dome, no plateau, no steep wall.
- **Rectangular (rounded corners):** flat top, near-vertical walls. Some have a bevel.

Attachments sit on the buccal side, about one per tooth; neighbouring centres are ~7 mm apart. Teeth hang toward −y; the palate/gingiva is the flat region at high y.

**Geometry** (±0.2 mm, from hand sections and contour maps):

| quantity | rectangular (cases 2, 8) | elliptical (case 5; case 1 similar) |
|---|---|---|
| height above tooth surface | ~1–1.3 mm by hand; detector h95 in case 2: 1.07–1.78 | ~1.1–1.5 mm |
| top size | ~2.5–3 mm along arch × 2 mm occlusal–gingival | dome ~2 × 3.5 mm at half height |
| top area | ~5–8 mm² (case 2 projected: 5.6–9.3) | – |
| top flatness | ±0.05 mm along arch | not flat |
| walls | ~1 mm drop within 0.2–0.3 mm | ~1 mm drop over ~1–1.5 mm |
| occlusal side | no wall; top ramps up over ~2 mm (bevel) | – |

- Volume per the Excel: 2.9–10.2 mm³ over all rows; per-case mean 4.6–7.0 mm³.
- The top is tilted: it rises ~0.5–0.7 mm from the occlusal to the gingival side, also for picks with local normals tilted 9°, 37°, 57°, so it is design, not a frame error [probable]. **Do not define the frame normal from the attachment top** (it differs ~15° from the tooth surface). Measure heights above a tooth base surface.
- Size and orientation differ between cases. Use ranges (~1.5–4 mm), not fixed values.

**Surroundings (what limits the base surface)**
- Gingival margin ~3.5–5 mm on the +y side; incisal edge ~3–4 mm on the other side. Interdental groove 2.5–3.5 mm deep at ~±4 mm along the arch. Some attachments (case 10) have an undercut at the gingival margin.
- Clean tooth surface exists only in a narrow strip ~0.3–1.5 mm outside the wall base (~2 mm toward occlusal/gingival, 0.5–1 mm along the arch). Along the arch the surface falls at 35–45° toward the groove. A 4 mm-radius neighbourhood contains several surface layers and is not a height field.

**T1 vs T2 from hand-picked profiles (before the automatic pipeline)**

Case 1 (4 elliptical attachments, both scans fine), heights in mm:

| pick | position | normal tilt | ring (registration error) | top | top − ring |
|---|---|---|---|---|---|
| 1 | posterior (x=+27, z=−13) | 40° | +0.102 | +0.144 | +0.042 |
| 2 | anterior (x=11, z=23) | 18° | −0.005 | +0.024 | +0.029 |
| 3 | anterior (x=−16, z=23) | 18° | −0.013 | +0.030 | +0.043 |
| 4 | posterior (x=−25, z=−14) | 41° | +0.077 | +0.114 | +0.037 |

- No visible wear: sections overlap within ~0.05 mm. This **contradicts the Excel** for case 1 (wear 5–19 %; Q14).
- `top − ring` is stable (spread ~0.007 mm), so referencing to the neighbouring surface cancels registration error. A constant bias of ≈ **+0.04 mm** remains (not real growth; probably ring slopes or sampling); it must be measured on a control patch and subtracted.
- Posterior teeth shifted ~+0.08..+0.10 mm buccally on both sides, anterior ~0: real arch widening or a small scale difference (Q2).
- Wall points in the difference map are noisy (±0.3 mm); use only the top (r < 1 mm).

Case 5 (4 elliptical): domes look the same; differences of 0.05–0.25 mm have the sign opposite to wear and match pick/frame error. The comparison was weak. Only "large wear (several tenths of a mm) not seen". Case 5 has no alignment.

Cases 2, 4, 8, 13: T1 profiles only; no wear conclusions.

---

## 5. Reference Excel (supervisor)

This file calculated volumes manually with ControlX.
Rows 1–14 in Excel order; volumes in mm³; loss volume = pre − post; row 10 is the one labelled "x.1" in the Excel.

```
case 1
pre : 6.3298 5.8484 6.2730 6.0297 5.1378 4.0718 5.1423 5.8148 4.4849 4.1057 4.0333 4.7562 5.3975 6.3628
post: 6.0059 5.0537 5.5068 4.9905 4.2433 3.6648 4.4226 4.7597 4.1998 3.7506 3.2702 4.2154 4.9591 5.9700
loss%: 5.12 13.59 12.21 17.23 17.41 9.99 14.00 18.14 6.36 8.65 18.92 11.37 8.12 6.17

case 2
pre : 7.5480 7.3974 6.4947 5.5584 5.2353 6.9859 8.2243 7.9928 6.6205 10.1573 6.9146 7.3965 5.9218 6.7856
post: 7.1065 6.4658 6.2038 3.9237 3.1452 6.5704 5.8233 6.4647 4.3051 5.2385 3.7109 5.1495 5.8998 5.7229
loss%: 5.85 12.59 4.48 29.41 39.92 5.95 29.19 19.12 34.97 48.43 46.33 30.38 0.37 15.66

case 3 (excluded)
pre : 5.2847 5.2386 4.1741 4.0707 2.9851 3.1601 5.5416 6.1223 4.3037 4.8248 5.0834 6.6128 6.1635 5.8718
post: 4.7425 4.6911 3.6966 3.8273 2.8982 2.9534 5.2978 5.9344 4.0461 4.5807 4.8812 5.9555 5.5308 5.3473
loss%: 10.26 10.45 11.44 5.98 2.91 6.54 4.40 3.07 5.99 5.06 3.98 9.94 10.26 8.93

case 4
pre : 7.0522 5.9511 4.1170 4.5258 4.0459 6.7125 8.5612 5.7775 4.5709 5.4002 6.7713 6.3436 6.8375 5.8146
post: 6.3393 5.5513 4.0537 4.1927 3.8304 5.9001 8.4936 5.6086 4.0725 4.9920 5.8034 5.8694 6.4759 5.2698
loss%: 10.11 6.72 1.54 7.36 5.32 12.10 0.79 2.92 10.90 7.56 14.29 7.48 5.29 9.37

case 5
pre : 6.2710 5.4082 4.6651 5.2622 3.4138 5.0982 6.4620 4.4674 3.0608 4.6775 5.9090 5.9447 6.3502 6.3731
post: 5.7331 4.7952 4.3074 4.6545 2.5842 4.6173 6.3723 4.4068 3.0078 3.9210 5.3724 5.6901 5.8799 6.3549
loss%: 8.58 11.34 7.67 11.55 24.30 9.43 1.39 1.36 1.73 16.17 9.08 4.28 7.41 0.29

case 6
pre : 5.9861 3.7542 4.0291 3.1447 3.3977 4.4147 4.7286 6.3587 3.1250 4.8562 5.9750 4.7129 5.2499 4.6264
post: 4.8216 3.6855 3.4806 3.0378 2.7508 4.2793 4.6507 5.3145 2.8650 4.8424 5.5844 4.6849 4.6870 4.2352
loss%: 19.45 1.83 13.61 3.40 19.04 3.07 1.65 16.42 8.32 0.28 6.54 0.59 10.72 8.46

case 8
pre : 6.6006 5.6466 4.0615 2.9349 4.3932 7.2516 8.6556 7.6596 5.3567 6.8518 9.1907 7.9975 7.2947 6.0865
post: 5.7539 4.3945 3.8244 2.7343 3.8067 6.4492 7.4897 7.0965 4.8390 5.9651 7.5212 6.8473 5.8146 5.3023
loss%: 12.83 22.17 5.84 6.84 13.35 11.07 13.47 7.35 9.67 12.94 18.17 14.38 20.29 12.88

case 10
pre : 5.3074 6.6189 5.5765 3.9183 3.2118 4.6640 7.4626 7.1048 6.6785 5.5808 7.2693 7.8393 6.6797 7.4981
post: 5.2067 6.2856 5.2357 3.4330 2.9241 4.0156 6.8953 6.8267 6.2953 4.8361 6.6208 7.1887 6.2644 7.2796
loss%: 1.90 5.04 6.11 12.39 8.96 13.90 7.60 3.91 5.74 13.34 8.92 8.30 6.22 2.91

case 11
pre : 6.4509 6.2020 5.3892 5.4938 4.0589 5.3071 5.4435 5.6522 2.9850 3.1439 4.3288 5.1231 6.1867 5.2583
post: 5.9050 4.7616 4.2405 3.9119 4.0123 4.2643 4.6840 4.8575 2.5919 2.9437 3.8949 4.6903 5.4853 4.4510
loss%: 8.46 23.22 21.31 28.79 1.15 19.65 13.95 14.06 13.17 6.37 10.02 8.45 11.34 15.36

case 12
pre : 8.9893 8.0761 8.0341 5.4844 3.5829 6.5780 8.5860 5.4308 3.8985 5.0692 6.5092 7.2623 8.4417 6.6001
post: 8.5126 7.0887 6.9684 5.0599 3.2898 5.6979 7.7287 4.6970 3.6082 4.7496 6.1364 7.0266 7.9737 6.4232
loss%: 5.30 12.23 13.27 7.74 8.18 13.38 9.98 13.51 7.45 6.30 5.73 3.24 5.54 2.68

case 13
pre : 6.5725 5.7263 5.3849 5.8024 3.9322 4.3422 6.0057 6.3959 5.0052 3.2751 3.7100 4.3253 6.7439 7.2561
post: 5.9935 4.9202 4.7515 5.5693 3.7319 4.2902 5.6301 6.0661 4.6581 2.7166 3.2922 3.5655 5.3016 6.3238
loss%: 8.81 14.08 11.76 4.02 5.09 1.20 6.25 5.16 6.93 17.05 11.26 17.57 21.39 12.85

case 14
pre : 5.5524 4.9616 4.0630 3.9116 2.9236 4.7711 4.8018 4.5698 2.9118 4.7782 4.9142 5.1881 5.4141 5.7679
post: 5.3137 4.3917 3.1380 2.8201 2.8757 4.5718 4.7220 4.4237 2.7718 3.5965 4.3346 4.4129 5.0487 4.9744
loss%: 4.30 11.49 22.77 27.91 1.64 4.18 1.66 3.20 4.81 24.73 11.79 14.94 6.75 13.76
```

Excel mean rows (pre / post / loss %): case 1: 5.28 / 4.60 / 12.11; 2: 7.01 / 5.42 / 21.86; 3: 4.96 / 4.46 / 7.74; 4: 5.89 / 5.32 / 7.27; 5: 5.24 / 4.69 / 8.18; 6: 4.69 / 4.21 / 9.52; 8: 6.38 / 5.49 / 13.37; 10: 5.96 / 5.38 / 7.52; 11: 5.13 / 4.41 / 14.04; 12: 6.60 / 5.94 / 8.08; 13: 5.38 / 4.62 / 14.10; 14: 4.60 / 4.17 / 10.71.

## 6. Code

| file | status |
|---|---|
| `src/config.py` | shared paths; content not reviewed by an agent (assumed `DATA_DIR = <root>/data`, `PROCESSED_DIR = <root>/data/processed`) |
| `src/align.py` | manual picking + ICP; works; GUI; output currently deleted |
| `src/detect_attachments.py` | automatic detection + arch ids; edited by the maintainer, untested by agents; run on case 2 T1 only |
| `src/measure_volume.py` | works; run on case 2 T1 |
| `src/match_attachments.py` | written by a contributor; **never run** |
| `src/test.py` | inspection script; needs `rtree` |
| `src/profile_attachment.py`, `src/profile_compare.py` | diagnostics (GUI / hand picks); not part of the pipeline |
| `requirements.txt` | should contain `numpy open3d trimesh scikit-learn scipy rtree matplotlib`; check it matches |

Planned, not written: `src/build_tables.py` (Q1), `src/compare_excel.py`, wear % and noise baseline.

### `detect_attachments.py`

CLI: `python src/detect_attachments.py --sample <id> [--scan T1|T2] [--show seg|heat|none] [--show-ids] [--reverse] [--debug] [other flags]`.
- `--scan` defaults to T1. Detection does not need alignment.
- `--show-ids` draws arch ids; `--reverse` flips the numbering direction; `--debug` also writes the debug PLY.
- Threshold flags exist for most constants (for example `--voxel`, `--ann-in`, `--ann-out`, `--seed-height`, `--min-seed-area`, `--min-separation`, `--edge-height`, `--edge-rel`, `--ring-in`, `--buccal-dot`, `--min-area`, `--max-area`, `--flip-normals`). Check the file for the exact list.
- Input: `DATA_DIR/<sample>/<scan>.stl`. **Overwrites** the `<scan>_*` files of that sample.
- If no attachment is kept, the stale `<scan>_attachments.csv` is deleted; label and height files are still written (all background / NaN).

Method:
1. **Stage 1.** Normals are oriented outward automatically (the outermost 15 % of the arch must face away from the centre; `--flip-normals` inverts). Working points: 0.3 mm voxel cloud, buccal, ≥ 1 mm from the scan border. For each, a quadric is fitted to the annulus `ANN_IN..ANN_OUT` using only points with a similar normal; the fit is rejected if points, angular sectors or residual rms are insufficient (gingival margins, grooves and crown edges become "invalid", not bumps). h = height above that fit. Seeds = valid points with h ≥ `SEED_HEIGHT`. DBSCAN on seeds; clusters under `SEED_MIN_AREA` are ignored silently, under `MIN_SEED_AREA` dropped as edge effects.
2. **Stage 2, per cluster (largest seed area first).** Ring fit on native vertices (thin ring just outside the region; wall removed by a normal filter). The region grows where h ≥ `EDGE_HEIGHT` (up to `REFINE_ITERS` iterations). **The last ring fit is THE base surface** and defines the saved h.
3. **Cleanup (selection only, never refits).** Keep vertices with `h >= max(EDGE_HEIGHT, EDGE_REL × h95)`, then the largest mesh-connected piece **by surface area**.
4. **Gates.** Overlap (> 30 % of vertices already labelled), distance to a kept attachment < `MIN_SEPARATION`, area outside `MIN_AREA..MAX_AREA`, h95 outside `MIN_TOP_H..MAX_TOP_H`. `qc_flag = "check"` if ring_rms > `QC_MAX_RING_RMS` or area > `QC_MAX_AREA`, otherwise "ok" (a warning, not a rejection).
5. **Numbering.** `arch_order` numbers the kept attachments along the arch. #0 is the end with the smaller lateral coordinate (≈ smaller x); ids grow toward +x.

CSV columns at last review (the file is the source of truth): `id, cx, cy, cz, area_mm2, h95_mm, h_top_mm, n_vertices, ring_rms_mm, nx, ny, nz, qc_flag`. Coordinates are in the scan frame. No volume column.

Constants (mm unless stated):
```
VOXEL 0.3  ANN_IN 2.0  ANN_OUT 3.5  ANN_NORMAL_DOT 0.65  ANN_MIN_POINTS 60
ANN_MIN_SECTORS 5  ANN_MAX_RMS 0.15  SEED_HEIGHT 0.5  BORDER_MARGIN 1.0
BUCCAL_DOT 0.35  DBSCAN_EPS 0.7  DBSCAN_MIN_SAMPLES 5  SEED_MIN_AREA 0.5
MIN_SEED_AREA 2.3  MIN_SEPARATION 5.5  PATCH_RADIUS 6.0  SEED_RADIUS 0.25
RING_IN 0.4  RING_OUT 1.6  RING_NORMAL_DOT 0.8  EDGE_HEIGHT 0.25  EDGE_REL 0.3
GROW_STEP 1.0  MAX_EXTENT 3.0  REFINE_ITERS 4  MIN_REGION_VERTICES 20
MIN_AREA 2.0  MAX_AREA 25.0  MIN_TOP_H 0.5  MAX_TOP_H 2.5
QC_MAX_RING_RMS 0.10  QC_MAX_AREA 16.0  LEADER_LENGTH 6.0
```

Fragile margins (thresholds were set from few cases):
- `MIN_SEED_AREA`: case 1 true ≥ 3.35, false ≤ 1.40; case 2 true ≥ 2.68, false ≤ 2.05; case 12 true ≥ 2.48.
- `MAX_TOP_H`: true h95 up to 2.23 (case 12); a wrong base gives ≥ 2.7.
- `MIN_SEPARATION`: smallest true spacing seen ~7.0 mm; a false duplicate in case 12 was 5.0 mm away.
- Heavily worn T2 attachments may fall below `SEED_HEIGHT` / `MIN_TOP_H` and be missed (not observed yet).
- Case 12: whether `EDGE_REL` removed the red patches on proximal surfaces of neighbours is unconfirmed; the effect of "largest piece by area" is unchecked.

Risks (untested code):
- One missed attachment shifts all later ids.
- Numbering direction is fragile when the in-plane spreads are similar (small jaw, case 14).
- "Left to right = +x" assumes the viewer is on the buccal side with +y up (Q10).
- `--show heat --show-ids` with no attachments passes `None` (crash).
- The file header reads `src/detect_attachment.py` (typo).

### `measure_volume.py`

Fits **no** surface. Volume = Σ over triangles with all three vertices in the mask of `0.5·|cross(b−a, c−a)·n_ref|` (area projected onto the plane normal to `n_ref`) × mean of the three saved h values. `n_ref` comes from CSV `nx, ny, nz`.
- Per scan (`SCANS = (("T1","pre"),("T2","post"))`) it needs `data/<id>/<scan>.stl`, `<scan>_attachment_labels.npy`, `<scan>_attachment_height.npy`, `<scan>_attachments.csv`.
- If one scan is missing, its column is empty and a warning is printed. If both are missing: error, no output.
- Warnings: mask vertex count vs CSV `n_vertices`; h95 from the height file vs `h95_mm` (tolerance 1e-3 mm).
- Output `volumes.csv`: `id, pre_volume_mm3, post_volume_mm3` (fixed by the maintainers; extra columns can be restored on request). Rows are joined **by id only**; ids from independent T1/T2 detections are not guaranteed to be the same attachment.
- `measure_scan(sample, scan, label)` returns `{id: volume or None}`, or `None` if the scan is missing.
- **Known bias:** border triangles and the low foot (below the detector cut) are not counted, so volumes are low.

Issues:
- Must import `arch_order` from `detect_attachments` instead of its own copy.
- The header says `# src/measure_scan.py` while the docstring and usage say `match_attachments.py`. Make them consistent. The docstring example `--samples 1 2 3` contains case 3 and must be fixed.
- The centre-only transform is fragile: equal counts with one false and one missed detection give a wrong transform; only the 2 mm warning catches it.
- Default `--teeth 14` blocks cases that give 13; use `--teeth` or `--force`.
- T2 is detected independently (own mask and base), unlike the T1-mask option (Q11). A strongly worn attachment missed in T2 becomes a T1-only row with an empty `post`.
- Terminal output is busy because `measure_scan` prints both tables.

### `align.py`
- Asks for ≥ 4 corresponding points (Shift+Click, same order, attachment-free areas) in `VisualizerWithEditing`. `kabsch()` gives the initial rigid transform (rejects collinear/coplanar picks). Then two-stage point-to-plane ICP on 100k sampled points (max correspondence 0.05 × diagonal, then 0.02 × diagonal ≈ 4.8 and 1.9 mm). T2 is fixed.
- Warns if pick RMSE > 0.02 × diagonal or fitness < 0.3. Saves `T1_to_T2.npy` and `T1_aligned.stl`.
- Weaknesses: manual picks (not reproducible); ICP over the whole mesh including gingiva/palate and crop borders (no trimming); distances scaled by diagonal; the 1.9 mm fine threshold is too loose for sub-0.1 mm goals; no tooth-level alignment; quality metrics are not saved.
- The detection/volume chain does not need alignment; only `match_attachments.py` optionally reads `T1_to_T2.npy`.

### Diagnostics
- `profile_attachment.py`: `--sample <id> --file T1.stl|T2.stl`, optional `--load-picks T1 --transform`, `--slab` (0.2 for coarse), `--grid` (0.3 coarse, 0.15 fine). Click an attachment centre from the buccal side; saves vertical section, horizontal section and height map (`profile_<scan>_<k>.png`, `picks_<scan>.npy`). Output names overwrite earlier files with the same prefix. Heights are relative to the picked point.
- `profile_compare.py`: `--sample <id>`; needs `T1_aligned.stl`, `T1_to_T2.npy`, `picks_T1.npy`. Shows T1 and T2 in one frame plus D = T2 − T1 (negative = wear); prints `top: mean D` (r < 1 mm) and `ring: median D`. Case 1 was run (~43 s).
- `test.py`: prints size, density, watertightness, T1/T2 consistency, saved-alignment angle and distance; writes `data/mesh_report.txt`. Usage: `python src/test.py [--samples 1 2 8]`.
- Picking does not belong in the final pipeline.

---

## 7. What failed (do not repeat)

**Detection**
1. **Surface variation + DBSCAN [failed].** Per-vertex surface variation (k = 30), top 8 %, DBSCAN. Result: tooth–gingiva border and interdental grooves, no attachments. Surface variation finds creases, not protrusions; a percentile always selects 8 %; k-NN scale depends on density.
2. **Taubin-smoothed surface + residual + seed/grow [failed].** Taubin smoothing preserves shapes larger than ~14 × edge length (~1.4 mm at 0.1 mm); a 2–3 mm attachment survived, residual ≈ 0 (p99.9 = 0.02 mm).
3. **Quadric on all neighbours within 4 mm with iterative rejection [failed].** h median +0.45, p99 3.4, max 4.8 mm (expected ~0 / 0.3–1 / ~1). A 4 mm radius covers several layers; the first fit includes the attachment; the 0.06 mm rejection equals noise, so the surface sinks.
4. **Ring-based approach (current).** Threshold history:
   - `MAX_TOP_H` 1.6 (from elliptical case 1) rejected true rectangular attachments of case 2 (h95 1.64, 1.77); raised to 2.2, then 2.5 (case 12 true h95 = 2.23).
   - `MIN_SEED_AREA` 2.5 dropped a true case-12 attachment (seed area 2.48) and kept a false anterior blob at (−3.3, 13.2, 22.1) (seed h max 0.80, ring_rms 0.131, 5.0 mm from kept neighbours). Now `MIN_SEED_AREA` 2.3 and `MIN_SEPARATION` 5.5.
   - Case 12 showed red patches on proximal surfaces of neighbours; the relative cut and "largest piece only" were added (effect unconfirmed).
   - A blocky h-map rendering bug was fixed by 4-neighbour inverse-distance interpolation.

Lesson: **any local smoother either swallows the attachment (large scale) or turns crown curvature into a false bump (small scale).** The surface under an attachment must be built from the narrow ring just outside its wall.

**Volume**
- Σ per-vertex surface area × h **overestimated** (11.9–17.7 mm³ vs Excel max 10.2) because surface area counts the near-vertical walls. Removed from detection; volume now uses projected area in `measure_volume.py`.
- **Two base surfaces for one attachment (fixed).** Detection fitted the ring around the pre-cleanup region while an earlier `measure_volume` refitted a ring around the smaller final mask; removed gentle foot vertices then raised the second base. Fix: one fit in detection.
- **Foot bias does not cancel in wear %.** If the missing foot volume is F, the computed wear is ΔV/(V1−F) instead of ΔV/V1, roughly 1.2× too large [estimate, not measured]. With different cuts (0.3 × h95) in T1 and T2 the foot is not even the same in both scans.
- `largest_component` chooses by surface area, not by area × h.

**Other**
- Picking points in a plain `PointCloud` shows nothing useful; give normals or use the mesh with `VisualizerWithEditing`.
- Comparing T1 and T2 with heights relative to a clicked point hides wear. Use one common frame and one common base reference.

---

## 8. Decisions

1. Rigid ICP only, no scale. Scale is consistent (mm); diagonal ratios 0.97–1.07 come from crops.
2. Attachment = protrusion above the underlying tooth surface. Measure height above a base surface, not curvature.
3. Measure each scan against its own base surface (thin ring in that scan), then compare. Registration error and tooth motion largely cancel.
4. One base fit, in detection. The last ring fit is saved as per-vertex h; nothing downstream refits.
5. Volume = Σ projected triangle area × mean h over fully-masked triangles. Ray casting or fixed-grid integration remain fallbacks.
6. Primary wear metric = mean top height above the base (resolution-insensitive). Wear % = (V1 − V2) / V1 × 100 over the same base and mask; also report mean and max height loss in mm.
7. Noise baseline: measure the same T1-vs-T2 difference on an attachment-free patch of the same tooth and subtract the constant bias (≈ +0.04 mm in case 1). Wear that does not clearly exceed it is "not measurable".
8. Alignment (goal sub-0.1 mm, optional for the current chain): trimmed ICP on a stable overlapping region; global on palatal rugae, then local per attachment on a 6–10 mm crown patch excluding the attachment. Manual picking stays as fallback/QC.
9. Parameters in mm, relative to attachment size; never relative to jaw diagonal or vertex counts.
10. `volumes.csv` columns are fixed: `id, pre_volume_mm3, post_volume_mm3` (T1 = pre, T2 = post).
11. Ids from independent detections are not comparable. Only matching (`match_attachments.py` or equivalent) defines correspondence.
12. **Open conflict (Q11):** the earlier plan was to detect on T1 only and compute V2 on the T1 mask with the T2 base surface (negative h clipped to 0). The current pipeline detects T2 independently. Decide before implementing wear %.

---

## 9. Next steps

1. The maintainer reruns case 2 with the current detector (overwrites the `T1_*` and `T2_*` files of case 2; T1 first, it needs a viewer):
   ```
   python src/detect_attachments.py --sample 2 --show-ids
   python src/detect_attachments.py --sample 2 --scan T2 --show none
   ```
   Send back: the `--show-ids` screenshot, the "Attachments in arch order" lists of both scans, and the T2 count. T2 is coarse; a count ≠ 14 means a detection miss.
2. Get the Excel row order on the arch (Q10). This fixes `--reverse`.
3. Make `match_attachments.py` import `arch_order` from `detect_attachments`, then run `python src/match_attachments.py --samples 2` (overwrites `volumes_matched.csv`).
4. Write `data/reference/excel_volumes.csv` (`case, row, pre_mm3, post_mm3, loss_pct`; 168 rows, from the appendix) and `src/compare_excel.py` with order-free checks (mean and median V1 per case vs the Excel pre values; sorted volume lists).
5. Decide Q11 and Q12, then implement wear % and the control-patch noise baseline.
6. Run detection on cases 8 and 12 (never run), then the fine/fine cases (1, 5, 6, 10), then the coarse-T1 cases 4, 11, 13, 14. Expect 14 attachments per scan.
7. Update this file after each step.

---

## 10. Open questions

- Q1: Which tables must `build_tables.py` produce? (Notes unreadable.) Proposed columns: `case, attachment id, shape, area_mm2, h95_T1, h95_T2, V1_mm3, V2_mm3, wear_pct, noise_baseline, T1_density, T2_density, qc_flag`.
- Q2: Do teeth move between T1 and T2, or is there a small scale difference? Case 1 shows a position-dependent offset (≈ 0.1 mm posterior, ≈ 0 anterior). Needs a distance heatmap on more cases.
- Q4: Are FDI tooth labels needed, or is an attachment id enough? STL files have no labels.
- Q5: Case 3: incomplete, a different patient, or recoverable?
- Q6: Which scans contain the worn attachments? The Excel gives wear per row but not how it was measured.
- Q10: Does "left to right" in the Excel match the script's #0 (smaller x, viewed from the buccal side)? Needed for matching rows to ids and for `--reverse`.
- Q11: Wear volume on the T1 mask with the T2 base (clip negatives), or independent T2 detection (current)? Affects edge wear, mixed-density cases and foot bias.
- Q12: Foot-bias correction for wear %: same mask and cut in both scans, or an estimated foot correction? Needs a decision and a measurement.
- Q14: Excel case 1 gives wear 5–19 %, hand profiles show none. Which is right?
- Q15: How was the Excel volume computed? What do the defect / excess / RMS columns of an earlier image of the case 2 sheet mean? (SBS, ARI and failure mode are irrelevant.)

---

## 11. Glossary

- **T1 / T2:** scan before / after ("pre" / "post"). Where alignment is used, T2 is the fixed frame; `T1_to_T2.npy` is a 4×4 matrix from raw T1 into the T2 frame; `T1_aligned.stl` is T1 already transformed.
- **Buccal:** cheek side of a tooth, where attachments sit. **Occlusal / incisal:** biting-edge side (−y). **Gingival:** gum side (+y).
- **Top / wall / foot:** the flat or crest surface of an attachment / its steep side / the low gentle skirt at its base (cut off by the detector, hence the volume bias).
- **Ring:** narrow band of tooth surface just outside the foot; the base reference. **Annulus:** the wider band (2.0–3.5 mm) used by stage 1 only.
- **Base surface:** estimated tooth surface under the attachment (the last ring fit in detection).
- **h:** height above the base surface in mm; positive = above tooth. **h95:** 95th percentile of h over the mask. **h_top:** mean h of mask vertices with h ≥ 0.8 × h95.
- **Seed / region / mask:** seed = working points with h ≥ `SEED_HEIGHT`; region = grown set (h ≥ `EDGE_HEIGHT`); mask = final cleaned vertex set used for volume.
- **qc_flag:** "check" if ring_rms > 0.10 mm or area > 16 mm², else "ok"; a warning, not a rejection.
- **D = T2 − T1:** height difference in one frame; negative = wear. **top / ring / top − ring** (`profile_compare.py`): mean D on the top, median D on the ring (= local registration error), and their difference.
- **fine / coarse:** mean mesh edge ≈ 0.1 mm / ≈ 0.2 mm.
- **Pick:** vertex clicked in an Open3D window (Shift+Left add, Shift+Right remove, Q/Esc finish).
- **FDI:** two-digit tooth numbering; not available in the files.

---

## 12. Environment and workflow

- Run everything from the repo root: `python src/<script>.py --sample <id>`. Git branch `main`. Python venv with `numpy open3d trimesh scikit-learn scipy rtree matplotlib`. Scripts do `from config import DATA_DIR, PROCESSED_DIR` (works because `src/` is the script directory).
- `rtree` must be installed, or `trimesh.proximity` fails.
- Harmless Open3D messages: "Couldn't open plugin directory", "No plugins found, falling back on no decorations".
- A run on a 500k-vertex mesh can take 30–60 s or more; say so.
- Workflow: a maintainer runs the command given by the agent and pastes back terminal output and screenshots; whole files are pasted when asked. Outputs are sometimes renamed by hand to avoid overwriting, so say explicitly which files a command overwrites.
- Typical window titles: "pick points WITHOUT attachments", "attachments (red)", "h map (red = above base, +-0.4 mm)". Colours: orange = T1, blue = T2, red = detected.
- Whether the latest edits (`config.py` paths, removal of version labels, the edited `detect_attachments.py`, `match_attachments.py`) are committed is unconfirmed.
- Also useful from maintainers when needed: the current content of `src/config.py`, any script before patching, and `python src/test.py --samples <ids>` output after new cases.

---

