# AGENTS.md — dental-wear-ml

Guide for AI coding agents (and humans) working on this repo. **Read it fully before changing code.**
It records what the project is, what the data looks like, what has been measured, what failed and why, and what to do next.

Status: **Phase 1 in progress.** Merged with the owner's earlier AGENTS.md draft (nothing in it was lost). Last update: after the profile/compare experiments on cases 1, 2, 4, 5, 8, 13 (T1 and T2 profiles) and after the supervisor's remark on expected wear (section 1).

## TL;DR for a new model (read this first)

- **Task:** measure wear (%) of orthodontic *attachments* (small composite bumps glued on teeth), not tooth wear, from two 3D scans of one upper jaw: T1 (before), T2 (after). Wear is expected to be small: at most about 15 %, rare (supervisor).
- **Data:** `data/<id>/T1.stl, T2.stl`, 11 active cases (1, 2, 4, 5, 6, 8, 10–14; 3 excluded; 7 and 9 do not exist). Millimetres. You usually do **not** have the STL files; the owner runs code and sends back terminal output and screenshots.
- **State:** alignment works globally (cases 1, 2 only). Attachment detection **does not work** (three failed attempts, section 5). Measurement of wear is **not implemented**. Diagnostic scripts exist (section 3).
- **Biggest facts:** (1) one scan of 6/11 cases has 4x fewer vertices; (2) global alignment error about 0.1 mm is as large as the wear; (3) attachment height is about 1–1.3 mm and a 4 mm-radius neighbourhood is not a height field; (4) measure each scan against its own thin-ring base surface (sections 4, 6).
- **Next step:** section 7 ("Immediate next steps"). Do not rewrite detection first.
- **Before asking the owner anything,** check section 11 (FAQ); most questions are already answered.

Legend: **[measured]** = seen in data/plots/terminal output. **[assumed]** = hypothesis, not verified. **[failed]** = tried and did not work. Anything not tagged is a decision or a fact from the owner.

---

## 0. Working rules

- **Language:** the owner writes Persian. Talk to the owner in Persian. Code, comments, docstrings, docs and commit messages are in English. Study/exam material is a separate matter and is not part of this repo.
- **Be concise.** The owner prefers short, direct, low-commentary answers and dislikes over-complicated solutions when a simpler documented one exists.
- **Never modify `data/<id>/T1.stl` or `T2.stl`.** They are raw inputs. All derived files go to `data/processed/<id>/`.
- **Do not paste or print full meshes.** Each STL is 10–65 MB (100k–650k vertices). Use summary statistics (see `src/test.py`).
- **Everything is in millimetres.** Define every threshold, radius and tolerance in mm, never in vertex counts or percentiles of vertices. Mesh density differs about 4x between files (section 2).
- **Ask before running GUI scripts.** `align.py`, `profile_attachment.py` and any visualisation open Open3D windows and need a human.
- **Do not guess; look at the data first.** Three detection algorithms were written from assumptions and all three failed (section 5). Before writing a new algorithm, produce a plot or number from the real data that supports its assumptions.
- **Say when code is untested.** Agents have no access to the STL files. State clearly which code was not run on real data, and give the owner the exact command and what output to send back.
- **Keep the existing script conventions:** one script per step in `src/`, `argparse` with `--sample <id>`, constants at the top of the file, paths via `PROJECT_ROOT = Path(__file__).resolve().parent.parent`, results under `data/processed/<id>/`.
- **Give full files, not patches,** when the owner will paste code (he pastes whole files). Ask for the current file before patching it; do not patch blind.
- **Be honest about uncertainty.** A wrong wear number that looks valid is worse than no number. Report quality metrics and flag suspicious cases instead of forcing a result.
- **Literature references in this file were written from memory and are unverified.** Check title and year before citing them in any report.

---

## 1. Project goal

Machine-learning dental project. **Final goal: compute the wear percentage of orthodontic attachments (not tooth wear)** by comparing two 3D scans of the same upper jaw: **T1 (before)** and **T2 (after)**.

**Phase 1 (current): geometric mesh processing, no deep learning.** Tools: Python, `open3d`, `trimesh`, `scikit-learn`. A later phase may use deep learning (e.g. MeshSegNet, Teeth3DS dataset) once Phase 1 gives labelled data and a baseline.

Original Phase 1 plan (from `docs/Phase_1.md` / `Phase_1.png`, handwritten notes):

1. Align T1 and T2: manual picking of points on regions without attachments, then automatic ICP (`open3d`).
2. Isolate attachments: curvature detection + DBSCAN (`open3d` or `scikit-learn`). **This idea was tested and does not work as written (section 5).**
3. Wear volume: computed with `trimesh`. **Needs a different method (section 4).**
4. Tables: unify and compare tables with `scikit-learn`. The notes were illegible here. Details unknown (Q1).

Note: the notes said "two points" for alignment, but `align.py` requires at least 4 (`MIN_PAIRS = 4`). Two points cannot define a rigid transform (rotation about their connecting line stays free); at least 3 non-collinear points are needed.

### Expected wear magnitude (from the supervisor)

- The supervisor says wear is **at most about 15 %, and only in very rare cases**. So most attachments probably have very little or no wear.
- The supervisor also gave the team an **Excel file**. Its content has **not** been shared with the agent yet. Ask the owner for it (or for its columns) before designing the tables or validation; do not assume what it contains. It may hold the ground truth asked for in Q6.
- Consequence: the pipeline must be able to say "no measurable wear" reliably. A noise baseline (section 4, item 8) is mandatory, not optional.

### Sensitivity budget

- Attachment volume is about **3–4 mm³**; height about 1–1.3 mm; top area about 5–8 mm².
- 10 % wear is about **0.1 mm** loss of mean height. 15 % is about 0.15 mm.
- Scan noise is about **0.03 mm** [measured on flat tops: ±0.05 mm], mesh edge length 0.1–0.2 mm. So only integrated quantities (mean height over the whole top patch, or volume) are meaningful, and wear below about 5 % is probably not detectable.

---

## 2. Data

### Layout

```
data/
  <id>/T1.stl, T2.stl          raw scans, ids: 1 2 3 4 5 6 8 10 11 12 13 14  (7 and 9 do not exist)
  processed/<id>/              derived outputs
      T1_aligned.stl           T1 transformed into the T2 frame            (cases 1, 2 only so far)
      T1_to_T2.npy             4x4 rigid transform, raw T1 -> T2 frame     (cases 1, 2 only so far)
      T1_segmented.ply         OUTDATED detection result (case 1), do not trust
      picks_T1.npy / picks_T2.npy   picks saved by profile_attachment.py
      profile_<T1|T2>_<k>.png, compare_<k>.png   diagnostic figures
  mesh_report.txt              output of src/test.py
docs/Phase_1.md, Phase_1.png   original notes
src/                           see section 3
```

Actual repo tree (as of the last `tree`; `processed/` content grows as cases are run):

```
dental-wear-ml/
├── data/
│   ├── 1 2 3 4 5 6 8 10 11 12 13 14/   each: T1.stl, T2.stl
│   └── processed/
│       ├── 1/ T1_aligned.stl, T1_segmented.ply (outdated), T1_to_T2.npy
│       └── 2/ T1_aligned.stl, T1_to_T2.npy
├── docs/ Phase_1.md, Phase_1.png
├── Readme.md
├── requirements.txt
└── src/ align.py, detect_attachments.py   (+ test.py, profile_attachment.py, profile_compare.py added later)
```

- One folder = one patient/jaw. Every file is an **upper jaw with 14 teeth**, plus palate and gingiva. STL files carry **no tooth labels**.
- **Active cases: 11** (1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14).
- Processing status: case 1 has alignment (+ an invalid segmentation); case 2 has alignment only; the rest are unprocessed (no alignment yet).
- Profile figures were made by hand for cases 1, 2, 4, 5, 8, 13 (see section 4).

### Case 3 is excluded for now

Its T1 is abnormal: bbox x from -85 to -29 (others about -34 to +34), different orientation (smallest extent on z, about 17 mm), surface area 3266 (others 4100–5300; T2 of the same case 6637), 59 mm centre offset to T2, and only about 70 % of the area and half the height (14 vs 27 mm) even after rotation. Probably a different frame plus incomplete or different data. **Skip it in all batch runs.** Revisit at the very end (Q5). If it is revisited: try a global initial alignment (PCA / FPFH+RANSAC) and check whether all 14 teeth and attachments are present before forcing any alignment.

### Measured scan properties (`src/test.py`, T1/T2 of all active cases)

- All files are **millimetres**. PCA extents about 64–70 x 52–66 x 20–28 mm. Case 14 is a smaller jaw (diagonal 85 mm vs 91–96 mm). Parameters must not depend on a fixed jaw size.
- Frame: x about -34..+34, z about -31..+31, y about 7..33. The arch lies in the x–z plane; **y is the occlusal–gingival axis, and +y points toward the gingiva/palate** [measured: confirmed from vertical sections of case 2; this closes the old Q7 for the cases seen]. Still verify in each new case before hard-coding axes.
- Every mesh is a **single connected component and NOT watertight**. Do not use `trimesh.Trimesh.volume`.
- STL triangle order and normal orientation are not guaranteed consistent. Orient normals explicitly before using their sign.
- **T2 always has a larger surface area than T1** (more gingiva and palate, different crop border), e.g. case 12: 5308 vs 6697. Raw centroid differences of 1–4 mm come from the crop, not from real motion. Whole-mesh ICP is biased by the trimmed borders.
- Raw T1/T2 bounding boxes are already close (centre offset 0.2–3 mm), but the saved rotations are **14.0° (case 1) and 9.8° (case 2)** and they are real.

### Mesh density (most important data finding)

Mean edge length about **0.10–0.12 mm** for "fine" scans (about 500–650k vertices) and about **0.20–0.23 mm** for "coarse" scans (about 120–165k vertices).

| case | T1 | T2 |
|---|---|---|
| 1, 5, 6, 10, 12 | fine | fine |
| 2, 8 | fine | **coarse** |
| 4, 11, 13, 14 | **coarse** | fine |

In 6 of 11 cases one scan has about 4x fewer vertices (probably lower export quality or decimation). Per-case numbers (vertices; mean edge mm; surface area; diagonal):

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

- Sharp edges of rectangular attachments round off on coarse meshes; this changes height and volume slightly. A T1-vs-T2 difference can be a **resolution artefact, not wear**.
- Edge **position** on a coarse mesh is unreliable (about ±0.1–0.2 mm). With an attachment perimeter of 8–9 mm and area 5–6 mm², a 0.15 mm edge error is about 15–20 % area error — as large as the wear itself. **Do not compute volume from separately thresholded edges in mixed-density cases** (2, 8, 4, 11, 13, 14).
- Attachment **top height** is reliable even on the coarse mesh [measured, case 4].
- Do not upsample coarse meshes to look finer (adds no information). Do not blindly decimate fine meshes to 0.2 mm either. Process on a common grid instead (section 4).
- Report mixed-density cases separately to expose resolution artefacts.

### Alignment quality of the existing global alignment [measured]

Nearest-point distance from aligned T1 to T2 (every 20th vertex), mm:

| case | mean | median | p90 | max |
|---|---|---|---|---|
| 1 | 0.051 | 0.026 | 0.076 | 4.57 |
| 2 | 0.097 | 0.050 | 0.186 | 4.25 |

Median is at the noise level; the max comes from regions present in one scan only (cut gingiva/palate). Case 2 is worse (coarse T2, or real tooth movement; cannot tell). Caveats: most of the surface is palate/gingiva so these numbers mostly describe that area; ICP minimises almost this exact metric so they are optimistic; case 2 p90 0.19 mm is comparable to the wear being measured.

**Local registration error is larger and position-dependent** [measured, case 1; see section 4]: posterior teeth are offset about +0.08 to +0.10 mm in the buccal direction, anterior teeth about 0. **Global alignment alone is not accurate enough for wear measurement.**

---

## 3. Code inventory

| file | status |
|---|---|
| `src/align.py` | step 1, working, to be improved |
| `src/detect_attachments.py` | step 2, **does not work** (three failed versions, section 5) |
| `src/test.py` | inspection script, working (needs `rtree`) |
| `src/profile_attachment.py` | diagnostic: user-picked centre -> local frame -> vertical/horizontal sections + height map; works |
| `src/profile_compare.py` | diagnostic: overlays aligned T1 on T2 in one frame; works |
| `src/dist_heatmap.py` | proposed (colour distance of aligned T1 to T2, red at 0.3 mm). Not confirmed to exist in repo |
| `requirements.txt` | should contain `numpy open3d trimesh scikit-learn rtree matplotlib`; add `scipy` when needed. Check it matches |

### `src/align.py`
- Loads T1, T2 (`open3d`), asks the user to pick **>= 4 corresponding points** (Shift+Click, same order, on non-attachment areas) in `VisualizerWithEditing`, shows a check view. Window title: "pick points WITHOUT attachments".
- `kabsch()` gives the initial rigid transform (rejects collinear/coplanar picks).
- Samples 100k points per mesh, estimates normals (radius 0.01 x diagonal), orients them outward, then **two-stage point-to-plane ICP** (max correspondence 0.05 x diagonal coarse, 0.02 x diagonal fine, about 4.8 mm and 1.9 mm).
- Warns if pick RMSE > 0.02 x diagonal or fitness < 0.3. Saves `T1_to_T2.npy` and `T1_aligned.stl`; shows an orange (T1) vs blue (T2) preview. T2 is the fixed reference.
- Weaknesses: manual picking (not reproducible, a bottleneck); ICP over the whole mesh including gingiva/palate and crop borders (no trimming); distances scaled by diagonal; 1.9 mm fine threshold is too loose for sub-0.1 mm goals; no tooth-level alignment; quality metrics are printed but not saved.
- It sets `pcd.normals = mesh.vertex_normals`; that is why its view is shaded. A bare `PointCloud` without normals renders flat and unreadable in Open3D.

### `src/profile_attachment.py` (diagnostic)
- Usage: `python src/profile_attachment.py --sample <id> --file T1.stl` (or `T2.stl`). `--file` sets the mesh and the output prefix (`profile_T1_<k>.png`, `profile_T2_<k>.png`); picks are saved to `picks_<tag>.npy`. `--load-picks T1 --transform` places the same picks on the other scan without re-clicking. `--slab` (slice thickness, e.g. 0.2 for coarse meshes) and `--grid` (height-map cell; use 0.3 on coarse meshes, 0.15 on fine) exist.
- Click the centre of an attachment from the buccal side (look at the model from the cheek side first). The script builds a local frame (normal outward from the tooth, occlusal-gingival axis, arch axis), then saves: vertical section, horizontal section, height map (0.15 mm cells, contours every 0.25 mm). Heights are relative to the picked point.
- Output names overwrite earlier files with the same prefix; copy the folder first. Old `profile_<k>.png` files were renamed by hand to `profile_T1_<k>.png` / `profile_T2_<k>.png`.
- Only for observation. It detects nothing and the picking step does **not** belong in the final pipeline.

### `src/profile_compare.py` (diagnostic)
- Usage: `python src/profile_compare.py --sample <id>`. Needs `T1_aligned.stl`, `T1_to_T2.npy`, `picks_T1.npy`. Builds **one** local frame from T2 only, draws T1 (blue) and T2 (red) in the same axes, plus a height-difference map `D = T2 - T1` (negative = T2 lower = wear). Prints, per pick, `snap` distance, `top: mean D` (r < 1 mm on the attachment top) and `ring: median D` (nearby tooth surface = local registration error).
- Case 1 was run (about 43 s); case 5 has no alignment yet, so its compare could not be run.

### `src/detect_attachments.py` (broken)
History in section 5. Whatever version is in the repo, **do not tune it**; replace it with the design in section 6.

### `src/test.py`
Prints size, unit guess, PCA extents, density, watertightness, T1/T2 consistency, saved-alignment angle/translation, and distance after alignment. Writes `data/mesh_report.txt`. Requires `rtree` (`pip install rtree`). Usage: `python src/test.py [--samples 1 2 8]`.

---

## 4. What the data says about the attachments [measured]

Two shapes occur, sometimes in different cases:

- **Elliptical:** smooth dome, no flat plateau, no steep wall. Seen in cases 1 and 5.
- **Rectangular (rounded corners):** flat top, near-vertical walls. Seen in cases 2 and 8; flat tops also in case 4 (coarse).

In the scans the palate/gingiva is the large flat region at high y; the teeth hang below it (toward low y). Attachments are on the outer (buccal) side of the arch. Some cases may mix or look different from the two labels; do not assume the shape from the case id. Attachments sit on the **buccal surface of the crown**, about one per tooth (at most about 14), centres of neighbouring teeth about **7 mm** apart.

### Geometry (±0.2 mm, from sections and contour maps)

| quantity | rectangular (cases 8, 2) | elliptical (case 5; case 1 similar) |
|---|---|---|
| height above tooth surface | about 1–1.3 mm | about 1.1–1.5 mm |
| top size | about 2.5–2.7 mm along arch x 2 mm occlusal-gingival (case 2: about 3 x 2) | dome about 2 x 3.5 mm at half height; width about 2.4 mm at -1 mm |
| top area | about 5–8 mm² | – |
| top flatness (along arch) | ±0.05 mm | not flat |
| walls | about 1 mm drop within 0.2–0.3 mm, near vertical, on both arch sides and the gingival side | gentle slope: about 1 mm drop over about 1–1.5 mm |
| occlusal side | **no wall**; top ramps up gently over about 2 mm (a wedge/bevel) | – |

- **Top is tilted:** in the vertical section the top rises about **0.5–0.7 mm** from the occlusal side to the gingival side. The slope is nearly the same for three picks whose local normals tilted 9°, 37° and 57°, so it is real design (a bevel), not a frame error [probable; not fully confirmed]. Case 4 shows less of a bevel (a short shoulder on the occlusal side) but a coarse mesh cannot tell whether the design differs or the slope was lost to sampling.
- **Do not define the frame normal from the attachment top** (it differs about 15° from the tooth surface). Heights must be measured relative to a **tooth base surface**, not relative to the top plane.
- Size/orientation differ between cases (case 4 looked about 1.5–2 mm along the arch and about 3 mm occlusal-gingival). Detection must use size **ranges** (about 1.5–4 mm), not fixed size or orientation.

### Surroundings (what limits the base surface)

- Gingival margin: about 3.5–5 mm on the +y side of the attachment. Incisal/occlusal edge: about 3–4 mm on the other side. Interdental groove: 2.5–3.5 mm deep at about ±4 mm along the arch. Some attachments (case 10) show an undercut at the gingival margin (curve folds back at about 5 mm).
- **Clean tooth surface exists only in a narrow strip:** about 0.3–1.5 mm outside the wall base (about 2 mm toward occlusal/gingival, 0.5–1 mm along the arch). Along the arch the surface falls at about 35–45° toward the groove for 3–3.5 mm. A 4 mm-radius neighbourhood therefore contains several surface layers (crown, groove, gingiva, neighbour tooth) and is not a height graph.

### T1 vs T2 results so far

**Case 1** (4 elliptical attachments, both scans fine), heights in mm:

| pick | position | normal tilt | ring (registration error) | top | top − ring |
|---|---|---|---|---|---|
| 1 | posterior (x=+27, z=-13) | 40° | +0.102 | +0.144 | +0.042 |
| 2 | anterior (x=11, z=23) | 18° | -0.005 | +0.024 | +0.029 |
| 3 | anterior (x=-16, z=23) | 18° | -0.013 | +0.030 | +0.043 |
| 4 | posterior (x=-25, z=-14) | 41° | +0.077 | +0.114 | +0.037 |

- No visible wear: sections overlap within about 0.05 mm; shape, width, height equal in T1 and T2. Wear larger than about 0.05 mm (about 4 % of height) is not present in this case.
- Posterior teeth are shifted about +0.08..+0.10 mm buccally in both left and right sides, anterior about 0: consistent with arch widening (real tooth movement) or a small scale difference. Cannot tell which. Either way a global fit leaves about 100 µm local error, **larger than the 0.1 mm wear for 10 %**. Therefore a local reference is required.
- `top − ring` is stable (spread about 0.007 mm) so referencing to the neighbouring surface cancels registration error well. A constant bias of about **+0.04 mm** remains: it is not real growth; probably the ring includes slopes/grooves, or T1/T2 sampling differs. It must be measured on a control patch and subtracted. This also gives the noise floor.
- Wall points in the difference map are noisy (±0.3 mm, red/blue speckle): tiny lateral shifts on steep slopes. Exclude edges from the measurement; use only the top (r < 1 mm).

**Case 5** (4 elliptical attachments, #14, #10, #5, #3): T1 and T2 domes look the same; differences 0.05–0.25 mm have the sign opposite to wear and match pick/frame error. That comparison was **weak**: the height reference was the clicked point itself (which also drops if the crest wears), T1/T2 picks differ by tenths of a mm, frame tilt differed (39 vs 43°, 30 vs 32°), curves were not on one plot. Only "large wear (several tenths mm) not seen" can be said. **Case 5 has no alignment yet**; run `align.py --sample 5` and use `profile_compare.py`.

**Cases 2, 4, 8, 13**: T1 profiles only (13 and 4 for geometry, 8 and 2 for rectangular shape); no wear conclusions.

---

## 5. What failed, and why (do not repeat)

All three detection attempts were written without having looked at real attachment sizes, and all failed on real data.

1. **v1 — surface variation + DBSCAN [failed].** Per-vertex surface variation (Pauly 2003; smallest-eigenvalue ratio, k=30), top 8 % (percentile 92), DBSCAN (eps = 0.01 x diagonal, min_samples 20), keep 150–20000 vertices. Result: no attachment red; red = thin lines on the tooth–gingiva border and interdental grooves. Reasons: (a) surface variation finds creases/edges, not protrusions; flat tops and smooth domes score low, gingival margins and fissures highest; (b) a percentile threshold always selects 8 % whether or not attachments exist; (c) k-NN scale depends on mesh density, vertex-count thresholds do not transfer between fine and coarse meshes; (d) whole mesh searched, not only the buccal crown; (e) per-vertex Python loop is slow. This is what the original plan "curvature + DBSCAN" produced.
2. **v2 — ROI (crown/radial percentile, normal filter) + residual to a Taubin-smoothed surface, seed/grow thresholds, DBSCAN [failed].** CLI options included `--seed-height`, `--grow-height`, `--grow-radius`, `--crown-percentile`, `--radial-percentile`, `--min-normal-dot`, `--dbscan-eps`, `--dbscan-min-samples`, `--smooth-iters`. Reason: **Taubin smoothing is a low-pass filter that deliberately preserves shapes larger than about 14 x edge length (about 1.4 mm on a 0.1 mm mesh), independent of iteration count.** A 2–3 mm attachment survived smoothing, so residual height was about 0 (p99.9 = 0.02 mm). The formula "iterations = (1/edge)²" used to reach 1 mm scale was wrong. Flipped normals were not the cause. Thresholds were not the cause either.
3. **v3 — local quadratic surface fit on all neighbours within r = 4 mm (voxel 0.3 mm cloud), iterative removal of points above the surface [failed].** Result: `h` median +0.45 mm, p99 3.4 mm, max 4.8 mm (expected about 0 / 0.3–1 / about 1); crown surfaces red, attachment interior white. Reasons: (a) a 4 mm radius (8 mm diameter, about a tooth width) covers crown + groove + gingiva + neighbour, i.e. several layers, which a single-height quadratic cannot model; (b) the first fit uses all points, so the attachment pulls the fit up from the start; (c) the 0.06 mm rejection threshold equals scan noise, so every pass removes the upper half of normal points and the surface sinks; (d) attachment width 3–4 mm vs 4 mm fit radius means the fit sits on top of the attachment (the white interior). Raising or lowering `HEIGHT_THRESHOLD` cannot help because `h` itself is wrong.

General lesson: **any local smoother either swallows the attachment (large scale) or turns crown curvature into a false bump (small scale).** The surface under an attachment must be built from the narrow ring just outside its wall, not from a big neighbourhood.

Other lessons:
- Picking points in a plain `PointCloud` shows nothing useful; give the cloud normals (or use the mesh with `VisualizerWithEditing`).
- Comparing T1 and T2 with heights relative to a clicked point hides wear. Use one common frame and one common base reference.

---

## 6. Key technical decisions

1. **Rigid ICP only, no scale.** Scale is consistent (mm); diagonal ratios 0.97–1.07 come from different crops. (A small scale difference between scans is nevertheless one explanation for the case 1 buccal offset; check before ruling it out.)
2. **Use Trimmed ICP** on a stable overlapping region (crop differences).
3. **Two-level alignment** (goal is sub-0.1 mm):
   - Global: on stable anatomy (palatal rugae region), initialised from the existing result or PCA/FPFH+RANSAC. Manual picking can stay as optional fallback and QC. Needed for the 9 unaligned cases. Raw boxes are already near each other (except case 3), so full FPFH from scratch is usually unnecessary.
   - Local: per tooth/attachment ICP on a crown-surface patch (about 6–10 mm radius around the attachment, **excluding the attachment itself**). Removes real tooth movement and global error.
4. **Attachment = protrusion above the underlying tooth surface.** Measure height above a base surface, not curvature.
5. **Measure each scan against its own base surface** (the thin ring around the attachment in that scan), then compare heights. Registration error and tooth motion largely cancel (`top − ring` was stable to 0.007 mm in case 1); alignment then only needs to carry the attachment region T1 → T2 to about 0.5 mm, not 0.05 mm. Keep the local-ICP method as a cross-check: if both agree, the number is trustworthy.
6. **Primary wear metric = mean height of the flat top above the base surface**, which is insensitive to resolution. Volume and wear % are the second step, computed on a **common mask (T1 ∩ T2)**, not on separately thresholded edges. Exclude edges/walls.
7. **Volume by height integration on a fixed grid** (e.g. 0.05–0.1 mm cells), not `mesh.volume` (open meshes; fine vs coarse must agree). Ray casting is the fallback.
8. **Detect attachments on T1 only** (T1 = before, intact), then reuse region and base surface to measure T2. No independent detection on T2.
9. **Wear % = (V1 − V2) / V1 × 100**, V1, V2 over the same base and mask; also report mean and max height loss in mm.
10. **Noise baseline:** measure the same T1-vs-T2 height/volume difference on a same-tooth patch without attachment and subtract the constant bias (about +0.04 mm in case 1). Wear that does not clearly exceed this baseline is reported as "not measurable".
11. **Mixed-density cases** (2, 8, 4, 11, 13, 14) are reported separately (section 2).
12. Parameters in mm, relative to attachment size, never relative to jaw diagonal or vertex counts. Do not fix size/orientation of attachments (ranges only).

---

## 7. Planned pipeline (next work)

Values are starting points, to be tuned on cases 1, 2, 8 (and 4/13 for coarse). All in mm.

1. **Preprocess:** clean mesh (duplicates, degenerate faces), keep native resolution for measurement. A resampled uniform copy (about 0.1–0.15 mm) may be used for detection only.
2. **Search region:** buccal crown surface only. PCA for the occlusal axis; keep vertices whose normals face outward from the arch centre; use the gingival-margin / incisal-edge window (attachments sit about 3.5–5 mm below the margin side, 3–4 mm above the edge). `+y` = gingiva side. Removes palate and lingual surfaces.
3. **Detection (new, not yet written or tested):** band-pass on surface height at scale about 1–2 mm (difference of two smoothings, sigma about 0.7 mm and about 2.5 mm): attachments (radius 1–2 mm) stand out, crown curvature (radius 6–8 mm) is removed. This is deliberately different from v2 (smoothing scale was too small there). Alternatives/complements: wall-based cue (normals deviating > 45° from the smoothed normal on a 0.5 mm scale, forming a U or closed ring 2–4 mm wide with a nearly flat inside) — gingival margin and grooves are long and open, so size/shape filters remove them. The occlusal side of rectangular attachments is open (U-shape, wedge), so do not require a closed ring. DBSCAN with `eps` in mm; filter by area (mm²), height (about 0.5–1.5 mm), compactness. Expect at most 14. Boundary from the sign change of curvature (convex top → concave foot) or half height; boundary uncertainty about ±0.3 mm for ellipses, ±0.3–0.5 mm on the open occlusal side.
4. **Base surface:** from a narrow ring (about 0.3–1.5 mm outside the wall foot; wider toward occlusal/gingival, narrow along the arch), fit a low-order surface (quadratic / thin-plate) using only ring points with similar normals; reject gingiva and neighbours. Sanity check: height of attachment must come out about 1–1.5 mm, never 3–4 mm.
5. **Alignment:** automatic initialisation + Trimmed ICP (global), then local ICP per attachment (6–10 mm patch, excluding the attachment). Save fitness/RMSE per case.
6. **Measurement:** mean top height above base for T1 and T2 (primary), volume on the common mask, wear %, noise baseline on a control patch.
7. **Tables:** one row per attachment. Proposed columns (confirm with Q1 and the supervisor's Excel): `case, attachment_id, tooth_label (if known), shape (elliptical/rectangular), area_mm2, height_T1, height_T2, V1_mm3, V2_mm3, wear_pct, noise_baseline, T1_density, T2_density, align_rmse, qc_flag`. Unify/compare across cases with `pandas`/`scikit-learn`.
8. **Validation:** overlays for every case (detected region on T1, T1-vs-T2 heatmap), distance heatmap of aligned T1 vs T2 to separate tooth movement (all tooth red, palate blue) from wear (only attachments red) from resolution artefact (everything noisy). Compare with the supervisor's Excel / any clinician rating if available.

Suggested scripts (same conventions): `src/dist_heatmap.py`, `src/align_auto.py`, `src/detect_attachments.py` (rewritten, step 3 above), `src/measure_wear.py`, `src/build_tables.py`.

**Immediate next steps, in order:**
1. Get the supervisor's Excel (Q6) and the Phase 1 "tables" definition (Q1).
2. Run `align.py` on the remaining cases (start with fine/fine cases 6, 10, 12; case 5 needed for the compare), then `profile_compare.py` / heatmap to see local registration error.
3. Write local ring-based base surface + top-height measurement, test on cases 1, 5 with the picked centres (a manual centre is acceptable for this measurement-validation step only).
4. Only then write automatic detection (step 3 above), validate on 1, 2, 8, then 4/13.

---

## 8. Open questions

- **Q1:** What do the Phase 1 "tables" contain (rows, columns, comparison)? Notes were unreadable.
- **Q2:** Do teeth really move between T1 and T2 (aligner treatment)? Case 1 shows a position-dependent offset (about 0.1 mm posterior, about 0 anterior) consistent with arch widening or a small scale difference. Needs the distance heatmap on more cases.
- **Q3:** Exact attachment size/thickness by type: refine from successful detections (current numbers in section 4 are ±0.2 mm from plots).
- **Q4:** Are tooth labels (FDI numbering) needed, or is `attachment_id` enough? STLs have no labels.
- **Q5:** Case 3: incomplete, different patient, or recoverable? Decide at the end.
- **Q6:** Ground truth for wear? The supervisor's Excel may answer this; its content is not yet known to the agent. Which cases have worn attachments? (Supervisor: wear at most about 15 %, rare.)
- **Q7:** Closed for the cases seen: y is the occlusal–gingival axis, +y toward gingiva/palate. Re-verify on new cases before relying on fixed axes.
- **Q8:** Which shape does each case have? Known: 1 and 5 elliptical; 2 and 8 rectangular; 4 flat-top but coarse. Unknown for 6, 10, 11, 12, 13, 14.
- **Q9:** Is the case 1 buccal offset real tooth movement or a small scale difference between scans?

---

## 9. Glossary and conventions

- **T1 / T2:** scan before / after. **T2 is the fixed reference frame**; `T1_to_T2.npy` is a 4x4 matrix from raw T1 into the T2 frame; `T1_aligned.stl` is T1 already transformed.
- **Buccal:** outer (cheek) side of a tooth, where attachments sit. **Occlusal / incisal:** biting edge side. **Gingival:** gum side. In the data, +y = gingiva/palate, -y = biting edge.
- **Attachment top:** the flat (rectangular) or crest (elliptical) surface of the attachment. **Wall / foot:** the steep side and its base.
- **Ring:** the narrow band of tooth surface just outside an attachment's foot, used as the base surface reference.
- **Base surface:** estimated tooth surface that would exist under the attachment.
- **h / d / residual:** height of a point above the base surface (mm). Positive = above tooth.
- **D = T2 - T1** height difference in the same frame; negative = T2 lower = wear.
- **top / ring / top - ring** (printed by `profile_compare.py`): mean D on the top (r < 1 mm), median D on the ring (= local registration error), and their difference (= wear estimate, before subtracting the control bias).
- **snap:** distance (mm) between the clicked point and the nearest mesh vertex.
- **tilt:** angle between the locally estimated normal and the radial direction of the arch.
- **fine / coarse:** mean edge about 0.1 mm / about 0.2 mm (section 2).
- **Pick:** a vertex clicked by the owner in an Open3D window (Shift+Left Click add, Shift+Right Click remove, Q/Esc finish).
- **FDI:** two-digit tooth numbering; not available in the files.
- **Wear %:** (V1 - V2) / V1 x 100 over the same base surface and mask.
- File naming: `profile_<T1|T2>_<k>.png` k = pick order; `picks_<T1|T2>.npy`; `compare_<k>.png`.

## 10. Environment and how the owner works

- Linux, project at `~/Desktop/dev/dental-wear-ml`, git repo on branch `main`, run from the repo root (`python src/<script>.py --sample <id>`). Python venv with `numpy open3d trimesh scikit-learn rtree matplotlib`.
- Harmless Open3D messages: "Couldn't open plugin directory", "No plugins found, falling back on no decorations".
- `rtree` must be installed or `trimesh.proximity` fails (it already caused a missing result once).
- The owner works by running the command you give him and pasting the **terminal output and screenshots** back; he pastes **whole files** when asked. He sometimes renames output files by hand (for example `profile_<k>.png` to `profile_T2_<k>.png`) to avoid overwriting; say explicitly which files a command will overwrite.
- A single run on a 500k-vertex mesh can take 30–60 s or more; say so and keep the number of points small (voxel 0.3 mm) when possible.
- Screenshots usually come as Open3D windows: titles like "pick points WITHOUT attachments", "Pick the CENTRE of attachments (Shift+Left Click, Q to finish)", orange = T1, blue = T2, red = detected.
- Reply format that works for him: short Persian text, numbers in mm, one table at most, then exactly what to run and what to send back. Say what you saw, what you are unsure about, and ask at most one question.
- When the owner reports "the output is wrong", assume he is right, look at the plot/numbers first and find the false assumption (see section 5) instead of changing thresholds.

## 11. FAQ — questions a future model tends to ask (already answered)

- **Which scan is the reference?** T2. T1 is moved onto T2.
- **Are units mm?** Yes, everywhere.
- **Are there tooth labels or segmentation masks?** No. Only raw STL. Ground truth for wear is unknown (the supervisor's Excel was mentioned but not yet seen).
- **Is the mesh watertight / can I use `mesh.volume`?** No and no (section 2).
- **Can I use the same vertex count or thresholds in vertices?** No; fine and coarse meshes differ 4x. Use mm.
- **Which axis is up?** y; +y toward gingiva/palate (verified on the cases profiled). Check again on new cases.
- **Why not curvature/surface variation?** Tested, finds gingival margins and grooves (section 5).
- **Why not smooth the mesh and subtract?** Tested twice; smoothing either keeps the attachment or turns crown curvature into false bumps (section 5).
- **Why four alignment picks and not two?** Two points leave a rotation free; code requires at least 4.
- **Why is case 3 skipped?** Section 2. Do not include it in batch runs.
- **Can the pipeline ask the user to click attachment centres?** Only for diagnostic/validation scripts. The owner objected to manual centres in the final detector; detection must be automatic.
- **How large is an attachment?** Section 4. Height about 1–1.3 mm, top about 2.5–3 x 2 mm, area about 5–8 mm², volume about 3–4 mm³.
- **How big is the wear I am looking for?** At most about 15 % of volume, rare; 10 % is about 0.1 mm of mean height, similar to scan noise and to the local registration error. Many cases may show none.
- **Is real tooth movement between T1 and T2 expected?** Possibly (aligner/orthodontic treatment); case 1 shows about 0.1 mm position-dependent offset. Unresolved (Q2, Q9).
- **Should T2 be detected separately?** No. Detect on T1, reuse region and base.
- **What language for code and docs?** English. Talk to the owner in Persian.
- **May I run GUI scripts?** Ask first; they need a human.
- **Which cases to start with?** 1 (fine/fine, 4 elliptical, results exist), 2 (rectangular, fine/coarse), 8 (rectangular, fine/coarse); then 4 or 13 (coarse T1). Fine/fine cases 6, 10, 12 need alignment first.
- **What to do if a result is suspicious?** Report it with its quality metrics and flag it; never force a number (section 0).

## 12. Things to collect from the owner when needed

- The supervisor's Excel (or its column list) and the definition of the Phase 1 "tables" (Q1, Q6).
- Which cases have worn attachments (Q6) and the attachment shape per case (Q8).
- The current content of any script before patching it (he pastes whole files).
- Output of `python src/test.py --samples <ids>` after new cases are added.

## 13. Reference list (unverified, from memory)

- Pauly, Keiser, Gross. *Multi-scale Feature Extraction on Point-Sampled Surfaces.* Eurographics 2003 (surface variation).
- Lee, Varshney, Jacobs. *Mesh Saliency.* SIGGRAPH 2005.
- Besl & McKay 1992; Rusinkiewicz & Levoy 2001; Chetverikov et al. 2002 (ICP and Trimmed ICP).
- Rusu et al. 2009 (FPFH).
- Choi et al. *Accuracy and reliability of palatal superimposition of 3D digital models.* Angle Orthodontist, 2010.
- Tooth segmentation by curvature: Kondo 2004; Zhao 2006; Kronfeld 2010.
- Taubin 1995 (mesh smoothing; low-pass behaviour, relevant to failure v2).
- Future deep learning: MeshSegNet (Lian et al. 2020), Teeth3DS dataset.