# AGENTS.md — dental-wear-ml

Guide for AI coding agents (and humans) working on this repo. Read it fully before changing code.
Last updated: Phase 1 in progress.

## 0. Working rules

- **Language:** the owner writes Persian. Talk to the owner in Persian. Code, comments, docstrings, docs and commit messages are in English.
- **Never modify `data/<id>/T1.stl` or `T2.stl`.** They are raw inputs. All derived files go to `data/processed/<id>/`.
- **Do not paste or print full meshes.** Each STL is 10–65 MB (100k–650k vertices). Use summary statistics (see `src/test.py`).
- **Everything is in millimetres.** Define every threshold, radius and tolerance in mm, never in vertex counts or percentiles of vertices. Mesh density differs 4x between files (section 3).
- **Ask before running GUI scripts.** `align.py` and any visualisation open Open3D windows and need a human.
- **Keep the existing script conventions:** one script per step in `src/`, `argparse` with `--sample <id>`, constants at the top of the file, paths via `PROJECT_ROOT = Path(__file__).resolve().parent.parent`, results saved under `data/processed/<id>/`.
- **Be honest about uncertainty.** A wrong wear number that looks valid is worse than no number. Report quality metrics and flag suspicious cases instead of forcing a result.
- **Literature references in this file were written from memory and are unverified.** Check title and year before citing them in any report.

## 1. Project goal

Machine-learning dental project. **Final goal: compute the wear percentage of orthodontic attachments (not tooth wear)** by comparing two 3D scans of the same upper jaw: T1 (before) and T2 (after).

**Phase 1 (current): geometric mesh processing, no deep learning.** Tools: Python, `open3d`, `trimesh`, `scikit-learn`. A later phase may use deep learning (e.g. MeshSegNet, Teeth3DS dataset) once Phase 1 gives labelled data and a baseline.

Original Phase 1 plan (from `docs/Phase_1.md` / `Phase_1.png`, handwritten notes):
1. Align T1 and T2: manual picking of points on regions without attachments, then automatic ICP (open3d).
2. Isolate attachments: curvature detection + DBSCAN (open3d or scikit-learn).
3. Wear volume: computed with trimesh.
4. Tables: unify and compare tables with scikit-learn. **The notes were illegible here. Details are unknown (open question Q1).**

Note: the notes said "two points", but `align.py` requires at least 4. Two points cannot define a rigid transform (rotation about their connecting line stays free), and at least 3 non-collinear points are needed.

## 2. Data

### Layout
```
data/
  <id>/T1.stl, T2.stl          raw scans, ids: 1 2 3 4 5 6 8 10 11 12 13 14  (7 and 9 do not exist)
  processed/<id>/              derived outputs
      T1_aligned.stl           T1 transformed into the T2 frame
      T1_to_T2.npy             4x4 rigid transform (T1 -> T2)
      T1_segmented.ply         attachment detection result (red = attachment), OUTDATED, see section 5
  mesh_report.txt              output of src/test.py
docs/Phase_1.md, Phase_1.png   original notes
src/align.py, detect_attachments.py, test.py
```

- One folder = one patient/jaw. Every file is an **upper jaw with 14 teeth**, plus palate and gingiva. STL files carry no tooth labels.
- **Active cases: 11** (1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14).
- **Case 3 is excluded for now.** Its T1 is abnormal: bbox x from -85 to -29 (others about -34 to +34), different orientation (smallest extent on z, about 17 mm), surface area 3266 (others 4100–5300), 59 mm centre offset to T2. Probably a different frame plus incomplete or different data. Revisit at the very end. Skip it in all batch runs.
- Processing status: case 1 has alignment + an (invalid) segmentation. Case 2 has alignment only. The rest are unprocessed.

### Attachments
Two shapes exist, sometimes in different cases:
- **Elliptical:** smooth dome-like bump, mild curvature.
- **Rectangular:** nearly flat top with sharp edges and corners, block-like.

They sit on the **buccal (outer) surface of the tooth crown**, typically one per tooth, so at most about 14 per jaw. In the scans, the palate/gingiva is the large flat region on top; the teeth hang below it.
Assumed physical size (unverified, measure it from data): a few mm long and wide, roughly 0.5–1.5 mm high. Expected wear is on the order of tenths of a millimetre.

### Measured scan properties 
- All files are in **millimetres**. PCA extents about 64–70 x 52–66 x 20–28 mm (case 14 is smaller, diagonal 85 vs 91–96 mm).
- Frame: x spans about -34..+34, z about -31..+31, y about 7..33. The smallest extent is along **y**, which is therefore probably the occlusal–gingival axis (verify before relying on it). The arch lies in the x–z plane.
- Every mesh is a **single connected component and NOT watertight**. Do not use `trimesh.Trimesh.volume` for attachment volume.
- **Mesh density is heterogeneous.** Mean edge length is about 0.10–0.12 mm for "fine" scans (about 500k vertices) and about 0.20–0.23 mm for "coarse" scans (about 120–165k vertices):

| case | T1 | T2 |
|---|---|---|
| 1, 5, 6, 10, 12 | fine | fine |
| 2, 8 | fine | **coarse** |
| 4, 11, 13, 14 | **coarse** | fine |

  Scanner noise and edge length are about 0.05–0.1 mm, the same order as expected wear. Single-vertex measurements are meaningless. Only integrated quantities (volume or mean height over the whole attachment patch) are trustworthy.
- **T2 always has a larger surface area than T1** (more gingiva and palate, different crop boundary). Raw centroid differences of 1–4 mm come from this, not from real motion. Whole-mesh ICP is biased by the trimmed borders.
- Raw T1/T2 bounding boxes are already close in all active cases (centre offset 0.2–3 mm), but the saved rotations are 14.0° (case 1) and 9.8° (case 2), and they are real: alignment quality is good (below).

### Current alignment quality (aligned T1 -> T2, nearest-point distance, every 20th vertex)
| case | mean | median | p90 | max |
|---|---|---|---|---|
| 1 | 0.051 | 0.026 | 0.076 | 4.57 |
| 2 | 0.097 | 0.050 | 0.186 | 4.25 |

Median is at the noise level. The max comes from regions present in only one scan. Case 2 is worse; it may be because T2 is coarse, or because teeth really moved. Caveats: most of the surface is palate/gingiva, so these numbers mostly reflect that area; ICP minimises almost this exact metric, so they are optimistic; the p90 of 0.19 mm for case 2 is comparable to the wear we want to measure. **Global alignment alone is not accurate enough for wear measurement.**

## 3. Existing code

### `src/align.py` (step 1, working but to be improved)
- Loads T1, T2 (`open3d`), asks the user to pick **>= 4 corresponding points** (Shift+Click, in the same order, on non-attachment areas) in `VisualizerWithEditing`, shows a check view.
- `kabsch()` gives the initial rigid transform (rejects collinear/coplanar picks).
- Samples 100k points per mesh, estimates normals (radius 0.01 x diagonal), orients them consistently/outward, then does **two-stage point-to-plane ICP** (max correspondence distance 0.05 x diagonal for coarse, 0.02 x diagonal for fine, about 4.8 mm and 1.9 mm).
- Warns if pick RMSE > 0.02 x diagonal or fitness < 0.3. Saves `T1_to_T2.npy` and `T1_aligned.stl`, shows an orange (T1) vs blue (T2) preview.
- Weaknesses: manual picking (not reproducible, a bottleneck); ICP over the whole mesh including gingiva/palate and crop borders (no trimming); distances scaled by diagonal; 1.9 mm fine threshold is loose for sub-0.1 mm goals; no tooth-level alignment, so real tooth movement between scans is not handled; the quality metrics printed are not saved.

### `src/detect_attachments.py` (step 2, **does not work; to be rewritten**)
Computes per-vertex surface variation (Pauly 2003; smallest-eigenvalue ratio over k=30 neighbours), takes the top 8% (percentile 92), clusters with DBSCAN (eps = 0.01 x diagonal, min_samples 20), keeps clusters of 150–20000 vertices, writes `T1_segmented.ply`. It takes no user input.

Why it fails (confirmed by screenshots: no attachment is red, red marks are thin lines on the tooth–gingiva border and interdental grooves):
1. Surface variation finds edges/creases, not protrusions. Flat tops (rectangular) and smooth domes (elliptical) give low values. Gingival margins and fissures give the highest values.
2. A percentile threshold always selects 8% of vertices, whether or not attachments exist.
3. k-nearest neighbours gives a scale that depends on mesh density, and vertex-count thresholds do not transfer between fine and coarse meshes.
4. The whole mesh is searched, not just the buccal crown surface.
5. The per-vertex Python loop is slow.
Do not tune this script. Replace it (section 5).

### `requirements.txt`
Should contain at least: `numpy`, `open3d`, `trimesh`, `scikit-learn`, `rtree`. Add `scipy` when needed. (Check that the file matches.)

## 4. Key technical decisions (agreed or strongly indicated by the data)

1. **Rigid ICP only, no scale.** The scale is consistent (mm) and diagonal ratios of 0.97–1.07 come from different crops.
2. **Use Trimmed ICP**, restricted to a stable overlapping region, because of the crop differences.
3. **Two-level alignment** (needed because the goal is sub-0.1 mm accuracy):
   - Global: on stable anatomy (palatal rugae region), initialised from the current result or from PCA/FPFH+RANSAC. Manual picking can stay as an optional fallback and QC, not as the main path. This matters for the 9 unprocessed cases.
   - Local: per tooth/attachment ICP on a crown-surface patch (about 6–10 mm radius around the attachment, **excluding the attachment itself**). This removes real tooth movement and global misalignment before measuring.
4. **Attachment = protrusion above the underlying tooth surface**, so measure height above a base surface, not curvature.
5. **Volume by height integration on a fixed grid, not `mesh.volume`** (meshes are open, and fine vs coarse meshes must give consistent results). Use ray casting at, for example, 0.05–0.1 mm grid spacing over the base surface.
6. **Detect attachments on T1 only** (T1 = before, with intact attachments), then reuse the same region and base surface to measure T2. No independent detection on T2.
7. **Wear % = (V1 − V2) / V1 x 100**, where V1 and V2 are volumes above the same base surface. Also report mean/max height loss in mm.
8. **Noise baseline:** measure the same T1-vs-T2 volume/height difference on a patch of the same tooth without an attachment. Wear that does not clearly exceed this baseline is not reported as wear.
9. **Track mesh-density mismatch** per case (fine/coarse combination in section 2). Report mismatched cases separately to expose any resolution artefact (coarse meshes round off sharp rectangular edges).
10. Do not upsample coarse meshes to look finer (adds no information). Do not blindly decimate fine meshes.

## 5. Planned pipeline (next work)

Parameter values below are starting points to be tuned on cases 1, 2 and 8 (all in mm).

1. **Preprocess:** clean the mesh (remove duplicates, degenerate faces), keep native resolution for measurement. A resampled copy with uniform density (about 0.1–0.15 mm) may be used only for detection.
2. **Search region:** crown buccal surface only. Use PCA for the occlusal axis, keep vertices whose normals face outward from the arch centre, and apply a height window between the gingival margin and the occlusal edge. Removes palate and lingual surfaces.
3. **Height residual (mesh-saliency style):** smooth the surface at a scale of about 2–3 x the attachment size (Taubin or MLS), compute `d = (v − v_smooth) · n` per vertex. Attachments give positive `d` of about 0.2–1 mm; gingival creases give negative `d`. Works for both elliptical and rectangular shapes.
4. **Cluster and filter:** DBSCAN on vertices with `d > ~0.15 mm`, `eps` in mm. Filter clusters by area (mm²), max height, compactness/aspect ratio. Expect at most 14. Refine the boundary at half the maximum height and fill holes. Output: one labelled region per attachment (and later per tooth).
5. **Alignment:** replace manual picking with an automatic initialisation plus Trimmed ICP (global), then local per-attachment ICP (section 4.3). Save fitness/RMSE per case.
6. **Volume and wear:** build the base surface under each attachment from the surrounding ring of vertices (quadratic or thin-plate interpolation), integrate height above it on a fixed grid for T1 and T2, compute wear % and the noise baseline.
7. **Tables:** per attachment, one row. Proposed columns (to confirm with Q1): `case, attachment_id, tooth_label (if known), shape (elliptical/rectangular), area_mm2, max_height_T1, max_height_T2, V1_mm3, V2_mm3, wear_pct, noise_baseline_mm3, T1_density, T2_density, align_rmse, qc_flag`. Unify and compare across cases with pandas/scikit-learn.
8. **Validation:** visual overlays for every case (detected region on T1, T1-vs-T2 heatmap), distance heatmap of aligned T1 vs T2 to tell tooth movement (teeth red, palate blue) from wear (only attachments red) from resolution effects (everything noisy).

Suggested new scripts, same conventions as existing ones: `src/dist_heatmap.py` (distance colouring of aligned T1 to T2; red at 0.3 mm), `src/align_auto.py`, `src/detect_attachments.py` (rewritten), `src/measure_wear.py`, `src/build_tables.py`.

## 6. Open questions

- **Q1:** What do the Phase 1 "tables" contain exactly (rows, columns, comparison)? The notes were unreadable.
- **Q2:** Do the teeth really move between T1 and T2 (aligner treatment)? To be answered by the distance heatmap on cases 1 and 2.
- **Q3:** Real attachment size and thickness. Measure it from the first successful detections.
- **Q4:** Are tooth labels (FDI numbering) needed in the output, or is "attachment_id" enough? There are no labels in the STLs.
- **Q5:** Case 3: is T1 incomplete, a different patient, or recoverable? Decide at the end.
- **Q6:** Is there ground truth for wear (manual measurements, a clinician's rating) to validate against?
- **Q7:** Is y the occlusal axis in all cases and is the palate always at higher y? Verify before using fixed axes anywhere.

## 7. Reference list (unverified, from memory)

- Pauly, Keiser, Gross. *Multi-scale Feature Extraction on Point-Sampled Surfaces.* Eurographics 2003 (surface variation).
- Lee, Varshney, Jacobs. *Mesh Saliency.* SIGGRAPH 2005.
- Besl & McKay 1992; Rusinkiewicz & Levoy 2001; Chetverikov et al. 2002 (ICP and Trimmed ICP).
- Rusu et al. 2009 (FPFH).
- Choi et al. *Accuracy and reliability of palatal superimposition of 3D digital models.* Angle Orthodontist, 2010.
- Tooth segmentation by curvature: Kondo 2004; Zhao 2006; Kronfeld 2010.
- Future deep learning: MeshSegNet (Lian et al. 2020), Teeth3DS dataset.