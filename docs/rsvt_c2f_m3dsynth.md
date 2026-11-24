# Coarse-to-fine RSVT on M3Dsynth

Updated 17 November 2026. This document describes the model and data path used
by `notebooks/01_RSVT_TRAINING.ipynb`. It replaces the 96³-token RSVT as the
proposed model; that design remains available as a negative control
(`configs/training/rsvt_real_ct.toml`, see [rsvt_theory.md](rsvt_theory.md)).
Nothing here is a result: no M3Dsynth image has been downloaded or trained on yet.

## Why the design changed

The architecture review of 17 November found three blocking issues and four
bugs in the 96³ RSVT:

| Issue | Evidence | Resolution |
|---|---|---|
| 96³ full-FOV input (≈3.1–4.6 mm voxels) averages away small edits | CNN/ViT baselines at chance on the same views; ViT memorized train | Evidence is read from **native-grid crops**; the 96³ view only gives context and proposes where to look |
| Sparse routing saved ~14–22% because the dense stem dominated | F06 profile 502→430 ms; ~8.8 vs ~6.9 GMAC | The expensive encoder runs only on **K selected crops**; ≈20–40× fewer MACs than the same encoder over the whole native volume |
| Objective misaligned with the endpoint (four-class CE incl. TB vs TM; CVaR ignored the authentic tail) | — | **Factorized heads** (tampered; removal vs insertion) and a **tail risk on both sides of the threshold** |
| Router started at ~8× the token budget | init gates ≈0.5 | Router bias initialized at the budget/prior (both models) |
| Soft-train/hard-inference attention mismatch | gate bias dropped at inference | `hard_gate_bias=true` (96³ model); fixed top-K crops (coarse-to-fine) |
| Mean pooling diluted one-voxel traces | 1/512 per 16³ patch | max+mean pooling (`token_pooling="max_mean"`; crop tokens use mean+max) |
| No augmentation | — | Left-right flips and crop jitter (training only) |

## Data: M3Dsynth store

M3Dsynth (Zingarini et al., ICASSP 2024) injects (FM) or removes (FB) lung nodules
in LIDC-IDRI scans with three generators: `cycle`, `pix2pix`, `diffusion`
(8,577 manipulated samples, 744 source series, 738 patients). The publisher's
patient-disjoint partitions are used: train 488 / validation 100 / test 150 patients.

`src/rsvt/data/m3dsynth_store.py` builds a verified store in
`MyDrive/Research/LUNG-CT-SCAN/datasets/m3dsynth_store/`:

1. **Real series** for the selected patients are downloaded from the public TCIA
   API by the exact SeriesInstanceUID recorded in the 11 Nov metadata audit, or
   read from an existing LIDC-IDRI folder (`LIDC_ROOT`). Conversion reproduces the
   authors' `convert_real_to_tiff.py`: `.dcm` files sorted by descending
   ImagePositionPatient z, raw stored pixels, `scan_to_uint16` offset rule.
   Headers are checked (PatientID, SeriesNumber, UID, CT, constant rescale).
2. **Author archives** (`cycle.tgz` 171 GB, `pix2pix.tgz` 142 GB, `diffusion.tgz`
   161 GB) are streamed once; they are not seekable, so the full archive is
   transferred even for a subset. For each selected sample only a box of
   ±48 mm around the author manipulation centre and the author voxel mask are kept.
3. **Verification while streaming**: every manipulated slice is compared with the
   real slice. A sample is eligible only if all changed voxels lie inside the box,
   the mask lies inside the box, slice counts and shapes match and the mask is not
   empty. This also proves that manipulated and authentic volumes are identical
   outside the edit (no global shortcut). The archive comparison is the only proof
   that the real-series conversion matches the authors' files.
4. `manifest.json` lists authentic volumes, manipulated volumes (eligible or not,
   with reasons) and authentic sites (TB/TM locations). Every file has a checksum
   receipt; `verify_store` reads them back.

Default caps: 200 train + 60 validation patients → 261 series, 2,993 manipulated
samples, about 19 GB in Drive and ~29 GB of LIDC download. Test patients are not
acquired. Change `PATIENTS_TRAIN` / `PATIENTS_VALIDATION` in the notebook.

`src/rsvt/data/m3dsynth_dataset.py` serves volumes:

* **Coarse view**: exact area-average resampling of the whole scan to 96³ after a
  −1000…400 HU window. Because it is linear, a manipulated volume's view is the
  stored real view plus the resampled box difference (tested to be exact).
* **Native crops**: ≈48 mm on the native voxel grid (z 16–64, y/x 32–96 voxels,
  multiples of 8), the box pasted in, no resampling.
* **Cells**: a 12³ grid over the coarse view (≈25–37 mm cells). Cells overlapping
  the author mask are router targets.

## Model

`src/rsvt/models/coarse_to_fine.py` (`CoarseToFineRSVT`, ~3.6 M parameters):

```text
96³ coarse view ─ CNN ─► 12³ cell features ─► router logits (top-K cells)
                                └──────────► 6³ context tokens
K native crops ─ CNN (full-res stem) ─► crop token, crop tamper logit,
                                        visible-nodule logit, half-res mask
[CLS] + context tokens + crop tokens ─ Transformer (3 layers) ─►
        tampered vs authentic  ·  removal vs insertion
```

Training crops per volume: 8 router top-K + 1 site crop (author mask centre for
manipulated volumes; a TB/TM site for authentic ones) + 4 random body cells, all
with jitter. Inference uses router top-K only (`crops_at_inference = 16`).
All pooling is reshape/amax-based and the mask decoder uses transposed
convolutions, so CUDA backward stays deterministic.

## Objective

Per volume: volume tamper BCE (class-weighted), removal-vs-insertion BCE
(manipulated only), crop tamper BCE, visible-nodule BCE (site crops), mask
BCE+Dice and router BCE (positive weight 50). From epoch 4 a **tail-risk term**
adds the empirical CVaR (α = 0.5) of the volume tamper loss within three groups:
authentic (weight 1), removal/FB (2) and insertion/FM (1), plus the variance of
per-generator mean loss within FB and FM (weight 0.1). Logical batches are 4
authentic + 12 manipulated volumes, stratified by generator × operation; the
batch objective is differentiated in scalar space and each volume is replayed
with its recorded RNG and crop selection (tested equal to direct backprop).

Checkpoint selection: lowest validation balanced volume BCE among risk-stage
epochs, using hard top-K inference. Validation metrics: tamper AUROC, removal and
insertion AUROC, per-generator and worst-generator AUROC, removal/insertion recall
at 5% and 10% authentic FPR (exploratory thresholds), removal-vs-insertion
accuracy, router recall@K and mask Dice on crops containing the site.

## Diagnostics before training

`src/rsvt/experiments/m3d_diagnostics.py`, train/validation only:

* **Resolution check**: paired crops at each manipulation centre (manipulated vs
  the same location in the authentic series), one small CNN trained at native
  resolution and one on the same crops degraded to the 96³ voxel size. Reports
  both validation AUROCs and the gap, overall and per generator × operation.
* **Shortcut check**: archive identity outside boxes, whether edits extend beyond
  the author mask box, and a whole-scan histogram classifier (expected ≈0.5).

## Running it

Open `notebooks/01_RSVT_TRAINING.ipynb` with a Colab GPU kernel. Review the data
plan (section 5), set `PREPARE_DATA = True` to download and verify, then run the
remaining sections. Data preparation and training both resume after interruption.
The release folder `outputs/rsvt_training/release_20261117_m3dsynth_c2f/` must
exist in Drive (see the delivery report).

## Limits

* Exploratory engineering pilot: hyperparameters are starting values, operating
  thresholds come from the same validation authentic volumes, no test evaluation.
* CT-GAN also derives from LIDC-IDRI and its source identities are unresolved, so
  CT-GAN is not an independent external test until identities are reconciled.
* The real-series conversion is a re-implementation; archive verification marks
  any mismatching sample ineligible rather than assuming equality.
* Author masks may not cover every changed voxel; the shortcut check reports how
  often edits extend beyond the mask box.
