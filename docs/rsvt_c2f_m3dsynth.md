# Coarse-to-fine RSVT on M3Dsynth

Updated 26 November 2026. This document describes the model and data path used
by `notebooks/01_RSVT_TRAINING.ipynb`. It replaces the 96³-token RSVT as the
proposed model; that design remains available as a negative control
(`configs/training/rsvt_real_ct.toml`, see [rsvt_theory.md](rsvt_theory.md)).
The store was built on 25–26 Nov 2026. One model has been trained on it, and it did
not learn to detect tampering (see [First training run](#first-training-run-26-november-2026)).
The [crop learning check](#crop-learning-check) then showed that the crop network learns
the edit from residual crops but not from the windowed image, so the model now reads
residual crops and checkpoints are chosen by AUROC.

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
   Every axis also covers 24 mm of in-plane pixels: in 26 receipts of the first run the
   changed voxels filled a cube of about 41 mm in plane with as many voxels along z, which
   on 1.8–3 mm slices is 99–228 mm deep. Label slices numbered past the scan depth are
   never read by the authors' loader; their content is counted, an empty one is accepted
   and recorded, and one holding mask voxels makes the sample ineligible.
   After a network error the download reconnects with an HTTP Range request at
   the byte where it stopped, but only if the server resumes the same file (same
   size and ETag/Last-Modified); otherwise it stops rather than splice data. A
   runtime restart reads the current archive again; finished samples are skipped.
3. **Verification while streaming**: every manipulated slice is compared with the
   real slice. A sample is eligible only if all changed voxels lie inside the box,
   the mask lies inside the box, slice counts and shapes match and the mask is not
   empty. This also proves that manipulated and authentic volumes are identical
   outside the edit (no global shortcut). The archive comparison is the only proof
   that the real-series conversion matches the authors' files.
4. `manifest.json` lists authentic volumes, manipulated volumes (eligible or not,
   with reasons) and authentic sites (TB/TM locations). Every file has a checksum
   receipt; `verify_store` reads them back.

Default caps: 200 train + 100 validation patients → 303 series and 3,594 manipulated
samples, of which 3,493 were eligible in the store of 26 Nov 2026 (785 removals,
2,708 insertions). Test patients are not acquired. Change `PATIENTS_TRAIN` /
`PATIENTS_VALIDATION` in the notebook.

`src/rsvt/data/m3dsynth_dataset.py` serves volumes:

* **Coarse view**: exact area-average resampling of the whole scan to 96³ after a
  −1000…400 HU window. Because it is linear, a manipulated volume's view is the
  stored real view plus the resampled box difference (tested to be exact).
* **Native crops**: ≈48 mm on the native voxel grid (z 16–64, y/x 32–96 voxels,
  multiples of 8), the box pasted in, no resampling.
* **Cells**: a 12³ grid over the coarse view (≈25–37 mm cells). Cells holding a
  changed voxel are router targets. Router candidates are the cells that pass a
  body-intensity test on the image alone; the label adds supervised positives but
  never candidates, in training or validation.
* **Edit labels**: the changed voxels (manipulated ≠ real), not the author mask. On
  removals the authors' mask is a ~33 mm cube, while the changed voxels span a cube
  with as many slices as pixels: median 126 mm deep, and 784 of 785 eligible removals
  change voxels outside the mask box; 2,466 of 2,708 insertions stay inside it. A label's
  bounding box must equal the one measured while streaming. Each training run saves
  `label_check.json`, comparing the changed voxels with the mask by operation and generator.

## Model

`src/rsvt/models/coarse_to_fine.py` (`CoarseToFineRSVT`, ~3.6 M parameters):

```text
96³ coarse view ─ CNN ─► 12³ cell features ─► router logits (top-K cells)
                                └──────────► 6³ context tokens
K native crops ─ residual ─ CNN (full-res stem) ─► crop token, crop tamper logit,
                                                   visible-nodule logit, half-res mask
[CLS] + context tokens + crop tokens ─ Transformer (3 layers) ─►
        tampered vs authentic  ·  removal vs insertion
```

The crop network reads each crop minus its 3×3×3 local mean (`fine_input =
"residual"`, from 26 Nov 2026), which removes anatomy and keeps the generator's
texture and seams; see [Crop learning check](#crop-learning-check).

Training crops per volume: 8 router top-K + 1 site crop (author mask centre for
manipulated volumes; a TB/TM site for authentic ones) + 4 random body cells, all
with jitter. Inference uses router top-K only (`crops_at_inference = 16`).
All pooling is reshape/amax-based and the mask decoder uses transposed
convolutions, so CUDA backward stays deterministic.

## Objective

Per volume: volume tamper BCE (weighted so that the 4 authentic and 12 manipulated
volumes of a batch carry equal total weight), removal-vs-insertion BCE
(manipulated only), crop tamper BCE (a crop is tampered when it holds at least a
quarter of the changed voxels a crop of its shape can hold), visible-nodule BCE
(site crops), mask BCE+Dice on the changed voxels and router BCE (positive weight 50). From epoch 4 a **tail-risk term**
adds the empirical CVaR (α = 0.5) of the volume tamper loss within three groups:
authentic (weight 1), removal/FB (2) and insertion/FM (1), plus the variance of
per-generator mean loss within FB and FM (weight 0.1). Logical batches are 4
authentic + 12 manipulated volumes, stratified by generator × operation; the
batch objective is differentiated in scalar space and each volume is replayed
with its recorded RNG and crop selection (tested equal to direct backprop).

Checkpoint selection (from 26 Nov 2026): the highest mean of removal-vs-authentic
and insertion-vs-authentic validation AUROC among risk-stage epochs, using hard
top-K inference. Selection starts at chance: an epoch is kept only if it beats 0.5
by `min_delta`, and early stopping counts epochs without a new best. A model that
gives every scan the same score therefore cannot be selected; if no epoch beats
chance, the last epoch is reported and the summary says so. (The earlier rule,
group-balanced validation BCE, let a constant model win: see
[First training run](#first-training-run-26-november-2026).)

Validation metrics: tamper AUROC, removal and insertion AUROC, per-generator and
worst-generator AUROC, removal/insertion recall at 5% and 10% authentic FPR
(exploratory thresholds), removal-vs-insertion accuracy, router recall@K (overall
and per operation) and mask Dice on crops containing the site.

**Site-crop diagnostic.** Each validation scan is scored a second time with one
extra crop: at the author manipulation centre for a manipulated scan, and at one
of the series' real TB/TM sites (chosen once per scan) for an authentic scan. It
reports the scan AUROC with that crop added and the crop network's AUROC on that
crop alone. The gap to the router-only scores is what crop selection loses. It is
never used for selection or training, and the router-only scores are unchanged by
it (tested).

## Diagnostics before training

`src/rsvt/experiments/m3d_diagnostics.py`, train/validation only:

* **Resolution check**: each manipulated crop (at its declared site centre) is
  paired with an authentic site of the same appearance: removals with TB sites (no
  nodule in either crop), insertions with TM sites (a nodule in both), from the same
  series when available. One small CNN is trained at native resolution and one on
  the same crops degraded to the 96³ voxel size. Reports both validation AUROCs and
  the gap, overall, per operation and per generator × operation. It tests whether
  the evidence at a known site survives coarse voxels, not whether the router can
  find the site.
* **Shortcut check**: archive identity outside boxes, whether edits extend beyond
  the author mask box, and a whole-scan histogram classifier (expected ≈0.5).

On the store of 26 Nov the resolution check gave native-crop AUROC 0.615 (removal
0.448, insertion 0.662) and 0.633 for the degraded crops: the small CNN barely
separated manipulated crops from matched authentic sites at either resolution.

## First training run (26 November 2026)

L4 GPU, commit `a732b1f`, default settings: 200 authentic and 2,024 manipulated
training volumes; 103 authentic and 1,469 manipulated validation volumes (309
removals, 1,160 insertions). Early stopping ended it after epoch 12 (1 h 45 min);
the selected checkpoint is epoch 4. No test volume was opened.

| Measure | Epoch 1 | Epoch 4 (selected) | Epoch 12 (last) |
|---|---|---|---|
| Validation tamper AUROC | 0.572 | 0.427 | 0.530 |
| Validation removal AUROC | 0.616 | 0.524 | 0.623 |
| Validation worst-generator AUROC | 0.551 | 0.380 | 0.526 |
| Router recall@16 | 8% | 38% | 37% |
| Training volume tamper loss (chance 0.693) | 0.698 | 0.706 | 0.698 |
| Training visible-nodule loss | 0.684 | 0.680 | 0.576 |

* The volume tamper, removal and crop tamper losses stayed at chance on the
  training volumes for all 600 optimizer steps. The edit-mask head predicted no
  voxel (Dice ≈ 0). Only the router (loss 2.85 → 0.60) and, from epoch 10, the
  visible-nodule head learned; both describe anatomy, not the edit.
* The selected model gives every validation scan about the same score (≈0.58–0.60),
  so its AUROCs below 0.5 are noise around a constant, not an inverted signal.
* The selection score (validation BCE averaged over authentic, removal and insertion)
  is 0.637 for the best constant prediction. Epoch 4 won by being a better constant,
  and early stopping counted from it. The rule was replaced the same day by an AUROC
  rule that a constant cannot win (see [Objective](#objective)).
* `label_check.json`: removals change 96% of the author mask box (median; 91% at the
  10th percentile) and reach 3.8× its depth, changing every slice in between;
  insertions stay inside their mask box.

## Crop learning check

`src/rsvt/experiments/m3d_crop_check.py` asks whether the model's fine crop network,
alone and with fresh weights, can learn the edit when every crop contains it. Each
pair is a manipulated crop at the author manipulation centre and the same crop of its
real series; the two are checked to differ only in the changed voxels. Removals and
insertions are balanced, so nodule presence cannot separate the classes (the real
crop of a removal holds a nodule, that of an insertion does not); per-operation
results expose that shortcut. Crops are 24×64×64 voxels, cut with a 4/8/8-voxel
margin for shifts. Three arms:

* **fit**: 32 training pairs, no augmentation, 600 steps: can it fit them at all?
* **image**: up to 384 training pairs (192 per operation; shift and flip shared by
  both crops of a pair), 2,000 steps, tested on up to 192 pairs from unseen
  validation patients, on the windowed image the model sees.
* **residual**: the same on the image minus its 3×3×3 local mean, which suppresses
  anatomy and keeps fine texture and seams.

It reports pairs ordered correctly (manipulated above its own real crop) per
operation and generator, crop AUROC, how much of the edit the HU window still shows,
and a one-line conclusion. Its thresholds (≥80% for both operations = learned)
describe the result; they are not pre-registered endpoints. It removes crop
selection, so it bounds what the crop stage can do and says nothing about routing.

**Result (26 Nov 2026, L4, commit `2884b66`).** 384 training pairs (144 patients)
and 192 validation pairs (64 patients), 64 per generator and operation in training
and 32 in validation; no pair was dropped and no test volume was opened. At least
97% of the changed voxels stayed visible after the HU window in 9 of 10 crops.

| Arm (unseen patients) | Pairs ordered correctly, removal / insertion | Crop AUROC, removal / insertion / all |
|---|---|---|
| fit (32 training pairs) | 100% / 100% (training pairs) | — |
| image | 67% / 81% | 0.69 / 0.69 / 0.69 |
| residual | 96% / 84% | 0.94 / 0.75 / 0.86 |

* On the windowed image the network fitted the training pairs (87.5%, crop AUROC 0.91)
  but did not transfer: removal on unseen patients peaked at 86% at step 200 and fell
  to 67%. It memorised anatomy.
* On the residual it learned the edit: every cycle and diffusion removal was ordered
  correctly, and pix2pix removals at 88%. Pix2pix insertion was weakest (crop AUROC
  0.64). Insertion was still improving at step 2,000.
* Early in training both arms ranked insertions below chance: they first learned
  "no nodule = manipulated". The residual arm outgrew it, but part of the remaining
  removal/insertion gap may be that bias; the rest is edit size (median changed
  voxels per crop: 47% of the crop for removals, 14% for insertions).
* Wilson 95% intervals for residual pairs ordered correctly: removal 90–98%,
  insertion 76–90%. They ignore that a patient contributes up to 19 pairs.

The first training run fed the crop network the windowed image, which explains much
of why its crop loss stayed at chance. The model now reads residual crops
(`fine_input = "residual"`).

## Running it

Open `notebooks/01_RSVT_TRAINING.ipynb` in Colab from GitHub and use Run all. On a
GPU, `RUN` chooses what the run does: `"train"` (the default) trains in section 13;
`"crop_check"` runs the crop learning check in section 12 and does not train.
With `RUN = "train"`, section 3 continues the latest unfinished training
run when there is one (`RESUME_SESSION = "auto"`), checking out the commit that
run started with; otherwise it clones the public repository at `CODE_REF`. Section
4 fetches the M3Dsynth metadata tables at a pinned commit of the authors'
repository and checks their SHA-256. Without the audited UID map, which stays out
of the repository, each LIDC series is found on TCIA by patient and series number;
on a sample of 30 series this gave the audited UID every time.

Section 7 downloads only what the saved store is missing for the current
selection, after checking free space in Google Drive and on the runtime disk;
otherwise it reads the store back against its checksums. Section 9 runs the data
checks once per store (saved beside it) and holds training back if most samples
failed archive verification or whole-scan statistics separate the classes.
Sections 12 and 13 run only on a GPU; on a CPU runtime they stop with a message, so
the cheap first run prepares and checks the data and a later GPU run does the rest.
Section 18 releases the Colab runtime (`google.colab.runtime.unassign`) after
section 17 has read every saved file back; if an earlier cell fails, the runtime is kept.

## Limits

* Exploratory engineering pilot: hyperparameters are starting values, operating
  thresholds come from the same validation authentic volumes, no test evaluation.
* CT-GAN also derives from LIDC-IDRI and its source identities are unresolved, so
  CT-GAN is not an independent external test until identities are reconciled.
* The real-series conversion is a re-implementation; archive verification marks
  any mismatching sample ineligible rather than assuming equality.
* The crop learning check places every crop on the edit. In the first training run
  the router's 16 crops held the edit for 37% of manipulated validation scans, so
  routing, not the crop stage, may now limit scan-level detection; the site-crop
  diagnostic measures this.
* Author masks do not cover the removal edits, so localization labels use the changed
  voxels. Why the removal edits are a voxel cube (as many slices as pixels) is not
  documented by the authors; `label_check.json` reports how much of each mask changed.
