# Risk-Aware Sparse Volumetric Transformer for Tampered Lung CT Detection

This repository contains the code for a study of detecting nodule injection and nodule removal in 3D chest CT scans (medical deepfakes).

**Research question.** Can a coarse-to-fine sparse transformer, which reads native-resolution crops chosen by a router, detect tampered lung CT more accurately and at lower compute than dense 3D CNN and ViT baselines? Removal of nodules is the endpoint of main interest.

**Status.** The study is under development. No results are available yet.

## Method

The proposed model is a coarse-to-fine RSVT (Risk-Aware Sparse Volumetric Transformer):

1. A 96³ view of the whole scan provides anatomical context and a router that scores a 12³ grid of cells.
2. A full-resolution encoder runs only on the top-K selected cells, cropped from the native voxel grid, so small edits are not averaged away.
3. A Transformer combines context tokens and crop tokens into two outputs: tampered versus authentic, and removal versus insertion.
4. Training adds a tail-risk (CVaR) term over authentic, removal, and insertion volumes and a penalty on the spread of per-generator losses.

Dense 3D ResNet and 3D ViT models serve as baselines. The earlier 96³-token RSVT is kept as a negative control. See [the model description](docs/rsvt_c2f_m3dsynth.md), [the design rationale](docs/rsvt_theory.md), and [the protocol decisions](docs/protocol_decisions.md).

## Repository structure

```
configs/          Dataset, model, training, evaluation, and ablation settings
data/             Dataset instructions (datasets are not distributed)
docs/             Method and protocol documentation
notebooks/        The notebook that prepares data and trains on Google Colab
requirements/     Pinned environments for data inventory and preprocessing
scripts/          Command-line tools for data acquisition, audits, and notebook builds
src/rsvt/         The experiment package
tests/            Unit tests on synthetic and procedural data
```

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[inventory,preprocessing,annotation_review,training]" matplotlib pillow
```

On Linux or macOS, use `.venv/bin/python`. Training runs on Google Colab; the local environment is used for development and tests.

## Development

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests use synthetic and procedural fixtures only. A few tests also check locally audited dataset metadata and are skipped when it is absent.

## Data

The study trains and validates on M3Dsynth, which edits LIDC-IDRI scans with three generators. CT-GAN is kept for comparison. Datasets are not included in this repository; see [`data/README.md`](data/README.md).

## Running the study

Open [`notebooks/01_RSVT_TRAINING.ipynb`](notebooks/01_RSVT_TRAINING.ipynb) in VS Code or Colab with a GPU runtime and mount Google Drive. The notebook shows the data plan, downloads and verifies the selected M3Dsynth samples and their LIDC-IDRI source series when `PREPARE_DATA = True`, runs resolution and shortcut checks, and trains the model. Data preparation and training resume after an interruption. The notebook loads the package from a checksum-verified code release in the project's Drive folder rather than from code in its cells.

Test patients from the publisher's partition are not downloaded or opened by data preparation, diagnostics, training, or model selection.
