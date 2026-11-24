# Data

Datasets are not stored in this repository, and each one remains under its original licence. Images are prepared and stored on Google Colab and Google Drive, not on the development machine.

| Dataset | Role | Local folder | Access |
| --- | --- | --- | --- |
| M3Dsynth | Training and validation (cycle, pix2pix, and diffusion generators) | `data/raw/m3dsynth/` | [GRIP, University of Naples Federico II](https://grip-unina.github.io/M3Dsynth/); see the [licence](https://www.grip.unina.it/download/prog/M3Dsynth/LICENSE.txt) |
| LIDC-IDRI | Authentic source series for M3Dsynth | `data/raw/lidc_idri/` | [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/lidc-idri/) |
| CT-GAN | Earlier pilot and comparison data | `data/raw/ct_gan/` | [UCI Machine Learning Repository, dataset 520](https://archive.ics.uci.edu/dataset/520) |

[`public_sources.json`](public_sources.json) records the source URLs, archive sizes, and publisher checksums. The M3Dsynth metadata tables are fetched with `scripts/fetch_m3dsynth_metadata.py`. The training notebook downloads the selected LIDC-IDRI series from the public TCIA API by SeriesInstanceUID and streams the M3Dsynth archives once, keeping only a verified box around each manipulation. Patients in the publisher's test partition are not downloaded.

[`manifests/manifest.template.csv`](manifests/manifest.template.csv) defines the manifest columns. Filled manifests, split assignments, keys, and anything derived from the images are written to this folder, which git ignores, and must not be committed or shared unless the dataset licence permits it.
