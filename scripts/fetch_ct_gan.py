"""Acquire the official UCI CT-GAN archive; no credentials or forgery code required."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsvt.data.acquisition import fetch_file, extract_zip  # noqa: E402
from rsvt.utils.runtime import require_colab_runtime  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("/content/rsvt-data/downloads/ct_gan_uci.zip"))
    parser.add_argument("--destination", type=Path, default=Path("/content/rsvt-data/raw/ct_gan/uci"))
    args = parser.parse_args()
    require_colab_runtime()
    archive = args.archive
    receipt = fetch_file(
        "https://archive.ics.uci.edu/static/public/520/"
        "deepfakes%2Bmedical%2Bimage%2Btamper%2Bdetection.zip",
        archive,
        max_bytes=12 * 1024**3,
        progress=lambda payload: print(json.dumps(payload), flush=True),
    )
    print(json.dumps(receipt, indent=2), flush=True)
    extraction = extract_zip(archive, args.destination, max_uncompressed_bytes=50 * 1024**3)
    print(json.dumps(extraction, indent=2), flush=True)
    # UCI's outer ZIP wraps a second ZIP containing DICOM data and ground truth.
    nested = args.destination / "data.zip"
    if nested.is_file():
        extraction = extract_zip(nested, args.destination / "dataset",
                                 max_uncompressed_bytes=50 * 1024**3)
        print(json.dumps(extraction, indent=2), flush=True)
    (args.destination / "acquisition_complete.json").write_text(
        json.dumps({"archive_receipt": receipt, "extraction": extraction,
                    "training_ready": False}, indent=2) + "\n", encoding="utf-8")
