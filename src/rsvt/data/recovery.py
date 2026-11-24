"""Locate CT-GAN inputs or restore a persistent archive on a Colab VM."""

from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path

from rsvt.data.acquisition import extract_zip, fetch_file, sha256_file
from rsvt.utils.runtime import require_colab_runtime

CTGAN_URL = ("https://archive.ics.uci.edu/static/public/520/"
             "deepfakes%2Bmedical%2Bimage%2Btamper%2Bdetection.zip")
MAX_ARCHIVE_BYTES = 12 * 1024**3


def is_ctgan_root(path: Path) -> bool:
    """Recognize layout only; F02 must still check counts and F01 source hashes."""
    folder = Path(path) / "Tampered Scans"
    return (all((folder / name).is_dir() for name in ("Experiment 1 - Blind", "Experiment 2 - Open"))
            and all((folder / name).is_file() for name in ("labels_exp1.csv", "labels_exp2.csv")))


def locate_ctgan_dataset(expected_root: Path, drive_root: Path, override=None) -> Path | None:
    if override is not None:
        selected = Path(override)
        if not is_ctgan_root(selected):
            raise ValueError("CT_ROOT_OVERRIDE must contain both CT-GAN experiment folders and label tables")
        return selected
    candidates = [Path(expected_root)]
    for parent in ("raw/ct_gan/uci", "data/raw/ct_gan/uci", "datasets/ct_gan/uci"):
        candidates += [Path(drive_root) / parent / "dataset", Path(drive_root) / parent]
    candidates += [Path(drive_root) / "datasets/ct_gan/dataset", Path(drive_root) / "dataset"]
    for path in candidates:
        if is_ctgan_root(path):
            return path
    return None


def restore_ctgan_archive(archive: Path, extraction_parent: Path, *,
                         download_budget_bytes: int, progress=None) -> dict:
    """Cache the official ZIP in Drive and extract into a new Colab directory.

    Never start from a local kernel. Incomplete archives/extractions are retained.
    Completed cached archives are checked against their acquisition receipt when
    present. Source equivalence is independently enforced against F01 by F02.
    """
    require_colab_runtime()
    if (isinstance(download_budget_bytes, bool) or not isinstance(download_budget_bytes, int)
            or download_budget_bytes < 0):
        raise ValueError("Download budget must be a nonnegative integer")
    archive, extraction_parent = Path(archive), Path(extraction_parent)
    if extraction_parent.exists():
        raise FileExistsError("Use a new restoration directory; prior outputs are preserved")
    if archive.is_symlink() or (archive.exists() and not archive.is_file()):
        raise ValueError("Archive must be a regular file")
    downloaded = 0
    receipt_path = archive.with_name(archive.name + ".receipt.json")
    if archive.is_file():
        fingerprint = sha256_file(archive)
        if receipt_path.is_file():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            if prior["sha256"] != fingerprint or prior["bytes"] != archive.stat().st_size:
                raise ValueError("Cached archive differs from its acquisition receipt")
            verification = "acquisition_sha256_matched"
        else:
            verification = "current_sha256_only_no_historical_receipt"
    else:
        if download_budget_bytes == 0:
            raise ValueError("No storage budget for restoring the missing source archive")
        receipt = fetch_file(CTGAN_URL, archive,
                             max_bytes=min(MAX_ARCHIVE_BYTES, download_budget_bytes), progress=progress)
        downloaded = receipt["bytes"]
        fingerprint = receipt["sha256"]
        verification = "download_sha256_recorded"
    if progress:
        progress({"stage": "restore_extract", "archive": str(archive)})
    outer = extraction_parent / "uci"
    outer_receipt = extract_zip(archive, outer, max_uncompressed_bytes=50 * 1024**3)
    nested = outer / "data.zip"
    if nested.is_file():
        inner_receipt = extract_zip(nested, outer / "dataset", max_uncompressed_bytes=50 * 1024**3)
        dataset_root = outer / "dataset"
    else:
        inner_receipt = None
        dataset_root = outer / "dataset" if is_ctgan_root(outer / "dataset") else outer
    if not is_ctgan_root(dataset_root):
        raise ValueError("Restored archive has no recognized CT-GAN dataset layout")
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_root": str(dataset_root), "archive": str(archive),
              "archive_sha256": fingerprint, "archive_verification": verification,
              "new_archive_bytes": downloaded, "publisher_checksum_verified": False,
              "outer_extraction": outer_receipt, "inner_extraction": inner_receipt,
              "f01_source_equivalence": "pending_phase2_hash_comparison", "training_ready": False}
    (extraction_parent / "restoration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
