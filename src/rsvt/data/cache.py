"""Validate previously generated F02 artifacts before reuse; never rewrite originals."""

from __future__ import annotations
import gzip
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from rsvt.data.acquisition import sha256_file


def artifact_names(save_native: bool) -> list[str]:
    names = ["hu_processed.nii.gz", "normalized.nii.gz", "valid_mask.nii.gz", "source_slices.csv"]
    if save_native:
        names += ["hu_native.nii.gz", "valid_native.nii.gz"]
    return names


def artifact_receipts(directory: Path, save_native: bool) -> dict:
    return {name: {"bytes": (directory / name).stat().st_size, "sha256": sha256_file(directory / name)}
            for name in artifact_names(save_native)}


def resume_index(run_directories) -> dict[str, list[dict]]:
    index = {}
    for run in run_directories:
        path = Path(run) / "cases.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Resume cases.json must contain a list")
        for row in rows:
            if row.get("status") not in {"preprocessed_pending_visual_review", "reused_cache_pending_review"}:
                continue
            if row.get("case_key") and row.get("output"):
                index.setdefault(row["case_key"], []).append({**row, "resume_run": str(run)})
    return index


def _check_header(path: Path, expected: dict) -> None:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    expected_type = sitk.sitkUInt8 if path.name.startswith("valid_") else sitk.sitkFloat32
    if reader.GetNumberOfComponents() != 1 or reader.GetPixelID() != expected_type:
        raise ValueError(f"Cached NIfTI pixel type mismatch: {path.name}")
    if list(reader.GetSize()) != expected["size_xyz"]:
        raise ValueError(f"Cached NIfTI size mismatch: {path.name}")
    for actual, key in ((reader.GetSpacing(), "spacing_xyz_mm"), (reader.GetOrigin(), "origin_lps_mm"),
                        (reader.GetDirection(), "direction")):
        if not np.allclose(actual, expected[key], rtol=1e-6, atol=1e-5):
            raise ValueError(f"Cached NIfTI geometry mismatch: {path.name}")


def validate_cache(entry: dict, candidate: dict, geometry, spec, annotation_rows, xy_index_base: int,
                   current_source_fingerprint: str) -> dict:
    directory = Path(entry["output"])
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "preprocessed_pending_visual_review":
        raise ValueError("Cache has no completed metadata marker")
    if metadata.get("case_key") != entry["case_key"] or metadata.get("source_candidate_id") != candidate["candidate_id"]:
        raise ValueError("Cached case/source identity mismatch")
    if metadata.get("source_file_set_sha256") != current_source_fingerprint:
        raise ValueError("Cached source fingerprint mismatch")
    if metadata.get("transform", {}).get("spec_fingerprint") != spec.fingerprint():
        raise ValueError("Cached preprocessing spec mismatch")
    expected_native = geometry.as_dict()
    for key in ("size_xyz", "spacing_xyz_mm", "origin_lps_mm", "direction"):
        if not np.allclose(metadata["native_geometry"][key], expected_native[key], rtol=1e-6, atol=1e-5):
            raise ValueError("Cached native geometry mismatch")
    cached_points = metadata.get("annotations", [])
    before = [(p["label"], str(p["source_slice_reference"]), str(p["source_x"]), str(p["source_y"]), p["xy_index_base"])
              for p in cached_points]
    after = [(r["location_label"], str(r["slice_reference"]), str(r["pixel_x"]), str(r["pixel_y"]), xy_index_base)
             for r in annotation_rows]
    if before != after:
        raise ValueError("Cached annotation inputs differ")
    prior = entry.get("cache_validation", {}).get("artifacts") or metadata.get("output_artifacts")
    receipts = {}
    for name in artifact_names(spec.save_native_hu):
        path = directory / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"Missing/empty cached artifact: {name}")
        if name.endswith(".nii.gz"):
            expected = metadata["native_geometry"] if name in {"hu_native.nii.gz", "valid_native.nii.gz"} else metadata["transform"]["processed"]
            _check_header(path, expected)
        if prior:
            recorded = prior.get(name)
            if not recorded or path.stat().st_size != recorded["bytes"] or sha256_file(path) != recorded["sha256"]:
                raise ValueError(f"Cached artifact checksum mismatch: {name}")
        else:
            # Legacy pilot outputs have no historical output checksum. Verify full
            # gzip integrity and geometry, then record an adoption fingerprint.
            if name.endswith(".gz"):
                with gzip.open(path, "rb") as stream:
                    while stream.read(1024 * 1024):
                        pass
        receipts[name] = {"bytes": path.stat().st_size, "sha256": prior[name]["sha256"] if prior else sha256_file(path)}
    return {"metadata": metadata, "artifacts": receipts,
            "verification": "recorded_sha256_and_header" if prior else "legacy_header_gzip_crc_and_adoption_sha256",
            "legacy_content_authenticity_proven": False if not prior else None}
