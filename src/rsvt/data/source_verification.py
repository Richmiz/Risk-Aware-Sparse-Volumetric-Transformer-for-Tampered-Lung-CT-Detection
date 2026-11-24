"""Pixel/source audit for the user-run Colab data-verification notebook.

No model is loaded, no predictions are made, and no split or historical file is
rewritten. Exact shared image slices are evidence; approximate matches remain
review candidates. Missing public identities are never filled with invented IDs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path, PurePosixPath
import re
import zipfile
import zlib

import numpy as np
import pydicom

from rsvt.data.acquisition import sha256_file
from rsvt.data.inventory import write_csv
from rsvt.data.verification import read_csv, scan_key
from rsvt.utils.runtime import require_colab_runtime


def public_identity(header, catalog):
    """Only accept a unique exact public series UID match, with patient agreement."""
    uid = str(getattr(header, "SeriesInstanceUID", ""))
    matches = [r for r in catalog if r["SeriesInstanceUID"] == uid]
    if len(matches) != 1:
        return {"canonical_patient_id": "", "canonical_series_uid": "",
                "public_mapping_status": "unresolved_no_unique_exact_public_series_UID"}
    row = matches[0]
    if not str(row.get("PatientID", "")).startswith("LIDC-IDRI-") or row.get("Collection") != "LIDC-IDRI":
        raise ValueError("Public source match has unexpected collection/patient namespace")
    native_patient = str(getattr(header, "PatientID", ""))
    if native_patient.startswith("LIDC-IDRI-") and native_patient != row["PatientID"]:
        raise ValueError("Native public patient ID conflicts with public series mapping")
    return {"canonical_patient_id": row["PatientID"], "canonical_series_uid": uid,
            "public_series_number": str(row.get("SeriesNumber", "")),
            "public_image_count": int(row["ImageCount"]), "public_mapping_status": "exact_public_series_UID_match"}


def slice_fingerprint(array, slope, intercept, padding_low=None, padding_high=None):
    """Hash native HU and valid support; exclude constant/padding-only slices."""
    array = np.asarray(array)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite single-frame slice")
    hu = array.astype(np.float64) * float(slope) + float(intercept)
    if not np.isfinite(hu).all():
        raise ValueError("Nonfinite native HU")
    valid = np.ones(array.shape, dtype=bool)
    if padding_low is not None:
        high = padding_low if padding_high is None else padding_high
        low, high = sorted((padding_low, high))
        valid &= ~((array >= low) & (array <= high))
    informative = bool(valid.any() and np.ptp(hu[valid]) > 0.)
    canonical = np.where(valid, hu, 0.).astype("<f8", copy=False)
    digest = hashlib.sha256(str(array.shape).encode() + canonical.tobytes() + valid.tobytes()).hexdigest()
    return digest, informative, hu, valid


def inspect_case_pixels(root, rows, catalog):
    """Recheck every source file against F01 before decoding its native pixels."""
    ordered = sorted(rows, key=lambda r: int(PurePosixPath(r["relative_path"].replace("\\", "/")).stem))
    wanted = set(np.linspace(0, len(ordered)-1, min(8, len(ordered))).round().astype(int).tolist())
    selected, pixel_hashes, facts, headers = [], [], [], []
    full_digest = hashlib.sha256()
    first = None
    for index, row in enumerate(ordered):
        relative = PurePosixPath(row["relative_path"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source image path")
        path = (Path(root) / Path(*relative.parts)).resolve()
        if not path.is_relative_to(Path(root).resolve()) or path.is_symlink():
            raise ValueError("Source image escapes dataset root")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"] or len(content) != int(row["bytes"]):
            raise ValueError("Source bytes differ from F01: " + str(relative))
        header = pydicom.dcmread(BytesIO(content))
        if first is None:
            first = header
        slope, intercept = float(header.RescaleSlope), float(header.RescaleIntercept)
        fingerprint, informative, hu, valid = slice_fingerprint(header.pixel_array, slope, intercept,
            getattr(header, "PixelPaddingValue", None), getattr(header, "PixelPaddingRangeLimit", None))
        full_digest.update(fingerprint.encode())
        if informative:
            pixel_hashes.append(fingerprint)
        if index in wanted:
            # Descriptive image similarity only; it never creates a source edge.
            yi = np.linspace(0, hu.shape[0]-1, min(32, hu.shape[0])).round().astype(int)
            xi = np.linspace(0, hu.shape[1]-1, min(32, hu.shape[1])).round().astype(int)
            selected.append(np.clip((hu[np.ix_(yi, xi)]+1000.)/1400., 0., 1.))
        facts.append({"path": str(relative), "bytes": len(content), "crc32": zlib.crc32(content) & 0xffffffff})
        headers.append({"filename": relative.name, "physical_position": [float(x) for x in header.ImagePositionPatient]})
    identity = public_identity(first, catalog)
    if identity["canonical_patient_id"]:
        identity["public_slice_count_matches"] = identity["public_image_count"] == len(ordered)
    return {"source_file_count": len(ordered), "all_source_bytes_match_F01": True,
            "all_pixels_decoded": True, "full_native_pixel_fingerprint": full_digest.hexdigest(),
            "informative_slice_hashes": sorted(set(pixel_hashes)), "public_identity": identity,
            "thumbnail_signature": np.stack(selected).astype(np.float32), "archive_facts": facts,
            "slice_positions": headers}


def compare_pixels(cases, records, *, minimum_shared_slices=3, similarity_mae=.01):
    """Expose cross-partition shared content and retain approximate matches as pending."""
    if minimum_shared_slices < 3 or similarity_mae <= 0:
        raise ValueError("Explicit conservative pixel-evidence thresholds required")
    record_by_key = {r["case_key"]: r for r in records}
    pairs, similarities = [], []
    for first, second in combinations(sorted(cases), 2):
        a, b = cases[first], cases[second]
        common = len(set(a["informative_slice_hashes"]) & set(b["informative_slice_hashes"]))
        full_equal = a["full_native_pixel_fingerprint"] == b["full_native_pixel_fingerprint"]
        first_public = a["public_identity"].get("canonical_patient_id", "")
        second_public = b["public_identity"].get("canonical_patient_id", "")
        same_public_patient = bool(first_public and first_public == second_public)
        if common >= minimum_shared_slices or (full_equal and common > 0) or same_public_patient:
            partitions = sorted({record_by_key[k]["split"] for k in (first, second)})
            pairs.append({"first": first, "second": second, "distinct_shared_informative_slices": common,
                          "full_native_pixel_equal": full_equal, "same_public_patient": same_public_patient,
                          "draft_partitions": partitions, "cross_partition_source_evidence": len(partitions) > 1,
                          "same_existing_group": record_by_key[first]["source_group_candidate"] == record_by_key[second]["source_group_candidate"],
                          "status": "shared_source_evidence_requires_identity_reconciliation"})
        x, y = a["thumbnail_signature"], b["thumbnail_signature"]
        if x.shape == y.shape:
            mae = float(np.mean(np.abs(x-y)))
            if mae <= similarity_mae and not (common >= minimum_shared_slices or full_equal):
                similarities.append({"first": first, "second": second, "thumbnail_MAE": mae,
                                     "status": "approximate_candidate_only_do_not_merge_or_certify"})
    return pairs, similarities


def compare_archive(inner_archive, facts):
    """Compare ZIP central-directory names, sizes and CRC with the actual source."""
    if inner_archive is None:
        return {"verified": False, "status": "inner_data_zip_not_available"}
    actual = {r["path"]: r for r in facts}
    with zipfile.ZipFile(inner_archive) as archive:
        if "data.zip" in archive.namelist():
            # Read the nested central directory on Colab without extracting or
            # changing the cached acquired archive.
            with archive.open("data.zip") as nested_stream, zipfile.ZipFile(nested_stream) as nested:
                members = [m for m in nested.infolist() if not m.is_dir() and m.filename.lower().endswith(".dcm")]
        else:
            members = [m for m in archive.infolist() if not m.is_dir() and m.filename.lower().endswith(".dcm")]
        expected = {str(PurePosixPath(m.filename.replace("\\", "/"))): m for m in members}
        if len(expected) != len(members):
            raise ValueError("Duplicate DICOM members in publisher archive")
        mismatches = sorted(name for name in set(actual) & set(expected)
                            if actual[name]["bytes"] != expected[name].file_size or actual[name]["crc32"] != expected[name].CRC)
        missing, extra = sorted(set(expected)-set(actual)), sorted(set(actual)-set(expected))
    return {"verified": not (missing or extra or mismatches), "archive_dicom_count": len(expected),
            "actual_dicom_count": len(actual), "missing_members": missing, "unexpected_files": extra,
            "size_or_CRC_mismatches": mismatches,
            "scope": "complete_relative_to_acquired_archive_not_proof_of_original_author_Drive_completeness"}


def run_source_verification(*, dataset_root, manifest_path, files_path, catalog_path,
                            metadata_summary_path, m3_metadata_dir, output_dir, inner_archive=None, progress=None):
    require_colab_runtime()
    output = Path(output_dir)
    if output.exists():
        raise ValueError("Use a fresh source-audit directory")
    records = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["records"]
    metadata = json.loads(Path(metadata_summary_path).read_text(encoding="utf-8"))
    if not metadata["metadata_checks_passed"]:
        raise ValueError("Resolve metadata audit failures before pixel verification")
    for path in (manifest_path, files_path, catalog_path,
                 Path(m3_metadata_dir) / "LIDC.csv", Path(m3_metadata_dir) / "sets.csv"):
        expected = metadata["input_sha256"].get(str(Path(path).resolve()))
        if not expected or expected != sha256_file(Path(path)):
            raise ValueError("Pixel-audit inputs differ from the completed metadata audit")
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("An actual public TCIA series catalog is required")
    files = read_csv(files_path)
    by_scan = defaultdict(list)
    for row in files:
        if row["dataset"] == "ct_gan" and row["kind"] == "dicom":
            by_scan[scan_key(PurePosixPath(row["relative_path"].replace("\\", "/")).parent)].append(row)
    if set(by_scan) != {r["case_key"] for r in records}:
        raise ValueError("Pixel audit coverage differs from manifest")
    output.mkdir(parents=True)
    cases, facts, public_rows = {}, [], []
    for index, key in enumerate(sorted(by_scan), 1):
        result = inspect_case_pixels(dataset_root, by_scan[key], catalog)
        cases[key] = result
        facts += result["archive_facts"]
        public_rows.append({"case_key": key, **result["public_identity"]})
        saved = {k: v for k, v in result.items() if k != "thumbnail_signature"}
        (output / (key.replace(":", "_") + ".json")).write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        if progress:
            progress({"case": key, "completed": index, "total": len(by_scan),
                      "public_mapping": result["public_identity"]["public_mapping_status"]})
    edges, candidates = compare_pixels(cases, records)
    archive = compare_archive(inner_archive, facts)
    m3_sources = read_csv(Path(m3_metadata_dir) / "LIDC.csv")
    m3_sets = {r["orig_id"]: r["set"] for r in read_csv(Path(m3_metadata_dir) / "sets.csv")}
    source_map = {(r["orig_id"], r["sdir_id"]): r for r in m3_sources}
    cross_dataset = []
    record_map = {r["case_key"]: r for r in records}
    for row in public_rows:
        patient = row["canonical_patient_id"]
        if not patient:
            continue
        source_key = patient, row["public_series_number"]
        cross_dataset.append({"case_key": row["case_key"], "canonical_patient_id": patient,
                              "canonical_series_uid": row["canonical_series_uid"],
                              "M3_same_patient": patient in m3_sets, "M3_exact_source_number": source_key in source_map,
                              "M3_publisher_split": m3_sets.get(patient, ""),
                              "CTGAN_draft_split": record_map[row["case_key"]]["split"],
                              "publisher_split_conflict": patient in m3_sets and
                                  {"valid": "validation"}.get(m3_sets[patient], m3_sets[patient]) != record_map[row["case_key"]]["split"]})
    identities_complete = all(r["canonical_patient_id"] for r in public_rows)
    summary = {"schema": "source_pixel_verification_v1", "all_cases_pixel_checked": len(cases) == len(records),
               "case_count": len(cases), "source_DICOM_files_checked": len(facts),
               "archive_member_comparison": archive, "shared_source_pixel_or_public_patient_edges": edges,
               "approximate_similarity_candidates": candidates,
               "public_identity_resolved": sum(bool(r["canonical_patient_id"]) for r in public_rows),
               "public_identity_unresolved": sum(not r["canonical_patient_id"] for r in public_rows),
               "cross_dataset_source_links": cross_dataset,
               "cross_partition_source_evidence": any(r["cross_partition_source_evidence"] for r in edges),
               "cross_dataset_publisher_split_conflicts": sum(r["publisher_split_conflict"] for r in cross_dataset),
               "complete_CTGAN_to_LIDC_mapping": identities_complete,
               "original_author_Drive_completeness_verified": False,
               "annotation_axis_visual_review_completed": False,
               "training_ready": False, "goal_verification_complete": False,
               "model_loaded": False, "training_performed": False, "test_performance_evaluated": False,
               "test_pixels_read_for_data_integrity_only": any(r["split"] == "test" for r in records),
               "historical_manifests_modified": False,
               "input_sha256": {str(Path(p)): sha256_file(Path(p)) for p in
                   (manifest_path, files_path, catalog_path, metadata_summary_path)}}
    columns = sorted(set().union(*(r.keys() for r in public_rows)))
    write_csv(output / "public_source_mapping.csv", columns, public_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def package_review_outputs(run_root, output_path):
    """Package completed audit metadata only for a user-controlled local download."""
    root, output = Path(run_root).resolve(), Path(output_path).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("Use a new review ZIP path outside the audit input directory")
    required = {"metadata/summary.json", "metadata/ctgan_reviewed_cases.csv", "metadata/ctgan_annotations.csv",
                "metadata/m3dsynth_lineage.csv", "metadata/m3dsynth_source_groups.csv",
                "source_pixels/summary.json", "source_pixels/public_source_mapping.csv", "notebook_receipt.json"}
    selected = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in required or re.fullmatch(r"source_pixels/exp[12]_[0-9]+\.json", relative):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError("Invalid audit metadata file")
            selected.append((relative, path))
    if not required.issubset({name for name, _ in selected}):
        raise ValueError("A completed metadata/source audit and notebook receipt are required")
    metadata = json.loads((root/"metadata/summary.json").read_text(encoding="utf-8"))
    source = json.loads((root/"source_pixels/summary.json").read_text(encoding="utf-8"))
    receipt = json.loads((root/"notebook_receipt.json").read_text(encoding="utf-8"))
    if (metadata.get("metadata_checks_passed") is not True or source.get("all_cases_pixel_checked") is not True
            or receipt.get("all_cases_pixel_checked") is not True
            or receipt.get("metadata_summary_sha256") != sha256_file(root/"metadata/summary.json")
            or receipt.get("source_summary_sha256") != sha256_file(root/"source_pixels/summary.json")):
        raise ValueError("Completed audit reports do not match the notebook receipt")
    if any(report.get("training_ready") is not False or report.get("goal_verification_complete") is not False
           for report in (metadata, source, receipt)):
        raise ValueError("Engineering audit must not silently certify research readiness")
    case_files = [name for name, _ in selected if re.fullmatch(r"source_pixels/exp[12]_[0-9]+\.json", name)]
    if len(case_files) != source["case_count"]:
        raise ValueError("Per-case source audit files are incomplete")
    manifest = {"schema": "metadata_only_data_audit_review_package_v1", "case_count": source["case_count"],
                "medical_images_included": False, "checkpoints_included": False,
                "training_ready": False, "goal_verification_complete": False,
                "files": {name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in selected}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in selected:
            archive.write(path, name)
        archive.writestr("review_package_manifest.json", json.dumps(manifest, indent=2)+"\n")
    return {"archive": str(output), "archive_sha256": sha256_file(output), "bytes": output.stat().st_size,
            "files": len(selected)+1, "medical_images_included": False, "goal_verification_complete": False}
