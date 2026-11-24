"""Inventory files and DICOM header groups without inferring clinical/source labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import tomllib

from rsvt.data.acquisition import sha256_file
from rsvt.data.manifest import MANIFEST_COLUMNS

FILE_COLUMNS = (
    "file_id", "dataset", "relative_path", "kind", "bytes", "sha256", "header_status",
    "patient_metadata_key", "study_metadata_key", "series_metadata_key", "sop_metadata_key",
    "modality", "rows", "columns", "frames", "pixel_spacing", "slice_thickness",
    "image_position", "image_orientation", "rescale_slope", "rescale_intercept", "error",
)
SERIES_COLUMNS = (
    "candidate_id", "dataset", "relative_directory", "series_metadata_key",
    "patient_metadata_keys", "file_count", "file_set_sha256", "review_status",
)


def write_csv(path: Path, columns, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _identity_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            with path.open("xb") as stream:
                stream.write(secrets.token_bytes(32))
        except FileExistsError:
            pass
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError("Inventory identity key must contain exactly 32 bytes")
    return key


def _token(key: bytes, namespace: str, value: str) -> str:
    if not value:
        return ""
    return hmac.new(key, f"{namespace}\0{value}".encode(), hashlib.sha256).hexdigest()


def file_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".dcm", ".dicom", ".ima")):
        return "dicom"
    if name.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".tif", ".tiff", ".npy", ".npz")):
        return "array_candidate"
    if name.endswith((".zip", ".tgz", ".tar", ".tar.gz", ".7z")):
        return "archive"
    if name.endswith((".csv", ".tsv", ".xml", ".json", ".xls", ".xlsx")):
        return "metadata_candidate"
    with path.open("rb") as stream:
        stream.seek(128)
        if stream.read(4) == b"DICM":
            return "dicom"
    return "other"


def _read_header(path: Path, dataset: str, key: bytes) -> dict:
    import pydicom

    fields = [
        "PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "Modality",
        "Rows", "Columns", "NumberOfFrames", "PixelSpacing", "SliceThickness",
        "ImagePositionPatient", "ImageOrientationPatient", "RescaleSlope", "RescaleIntercept",
    ]
    try:
        header = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=fields)
    except Exception as exc:
        # Normalize parser errors without copying potentially identifying tag values.
        raise ValueError("DICOM header cannot be read") from exc
    result = {"header_status": "read"}
    for tag, output, namespace in (
        ("PatientID", "patient_metadata_key", f"patient:{dataset}"),
        ("StudyInstanceUID", "study_metadata_key", "study"),
        ("SeriesInstanceUID", "series_metadata_key", "series"),
        ("SOPInstanceUID", "sop_metadata_key", "sop"),
    ):
        result[output] = _token(key, namespace, str(getattr(header, tag, "")))
    for tag, output in (
        ("Modality", "modality"), ("Rows", "rows"), ("Columns", "columns"),
        ("NumberOfFrames", "frames"), ("PixelSpacing", "pixel_spacing"),
        ("SliceThickness", "slice_thickness"), ("ImagePositionPatient", "image_position"),
        ("ImageOrientationPatient", "image_orientation"),
        ("RescaleSlope", "rescale_slope"), ("RescaleIntercept", "rescale_intercept"),
    ):
        value = getattr(header, tag, "")
        result[output] = str(value)
    return result


def inventory_data(config_path: Path, project_root: Path, output_dir: Path) -> dict:
    """Write file inventory, candidate groups, overlap evidence, and an unready manifest.

    PatientID keys are dataset-scoped; UID keys are global candidate overlap evidence.
    Neither represents reconciled canonical identity. Pixel data is not decoded.
    """
    root = project_root.resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    settings, datasets = config.get("inventory", {}), config.get("datasets", {})
    if not isinstance(settings, dict) or not isinstance(datasets, dict) or not datasets:
        raise ValueError("Inventory config needs inventory and datasets tables")
    for flag in ("hash_files", "read_dicom_headers"):
        if not isinstance(settings.get(flag, True), bool):
            raise ValueError(f"inventory.{flag} must be boolean")
    output_dir = output_dir.resolve()
    roots, symlink_roots = {}, {}
    for name, spec in datasets.items():
        if not isinstance(spec, dict) or not spec.get("root"):
            raise ValueError(f"Missing dataset root: {name}")
        if spec.get("role") not in {"forensic", "anatomy_reference"}:
            raise ValueError(f"Invalid inventory role: {name}")
        configured_root = root / spec["root"]
        symlink_roots[name] = configured_root.is_symlink()
        location = configured_root.resolve()
        if output_dir.is_relative_to(location) or location.is_relative_to(output_dir):
            raise ValueError("Inventory output must be outside dataset roots")
        if any(location.is_relative_to(existing) or existing.is_relative_to(location)
               for existing in roots.values()):
            raise ValueError("Dataset roots must not overlap")
        roots[name] = location
    key_path = (root / settings.get("identity_key_file", "data/private/inventory_identity.key")).resolve()
    if not key_path.is_relative_to(root):
        raise ValueError("Identity key must be stored inside the project root")
    if any(key_path.is_relative_to(location) for location in roots.values()):
        raise ValueError("Identity key must be stored outside dataset roots")
    output_dir.mkdir(parents=True, exist_ok=False)
    key = _identity_key(key_path)
    rows, errors, summaries = [], [], {}
    for name, location in roots.items():
        before = len(rows)
        if symlink_roots[name]:
            errors.append({"dataset": name, "error": "symlink_root_skipped"})
        elif location.is_dir():
            def walk_error(error):
                errors.append({"dataset": name, "error": type(error).__name__})
            for directory, dirs, filenames in os.walk(location, followlinks=False, onerror=walk_error):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(directory) / d).is_symlink())
                for filename in sorted(filenames):
                    path = Path(directory) / filename
                    if filename.startswith(".") or path.is_symlink():
                        continue
                    relative = path.relative_to(location).as_posix()
                    row = dict.fromkeys(FILE_COLUMNS, "")
                    row.update(file_id=hashlib.sha256(f"{name}/{relative}".encode()).hexdigest(),
                               dataset=name, relative_path=relative, header_status="not_applicable")
                    try:
                        initial = path.stat()
                        row["bytes"] = initial.st_size
                        row["kind"] = file_kind(path)
                        if settings.get("hash_files", True):
                            row["sha256"] = sha256_file(path)
                        if row["kind"] == "dicom":
                            if settings.get("read_dicom_headers", True):
                                row.update(_read_header(path, name, key))
                            else:
                                row["header_status"] = "disabled"
                        final = path.stat()
                        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
                            row["sha256"] = ""
                            raise ValueError("file_changed_during_inventory")
                    except (OSError, ValueError, ImportError, EOFError) as exc:
                        row["error"] = type(exc).__name__
                        if row["kind"] == "dicom":
                            row["header_status"] = "error"
                        errors.append({"dataset": name, "relative_path": relative,
                                       "error": type(exc).__name__})
                    rows.append(row)
        own = rows[before:]
        image_files = sum(row["kind"] in {"dicom", "array_candidate"} for row in own)
        availability = "available_unverified" if image_files else "metadata_only" if own else "empty" if location.is_dir() else "missing"
        summaries[name] = {
            "root": str(location),
            "status": availability, "image_candidate_files": image_files,
            "files": len(own), "bytes": sum(int(r["bytes"] or 0) for r in own),
            "kinds": dict(Counter(r["kind"] for r in own)),
            "observed_patient_metadata_keys": len({r["patient_metadata_key"] for r in own if r["patient_metadata_key"]}),
            "independent_patient_count": None,
        }

    grouped = defaultdict(list)
    hashes = defaultdict(list)
    for row in rows:
        if row["sha256"]:
            hashes[row["sha256"]].append(row)
        if row["series_metadata_key"] and row["modality"] == "CT" and not row["error"]:
            grouped[(row["dataset"], str(Path(row["relative_path"]).parent), row["series_metadata_key"])].append(row)
    series, drafts = [], []
    for (dataset, directory, series_key), members in sorted(grouped.items()):
        candidate_id = hashlib.sha256(f"{dataset}/{directory}/{series_key}".encode()).hexdigest()
        complete_hashes = all(member["sha256"] for member in members)
        content_hash = hashlib.sha256("\n".join(sorted(m["sha256"] for m in members)).encode()).hexdigest() if complete_hashes else ""
        series.append(dict(zip(SERIES_COLUMNS, (
            candidate_id, dataset, directory, series_key,
            ";".join(sorted({m["patient_metadata_key"] for m in members if m["patient_metadata_key"]})),
            len(members), content_hash, "identity_and_completeness_unverified",
        ))))
        draft = dict.fromkeys(MANIFEST_COLUMNS, "")
        draft.update(sample_id=candidate_id, dataset=dataset,
                     volume_path=str(roots[dataset] / directory), role=datasets[dataset]["role"])
        if draft["role"] == "anatomy_reference":
            draft["manipulation_type"] = "none"
        drafts.append(draft)
    shared_uids = defaultdict(list)
    for group in series:
        shared_uids[group["series_metadata_key"]].append(group["candidate_id"])
    overlap = {
        "exact_file_duplicates": [{"sha256": digest, "files": [{"dataset": r["dataset"], "path": r["relative_path"]} for r in members]}
                                  for digest, members in hashes.items() if len(members) > 1],
        "shared_series_uid_candidates": [{"series_metadata_key": key, "candidates": candidates}
                                         for key, candidates in shared_uids.items() if len(candidates) > 1],
        "interpretation": "Evidence for review only; UID matches can include modified derivatives; no voxel-similarity audit performed.",
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": summaries, "files": len(rows), "ct_series_candidates": len(series),
        "metadata_errors": errors, "phase_f01_complete": False,
        "training_ready": False, "identity_key_fingerprint": hashlib.sha256(key).hexdigest(),
        "pending": ["reconcile canonical source/patient identities", "verify label and annotation levels",
                    "verify series completeness and pixel-level overlap", "define final partitions"],
    }
    write_csv(output_dir / "files.csv", FILE_COLUMNS, rows)
    write_csv(output_dir / "series_candidates.csv", SERIES_COLUMNS, series)
    write_csv(output_dir / "manifest_candidates.csv", MANIFEST_COLUMNS, drafts)
    for name, content in (("summary.json", summary), ("overlap.json", overlap)):
        (output_dir / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    lines = ["# Dataset inventory", "", "Inventory candidates require review before training.", "",
             "| Dataset | Status | Files | Bytes |", "|---|---|---:|---:|"]
    lines += [f"| {name} | {item['status']} | {item['files']} | {item['bytes']} |" for name, item in summaries.items()]
    lines += ["", f"CT series candidates: {len(series)}. Independent patient count remains unverified.",
              f"Read/hash errors: {len(errors)}. Training ready: false.", "",
              "See summary.json, files.csv, series_candidates.csv, overlap.json, and manifest_candidates.csv."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
