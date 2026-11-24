"""Audit declared identities; this cannot replace a provenance or image-similarity audit."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from rsvt.data.labels import MANIPULATION_BY_CLASS

MANIFEST_COLUMNS = (
    "sample_id", "dataset", "patient_id", "source_volume_id", "series_uid",
    "volume_path", "role", "label", "generator_id", "manipulation_type",
    "annotation_type", "annotation_path", "split", "sha256",
)


@dataclass
class ManifestAudit:
    sample_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def audit_manifest(
    manifest_path: str | Path,
    project_root: str | Path,
    *,
    check_paths: bool = False,
    held_out_generator: str | None = None,
) -> ManifestAudit:
    """Check schema, label semantics, grouped leakage, and optional held-out identity."""
    result = ManifestAudit()
    root = Path(project_root).resolve()
    seen_samples: set[str] = set()
    group_splits: dict[tuple[str, str], set[str]] = {}
    source_patients: dict[str, set[str]] = {}
    held_out_count = 0
    missing_hash_count = 0

    with Path(manifest_path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            result.errors.append(f"Missing columns: {', '.join(sorted(missing))}")
            return result
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            result.errors.append("Duplicate column names")
            return result
        for line, record in enumerate(reader, start=2):
            result.sample_count += 1
            prefix = f"Row {line}: "
            if None in record:
                result.errors.append(prefix + "Too many CSV fields")
            row = {key: (record.get(key) or "").strip() for key in MANIFEST_COLUMNS}
            for key in ("sample_id", "dataset", "patient_id", "source_volume_id", "volume_path"):
                if not row[key]:
                    result.errors.append(prefix + f"{key} is required")
            if row["sample_id"] in seen_samples:
                result.errors.append(prefix + "Duplicate sample_id")
            seen_samples.add(row["sample_id"])
            if row["split"] not in {"train", "validation", "test"}:
                result.errors.append(prefix + "split must be train, validation, or test")

            role = row["role"]
            label = row["label"]
            if role == "forensic":
                if label not in MANIPULATION_BY_CLASS:
                    result.errors.append(prefix + "Forensic label must be TB, TM, FB, or FM")
                elif row["manipulation_type"] != MANIPULATION_BY_CLASS[label]:
                    result.errors.append(prefix + "Label and manipulation_type disagree")
                if label in {"FB", "FM"} and not row["generator_id"]:
                    result.errors.append(prefix + "Manipulated samples need a generator_id")
                if label in {"TB", "TM"} and row["generator_id"]:
                    result.errors.append(prefix + "Authentic samples must have blank generator_id")
            elif role == "anatomy_reference":
                if label or row["generator_id"] or row["manipulation_type"] != "none":
                    result.errors.append(prefix + "Anatomy references need blank label/generator and manipulation_type=none")
            else:
                result.errors.append(prefix + "role must be forensic or anatomy_reference")

            annotation_type = row["annotation_type"]
            if annotation_type not in {"none", "point", "box", "mask"}:
                result.errors.append(prefix + "Invalid annotation_type")
            if annotation_type in {"point", "box", "mask"} and not row["annotation_path"]:
                result.errors.append(prefix + "Spatial annotations need annotation_path")
            if annotation_type == "none" and row["annotation_path"]:
                result.errors.append(prefix + "annotation_type=none must have blank annotation_path")
            if row["sha256"] and not re.fullmatch(r"[0-9a-fA-F]{64}", row["sha256"]):
                result.errors.append(prefix + "sha256 must be a 64-character hexadecimal digest")

            for key in ("patient_id", "source_volume_id", "sha256"):
                if row[key]:
                    identity = row[key].lower() if key == "sha256" else row[key]
                    group_splits.setdefault((key, identity), set()).add(row["split"])
            if row["source_volume_id"] and row["patient_id"]:
                source_patients.setdefault(row["source_volume_id"], set()).add(row["patient_id"])
            if not row["sha256"]:
                missing_hash_count += 1

            if held_out_generator and row["generator_id"] == held_out_generator:
                held_out_count += 1
                if row["split"] != "test":
                    result.errors.append(prefix + "Held-out generator appears outside test")
            if check_paths:
                for key in ("volume_path", "annotation_path"):
                    if row[key] and not (root / row[key]).exists():
                        result.errors.append(prefix + f"{key} does not exist: {row[key]}")

    if not result.sample_count:
        result.errors.append("Manifest is empty; add audited samples before experimentation")
    for (kind, identity), splits in group_splits.items():
        if len(splits) > 1:
            result.errors.append(f"Partition leakage: {kind}={identity} in {', '.join(sorted(splits))}")
    for source, patients in source_patients.items():
        if len(patients) > 1:
            result.errors.append(f"Source volume {source} maps to multiple patient identities")
    if held_out_generator and held_out_count == 0:
        result.errors.append(f"Held-out generator not found: {held_out_generator}")
    if missing_hash_count:
        result.warnings.append(f"{missing_hash_count} samples lack hashes; duplicate-file checks are incomplete")
    result.warnings.append("Canonical source/patient identities and spatial annotations require independent verification")
    return result
