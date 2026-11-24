"""Audit author-supplied M3Dsynth metadata; metadata rows are not downloaded volumes."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from rsvt.data.acquisition import sha256_file
from rsvt.data.inventory import write_csv


def _table(path: Path, required: set[str]) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid metadata columns: {path.name}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty metadata: {path.name}")
    return rows


def audit_m3dsynth(metadata_dir: Path, dataset_root: Path, output_dir: Path) -> dict:
    """Preserve sample/source relations, operations, and publisher splits for review.

    Publisher train/valid/test sets are recorded as evidence, never silently adopted
    as this project's LOGO protocol. Coordinates refer to source metadata and require
    transformation checks before localization use. Real rows get no clinical label.
    """
    data = _table(metadata_dir / "data.csv", {"img_id", "ty", "mod", "orig_id", "sdir_id", "coord_z", "coord_y", "coord_x"})
    lineage = _table(metadata_dir / "LIDC.csv", {"orig_id", "sdir_id", "dicom"})
    partitions = _table(metadata_dir / "sets.csv", {"orig_id", "set"})
    sources = {}
    for row in lineage:
        key = (row["orig_id"], row["sdir_id"])
        if key in sources:
            raise ValueError(f"Duplicate LIDC source mapping: {key}")
        sources[key] = row
    publisher_sets = {}
    for row in partitions:
        if row["set"] not in {"train", "valid", "test"}:
            raise ValueError("Unknown publisher partition")
        if row["orig_id"] in publisher_sets:
            raise ValueError("Duplicate publisher patient partition")
        publisher_sets[row["orig_id"]] = row["set"]
    seen, output, groups = set(), [], defaultdict(set)
    for row in data:
        identity = (row["mod"], row["img_id"])
        source = (row["orig_id"], row["sdir_id"])
        if identity in seen:
            raise ValueError(f"Duplicate M3Dsynth sample: {identity}")
        seen.add(identity)
        if row["mod"] not in {"real", "cycle", "pix2pix", "diffusion"} or row["ty"] not in {"inj", "rem"}:
            raise ValueError("Unknown M3Dsynth generator or operation")
        if source not in sources or row["orig_id"] not in publisher_sets:
            raise ValueError("Unresolved source or publisher partition mapping")
        if not all(math.isfinite(float(row[key])) for key in ("coord_z", "coord_y", "coord_x")):
            raise ValueError("Non-finite source coordinate")
        scan = dataset_root / row["mod"] / "scan" / row["img_id"]
        label = dataset_root / row["mod"] / "label" / row["img_id"]
        item = {
            "sample_id": f"m3dsynth:{row['mod']}:{row['img_id']}",
            "author_patient_id": row["orig_id"],
            "author_source_id": f"{row['orig_id']}:{row['sdir_id']}",
            "author_image_id": row["img_id"], "variant": row["mod"],
            "generator_id": "" if row["mod"] == "real" else row["mod"],
            "operation": row["ty"],
            "proposed_forensic_label": "" if row["mod"] == "real" else "FB" if row["ty"] == "rem" else "FM",
            "publisher_split": publisher_sets[row["orig_id"]],
            "source_dicom_directory": sources[source]["dicom"],
            "scan_path": str(scan), "scan_directory_present": scan.is_dir(),
            "label_directory_present": label.is_dir(),
            "source_z": row["coord_z"], "source_y": row["coord_y"], "source_x": row["coord_x"],
            "review_status": "metadata_only_not_training_ready",
        }
        output.append(item)
        groups[source].add(row["mod"])
    summary = {
        "metadata_rows": len(output),
        "manipulated_rows": sum(bool(row["generator_id"]) for row in output),
        "real_reference_rows": sum(not row["generator_id"] for row in output),
        "author_patient_ids": len({row["author_patient_id"] for row in output}),
        "author_source_series": len(groups),
        "by_variant": dict(Counter(row["variant"] for row in output)),
        "by_variant_operation": dict(Counter(f"{row['variant']}:{row['operation']}" for row in output)),
        "sources_shared_by_multiple_manipulation_generators": sum(len(variants - {"real"}) > 1 for variants in groups.values()),
        "scan_directories_present": sum(row["scan_directory_present"] for row in output),
        "training_ready": False,
        "metadata_sha256": {name: sha256_file(metadata_dir / name) for name in ("data.csv", "LIDC.csv", "sets.csv")},
        "interpretation": "Author-declared LIDC lineage; no independent volume matching or clinical malignancy verification performed.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "metadata_inventory.csv", list(output[0]), output)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
