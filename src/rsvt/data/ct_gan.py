"""Extract a bounded CT-GAN pilot from a locally acquired UCI ZIP prefix."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath

from rsvt.data.acquisition import safe_member_path
from rsvt.data.inventory import write_csv


def extract_ctgan_prefix(archive_prefix: Path, destination: Path, *, max_bytes=512 * 1024**2) -> dict:
    """Extract metadata and the first contiguous DICOM directory, verifying member CRCs.

    A prefix cannot prove whole-archive integrity or series completeness. The output
    is a pilot acquisition only and must not be represented as the entire dataset.
    Snapshot the file length so a concurrent downloader cannot create a moving input.
    """
    from stream_unzip import stream_unzip, TruncatedDataError

    if destination.exists():
        raise FileExistsError(f"Pilot destination already exists: {destination}")
    limit = archive_prefix.stat().st_size
    destination.mkdir(parents=True)
    files = []
    total = 0
    first_series = None
    stopped_at_next_series = False
    consumed = 0
    prefix_digest = hashlib.sha256()

    def save(name: str, body) -> None:
        nonlocal total
        target = safe_member_path(destination, name)
        if name.endswith("/"):
            for _ in body:
                pass
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        digest, size = hashlib.sha256(), 0
        with partial.open("xb") as output:
            for chunk in body:
                total += len(chunk)
                size += len(chunk)
                if total > max_bytes:
                    raise ValueError("Pilot extraction exceeded byte limit")
                output.write(chunk)
                digest.update(chunk)
        partial.rename(target)
        files.append({"member": name, "bytes": size, "sha256": digest.hexdigest(),
                      "member_crc_verified": True})

    with archive_prefix.open("rb") as source:
        def chunks():
            nonlocal consumed
            while consumed < limit:
                chunk = source.read(min(65536, limit - consumed))
                if not chunk:
                    break
                consumed += len(chunk)
                prefix_digest.update(chunk)
                yield chunk
        try:
            for raw_name, _, body in stream_unzip(chunks()):
                name = raw_name.decode("utf-8")
                if name == "data.zip":
                    for inner_name, _, inner_body in stream_unzip(body):
                        member = inner_name.decode("utf-8")
                        if member.lower().endswith(".dcm"):
                            directory = str(PurePosixPath(member).parent)
                            if first_series is None:
                                first_series = directory
                            elif directory != first_series:
                                stopped_at_next_series = True
                                break
                            save(member, inner_body)
                        elif member.endswith("/"):
                            for _ in inner_body:
                                pass
                        elif member.lower().endswith((".csv", ".txt", ".docx")):
                            save(member, inner_body)
                        else:
                            for _ in inner_body:
                                pass
                    break
                else:
                    save(name, body)
        except TruncatedDataError:
            stopped_at_next_series = False
    receipt = {
        "source_url": "https://archive.ics.uci.edu/static/public/520/deepfakes%2Bmedical%2Bimage%2Btamper%2Bdetection.zip",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_prefix": str(archive_prefix), "snapshot_bytes": limit,
        "read_prefix_bytes": consumed, "read_prefix_sha256": prefix_digest.hexdigest(),
        "selected_directory": first_series, "next_series_boundary_seen": stopped_at_next_series,
        "dicom_files": sum(item["member"].lower().endswith(".dcm") for item in files),
        "whole_archive_checksum_verified": False, "series_completeness_verified": False,
        "training_ready": False, "files": files,
    }
    (destination / "pilot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def audit_ctgan_labels(dataset_root: Path, output_dir: Path) -> dict:
    """Preserve location labels and flag mixed-label scans; do not assign volume labels."""
    from collections import Counter, defaultdict
    import csv

    labels = []
    scan_groups = defaultdict(list)
    seen = set()
    sources = (
        ("exp1", "labels_exp1.csv", "Experiment 1 - Blind"),
        ("exp2", "labels_exp2.csv", "Experiment 2 - Open"),
    )
    for experiment, filename, directory in sources:
        table = dataset_root / "Tampered Scans" / filename
        with table.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"type", "uuid", "slice", "x", "y"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Invalid CT-GAN label columns: {filename}")
            count = 0
            for row in reader:
                count += 1
                category, scan = row["type"].strip(), row["uuid"].strip()
                if category not in {"TB", "TM", "FB", "FM"} or not scan.isdecimal():
                    raise ValueError("Invalid CT-GAN category or publisher scan ID")
                coords = tuple(int(row[key]) for key in ("slice", "x", "y"))
                if min(coords) < 0:
                    raise ValueError("CT-GAN coordinates must be nonnegative")
                identity = (experiment, scan, category, *coords)
                duplicate = identity in seen
                seen.add(identity)
                scan_key = f"{experiment}:{scan}"
                scan_path = dataset_root / "Tampered Scans" / directory / scan
                item = {
                    "publisher_scan_key": scan_key, "publisher_experiment": experiment,
                    "publisher_scan_id": scan, "location_label": category,
                    "slice_reference": coords[0], "pixel_x": coords[1], "pixel_y": coords[2],
                    "scan_path": str(scan_path), "scan_directory_present": scan_path.is_dir(),
                    "annotation_type": "point_candidate" if any(coords) else "unverified_zero_reference",
                    "duplicate_label_record": duplicate, "volume_label": "",
                    "review_status": "location_label_only_geometry_and_source_pending",
                }
                labels.append(item)
                scan_groups[scan_key].append(item)
            if not count:
                raise ValueError(f"Empty CT-GAN label file: {filename}")
    scans = []
    for key, members in sorted(scan_groups.items()):
        categories = sorted({row["location_label"] for row in members})
        scans.append({"publisher_scan_key": key, "label_rows": len(members),
                      "location_classes": ";".join(categories), "mixed_location_classes": len(categories) > 1,
                      "scan_directory_present": members[0]["scan_directory_present"],
                      "volume_label": "", "canonical_patient_id": "", "canonical_source_id": ""})
    summary = {
        "label_rows": len(labels), "distinct_label_records": len(seen),
        "duplicate_label_rows": sum(row["duplicate_label_record"] for row in labels),
        "zero_coordinate_rows": sum(row["annotation_type"] == "unverified_zero_reference" for row in labels),
        "publisher_scan_groups": len(scans),
        "location_class_counts": dict(Counter(row["location_label"] for row in labels)),
        "scan_groups_by_experiment": dict(Counter(key.split(":")[0] for key in scan_groups)),
        "mixed_location_class_scan_groups": sum(row["mixed_location_classes"] for row in scans),
        "scan_directories_present": sum(row["scan_directory_present"] for row in scans),
        "independent_patient_count": None, "training_ready": False,
        "interpretation": "Official location-level annotations; scan IDs and experiment partitions do not establish patient independence or four-class volume targets.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "location_labels.csv", list(labels[0]), labels)
    write_csv(output_dir / "scan_label_summary.csv", list(scans[0]), scans)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
