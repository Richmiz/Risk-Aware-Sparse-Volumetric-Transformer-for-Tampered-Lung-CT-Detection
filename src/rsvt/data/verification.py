"""Reproducible publisher-label, header, source-group and lineage verification.

This audit reads metadata only. It never upgrades historical manifests or claims
pixel equivalence, patient independence, clinical diagnosis, or training readiness.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

from rsvt.data.acquisition import sha256_file
from rsvt.data.inventory import write_csv
from rsvt.data.splits import CLASSES, digest_json


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def scan_key(relative):
    path = PurePosixPath(str(relative).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe source path")
    matches = [short for long, short in (("Experiment 1 - Blind", "exp1"),
               ("Experiment 2 - Open", "exp2")) if long in path.parts]
    if len(matches) != 1 or not path.name.isdecimal():
        raise ValueError("Unrecognized CT-GAN scan directory")
    return matches[0] + ":" + path.name


def _vector(value, size):
    result = np.asarray(ast.literal_eval(value), dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Invalid header geometry vector")
    return result


def verify_labels(label_paths, policy):
    """Preserve duplicates and zero sentinels, and validate publisher scan counts."""
    annotated, grouped, seen = [], defaultdict(list), set()
    for experiment, path in sorted(label_paths.items()):
        if experiment not in policy["expected_scan_counts"]:
            raise ValueError("Unknown publisher experiment")
        if sha256_file(Path(path)) != policy["label_file_sha256"][experiment]:
            raise ValueError("Label table differs from the reviewed publisher table")
        for row_number, row in enumerate(read_csv(path), start=2):
            label, scan = row["type"].strip(), row["uuid"].strip()
            if label not in CLASSES or not scan.isdecimal():
                raise ValueError("Invalid publisher label or scan ID")
            coordinates = tuple(int(row[k]) for k in ("slice", "x", "y"))
            if min(coordinates) < 0:
                raise ValueError("Negative publisher coordinate")
            zero = coordinates == (0, 0, 0)
            if zero and label != "TB":
                raise ValueError("Unexpected zero-location class; explicit review required")
            identity = (experiment, scan, label, *coordinates)
            item = {"case_key": experiment + ":" + scan, "label": label,
                    "source_row": row_number, "source_table": Path(path).name,
                    "slice": coordinates[0], "x": coordinates[1], "y": coordinates[2],
                    "duplicate": identity in seen, "zero_location_placeholder": zero,
                    "localization_eligible": False}
            seen.add(identity)
            annotated.append(item)
            grouped[item["case_key"]].append(item)
    scans = {}
    for key, rows in sorted(grouped.items()):
        labels = {r["label"] for r in rows}
        if len(labels) != 1:
            raise ValueError("Mixed location labels cannot become a single volume label")
        scans[key] = {"label": next(iter(labels)), "raw_rows": len(rows),
                      "unique_rows": sum(not r["duplicate"] for r in rows),
                      "nonzero_unique_points": sum(not r["duplicate"] and not r["zero_location_placeholder"] for r in rows)}
    actual = {experiment: dict(Counter(v["label"] for k, v in scans.items()
              if k.startswith(experiment + ":"))) for experiment in label_paths}
    if actual != policy["expected_scan_counts"]:
        raise ValueError("Scan-level labels do not reproduce publisher class counts")
    return scans, annotated


def verify_headers(files, sources, manifest_records):
    """Recompute file-set binding and header consistency for every original scan."""
    files_by_scan = defaultdict(list)
    for row in files:
        if row["dataset"] == "ct_gan" and row["kind"] == "dicom":
            directory = PurePosixPath(row["relative_path"].replace("\\", "/")).parent
            files_by_scan[scan_key(directory)].append(row)
    sources_by_key = {scan_key(r["relative_directory"]): r for r in sources if r["dataset"] == "ct_gan"}
    manifest_by_key = {r["case_key"]: r for r in manifest_records}
    if (len(manifest_by_key) != len(manifest_records) or len(sources_by_key) != len(sources)
            or set(files_by_scan) != set(sources_by_key) or set(files_by_scan) != set(manifest_by_key)):
        raise ValueError("File, series and manifest scan coverage differs or repeats")
    profiles, geometry_candidates = {}, defaultdict(list)
    for key, rows in sorted(files_by_scan.items()):
        source, record = sources_by_key[key], manifest_by_key[key]
        if any(r["header_status"] != "read" or r["modality"] != "CT" for r in rows):
            raise ValueError("Unreadable or non-CT source header")
        digests = sorted(r["sha256"] for r in rows)
        import hashlib
        fingerprint = hashlib.sha256("\n".join(digests).encode()).hexdigest()
        if (fingerprint != source["file_set_sha256"] or fingerprint != record["source_file_set_sha256"]
                or source["candidate_id"] != record["source_candidate_id"] or len(rows) != int(source["file_count"])):
            raise ValueError("Source file set differs from series/F03 evidence")
        for field, source_field in (("patient_metadata_key", "patient_metadata_keys"),
                                    ("series_metadata_key", "series_metadata_key")):
            values = {r[field] for r in rows}
            if len(values) != 1 or "" in values or next(iter(values)) != record[source_field]:
                raise ValueError("Inconsistent patient/series metadata within scan")
        if len({r["sop_metadata_key"] for r in rows}) != len(rows) or any(not r["sop_metadata_key"] for r in rows):
            raise ValueError("Missing or repeated SOP identity within a scan")
        indexed = {}
        for row in rows:
            stem = PurePosixPath(row["relative_path"].replace("\\", "/")).stem
            if not stem.isdecimal() or int(stem) in indexed:
                raise ValueError("Ambiguous numeric source filenames")
            indexed[int(stem)] = row
        indices = sorted(indexed)
        if indices != list(range(len(rows))):
            raise ValueError("Source slice filenames are not contiguous from zero")
        first = rows[0]
        direction = _vector(first["image_orientation"], 6)
        normal = np.cross(direction[:3], direction[3:])
        if not np.isclose(np.linalg.norm(normal), 1., atol=1e-4):
            raise ValueError("Invalid DICOM slice normal")
        positions = []
        for row in rows:
            if (not np.allclose(_vector(row["image_orientation"], 6), direction, atol=1e-4)
                    or row["rows"] != first["rows"] or row["columns"] != first["columns"]
                    or not np.allclose(_vector(row["pixel_spacing"], 2), _vector(first["pixel_spacing"], 2), atol=1e-5)):
                raise ValueError("Inconsistent source dimensions/orientation/spacing")
            positions.append(float(_vector(row["image_position"], 3) @ normal))
        positions = sorted(positions)
        gaps = np.diff(positions)
        if not len(gaps) or min(gaps) <= 0 or not np.allclose(gaps, np.median(gaps), atol=.05, rtol=.01):
            raise ValueError("Repeated or irregular source slice planes")
        profile = {"case_key": key, "slice_count": len(rows), "filename_start": indices[0],
                   "filename_end": indices[-1], "rows": int(first["rows"]), "columns": int(first["columns"]),
                   "spacing_z_mm": float(np.median(gaps)), "header_geometry_verified": True,
                   "source_file_set_verified": True, "pixel_data_reverified": False}
        profiles[key] = profile
        # Geometry similarity is a review candidate, never a source-identity edge.
        signature = digest_json([len(rows), first["rows"], first["columns"],
                                  np.round(_vector(first["pixel_spacing"], 2), 5).tolist(),
                                  round(float(np.median(gaps)), 4), np.round(direction, 5).tolist(),
                                  np.round(positions, 3).tolist()])
        geometry_candidates[signature].append(key)
    return profiles, files_by_scan, [v for v in geometry_candidates.values() if len(v) > 1]


def source_edges(files_by_scan, manifest_records):
    evidence = defaultdict(set)
    records = {r["case_key"]: r for r in manifest_records}
    for key, rows in files_by_scan.items():
        for field in ("patient_metadata_key", "study_metadata_key", "series_metadata_key", "sop_metadata_key", "sha256"):
            for row in rows:
                if row[field]:
                    evidence[field, row[field]].add(key)
    pairs = defaultdict(Counter)
    for (field, _), members in evidence.items():
        for pair in combinations(sorted(members), 2):
            pairs[pair][field] += 1
    output = []
    for (first, second), counts in sorted(pairs.items()):
        output.append({"first": first, "second": second, "evidence_counts": dict(counts),
                       "labels": [records[first]["label_candidate"], records[second]["label_candidate"]],
                       "same_draft_group": records[first]["source_group_candidate"] == records[second]["source_group_candidate"],
                       "partitions": sorted({records[first]["split"], records[second]["split"]}),
                       "identity_status": "shared_source_evidence_not_full_pixel_identity"})
    return output


def verify_m3_lineage(metadata_dir, catalog=None):
    root = Path(metadata_dir)
    data, sources, partitions, centers = [read_csv(root / n) for n in ("data.csv", "LIDC.csv", "sets.csv", "centers.csv")]
    source_map = {(r["orig_id"], r["sdir_id"]): r for r in sources}
    sets = {r["orig_id"]: r["set"] for r in partitions}
    if len(source_map) != len(sources) or len(sets) != len(partitions) or not set(sets.values()).issubset({"train", "valid", "test"}):
        raise ValueError("M3Dsynth source or patient split mapping is ambiguous")
    for source, row in source_map.items():
        path = PurePosixPath(row["dicom"].replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or path.parts[0] != source[0] or path.name.split(".")[0] != source[1]:
            raise ValueError("M3Dsynth source directory conflicts with patient/series identity")
    public_sources = defaultdict(list)
    for row in catalog or []:
        if row.get("SeriesNumber") is not None and row.get("Collection") == "LIDC-IDRI":
            public_sources[row["PatientID"], str(row["SeriesNumber"])].append(row)
    public_map = {key: public_sources[key][0]["SeriesInstanceUID"] for key in source_map if len(public_sources[key]) == 1}
    seen, lineage, members, per_image = set(), [], defaultdict(list), defaultdict(set)
    for row in data:
        key, source = (row["mod"], row["img_id"]), (row["orig_id"], row["sdir_id"])
        if key in seen or source not in source_map or row["orig_id"] not in sets:
            raise ValueError("M3Dsynth duplicate sample or unresolved source")
        if row["mod"] not in {"real", "cycle", "pix2pix", "diffusion"} or row["ty"] not in {"inj", "rem"}:
            raise ValueError("Unknown M3Dsynth operation/generator")
        if any(not math.isfinite(float(row[k])) for k in ("coord_z", "coord_y", "coord_x")):
            raise ValueError("Nonfinite M3Dsynth source coordinate")
        seen.add(key)
        per_image[row["img_id"]].add((row["ty"], row["orig_id"], row["sdir_id"], row["coord_z"], row["coord_y"], row["coord_x"]))
        item = {"sample_id": "m3dsynth:" + row["mod"] + ":" + row["img_id"],
                "canonical_patient_id": row["orig_id"], "publisher_source_id": ":".join(source),
                "canonical_series_uid": public_map.get(source, ""),
                "source_dicom_directory": source_map[source]["dicom"],
                "generator": "" if row["mod"] == "real" else row["mod"], "operation": row["ty"],
                "forensic_label_from_operation": "" if row["mod"] == "real" else "FB" if row["ty"] == "rem" else "FM",
                "publisher_split": sets[row["orig_id"]], "identity_basis": "publisher_LIDC_lineage",
                "clinical_label_verified": False, "image_verified": False}
        lineage.append(item)
        members[source].append(item)
    if any(len(identities) != 1 for identities in per_image.values()):
        raise ValueError("M3Dsynth generator variants disagree about source/operation/coordinates")
    real_ids = {row["img_id"] for row in data if row["mod"] == "real"}
    if any(row["img_id"] not in real_ids for row in data if row["mod"] != "real"):
        raise ValueError("M3Dsynth manipulated sample lacks its real reference row")
    center_keys = [(r["mod"], r["img_id"]) for r in centers]
    if len(set(center_keys)) != len(center_keys):
        raise ValueError("M3Dsynth center table has duplicate sample IDs")
    orphan_centers = [r for r in centers if (r["mod"], r["img_id"]) not in seen]
    center_set = set(center_keys)
    missing_test_centers = [r["sample_id"] for r, original in zip(lineage, data)
                            if r["publisher_split"] == "test" and (original["mod"], original["img_id"]) not in center_set]
    if missing_test_centers:
        raise ValueError("M3Dsynth test samples lack publisher evaluation centers")
    grouped = [{"publisher_source_id": ":".join(k), "canonical_patient_id": k[0],
                "canonical_series_uid": public_map.get(k, ""),
                "sample_rows": len(v), "real_reference_rows": sum(not r["generator"] for r in v),
                "generators": ";".join(sorted({r["generator"] for r in v if r["generator"]})),
                "publisher_split": v[0]["publisher_split"]} for k, v in sorted(members.items())]
    summary = {"metadata_rows": len(data), "patient_ids": len(sets), "source_series": len(members),
               "by_variant": dict(Counter(r["mod"] for r in data)),
               "publisher_patient_split_counts": dict(Counter(sets.values())),
               "shared_multiple_generators": sum(len({r["generator"] for r in v if r["generator"]}) > 1 for v in members.values()),
               "center_rows": len(centers), "orphan_center_rows": orphan_centers,
               "public_TCIA_series_matches": len(public_map),
               "public_TCIA_source_mapping_complete": bool(catalog) and len(public_map) == len(source_map),
               "orphan_center_policy": "retained_as_publisher_inconsistency_excluded_from_sample_lineage",
               "missing_test_centers": missing_test_centers, "clinical_labels_inferred_for_real": 0,
               "image_availability_verified": False, "cross_dataset_CTGAN_mapping_complete": False}
    return summary, lineage, grouped


def run_metadata_verification(*, manifest_path, files_path, series_path, overlap_path,
                              label_root, readme_path, m3_metadata_dir, tcia_catalog_path, policy_path, output_dir):
    """Audit all scans and write a separate evidence package; originals stay intact."""
    paths = [Path(p) for p in (manifest_path, files_path, series_path, overlap_path, policy_path, readme_path, tcia_catalog_path)]
    labels = {e: Path(label_root) / ("labels_" + e + ".csv") for e in ("exp1", "exp2")}
    paths += list(labels.values()) + [Path(m3_metadata_dir) / n for n in ("data.csv", "LIDC.csv", "sets.csv", "centers.csv")]
    output = Path(output_dir).resolve()
    if output.exists() or any(p.resolve().is_relative_to(output) for p in paths):
        raise ValueError("Use a new audit output directory separate from the evidence")
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if sha256_file(Path(readme_path)) != policy["source"]["readme_sha256"]:
        raise ValueError("Publisher README differs from reviewed evidence")
    # Bind independent downloads to receipts in the actual historical manifest.
    for path, suffix in ((series_path, "/files/series_candidates.csv"), (overlap_path, "/files/overlap.json")):
        expected = [v for k, v in manifest["input_receipts"].items() if k.replace("\\", "/").endswith(suffix)]
        if len(expected) != 1 or sha256_file(Path(path)) != expected[0]:
            raise ValueError("Downloaded evidence differs from F03 receipts")
    records = manifest["records"]
    scanned, annotation_rows = verify_labels(labels, policy)
    if set(scanned) != {r["case_key"] for r in records}:
        raise ValueError("Publisher and F03 case coverage differs")
    files, sources = read_csv(files_path), read_csv(series_path)
    profiles, by_scan, geometry_candidates = verify_headers(files, sources, records)
    for experiment, path in labels.items():
        matches = {r["sha256"] for r in files if r["dataset"] == "ct_gan"
                   and r["relative_path"].replace("\\", "/") == "Tampered Scans/" + Path(path).name}
        if matches != {sha256_file(path)}:
            raise ValueError("Local label file does not match the remote raw inventory")
    reviewed = []
    for record in sorted(records, key=lambda r: r["case_key"]):
        key = record["case_key"]
        if scanned[key]["label"] != record["label_candidate"]:
            raise ValueError("F03 target differs from publisher label")
        reviewed.append({"case_key": key, "publisher_label": scanned[key]["label"],
                         "publisher_volume_label_verified": True, **scanned[key], **profiles[key],
                         "draft_split": record["split"], "source_group_candidate": record["source_group_candidate"],
                         "canonical_patient_id": "", "canonical_source_id": "",
                         "identity_reconciled": False, "classification_eligible": False,
                         "localization_eligible": False})
    for row in annotation_rows:
        members = by_scan[row["case_key"]]
        names = {int(PurePosixPath(r["relative_path"].replace("\\", "/")).stem): r for r in members}
        if row["zero_location_placeholder"]:
            row["coordinate_status"] = "absent_location_TB_placeholder"
        elif row["slice"] not in names:
            row["coordinate_status"] = "source_slice_missing"
        else:
            member = names[row["slice"]]
            inside = 0 <= row["x"] < int(member["columns"]) and 0 <= row["y"] < int(member["rows"])
            row["coordinate_status"] = "filename_and_zero_based_bounds_pass_visual_pending" if inside else "coordinate_out_of_bounds"
    edges = source_edges(by_scan, records)
    catalog = json.loads(Path(tcia_catalog_path).read_text(encoding="utf-8"))
    m3, lineage, source_groups = verify_m3_lineage(m3_metadata_dir, catalog)
    actual_slices = {e: sum(p["slice_count"] for k, p in profiles.items() if k.startswith(e + ":")) for e in labels}
    checks = {"all_publisher_volume_labels_match": True, "all_source_sets_match_F03": True,
              "all_headers_and_filename_sequences_consistent": True,
              "all_known_shared_sources_within_one_partition": all(len(e["partitions"]) == 1 for e in edges),
              "all_known_shared_sources_in_same_F03_group": all(e["same_draft_group"] for e in edges),
              "all_M3_sources_uniquely_match_public_TCIA_series": m3["public_TCIA_source_mapping_complete"],
              "all_nonzero_points_within_declared_slice_bounds": all(r["coordinate_status"] not in {"source_slice_missing", "coordinate_out_of_bounds"} for r in annotation_rows)}
    summary = {"schema": "data_label_verification_v1", "scope": "complete_metadata_audit_pixel_and_cross_dataset_reconciliation_pending",
               "metadata_checks_passed": all(checks.values()), "checks": checks,
               "ctgan": {"volumes": len(reviewed), "class_counts": dict(Counter(r["publisher_label"] for r in reviewed)),
                          "annotation_rows": len(annotation_rows), "unique_annotation_rows": sum(not r["duplicate"] for r in annotation_rows),
                          "duplicate_rows": sum(r["duplicate"] for r in annotation_rows),
                          "zero_rows": sum(r["zero_location_placeholder"] for r in annotation_rows),
                          "unique_nonzero_points": sum(not r["duplicate"] and not r["zero_location_placeholder"] for r in annotation_rows),
                          "actual_slices": actual_slices, "readme_slices": policy["readme_slice_counts"],
                          "slice_difference_actual_minus_readme": {e: actual_slices[e] - policy["readme_slice_counts"][e] for e in labels},
                          "shared_source_pairs": edges, "source_groups_in_draft": len({r["source_group_candidate"] for r in records}),
                          "geometry_only_review_candidates": geometry_candidates,
                          "independent_patient_count_verified": False},
               "m3dsynth": m3, "training_ready": False, "goal_verification_complete": False,
               "pending": ["CTGAN-to-public-LIDC source mapping and cross-dataset split audit",
                           "pixel-content source verification for relabeled or rewritten UIDs",
                           "archive-member versus actual source comparison for README slice-count discrepancy",
                           "visual confirmation of annotation axes and index base"],
               "input_sha256": {str(p.resolve()): sha256_file(p) for p in paths}}
    output.mkdir(parents=True)
    for name, rows in (("ctgan_reviewed_cases.csv", reviewed), ("ctgan_annotations.csv", annotation_rows),
                       ("m3dsynth_lineage.csv", lineage), ("m3dsynth_source_groups.csv", source_groups)):
        write_csv(output / name, tuple(rows[0]), rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
