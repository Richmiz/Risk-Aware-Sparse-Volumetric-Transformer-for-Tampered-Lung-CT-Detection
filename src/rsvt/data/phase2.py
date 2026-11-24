"""F02 CT-GAN geometry audit and bounded preprocessing, without model training."""

from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gc
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tomllib
import numpy as np
import SimpleITK as sitk

from rsvt.data.inventory import write_csv
from rsvt.data.io import GeometryPolicy, inspect_series, read_hu_volume, source_fingerprint
from rsvt.data.preprocessing import PreprocessingSpec, preprocess_volume, transform_points, canonical_grid_geometry
from rsvt.data.intensity import series_intensity_evidence, sample_rescale_qa
from rsvt.data.cache import artifact_receipts, resume_index, validate_cache


def _json(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid columns in {path.name}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty evidence file: {path.name}")
    return rows


def load_preprocessing_config(path: Path):
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for key in ("geometry", "preprocessing", "annotations"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Missing config table: {key}")
    settings = dict(config["preprocessing"])
    for key in ("target_spacing_xyz_mm", "hu_window"):
        if key in settings:
            settings[key] = tuple(settings[key])
    try:
        policy = GeometryPolicy(**config["geometry"])
        spec = PreprocessingSpec(**settings)
    except TypeError as exc:
        raise ValueError("Unknown geometry/preprocessing parameter") from exc
    if config["annotations"].get("slice_reference") != "numeric_filename":
        raise ValueError("Only explicit numeric-filename slice mapping is implemented")
    base = config["annotations"].get("xy_index_base")
    if isinstance(base, bool) or base not in (0, 1):
        raise ValueError("xy_index_base must be 0 or 1")
    return config, policy, spec


def source_group_candidates(rows: list[dict]) -> dict[str, str]:
    """Conservatively union shared F01 patient/series/file-set evidence; no split yet."""
    ids = [row["candidate_id"] for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Missing/duplicate F01 candidate identity")
    parents = {item: item for item in ids}

    def find(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    known = {}
    for row in rows:
        for kind in ("patient_metadata_keys", "series_metadata_key", "file_set_sha256"):
            for value in row.get(kind, "").split(";"):
                if not value:
                    continue
                evidence = (kind, value)
                if evidence in known:
                    parents[find(row["candidate_id"])] = find(known[evidence])
                known[evidence] = row["candidate_id"]
    groups = defaultdict(list)
    for item in ids:
        groups[find(item)].append(item)
    result = {}
    for members in groups.values():
        key = "ctgan-source-candidate-" + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:20]
        result.update({item: key for item in members})
    return result


def map_ctgan_annotations(rows: list[dict], geometry, source, target, xy_index_base: int) -> list[dict]:
    """Filename-based candidates only; never invent masks or infer a sorted z ordinal."""
    slice_names = defaultdict(list)
    for index, record in enumerate(geometry.slices):
        if record.path.stem.isdecimal():
            slice_names[int(record.path.stem)].append(index)
    output = []
    for row in rows:
        item = {"label": row["location_label"], "source_slice_reference": row["slice_reference"],
                "source_x": row["pixel_x"], "source_y": row["pixel_y"],
                "supervision_eligible": False, "slice_mapping": "numeric_filename",
                "xy_index_base": xy_index_base, "status": "pending_visual_review"}
        try:
            ref, x, y = (int(row[k]) for k in ("slice_reference", "pixel_x", "pixel_y"))
            if not any((ref, x, y)):
                item["status"] = "zero_reference_not_localization_target"
            elif (row.get("duplicate_label_record") or "").strip().lower() == "true":
                item["status"] = "duplicate_label_record_not_repeated_supervision"
            elif len(slice_names.get(ref, [])) != 1:
                item["status"] = "slice_reference_missing_or_ambiguous"
            else:
                index = [x - xy_index_base, y - xy_index_base, slice_names[ref][0]]
                mapped = transform_points([index], source, target)
                item.update(native_index_xyz=index, physical_lps_mm=list(mapped["physical_lps_mm"][0]),
                            processed_index_xyz=list(mapped["target_index_xyz"][0]),
                            inside_processed_support=mapped["inside_target_support"][0],
                            semantic_role="manipulation_candidate" if row["location_label"] in {"FB", "FM"} else "authentic_reference")
        except ValueError:
            item["status"] = "invalid_or_out_of_bounds_coordinate"
        output.append(item)
    return output


def _case_location(root: Path, relative: str):
    parts = PurePosixPath(relative.replace("\\", "/"))
    if parts.is_absolute() or ".." in parts.parts:
        raise ValueError("Unsafe relative series directory")
    folder = (root / Path(*parts.parts)).resolve()
    if not folder.is_relative_to(root.resolve()):
        raise ValueError("Series directory escapes dataset root")
    experiment = "exp1" if "Experiment 1 - Blind" in parts.parts else "exp2" if "Experiment 2 - Open" in parts.parts else None
    if experiment is None or not parts.name.isdecimal():
        raise ValueError("Unknown CT-GAN directory layout")
    return folder, f"{experiment}:{parts.name}", f"{experiment}_{parts.name}"


def run_phase2(dataset_root: Path, inventory_run: Path, config_path: Path,
               output_dir: Path, processed_root: Path, *, max_cases: int = 4, progress=None,
               resume_runs=(), max_new_output_bytes: int | None = None) -> dict:
    """Audit every F01 CT series; decode/preprocess only the requested pilot or all (0)."""
    if isinstance(max_cases, bool) or max_cases < 0:
        raise ValueError("max_cases must be nonnegative; zero processes all valid cases")
    if max_new_output_bytes is not None and (isinstance(max_new_output_bytes, bool)
            or not isinstance(max_new_output_bytes, int) or max_new_output_bytes < 0):
        raise ValueError("Output byte budget must be a nonnegative integer or None")
    cache_index = resume_index(resume_runs)
    config, policy, spec = load_preprocessing_config(config_path)
    candidates = _csv(inventory_run / "files/series_candidates.csv",
                      {"candidate_id", "dataset", "relative_directory", "file_count", "file_set_sha256"})
    candidates = [row for row in candidates if row["dataset"] == "ct_gan"]
    if not candidates:
        raise ValueError("No CT-GAN candidates in F01 evidence")
    labels = _csv(inventory_run / "ct_gan_labels/location_labels.csv",
                  {"publisher_scan_key", "location_label", "slice_reference", "pixel_x", "pixel_y"})
    if not dataset_root.is_dir():
        raise ValueError("CT-GAN data is absent; reconnect the same Colab runtime or restore it remotely")
    dataset_root, output_dir, processed_root = dataset_root.resolve(), output_dir.resolve(), processed_root.resolve()
    for destination in (output_dir, processed_root):
        if destination.is_relative_to(dataset_root) or dataset_root.is_relative_to(destination):
            raise ValueError("F02 outputs must be outside original DICOM data")
    if output_dir == processed_root or output_dir.is_relative_to(processed_root) or processed_root.is_relative_to(output_dir):
        raise ValueError("Report and volume output directories must be separate")
    if output_dir.exists() or processed_root.exists():
        raise FileExistsError("Run outputs already exist; use a new run ID")
    output_dir.mkdir(parents=True)
    processed_root.mkdir(parents=True)
    _json(output_dir / "resolved_config.json", config)
    groups = source_group_candidates(candidates)
    by_scan = defaultdict(list)
    for row in labels:
        by_scan[row["publisher_scan_key"]].append(row)
    audits, available = [], []
    for row in sorted(candidates, key=lambda r: r["relative_directory"]):
        audit = {"candidate_id": row["candidate_id"], "directory": row["relative_directory"],
                 "source_group_candidate": groups[row["candidate_id"]],
                 "status": "rejected", "reason": "", "case_key": "", "slice_count": 0}
        try:
            folder, scan_key, slug = _case_location(dataset_root, row["relative_directory"])
            audit["case_key"] = scan_key
            geometry = inspect_series(folder, policy)
            if len(geometry.slices) != int(row["file_count"]):
                raise ValueError("Slice count differs from F01")
            audit.update(status="geometry_valid", slice_count=len(geometry.slices),
                         native_geometry=geometry.as_dict(), label_record_count=len(by_scan[scan_key]),
                         intensity_evidence=series_intensity_evidence(geometry))
            available.append((row, geometry, scan_key, slug))
        except (ValueError, OSError, RuntimeError) as exc:
            audit["reason"] = str(exc)
        audits.append(audit)
        if progress:
            progress({"stage": "geometry", "case": audit["case_key"] or audit["directory"], "status": audit["status"]})
    _json(output_dir / "geometry_audit.json", audits)
    write_csv(output_dir / "source_groups.csv", ("candidate_id", "relative_directory", "source_group_candidate"),
              [{"candidate_id": r["candidate_id"], "relative_directory": r["relative_directory"],
                "source_group_candidate": groups[r["candidate_id"]]} for r in candidates])
    selected = available[:max_cases] if max_cases else available
    outcomes = []
    output_bytes_written = 0
    _json(output_dir / "run_plan.json", {"processed_root": str(processed_root),
          "resume_runs": [str(p) for p in resume_runs], "max_cases": max_cases,
          "max_new_output_bytes": max_new_output_bytes, "spec_fingerprint": spec.fingerprint()})
    _json(output_dir / "cases.json", [])
    for row, geometry, scan_key, slug in selected:
        outcome = {"case_key": scan_key, "status": "rejected", "reason": "", "output": ""}
        volume = native = valid = result = cached_native = case_dir = None
        cache_failures = []
        fingerprint = ""
        try:
            fingerprint = source_fingerprint(geometry)
            if row["file_set_sha256"] and fingerprint != row["file_set_sha256"]:
                raise ValueError("Source file-set differs from F01 fingerprint")
            for entry in reversed(cache_index.get(scan_key, [])):
                try:
                    verified = validate_cache(entry, row, geometry, spec, by_scan[scan_key],
                                              config["annotations"]["xy_index_base"], fingerprint)
                    if spec.save_native_hu:
                        cached_native = sitk.ReadImage(str(Path(entry["output"]) / "hu_native.nii.gz"))
                    checks = sample_rescale_qa(geometry, sitk.GetArrayViewFromImage(cached_native)
                                              if cached_native is not None else None)
                    outcome.update(status="reused_cache_pending_review", output=entry["output"],
                                   reused=True, resume_run=entry["resume_run"],
                                   cache_validation={k: v for k, v in verified.items() if k != "metadata"},
                                   statistics=verified["metadata"]["statistics"],
                                   intensity_qa={"header_evidence": series_intensity_evidence(geometry), **checks})
                    break
                except (ValueError, OSError, KeyError, TypeError, EOFError, RuntimeError) as exc:
                    cache_failures.append(str(exc))
                finally:
                    cached_native = None
            if outcome["status"] == "reused_cache_pending_review":
                continue
            expected_grid = canonical_grid_geometry(geometry.as_dict(), spec)
            estimated = int(np.prod(expected_grid["size_xyz"], dtype=object)) * 9
            if spec.save_native_hu:
                estimated += int(np.prod(geometry.size_xyz, dtype=object)) * 5
            estimated += 1024**2
            if max_new_output_bytes is not None and estimated > max_new_output_bytes - output_bytes_written:
                outcome.update(status="deferred_storage_budget", reason="Conservative next-volume footprint exceeds remaining output budget")
                continue
            volume = read_hu_volume(geometry)
            checks = sample_rescale_qa(geometry, volume.hu_zyx)
            native, valid = volume.as_images()
            result = preprocess_volume(native, valid, spec)
            annotations = map_ctgan_annotations(by_scan[scan_key], geometry, native, result["hu"],
                                                config["annotations"]["xy_index_base"])
            case_dir = processed_root / slug
            case_dir.mkdir()
            # NIfTI compression can vary. Reserve the conservative uncompressed output size.
            required = int(np.prod(result["hu"].GetSize(), dtype=object)) * 9
            if spec.save_native_hu:
                required += int(np.prod(native.GetSize(), dtype=object)) * 5
            if shutil.disk_usage(case_dir).free < required + 256 * 1024**2:
                raise ValueError("Insufficient storage reserve for processed outputs")
            if spec.save_native_hu:
                sitk.WriteImage(native, str(case_dir / "hu_native.nii.gz"), True)
                sitk.WriteImage(valid, str(case_dir / "valid_native.nii.gz"), True)
            for key, filename in (("hu", "hu_processed.nii.gz"), ("normalized", "normalized.nii.gz"), ("valid", "valid_mask.nii.gz")):
                sitk.WriteImage(result[key], str(case_dir / filename), True)
            write_csv(case_dir / "source_slices.csv", ("native_z", "filename", "position_lps_mm", "slope", "intercept", "padding_low", "padding_high"),
                      [{"native_z": i, "filename": r.path.name, "position_lps_mm": json.dumps(r.position_lps),
                        "slope": r.slope, "intercept": r.intercept,
                        "padding_low": r.padding_low, "padding_high": r.padding_high} for i, r in enumerate(geometry.slices)])
            stats = {"valid_voxels": int(np.count_nonzero(volume.valid_zyx)),
                     "hu_min_valid": float(np.min(volume.hu_zyx, where=volume.valid_zyx, initial=np.inf)),
                     "hu_max_valid": float(np.max(volume.hu_zyx, where=volume.valid_zyx, initial=-np.inf))}
            metadata = {"case_key": scan_key, "source_candidate_id": row["candidate_id"],
                        "source_group_candidate": groups[row["candidate_id"]], "source_file_set_sha256": fingerprint,
                        "native_geometry": geometry.as_dict(), "transform": result["transform"],
                        "statistics": stats, "annotations": annotations, "training_ready": False,
                        "intensity_qa": {"header_evidence": series_intensity_evidence(geometry), **checks},
                        "output_artifacts": artifact_receipts(case_dir, spec.save_native_hu),
                        "status": "preprocessed_pending_visual_review"}
            _json(case_dir / "metadata.json", metadata)
            output_bytes_written += sum(item["bytes"] for item in metadata["output_artifacts"].values()) + (case_dir / "metadata.json").stat().st_size
            outcome.update(status="preprocessed_pending_visual_review", output=str(case_dir), statistics=stats,
                           reused=False, intensity_qa=metadata["intensity_qa"])
        except (ValueError, OSError, RuntimeError, MemoryError) as exc:
            outcome["reason"] = str(exc)
        finally:
            volume = native = valid = result = None
            gc.collect()
            if outcome["status"] == "rejected" and case_dir is not None:
                outcome["partial_output"] = str(case_dir)
                try:
                    output_bytes_written += sum(p.stat().st_size for p in case_dir.iterdir() if p.is_file())
                except OSError:
                    outcome["partial_storage_accounting_incomplete"] = True
            if cache_failures:
                outcome["cache_rejection_reasons"] = cache_failures
            outcome.update(source_candidate_id=row["candidate_id"], source_group_candidate=groups[row["candidate_id"]],
                           source_file_set_sha256=fingerprint or row["file_set_sha256"])
            outcomes.append(outcome)
            # Checkpoint every completed case, including reused/deferred cases.
            _json(output_dir / "cases.json", outcomes)
            if progress:
                progress({"stage": "preprocess", "case_key": scan_key, "status": outcome["status"],
                          "reason": outcome["reason"], "new_output_bytes": output_bytes_written})
    failed_geometry = sum(row["status"] == "rejected" for row in audits)
    failed_process = sum(row["status"] == "rejected" for row in outcomes)
    successes = [row for row in outcomes if row["status"] in {"preprocessed_pending_visual_review", "reused_cache_pending_review"}]
    reused_count = sum(row.get("reused", False) for row in successes)
    deferred_count = sum(row["status"] == "deferred_storage_budget" for row in outcomes)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "f01_run": str(inventory_run), "dataset_root": str(dataset_root),
               "candidate_series": len(candidates), "geometry_valid": len(available),
               "geometry_rejected": failed_geometry,
               "geometry_with_warnings": sum(bool(a.get("native_geometry", {}).get("warnings")) for a in audits),
               "geometry_warning_counts": dict(Counter(warning for a in audits for warning in a.get("native_geometry", {}).get("warnings", []))),
               "geometry_rejection_reasons": dict(Counter(a["reason"] for a in audits if a["status"] == "rejected")),
               "source_groups_candidates": len(set(groups.values())),
               "processed_attempted": len(selected), "processed_success": len(successes),
               "newly_processed": len(successes) - reused_count, "reused_cache": reused_count,
               "deferred_storage_budget": deferred_count, "new_output_bytes": output_bytes_written,
               "intensity_header_status_counts": dict(Counter(a["intensity_evidence"]["status"] for a in audits if "intensity_evidence" in a)),
               "rescale_sample_checks_passed": sum(r.get("intensity_qa", {}).get("numerical_rescale_check") == "passed" for r in successes),
               "processed_rejected": failed_process, "processed_root": str(processed_root),
               "unprocessed_geometry_valid": len(available) - len(successes),
               "all_geometry_valid_covered": len(successes) == len(available) and bool(available),
               "ok": not failed_geometry and not failed_process and not deferred_count and bool(selected),
               "phase_f02_complete": False, "training_ready": False,
               "spec_fingerprint": spec.fingerprint(), "cases": outcomes,
               "pending": ["visual review of HU and spatial annotation alignment", "source identity reconciliation",
                           "final preprocessing choices on source validation", "remaining volumes if this is a pilot"]}
    _json(output_dir / "summary.json", summary)
    _json(output_dir / "cases.json", outcomes)
    columns = ("case_key", "source_candidate_id", "source_group_candidate", "source_file_set_sha256", "output", "status")
    write_csv(output_dir / "processing_manifest.csv", columns,
              [{key: row[key] for key in columns} for row in successes])
    lines = ["# F02 Geometry and Preprocessing", "",
             f"Geometry valid: {len(available)}/{len(candidates)}; rejected: {failed_geometry}.",
             f"Preprocessing successful: {len(successes)}/{len(selected)}; reused: {reused_count}; storage deferred: {deferred_count}.",
             "", "See geometry_audit.json and summary.json for rejection reasons.",
             "Pilot parameters and coordinate mappings require review before training."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
