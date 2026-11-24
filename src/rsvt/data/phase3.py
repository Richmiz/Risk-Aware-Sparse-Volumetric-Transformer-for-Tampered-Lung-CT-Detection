"""F03 source-grouped draft manifests and train-only data-loader smoke checks."""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
import json
import re
from pathlib import Path, PurePosixPath
import tomllib
import numpy as np

from rsvt.data.acquisition import sha256_file
from rsvt.data.cache import _check_header
from rsvt.data.dataset import CTVolumeDataset, pad_collate
from rsvt.data.inventory import write_csv
from rsvt.data.phase2 import _csv, _json
from rsvt.data.splits import CLASS_INDEX, audit_partitions, digest_json, evidence_groups, grouped_split, validate_fractions


def load_phase3_config(path: Path) -> dict:
    config = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    partition, loader = config["partition"], config["loader"]
    if (partition["status"] != "engineering_draft"
            or partition["label_rule"] != "unanimous_location_label_candidate"):
        raise ValueError("F03 currently supports explicit engineering drafts with unanimous location-label candidates")
    validate_fractions({key: partition[key] for key in ("train", "validation", "test")})
    for value, name, minimum in ((partition["seed"], "seed", 0), (loader["smoke_samples"], "smoke_samples", 0),
            (loader["max_decode_bytes"], "max_decode_bytes", 1), (loader["max_batch_bytes"], "max_batch_bytes", 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid {name}")
    if (len(set(loader["flip_axes"])) != len(loader["flip_axes"])
            or not set(loader["flip_axes"]).issubset({"x", "y", "z"})
            or isinstance(loader["flip_probability"], bool) or not np.isfinite(loader["flip_probability"])
            or not 0 <= loader["flip_probability"] <= 1):
        raise ValueError("Invalid training flip policy")
    return config


def _scan_key(relative: str) -> str:
    parts = PurePosixPath(relative.replace("\\", "/"))
    if parts.is_absolute() or ".." in parts.parts or not parts.name.isdecimal():
        raise ValueError("Invalid source directory identity")
    experiments = [token for name, token in (("Experiment 1 - Blind", "exp1"), ("Experiment 2 - Open", "exp2"))
                   if name in parts.parts]
    if len(experiments) != 1:
        raise ValueError("Unknown CT-GAN experiment directory")
    return experiments[0] + ":" + parts.name


def _record(source, case, labels, spec_fingerprint, output_dir):
    key = _scan_key(source["relative_directory"])
    if not re.fullmatch(r"[0-9a-f]{64}", source["file_set_sha256"]):
        raise ValueError("Malformed F01 source fingerprint")
    if (case["case_key"] != key or case["source_candidate_id"] != source["candidate_id"]
            or case["source_file_set_sha256"] != source["file_set_sha256"]):
        raise ValueError("F01/F02 source identity or fingerprint mismatch")
    categories = {row["location_label"] for row in labels}
    if len(categories) != 1 or not categories.issubset(CLASS_INDEX):
        raise ValueError("Missing/mixed location labels cannot silently become one volume target")
    label = next(iter(categories))
    directory = Path(case["output"]).resolve()
    if output_dir.is_relative_to(directory) or directory.is_relative_to(output_dir):
        raise ValueError("F03 reports must be separate from processed case directories")
    metadata_path = directory / "metadata.json"
    if metadata_path.is_symlink():
        raise ValueError("Symlinked case metadata is not supported")
    metadata_digest = sha256_file(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("status") != "preprocessed_pending_visual_review" or metadata["case_key"] != key
            or metadata["source_candidate_id"] != source["candidate_id"]
            or metadata["source_file_set_sha256"] != source["file_set_sha256"]
            or metadata["transform"]["spec_fingerprint"] != spec_fingerprint
            or case["source_group_candidate"] != metadata["source_group_candidate"]):
        raise ValueError("Completed F02 metadata differs from run/source evidence")
    if {a["label"] for a in metadata.get("annotations", [])} != categories:
        raise ValueError("F01 label evidence differs from F02 annotation labels")
    if case.get("intensity_qa", {}).get("numerical_rescale_check") != "passed":
        raise ValueError("F02 numerical rescale QA is missing")
    receipts = case.get("cache_validation", {}).get("artifacts") or metadata.get("output_artifacts")
    if not receipts:
        raise ValueError("F02 output receipts are required; resume/validate legacy cache first")
    actual = {}
    for name, prefix in (("normalized.nii.gz", "normalized"), ("valid_mask.nii.gz", "valid_mask")):
        path = directory / name
        expected = receipts[name]
        stamp = (path.stat().st_size, path.stat().st_mtime_ns)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing/size-mismatched model input: {name}")
        fingerprint = sha256_file(path)
        if fingerprint != expected["sha256"]:
            raise ValueError(f"Model input checksum differs from F02: {name}")
        _check_header(path, metadata["transform"]["processed"])
        if stamp != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError("Model input changed while building the partition snapshot")
        actual.update({prefix + "_sha256": fingerprint, prefix + "_bytes": path.stat().st_size})
    if sha256_file(metadata_path) != metadata_digest:
        raise ValueError("Case metadata changed while building the partition snapshot")
    return {"case_key": key, "dataset": "ct_gan", "source_candidate_id": source["candidate_id"],
            "relative_directory": source["relative_directory"],
            "patient_metadata_keys": source.get("patient_metadata_keys", ""),
            "series_metadata_key": source.get("series_metadata_key", ""),
            "source_file_set_sha256": source["file_set_sha256"],
            "f02_source_group_candidate": case["source_group_candidate"],
            "volume_path": str(directory / "normalized.nii.gz"), "valid_mask_path": str(directory / "valid_mask.nii.gz"),
            "metadata_path": str(metadata_path), "metadata_sha256": metadata_digest,
            "metadata_bytes": metadata_path.stat().st_size, **actual,
            "size_xyz": metadata["transform"]["processed"]["size_xyz"],
            "label_candidate": label, "candidate_label_index": CLASS_INDEX[label],
            "generator_candidate": "ct_gan" if label in {"FB", "FM"} else "",
            "label_rule": "unanimous_location_label_candidate", "label_verified": False,
            "classification_eligible": False, "localization_eligible": False,
            "identity_reconciled": False, "partition_status": "engineering_draft"}


def run_phase3(f02_run: Path, inventory_run: Path, config_path: Path, output_dir: Path,
               *, torch_smoke=False, progress=None) -> dict:
    config = load_phase3_config(config_path)
    f02_run, inventory_run, output_dir = (Path(p).resolve() for p in (f02_run, inventory_run, output_dir))
    if output_dir.exists():
        raise FileExistsError("F03 run already exists; use a new ID and preserve the previous partition manifest")
    for original in (f02_run, inventory_run):
        if output_dir.is_relative_to(original) or original.is_relative_to(output_dir):
            raise ValueError("F03 reports must be separate from historical run evidence")
    summary2 = json.loads((f02_run / "summary.json").read_text(encoding="utf-8"))
    if (summary2.get("ok") is not True or summary2.get("all_geometry_valid_covered") is not True
            or summary2.get("processed_rejected") != 0 or summary2.get("deferred_storage_budget") != 0
            or Path(summary2["f01_run"]).resolve() != inventory_run):
        raise ValueError("Use a complete successful F02 run and its original F01 evidence")
    cases = json.loads((f02_run / "cases.json").read_text(encoding="utf-8"))
    manifest2 = _csv(f02_run / "processing_manifest.csv", {"case_key", "source_candidate_id", "source_group_candidate", "source_file_set_sha256", "output", "status"})
    sources = _csv(inventory_run / "files/series_candidates.csv", {"candidate_id", "dataset", "relative_directory", "file_set_sha256"})
    sources = [row for row in sources if row["dataset"] == "ct_gan"]
    labels = _csv(inventory_run / "ct_gan_labels/location_labels.csv", {"publisher_scan_key", "location_label"})
    overlap = json.loads((inventory_run / "files/overlap.json").read_text(encoding="utf-8"))
    if (len(cases) != len(manifest2) or len(cases) != len(sources) or len(cases) != summary2["processed_success"]
            or len({r["source_candidate_id"] for r in cases}) != len(cases)
            or len({r["candidate_id"] for r in sources}) != len(sources)
            or len({r["case_key"] for r in manifest2}) != len(manifest2)):
        raise ValueError("Incomplete or duplicated F01/F02 manifest coverage")
    case_index = {r["source_candidate_id"]: r for r in cases}
    by_key = {r["case_key"]: r for r in manifest2}
    by_scan = defaultdict(list)
    for row in labels:
        by_scan[row["publisher_scan_key"]].append(row)
    input_files = [f02_run / name for name in ("summary.json", "cases.json", "processing_manifest.csv")]
    input_files += [inventory_run / name for name in ("files/series_candidates.csv", "files/overlap.json", "ct_gan_labels/location_labels.csv")]
    input_files.append(Path(config_path))
    input_receipts = {str(p): sha256_file(p) for p in input_files}
    for case in cases:
        original = Path(case["output"]).resolve()
        if output_dir.is_relative_to(original) or original.is_relative_to(output_dir):
            raise ValueError("F03 reports must be separate from processed case directories")
    output_dir.mkdir(parents=True)
    _json(output_dir / "resolved_config.json", config)
    records, audits = [], []
    for source in sorted(sources, key=lambda r: r["candidate_id"]):
        audit = {"source_candidate_id": source["candidate_id"], "status": "rejected", "reason": ""}
        try:
            case = case_index[source["candidate_id"]]
            if case["status"] not in {"preprocessed_pending_visual_review", "reused_cache_pending_review"}:
                raise ValueError("F02 case is not successful")
            if any(str(case[key]) != value for key, value in by_key[case["case_key"]].items()):
                raise ValueError("F02 cases and processing manifest disagree")
            record = _record(source, case, by_scan[_scan_key(source["relative_directory"])], summary2["spec_fingerprint"], output_dir)
            records.append(record)
            audit.update(status="inputs_verified", case_key=record["case_key"])
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            audit["reason"] = str(exc)
        audits.append(audit)
        _json(output_dir / "input_audit.json", audits)
        if progress:
            progress({"stage": "verify_inputs", **audit})
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "f02_run": str(f02_run),
               "f01_run": str(inventory_run), "input_receipts": input_receipts,
               "input_cases": len(sources), "verified_inputs": len(records), "rejected_inputs": len(sources)-len(records),
               "training_ready": False, "phase_f03_complete": False, "status": "engineering_draft",
               "raw_dicom_required": False, "model_training_run": False, "ok": False}
    if len(records) != len(sources):
        summary["reason"] = "Input audit failed; no partition manifest was produced"
        _json(output_dir / "summary.json", summary)
        return summary
    # Refuse a moving evidence snapshot rather than silently mixing run versions.
    if any(sha256_file(p) != input_receipts[str(p)] for p in input_files):
        raise ValueError("Input run evidence changed while building F03")
    groups = evidence_groups(records, overlap)
    records = [{**r, "source_group_candidate": groups[r["source_candidate_id"]]} for r in records]
    part = config["partition"]
    records = grouped_split(records, {key: part[key] for key in ("train", "validation", "test")}, part["seed"])
    audit = audit_partitions(records)
    if not audit["ok"]:
        raise ValueError("Source evidence leakage in draft partition")
    assignment_digest = digest_json([{key: r[key] for key in ("case_key", "source_group_candidate", "split", "label_candidate", "source_file_set_sha256", "normalized_sha256")} for r in records])
    resource_failures = [{"case_key": r["case_key"], "voxels": int(np.prod(r["size_xyz"], dtype=object))}
                         for r in records if (int(np.prod(r["size_xyz"], dtype=object))*12 > config["loader"]["max_decode_bytes"]
                         or int(np.prod(r["size_xyz"], dtype=object))*11 > config["loader"]["max_batch_bytes"])]
    document = {"schema_version": "f03_candidate_v1", "status": "engineering_draft", "training_ready": False,
                "class_index": CLASS_INDEX, "seed": part["seed"], "assignments_sha256": assignment_digest,
                "f02_run": str(f02_run), "input_receipts": input_receipts, "records": records}
    manifest_path = output_dir / "manifest.draft.json"
    _json(manifest_path, document)
    columns = tuple(key for key in records[0] if key != "size_xyz")
    write_csv(output_dir / "manifest.draft.csv", columns, [{key: r[key] for key in columns} for r in records])
    _json(output_dir / "split_audit.json", audit)
    smoke = {"ok": False, "status": "not_run", "samples": [], "partition": "train"}
    try:
        settings = config["loader"]
        dataset = CTVolumeDataset(manifest_path, "train", seed=part["seed"], max_decode_bytes=settings["max_decode_bytes"],
                                  flip_axes=settings["flip_axes"], flip_probability=settings["flip_probability"])
        # Smallest training volumes bound smoke memory; no held-out pixel reads.
        indices = sorted(range(len(dataset)), key=lambda i: (int(np.prod(dataset.rows[i]["size_xyz"], dtype=object)), dataset.rows[i]["case_key"]))[:settings["smoke_samples"]]
        if torch_smoke:
            from torch.utils.data import DataLoader, Subset
            batches = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=partial(pad_collate, max_batch_bytes=settings["max_batch_bytes"], as_torch=True))
        else:
            batches = (pad_collate([dataset[index]], max_batch_bytes=settings["max_batch_bytes"]) for index in indices)
        for batch in batches:
            info = {"case_key": batch["case_key"][0], "shape_bczyx": list(batch["image"].shape),
                    "valid_voxels": int(batch["valid_mask"].sum()),
                    "classification_eligible": bool(batch["classification_eligible"][0])}
            if torch_smoke:
                info["torch_dtype"] = str(batch["image"].dtype)
            smoke["samples"].append(info)
            del batch
            if progress:
                progress({"stage": "loader_smoke", **info})
        smoke.update(ok=bool(indices), status="passed" if indices else "not_run", torch_requested=torch_smoke)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError, MemoryError) as exc:
        smoke.update(status="failed", reason=str(exc))
    _json(output_dir / "loader_smoke.json", smoke)
    summary.update(ok=smoke["ok"] and not resource_failures, source_groups=len(set(groups.values())), seed=part["seed"],
                   assignments_sha256=assignment_digest, split_audit=audit, loader_smoke=smoke,
                   loader_resource_preflight={"batch_size": 1, "inadmissible_cases": resource_failures,
                       "max_processed_voxels": max(int(np.prod(r["size_xyz"], dtype=object)) for r in records)},
                   manifest_path=str(manifest_path), manifest_sha256=sha256_file(manifest_path),
                   logo_status="not_applicable_single_generator_ctgan", classification_rule=part["label_rule"],
                   pending=["canonical patient/source reconciliation", "candidate volume label validation",
                            "spatial annotation eligibility and preprocessing signal-preservation QA", "multi-generator image availability for LOGO"])
    _json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text("# F03 engineering draft\n\n" +
        f"Verified model inputs: {len(records)}; groups: {len(set(groups.values()))}; seed: {part['seed']}.\n\n" +
        "See split_audit.json for class/source distribution and loader_smoke.json for train-only checks.\n\n" +
        "Candidate labels and source groups remain provisional; training_ready=false.\n", encoding="utf-8")
    return summary
