"""Review a user-downloaded metadata audit ZIP locally, without extracting images."""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def review_archive(path, expected_bundle_sha256, *, expected_cases=100):
    """Verify the transport and summarize open scientific findings, never certify readiness."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or sum(info.file_size for info in archive.infolist()) > 64*1024**2:
            raise ValueError("Repeated filenames or oversized metadata-only review package")
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError("Unsafe review package member")
        manifest = json.loads(archive.read("review_package_manifest.json"))
        if (manifest.get("schema") != "metadata_only_data_audit_review_package_v1"
                or manifest.get("medical_images_included") is not False
                or manifest.get("checkpoints_included") is not False):
            raise ValueError("Unexpected review-package schema or payload scope")
        if set(names) != set(manifest["files"]) | {"review_package_manifest.json"}:
            raise ValueError("Review package has missing or undeclared files")
        allowed = {"metadata/summary.json", "metadata/ctgan_reviewed_cases.csv", "metadata/ctgan_annotations.csv",
                   "metadata/m3dsynth_lineage.csv", "metadata/m3dsynth_source_groups.csv",
                   "source_pixels/summary.json", "source_pixels/public_source_mapping.csv", "notebook_receipt.json"}
        for name in manifest["files"]:
            if name not in allowed and not re.fullmatch(r"source_pixels/exp[12]_[0-9]+\.json", name):
                raise ValueError("Unrecognized file in metadata-only review package")
        for name, receipt in manifest["files"].items():
            content = archive.read(name)
            if len(content) != receipt["bytes"] or hashlib.sha256(content).hexdigest() != receipt["sha256"]:
                raise ValueError("Review metadata checksum mismatch: " + name)
        def document(name):
            return json.loads(archive.read(name))
        metadata = document("metadata/summary.json")
        source = document("source_pixels/summary.json")
        receipt = document("notebook_receipt.json")
        if receipt.get("bundle_sha256") != expected_bundle_sha256:
            raise ValueError("Returned run did not use the expected local notebook code bundle")
        if (receipt.get("metadata_summary_sha256") != manifest["files"]["metadata/summary.json"]["sha256"]
                or receipt.get("source_summary_sha256") != manifest["files"]["source_pixels/summary.json"]["sha256"]):
            raise ValueError("Notebook receipt does not bind the returned summaries")
        if (metadata.get("metadata_checks_passed") is not True or source.get("all_cases_pixel_checked") is not True
                or receipt.get("all_cases_pixel_checked") is not True):
            raise ValueError("Metadata or pixel run is incomplete")
        if any(report.get("training_ready") is not False or report.get("goal_verification_complete") is not False
               for report in (metadata, source, receipt, manifest)):
            raise ValueError("Audit artifact improperly claims research readiness")
        def rows(name):
            return list(csv.DictReader(StringIO(archive.read(name).decode("utf-8-sig"))))
        cases = rows("metadata/ctgan_reviewed_cases.csv")
        public_mapping = rows("source_pixels/public_source_mapping.csv")
        case_ids = {row["case_key"] for row in cases}
        mapping_ids = {row["case_key"] for row in public_mapping}
        case_members = [name for name in names if re.fullmatch(r"source_pixels/exp[12]_[0-9]+\.json", name)]
        file_case_ids = {PurePosixPath(name).stem.replace("_", ":", 1) for name in case_members}
        if (len(cases) != expected_cases or len(case_ids) != expected_cases
                or len(public_mapping) != expected_cases or len(mapping_ids) != expected_cases
                or source.get("case_count") != expected_cases or manifest.get("case_count") != expected_cases
                or metadata["ctgan"]["volumes"] != expected_cases
                or case_ids != mapping_ids or case_ids != file_case_ids):
            raise ValueError("Returned per-scan coverage does not match the requested CT-GAN cohort")
        checked_slices = 0
        for name in case_members:
            case = document(name)
            if case.get("all_source_bytes_match_F01") is not True or case.get("all_pixels_decoded") is not True:
                raise ValueError("Per-scan source bytes or pixel decoding not verified")
            checked_slices += case["source_file_count"]
        if (checked_slices != source["source_DICOM_files_checked"]
                or checked_slices != sum(metadata["ctgan"]["actual_slices"].values())):
            raise ValueError("Per-case slice coverage does not reproduce source/inventory totals")
        known_edges = source["shared_source_pixel_or_public_patient_edges"]
        cross_partition = [edge for edge in known_edges if edge["cross_partition_source_evidence"]]
        unresolved = sum(not row["canonical_patient_id"] for row in public_mapping)
        if unresolved != source["public_identity_unresolved"]:
            raise ValueError("Public-source mapping summary disagrees with per-case records")
        archive_comparison = source["archive_member_comparison"]
        pending = []
        if cross_partition:
            pending.append("shared_source_evidence_crosses_existing_partitions")
        if unresolved:
            pending.append("complete_CTGAN_to_LIDC_mapping_missing")
        if source["approximate_similarity_candidates"]:
            pending.append("approximate_source_candidates_require_review")
        if not archive_comparison["verified"]:
            pending.append("acquired_archive_completeness_not_confirmed")
        if not source["original_author_Drive_completeness_verified"]:
            pending.append("original_author_source_completeness_not_confirmed")
        if not source["annotation_axis_visual_review_completed"]:
            pending.append("annotation_axis_review_pending")
        if source["cross_dataset_publisher_split_conflicts"]:
            pending.append("cross_dataset_publisher_partitions_require_reconciliation")
    return {"schema": "local_review_of_returned_data_audit_v1", "archive": str(path.resolve()),
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "expected_code_bundle_sha256": expected_bundle_sha256, "artifact_integrity_verified": True,
            "reported_CT_volumes_pixel_checked": expected_cases, "reported_DICOM_files_checked": checked_slices,
            "shared_source_edges": known_edges, "cross_partition_source_edges": cross_partition,
            "public_identity_unresolved": unresolved, "archive_member_comparison": archive_comparison,
            "readme_slice_difference": metadata["ctgan"]["slice_difference_actual_minus_readme"],
            "pending_findings": pending, "training_ready": False, "goal_verification_complete": False,
            "basis": "verified_metadata_from_user_run_not_independent_redecoding_of_remote_pixels"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = args.expected_bundle_sha256
    if expected is None:
        delivery = json.loads((ROOT/"reports/supplementary/data_label_audit_20261111/delivery.json").read_text())
        known = delivery.get("verified_completed_runs", {}).get(hashlib.sha256(args.archive.read_bytes()).hexdigest(), {})
        expected = known.get("bundle_sha256", delivery["bundle_sha256"])
    result = review_archive(args.archive, expected)
    if args.output:
        if args.output.exists() or args.output.resolve() == args.archive.resolve():
            raise FileExistsError("Use a new local review output file")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
