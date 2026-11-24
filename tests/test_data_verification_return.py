"""Validate local readback using only procedural DICOM and metadata fixtures."""

import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

import test_data_verification as fixtures

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/review_data_verification_return.py"
spec = importlib.util.spec_from_file_location("local_return_review", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LocalReturnReview(unittest.TestCase):
    def fixture(self, root):
        output = fixtures.LocalReviewPackage.fixture(self, root)
        self.addCleanup(lambda: output.unlink(missing_ok=True))
        metadata_path = root/"metadata/summary.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["ctgan"] = {"volumes": 2, "actual_slices": {"exp1": 6},
                             "slice_difference_actual_minus_readme": {"exp1": 0}}
        metadata_path.write_text(json.dumps(metadata))
        fixtures.table(root/"metadata/ctgan_reviewed_cases.csv", [{"case_key": "exp1:1"}, {"case_key": "exp1:2"}])
        receipt_path = root/"notebook_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["metadata_summary_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        receipt["bundle_sha256"] = "procedural-code-sha256"
        receipt_path.write_text(json.dumps(receipt))
        from rsvt.data.source_verification import package_review_outputs
        package_review_outputs(root, output)
        return output

    def test_complete_transport_preserves_open_source_findings_without_certifying_data(self):
        with TemporaryDirectory() as tmp:
            output = self.fixture(Path(tmp))
            result = module.review_archive(output, "procedural-code-sha256", expected_cases=2)
            self.assertTrue(result["artifact_integrity_verified"])
            self.assertEqual(result["reported_DICOM_files_checked"], 6)
            self.assertEqual(len(result["cross_partition_source_edges"]), 1)
            self.assertIn("shared_source_evidence_crosses_existing_partitions", result["pending_findings"])
            self.assertFalse(result["goal_verification_complete"])

    def test_old_or_different_notebook_bundle_is_rejected(self):
        with TemporaryDirectory() as tmp:
            output = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "expected local notebook"):
                module.review_archive(output, "different-bundle", expected_cases=2)

    def test_modified_returned_metadata_is_detected_before_source_claims(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.fixture(root)
            changed = root/"changed.review.zip"
            with zipfile.ZipFile(output) as source, zipfile.ZipFile(changed, "w") as target:
                for name in source.namelist():
                    value = source.read(name)
                    if name == "metadata/ctgan_reviewed_cases.csv":
                        value = value.replace(b"exp1:2", b"exp1:9")
                    target.writestr(name, value)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                module.review_archive(changed, "procedural-code-sha256", expected_cases=2)

    def test_even_declared_nonmetadata_files_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.fixture(root)
            changed = root/"extra.review.zip"
            content = b"procedural placeholder"
            with zipfile.ZipFile(output) as source, zipfile.ZipFile(changed, "w") as target:
                manifest = json.loads(source.read("review_package_manifest.json"))
                manifest["files"]["unexpected.dcm"] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                for name in source.namelist():
                    target.writestr(name, json.dumps(manifest).encode() if name == "review_package_manifest.json" else source.read(name))
                target.writestr("unexpected.dcm", content)
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                module.review_archive(changed, "procedural-code-sha256", expected_cases=2)


if __name__ == "__main__":
    unittest.main()
