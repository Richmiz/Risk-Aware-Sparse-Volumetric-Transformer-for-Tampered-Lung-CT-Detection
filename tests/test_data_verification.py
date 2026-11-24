"""Source and label verification tests use procedural fixtures, never patient images."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile
import zlib

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsvt.data.source_verification import compare_archive, compare_pixels, inspect_case_pixels, package_review_outputs, public_identity, run_source_verification, slice_fingerprint
from rsvt.data.verification import scan_key, verify_labels, verify_m3_lineage


def table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PublisherLabels(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "labels_exp1.csv"
        self.rows = [dict(type=c, uuid=str(i+1), slice=0 if c == "TB" else 1,
                          x=0 if c == "TB" else 2, y=0 if c == "TB" else 3)
                     for i, c in enumerate(("TB", "TM", "FB", "FM"))]
        self.policy = {"expected_scan_counts": {"exp1": {c: 1 for c in ("TB", "TM", "FB", "FM")}},
                       "label_file_sha256": {}}
        self.addCleanup(self.tmp.cleanup)

    def prepare(self, rows):
        table(self.path, rows)
        self.policy["label_file_sha256"]["exp1"] = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_duplicate_zero_rows_preserved_without_extra_cases_or_supervision(self):
        self.prepare(self.rows + [self.rows[0], self.rows[0]])
        scans, annotations = verify_labels({"exp1": self.path}, self.policy)
        self.assertEqual(len(scans), 4)
        self.assertEqual(scans["exp1:1"]["unique_rows"], 1)
        self.assertEqual(sum(r["duplicate"] for r in annotations), 2)
        self.assertEqual(sum(r["zero_location_placeholder"] for r in annotations), 3)
        self.assertFalse(any(r["localization_eligible"] for r in annotations))

    def test_changed_label_file_rejected_before_adopting_targets(self):
        self.prepare(self.rows)
        self.path.write_text(self.path.read_text().replace("FM", "FB"))
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_labels({"exp1": self.path}, self.policy)

    def test_mixed_location_labels_not_silently_majority_voted(self):
        self.prepare(self.rows + [{**self.rows[2], "type": "FM"}])
        with self.assertRaisesRegex(ValueError, "Mixed"):
            verify_labels({"exp1": self.path}, self.policy)

    def test_zero_on_fake_class_requires_explicit_review(self):
        self.prepare([{**r, "slice": 0, "x": 0, "y": 0} if r["type"] == "FB" else r for r in self.rows])
        with self.assertRaisesRegex(ValueError, "zero-location"):
            verify_labels({"exp1": self.path}, self.policy)

    def test_publisher_counts_checked_at_scan_instead_of_location_level(self):
        self.prepare(self.rows + [{**self.rows[2], "uuid": "999"}])
        with self.assertRaisesRegex(ValueError, "class counts"):
            verify_labels({"exp1": self.path}, self.policy)

    def test_experiment_namespaces_and_paths_are_explicit(self):
        self.assertEqual(scan_key("Tampered Scans\\Experiment 1 - Blind\\6031"), "exp1:6031")
        self.assertEqual(scan_key("Tampered Scans/Experiment 2 - Open/6031"), "exp2:6031")
        with self.assertRaises(ValueError):
            scan_key("../Experiment 1 - Blind/6031")


class PixelSources(unittest.TestCase):
    def records(self):
        return [{"case_key": "a", "split": "train", "source_group_candidate": "g1"},
                {"case_key": "b", "split": "test", "source_group_candidate": "g2"}]

    def cases(self):
        return {k: {"informative_slice_hashes": [], "full_native_pixel_fingerprint": k,
                    "public_identity": {"canonical_patient_id": ""},
                    "thumbnail_signature": np.full((2, 4, 4), float(i))}
                for i, k in enumerate(("a", "b"))}

    def test_native_HU_equivalence_independent_of_storage_rescale(self):
        image = np.arange(16, dtype=np.int16).reshape(4, 4)
        a = slice_fingerprint(image, 2, -100)
        b = slice_fingerprint(image*2, 1, -100)
        self.assertEqual(a[0], b[0])
        self.assertTrue(a[1])

    def test_constant_and_padding_only_slices_do_not_create_source_edges(self):
        self.assertFalse(slice_fingerprint(np.zeros((4, 4)), 1, -1000)[1])
        self.assertFalse(slice_fingerprint(np.zeros((4, 4)), 1, -1000, 0)[1])
        cases = self.cases()
        cases["b"]["full_native_pixel_fingerprint"] = cases["a"]["full_native_pixel_fingerprint"]
        self.assertEqual(compare_pixels(cases, self.records())[0], [])

    def test_shared_native_pixels_detect_cross_partition_evidence_with_rewritten_IDs(self):
        cases = self.cases()
        for case in cases.values():
            case["informative_slice_hashes"] = ["slice1", "slice2", "slice3"]
        pairs, _ = compare_pixels(cases, self.records())
        self.assertTrue(pairs[0]["cross_partition_source_evidence"])
        self.assertFalse(pairs[0]["same_existing_group"])
        self.assertEqual(pairs[0]["distinct_shared_informative_slices"], 3)

    def test_repeated_single_slice_does_not_count_as_multiple_independent_matches(self):
        cases = self.cases()
        for case in cases.values():
            case["informative_slice_hashes"] = ["same"]*10
        self.assertEqual(compare_pixels(cases, self.records())[0], [])

    def test_approximate_similarity_is_a_candidate_and_never_an_identity(self):
        cases = self.cases()
        cases["b"]["thumbnail_signature"] = np.full((2, 4, 4), .001)
        pairs, candidates = compare_pixels(cases, self.records())
        self.assertEqual(pairs, [])
        self.assertEqual(len(candidates), 1)
        self.assertIn("do_not_merge", candidates[0]["status"])

    def test_unique_exact_public_UID_required_for_canonical_patient(self):
        row = {"SeriesInstanceUID": "public.uid", "PatientID": "LIDC-IDRI-0001", "Collection": "LIDC-IDRI",
               "SeriesNumber": "12", "ImageCount": 10}
        self.assertEqual(public_identity(SimpleNamespace(SeriesInstanceUID="public.uid"), [row])["canonical_patient_id"], "LIDC-IDRI-0001")
        self.assertEqual(public_identity(SimpleNamespace(SeriesInstanceUID="public"), [row])["canonical_patient_id"], "")
        self.assertEqual(public_identity(SimpleNamespace(SeriesInstanceUID="public.uid"), [row, row])["canonical_patient_id"], "")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            public_identity(SimpleNamespace(SeriesInstanceUID="public.uid", PatientID="LIDC-IDRI-0002"), [row])

    def test_real_DICOM_decoder_and_file_binding_on_procedural_images(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"scan").mkdir()
            rows = []
            for i in range(3):
                path = root/"scan"/(str(i)+".dcm")
                meta = FileMetaDataset()
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID = CTImageStorage
                meta.MediaStorageSOPInstanceUID = "1.2.3.4."+str(i+1)
                image = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0"*128)
                image.SOPClassUID = CTImageStorage
                image.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
                image.SeriesInstanceUID = "1.2.3.4.99"
                image.PatientID = "procedural"
                image.Rows = image.Columns = 4
                image.SamplesPerPixel = 1
                image.PhotometricInterpretation = "MONOCHROME2"
                image.BitsAllocated = image.BitsStored = 16
                image.HighBit = 15
                image.PixelRepresentation = 1
                image.RescaleSlope, image.RescaleIntercept = 1, -1024
                image.ImagePositionPatient = [0, 0, i]
                image.PixelData = (np.arange(16, dtype=np.int16).reshape(4,4)+i).tobytes()
                image.save_as(path, enforce_file_format=True)
                rows.append({"relative_path": "scan/"+path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
            result = inspect_case_pixels(root, rows, [])
            self.assertEqual(result["source_file_count"], 3)
            self.assertEqual(len(result["informative_slice_hashes"]), 3)
            self.assertTrue(result["all_pixels_decoded"])
            self.assertEqual(result["public_identity"]["canonical_patient_id"], "")
            rows[0]["sha256"] = "incorrect"
            with self.assertRaisesRegex(ValueError, "differ from F01"):
                inspect_case_pixels(root, rows, [])

    def test_same_public_patient_is_grouped_even_for_different_series(self):
        cases = self.cases()
        for case in cases.values():
            case["public_identity"]["canonical_patient_id"] = "LIDC-IDRI-0001"
        self.assertTrue(compare_pixels(cases, self.records())[0][0]["same_public_patient"])

    def test_archive_audit_detects_missing_member_and_changed_bytes(self):
        with TemporaryDirectory() as tmp:
            archive = Path(tmp)/"data.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("scan/0.dcm", b"abc")
                z.writestr("scan/1.dcm", b"def")
            facts = [{"path": "scan/0.dcm", "bytes": 3, "crc32": zlib.crc32(b"abc")}]
            self.assertEqual(compare_archive(archive, facts)["missing_members"], ["scan/1.dcm"])
            facts += [{"path": "scan/1.dcm", "bytes": 3, "crc32": zlib.crc32(b"def")}]
            self.assertTrue(compare_archive(archive, facts)["verified"])
            facts[1]["crc32"] = 0
            self.assertFalse(compare_archive(archive, facts)["verified"])

    def test_nested_UCI_archive_is_checked_without_requiring_extraction(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            inner = root/"data.zip"
            outer = root/"uci.zip"
            with zipfile.ZipFile(inner, "w") as z:
                z.writestr("scan/0.dcm", b"abc")
            with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.write(inner, "data.zip")
            self.assertTrue(compare_archive(outer, [{"path": "scan/0.dcm", "bytes": 3, "crc32": zlib.crc32(b"abc")}])["verified"])


class M3Lineage(unittest.TestCase):
    def fixture(self, root):
        rows = [{"img_id": image, "ty": "rem", "mod": variant, "orig_id": "LIDC-IDRI-0001",
                 "sdir_id": "12", "coord_z": "1", "coord_y": "2", "coord_x": "3"}
                for image, variant in (("rem_1", "real"), ("rem_2", "real"), ("rem_1", "cycle"))]
        table(root/"data.csv", rows)
        table(root/"LIDC.csv", [{"orig_id": "LIDC-IDRI-0001", "sdir_id": "12", "dicom": "LIDC-IDRI-0001/date/12.000000-procedural"}])
        table(root/"sets.csv", [{"orig_id": "LIDC-IDRI-0001", "set": "test"}])
        centers = [{"img_id": r["img_id"], "mod": r["mod"]} for r in rows]
        table(root/"centers.csv", centers)
        return centers

    def test_repeated_real_reference_rows_share_one_original_series_without_clinical_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            summary, lineage, sources = verify_m3_lineage(root)
            self.assertEqual(summary["metadata_rows"], 3)
            self.assertEqual(summary["source_series"], 1)
            self.assertEqual(sources[0]["real_reference_rows"], 2)
            self.assertTrue(all(not r["forensic_label_from_operation"] for r in lineage if not r["generator"]))

    def test_orphan_center_is_reported_and_never_invented_as_a_sample(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            centers = self.fixture(root)
            table(root/"centers.csv", centers + [{"img_id": "inj_orphan", "mod": "real"}])
            summary, lineage, _ = verify_m3_lineage(root)
            self.assertEqual(len(summary["orphan_center_rows"]), 1)
            self.assertEqual(len(lineage), 3)

    def test_publisher_lineage_is_independently_linked_to_public_series_UID(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            catalog = [{"PatientID": "LIDC-IDRI-0001", "SeriesNumber": "12",
                        "SeriesInstanceUID": "public.uid", "Collection": "LIDC-IDRI"},
                       {"PatientID": "LIDC-IDRI-9999", "Collection": "LIDC-IDRI"}]
            summary, lineage, _ = verify_m3_lineage(root, catalog)
            self.assertTrue(summary["public_TCIA_source_mapping_complete"])
            self.assertTrue(all(r["canonical_series_uid"] == "public.uid" for r in lineage))
            summary, lineage, _ = verify_m3_lineage(root, catalog + [catalog[0]])
            self.assertFalse(summary["public_TCIA_source_mapping_complete"])
            self.assertTrue(all(not r["canonical_series_uid"] for r in lineage))

    def test_missing_test_center_rejected_instead_of_silent_inner_join(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            centers = self.fixture(root)
            table(root/"centers.csv", centers[:2])
            with self.assertRaisesRegex(ValueError, "lack publisher"):
                verify_m3_lineage(root)


class CompleteSourceAudit(unittest.TestCase):
    """Exercise report generation and public/source bindings on tiny DICOM fixtures."""

    def fixture(self, root):
        records, files, catalog = [], [], []
        for case_number, split in ((1, "train"), (2, "test")):
            directory = root / "raw/Tampered Scans/Experiment 1 - Blind" / str(case_number)
            directory.mkdir(parents=True)
            uid = "1.2.3.4." + str(90 + case_number)
            records.append({"case_key": "exp1:"+str(case_number), "split": split,
                            "source_group_candidate": "old-group-"+str(case_number)})
            catalog.append({"SeriesInstanceUID": uid, "PatientID": "LIDC-IDRI-0001",
                            "Collection": "LIDC-IDRI", "SeriesNumber": str(11+case_number), "ImageCount": 3})
            for i in range(3):
                path = directory / (str(i)+".dcm")
                meta = FileMetaDataset()
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID = CTImageStorage
                meta.MediaStorageSOPInstanceUID = uid + "."+str(i+1)
                image = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0"*128)
                image.SOPClassUID = CTImageStorage
                image.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
                image.SeriesInstanceUID, image.PatientID = uid, "procedural"
                image.Rows = image.Columns = 4
                image.SamplesPerPixel, image.PhotometricInterpretation = 1, "MONOCHROME2"
                image.BitsAllocated = image.BitsStored = 16
                image.HighBit, image.PixelRepresentation = 15, 1
                image.RescaleSlope, image.RescaleIntercept = 1, -1024
                image.ImagePositionPatient = [0, 0, i]
                image.PixelData = (np.arange(16, dtype=np.int16).reshape(4, 4)+i).tobytes()
                image.save_as(path, enforce_file_format=True)
                files.append({"dataset": "ct_gan", "kind": "dicom",
                              "relative_path": path.relative_to(root/"raw").as_posix(),
                              "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (root/"manifest.json").write_text(json.dumps({"records": records}))
        table(root/"files.csv", files)
        (root/"catalog.json").write_text(json.dumps(catalog))
        (root/"m3").mkdir()
        table(root/"m3/LIDC.csv", [{"orig_id": "LIDC-IDRI-0001", "sdir_id": "12"}])
        table(root/"m3/sets.csv", [{"orig_id": "LIDC-IDRI-0001", "set": "train"}])
        bound_paths = [root/n for n in ("manifest.json", "files.csv", "catalog.json", "m3/LIDC.csv", "m3/sets.csv")]
        (root/"metadata.json").write_text(json.dumps({"metadata_checks_passed": True,
            "training_ready": False, "goal_verification_complete": False,
            "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in bound_paths}}))
        return {"dataset_root": root/"raw", "manifest_path": root/"manifest.json",
                "files_path": root/"files.csv", "catalog_path": root/"catalog.json",
                "metadata_summary_path": root/"metadata.json", "m3_metadata_dir": root/"m3",
                "output_dir": root/"source_audit"}

    def test_full_audit_exposes_cross_partition_source_without_promoting_readiness(self):
        with TemporaryDirectory() as tmp, patch("rsvt.data.source_verification.require_colab_runtime"):
            args = self.fixture(Path(tmp))
            report = run_source_verification(**args)
            self.assertEqual(report["source_DICOM_files_checked"], 6)
            self.assertTrue(report["all_cases_pixel_checked"])
            self.assertTrue(report["cross_partition_source_evidence"])
            self.assertEqual(report["public_identity_resolved"], 2)
            self.assertEqual(report["cross_dataset_publisher_split_conflicts"], 1)
            self.assertFalse(report["training_ready"])
            self.assertFalse(report["goal_verification_complete"])
            self.assertFalse(report["model_loaded"])
            self.assertFalse(report["test_performance_evaluated"])
            self.assertEqual(json.loads((args["output_dir"]/"summary.json").read_text()), report)
            with (args["output_dir"]/"public_source_mapping.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_changed_public_catalog_cannot_relabel_sources_after_metadata_audit(self):
        with TemporaryDirectory() as tmp, patch("rsvt.data.source_verification.require_colab_runtime"):
            args = self.fixture(Path(tmp))
            catalog = json.loads(args["catalog_path"].read_text())
            catalog[0]["PatientID"] = "LIDC-IDRI-9999"
            args["catalog_path"].write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, "differ from the completed metadata"):
                run_source_verification(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_changed_M3_source_or_split_cannot_change_cross_dataset_conclusions(self):
        for name in ("LIDC.csv", "sets.csv"):
            with self.subTest(name=name), TemporaryDirectory() as tmp, patch("rsvt.data.source_verification.require_colab_runtime"):
                args = self.fixture(Path(tmp))
                path = args["m3_metadata_dir"]/name
                path.write_text(path.read_text()+"\n")
                with self.assertRaisesRegex(ValueError, "differ from the completed metadata"):
                    run_source_verification(**args)
                self.assertFalse(args["output_dir"].exists())


class LocalReviewPackage(unittest.TestCase):
    def fixture(self, root):
        args = CompleteSourceAudit.fixture(self, root)
        args["output_dir"] = root/"source_pixels"
        with patch("rsvt.data.source_verification.require_colab_runtime"):
            run_source_verification(**args)
        (root/"metadata").mkdir()
        shutil.copyfile(root/"metadata.json", root/"metadata/summary.json")
        for name in ("ctgan_reviewed_cases.csv", "ctgan_annotations.csv", "m3dsynth_lineage.csv", "m3dsynth_source_groups.csv"):
            table(root/"metadata"/name, [{"sample_id": "procedural_example"}])
        receipt = {"metadata_summary_sha256": hashlib.sha256((root/"metadata/summary.json").read_bytes()).hexdigest(),
                   "source_summary_sha256": hashlib.sha256((root/"source_pixels/summary.json").read_bytes()).hexdigest(),
                   "all_cases_pixel_checked": True, "training_ready": False, "goal_verification_complete": False}
        (root/"notebook_receipt.json").write_text(json.dumps(receipt))
        return root.parent/(root.name+".review.zip")

    def test_review_download_contains_only_declared_metadata_not_images_or_other_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.fixture(root)
            self.addCleanup(lambda: output.unlink(missing_ok=True))
            (root/"source_pixels/preview.png").write_bytes(b"procedural placeholder")
            (root/"credentials.json").write_text('{"example":"not_a_secret"}')
            (root/"checkpoint.pt").write_bytes(b"not a model")
            result = package_review_outputs(root, output)
            with zipfile.ZipFile(output) as z:
                self.assertEqual(len(z.namelist()), 11)
                self.assertNotIn("credentials.json", z.namelist())
                self.assertNotIn("source_pixels/preview.png", z.namelist())
                self.assertFalse(any(n.endswith((".dcm", ".pt", ".png")) for n in z.namelist()))
                manifest = json.loads(z.read("review_package_manifest.json"))
                self.assertFalse(manifest["medical_images_included"])
                self.assertFalse(manifest["goal_verification_complete"])
                for name, ref in manifest["files"].items():
                    self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), ref["sha256"])
            self.assertFalse(result["medical_images_included"])

    def test_changed_report_without_matching_receipt_cannot_be_exported_as_reviewed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.fixture(root)
            (root/"source_pixels/summary.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "receipt"):
                package_review_outputs(root, output)
            self.assertFalse(output.exists())

    def test_missing_case_report_cannot_be_hidden_by_a_success_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.fixture(root)
            (root/"source_pixels/exp1_2.json").unlink()
            with self.assertRaisesRegex(ValueError, "Per-case"):
                package_review_outputs(root, output)
            self.assertFalse(output.exists())

    def test_review_zip_does_not_overwrite_an_existing_user_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.fixture(root)
            output.write_bytes(b"keep existing file")
            self.addCleanup(lambda: output.unlink(missing_ok=True))
            with self.assertRaisesRegex(ValueError, "new review ZIP"):
                package_review_outputs(root, output)
            self.assertEqual(output.read_bytes(), b"keep existing file")


if __name__ == "__main__":
    unittest.main()
