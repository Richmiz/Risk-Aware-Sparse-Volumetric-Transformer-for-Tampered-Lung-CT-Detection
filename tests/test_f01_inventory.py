"""Behavior checks for data acquisition and provenance; all fixtures are synthetic."""

from io import BytesIO
from pathlib import Path
import csv
import json
import sys
from tempfile import TemporaryDirectory
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsvt.data.acquisition import extract_zip, safe_member_path
from rsvt.data.ct_gan import audit_ctgan_labels, extract_ctgan_prefix
from rsvt.data.inventory import inventory_data
from rsvt.data.m3dsynth import audit_m3dsynth
from rsvt.data.manifest import MANIFEST_COLUMNS, audit_manifest


class WorkspaceCase(unittest.TestCase):
    def setUp(self):
        self.work = TemporaryDirectory(prefix="rsvt_f01_")
        self.root = Path(self.work.name)

    def tearDown(self):
        self.work.cleanup()

    def csv(self, path, columns, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


class ArchiveChecks(WorkspaceCase):
    def test_archive_paths_reject_escape(self):
        for name in ("../escape", "/absolute", "C:/windows", "a\\..\\escape", "x:stream"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_member_path(self.root, name)

    def test_zip_traversal_rejected_before_writing(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("../escape.txt", "bad")
        target = self.root / "output"
        with self.assertRaises(ValueError):
            extract_zip(archive, target, max_uncompressed_bytes=100)
        self.assertFalse(target.exists())

    def test_zip_size_limit_and_extraction(self):
        archive = self.root / "good.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("nested/file.txt", "payload")
        with self.assertRaises(ValueError):
            extract_zip(archive, self.root / "too_large", max_uncompressed_bytes=1)
        extract_zip(archive, self.root / "ok", max_uncompressed_bytes=100)
        self.assertEqual((self.root / "ok/nested/file.txt").read_text(), "payload")

    def test_nested_prefix_extracts_only_first_series(self):
        inner = BytesIO()
        with zipfile.ZipFile(inner, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            stream.writestr("Tampered Scans/labels_exp1.csv", "type,uuid,slice,x,y\n")
            stream.writestr("scan/a/1.dcm", b"synthetic1")
            stream.writestr("scan/a/2.dcm", b"synthetic2")
            stream.writestr("scan/b/1.dcm", b"synthetic3")
        archive = self.root / "outer.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            stream.writestr("Dataset Access.txt", "synthetic fixture")
            stream.writestr("data.zip", inner.getvalue())
        report = extract_ctgan_prefix(archive, self.root / "pilot")
        self.assertEqual(report["dicom_files"], 2)
        self.assertTrue(report["next_series_boundary_seen"])
        self.assertFalse(report["series_completeness_verified"])
        self.assertFalse((self.root / "pilot/scan/b/1.dcm").exists())


class InventoryChecks(WorkspaceCase):
    def config(self):
        path = self.root / "inventory.toml"
        path.write_text('[inventory]\nhash_files=true\nread_dicom_headers=true\n'
                        '[datasets.a]\nroot="raw/a"\nrole="forensic"\n'
                        '[datasets.b]\nroot="raw/b"\nrole="anatomy_reference"\n')
        return path

    def test_empty_and_missing_are_not_ready(self):
        (self.root / "raw/a").mkdir(parents=True)
        report = inventory_data(self.config(), self.root, self.root / "out")
        self.assertEqual(report["datasets"]["a"]["status"], "empty")
        self.assertEqual(report["datasets"]["b"]["status"], "missing")
        self.assertFalse(report["training_ready"])

    def test_duplicate_hashes_across_datasets(self):
        for name in ("a", "b"):
            folder = self.root / "raw" / name
            folder.mkdir(parents=True)
            (folder / "same.bin").write_bytes(b"identical fixture")
        inventory_data(self.config(), self.root, self.root / "out")
        overlap = json.loads((self.root / "out/overlap.json").read_text())
        self.assertEqual(len(overlap["exact_file_duplicates"]), 1)

    def test_bad_dicom_reported_without_crashing(self):
        folder = self.root / "raw/a"
        folder.mkdir(parents=True)
        (folder / "invalid.dcm").write_bytes(b"not a DICOM")
        report = inventory_data(self.config(), self.root, self.root / "out")
        self.assertEqual(len(report["metadata_errors"]), 1)
        self.assertEqual(report["ct_series_candidates"], 0)

    def test_valid_dicom_header_group_and_pseudonymization(self):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid

        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        uid = generate_uid()
        for name in ("a", "b"):
            folder = self.root / "raw" / name
            folder.mkdir(parents=True)
            meta.MediaStorageSOPInstanceUID = generate_uid()
            image = FileDataset(str(folder / "slice.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
            image.PatientID = "SYNTHETIC_PATIENT"
            image.SeriesInstanceUID = uid
            image.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            image.Modality, image.Rows, image.Columns = "CT", 2, 2
            image.save_as(folder / "slice.dcm", enforce_file_format=True)
        report = inventory_data(self.config(), self.root, self.root / "out")
        self.assertEqual(report["ct_series_candidates"], 2)
        overlap = json.loads((self.root / "out/overlap.json").read_text())
        self.assertEqual(len(overlap["shared_series_uid_candidates"]), 1)
        self.assertNotIn("SYNTHETIC_PATIENT", (self.root / "out/files.csv").read_text())
        self.assertNotIn(uid, (self.root / "out/files.csv").read_text())

    def test_no_output_inside_input_and_no_overwrite(self):
        config = self.config()
        with self.assertRaises(ValueError):
            inventory_data(config, self.root, self.root / "raw/a/out")
        inventory_data(config, self.root, self.root / "out")
        with self.assertRaises(FileExistsError):
            inventory_data(config, self.root, self.root / "out")


class MetadataChecks(WorkspaceCase):
    def test_ctgan_mixed_location_labels_do_not_become_volume_label(self):
        folder = self.root / "ct/Tampered Scans"
        columns = ["type", "uuid", "slice", "x", "y"]
        self.csv(folder / "labels_exp1.csv", columns, [
            dict(zip(columns, ("FB", "1", "2", "3", "4"))),
            dict(zip(columns, ("TB", "1", "3", "4", "5"))),
        ])
        self.csv(folder / "labels_exp2.csv", columns, [dict(zip(columns, ("FM", "2", "2", "3", "4")))])
        report = audit_ctgan_labels(self.root / "ct", self.root / "out")
        self.assertEqual(report["mixed_location_class_scan_groups"], 1)
        with (self.root / "out/scan_label_summary.csv").open() as stream:
            self.assertTrue(all(not row["volume_label"] for row in csv.DictReader(stream)))

    def m3_metadata(self):
        folder = self.root / "metadata"
        columns = ["img_id", "ty", "mod", "orig_id", "sdir_id", "coord_z", "coord_y", "coord_x"]
        rows = [dict(zip(columns, ("rem_1", "rem", mod, "LIDC-IDRI-0001", "5", "1", "2", "3")))
                for mod in ("real", "cycle", "diffusion")]
        self.csv(folder / "data.csv", columns, rows)
        self.csv(folder / "LIDC.csv", ["orig_id", "sdir_id", "dicom"],
                 [{"orig_id": "LIDC-IDRI-0001", "sdir_id": "5", "dicom": "source/series"}])
        self.csv(folder / "sets.csv", ["orig_id", "set"], [{"orig_id": "LIDC-IDRI-0001", "set": "train"}])
        return folder

    def test_m3_lineage_is_shared_across_generators(self):
        report = audit_m3dsynth(self.m3_metadata(), self.root / "volumes", self.root / "out")
        self.assertEqual(report["author_source_series"], 1)
        self.assertEqual(report["manipulated_rows"], 2)
        self.assertEqual(report["sources_shared_by_multiple_manipulation_generators"], 1)
        self.assertEqual(report["scan_directories_present"], 0)
        self.assertFalse(report["training_ready"])

    def test_ctgan_duplicate_zero_rows_are_evidence_not_valid_points(self):
        folder = self.root / "ct/Tampered Scans"
        columns = ["type", "uuid", "slice", "x", "y"]
        row = dict(zip(columns, ("TB", "1", "0", "0", "0")))
        self.csv(folder / "labels_exp1.csv", columns, [row, row])
        self.csv(folder / "labels_exp2.csv", columns,
                 [dict(zip(columns, ("FM", "2", "1", "2", "3")))])
        report = audit_ctgan_labels(self.root / "ct", self.root / "out")
        self.assertEqual(report["duplicate_label_rows"], 1)
        self.assertEqual(report["zero_coordinate_rows"], 2)
        self.assertEqual(report["label_rows"], 3)

    def test_m3_missing_lineage_rejected(self):
        folder = self.m3_metadata()
        (folder / "LIDC.csv").write_text("orig_id,sdir_id,dicom\nother,5,source\n")
        with self.assertRaises(ValueError):
            audit_m3dsynth(folder, self.root, self.root / "out")

    def test_training_manifest_rejects_draft_and_malformed_hash(self):
        row = dict.fromkeys(MANIFEST_COLUMNS, "")
        row.update(sample_id="sample", dataset="fixture", patient_id="p", source_volume_id="s",
                   volume_path="v.nii.gz", role="forensic", label="TB", manipulation_type="none",
                   annotation_type="none", split="train", sha256="not_a_hash")
        path = self.root / "manifest.csv"
        self.csv(path, MANIFEST_COLUMNS, [row])
        self.assertFalse(audit_manifest(path, self.root).ok)
        row.update(sha256="", patient_id="", source_volume_id="", split="")
        self.csv(path, MANIFEST_COLUMNS, [row])
        self.assertFalse(audit_manifest(path, self.root).ok)


if __name__ == "__main__":
    unittest.main()
