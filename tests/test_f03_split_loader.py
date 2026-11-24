"""F03 grouping, immutable draft manifests, and volume-loader behavior on synthetic data."""

from copy import deepcopy
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsvt.data.acquisition import sha256_file
from rsvt.data.dataset import CTVolumeDataset, pad_collate
from rsvt.data.inventory import write_csv
from rsvt.data.manifest import audit_manifest
from rsvt.data.phase3 import run_phase3
from rsvt.data.preprocessing import image_geometry
from rsvt.data.splits import CLASS_INDEX, audit_partitions, evidence_groups, grouped_split, leave_one_generator_out


def records(n=100):
    labels = ["FB"]*35 + ["FM"]*35 + ["TB"]*15 + ["TM"]*15
    return [{"case_key": f"case:{i}", "source_candidate_id": f"id-{i}",
             "relative_directory": f"Tampered Scans/Experiment 1 - Blind/{i}",
             "source_group_candidate": f"g-{i}", "f02_source_group_candidate": f"f02-{i}",
             "patient_metadata_keys": f"patient-{i}", "series_metadata_key": f"series-{i}",
             "source_file_set_sha256": f"source-{i}", "normalized_sha256": f"normalized-{i}",
             "label_candidate": labels[i], "generator_candidate": "ct_gan" if labels[i] in {"FB", "FM"} else ""}
            for i in range(n)]


class GroupSplits(unittest.TestCase):
    def test_grouped_split_is_row_order_independent_and_keeps_mixed_label_pair(self):
        rows = records()
        rows[85]["source_group_candidate"] = rows[0]["source_group_candidate"]
        rows[85]["patient_metadata_keys"] = rows[0]["patient_metadata_keys"]
        rows[85]["f02_source_group_candidate"] = rows[0]["f02_source_group_candidate"]
        fractions = {"train": .7, "validation": .15, "test": .15}
        first = grouped_split(rows, fractions, 13)
        second = grouped_split(list(reversed(rows)), fractions, 13)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["split"], next(r["split"] for r in first if r["case_key"] == "case:85"))
        audit = audit_partitions(first)
        self.assertTrue(audit["ok"])
        for part, target in (("train", 70), ("validation", 15), ("test", 15)):
            self.assertLessEqual(abs(audit["distribution"][part]["cases"]-target), 3)
            self.assertFalse(audit["missing_classes"][part])

    def test_transitive_metadata_and_partial_file_overlap_are_merged(self):
        rows = records(4)
        rows[1]["patient_metadata_keys"] = rows[0]["patient_metadata_keys"]
        rows[2]["series_metadata_key"] = rows[1]["series_metadata_key"]
        overlap = {"exact_file_duplicates": [{"files": [
            {"dataset": "ct_gan", "path": rows[2]["relative_directory"]+"/1.dcm"},
            {"dataset": "ct_gan", "path": rows[3]["relative_directory"]+"/3.dcm"}]}]}
        self.assertEqual(len(set(evidence_groups(rows, overlap).values())), 1)

    def test_identical_validity_masks_do_not_merge_unrelated_sources(self):
        rows = [{**r, "valid_mask_sha256": "all-valid-mask"} for r in records(4)]
        self.assertEqual(len(set(evidence_groups(rows, {}).values())), 4)

    def test_identical_normalized_content_is_grouped(self):
        rows = records(4)
        rows[2]["normalized_sha256"] = rows[0]["normalized_sha256"]
        groups = evidence_groups(rows, {})
        self.assertEqual(groups[rows[2]["source_candidate_id"]], groups[rows[0]["source_candidate_id"]])

    def test_leakage_audit_detects_metadata_across_declared_distinct_groups(self):
        rows = records(2)
        rows[0]["split"], rows[1]["split"] = "train", "test"
        rows[1]["patient_metadata_keys"] = rows[0]["patient_metadata_keys"]
        self.assertFalse(audit_partitions(rows)["ok"])

    def test_invalid_fractions_and_insufficient_groups_are_rejected(self):
        for fractions in ({"train": .8, "test": .3}, {"train": 1., "test": 0.}, {"train": float("nan"), "test": .2}):
            with self.assertRaises(ValueError):
                grouped_split(records(4), fractions, 13)
        with self.assertRaises(ValueError):
            grouped_split(records(2), {"train": .7, "validation": .15, "test": .15}, 13)

    def test_logo_holds_entire_source_group_including_other_generator(self):
        rows = records(8)
        for row in rows:
            row["generator_candidate"] = "B"
        rows[0]["generator_candidate"] = "A"
        rows[1]["source_group_candidate"] = rows[0]["source_group_candidate"]
        result = leave_one_generator_out(rows, "A")
        self.assertEqual({r["case_key"] for r in result if r["split"] == "test"}, {"case:0", "case:1"})
        self.assertTrue(audit_partitions(result)["ok"])

    def test_single_generator_logo_is_not_misrepresented_as_valid(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            leave_one_generator_out(records(), "ct_gan")


class LoaderFixture(unittest.TestCase):
    def setUp(self):
        self.work = TemporaryDirectory(prefix="rsvt_f03_")
        self.root = Path(self.work.name)
        self.inventory, self.f02 = self.root / "inventory", self.root / "f02"
        (self.inventory / "files").mkdir(parents=True)
        (self.inventory / "ct_gan_labels").mkdir()
        self.f02.mkdir()
        sources, labels, cases, manifest = [], [], [], []
        for i in range(12):
            key, candidate = f"exp1:{i+1}", f"candidate-{i}"
            category = tuple(CLASS_INDEX)[i % 4]
            rawhash = hashlib.sha256(f"source-{i}".encode()).hexdigest()
            directory = self.root / "processed" / f"exp1_{i+1}"
            directory.mkdir(parents=True)
            array = np.linspace(.01+i*.01, .6+i*.01, 60, dtype=np.float32).reshape(3, 4, 5)
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((1., 1., 1.))
            image.SetOrigin((10., 20., 30.))
            mask = sitk.GetImageFromArray(np.ones_like(array, dtype=np.uint8))
            mask.CopyInformation(image)
            mask[0, 0, 0] = 0
            sitk.WriteImage(image, str(directory / "normalized.nii.gz"))
            sitk.WriteImage(mask, str(directory / "valid_mask.nii.gz"))
            receipts = {name: {"bytes": (directory/name).stat().st_size, "sha256": sha256_file(directory/name)}
                        for name in ("normalized.nii.gz", "valid_mask.nii.gz")}
            metadata = {"status": "preprocessed_pending_visual_review", "case_key": key,
                "source_candidate_id": candidate, "source_group_candidate": f"f02-group-{i}",
                "source_file_set_sha256": rawhash,
                "transform": {"spec_fingerprint": "synthetic-spec", "processed": image_geometry(image)},
                "annotations": [{"label": category, "processed_index_xyz": [1., 2., 1.],
                                 "supervision_eligible": False, "status": "pending_visual_review"}],
                "output_artifacts": receipts, "training_ready": False}
            (directory / "metadata.json").write_text(json.dumps(metadata))
            source = {"candidate_id": candidate, "dataset": "ct_gan", "relative_directory": f"Tampered Scans/Experiment 1 - Blind/{i+1}",
                      "patient_metadata_keys": f"p-{i}", "series_metadata_key": f"s-{i}", "file_set_sha256": rawhash}
            sources.append(source)
            labels.append({"publisher_scan_key": key, "location_label": category})
            case = {"case_key": key, "source_candidate_id": candidate, "source_group_candidate": f"f02-group-{i}",
                    "source_file_set_sha256": rawhash, "output": str(directory), "status": "preprocessed_pending_visual_review"}
            manifest.append(case.copy())
            case["intensity_qa"] = {"numerical_rescale_check": "passed"}
            cases.append(case)
        for path, data in ((self.inventory / "files/series_candidates.csv", sources),
                           (self.inventory / "ct_gan_labels/location_labels.csv", labels),
                           (self.f02 / "processing_manifest.csv", manifest)):
            write_csv(path, tuple(data[0]), data)
        (self.inventory / "files/overlap.json").write_text(json.dumps({"exact_file_duplicates": []}))
        (self.f02 / "cases.json").write_text(json.dumps(cases))
        (self.f02 / "summary.json").write_text(json.dumps({"ok": True, "all_geometry_valid_covered": True,
            "processed_rejected": 0, "deferred_storage_budget": 0, "processed_success": 12,
            "f01_run": str(self.inventory), "spec_fingerprint": "synthetic-spec"}))
        self.config = ROOT / "configs/splits/ct_gan_draft.toml"

    def tearDown(self):
        self.work.cleanup()

    def build(self, output="report", **kwargs):
        return run_phase3(self.f02, self.inventory, self.config, self.root / output, **kwargs)

    def dataset(self, **kwargs):
        self.build()
        return CTVolumeDataset(self.root / "report/manifest.draft.json", "train", **kwargs)


class Phase3Integration(LoaderFixture):
    def test_complete_pipeline_records_draft_flags_and_train_only_smoke(self):
        result = self.build()
        self.assertTrue(result["ok"])
        self.assertEqual(result["verified_inputs"], 12)
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["phase_f03_complete"])
        doc = json.loads((self.root / "report/manifest.draft.json").read_text())
        train = {r["case_key"] for r in doc["records"] if r["split"] == "train"}
        self.assertTrue(all(row["case_key"] in train for row in result["loader_smoke"]["samples"]))
        self.assertTrue(all(not r["classification_eligible"] for r in doc["records"]))
        self.assertFalse(audit_manifest(self.root / "report/manifest.draft.csv", self.root).ok)
        with self.assertRaises(FileExistsError):
            self.build()

    def test_repeated_run_keeps_same_assignments(self):
        first, second = self.build(), self.build("second")
        self.assertEqual(first["assignments_sha256"], second["assignments_sha256"])

    def test_corrupt_input_prevents_any_partition_manifest(self):
        with (self.root / "processed/exp1_1/normalized.nii.gz").open("ab") as stream:
            stream.write(b"synthetic corruption")
        result = self.build()
        self.assertFalse(result["ok"])
        self.assertEqual(result["rejected_inputs"], 1)
        self.assertFalse((self.root / "report/manifest.draft.json").exists())

    def test_mixed_location_labels_are_not_silently_aggregated(self):
        path = self.inventory / "ct_gan_labels/location_labels.csv"
        with path.open("a") as stream:
            stream.write("exp1:1,FM\n")
        result = self.build()
        self.assertEqual(result["rejected_inputs"], 1)
        self.assertFalse(result["ok"])

    def test_incomplete_f02_run_is_rejected(self):
        path = self.f02 / "summary.json"
        value = json.loads(path.read_text())
        value["all_geometry_valid_covered"] = False
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "complete successful"):
            self.build()


class VolumeLoader(LoaderFixture):
    def test_default_full_fov_dtype_mask_and_pending_point_contract(self):
        dataset = self.dataset()
        sample = dataset[0]
        self.assertEqual(sample["image"].shape, (1, 3, 4, 5))
        self.assertEqual(sample["image"].dtype, np.float32)
        self.assertEqual(sample["valid_mask"].dtype, bool)
        self.assertFalse(sample["valid_mask"][0, 0, 0, 0])
        self.assertEqual(sample["image"][0, 0, 0, 0], 0)
        self.assertFalse(sample["classification_eligible"])
        self.assertFalse(sample["annotation_candidates"][0]["supervision_eligible"])
        self.assertEqual(sample["augmentation"]["flipped_axes"], [])

    def test_spatial_flip_preserves_point_physical_position_and_is_repeatable(self):
        original = self.dataset()
        flipped = CTVolumeDataset(self.root / "report/manifest.draft.json", "train", flip_axes=("x", "y", "z"), flip_probability=1.)
        before, after = original[0], flipped[0]
        point_before = [*before["annotation_candidates"][0]["index_xyz"], 1.]
        point_after = [*after["annotation_candidates"][0]["index_xyz"], 1.]
        np.testing.assert_allclose(before["voxel_to_lps"] @ point_before, after["voxel_to_lps"] @ point_after)
        np.testing.assert_array_equal(after["image"], before["image"][:, ::-1, ::-1, ::-1])
        np.testing.assert_array_equal(after["image"], flipped[0]["image"])
        self.assertTrue(after["image"].flags.c_contiguous)

    def test_draft_cannot_be_used_for_training_or_test_smoke(self):
        self.build()
        path = self.root / "report/manifest.draft.json"
        with self.assertRaisesRegex(ValueError, "not ready"):
            CTVolumeDataset(path, "train", purpose="training")
        with self.assertRaisesRegex(ValueError, "training inputs only"):
            CTVolumeDataset(path, "test")

    def test_validation_augmentation_is_rejected_even_for_a_ready_synthetic_manifest(self):
        self.build()
        path = self.root / "report/manifest.draft.json"
        doc = json.loads(path.read_text())
        doc["training_ready"] = True
        doc["status"] = "reviewed"
        for row in doc["records"]:
            row["classification_eligible"] = row["identity_reconciled"] = True
            row["label_verified"] = True
            row["partition_status"] = "reviewed"
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "training inputs only"):
            CTVolumeDataset(path, "validation", purpose="evaluation", flip_axes=("x",))

    def test_string_eligibility_flag_is_not_a_valid_boolean(self):
        self.build()
        path = self.root / "report/manifest.draft.json"
        doc = json.loads(path.read_text())
        doc["records"][0]["classification_eligible"] = "false"
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "must be booleans"):
            CTVolumeDataset(path, "train")

    def test_changed_partition_is_detected_by_assignment_fingerprint(self):
        self.build()
        path = self.root / "report/manifest.draft.json"
        doc = json.loads(path.read_text())
        next(row for row in doc["records"] if row["split"] == "train")["split"] = "test"
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "assignment fingerprint"):
            CTVolumeDataset(path, "train")

    def test_changed_artifact_is_detected_after_manifest_creation(self):
        dataset = self.dataset()
        path = Path(dataset.rows[0]["volume_path"])
        with path.open("ab") as stream:
            stream.write(b"changed after split")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            dataset[0]

    def test_decode_budget_rejects_before_image_read(self):
        dataset = self.dataset(max_decode_bytes=1)
        with patch("rsvt.data.dataset.sitk.ReadImage", side_effect=AssertionError("must not decode")):
            with self.assertRaisesRegex(ValueError, "decode memory"):
                dataset[0]

    def test_padding_mask_is_distinct_from_pixel_validity(self):
        dataset = self.dataset()
        first = dataset[0]
        second = deepcopy(first)
        second["image"] = first["image"][:, :2, :3, :4].copy()
        second["valid_mask"] = first["valid_mask"][:, :2, :3, :4].copy()
        batch = pad_collate([first, second])
        self.assertEqual(batch["image"].shape, (2, 1, 3, 4, 5))
        self.assertTrue(batch["padding_mask"][0, 0, 0, 0, 0])
        self.assertFalse(batch["valid_mask"][0, 0, 0, 0, 0])
        self.assertFalse(batch["padding_mask"][1, 0, 2, 3, 4])
        self.assertEqual(batch["image"][1, 0, 2, 3, 4], 0.)

    def test_collation_budget_rejects_before_padded_allocation(self):
        sample = self.dataset()[0]
        with patch("rsvt.data.dataset.np.zeros", side_effect=AssertionError("must not allocate")):
            with self.assertRaisesRegex(ValueError, "Padded batch"):
                pad_collate([sample], max_batch_bytes=1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch adapter is exercised on Colab; torch not installed locally")
    def test_torch_dataloader_with_custom_collator(self):
        from torch.utils.data import DataLoader
        dataset = self.dataset()
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=partial(pad_collate, as_torch=True))))
        self.assertEqual(tuple(batch["image"].shape), (2, 1, 3, 4, 5))
        self.assertEqual(str(batch["image"].dtype), "torch.float32")


if __name__ == "__main__":
    unittest.main()
