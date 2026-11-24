"""Resume, cache integrity, storage deferral, and CT rescale evidence tests."""

from pathlib import Path
import csv
import json
import sys
import unittest
from unittest.mock import patch
from pydicom.dataset import Dataset
from pydicom.uid import CTImageStorage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import test_f02_preprocessing as fixtures
from rsvt.data.intensity import header_intensity_evidence, series_intensity_evidence, sample_rescale_qa
from rsvt.data.io import inspect_series, read_hu_volume
from rsvt.data.phase2 import run_phase2
import rsvt.data.phase2 as phase2


class UnitEvidenceTests(unittest.TestCase):
    def header(self, image_type=("ORIGINAL", "PRIMARY", "AXIAL"), units=None, energy=None):
        header = Dataset()
        header.SOPClassUID = CTImageStorage
        header.ImageType = list(image_type)
        if units is not None:
            header.RescaleType = units
        if energy is not None:
            header.add_new((0x0018, 0x9361), "CS", energy)
        return header

    def test_missing_rescale_type_can_follow_ct_module_conditions(self):
        evidence = header_intensity_evidence(self.header())
        self.assertEqual(evidence["status"], "hu_by_ct_image_module_conditions")
        self.assertFalse(evidence["calibration_verified"])

    def test_explicit_hu_and_explicit_non_hu_are_distinct(self):
        self.assertEqual(header_intensity_evidence(self.header(units="HU"))["status"], "explicit_hu")
        self.assertEqual(header_intensity_evidence(self.header(units="US"))["status"], "declared_non_hu")

    def test_incomplete_localizer_derived_or_multienergy_is_not_inferred_hu(self):
        for header in (self.header(image_type=[]), self.header(image_type=("ORIGINAL", "PRIMARY", "LOCALIZER")),
                       self.header(image_type=("DERIVED", "PRIMARY", "AXIAL")), self.header(energy="YES")):
            with self.subTest(header=str(header.ImageType)):
                self.assertEqual(header_intensity_evidence(header)["status"], "units_unspecified_requires_review")


class ResumeTests(fixtures.SyntheticFixture):
    def prepare(self):
        return fixtures.Phase2Integration.prepare_run(self)

    def pilot(self, dataset, inventory):
        return run_phase2(dataset, inventory, ROOT / "configs/preprocessing/ct_gan_pilot.toml",
                          self.root / "pilot_report", self.root / "pilot_volumes", max_cases=1)

    def bulk(self, dataset, inventory, **kwargs):
        return run_phase2(dataset, inventory, ROOT / "configs/preprocessing/ct_gan_pilot.toml",
                          self.root / "bulk_report", self.root / "bulk_volumes", max_cases=0,
                          resume_runs=[self.root / "pilot_report"], **kwargs)

    def test_resume_reuses_one_and_only_processes_missing_case(self):
        dataset, inventory = self.prepare()
        pilot = self.pilot(dataset, inventory)
        old_metadata = Path(pilot["cases"][0]["output"]) / "metadata.json"
        original_bytes = old_metadata.read_bytes()
        with patch("rsvt.data.phase2.preprocess_volume", wraps=phase2.preprocess_volume) as compute:
            report = self.bulk(dataset, inventory)
        self.assertEqual(compute.call_count, 1)
        self.assertEqual((report["processed_success"], report["newly_processed"], report["reused_cache"]), (2, 1, 1))
        self.assertTrue(report["all_geometry_valid_covered"])
        self.assertEqual(old_metadata.read_bytes(), original_bytes)
        self.assertFalse((self.root / "bulk_volumes/exp1_1").exists())
        with (self.root / "bulk_report/processing_manifest.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_legacy_pilot_gets_crc_and_adoption_receipts_without_rewrite(self):
        dataset, inventory = self.prepare()
        pilot = self.pilot(dataset, inventory)
        path = Path(pilot["cases"][0]["output"]) / "metadata.json"
        data = json.loads(path.read_text())
        del data["output_artifacts"]
        path.write_text(json.dumps(data))
        before = path.read_bytes()
        report = self.bulk(dataset, inventory)
        reused = next(row for row in report["cases"] if row.get("reused"))
        self.assertEqual(reused["cache_validation"]["verification"], "legacy_header_gzip_crc_and_adoption_sha256")
        self.assertFalse(reused["cache_validation"]["legacy_content_authenticity_proven"])
        self.assertEqual(path.read_bytes(), before)

    def test_corrupt_cache_is_rejected_and_recomputed_in_new_directory(self):
        dataset, inventory = self.prepare()
        pilot = self.pilot(dataset, inventory)
        corrupt = Path(pilot["cases"][0]["output"]) / "normalized.nii.gz"
        with corrupt.open("ab") as stream:
            stream.write(b"tampered synthetic artifact")
        report = self.bulk(dataset, inventory)
        self.assertEqual(report["reused_cache"], 0)
        self.assertEqual(report["newly_processed"], 2)
        self.assertTrue(report["cases"][0]["cache_rejection_reasons"])
        self.assertTrue(corrupt.read_bytes().endswith(b"tampered synthetic artifact"))

    def test_changed_source_does_not_reuse_cache_or_bypass_f01(self):
        dataset, inventory = self.prepare()
        self.pilot(dataset, inventory)
        path = dataset / "Tampered Scans/Experiment 1 - Blind/1/1.dcm"
        with path.open("ab") as stream:
            stream.write(b"source changed")
        report = self.bulk(dataset, inventory)
        self.assertEqual(report["processed_rejected"], 1)
        self.assertEqual(report["reused_cache"], 0)
        self.assertIn("differs from F01", report["cases"][0]["reason"])

    def test_changed_config_requires_new_outputs(self):
        dataset, inventory = self.prepare()
        self.pilot(dataset, inventory)
        config = self.root / "changed.toml"
        config.write_text((ROOT / "configs/preprocessing/ct_gan_pilot.toml").read_text().replace(
            "[1.0, 1.0, 1.0]", "[1.5, 1.0, 1.0]"))
        report = run_phase2(dataset, inventory, config, self.root / "bulk_report", self.root / "bulk_volumes",
                            max_cases=0, resume_runs=[self.root / "pilot_report"])
        self.assertEqual(report["reused_cache"], 0)
        self.assertEqual(report["newly_processed"], 2)

    def test_zero_budget_still_reuses_cache_but_defers_new_case(self):
        dataset, inventory = self.prepare()
        self.pilot(dataset, inventory)
        with patch("rsvt.data.phase2.read_hu_volume", side_effect=AssertionError("new decode must not run")):
            report = self.bulk(dataset, inventory, max_new_output_bytes=0)
        self.assertEqual((report["reused_cache"], report["newly_processed"], report["deferred_storage_budget"]), (1, 0, 1))
        self.assertFalse(report["ok"])
        self.assertEqual(report["new_output_bytes"], 0)

    def test_changed_annotation_inputs_are_not_reused(self):
        dataset, inventory = self.prepare()
        self.pilot(dataset, inventory)
        path = inventory / "ct_gan_labels/location_labels.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        rows[0]["pixel_x"] = "2"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        report = self.bulk(dataset, inventory)
        self.assertEqual(report["reused_cache"], 0)
        self.assertTrue(any("annotation inputs" in reason for reason in report["cases"][0]["cache_rejection_reasons"]))

    def test_legacy_cache_with_corrupt_gzip_is_not_adopted(self):
        dataset, inventory = self.prepare()
        pilot = self.pilot(dataset, inventory)
        folder = Path(pilot["cases"][0]["output"])
        path = folder / "metadata.json"
        data = json.loads(path.read_text())
        del data["output_artifacts"]
        path.write_text(json.dumps(data))
        artifact = folder / "normalized.nii.gz"
        content = bytearray(artifact.read_bytes())
        content[-8] ^= 1
        artifact.write_bytes(content)
        report = self.bulk(dataset, inventory)
        self.assertEqual(report["reused_cache"], 0)
        self.assertTrue(report["cases"][0]["cache_rejection_reasons"])

    def test_checkpoint_allows_resume_after_interruption(self):
        dataset, inventory = self.prepare()
        def interrupt(item):
            if item["stage"] == "preprocess":
                raise KeyboardInterrupt("synthetic interruption")
        with self.assertRaises(KeyboardInterrupt):
            run_phase2(dataset, inventory, ROOT / "configs/preprocessing/ct_gan_pilot.toml",
                       self.root / "pilot_report", self.root / "pilot_volumes", max_cases=0, progress=interrupt)
        rows = json.loads((self.root / "pilot_report/cases.json").read_text())
        self.assertEqual(len(rows), 1)
        self.assertFalse((self.root / "pilot_report/summary.json").exists())
        report = self.bulk(dataset, inventory)
        self.assertEqual((report["reused_cache"], report["newly_processed"]), (1, 1))

    def test_reference_check_finds_an_incorrect_native_array(self):
        geometry = inspect_series(self.series())
        volume = read_hu_volume(geometry)
        good = sample_rescale_qa(geometry, volume.hu_zyx)
        self.assertEqual(good["numerical_rescale_check"], "passed")
        volume.hu_zyx[0, 0, 0] += 100
        with self.assertRaisesRegex(ValueError, "differs from pydicom"):
            sample_rescale_qa(geometry, volume.hu_zyx)

    def test_header_evidence_is_aggregated_without_assuming_calibration(self):
        folder = self.series()
        self.change(folder / "1.dcm", RescaleType=None, ImageType=["ORIGINAL", "PRIMARY", "AXIAL"])
        geometry = inspect_series(folder)
        evidence = series_intensity_evidence(geometry)
        self.assertEqual(evidence["status"], "hu_supported_by_header_evidence")
        self.assertEqual(evidence["slice_status_counts"]["hu_by_ct_image_module_conditions"], 1)
        self.assertFalse(evidence["calibration_verified"])


if __name__ == "__main__":
    unittest.main()
