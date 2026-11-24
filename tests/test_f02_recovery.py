"""Synthetic recovery checks: no medical data or network access."""

from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsvt.data.acquisition import sha256_file
from rsvt.data.recovery import locate_ctgan_dataset, restore_ctgan_archive


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.work = TemporaryDirectory(prefix="rsvt_recovery_")
        self.root = Path(self.work.name)

    def tearDown(self):
        self.work.cleanup()

    def layout(self, root):
        for name in ("Experiment 1 - Blind", "Experiment 2 - Open"):
            (root / "Tampered Scans" / name).mkdir(parents=True)
        for name in ("labels_exp1.csv", "labels_exp2.csv"):
            (root / "Tampered Scans" / name).write_text("type,uuid,slice,x,y\n")

    def archive(self, name="cached.zip", unsafe=False):
        inner = BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            for name_part in ("Experiment 1 - Blind", "Experiment 2 - Open"):
                z.writestr("Tampered Scans/" + name_part + "/1/synthetic.txt", "fixture")
            for label in ("labels_exp1.csv", "labels_exp2.csv"):
                z.writestr("Tampered Scans/" + label, "type,uuid,slice,x,y\n")
            if unsafe:
                z.writestr("../escape.txt", "fixture")
        archive = self.root / name
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("data.zip", inner.getvalue())
        return archive

    def test_drive_raw_is_located_without_confusing_processed_outputs(self):
        drive = self.root / "drive"
        self.layout(drive / "raw/ct_gan/uci/dataset")
        found = locate_ctgan_dataset(self.root / "missing", drive)
        self.assertEqual(found, drive / "raw/ct_gan/uci/dataset")
        with self.assertRaises(ValueError):
            locate_ctgan_dataset(self.root / "missing", drive, drive / "processed")

    def test_missing_raw_returns_none_even_if_processed_exists(self):
        (self.root / "processed/run/exp1_1").mkdir(parents=True)
        self.assertIsNone(locate_ctgan_dataset(self.root / "missing", self.root))

    def test_cached_nested_archive_restores_with_zero_download_budget(self):
        archive = self.archive()
        original = archive.read_bytes()
        with patch("rsvt.data.recovery.require_colab_runtime"), patch("rsvt.data.recovery.fetch_file") as fetch:
            result = restore_ctgan_archive(archive, self.root / "restored", download_budget_bytes=0)
        fetch.assert_not_called()
        self.assertEqual(result["new_archive_bytes"], 0)
        self.assertTrue(Path(result["dataset_root"]).is_dir())
        self.assertEqual(archive.read_bytes(), original)
        self.assertEqual(result["f01_source_equivalence"], "pending_phase2_hash_comparison")

    def test_mismatched_cached_receipt_is_rejected_before_extraction(self):
        archive = self.archive()
        archive.with_name(archive.name + ".receipt.json").write_text(json.dumps({"bytes": archive.stat().st_size, "sha256": "0"*64}))
        with patch("rsvt.data.recovery.require_colab_runtime"), self.assertRaisesRegex(ValueError, "differs"):
            restore_ctgan_archive(archive, self.root / "restored", download_budget_bytes=0)
        self.assertFalse((self.root / "restored").exists())

    def test_unsafe_nested_archive_does_not_escape(self):
        archive = self.archive(unsafe=True)
        with patch("rsvt.data.recovery.require_colab_runtime"), self.assertRaises(ValueError):
            restore_ctgan_archive(archive, self.root / "restored", download_budget_bytes=0)
        self.assertFalse((self.root / "restored/uci/escape.txt").exists())

    def test_local_runtime_rejected_before_network_or_disk_activity(self):
        with patch("rsvt.data.recovery.require_colab_runtime", side_effect=RuntimeError("remote only")):
            with patch("rsvt.data.recovery.fetch_file") as fetch, self.assertRaises(RuntimeError):
                restore_ctgan_archive(self.root / "new.zip", self.root / "new", download_budget_bytes=100)
        fetch.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_new_download_respects_budget_and_writes_restoration_receipt(self):
        source = self.archive()
        destination = self.root / "drive/new.zip"
        def fake_fetch(url, target, **kwargs):
            self.assertEqual(kwargs["max_bytes"], 50000)
            target.parent.mkdir(parents=True)
            target.write_bytes(source.read_bytes())
            return {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
        with patch("rsvt.data.recovery.require_colab_runtime"), patch("rsvt.data.recovery.fetch_file", side_effect=fake_fetch):
            result = restore_ctgan_archive(destination, self.root / "restored", download_budget_bytes=50000)
        self.assertEqual(result["new_archive_bytes"], destination.stat().st_size)
        self.assertTrue((self.root / "restored/restoration.json").is_file())


if __name__ == "__main__":
    unittest.main()
