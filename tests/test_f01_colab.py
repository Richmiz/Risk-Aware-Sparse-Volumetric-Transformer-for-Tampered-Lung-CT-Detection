"""Check remote-only acquisition and exclusion of medical data from code bundles."""

from pathlib import Path
import importlib.util
import json
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsvt.utils.runtime import require_colab_runtime


class ColabPolicyChecks(unittest.TestCase):
    def test_windows_kernel_cannot_start_acquisition(self):
        with patch("rsvt.utils.runtime.platform.system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "local dataset downloads are disabled"):
                require_colab_runtime()

    def test_non_colab_linux_kernel_is_rejected(self):
        with patch("rsvt.utils.runtime.platform.system", return_value="Linux"):
            with patch("rsvt.utils.runtime.importlib.util.find_spec", return_value=None):
                with self.assertRaises(RuntimeError):
                    require_colab_runtime()

    def test_bundle_omits_datasets_keys_outputs_and_environments(self):
        spec = importlib.util.spec_from_file_location("rsvt_bundle_builder", ROOT / "scripts/build_colab_bundle.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with TemporaryDirectory(prefix="rsvt_bundle_") as work:
            root = Path(work)
            fixtures = {
                "README.md": "source instructions",
                "src/rsvt/module.py": '"""Source only."""',
                "notebooks/executed.ipynb": json.dumps({"nbformat": 4, "nbformat_minor": 5,
                    "metadata": {"widgets": {"private": "fixture"}}, "cells": [
                        {"cell_type": "code", "source": ["print('code')"], "metadata": {"execution": "fixture"},
                         "execution_count": 1, "outputs": [{"data": {"image/png": "medical preview fixture"}}]},
                        {"cell_type": "markdown", "source": ["Instructions"], "metadata": {},
                         "attachments": {"image": {"image/png": "medical preview fixture"}}}]}),
                "data/public_sources.json": "{}",
                "data/manifests/manifest.template.csv": "sample_id\n",
                "data/raw/patient.dcm": "medical data fixture",
                "data/private/inventory_identity.key": "private fixture",
                "data/manifests/local.csv": "private manifest fixture",
                ".venv/local.py": "environment fixture",
                "outputs/predictions.csv": "prediction fixture",
                ".env": "secret fixture",
            }
            for name, content in fixtures.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            result = builder.build_bundle(root, root / "artifacts/code.zip")
            with zipfile.ZipFile(result["bundle"]) as archive:
                members = set(archive.namelist())
                notebook = json.loads(archive.read("notebooks/executed.ipynb"))
            self.assertEqual(notebook["cells"][0]["outputs"], [])
            self.assertIsNone(notebook["cells"][0]["execution_count"])
            self.assertEqual(notebook["cells"][0]["source"], ["print('code')"])
            self.assertNotIn("attachments", notebook["cells"][1])
            self.assertEqual(notebook["metadata"], {})
            self.assertEqual(members, {"README.md", "src/rsvt/module.py",
                                       "notebooks/executed.ipynb",
                                       "data/public_sources.json", "data/manifests/manifest.template.csv"})
            self.assertFalse(result["contains_datasets"])


if __name__ == "__main__":
    unittest.main()
