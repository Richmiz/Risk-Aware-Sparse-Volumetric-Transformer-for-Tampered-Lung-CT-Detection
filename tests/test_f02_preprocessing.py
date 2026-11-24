"""Synthetic CT geometry/HU checks; no real medical data or network requests."""

from itertools import product
from pathlib import Path
import csv
import json
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsvt.data.io import GeometryPolicy, SeriesValidationError, inspect_series, read_hu_volume, source_fingerprint
from rsvt.data.preprocessing import PreprocessingSpec, canonical_reference, preprocess_volume, resample_mask, transform_points, transform_box
from rsvt.data.phase2 import run_phase2, map_ctgan_annotations, source_group_candidates
from rsvt.data.inventory import inventory_data
from rsvt.data.ct_gan import audit_ctgan_labels


class SyntheticFixture(unittest.TestCase):
    def setUp(self):
        self.work = TemporaryDirectory(prefix="rsvt_f02_")
        self.root = Path(self.work.name)

    def tearDown(self):
        self.work.cleanup()

    def series(self, folder=None, *, zs=(4, 0, 2), names=("9", "2", "1"),
               orientation=(1, 0, 0, 0, 1, 0), uid=None, patient="SYNTHETIC", padding=None):
        folder = folder or self.root / "series"
        folder.mkdir(parents=True)
        uid, frame = uid or generate_uid(), generate_uid()
        for i, (z, name) in enumerate(zip(zs, names)):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            path = folder / f"{name}.dcm"
            image = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
            image.SOPClassUID, image.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
            image.SeriesInstanceUID, image.FrameOfReferenceUID, image.PatientID = uid, frame, patient
            image.Modality, image.Rows, image.Columns = "CT", 3, 4
            image.SamplesPerPixel, image.PhotometricInterpretation = 1, "MONOCHROME2"
            image.BitsAllocated, image.BitsStored, image.HighBit, image.PixelRepresentation = 16, 16, 15, 1
            image.ImageOrientationPatient = list(orientation)
            normal = np.cross(orientation[:3], orientation[3:])
            image.ImagePositionPatient = (np.array([10., 20., 0.]) + z * normal).tolist()
            image.PixelSpacing = [2., .7]
            image.RescaleSlope, image.RescaleIntercept, image.RescaleType = i + 1, -1000 + i * 100, "HU"
            image.InstanceNumber = 100 - i
            pixels = np.full((3, 4), 10 * (i + 1), dtype=np.int16)
            if padding is not None:
                image.PixelPaddingValue = padding
                pixels[0, 0] = padding
            image.PixelData = pixels.tobytes()
            image.save_as(path, enforce_file_format=True)
        return folder

    def change(self, path, **values):
        image = pydicom.dcmread(path)
        for key, value in values.items():
            if value is None:
                delattr(image, key)
            else:
                setattr(image, key, value)
        image.save_as(path, enforce_file_format=True)


class SyntheticCT(SyntheticFixture):
    def test_physical_sort_and_per_slice_hu(self):
        geometry = inspect_series(self.series())
        self.assertEqual([r.path.stem for r in geometry.slices], ["2", "1", "9"])
        self.assertEqual(geometry.spacing_xyz_mm, (.7, 2., 2.))
        volume = read_hu_volume(geometry)
        np.testing.assert_allclose(volume.hu_zyx[:, 1, 1], [-860, -710, -990])
        self.assertEqual(volume.hu_zyx.dtype, np.float32)
        image, valid = volume.as_images()
        np.testing.assert_allclose(image.TransformIndexToPhysicalPoint((1, 1, 1)), [10.7, 22, 2])

    def test_unsigned_pixels_are_not_reinterpreted_as_signed(self):
        folder = self.series()
        image = pydicom.dcmread(folder / "2.dcm")
        image.PixelRepresentation = 0
        image.PixelData = np.full((3, 4), 40000, dtype=np.uint16).tobytes()
        image.save_as(folder / "2.dcm", enforce_file_format=True)
        volume = read_hu_volume(inspect_series(folder))
        self.assertEqual(volume.hu_zyx[0, 1, 1], 79100.)

    def test_padding_has_separate_validity(self):
        volume = read_hu_volume(inspect_series(self.series(padding=-2000)))
        self.assertFalse(volume.valid_zyx[0, 0, 0])
        self.assertTrue(volume.valid_zyx[0, 1, 1])
        self.assertEqual(volume.hu_zyx[0, 0, 0], -4900.)

    def test_missing_units_are_visible_as_a_warning(self):
        folder = self.series()
        self.change(folder / "1.dcm", RescaleType=None)
        self.assertTrue(any("HU_assumption" in warning for warning in inspect_series(folder).warnings))

    def test_rejects_duplicate_planes(self):
        with self.assertRaisesRegex(SeriesValidationError, "Duplicate"):
            inspect_series(self.series(zs=(0, 0, 2)))

    def test_rejects_irregular_spacing(self):
        with self.assertRaisesRegex(SeriesValidationError, "Irregular"):
            inspect_series(self.series(zs=(0, 1, 3)))

    def test_rejects_missing_hu_rescale(self):
        folder = self.series()
        self.change(folder / "1.dcm", RescaleSlope=None)
        with self.assertRaisesRegex(SeriesValidationError, "RescaleSlope"):
            inspect_series(folder)

    def test_rejects_wrong_units_and_multiframe(self):
        folder = self.series()
        self.change(folder / "1.dcm", RescaleType="US")
        with self.assertRaisesRegex(SeriesValidationError, "not HU"):
            inspect_series(folder)
        self.change(folder / "1.dcm", RescaleType="HU", NumberOfFrames=2)
        with self.assertRaisesRegex(SeriesValidationError, "multi-frame"):
            inspect_series(folder)

    def test_rejects_mixed_series(self):
        folder = self.series()
        self.change(folder / "1.dcm", SeriesInstanceUID=generate_uid())
        with self.assertRaisesRegex(SeriesValidationError, "Mixed"):
            inspect_series(folder)

    def test_rejects_orientation_change_and_tilt(self):
        folder = self.series()
        self.change(folder / "1.dcm", ImagePositionPatient=[10.2, 20., 2.])
        with self.assertRaisesRegex(SeriesValidationError, "gantry"):
            inspect_series(folder)
        self.change(folder / "1.dcm", ImagePositionPatient=[10., 20., 2.], ImageOrientationPatient=[0, 1, 0, -1, 0, 0])
        with self.assertRaisesRegex(SeriesValidationError, "orientations"):
            inspect_series(folder)

    def test_source_change_is_detected_before_decode_or_hash(self):
        geometry = inspect_series(self.series())
        with geometry.slices[0].path.open("ab") as stream:
            stream.write(b"changed")
        for operation in (read_hu_volume, source_fingerprint):
            with self.assertRaisesRegex(SeriesValidationError, "changed"):
                operation(geometry)

    def test_memory_limit_before_allocation(self):
        with self.assertRaisesRegex(SeriesValidationError, "voxel count"):
            inspect_series(self.series(), GeometryPolicy(max_input_voxels=1))

    def test_oblique_geometry_preserves_physical_positions(self):
        a = 2 ** -.5
        geometry = inspect_series(self.series(orientation=(1, 0, 0, 0, a, a)))
        image, _ = read_hu_volume(geometry).as_images()
        np.testing.assert_allclose(image.TransformIndexToPhysicalPoint((1, 1, 1)), [10.7, 20., 4*a], atol=1e-6)


class SpatialTransforms(unittest.TestCase):
    def image(self, direction=None):
        array = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6) - 1050
        image = sitk.GetImageFromArray(array)
        image.SetOrigin((10., -8., 3.))
        image.SetSpacing((.7, 1.2, 2.5))
        if direction:
            image.SetDirection(direction)
        valid = sitk.GetImageFromArray(np.ones_like(array, dtype=np.uint8))
        valid.CopyInformation(image)
        return image, valid

    def test_identity_resampling_and_window(self):
        image, valid = self.image()
        spec = PreprocessingSpec(target_spacing_xyz_mm=image.GetSpacing(), hu_window=(-1040., -960.))
        result = preprocess_volume(image, valid, spec)
        np.testing.assert_allclose(sitk.GetArrayFromImage(result["hu"]), sitk.GetArrayFromImage(image), atol=1e-5)
        expected = np.clip((sitk.GetArrayFromImage(image) + 1040) / 80, 0, 1)
        np.testing.assert_allclose(sitk.GetArrayFromImage(result["normalized"]), expected, atol=1e-6)

    def test_full_fov_and_point_roundtrip_for_oblique_image(self):
        a = 2 ** -.5
        image, _ = self.image((a, -a, 0., a, a, 0., 0., 0., 1.))
        target = canonical_reference(image, PreprocessingSpec())
        for corner in product(*[(-.5, size - .5) for size in image.GetSize()]):
            out = target.TransformPhysicalPointToContinuousIndex(image.TransformContinuousIndexToPhysicalPoint(corner))
            self.assertTrue(np.all(np.asarray(out) >= -.5 - 1e-6))
            self.assertTrue(np.all(np.asarray(out) <= np.asarray(target.GetSize()) - .5 + 1e-6))
        mapped = transform_points([[2., 3., 1.]], image, target)
        recovered = image.TransformPhysicalPointToContinuousIndex(mapped["physical_lps_mm"][0])
        np.testing.assert_allclose(recovered, [2, 3, 1], atol=1e-6)
        self.assertTrue(mapped["inside_target_support"][0])

    def test_mask_keeps_categorical_labels(self):
        image, _ = self.image()
        mask = sitk.GetImageFromArray(np.where(sitk.GetArrayFromImage(image) > -1000, 2, 0).astype(np.uint8))
        mask.CopyInformation(image)
        result = resample_mask(mask, canonical_reference(image, PreprocessingSpec()))
        self.assertTrue(set(np.unique(sitk.GetArrayFromImage(result))).issubset({0, 2}))

    def test_invalid_points_boxes_specs_and_output_size(self):
        image, _ = self.image()
        with self.assertRaises(ValueError):
            transform_points([[999, 0, 0]], image, image)
        with self.assertRaises(ValueError):
            transform_box([2, 0, 0], [1, 2, 3], image, image)
        for args in ({"hu_window": (1, 1)}, {"target_spacing_xyz_mm": (0, 1, 1)}):
            with self.assertRaises(ValueError):
                PreprocessingSpec(**args)
        with self.assertRaisesRegex(ValueError, "voxel count"):
            canonical_reference(image, PreprocessingSpec(max_output_voxels=1))

    def test_validity_geometry_mismatch_rejected(self):
        image, valid = self.image()
        valid.SetOrigin((0., 0., 0.))
        with self.assertRaisesRegex(ValueError, "geometry"):
            preprocess_volume(image, valid, PreprocessingSpec())

    def test_box_corners_are_conservatively_mapped(self):
        image, _ = self.image()
        box = transform_box([0, 1, 0], [3, 4, 2], image, image)
        np.testing.assert_allclose(box["lower_index_xyz"], [0, 1, 0])
        np.testing.assert_allclose(box["upper_index_xyz"], [3, 4, 2])

    def test_padding_is_background_before_resampling(self):
        image, valid = self.image()
        image[0, 0, 0] = -9000
        valid[0, 0, 0] = 0
        result = preprocess_volume(image, valid, PreprocessingSpec(target_spacing_xyz_mm=image.GetSpacing()))
        self.assertEqual(result["hu"][0, 0, 0], -1024.)
        self.assertEqual(result["valid"][0, 0, 0], 0)


class Phase2Integration(SyntheticFixture):
    def prepare_run(self):
        dataset = self.root / "data"
        uid = generate_uid()
        for experiment in ("Experiment 1 - Blind", "Experiment 2 - Open"):
            self.series(dataset / "Tampered Scans" / experiment / "1", uid=uid)
        columns = ("type", "uuid", "slice", "x", "y")
        for filename, category in (("labels_exp1.csv", "TM"), ("labels_exp2.csv", "FB")):
            with (dataset / "Tampered Scans" / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerow(dict(zip(columns, (category, "1", "9", "1", "1"))))
        config = self.root / "inventory.toml"
        config.write_text('[inventory]\nhash_files=true\nidentity_key_file="private/key"\n[datasets.ct_gan]\nroot="data"\nrole="forensic"\n')
        run = self.root / "f01"
        inventory_data(config, self.root, run / "files")
        audit_ctgan_labels(dataset, run / "ct_gan_labels")
        return dataset, run

    def test_batch_pilot_export_and_source_grouping(self):
        dataset, inventory = self.prepare_run()
        report = run_phase2(dataset, inventory, ROOT / "configs/preprocessing/ct_gan_pilot.toml",
                            self.root / "reports", self.root / "volumes", max_cases=1)
        self.assertEqual(report["geometry_valid"], 2)
        self.assertEqual(report["processed_success"], 1)
        self.assertEqual(report["unprocessed_geometry_valid"], 1)
        self.assertEqual(report["source_groups_candidates"], 1)
        self.assertFalse(report["training_ready"])
        output = Path(report["cases"][0]["output"])
        metadata = json.loads((output / "metadata.json").read_text())
        point = metadata["annotations"][0]
        self.assertEqual(point["native_index_xyz"], [1, 1, 2])
        self.assertFalse(point["supervision_eligible"])
        reread = sitk.ReadImage(str(output / "hu_processed.nii.gz"))
        self.assertEqual(list(reread.GetSize()), metadata["transform"]["processed"]["size_xyz"])
        np.testing.assert_allclose(reread.GetOrigin(), metadata["transform"]["processed"]["origin_lps_mm"], atol=1e-5)
        np.testing.assert_allclose(reread.GetSpacing(), metadata["transform"]["processed"]["spacing_xyz_mm"], atol=1e-5)
        np.testing.assert_allclose(reread.GetDirection(), metadata["transform"]["processed"]["direction"], atol=1e-6)
        self.assertTrue((output / "valid_native.nii.gz").is_file())
        with self.assertRaises(FileExistsError):
            run_phase2(dataset, inventory, ROOT / "configs/preprocessing/ct_gan_pilot.toml",
                       self.root / "reports", self.root / "volumes")

    def test_filename_annotation_mapping_and_invalid_reference(self):
        geometry = inspect_series(self.series())
        source, valid = read_hu_volume(geometry).as_images()
        target = preprocess_volume(source, valid, PreprocessingSpec())["hu"]
        rows = [{"location_label": "FB", "slice_reference": "9", "pixel_x": "1", "pixel_y": "1"},
                {"location_label": "TB", "slice_reference": "0", "pixel_x": "0", "pixel_y": "0"},
                {"location_label": "FB", "slice_reference": "999", "pixel_x": "1", "pixel_y": "1"}]
        mapped = map_ctgan_annotations(rows, geometry, source, target, 0)
        self.assertEqual(mapped[0]["native_index_xyz"], [1, 1, 2])
        self.assertEqual(mapped[1]["status"], "zero_reference_not_localization_target")
        self.assertEqual(mapped[2]["status"], "slice_reference_missing_or_ambiguous")

    def test_transitive_source_groups(self):
        rows = [{"candidate_id": "a", "patient_metadata_keys": "p", "series_metadata_key": "s1"},
                {"candidate_id": "b", "patient_metadata_keys": "p", "series_metadata_key": "s2"},
                {"candidate_id": "c", "patient_metadata_keys": "q", "series_metadata_key": "s2"}]
        groups = source_group_candidates(rows)
        self.assertEqual(len(set(groups.values())), 1)

    def test_rejected_geometry_produces_a_report_without_fake_success(self):
        dataset, inventory = self.prepare_run()
        for folder in dataset.glob('Tampered Scans/Experiment */1'):
            self.change(folder / "1.dcm", ImagePositionPatient=[10., 20., 0.])
        report = run_phase2(dataset, inventory, ROOT / "configs/preprocessing/ct_gan_pilot.toml",
                            self.root / "reports", self.root / "volumes", max_cases=0)
        self.assertFalse(report["ok"])
        self.assertEqual(report["geometry_rejected"], 2)
        self.assertEqual(report["processed_success"], 0)
        self.assertTrue((self.root / "reports/geometry_audit.json").exists())


if __name__ == "__main__":
    unittest.main()
