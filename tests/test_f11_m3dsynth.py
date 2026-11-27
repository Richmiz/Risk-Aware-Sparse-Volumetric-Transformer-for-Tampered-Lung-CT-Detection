"""F11: M3Dsynth store, coarse-to-fine RSVT, endpoint losses and trainer.

Procedural CT series and TIFF archives only; no medical images are read.
"""

from collections import Counter
import csv
import errno
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsvt.data import m3dsynth_store as store_mod
from rsvt.data.m3dsynth_dataset import (AreaResampler, M3DDataset, M3DStore, VolumeCase, _rng, coarse_builder,
                                        crop_tamper_threshold, intensity_lut)
from rsvt.losses.endpoint_risk import dice_bce, generator_variance, grouped_tail_risk
from rsvt.models.coarse_to_fine import CoarseToFineRSVT, compute_estimate
from rsvt.training import coarse_to_fine as trainer

SHAPE = (20, 64, 64)
SPACING = (2.5, .7, .7)
PATIENTS = {"LIDC-IDRI-0001": ("train", "3000001"), "LIDC-IDRI-0002": ("train", "3000002"),
            "LIDC-IDRI-0003": ("valid", "3000003")}
SITES = [("rem_1", "LIDC-IDRI-0001", (10, 30, 30)), ("inj_2", "LIDC-IDRI-0001", (6, 20, 44)),
         ("rem_3", "LIDC-IDRI-0002", (12, 34, 26)), ("inj_4", "LIDC-IDRI-0002", (9, 40, 40)),
         ("rem_5", "LIDC-IDRI-0003", (10, 32, 32)), ("inj_6", "LIDC-IDRI-0003", (8, 24, 36))]
GENS = ("cycle", "pix2pix")
REPO = Path(__file__).resolve().parents[1]
# Author metadata and the audited series map are local-only (not redistributed).
AUDITED_METADATA = REPO / "data/raw/m3dsynth/metadata"
AUDITED_UID_MAP = REPO / "reports/supplementary/data_label_audit_20261111/metadata_review_v2/m3dsynth_source_groups.csv"


def dicom_path(patient):
    return f"{patient}/01-01-2000-NA-NA-12345/{PATIENTS[patient][1]}.000000-NA-54321"


def raw_volume(seed):
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, n) for n in SHAPE], indexing="ij")
    body = (y ** 2 + x ** 2) < .8
    values = np.where(body, -850 + 300 * np.sin(4 * z) * np.cos(3 * x) + rng.normal(0, 30, SHAPE), -1024)
    return np.clip(values, -1024, 1500).astype(np.int16)


def write_series(root, patient, raw):
    folder = root / dicom_path(patient)
    folder.mkdir(parents=True)
    for i, plane in enumerate(raw):  # file order deliberately scrambled
        path = folder / f"1-{(7 * i) % len(raw):03d}{i}.dcm"
        meta = FileMetaDataset()
        meta.TransferSyntaxUID, meta.MediaStorageSOPClassUID = ExplicitVRLittleEndian, CTImageStorage
        meta.MediaStorageSOPInstanceUID = f"1.2.826.0.1.{int(PATIENTS[patient][1])}.{i}"
        d = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        d.SOPClassUID, d.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        d.PatientID, d.Modality, d.SeriesNumber = patient, "CT", int(PATIENTS[patient][1])
        d.SeriesInstanceUID, d.StudyInstanceUID = f"1.2.826.0.9.{PATIENTS[patient][1]}54321", "1.2.826.0.8"
        d.Rows, d.Columns = SHAPE[1], SHAPE[2]
        d.SamplesPerPixel, d.PhotometricInterpretation = 1, "MONOCHROME2"
        d.BitsAllocated, d.BitsStored, d.HighBit, d.PixelRepresentation = 16, 16, 15, 1
        d.RescaleSlope, d.RescaleIntercept = 1, 0
        # Descending z in the authors' order: stack index i has z = -i * 2.5.
        d.ImagePositionPatient = [-22., -22., -SPACING[0] * i]
        d.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        d.PixelSpacing = [SPACING[1], SPACING[2]]
        d.SliceThickness = SPACING[0]
        d.PixelData = plane.astype(np.int16).tobytes()
        d.save_as(path, enforce_file_format=True)


def tiff(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="TIFF", compression="tiff_lzma")
    return buffer.getvalue()


def manipulate(u16, center, seed):
    rng = np.random.default_rng(seed)
    out, mask = u16.copy(), np.zeros(u16.shape, bool)
    z, y, x = center
    region = (slice(z - 2, z + 3), slice(y - 4, y + 5), slice(x - 4, x + 5))
    out[region] = (rng.integers(900, 1400, out[region].shape)).astype(np.uint16)
    mask[(slice(z - 1, z + 2), slice(y - 3, y + 4), slice(x - 3, x + 4))] = True
    return out, mask


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.meta_dir, self.lidc, self.store, self.work = (self.root / n for n in ("meta", "lidc", "store", "work"))
        self.meta_dir.mkdir()
        self.raw = {p: raw_volume(i) for i, p in enumerate(PATIENTS)}
        for patient, raw in self.raw.items():
            write_series(self.lidc, patient, raw)
        with (self.meta_dir / "LIDC.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["orig_id", "sdir_id", "spacing_z", "spacing_y", "spacing_x", "dicom"])
            for p, (_, s) in PATIENTS.items():
                w.writerow([p, s, SPACING[0], SPACING[1], SPACING[2], dicom_path(p)])
        with (self.meta_dir / "sets.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["orig_id", "set"])
            for p, (split, _) in PATIENTS.items():
                w.writerow([p, split])
        with (self.meta_dir / "data.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["img_id", "ty", "mod", "orig_id", "sdir_id", "coord_z", "coord_y", "coord_x"])
            for img_id, patient, (z, y, x) in SITES:
                for mod in ("real", *GENS):
                    w.writerow([img_id, img_id.split("_")[0], mod, patient, PATIENTS[patient][1], z, y, x])

    def acquire(self, *, outside_diff=None, half_mm=12.):
        meta = store_mod.load_metadata(self.meta_dir)
        selection = store_mod.select_subset(meta, patients_per_split={"train": None, "validation": None, "test": 0},
                                            generators=GENS, seed=13)
        build = coarse_builder()
        real_meta = {}
        for item in selection["series"]:
            real_meta[item["series_key"]] = store_mod.acquire_real_series(
                item, self.store, self.work, lidc_root=self.lidc, coarse_builder=build,
                fetch=lambda url: (_ for _ in ()).throw(AssertionError("network used")))
        self.u16 = {k: np.load(self.work / "real" / (k + ".npy")) for k in real_meta}
        self.edits = {}
        for generator in GENS:
            members = []
            for img_id, patient, center in SITES:
                key = store_mod.series_key(patient, PATIENTS[patient][1])
                edited, mask = manipulate(self.u16[key], center, hash((generator, img_id)) % 1000)
                if outside_diff == (generator, img_id):
                    edited[0, 0, 0] += 7  # far outside the stored box
                self.edits[(generator, img_id)] = (edited, mask)
                members += [(f"{generator}/label/{img_id}/slide{z:04d}.tiff", tiff(mask[z])) for z in range(SHAPE[0])]
                members += [(f"{generator}/scan/{img_id}/slide{z:04d}.tiff", tiff(edited[z])) for z in range(SHAPE[0])]
            members.sort(key=lambda m: (0 if "/label/" in m[0] else 1, m[0]))  # labels first: spill path
            archive = self.root / f"{generator}.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                for name, data in members:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, BytesIO(data))
            store_mod.stream_generator(generator, selection["samples"], real_meta, self.store, self.work,
                                       source=archive, half_mm=half_mm)
        self.selection, self.meta = selection, meta
        return store_mod.build_manifest(self.store, selection, meta)


def small_config(**training):
    config = {
        "data": {"generators": list(GENS), "crop_mm": 16., "grid": 12, "flip_probability": .5, "forced_sites": 1,
                 "random_crops": 1, "jitter_fraction": .25},
        "acquisition": {"patients_train": 2, "patients_validation": 1, "patients_test": 0, "half_box_mm": 12., "seed": 13},
        "model": {"dimension": 16, "heads": 2, "layers": 1, "dropout": .1, "coarse_widths": [4, 8, 8],
                  "fine_widths": [4, 8, 8], "router_hidden": 8, "router_prior": .01, "crops_at_inference": 2},
        "training": {"mode": "engineering_pilot", "device": "cpu", "seed": 13, "threads": 1, "epochs": 2,
                     "learning_rate": 1e-3, "weight_decay": 1e-4, "gradient_clip": 5., "patience": 0, "min_delta": 0.,
                     "exclude_norm_bias_decay": True, "lr_schedule": "warmup_cosine", "warmup_epochs": 1,
                     "warmup_start_ratio": .5, "min_lr_ratio": .1, "authentic_per_batch": 1, "manipulated_per_batch": 2,
                     "max_batches_per_epoch": 1, "router_crops_training": 1, "risk_start_epoch": 1},
        "loss": {"volume": 1., "removal": .5, "crop_tamper": .5, "crop_visible": .25, "mask": .5, "router": .5,
                 "router_pos_weight": 20.},
        "risk": {"alpha": .5, "cvar_weight": .3, "authentic_weight": 1., "fb_weight": 2., "fm_weight": 1.,
                 "generator_weight": .1, "min_generator_samples": 1},
        "evaluation": {"fpr_targets": [.5]}}
    config["training"].update(training)
    return config


class FakeResponse:
    def __init__(self, data, start, status, headers, stop):
        self.data, self.pos, self.status, self.headers, self.stop = data, start, status, headers, stop

    def readinto(self, buffer):
        if self.stop is not None and self.pos >= self.stop:
            raise ConnectionResetError("simulated dropped connection")
        end = len(self.data) if self.stop is None else min(len(self.data), self.stop)
        count = min(len(buffer), end - self.pos)
        buffer[:count] = self.data[self.pos:self.pos + count]
        self.pos += count
        return count

    def close(self):
        pass


class FakeServer:
    """Serves bytes with HTTP Range support and drops each connection at the next listed offset."""

    def __init__(self, data, *, drops=(), honour_range=True, change_etag_on_reconnect=False):
        self.data, self.drops = data, list(drops)
        self.honour_range, self.change_etag = honour_range, change_etag_on_reconnect
        self.ranges = []

    def __call__(self, request, timeout=None):
        wanted = request.get_header("Range")
        self.ranges.append(wanted)
        etag = '"v2"' if self.change_etag and len(self.ranges) > 1 else '"v1"'
        stop = self.drops.pop(0) if self.drops else None
        if not wanted or not self.honour_range:
            return FakeResponse(self.data, 0, 200, {"Content-Length": str(len(self.data)), "ETag": etag}, stop)
        start = int(re.fullmatch(r"bytes=(\d+)-", wanted).group(1))
        return FakeResponse(self.data, start, 206,
                            {"Content-Range": f"bytes {start}-{len(self.data) - 1}/{len(self.data)}", "ETag": etag}, stop)


class ResumableArchiveStream(unittest.TestCase):
    @staticmethod
    def archive():
        rng = np.random.default_rng(3)
        members = {f"cycle/scan/inj_{i}/slide{z:04d}.tiff": rng.integers(0, 256, 70_000, dtype=np.uint8).tobytes()
                   for i in range(3) for z in range(4)}
        buffer = BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, BytesIO(data))
        return buffer.getvalue(), members

    def test_reconnects_at_the_same_byte_and_decodes_every_member(self):
        data, members = self.archive()
        drops = [len(data) // 5, len(data) // 2, len(data) * 4 // 5]
        server = FakeServer(data, drops=drops)
        stream = store_mod.open_archive_stream("https://example.invalid/cycle.tgz", opener=server, sleep=lambda s: None)
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            seen = {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}
        self.assertEqual(seen, members)
        self.assertEqual(server.ranges, [None] + [f"bytes={d}-" for d in drops])  # resumed exactly where each drop hit
        self.assertEqual((stream.raw.reconnects, stream.raw.position), (3, len(data)))
        stream.close()

    def test_tcia_downloads_retry_transient_errors(self):
        class Body:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b"series zip"

        calls = []

        def flaky(request, timeout=None):
            calls.append(request.full_url)
            if len(calls) < 3:
                raise ConnectionResetError("simulated drop")
            return Body()

        with mock.patch.object(store_mod.urllib.request, "urlopen", flaky):
            self.assertEqual(store_mod._fetch("https://example.invalid/getImage", sleep=lambda s: None), b"series zip")
        self.assertEqual(len(calls), 3)
        with mock.patch.object(store_mod.urllib.request, "urlopen", side_effect=TimeoutError("stalled")):
            with self.assertRaises(TimeoutError):
                store_mod._fetch("https://example.invalid/getImage", sleep=lambda s: None)

    def test_refuses_to_splice_or_retry_forever(self):
        data, _ = self.archive()
        cases = ((dict(honour_range=False), store_mod.StreamResumeError, "did not resume"),
                 (dict(change_etag_on_reconnect=True), store_mod.StreamResumeError, "changed on the server"),
                 (dict(drops=[100] * 20), OSError, "after 10 reconnect attempts"))
        for settings, error, message in cases:
            with self.subTest(settings=settings):
                settings.setdefault("drops", [len(data) // 3])
                stream = store_mod.open_archive_stream("https://example.invalid/x.tgz", opener=FakeServer(data, **settings),
                                                       sleep=lambda s: None)
                with self.assertRaisesRegex(error, message):
                    stream.read()


class Conversion(unittest.TestCase):
    def test_uint16_rules_match_the_author_conversion(self):
        scan = np.array([[[-1024, 0], [100, 3000]]], dtype=np.int16)
        u16, rule = store_mod.authors_uint16(scan)
        self.assertEqual(rule, {"rule": "negated_minimum", "offset": 1024})
        self.assertEqual(u16.tolist(), [[[0, 1024], [1124, 4024]]])
        same, rule = store_mod.authors_uint16(np.array([[[5]]], dtype=np.uint16))
        self.assertEqual((same.tolist(), rule["offset"]), ([[[5]]], 0))
        padded = np.full((12, 12, 12), -1024, np.int16)
        padded[5:7, 5:7, 5:7] = -3000  # minimum is not a candidate; corners decide
        u16, rule = store_mod.authors_uint16(padded)
        self.assertEqual((rule["rule"], rule["offset"]), ("negated_corner_median", 1024))
        self.assertEqual(int(u16[6, 6, 6]), 0)  # clipped to -offset before shifting
        wrap = np.array([[[-1024, 32000]]], dtype=np.int16)
        self.assertEqual(int(store_mod.authors_uint16(wrap)[0][0, 0, 1]), (32000 + 1024 - 65536) % 65536)
        with self.assertRaises(ValueError):
            store_mod.authors_uint16(np.zeros((1, 1, 1), np.float32))

    def test_metadata_fetch_is_pinned_verified_and_kept(self):
        payload = {name: f"{name} table".encode() for name in store_mod.METADATA_FILES}
        pins = {name: (remote, hashlib.sha256(payload[name]).hexdigest())
                for name, (remote, _) in store_mod.METADATA_FILES.items()}
        urls = []

        def fetch(url):
            urls.append(url)
            return payload[url.rsplit("/", 1)[-1]]

        with TemporaryDirectory() as tmp, mock.patch.dict(store_mod.METADATA_FILES, pins):
            folder = Path(tmp)
            self.assertEqual(sorted(store_mod.fetch_metadata(folder, fetch=fetch)["fetched"]), sorted(pins))
            self.assertTrue(all(f"/M3Dsynth/{store_mod.METADATA_COMMIT}/" in url for url in urls))
            self.assertEqual(store_mod.fetch_metadata(folder, fetch=fetch)["fetched"], [])  # verified copies kept
            (folder / "sets.csv").write_bytes(b"edited locally")
            with self.assertRaisesRegex(ValueError, "differs from the audited copy"):
                store_mod.fetch_metadata(folder, fetch=lambda url: b"tampered")
            self.assertEqual((folder / "sets.csv").read_bytes(), b"edited locally")  # nothing replaced

    @unittest.skipUnless((AUDITED_METADATA / "data.csv").is_file(), "local audited M3Dsynth metadata is not present")
    def test_pinned_metadata_checksums_equal_the_audited_copies(self):
        for name, (_, expected) in store_mod.METADATA_FILES.items():
            self.assertEqual(store_mod.sha256_path(AUDITED_METADATA / name), expected, name)

    @unittest.skipUnless((AUDITED_METADATA / "data.csv").is_file() and AUDITED_UID_MAP.is_file(),
                         "local M3Dsynth metadata and audited series map are not present")
    def test_audited_uid_map_covers_every_author_series(self):
        meta = store_mod.load_metadata(AUDITED_METADATA, AUDITED_UID_MAP)
        self.assertEqual(len(meta["series"]), 744)
        self.assertTrue(all(v["series_instance_uid"].startswith("1.3.6.1.4.1.14519") for v in meta["series"].values()))
        selection = store_mod.select_subset(meta, patients_per_split={"train": 3, "validation": 2, "test": 0})
        self.assertEqual(selection["counts"]["patients"], {"train": 3, "validation": 2})
        self.assertTrue(all(s["split"] in ("train", "validation") for s in selection["samples"]))
        again = store_mod.select_subset(meta, patients_per_split={"train": 3, "validation": 2, "test": 0})
        self.assertEqual(selection["patients"], again["patients"])

    def test_series_hint_and_box_clip(self):
        self.assertEqual(store_mod.folder_series_hint("P/x/3000611.000000-NA-03264"), (3000611, "03264"))
        start, stop = store_mod.box_for((1, 30, 62), (20, 64, 64), (2.5, .7, .7), 5.)
        self.assertEqual(start, [0, 22, 54])
        self.assertEqual(stop, [4, 39, 64])


class StoreAndDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.fx = Fixture(cls.tmp.name)
        cls.manifest = cls.fx.acquire(outside_diff=("pix2pix", "inj_4"))
        cls.store = M3DStore(cls.fx.store, Path(cls.tmp.name) / "cache")

    @classmethod
    def tearDownClass(cls):
        cls.store = None  # release Windows memory-map handles before cleanup
        import gc
        gc.collect()
        cls.tmp.cleanup()

    def test_router_candidates_never_come_from_the_label(self):
        volume = next(v for v in self.manifest["volumes"] if v["kind"] == "manipulated" and v["eligible"])
        common = dict(flip=False, grid=12, crop_mm=16., forced=0, random_centers=0, rng=_rng(1), jitter_fraction=0.)
        blind = VolumeCase(self.store, volume, body_threshold=2., **common)  # no cell passes the image test
        self.assertFalse(blind.cell_valid.any())
        self.assertTrue(blind.cell_target.any())
        np.testing.assert_array_equal(blind.router_cells, blind.cell_target > 0)  # positives still supervised
        normal = VolumeCase(self.store, volume, **common)
        self.assertTrue(np.all(normal.router_cells[normal.cell_valid]))
        authentic = {"kind": "authentic", "series_key": volume["series_key"], "volume_id": "real:" + volume["series_key"],
                     "operation": ""}
        reference = VolumeCase(self.store, authentic, body_threshold=2., **common)
        np.testing.assert_array_equal(reference.cell_valid, blind.cell_valid)

    def test_real_series_follow_descending_z_and_receipts(self):
        key = store_mod.series_key("LIDC-IDRI-0001", "3000001")
        info = self.manifest["series"][key]
        np.testing.assert_array_equal(self.fx.u16[key], (self.fx.raw["LIDC-IDRI-0001"].astype(np.int32) + 1024).astype(np.uint16))
        self.assertEqual(info["u16_to_hu"]["offset"], 1024)
        self.assertAlmostEqual(info["geometry"]["voxel_to_lps"][2][2], -2.5)
        self.assertTrue(info["geometry"]["uniform_spacing"] and info["inplane_spacing_matches_metadata"])
        self.assertEqual(info["origin"]["source"], "existing_lidc_root")
        expected = dict(self.fx.meta["series"][key], series_instance_uid="1.2.3.wrong")
        with self.assertRaisesRegex(ValueError, "audited TCIA mapping"):
            store_mod.convert_series(self.fx.lidc / dicom_path("LIDC-IDRI-0001"), expected)

    def test_boxes_are_verified_and_outside_differences_rejected(self):
        volumes = {v["volume_id"]: v for v in self.manifest["volumes"]}
        bad = volumes["pix2pix:inj_4"]
        self.assertFalse(bad["eligible"])
        self.assertTrue(any("outside the stored box" in p for p in bad["problems"]))
        good = [v for v in volumes.values() if v["kind"] == "manipulated" and v["volume_id"] != "pix2pix:inj_4"]
        self.assertEqual(len(good), len(SITES) * len(GENS) - 1)
        self.assertTrue(all(v["eligible"] and v["mask_voxels"] == 3 * 7 * 7 for v in good))
        self.assertTrue(store_mod.verify_store(self.fx.store, self.manifest)["ok"])
        self.assertEqual(self.manifest["patient_overlap_across_splits"], [])
        summary = store_mod.stream_generator("cycle", self.fx.selection["samples"], {}, self.fx.store, self.fx.work, half_mm=12.,
                                             source=self.fx.root / "cycle.tgz")
        self.assertEqual(summary["status"], "already_complete")  # resume skips verified samples

    def _reconstructed(self, volume):
        key = volume["series_key"]
        if volume["kind"] == "authentic":
            return self.fx.u16[key]
        return self.fx.edits[(volume["generator"], volume["img_id"])][0]

    def test_coarse_view_delta_is_exact(self):
        volume = next(v for v in self.manifest["volumes"] if v["volume_id"] == "cycle:rem_1")
        lut = intensity_lut(self.manifest["series"][volume["series_key"]]["u16_to_hu"])
        direct = AreaResampler().view(self._reconstructed(volume), lut)
        np.testing.assert_allclose(self.store.coarse(volume), direct, atol=2e-3)  # float16 real view
        self.assertEqual(direct.shape, (96, 96, 96))

    def test_crops_paste_box_and_flip_consistently(self):
        volume = next(v for v in self.manifest["volumes"] if v["volume_id"] == "cycle:rem_3")
        lut = intensity_lut(self.manifest["series"][volume["series_key"]]["u16_to_hu"])
        center, shape = (12., 34., 26.), (16, 32, 32)
        image, valid, mask = self.store.crop(volume, center, shape)
        full = lut[self._reconstructed(volume)]
        start = [int(round(c - s / 2)) for c, s in zip(center, shape)]
        expected = np.zeros(shape, np.float32)
        lo = [max(a, 0) for a in start]
        hi = [min(a + s, n) for a, s, n in zip(start, shape, SHAPE)]
        expected[tuple(slice(l - a, h - a) for l, h, a in zip(lo, hi, start))] = full[tuple(slice(l, h) for l, h in zip(lo, hi))]
        np.testing.assert_array_equal(image, expected)
        self.assertEqual(int(mask.sum()), 5 * 9 * 9)  # the changed voxels, wider than the 3x7x7 author mask
        dataset = M3DDataset(self.store, "train", train=True, flip_probability=1., crop_mm=16., forced_sites=1, random_crops=0)
        case = dataset[dataset.rows.index(volume)]
        self.assertTrue(case.flip)
        mirrored = case.to_view(np.asarray(center))
        flipped = case.crops([mirrored])
        np.testing.assert_array_equal(flipped["image"][0, 0].numpy(), image[..., ::-1])
        self.assertEqual(float(flipped["tamper"][0]), 1.)
        self.assertEqual(case.cell_target.sum() > 0, True)
        self.assertEqual(len(case.forced), 1)
        self.assertEqual(case.forced[0][1], 0.)  # removal: no visible nodule

    def test_labels_follow_the_changed_voxels_not_the_author_mask(self):
        volume = next(v for v in self.manifest["volumes"] if v["volume_id"] == "cycle:rem_1")
        edited, mask = self.fx.edits[("cycle", "rem_1")]
        changed = np.argwhere(edited != self.fx.u16[volume["series_key"]])
        g = 12

        def cells(points, flip=False):
            points = points.astype(float)
            if flip:
                points[:, 2] = SHAPE[2] - 1 - points[:, 2]
            c = np.floor((points + .5) * g / np.asarray(SHAPE)).astype(int)
            return sorted({(int(z) * g + int(y)) * g + int(x) for z, y, x in c})

        common = dict(grid=g, crop_mm=16., forced=0, random_centers=0, rng=_rng(1), jitter_fraction=0.)
        case = VolumeCase(self.store, volume, flip=False, **common)
        self.assertEqual(np.flatnonzero(case.cell_target).tolist(), cells(changed))
        # The author mask (3 slices) misses the cells reached only by the 5-slice edit.
        self.assertLess(len(cells(np.argwhere(mask))), len(cells(changed)))
        flipped = VolumeCase(self.store, volume, flip=True, **common)
        self.assertEqual(np.flatnonzero(flipped.cell_target).tolist(), cells(changed, flip=True))
        self.assertEqual(case.changed_total, 5 * 9 * 9)
        self.assertEqual(case.tamper_threshold, .25 * 5 * 9 * 9)  # the edit fits in one crop: the earlier rule

    def test_a_crop_needs_a_quarter_of_what_it_can_hold(self):
        self.assertEqual(crop_tamper_threshold(400, (10, 20, 20), (16, 32, 32)), 100.)
        # Edit three crop depths deep (a removal on thick slices): a centred crop holds a third.
        self.assertAlmostEqual(crop_tamper_threshold(900, (48, 40, 40), (16, 64, 64)), .25 * 900 / 3)

    def test_labels_must_match_the_edit_verified_while_streaming(self):
        volume = dict(next(v for v in self.manifest["volumes"] if v["volume_id"] == "pix2pix:rem_3"))
        (z0, y0, x0), hi = volume["diff_bbox"]
        volume["diff_bbox"] = [[z0 + 1, y0, x0], hi]
        store = M3DStore(self.fx.store, Path(self.tmp.name) / "cache_mismatch")
        with self.assertRaisesRegex(ValueError, "verified while streaming"):
            store.box(volume)
        report = self.store.label_check([v for v in self.manifest["volumes"] if v["kind"] == "manipulated" and v["eligible"]])
        removal = report["groups"]["removal"]
        self.assertEqual(removal["volumes"], len(GENS) * 3)
        self.assertEqual(removal["mask_changed"]["median"], 1.)
        self.assertAlmostEqual(removal["changed_in_mask"]["median"], 147 / 405)
        self.assertAlmostEqual(removal["depth_ratio"]["median"], 5 / 3)
        self.assertEqual(removal["changed_slice_fill"]["median"], 1.)
        self.assertIn("insertion:pix2pix", report["groups"])

    def test_validation_cases_are_unaugmented(self):
        dataset = M3DDataset(self.store, "validation", train=False)
        case = dataset[0]
        self.assertFalse(case.flip)
        self.assertEqual((case.forced, case.random), ([], []))
        self.assertTrue(all(v["split"] == "validation" for v in dataset.rows))

    def test_validation_site_crop_is_the_edit_or_a_real_site_and_never_trains(self):
        with self.assertRaises(ValueError):
            M3DDataset(self.store, "train", train=True, check_site=True)
        plain = M3DDataset(self.store, "validation", train=False)
        checked = M3DDataset(self.store, "validation", train=False, check_site=True)
        sources = Counter()
        for index in range(len(checked)):
            case, again, base = checked[index], checked[index], plain[index]
            self.assertIsNone(base.check_center)
            np.testing.assert_array_equal(case.check_center, again.check_center)  # the same site every epoch
            np.testing.assert_array_equal(case.cell_target, base.cell_target)  # nothing else about the case changes
            self.assertEqual((case.forced, case.random), ([], []))
            sources[case.check_source] += 1
            if case.tampered:
                self.assertEqual(case.check_source, "author_site")
                np.testing.assert_array_equal(case.check_center, case.site_center)
                self.assertEqual(float(case.crops([case.check_center])["tamper"][0]), 1.)  # the crop holds the edit
            else:
                self.assertIn(case.check_source, ("real_site_TB", "real_site_TM"))
        self.assertTrue(sources["author_site"] and sum(sources.values()) > sources["author_site"])


class ModelAndLosses(unittest.TestCase):
    def test_forward_backward_selection_and_estimate(self):
        torch.manual_seed(0)
        model = CoarseToFineRSVT(dimension=16, heads=2, layers=1, coarse_widths=(4, 8, 8), fine_widths=(4, 8, 8),
                                 router_hidden=8, router_prior=.01, crops_at_inference=3)
        coarse = model.encode_coarse(torch.rand(1, 1, 96, 96, 96))
        self.assertEqual(tuple(coarse["router_logits"].shape), (12 ** 3,))
        self.assertTrue(torch.allclose(torch.sigmoid(coarse["router_logits"]), torch.tensor(.01), atol=1e-6))
        crops = model.encode_crops(torch.rand(3, 1, 16, 32, 32), torch.ones(3, 1, 16, 32, 32, dtype=torch.bool))
        self.assertEqual(tuple(crops["mask_logits"].shape), (3, 1, 8, 16, 16))
        fused = model.fuse(coarse, crops, torch.zeros(3, 3))
        (fused["tamper_logit"] + fused["removal_logit"] + crops["mask_logits"].mean()
         + crops["visible_logits"].sum()).backward()
        self.assertTrue(all(p.grad is not None for n, p in model.named_parameters() if not n.startswith("coarse.router")))
        valid = torch.zeros(12 ** 3, dtype=torch.bool)
        valid[[5, 9, 40]] = True
        logits = torch.zeros(12 ** 3)
        logits[40] = 2.
        self.assertEqual(model.select_cells(logits, valid, 2).tolist(), [40, 5])
        estimate = compute_estimate(model, (96, 96, 96), (16, 32, 32), 3, (300, 512, 512))
        self.assertGreater(estimate["dense_over_sparse"], 1.)
        self.assertLess(estimate["native_voxels_read_fraction"], .01)

    def test_fine_branch_reads_the_residual_by_default(self):
        from rsvt.experiments.m3d_crop_check import residual
        from rsvt.models.coarse_to_fine import FineBranch, local_residual
        torch.manual_seed(0)
        model = CoarseToFineRSVT(dimension=16, heads=2, layers=1, coarse_widths=(4, 8, 8), fine_widths=(4, 8, 8),
                                 router_hidden=8, router_prior=.01, crops_at_inference=3).eval()
        self.assertEqual(model.fine.input, "residual")
        self.assertEqual(FineBranch(16, (4, 8, 8)).input, "image")  # the crop check's arms filter their own input
        with self.assertRaises(ValueError):
            FineBranch(16, (4, 8, 8), input="gradient")
        image = torch.rand(2, 1, 16, 32, 32)
        valid = torch.ones_like(image, dtype=torch.bool)
        torch.testing.assert_close(residual(image), local_residual(image), rtol=0, atol=0)  # the check's filter
        with torch.no_grad():
            # A uniform brightness change is not evidence for the residual branch, but moves the image branch.
            torch.testing.assert_close(model.encode_crops(image, valid)["tamper_logits"],
                                       model.encode_crops(image + .3, valid)["tamper_logits"])
            image_branch = FineBranch(16, (4, 8, 8)).eval()
            self.assertFalse(torch.allclose(image_branch(image, valid)["tamper_logits"],
                                            image_branch(image + .3, valid)["tamper_logits"]))

    def test_tail_risk_covers_both_sides_of_the_threshold(self):
        losses = torch.tensor([.1, 2., .3, .2, 1.5], requires_grad=True)
        groups = ["authentic", "authentic", "FB", "FB", "FM"]
        value, details = grouped_tail_risk(losses, groups, {"authentic": 1., "FB": 2., "FM": 1.}, .5)
        self.assertAlmostEqual(float(value.detach()), (2. + 2 * .3 + 1.5) / 4, places=6)
        value.backward()
        self.assertGreater(float(losses.grad[1]), 0.)  # the worst authentic case is penalized
        absent, _ = grouped_tail_risk(torch.tensor([.4]), ["authentic"], {"authentic": 1., "FB": 2.}, .5)
        self.assertAlmostEqual(float(absent), .4 / 3, places=6)
        variance, info = generator_variance(torch.tensor([1., 3., 2., 2.]), ["FB", "FB", "FM", "FM"],
                                            ["cycle", "pix2pix", "cycle", "cycle"], min_samples=1)
        self.assertAlmostEqual(float(variance), 1. / 2, places=6)
        self.assertEqual((info["FB"]["enabled"], info["FM"]["enabled"]), (True, False))

    def test_dice_bce_ignores_dice_on_empty_targets(self):
        logits = torch.full((2, 1, 2, 2, 2), -10.)
        target = torch.zeros(2, 1, 2, 2, 2)
        target[1, 0, 0, 0, 0] = 1.
        loss = dice_bce(logits, target, torch.ones_like(target, dtype=torch.bool))
        self.assertLess(float(loss[0]), 1e-3)
        self.assertGreater(float(loss[1]), .9)


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.fx = Fixture(cls.tmp.name)
        cls.fx.acquire()

    @classmethod
    def tearDownClass(cls):
        import gc
        gc.collect()
        cls.tmp.cleanup()

    def _setup(self, config):
        store = M3DStore(self.fx.store, Path(self.tmp.name) / "cache")
        train, validation = trainer.build_datasets(store, config)
        torch.manual_seed(1)
        model = CoarseToFineRSVT(**config["model"])
        return store, train, validation, model

    def test_replay_gradient_equals_direct_logical_batch_gradient(self):
        config = small_config(gradient_clip=1e9)  # compare raw gradients, not clipped ones
        _, train, _, model = self._setup(config)
        model.train()
        indices = trainer.epoch_batches(train, config, 0)[0]
        weights = trainer.class_weights(config)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        torch.manual_seed(5)
        terms, metas = {k: [] for k in trainer.TERMS}, []
        selections = []
        for index in indices:  # direct graph over the whole logical batch
            case = train[index]
            with torch.random.fork_rng():
                _, _, selection, _ = trainer.case_terms(model, case, "cpu", config, train=True)
            selections.append(selection)
        torch.manual_seed(5)
        for index, selection in zip(indices, selections):
            t, _, _, meta = trainer.case_terms(model, train[index], "cpu", config, train=True, selection=selection)
            for k in trainer.TERMS:
                terms[k].append(t[k])
            metas.append(meta)
        total, _ = trainer.objective({k: torch.stack(v) for k, v in terms.items()}, metas, config, weights, risk_active=True)
        direct = torch.autograd.grad(total, list(model.parameters()), allow_unused=True)
        model.load_state_dict(state)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        torch.manual_seed(5)
        trainer.replay_step(model, optimizer, train, indices, "cpu", config, weights, risk_active=True)
        for (name, p), g in zip(model.named_parameters(), direct):
            if g is None:
                self.assertTrue(p.grad is None or float(p.grad.abs().max()) == 0., name)
            else:
                torch.testing.assert_close(p.grad, g, rtol=1e-4, atol=1e-6, msg=name)

    def test_class_weights_balance_each_logical_batch(self):
        config = small_config(authentic_per_batch=4, manipulated_per_batch=12)
        weights = trainer.class_weights(config)
        self.assertAlmostEqual(4 * weights["authentic"], 12 * weights["manipulated"])
        metas = [{"tampered": i >= 4, "has_removal": i >= 4, "has_visible": False,
                  "group": "authentic" if i < 4 else "FM", "generator": "" if i < 4 else "cycle"} for i in range(16)]
        probes = {k: torch.zeros(16) for k in trainer.TERMS}
        probes["volume"] = torch.tensor([1.] * 4 + [0.] * 12)
        _, details = trainer.objective(probes, metas, config, weights, risk_active=False)
        self.assertAlmostEqual(details["volume"], .5, places=6)  # authentic carries half, not 78%

    def test_selection_balances_authentic_removal_and_insertion(self):
        authentic = [{"kind": "authentic", "generator": "", "operation": "", "volume_loss": 1.,
                      "tamper_probability": p, "max_crop_probability": p} for p in (.2, .3)]
        manipulated = [{"kind": "manipulated", "generator": "cycle", "operation": op, "volume_loss": loss,
                        "tamper_probability": p, "max_crop_probability": p, "removal_probability": .5, "router_hit": False}
                       for op, loss, p in [("removal", 4., .1)] + [("insertion", 0., .9)] * 4]
        metrics = trainer.volume_metrics(authentic + manipulated, [.5])
        self.assertAlmostEqual(metrics["balanced_volume_bce"], (1. + .8) / 2)
        self.assertAlmostEqual(metrics["group_balanced_volume_bce"], (1. + 4. + 0.) / 3)
        self.assertEqual(metrics["group_volume_bce"], {"authentic": 1., "insertion": 0., "removal": 4.})

    def test_selection_auroc_weighs_operations_equally_and_a_constant_scores_chance(self):
        def row(kind, operation, p, with_site, site_crop, source=None):
            return {"kind": kind, "generator": "" if kind == "authentic" else "cycle", "operation": operation,
                    "volume_loss": .5, "tamper_probability": p, "max_crop_probability": p, "removal_probability": .5,
                    "router_hit": None if kind == "authentic" else operation == "removal",
                    "tamper_probability_with_site": with_site, "site_crop_probability": site_crop, "site_source": source}
        authentic = [row("authentic", "", .2, .1, .1, "real_site_TM"), row("authentic", "", .4, .2, .2, "real_site_TB")]
        edited = [row("manipulated", "removal", .9, .8, .9)] + [row("manipulated", "insertion", p, .8, .9)
                                                                 for p in (.1, .3, .3, .3)]
        metrics = trainer.volume_metrics(authentic + edited, [.5])
        self.assertAlmostEqual(metrics["removal_auroc"], 1.)
        self.assertAlmostEqual(metrics["insertion_auroc"], 3 / 8)
        self.assertAlmostEqual(metrics["tamper_auroc"], .5)  # pooled: the four insertions outweigh the removal
        self.assertAlmostEqual(metrics["selection_auroc"], (1. + 3 / 8) / 2)  # one removal counts as much as four insertions
        self.assertEqual(metrics["router_recall_by_operation"], {"insertion": 0., "removal": 1.})
        site = metrics["with_site_crop"]
        self.assertEqual((site["scans"], site["tamper_auroc"], site["crop_removal_auroc"]), (7, 1., 1.))
        self.assertEqual(site["authentic_site_sources"], {"real_site_TB": 1, "real_site_TM": 1})
        constant = [dict(r, tamper_probability=.6) for r in authentic + edited]
        self.assertEqual(trainer.volume_metrics(constant, [.5])["selection_auroc"], .5)  # cannot beat the 0.5 start

    def test_site_crop_diagnostic_leaves_router_only_scores_unchanged(self):
        config = small_config()
        store, _, checked, model = self._setup(config)
        d = config["data"]
        plain = M3DDataset(store, "validation", train=False, seed=config["training"]["seed"], grid=d["grid"],
                           crop_mm=d["crop_mm"], generators=d["generators"])
        model.eval()
        with torch.inference_mode():
            for index in range(len(checked)):
                terms, outputs, _, _ = trainer.case_terms(model, checked[index], "cpu", config, train=False)
                base_terms, base, _, _ = trainer.case_terms(model, plain[index], "cpu", config, train=False)
                for key in ("tamper_probability", "removal_probability", "max_crop_probability", "router_hit"):
                    self.assertEqual(outputs[key], base[key], key)
                self.assertEqual(float(terms["volume"]), float(base_terms["volume"]))
                self.assertNotIn("tamper_probability_with_site", base)
                self.assertTrue(0. < outputs["tamper_probability_with_site"] < 1.)
                self.assertTrue(0. < outputs["site_crop_probability"] < 1.)

    def test_fit_resume_matches_uninterrupted_and_reports(self):
        config = small_config()
        root = Path(self.tmp.name)
        summary = trainer.run_training(self.fx.store, root / "cache", config, root / "runA")
        self.assertTrue(summary["ok"] and summary["epochs_completed"] == 2)
        metrics = summary["validation_metrics"]
        self.assertEqual((metrics["authentic"], metrics["manipulated"]), (1, 4))
        for key in ("tamper_auroc", "removal_auroc", "worst_generator_auroc", "router_recall_at_k", "operating_points",
                    "selection_auroc", "router_recall_by_operation"):
            self.assertIn(key, metrics)
        self.assertEqual(summary["selection_rule"], trainer.SELECTION_RULE)
        if summary["best_epoch"] is None:  # nothing beat chance: the last epoch is reported, and says so
            self.assertIsNone(summary["best_selection_auroc"])
            self.assertIn("beat chance", summary["selection_note"])
        else:
            self.assertGreater(summary["best_selection_auroc"], .5)
        self.assertEqual(metrics["with_site_crop"]["scans"], 5)
        self.assertEqual(sum(metrics["with_site_crop"]["authentic_site_sources"].values()), 1)
        predictions = json.loads((root / "runA/selected_validation_predictions.json").read_text())
        self.assertTrue(all("tamper_probability_with_site" in p for p in predictions))
        self.assertIn("With the edit's crop added (diagnostic)", (root / "runA/report.md").read_text(encoding="utf-8"))
        self.assertTrue((root / "runA/persistence_receipt.json").exists())
        estimate = json.loads((root / "runA/compute_estimate.json").read_text())
        self.assertGreater(estimate["dense_over_sparse"], 1.)
        trainer.run_training(self.fx.store, root / "cache", config, root / "runB", stop_after_epoch=1)
        trainer.run_training(self.fx.store, root / "cache", config, root / "runB", resume=True)
        a = torch.load(root / "runA/last.pt", weights_only=True)["model"]
        b = torch.load(root / "runB/last.pt", weights_only=True)["model"]
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
        with self.assertRaises(FileExistsError):
            trainer.run_training(self.fx.store, root / "cache", config, root / "runA")

    def test_config_validation(self):
        config = small_config()
        trainer.validate_config(config)
        bad = small_config(risk_start_epoch=2)
        with self.assertRaises(ValueError):
            trainer.validate_config(bad)
        bad = small_config()
        bad["data"]["generators"] = ["stylegan"]
        with self.assertRaises(ValueError):
            trainer.validate_config(bad)


class DiagnosticsAndNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.fx = Fixture(cls.tmp.name)
        cls.fx.acquire()

    @classmethod
    def tearDownClass(cls):
        import gc
        gc.collect()
        cls.tmp.cleanup()

    def test_diagnostics_use_train_and_validation_only(self):
        from rsvt.experiments.m3d_diagnostics import degrade, resolution_check, shortcut_check
        store = M3DStore(self.fx.store, Path(self.tmp.name) / "cache")
        out = Path(self.tmp.name) / "diagnostics"
        shortcut = shortcut_check(store, GENS, out)
        identity = shortcut["archive_identity"]
        self.assertEqual((identity["eligible"], identity["eligible_changed_voxels_outside_box"]), (len(SITES) * len(GENS), 0))
        self.assertEqual(identity["eligible_center_inside_mask_bbox"], 1.)
        self.assertIn("validation_auroc", shortcut["global_histogram_classifier"])
        result = resolution_check(store, GENS, out, pairs_train=0, pairs_validation=0, epochs=1, crop_shape=(8, 16, 16))
        self.assertEqual(result["pairs"], {"train": 8, "validation": 4})
        self.assertEqual(set(result["by_operation"]), {"removal", "insertion"})
        self.assertTrue((out / "resolution_check.json").exists())
        image = torch.zeros(1, 8, 16, 16)
        image[0, 4, 8, 8] = 1.
        degraded = degrade(image[None], [[4., 4., 4.]])
        self.assertLess(float(degraded.max()), .2)  # a one-voxel trace does not survive coarse voxels

    def test_flaky_http_stream_stores_the_same_boxes_as_a_clean_stream(self):
        data = (self.fx.root / "cycle.tgz").read_bytes()
        server = FakeServer(data, drops=[len(data) // 4, len(data) // 2, len(data) * 3 // 4])
        real_meta = {s["series_key"]: json.loads((self.fx.store / "real" / (s["series_key"] + ".json")).read_text(encoding="utf-8"))
                     for s in self.fx.selection["series"]}
        store = Path(self.tmp.name) / "store_http"
        with mock.patch.object(store_mod.urllib.request, "urlopen", server), mock.patch.object(store_mod.time, "sleep", lambda s: None):
            summary = store_mod.stream_generator("cycle", self.fx.selection["samples"], real_meta, store, self.fx.work,
                                                 source="https://example.invalid/cycle.tgz", half_mm=12.)
        self.assertEqual((summary["status"], summary["reconnects"], summary["archive_bytes_read"]), ("complete", 3, len(data)))
        clean = sorted(p for p in (self.fx.store / store_mod.BOX_FOLDER / "cycle").glob("*.json") if p.name != "stream_summary.json")
        self.assertEqual(len(clean), len(SITES))
        for path in clean:
            a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (path, store / store_mod.BOX_FOLDER / "cycle" / path.name))
            for key in ("box_content_sha256", "eligible", "diff_voxels", "mask_voxels", "mask_bbox"):
                self.assertEqual(a[key], b[key], (path.name, key))

    def test_resolution_pairs_match_appearance_not_location(self):
        from rsvt.experiments.m3d_diagnostics import MATCHED_SITE, _pairs
        store = M3DStore(self.fx.store, Path(self.tmp.name) / "cache_pairs")
        sites = {s["img_id"]: s for s in store.manifest["sites"]}
        _, labels, _, _, records = _pairs(store, "validation", GENS, 0, 13, (8, 16, 16))
        paired = [r for r in records if r["paired"]]
        self.assertEqual(len(paired), 4)
        for record in paired:
            site = sites[record["authentic_site"]]
            self.assertEqual(site["site_label"], MATCHED_SITE[record["operation"]])  # removal vs TB, insertion vs TM
            self.assertNotEqual(record["volume_id"].split(":")[1], record["authentic_site"])  # never its own location
            self.assertTrue(record["same_series"])
        self.assertEqual(labels.tolist(), [1., 0.] * 4)

    def drive_session(self, name):
        from rsvt.training.m3d_notebook import M3DNotebookSession
        root = Path(self.tmp.name)
        (root / name / "outputs/rsvt_training").mkdir(parents=True)
        return M3DNotebookSession(root / name, Path(__file__).resolve().parents[1], configuration=small_config(), epochs=2,
                                  learning_rate=1e-3, patience=0, seed=13, patients_train=2, patients_validation=1,
                                  generators=GENS, metadata_dir=self.fx.meta_dir, lidc_root=self.fx.lidc, enforce_drive=False,
                                  display_enabled=False, store_root=root / (name + "_store"), cache_root=root / (name + "_cache"))

    def test_space_needed_counts_only_what_is_missing(self):
        session = self.drive_session("space")
        need = lambda: session.check_storage(stop=False)["locations"]["Google Drive"]["needed_gb"]
        everything = need()
        for series in session.selection["series"]:
            path = session.store_root / "real" / (series["series_key"] + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        self.assertLess(need(), everything)
        (session.store_root / "boxes" / "cycle").mkdir(parents=True)  # boxes kept from an earlier rule are not progress
        before = need()
        (session.store_root / "boxes/cycle/rem_1.npz").write_bytes(b"x" * 10 ** 6)
        self.assertEqual(need(), before)
        self.assertIs(session.check_storage(stop=False)["locations"]["Google Drive"]["quota_reported"], False)  # same disk here

    def test_lost_drive_stops_at_once_and_never_streams(self):
        session = self.drive_session("drive_lost")
        calls = []

        def gone(*args, **kwargs):
            calls.append(1)
            raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")

        with mock.patch.object(store_mod, "acquire_real_series", gone),                 mock.patch.object(store_mod, "stream_generator", side_effect=AssertionError("streamed without the scans")):
            with self.assertRaisesRegex(RuntimeError, "Google Drive disconnected"):
                session.prepare_data()
        self.assertEqual(len(calls), 1)  # the first lost read, not one per scan
        self.assertFalse((session.store_root / "real_acquisition_failures.json").exists())

    def test_drive_lost_while_streaming_is_named(self):
        session = self.drive_session("drive_lost_stream")
        with mock.patch.object(store_mod, "stream_generator", side_effect=OSError(errno.ENOTCONN, "Transport endpoint is not connected")):
            with self.assertRaisesRegex(RuntimeError, "Google Drive disconnected"):
                session.prepare_data()
        with mock.patch.object(store_mod, "stream_generator", side_effect=OSError("Archive stream failed at byte 5")):
            with self.assertRaisesRegex(OSError, "Archive stream failed"):  # a server problem keeps its own message
                session.prepare_data()

    def test_too_many_failed_scans_stop_before_streaming(self):
        session = self.drive_session("many_failed")
        with mock.patch.object(store_mod, "acquire_real_series", side_effect=ValueError("SeriesNumber differs")),                 mock.patch.object(store_mod, "stream_generator", side_effect=AssertionError("streamed without the scans")):
            with self.assertRaisesRegex(RuntimeError, "3 of 3 LIDC scans failed.*SeriesNumber differs"):
                session.prepare_data()
        recorded = json.loads((session.store_root / "real_acquisition_failures.json").read_text(encoding="utf-8"))
        self.assertEqual(len(recorded), 3)  # genuine scan failures are still listed

    def test_run_all_decides_stages_from_saved_state(self):
        from rsvt.experiments import m3d_diagnostics
        from rsvt.training.m3d_notebook import M3DNotebookSession
        root = Path(self.tmp.name)
        (root / "auto_project/outputs/rsvt_training").mkdir(parents=True)
        repo = Path(__file__).resolve().parents[1]

        def session(**changes):
            settings = dict(configuration=small_config(), epochs=2, learning_rate=1e-3, patience=0, seed=13,
                            patients_train=2, patients_validation=1, generators=GENS, metadata_dir=self.fx.meta_dir,
                            lidc_root=self.fx.lidc, enforce_drive=False, display_enabled=False,
                            store_root=self.fx.store, cache_root=root / "auto_cache")
            settings.update(changes)
            return M3DNotebookSession(root / "auto_project", repo, **settings)

        ready = session()
        self.assertTrue(ready.store_status()["complete"])
        with mock.patch.object(store_mod, "acquire_real_series", side_effect=AssertionError("downloaded again")), \
                mock.patch.object(store_mod, "stream_generator", side_effect=AssertionError("streamed again")):
            self.assertEqual(ready.prepare_data()["schema_version"], "m3dsynth_store_v1")  # verified, not downloaded
        self.assertEqual(session(generators=("cycle",)).store_status()["reason"],
                         "the selection changed (patients, generators or seed)")
        first = ready.run_diagnostics(pairs_train=0, pairs_validation=0, epochs=1)
        with mock.patch.object(m3d_diagnostics, "resolution_check", side_effect=AssertionError("recomputed")):
            again = session().run_diagnostics(pairs_train=0, pairs_validation=0, epochs=1)
        self.assertEqual((again["folder"], again["problems"]), (first["folder"], first["problems"]))
        self.assertTrue(Path(first["folder"]).is_relative_to(self.fx.store))  # saved with the dataset, shared by runs
        cuda = small_config()
        cuda["training"]["device"] = "cuda"
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertIsNone(session(configuration=cuda).train_when_ready())  # CPU runtime: training waits
        self.assertIsNone(ready.train_when_ready({"problems": ["whole-scan statistics separate the classes"]}))
        fresh = session(store_root=root / "auto_store_empty")
        tiny = type("Usage", (), {"total": 10 ** 12, "used": 10 ** 12, "free": 10 ** 8})()
        with mock.patch("shutil.disk_usage", return_value=tiny), \
                mock.patch.object(store_mod, "acquire_real_series", side_effect=AssertionError("downloaded")):
            with self.assertRaisesRegex(RuntimeError, "Not enough space, so nothing was downloaded"):
                fresh.prepare_data()

    def test_recheck_false_skips_only_after_a_full_readback_of_the_same_store(self):
        from rsvt.training.m3d_notebook import M3DNotebookSession
        root = Path(self.tmp.name)
        runs = root / "recheck_project/outputs/rsvt_training"
        runs.mkdir(parents=True)
        repo = Path(__file__).resolve().parents[1]

        def session(recheck):
            return M3DNotebookSession(root / "recheck_project", repo, configuration=small_config(), epochs=2,
                                      learning_rate=1e-3, patience=0, seed=13, patients_train=2, patients_validation=1,
                                      generators=GENS, metadata_dir=self.fx.meta_dir, lidc_root=self.fx.lidc,
                                      enforce_drive=False, display_enabled=False, store_root=self.fx.store,
                                      cache_root=root / "recheck_cache", recheck_dataset=recheck)

        record = lambda s: json.loads((s.root / "dataset_verification.json").read_text(encoding="utf-8"))
        with mock.patch.object(store_mod, "verify_store", wraps=store_mod.verify_store) as readback:
            first = session(False)  # nothing earlier to rely on, so it reads everything back
            first.prepare_data()
            self.assertEqual(readback.call_count, 1)
            full = record(first)
            self.assertEqual((full["full_readback"], full["ok"]), (True, True))
            # Newer runs that must not count: another manifest, failures, another store folder.
            manifest = store_mod.sha256_path(self.fx.store / "manifest.json")
            for name, store, result in (("c2f_99990103T000000_000000Z", self.fx.store, {"manifest_sha256": "0" * 64}),
                                        ("c2f_99990102T000000_000000Z", self.fx.store, {"ok": False, "failures": ["x"]}),
                                        ("c2f_99990101T000000_000000Z", root / "elsewhere", {})):
                (runs / name).mkdir()
                store_mod.write_json(runs / name / "notebook_session.json", {"store": str(store)})
                store_mod.write_json(runs / name / "dataset_verification.json",
                           {"manifest_sha256": manifest, "ok": True, "failures": [], "files_checked": 5, **result})
            skipped = session(False)
            self.assertEqual(skipped.prepare_data()["schema_version"], "m3dsynth_store_v1")
            self.assertEqual(readback.call_count, 1)  # not read back again
            reused = record(skipped)
            self.assertEqual((reused["full_readback"], reused["files_checked"]), (False, 0))
            self.assertEqual(reused["reused_full_readback"]["run"], first.root.name)
            self.assertEqual(reused["reused_full_readback"]["files_checked"], full["files_checked"])
            again = session(False)  # a skip never vouches for a later skip: it points at the full read-back
            again.prepare_data()
            self.assertEqual(record(again)["reused_full_readback"]["run"], first.root.name)
            session(True).prepare_data()  # RECHECK_DATASET = True always reads everything back
            self.assertEqual(readback.call_count, 2)
        skipped.verify_saved()
        self.assertIn("full read-back skipped", (skipped.root / "run_overview.md").read_text(encoding="utf-8"))
        first.verify_saved()
        self.assertIn("read back against its checksum in this run", (first.root / "run_overview.md").read_text(encoding="utf-8"))

    def test_notebook_session_end_to_end_in_fixture_mode(self):
        from rsvt.training.m3d_notebook import M3DNotebookSession
        root = Path(self.tmp.name)
        (root / "project/outputs/rsvt_training").mkdir(parents=True)
        common = dict(configuration=small_config(), epochs=2, learning_rate=1e-3, patience=0, seed=13,
                      patients_train=2, patients_validation=1, generators=GENS, metadata_dir=self.fx.meta_dir,
                      lidc_root=self.fx.lidc, enforce_drive=False, display_enabled=False)
        repo = Path(__file__).resolve().parents[1]
        fresh = M3DNotebookSession(root / "project", repo, store_root=root / "fresh_store", cache_root=root / "fresh_local",
                                   **common)
        self.assertFalse(fresh.has_store())
        self.assertIsNone(fresh.prepare_data(download=False))
        manifest = fresh.prepare_data(download=True, archive_sources={g: self.fx.root / f"{g}.tgz" for g in GENS})
        self.assertEqual(manifest["counts"]["ineligible"], 0)
        session = M3DNotebookSession(root / "project", repo, store_root=self.fx.store, cache_root=root / "cache2", **common)
        session.show_runtime()
        self.assertEqual(session.plan_data()["real_series"], 3)
        self.assertTrue(session.has_store())
        session.prepare_data(download=False)
        self.assertEqual(len(session.show_dataset()), 3)
        self.assertEqual(len(session.show_previews()), 2)
        diagnostics = session.run_diagnostics(pairs_train=0, pairs_validation=0, epochs=1)
        self.assertIn("resolution", diagnostics)
        session.show_configuration()
        self.assertGreater(session.show_model(), 0)
        summary = session.train()
        self.assertTrue(summary["training_finished"])
        self.assertEqual(len(session.show_history()), 2)
        self.assertIsNotNone(session.show_results())
        records = session.verify_saved()
        self.assertIn("training/summary.json", records)
        self.assertIn("figures/learning_curves.png", records)
        self.assertTrue(session.train()["training_finished"])  # re-run reuses the finished session


class CropCheck(unittest.TestCase):
    """The crop learning check: paired crops at known edits, three arms, train/validation only."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.fx = Fixture(cls.tmp.name)
        cls.fx.acquire()
        cls.store = M3DStore(cls.fx.store, Path(cls.tmp.name) / "cache")

    @classmethod
    def tearDownClass(cls):
        import gc
        gc.collect()
        cls.tmp.cleanup()

    def tiny(self, **changes):
        settings = dict(device="cpu", per_operation_train=0, per_operation_validation=0, fit_per_operation=2, steps=4,
                        fit_steps=4, batch_pairs=2, shape=(8, 16, 16), margin=(2, 4, 4), eval_every=2)
        settings.update(changes)
        return settings

    def test_pairs_are_balanced_spread_over_generators_and_prefix_stable(self):
        from rsvt.experiments.m3d_crop_check import select_pairs
        everything = select_pairs(self.store, "train", GENS, 0, 13)
        self.assertEqual(Counter(v["operation"] for v in everything), {"removal": 4, "insertion": 4})
        self.assertEqual({v["split"] for v in everything}, {"train"})
        two = select_pairs(self.store, "train", GENS, 2, 13)
        by_operation = lambda volumes, op: [v["volume_id"] for v in volumes if v["operation"] == op]
        for operation in ("removal", "insertion"):
            self.assertEqual(by_operation(two, operation), by_operation(everything, operation)[:2])  # prefix
            self.assertEqual({v["generator"] for v in two if v["operation"] == operation}, set(GENS))  # round-robin

    def test_pairs_differ_only_in_the_changed_voxels(self):
        from rsvt.experiments.m3d_crop_check import extract_pairs, select_pairs
        volumes = select_pairs(self.store, "validation", GENS, 0, 13)
        shape, margin = (8, 16, 16), (2, 4, 4)
        data = extract_pairs(self.store, volumes, shape, margin)
        self.assertEqual(tuple(data["fake"].shape), (4, 1, 12, 24, 24))
        core = tuple(slice(m, m + s) for s, m in zip(shape, margin))
        for volume, record in zip(volumes, data["records"]):
            image, _, edit = self.store.crop(volume, np.asarray(volume["center_zyx"], float), (12, 24, 24))
            reference, _, _ = self.store.crop({"kind": "authentic", "series_key": volume["series_key"]},
                                              np.asarray(volume["center_zyx"], float), (12, 24, 24))
            self.assertEqual(record["changed_voxels_in_crop"], int(edit[core].sum()))
            self.assertGreater(record["changed_voxels_in_crop"], 0)
            self.assertAlmostEqual(record["edit_visible_after_window"],
                                   ((image != reference) & edit)[core].sum() / edit[core].sum())
        original = self.store.crop

        def shifted(volume, center, shape):  # a real crop that is not the same location
            image, inside, edit = original(volume, center, shape)
            return (image + .01 if volume["kind"] == "authentic" else image), inside, edit
        with mock.patch.object(self.store, "crop", shifted):
            with self.assertRaisesRegex(ValueError, "differs outside its changed voxels"):
                extract_pairs(self.store, volumes[:1], shape, margin)

    def test_metrics_verdicts_and_conclusions(self):
        from rsvt.experiments.m3d_crop_check import interpret, pair_metrics, verdict
        records = [{"operation": op, "generator": "cycle"} for op in ("removal", "removal", "insertion", "insertion")]
        fake, real = np.array([2., 0., 1., 1.]), np.array([1., 1., 0., 1.])
        metrics = pair_metrics(fake, real, records)
        self.assertEqual(metrics["removal"]["paired_accuracy"], .5)   # one above its real crop, one below
        self.assertEqual(metrics["insertion"]["paired_accuracy"], .75)  # one above, one tie
        self.assertAlmostEqual(metrics["all"]["bce"], (np.logaddexp(0, -fake).sum() + np.logaddexp(0, real).sum()) / 8)
        self.assertEqual(metrics["cycle:removal"]["pairs"], 2)
        case = lambda overall, removal, insertion: {"all": {"paired_accuracy": overall}, "removal": {"paired_accuracy": removal},
                                                    "insertion": {"paired_accuracy": insertion}}
        self.assertEqual([verdict(case(.9, .85, .95)), verdict(case(.5, .1, .9)), verdict(case(.65, .6, .7)),
                          verdict(case(.5, .5, .5)), verdict({"all": {"paired_accuracy": .5}, "removal": None})],
                         ["learned", "nodule_presence", "partly", "not_learned", "incomplete"])
        fits, fails = {"bce": .01, "paired_accuracy": 1.}, {"bce": .6, "paired_accuracy": .7}
        self.assertIn("could not fit", interpret(fails, {"image": "learned", "residual": "learned"}))
        self.assertIn("lies upstream", interpret(fits, {"image": "learned", "residual": "not_learned"}))
        self.assertIn("residual input", interpret(fits, {"image": "not_learned", "residual": "learned"}))
        text = interpret(fits, {"image": "nodule_presence", "residual": "partly"})
        self.assertIn("partly on residual", text)
        self.assertIn("On image it scores one operation well and the other badly", text)

    def test_residual_removes_flat_anatomy_and_keeps_a_one_voxel_trace(self):
        from rsvt.experiments.m3d_crop_check import residual
        flat = torch.full((1, 1, 6, 6, 6), .4)
        self.assertLess(float(residual(flat).abs().max()), 1e-6)
        spike = flat.clone()
        spike[0, 0, 3, 3, 3] = 1.
        self.assertAlmostEqual(float(residual(spike)[0, 0, 3, 3, 3]), .6 * 26 / 27, places=5)

    def test_crop_check_runs_every_arm_on_train_and_validation_only(self):
        from rsvt.experiments.m3d_crop_check import crop_check
        out = Path(self.tmp.name) / "crop_check"
        events = []
        result = crop_check(self.store, GENS, out, model_kwargs={"dimension": 8, "widths": (4, 8, 8)},
                            progress=events.append, **self.tiny())
        saved = json.loads((out / "crop_check.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["schema_version"], saved["interpretation"]), (result["schema_version"], result["interpretation"]))
        self.assertEqual((sum(result["pairs"]["train"].values()), sum(result["pairs"]["validation"].values()),
                          result["pairs"]["fit"]), (8, 4, 4))
        self.assertEqual((result["pairs"]["train_patients"], result["pairs"]["validation_patients"]), (2, 1))
        validation_ids = {v["volume_id"] for v in self.store.manifest["volumes"] if v["split"] == "validation"}
        self.assertTrue({r["volume_id"] for r in result["records"]["validation"]} <= validation_ids)
        self.assertEqual(set(result["arms"]), {"fit", "image", "residual"})
        for name, arm in result["arms"].items():
            # the fit arm reports six times over its steps; the others every `eval_every` steps
            self.assertEqual([p["step"] for p in arm["curve"]], [0, 1, 2, 3, 4] if name == "fit" else [0, 2, 4])
            self.assertEqual(set(arm["final"]) - {"step", "train_loss", "learning_rate"},
                             {"train"} if name == "fit" else {"validation", "train_sample"})
        self.assertEqual(set(result["verdicts"]), {"image", "residual"})
        self.assertEqual(result["test_volumes_opened"], 0)
        self.assertEqual({e.get("arm") for e in events if e["stage"] == "crop_check"}, {"fit", "image", "residual"})

    def test_notebook_runs_the_check_instead_of_training(self):
        from rsvt.training.m3d_notebook import M3DNotebookSession
        root = Path(self.tmp.name)
        (root / "check_project/outputs/rsvt_training").mkdir(parents=True)
        common = dict(configuration=small_config(), epochs=2, learning_rate=1e-3, patience=0, seed=13,
                      patients_train=2, patients_validation=1, generators=GENS, metadata_dir=self.fx.meta_dir,
                      lidc_root=self.fx.lidc, enforce_drive=False, display_enabled=False,
                      store_root=self.fx.store, cache_root=root / "check_cache")
        with self.assertRaisesRegex(ValueError, "RUN must be one of"):
            M3DNotebookSession(root / "check_project", REPO, run="both", **common)
        training = M3DNotebookSession(root / "check_project", REPO, **common)
        self.assertIsNone(training.run_crop_check())  # RUN = "train": skipped
        session = M3DNotebookSession(root / "check_project", REPO, run="crop_check", **common)
        with mock.patch.object(trainer, "run_training", side_effect=AssertionError("trained")):
            self.assertIsNone(session.train_when_ready())
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertIsNone(session.run_crop_check())  # waits for a GPU
        settings = self.tiny()
        result = session.run_crop_check(**settings)
        self.assertEqual(result["settings"]["model"], {"dimension": 16, "widths": [4, 8, 8]})  # the model's fine branch
        records = session.verify_saved()
        self.assertIn("crop_check/crop_check.json", records)
        self.assertIn("figures/crop_check.png", records)
        self.assertIn("Crop learning check: " + result["interpretation"],
                      (session.root / "run_overview.md").read_text(encoding="utf-8"))
        self.assertEqual(json.loads((session.root / "notebook_session.json").read_text(encoding="utf-8"))["run"], "crop_check")


class PretrainCheck(unittest.TestCase):
    """Crop pretraining on the model's own crops, the transfer check and the crop-swap test (27 Nov 2026)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.fx = Fixture(cls.tmp.name)
        cls.fx.acquire()
        cls.store = M3DStore(cls.fx.store, Path(cls.tmp.name) / "cache")

    @classmethod
    def tearDownClass(cls):
        import gc
        gc.collect()
        cls.tmp.cleanup()

    def tiny(self, **changes):
        settings = dict(device="cpu", per_operation_train=0, per_operation_validation=0, steps=2, batch_pairs=2,
                        eval_every=1, fusion_steps=2, resamples=50)
        settings.update(changes)
        return settings

    def pairs(self, split="validation"):
        from rsvt.experiments.m3d_crop_check import select_pairs
        from rsvt.experiments.m3d_pretrain_check import site_pairs
        return site_pairs(self.store, select_pairs(self.store, split, GENS, 0, 13), small_config()["data"]["crop_mm"])

    def test_site_pairs_are_the_models_site_crop_and_its_unedited_twin(self):
        from rsvt.experiments.m3d_pretrain_check import _core, unedited
        data = self.pairs()
        self.assertEqual(len(data["items"]), 4)
        for volume, item in zip(data["volumes"], data["items"]):
            self.assertTrue(all(m % 2 == 0 for m in item["margin"]))  # even: the big crop rounds like the model's
            case = VolumeCase(self.store, volume, flip=False, grid=12, crop_mm=16., forced=0, random_centers=0,
                              rng=_rng(0), jitter_fraction=0.)
            self.assertEqual(item["shape"], case.crop_shape)
            edited = case.crops([case.site_center])
            original = case.crops([case.site_center], source=unedited(volume))
            torch.testing.assert_close(edited["image"][0], _core(item, "fake"), rtol=0, atol=0)
            torch.testing.assert_close(original["image"][0], _core(item, "real"), rtol=0, atol=0)
            self.assertEqual(float(original["tamper"][0]), 0.)
            differs = edited["image"] != original["image"]
            self.assertTrue(bool(differs.any()))
            self.assertFalse(bool((differs & ~edited["mask"].bool()).any()))  # only the changed voxels differ
        with self.assertRaisesRegex(ValueError, "same series"):
            case.crops([case.site_center], source={"kind": "authentic", "series_key": "other"})

    def test_transfer_check_matches_and_catches_a_wrong_or_doubled_residual(self):
        from rsvt.experiments import m3d_pretrain_check as check
        from rsvt.models import coarse_to_fine as model_mod
        config = small_config()
        train, validation = self.pairs("train"), self.pairs()
        kwargs = check.fine_kwargs(config["model"])
        self.assertEqual(kwargs["input"], "residual")  # the model default reaches pretraining
        branch, curve, reference = check.pretrain(train, validation, steps=2, batch_pairs=2, lr=1e-3, seed=13,
                                                  device="cpu", model_kwargs=kwargs, eval_every=1)
        self.assertEqual([p["step"] for p in curve], [0, 1, 2])
        state = {k: v.detach().clone() for k, v in branch.state_dict().items()}
        common = dict(grid=12, crop_mm=16., tolerance=1e-4)
        result = check.transfer_check(state, config["model"], self.store, validation, reference, "cpu", **common)
        self.assertTrue(result["match"])
        self.assertEqual(result["model_data_path"]["max_image_difference"], 0.)
        image_input = dict(config["model"], fine_input="image")  # residual missing in the full model
        self.assertFalse(check.transfer_check(state, image_input, self.store, validation, reference, "cpu",
                                              **common)["loaded_weights"]["match"])
        once = model_mod.local_residual
        with mock.patch.object(model_mod, "local_residual", lambda x: once(once(x))):  # residual applied twice
            self.assertFalse(check.transfer_check(state, config["model"], self.store, validation, reference, "cpu",
                                                  **common)["match"])

    def test_swap_test_changes_only_the_slot_and_the_frozen_crops_stay_put(self):
        from rsvt.experiments import m3d_pretrain_check as check
        config = small_config()
        validation = self.pairs()
        torch.manual_seed(3)
        model = CoarseToFineRSVT(**config["model"])
        rows = check.swap_test(model, self.store, validation["volumes"], "cpu", grid=12, crop_mm=16.)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["other_crops"] == config["model"]["crops_at_inference"] - 1 for r in rows))
        self.assertTrue(any(r["edited"]["slot_crop_logit"] != r["unedited"]["slot_crop_logit"] for r in rows))
        original = VolumeCase.crops

        def same(case, centers, source=None):  # the "unedited" slot gets the edited crop: nothing may change
            return original(case, centers)
        with mock.patch.object(VolumeCase, "crops", same):
            for row in check.swap_test(model, self.store, validation["volumes"], "cpu", grid=12, crop_mm=16.):
                self.assertEqual(row["edited"], row["unedited"])
        frozen = {k: v.clone() for k, v in model.fine.state_dict().items()}
        for parameter in model.fine.parameters():
            parameter.requires_grad_(False)
        curve = check.train_fusion(model, self.store, config, steps=2, device="cpu")
        self.assertEqual([c["step"] for c in curve], [1, 2])
        for key, value in model.fine.state_dict().items():
            torch.testing.assert_close(value, frozen[key], rtol=0, atol=0, msg=key)

    def test_patient_intervals_metrics_and_conclusions(self):
        from rsvt.experiments.m3d_pretrain_check import cluster_interval, interpret, swap_metrics
        self.assertEqual(cluster_interval([1., 0., 1.], ["a", "a", "a"], 20), [2 / 3, 2 / 3])  # one patient: no spread
        low, high = cluster_interval([1., 1., 0., 0.], ["a", "a", "b", "b"], 400)
        self.assertTrue(0. <= low < .5 < high <= 1.)

        def row(op, patient, scan, crop):
            return {"operation": op, "patient_id": patient, "other_crops_holding_edit": 0,
                    "edited": {"scan_logit": scan, "max_crop_logit": crop, "slot_crop_logit": crop},
                    "unedited": {"scan_logit": 0., "max_crop_logit": 0., "slot_crop_logit": 0.}}
        rows = ([row("removal", f"p{i}", 1., 2.) for i in range(6)]
                + [row("insertion", f"p{i}", -1. if i % 2 else 0., 2.) for i in range(6)])
        metrics = swap_metrics(rows, resamples=200)
        self.assertEqual(metrics["removal"]["scan_logit"]["paired_accuracy"], 1.)
        self.assertEqual(metrics["insertion"]["scan_logit"]["paired_accuracy"], .25)  # three ties, three drops
        self.assertEqual((metrics["all"]["scans"], metrics["all"]["patients"]), (12, 6))
        ok = {"match": True}
        self.assertIn("implementation error", interpret({"match": False}, "learned", None))
        self.assertIn("did not reproduce", interpret(ok, "not_learned", None))
        text = interpret(ok, "learned", {"after_training": metrics})
        self.assertIn("for insertion", text)
        self.assertIn("not using evidence the crops carry", text)  # the largest crop score did respond
        everything = swap_metrics([row("removal", f"p{i}", 1., 2.) for i in range(6)]
                                  + [row("insertion", f"p{i}", 1., 2.) for i in range(6)], resamples=200)
        self.assertIn("Proceed with run 3", interpret(ok, "learned", {"after_training": everything}))

    def test_pretrain_check_saves_weights_and_uses_train_and_validation_only(self):
        from rsvt.experiments import m3d_pretrain_check as check
        out = Path(self.tmp.name) / "pretrain_check"
        events = []
        with mock.patch.object(check, "verdict", return_value="learned"):  # two steps learn nothing; run check 2 anyway
            result = check.pretrain_check(self.store, small_config(), out, progress=events.append, **self.tiny())
        saved = json.loads((out / "pretrain_check.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["interpretation"], result["interpretation"])
        self.assertTrue(result["transfer"]["match"])
        self.assertEqual((sum(result["pairs"]["train"].values()), sum(result["pairs"]["validation"].values())), (8, 4))
        validation_ids = {v["volume_id"] for v in self.store.manifest["volumes"] if v["split"] == "validation"}
        self.assertTrue({r["volume_id"] for r in result["validation_pairs"]} <= validation_ids)
        swap = result["swap_test"]
        self.assertTrue(swap["crop_network_unchanged"])
        self.assertEqual(swap["after_training"]["all"]["scans"], 4)
        self.assertEqual(len(swap["fusion_curve"]), 2)
        receipt = json.loads((out / "fine_branch.pt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["sha256"], hashlib.sha256((out / "fine_branch.pt").read_bytes()).hexdigest())
        weights = torch.load(out / "fine_branch.pt", weights_only=True)
        model = CoarseToFineRSVT(**small_config()["model"])
        model.fine.load_state_dict(weights["fine"])  # loads into the full model's crop network as saved
        self.assertEqual(weights["format_version"], check.WEIGHTS_FORMAT)
        self.assertEqual(result["test_volumes_opened"], 0)
        stages = {e["stage"] for e in events}
        self.assertTrue({"pretrain", "transfer", "fusion", "swap_test"} <= stages)
        skipped = check.pretrain_check(self.store, small_config(), Path(self.tmp.name) / "skipped", **self.tiny())
        if skipped["pretraining"]["verdict"] not in ("learned", "partly"):
            self.assertIn("did not learn", skipped["swap_test"]["skipped"])

    def test_notebook_runs_the_pretrain_check_instead_of_training(self):
        from rsvt.training.m3d_notebook import M3DNotebookSession
        root = Path(self.tmp.name)
        (root / "pre_project/outputs/rsvt_training").mkdir(parents=True)
        common = dict(configuration=small_config(), epochs=2, learning_rate=1e-3, patience=0, seed=13,
                      patients_train=2, patients_validation=1, generators=GENS, metadata_dir=self.fx.meta_dir,
                      lidc_root=self.fx.lidc, enforce_drive=False, display_enabled=False,
                      store_root=self.fx.store, cache_root=root / "pre_cache")
        training = M3DNotebookSession(root / "pre_project", REPO, **common)
        self.assertIsNone(training.run_check())  # RUN = "train": no check
        self.assertIsNone(training.run_pretrain_check())
        session = M3DNotebookSession(root / "pre_project", REPO, run="pretrain_check", **common)
        with mock.patch.object(trainer, "run_training", side_effect=AssertionError("trained")):
            self.assertIsNone(session.train_when_ready())
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertIsNone(session.run_check())  # waits for a GPU
        from rsvt.experiments import m3d_pretrain_check as check
        with mock.patch.object(check, "verdict", return_value="learned"):
            result = session.run_check(**self.tiny())
        records = session.verify_saved()
        for name in ("pretrain_check/pretrain_check.json", "pretrain_check/fine_branch.pt", "figures/pretrain_check.png"):
            self.assertIn(name, records)
        self.assertIn("Crop pretraining check: " + result["interpretation"],
                      (session.root / "run_overview.md").read_text(encoding="utf-8"))


class BoxRuleAndLabelTail(unittest.TestCase):
    """Deeper boxes, inspected label tails and re-streaming under a new rule (25 Nov 2026)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        cls.fx = Fixture(cls.tmp.name)
        cls.fx.acquire()
        cls.real_meta = {s["series_key"]: json.loads((cls.fx.store / "real" / (s["series_key"] + ".json")).read_text(encoding="utf-8"))
                         for s in cls.fx.selection["series"]}

    @classmethod
    def tearDownClass(cls):
        import gc
        gc.collect()
        cls.tmp.cleanup()

    def stream(self, name, *, tails, labels_first=True, edit=None, progress=None):
        """Re-stream 'cycle'; `tails` maps img_id -> mask voxels in one label slice past the scan end."""
        members = []
        for img_id, patient, center in SITES:
            scan, mask = self.fx.edits[("cycle", img_id)]
            if edit and img_id in edit:
                scan = edit[img_id](scan.copy())
            label = [(f"cycle/label/{img_id}/slide{z:04d}.tiff", tiff(mask[z])) for z in range(SHAPE[0])]
            scans = [(f"cycle/scan/{img_id}/slide{z:04d}.tiff", tiff(scan[z])) for z in range(SHAPE[0])]
            extra = []
            if img_id in tails:
                tail = np.zeros(SHAPE[1:], dtype=mask.dtype)
                tail.flat[:tails[img_id]] = 1
                extra = [(f"cycle/label/{img_id}/slide{SHAPE[0]:04d}.tiff", tiff(tail))]
            members += extra + label + scans if labels_first else scans + label + extra  # second: arrives after completion
        archive = Path(self.tmp.name) / f"{name}.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            for member, data in members:
                info = tarfile.TarInfo(member)
                info.size = len(data)
                tar.addfile(info, BytesIO(data))
        store = Path(self.tmp.name) / name
        summary = store_mod.stream_generator("cycle", self.fx.selection["samples"], self.real_meta, store, self.fx.work,
                                             source=archive, half_mm=12., progress=progress, progress_every=1)
        folder = store / store_mod.BOX_FOLDER / "cycle"
        return summary, {i: json.loads((folder / f"{i}.json").read_text(encoding="utf-8")) for i, _, _ in SITES}

    def test_box_covers_a_voxel_cube_on_thick_slices_only_when_needed(self):
        shape, center = (133, 512, 512), (60, 250, 250)
        for spacing, depth in (((2.5, .78, .78), 31), ((3., .55, .55), 44), ((1., .78, .78), 48)):
            start, stop = store_mod.box_for(center, shape, spacing, 48., 24.)
            self.assertEqual((center[0] - start[0], stop[0] - 1 - center[0]), (depth, depth))
            old = store_mod.box_for(center, shape, spacing, 48.)
            self.assertEqual((start[1:], stop[1:]), (old[0][1:], old[1][1:]))  # in-plane extent unchanged
        self.assertEqual(store_mod.box_for((3, 250, 250), shape, (2.5, .78, .78), 48., 24.)[0][0], 0)  # still clipped

    def test_an_empty_label_tail_is_accepted_and_recorded_a_filled_one_is_not(self):
        for order in (True, False):
            with self.subTest(labels_first=order):
                summary, receipts = self.stream(f"tail_{order}", tails={"rem_1": 0, "inj_2": 5}, labels_first=order)
                self.assertTrue(receipts["rem_1"]["eligible"])
                self.assertEqual((receipts["rem_1"]["label_tail_slices"], receipts["rem_1"]["label_tail_voxels"]), (1, 0))
                self.assertFalse(receipts["inj_2"]["eligible"])
                self.assertEqual(receipts["inj_2"]["problems"], ["5 mask voxels in label slices past the scan end"])
                self.assertEqual((receipts["rem_3"]["eligible"], receipts["rem_3"]["label_tail_slices"]), (True, 0))
                self.assertEqual(summary["label_tails"], {"empty": 1, "nonempty": 1})
                self.assertEqual((summary["finished"]["eligible"], summary["finished"]["ineligible"]), (len(SITES) - 1, 1))
                for img_id, record in receipts.items():  # a late slice must not replace the stored sample
                    self.assertGreater(record["diff_voxels"], 0, img_id)

    def test_label_tail_counts_are_visible_while_streaming(self):
        events = []
        self.stream("live", tails={"rem_1": 0, "inj_2": 5}, progress=events.append)
        self.assertEqual(events[-1]["label_tails"], {"empty": 1, "nonempty": 1})
        self.assertTrue(any(e["remaining"] and sum(e["label_tails"].values()) for e in events))  # before the end

    def test_other_checks_stay_strict(self):
        def far(scan):
            scan[0, 0, 0] += 7
            return scan
        _, receipts = self.stream("strict", tails={"rem_1": 0}, edit={"rem_1": far})
        self.assertFalse(receipts["rem_1"]["eligible"])  # an empty tail does not excuse a change outside the box
        self.assertIn("changed voxels outside the stored box", " ".join(receipts["rem_1"]["problems"]))

    def test_old_boxes_are_kept_and_an_earlier_rule_is_streamed_again(self):
        store = Path(self.tmp.name) / "upgrade"
        old = store / "boxes" / "cycle"
        old.mkdir(parents=True)
        (old / "rem_1.json").write_text('{"eligible": false}', encoding="utf-8")
        (old / "rem_1.npz").write_bytes(b"old box")
        summary, receipts = self.stream("upgrade", tails={})
        self.assertEqual(sum(summary["finished"].values()), len(SITES))
        self.assertEqual(((old / "rem_1.json").read_text(encoding="utf-8"), (old / "rem_1.npz").read_bytes()),
                         ('{"eligible": false}', b"old box"))
        self.assertEqual(receipts["rem_1"]["box_rule"], store_mod.box_rule(12., 0.))
        common = dict(source=Path(self.tmp.name) / "upgrade.tgz", half_mm=12.)
        args = ("cycle", self.fx.selection["samples"], self.real_meta, store, self.fx.work)
        self.assertEqual(store_mod.stream_generator(*args, **common)["status"], "already_complete")  # same rule: skipped
        again = store_mod.stream_generator(*args, **common, cube_half_mm=6.)
        self.assertEqual(sum(again["finished"].values()), len(SITES))  # new rule: streamed again

    def test_earlier_manifest_is_preserved_and_reported_as_an_earlier_rule(self):
        from rsvt.training.m3d_notebook import M3DNotebookSession
        root = Path(self.tmp.name)
        (root / "rule_project/outputs/rsvt_training").mkdir(parents=True)
        config = small_config()
        config["acquisition"]["cube_half_mm"] = 6.
        settings = dict(epochs=2, learning_rate=1e-3, patience=0, seed=13, patients_train=2, patients_validation=1,
                        generators=GENS, metadata_dir=self.fx.meta_dir, lidc_root=self.fx.lidc, enforce_drive=False,
                        display_enabled=False, store_root=self.fx.store, cache_root=root / "rule_cache")
        repo = Path(__file__).resolve().parents[1]
        same = M3DNotebookSession(root / "rule_project", repo, configuration=small_config(), **settings)
        self.assertTrue(same.store_status()["complete"])
        other = M3DNotebookSession(root / "rule_project", repo, configuration=config, **settings)
        self.assertFalse(other.store_status()["complete"])
        self.assertIn("earlier rule", other.store_status()["reason"])
        before = store_mod.sha256_path(self.fx.store / "manifest.json")
        store_mod.build_manifest(self.fx.store, self.fx.selection, self.fx.meta)
        kept = self.fx.store / "manifest_history" / f"manifest_{before[:12]}.json"
        self.assertEqual(store_mod.sha256_path(kept), before)


class LegacyRouterFixes(unittest.TestCase):
    def test_initial_gates_start_at_budget_and_hard_gate_bias_runs(self):
        from rsvt.models.proposed import RSVT
        torch.manual_seed(0)
        kwargs = dict(dimension=16, heads=2, layers=1, coarse_width=4, candidate_width=4, sequence_cap=40,
                      max_tokens=2048, token_pooling="max_mean", router_initial_keep_fraction=.05, hard_gate_bias=True)
        model = RSVT(**kwargs)
        image = torch.rand(1, 1, 32, 32, 32)
        valid = torch.ones_like(image, dtype=torch.bool)
        model.train()
        out = model(image, valid, torch.eye(4, dtype=torch.float64)[None])
        candidates = int(out["routing"]["candidate_mask"].sum())
        coarse = int(out["routing"]["coarse_mask"].sum())
        self.assertAlmostEqual(float(out["routing"]["expected_sequence_counts"][0].detach()),
                               1 + coarse + .05 * candidates, places=3)
        model.eval()
        hard = model(image, valid, torch.eye(4, dtype=torch.float64)[None])
        self.assertTrue(torch.isfinite(hard["logits"]).all())
        legacy = RSVT(**{**kwargs, "hard_gate_bias": False})
        legacy.load_state_dict(model.state_dict())
        legacy.eval()
        plain = legacy(image, valid, torch.eye(4, dtype=torch.float64)[None])
        self.assertFalse(torch.allclose(plain["logits"], hard["logits"]))

    def test_max_mean_pooling_keeps_a_one_voxel_spike(self):
        from rsvt.models.geometry import masked_max_pool_tokens, masked_pool_tokens
        features = torch.zeros(1, 1, 8, 8, 8)
        features[0, 0, 3, 3, 3] = 1.
        support = torch.ones_like(features, dtype=torch.bool)
        self.assertAlmostEqual(float(masked_pool_tokens(features, support, 8, (1, 1, 1))[0, 0]), 1 / 512)
        self.assertEqual(float(masked_max_pool_tokens(features, support, 8, (1, 1, 1))[0, 0]), 1.)


if __name__ == "__main__":
    unittest.main()
