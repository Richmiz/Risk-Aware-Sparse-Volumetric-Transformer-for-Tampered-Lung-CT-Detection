"""F11: M3Dsynth store, coarse-to-fine RSVT, endpoint losses and trainer.

Procedural CT series and TIFF archives only; no medical images are read.
"""

import csv
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
                                        intensity_lut)
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
        self.assertEqual(int(mask.sum()), 3 * 7 * 7)
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

    def test_validation_cases_are_unaugmented(self):
        dataset = M3DDataset(self.store, "validation", train=False)
        case = dataset[0]
        self.assertFalse(case.flip)
        self.assertEqual((case.forced, case.random), ([], []))
        self.assertTrue(all(v["split"] == "validation" for v in dataset.rows))


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

    def test_fit_resume_matches_uninterrupted_and_reports(self):
        config = small_config()
        root = Path(self.tmp.name)
        summary = trainer.run_training(self.fx.store, root / "cache", config, root / "runA")
        self.assertTrue(summary["ok"] and summary["epochs_completed"] == 2)
        metrics = summary["validation_metrics"]
        self.assertEqual((metrics["authentic"], metrics["manipulated"]), (1, 4))
        for key in ("tamper_auroc", "removal_auroc", "worst_generator_auroc", "router_recall_at_k", "operating_points"):
            self.assertIn(key, metrics)
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
