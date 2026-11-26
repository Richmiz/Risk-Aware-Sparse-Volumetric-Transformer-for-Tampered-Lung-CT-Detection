"""M3Dsynth volumes for coarse-to-fine RSVT: 96^3 coarse views and native crops.

* Coarse view: exact area-average resampling (anti-aliased) of the whole scan to
  96^3, after a fixed HU window. Area averaging is linear, so a manipulated
  volume's view equals the real view plus the resampled box difference; per-sample
  views are never stored.
* Native crops: the router's cells and the training sites are cut from the 1 mm-
  class native voxel grid (no resampling), so generator traces are kept.
* Flips: left-right only, deterministic per (seed, epoch, volume).
* Labels: where a volume was edited comes from its changed voxels (manipulated !=
  real), not from the author mask. On M3Dsynth removals the authors' mask is a
  ~33 mm cube while the changed voxels span a cube with as many slices as pixels,
  3-5x deeper on 2-3 mm slices (store of 26 Nov 2026).
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rsvt.data.m3dsynth_store import materialize_local, sha256_path, write_json

WINDOW_HU = (-1000., 400.)
COARSE_SHAPE = (96, 96, 96)


def intensity_lut(u16_to_hu, window=WINDOW_HU):
    values = np.arange(65536, dtype=np.float64)
    hu = (values - u16_to_hu["offset"]) * u16_to_hu["slope"] + u16_to_hu["intercept"]
    return np.clip((hu - window[0]) / (window[1] - window[0]), 0., 1.).astype(np.float32)


def area_matrix(n_in, n_out):
    """(n_out, n_in) weights: exact overlap of unit input cells with each output bin."""
    edges = np.arange(n_out + 1, dtype=np.float64) * n_in / n_out
    cells = np.arange(n_in, dtype=np.float64)
    lo = np.maximum(edges[:-1, None], cells[None])
    hi = np.minimum(edges[1:, None], cells[None] + 1.)
    return (np.clip(hi - lo, 0., None) / (edges[1:] - edges[:-1])[:, None]).astype(np.float32)


class AreaResampler:
    def __init__(self, shape=COARSE_SHAPE):
        self.shape = tuple(shape)
        self._cache = {}

    def matrices(self, native_shape):
        out = []
        for n, m in zip(native_shape, self.shape):
            if (n, m) not in self._cache:
                self._cache[(n, m)] = torch.from_numpy(area_matrix(n, m))
            out.append(self._cache[(n, m)])
        return out

    def view(self, u16, lut, *, chunk=32):
        """Whole-volume coarse view; chunks along z keep memory near one slab."""
        az, ay, ax = self.matrices(u16.shape)
        lut_t = torch.from_numpy(lut)
        partial = torch.zeros((u16.shape[0], self.shape[1], self.shape[2]))
        for z0 in range(0, u16.shape[0], chunk):
            block = lut_t[torch.from_numpy(np.asarray(u16[z0:z0 + chunk]).astype(np.int64))]
            partial[z0:z0 + chunk] = torch.einsum("zyx,ix,jy->zji", block, ax, ay)
        return torch.einsum("zji,kz->kji", partial, az).numpy()

    def delta(self, difference, start, native_shape):
        """Coarse-view change produced by a box difference starting at `start`."""
        az, ay, ax = self.matrices(native_shape)
        (z0, y0, x0), (dz, dy, dx) = start, difference.shape
        d = torch.from_numpy(np.ascontiguousarray(difference, dtype=np.float32))
        return torch.einsum("zyx,kz,jy,ix->kji", d, az[:, z0:z0 + dz], ay[:, y0:y0 + dy], ax[:, x0:x0 + dx]).numpy()


def coarse_builder(shape=COARSE_SHAPE):
    """Callback for acquisition: store each real series' coarse view once (float16)."""
    resampler = AreaResampler(shape)

    def build(u16, meta, store_root):
        view = resampler.view(u16, intensity_lut(meta["u16_to_hu"])).astype(np.float16)
        path = Path(store_root) / "coarse" / (meta["series_key"] + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part.npy")
        np.save(part, view)
        part.replace(path)
        return {"file": f"coarse/{path.name}", "sha256": sha256_path(path), "shape": list(shape),
                "window_hu": list(WINDOW_HU), "resampling": "exact_area_average"}
    return build


def round8(value, low, high):
    return int(min(high, max(low, 8 * round(value / 8))))


def crop_shape_for(spacing_zyx, crop_mm=48.):
    """Native-grid crop covering ~crop_mm: z in [16, 64], y/x in [32, 96] voxels, multiples of 8."""
    sz, sy, sx = spacing_zyx
    return (round8(crop_mm / sz, 16, 64), round8(crop_mm / sy, 32, 96), round8(crop_mm / sx, 32, 96))


def _rng(*parts):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little"))


def changed_bbox(changed, start):
    """Bounding box [lo, hi) of the changed voxels in scan coordinates, as streaming records it."""
    points = np.argwhere(changed)
    if not len(points):
        return None
    return [[int(v) for v in points.min(0) + start], [int(v) for v in points.max(0) + 1 + start]]


def crop_tamper_threshold(changed_total, changed_extent, crop_shape, fraction=.25):
    """Changed voxels a crop must hold to count as tampered.

    A quarter of the changed voxels a crop of this shape can hold: all of them when the
    edit fits inside the crop (the earlier rule), and the crop's share along each axis
    when the edit is longer, as M3Dsynth removals are on thick slices.
    """
    share = float(np.prod([min(1., c / max(e, 1)) for c, e in zip(crop_shape, changed_extent)]))
    return fraction * changed_total * share


LABEL_FIELDS = ("changed_voxels", "mask_voxels", "mask_changed", "changed_in_mask", "depth_ratio", "changed_slice_fill")


def label_statistics(changed, mask):
    """How one sample's changed voxels relate to the author mask (both in box coordinates)."""
    both = int(np.count_nonzero(changed & mask))
    n_changed, n_mask = int(np.count_nonzero(changed)), int(np.count_nonzero(mask))
    changed_z = np.flatnonzero(changed.any(axis=(1, 2)))
    mask_z = np.flatnonzero(mask.any(axis=(1, 2)))
    changed_span = int(changed_z[-1] - changed_z[0] + 1) if len(changed_z) else 0
    mask_span = int(mask_z[-1] - mask_z[0] + 1) if len(mask_z) else 0
    return {"changed_voxels": n_changed, "mask_voxels": n_mask,
            "mask_changed": both / n_mask if n_mask else None,            # share of the author mask that changed
            "changed_in_mask": both / n_changed if n_changed else None,   # share of the edit inside the author mask
            "depth_ratio": changed_span / mask_span if mask_span else None,
            "changed_slice_fill": len(changed_z) / changed_span if changed_span else None}  # < 1: skipped slices


class M3DStore:
    """Read-only access to a verified store, with a local cache for random access."""

    def __init__(self, store_root, cache_root, *, manifest=None, resampler=None, box_cache=64):
        self.root, self.cache = Path(store_root), Path(cache_root)
        self.manifest = manifest or json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "m3dsynth_store_v1":
            raise ValueError("Unsupported M3Dsynth manifest")
        self.series = self.manifest["series"]
        self.resampler = resampler or AreaResampler(COARSE_SHAPE)
        self._luts, self._real, self._coarse = {}, {}, {}
        self._boxes, self._box_cache = OrderedDict(), box_cache
        self.label_stats = {}
        self.sites = {}
        for site in self.manifest["sites"]:
            self.sites.setdefault(site["series_key"], []).append(site)

    def lut(self, key):
        if key not in self._luts:
            self._luts[key] = intensity_lut(self.series[key]["u16_to_hu"])
        return self._luts[key]

    def real(self, key):
        if key not in self._real:
            local = self.cache / "real" / (key + ".npy")
            if not local.exists():
                materialize_local(self.root / "real" / (key + ".npz"), local, self.series[key]["content_sha256"])
            array = np.load(local, mmap_mode="r")
            if list(array.shape) != self.series[key]["shape_zyx"] or array.dtype != np.uint16:
                raise ValueError("Local real series differs from its receipt: " + key)
            self._real[key] = array
        return self._real[key]

    def coarse_real(self, key):
        if key not in self._coarse:
            info = self.series[key]["coarse"]
            path = self.root / info["file"]
            if sha256_path(path) != info["sha256"]:
                raise ValueError("Coarse view checksum mismatch: " + key)
            self._coarse[key] = np.load(path).astype(np.float32)
        return self._coarse[key]

    def box(self, volume):
        """(manipulated box, changed voxels, author mask) for one sample, in box coordinates.

        The changed voxels (manipulated != real) are the training labels. For an eligible
        sample their bounding box must equal the one measured while streaming, so the labels
        describe exactly the edit that was verified.
        """
        key = volume["volume_id"]
        if key in self._boxes:
            self._boxes.move_to_end(key)
            return self._boxes[key]
        path = self.root / volume["box_file"]
        local = self.cache / volume["box_file"]
        if not local.exists():
            if sha256_path(path) != volume["box_file_sha256"]:
                raise ValueError("Box checksum mismatch: " + key)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(path.read_bytes())
        with np.load(local, allow_pickle=False) as data:
            scan, mask = data["scan"].copy(), data["mask"].copy()
        expected = [b - a for a, b in zip(volume["box_start_zyx"], volume["box_stop_zyx"])]
        if list(scan.shape) != expected or mask.shape != scan.shape:
            raise ValueError("Box shape differs from manifest: " + key)
        (z0, y0, x0), (z1, y1, x1) = volume["box_start_zyx"], volume["box_stop_zyx"]
        changed = scan != np.asarray(self.real(volume["series_key"])[z0:z1, y0:y1, x0:x1])
        if volume.get("eligible") and changed_bbox(changed, volume["box_start_zyx"]) != volume.get("diff_bbox"):
            raise ValueError("Changed voxels differ from the edit verified while streaming: " + key)
        if key not in self.label_stats:
            self.label_stats[key] = label_statistics(changed, mask)
        item = (scan, changed, mask)
        self._boxes[key] = item
        if len(self._boxes) > self._box_cache:
            self._boxes.popitem(last=False)
        return item

    def localize(self, volumes, progress=None):
        """Copy/verify everything a split needs into the local cache before training."""
        keys = sorted({v["series_key"] for v in volumes})
        for i, key in enumerate(keys, 1):
            self.real(key)
            self.coarse_real(key)
            if progress and (i % 10 == 0 or i == len(keys)):
                progress({"stage": "local_cache", "series": i, "of": len(keys)})
        for volume in volumes:
            if volume["kind"] == "manipulated":
                self.box(volume)

    def label_check(self, volumes):
        """Training labels (changed voxels) against the author masks, by operation and generator."""
        groups = defaultdict(list)
        for volume in volumes:
            if volume["kind"] != "manipulated":
                continue
            if volume["volume_id"] not in self.label_stats:
                self.box(volume)
            stats = self.label_stats[volume["volume_id"]]
            groups[volume["operation"]].append(stats)
            groups[f"{volume['operation']}:{volume['generator']}"].append(stats)

        def summary(rows):
            out = {"volumes": len(rows)}
            for name in LABEL_FIELDS:
                values = [r[name] for r in rows if r[name] is not None]
                out[name] = (dict(zip(("p10", "median", "p90"), (float(v) for v in np.percentile(values, [10, 50, 90]))))
                             if values else None)
            return out
        return {"schema_version": "m3d_label_check_v1",
                "labels": "changed voxels (manipulated != real); the author mask only places the site crop",
                "fields": {"mask_changed": "share of the author mask that changed",
                           "changed_in_mask": "share of the changed voxels inside the author mask",
                           "depth_ratio": "slices spanned by the changes / slices spanned by the mask",
                           "changed_slice_fill": "slices with a change / slices spanned by the changes"},
                "groups": {k: summary(v) for k, v in sorted(groups.items())}, "test_volumes_opened": 0}

    def coarse(self, volume):
        """96^3 normalized view of the (possibly manipulated) volume, unflipped."""
        key = volume["series_key"]
        view = self.coarse_real(key).copy()
        if volume["kind"] == "manipulated":
            scan = self.box(volume)[0]
            (z0, y0, x0), (z1, y1, x1) = volume["box_start_zyx"], volume["box_stop_zyx"]
            lut = self.lut(key)
            real = np.asarray(self.real(key)[z0:z1, y0:y1, x0:x1])
            view += self.resampler.delta(lut[scan] - lut[real], (z0, y0, x0), self.series[key]["shape_zyx"])
        return view

    def crop(self, volume, center_native, shape):
        """Native crop (uint16 -> window) with the manipulated box pasted; zero outside the scan.

        Returns the image, the in-scan voxels and the changed voxels (the edit label) in the crop.
        """
        key = volume["series_key"]
        real = self.real(key)
        full = self.series[key]["shape_zyx"]
        start = [int(round(c - s / 2.)) for c, s in zip(center_native, shape)]
        u16 = np.zeros(shape, dtype=np.uint16)
        valid = np.zeros(shape, dtype=bool)
        edit = np.zeros(shape, dtype=bool)
        src, dst = [], []
        for a, s, n in zip(start, shape, full):
            lo, hi = max(a, 0), min(a + s, n)
            if hi <= lo:
                return np.zeros(shape, np.float32), valid, edit
            src.append(slice(lo, hi))
            dst.append(slice(lo - a, hi - a))
        u16[tuple(dst)] = real[tuple(src)]
        valid[tuple(dst)] = True
        if volume["kind"] == "manipulated":
            scan, changed, _ = self.box(volume)
            b0, b1 = volume["box_start_zyx"], volume["box_stop_zyx"]
            src, dst, empty = [], [], False
            for a, s, lo_b, hi_b in zip(start, shape, b0, b1):
                lo, hi = max(a, lo_b), min(a + s, hi_b)
                if hi <= lo:
                    empty = True
                    break
                src.append(slice(lo - lo_b, hi - lo_b))
                dst.append(slice(lo - a, hi - a))
            if not empty:
                u16[tuple(dst)] = scan[tuple(src)]
                edit[tuple(dst)] = changed[tuple(src)]
        image = self.lut(key)[u16]
        image[~valid] = 0.
        return image, valid, edit & valid


class VolumeCase:
    """One volume for one epoch: coarse view, cells, labels and a crop provider."""

    def __init__(self, store, volume, *, flip, grid, crop_mm, forced, random_centers, rng, jitter_fraction,
                 body_threshold=.04):
        self.store, self.volume, self.flip, self.grid = store, volume, flip, grid
        info = store.series[volume["series_key"]]
        self.native_shape = tuple(info["shape_zyx"])
        self.crop_shape = crop_shape_for(info["geometry"]["spacing_zyx"], crop_mm)
        coarse = store.coarse(volume)
        self.coarse = np.ascontiguousarray(coarse[..., ::-1] if flip else coarse)
        g, (cz, cy, cx) = grid, self.coarse.shape
        if cz % g or cy % g or cx % g:
            raise ValueError("Coarse view must divide into the cell grid")
        blocks = self.coarse.reshape(g, cz // g, g, cy // g, g, cx // g).mean(axis=(1, 3, 5))
        # Router candidates come from the image alone, in training and at inference alike.
        self.cell_valid = (blocks > body_threshold).reshape(-1)
        idx = np.stack(np.meshgrid(*[np.arange(g)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.cell_index = idx
        self.cell_centers = (idx + .5) * np.asarray(self.native_shape) / g - .5  # view == native up to x-flip
        self.cell_positions = ((idx + .5) / g * 2. - 1.).astype(np.float32)
        self.tampered = float(volume["kind"] == "manipulated")
        self.removal = (1. if volume["operation"] == "removal" else 0.) if self.tampered else -1.
        self.cell_target = np.zeros(g ** 3, dtype=np.float32)
        self.changed_total, self.tamper_threshold = 0, 0.
        self.site_center = None
        if self.tampered:
            # Router targets: every cell holding a changed voxel.
            points = np.argwhere(store.box(volume)[1]) + np.asarray(volume["box_start_zyx"])
            self.changed_total = len(points)
            if self.changed_total:
                cells = np.floor((self.to_view(points) + .5) * g / np.asarray(self.native_shape)).astype(int).clip(0, g - 1)
                self.cell_target[np.unique((cells[:, 0] * g + cells[:, 1]) * g + cells[:, 2])] = 1.
                extent = points.max(0) - points.min(0) + 1
                self.tamper_threshold = crop_tamper_threshold(self.changed_total, extent, self.crop_shape)
            # The site crop stays at the authors' declared site (the centre of their mask box);
            # on the first full store it lay within a voxel of the centre of the changed voxels.
            lo, hi = (np.asarray(v, dtype=float) for v in volume["mask_bbox"])
            lo_v, hi_v = self.to_view(lo), self.to_view(hi - 1)
            self.site_center = (np.minimum(lo_v, hi_v) + np.maximum(lo_v, hi_v)) / 2.
        # Cells supervised by the router loss: every true cell is a positive even when it
        # fails the body test, but the label never adds selection candidates (that leaked
        # the answer into validation routing; review of 24 Nov 2026).
        self.router_cells = self.cell_valid | (self.cell_target > 0)
        jitter = np.asarray(self.crop_shape, dtype=float) * jitter_fraction
        self.forced = []
        if forced:
            if self.tampered:
                visible = 0. if self.removal == 1. else 1.
                self.forced.append((self.site_center + rng.uniform(-jitter, jitter), visible))
            else:
                sites = store.sites.get(volume["series_key"], [])
                for site in (rng.choice(len(sites), size=min(forced, len(sites)), replace=False) if sites else []):
                    item = sites[int(site)]
                    center = self.to_view(np.asarray(item["center_zyx"], dtype=float))
                    self.forced.append((center + rng.uniform(-jitter, jitter), 1. if item["site_label"] == "TM" else 0.))
        valid_cells = np.flatnonzero(self.cell_valid)
        self.random = []
        if random_centers and len(valid_cells):
            chosen = rng.choice(valid_cells, size=min(random_centers, len(valid_cells)), replace=False)
            self.random = [self.cell_centers[c] + rng.uniform(-jitter, jitter) for c in chosen]

    def to_view(self, native_zyx):
        out = np.array(native_zyx, dtype=float)
        if self.flip:
            out[..., 2] = self.native_shape[2] - 1 - out[..., 2]
        return out

    def crops(self, centers_view):
        """K crops at view-space centres -> tensors and per-crop supervision."""
        images, valids, masks, tamper, positions = [], [], [], [], []
        for center in centers_view:
            native = self.to_view(center)  # the x-flip is its own inverse
            image, valid, edit = self.store.crop(self.volume, native, self.crop_shape)
            if self.flip:
                image, valid, edit = image[..., ::-1], valid[..., ::-1], edit[..., ::-1]
            images.append(np.ascontiguousarray(image))
            valids.append(np.ascontiguousarray(valid))
            masks.append(np.ascontiguousarray(edit))
            inside = int(edit.sum())
            tamper.append(float(bool(self.tampered and self.changed_total and inside >= self.tamper_threshold)))
            positions.append(np.clip(np.asarray(center) / (np.asarray(self.native_shape) - 1) * 2. - 1., -1, 1))
        return {"image": torch.from_numpy(np.stack(images))[:, None], "valid": torch.from_numpy(np.stack(valids))[:, None],
                "mask": torch.from_numpy(np.stack(masks))[:, None].float(),
                "tamper": torch.tensor(tamper, dtype=torch.float32),
                "positions": torch.tensor(np.stack(positions), dtype=torch.float32)}


class M3DDataset:
    def __init__(self, store, split, *, seed=13, train=False, flip_probability=.5, grid=12, crop_mm=48.,
                 forced_sites=1, random_crops=4, jitter_fraction=.25, generators=None, eligible_only=True):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        if not train and (flip_probability or forced_sites or random_crops):
            flip_probability, forced_sites, random_crops = 0., 0, 0
        self.store, self.split, self.seed, self.train = store, split, seed, train
        self.flip_probability, self.grid, self.crop_mm = flip_probability, grid, crop_mm
        self.forced_sites, self.random_crops, self.jitter = forced_sites, random_crops, jitter_fraction
        self.rows = [v for v in store.manifest["volumes"] if v["split"] == split and (v["eligible"] or not eligible_only)
                     and (generators is None or v["kind"] == "authentic" or v["generator"] in generators)]
        if not self.rows:
            raise ValueError(f"No eligible {split} volumes in the store")
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        volume = self.rows[index]
        rng = _rng(self.seed, self.epoch, volume["volume_id"], "case")
        flip = bool(self.train and rng.random() < self.flip_probability)
        return VolumeCase(self.store, volume, flip=flip, grid=self.grid, crop_mm=self.crop_mm,
                          forced=self.forced_sites if self.train else 0,
                          random_centers=self.random_crops if self.train else 0, rng=rng,
                          jitter_fraction=self.jitter if self.train else 0.)
