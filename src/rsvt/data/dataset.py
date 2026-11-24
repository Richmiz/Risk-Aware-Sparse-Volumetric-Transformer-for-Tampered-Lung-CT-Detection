"""Lazy full-volume loading, bounded padding, and optional PyTorch conversion."""

from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk

from rsvt.data.acquisition import sha256_file
from rsvt.data.cache import _check_header
from rsvt.data.splits import CLASS_INDEX, audit_partitions, digest_json


def _stat(path: Path):
    item = path.stat()
    return item.st_size, item.st_mtime_ns


class CTVolumeDataset:
    """Map-style dataset returning float32 CZYX arrays; no voxel cache or resizing.

    Draft manifests permit smoke inspection of training inputs. Actual training
    or evaluation requires a separately reviewed manifest marked ready.
    """

    def __init__(self, manifest_path: Path, split: str, *, purpose="smoke", seed=13,
                 flip_axes=(), flip_probability=0., max_decode_bytes=3*1024**3,
                 allow_candidate_targets=False):
        content = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if content.get("schema_version") != "f03_candidate_v1":
            raise ValueError("Unsupported F03 manifest schema")
        if purpose not in {"smoke", "training", "evaluation", "engineering_pilot"}:
            raise ValueError("Unknown dataset purpose")
        if split not in {"train", "validation", "test"}:
            raise ValueError("Invalid dataset partition")
        if purpose == "smoke" and split != "train":
            raise ValueError("Draft smoke inspection reads training inputs only")
        if purpose == "engineering_pilot" and (allow_candidate_targets is not True or split == "test"):
            raise ValueError("Engineering candidate-target mode requires explicit acknowledgement and train/validation only")
        if purpose in {"training", "evaluation"} and (content.get("training_ready") is not True or content.get("status") != "reviewed"):
            raise ValueError("Draft manifest is not ready for training/evaluation")
        if content.get("class_index") != CLASS_INDEX:
            raise ValueError("Manifest class mapping differs from the loader contract")
        records = content["records"]
        assignments = [{key: r[key] for key in ("case_key", "source_group_candidate", "split", "label_candidate",
                        "source_file_set_sha256", "normalized_sha256")} for r in sorted(records, key=lambda r: r["case_key"])]
        if digest_json(assignments) != content.get("assignments_sha256"):
            raise ValueError("Manifest assignment fingerprint mismatch")
        if any(not isinstance(r.get(key), bool) for r in records
               for key in ("classification_eligible", "identity_reconciled", "label_verified")):
            raise ValueError("Eligibility and identity flags must be booleans")
        if len({r["case_key"] for r in records}) != len(records) or not audit_partitions(records)["ok"]:
            raise ValueError("Manifest has duplicate cases or partition leakage")
        if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                or not isinstance(max_decode_bytes, int) or isinstance(max_decode_bytes, bool) or max_decode_bytes <= 0):
            raise ValueError("Invalid seed or decode memory limit")
        if (len(set(flip_axes)) != len(flip_axes) or not set(flip_axes).issubset({"x", "y", "z"})
                or isinstance(flip_probability, bool) or not np.isfinite(flip_probability)
                or not 0 <= flip_probability <= 1):
            raise ValueError("Invalid spatial augmentation policy")
        if split != "train" and (flip_axes or flip_probability):
            raise ValueError("Augmentation is allowed on training inputs only")
        self.rows = [r for r in records if r["split"] == split]
        if not self.rows:
            raise ValueError("Requested partition is empty")
        if purpose in {"training", "evaluation"} and any(r.get("classification_eligible") is not True or r.get("identity_reconciled") is not True
                                     or r.get("label_verified") is not True or r.get("partition_status") != "reviewed" for r in self.rows):
            raise ValueError("Unverified labels/source identities cannot be supervised targets")
        self.seed, self.epoch = seed, 0
        self.flip_axes, self.flip_probability = tuple(flip_axes), flip_probability
        self.max_decode_bytes = max_decode_bytes

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch: int):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        self.epoch = epoch

    def __getitem__(self, index):
        row = self.rows[index]
        paths = [Path(row[key]) for key in ("volume_path", "valid_mask_path", "metadata_path")]
        stamps = [_stat(path) for path in paths]
        for path, hash_key, size_key in zip(paths,
                ("normalized_sha256", "valid_mask_sha256", "metadata_sha256"),
                ("normalized_bytes", "valid_mask_bytes", "metadata_bytes")):
            if path.is_symlink() or path.stat().st_size != row[size_key] or sha256_file(path) != row[hash_key]:
                raise ValueError(f"Dataset artifact changed: {path.name}")
        metadata = json.loads(paths[2].read_text(encoding="utf-8"))
        if (metadata["case_key"] != row["case_key"] or metadata["source_candidate_id"] != row["source_candidate_id"]
                or metadata["source_file_set_sha256"] != row["source_file_set_sha256"]):
            raise ValueError("Dataset metadata identity mismatch")
        grid = metadata["transform"]["processed"]
        size = grid["size_xyz"]
        # Includes decode/copy temporaries, rather than only the final arrays.
        if int(np.prod(size, dtype=object)) * 12 > self.max_decode_bytes:
            raise ValueError("Volume exceeds decode memory limit before pixel allocation")
        _check_header(paths[0], grid)
        _check_header(paths[1], grid)
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(paths[0])))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(paths[1])))
        if image.dtype != np.float32 or image.shape != tuple(reversed(size)) or mask.shape != image.shape:
            raise ValueError("Decoded array shape/type mismatch")
        if (not np.isfinite(image).all() or float(image.min()) < -1e-6 or float(image.max()) > 1+1e-6
                or int(mask.max()) > 1 or not np.any(mask)):
            raise ValueError("Invalid normalized values or binary validity mask")
        if stamps != [_stat(path) for path in paths]:
            raise ValueError("Dataset artifact changed during decoding")
        valid = mask.astype(bool)
        del mask
        image[~valid] = 0.
        affine = np.eye(4)
        affine[:3, :3] = np.asarray(grid["direction"]).reshape(3, 3) @ np.diag(grid["spacing_xyz_mm"])
        affine[:3, 3] = grid["origin_lps_mm"]
        augmentation = np.eye(4)
        salt = f"{self.seed}:{self.epoch}:{row['case_key']}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(salt).digest()[:8], "little"))
        flipped = []
        for name in sorted(self.flip_axes):
            if rng.random() < self.flip_probability:
                axis = ("x", "y", "z").index(name)
                image, valid = np.flip(image, 2-axis), np.flip(valid, 2-axis)
                augmentation[axis, axis] = -1
                augmentation[axis, 3] = size[axis]-1
                flipped.append(name)
        points = []
        for candidate in metadata.get("annotations", []):
            if "processed_index_xyz" in candidate:
                item = deepcopy(candidate)
                original = np.array([*item["processed_index_xyz"], 1.])
                item["index_xyz"] = (np.linalg.inv(augmentation) @ original)[:3].tolist()
                # F03 does not promote F02 points to localization supervision.
                item["supervision_eligible"] = False
                points.append(item)
        return {"image": np.ascontiguousarray(image[None]), "valid_mask": np.ascontiguousarray(valid[None]),
                "candidate_label_index": np.int64(CLASS_INDEX[row["label_candidate"]]),
                "candidate_tampered": np.bool_(row["label_candidate"] in {"FB", "FM"}),
                "classification_eligible": np.bool_(row["classification_eligible"]),
                "case_key": row["case_key"], "source_group_candidate": row["source_group_candidate"],
                "generator_candidate": row["generator_candidate"], "split": row["split"],
                "voxel_to_lps": affine @ augmentation, "annotation_candidates": points,
                "augmentation": {"flipped_axes": flipped, "augmented_to_processed_xyz": augmentation.tolist()}}


def pad_collate(samples: list[dict], *, max_batch_bytes=2*1024**3, as_torch=False) -> dict:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    if isinstance(max_batch_bytes, bool) or not isinstance(max_batch_bytes, int) or max_batch_bytes <= 0:
        raise ValueError("Batch memory limit must be a positive integer")
    for sample in samples:
        if (sample["image"].ndim != 4 or sample["image"].shape[0] != 1
                or sample["image"].shape != sample["valid_mask"].shape
                or sample["image"].dtype != np.float32 or sample["valid_mask"].dtype != bool):
            raise ValueError("Samples must contain float32 CZYX images and matching bool masks")
    shape = tuple(max(sample["image"].shape[axis] for sample in samples) for axis in range(4))
    required = int(np.prod(shape, dtype=object)) * len(samples) * 6
    required += sum(sample["image"].nbytes + sample["valid_mask"].nbytes for sample in samples)
    if required > max_batch_bytes:
        raise ValueError("Padded batch exceeds memory limit before allocation")
    batch_shape = (len(samples), *shape)
    images = np.zeros(batch_shape, dtype=np.float32)
    valid = np.zeros(batch_shape, dtype=bool)
    support = np.zeros(batch_shape, dtype=bool)
    for index, sample in enumerate(samples):
        region = (index, *[slice(0, n) for n in sample["image"].shape])
        images[region] = sample["image"]
        valid[region] = sample["valid_mask"]
        support[region] = True
    output = {"image": images, "valid_mask": valid, "padding_mask": support,
              "candidate_label_index": np.array([s["candidate_label_index"] for s in samples], dtype=np.int64),
              "candidate_tampered": np.array([s["candidate_tampered"] for s in samples], dtype=bool),
              "classification_eligible": np.array([s["classification_eligible"] for s in samples], dtype=bool),
              "original_shape_czyx": np.array([s["image"].shape for s in samples], dtype=np.int64),
              "voxel_to_lps": np.stack([s["voxel_to_lps"] for s in samples]),
              **{key: [s[key] for s in samples] for key in ("case_key", "source_group_candidate", "split",
                 "generator_candidate", "annotation_candidates", "augmentation")}}
    return to_torch_batch(output) if as_torch else output


def to_torch_batch(batch: dict) -> dict:
    import torch
    return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value for key, value in batch.items()}
