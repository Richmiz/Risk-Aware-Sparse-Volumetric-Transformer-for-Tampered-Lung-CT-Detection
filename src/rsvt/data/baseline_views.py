"""Deterministic full-FOV coarse views for exploratory baseline training."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch

from rsvt.data.acquisition import sha256_file
from rsvt.data.dataset import CTVolumeDataset
from rsvt.data.phase2 import _json
from rsvt.data.splits import CLASS_INDEX, digest_json


def select_pilot_rows(records, seed, train_per_class, validation_per_class):
    selected = []
    for split, limit in (("train",train_per_class),("validation",validation_per_class)):
        if isinstance(limit,bool) or not isinstance(limit,int) or limit < 0:
            raise ValueError("Subset limits must be nonnegative integers; zero selects the full partition")
        for label in CLASS_INDEX:
            own = [r for r in records if r["split"] == split and r["label_candidate"] == label]
            if not own:
                raise ValueError("Train/validation must represent all four candidate classes")
            own.sort(key=lambda r:digest_json([seed,split,r["case_key"]]))
            selected += own[:limit] if limit else own
    return sorted(selected,key=lambda r:(r["split"],r["case_key"]))


def coarse_full_fov(sample, shape_zyx):
    if len(shape_zyx) != 3 or any(isinstance(n,bool) or not isinstance(n,int) or n < 2 for n in shape_zyx):
        raise ValueError("Coarse shape needs three integer sizes >=2")
    affine = np.asarray(sample["voxel_to_lps"],dtype=float)
    old_size = np.asarray(sample["image"].shape[-3:][::-1],dtype=float)
    new_size = np.asarray(tuple(reversed(shape_zyx)),dtype=float)
    spacing = np.linalg.norm(affine[:3,:3],axis=0)
    direction = affine[:3,:3]/spacing
    original = sitk.GetImageFromArray(sample["image"][0])
    original.SetSpacing(tuple(spacing)); original.SetOrigin(tuple(affine[:3,3])); original.SetDirection(tuple(direction.ravel()))
    valid = sitk.GetImageFromArray(sample["valid_mask"][0].astype(np.uint8))
    valid.CopyInformation(original)
    mapping = np.eye(4)
    mapping[:3,:3] = np.diag(old_size/new_size)
    mapping[:3,3] = (old_size/new_size-1.)/2.
    coarse_affine = affine @ mapping
    reference = sitk.Image([int(n) for n in new_size],sitk.sitkFloat32)
    reference.SetDirection(tuple(direction.ravel()))
    reference.SetSpacing(tuple(spacing*old_size/new_size))
    reference.SetOrigin(tuple(coarse_affine[:3,3]))
    image = sitk.GetArrayFromImage(sitk.Resample(original,reference,sitk.Transform(3,sitk.sitkIdentity),sitk.sitkLinear,0.,sitk.sitkFloat32))
    mask = sitk.GetArrayFromImage(sitk.Resample(valid,reference,sitk.Transform(3,sitk.sitkIdentity),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)).astype(bool)
    if not mask.any() or not np.isfinite(image).all():
        raise ValueError("Coarse view lost all valid support or contains nonfinite pixels")
    image[~mask] = 0.
    return {"image":image[None],"valid_mask":mask[None],"voxel_to_lps":coarse_affine,
            "coarse_to_processed_xyz":mapping}


def prepare_views(manifest_path, selected, cache_root, shape_zyx, *, max_decode_bytes=3*1024**3, progress=None):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True,exist_ok=True)
    datasets = {split:CTVolumeDataset(manifest_path,split,purpose="engineering_pilot",allow_candidate_targets=True,
                                    max_decode_bytes=max_decode_bytes) for split in ("train","validation")}
    output = []
    for row in selected:
        if row["split"] not in datasets:
            raise ValueError("Baseline view preparation must not access test pixels")
        name = "case-"+digest_json(row["case_key"])[:20]
        path, marker = cache_root/(name+".npz"), cache_root/(name+".json")
        binding = digest_json({"implementation":"f04_coarse_full_support_v1","shape_zyx":list(shape_zyx),
                    **{key:row[key] for key in ("case_key","split","normalized_sha256","valid_mask_sha256","metadata_sha256")}})
        if marker.is_file():
            receipt = json.loads(marker.read_text(encoding="utf-8"))
            if (receipt["binding"] != binding or path.is_symlink() or path.stat().st_size != receipt["bytes"]
                    or sha256_file(path) != receipt["sha256"]):
                raise ValueError("Cached coarse view differs from its source/policy receipt")
        else:
            if path.exists():
                raise ValueError("Coarse view has no completion receipt; use a new run or inspect partial cache")
            dataset = datasets[row["split"]]
            index = next(i for i,r in enumerate(dataset.rows) if r["case_key"] == row["case_key"])
            sample = dataset[index]
            view = coarse_full_fov(sample,shape_zyx)
            temporary = path.with_name(path.name+".part")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream,**view)
            temporary.replace(path)
            receipt = {"binding":binding,"sha256":sha256_file(path),"bytes":path.stat().st_size,
                       "shape_zyx":list(shape_zyx),"classification_eligible":False,"full_field_of_view":True,
                       "case_key":row["case_key"],"source_normalized_sha256":row["normalized_sha256"]}
            _json(marker,receipt)
            del sample,view
        output.append({"case_key":row["case_key"],"split":row["split"],"label_candidate":row["label_candidate"],
                       "source_group_candidate":row["source_group_candidate"],"label_index":CLASS_INDEX[row["label_candidate"]],
                       "path":str(path),**receipt})
        _json(cache_root/"index.json",output)
        if progress:
            progress({"stage":"coarse_view","case_key":row["case_key"],"split":row["split"]})
    return output


class CachedViews:
    def __init__(self, entries, split, *, include_geometry=False):
        if split not in {"train","validation"}:
            raise ValueError("F04 pilot does not open test inputs")
        self.rows = [r for r in entries if r["split"] == split]
        self.include_geometry = include_geometry
        if not self.rows:
            raise ValueError("Empty cached-view partition")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        path = Path(row["path"])
        if path.is_symlink() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("Coarse training input changed after caching")
        with np.load(path,allow_pickle=False) as data:
            image,valid = data["image"].copy(),data["valid_mask"].copy()
            affine = data['voxel_to_lps'].copy() if self.include_geometry else None
        if (image.dtype != np.float32 or valid.dtype != bool or image.shape != valid.shape
                or image.shape != (1,*row["shape_zyx"]) or not np.isfinite(image).all() or not valid.any()):
            raise ValueError("Invalid coarse-view array contract")
        result = {"image":torch.from_numpy(image),"valid_mask":torch.from_numpy(valid),
                  "target":int(row["label_index"]),"case_key":row["case_key"]}
        if self.include_geometry:
            if (affine.shape != (4,4) or not np.isfinite(affine).all()
                    or not np.allclose(affine[3],[0,0,0,1]) or abs(np.linalg.det(affine[:3,:3])) < 1e-12):
                raise ValueError('Invalid cached physical geometry')
            result['voxel_to_lps'] = torch.as_tensor(affine,dtype=torch.float64)
        return result
