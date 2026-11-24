"""Numerical input and gradient audits for exploratory fitting controls."""

from collections import defaultdict
import hashlib
import json
import numpy as np
import torch

from rsvt.data.baseline_views import CachedViews


def audit_model_inputs(entries):
    """Inspect cached inputs only. These checks do not validate lesion preservation."""
    if not entries or any(r["split"] not in {"train","validation"} for r in entries):
        raise ValueError("Input audit accepts nonempty train/validation entries only")
    records,groups = [],defaultdict(list)
    for split in ("train","validation"):
        dataset = CachedViews(entries,split)
        for index in range(len(dataset)):
            sample = dataset[index]
            image = np.asarray(sample["image"].numpy(),dtype="<f4",order="C").copy()
            mask = sample["valid_mask"].numpy()
            image[~mask] = 0.  # The models zero invalid voxels before their encoder.
            valid = image[mask]
            fingerprint = hashlib.sha256(json.dumps(list(image.shape),separators=(",",":")).encode())
            fingerprint.update(image.tobytes(order="C"))
            fingerprint.update(mask.astype(np.uint8).tobytes(order="C"))
            row = {"case_key":sample["case_key"],"split":split,"candidate_target":sample["target"],
                   "model_input_sha256":fingerprint.hexdigest(),"valid_voxels":int(mask.sum()),
                   "valid_fraction":float(mask.mean()),"minimum":float(valid.min()),"maximum":float(valid.max()),
                   "mean":float(valid.mean(dtype=np.float64)),"std":float(valid.std(dtype=np.float64)),
                   "constant_valid_intensity":bool(valid.min()==valid.max()),
                   "out_of_range_valid_voxels":int(((valid<0)|(valid>1)).sum())}
            records.append(row);groups[row["model_input_sha256"]].append(row)
    repeated = [{"model_input_sha256":key,"case_keys":[r["case_key"] for r in rows],
                 "candidate_targets":sorted({r["candidate_target"] for r in rows}),
                 "partitions":sorted({r["split"] for r in rows})}
                for key,rows in groups.items() if len(rows)>1]
    conflicts = [g for g in repeated if len(g["candidate_targets"])>1]
    overlap = [g for g in repeated if len(g["partitions"])>1]
    outside = sum(r["out_of_range_valid_voxels"] for r in records)
    return {"ok":not conflicts and not overlap and outside==0,"cases":len(records),"records":records,
            "duplicate_model_input_groups":repeated,"conflicting_target_groups":conflicts,
            "train_validation_overlap_groups":overlap,"out_of_range_valid_voxels":outside,
            "constant_valid_intensity_cases":[r["case_key"] for r in records if r["constant_valid_intensity"]],
            "fingerprint_policy":"shape_then_little_endian_float32_zeroed_invalid_image_then_uint8_validity_C_order",
            "scope":"numerical_model_input_sanity_not_lesion_preservation_or_target_validation","test_cases_opened":0}


def gradient_groups(model):
    """Read unscaled gradients by top-level module without changing them or RNG."""
    groups = {}
    for name,parameter in model.named_parameters():
        root = name.split(".",1)[0]
        item = groups.setdefault(root,{"parameter_tensors":0,"with_gradient":0,"norms":[]})
        item["parameter_tensors"] += 1
        if parameter.grad is not None:
            item["with_gradient"] += 1
            item["norms"].append(torch.linalg.vector_norm(parameter.grad.detach().float()))
    output = {}
    for root,item in groups.items():
        norm = torch.linalg.vector_norm(torch.stack(item["norms"])) if item["norms"] else None
        finite = bool(torch.isfinite(norm)) if norm is not None else None
        output[root] = {"parameter_tensors":item["parameter_tensors"],"with_gradient":item["with_gradient"],
                        "l2_norm":float(norm) if finite else None,"finite":finite}
    return output
