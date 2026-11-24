"""Pre-training diagnostics on train/validation only (no test volumes).

1. Resolution check: does the tampering evidence survive the 96^3 coarse view?
   Paired crops at each manipulation centre (manipulated vs the same location in
   the authentic series) are classified by the same small CNN twice: at native
   resolution and after degrading to the coarse-view voxel size. A large gap
   confirms that the evidence lives at native resolution.
2. Shortcut check: are manipulated and authentic volumes identical outside the
   edit (archive verification), do author masks cover the edits, and can global
   intensity statistics alone separate the classes?
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

from rsvt.data.m3dsynth_dataset import COARSE_SHAPE, _rng
from rsvt.data.m3dsynth_store import write_json
from rsvt.evaluation.classification import binary_auroc
from rsvt.models.coarse_to_fine import FineBranch


# Each manipulated crop is paired with an authentic site of the same appearance class:
# a removal (nodule replaced by tissue) with an authentic TB site (no nodule), an
# insertion (synthetic nodule) with an authentic TM site (a real nodule). Pairing with
# the same location in the source scan compared "nodule vs no nodule", which any
# resolution separates, so it could not show whether forensic evidence needs native
# resolution (review of 24 Nov 2026).
MATCHED_SITE = {"removal": "TB", "insertion": "TM"}


def _pairs(store, split, generators, limit, seed, shape):
    volumes = [v for v in store.manifest["volumes"] if v["split"] == split and v["kind"] == "manipulated"
               and v["eligible"] and v["generator"] in generators]
    rng = _rng(seed, split, "resolution_pairs")
    if limit and len(volumes) > limit:
        volumes = [volumes[i] for i in sorted(rng.choice(len(volumes), size=limit, replace=False))]
    same_series, same_split = {}, {}
    for site in store.manifest["sites"]:
        if site["split"] == split:
            same_series.setdefault((site["series_key"], site["site_label"]), []).append(site)
            same_split.setdefault(site["site_label"], []).append(site)
    images, labels, factors, groups, records = [], [], [], [], []
    for volume in volumes:
        wanted = MATCHED_SITE[volume["operation"]]
        candidates = same_series.get((volume["series_key"], wanted)) or same_split.get(wanted)
        if not candidates:
            records.append({"volume_id": volume["volume_id"], "paired": False, "reason": f"no authentic {wanted} site"})
            continue
        site = candidates[int(rng.integers(len(candidates)))]
        authentic = {"kind": "authentic", "series_key": site["series_key"], "volume_id": "real:" + site["series_key"]}
        group = f"{volume['generator']}:{volume['operation']}"
        # Both crops are centred on declared site centres, so neither class carries a centring cue.
        for item, center, label in ((volume, volume["center_zyx"], 1.), (authentic, site["center_zyx"], 0.)):
            image, valid, _ = store.crop(item, np.asarray(center, dtype=float), shape)
            images.append(np.where(valid, image, 0.).astype(np.float32))
            labels.append(label)
            factors.append([n / c for n, c in zip(store.series[item["series_key"]]["shape_zyx"], COARSE_SHAPE)])
            groups.append(group)
        records.append({"volume_id": volume["volume_id"], "paired": True, "operation": volume["operation"],
                        "authentic_site": site["img_id"], "authentic_site_label": wanted,
                        "same_series": site["series_key"] == volume["series_key"]})
    if not images:
        raise ValueError(f"No eligible {split} manipulated volumes with matched authentic sites for the resolution check")
    return torch.from_numpy(np.stack(images))[:, None], torch.tensor(labels), factors, groups, records


def degrade(images, factors):
    """Area-average to the coarse-view voxel size, then trilinear back to the crop grid."""
    out = []
    with torch.no_grad():
        for image, factor in zip(images, factors):
            size = [max(1, int(round(n / max(f, 1.)))) for n, f in zip(image.shape[1:], factor)]
            small = F.interpolate(image[None], size=size, mode="area")
            out.append(F.interpolate(small, size=image.shape[1:], mode="trilinear", align_corners=False)[0])
    return torch.stack(out)


def _train_classifier(train_x, train_y, val_x, val_y, *, device, epochs, seed, batch=16, lr=1e-3):
    torch.manual_seed(seed)
    model = FineBranch(dimension=32, widths=(8, 16, 16)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        losses = []
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            x = train_x[index].to(device)
            flip = torch.rand(len(index), generator=generator) < .5
            x = torch.where(flip[:, None, None, None, None].to(device), x.flip(-1), x)
            out = model(x, torch.ones_like(x, dtype=torch.bool))
            loss = F.binary_cross_entropy_with_logits(out["tamper_logits"], train_y[index].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(val_x), 64):
            x = val_x[start:start + 64].to(device)
            scores.append(torch.sigmoid(model(x, torch.ones_like(x, dtype=torch.bool))["tamper_logits"]).cpu())
    scores = torch.cat(scores).numpy()
    return {"validation_auroc": binary_auroc(val_y.numpy().astype(int), scores), "train_loss_by_epoch": history}, scores


def resolution_check(store, generators, output_dir, *, device="cpu", pairs_train=600, pairs_validation=200,
                     epochs=6, crop_shape=(16, 48, 48), seed=13, progress=None):
    train_x, train_y, train_f, _, train_records = _pairs(store, "train", generators, pairs_train, seed, crop_shape)
    val_x, val_y, val_f, val_groups, val_records = _pairs(store, "validation", generators, pairs_validation, seed, crop_shape)
    if progress:
        progress({"stage": "resolution_check", "train_crops": len(train_x), "validation_crops": len(val_x)})
    native, native_scores = _train_classifier(train_x, train_y, val_x, val_y, device=device, epochs=epochs, seed=seed)
    if progress:
        progress({"stage": "resolution_check", "arm": "native", "auroc": native["validation_auroc"]})
    coarse, coarse_scores = _train_classifier(degrade(train_x, train_f), train_y, degrade(val_x, val_f), val_y,
                                              device=device, epochs=epochs, seed=seed)
    labels = val_y.numpy().astype(int)

    def arms(index):
        result = {"native": binary_auroc(labels[index], native_scores[index]),
                  "coarse_equivalent": binary_auroc(labels[index], coarse_scores[index])}
        result["gap"] = (None if None in result.values() else result["native"] - result["coarse_equivalent"])
        return result

    by_group = {group: arms([i for i, g in enumerate(val_groups) if g == group]) for group in sorted(set(val_groups))}
    by_operation = {op: arms([i for i, g in enumerate(val_groups) if g.endswith(":" + op)])
                    for op in ("removal", "insertion") if any(g.endswith(":" + op) for g in val_groups)}
    gap = (None if native["validation_auroc"] is None or coarse["validation_auroc"] is None
           else native["validation_auroc"] - coarse["validation_auroc"])
    removal_gap = by_operation.get("removal", {}).get("gap")
    result = {"schema_version": "m3d_resolution_check_v2", "crop_shape_zyx": list(crop_shape),
              "pairs": {"train": len(train_x) // 2, "validation": len(val_x) // 2},
              "unpaired": {"train": sum(not r["paired"] for r in train_records),
                           "validation": sum(not r["paired"] for r in val_records)},
              "native": native, "coarse_equivalent": coarse, "auroc_gap_native_minus_coarse": gap,
              "by_operation": by_operation, "by_generator_operation": by_group,
              "design": ("manipulated crop at the declared site centre vs an authentic site of the same appearance "
                         "(removal vs TB, insertion vs TM; same series when available); identical small CNN, seed and "
                         "schedule for both arms; validation patients unseen in training"),
              "scope": ("tests whether the evidence at a known site survives coarse voxels; it does not test whether "
                        "the router can localize the site from the 96^3 view"),
              "interpretation": ("removal evidence is resolution-dependent: native crops are needed for removals"
                                 if removal_gap is not None and removal_gap >= .1 else
                                 "no large removal resolution gap at this sample size; inspect per-operation AUROCs"),
              "pairing_records": {"validation": val_records},
              "test_volumes_opened": 0}
    write_json(Path(output_dir) / "resolution_check.json", result)
    return result


def _histogram_features(store, volume):
    view = store.coarse(volume)
    histogram, _ = np.histogram(view, bins=32, range=(0., 1.))
    histogram = histogram / histogram.sum()
    return np.concatenate([histogram, [view.mean(), view.std()]]).astype(np.float32)


def _logistic(train_x, train_y, test_x, steps=400, lr=.1, l2=1e-2):
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    x, z = torch.from_numpy((train_x - mean) / std), torch.from_numpy((test_x - mean) / std)
    y = torch.from_numpy(train_y.astype(np.float32))
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    positive = float(y.mean())
    pos_weight = torch.tensor((1 - positive) / max(positive, 1e-6))
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(x @ weight + bias, y, pos_weight=pos_weight) + l2 * weight.square().sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return torch.sigmoid(z @ weight + bias).numpy()


def shortcut_check(store, generators, output_dir, *, manipulated_cap=400, seed=13):
    root = Path(store.root)
    volumes = [v for v in store.manifest["volumes"] if v["kind"] == "authentic" or v["generator"] in generators]
    receipts = []
    for volume in volumes:
        if volume["kind"] == "manipulated":
            path = root / volume["box_file"].replace(".npz", ".json")
            receipts.append(json.loads(path.read_text(encoding="utf-8")))
    eligible = [r for r in receipts if r["eligible"]]
    beyond_mask = []
    for r in eligible:
        if r["diff_bbox"] and r["mask_bbox"]:
            (dlo, dhi), (mlo, mhi) = r["diff_bbox"], r["mask_bbox"]
            beyond_mask.append(any(a < b for a, b in zip(dlo, mlo)) or any(a > b for a, b in zip(dhi, mhi)))
    identity = {"manipulated_receipts": len(receipts), "eligible": len(eligible),
                "ineligible_reasons": dict(Counter(re.sub(r"\d+", "N", p)
                                                   for r in receipts if not r["eligible"] for p in r["problems"])),
                "eligible_changed_voxels_outside_box": sum(r["diff_outside_box"] for r in eligible),
                "eligible_center_inside_mask_bbox": float(np.mean([r["center_inside_mask_bbox"] for r in eligible])) if eligible else None,
                "edits_extending_beyond_author_mask_bbox": float(np.mean(beyond_mask)) if beyond_mask else None}
    features = {}
    for split in ("train", "validation"):
        chosen = [v for v in volumes if v["split"] == split and v["eligible"]]
        manipulated = [v for v in chosen if v["kind"] == "manipulated"]
        rng = _rng(seed, split, "shortcut")
        if manipulated_cap and len(manipulated) > manipulated_cap:
            manipulated = [manipulated[i] for i in sorted(rng.choice(len(manipulated), manipulated_cap, replace=False))]
        rows = [v for v in chosen if v["kind"] == "authentic"] + manipulated
        features[split] = (np.stack([_histogram_features(store, v) for v in rows]),
                           np.array([v["kind"] == "manipulated" for v in rows], dtype=int))
    scores = _logistic(features["train"][0], features["train"][1], features["validation"][0])
    auroc = binary_auroc(features["validation"][1], scores)
    result = {"schema_version": "m3d_shortcut_check_v1", "archive_identity": identity,
              "global_histogram_classifier": {"features": "32-bin coarse-view histogram + mean + std",
                                              "train_volumes": int(len(features["train"][1])),
                                              "validation_volumes": int(len(features["validation"][1])),
                                              "validation_auroc": auroc},
              "interpretation": ("no global intensity shortcut detected" if auroc is not None and abs(auroc - .5) < .1
                                 else "global statistics separate classes: investigate before trusting detector results"),
              "test_volumes_opened": 0}
    write_json(Path(output_dir) / "shortcut_check.json", result)
    return result
