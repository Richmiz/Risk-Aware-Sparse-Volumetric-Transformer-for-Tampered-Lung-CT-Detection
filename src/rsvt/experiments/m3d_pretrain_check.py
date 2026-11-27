"""Crop pretraining with a transfer check, then a crop-swap test of the scan classifier.

The second full training run (26 Nov 2026) left its crop network at the loss of a constant
prediction: its crop scores ranked edits below real nodules (crop AUROC 0.37), and adding the
edit's crop to a scan changed its score by at most 0.0007. The crop learning check had shown
that the same network, trained alone on paired residual crops, learns the edit on unseen
patients. Before a third full run, two short checks (agreed 27 Nov 2026):

1. Pretrain and verify the transfer. The model's crop network is trained alone with the crop
   check's recipe, but on crops cut the way the full model cuts its site crop: each scan's own
   ~48 mm crop shape, centred on the authors' mask box, with the residual computed inside the
   network on the crop itself. The weights are saved and loaded into a fresh full model, and
   the same validation crops are scored again before any optimizer step; the scores must
   match (this catches loading errors and a residual applied twice). The validation crops are
   also cut again through the model's own data path and compared.
2. Give the scan classifier a fair test. With the verified crop network frozen, the rest of
   the model (coarse branch and router, fusion and heads) trains briefly on training patients.
   On held-out patients each edited scan is then scored twice with identical context and the
   same number of crops: one slot holds the edited crop, or the same crop of the unedited
   scan. The paired change in the scan score shows whether the scan classifier uses the crop
   evidence; the largest crop score among the scan's crops is a yardstick that needs no
   fusion. The unedited crop is used only in this test, never in training.

Train/validation only; no test volume is opened. Exploratory: the rules in `interpret`
describe the result; they are not pre-registered endpoints.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from rsvt.data.m3dsynth_dataset import VolumeCase, _rng, crop_shape_for, site_center_native
from rsvt.data.m3dsynth_store import digest_json, write_json
from rsvt.experiments.m3d_crop_check import OPERATIONS, pair_metrics, select_pairs, verdict
from rsvt.models.coarse_to_fine import CoarseToFineRSVT, FineBranch
from rsvt.training import coarse_to_fine as trainer
from rsvt.training.checkpoint import save_checkpoint
from rsvt.training.optimization import build_optimizer
from rsvt.utils.reproducibility import seed_everything

SCHEMA = "m3d_pretrain_check_v1"
WEIGHTS = "fine_branch.pt"
WEIGHTS_FORMAT = "rsvt_fine_branch_v1"


def unedited(volume):
    """The real (unedited) scan of a manipulated volume's series."""
    return {"kind": "authentic", "series_key": volume["series_key"], "volume_id": "real:" + volume["series_key"]}


def fine_kwargs(model_config):
    """The full model's crop network settings, so pretrained weights load into it unchanged."""
    return {"dimension": model_config["dimension"], "widths": tuple(model_config["fine_widths"]),
            "input": model_config.get("fine_input", "residual")}


def cluster_interval(values, clusters, resamples=2000, seed=13):
    """95% percentile interval of a mean, resampling whole patients (a patient's scans move together)."""
    sums, counts = {}, {}
    for value, cluster in zip(values, clusters):
        sums[cluster] = sums.get(cluster, 0.) + float(value)
        counts[cluster] = counts.get(cluster, 0) + 1
    totals, sizes = np.asarray(list(sums.values())), np.asarray([counts[c] for c in sums])
    picks = np.random.default_rng(seed).integers(0, len(totals), size=(resamples, len(totals)))
    means = totals[picks].sum(1) / sizes[picks].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# --------------------------------------------------------------------------- pairs


def site_pairs(store, volumes, crop_mm, margin_fraction=.125, progress=None, split=""):
    """Edited and unedited crops at each site, cut as the full model cuts its site crop.

    Each crop has its scan's own shape for ~crop_mm (`crop_shape_for`) and is centred where
    the model's site crop is (`site_center_native`), with a margin on every side for training
    shifts. Margins are even, so the centre of the larger crop rounds exactly like the model's
    crop. A pair whose centre crop holds no changed voxel is dropped and counted.
    """
    items, records, kept, dropped = [], [], [], []
    for i, volume in enumerate(volumes, 1):
        shape = crop_shape_for(store.series[volume["series_key"]]["geometry"]["spacing_zyx"], crop_mm)
        margin = tuple(2 * int(round(s * margin_fraction / 2.)) for s in shape)
        big = tuple(s + 2 * m for s, m in zip(shape, margin))
        core = tuple(slice(m, m + s) for s, m in zip(shape, margin))
        center = site_center_native(volume)
        image, inside, edit = store.crop(volume, center, big)
        reference, inside_real, _ = store.crop(unedited(volume), center, big)
        if not np.array_equal(inside, inside_real) or ((image != reference) & ~edit).any():
            raise ValueError("A pair differs outside its changed voxels: " + volume["volume_id"])
        changed = int(edit[core].sum())
        if not changed:
            dropped.append(volume["volume_id"])
            continue
        items.append({"fake": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None],
                      "real": torch.from_numpy(np.ascontiguousarray(reference, dtype=np.float32))[None],
                      "valid": torch.from_numpy(np.ascontiguousarray(inside))[None], "shape": shape, "margin": margin})
        records.append({"volume_id": volume["volume_id"], "operation": volume["operation"],
                        "generator": volume["generator"], "patient_id": volume["patient_id"],
                        "crop_shape_zyx": list(shape), "changed_voxels_in_crop": changed})
        kept.append(volume)
        if progress and (i % 50 == 0 or i == len(volumes)):
            progress({"stage": "pretrain_check_pairs", "split": split, "done": i, "of": len(volumes)})
    if not records:
        raise ValueError("No pair holds changed voxels in its crop")
    return {"items": items, "records": records, "volumes": kept, "dropped": dropped}


def _core(item, key, offset=None):
    (oz, oy, ox), (sz, sy, sx) = (item["margin"] if offset is None else offset), item["shape"]
    return item[key][..., oz:oz + sz, oy:oy + sy, ox:ox + sx]


def pair_logits(encode, items, device):
    """Tamper logits of each pair's centre crops (edited, unedited); `encode` is a crop network."""
    fake, real = [], []
    with torch.inference_mode():
        for item in items:
            images = torch.stack((_core(item, "fake"), _core(item, "real"))).to(device)
            valid = torch.stack((_core(item, "valid"),) * 2).to(device)
            logits = encode(images, valid)["tamper_logits"].float().cpu()
            fake.append(float(logits[0]))
            real.append(float(logits[1]))
    return np.asarray(fake), np.asarray(real)


def paired_intervals(fake, real, records, resamples, seed):
    """Share of pairs ordered correctly, with a patient-level 95% interval, per operation."""
    out = {}
    for operation in OPERATIONS:
        index = [i for i, r in enumerate(records) if r["operation"] == operation]
        if index:
            wins = [(fake[i] > real[i]) + .5 * (fake[i] == real[i]) for i in index]
            out[operation] = cluster_interval(wins, [records[i]["patient_id"] for i in index], resamples, seed)
    return out


# --------------------------------------------------------------------------- check 1: pretrain, transfer


def _schedule(optimizer, steps):
    """The crop check's schedule: linear warmup over 10% of steps (at most 100), then cosine to 5%."""
    warmup = max(1, min(100, steps // 10))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
        (step + 1) / warmup if step < warmup else
        .05 + .95 * .5 * (1. + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))))


def pretrain(train, validation, *, steps, batch_pairs, lr, seed, device, model_kwargs, eval_every, progress=None):
    """Train a fresh crop network on edited (1) vs unedited (0) crops; score validation pairs every `eval_every` steps."""
    torch.manual_seed(seed)
    model = FineBranch(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = _schedule(optimizer, steps)
    generator = torch.Generator().manual_seed(seed)
    order, losses, curve, scores = torch.empty(0, dtype=torch.long), [], [], None

    def evaluate(step):
        model.eval()
        fake, real = pair_logits(model, validation["items"], device)
        model.train()
        curve.append({"step": step, "train_loss": float(np.mean(losses)) if losses else None,
                      "learning_rate": optimizer.param_groups[0]["lr"],
                      "validation": pair_metrics(fake, real, validation["records"])})
        return fake, real

    scores = evaluate(0)
    model.train()
    for step in range(1, steps + 1):
        if len(order) < batch_pairs:
            order = torch.cat((order, torch.randperm(len(train["items"]), generator=generator)))
        index, order = order[:batch_pairs], order[batch_pairs:]
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for i in index.tolist():
            item = train["items"][i]
            # Both crops of a pair get the same shift and flip, so they still differ only in the edit.
            offset = [int(torch.randint(0, 2 * m + 1, (1,), generator=generator)) for m in item["margin"]]
            flip = bool(torch.rand(1, generator=generator) < .5)
            images = torch.stack((_core(item, "fake", offset), _core(item, "real", offset)))
            valid = torch.stack((_core(item, "valid", offset),) * 2)
            if flip:
                images, valid = images.flip(-1), valid.flip(-1)
            # Crops differ in shape between scans, so each pair is its own forward pass; GroupNorm
            # keeps samples independent, so this equals one batch.
            logits = model(images.to(device), valid.to(device))["tamper_logits"]
            loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.], device=device)) / len(index)
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        scheduler.step()
        losses.append(total)
        if step % eval_every == 0 or step == steps:
            scores = evaluate(step)
            losses = []
            if progress:
                last = curve[-1]["validation"]
                progress({"stage": "pretrain", "step": step, "of": steps, "train_loss": curve[-1]["train_loss"],
                          **{f"{op}_paired_accuracy": (last.get(op) or {}).get("paired_accuracy") for op in OPERATIONS}})
    model.eval()
    return model, curve, scores


def _case(store, volume, grid, crop_mm):
    """An unaugmented validation case, as evaluation builds it."""
    return VolumeCase(store, volume, flip=False, grid=grid, crop_mm=crop_mm, forced=0, random_centers=0,
                      rng=_rng(volume["volume_id"], "pretrain_check"), jitter_fraction=0.)


def transfer_check(fine_state, model_config, store, validation, reference, device, *, grid, crop_mm, tolerance):
    """Load the pretrained crop network into a fresh full model and score the same crops again.

    `loaded_weights` scores the very tensors pretraining scored; `model_data_path` cuts the
    validation crops again as the model's data path does (`VolumeCase.crops` at the site).
    """
    full = CoarseToFineRSVT(**model_config)
    full.fine.load_state_dict(fine_state)  # strict: every key and shape must match
    full.to(device).eval()
    fake, real = pair_logits(full.encode_crops, validation["items"], device)
    loaded = float(max(np.abs(fake - reference[0]).max(), np.abs(real - reference[1]).max()))
    image_difference, path_fake, path_real = 0., [], []
    with torch.inference_mode():
        for volume, item in zip(validation["volumes"], validation["items"]):
            case = _case(store, volume, grid, crop_mm)
            edited = case.crops([case.site_center])
            original = case.crops([case.site_center], source=unedited(volume))
            for crops, key in ((edited, "fake"), (original, "real")):
                expected, valid = _core(item, key), _core(item, "valid")
                if crops["image"][0].shape != expected.shape or not torch.equal(crops["valid"][0], valid):
                    image_difference = math.inf
                else:
                    image_difference = max(image_difference, float((crops["image"][0] - expected).abs().max()))
            images = torch.cat((edited["image"], original["image"])).to(device)
            valid = torch.cat((edited["valid"], original["valid"])).to(device)
            logits = full.encode_crops(images, valid)["tamper_logits"].float().cpu()
            path_fake.append(float(logits[0]))
            path_real.append(float(logits[1]))
    path_fake, path_real = np.asarray(path_fake), np.asarray(path_real)
    path = (float(max(np.abs(path_fake - reference[0]).max(), np.abs(path_real - reference[1]).max()))
            if image_difference != math.inf else math.inf)
    result = {"tolerance": tolerance,
              "loaded_weights": {"max_logit_difference": loaded, "match": loaded <= tolerance},
              "model_data_path": {"max_image_difference": image_difference, "max_logit_difference": path,
                                  "match": image_difference == 0. and path <= tolerance,
                                  "metrics": pair_metrics(path_fake, path_real, validation["records"])}}
    result["match"] = result["loaded_weights"]["match"] and result["model_data_path"]["match"]
    return result


# --------------------------------------------------------------------------- check 2: frozen crops, swap test


def train_fusion(model, store, config, *, steps, device, progress=None):
    """Train everything but the frozen crop network on training patients for `steps` logical batches.

    The full model's own data, objective and scalar replay are used, without the tail-risk
    stage. Frozen weights receive no gradient, so AdamW leaves them exactly as loaded.
    """
    train, _ = trainer.build_datasets(store, config)
    store.localize(train.rows, progress=progress)
    optimizer = build_optimizer(model, config["training"])
    scheduler = _schedule(optimizer, steps)
    weights = trainer.class_weights(config)
    curve, step, epoch = [], 0, 0
    while step < steps:
        train.set_epoch(epoch)
        for indices in trainer.epoch_batches(train, config, epoch):
            result = trainer.replay_step(model, optimizer, train, indices, device, config, weights, risk_active=False)
            scheduler.step()
            step += 1
            curve.append({"step": step, "epoch": epoch + 1, "objective": result["objective"],
                          **{"train_" + k: result["terms"][k] for k in ("volume", "removal", "router")}})
            if progress and (step % 10 == 0 or step == steps):
                progress({"stage": "fusion", "step": step, "of": steps, "objective": result["objective"],
                          "volume_loss": result["terms"]["volume"]})
            if step >= steps:
                break
        epoch += 1
    model.eval()
    return curve


def swap_test(model, store, volumes, device, *, grid, crop_mm, progress=None, label=""):
    """Score each edited scan twice: one crop slot holds the edited crop, or the same crop of the unedited scan.

    Context (the coarse view of the edited scan) and the other K-1 crops are identical in both
    conditions. Those crops come from the router, excluding cells that hold changed voxels, so
    the slot is the only place the two conditions differ.
    """
    model.eval()
    rows = []
    with torch.inference_mode():
        for i, volume in enumerate(volumes, 1):
            case = _case(store, volume, grid, crop_mm)
            coarse = model.encode_coarse(torch.from_numpy(case.coarse)[None, None].to(device))
            cells = model.select_cells(coarse["router_logits"], torch.from_numpy(case.cell_valid).to(device),
                                       model.crops_at_inference - 1,
                                       exclude=torch.from_numpy(case.cell_target > 0).to(device)).cpu().numpy().tolist()
            others = case.crops([case.cell_centers[c] for c in cells]) if cells else None
            encoded = model.encode_crops(others["image"].to(device), others["valid"].to(device)) if others else None
            row = {"volume_id": volume["volume_id"], "operation": volume["operation"], "generator": volume["generator"],
                   "patient_id": volume["patient_id"], "other_crops": len(cells),
                   "other_crops_holding_edit": int(others["tamper"].sum()) if others else 0}
            for condition, source in (("edited", None), ("unedited", unedited(volume))):
                site = case.crops([case.site_center], source=source)
                fine = model.encode_crops(site["image"].to(device), site["valid"].to(device))
                positions = site["positions"]
                if encoded is not None:
                    fine = {k: torch.cat((encoded[k], fine[k])) for k in ("tokens", "tamper_logits")}
                    positions = torch.cat((others["positions"], positions))
                fused = model.fuse(coarse, fine, positions.to(device))
                row[condition] = {"scan_logit": float(fused["tamper_logit"]),
                                  "slot_crop_logit": float(fine["tamper_logits"][-1]),
                                  "max_crop_logit": float(fine["tamper_logits"].max())}
            rows.append(row)
            if progress and (i % 50 == 0 or i == len(volumes)):
                progress({"stage": "swap_test", "when": label, "done": i, "of": len(volumes)})
    return rows


SWAP_SCORES = ("scan_logit", "max_crop_logit", "slot_crop_logit")


def swap_metrics(rows, *, resamples=2000, seed=13):
    """Paired change (edited minus unedited slot) of each score, per operation, with patient-level intervals."""
    out = {}
    for group in ("all",) + OPERATIONS:
        members = [r for r in rows if group == "all" or r["operation"] == group]
        if not members:
            out[group] = None
            continue
        patients = [r["patient_id"] for r in members]
        entry = {"scans": len(members), "patients": len(set(patients)),
                 "scans_with_edit_in_other_crops": sum(r["other_crops_holding_edit"] > 0 for r in members)}
        for key in SWAP_SCORES:
            change = np.asarray([r["edited"][key] - r["unedited"][key] for r in members])
            wins = (change > 0) + .5 * (change == 0)
            entry[key] = {"paired_accuracy": float(wins.mean()),
                          "paired_accuracy_ci95": cluster_interval(wins, patients, resamples, seed),
                          "mean_change": float(change.mean()),
                          "mean_change_ci95": cluster_interval(change, patients, resamples, seed),
                          "median_change": float(np.median(change))}
        out[group] = entry
    return out


def responds(metrics, key="scan_logit"):
    """Operations whose score rises with the edited crop: the patient-level interval of the share lies above 50%."""
    return [op for op in OPERATIONS if metrics.get(op) and metrics[op][key]["paired_accuracy_ci95"][0] > .5]


def passed(result):
    """Did a saved check reach "proceed with run 3" (the transfer matched and the scan score responds for both operations)?"""
    swap = result.get("swap_test") or {}
    after = swap.get("after_training")
    return bool((result.get("transfer") or {}).get("match") and (result.get("pretraining") or {}).get("verdict")
                in ("learned", "partly") and after and swap.get("crop_network_unchanged")
                and len(responds(after)) == len(OPERATIONS))


def interpret(transfer, crop_verdict, swap):
    if not transfer["match"]:
        return ("The pretrained crop network scores differently once loaded into the full model (or the model's data "
                "path cuts different crops). That is an implementation error: fix it before any training run.")
    if crop_verdict not in ("learned", "partly"):
        return ("Pretraining on the model's own crops did not reproduce the crop learning check ("
                + crop_verdict.replace("_", " ") + " on unseen patients). Look at the crop recipe before another full run.")
    after = swap["after_training"]
    scan, yardstick = responds(after), responds(after, "max_crop_logit")
    if len(scan) == len(OPERATIONS):
        return ("The crop network survives the transfer, and after brief training the scan classifier raises its score "
                "when the edited crop replaces the unedited one, for removal and insertion, on unseen patients. "
                "Proceed with run 3, changing only the crop network's starting weights.")
    missing = [op for op in OPERATIONS if op not in scan]
    text = ("The crop network survives the transfer, but after brief training the scan score does not respond to the "
            "edited crop for " + " and ".join(missing) + " on unseen patients")
    lost = [op for op in missing if op in yardstick]
    if lost:
        text += (", although the largest crop score does for " + " and ".join(lost)
                 + ": the scan classifier is not using evidence the crops carry")
    return text + ". Investigate how the full model combines crop evidence before another full run."


# --------------------------------------------------------------------------- run


def pretrain_check(store, config, output_dir, *, device="cuda", per_operation_train=192, per_operation_validation=96,
                   steps=2000, batch_pairs=8, lr=1e-3, margin_fraction=.125, eval_every=200, fusion_steps=300,
                   tolerance=1e-3, resamples=2000, progress=None):
    t, d = config["training"], config["data"]
    seed = t["seed"]
    seed_everything(seed, t["threads"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generators = sorted(d["generators"])
    train_volumes = select_pairs(store, "train", generators, per_operation_train, seed)
    validation_volumes = select_pairs(store, "validation", generators, per_operation_validation, seed)
    if {v["patient_id"] for v in train_volumes} & {v["patient_id"] for v in validation_volumes}:
        raise ValueError("A patient appears in both the training and the validation pairs")
    train = site_pairs(store, train_volumes, d["crop_mm"], margin_fraction, progress, "train")
    validation = site_pairs(store, validation_volumes, d["crop_mm"], margin_fraction, progress, "validation")
    kwargs = fine_kwargs(config["model"])

    # Check 1: pretrain the crop network alone, save it, and score the same crops inside a fresh full model.
    branch, curve, reference = pretrain(train, validation, steps=steps, batch_pairs=batch_pairs, lr=lr, seed=seed,
                                        device=device, model_kwargs=kwargs, eval_every=eval_every, progress=progress)
    fine_state = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
    contract = digest_json({"schema": SCHEMA, "config_sha256": digest_json(config),
                            "implementation_sha256": trainer.implementation_fingerprint(),
                            "train_pairs": [r["volume_id"] for r in train["records"]]})
    save_checkpoint(output / WEIGHTS, {"format_version": WEIGHTS_FORMAT, "contract_sha256": contract, "epoch": steps,
                                       "model_kwargs": {k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items()},
                                       "fine": fine_state})
    transfer = transfer_check(fine_state, config["model"], store, validation, reference, device,
                              grid=d["grid"], crop_mm=d["crop_mm"], tolerance=tolerance)
    if progress:
        progress({"stage": "transfer", "match": transfer["match"],
                  "max_logit_difference": transfer["loaded_weights"]["max_logit_difference"]})
    final = curve[-1]["validation"]
    crop_verdict = verdict(final)

    # Check 2: freeze the crop network, train the rest briefly, and swap the edit's crop for the unedited one.
    # It only means something when the crop network carries the edit and keeps it inside the model.
    swap, skipped = None, None
    if not transfer["match"]:
        skipped = "the transfer did not match"
    elif crop_verdict not in ("learned", "partly"):
        skipped = "pretraining did not learn the edit on unseen patients"
    else:
        seed_everything(seed, t["threads"])
        model = CoarseToFineRSVT(**config["model"])
        model.fine.load_state_dict(fine_state)
        for parameter in model.fine.parameters():
            parameter.requires_grad_(False)
        model.to(device)
        common = dict(grid=d["grid"], crop_mm=d["crop_mm"], progress=progress)
        before = swap_test(model, store, validation["volumes"], device, label="before training", **common)
        fusion = train_fusion(model, store, config, steps=fusion_steps, device=device, progress=progress)
        after = swap_test(model, store, validation["volumes"], device, label="after training", **common)
        if any(not torch.equal(v.detach().cpu(), fine_state[k]) for k, v in model.fine.state_dict().items()):
            raise ValueError("The frozen crop network changed during scan-classifier training")
        swap = {"fusion_steps": fusion_steps, "fusion_curve": fusion, "crop_network_unchanged": True,
                "before_training": swap_metrics(before, resamples=resamples, seed=seed),
                "after_training": swap_metrics(after, resamples=resamples, seed=seed), "rows_after_training": after}

    result = {
        "schema_version": SCHEMA,
        "questions": ["does the crop network learn the edit from the model's own crops, and keep its scores inside the "
                      "full model?", "does a briefly trained scan classifier use that crop evidence on unseen patients?"],
        "design": ("check 1: edited crop at the model's site-crop centre vs the same crop of the unedited scan, each "
                   "scan's own ~crop_mm shape, residual inside the network, removals and insertions balanced, validation "
                   "patients unseen; check 2: crop network frozen, the rest trained briefly on training patients, then "
                   "each validation edit scored with one crop slot swapped between edited and unedited, identical "
                   "context and crop count; the unedited crop is used only in this test"),
        "settings": {"steps": steps, "batch_pairs": batch_pairs, "learning_rate": lr, "eval_every": eval_every,
                     "schedule": "warmup then cosine to 5%", "margin_fraction": margin_fraction,
                     "augmentation": "same random shift and left-right flip for both crops of a pair",
                     "crop_mm": d["crop_mm"], "model": {k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items()},
                     "fusion_steps": fusion_steps, "fusion_learning_rate": t["learning_rate"],
                     "crops_per_scan_in_swap_test": config["model"].get("crops_at_inference", 16),
                     "tolerance": tolerance, "bootstrap_resamples": resamples, "seed": seed, "device": str(device),
                     "generators": generators},
        "pairs": {"train": dict(Counter(f"{r['generator']}:{r['operation']}" for r in train["records"])),
                  "validation": dict(Counter(f"{r['generator']}:{r['operation']}" for r in validation["records"])),
                  "dropped_without_edit_in_crop": train["dropped"] + validation["dropped"],
                  "train_patients": len({r["patient_id"] for r in train["records"]}),
                  "validation_patients": len({r["patient_id"] for r in validation["records"]}),
                  "crop_shapes": dict(Counter("x".join(map(str, r["crop_shape_zyx"]))
                                              for r in train["records"] + validation["records"]))},
        "pretraining": {"curve": curve, "final": final, "verdict": crop_verdict,
                        "paired_accuracy_ci95": paired_intervals(reference[0], reference[1], validation["records"],
                                                                 resamples, seed)},
        "weights": {"file": WEIGHTS, "format": WEIGHTS_FORMAT, "contract_sha256": contract},
        "transfer": transfer,
        "swap_test": swap if swap is not None else {"skipped": skipped},
        "interpretation": interpret(transfer, crop_verdict, swap),
        "scope": ("exploratory; the edit's location is given in both checks, so they say nothing about routing; the "
                  "unedited crop is a diagnostic reference never seen in training"),
        "validation_pairs": validation["records"],
        "test_volumes_opened": 0}
    write_json(output / "pretrain_check.json", result)
    return result
