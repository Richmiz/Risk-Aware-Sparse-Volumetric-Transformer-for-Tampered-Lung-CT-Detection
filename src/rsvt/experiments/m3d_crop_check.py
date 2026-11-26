"""Crop learning check: can the fine crop network learn the edit when every crop contains it?

The first full training run (26 Nov 2026) never moved its tamper losses off chance, even on
training volumes. This check removes the router, the transformer and the other heads and asks
the crop network alone.

Pairs: one manipulated sample's crop at the author manipulation centre and the same crop from
its real LIDC series, so the two differ only in the changed voxels (checked for every pair).
Removals and insertions are balanced. The real crop of a removal holds a nodule and that of an
insertion does not, so "is there a nodule here" cannot separate the classes; only how the
edited tissue looks can. Per-operation results expose a nodule-presence shortcut: it would
score one operation well and the other badly.

Arms, each with fresh weights of the model's own fine branch:

* fit: 32 training pairs, no augmentation. Can the network and optimizer fit them at all?
* image: training pairs, tested on unseen validation patients, on the windowed image the
  model sees.
* residual: the same on the image minus its 3x3x3 local mean, the usual forensic pre-filter
  that suppresses anatomy and keeps fine texture and seams.

Train/validation only; no test volume is opened. Exploratory: the thresholds in `interpret`
describe the result; they are not pre-registered endpoints.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from rsvt.data.m3dsynth_dataset import _rng
from rsvt.data.m3dsynth_store import write_json
from rsvt.evaluation.classification import binary_auroc
from rsvt.models.coarse_to_fine import FineBranch

SCHEMA = "m3d_crop_check_v1"
OPERATIONS = ("removal", "insertion")
INPUTS = ("image", "residual")


def select_pairs(store, split, generators, per_operation, seed):
    """Eligible manipulated volumes, `per_operation` per operation (0 = all), round-robin over generators.

    The order is fixed by the seed, and a smaller count is a prefix of a larger one.
    """
    volumes = [v for v in store.manifest["volumes"] if v["split"] == split and v["kind"] == "manipulated"
               and v["eligible"] and v["generator"] in generators]
    chosen = []
    for operation in OPERATIONS:
        queues = []
        for generator in sorted(generators):
            members = sorted((v for v in volumes if v["operation"] == operation and v["generator"] == generator),
                             key=lambda v: v["volume_id"])
            order = _rng(seed, split, operation, generator, "crop_check").permutation(len(members))
            queues.append([members[i] for i in order])
        picked = []
        while any(queues) and (not per_operation or len(picked) < per_operation):
            for queue in queues:
                if queue and (not per_operation or len(picked) < per_operation):
                    picked.append(queue.pop(0))
        chosen += picked
    return chosen


def extract_pairs(store, volumes, shape, margin, progress=None):
    """Manipulated and real crops (shape + 2*margin) at each manipulation centre, with records.

    The larger crop leaves room for random shifts in training; evaluation uses its centre.
    A pair whose centre crop holds no changed voxel is dropped and counted.
    """
    big = tuple(int(s + 2 * m) for s, m in zip(shape, margin))
    core = tuple(slice(m, m + s) for s, m in zip(shape, margin))
    fake, real, valid, records, dropped = [], [], [], [], []
    for i, volume in enumerate(volumes, 1):
        center = np.asarray(volume["center_zyx"], dtype=float)
        image, inside, edit = store.crop(volume, center, big)
        source = {"kind": "authentic", "series_key": volume["series_key"], "volume_id": "real:" + volume["series_key"]}
        reference, inside_real, _ = store.crop(source, center, big)
        difference = image != reference
        if not np.array_equal(inside, inside_real) or (difference & ~edit).any():
            raise ValueError("A pair differs outside its changed voxels: " + volume["volume_id"])
        changed = int(edit[core].sum())
        if not changed:
            dropped.append(volume["volume_id"])
            continue
        records.append({"volume_id": volume["volume_id"], "operation": volume["operation"],
                        "generator": volume["generator"], "patient_id": volume["patient_id"],
                        "changed_voxels_in_crop": changed,
                        # changed voxels the HU window still shows (both values clipped to the same end hide them)
                        "edit_visible_after_window": float((difference & edit)[core].sum() / changed)})
        fake.append(image.astype(np.float32))
        real.append(reference.astype(np.float32))
        valid.append(inside)
        if progress and (i % 50 == 0 or i == len(volumes)):
            progress({"stage": "crop_check_pairs", "done": i, "of": len(volumes)})
    if not records:
        raise ValueError("No pair holds changed voxels in its crop")
    tensor = lambda items: torch.from_numpy(np.stack(items))[:, None]
    return {"fake": tensor(fake), "real": tensor(real), "valid": tensor(valid), "records": records, "dropped": dropped}


def residual(images):
    """Image minus its 3x3x3 local mean: suppresses anatomy, keeps fine texture and seams."""
    return images - F.avg_pool3d(images, 3, stride=1, padding=1, count_include_pad=False)


def _inputs(images, kind):
    if kind == "image":
        return images
    if kind == "residual":
        return residual(images)
    raise ValueError("Unknown input: " + str(kind))


def _crop(tensor, offset, shape):
    return tensor[..., offset[0]:offset[0] + shape[0], offset[1]:offset[1] + shape[1], offset[2]:offset[2] + shape[2]]


def scores(model, data, kind, shape, margin, device, batch=16):
    """Tamper logits of the centre crops (logits keep their order where probabilities saturate)."""
    was_training = model.training
    model.eval()
    out = {"fake": [], "real": []}
    with torch.inference_mode():
        for start in range(0, len(data["valid"]), batch):
            valid = _crop(data["valid"][start:start + batch].to(device), margin, shape)
            for key in out:
                x = _crop(_inputs(data[key][start:start + batch].to(device), kind), margin, shape)
                out[key].append(model(x, valid)["tamper_logits"].float().cpu())
    model.train(was_training)
    return torch.cat(out["fake"]).numpy().astype(float), torch.cat(out["real"]).numpy().astype(float)


def pair_metrics(fake, real, records):
    """Paired accuracy (manipulated crop scored above its own real crop; ties count half), crop AUROC and BCE."""
    fake, real = np.asarray(fake, dtype=float), np.asarray(real, dtype=float)

    def summary(index):
        if not index:
            return None
        f, r = fake[index], real[index]
        bce = (np.logaddexp(0., -f).sum() + np.logaddexp(0., r).sum()) / (2 * len(index))
        return {"pairs": len(index), "paired_accuracy": float(np.mean((f > r) + .5 * (f == r))),
                "crop_auroc": binary_auroc([1] * len(f) + [0] * len(r), np.concatenate((f, r))), "bce": float(bce)}

    groups = {"all": list(range(len(records)))}
    for operation in OPERATIONS:
        groups[operation] = [i for i, r in enumerate(records) if r["operation"] == operation]
    for name in sorted({f"{r['generator']}:{r['operation']}" for r in records}):
        groups[name] = [i for i, r in enumerate(records) if f"{r['generator']}:{r['operation']}" == name]
    return {name: summary(index) for name, index in groups.items()}


def subset(data, index):
    index = list(index)
    return {"fake": data["fake"][index], "real": data["real"][index], "valid": data["valid"][index],
            "records": [data["records"][i] for i in index]}


def train_arm(train, evaluations, *, kind, steps, batch_pairs, lr, seed, device, shape, margin, augment,
              model_kwargs, eval_every, name="", progress=None):
    """Train a fresh fine branch on manipulated (1) vs real (0) crops; evaluate every `eval_every` steps."""
    torch.manual_seed(seed)
    model = FineBranch(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, min(100, steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
        (step + 1) / warmup if step < warmup else
        .05 + .95 * .5 * (1. + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))))
    generator = torch.Generator().manual_seed(seed)
    count = len(train["valid"])
    order, losses = torch.empty(0, dtype=torch.long), []

    def evaluate(step):
        point = {"step": step, "train_loss": float(np.mean(losses)) if losses else None,
                 "learning_rate": optimizer.param_groups[0]["lr"]}
        for split, data in evaluations.items():
            fake, real = scores(model, data, kind, shape, margin, device)
            point[split] = pair_metrics(fake, real, data["records"])
        return point

    curve = [evaluate(0)]
    model.train()
    for step in range(1, steps + 1):
        if len(order) < batch_pairs:
            order = torch.cat((order, torch.randperm(count, generator=generator)))
        index, order = order[:batch_pairs], order[batch_pairs:]
        images = _inputs(torch.cat((train["fake"][index], train["real"][index])).to(device), kind)
        valid = torch.cat((train["valid"][index], train["valid"][index])).to(device)
        pieces, masks = [None] * (2 * len(index)), [None] * (2 * len(index))
        for i in range(len(index)):
            # Both members of a pair get the same shift and flip, so they still differ only in the edit.
            if augment:
                offset = [int(torch.randint(0, 2 * m + 1, (1,), generator=generator)) for m in margin]
                flip = bool(torch.rand(1, generator=generator) < .5)
            else:
                offset, flip = list(margin), False
            for j in (i, i + len(index)):
                x, v = _crop(images[j], offset, shape), _crop(valid[j], offset, shape)
                pieces[j], masks[j] = (x.flip(-1), v.flip(-1)) if flip else (x, v)
        target = torch.cat((torch.ones(len(index)), torch.zeros(len(index)))).to(device)
        loss = F.binary_cross_entropy_with_logits(model(torch.stack(pieces), torch.stack(masks))["tamper_logits"], target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if step % eval_every == 0 or step == steps:
            curve.append(evaluate(step))
            losses = []
            if progress:
                last = curve[-1]
                progress({"stage": "crop_check", "arm": name, "step": step, "of": steps, "train_loss": last["train_loss"],
                          **{f"{split}_paired_accuracy": last[split]["all"]["paired_accuracy"] for split in evaluations}})
    return {"input": kind, "steps": steps, "augment": augment, "pairs": count, "curve": curve, "final": curve[-1]}


def verdict(metrics):
    """Describe validation results on one input: learned, nodule-presence shortcut, partly, or not learned."""
    removal, insertion = (metrics[op]["paired_accuracy"] if metrics.get(op) else None for op in OPERATIONS)
    if removal is None or insertion is None:
        return "incomplete"
    if removal >= .8 and insertion >= .8:
        return "learned"
    if abs(removal - insertion) >= .5:
        return "nodule_presence"
    if metrics["all"]["paired_accuracy"] >= .6:
        return "partly"
    return "not_learned"


def interpret(fit, verdicts):
    fits = fit["bce"] < .1 and fit["paired_accuracy"] == 1.
    if not fits:
        return ("The crop network could not fit 32 training pairs, so the input, the network or the optimizer fails "
                "before any question of data. Fix that first.")
    if verdicts["image"] == "learned":
        return ("The crop network learns the edit from crops that contain it, on unseen patients. The full model's "
                "failure lies upstream: crop selection, the multi-task objective or its schedule.")
    if verdicts["residual"] == "learned":
        return ("The edit is learnable once image content is suppressed: give the fine branch a residual input, "
                "then train the full model again.")
    shortcut = [kind for kind, v in verdicts.items() if v == "nodule_presence"]
    text = ("The crop network fits training pairs but does not learn the edit for unseen patients on either input"
            + (" (partly on " + " and ".join(k for k, v in verdicts.items() if v == "partly") + ")"
               if "partly" in verdicts.values() else "") + ". ")
    if shortcut:
        text += "On " + " and ".join(shortcut) + " it scores one operation well and the other badly: it uses nodule presence. "
    return text + "A stronger forensic crop encoder or more training pairs are needed before the coarse-to-fine design can work."


def _medians(records, key):
    out = {}
    for operation in OPERATIONS:
        values = [r[key] for r in records if r["operation"] == operation and r[key] is not None]
        out[operation] = float(np.median(values)) if values else None
    return out


def crop_check(store, generators, output_dir, *, device="cuda", seed=13, per_operation_train=192,
               per_operation_validation=96, fit_per_operation=16, steps=2000, fit_steps=600, batch_pairs=8, lr=1e-3,
               shape=(24, 64, 64), margin=(4, 8, 8), eval_every=200, model_kwargs=None, progress=None):
    model_kwargs = dict(model_kwargs or {"dimension": 128, "widths": (16, 32, 64)})
    generators = sorted(generators)
    train_volumes = select_pairs(store, "train", generators, per_operation_train, seed)
    validation_volumes = select_pairs(store, "validation", generators, per_operation_validation, seed)
    if {v["patient_id"] for v in train_volumes} & {v["patient_id"] for v in validation_volumes}:
        raise ValueError("A patient appears in both the training and the validation pairs")
    train = extract_pairs(store, train_volumes, shape, margin, progress)
    validation = extract_pairs(store, validation_volumes, shape, margin, progress)
    fit_index = [i for op in OPERATIONS for i in [j for j, r in enumerate(train["records"]) if r["operation"] == op][:fit_per_operation]]
    fit_data = subset(train, fit_index)
    sample = subset(train, [i for op in OPERATIONS
                            for i in [j for j, r in enumerate(train["records"]) if r["operation"] == op][:48]])
    common = dict(batch_pairs=batch_pairs, lr=lr, seed=seed, device=device, shape=shape, margin=margin,
                  model_kwargs=model_kwargs, progress=progress)
    arms = {"fit": train_arm(fit_data, {"train": fit_data}, kind="image", steps=fit_steps, augment=False,
                             eval_every=max(1, fit_steps // 6), name="fit", **common)}
    for kind in INPUTS:
        arms[kind] = train_arm(train, {"validation": validation, "train_sample": sample}, kind=kind, steps=steps,
                               augment=True, eval_every=eval_every, name=kind, **common)
    verdicts = {kind: verdict(arms[kind]["final"]["validation"]) for kind in INPUTS}
    records = train["records"] + validation["records"]
    result = {
        "schema_version": SCHEMA,
        "question": "can the model's fine crop network learn the edit when every crop contains it?",
        "design": ("manipulated crop at the author manipulation centre vs the same crop of its real series (they differ "
                   "only in the changed voxels); removals and insertions balanced, so nodule presence cannot separate "
                   "the classes; fresh fine branch per arm; validation patients unseen in training"),
        "settings": {"crop_shape_zyx": list(shape), "shift_margin_zyx": list(margin), "steps": steps, "fit_steps": fit_steps,
                     "batch_pairs": batch_pairs, "learning_rate": lr, "schedule": "warmup then cosine to 5%",
                     "augmentation": "same random shift and left-right flip for both crops of a pair",
                     "model": {k: list(v) if isinstance(v, tuple) else v for k, v in model_kwargs.items()},
                     "seed": seed, "device": str(device), "generators": generators},
        "pairs": {"train": dict(Counter(f"{r['generator']}:{r['operation']}" for r in train["records"])),
                  "validation": dict(Counter(f"{r['generator']}:{r['operation']}" for r in validation["records"])),
                  "fit": len(fit_index), "dropped_without_edit_in_crop": train["dropped"] + validation["dropped"],
                  "train_patients": len({r["patient_id"] for r in train["records"]}),
                  "validation_patients": len({r["patient_id"] for r in validation["records"]})},
        "edit_in_crop": {"changed_voxels_median": _medians(records, "changed_voxels_in_crop"),
                         "visible_after_window_median": _medians(records, "edit_visible_after_window")},
        "arms": arms,
        "verdicts": verdicts,
        "interpretation": interpret(arms["fit"]["final"]["train"]["all"], verdicts),
        "scope": ("exploratory; the edit's location is given, so this bounds what the crop stage can do and says "
                  "nothing about routing"),
        "records": {"validation": validation["records"]},
        "test_volumes_opened": 0}
    write_json(Path(output_dir) / "crop_check.json", result)
    return result
