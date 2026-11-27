"""Training and evaluation of coarse-to-fine RSVT on a verified M3Dsynth store.

Logical batches mix authentic and manipulated volumes (stratified by generator and
operation). Tail-risk terms need the whole batch, but one GPU holds one volume, so
the objective is differentiated in per-case scalar space and each case is replayed
with its recorded RNG state and recorded crop selection (scalar replay, as in the
earlier RSVT trainer). Crop selection is non-differentiable and reused exactly.

26 Nov 2026 (after the first run and the crop learning check):
* Checkpoint selection maximises the mean of removal-vs-authentic and
  insertion-vs-authentic validation AUROC, starting from chance (0.5). The earlier
  rule (group-balanced validation BCE) let a constant model win, since the best
  constant scores 0.637 there; a constant scores 0.5 here and can never be selected.
* Validation also scores each scan with one extra crop at the edit (or, for an
  authentic scan, at one of its real sites). This diagnostic separates what routing
  loses from what the crop stage misses; it is never used for selection.

27 Nov 2026 (after the crop pretraining check passed): run 3 changes only the crop
network's starting weights. `training.fine_init_sha256` names the weights a crop
pretraining check saved; `run_training(pretrained=<that check's folder>)` verifies them
and loads them into the crop network before the first step (a resumed run keeps its own
weights). Every epoch also scores that check's fixed validation pairs (edited crop vs
the same crop of the unedited scan), to show whether joint training erodes what the crop
network learned. Diagnostic only, never used for selection.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import importlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from rsvt.data.m3dsynth_dataset import M3DDataset, M3DStore, _rng, crop_shape_for
from rsvt.data.m3dsynth_store import digest_json, sha256_path, write_json
from rsvt.evaluation.classification import binary_auroc
from rsvt.losses.endpoint_risk import dice_bce, generator_variance, grouped_tail_risk
from rsvt.models.coarse_to_fine import CoarseToFineRSVT, compute_estimate
from rsvt.training.checkpoint import load_checkpoint, save_checkpoint
from rsvt.training.optimization import build_optimizer, build_scheduler, optimization_report, validate_optimization
from rsvt.utils.config import load_config
from rsvt.utils.reproducibility import restore_rng, rng_state, seed_everything

TERMS = ("volume", "removal", "crop_tamper", "crop_visible", "mask", "router")
TABLES = ("data", "acquisition", "model", "training", "loss", "risk", "evaluation")
FORMAT = "rsvt_c2f_state_v2"  # v2: `best` is the selection AUROC (higher is better), not a BCE
SELECTION_RULE = ("highest mean of removal-vs-authentic and insertion-vs-authentic validation AUROC among "
                  "risk-stage epochs, above chance (0.5) by at least min_delta (hard top-K inference)")
SITE_CHECK_SCOPE = ("diagnostic only, never used for selection: each validation scan is scored again with one extra "
                    "crop at the author manipulation centre (manipulated) or at one of its series' real TB/TM sites "
                    "(authentic). The gap to router-only scores is what routing loses.")


# --------------------------------------------------------------------------- config


def _positive(value, name, *, zero=False, integer=False):
    kinds = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds) or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError("Invalid " + name)
    return value


def validate_config(config):
    if set(config) != set(TABLES):
        raise ValueError("Config needs tables: " + ", ".join(TABLES))
    t, loss, risk, data = config["training"], config["loss"], config["risk"], config["data"]
    if t.get("mode") != "engineering_pilot":
        raise ValueError("Explicit exploratory training mode required")
    for key in ("epochs", "threads", "authentic_per_batch", "manipulated_per_batch", "router_crops_training"):
        _positive(t[key], key, integer=True)
    for key in ("seed", "max_batches_per_epoch", "risk_start_epoch", "patience"):
        _positive(t[key], key, zero=True, integer=True)
    for key in ("learning_rate", "gradient_clip"):
        _positive(t[key], key)
    _positive(t["weight_decay"], "weight_decay", zero=True)
    _positive(t["min_delta"], "min_delta", zero=True)
    if t["device"] not in ("cpu", "cuda") or t["risk_start_epoch"] >= t["epochs"]:
        raise ValueError("Explicit device and a risk stage before the last epoch required")
    init = t.get("fine_init_sha256", "")
    if not isinstance(init, str) or (init and (len(init) != 64 or set(init) - set("0123456789abcdef"))):
        raise ValueError("fine_init_sha256 must be empty (random start) or the SHA-256 of pretrained crop weights")
    validate_optimization(t)
    for key in TERMS:
        _positive(loss[key], "loss." + key, zero=True)
    _positive(loss["router_pos_weight"], "router_pos_weight")
    if not 0 <= risk["alpha"] < 1:
        raise ValueError("CVaR alpha must be in [0,1)")
    for key in ("cvar_weight", "authentic_weight", "fb_weight", "fm_weight", "generator_weight"):
        _positive(risk[key], "risk." + key, zero=True)
    _positive(risk["min_generator_samples"], "min_generator_samples", integer=True)
    if not data["generators"] or set(data["generators"]) - {"cycle", "pix2pix", "diffusion"}:
        raise ValueError("Unknown training generators")
    if not all(0 < f < 1 for f in config["evaluation"]["fpr_targets"]):
        raise ValueError("FPR targets must be in (0,1)")
    CoarseToFineRSVT(**config["model"])
    return config


def load_c2f_config(path):
    return validate_config(load_config(path))


def implementation_fingerprint():
    names = ("rsvt.training.coarse_to_fine", "rsvt.models.coarse_to_fine", "rsvt.data.m3dsynth_dataset",
             "rsvt.data.m3dsynth_store", "rsvt.losses.endpoint_risk", "rsvt.losses.cvar", "rsvt.training.checkpoint",
             "rsvt.training.optimization", "rsvt.evaluation.classification", "rsvt.utils.reproducibility",
             "rsvt.experiments.m3d_pretrain_check", "rsvt.experiments.m3d_crop_check")  # pretrained start, fixed pairs
    return digest_json({n: sha256_path(Path(importlib.import_module(n).__file__)) for n in names})


# --------------------------------------------------------------------------- one case


def _group(case):
    if not case.tampered:
        return "authentic"
    return "FB" if case.removal == 1. else "FM"


def case_terms(model, case, device, config, *, train, selection=None):
    """Forward one volume. Returns scalar loss terms, outputs, and the crop selection."""
    loss_cfg = config["loss"]
    coarse = model.encode_coarse(torch.from_numpy(case.coarse)[None, None].to(device))
    valid_cells = torch.from_numpy(case.cell_valid).to(device)
    if selection is None:
        k = config["training"]["router_crops_training"] if train else model.crops_at_inference
        cells = model.select_cells(coarse["router_logits"], valid_cells, k).cpu().numpy().tolist()
    else:
        cells = selection["cells"]
    centers = [case.cell_centers[c] for c in cells] + [c for c, _ in case.forced] + list(case.random)
    visible = [-1.] * len(cells) + [v for _, v in case.forced] + [-1.] * len(case.random)
    if not centers:
        raise ValueError("No crop selected; the coarse view has no valid cells")
    crops = case.crops(centers)
    fine = model.encode_crops(crops["image"].to(device), crops["valid"].to(device))
    fused = model.fuse(coarse, fine, crops["positions"].to(device))
    zero = fused["tamper_logit"] * 0.
    tampered = torch.tensor(case.tampered, device=device)
    terms = {"volume": F.binary_cross_entropy_with_logits(fused["tamper_logit"], tampered)}
    terms["removal"] = (F.binary_cross_entropy_with_logits(fused["removal_logit"], torch.tensor(case.removal, device=device))
                        if case.tampered else zero)
    terms["crop_tamper"] = F.binary_cross_entropy_with_logits(fine["tamper_logits"], crops["tamper"].to(device))
    visible_t = torch.tensor(visible, device=device)
    labelled = visible_t >= 0
    terms["crop_visible"] = (F.binary_cross_entropy_with_logits(fine["visible_logits"][labelled], visible_t[labelled])
                             if bool(labelled.any()) else zero)
    with torch.no_grad():
        target = F.max_pool3d(crops["mask"].to(device), 2)
        support = F.max_pool3d(crops["valid"].float().to(device), 2)
    terms["mask"] = dice_bce(fine["mask_logits"], target, support).mean()
    cell_target = torch.from_numpy(case.cell_target).to(device)
    router = F.binary_cross_entropy_with_logits(coarse["router_logits"], cell_target, reduction="none",
                                                pos_weight=torch.tensor(loss_cfg["router_pos_weight"], device=device))
    terms["router"] = router[torch.from_numpy(case.router_cells).to(device)].mean()
    outputs = {}
    if not train:
        with torch.no_grad():
            selected_targets = case.cell_target[cells] if cells else np.zeros(0)
            outputs = {"tamper_probability": float(torch.sigmoid(fused["tamper_logit"])),
                       "removal_probability": float(torch.sigmoid(fused["removal_logit"])),
                       "max_crop_probability": float(torch.sigmoid(fine["tamper_logits"]).max()),
                       "router_hit": bool(selected_targets.any()) if case.tampered else None,
                       "selected_cells": len(cells)}
            if case.tampered and selected_targets.any():
                index = torch.nonzero(torch.from_numpy(selected_targets > 0)).flatten().to(device)
                predicted = (torch.sigmoid(fine["mask_logits"][index]) > .5).float()
                truth = target[index]
                outputs["mask_dice_selected"] = float((2 * (predicted * truth).sum() + 1e-6)
                                                      / (predicted.sum() + truth.sum() + 1e-6))
            check = getattr(case, "check_center", None)
            if check is not None:
                # Diagnostic: the same scan with the edit's crop (or a real site's) added to the router's crops.
                site = case.crops([check])
                site_fine = model.encode_crops(site["image"].to(device), site["valid"].to(device))
                joined = {k: torch.cat((fine[k], site_fine[k])) for k in ("tokens", "tamper_logits")}
                with_site = model.fuse(coarse, joined, torch.cat((crops["positions"], site["positions"])).to(device))
                outputs.update(tamper_probability_with_site=float(torch.sigmoid(with_site["tamper_logit"])),
                               site_crop_probability=float(torch.sigmoid(site_fine["tamper_logits"][0])),
                               site_source=case.check_source)
    meta = {"group": _group(case), "generator": case.volume["generator"], "tampered": case.tampered,
            "has_removal": bool(case.tampered), "has_visible": bool(labelled.any())}
    return terms, outputs, {"cells": cells}, meta


# --------------------------------------------------------------------------- objective


def class_weights(config):
    """Per-case volume-loss weights giving both classes equal total weight in a logical batch.

    `epoch_batches` already fixes each batch at `authentic_per_batch` authentic and
    `manipulated_per_batch` manipulated volumes, so the weights follow that composition.
    Dataset-level counts must not be applied on top: with 200 authentic and 2,072
    manipulated training volumes and 4+12 batches they gave the authentic class 78%
    of the volume loss (review of 24 Nov 2026).
    """
    t = config["training"]
    total = t["authentic_per_batch"] + t["manipulated_per_batch"]
    return {"authentic": total / (2. * t["authentic_per_batch"]),
            "manipulated": total / (2. * t["manipulated_per_batch"])}


def objective(probes, metas, config, weights, *, risk_active):
    loss_cfg, risk = config["loss"], config["risk"]
    device = probes["volume"].device
    w = torch.tensor([weights["manipulated" if m["tampered"] else "authentic"] for m in metas], device=device)
    parts = {"volume": (w * probes["volume"]).sum() / w.sum()}

    def masked_mean(values, flags):
        flags = torch.tensor(flags, device=device)
        return values[flags].mean() if bool(flags.any()) else values.sum() * 0.

    parts["removal"] = masked_mean(probes["removal"], [m["has_removal"] for m in metas])
    parts["crop_visible"] = masked_mean(probes["crop_visible"], [m["has_visible"] for m in metas])
    for key in ("crop_tamper", "mask", "router"):
        parts[key] = probes[key].mean()
    total = sum(loss_cfg[k] * parts[k] for k in TERMS)
    details = {k: float(v.detach()) for k, v in parts.items()}
    if risk_active:
        groups = [m["group"] for m in metas]
        tail, tail_details = grouped_tail_risk(probes["volume"], groups, {"authentic": risk["authentic_weight"],
                                               "FB": risk["fb_weight"], "FM": risk["fm_weight"]}, risk["alpha"])
        variance, variance_details = generator_variance(probes["volume"], groups, [m["generator"] for m in metas],
                                                        min_samples=risk["min_generator_samples"])
        total = total + risk["cvar_weight"] * tail + risk["generator_weight"] * variance
        details.update(tail_risk=float(tail.detach()), tail_groups=tail_details,
                       generator_variance=float(variance.detach()), generator_groups=variance_details)
    return total, details


def epoch_batches(dataset, config, epoch):
    """Each authentic training volume once per epoch; manipulated volumes stratified by generator x operation."""
    t = config["training"]
    rng = _rng(t["seed"], epoch, "batches")
    authentic = [i for i, v in enumerate(dataset.rows) if v["kind"] == "authentic"]
    strata = defaultdict(list)
    for i, v in enumerate(dataset.rows):
        if v["kind"] == "manipulated":
            strata[(v["generator"], v["operation"])].append(i)
    if not authentic or not strata:
        raise ValueError("Training needs authentic and manipulated volumes")
    rng.shuffle(authentic)
    order = sorted(strata)
    for key in order:
        rng.shuffle(strata[key])
    count = math.ceil(len(authentic) / t["authentic_per_batch"])
    if t["max_batches_per_epoch"]:
        count = min(count, t["max_batches_per_epoch"])
    pointers, batches, cursor = Counter(), [], 0
    for b in range(count):
        batch = [authentic[(b * t["authentic_per_batch"] + j) % len(authentic)] for j in range(t["authentic_per_batch"])]
        for _ in range(t["manipulated_per_batch"]):
            key = order[cursor % len(order)]
            cursor += 1
            items = strata[key]
            batch.append(items[pointers[key] % len(items)])
            pointers[key] += 1
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def replay_step(model, optimizer, dataset, indices, device, config, weights, *, risk_active):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    states, cases, records, selections, metas = [], [], [], [], []
    for index in indices:
        states.append(rng_state())
        case = dataset[index]
        with torch.no_grad():
            terms, _, selection, meta = case_terms(model, case, device, config, train=True)
        cases.append(case)
        records.append({k: v.detach() for k, v in terms.items()})
        selections.append(selection)
        metas.append(meta)
    after = rng_state()
    probes = {k: torch.stack([r[k] for r in records]).requires_grad_() for k in TERMS}
    total, details = objective(probes, metas, config, weights, risk_active=risk_active)
    if not torch.isfinite(total):
        raise ValueError("Non-finite logical-batch objective")
    gradients = dict(zip(TERMS, torch.autograd.grad(total, [probes[k] for k in TERMS], allow_unused=True)))
    try:
        for position, case in enumerate(cases):
            restore_rng(states[position])
            terms, _, _, _ = case_terms(model, case, device, config, train=True, selection=selections[position])
            for key in TERMS:
                if not torch.isclose(terms[key].detach(), records[position][key], rtol=1e-4, atol=1e-5):
                    raise ValueError(f"Replay differs from measurement ({key}); optimizer not stepped")
            surrogate = sum(gradients[k][position].detach() * terms[k] for k in TERMS if gradients[k] is not None)
            surrogate.backward()
    finally:
        restore_rng(after)
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip"], error_if_nonfinite=True))
    optimizer.step()
    return {"objective": float(total.detach()), "terms": details, "gradient_norm": norm, "cases": len(indices),
            "groups": dict(Counter(m["group"] for m in metas))}


# --------------------------------------------------------------------------- evaluation


def _threshold(authentic_scores, target_fpr):
    """Strict '>' threshold with empirical authentic FPR <= target (exploratory)."""
    scores = np.sort(np.asarray(authentic_scores, dtype=float))
    allowed = int(math.floor(target_fpr * len(scores)))
    return float(scores[len(scores) - 1 - allowed]) if len(scores) else None


def volume_metrics(rows, fpr_targets):
    authentic = [r for r in rows if r["kind"] == "authentic"]
    manipulated = [r for r in rows if r["kind"] == "manipulated"]
    out = {"cases": len(rows), "authentic": len(authentic), "manipulated": len(manipulated),
           "by_generator_operation": dict(Counter(f"{r['generator']}:{r['operation']}" for r in manipulated))}

    def auroc(positive, negative=authentic, key="tamper_probability"):
        if not positive or not negative:
            return None
        return binary_auroc([0] * len(negative) + [1] * len(positive), [r[key] for r in negative + positive])

    out["tamper_auroc"] = auroc(manipulated)
    out["tamper_auroc_max_crop"] = auroc(manipulated, key="max_crop_probability")
    out["removal_auroc"] = auroc([r for r in manipulated if r["operation"] == "removal"])
    out["insertion_auroc"] = auroc([r for r in manipulated if r["operation"] == "insertion"])
    per_generator = {g: auroc([r for r in manipulated if r["generator"] == g]) for g in sorted({r["generator"] for r in manipulated})}
    out["per_generator_auroc"] = per_generator
    known = [v for v in per_generator.values() if v is not None]
    out["worst_generator_auroc"] = min(known) if known else None
    # Checkpoint selection: removal and insertion count equally (insertions outnumber removals
    # about 4:1), and a constant score gives 0.5, so a model that ignores the scan cannot win.
    operations = [v for v in (out["removal_auroc"], out["insertion_auroc"]) if v is not None]
    out["selection_auroc"] = float(np.mean(operations)) if operations else None
    checked = [r for r in rows if r.get("tamper_probability_with_site") is not None]
    if checked:
        real = [r for r in checked if r["kind"] == "authentic"]
        edited = [r for r in checked if r["kind"] == "manipulated"]
        site = {"scans": len(checked), "authentic_site_sources": dict(Counter(r["site_source"] for r in real)),
                "scope": SITE_CHECK_SCOPE}
        for name, items in (("tamper", edited), ("removal", [r for r in edited if r["operation"] == "removal"]),
                            ("insertion", [r for r in edited if r["operation"] == "insertion"])):
            site[name + "_auroc"] = auroc(items, real, "tamper_probability_with_site")
            site["crop_" + name + "_auroc"] = auroc(items, real, "site_crop_probability")
        out["with_site_crop"] = site
    losses = defaultdict(list)
    for r in rows:
        losses[r["kind"]].append(r["volume_loss"])
    out["balanced_volume_bce"] = float(np.mean([np.mean(v) for v in losses.values()])) if losses else None
    out["volume_bce"] = float(np.mean([r["volume_loss"] for r in rows])) if rows else None
    # Reported only (it chose checkpoints until 26 Nov 2026): BCE with authentic, removal and
    # insertion weighted equally. Its best constant prediction scores 0.637.
    groups = defaultdict(list)
    for r in rows:
        groups["authentic" if r["kind"] == "authentic" else r["operation"]].append(r["volume_loss"])
    out["group_volume_bce"] = {g: float(np.mean(v)) for g, v in sorted(groups.items())}
    out["group_balanced_volume_bce"] = float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
    operating = {}
    for target in fpr_targets:
        threshold = _threshold([r["tamper_probability"] for r in authentic], target)
        if threshold is None:
            continue
        def recall(items):
            return (sum(r["tamper_probability"] > threshold for r in items) / len(items)) if items else None
        operating[f"fpr_{target:g}"] = {
            "threshold": threshold, "observed_authentic_fpr": sum(r["tamper_probability"] > threshold for r in authentic) / len(authentic),
            "authentic_cases": len(authentic),
            "removal_recall": recall([r for r in manipulated if r["operation"] == "removal"]),
            "insertion_recall": recall([r for r in manipulated if r["operation"] == "insertion"]),
            "removal_recall_by_generator": {g: recall([r for r in manipulated if r["operation"] == "removal" and r["generator"] == g])
                                            for g in sorted({r["generator"] for r in manipulated})}}
    out["operating_points"] = operating
    out["operating_point_scope"] = "thresholds set on these same authentic cases; exploratory, not a frozen endpoint"
    if manipulated:
        out["removal_vs_insertion_accuracy"] = float(np.mean([(r["removal_probability"] > .5) == (r["operation"] == "removal")
                                                              for r in manipulated]))
        out["router_recall_at_k"] = float(np.mean([bool(r["router_hit"]) for r in manipulated]))
        out["router_recall_by_operation"] = {op: float(np.mean([bool(r["router_hit"]) for r in manipulated
                                                                 if r["operation"] == op]))
                                             for op in sorted({r["operation"] for r in manipulated})}
        dice = [r["mask_dice_selected"] for r in manipulated if r.get("mask_dice_selected") is not None]
        out["mask_dice_on_hit_crops"] = float(np.mean(dice)) if dice else None
    return out


def evaluate(model, dataset, device, config, *, progress=None, partition="validation"):
    model.eval()
    rows = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            case = dataset[index]
            terms, outputs, _, _ = case_terms(model, case, device, config, train=False)
            v = case.volume
            rows.append({"volume_id": v["volume_id"], "kind": v["kind"], "generator": v["generator"],
                         "operation": v["operation"], "patient_id": v["patient_id"],
                         "volume_loss": float(terms["volume"]), **outputs})
            if progress and (index + 1) % 100 == 0:
                progress({"event": "evaluation", "partition": partition, "done": index + 1, "of": len(dataset)})
    return volume_metrics(rows, config["evaluation"]["fpr_targets"]), rows


# --------------------------------------------------------------------------- pretrained crop network


def load_pretrained(folder, config):
    """Verified crop-network weights and the check that saved them, from a crop pretraining check folder.

    The file must have the checksum the configuration names, match its own receipt, and have
    been built with this model's crop-network settings.
    """
    from rsvt.experiments import m3d_pretrain_check as pretrain  # it imports this module
    folder = Path(folder)
    weights = folder / pretrain.WEIGHTS
    expected = config["training"].get("fine_init_sha256", "")
    if not expected or sha256_path(weights) != expected:
        raise ValueError("The pretrained crop weights do not have the checksum the configuration names (fine_init_sha256)")
    receipt = json.loads(weights.with_name(weights.name + ".json").read_text(encoding="utf-8"))
    state = load_checkpoint(weights, receipt["contract_sha256"], expected_format=pretrain.WEIGHTS_FORMAT)
    settings = {k: list(v) if isinstance(v, tuple) else v for k, v in pretrain.fine_kwargs(config["model"]).items()}
    if state["model_kwargs"] != settings:
        raise ValueError("The pretrained crop network has other settings: " + json.dumps(state["model_kwargs"]))
    check_path = folder / "pretrain_check.json"
    check = json.loads(check_path.read_text(encoding="utf-8"))
    return {"fine": state["fine"], "check": check,
            "record": {"folder": str(folder), "weights_sha256": expected, "weights_contract_sha256": receipt["contract_sha256"],
                       "pretrain_check_sha256": sha256_path(check_path),
                       "pretrain_check_interpretation": check.get("interpretation")}}


def fixed_pairs(store, validation, check, crop_mm, progress=None):
    """The pretraining check's validation pairs, cut exactly as the model cuts its site crop."""
    from rsvt.experiments.m3d_pretrain_check import site_pairs
    rows = {v["volume_id"]: v for v in validation.rows}
    wanted = [(r["volume_id"], r["crop_shape_zyx"], r["changed_voxels_in_crop"]) for r in check["validation_pairs"]]
    missing = [w[0] for w in wanted if w[0] not in rows]
    if missing:
        raise ValueError(f"{len(missing)} fixed validation pairs are not in this run's validation set (e.g. {missing[0]})")
    pairs = site_pairs(store, [rows[w[0]] for w in wanted], crop_mm, margin_fraction=0., progress=progress, split="fixed")
    if [(r["volume_id"], r["crop_shape_zyx"], r["changed_voxels_in_crop"]) for r in pairs["records"]] != wanted:
        raise ValueError("The fixed validation pairs differ from the crops the pretraining check scored")
    return pairs


def score_fixed_pairs(model, pairs, device):
    """Share of fixed pairs whose edited crop outscores its unedited twin, crop AUROC and BCE, per operation and generator."""
    from rsvt.experiments.m3d_crop_check import pair_metrics
    from rsvt.experiments.m3d_pretrain_check import pair_logits
    training = model.training
    model.eval()
    fake, real = pair_logits(model.encode_crops, pairs["items"], device)
    model.train(training)
    return pair_metrics(fake, real, pairs["records"])


def _pair_share(scores, group):
    return ((scores or {}).get(group) or {}).get("paired_accuracy")


# --------------------------------------------------------------------------- fit / run


def fit(model, train, validation, config, output_dir, contract, *, resume=False, progress=None, stop_after_epoch=None,
        pretrained=None):
    t = config["training"]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    seed_everything(t["seed"], t["threads"])
    if t["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("A Colab GPU is required")
    device = torch.device(t["device"])
    model.to(device)
    optimizer = build_optimizer(model, t)
    scheduler = build_scheduler(optimizer, t)
    weights = class_weights(config)
    binding = digest_json(contract)
    # Selection starts at chance: an epoch is kept only if it beats 0.5 by min_delta.
    history, best, bad, start, best_epoch = [], .5, 0, 0, None
    if resume:
        state = load_checkpoint(destination / "last.pt", binding, expected_format=FORMAT)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        history, best, bad, start, best_epoch = (state[k] for k in ("history", "best", "bad", "epoch", "best_epoch"))
    elif (destination / "last.pt").exists():
        raise FileExistsError("Checkpoints exist; resume explicitly")
    if pretrained and not resume:
        # Run 3: only the crop network's starting weights change. A resumed run keeps the weights it saved.
        model.fine.load_state_dict(pretrained["fine"])  # strict: every key and shape must match
        at_start = score_fixed_pairs(model, pretrained["pairs"], device)
        reference = pretrained["check"]["pretraining"]["final"]["all"]
        reproduces = all(abs(at_start["all"][k] - reference[k]) <= .01 for k in ("paired_accuracy", "crop_auroc"))
        write_json(destination / "fine_init.json", {**pretrained["record"], "fixed_pairs": len(pretrained["pairs"]["records"]),
                                                    "at_start": at_start, "pretraining_final": pretrained["check"]["pretraining"]["final"],
                                                    "reproduces_pretraining": reproduces})
        if not reproduces:
            raise ValueError("The loaded crop network does not reproduce the pretraining check's scores on its own "
                             "validation pairs; the weights or the crops differ")
        if progress:
            progress({"event": "fine_init", "fixed_pairs_removal": _pair_share(at_start, "removal"),
                      "fixed_pairs_insertion": _pair_share(at_start, "insertion")})
    began = time.perf_counter()
    for epoch in range(start, t["epochs"]):
        if t["patience"] and bad >= t["patience"]:
            break
        train.set_epoch(epoch)
        risk_active = epoch >= t["risk_start_epoch"]
        batches = epoch_batches(train, config, epoch)
        steps = []
        for number, indices in enumerate(batches, 1):
            steps.append(replay_step(model, optimizer, train, indices, device, config, weights, risk_active=risk_active))
            if progress:
                progress({"event": "logical_batch", "epoch": epoch + 1, "batch": number, "batches": len(batches),
                          "objective": steps[-1]["objective"], "risk_active": risk_active})
        metrics, predictions = evaluate(model, validation, device, config, progress=progress)
        pair_scores = score_fixed_pairs(model, pretrained["pairs"], device) if pretrained else None
        score = metrics["selection_auroc"]
        eligible = risk_active
        improved = eligible and score is not None and score > best + t["min_delta"]
        if improved:
            best, bad, best_epoch = score, 0, epoch + 1
        elif eligible:
            bad += 1
        mean_terms = {k: float(np.mean([s["terms"][k] for s in steps])) for k in TERMS}
        row = {"epoch": epoch + 1, "stage": "risk" if risk_active else "warmup", "checkpoint_eligible": eligible,
               "learning_rate": optimizer.param_groups[0]["lr"], "optimizer_steps": len(steps),
               "training_objective": float(np.mean([s["objective"] for s in steps])), **{"train_" + k: v for k, v in mean_terms.items()},
               "validation_selection_auroc": score, "validation_balanced_bce": metrics["balanced_volume_bce"],
               "validation_group_balanced_bce": metrics["group_balanced_volume_bce"],
               "validation_tamper_auroc": metrics["tamper_auroc"],
               "validation_removal_auroc": metrics["removal_auroc"], "validation_insertion_auroc": metrics["insertion_auroc"],
               "validation_worst_generator_auroc": metrics["worst_generator_auroc"],
               "validation_tamper_auroc_with_site": (metrics.get("with_site_crop") or {}).get("tamper_auroc"),
               "validation_site_crop_auroc": (metrics.get("with_site_crop") or {}).get("crop_tamper_auroc"),
               "validation_router_recall": metrics.get("router_recall_at_k"),
               "validation_router_recall_removal": (metrics.get("router_recall_by_operation") or {}).get("removal"),
               "validation_router_recall_insertion": (metrics.get("router_recall_by_operation") or {}).get("insertion"),
               "validation_removal_recall_fpr10": metrics["operating_points"].get("fpr_0.1", {}).get("removal_recall"),
               "validation_pairs_removal": _pair_share(pair_scores, "removal"),
               "validation_pairs_insertion": _pair_share(pair_scores, "insertion"),
               "validation_pairs_crop_auroc": ((pair_scores or {}).get("all") or {}).get("crop_auroc"),
               "gradient_norm_mean": float(np.mean([s["gradient_norm"] for s in steps]))}
        history.append(row)
        scheduler.step()
        state = {"format_version": FORMAT, "contract_sha256": binding, "epoch": epoch + 1,
                 "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(),
                 "history": history, "best": best, "bad": bad, "best_epoch": best_epoch}
        write_json(destination / f"epoch_{epoch + 1:03d}.json", {"row": row, "validation": metrics, "steps": steps,
                                                                "fixed_pairs": pair_scores})
        if improved:
            save_checkpoint(destination / "best.pt", state)
            write_json(destination / "best_validation_predictions.json", predictions)
        save_checkpoint(destination / "last.pt", state)
        write_json(destination / "history.json", history)
        if progress:
            progress({"event": "epoch", **row})
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            break
    selected = destination / ("best.pt" if (destination / "best.pt").exists() else "last.pt")
    state = load_checkpoint(selected, binding, expected_format=FORMAT)
    model.load_state_dict(state["model"])
    metrics, predictions = evaluate(model, validation, device, config, progress=progress)
    selected_pairs = score_fixed_pairs(model, pretrained["pairs"], device) if pretrained else None
    write_json(destination / "selected_validation_predictions.json", predictions)
    last = history[-1]["epoch"] if history else 0
    summary = {"ok": True, "model": "Coarse-to-fine Risk-Aware Sparse Volumetric Transformer",
               "dataset": "M3Dsynth", "scope": "exploratory_volume_forensics", "training_ready": False,
               "epochs_completed": last, "training_finished": last >= t["epochs"] or bool(t["patience"] and bad >= t["patience"]),
               "best_epoch": best_epoch, "reported_checkpoint_epoch": state["epoch"],
               "checkpoint_stage": "risk" if best_epoch else "provisional_warmup_or_last",
               "selection_rule": SELECTION_RULE, "best_selection_auroc": best if best_epoch else None,
               "selection_note": (None if best_epoch else
                                  "No risk-stage epoch beat chance on the selection AUROC; the last epoch is reported."),
               "site_crop_check": SITE_CHECK_SCOPE,
               "fine_init": pretrained["record"] if pretrained else None,
               "fixed_pairs": selected_pairs,
               "train_volumes": dict(Counter(v["kind"] for v in train.rows)),
               "validation_volumes": dict(Counter(v["kind"] for v in validation.rows)),
               "class_weights": weights, "validation_metrics": metrics, "history": history,
               "optimization": optimization_report(optimizer, t), "parameters": sum(p.numel() for p in model.parameters()),
               "wall_seconds_this_invocation": time.perf_counter() - began, "device": str(device),
               "test_volumes_opened": 0, "annotation_points_used": False,
               "supervision": ("changed voxels (manipulated != real) for router, crop and mask targets; author "
                               "manipulation centres for the site crop (M3Dsynth), training only; inference uses router top-K"),
               "limitations": ["Exploratory: thresholds are set on the same validation authentic volumes.",
                               "LIDC-IDRI sources may overlap CT-GAN scans; do not treat CT-GAN as independent without identity checks.",
                               "Volume-level authentic count limits the resolution of low-FPR operating points."]}
    write_json(destination / "summary.json", summary)
    return summary


def persist_receipt(directory):
    """Read every report and checkpoint back against checksums before declaring success."""
    directory = Path(directory)
    records = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "persistence_receipt.json" or path.suffix not in (".json", ".md", ".pt", ".csv", ".png", ".toml"):
            continue
        digest = sha256_path(path)
        if path.suffix == ".pt":
            receipt = json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))
            if receipt["sha256"] != digest or receipt["bytes"] != path.stat().st_size:
                raise OSError("Checkpoint readback mismatch: " + path.name)
        records[path.relative_to(directory).as_posix()] = {"sha256": digest, "bytes": path.stat().st_size}
    receipt = write_json(directory / "persistence_receipt.json", {"schema_version": "rsvt_c2f_persistence_v1", "files": records})
    for name, record in records.items():
        if sha256_path(directory / name) != record["sha256"]:
            raise OSError("Readback mismatch: " + name)
    return receipt


def build_datasets(store, config):
    d = config["data"]
    common = dict(seed=config["training"]["seed"], grid=d["grid"], crop_mm=d["crop_mm"], generators=d["generators"])
    train = M3DDataset(store, "train", train=True, flip_probability=d["flip_probability"], forced_sites=d["forced_sites"],
                       random_crops=d["random_crops"], jitter_fraction=d["jitter_fraction"], **common)
    validation = M3DDataset(store, "validation", train=False, check_site=True, **common)
    return train, validation


def run_training(store_root, cache_root, config, output_dir, *, resume=False, progress=None, stop_after_epoch=None,
                 pretrained=None):
    """Train (or resume); `pretrained` is a crop pretraining check folder, required when the config names its weights."""
    config = validate_config(config)
    if bool(config["training"].get("fine_init_sha256")) != (pretrained is not None):
        raise ValueError("Pass the pretraining check folder exactly when the configuration names pretrained crop weights")
    loaded = load_pretrained(pretrained, config) if pretrained is not None else None  # verified before any data work
    store = M3DStore(store_root, cache_root)
    train, validation = build_datasets(store, config)
    store.localize(train.rows + validation.rows, progress=progress)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = store.label_check(train.rows + validation.rows)
    write_json(output / "label_check.json", labels)
    if progress:
        progress({"stage": "labels", **{op: {"volumes": g["volumes"], "mask_changed": (g["mask_changed"] or {}).get("median"),
                                              "depth_ratio": (g["depth_ratio"] or {}).get("median")}
                                         for op, g in labels["groups"].items() if ":" not in op}})
    contract = {"model": "RSVT-C2F", "implementation_sha256": implementation_fingerprint(),
                "config_sha256": digest_json(config), "manifest_sha256": sha256_path(Path(store_root) / "manifest.json"),
                "assignments_sha256": store.manifest["assignments_sha256"]}
    if loaded:
        contract["pretrain_check_sha256"] = loaded["record"]["pretrain_check_sha256"]
        loaded["pairs"] = fixed_pairs(store, validation, loaded["check"], config["data"]["crop_mm"], progress)
    if resume:
        stored = json.loads((output / "run_contract.json").read_text(encoding="utf-8"))
        if stored != contract:
            raise ValueError("Resume needs the same code, configuration and manifest")
    else:
        if (output / "run_contract.json").exists():
            raise FileExistsError("Use a new run folder or resume explicitly")
        write_json(output / "run_contract.json", contract)
        write_json(output / "resolved_config.json", config)
    seed_everything(config["training"]["seed"], config["training"]["threads"])
    model = CoarseToFineRSVT(**config["model"])
    info = store.series[train.rows[0]["series_key"]]
    write_json(output / "compute_estimate.json", compute_estimate(
        model, (96, 96, 96), crop_shape_for(info["geometry"]["spacing_zyx"], config["data"]["crop_mm"]),
        model.crops_at_inference, info["shape_zyx"]) | {"example_series": info["series_key"]})
    try:
        summary = fit(model, train, validation, config, output, contract, resume=resume and (output / "last.pt").exists(),
                      progress=progress, stop_after_epoch=stop_after_epoch, pretrained=loaded)
    except Exception as exc:
        write_json(output / "failure.json", {"ok": False, "reason": str(exc)})
        raise
    metrics = summary["validation_metrics"]
    lines = ["# RSVT coarse-to-fine run (M3Dsynth)", "",
             f"Train volumes: {summary['train_volumes']}; validation: {summary['validation_volumes']}; test opened: 0.", "",
             f"Selected epoch: {summary['best_epoch']} ({summary['selection_rule']}).", "",
             f"Validation tamper AUROC {metrics['tamper_auroc']}; removal AUROC {metrics['removal_auroc']}; "
             f"insertion AUROC {metrics['insertion_auroc']}; worst-generator AUROC {metrics['worst_generator_auroc']}; "
             f"router recall@K {metrics.get('router_recall_at_k')}.", "",
             f"With the edit's crop added (diagnostic): tamper AUROC {(metrics.get('with_site_crop') or {}).get('tamper_auroc')}; "
             f"that crop alone {(metrics.get('with_site_crop') or {}).get('crop_tamper_auroc')}.", ""]
    if loaded:
        pairs = summary["fixed_pairs"]
        lines += [f"Crop network started from the weights of the pretraining check in `{loaded['record']['folder']}`. "
                  f"Fixed validation pairs ordered correctly at the selected epoch: removal {_pair_share(pairs, 'removal')}, "
                  f"insertion {_pair_share(pairs, 'insertion')}.", ""]
    lines += ["Operating points are exploratory (thresholds from the same validation authentic volumes).", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    persist_receipt(output)
    return summary
