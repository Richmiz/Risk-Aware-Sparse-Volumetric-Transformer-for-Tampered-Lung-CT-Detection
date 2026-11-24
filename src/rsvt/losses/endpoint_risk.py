"""Losses aligned with the forensic endpoint: removal recall at a fixed authentic FPR.

The earlier class-conditional CVaR acted on four-class CE, so it also pushed on
harmless FB<->FM confusions and never looked at the authentic scans that set the
threshold. Here the tail terms act on the binary tampered-vs-authentic loss and
cover both sides of the threshold: the worst missed removals (FB), the worst
missed insertions (FM) and the highest-scoring authentic scans.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from rsvt.losses.cvar import empirical_cvar

GROUPS = ("authentic", "FB", "FM")


def grouped_tail_risk(losses, groups, weights, alpha):
    """Weighted mean of per-group empirical CVaR; absent groups add 0 (fixed denominator)."""
    if losses.ndim != 1 or len(groups) != len(losses):
        raise ValueError("One loss and one group per case required")
    if set(weights) - set(GROUPS) or any(w < 0 for w in weights.values()) or not sum(weights.values()):
        raise ValueError("Non-negative authentic/FB/FM weights with a positive sum required")
    terms, details = [], {}
    for name, weight in weights.items():
        selected = torch.tensor([g == name for g in groups], device=losses.device)
        values = losses[selected]
        value = empirical_cvar(values, alpha) if len(values) else losses.sum() * 0.
        terms.append(weight * value)
        details[name] = {"count": int(selected.sum()), "cvar": float(value.detach()), "weight": weight}
    return sum(terms) / sum(weights.values()), details


def generator_variance(losses, groups, generators, *, min_samples=2):
    """Population variance of per-generator mean loss within FB and within FM."""
    zero = losses.sum() * 0.
    terms, details = [], {}
    for label in ("FB", "FM"):
        means = []
        for generator in sorted({g for g, group in zip(generators, groups) if group == label and g}):
            selected = torch.tensor([group == label and g == generator for g, group in zip(generators, groups)],
                                    device=losses.device)
            if int(selected.sum()) >= min_samples:
                means.append(losses[selected].mean())
        enabled = len(means) >= 2
        terms.append(torch.stack(means).var(unbiased=False) if enabled else zero)
        details[label] = {"generators_with_support": len(means), "enabled": enabled}
    return sum(terms) / 2., details


def dice_bce(logits, target, valid, eps=1.):
    """Per-crop BCE on valid voxels plus soft Dice on crops whose target is non-empty."""
    v = valid.float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * v).flatten(1).sum(1)
    bce = bce / v.flatten(1).sum(1).clamp_min(1.)
    probability = torch.sigmoid(logits) * v
    t = target * v
    intersection = (probability * t).flatten(1).sum(1)
    total = probability.flatten(1).sum(1) + t.flatten(1).sum(1)
    dice = 1. - (2. * intersection + eps) / (total + eps)
    has_target = t.flatten(1).sum(1) > 0
    return bce + torch.where(has_target, dice, torch.zeros_like(dice))
