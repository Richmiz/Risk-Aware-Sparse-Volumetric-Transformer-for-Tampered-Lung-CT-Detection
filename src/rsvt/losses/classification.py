"""Inverse-frequency class weighting from selected training candidates only."""

import torch
from torch.nn import functional as F


def training_class_weights(labels, classes=4):
    labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.ndim != 1 or not len(labels) or bool(((labels < 0) | (labels >= classes)).any()):
        raise ValueError("Invalid training label indices")
    counts = torch.bincount(labels, minlength=classes)
    if bool((counts == 0).any()):
        raise ValueError("Selected training subset must contain every class")
    weights = counts.sum().float() / (classes * counts.float())
    return weights, counts


def balanced_cross_entropy(logits, targets, weights):
    # Multiplying unreduced losses keeps class weights effective at batch size 1.
    # PyTorch's weighted-mean CE would divide by the sole example's weight.
    values = F.cross_entropy(logits, targets, reduction="none")
    return (values * weights[targets]).mean()
