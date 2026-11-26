"""Coarse-to-fine RSVT: budgeted native-resolution crops chosen by a coarse router.

Why this replaces the 96^3 token design (review, 17 Nov 2026)
-------------------------------------------------------------
* The 96^3 full-FOV view has ~3-4.6 mm voxels; a CT-GAN/M3Dsynth edit spans only
  a few such voxels and its texture traces are averaged away. Here the coarse
  view only provides anatomy context and proposes where to look; evidence is read
  from native-grid crops.
* Sparse selection now saves real compute: the expensive encoder runs on K crops
  instead of the whole native volume (see `compute_estimate`).
* Heads are factorized to match the forensic endpoint: tampered vs authentic
  (primary) and removal vs insertion; per-crop tamper, visible-nodule and voxel
  mask heads use the author masks as supervision.

26 Nov 2026: the fine branch reads the crop's residual (image minus its 3x3x3 local
mean) by default. In the crop learning check a fresh fine branch learned the edit
from residual crops on unseen patients (removal crop AUROC 0.94, insertion 0.75) but
not from the windowed image (0.69 for both), which it fitted by memorising anatomy.

All pooling is reshape/amax based and the decoder uses transposed convolutions, so
CUDA backward stays deterministic under `torch.use_deterministic_algorithms`.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


FINE_INPUTS = ("image", "residual")


def _groups(channels):
    return max(1, min(8, channels // 4))


def local_residual(images):
    """Image minus its 3x3x3 local mean: suppresses anatomy, keeps fine texture and seams."""
    return images - F.avg_pool3d(images, 3, stride=1, padding=1, count_include_pad=False)


class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(inputs, outputs, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(_groups(outputs), outputs)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = ConvBlock(channels, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_groups(channels), channels)

    def forward(self, x):
        return F.gelu(x + self.norm(self.conv(self.first(x))))


def _logit(probability):
    if not 0 < probability < 1:
        raise ValueError("Prior probability must be in (0,1)")
    return math.log(probability / (1 - probability))


class CoarseBranch(nn.Module):
    """96^3 view -> cell features on a 12^3 grid (stride 8) and 6^3 context tokens."""

    def __init__(self, dimension=128, widths=(16, 32, 64), router_hidden=64, router_prior=.003):
        super().__init__()
        a, b, c = widths
        self.stem = ConvBlock(1, a)
        self.down = nn.Sequential(ConvBlock(a, b, 2), ResBlock(b), ConvBlock(b, c, 2), ResBlock(c),
                                  ConvBlock(c, dimension, 2), ResBlock(dimension))
        self.context = nn.Linear(dimension, dimension)
        self.router = nn.Sequential(nn.Linear(2 * dimension, router_hidden), nn.GELU(), nn.Linear(router_hidden, 1))
        # Start at the expected share of manipulated cells (~1-5 of 1,728); with a
        # zero bias every cell starts at p=0.5 and the rare positives are swamped.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, _logit(router_prior))

    def forward(self, coarse):
        if coarse.ndim != 5 or coarse.shape[:2] != (1, 1):
            raise ValueError("Coarse branch expects one 1x1xZxYxX view")
        cells = self.down(self.stem(coarse))[0]  # (C, g, g, g)
        c, g = cells.shape[0], cells.shape[1]
        if g % 2 or cells.shape[1:] != (g, g, g):
            raise ValueError("Coarse view must produce an even cubic cell grid")
        flat = cells.flatten(1).transpose(0, 1)  # (g^3, C)
        context_grid = cells.reshape(c, g // 2, 2, g // 2, 2, g // 2, 2).mean(dim=(2, 4, 6))
        context = self.context(context_grid.flatten(1).transpose(0, 1))
        global_mean = flat.mean(0, keepdim=True).expand_as(flat)
        router = self.router(torch.cat((flat, global_mean), -1)).squeeze(-1)
        h = g // 2
        index = torch.stack(torch.meshgrid(*[torch.arange(h, device=coarse.device)] * 3, indexing="ij"), -1).reshape(-1, 3)
        positions = ((index.float() + .5) / h * 2. - 1.)
        return {"cells": flat, "router_logits": router, "context": context, "context_positions": positions, "grid": g}


class FineBranch(nn.Module):
    """Native crops -> token, crop tamper/visible logits and a half-resolution mask.

    `input` chooses what the stem reads: the windowed image or its local residual.
    """

    def __init__(self, dimension=128, widths=(16, 32, 64), input="image"):
        super().__init__()
        if input not in FINE_INPUTS:
            raise ValueError("Fine input must be one of: " + ", ".join(FINE_INPUTS))
        self.input = input
        a, b, c = widths
        self.stem = nn.Sequential(ConvBlock(2, a), ConvBlock(a, a))  # full-resolution texture
        self.s1 = nn.Sequential(ConvBlock(a, b, 2), ResBlock(b))
        self.s2 = nn.Sequential(ConvBlock(b, c, 2), ResBlock(c))
        self.s3 = nn.Sequential(ConvBlock(c, dimension, 2), ResBlock(dimension))
        self.token = nn.Linear(2 * dimension, dimension)
        self.tamper = nn.Linear(dimension, 1)
        self.visible = nn.Linear(dimension, 1)
        self.up = nn.ConvTranspose3d(c, b, 2, stride=2)
        self.mask = nn.Sequential(ConvBlock(b, a), nn.Conv3d(a, 1, 1))

    def forward(self, image, valid):
        if image.ndim != 5 or image.shape[1] != 1 or valid.shape != image.shape:
            raise ValueError("Crops must be Kx1xZxYxX with a matching validity mask")
        if any(n % 8 for n in image.shape[2:]):
            raise ValueError("Crop sides must be multiples of 8")
        v = valid.float()
        signal = local_residual(image) if self.input == "residual" else image
        x = self.stem(torch.cat((signal * v, v), 1))
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        pooled = torch.cat((f3.mean(dim=(2, 3, 4)), f3.amax(dim=(2, 3, 4))), -1)
        token = F.gelu(self.token(pooled))
        mask = self.mask(f1 + self.up(f2))
        return {"tokens": token, "tamper_logits": self.tamper(token).squeeze(-1),
                "visible_logits": self.visible(token).squeeze(-1), "mask_logits": mask}


class CoarseToFineRSVT(nn.Module):
    def __init__(self, dimension=128, heads=4, layers=3, dropout=.1, coarse_widths=(16, 32, 64),
                 fine_widths=(16, 32, 64), router_hidden=64, router_prior=.003, crops_at_inference=16,
                 fine_input="residual"):
        super().__init__()
        if dimension % heads:
            raise ValueError("Transformer dimension must divide into heads")
        if isinstance(crops_at_inference, bool) or not isinstance(crops_at_inference, int) or crops_at_inference < 1:
            raise ValueError("crops_at_inference must be a positive integer")
        self.crops_at_inference = crops_at_inference
        self.coarse = CoarseBranch(dimension, coarse_widths, router_hidden, router_prior)
        self.fine = FineBranch(dimension, fine_widths, fine_input)
        self.position = nn.Sequential(nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.kind = nn.Embedding(2, dimension)
        self.crop_score = nn.Linear(1, dimension)
        self.cls = nn.Parameter(torch.zeros(1, 1, dimension))
        nn.init.trunc_normal_(self.cls, std=.02)
        self.transformer = nn.ModuleList(nn.TransformerEncoderLayer(dimension, heads, 4 * dimension, dropout,
                                         activation="gelu", batch_first=True, norm_first=True) for _ in range(layers))
        self.norm = nn.LayerNorm(dimension)
        self.tamper = nn.Linear(dimension, 1)
        self.removal = nn.Linear(dimension, 1)

    def encode_coarse(self, coarse):
        return self.coarse(coarse)

    def encode_crops(self, image, valid):
        return self.fine(image, valid)

    def fuse(self, coarse, crops, crop_positions):
        context = coarse["context"] + self.position(coarse["context_positions"]) + self.kind.weight[0]
        tokens = (crops["tokens"] + self.position(crop_positions) + self.kind.weight[1]
                  + self.crop_score(crops["tamper_logits"][:, None]))
        sequence = torch.cat((self.cls[0], context, tokens), 0)[None]
        for block in self.transformer:
            sequence = block(sequence)
        cls = self.norm(sequence[0, 0])
        return {"tamper_logit": self.tamper(cls).squeeze(-1), "removal_logit": self.removal(cls).squeeze(-1)}

    @staticmethod
    def select_cells(router_logits, cell_valid, k, *, exclude=None):
        """Hard top-k among valid cells (stable for ties)."""
        scores = router_logits.detach().float().clone()
        allowed = cell_valid.clone()
        if exclude is not None:
            allowed &= ~exclude
        scores[~allowed] = float("-inf")
        count = min(int(k), int(allowed.sum()))
        if count <= 0:
            return torch.zeros(0, dtype=torch.long, device=router_logits.device)
        order = torch.argsort(scores, descending=True, stable=True)
        return order[:count]


def compute_estimate(model, coarse_shape, crop_shape, crops, native_shape):
    """Multiply-accumulates from conv/linear/transposed-conv hooks on a dummy forward.

    Returns coarse, per-crop fine, fusion and the dense-native equivalent of the
    fine encoder (the comparison that sparse selection should beat). Attention
    matmuls are added analytically. These are operator estimates, not latency.
    """
    counts = {"value": 0}

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv3d):
            k = math.prod(module.kernel_size)
            counts["value"] += output.numel() * module.in_channels * k // module.groups
        elif isinstance(module, nn.ConvTranspose3d):
            k = math.prod(module.kernel_size)
            counts["value"] += inputs[0].numel() * module.out_channels * k // module.groups
        elif isinstance(module, nn.Linear):
            counts["value"] += output.numel() * module.in_features

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear))]
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            counts["value"] = 0
            coarse = model.encode_coarse(torch.zeros(1, 1, *coarse_shape))
            coarse_macs = counts["value"]
            counts["value"] = 0
            one = model.encode_crops(torch.zeros(1, 1, *crop_shape), torch.ones(1, 1, *crop_shape, dtype=torch.bool))
            crop_macs = counts["value"]
            counts["value"] = 0
            many = {k: v.repeat(crops, *[1] * (v.ndim - 1)) for k, v in one.items()}
            model.fuse(coarse, many, torch.zeros(crops, 3))
            fusion_macs = counts["value"]
        model.train(was_training)
    finally:
        for handle in handles:
            handle.remove()
    sequence = 1 + coarse["context"].shape[0] + crops
    dimension = model.norm.normalized_shape[0]
    fusion_macs += len(model.transformer) * 2 * sequence * sequence * dimension
    per_voxel = crop_macs / math.prod(crop_shape)
    native = math.prod(native_shape)
    sparse = coarse_macs + crops * crop_macs + fusion_macs
    return {"coarse_gmac": coarse_macs / 1e9, "fine_gmac_per_crop": crop_macs / 1e9, "crops": crops,
            "fusion_gmac": fusion_macs / 1e9, "total_gmac": sparse / 1e9,
            "dense_native_fine_gmac": per_voxel * native / 1e9,
            "dense_over_sparse": (per_voxel * native + coarse_macs) / sparse,
            "native_voxels_read_fraction": crops * math.prod(crop_shape) / native,
            "scope": "conv/linear/transposed-conv MACs from hooks plus analytic attention; not measured latency"}
