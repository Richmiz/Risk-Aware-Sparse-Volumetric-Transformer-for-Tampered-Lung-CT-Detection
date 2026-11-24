"""B0 3D ResNet-50/GN and B1 dense fixed-patch 3D ViT engineering baselines."""

import math
import torch
from torch import nn
from torch.nn import functional as F


def _norm(channels):
    groups = math.gcd(8, channels)
    if channels // groups < 2:
        groups = max(1, groups // 2)
    return nn.GroupNorm(groups, channels)


def _inputs(image, valid):
    if image.ndim != 5 or image.shape[1] != 1 or valid.shape != image.shape or valid.dtype != torch.bool:
        raise ValueError("Expected B/1/Z/Y/X images and matching bool validity mask")
    if not valid.flatten(1).any(dim=1).all():
        raise ValueError("Every input volume needs valid voxels")
    return image * valid.to(image.dtype)


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, incoming, width, stride=1):
        super().__init__()
        outgoing = width * self.expansion
        self.layers = nn.Sequential(nn.Conv3d(incoming, width, 1, bias=False), _norm(width), nn.ReLU(),
            nn.Conv3d(width, width, 3, stride=stride, padding=1, bias=False), _norm(width), nn.ReLU(),
            nn.Conv3d(width, outgoing, 1, bias=False), _norm(outgoing))
        self.shortcut = (nn.Identity() if incoming == outgoing and stride == 1 else
                         nn.Sequential(nn.Conv3d(incoming, outgoing, 1, stride=stride, bias=False), _norm(outgoing)))

    def forward(self, image):
        return F.relu(self.layers(image) + self.shortcut(image))


class ResNet50Volume(nn.Module):
    """Bottleneck stage counts 3/4/6/3, 3D convolutions, GroupNorm, masked pooling.

    Width is explicit; the narrow pilot is not presented as a pretrained or
    standard-width benchmark reproduction.
    """
    def __init__(self, base_width=16, classes=4):
        super().__init__()
        if not isinstance(base_width, int) or isinstance(base_width, bool) or base_width < 2 or classes != 4:
            raise ValueError("Invalid CNN width or class contract")
        self.stem = nn.Sequential(nn.Conv3d(1, base_width, 7, stride=2, padding=3, bias=False),
                                  _norm(base_width), nn.ReLU(), nn.MaxPool3d(3, stride=2, padding=1))
        stages, incoming = [], base_width
        for stage, count in enumerate((3, 4, 6, 3)):
            width = base_width * 2**stage
            blocks = [Bottleneck3D(incoming, width, 1 if stage == 0 else 2)]
            blocks += [Bottleneck3D(width*4, width) for _ in range(count-1)]
            stages.append(nn.Sequential(*blocks))
            incoming = width*4
        self.stages = nn.Sequential(*stages)
        self.classifier = nn.Linear(incoming, classes)

    def forward(self, image, valid_mask):
        features = self.stages(self.stem(_inputs(image, valid_mask)))
        support = F.adaptive_max_pool3d(valid_mask.float(), features.shape[-3:])
        pooled = (features*support).sum((2, 3, 4)) / support.sum((2, 3, 4)).clamp_min(1.)
        return {"logits": self.classifier(pooled), "feature_cells": math.prod(features.shape[-3:])}


class DenseViT3D(nn.Module):
    def __init__(self, input_shape=(64,64,64), patch_size=8, dimension=96, layers=2, heads=4, dropout=.1):
        super().__init__()
        if (not isinstance(heads,int) or isinstance(heads,bool) or heads <= 0
                or not isinstance(dimension,int) or isinstance(dimension,bool)
                or not isinstance(layers,int) or isinstance(layers,bool)
                or len(input_shape) != 3 or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in input_shape)
                or not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size <= 0
                or any(n % patch_size for n in input_shape) or dimension < 4 or dimension % heads or layers < 1
                or not 0 <= dropout < 1):
            raise ValueError("Invalid ViT input grid or transformer configuration")
        self.input_shape, self.patch_size = tuple(input_shape), patch_size
        self.token_count = math.prod(n//patch_size for n in input_shape)
        if self.token_count > 4096:
            raise ValueError("Dense ViT grid exceeds the engineering limit of 4096 patch tokens")
        self.embedding = nn.Conv3d(1, dimension, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1,1,dimension))
        self.position = nn.Parameter(torch.empty(1,self.token_count+1,dimension))
        nn.init.trunc_normal_(self.position, std=.02)
        layer = nn.TransformerEncoderLayer(dimension, heads, dimension*4, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dimension), enable_nested_tensor=False)
        # TransformerEncoder clones its prototype layer; initialize each clone
        # independently rather than starting every depth with identical weights.
        for block in self.encoder.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            for module in block.modules():
                if isinstance(module,nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        self.head = nn.Linear(dimension, 4)

    def forward(self, image, valid_mask):
        image = _inputs(image, valid_mask)
        if tuple(image.shape[-3:]) != self.input_shape:
            raise ValueError("ViT input shape differs from its fixed positional grid")
        tokens = self.embedding(image).flatten(2).transpose(1,2)
        valid_tokens = F.max_pool3d(valid_mask.float(), self.patch_size, self.patch_size).flatten(1) > 0
        padding = torch.cat([torch.zeros((len(image),1),dtype=torch.bool,device=image.device), ~valid_tokens], dim=1)
        tokens = torch.cat([self.cls_token.expand(len(image),-1,-1), tokens], dim=1) + self.position
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        return {"logits": self.head(encoded[:,0]), "active_token_counts": valid_tokens.sum(1),
                "dense_token_count_including_cls": self.token_count+1}


def build_baseline(name, config, input_shape):
    if name == "resnet50_3d":
        return ResNet50Volume(base_width=config["cnn_width"])
    if name == "vit3d":
        return DenseViT3D(input_shape=tuple(input_shape), patch_size=config["patch_size"],
                          dimension=config["dimension"], layers=config["layers"], heads=config["heads"], dropout=config["dropout"])
    raise ValueError("Unknown baseline architecture")
