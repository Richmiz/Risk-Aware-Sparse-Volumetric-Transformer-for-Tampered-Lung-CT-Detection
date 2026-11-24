"""Opt-in AdamW grouping and warmup/cosine scheduling, shared by baseline/RSVT."""

import math
import torch


def validate_optimization(config):
    policy = config.get("lr_schedule", "cosine")
    if policy not in {"cosine", "warmup_cosine"}:
        raise ValueError("Unknown learning-rate schedule")
    warmup = config.get("warmup_epochs", 0)
    epochs = config["epochs"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup < epochs:
        raise ValueError("Warmup must be an integer below the epoch budget")
    if policy == "cosine" and warmup:
        raise ValueError("Warmup requires warmup_cosine")
    for key, default, lower_inclusive in (("warmup_start_ratio", .1, False), ("min_lr_ratio", 0., True)):
        value = config.get(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value > 1 or value < 0 or not lower_inclusive and value == 0):
            raise ValueError("Invalid optimization ratio: " + key)
    if not isinstance(config.get("exclude_norm_bias_decay", False), bool):
        raise ValueError("exclude_norm_bias_decay must be bool")


def build_optimizer(model, config):
    """Keep parameters in the optimizer even when RSVT stages temporarily freeze them."""
    if not config.get("exclude_norm_bias_decay", False):
        return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        # Normalization/bias plus learned positional and CLS embeddings are not
        # convolution/linear weights. Use names AND dimensionality for tokens.
        exempt = (parameter.ndim <= 1 or name.endswith('position') or 'cls_token' in name
                  or name.endswith('scale_embedding.weight'))
        (no_decay if exempt else decay).append(parameter)
    groups = [{"params": decay, "weight_decay": config["weight_decay"], "group_name": "weights"},
              {"params": no_decay, "weight_decay": 0., "group_name": "norm_bias_position"}]
    return torch.optim.AdamW(groups, lr=config["learning_rate"])


def build_scheduler(optimizer, config):
    validate_optimization(config)
    if config.get("lr_schedule", "cosine") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"],
            eta_min=config["learning_rate"] * config.get("min_lr_ratio", 0.))
    warmup = config.get("warmup_epochs", 0)
    start, floor = config.get("warmup_start_ratio", .1), config.get("min_lr_ratio", 0.)
    def multiplier(epoch):
        if warmup and epoch < warmup:
            return start + (1. - start) * epoch / warmup
        progress = min(1., max(0., (epoch - warmup) / (config["epochs"] - warmup)))
        return floor + (1. - floor) * (1. + math.cos(math.pi * progress)) / 2.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def optimization_report(optimizer, config):
    return {"optimizer": "AdamW", "lr_schedule": config.get("lr_schedule", "cosine"),
            "warmup_epochs": config.get("warmup_epochs", 0), "base_learning_rate": config["learning_rate"],
            "min_lr_ratio": config.get("min_lr_ratio", 0.),
            "parameter_groups": [{"name": g.get("group_name", "all"), "weight_decay": g["weight_decay"],
                                  "parameter_count": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups]}
