"""Load composable TOML configurations without deep-learning dependencies."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tomllib
from typing import Any


class ConfigError(ValueError):
    """An experiment configuration is malformed or violates a protocol rule."""


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Resolve parent files relative to each TOML; later values override earlier ones."""
    source = Path(path).resolve()
    if source in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, source))
        raise ConfigError(f"Circular config inheritance: {chain}")
    try:
        with source.open("rb") as stream:
            current = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot load {source}: {exc}") from exc

    parents = current.pop("extends", [])
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise ConfigError("extends must be a list of TOML paths")
    result: dict[str, Any] = {}
    for parent in parents:
        result = _merge(result, load_config(source.parent / parent, (*_stack, source)))
    return _merge(result, current)


def validate_config(config: dict[str, Any]) -> list[str]:
    """Check scaffold protocol fields and report decisions that still require a pilot."""
    required = ("project", "data", "input", "training", "evaluation", "protocol")
    for section in required:
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"Missing or invalid section: {section}")
    for section in ("experiment", "dataset", "model", "loss", "compute"):
        if section in config and not isinstance(config[section], dict):
            raise ConfigError(f"Invalid section: {section}")
    if config["data"].get("split_unit") != "source_volume_id":
        raise ConfigError("Split unit must be source_volume_id")
    if config["data"].get("require_patient_disjoint") is not True:
        raise ConfigError("Patient-disjoint partitioning must be enabled")
    if config["training"].get("selection_data") != "source_validation_only":
        raise ConfigError("Model selection must use source validation data only")
    if config["evaluation"].get("bootstrap_unit") != "source_volume_id":
        raise ConfigError("Bootstrap must resample source CT groups")
    if config["evaluation"].get("primary_metric") != "fb_recall_at_fixed_fpr":
        raise ConfigError("Primary metric must be fb_recall_at_fixed_fpr")
    pending = config["protocol"].get("pending_decisions", [])
    if not isinstance(pending, list) or not all(isinstance(item, str) for item in pending):
        raise ConfigError("protocol.pending_decisions must be a list of strings")
    warnings = [f"Pilot decision pending: {item}" for item in pending]
    experiment = config.get("experiment", {})
    if experiment.get("protocol") == "leave_one_generator_out":
        if experiment.get("held_out_generator_id") in (None, "", "UNSET"):
            warnings.append("Set held_out_generator_id from the audited generator inventory")
    warnings.append("Configuration validation does not establish training readiness")
    return warnings
