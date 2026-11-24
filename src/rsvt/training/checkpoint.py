"""Checksummed PyTorch state dictionaries with strict run-contract binding."""

import json
from pathlib import Path
import torch
from rsvt.data.acquisition import sha256_file


def save_checkpoint(path: Path, state: dict):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    path.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open("wb") as stream:
        torch.save(state,stream)
    temporary.replace(path)
    receipt = {"sha256":sha256_file(path), "bytes":path.stat().st_size,
               "contract_sha256":state["contract_sha256"], "epoch":state["epoch"]}
    target = path.with_name(path.name+".json")
    temporary = target.with_name(target.name+".tmp")
    temporary.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    temporary.replace(target)
    return receipt


def load_checkpoint(path: Path, contract_sha256: str, *, expected_format="f04_state_v1"):
    path = Path(path)
    receipt = json.loads(path.with_name(path.name+".json").read_text(encoding="utf-8"))
    if (path.is_symlink() or receipt["contract_sha256"] != contract_sha256
            or path.stat().st_size != receipt["bytes"] or sha256_file(path) != receipt["sha256"]):
        raise ValueError("Checkpoint checksum or run contract differs")
    state = torch.load(path,map_location="cpu",weights_only=True)
    if state.get("format_version") != expected_format or state["contract_sha256"] != contract_sha256:
        raise ValueError("Unsupported or mismatched checkpoint contract")
    return state
