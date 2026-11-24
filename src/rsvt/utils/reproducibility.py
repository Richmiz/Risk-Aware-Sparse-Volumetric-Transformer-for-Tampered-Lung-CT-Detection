"""Deterministic engineering runs and safe primitive/tensor RNG serialization."""

import os
import random
import numpy as np
import torch


def seed_everything(seed, threads=2):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    state = np.random.get_state()
    return {"python":random.getstate(), "numpy":{"algorithm":state[0],"keys":state[1].tolist(),
            "position":state[2],"has_gauss":state[3],"cached_gaussian":state[4]},
            "torch_cpu":torch.get_rng_state(),
            "torch_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["algorithm"],np.asarray(n["keys"],dtype=np.uint32),n["position"],n["has_gauss"],n["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] and torch.cuda.is_available():
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs from checkpoint")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
