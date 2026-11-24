"""Deterministic soft attention-prior training and hard top-k inference packing."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def validate_cap(cap):
    if isinstance(cap,bool) or not isinstance(cap,int) or cap<1:
        raise ValueError("Budget must be a positive integer sequence cap including CLS")


def keep_fraction_logit(fraction):
    """Bias that makes every initial gate equal `fraction` (before input-dependent terms)."""
    if isinstance(fraction,bool) or not isinstance(fraction,(int,float)) or not math.isfinite(fraction) or not 0<fraction<1:
        raise ValueError("Initial keep fraction must be in (0,1)")
    return math.log(fraction/(1.-fraction))


class ContextTokenRouter(nn.Module):
    def __init__(self,dimension=64,hidden=32,temperature=1.,initial_keep_fraction=None):
        super().__init__()
        if isinstance(hidden,bool) or not isinstance(hidden,int) or hidden<1:
            raise ValueError("Router width must be a positive integer")
        if isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or temperature<=0:
            raise ValueError("Temperature must be positive finite")
        self.temperature=temperature
        self.score=nn.Sequential(nn.Linear(2*dimension+5,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.initial_keep_fraction=initial_keep_fraction
        if initial_keep_fraction is not None:
            # Start at the token budget instead of ~50% of every candidate. With
            # default zero-ish logits the expected count starts ~8x the cap and the
            # dual multiplier dominates the classification gradient.
            nn.init.zeros_(self.score[-1].weight)
            nn.init.constant_(self.score[-1].bias,keep_fraction_logit(initial_keep_fraction)*temperature)

    def forward(self,embeddings,cues,valid_fraction,valid,scale_ids):
        coarse=valid&(scale_ids==0);candidate=valid&(scale_ids>0)
        context=(embeddings*coarse[...,None]).sum(1)/coarse.sum(1,keepdim=True).clamp_min(1)
        features=torch.cat((embeddings,cues,valid_fraction[...,None],context[:,None,:].expand_as(embeddings)),-1)
        logits=self.score(features).squeeze(-1)/self.temperature
        probabilities=torch.where(candidate,logits.sigmoid(),torch.zeros_like(logits))
        log_bias=torch.where(candidate,F.logsigmoid(logits),torch.zeros_like(logits))
        return {"logits":logits,"probabilities":probabilities,"key_log_bias":log_bias,
                "candidate_mask":candidate,"coarse_mask":coarse,
                "expected_sequence_counts":1.+coarse.sum(1)+probabilities.sum(1)}


def hard_selection(logits,valid,scale_ids,sequence_cap,*,all_valid=False):
    validate_cap(sequence_cap)
    if (logits.ndim!=2 or valid.shape!=logits.shape or scale_ids.shape!=logits.shape or valid.dtype!=torch.bool
            or not torch.isfinite(logits[valid]).all()):raise ValueError("Finite routing scores and matching validity/scales required")
    indices=[]
    for row in range(len(valid)):
        coarse=(valid[row]&(scale_ids[row]==0)).nonzero().flatten()
        candidates=(valid[row]&(scale_ids[row]>0)).nonzero().flatten()
        mandatory=len(coarse)+1
        if not all_valid and sequence_cap<mandatory:
            raise ValueError(f"Budget {sequence_cap} below mandatory coarse+CLS floor {mandatory} for case {row}")
        count=len(candidates) if all_valid else min(len(candidates),sequence_cap-mandatory)
        rank=torch.argsort(logits[row,candidates],descending=True,stable=True)[:count]
        indices.append(torch.cat((coarse,candidates[rank])).sort().values)
    return indices


def pack_tokens(tokens,indices):
    """Gather before attention; sequence padding reaches only the selected maximum."""
    packed={key:pad_sequence([value[i].index_select(0,chosen) for i,chosen in enumerate(indices)],
                            batch_first=True,padding_value=-1 if key=="scale_ids" else 0)
            for key,value in tokens.items()}
    lengths=torch.tensor([len(row) for row in indices],device=indices[0].device)
    presence=torch.arange(int(lengths.max()),device=lengths.device)[None]<lengths[:,None]
    return packed,presence,pad_sequence(indices,batch_first=True,padding_value=-1)
