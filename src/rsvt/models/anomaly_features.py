"""Deterministic intensity/validity cues; not validated forensic detectors."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class ForensicCues(nn.Module):
    names=("absolute_local_residual","adjacent_tensor_z_difference","local_std","neighborhood_valid_fraction")
    def forward(self,image,valid):
        clean=torch.where(valid,image,torch.zeros_like(image))
        # Explicit padding also supports one/two-voxel spatial dimensions.
        def neighborhood_sum(value):
            return F.avg_pool3d(F.pad(value,(1,1,1,1,1,1)),3,1,divisor_override=1)
        count=neighborhood_sum(valid.float())
        mean=neighborhood_sum(clean)/count.clamp_min(1.)
        square=neighborhood_sum(clean.square())/count.clamp_min(1.)
        std=((square-mean.square()).clamp_min(0)+1e-6).sqrt()-math.sqrt(1e-6)
        pairs=valid[:,:,1:]&valid[:,:,:-1]
        delta=(clean[:,:,1:]-clean[:,:,:-1]).abs()*pairs
        difference=F.pad(delta,(0,0,0,0,0,1))+F.pad(delta,(0,0,0,0,1,0))
        degree=F.pad(pairs.float(),(0,0,0,0,0,1))+F.pad(pairs.float(),(0,0,0,0,1,0))
        result=torch.cat(((clean-mean).abs(),difference/degree.clamp_min(1.),std,count/27.),1)
        return torch.where(valid.expand_as(result),result,torch.zeros_like(result))
