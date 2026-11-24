"""Shared local feature stem and dense multi-scale candidate grids."""

import torch
from torch import nn
from torch.nn import functional as F
from rsvt.models.anomaly_features import ForensicCues
from rsvt.models.coarse_encoder import ChannelNorm3D
from rsvt.models.geometry import grid_shape,masked_max_pool_tokens,masked_pool_tokens,pad_volume,patch_layout,power_of_two,sum_pool

POOLING=("mean","max_mean")

TOKEN_FIELDS=("embeddings","valid","scale_ids","centers_voxel_xyz","centers_lps_mm","starts_xyz",
              "ends_xyz","size_mm_xyz","position_features","cues","valid_fraction")


def merge_packets(packets):
    return {key:torch.cat([packet[key] for packet in packets],0) for key in TOKEN_FIELDS}


class MultiscaleTokenizer(nn.Module):
    def __init__(self,dimension=64,width=16,patch_sizes=(8,16,32),pooling="mean"):
        super().__init__()
        if len(patch_sizes)!=3 or list(patch_sizes)!=sorted(set(patch_sizes)) or not all(power_of_two(p) for p in patch_sizes):
            raise ValueError("Use three increasing power-of-two patch sizes in [4,64]")
        if pooling not in POOLING:raise ValueError("Token pooling must be mean or max_mean")
        self.patch_sizes=tuple(patch_sizes);self.cue_extractor=ForensicCues();self.pooling=pooling
        self.first=nn.Conv3d(5,width,3,padding=1);self.norm1=ChannelNorm3D(width)
        self.second=nn.Conv3d(width,dimension,3,stride=2,padding=1);self.norm2=ChannelNorm3D(dimension)
        inputs=dimension*(2 if pooling=="max_mean" else 1)
        self.projections=nn.ModuleList(nn.Linear(inputs,dimension) for _ in patch_sizes)

    def forward(self,image,valid,affine):
        shape=tuple(image.shape[-3:]);cues=self.cue_extractor(image,valid)
        padded,mask=pad_volume(torch.cat((image,cues),1),valid,max(self.patch_sizes))
        features=F.gelu(self.norm1(self.first(padded)))*mask
        support=F.max_pool3d(mask.float(),2,2)>0
        features=F.gelu(self.norm2(self.second(features)))*support
        # Cue summaries use raw-grid validity; learned features use stride-two support.
        cue_padded,_=pad_volume(cues,valid,max(self.patch_sizes))
        packets=[]
        for index,(patch,projection) in enumerate(zip(self.patch_sizes,self.projections)):
            grid=grid_shape(shape,patch);z,y,x=grid
            layout=patch_layout(shape,patch,affine)
            counts=sum_pool(mask,patch)[0,0,:z,:y,:x].flatten()
            volumes=(layout["ends_xyz"]-layout["starts_xyz"]).prod(1)
            pooled=masked_pool_tokens(features,support,patch//2,grid)
            if self.pooling=="max_mean":
                pooled=torch.cat((pooled,masked_max_pool_tokens(features,support,patch//2,grid)),-1)
            embeddings=projection(pooled)
            packets.append({**layout,"embeddings":embeddings,"valid":counts>0,
                "scale_ids":torch.full((len(counts),),index+1,dtype=torch.long,device=image.device),
                "cues":masked_pool_tokens(cue_padded,mask,patch,grid),"valid_fraction":counts/volumes})
        return merge_packets(packets),[p["grid_shape_zyx"] for p in packets],int(features[0,0].numel())
