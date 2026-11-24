"""Lightweight always-on coarse context; randomly initialized, no lung segmentation."""

import torch
from torch import nn
from torch.nn import functional as F
from rsvt.models.geometry import grid_shape,masked_pool_tokens,pad_volume,patch_layout,power_of_two,sum_pool


class ChannelNorm3D(nn.Module):
    """Per-voxel channel normalization, independent of other spatial positions."""
    def __init__(self,channels):
        super().__init__();self.norm=nn.LayerNorm(channels)
    def forward(self,value):
        return self.norm(value.permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()


class CoarseContextEncoder(nn.Module):
    def __init__(self,dimension=64,width=8,patch_size=32):
        super().__init__()
        if not power_of_two(patch_size):raise ValueError("Coarse patch size must be a power of two in [4,64]")
        self.patch_size=patch_size
        self.first=nn.Conv3d(1,width,3,stride=2,padding=1)
        self.norm1=ChannelNorm3D(width)
        self.second=nn.Conv3d(width,dimension,3,stride=2,padding=1)
        self.norm2=ChannelNorm3D(dimension)
        self.projection=nn.Linear(dimension,dimension)

    def forward(self,image,valid,affine):
        shape=tuple(image.shape[-3:]);image,valid=pad_volume(image,valid,self.patch_size)
        mask1=F.max_pool3d(valid.float(),2,2)>0
        features=F.gelu(self.norm1(self.first(image)))*mask1
        mask2=F.max_pool3d(mask1.float(),2,2)>0
        features=F.gelu(self.norm2(self.second(features)))*mask2
        grid=grid_shape(shape,self.patch_size)
        embeddings=self.projection(masked_pool_tokens(features,mask2,self.patch_size//4,grid))
        layout=patch_layout(shape,self.patch_size,affine)
        z,y,x=grid
        counts=sum_pool(valid,self.patch_size)[0,0,:z,:y,:x].flatten()
        volumes=(layout["ends_xyz"]-layout["starts_xyz"]).prod(1)
        return {**layout,"embeddings":embeddings,"valid":counts>0,
                "scale_ids":torch.zeros(len(counts),dtype=torch.long,device=image.device),
                "valid_fraction":counts/volumes,"cues":embeddings.new_zeros((len(counts),4)),
                "feature_voxels":features.shape[-3]*features.shape[-2]*features.shape[-1]}
