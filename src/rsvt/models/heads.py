"""Four-class head and blockwise voxel logits from the fine token grid."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class ClassificationHead(nn.Module):
    def __init__(self,dimension):
        super().__init__();self.linear=nn.Linear(dimension,4)
    def forward(self,cls):return self.linear(cls)


class PatchLocalizationHead(nn.Module):
    """Each fine token labels its exact voxel support; no interpolated subpatch precision."""
    def __init__(self,dimension,patch_size):
        super().__init__();self.linear=nn.Linear(dimension,1);self.patch_size=patch_size
    def forward(self,encoded,scale_ids,presence,token_valid,grid_shapes,shapes,valid_mask):
        maps,patch_logits,patch_valid=[],[],[]
        batch_shape=valid_mask.shape[-3:]
        for index,shape in enumerate(shapes):
            select=(scale_ids[index]==1)&presence[index]
            logits=self.linear(encoded[index,select]).squeeze(-1)
            valid=token_valid[index,select]
            logits=torch.where(valid,logits,torch.zeros_like(logits))
            patch_logits.append(logits);patch_valid.append(valid)
            grid=tuple(int(n) for n in grid_shapes[index,1])
            field=logits.reshape(1,1,*grid)
            for axis in (2,3,4):field=field.repeat_interleave(self.patch_size,dim=axis)
            z,y,x=shape
            field=field[:,:,:z,:y,:x]
            field=F.pad(field,(0,batch_shape[2]-x,0,batch_shape[1]-y,0,batch_shape[0]-z))
            maps.append(field)
        spatial=torch.cat(maps,0)
        spatial=torch.where(valid_mask,spatial,torch.zeros_like(spatial))
        return {"spatial_logits":spatial,"spatial_valid_mask":valid_mask,
                "fine_token_logits":pad_sequence(patch_logits,batch_first=True),
                "fine_token_valid_mask":pad_sequence(patch_valid,batch_first=True),
                "spatial_output_scope":"unvalidated_blockwise_fine_patch_logits_no_subpatch_localization_claim"}
