"""F05-compatible dense attention with optional F06 differentiable log-gates."""

import torch
from torch import nn


class PhysicalPositionEncoding(nn.Module):
    def __init__(self,dimension):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(12,dimension),nn.GELU(),nn.Linear(dimension,dimension))
        self.scale_embedding=nn.Embedding(4,dimension)
    def forward(self,features,scale_ids,presence):
        encoded=self.project(features)+self.scale_embedding(scale_ids.clamp_min(0))
        return torch.where(presence[...,None],encoded,torch.zeros_like(encoded))


class DenseVolumetricTransformer(nn.Module):
    def __init__(self,dimension=64,heads=4,layers=2,dropout=0.):
        super().__init__()
        self.cls_token=nn.Parameter(torch.empty(1,1,dimension));nn.init.trunc_normal_(self.cls_token,std=.02)
        # Separate constructors avoid identical cloned layer initialization.
        self.blocks=nn.ModuleList(nn.TransformerEncoderLayer(dimension,heads,dimension*4,dropout,
                     activation="gelu",batch_first=True,norm_first=True) for _ in range(layers))
        self.norm=nn.LayerNorm(dimension)
    def forward(self,tokens,valid,*,key_log_bias=None):
        cls=self.cls_token.to(tokens.dtype).expand(len(tokens),-1,-1)
        sequence=torch.cat((cls,tokens),1)
        mask=torch.cat((torch.ones((len(tokens),1),dtype=torch.bool,device=tokens.device),valid),1)
        attention_bias=None
        if key_log_bias is not None:
            if key_log_bias.shape!=valid.shape or not torch.isfinite(key_log_bias[valid]).all():
                raise ValueError("Finite key log-gates must match token validity")
            bias=torch.cat((key_log_bias.new_zeros((len(tokens),1)),key_log_bias),1)
            bias=bias.masked_fill(~mask,float("-inf"))
            heads=self.blocks[0].self_attn.num_heads;length=sequence.shape[1]
            attention_bias=bias[:,None,None,:].expand(-1,heads,length,-1).reshape(len(tokens)*heads,length,length)
        for block in self.blocks:
            sequence=block(sequence,src_key_padding_mask=~mask) if attention_bias is None else block(sequence,src_mask=attention_bias)
        sequence=self.norm(sequence)
        sequence=torch.where(mask[...,None],sequence,torch.zeros_like(sequence))
        return sequence[:,0],sequence[:,1:]
