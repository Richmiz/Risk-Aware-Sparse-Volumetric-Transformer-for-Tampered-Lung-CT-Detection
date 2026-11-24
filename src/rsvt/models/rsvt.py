"""RSVT F05 dense foundation: mandatory coarse context and every multiscale candidate."""

import math
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from rsvt.models.coarse_encoder import CoarseContextEncoder
from rsvt.models.geometry import grid_shape,power_of_two,validate_inputs
from rsvt.models.heads import ClassificationHead,PatchLocalizationHead
from rsvt.models.tokenization import MultiscaleTokenizer,TOKEN_FIELDS,merge_packets
from rsvt.models.transformer import DenseVolumetricTransformer,PhysicalPositionEncoding

CLASS_INDEX={"TB":0,"TM":1,"FB":2,"FM":3}


class RSVTDenseFoundation(nn.Module):
    def __init__(self,dimension=64,heads=4,layers=2,coarse_width=8,candidate_width=16,
                 coarse_patch_size=32,patch_sizes=(8,16,32),dropout=0.,localization_head=True,
                 max_tokens=2048,max_batch_voxels=2097152,max_attention_elements=33554432,
                 max_feature_grid_voxels=8388608,token_pooling="mean"):
        super().__init__()
        integers={"dimension":dimension,"heads":heads,"layers":layers,"coarse_width":coarse_width,
                  "candidate_width":candidate_width,"max_tokens":max_tokens,"max_batch_voxels":max_batch_voxels,
                  "max_attention_elements":max_attention_elements,"max_feature_grid_voxels":max_feature_grid_voxels}
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in integers.values()):
            raise ValueError("Model sizes/bounds must be positive integers")
        if dimension<4 or dimension%heads or not power_of_two(coarse_patch_size) or max_tokens>4096:
            raise ValueError("Invalid dimension, coarse grid, or dense token bound")
        if (len(patch_sizes)!=3 or list(patch_sizes)!=sorted(set(patch_sizes))
                or not all(power_of_two(n) for n in patch_sizes)):
            raise ValueError("Three increasing power-of-two candidate patches are required")
        if isinstance(dropout,bool) or not isinstance(dropout,(int,float)) or not math.isfinite(dropout) or not 0<=dropout<1:
            raise ValueError("Dropout must be finite in [0,1)")
        if not isinstance(localization_head,bool):raise ValueError("localization_head must be bool")
        self.patch_sizes=tuple(patch_sizes);self.coarse_patch_size=coarse_patch_size
        self.max_tokens=max_tokens;self.max_batch_voxels=max_batch_voxels
        self.heads=heads;self.layers=layers;self.max_attention_elements=max_attention_elements
        self.max_feature_grid_voxels=max_feature_grid_voxels
        self.coarse_encoder=CoarseContextEncoder(dimension,coarse_width,coarse_patch_size)
        self.tokenizer=MultiscaleTokenizer(dimension,candidate_width,self.patch_sizes,token_pooling)
        self.position=PhysicalPositionEncoding(dimension)
        self.transformer=DenseVolumetricTransformer(dimension,heads,layers,dropout)
        self.classifier=ClassificationHead(dimension)
        self.localizer=PatchLocalizationHead(dimension,self.patch_sizes[0]) if localization_head else None

    def forward(self,image,valid_mask,voxel_to_lps,*,padding_mask=None):
        shapes,affines=validate_inputs(image,valid_mask,voxel_to_lps,padding_mask,self.max_batch_voxels)
        counts=[sum(math.prod(grid_shape(shape,p)) for p in (self.coarse_patch_size,*self.patch_sizes)) for shape in shapes]
        sequence_length=max(counts)+1
        attention_slots=len(image)*self.heads*sequence_length**2
        # Bound padded stem grids too: tiny original boxes may pad to large cubes.
        feature_voxels_bound=sum(math.prod(grid_shape(shape,self.coarse_patch_size))*(self.coarse_patch_size**3//8+self.coarse_patch_size**3//64)
            +math.prod(grid_shape(shape,max(self.patch_sizes)))*(max(self.patch_sizes)**3+max(self.patch_sizes)**3//8)
            for shape in shapes)
        if (sequence_length>self.max_tokens or attention_slots>self.max_attention_elements
                or feature_voxels_bound>self.max_feature_grid_voxels):
            raise ValueError("Dense token/attention budget exceeded before feature construction")
        packets,grids,coarse_counts,feature_voxels=[],[],[],[]
        for index,(z,y,x) in enumerate(shapes):
            valid=valid_mask[index:index+1,:,:z,:y,:x]
            value=image[index:index+1,:,:z,:y,:x]
            clean=torch.where(valid,value,torch.zeros_like(value))
            coarse=self.coarse_encoder(clean,valid,affines[index])
            candidates,candidate_grids,local_voxels=self.tokenizer(clean,valid,affines[index])
            packets.append(merge_packets([coarse,candidates]))
            grids.append([coarse["grid_shape_zyx"],*candidate_grids])
            coarse_counts.append(len(coarse["valid"]))
            feature_voxels.append([coarse["feature_voxels"],local_voxels])
        tokens={key:pad_sequence([p[key] for p in packets],batch_first=True,
                                padding_value=-1 if key=="scale_ids" else 0) for key in TOKEN_FIELDS}
        lengths=torch.tensor(counts,device=image.device,dtype=torch.long)
        presence=torch.arange(max(counts),device=image.device)[None,:]<lengths[:,None]
        embeddings=tokens["embeddings"]+self.position(tokens["position_features"],tokens["scale_ids"],presence)
        cls,encoded=self.transformer(embeddings,tokens["valid"])
        grid_tensor=torch.tensor(grids,device=image.device,dtype=torch.long)
        coarse=(tokens["scale_ids"]==0)&presence
        output={"schema_version":"rsvt_f05_dense_v1","logits":self.classifier(cls),"class_index":dict(CLASS_INDEX),
                "token_features":encoded,"pre_attention_embeddings":embeddings,
                "token_valid_mask":tokens["valid"],"token_presence_mask":presence,
                "token_geometry":{key:tokens[key] for key in ("scale_ids","centers_voxel_xyz","centers_lps_mm",
                    "starts_xyz","ends_xyz","size_mm_xyz","valid_fraction")},
                "candidate_cues":tokens["cues"],"candidate_cue_names":self.tokenizer.cue_extractor.names,
                "candidate_cue_valid_mask":tokens["valid"]&(tokens["scale_ids"]>0),
                "grid_shapes_zyx":grid_tensor,"voxel_to_lps":affines,
                "original_shape_zyx":torch.tensor(shapes,device=image.device,dtype=torch.long),
                "active_token_counts":tokens["valid"].sum(1),
                "compute":{"routing_enabled":False,"coarse_tokens_pruned":0,"saved_attention_slots":0,
                    "constructed_coarse_tokens":torch.tensor(coarse_counts,device=image.device),
                    "constructed_candidate_tokens":lengths-torch.tensor(coarse_counts,device=image.device),
                    "valid_coarse_tokens":(tokens["valid"]&coarse).sum(1),
                    "valid_candidate_tokens":(tokens["valid"]&~coarse).sum(1),
                    "dense_sequence_length_including_cls":sequence_length,
                    "attention_pair_slots_per_layer":attention_slots,"attention_pair_slots_all_layers":attention_slots*self.layers,
                    "padded_stem_grid_voxels_all_stages":feature_voxels_bound,
                    "constructed_feature_grid_voxels_coarse_candidate":torch.tensor(feature_voxels,device=image.device),
                    "scope":"dense_all_candidates_attention_slots_not_FLOPs_or_measured_latency"},
                "training_ready":False,"spatial_logits":None,"spatial_valid_mask":None}
        if self.localizer is not None:
            output.update(self.localizer(encoded,tokens["scale_ids"],presence,tokens["valid"],grid_tensor,shapes,valid_mask))
        return output
