"""F06 routing-only RSVT: dense soft training, genuinely packed hard inference."""

import math
import torch
from torch.nn.utils.rnn import pad_sequence

from rsvt.evaluation.efficiency import StageTimer,estimated_operator_flops
from rsvt.models.geometry import grid_shape,validate_inputs
from rsvt.models.routing import ContextTokenRouter,hard_selection,pack_tokens,validate_cap
from rsvt.models.rsvt import CLASS_INDEX,RSVTDenseFoundation
from rsvt.models.tokenization import TOKEN_FIELDS,merge_packets

GEOMETRY_FIELDS=("scale_ids","centers_voxel_xyz","centers_lps_mm","starts_xyz","ends_xyz","size_mm_xyz","valid_fraction")


class RSVTRoutingFoundation(RSVTDenseFoundation):
    def __init__(self,*,router_hidden=32,temperature=1.,sequence_cap=128,router_initial_keep_fraction=None,
                 hard_gate_bias=False,**kwargs):
        if kwargs.get("localization_head",False) is not False:
            raise ValueError("F06 B3 is classification/routing-only; localization is disabled")
        if not isinstance(hard_gate_bias,bool):raise ValueError("hard_gate_bias must be bool")
        kwargs["localization_head"]=False
        super().__init__(**kwargs)
        validate_cap(sequence_cap)
        if sequence_cap>self.max_tokens:raise ValueError("Sequence cap exceeds model resource bound")
        self.sequence_cap=sequence_cap
        # True: hard inference keeps the same log-gate attention bias that soft
        # training used, restricted to the packed tokens (removes train/test drift).
        self.hard_gate_bias=hard_gate_bias
        self.router=ContextTokenRouter(self.classifier.linear.in_features,router_hidden,temperature,
                                       router_initial_keep_fraction)

    def forward(self,image,valid_mask,voxel_to_lps,*,padding_mask=None,routing_mode=None,
                sequence_cap=None,profile_stages=False):
        mode=routing_mode or ("soft" if self.training else "hard")
        if mode not in ("soft","hard","dense","all_valid"):raise ValueError("Unknown routing mode")
        if mode=="hard" and self.training:raise ValueError("Hard packing is inference-only; use eval()")
        cap=self.sequence_cap if sequence_cap is None else sequence_cap
        validate_cap(cap)
        if cap>self.max_tokens:raise ValueError("Sequence cap exceeds model resource bound")
        timer=StageTimer(image.device,profile_stages)
        with timer.stage("validation_and_preflight"):
            shapes,affines=validate_inputs(image,valid_mask,voxel_to_lps,padding_mask,self.max_batch_voxels)
            grids=[[grid_shape(shape,p) for p in (self.coarse_patch_size,*self.patch_sizes)] for shape in shapes]
            counts=[[math.prod(g) for g in row] for row in grids]
            lengths=[sum(row) for row in counts];dense_sequence=max(lengths)+1
            feature_bound=sum(math.prod(grid_shape(shape,self.coarse_patch_size))*(self.coarse_patch_size**3//8+self.coarse_patch_size**3//64)
                +math.prod(grid_shape(shape,max(self.patch_sizes)))*(max(self.patch_sizes)**3+max(self.patch_sizes)**3//8) for shape in shapes)
            if (dense_sequence>self.max_tokens or len(image)*self.heads*dense_sequence**2>self.max_attention_elements
                    or feature_bound>self.max_feature_grid_voxels):raise ValueError("Candidate/dense training resource bound exceeded")
        packets=[]
        for i,(z,y,x) in enumerate(shapes):
            with timer.stage("input_masking"):
                valid=valid_mask[i:i+1,:,:z,:y,:x];value=image[i:i+1,:,:z,:y,:x]
                clean=torch.where(valid,value,torch.zeros_like(value))
            with timer.stage("coarse_encoder"):
                coarse=self.coarse_encoder(clean,valid,affines[i])
            with timer.stage("candidate_encoder_and_cues"):
                candidates,_,_=self.tokenizer(clean,valid,affines[i])
            with timer.stage("geometry_and_batching"):packets.append(merge_packets([coarse,candidates]))
        with timer.stage("geometry_and_batching"):
            tokens={key:pad_sequence([p[key] for p in packets],batch_first=True,
                         padding_value=-1 if key=="scale_ids" else 0) for key in TOKEN_FIELDS}
            presence=torch.arange(max(lengths),device=image.device)[None]<torch.tensor(lengths,device=image.device)[:,None]
        with timer.stage("physical_position"):
            embeddings=tokens["embeddings"]+self.position(tokens["position_features"],tokens["scale_ids"],presence)
            tokens["embeddings"]=embeddings
        with timer.stage("controller_and_packing"):
            valid=tokens["valid"];coarse_mask=valid&(tokens["scale_ids"]==0)
            minimum=coarse_mask.sum(1)+1
            if mode in ("soft","hard") and (minimum>cap).any():
                raise ValueError(f"Sequence budget below mandatory coarse+CLS floor {int(minimum.max())}")
            route=self.router(embeddings,tokens["cues"],tokens["valid_fraction"],valid,tokens["scale_ids"]) if mode in ("soft","hard") else None
            selected=valid.clone()
            source=torch.arange(max(lengths),device=image.device)[None].expand(len(image),-1).clone()
            source=source.masked_fill(~presence,-1)
            packed=tokens
            if mode in ("hard","all_valid"):
                scores=route["logits"] if route else embeddings.new_zeros(valid.shape)
                indices=hard_selection(scores,valid,tokens["scale_ids"],cap,all_valid=mode=="all_valid")
                packed,packed_presence,source=pack_tokens(tokens,indices)
                selected=torch.zeros_like(valid)
                for i,index in enumerate(indices):selected[i,index]=True
            else:packed_presence=presence
        bias=None
        if mode=="soft":bias=route["key_log_bias"]
        elif mode=="hard" and self.hard_gate_bias:
            bias=pad_sequence([route["key_log_bias"][i].index_select(0,chosen) for i,chosen in enumerate(indices)],
                              batch_first=True,padding_value=0.)
        with timer.stage("transformer"):
            cls,encoded=self.transformer(packed["embeddings"],packed["valid"],key_log_bias=bias)
        with timer.stage("classification"):logits=self.classifier(cls)
        with timer.stage("output_metadata"):
            sequence=encoded.shape[1]+1
            slots=len(image)*self.heads*sequence**2
            routing={"mode":mode,"sequence_cap_including_cls":cap,"soft_training_is_dense":mode=="soft",
                "minimum_sequence_counts":minimum,"selected_source_mask":selected,"source_valid_mask":valid,
                "source_geometry":{key:tokens[key] for key in GEOMETRY_FIELDS},
                "source_candidate_mask":valid&(tokens["scale_ids"]>0)}
            if route is not None:routing.update(route)
            output={"schema_version":"rsvt_f06_routing_v1","logits":logits,"class_index":dict(CLASS_INDEX),
                "token_features":encoded,"token_valid_mask":packed["valid"],"token_presence_mask":packed_presence,
                "token_geometry":{key:packed[key] for key in GEOMETRY_FIELDS},"source_token_indices":source,
                "grid_shapes_zyx":torch.tensor(grids,device=image.device),"original_shape_zyx":torch.tensor(shapes,device=image.device),
                "voxel_to_lps":affines,"active_token_counts":packed["valid"].sum(1),"routing":routing,
                "compute":{"routing_enabled":mode in ("soft","hard"),"coarse_tokens_pruned":0,
                    "constructed_coarse_tokens":[row[0] for row in counts],"constructed_candidate_tokens":[sum(row[1:]) for row in counts],
                    "valid_coarse_tokens":coarse_mask.sum(1),"valid_candidate_tokens":(valid&~coarse_mask).sum(1),
                    "selected_candidate_tokens":(selected&~coarse_mask).sum(1),
                    "actual_sequence_length_including_cls":sequence,"dense_sequence_length_including_cls":dense_sequence,
                    "attention_pair_slots_per_layer":slots,"dense_attention_pair_slots_per_layer":len(image)*self.heads*dense_sequence**2,
                    "saved_attention_slots":len(image)*self.heads*dense_sequence**2-slots,
                    "padded_stem_grid_voxels_all_stages":feature_bound,
                    "operator_flops_estimate":estimated_operator_flops(self,shapes,counts,sequence,router_used=route is not None),
                    "scope":"dense_soft_training_or_actual_packed_inference_stems_unchanged"},
                "training_ready":False,"spatial_logits":None,"spatial_valid_mask":None}
        output["stage_ms"]=dict(timer.ms)
        return output
