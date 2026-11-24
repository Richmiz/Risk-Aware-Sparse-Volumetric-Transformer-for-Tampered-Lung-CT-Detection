"""F07 optional verified-mask head over soft or hard routed features."""

import torch
from torch.nn import functional as F
from rsvt.models.heads import PatchLocalizationHead
from rsvt.models.routed_rsvt import RSVTRoutingFoundation


class RSVTRiskFoundation(RSVTRoutingFoundation):
    def __init__(self,*,localization_head=True,**kwargs):
        if not isinstance(localization_head,bool):raise ValueError("localization_head must be bool")
        super().__init__(localization_head=False,**kwargs)
        self.localizer=PatchLocalizationHead(self.classifier.linear.in_features,self.patch_sizes[0]) if localization_head else None

    def forward(self,image,valid_mask,voxel_to_lps,*,padding_mask=None,routing_mode=None,sequence_cap=None,profile_stages=False):
        out=super().forward(image,valid_mask,voxel_to_lps,padding_mask=padding_mask,routing_mode=routing_mode,
                            sequence_cap=sequence_cap,profile_stages=profile_stages)
        out['schema_version']='rsvt_f07_risk_v1'
        if self.localizer is None:return out
        route=out['routing'];scales=route['source_geometry']['scale_ids'];presence=scales>=0
        selected=route['selected_source_mask'];rows=[];coverage=[]
        for i,shape in enumerate(out['original_shape_zyx'].tolist()):
            valid_source=out['source_token_indices'][i]>=0
            source=out['source_token_indices'][i,valid_source]
            encoded=out['token_features'].new_zeros((scales.shape[1],out['token_features'].shape[-1]))
            rows.append(encoded.index_copy(0,source,out['token_features'][i,valid_source]))
            fine=selected[i,(scales[i]==1)&presence[i]]
            grid=tuple(int(n) for n in out['grid_shapes_zyx'][i,1])
            field=fine.reshape(1,1,*grid)
            for axis in (2,3,4):field=field.repeat_interleave(self.patch_sizes[0],dim=axis)
            z,y,x=shape;field=field[:,:,:z,:y,:x]
            coverage.append(F.pad(field,(0,image.shape[-1]-x,0,image.shape[-2]-y,0,image.shape[-3]-z)))
        predicted_valid=torch.cat(coverage,0)&valid_mask
        out.update(self.localizer(torch.stack(rows),scales,presence,selected,out['grid_shapes_zyx'],
            out['original_shape_zyx'].tolist(),predicted_valid))
        out['spatial_output_scope']='piecewise_fine_patch_logits_only_where_selected_fine_support_is_valid_else_unknown'
        out['compute']['localization_cost_scope']='additional_scatter_head_voxel_expansion_not_in_F06_operator_estimate'
        return out
