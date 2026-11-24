"""F07 training objective; raw CE supplies CVaR and generator terms exactly once."""

import math
import torch
from torch.nn import functional as F
from rsvt.losses.cvar import class_conditional_cvar
from rsvt.losses.generator import generator_risk_variance
from rsvt.losses.localization import verified_voxel_mask_loss


def risk_objective(output,targets,class_weights,*,loss_config,generator_ids,generator_verified,
                   training_generator_ids,risk_enabled,localization=None,dual=None,sequence_cap=None):
    if not isinstance(risk_enabled,bool) or any(isinstance(loss_config[k],bool) or not isinstance(loss_config[k],(int,float))
        or not math.isfinite(loss_config[k]) or loss_config[k]<0 for k in ('cvar_weight','generator_weight','localization_weight')):
        raise ValueError("Risk enablement must be bool and loss coefficients finite nonnegative")
    logits=output['logits']
    if (logits.ndim!=2 or logits.shape[1]!=4 or not len(logits) or not torch.isfinite(logits).all()
            or targets.shape!=(len(logits),) or targets.dtype!=torch.long or targets.device!=logits.device
            or ((targets<0)|(targets>3)).any() or class_weights.shape!=(4,) or class_weights.device!=logits.device
            or not torch.isfinite(class_weights).all() or (class_weights<=0).any()):raise ValueError("Invalid four-class risk objective inputs")
    values=F.cross_entropy(logits,targets,reduction='none')
    classification=(values*class_weights[targets]).mean()
    cvar=class_conditional_cvar(values,targets,alpha=loss_config['alpha'],fb_weight=loss_config['fb_weight'],
        fm_weight=loss_config['fm_weight'],min_class_samples=loss_config['min_class_samples'])
    generator=generator_risk_variance(values,targets,generator_ids,generator_verified,training_generator_ids=training_generator_ids,
        partition='train',min_group_samples=loss_config['min_group_samples'])
    zero=logits.reshape(-1)[:0].sum()
    loc={"loss":zero,"eligible_cases":0,"reason":"no_voxel_supervision_requested"}
    if localization is not None:
        if output.get('spatial_logits') is None:raise ValueError("Voxel supervision requires an available localization head")
        loc=verified_voxel_mask_loss(output['spatial_logits'],localization['targets'],output['spatial_valid_mask'],
            localization['annotation_verified'],annotation_kind=localization['annotation_kind'])
    budget={"loss":zero,"constraint":zero,"multiplier":0.,"scope":"disabled_for_dense_stage"}
    if dual is not None:
        route=output.get('routing',{})
        if route.get('mode')!='soft':raise ValueError("Training dual requires differentiable soft-routing expected counts")
        budget=dual.terms(route['expected_sequence_counts'],sequence_cap)
    total=classification+loss_config['localization_weight']*loc['loss']+budget['loss']
    if risk_enabled:total=total+loss_config['cvar_weight']*cvar['loss']+loss_config['generator_weight']*generator['loss']
    return {"loss":total,"classification":classification,"per_sample_unweighted_ce":values,
        "cvar":cvar,"generator":generator,"localization":loc,"budget":budget,"risk_terms_active":bool(risk_enabled)}
