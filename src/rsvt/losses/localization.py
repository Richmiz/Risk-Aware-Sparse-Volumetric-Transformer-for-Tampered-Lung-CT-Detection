"""Verified voxel-mask Dice/Focal gate. Points/boxes are not segmentation targets."""

import math
import torch
from torch.nn import functional as F


def verified_voxel_mask_loss(logits,targets,valid_mask,annotation_verified,*,annotation_kind,
                             focal_alpha=.25,focal_gamma=2.,dice_weight=1.,focal_weight=1.):
    if annotation_kind!="voxel_mask":raise ValueError("Dice/Focal requires verified voxel_mask annotations, not points/boxes")
    if logits.ndim!=5 or logits.shape[1]!=1 or targets.shape!=logits.shape or valid_mask.shape!=logits.shape:
        raise ValueError("Localization logits, voxel targets, and validity require matching B/1/Z/Y/X")
    if (valid_mask.dtype!=torch.bool or annotation_verified.dtype!=torch.bool or annotation_verified.shape!=(len(logits),)
            or any(t.device!=logits.device for t in (targets,valid_mask,annotation_verified))):
        raise ValueError("Mask/verification flags must be bool and all tensors on the logits device")
    for value in (focal_alpha,focal_gamma,dice_weight,focal_weight):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError("Invalid localization loss scalar")
    if not 0<focal_alpha<1 or focal_gamma<0 or min(dice_weight,focal_weight)<0 or dice_weight+focal_weight==0:
        raise ValueError("Invalid Dice/Focal loss weights")
    dice,focal=[],[]
    for index in range(len(logits)):
        if not annotation_verified[index] or not valid_mask[index].any():continue
        selected=logits[index][valid_mask[index]].float()
        truth=targets[index][valid_mask[index]].float()
        if (not torch.isfinite(selected).all() or not torch.isfinite(truth).all()
                or not ((truth==0)|(truth==1)).all()):
            raise ValueError("Verified voxel masks must be finite binary values and logits finite")
        probability=selected.sigmoid()
        dice.append(1.-(2.*(probability*truth).sum()+1e-6)/(probability.sum()+truth.sum()+1e-6))
        pt=probability*truth+(1.-probability)*(1.-truth)
        alpha=focal_alpha*truth+(1.-focal_alpha)*(1.-truth)
        focal.append((alpha*(1.-pt).pow(focal_gamma)*F.binary_cross_entropy_with_logits(selected,truth,reduction="none")).mean())
    zero=logits.reshape(-1)[:0].sum()
    dice_value=torch.stack(dice).mean() if dice else zero
    focal_value=torch.stack(focal).mean() if focal else zero
    return {"loss":dice_weight*dice_value+focal_weight*focal_value,"dice":dice_value,"focal":focal_value,
            "eligible_cases":len(dice),"annotation_kind":"voxel_mask"}
