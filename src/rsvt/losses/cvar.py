"""Fractional empirical upper-tail CVaR, conditioned separately on FB and FM."""

import math
import torch


def empirical_cvar(losses,alpha=.9):
    if (losses.ndim!=1 or not losses.is_floating_point() or not torch.isfinite(losses).all()
            or isinstance(alpha,bool) or not isinstance(alpha,(float,int)) or not math.isfinite(alpha) or not 0<=alpha<1):
        raise ValueError("CVaR requires a finite loss vector and confidence alpha in [0,1)")
    if not losses.numel():return losses.sum()
    mass=len(losses)*(1.-alpha)
    ordered=torch.sort(losses,descending=True,stable=True).values
    if mass<=1.:return ordered[0]
    whole=math.floor(mass);fraction=mass-whole
    result=ordered[:whole].sum()
    if fraction>0 and whole<len(losses):result=result+fraction*ordered[whole]
    return result/mass


def class_conditional_cvar(losses,targets,*,alpha=.9,fb_weight=2.,fm_weight=1.,min_class_samples=4):
    if targets.shape!=losses.shape or targets.dtype!=torch.long or targets.device!=losses.device or ((targets<0)|(targets>3)).any():
        raise ValueError("CVaR targets must be matching long class indices 0..3")
    if (isinstance(min_class_samples,bool) or not isinstance(min_class_samples,int) or min_class_samples<1
            or any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<=0 for w in (fb_weight,fm_weight))):
        raise ValueError("Positive class weights and integer minimum support required")
    # Validate alpha even when both requested classes are absent.
    empirical_cvar(losses,alpha)
    terms=[];details={}
    for label,name,weight in ((2,'FB',fb_weight),(3,'FM',fm_weight)):
        selected=losses[targets==label];count=len(selected)
        value=empirical_cvar(selected,alpha);terms.append(weight*value)
        details[name]={"loss":value,"count":count,"tail_mass":count*(1.-alpha),"missing":count==0,
            "low_support":count<min_class_samples,"max_only":0<count*(1.-alpha)<=1.}
    # Fixed denominator: an absent class contributes differentiable zero, without
    # silently amplifying the other class's coefficient when batch composition changes.
    return {"loss":sum(terms)/(fb_weight+fm_weight),"classes":details,"alpha_confidence":alpha,
            "normalization":"fixed_sum_requested_FB_FM_weights","estimator":"fractional_empirical_upper_tail"}
