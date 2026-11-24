"""Class-conditioned population variance across verified training-generator risks."""

import torch


def generator_risk_variance(losses,targets,generator_ids,generator_verified,*,training_generator_ids,
                            partition,min_group_samples=1):
    if partition!='train':raise ValueError("Generator-risk loss may use training generators only")
    allowed=tuple(training_generator_ids)
    if (not allowed or len(allowed)!=len(set(allowed)) or any(isinstance(g,bool) or not isinstance(g,int) or g<0 for g in allowed)
            or isinstance(min_group_samples,bool) or not isinstance(min_group_samples,int) or min_group_samples<1):
        raise ValueError("Declare unique nonnegative training-generator IDs and minimum group support")
    if (losses.ndim!=1 or not losses.is_floating_point() or not torch.isfinite(losses).all()
            or any(t.shape!=losses.shape or t.device!=losses.device for t in (targets,generator_ids,generator_verified))
            or targets.dtype!=torch.long or generator_ids.dtype!=torch.long or generator_verified.dtype!=torch.bool
            or ((targets<0)|(targets>3)).any()):raise ValueError("Invalid generator-risk tensors")
    known=torch.zeros_like(generator_verified)
    for identity in allowed:known|=generator_ids==identity
    if (generator_verified&~known).any():raise ValueError("Verified generator is outside the training allowlist")
    zero=losses[:0].sum();terms=[];details={}
    for label,name in ((2,'FB'),(3,'FM')):
        means=[];counts={}
        for identity in allowed:
            selected=(targets==label)&generator_verified&(generator_ids==identity)
            count=int(selected.sum());counts[str(identity)]=count
            if count>=min_group_samples:means.append(losses[selected].mean())
        enabled=len(means)>=2
        value=torch.stack(means).var(unbiased=False) if enabled else zero
        terms.append(value)
        details[name]={"loss":value,"enabled":enabled,"eligible_groups":len(means),"counts":counts,
            "reason":"active" if enabled else "fewer_than_two_verified_training_groups_with_support"}
    return {"loss":sum(terms)/2.,"classes":details,"enabled_classes":sum(item['enabled'] for item in details.values()),
            "formulation":"equal_group_population_variance_within_FB_and_FM_fixed_class_denominator",
            "partition":"train","training_generator_ids":list(allowed)}
