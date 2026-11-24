"""F07 staged step engine. Medical data/selection orchestration remains separate."""

import math
import torch
from rsvt.experiments.phase5 import _plain
from rsvt.losses.combined import risk_objective


def configure_stage(model,variant,stage,*,localization_active):
    if variant not in ('B4_risk_only','B5_full') or stage not in ('baseline','routing_warmup','risk_finetune'):
        raise ValueError("Unknown F07 variant/stage")
    if variant=='B4_risk_only' and stage=='routing_warmup':raise ValueError("B4 has no routing warmup")
    model.train();model.requires_grad_(True)
    if stage=='routing_warmup':model.requires_grad_(False);model.router.requires_grad_(True)
    elif variant=='B4_risk_only' or stage=='baseline':model.router.requires_grad_(False)
    if model.localizer is not None and not localization_active:model.localizer.requires_grad_(False)
    return {"routing_mode":"dense" if variant=='B4_risk_only' or stage=='baseline' else "soft",
            "risk_enabled":stage=='risk_finetune',"dual_enabled":variant=='B5_full' and stage!='baseline'}


def risk_training_step(model,optimizer,batch,class_weights,*,variant,stage,loss_config,dual,gradient_clip_norm=5.):
    if isinstance(gradient_clip_norm,bool) or not isinstance(gradient_clip_norm,(float,int)) or not math.isfinite(gradient_clip_norm) or gradient_clip_norm<=0:
        raise ValueError("Gradient clip norm must be positive finite")
    localization=batch.get('localization')
    loc_active=localization is not None and bool(localization['annotation_verified'].any()) and loss_config['localization_weight']>0
    settings=configure_stage(model,variant,stage,localization_active=loc_active)
    # Optimizer owns all parameters across stages. Clearing stale grads prevents
    # AdamW from updating a parameter frozen for the current stage.
    optimizer.zero_grad(set_to_none=True)
    frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
    router_before=model.router.score[0].weight.detach().clone()
    image=batch['image'].detach().requires_grad_(True)
    output=model(image,batch['valid_mask'],batch['voxel_to_lps'],padding_mask=batch['padding_mask'],routing_mode=settings['routing_mode'])
    objective=risk_objective(output,batch['labels'],class_weights,loss_config=loss_config,
        generator_ids=batch['generator_ids'],generator_verified=batch['generator_verified'],training_generator_ids=batch['training_generator_ids'],
        risk_enabled=settings['risk_enabled'],localization=localization,dual=dual if settings['dual_enabled'] else None,sequence_cap=model.sequence_cap)
    risk_gradients={}
    if settings['risk_enabled'] and settings['routing_mode']=='soft':
        for name,term in (('cvar',objective['cvar']['loss']),('generator',objective['generator']['loss'])):
            grad=torch.autograd.grad(term,model.router.score[0].weight,retain_graph=True,allow_unused=True)[0]
            active=(any(item['count']>0 for item in objective['cvar']['classes'].values()) and loss_config['cvar_weight']>0) if name=='cvar' else (objective['generator']['enabled_classes']>0 and loss_config['generator_weight']>0)
            risk_gradients[name]={"active":active,"present_finite":grad is not None and bool(torch.isfinite(grad).all()),
                                  "l2_norm":float(grad.detach().double().norm()) if grad is not None else 0.}
    if not torch.isfinite(objective['loss']):raise ValueError("Nonfinite F07 objective before optimizer step")
    objective['loss'].backward()
    groups={}
    modules={'coarse':model.coarse_encoder,'candidate':model.tokenizer,'position':model.position,'transformer':model.transformer,
             'classifier':model.classifier,'router':model.router}
    if model.localizer is not None:modules['localization']=model.localizer
    for name,module in modules.items():
        parameters=[p for p in module.parameters() if p.requires_grad]
        if not parameters:groups[name]={"active":False,"all_present_finite":True,"l2_norm":0.};continue
        grads=[p.grad for p in parameters]
        groups[name]={"active":True,"all_present_finite":all(g is not None and bool(torch.isfinite(g).all()) for g in grads),
                      "l2_norm":math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))}
    if not all(g['all_present_finite'] for g in groups.values()):raise ValueError("Nonfinite or missing active-parameter gradient; weights not updated")
    before_clip=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],gradient_clip_norm,error_if_nonfinite=True))
    optimizer.step()
    dual_before=dual.state_dict()
    if settings['dual_enabled']:dual.update(objective['budget']['constraint'])
    unchanged=all(torch.equal(frozen[name],p.detach()) for name,p in model.named_parameters() if name in frozen)
    row={"variant":variant,"stage":stage,**settings,"loss":float(objective['loss'].detach()),
        "classification":float(objective['classification'].detach()),"cvar":_plain(objective['cvar']),
        "generator":_plain(objective['generator']),"localization":_plain(objective['localization']),
        "budget":_plain(objective['budget']),"dual_before":dual_before,"dual_after":dual.state_dict(),
        "gradient_groups":groups,"risk_gradients_to_router":risk_gradients,"gradient_norm_before_clip":before_clip,
        "gradient_clipped":before_clip>gradient_clip_norm,"frozen_parameters_unchanged":unchanged,
        "router_weight_changed":not torch.equal(router_before,model.router.score[0].weight.detach()),
        "invalid_image_gradient_zero":image.grad is not None and bool((image.grad[~batch['valid_mask']]==0).all()),
        "expected_sequence_counts":_plain(output['routing'].get('expected_sequence_counts')),
        "actual_training_sequence":output['compute']['actual_sequence_length_including_cls'],
        "training_remains_dense":output['routing']['mode'] in ('soft','dense')}
    if not unchanged:raise ValueError("A frozen parameter changed unexpectedly")
    return row
