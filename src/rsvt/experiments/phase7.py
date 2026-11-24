"""F07 synthetic logical-batch B4/B5 stages; no dataset or holdout reader."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import torch
from torch.nn import functional as F

from rsvt.experiments.phase5 import _plain,coarse_coverage,synthetic_batch
from rsvt.experiments.phase6 import choose_device,weights_digest
from rsvt.losses.budget import ProjectedTokenDual
from rsvt.losses.classification import training_class_weights
from rsvt.losses.generator import generator_risk_variance
from rsvt.models.risk_rsvt import RSVTRiskFoundation
from rsvt.training.trainer import risk_training_step
from rsvt.training.optimization import build_optimizer, build_scheduler, validate_optimization, optimization_report
from rsvt.utils.config import load_config
from rsvt.utils.reproducibility import seed_everything


def load_phase7_config(path):
    c=load_config(path)
    if set(c)-{'optimization'}!={'model','loss','dual','engineering','stages','protocol'} or any(not isinstance(v,dict) for v in c.values()):
        raise ValueError("F07 requires explicit model/loss/dual/engineering/stages/protocol tables")
    model=c['model'];loss=c['loss'];e=c['engineering'];stages=c['stages'];protocol=c['protocol']
    model_fields={'name','dimension','heads','layers','coarse_width','candidate_width','coarse_patch_size','patch_sizes','dropout',
        'localization_head','max_tokens','max_batch_voxels','max_attention_elements','max_feature_grid_voxels','router_hidden','temperature','sequence_cap'}
    if set(model)!=model_fields or model['name']!='rsvt_f07_risk' or model['localization_head'] is not True:
        raise ValueError("Use the explicit F07 model with verified-mask head for synthetic checks")
    flags={'medical_training_ready','project_multi_generator_ready','alpha_source_validation_selected','risk_weights_source_validation_selected'}
    if set(protocol)!=flags|{'status'} or protocol['status']!='synthetic_F07_B4_B5_engineering' or any(protocol[k] is not False for k in flags):
        raise ValueError("F07 must retain synthetic status and pending research-selection flags")
    def integer(v,lo,hi):return isinstance(v,int) and not isinstance(v,bool) and lo<=v<=hi
    def positive(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0
    if set(loss)!={'alpha','fb_weight','fm_weight','min_class_samples','min_group_samples','cvar_weight','generator_weight','localization_weight'}:
        raise ValueError("Unknown/missing risk-loss fields")
    if (isinstance(loss['alpha'],bool) or not isinstance(loss['alpha'],(int,float)) or not math.isfinite(loss['alpha']) or not 0<=loss['alpha']<1
            or not all(positive(loss[k]) for k in ('fb_weight','fm_weight','cvar_weight','generator_weight','localization_weight'))
            or not integer(loss['min_class_samples'],1,1000) or not integer(loss['min_group_samples'],1,1000)):
        raise ValueError("Invalid risk-loss parameters")
    if set(c['dual'])!={'initial','learning_rate','maximum'}:raise ValueError("Unknown dual fields")
    ProjectedTokenDual(**c['dual'])
    if set(e)!={'seed','threads','device','synthetic_shapes_zyx','learning_rate','group_intensity_offset','group_noise_std','gradient_clip_norm'}:
        raise ValueError("Unknown/missing synthetic engineering fields")
    if (not integer(e['seed'],0,2**32-1) or not integer(e['threads'],1,16) or e['device'] not in ('auto','cpu','cuda')
            or not all(positive(e[k]) for k in ('learning_rate','group_intensity_offset','group_noise_std','gradient_clip_norm'))):
        raise ValueError("Invalid F07 engineering parameters")
    shapes=e['synthetic_shapes_zyx']
    if not isinstance(shapes,list) or len(shapes)!=4 or any(not isinstance(s,list) or len(s)!=3 or not all(integer(n,1,96) for n in s) for s in shapes):
        raise ValueError("Declare four bounded procedural shapes")
    if set(stages)!={'baseline_steps','routing_warmup_steps','risk_finetune_steps'} or not all(integer(n,1,20) for n in stages.values()):
        raise ValueError("Bound each synthetic stage to 1..20 steps")
    if 'optimization' in c:
        options=c['optimization']
        if (set(options)-{'weight_decay','lr_schedule','warmup_epochs','warmup_start_ratio','min_lr_ratio','exclude_norm_bias_decay'}
                or isinstance(options.get('weight_decay',0.),bool)
                or not isinstance(options.get('weight_decay',0.),(int,float))
                or not math.isfinite(options.get('weight_decay',0.)) or options.get('weight_decay',0.)<0):
            raise ValueError("Invalid RSVT optimization options")
        validate_optimization({'epochs':stages['baseline_steps']+stages['risk_finetune_steps'],**options})
    return c


def risk_synthetic_batch(shapes,device='cpu',*,offset=.08,noise_std=.01):
    base=synthetic_batch(shapes,device)
    image=base['image'];second=torch.where(base['valid_mask'],(image+offset+torch.randn_like(image)*noise_std).clamp(0,1),0.)
    result={key:torch.cat((base[key],base[key]),0) for key in ('valid_mask','padding_mask','voxel_to_lps','labels')}
    result['image']=torch.cat((image,second),0)
    result['localization']={'targets':torch.cat((base['targets'],base['targets']),0),
        'annotation_verified':torch.cat((base['annotation_verified'],base['annotation_verified']),0),'annotation_kind':'voxel_mask'}
    fake=result['labels']>=2
    result['generator_ids']=torch.where(fake,torch.arange(8,device=device)//4,-1)
    result['generator_verified']=fake;result['training_generator_ids']=(0,1)
    result['case_ids']=[f'procedural_group{group}_class{label}' for group in (0,1) for label in range(4)]
    return result


def run_phase7(config_path,output_dir,*,device=None,progress=None):
    c=load_phase7_config(config_path);e=c['engineering'];device=choose_device(device or e['device'])
    destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()):raise ValueError("F07 output directory must be new/empty")
    seed_everything(e['seed'],e['threads'])
    if device=='cuda':torch.use_deterministic_algorithms(True,warn_only=True)
    started=time.perf_counter()
    initial=RSVTRiskFoundation(**{k:v for k,v in c['model'].items() if k!='name'}).to(device)
    initial_digest=weights_digest(initial)
    batch=risk_synthetic_batch(e['synthetic_shapes_zyx'],device,offset=e['group_intensity_offset'],noise_std=e['group_noise_std'])
    weights,counts=training_class_weights(batch['labels'].cpu());weights=weights.to(device)
    variants={};checks={}
    for variant in ('B4_risk_only','B5_full'):
        model=deepcopy(initial)
        dual=ProjectedTokenDual(**c['dual']);history=[]
        plan=[('baseline',c['stages']['baseline_steps'])]
        if variant=='B5_full':plan.append(('routing_warmup',c['stages']['routing_warmup_steps']))
        plan.append(('risk_finetune',c['stages']['risk_finetune_steps']))
        options={'epochs':sum(steps for _,steps in plan),'learning_rate':e['learning_rate'],
                 'weight_decay':0.,**c.get('optimization',{})}
        optimizer=build_optimizer(model,options)
        scheduler=build_scheduler(optimizer,options) if 'optimization' in c else None
        for stage,steps in plan:
            for number in range(1,steps+1):
                learning_rate=optimizer.param_groups[0]['lr']
                row=risk_training_step(model,optimizer,batch,weights,variant=variant,stage=stage,loss_config=c['loss'],
                    dual=dual,gradient_clip_norm=e['gradient_clip_norm'])
                row['stage_step']=number
                if scheduler is not None:
                    row['learning_rate']=learning_rate
                    row['optimization']={**optimization_report(optimizer,options),'schedule_unit':'optimizer_step'}
                    scheduler.step()
                history.append(row)
                (destination/f'{variant}_history.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
                if progress:progress({'event':'stage_step','variant':variant,'stage':stage,'step':number,'loss':row['loss'],
                    'constraint':row['budget']['constraint'],'dual_after':row['dual_after']['multiplier']})
        model.eval()
        with torch.inference_mode():
            output=model(batch['image'],batch['valid_mask'],batch['voxel_to_lps'],padding_mask=batch['padding_mask'],
                         routing_mode='dense' if variant=='B4_risk_only' else 'hard')
            ce=F.cross_entropy(output['logits'],batch['labels'],reduction='none')
            guard=generator_risk_variance(ce,batch['labels'],torch.where(batch['generator_verified'],0,-1),batch['generator_verified'],
                                         training_generator_ids=(0,),partition='train',min_group_samples=c['loss']['min_group_samples'])
            inference={'routing_mode':output['routing']['mode'],'logits_shape':list(output['logits'].shape),
                'actual_sequence':output['compute']['actual_sequence_length_including_cls'],'active_token_counts':_plain(output['active_token_counts']),
                'coarse_coverage':coarse_coverage(output,batch['valid_mask']),
                'fine_voxel_prediction_coverage_per_case':_plain(output['spatial_valid_mask'].flatten(1).sum(1)/batch['valid_mask'].flatten(1).sum(1)),
                'spatial_mask_inside_valid':bool((~output['spatial_valid_mask']|batch['valid_mask']).all()),
                'spatial_logits_zero_outside_prediction_mask':bool((output['spatial_logits'][~output['spatial_valid_mask']]==0).all()),
                'single_generator_guard_disabled':guard['enabled_classes']==0 and float(guard['loss'])==0.,
                'scope':'synthetic_contract_only_no_medical_performance_or_timing_claim'}
        variants[variant]={'initial_weights_sha256':initial_digest,'optimizer_steps':len(history),'stages':[name for name,_ in plan],
            'history':history,'inference':inference,'final_dual':dual.state_dict()}
        checks[variant+'_finite_and_frozen']=all(math.isfinite(row['loss']) and row['frozen_parameters_unchanged'] and row['invalid_image_gradient_zero'] for row in history)
        checks[variant+'_active_gradient_groups']=all(g['all_present_finite'] and g['l2_norm']>0 for row in history for g in row['gradient_groups'].values() if g['active'])
        checks[variant+'_coarse_and_localization_masks']=all(inference['coarse_coverage']) and inference['spatial_mask_inside_valid'] and inference['spatial_logits_zero_outside_prediction_mask']
        checks[variant+'_single_generator_guard']=inference['single_generator_guard_disabled']
        if variant=='B4_risk_only':
            checks['B4_router_frozen_and_dual_disabled']=all(not row['router_weight_changed'] and not row['dual_enabled'] for row in history)
        else:
            risk_rows=[row for row in history if row['stage']=='risk_finetune']
            checks['risk_CVaR_and_generator_reach_router']=all(g['present_finite'] and g['l2_norm']>0 for row in risk_rows for g in row['risk_gradients_to_router'].values() if g['active'])
            checks['B5_hard_cap_enforced']=inference['actual_sequence']<=model.sequence_cap
            checks['B5_router_learns_only_after_baseline']=all(row['router_weight_changed']==(row['stage']!='baseline') for row in history)
            checks['B5_dual_updates_count']=dual.updates==c['stages']['routing_warmup_steps']+c['stages']['risk_finetune_steps']
        del model,optimizer,output
    resolved=json.dumps(c,sort_keys=True,indent=2)+'\n'
    source=hashlib.sha256();root=Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob('*.py')):source.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    summary={'schema_version':'rsvt_f07_engineering_v1','phase':'F07','ok':all(checks.values()),'checks':checks,
        'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':device,'threads':e['threads']},
        'medical_data_read':False,'medical_training_ready':False,'project_multi_generator_ready':False,
        'synthetic_training_generator_ids':[0,1],'generator_ids_are_procedural_fixtures_only':True,
        'config_sha256':hashlib.sha256(resolved.encode()).hexdigest(),'rsvt_python_source_sha256':source.hexdigest(),
        'initial_weights_sha256':initial_digest,'training_class_counts':counts.tolist(),'training_class_weights':_plain(weights),
        'logical_batch_size':len(batch['labels']),'cases':batch['case_ids'],'sequence_cap':c['model']['sequence_cap'],
        'variants':variants,'wall_seconds':time.perf_counter()-started,
        'comparison_scope':'shared_initial_weights_and_cases_different_stage_update_counts_no_comparative_performance_claim',
        'risk_parameter_scope':'engineering_defaults_not_source_validation_selected_not_clinical_costs',
        'pending':['Colab run/review','source-validation alpha and weights','medical data/annotation/identity QA and F04 CNN optimization',
            'actual multi-generator data','B4/B5 matched medical comparisons','full pipeline profiling with localization head','F08 locked experiment orchestration']}
    (destination/'resolved_config.json').write_text(resolved,encoding='utf-8')
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary
