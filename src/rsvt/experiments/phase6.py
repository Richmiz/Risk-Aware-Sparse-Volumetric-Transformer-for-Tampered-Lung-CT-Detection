"""F06 source-only synthetic B3 smoke, selection audit and isolated profiling."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from rsvt.evaluation.efficiency import percentile,process_memory,synchronize
from rsvt.experiments.phase5 import _plain,coarse_coverage,synthetic_batch
from rsvt.losses.budget import expected_token_budget_loss
from rsvt.models.routed_rsvt import RSVTRoutingFoundation
from rsvt.utils.config import load_config
from rsvt.utils.reproducibility import seed_everything


def load_phase6_config(path):
    c=load_config(path)
    if set(c)!={"model","engineering","profiling","protocol"} or any(not isinstance(v,dict) for v in c.values()):
        raise ValueError("F06 requires model/engineering/profiling/protocol tables")
    m,e,p,protocol=(c[k] for k in ("model","engineering","profiling","protocol"))
    allowed={"name","dimension","heads","layers","coarse_width","candidate_width","coarse_patch_size","patch_sizes",
        "dropout","localization_head","max_tokens","max_batch_voxels","max_attention_elements","max_feature_grid_voxels",
        "router_hidden","temperature","sequence_cap"}
    if set(m)!=allowed or m['name']!='rsvt_f06_routing' or m['localization_head'] is not False:
        raise ValueError("Use the complete F06 routing-only model config")
    if (set(protocol)!={"status","medical_training_ready","risk_and_generator_losses"}
            or protocol['status']!='synthetic_routing_only_B3_engineering'
            or protocol['medical_training_ready'] is not False or protocol['risk_and_generator_losses'] is not False):
        raise ValueError("F06 must retain synthetic B3/no-risk protocol flags")
    if set(e)!={"seed","threads","device","synthetic_shapes_zyx","learning_rate","budget_loss_weight","audit_trials","audit_noise_std"}:
        raise ValueError("Unknown/missing F06 engineering fields")
    def integer(n,low,high):return isinstance(n,int) and not isinstance(n,bool) and low<=n<=high
    def positive(n):return isinstance(n,(float,int)) and not isinstance(n,bool) and math.isfinite(n) and n>0
    if not integer(e['seed'],0,2**32-1) or not integer(e['threads'],1,16) or not integer(e['audit_trials'],2,20):
        raise ValueError("Invalid seed, threads, or audit trials")
    if e['device'] not in ('auto','cpu','cuda') or any(not positive(e[k]) for k in ('learning_rate','budget_loss_weight','audit_noise_std')):
        raise ValueError("Invalid device/engineering scalar")
    shapes=e['synthetic_shapes_zyx']
    if not isinstance(shapes,list) or len(shapes)!=4 or any(not isinstance(s,list) or len(s)!=3 or any(not integer(n,1,96) for n in s) for s in shapes):
        raise ValueError("Four synthetic shapes with axes 1..96 are required")
    if set(p)!={"shape_zyx","warmup","iterations","stage_iterations","low","medium","high","latency_target_factors"}:
        raise ValueError("Unknown/missing profiling fields")
    if (not isinstance(p['shape_zyx'],list) or len(p['shape_zyx'])!=3 or any(not integer(n,1,96) for n in p['shape_zyx'])
            or not integer(p['warmup'],1,20) or not integer(p['iterations'],5,100) or not integer(p['stage_iterations'],1,10)):
        raise ValueError("Invalid bounded profiling shape/repetitions")
    caps=[p[k] for k in ('low','medium','high')]
    if not all(integer(n,1,2048) for n in caps) or caps!=sorted(set(caps)):
        raise ValueError("Use three increasing token caps")
    factors=p['latency_target_factors']
    if not isinstance(factors,list) or len(factors)!=3 or not all(positive(n) and n<=1 for n in factors) or factors!=sorted(set(factors)):
        raise ValueError("Three increasing latency target factors in (0,1] required")
    return c


def choose_device(requested):
    if requested not in ('auto','cpu','cuda'):raise ValueError("Invalid device")
    actual='cuda' if requested=='auto' and torch.cuda.is_available() else 'cpu' if requested=='auto' else requested
    if actual=='cuda' and not torch.cuda.is_available():raise ValueError("CUDA unavailable; select auto/cpu")
    return actual


def make_model(config,device):
    return RSVTRoutingFoundation(**{k:v for k,v in config['model'].items() if k!='name'}).to(device)


def profile_fixture(shape,device):
    axes=[(torch.arange(n,device=device)-(n-1)/2)/max(n/2,1) for n in shape]
    z,y,x=torch.meshgrid(*axes,indexing='ij')
    valid=(z.square()+y.square()+x.square()<=1.)[None,None]
    valid[0,0,shape[0]//2,shape[1]//2,shape[2]//2]=True
    image=(.4+.1*torch.sin(x*4)+.1*torch.cos(y*3)+.05*torch.rand(shape,device=device))[None,None]
    image=torch.where(valid,image,0.).float()
    affine=torch.diag(torch.tensor([.8,1.,1.5,1.],dtype=torch.float64,device=device))[None]
    return dict(image=image,valid_mask=valid,voxel_to_lps=affine,padding_mask=torch.ones_like(valid))


def weights_digest(model):
    digest=hashlib.sha256()
    for key,value in model.state_dict().items():digest.update(key.encode()+value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def profile_worker(config_path,device,profile,output_path):
    c=load_phase6_config(config_path);device=choose_device(device);e=c['engineering'];p=c['profiling']
    seed_everything(e['seed'],e['threads'])
    if device=='cuda':torch.use_deterministic_algorithms(True,warn_only=True)
    model=make_model(c,device).eval();digest=weights_digest(model)
    batch=profile_fixture(p['shape_zyx'],device)
    # This implementation always constructs candidate grids, including at its floor.
    shape=p['shape_zyx'];patch=c['model']['coarse_patch_size']
    floor=1
    for z in range(0,shape[0],patch):
        for y in range(0,shape[1],patch):
            for x in range(0,shape[2],patch):floor+=int(batch['valid_mask'][0,0,z:z+patch,y:y+patch,x:x+patch].any())
    mode=profile if profile in ('dense','all_valid') else 'hard'
    cap=floor if profile=='minimum' else p.get(profile,c['model']['sequence_cap'])
    if profile not in ('dense','all_valid','minimum','low','medium','high'):raise ValueError("Unknown profile")
    arguments=dict(routing_mode=mode,sequence_cap=cap)
    with torch.inference_mode():
        for _ in range(p['warmup']):
            temporary=model(**batch,**arguments);del temporary
        synchronize(device);memory_before=process_memory()
        if device=='cuda':
            torch.cuda.reset_peak_memory_stats();cuda_baseline=torch.cuda.memory_allocated()
        samples=[]
        for _ in range(p['iterations']):
            synchronize(device);start=time.perf_counter()
            temporary=model(**batch,**arguments);synchronize(device)
            samples.append((time.perf_counter()-start)*1000);del temporary
        memory_after=process_memory()
        cuda_memory={"baseline_allocated_bytes":cuda_baseline,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                     "peak_reserved_bytes":torch.cuda.max_memory_reserved()} if device=='cuda' else None
        stages={};compute=None;active=None
        for _ in range(p['stage_iterations']):
            temporary=model(**batch,**arguments,profile_stages=True)
            compute=_plain(temporary['compute']);active=_plain(temporary['active_token_counts'])
            for key,value in temporary['stage_ms'].items():stages.setdefault(key,[]).append(value)
            del temporary
    result={"profile":profile,"routing_mode":mode,"sequence_cap":cap if mode=='hard' else None,
        "model_initial_weights_sha256":digest,"shape_zyx":shape,"batch_size":1,"device":device,"torch":str(torch.__version__),
        "warmup":p['warmup'],"iterations":p['iterations'],"latency_samples_ms":samples,
        "p50_ms":percentile(samples,.5),"p95_ms":percentile(samples,.95),
        "timing_scope":"preloaded_FP32_model_forward_including_validation_stems_controller_packing_attention_metadata_no_IO",
        "stage_ms_instrumented_p50":{k:percentile(v,.5) for k,v in stages.items()},
        "stage_scope":"separate_synchronized_instrumented_forwards_not_additive_latency_estimate",
        "memory_before":memory_before,"memory_after":memory_after,"cuda_memory":cuda_memory,
        "memory_scope":"fresh_process_per_profile_peak_RSS_includes_imports_and_initialization_CUDA_peaks_reset_after_warmup",
        "active_token_counts":active,"compute":compute,"medical_data_read":False}
    Path(output_path).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def calibrate_latency(profiles,factors):
    dense=next(row for row in profiles if row['profile']=='dense')
    hard=[row for row in profiles if row['routing_mode']=='hard']
    result={}
    for name,factor in zip(('low','medium','high'),factors):
        target=dense['p95_ms']*factor
        feasible=[row for row in hard if row['p95_ms']<=target]
        best=max(feasible,key=lambda row:row['sequence_cap']) if feasible else None
        result[name]={"target_factor_of_dense_p95":factor,"target_ms":target,
            "feasible_measured_profile":best['profile'] if best else None,
            "sequence_cap":best['sequence_cap'] if best else None,
            "observed_p95_ms":best['p95_ms'] if best else None,"infeasible_among_measured_profiles":not feasible}
    return {"proposals":result,"minimum_observed_hard_p95_ms":min(row['p95_ms'] for row in hard),
        "scope":"exploratory_same_fixture_hardware_calibration_not_a_guaranteed_latency_constraint"}


def selection_audit(model,batch,trials,noise_std,cap,destination):
    model.eval();frequency=None;base=None;fine_coverage=[]
    with torch.inference_mode():
        for _ in range(trials):
            noisy=(batch['image']+torch.randn_like(batch['image'])*noise_std).clamp(0,1)
            out=model(noisy,batch['valid_mask'],batch['voxel_to_lps'],padding_mask=batch['padding_mask'],sequence_cap=cap)
            route=out['routing'];selected=route['selected_source_mask'];candidate=route['source_candidate_mask']
            frequency=selected.int() if frequency is None else frequency+selected.int()
            base=route
            case_coverage=[]
            for i,shape in enumerate(out['original_shape_zyx'].tolist()):
                covered=torch.zeros(shape,dtype=torch.bool,device=noisy.device)
                fine=selected[i]&(route['source_geometry']['scale_ids'][i]==1)
                for start,end in zip(route['source_geometry']['starts_xyz'][i,fine].tolist(),route['source_geometry']['ends_xyz'][i,fine].tolist()):
                    x,y,z=start;ex,ey,ez=end;covered[z:ez,y:ey,x:ex]=True
                z,y,x=shape;valid=batch['valid_mask'][i,0,:z,:y,:x]
                case_coverage.append(float((covered&valid).sum()/valid.sum()))
            fine_coverage.append(case_coverage)
    rows=[];bins=[]
    for i in range(len(batch['image'])):
        # Coordinates normalized by each original box, not the padded batch size.
        support=batch['padding_mask'][i,0]
        original=[int(support.any(dim=tuple(k for k in range(3) if k!=axis)).nonzero().flatten()[-1])+1 for axis in range(3)]
        occupied=set();visited=set()
        for j in base['source_candidate_mask'][i].nonzero().flatten().tolist():
            g=base['source_geometry'];xyz=g['centers_voxel_xyz'][i,j].tolist()
            bin_xyz=tuple(min(3,int((xyz[a]+.5)/original[2-a]*4)) for a in range(3))
            occupied.add(bin_xyz)
            if int(frequency[i,j])>0:visited.add(bin_xyz)
            rows.append({"case":i,"source_index":j,"scale_id":int(g['scale_ids'][i,j]),
                "center_x":xyz[0],"center_y":xyz[1],"center_z":xyz[2],"selected_trials":int(frequency[i,j]),"trials":trials})
        bins.append({"case":i,"candidate_center_bins_occupied":len(occupied),"bins_never_selected":len(occupied-visited)})
    destination=Path(destination)
    with (destination/'selection_frequency.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {"trials":trials,"sequence_cap":cap,"noise_std":noise_std,"candidate_rows":len(rows),
        "candidates_never_selected":sum(row['selected_trials']==0 for row in rows),
        "by_scale":{str(scale):{"candidates":sum(row['scale_id']==scale for row in rows),
            "selected_trial_occurrences":sum(row['selected_trials'] for row in rows if row['scale_id']==scale)} for scale in (1,2,3)},
        "coarse_center_spatial_bins_4x4x4":bins,"fine_voxel_coverage_per_trial_case":fine_coverage,
        "scope":"same_synthetic_cases_with_intensity_noise_trained_one_step_no_medical_bias_or_recall_claim"}


def run_phase6(config_path,output_dir,*,device=None,progress=None):
    c=load_phase6_config(config_path);e=c['engineering'];device=choose_device(device or e['device'])
    destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()):raise ValueError("F06 output directory must be new/empty")
    seed_everything(e['seed'],e['threads'])
    if device=='cuda':torch.use_deterministic_algorithms(True,warn_only=True)
    model=make_model(c,device).train();batch=synthetic_batch(e['synthetic_shapes_zyx'],device)
    batch['image'].requires_grad_(True)
    output=model(batch['image'],batch['valid_mask'],batch['voxel_to_lps'],padding_mask=batch['padding_mask'])
    ce=F.cross_entropy(output['logits'],batch['labels'])
    router_task_grad=torch.autograd.grad(ce,model.router.score[0].weight,retain_graph=True)[0]
    penalty=expected_token_budget_loss(output['routing']['expected_sequence_counts'],model.sequence_cap)
    loss=ce+e['budget_loss_weight']*penalty
    optimizer=torch.optim.AdamW(model.parameters(),lr=e['learning_rate'],weight_decay=0.)
    before=model.router.score[0].weight.detach().clone();loss.backward()
    groups={}
    for name,module in (("coarse",model.coarse_encoder),("candidate",model.tokenizer),("position",model.position),
                        ("transformer",model.transformer),("classifier",model.classifier),("router",model.router)):
        grads=[p.grad for p in module.parameters()]
        groups[name]={"all_present_finite":all(g is not None and bool(torch.isfinite(g).all()) for g in grads),
                      "l2_norm":math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))}
    checks={"finite_loss":bool(torch.isfinite(loss)),"all_module_gradients_finite_nonzero":all(g['all_present_finite'] and g['l2_norm']>0 for g in groups.values()),
        "classification_gradient_reaches_router":bool(torch.isfinite(router_task_grad).all() and router_task_grad.abs().sum()>0),
        "invalid_image_gradient_zero":bool((batch['image'].grad[~batch['valid_mask']]==0).all()),
        "soft_training_does_not_claim_sparse_saving":output['compute']['saved_attention_slots']==0}
    if all(checks.values()):optimizer.step()
    checks['router_weights_updated']=not torch.equal(before,model.router.score[0].weight.detach())
    soft={"loss":float(loss.detach()),"classification":float(ce.detach()),"budget_penalty":float(penalty.detach()),
          "expected_sequence_counts":_plain(output['routing']['expected_sequence_counts']),"compute":_plain(output['compute'])}
    del output,loss,ce,penalty
    model.eval()
    with torch.inference_mode():
        hard=model(batch['image'],batch['valid_mask'],batch['voxel_to_lps'],padding_mask=batch['padding_mask'])
        dense=model(batch['image'],batch['valid_mask'],batch['voxel_to_lps'],padding_mask=batch['padding_mask'],routing_mode='dense')
        compact=model(batch['image'],batch['valid_mask'],batch['voxel_to_lps'],padding_mask=batch['padding_mask'],routing_mode='all_valid')
        checks.update(hard_sequence_cap_enforced=hard['compute']['actual_sequence_length_including_cls']<=model.sequence_cap,
            hard_transformer_input_shorter=hard['token_features'].shape[1]<dense['token_features'].shape[1],
            coarse_coverage=all(coarse_coverage(hard,batch['valid_mask'])),
            all_valid_packing_matches_dense=bool(torch.allclose(compact['logits'],dense['logits'],atol=1e-6,rtol=1e-5)))
        hard_compute=_plain(hard['compute'])
    audit=selection_audit(model,batch,e['audit_trials'],e['audit_noise_std'],model.sequence_cap,destination)
    (destination/'selection_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    if progress:progress({"event":"smoke_and_selection_audit","checks":checks})
    # Release training/optimizer references before isolated workers, particularly
    # the parent's CUDA caching allocator, so profiling has room on the device.
    del hard,dense,compact,model,batch,optimizer,before,router_task_grad,grads,module
    if device=='cuda':torch.cuda.empty_cache()
    profiles=[];profile_dir=destination/'profiles';profile_dir.mkdir()
    env=os.environ.copy();src=str(Path(__file__).resolve().parents[2]);env['PYTHONPATH']=src+os.pathsep+env.get('PYTHONPATH','')
    for profile in ('dense','all_valid','minimum','low','medium','high'):
        target=profile_dir/f'{profile}.json'
        command=[sys.executable,'-m','rsvt.experiments.phase6','--profile-worker','--config',str(Path(config_path).resolve()),
                 '--device',device,'--profile',profile,'--output',str(target.resolve())]
        completed=subprocess.run(command,env=env,capture_output=True,text=True,timeout=600)
        (profile_dir/f'{profile}.log').write_text(completed.stdout+completed.stderr,encoding='utf-8')
        if completed.returncode or not target.is_file():raise ValueError(f"Profiling {profile} failed; inspect {profile_dir / (profile+'.log')}")
        row=json.loads(target.read_text(encoding='utf-8'));profiles.append(row)
        if progress:progress({"event":"profile_complete","profile":profile,"p50_ms":row['p50_ms'],"sequence":row['compute']['actual_sequence_length_including_cls']})
    checks['profile_weights_identical']=len({row['model_initial_weights_sha256'] for row in profiles})==1
    checks['fixed_stem_flops_identical']=len({(row['compute']['operator_flops_estimate']['by_component']['coarse_encoder'],
        row['compute']['operator_flops_estimate']['by_component']['candidate_encoder_and_projections']) for row in profiles})==1
    calibration=calibrate_latency(profiles,c['profiling']['latency_target_factors'])
    dense_profile=profiles[0]
    for row in profiles:row['measured_p50_speed_ratio_vs_dense']=dense_profile['p50_ms']/row['p50_ms']
    resolved=json.dumps(c,sort_keys=True,indent=2)+'\n'
    summary={"schema_version":"rsvt_f06_engineering_v1","ok":all(checks.values()),"phase":"F06","scope":"synthetic_routing_only_B3",
        "medical_data_read":False,"medical_training_ready":False,"risk_and_generator_losses":False,
        "environment":{"python":platform.python_version(),"torch":str(torch.__version__),"device":device,"threads":e['threads']},
        "config_sha256":hashlib.sha256(resolved.encode()).hexdigest(),"checks":checks,"gradient_groups":groups,
        "soft_training":soft,"hard_inference_compute":hard_compute,"selection_audit":audit,"profiles":profiles,
        "latency_calibration":calibration,"profile_weights_scope":"identical_seeded_random_initial_weights_fresh_process_each_mode_not_the_one_step_smoke_weights",
        "pending":["Colab device-specific profile review","medical B3 training/comparison and QA","coarse pretraining",
                   "risk/CVaR/generator losses F07","final budgets from source validation","full operator FLOPs coverage"]}
    (destination/'resolved_config.json').write_text(resolved,encoding='utf-8')
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile-worker',action='store_true',required=True)
    parser.add_argument('--config',required=True);parser.add_argument('--device',required=True)
    parser.add_argument('--profile',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();profile_worker(args.config,args.device,args.profile,args.output)
