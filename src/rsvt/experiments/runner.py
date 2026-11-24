"""F08 source-bound synthetic jobs with immutable attempts and job-boundary resume."""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
import torch
from torch.nn import functional as F

from rsvt.data.acquisition import sha256_file
from rsvt.data.splits import digest_json
from rsvt.evaluation.classification import classification_metrics
from rsvt.evaluation.thresholds import calibrate_source_threshold,evaluate_frozen_threshold
from rsvt.experiments.ablations import load_catalog
from rsvt.experiments.phase6 import choose_device,weights_digest
from rsvt.experiments.phase7 import risk_synthetic_batch
from rsvt.experiments.protocol import audit_f03_metadata,research_readiness
from rsvt.losses.budget import ProjectedTokenDual
from rsvt.losses.classification import training_class_weights
from rsvt.losses.combined import risk_objective
from rsvt.models.baselines import ResNet50Volume,DenseViT3D
from rsvt.models.risk_rsvt import RSVTRiskFoundation
from rsvt.training.checkpoint import save_checkpoint
from rsvt.utils.config import load_config
from rsvt.utils.reproducibility import rng_state,seed_everything


def atomic_json(path,value,*,replace=False):
    path=Path(path)
    if path.exists() and not replace:raise ValueError(f"Artifact already exists: {path.name}")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def source_digest():
    root=Path(__file__).resolve().parents[1];digest=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):digest.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


def load_phase8_config(path):
    c=load_config(path)
    if set(c)!={'engineering','catalog','demo_policy','demo_loss','demo_dual','research_protocol'} or any(not isinstance(v,dict) for v in c.values()):
        raise ValueError("Use the explicit F08 orchestration config")
    e=c['engineering'];p=c['demo_policy']
    if set(e)!={'scope','device','threads','seeds','shape_zyx','steps','learning_rate','execute_ids'} or e['scope']!='synthetic_only':
        raise ValueError("F08 execution supports synthetic fixtures only")
    def integer(v,lo,hi):return isinstance(v,int) and not isinstance(v,bool) and lo<=v<=hi
    if (e['device'] not in ('auto','cpu','cuda') or not integer(e['threads'],1,8) or not integer(e['steps'],1,5)
            or e['shape_zyx']!=[32,32,32] or not isinstance(e['seeds'],list) or not 1<=len(e['seeds'])<=5
            or len(set(e['seeds']))!=len(e['seeds']) or not all(integer(v,0,2**32-10001) for v in e['seeds'])
            or isinstance(e['learning_rate'],bool) or not isinstance(e['learning_rate'],(float,int))
            or not math.isfinite(e['learning_rate']) or not 0<e['learning_rate']<=.01):raise ValueError("Invalid bounded F08 synthetic settings")
    if (not isinstance(e['execute_ids'],list) or not e['execute_ids'] or len(e['execute_ids'])!=len(set(e['execute_ids']))):
        raise ValueError("Declare unique baseline job IDs")
    if p!={'target_fpr':p.get('target_fpr'),'score':'p_FB_plus_p_FM','comparison':'strict_gt',
        'checkpoint_selection':'minimum_unweighted_source_validation_CE','scope':'procedural_interface_check_not_research_selected_FPR'}:
        raise ValueError("Unsupported demo evaluation policy")
    if isinstance(p['target_fpr'],bool) or not isinstance(p['target_fpr'],(int,float)) or not 0<=p['target_fpr']<1:raise ValueError("Invalid demo FPR")
    ProjectedTokenDual(**c['demo_dual'])
    loss=c['demo_loss']
    if set(loss)!={'alpha','fb_weight','fm_weight','min_class_samples','min_group_samples','cvar_weight','generator_weight','localization_weight'}:
        raise ValueError("Invalid demo-loss fields")
    if (isinstance(loss['alpha'],bool) or not isinstance(loss['alpha'],(int,float)) or not 0<=loss['alpha']<1
            or any(isinstance(loss[k],bool) or not isinstance(loss[k],(float,int)) or not math.isfinite(loss[k]) or loss[k]<=0
                for k in ('fb_weight','fm_weight','cvar_weight','generator_weight'))
            or any(not integer(loss[k],1,1000) for k in ('min_class_samples','min_group_samples'))
            or loss['localization_weight']!=0.):raise ValueError("Invalid bounded demo loss; localization must be off")
    return c


def build_model(entry,shape):
    m=entry['model'];family=m['family']
    for key,lo,hi in (('dimension',4,32),('layers',1,2),('heads',1,4),('cnn_width',2,4)):
        value=m[key]
        if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:raise ValueError("Catalog model exceeds prototype bounds")
    if m['dimension']%m['heads'] or m['patch_size'] not in (8,16) or m['coarse_patch_size']!=16 or m['candidate_patch_sizes']!=[4,8,16]:
        raise ValueError("Unsupported catalog token geometry")
    if isinstance(m['sequence_cap'],bool) or not isinstance(m['sequence_cap'],int) or not 16<=m['sequence_cap']<=256:raise ValueError("Invalid prototype sequence cap")
    if family=='resnet50_3d':return ResNet50Volume(m['cnn_width'])
    if family=='vit3d':return DenseViT3D(tuple(shape),m['patch_size'],m['dimension'],m['layers'],m['heads'],0.)
    if family=='rsvt':return RSVTRiskFoundation(dimension=m['dimension'],layers=m['layers'],heads=m['heads'],coarse_width=4,
        candidate_width=4,coarse_patch_size=16,patch_sizes=(4,8,16),localization_head=False,sequence_cap=m['sequence_cap'])
    raise ValueError("Requested baseline backend is not implemented")


def seed_runtime(seed,threads,device):
    seed_everything(seed,threads)
    if device=='cuda':torch.use_deterministic_algorithms(True,warn_only=True)


def fixtures(seed,shape,device,threads):
    batches={};metadata={}
    for offset,partition in ((0,'train'),(10000,'validation')):
        seed_runtime(seed+offset,threads,device)
        batch=risk_synthetic_batch([shape]*4,device);batch.pop('localization')
        # F07 varies geometry by case to exercise transforms. For F08's shared
        # interface fixtures, remove the case/class-to-affine correlation.
        batch['voxel_to_lps']=batch['voxel_to_lps'][:1].expand_as(batch['voxel_to_lps']).clone()
        labels=batch['labels'].tolist()
        rows=[{'case_id':f'{partition}:g{i//4}:class{label}','source_group':f'synthetic:{seed}:{partition}:base{label}',
               'partition':partition,'label':label,'generator_id':int(batch['generator_ids'][i]),'generator_verified':bool(batch['generator_verified'][i])}
              for i,label in enumerate(labels)]
        digest=hashlib.sha256()
        for key in ('image','valid_mask','voxel_to_lps'):digest.update(key.encode()+batch[key].detach().cpu().contiguous().numpy().tobytes())
        batch['records']=rows;batches[partition]=batch
        metadata[partition]={'records':rows,'tensor_sha256':digest.hexdigest(),'assignment_sha256':digest_json(rows),
            'volume_cases':len(rows),'source_groups':len({r['source_group'] for r in rows})}
    if set(r['source_group'] for r in metadata['train']['records'])&set(r['source_group'] for r in metadata['validation']['records']):raise ValueError("Synthetic source leakage")
    return batches,metadata


def forward(model,entry,batch,*,training):
    if entry['model']['family']!='rsvt':return model(batch['image'],batch['valid_mask'])
    mode=('soft' if training else 'hard') if entry['objective']['routing'] else 'dense'
    return model(batch['image'],batch['valid_mask'],batch['voxel_to_lps'],padding_mask=batch['padding_mask'],routing_mode=mode)


def runtime_signature(device):
    return {'python':platform.python_version(),'torch':str(torch.__version__),'device':device,
            'gpu_name':torch.cuda.get_device_name() if device=='cuda' else None}


def hardware_record(device):
    cpu=platform.processor()
    if platform.system()=='Linux' and Path('/proc/cpuinfo').is_file():
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),cpu)
    return {'platform':platform.platform(),'machine':platform.machine(),'cpu_model':cpu,'logical_cpu_count':os.cpu_count(),
        'torch_threads':torch.get_num_threads(),'torch_cuda_build':torch.version.cuda,
        'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None}


def build_plan(config_path,*,device=None,f03_manifest=None,forward_smoke=True):
    c=load_phase8_config(config_path);catalog=load_catalog(config_path,c);e=c['engineering'];actual=choose_device(device or e['device'])
    for identity in e['execute_ids']:
        if identity not in catalog or not catalog[identity]['entry']['synthetic_backend_supported'] or not identity.startswith('B'):
            raise ValueError(f"No synthetic execution backend for {identity}")
    metadata=audit_f03_metadata(f03_manifest);readiness=research_readiness(c['research_protocol'],metadata,catalog)
    payload={'schema_version':'f08_engineering_plan_v1','kind':'engineering_draft_snapshot_not_final_protocol',
        'resolved_config':c,'catalog':catalog,'source_sha256':source_digest(),'runtime':runtime_signature(actual),'f03_metadata':metadata,
        'test_volume_access_allowed':False,'research_readiness':readiness,
        'jobs':[{'id':f'{identity}_s{seed}','experiment_id':identity,'seed':seed} for seed in e['seeds'] for identity in e['execute_ids']]}
    probes={}
    if forward_smoke:
        batches,synthetic=fixtures(e['seeds'][0],e['shape_zyx'],actual,e['threads'])
        for identity in e['execute_ids']:
            seed_runtime(e['seeds'][0],e['threads'],actual);model=build_model(catalog[identity],e['shape_zyx']).to(actual).eval()
            with torch.inference_mode():out=forward(model,catalog[identity],batches['train'],training=False)
            if out['logits'].shape!=(8,4) or not torch.isfinite(out['logits']).all():raise ValueError("Dry-run forward contract failed")
            probes[identity]={'input_shape':list(batches['train']['image'].shape),'logits_shape':list(out['logits'].shape),'finite':True}
            del model,out
        payload['synthetic_dataset']=synthetic
    return {'payload':payload,'contract_sha256':digest_json(payload),'forward_probes':probes,'research_ready':False,
            'hardware_record':hardware_record(actual)}


@contextmanager
def exclusive_run(path):
    # Local-runtime advisory lock avoids relying on cloud-mount lock operations.
    # It does not coordinate two different Colab VMs writing the same Drive run.
    directory=Path(tempfile.gettempdir())/'rsvt-f08-locks';directory.mkdir(exist_ok=True)
    key=hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
    stream=(directory/(key+'.lock')).open('a+b')
    if stream.tell()==0:stream.write(b'0');stream.flush()
    stream.seek(0)
    try:
        try:
            if platform.system()=='Windows':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:raise ValueError("Another runner is using this output directory") from exc
        yield
    finally:stream.close()


def checkpoint_load(path,contract=None):
    path=Path(path);receipt_path=path.with_name(path.name+'.json')
    if path.is_symlink() or receipt_path.is_symlink():raise ValueError("Checkpoint/receipt symlink rejected")
    receipt=json.loads(receipt_path.read_text())
    if path.stat().st_size!=receipt['bytes'] or sha256_file(path)!=receipt['sha256']:raise ValueError("Checkpoint checksum mismatch")
    state=torch.load(path,map_location='cpu',weights_only=True)
    binding=digest_json(state['job_spec'])
    if (state.get('format_version')!='f08_synthetic_state_v1' or binding!=state['contract_sha256'] or binding!=receipt['contract_sha256']
            or (contract is not None and binding!=contract) or state['job_spec']['source_sha256']!=source_digest()):
        raise ValueError("Checkpoint source or contract mismatch")
    return state,receipt


def prediction_rows(model,entry,batch):
    model.eval()
    with torch.inference_mode():probabilities=forward(model,entry,batch,training=False)['logits'].softmax(1).cpu().tolist()
    return [dict(record,probabilities=probability) for record,probability in zip(batch['records'],probabilities)]


def run_job(entry,seed,config,plan_hash,destination,device):
    e=config['engineering'];batches,data=fixtures(seed,e['shape_zyx'],device,e['threads'])
    seed_runtime(seed,e['threads'],device);model=build_model(entry,e['shape_zyx']).to(device)
    if entry['model']['family']=='rsvt' and not entry['objective']['routing']:model.router.requires_grad_(False)
    initial=weights_digest(model);optimizer=torch.optim.AdamW(model.parameters(),lr=e['learning_rate'],weight_decay=0.)
    dual=ProjectedTokenDual(**config['demo_dual']);weights,counts=training_class_weights(batches['train']['labels'].cpu());weights=weights.to(device)
    loss_config=dict(config['demo_loss']);loss_config['cvar_weight']*=int(entry['objective']['cvar']);loss_config['generator_weight']*=int(entry['objective']['generator_variance'])
    spec={'schema':'f08_synthetic_job_v1','entry':entry,'seed':seed,'shape_zyx':e['shape_zyx'],'threads':e['threads'],
        'steps':e['steps'],'learning_rate':e['learning_rate'],'loss_config':loss_config,'dual_config':config['demo_dual'],
        'data':data,'plan_sha256':plan_hash,'source_sha256':source_digest(),'runtime':runtime_signature(device)}
    binding=digest_json(spec);history=[];best=None;started=time.perf_counter();checkpoint_dir=destination/'checkpoints';checkpoint_dir.mkdir()
    for step in range(1,e['steps']+1):
        model.train();optimizer.zero_grad(set_to_none=True);output=forward(model,entry,batches['train'],training=True)
        if entry['model']['family']=='rsvt':
            obj=risk_objective(output,batches['train']['labels'],weights,loss_config=loss_config,
                generator_ids=batches['train']['generator_ids'],generator_verified=batches['train']['generator_verified'],training_generator_ids=(0,1),
                risk_enabled=entry['objective']['cvar'] or entry['objective']['generator_variance'],
                dual=dual if entry['objective']['routing'] else None,sequence_cap=entry['model']['sequence_cap'])
            loss=obj['loss']
        else:obj=None;loss=(F.cross_entropy(output['logits'],batches['train']['labels'],reduction='none')*weights[batches['train']['labels']]).mean()
        if not torch.isfinite(loss):raise ValueError("Nonfinite synthetic loss")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise ValueError("Invalid active-parameter gradients")
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True));optimizer.step()
        if obj is not None and entry['objective']['routing']:dual.update(obj['budget']['constraint'])
        model.eval()
        with torch.inference_mode():validation=forward(model,entry,batches['validation'],training=False)['logits'];ce=float(F.cross_entropy(validation,batches['validation']['labels']))
        if not torch.isfinite(validation).all() or not math.isfinite(ce):raise ValueError("Nonfinite validation output")
        row={'step':step,'training_loss':float(loss.detach()),'source_validation_unweighted_CE':ce,'gradient_norm_before_clip':norm,'dual':dual.state_dict()};history.append(row)
        state={'format_version':'f08_synthetic_state_v1','contract_sha256':binding,'job_spec':spec,'epoch':step,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'dual':dual.state_dict(),'rng':rng_state(),'history':history,'class_weights':weights.cpu(),'class_counts':counts}
        checkpoint=checkpoint_dir/f'step_{step:04d}.pt';save_checkpoint(checkpoint,state)
        if best is None or ce<best['validation_CE']:best={'step':step,'validation_CE':ce,'path':checkpoint.relative_to(destination).as_posix()}
        del output,loss,obj,validation,state
    best_path=destination/best['path'];state,receipt=checkpoint_load(best_path,binding);model.load_state_dict(state['model'])
    predictions=prediction_rows(model,entry,batches['validation']);prediction_hash=digest_json(predictions)
    labels=[r['label'] for r in predictions];probs=[r['probabilities'] for r in predictions]
    threshold=calibrate_source_threshold(labels,probs,target_fpr=config['demo_policy']['target_fpr'],partition='validation',
        checkpoint_sha256=receipt['sha256'],prediction_sha256=prediction_hash)
    metrics={'classification':classification_metrics(labels,probs,partition='validation'),
        'frozen_threshold':evaluate_frozen_threshold(labels,probs,threshold,checkpoint_sha256=receipt['sha256'],partition='validation')}
    for name,value in [('job_contract.json',spec),('history.json',history),('predictions.json',predictions),('threshold.json',threshold),('metrics.json',metrics)]:atomic_json(destination/name,value)
    artifacts={}
    for path in sorted(destination.rglob('*')):
        if path.is_file():artifacts[path.relative_to(destination).as_posix()]={'sha256':sha256_file(path),'bytes':path.stat().st_size}
    result={'ok':True,'job_contract_sha256':binding,'plan_sha256':plan_hash,'experiment_id':entry['entry']['id'],'seed':seed,
        'initial_weights_sha256':initial,'data':data,'best_checkpoint':best,'best_checkpoint_sha256':receipt['sha256'],
        'checkpoint_selection':'minimum_unweighted_source_validation_CE','last_step':e['steps'],'metrics':metrics,'artifacts':artifacts,
        'wall_seconds':time.perf_counter()-started,'hardware_record':hardware_record(device),
        'scope':'synthetic_train_validation_interface_not_medical_results','test_cases_opened':0}
    atomic_json(destination/'result.json',result)
    return result


def verify_completed(attempt,expected_result_sha,plan_hash):
    if sha256_file(attempt/'result.json')!=expected_result_sha:raise ValueError("Completed result changed")
    result=json.loads((attempt/'result.json').read_text())
    if result['plan_sha256']!=plan_hash:raise ValueError("Completed job belongs to another plan")
    for name,reference in result['artifacts'].items():
        path=attempt/name
        if path.is_symlink() or not path.resolve().is_relative_to(attempt.resolve()) or path.stat().st_size!=reference['bytes'] or sha256_file(path)!=reference['sha256']:
            raise ValueError("Completed job artifact changed")
    return result


def run_phase8(config_path,output_dir,*,device=None,f03_manifest=None,resume=False,progress=None):
    output=Path(output_dir)
    if resume:
        if not output.is_dir():raise ValueError("Resume directory does not exist")
    else:output.mkdir(parents=True,exist_ok=False)
    with exclusive_run(output):
        plan=build_plan(config_path,device=device,f03_manifest=f03_manifest,forward_smoke=True)
        if resume:
            old=json.loads((output/'plan.lock.json').read_text())
            if digest_json(old['payload'])!=old['contract_sha256'] or old['contract_sha256']!=plan['contract_sha256']:
                raise ValueError("Resume requires identical code, catalog, config, metadata, runtime and fixtures")
            ledger=json.loads((output/'ledger.json').read_text())
            if ledger['plan_sha256']!=plan['contract_sha256']:raise ValueError("Ledger plan mismatch")
        else:
            atomic_json(output/'plan.lock.json',plan);ledger={'schema_version':'f08_ledger_v1','plan_sha256':plan['contract_sha256'],'jobs':{}}
            atomic_json(output/'ledger.json',ledger)
        c=plan['payload']['resolved_config'];catalog=plan['payload']['catalog'];actual=plan['payload']['runtime']['device'];results={};skipped=[]
        for job in plan['payload']['jobs']:
            identity=job['id'];record=ledger['jobs'].get(identity,{'status':'pending','attempts':[]})
            base=output/'jobs'/identity
            if record.get('status') not in ('pending','running','failed','succeeded') or not isinstance(record.get('attempts'),list):raise ValueError("Invalid job ledger state")
            if record['status']=='succeeded':
                name=record['completed_attempt']
                if not isinstance(name,str) or not name.startswith('attempt_') or not name.removeprefix('attempt_').isdigit():raise ValueError("Invalid completed attempt path")
                attempt=base/record['completed_attempt'];results[identity]=verify_completed(attempt,record['result_sha256'],plan['contract_sha256']);skipped.append(identity);continue
            if record['status']=='running':record['attempts'][-1]['status']='interrupted'
            attempt_name=f'attempt_{len(record["attempts"])+1:04d}';attempt=base/attempt_name;attempt.mkdir(parents=True,exist_ok=False)
            record['status']='running';record['attempts'].append({'directory':attempt_name,'status':'running'});ledger['jobs'][identity]=record
            atomic_json(output/'ledger.json',ledger,replace=True)
            try:result=run_job(catalog[job['experiment_id']],job['seed'],c,plan['contract_sha256'],attempt,actual)
            except Exception as exc:
                record['status']='failed';record['attempts'][-1]['status']='failed';record['attempts'][-1]['error']=f'{type(exc).__name__}: {exc}'
                atomic_json(output/'ledger.json',ledger,replace=True);raise
            record.update(status='succeeded',completed_attempt=attempt_name,result_sha256=sha256_file(attempt/'result.json'))
            record['attempts'][-1]['status']='succeeded';atomic_json(output/'ledger.json',ledger,replace=True);results[identity]=result
            if progress:progress({'event':'job_complete','id':identity,'best_step':result['best_checkpoint']['step'],'scope':'synthetic_only'})
        by_seed={seed:[v for v in results.values() if v['seed']==seed] for seed in c['engineering']['seeds']}
        checks={'requested_jobs_complete':len(results)==len(plan['payload']['jobs']) and all(v['ok'] for v in results.values()),
            'catalog_13_declared':len(catalog)==13,'medical_and_test_access_disabled':not plan['research_ready'] and all(v['test_cases_opened']==0 for v in results.values()),
            'same_input_data_per_seed':all(len({digest_json(v['data']) for v in rows})==1 for rows in by_seed.values()),
            'rsvt_initial_weights_matched':all(len({v['initial_weights_sha256'] for v in rows if v['experiment_id'] in ('B3','B4','B5')})<=1 for rows in by_seed.values()),
            'checkpoint_selection_source_validation':all(v['checkpoint_selection']=='minimum_unweighted_source_validation_CE' for v in results.values())}
        summary={'schema_version':'rsvt_f08_engineering_v1','phase':'F08','ok':all(checks.values()),'checks':checks,'plan_sha256':plan['contract_sha256'],
            'jobs_completed':len(results),'jobs_skipped_on_resume':skipped,'resume_scope':'completed_job_boundary_partial_attempts_preserved_and_restarted',
            'catalog_entries':len(catalog),'unsupported_entries':[k for k,v in catalog.items() if not v['entry']['synthetic_backend_supported']],
            'forward_probes':plan['forward_probes'],'research_readiness':plan['payload']['research_readiness'],'f03_metadata':plan['payload']['f03_metadata'],
            'runtime':plan['payload']['runtime'],'medical_data_read':False,'test_cases_opened':0,'medical_training_ready':False,
            'session_hardware_record':hardware_record(actual),'writer_lock_scope':'same_local_runtime_only_one_VM_writer_per_Drive_run',
            'results':{k:{'experiment_id':v['experiment_id'],'best_checkpoint':v['best_checkpoint'],'metrics':v['metrics'],'test_cases_opened':0} for k,v in results.items()},
            'scope':'engineering_orchestration_and_synthetic_interfaces_not_final_protocol_or_model_comparison'}
        atomic_json(output/'summary.json',summary,replace=True)
        return summary


def infer_checkpoint(checkpoint,output_path,*,device='auto',partition='validation'):
    if partition!='validation':raise ValueError("F08 inference is synthetic validation-only")
    if Path(output_path).exists():raise ValueError("Inference output must be new")
    state,receipt=checkpoint_load(checkpoint);spec=state['job_spec'];actual=choose_device(device)
    if spec['runtime']!=runtime_signature(actual):raise ValueError("Checkpoint replay requires the same runtime signature")
    batches,data=fixtures(spec['seed'],spec['shape_zyx'],actual,spec['threads'])
    if data!=spec['data']:raise ValueError("Inference procedural data fingerprint differs")
    seed_runtime(spec['seed'],spec['threads'],actual);model=build_model(spec['entry'],spec['shape_zyx']).to(actual);model.load_state_dict(state['model'])
    rows=prediction_rows(model,spec['entry'],batches['validation'])
    result={'schema_version':'f08_inference_v1','partition':'validation','checkpoint_sha256':receipt['sha256'],'predictions':rows,
        'prediction_sha256':digest_json(rows),'medical_data_read':False,'test_cases_opened':0}
    atomic_json(output_path,result);return result


def evaluate_inference(inference_path,threshold_path,output_path):
    if Path(output_path).exists():raise ValueError("Evaluation output must be new")
    prediction=json.loads(Path(inference_path).read_text());threshold=json.loads(Path(threshold_path).read_text())
    if prediction.get('schema_version')!='f08_inference_v1' or prediction.get('partition')!='validation' or digest_json(prediction['predictions'])!=prediction['prediction_sha256']:
        raise ValueError("Invalid validation prediction contract")
    rows=prediction['predictions'];y=[row['label'] for row in rows];p=[row['probabilities'] for row in rows]
    if (any(row.get('partition')!='validation' for row in rows) or len({r['case_id'] for r in rows})!=len(rows)
            or prediction['prediction_sha256']!=threshold.get('calibration_prediction_sha256')):
        raise ValueError("Validation replay cases or calibration prediction binding differ")
    metrics={'classification':classification_metrics(y,p,partition='validation'),
        'frozen_threshold':evaluate_frozen_threshold(y,p,threshold,checkpoint_sha256=prediction['checkpoint_sha256'],partition='validation')}
    atomic_json(output_path,metrics);return metrics
