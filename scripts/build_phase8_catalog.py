"""Generate explicit proposal-aligned B0-B5/E1-E7 declarations, not final results."""

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build():
    baseline={
        'B0':('3D ResNet50/GN; DenseNet121 alternative pending','resnet50_3d',True,False,False,False),
        'B1':('Fixed-patch dense 3D ViT reference','vit3d',True,False,False,False),
        'B2':('Small-region local-global two-stage cascade','two_stage_cascade',False,False,False,False),
        'B3':('Routing-only RSVT foundation','rsvt',True,True,False,False),
        'B4':('Asymmetric CVaR without routing or generator regularization','rsvt',True,False,True,False),
        'B5':('Routing plus CVaR plus generator regularization','rsvt',True,True,True,True)}
    written=[]
    for identity,(description,family,supported,routing,risk,generator) in baseline.items():
        data={'entry':{'id':identity,'kind':'baseline','description':description,'synthetic_backend_supported':supported,
            'medical_backend_enabled':False,'exact_reference_reproduction':False,
            'unsupported_reason':'' if supported else 'Two-stage cascade not implemented; no substituted baseline'},
            'model':{'family':family,'dimension':16,'layers':1,'heads':2,'cnn_width':2,'patch_size':8,
                'coarse_patch_size':16,'candidate_patch_sizes':[4,8,16],'sequence_cap':64,'localization_head':False},
            'objective':{'routing':routing,'cvar':risk,'generator_variance':generator},
            'controls':{'source_grouped':True,'patient_disjoint_required':True,'selection_partition':'validation',
                'preprocessing_shared':True,'scope':'synthetic_adapter_not_paper_reproduction_or_matched_cost_claim'}}
        path=ROOT/f'configs/experiments/registry/{identity}.toml';write_toml(path,data);written.append(path)
    contrasts={
        'E1':('B5','Fixed versus adaptive multiscale candidate selection',['fixed_scale','multiscale_adaptive'],
            'Single-scale paired executor and capacity controls pending'),
        'E2':('B5','With versus without coarse anatomical context',['with_coarse','without_coarse'],
            'Explicit no-coarse ablation must disable both mandatory tokens and router context; pending'),
        'E3':('B5','Class-balanced CE versus FB/FM CVaR',['without_cvar','with_cvar'],'Paired ablation executor pending F10'),
        'E4':('B5','Without versus with generator variance',['without_generator','with_generator'],'Verified multi-generator images and paired executor pending'),
        'E5':('B5','Budget performance/cost frontier',['cap64','cap128','cap256'],'Paired source-validation and full-cost profiling pending'),
        'E6':('B5','Held-out generators and preprocessing shifts',['generator_holdout','preprocessing_shifts'],'LOGO images, held-out ID and shift policy pending'),
        'E7':('B5','Repeated seeds/partitions and leakage audit',['seeds','partitions','leakage_audit'],'Final canonical groups, split policy and paired executor pending')}
    for identity,(reference,description,conditions,reason) in contrasts.items():
        data={'entry':{'id':identity,'kind':'ablation','reference':reference,'description':description,
            'synthetic_backend_supported':False,'medical_backend_enabled':False,'exact_reference_reproduction':False,'unsupported_reason':reason},
            'contrast':{'conditions':conditions,'candidate_seeds':[13,37,73,101,137],'candidate_sequence_caps':[64,128,256],
                'split_unit':'source_volume_id','matched_partitions_required':True,'matched_preprocessing_required':True,
                'selection_partition':'validation','status':'design_only_not_executed'}}
        path=ROOT/f'configs/ablations/{identity}.toml';write_toml(path,data);written.append(path)
    return written


def write_toml(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);lines=[]
    for table,values in data.items():
        lines.append(f'[{table}]')
        for key,value in values.items():lines.append(f'{key} = {json.dumps(value,ensure_ascii=False)}')
        lines.append('')
    path.write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    for path in build():print(path.relative_to(ROOT))
