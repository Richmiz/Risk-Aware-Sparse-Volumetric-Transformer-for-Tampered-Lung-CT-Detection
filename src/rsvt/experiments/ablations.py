"""Validated experiment declarations; design-only entries never silently execute."""

from pathlib import Path
from rsvt.utils.config import load_config


def load_catalog(config_path,config):
    root=Path(config_path).resolve().parent;entries={}
    paths=config['catalog'].get('baselines',[])+config['catalog'].get('ablations',[])
    for relative in paths:
        if not isinstance(relative,str):raise ValueError("Catalog paths must be strings")
        data=load_config(root/relative);entry=data.get('entry',{});identity=entry.get('id')
        if identity in entries:raise ValueError("Duplicate experiment ID")
        if identity not in {f'B{i}' for i in range(6)}|{f'E{i}' for i in range(1,8)}:
            raise ValueError("Unknown catalog ID")
        kind='baseline' if identity.startswith('B') else 'ablation'
        if entry.get('kind')!=kind or any(not isinstance(entry.get(key),bool) for key in
            ('synthetic_backend_supported','medical_backend_enabled','exact_reference_reproduction')):
            raise ValueError("Invalid catalog kind/flags")
        if entry['medical_backend_enabled'] is not False:raise ValueError("F08 has no medical execution backend")
        if kind=='ablation' and (entry['synthetic_backend_supported'] or not data.get('contrast',{}).get('conditions')):
            raise ValueError("E1-E7 remain explicit paired-analysis designs, not executors")
        if not entry['synthetic_backend_supported'] and not entry.get('unsupported_reason'):
            raise ValueError("Unsupported entries require a reason")
        if kind=='baseline':
            model=data.get('model',{});objective=data.get('objective',{})
            if set(objective)!={'routing','cvar','generator_variance'} or any(not isinstance(v,bool) for v in objective.values()):
                raise ValueError("Baseline objective switches must be booleans")
            if model.get('family') not in ('resnet50_3d','vit3d','rsvt','two_stage_cascade'):
                raise ValueError("Unknown model family")
            if entry['synthetic_backend_supported'] and model['family']=='two_stage_cascade':raise ValueError("Cascade backend unavailable")
            if model.get('localization_head') is not False:raise ValueError("F08 matched interface smoke uses classification only")
        entries[identity]=data
    if set(entries)!={f'B{i}' for i in range(6)}|{f'E{i}' for i in range(1,8)}:raise ValueError("Catalog must declare B0-B5 and E1-E7")
    if not (entries['B3']['model']==entries['B4']['model']==entries['B5']['model']):raise ValueError("B3/B4/B5 must share the prototype architecture/budget")
    expected={'B0':(False,False,False),'B1':(False,False,False),'B2':(False,False,False),
              'B3':(True,False,False),'B4':(False,True,False),'B5':(True,True,True)}
    for key,settings in expected.items():
        if tuple(entries[key]['objective'][name] for name in ('routing','cvar','generator_variance'))!=settings:
            raise ValueError("Baseline objective switches do not match the proposal-aligned catalog")
    return entries
