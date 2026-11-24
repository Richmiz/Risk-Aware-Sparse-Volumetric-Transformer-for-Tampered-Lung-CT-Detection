"""Bind the proposed RSVT to the existing pinned F03 CT manifest."""

import importlib
import json
import math
from pathlib import Path
import re

from rsvt.data.acquisition import sha256_file
from rsvt.data.baseline_views import CachedViews, prepare_views, select_pilot_rows
from rsvt.data.phase2 import _json
from rsvt.data.splits import CLASS_INDEX, digest_json
from rsvt.models.proposed import RSVT
from rsvt.training.optimization import validate_optimization
from rsvt.training.proposed import fit_rsvt
from rsvt.utils.config import load_config
from rsvt.utils.reproducibility import seed_everything


def load_proposed_config(path):
    config = load_config(path)
    if set(config) != {'training', 'data', 'model', 'risk', 'stages', 'dual'}:
        raise ValueError('Explicit RSVT training/data/model/risk/stages/dual tables required')
    t, data, risk, stages = (config[k] for k in ('training', 'data', 'risk', 'stages'))
    if t.get('mode') != 'engineering_pilot' or t.get('candidate_targets_acknowledged') is not True:
        raise ValueError('Explicit exploratory training acknowledgement required')
    for key, minimum in [('epochs', 3), ('patience', 1), ('threads', 1), ('seed', 0), ('logical_batch_size', 4)]:
        value = t[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum: raise ValueError('Invalid '+key)
    for key, zero in [('learning_rate', False), ('weight_decay', True), ('min_delta', True), ('gradient_clip', False)]:
        value = t[key]
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0 or not zero and value == 0:
            raise ValueError('Invalid '+key)
    validate_optimization(t)
    if t['device'] not in {'cpu', 'cuda'}: raise ValueError('Explicit cpu/cuda device required')
    if (not re.fullmatch('[0-9a-f]{64}', data['expected_assignment_sha256']) or len(data['shape_zyx']) != 3
            or any(isinstance(n,bool) or not isinstance(n,int) or not 32 <= n <= 96 for n in data['shape_zyx'])):
        raise ValueError('Pin source assignments and a bounded full-FOV view')
    for key in ('train_per_class', 'validation_per_class', 'max_decode_bytes'):
        if isinstance(data[key],bool) or not isinstance(data[key],int) or data[key] < (1 if key == 'max_decode_bytes' else 0):
            raise ValueError('Invalid data bound')
    if config['model'].get('localization_head') is not False: raise ValueError('Voxel localization is disabled')
    for key in ('baseline_epochs', 'routing_warmup_epochs'):
        if isinstance(stages[key],bool) or not isinstance(stages[key],int) or stages[key] < 1: raise ValueError('Invalid stages')
    if sum(stages.values()) >= t['epochs']: raise ValueError('At least one risk-finetuning epoch required')
    if t['patience'] <= sum(stages.values()): raise ValueError('Patience must permit reaching risk finetuning')
    if (isinstance(risk['alpha'],bool) or not isinstance(risk['alpha'],(int,float)) or not math.isfinite(risk['alpha'])
            or not 0 <= risk['alpha'] < 1): raise ValueError('Invalid CVaR confidence')
    for key in ('cvar_weight', 'fb_weight', 'fm_weight'):
        value = risk[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite CVaR coefficients required')
    if isinstance(risk['min_class_samples'],bool) or not isinstance(risk['min_class_samples'],int) or risk['min_class_samples'] < 1:
        raise ValueError('Invalid risk support')
    # Constructor validates architectural dimensions and resource bounds.
    seed_everything(t['seed'], t['threads'])
    model = RSVT(**config['model'])
    shape = data['shape_zyx']; count = lambda p: math.prod(math.ceil(n/p) for n in shape)
    sequence = 1 + sum(count(p) for p in (model.coarse_patch_size, *model.patch_sizes))
    if (sequence > model.max_tokens or model.heads*sequence**2 > model.max_attention_elements
            or model.sequence_cap < count(model.coarse_patch_size)+1):
        raise ValueError('View violates dense-training or mandatory coarse+CLS token budget')
    from rsvt.losses.budget import ProjectedTokenDual
    ProjectedTokenDual(**config['dual'])
    return config


def implementation_fingerprint():
    names = ('rsvt.training.proposed_run', 'rsvt.training.proposed', 'rsvt.training.optimization',
        'rsvt.training.risk_trainer', 'rsvt.training.checkpoint', 'rsvt.data.baseline_views', 'rsvt.data.dataset',
        'rsvt.models.proposed', 'rsvt.models.risk_rsvt', 'rsvt.models.routed_rsvt', 'rsvt.models.rsvt', 'rsvt.models.routing',
        'rsvt.models.transformer', 'rsvt.models.coarse_encoder', 'rsvt.models.tokenization', 'rsvt.models.anomaly_features',
        'rsvt.models.geometry', 'rsvt.models.heads', 'rsvt.losses.cvar', 'rsvt.losses.budget',
        'rsvt.losses.classification', 'rsvt.evaluation.classification', 'rsvt.utils.reproducibility')
    return digest_json({name: sha256_file(Path(importlib.import_module(name).__file__)) for name in names})


def run_proposed(f03_run, config_path, output_dir, *, resume=False, stop_after_epoch=None, progress=None):
    import torch
    config = load_proposed_config(config_path)
    if config['training']['device'] == 'cuda' and not torch.cuda.is_available(): raise ValueError('Select a Colab GPU')
    source, output = Path(f03_run).resolve(), Path(output_dir).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source): raise ValueError('Separate training from source evidence')
    manifest = source/'manifest.draft.json'; before = sha256_file(manifest)
    document = json.loads(manifest.read_text(encoding='utf-8'))
    evidence = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if (evidence.get('ok') is not True or evidence['manifest_sha256'] != before
            or document['assignments_sha256'] != evidence['assignments_sha256']
            or document['assignments_sha256'] != config['data']['expected_assignment_sha256']
            or document['class_index'] != CLASS_INDEX): raise ValueError('Pinned F03 evidence differs')
    selected = select_pilot_rows(document['records'], config['training']['seed'],
        config['data']['train_per_class'], config['data']['validation_per_class'])
    for row in document['records']:
        original = Path(row['volume_path']).parent.resolve()
        if output.is_relative_to(original) or original.is_relative_to(output): raise ValueError('Preserve processed originals')
    contract = {'model': 'RSVT', 'implementation_sha256': implementation_fingerprint(), 'config_sha256': digest_json(config),
        'manifest_sha256': before, 'assignments_sha256': document['assignments_sha256'],
        'selected_cases_sha256': digest_json([{k: r[k] for k in ('case_key','split','label_candidate','normalized_sha256','valid_mask_sha256','metadata_sha256')} for r in selected])}
    if output.exists() and not resume: raise FileExistsError('Use a new run or explicit resume')
    if resume:
        if json.loads((output/'run_contract.json').read_text()) != contract: raise ValueError('Resume contract differs')
    else:
        output.mkdir(parents=True)
        _json(output/'run_contract.json', contract); _json(output/'resolved_config.json', config)
        _json(output/'selected_cases.json', [{k:r[k] for k in ('case_key','split','label_candidate')} for r in selected])
    try:
        views = prepare_views(manifest, selected, output/'views', config['data']['shape_zyx'],
            max_decode_bytes=config['data']['max_decode_bytes'], progress=progress)
        train, validation = (CachedViews(views, split, include_geometry=True) for split in ('train','validation'))
        seed_everything(config['training']['seed'], config['training']['threads'])
        model = RSVT(**config['model'])
        result = fit_rsvt(model, train, validation, config, output,
            {**contract, 'view_index_sha256': sha256_file(output/'views/index.json')},
            resume=resume and (output/'last.pt').is_file(), stop_after_epoch=stop_after_epoch, progress=progress)
        result['source_manifest_unchanged'] = sha256_file(manifest) == before
        if not result['source_manifest_unchanged']: raise ValueError('Source manifest changed')
        _json(output/'summary.json', result)
        (output/'report.md').write_text('# RSVT volume-classification run\n\n'
            f"Train: {len(train)}; validation: {len(validation)}; test opened: 0.\n\n"
            'Hard-packed validation; minimum validation CE selects the checkpoint.\n'
            'Exploratory results; source identity remains unresolved.\n'
            'Generator variance and spatial localization are disabled.\n'
            'See docs/rsvt_theory.md for theory, objective and limitations.\n', encoding='utf-8')
        return result
    except Exception as exc:
        _json(output/'failure.json', {'ok': False, 'reason': str(exc), 'training_ready': False})
        raise
