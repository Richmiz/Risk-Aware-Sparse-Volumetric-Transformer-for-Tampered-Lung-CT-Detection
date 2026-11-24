"""Paired synthetic E3/E4/E5 controls, trained checkpoint profiles and statistics."""

import argparse
from copy import deepcopy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

from rsvt.data.acquisition import sha256_file
from rsvt.data.splits import digest_json
from rsvt.evaluation.efficiency import percentile, process_memory, synchronize
from rsvt.evaluation.statistics import paired_cluster_bootstrap, noninferiority_status, pareto_frontier
from rsvt.evaluation.thresholds import calibrate_source_threshold
from rsvt.experiments.ablations import load_catalog
from rsvt.experiments.phase6 import choose_device, weights_digest, _plain
from rsvt.experiments.phase9 import read_f08_jobs, run_phase9
from rsvt.experiments.runner import (atomic_json, build_model, checkpoint_load, fixtures, hardware_record,
    load_phase8_config, run_job, runtime_signature, seed_runtime, source_digest, verify_completed)
from rsvt.utils.config import load_config

ROOT = Path(__file__).resolve().parents[3]


def load_phase10_config(path):
    c = load_config(path)
    if set(c) != {'engineering', 'statistics', 'profiling'}: raise ValueError('Explicit F10 config required')
    e, s, p = (c[k] for k in ('engineering', 'statistics', 'profiling'))
    def integer(v, low, high): return type(v) is int and low <= v <= high
    if (set(e) != {'scope', 'device', 'threads', 'seeds', 'steps', 'sequence_caps', 'read_medical_volumes', 'allow_final_test'}
            or e['scope'] != 'synthetic_paired_ablation_and_cluster_statistics' or e['device'] not in ('auto', 'cpu', 'cuda')
            or e['read_medical_volumes'] is not False or e['allow_final_test'] is not False
            or not integer(e['threads'], 1, 4) or not integer(e['steps'], 1, 2)
            or not isinstance(e['seeds'], list) or not 1 <= len(e['seeds']) <= 5
            or any(type(v) is not int or v not in (13, 37, 73, 101, 137) for v in e['seeds'])
            or len(set(e['seeds'])) != len(e['seeds']) or e['sequence_caps'] != [64, 128, 256]):
        raise ValueError('F10 executes bounded procedural jobs only')
    if (set(s) != {'replicates', 'seed', 'confidence_level', 'cluster_key', 'method', 'undefined_policy', 'noninferiority_margin'}
            or not integer(s['replicates'], 1000, 10000) or not integer(s['seed'], 0, 2**32-1)
            or type(s['confidence_level']) is not float or s['confidence_level'] != .95
            or s['cluster_key'] != 'source_group' or s['method'] != 'paired_whole_cluster_percentile'
            or s['undefined_policy'] != 'suppress_interval_if_any_undefined' or s['noninferiority_margin'] != 'PENDING'):
        raise ValueError('Research margin remains pending; use the explicit demo bootstrap policy')
    if (set(p) != {'warmup', 'iterations', 'stage_iterations', 'batch_size', 'case_index', 'timing_scope'}
            or not integer(p['warmup'], 1, 5) or not integer(p['iterations'], 3, 30)
            or not integer(p['stage_iterations'], 1, 5) or type(p['batch_size']) is not int or p['batch_size'] != 1
            or type(p['case_index']) is not int or p['case_index'] != 0 or p['timing_scope'] != 'preloaded_FP32_forward_no_IO'):
        raise ValueError('Explicit bounded profiling scope required')
    return c


def paired_variants():
    base_path = ROOT / 'configs/orchestration/phase08.toml'
    base = load_phase8_config(base_path); reference = load_catalog(base_path, base)['B5']
    variants = {}
    for name in ('FULL', 'WITHOUT_CVAR', 'WITHOUT_GENERATOR', 'CAP128', 'CAP256'):
        entry = deepcopy(reference)
        entry['entry'].update(id='F10_' + name, kind='engineering_ablation_variant',
            description='F10 procedural paired control: ' + name)
        if name == 'WITHOUT_CVAR': entry['objective']['cvar'] = False
        if name == 'WITHOUT_GENERATOR': entry['objective']['generator_variance'] = False
        if name.startswith('CAP'): entry['model']['sequence_cap'] = int(name[3:])
        variants[name] = entry
    return base, variants


def file_receipt(path):
    return {'sha256': sha256_file(path), 'bytes': Path(path).stat().st_size}


def verify_receipt(root, name, expected):
    path = Path(root) / name
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()) or file_receipt(path) != expected:
        raise ValueError('Changed/escaped F10 artifact: ' + name)
    return path


def profile_checkpoint(checkpoint, config_path, output_path, *, device):
    c = load_phase10_config(config_path); p = c['profiling']; actual = choose_device(device)
    state, receipt = checkpoint_load(checkpoint); spec = state['job_spec']
    if spec['runtime'] != runtime_signature(actual) or spec['entry']['model']['family'] != 'rsvt':
        raise ValueError('Profiling requires matching runtime and a current-source synthetic RSVT checkpoint')
    batches, data = fixtures(spec['seed'], spec['shape_zyx'], actual, spec['threads'])
    if data != spec['data']: raise ValueError('Checkpoint input fingerprint differs')
    batch = {k: batches['validation'][k][:1].clone() for k in ('image', 'valid_mask', 'voxel_to_lps', 'padding_mask')}
    tensor_digest = hashlib.sha256()
    for key, value in batch.items(): tensor_digest.update(key.encode()+value.cpu().contiguous().numpy().tobytes())
    seed_runtime(spec['seed'], spec['threads'], actual)
    model = build_model(spec['entry'], spec['shape_zyx']).to(actual).eval(); model.load_state_dict(state['model'])
    trained_digest = weights_digest(model); del batches, state; gc.collect()
    if actual == 'cuda': torch.cuda.empty_cache()
    args = {'routing_mode': 'hard', 'sequence_cap': spec['entry']['model']['sequence_cap']}
    with torch.inference_mode():
        for _ in range(p['warmup']):
            temporary = model(**batch, **args); del temporary
        synchronize(actual); memory_before = process_memory()
        if actual == 'cuda':
            torch.cuda.reset_peak_memory_stats(); cuda_baseline = torch.cuda.memory_allocated()
        samples = []
        for _ in range(p['iterations']):
            synchronize(actual); start = time.perf_counter(); temporary = model(**batch, **args); synchronize(actual)
            samples.append((time.perf_counter()-start)*1000); del temporary
        memory_after = process_memory()
        cuda_memory = {'baseline_allocated_bytes': cuda_baseline, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()} if actual == 'cuda' else None
        stages = {}
        for _ in range(p['stage_iterations']):
            temporary = model(**batch, **args, profile_stages=True); compute = _plain(temporary['compute'])
            for key, value in temporary['stage_ms'].items(): stages.setdefault(key, []).append(value)
            del temporary
    signature = {'runtime': runtime_signature(actual), 'hardware': hardware_record(actual),
        'input_sha256': tensor_digest.hexdigest(), 'shape_zyx': spec['shape_zyx'], 'profile_policy': p}
    report = {'schema_version': 'f10_checkpoint_profile_v1', 'checkpoint_sha256': receipt['sha256'],
        'trained_weights_sha256': trained_digest, 'source_sha256': source_digest(), 'signature': signature,
        'matched_profile_signature': digest_json(signature), 'batch_size': 1, 'case_index': 0,
        'selection': 'minimum_unweighted_source_validation_CE_not_latency_selected',
        'sequence_cap': args['sequence_cap'], 'warmup': p['warmup'], 'iterations': p['iterations'],
        'latency_samples_ms': samples, 'p50_ms': percentile(samples, .5), 'p95_ms': percentile(samples, .95),
        'stage_ms_instrumented_p50': {k: percentile(v, .5) for k, v in stages.items()},
        'timing_scope': 'preloaded_FP32_forward_validation_stems_controller_packing_attention_metadata_no_IO',
        'stage_scope': 'separate_synchronized_forwards_not_additive_latency', 'compute': compute,
        'memory_before': memory_before, 'memory_after': memory_after, 'cuda_memory': cuda_memory,
        'memory_scope': 'fresh_worker_process_RSS_includes_imports_checkpoint_load_fixture_reconstruction_CUDA_peak_reset_after_warmup',
        'medical_data_read': False, 'scope': 'two_step_synthetic_checkpoint_not_medical_baseline_profile'}
    atomic_json(output_path, report); return report


def launch_profile(checkpoint, config_path, output_path, device):
    env = dict(os.environ); env['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + env.get('PYTHONPATH', '')
    command = [sys.executable, '-m', 'rsvt.experiments.phase10', '--profile-checkpoint', str(Path(checkpoint).resolve()),
        '--config', str(Path(config_path).resolve()), '--output', str(Path(output_path).resolve()), '--device', device]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    Path(output_path).with_suffix('.log').write_text(completed.stdout+completed.stderr, encoding='utf-8')
    if completed.returncode or not Path(output_path).is_file(): raise ValueError('Checkpoint profiling failed; inspect profile.log')
    return json.loads(Path(output_path).read_text(encoding='utf-8'))


def read_phase10(directory):
    root = Path(directory).resolve(); plan = json.loads((root/'plan.lock.json').read_text()); payload = plan['payload']
    ledger = json.loads((root/'ledger.json').read_text())
    if (digest_json(payload) != plan['contract_sha256'] or ledger['plan_sha256'] != plan['contract_sha256']
            or payload['scope'] != 'synthetic_paired_ablation_and_cluster_statistics'
            or payload['final_test_access_allowed'] is not False or set(ledger['jobs']) != {j['id'] for j in payload['jobs']}):
        raise ValueError('Incomplete/invalid F10 plan or ledger')
    for name, ref in ledger['prerequisite_artifacts'].items(): verify_receipt(root, name, ref)
    jobs = {}
    for job in payload['jobs']:
        identity = job['id']; record = ledger['jobs'][identity]
        if record['status'] != 'succeeded': raise ValueError('F10 job is not complete')
        base = root/'jobs'/identity
        path = verify_receipt(root, 'jobs/'+identity+'/complete.json', record['complete_receipt'])
        complete = json.loads(path.read_text()); result = verify_completed(base/'training', complete['training_result_sha256'], plan['contract_sha256'])
        for name, ref in complete['profile_artifacts'].items(): verify_receipt(base, name, ref)
        spec = json.loads((base/'training/job_contract.json').read_text())
        profile = json.loads((base/'profile.json').read_text())
        if (complete['plan_sha256'] != plan['contract_sha256']
                or spec['entry'] != payload['variants'][job['variant']] or spec['source_sha256'] != payload['source_sha256']
                or spec['seed'] != job['seed'] or result['seed'] != job['seed'] or profile['source_sha256'] != payload['source_sha256']
                or profile['checkpoint_sha256'] != result['best_checkpoint_sha256'] or spec['data'] != result['data']
                or profile['matched_profile_signature'] != digest_json(profile['signature'])
                or profile['signature']['runtime'] != payload['runtime']
                or profile['signature']['profile_policy'] != payload['config']['profiling']
                or profile['sequence_cap'] != spec['entry']['model']['sequence_cap']):
            raise ValueError('F10 variant/checkpoint/source mismatch')
        predictions = json.loads((base/'training/predictions.json').read_text())
        threshold = json.loads((base/'training/threshold.json').read_text())
        assignments = [{k: v for k, v in row.items() if k != 'probabilities'} for row in predictions]
        if assignments != spec['data']['validation']['records']: raise ValueError('Prediction assignment changed')
        recalculated = calibrate_source_threshold([r['label'] for r in predictions], [r['probabilities'] for r in predictions],
            target_fpr=payload['shared_executor_config']['demo_policy']['target_fpr'], partition='validation',
            checkpoint_sha256=result['best_checkpoint_sha256'], prediction_sha256=digest_json(predictions))
        if recalculated != threshold: raise ValueError('Frozen threshold/checkpoint/prediction binding changed')
        jobs[identity] = {'result': result, 'profile': profile, 'predictions': predictions,
            'threshold': threshold, 'seed': job['seed'], 'variant': job['variant']}
    return plan, jobs


def analyze_phase10(directory, output_dir):
    plan, jobs = read_phase10(directory); root = Path(directory); payload = plan['payload']; policy = payload['config']['statistics']
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False); contrasts = {}; cost_tables = {}
    def contrast(name, first, second):
        if first['result']['data'] != second['result']['data'] or first['result']['initial_weights_sha256'] != second['result']['initial_weights_sha256']:
            raise ValueError('Paired ablations require identical input tensors/assignments and initial weights')
        if first['profile']['matched_profile_signature'] != second['profile']['matched_profile_signature']:
            raise ValueError('Paired checkpoint profiles require matched hardware/input/policy')
        report = paired_cluster_bootstrap(first['predictions'], second['predictions'], first['threshold'], second['threshold'],
            replicates=policy['replicates'], seed=policy['seed'], confidence=policy['confidence_level'], cluster_key=policy['cluster_key'])
        report['noninferiority'] = noninferiority_status(report['metrics']['macro_f1_all_four']['percentile_interval'])
        atomic_json(output/'paired'/(name+'.json'), report)
        contrasts[name] = {k: report[k] for k in ('metrics', 'replicates', 'clusters', 'draw_sha256', 'noninferiority', 'research_inference_allowed')}
    for seed in payload['config']['engineering']['seeds']:
        subset = {j['variant']: j for j in jobs.values() if j['seed'] == seed}; full = subset['FULL']
        contrast(f'E3_with_minus_without_CVAR_s{seed}', full, subset['WITHOUT_CVAR'])
        contrast(f'E4_with_minus_without_generator_s{seed}', full, subset['WITHOUT_GENERATOR'])
        for cap in (128, 256): contrast(f'E5_cap{cap}_minus_cap64_s{seed}', subset['CAP'+str(cap)], full)
        rows = []
        for variant in ('FULL', 'CAP128', 'CAP256'):
            j = subset[variant]; profile = j['profile']; metrics = j['result']['metrics']['classification']
            rows.append({'id': variant, 'sequence_cap': profile['sequence_cap'], 'quality': metrics['macro_f1'],
                'latency_p95_ms': profile['p95_ms'], 'operator_flops': profile['compute']['operator_flops_estimate']['total'],
                'matched_profile_signature': profile['matched_profile_signature']})
        cost_tables[str(seed)] = pareto_frontier(rows)
    # B5/B1 is an architecture comparison, never a component ablation.
    saved = json.loads((root/'prerequisite_f09/predictions.json').read_text())
    calibrations = json.loads((root/'f08_calibrations.json').read_text()); baseline_comparisons = {}
    for identity in sorted(calibrations):
        if not identity.startswith('B5_'): continue
        other = identity.replace('B5_', 'B1_', 1)
        if other not in calibrations: continue
        pair = [[r for r in saved if r['experiment_id'] == name] for name in (identity, other)]
        report = paired_cluster_bootstrap(*pair, calibrations[identity], calibrations[other],
            replicates=policy['replicates'], seed=policy['seed'], confidence=policy['confidence_level'])
        name = identity+'_versus_'+other; atomic_json(output/'paired'/(name+'.json'), report)
        baseline_comparisons[name] = {'metrics': report['metrics'], 'scope': 'F08_architecture_comparison_not_component_ablation'}
    atomic_json(output/'budget_frontiers.json', cost_tables)
    with (output/'budget_costs.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['seed', 'id', 'sequence_cap', 'quality', 'latency_p95_ms', 'operator_flops', 'matched_profile_signature'])
        writer.writeheader()
        for seed, table in cost_tables.items():
            for row in table['rows']: writer.writerow({'seed': seed, **row})
    statuses = {'E1': 'pending_single_scale_capacity_controls', 'E2': 'pending_coarse_token_and_context_removal',
        'E3': 'synthetic_paired_CVAR_removal_executed_other_terms_fixed',
        'E4': 'synthetic_paired_generator_variance_removal_executed_other_terms_fixed',
        'E5': 'synthetic_trained_caps64_128_256_profiled', 'E6': 'pending_actual_LOGO_and_prespecified_shifts',
        'E7': 'pending_optimizer_seed_only_and_partition_robustness_canonical_QA'}
    checks = {'all_paired_data_and_initial_weights_matched': True,
        'four_contrasts_per_seed': len(contrasts) == 4*len(payload['config']['engineering']['seeds']),
        'at_least_1000_bootstrap_replicates': policy['replicates'] >= 1000,
        'noninferiority_not_invented': all(v['noninferiority']['status'] == 'not_assessed_margin_not_locked' for v in contrasts.values()),
        'trained_profile_hard_caps_enforced': all(j['profile']['compute']['actual_sequence_length_including_cls'] <= j['profile']['sequence_cap'] for j in jobs.values()),
        'profile_checkpoint_bindings_verified': True, 'medical_and_final_test_disabled': payload['final_test_access_allowed'] is False}
    summary = {'schema_version': 'rsvt_f10_analysis_v1', 'ok': all(checks.values()), 'checks': checks,
        'plan_sha256': plan['contract_sha256'], 'analysis_source_sha256': source_digest(), 'paired_contrasts': contrasts,
        'F08_baseline_comparisons': baseline_comparisons, 'ablation_status': statuses, 'budget_frontiers': cost_tables,
        'replicate_interpretation': 'each_seed_changes_fixture_and_initialization_analyze_separately_not_optimizer_seed_only_E7',
        'seed_pooling': 'disabled_no_patient_uncertainty_or_extra_independent_volume_claim',
        'research_inference_allowed': False, 'final_test_access_allowed': False,
        'scope': 'two_step_synthetic_engineering_not_confirmatory_medical_statistics',
        'artifacts': {p.relative_to(output).as_posix(): file_receipt(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    atomic_json(output/'summary.json', summary); return summary


def run_phase10(f08_run, config_path, output_dir, *, device=None, progress=None):
    c = load_phase10_config(config_path); actual = choose_device(device or c['engineering']['device'])
    f08_plan, old_jobs = read_f08_jobs(f08_run); base, variants = paired_variants()
    base['engineering'].update(seeds=c['engineering']['seeds'], steps=c['engineering']['steps'], threads=c['engineering']['threads'])
    payload = {'schema_version': 'f10_engineering_plan_v1', 'scope': c['engineering']['scope'], 'config': c,
        'variants': variants, 'shared_executor_config': base, 'source_sha256': source_digest(), 'runtime': runtime_signature(actual),
        'f08_plan_sha256': f08_plan['contract_sha256'], 'final_test_access_allowed': False,
        'jobs': [{'id': f'{v}_s{s}', 'variant': v, 'seed': s} for s in c['engineering']['seeds'] for v in variants]}
    plan = {'payload': payload, 'contract_sha256': digest_json(payload)}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False); atomic_json(output/'plan.lock.json', plan)
    run_phase9(f08_run, ROOT/'configs/evaluation/phase09.toml', output/'prerequisite_f09')
    atomic_json(output/'f08_calibrations.json', {name: j['threshold'] for name, j in old_jobs.items()})
    refs = {p.relative_to(output).as_posix(): file_receipt(p) for p in sorted((output/'prerequisite_f09').rglob('*')) if p.is_file()}
    refs['f08_calibrations.json'] = file_receipt(output/'f08_calibrations.json')
    ledger = {'plan_sha256': plan['contract_sha256'], 'prerequisite_artifacts': refs, 'jobs': {}}
    atomic_json(output/'ledger.json', ledger)
    for job in payload['jobs']:
        destination = output/'jobs'/job['id']; (destination/'training').mkdir(parents=True)
        result = run_job(variants[job['variant']], job['seed'], base, plan['contract_sha256'], destination/'training', actual)
        gc.collect()
        if actual == 'cuda': torch.cuda.empty_cache()
        profile = launch_profile(destination/'training'/result['best_checkpoint']['path'], config_path, destination/'profile.json', actual)
        complete = {'plan_sha256': plan['contract_sha256'], 'training_result_sha256': sha256_file(destination/'training/result.json'),
            'profile_artifacts': {n: file_receipt(destination/n) for n in ('profile.json', 'profile.log')}}
        atomic_json(destination/'complete.json', complete)
        ledger['jobs'][job['id']] = {'status': 'succeeded', 'complete_receipt': file_receipt(destination/'complete.json')}
        atomic_json(output/'ledger.json', ledger, replace=True)
        if progress: progress({'event': 'paired_job_and_profile_complete', 'job': job['id'], 'p95_ms': profile['p95_ms']})
    analysis = analyze_phase10(output, output/'analysis')
    summary = {'schema_version': 'rsvt_f10_engineering_v1', 'phase': 'F10', 'ok': analysis['ok'], 'checks': analysis['checks'],
        'jobs_completed': len(ledger['jobs']), 'source_sha256': payload['source_sha256'], 'plan_sha256': plan['contract_sha256'],
        'analysis_summary': file_receipt(output/'analysis/summary.json'), 'ablation_status': analysis['ablation_status'],
        'medical_data_read': False, 'test_cases_opened': 0, 'final_test_access_allowed': False, 'research_ready': False,
        'research_readiness': f08_plan['payload']['research_readiness'],
        'prerequisite_F09': 'fresh_recalculation_from_F08_not_review_of_standalone_F09_Colab_run',
        'rerun_policy': 'new_run_only_preserve_partial_outputs_no_optimizer_resume',
        'scope': 'synthetic_E3_E4_E5_and_cluster_statistics_only'}
    atomic_json(output/'summary.json', summary); return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--profile-checkpoint', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True); args = parser.parse_args()
    profile_checkpoint(args.profile_checkpoint, args.config, args.output, device=args.device)
