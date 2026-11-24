"""F09 sealed-prediction evaluation and explicit synthetic metric demonstrations."""

import csv
import json
from pathlib import Path
import re
import numpy as np

from rsvt.data.acquisition import sha256_file
from rsvt.data.splits import digest_json
from rsvt.evaluation.classification import classification_metrics, four_class_report
from rsvt.evaluation.calibration import calibration_report
from rsvt.evaluation.cross_generator import validate_rows, audit_source_separation, audit_logo_fold, per_generator_report
from rsvt.evaluation.localization import point_localization, verified_mask_metrics, voxel_xyz_to_lps
from rsvt.evaluation.thresholds import calibrate_source_threshold, bound_threshold_report, evaluate_frozen_threshold
from rsvt.experiments.runner import atomic_json, source_digest, verify_completed
from rsvt.utils.config import load_config


def load_phase9_config(path):
    c = load_config(path)
    expected = {'scope': 'engineering_saved_synthetic_validation_only', 'ece_bins': 10,
        'demo_point_radius_mm': 5.0, 'read_medical_volumes': False, 'allow_final_test': False,
        'fit_probability_calibration': False}
    protocol = {'status': 'draft', 'research_protocol_locked': False, 'actual_LOGO_training': False,
        'independent_patient_inference': False, 'matched_trained_model_profiling': False}
    # This release has one explicit demonstration policy. No research override.
    if (set(c) != {'evaluation', 'protocol'} or c['evaluation'] != expected or c['protocol'] != protocol
            or any(type(c['evaluation'][k]) is not type(v) for k, v in expected.items())
            or any(type(c['protocol'][k]) is not type(v) for k, v in protocol.items())):
        raise ValueError('F09 supports the explicit engineering policy only; final test is unavailable')
    return c


def evaluate_rows(rows, calibration, checkpoint_sha256, *, bins):
    if not isinstance(rows, list) or not rows: raise ValueError('Nonempty prediction rows required')
    validate_rows(rows, partition=rows[0]['partition'])
    y, p = [r['label'] for r in rows], [r['probabilities'] for r in rows]
    return {'classification': four_class_report(y, p),
            'frozen_threshold': bound_threshold_report(y, p, calibration, checkpoint_sha256=checkpoint_sha256),
            'calibration': calibration_report(y, p, bins=bins),
            'generators': per_generator_report(rows, threshold=calibration['threshold']),
            'volume_cases': len(rows), 'supplied_source_groups': len({r['source_group'] for r in rows}),
            'unit': 'one_saved_probability_vector_per_volume_no_group_aggregation',
            'independent_patients_verified': False}


def read_f08_jobs(directory):
    """Hash synthetic checkpoints as opaque bytes; never torch.load or open images."""
    root = Path(directory).resolve()
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    plan, ledger, summary = read('plan.lock.json'), read('ledger.json'), read('summary.json')
    payload = plan['payload']; binding = plan['contract_sha256']
    if (digest_json(payload) != binding or ledger['plan_sha256'] != binding or summary['plan_sha256'] != binding
            or payload['kind'] != 'engineering_draft_snapshot_not_final_protocol'
            or payload['resolved_config']['engineering']['scope'] != 'synthetic_only'
            or payload['test_volume_access_allowed'] is not False
            or summary.get('ok') is not True or not all(summary['checks'].values())
            or summary.get('medical_data_read') is not False or summary.get('test_cases_opened') != 0):
        raise ValueError('Invalid/uncompleted F08 engineering snapshot')
    ids = {j['id'] for j in payload['jobs']}
    if len(ids) != len(payload['jobs']) or ids != set(ledger['jobs']) or ids != set(summary['results']):
        raise ValueError('F08 job ledger/plan mismatch')
    jobs = {}
    for identity in sorted(ids):
        record = ledger['jobs'][identity]
        if (not re.fullmatch(r'B[01345]_s[0-9]+', identity) or record.get('status') != 'succeeded'
                or not re.fullmatch(r'attempt_[0-9]+', record.get('completed_attempt', ''))):
            raise ValueError('Invalid completed synthetic job path/state')
        attempt = root / 'jobs' / identity / record['completed_attempt']
        if not attempt.resolve().is_relative_to(root): raise ValueError('Job path escapes run')
        result = verify_completed(attempt, record['result_sha256'], binding)
        read_job = lambda name: json.loads((attempt / name).read_text(encoding='utf-8'))
        spec, rows, threshold = read_job('job_contract.json'), read_job('predictions.json'), read_job('threshold.json')
        if (digest_json(spec) != result['job_contract_sha256'] or spec['source_sha256'] != payload['source_sha256']
                or spec['plan_sha256'] != binding or spec['entry'] != payload['catalog'][result['experiment_id']]
                or result['data'] != spec['data'] or result['ok'] is not True or result['test_cases_opened'] != 0):
            raise ValueError('Job contract/source/data mismatch')
        if spec['seed'] == payload['resolved_config']['engineering']['seeds'][0] and spec['data'] != payload['synthetic_dataset']:
            raise ValueError('Job fixture differs from plan snapshot')
        validate_rows(rows, partition='validation')
        assignments = [{k: v for k, v in row.items() if k != 'probabilities'} for row in rows]
        if assignments != spec['data']['validation']['records']:
            raise ValueError('Saved predictions differ from bound validation assignment')
        audit = audit_source_separation({p: spec['data'][p]['records'] for p in ('train', 'validation')})
        checkpoint = result['best_checkpoint_sha256']; checkpoint_path = result['best_checkpoint']['path']
        if result['artifacts'][checkpoint_path]['sha256'] != checkpoint:
            raise ValueError('Best checkpoint hash mismatch')
        y, p = [r['label'] for r in rows], [r['probabilities'] for r in rows]
        recomputed = calibrate_source_threshold(y, p, target_fpr=payload['resolved_config']['demo_policy']['target_fpr'],
            partition='validation', checkpoint_sha256=checkpoint, prediction_sha256=digest_json(rows))
        if recomputed != threshold: raise ValueError('Frozen threshold does not match source validation')
        legacy = {'classification': classification_metrics(y, p, partition='validation'),
            'frozen_threshold': evaluate_frozen_threshold(y, p, threshold, checkpoint_sha256=checkpoint, partition='validation')}
        if legacy != result['metrics'] or legacy != summary['results'][identity]['metrics']:
            raise ValueError('Recomputed F08 metrics mismatch')
        jobs[identity] = {'result': result, 'rows': rows, 'threshold': threshold, 'source_audit': audit}
    return plan, jobs


def metric_demonstrations(config):
    folds = []
    for heldout in (0, 1):
        partitions = {}
        for partition in ('train', 'validation', 'synthetic_holdout'):
            rows = []
            scores = [.1, .3, .8, .9] if partition == 'synthetic_holdout' else [.1, .2, .7, .8]
            for label, score in enumerate(scores):
                rows.append({'case_id': f'demo:fold{heldout}:{partition}:c{label}',
                    'source_group': f'demo:fold{heldout}:{partition}:parent{label}', 'partition': partition,
                    'label': label, 'generator_id': (heldout if partition == 'synthetic_holdout' else 1 - heldout) if label >= 2 else -1,
                    'generator_verified': label >= 2,
                    'probabilities': [(1-score)*.5, (1-score)*.5, score*.1, score*.9]})
            partitions[partition] = rows
        audit = audit_logo_fold(partitions, held_out_generator=heldout)
        val = partitions['validation']; fake_checkpoint = digest_json({'probability_fixture_no_model': heldout})
        threshold = calibrate_source_threshold([r['label'] for r in val], [r['probabilities'] for r in val],
            target_fpr=.05, partition='validation', checkpoint_sha256=fake_checkpoint, prediction_sha256=digest_json(val))
        report = evaluate_rows(partitions['synthetic_holdout'], threshold, fake_checkpoint, bins=config['ece_bins'])
        folds.append({'audit': audit, 'partitions': partitions, 'frozen_threshold': threshold, 'report': report,
                      'checkpoint_digest_kind': 'placeholder_digest_no_model_checkpoint_exists'})
    affine = np.array([[0, -2, 0, 10], [1, 0, 0, -20], [0, 0, 3, 5], [0, 0, 0, 1.]])
    reference = voxel_xyz_to_lps([[1, 2, 3], [4, 5, 6]], affine)
    peaks = reference + [[1, 0, 0], [0, 0, 6]]
    points = point_localization(peaks, reference, verified=True, radius_mm=config['demo_point_radius_mm'], coordinate_system='LPS_mm')
    target = np.zeros((4, 4, 4), dtype=bool); target[1:3, 1:3, 1:3] = True
    prediction = target.copy(); prediction[0, 0, 0] = True
    masks = verified_mask_metrics(prediction, target, np.ones_like(target), verified=True,
                                 prediction_frame='procedural_grid', reference_frame='procedural_grid')
    return {'scope': 'handcrafted_probability_and_spatial_fixtures_not_trained_LOGO_or_medical_results',
            'model_trained': False, 'folds': folds, 'point_localization': points, 'mask_localization': masks,
            'box_metrics': 'unavailable_no_box_to_mask_conversion',
            'fold_pooling': 'disabled_no_repeated_control_or_case_independence_claim'}


def run_phase9(f08_run, config_path, output_dir, *, progress=None):
    config = load_phase9_config(config_path); plan, jobs = read_f08_jobs(f08_run)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    input_root = Path(f08_run).resolve()
    origin_run = input_root.parent.name if input_root.name == 'orchestration' else input_root.name
    reports, enriched = {}, []
    for identity, job in jobs.items():
        checkpoint = job['result']['best_checkpoint_sha256']
        report = evaluate_rows(job['rows'], job['threshold'], checkpoint, bins=config['evaluation']['ece_bins'])
        report.update(scope='saved_F08_synthetic_in_domain_validation_not_final_test_or_LOGO',
                      checkpoint_sha256=checkpoint, source_audit=job['source_audit'])
        atomic_json(output / 'models' / (identity + '.json'), report); reports[identity] = report
        for row in job['rows']:
            enriched.append({**row, 'experiment_id': identity, 'origin_run': origin_run,
                'fold_id': 'in_domain_validation', 'checkpoint_sha256': checkpoint,
                'threshold_binding_sha256': job['threshold']['binding_sha256']})
        if progress: progress({'event': 'saved_predictions_evaluated', 'job': identity, 'volumes': len(job['rows'])})
    atomic_json(output / 'predictions.json', enriched)
    fields = ['experiment_id', 'origin_run', 'fold_id', 'case_id', 'source_group', 'partition', 'label',
              'generator_id', 'generator_verified', 'p_TB', 'p_TM', 'p_FB', 'p_FM', 'checkpoint_sha256', 'threshold_binding_sha256']
    with (output / 'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in enriched:
            csvrow = {k: v for k, v in row.items() if k != 'probabilities'}
            csvrow.update(zip(('p_TB', 'p_TM', 'p_FB', 'p_FM'), row['probabilities'])); writer.writerow(csvrow)
    demo = metric_demonstrations(config['evaluation']); atomic_json(output / 'metric_demonstrations.json', demo)
    # No old checkpoint inference or latency claims after changing the source tree.
    efficiency = {'new_profiling_performed': False, 'matched_trained_model_profile_available': False,
        'available_engineering_profiler': 'phase6-routing_isolated_workers',
        'prior_evidence': 'reports/phase06_colab_review.md',
        'scope': 'F06_random_weight_routing_profiles_are_not_F08_trained_model_profiles',
        'required_before_research_comparison': ['same_hardware_and_input', 'matched_trained_checkpoints',
            'warmup_device_synchronization', 'batch_size', 'IO_scope', 'p50_p95_samples',
            'process_RSS_CUDA_memory_scope', 'controller_overhead', 'operator_FLOP_exclusions']}
    atomic_json(output / 'efficiency_status.json', efficiency)
    checks = {'all_saved_jobs_evaluated': set(reports) == set(jobs),
        'legacy_F08_metrics_recomputed': True,
        'all_prediction_rows_exported': len(enriched) == sum(len(j['rows']) for j in jobs.values()),
        'calibration_bins_account_for_all_volumes': all(sum(b['count'] for b in r['calibration']['top_label']['bins']) == r['volume_cases'] for r in reports.values()),
        'LOGO_metadata_two_rotations': len(demo['folds']) == 2 and all(f['audit']['ok'] for f in demo['folds']),
        'holdout_FPR_not_forced_to_validation_target': all(f['report']['frozen_threshold']['authentic_FPR'] == .5 for f in demo['folds']),
        'FB_detection_distinct_from_classification': all(f['report']['frozen_threshold']['FB_tamper_recall'] == 1 and f['report']['frozen_threshold']['FB_classification_recall'] == 0 for f in demo['folds']),
        'physical_point_radius_and_mask_counts': demo['point_localization']['matched_points'] == 1 and demo['mask_localization']['TP'] == 8 and demo['mask_localization']['FP'] == 1}
    summary = {'schema_version': 'rsvt_f09_engineering_v1', 'phase': 'F09', 'ok': all(checks.values()), 'checks': checks,
        'source_sha256': source_digest(), 'config_sha256': digest_json(config), 'f08_plan_sha256': plan['contract_sha256'],
        'models_evaluated': len(reports), 'prediction_rows': len(enriched), 'results': reports,
        'medical_data_read': False, 'test_cases_opened': 0, 'research_ready': False, 'final_test_access_allowed': False,
        'actual_LOGO_model_trained': False, 'research_readiness': plan['payload']['research_readiness'],
        'efficiency': efficiency, 'scope': 'engineering_evaluation_not_research_final_test',
        'pending': ['canonical_identity_label_annotation_QA', 'locked_source_validation_research_protocol',
            'actual_multigenerator_LOGO_training_evaluation', 'matched_trained_model_profiling',
            'medical_localization_verified_annotations', 'paired_ablation_and_cluster_statistics_F10']}
    artifacts = {p.relative_to(output).as_posix(): {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
                 for p in sorted(output.rglob('*')) if p.is_file()}
    summary['artifacts'] = artifacts; atomic_json(output / 'summary.json', summary)
    return summary
