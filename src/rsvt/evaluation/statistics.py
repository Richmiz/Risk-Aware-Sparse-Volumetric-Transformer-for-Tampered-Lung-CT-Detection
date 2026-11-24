"""Paired whole-cluster percentile bootstrap of frozen volume predictions."""

import math
import numpy as np
from rsvt.data.splits import digest_json
from rsvt.evaluation.classification import four_class_report
from rsvt.evaluation.cross_generator import validate_rows
from rsvt.evaluation.thresholds import bound_threshold_report


def align_pairs(first, second, *, cluster_key='source_group'):
    if cluster_key not in ('source_group', 'patient_id'):
        raise ValueError('Use an explicitly supplied source or patient cluster')
    for rows in (first, second):
        if not isinstance(rows, list) or not rows: raise ValueError('Nonempty paired rows required')
        partitions = {r.get('partition') for r in rows}
        if len(partitions) != 1 or next(iter(partitions)) not in ('validation', 'synthetic_holdout'):
            raise ValueError('One engineering evaluation partition per contrast required')
        validate_rows(rows, partition=next(iter(partitions)))
        if any(not isinstance(r.get(cluster_key), str) or not r[cluster_key].strip() for r in rows):
            raise ValueError('Cluster identifiers must be present; no patient fallback')
        folds = {r.get('fold_id', 'single_fold') for r in rows}
        if len(folds) != 1: raise ValueError('Repeated folds require a separate prespecified correlation model')
    a, b = ({r['case_id']: r for r in rows} for rows in (first, second))
    if set(a) != set(b): raise ValueError('Models must predict exactly the same volumes')
    keys = sorted(a)
    required = {'label', 'source_group', 'partition', 'generator_id', 'generator_verified', cluster_key}
    optional = {'patient_id', 'volume_sha256', 'fold_id'}
    for key in keys:
        if any(a[key].get(k) != b[key].get(k) for k in required | optional):
            raise ValueError('Paired labels/source/generator/fold metadata differ')
    return [a[k] for k in keys], [b[k] for k in keys]


def endpoint_values(rows, calibration, checkpoint_sha256):
    labels = [r['label'] for r in rows]; probabilities = [r['probabilities'] for r in rows]
    classification = four_class_report(labels, probabilities)
    threshold = bound_threshold_report(labels, probabilities, calibration, checkpoint_sha256=checkpoint_sha256)
    return {'FB_tamper_recall': threshold['FB_tamper_recall'], 'authentic_FPR': threshold['authentic_FPR'],
            'macro_f1_all_four': classification['macro_f1_all_four'], 'tamper_auroc': classification['tamper_auroc']}


def paired_cluster_bootstrap(first, second, calibration_first, calibration_second, *, replicates=1000,
                             seed=113, confidence=.95, cluster_key='source_group'):
    """Delta = first minus second; identical cluster draws for both models.

    Any undefined replicate suppresses that metric's interval rather than
    reporting a conditional interval after silently dropping missing classes.
    Thresholds remain fixed, conditioning on the fitted/calibrated models.
    """
    if (type(replicates) is not int or not 1000 <= replicates <= 10000 or type(seed) is not int
            or not 0 <= seed < 2**32 or type(confidence) not in (int, float)
            or not math.isfinite(confidence) or not 0 < confidence < 1):
        raise ValueError('Require 1000-10000 replicates, valid seed and confidence')
    a, b = align_pairs(first, second, cluster_key=cluster_key)
    checkpoints = [c['checkpoint_sha256'] for c in (calibration_first, calibration_second)]
    observed = [endpoint_values(r, c, s) for r, c, s in zip((a, b), (calibration_first, calibration_second), checkpoints)]
    groups = sorted({r[cluster_key] for r in a})
    if len(groups) < 2: raise ValueError('At least two supplied clusters required')
    members = [[i for i, r in enumerate(a) if r[cluster_key] == g] for g in groups]
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(replicates, len(groups)))
    distribution = []
    for draw in draws:
        indices = [i for group in draw for i in members[group]]
        values = [endpoint_values([r[i] for i in indices], c, s)
                  for r, c, s in zip((a, b), (calibration_first, calibration_second), checkpoints)]
        distribution.append({'sampled_volumes': len(indices), 'delta': {
            k: values[0][k] - values[1][k] if values[0][k] is not None and values[1][k] is not None else None
            for k in observed[0]}})
    metrics = {}; alpha = (1 - confidence) / 2
    for key in observed[0]:
        values = [d['delta'][key] for d in distribution if d['delta'][key] is not None]
        point = observed[0][key] - observed[1][key] if observed[0][key] is not None and observed[1][key] is not None else None
        interval = np.quantile(values, [alpha, 1-alpha], method='linear').tolist() if len(values) == replicates and point is not None else None
        metrics[key] = {'first': observed[0][key], 'second': observed[1][key], 'delta_first_minus_second': point,
            'percentile_interval': interval, 'valid_replicates': len(values), 'undefined_replicates': replicates-len(values),
            'degenerate_distribution': len(set(values)) == 1 if values else None,
            'interval_status': 'computed_descriptive_only' if interval is not None else 'unavailable_unsupported_observed_or_resampled_class'}
    return {'schema_version': 'f10_paired_cluster_bootstrap_v1', 'metrics': metrics, 'replicates': replicates,
        'seed': seed, 'confidence_level': confidence, 'method': 'paired_whole_cluster_percentile_linear_quantile',
        'cluster_key': cluster_key, 'clusters': groups, 'volumes': len(a),
        'cluster_members': {g: [a[i]['case_id'] for i in member] for g, member in zip(groups, members)},
        'draw_indices': draws.tolist(), 'draw_sha256': digest_json(draws.tolist()), 'distribution': distribution,
        'threshold_bindings': [c['binding_sha256'] for c in (calibration_first, calibration_second)],
        'uncertainty_scope': 'conditional_on_fixed_fitted_models_and_validation_thresholds_not_training_or_calibration_uncertainty',
        'undefined_policy': 'any_undefined_replicate_suppresses_interval_no_conditional_filtering',
        'multiplicity': 'exploratory_pointwise_intervals_no_adjustment_no_confirmatory_decision',
        'canonical_patient_independence_verified': False,
        'research_inference_allowed': False}


def noninferiority_status(interval, *, margin=None):
    if margin is None:
        return {'status': 'not_assessed_margin_not_locked', 'margin': None}
    if type(margin) not in (float, int) or not math.isfinite(margin) or not 0 < margin < 1:
        raise ValueError('Explicit positive margin in metric units required')
    if interval is None: return {'status': 'not_assessed_interval_unavailable', 'margin': margin}
    x = np.asarray(interval, dtype=float)
    if x.shape != (2,) or not np.isfinite(x).all() or x[0] > x[1]: raise ValueError('Ordered finite interval required')
    status = 'noninferior_by_supplied_rule' if x[0] > -margin else 'inferior_by_supplied_rule' if x[1] < -margin else 'inconclusive'
    return {'status': status, 'margin': margin, 'delta_direction': 'higher_is_better_first_minus_second',
            'rule': 'lower_bound_strictly_greater_than_negative_margin', 'scope': 'numeric_helper_not_a_research_protocol_decision'}


def pareto_frontier(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows): raise ValueError('Unique nonempty cost rows required')
    if len({r['matched_profile_signature'] for r in rows}) != 1: raise ValueError('Matched hardware/input/profile scope required')
    for row in rows:
        if any(type(row[k]) not in (float, int) or not math.isfinite(row[k]) for k in ('quality', 'latency_p95_ms', 'operator_flops')):
            raise ValueError('Finite quality/cost required')
        if not 0 <= row['quality'] <= 1 or row['latency_p95_ms'] <= 0 or row['operator_flops'] <= 0:
            raise ValueError('Invalid quality or nonpositive costs')
    def dominates(a, b):
        weak = a['quality'] >= b['quality'] and a['latency_p95_ms'] <= b['latency_p95_ms'] and a['operator_flops'] <= b['operator_flops']
        strict = a['quality'] > b['quality'] or a['latency_p95_ms'] < b['latency_p95_ms'] or a['operator_flops'] < b['operator_flops']
        return weak and strict
    return {'frontier_ids': sorted(r['id'] for r in rows if not any(dominates(o, r) for o in rows if o['id'] != r['id'])),
            'rows': rows, 'scope': 'descriptive_measured_fixture_frontier_no_latency_guarantee_or_research_superiority'}
