"""Supplied-metadata LOGO audits and descriptive generator reports; no training."""

import numpy as np
from rsvt.evaluation.classification import validate_probabilities, binary_auroc


def validate_rows(rows, *, partition):
    if partition not in ('train', 'validation', 'synthetic_holdout'):
        raise ValueError('F09 dispatcher accepts engineering partitions only')
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError('Nonempty prediction/assignment rows required')
    for row in rows:
        if (any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('case_id', 'source_group'))
                or row.get('partition') != partition or type(row.get('label')) is not int
                or row['label'] not in range(4) or type(row.get('generator_id')) is not int
                or type(row.get('generator_verified')) is not bool):
            raise ValueError('Invalid case/source/partition/label/generator metadata')
        if row['label'] < 2:
            if row['generator_id'] != -1 or row['generator_verified']:
                raise ValueError('Authentic cases must not be attributed to a generator')
        elif row['generator_id'] < 0 or not row['generator_verified']:
            raise ValueError('Tampered generator identity must be verified')
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate volume case IDs')
    return rows


def audit_source_separation(partitions):
    owners = {}; identities = set()
    for partition, rows in partitions.items():
        validate_rows(rows, partition=partition)
        for row in rows:
            if row['case_id'] in identities: raise ValueError('Case reused across partitions')
            identities.add(row['case_id'])
            for key in ('source_group', 'patient_id', 'volume_sha256'):
                value = row.get(key)
                if value is None: continue
                if not isinstance(value, str) or not value.strip(): raise ValueError('Invalid lineage key')
                token = (key, value)
                if token in owners and owners[token] != partition:
                    raise ValueError('Source/patient/content leakage across partitions')
                owners[token] = partition
    return {'ok': True, 'groups_per_partition': {
        p: len({r['source_group'] for r in rows}) for p, rows in partitions.items()},
        'independence_basis': 'supplied_metadata_only_no_identity_reconciliation',
        'canonical_patient_independence_verified': False}


def audit_logo_fold(partitions, *, held_out_generator):
    if set(partitions) != {'train', 'validation', 'synthetic_holdout'}:
        raise ValueError('Explicit train/validation/synthetic_holdout required')
    if type(held_out_generator) is not int or held_out_generator < 0:
        raise ValueError('Invalid held-out generator')
    audit = audit_source_separation(partitions)
    sets = {}
    for partition, rows in partitions.items():
        if {r['label'] for r in rows} != {0, 1, 2, 3}:
            raise ValueError('All four classes required in each demonstration partition')
        sets[partition] = {r['generator_id'] for r in rows if r['label'] >= 2}
    if (held_out_generator in sets['train'] | sets['validation']
            or not sets['validation'] <= sets['train']
            or sets['synthetic_holdout'] != {held_out_generator}):
        raise ValueError('Target generator leaks into source training/calibration, or invalid source set')
    return {**audit, 'held_out_generator': held_out_generator,
            'source_generators': sorted(sets['train']),
            'scope': 'metadata_fold_contract_only_no_LOGO_model_trained',
            'external_validation_demonstrated': False}


def per_generator_report(rows, *, threshold):
    partitions = {r.get('partition') for r in rows}
    if len(partitions) != 1: raise ValueError('A report must contain one partition')
    validate_rows(rows, partition=next(iter(partitions)))
    y, p = validate_probabilities([r['label'] for r in rows], [r['probabilities'] for r in rows])
    if isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid frozen threshold')
    auth = y < 2; scores = p[:, 2:].sum(1); detected = scores > threshold
    generators = np.array([r['generator_id'] for r in rows]); output = {}
    for generator in sorted(set(generators[y >= 2].tolist())):
        own = (y >= 2) & (generators == generator); fb = own & (y == 2); fm = own & (y == 3)
        subset = own | auth
        output[str(generator)] = {'tampered_volumes': int(own.sum()), 'FB_volumes': int(fb.sum()),
            'FM_volumes': int(fm.sum()),
            'source_groups': len({r['source_group'] for i, r in enumerate(rows) if own[i]}),
            'FB_tamper_recall': float(detected[fb].mean()) if fb.any() else None,
            'FB_classification_recall': float((p.argmax(1)[fb] == 2).mean()) if fb.any() else None,
            'FM_tamper_recall': float(detected[fm].mean()) if fm.any() else None,
            'tamper_auroc_with_shared_authentic': binary_auroc((y[subset] >= 2).astype(int), scores[subset])}
    values = [r['FB_tamper_recall'] for r in output.values()]
    worst = min(values) if values and all(v is not None for v in values) else None
    return {'by_generator': output, 'worst_generator_FB_tamper_recall': worst,
            'worst_generator_policy': 'null_if_any_generator_has_no_FB_support',
            'shared_authentic_volumes': int(auth.sum()),
            'authentic_FPR_once': float(detected[auth].mean()) if auth.any() else None,
            'control_reuse': 'same_authentic_controls_per_generator_AUC_never_sum_as_independent_volumes',
            'scope': 'descriptive_volume_metrics_not_independent_patient_or_external_validation'}
