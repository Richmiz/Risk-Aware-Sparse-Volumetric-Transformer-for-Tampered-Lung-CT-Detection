"""Fixed-bin calibration diagnostics; no fitted calibration or target tuning."""

import numpy as np
from rsvt.evaluation.classification import validate_probabilities


def reliability_bins(outcomes, confidence, *, bins=10):
    y, p = np.asarray(outcomes), np.asarray(confidence, dtype=float)
    if (isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 100
            or y.ndim != 1 or not len(y) or p.shape != y.shape
            or not np.isin(y, (0, 1)).all() or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()):
        raise ValueError('Invalid reliability inputs or bin count')
    # [left,right), except the final bin includes 1. Both 0 and 1 are counted.
    assignment = np.minimum((p * bins).astype(int), bins - 1)
    rows = []; ece = 0.
    for i in range(bins):
        mask = assignment == i; count = int(mask.sum())
        observed = float(y[mask].mean()) if count else None
        mean = float(p[mask].mean()) if count else None
        gap = abs(observed - mean) if count else None
        if count: ece += count / len(y) * gap
        rows.append({'left': i / bins, 'right': (i + 1) / bins, 'count': count,
                     'mean_probability': mean, 'observed_frequency': observed, 'absolute_gap': gap})
    return {'ece': ece, 'bins': rows, 'bin_policy': '[left,right), final includes 1',
            'cases': len(y), 'scope': 'descriptive_bin_dependent_not_population_calibration_guarantee'}


def calibration_report(targets, probabilities, *, bins=10):
    y, p = validate_probabilities(targets, probabilities)
    one_hot = np.eye(4)[y]; tamper = np.clip(p[:, 2:].sum(1), 0, 1); binary = (y >= 2).astype(int)
    epsilon = 1e-12
    return {'multiclass_brier_sum': float(np.square(p - one_hot).sum(1).mean()),
            'multiclass_brier_range': [0, 2],
            'multiclass_nll': float(-np.log(np.clip(p[np.arange(len(y)), y], epsilon, 1)).mean()),
            'nll_clip_epsilon': epsilon,
            'top_label': reliability_bins((p.argmax(1) == y).astype(int), p.max(1), bins=bins),
            'tamper_brier': float(np.square(tamper - binary).mean()),
            'tamper_reliability': reliability_bins(binary, tamper, bins=bins),
            'probability_transform': 'none_no_temperature_fitting', 'bins_locked_for_demo': bins,
            'aggregate_roundoff_policy': 'clip_tamper_sum_to_unit_interval_for_Brier_and_reliability_only'}
