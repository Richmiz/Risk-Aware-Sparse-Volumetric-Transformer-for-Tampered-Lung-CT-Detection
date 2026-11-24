"""Source-validation-only empirical fixed-FPR calibration for synthetic interfaces."""

import math
import numpy as np
from rsvt.data.splits import digest_json


def calibrate_source_threshold(targets,probabilities,*,target_fpr,partition,checkpoint_sha256,prediction_sha256):
    if partition!='validation':raise ValueError("Threshold calibration requires source validation only")
    if isinstance(target_fpr,bool) or not isinstance(target_fpr,(float,int)) or not math.isfinite(target_fpr) or not 0<=target_fpr<1:
        raise ValueError("Target FPR must be finite in [0,1)")
    y=np.asarray(targets);p=np.asarray(probabilities,dtype=float)
    if (y.ndim!=1 or p.shape!=(len(y),4) or not len(y) or not np.isfinite(p).all() or (p<0).any() or (p>1).any()
            or not np.allclose(p.sum(1),1.,atol=1e-6) or not np.isin(y,[0,1,2,3]).all()):raise ValueError("Invalid calibration probabilities/targets")
    scores=p[:,2:].sum(1);authentic=scores[y<2]
    if not len(authentic):raise ValueError("Authentic validation cases are required")
    allowed=math.floor(target_fpr*len(authentic))
    threshold=float(np.sort(authentic)[::-1][allowed]);false_positives=int((authentic>threshold).sum())
    payload={'schema_version':'f08_threshold_v1','score':'p_FB_plus_p_FM','comparison':'strict_gt','threshold':threshold,
        'target_fpr':target_fpr,'authentic_cases':len(authentic),'allowed_validation_FP':allowed,'observed_validation_FP':false_positives,
        'empirical_validation_fpr':false_positives/len(authentic),'volume_fpr_resolution':1/len(authentic),
        'selection_partition':'validation','checkpoint_sha256':checkpoint_sha256,'calibration_prediction_sha256':prediction_sha256,
        'scope':'synthetic_source_validation_empirical_not_population_FPR_guarantee'}
    return {**payload,'binding_sha256':digest_json(payload)}


def evaluate_frozen_threshold(targets,probabilities,calibration,*,checkpoint_sha256,partition):
    if partition!='validation':raise ValueError("F08 evaluation is validation-only; final test is unavailable")
    payload={k:v for k,v in calibration.items() if k!='binding_sha256'}
    if digest_json(payload)!=calibration.get('binding_sha256') or calibration.get('checkpoint_sha256')!=checkpoint_sha256:
        raise ValueError("Threshold binding/checkpoint mismatch")
    if calibration.get('score')!='p_FB_plus_p_FM' or calibration.get('comparison')!='strict_gt':raise ValueError("Unsupported score/comparison")
    y=np.asarray(targets);p=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or not len(y) or not np.isin(y,[0,1,2,3]).all() or p.shape!=(len(y),4) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1.,atol=1e-6):raise ValueError("Invalid frozen-threshold probabilities")
    decisions=p[:,2:].sum(1)>calibration['threshold'];auth=y<2;fb=y==2;predicted=p.argmax(1)
    return {'scope':'synthetic_validation_only','score':calibration['score'],'threshold':calibration['threshold'],
        'authentic_FPR':float(decisions[auth].mean()) if auth.any() else None,
        'FB_tamper_recall':float(decisions[fb].mean()) if fb.any() else None,
        'FB_classification_recall':float((predicted[fb]==2).mean()) if fb.any() else None,
        'threshold_binding_sha256':calibration['binding_sha256']}


def bound_threshold_report(targets, probabilities, calibration, *, checkpoint_sha256):
    """Apply only a sealed source threshold; never select one on the target rows."""
    from rsvt.evaluation.classification import validate_probabilities
    payload = {k: v for k, v in calibration.items() if k != 'binding_sha256'}
    if (calibration.get('schema_version') != 'f08_threshold_v1'
            or digest_json(payload) != calibration.get('binding_sha256')
            or calibration.get('checkpoint_sha256') != checkpoint_sha256
            or calibration.get('selection_partition') != 'validation'
            or calibration.get('score') != 'p_FB_plus_p_FM'
            or calibration.get('comparison') != 'strict_gt'
            or type(calibration.get('threshold')) not in (int, float)
            or not math.isfinite(calibration['threshold']) or not 0 <= calibration['threshold'] <= 1):
        raise ValueError('Invalid source threshold/checkpoint binding')
    for name in ('checkpoint_sha256', 'calibration_prediction_sha256'):
        value = calibration.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('SHA256 binding required')
    y, p = validate_probabilities(targets, probabilities)
    decisions = p[:, 2:].sum(1) > calibration['threshold']; auth = y < 2; fake = y >= 2
    counts = {'authentic_volumes': int(auth.sum()), 'tampered_volumes': int(fake.sum()),
              'FP': int(decisions[auth].sum()), 'TN': int((~decisions[auth]).sum()),
              'TP': int(decisions[fake].sum()), 'FN': int((~decisions[fake]).sum())}
    report = {**counts, 'threshold': calibration['threshold'], 'comparison': 'strict_gt',
              'threshold_binding_sha256': calibration['binding_sha256'],
              'authentic_FPR': float(decisions[auth].mean()) if auth.any() else None,
              'tamper_recall': float(decisions[fake].mean()) if fake.any() else None,
              'volume_FPR_resolution': 1 / int(auth.sum()) if auth.any() else None,
              'target_fpr_from_validation': calibration['target_fpr'],
              'scope': 'empirical_volume_endpoint_not_population_FPR_guarantee'}
    for label, name in ((2, 'FB'), (3, 'FM')):
        mask = y == label
        report[name + '_volumes'] = int(mask.sum())
        report[name + '_tamper_recall'] = float(decisions[mask].mean()) if mask.any() else None
        report[name + '_classification_recall'] = float((p.argmax(1)[mask] == label).mean()) if mask.any() else None
    return report
