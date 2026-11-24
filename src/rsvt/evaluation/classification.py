"""Exploratory train/validation metrics; no final-test or clinical claims."""

import numpy as np

LABELS = ("TB", "TM", "FB", "FM")


def classification_metrics(targets, probabilities, *, partition):
    if partition not in {"train", "validation"}:
        raise ValueError("Exploratory metrics accept train/validation only")
    y, p = np.asarray(targets, dtype=int), np.asarray(probabilities, dtype=float)
    if (y.ndim != 1 or not len(y) or p.shape != (len(y),4) or not np.isfinite(p).all()
            or np.any((y < 0) | (y > 3)) or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(1),1.,atol=1e-6)):
        raise ValueError("Invalid four-class probabilities")
    prediction = p.argmax(1)
    matrix = np.zeros((4,4),dtype=int)
    np.add.at(matrix,(y,prediction),1)
    recalls, f1 = {}, {}
    for index, name in enumerate(LABELS):
        support, predicted = matrix[index].sum(), matrix[:,index].sum()
        recalls[name] = float(matrix[index,index]/support) if support else None
        f1[name] = float(2*matrix[index,index]/(support+predicted)) if support else None
    authentic, manipulated = y < 2, y >= 2
    scores = p[:,2:].sum(1)
    auroc = None
    if authentic.any() and manipulated.any():
        difference = scores[manipulated,None]-scores[None,authentic]
        auroc = float(np.mean((difference > 0) + .5*(difference == 0)))
    threshold_info = None
    if partition == "validation" and authentic.any():
        threshold = float(scores[authentic].max())
        detected = scores > threshold
        fb = y == 2
        threshold_info = {"policy":"maximum_authentic_validation_score_strict_gt", "threshold":threshold,
            "comparison":">", "authentic_candidates":int(authentic.sum()),
            "empirical_fpr":float(detected[authentic].mean()), "fpr_resolution":1./int(authentic.sum()),
            "fb_tamper_recall":float(detected[fb].mean()) if fb.any() else None,
            "scope":"exploratory_validation_only_not_confirmed_population_fpr"}
    return {"cases":len(y), "accuracy":float(np.mean(prediction==y)), "confusion_matrix":matrix.tolist(),
            "class_recall":recalls, "class_f1":f1,
            "class_support":{name:int(matrix[i].sum()) for i,name in enumerate(LABELS)},
            "predicted_class_counts":{name:int(matrix[:,i].sum()) for i,name in enumerate(LABELS)},
            "balanced_accuracy":float(np.mean(list(recalls.values()))) if all(v is not None for v in recalls.values()) else None,
            "dominant_prediction_fraction":float(matrix.sum(0).max()/len(y)),
            "constant_argmax":bool(matrix.sum(0).max()==len(y)),
            "hard_tamper_authentic_fpr":float(np.mean(prediction[authentic]>=2)) if authentic.any() else None,
            "macro_f1":float(np.mean(list(f1.values()))) if all(v is not None for v in f1.values()) else None,
            "tamper_auroc":auroc, "validation_threshold":threshold_info,
            "label_status":"candidate", "scope":partition+"_only"}


def validation_metrics(targets, probabilities):
    return classification_metrics(targets, probabilities, partition="validation")


def validate_probabilities(targets, probabilities):
    """Strict four-class array contract shared by F09 offline metrics."""
    y, p = np.asarray(targets), np.asarray(probabilities, dtype=float)
    if (y.ndim != 1 or not len(y) or y.dtype.kind not in 'iu'
            or not np.isin(y, (0, 1, 2, 3)).all() or p.shape != (len(y), 4)
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(1), 1., atol=1e-6, rtol=0)):
        raise ValueError('Invalid integer targets/four-class probabilities')
    return y, p


def binary_auroc(targets, scores):
    """Mann-Whitney AUC with average ranks for ties, O(n log n) memory/time."""
    y, s = np.asarray(targets), np.asarray(scores, dtype=float)
    if (y.ndim != 1 or s.shape != y.shape or not len(y)
            or not np.isin(y, (0, 1)).all() or not np.isfinite(s).all()):
        raise ValueError('Invalid binary targets/scores')
    positive = int(y.sum()); negative = len(y) - positive
    if not positive or not negative:
        return None
    order = np.argsort(s, kind='stable'); sorted_scores = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(y)]
    ranks = np.repeat((starts + 1 + ends) / 2., ends - starts)
    return float((ranks[y[order] == 1].sum() - positive * (positive + 1) / 2.)
                 / (positive * negative))


def four_class_report(targets, probabilities):
    """Pure array metrics; filesystem dispatch/protocol authorization is separate."""
    y, p = validate_probabilities(targets, probabilities)
    predicted = p.argmax(1); matrix = np.zeros((4, 4), dtype=int)
    np.add.at(matrix, (y, predicted), 1)
    recalls, f1, auc = {}, {}, {}
    for i, name in enumerate(LABELS):
        support, predicted_count = int(matrix[i].sum()), int(matrix[:, i].sum())
        recalls[name] = float(matrix[i, i] / support) if support else None
        f1[name] = float(2 * matrix[i, i] / (support + predicted_count)) if support else None
        auc[name] = binary_auroc((y == i).astype(int), p[:, i])
    def strict_mean(values):
        return float(np.mean(list(values.values()))) if all(v is not None for v in values.values()) else None
    return {'cases': len(y), 'class_order': list(LABELS), 'confusion_matrix': matrix.tolist(),
            'accuracy': float((y == predicted).mean()), 'class_recall': recalls,
            'class_f1': f1, 'macro_f1_all_four': strict_mean(f1),
            'balanced_accuracy_all_four': strict_mean(recalls),
            'class_support': {n: int(matrix[i].sum()) for i, n in enumerate(LABELS)},
            'class_ovr_auroc': auc, 'macro_ovr_auroc_all_four': strict_mean(auc),
            'tamper_auroc': binary_auroc((y >= 2).astype(int), p[:, 2:].sum(1)),
            'missing_class_policy': 'null_for_unsupported_class_and_all_four_macro',
            'argmax_tie_policy': 'lowest_class_index'}
