"""Verified, annotation-matched spatial diagnostics with explicit coordinate units."""

import numpy as np


def voxel_xyz_to_lps(points, affine):
    p, a = np.asarray(points, dtype=float), np.asarray(affine, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or a.shape != (4, 4)
            or not np.isfinite(a).all() or not np.allclose(a[3], [0, 0, 0, 1])
            or abs(np.linalg.det(a[:3, :3])) < 1e-12):
        raise ValueError('Finite XYZ points and nonsingular voxel-to-LPS affine required')
    return p @ a[:3, :3].T + a[:3, 3]


def point_localization(predicted_lps_mm, target_lps_mm, *, verified, radius_mm, coordinate_system):
    if verified is not True or coordinate_system != 'LPS_mm':
        raise ValueError('Verified LPS_mm point annotations required')
    if isinstance(radius_mm, bool) or not np.isfinite(radius_mm) or radius_mm < 0:
        raise ValueError('Explicit nonnegative matching radius in mm required')
    p, t = np.asarray(predicted_lps_mm, dtype=float), np.asarray(target_lps_mm, dtype=float)
    if any(a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() for a in (p, t)):
        raise ValueError('Point arrays must be finite N by 3, including empty (0,3) arrays')
    distances = np.linalg.norm(t[:, None, :] - p[None, :, :], axis=2)
    # Maximum-cardinality one-to-one matching within radius, not greedy nearest.
    edges = distances <= radius_mm; owner = {}
    def augment(target, seen):
        for peak in np.flatnonzero(edges[target]).tolist():
            if peak in seen: continue
            seen.add(peak)
            if peak not in owner or augment(owner[peak], seen):
                owner[peak] = target; return True
        return False
    for index in range(len(t)): augment(index, set())
    matched = len(owner)
    return {'annotation_type': 'verified_points', 'coordinate_system': 'LPS_mm',
            'radius_mm': float(radius_mm), 'target_points': len(t), 'predicted_points': len(p),
            'matched_points': matched, 'lesion_sensitivity': matched / len(t) if len(t) else None,
            'unmatched_prediction_count': len(p) - matched,
            'minimum_target_distance_mm': distances.min(1).tolist() if len(p) else [None] * len(t),
            'matching_policy': 'maximum_cardinality_one_to_one_distance_le_radius',
            'dice_or_iou': None, 'scope': 'supplied_peaks_no_peak_threshold_tuning_or_mask_conversion'}


def verified_mask_metrics(prediction, target, valid, *, verified, prediction_frame, reference_frame, coverage=None):
    if verified is not True or not isinstance(reference_frame, str) or not reference_frame or prediction_frame != reference_frame:
        raise ValueError('Verified masks on the identical reference grid required')
    p, t, v = np.asarray(prediction), np.asarray(target), np.asarray(valid)
    if (p.ndim != 3 or p.shape != t.shape or p.shape != v.shape
            or any(a.dtype != bool for a in (p, t, v)) or not v.any()):
        raise ValueError('Boolean 3D masks with nonempty valid support required')
    if coverage is not None:
        c = np.asarray(coverage)
        if c.dtype != bool or c.shape != p.shape or not c[v].all():
            raise ValueError('Incomplete fine-token coverage cannot be treated as background')
    tp, fp, fn = int((p & t & v).sum()), int((p & ~t & v).sum()), int((~p & t & v).sum())
    denominator = 2 * tp + fp + fn; union = tp + fp + fn
    return {'annotation_type': 'verified_voxel_mask', 'reference_frame': reference_frame,
            'valid_voxels': int(v.sum()), 'TP': tp, 'FP': fp, 'FN': fn,
            'dice': 2 * tp / denominator if denominator else None,
            'iou': tp / union if union else None,
            'empty_union_policy': 'null_no_positive_union', 'coverage_policy': 'all_valid_voxels_required'}
