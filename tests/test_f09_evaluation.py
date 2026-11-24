"""Analytic metric oracles, provenance failures and sealed synthetic integration."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from rsvt.evaluation.classification import binary_auroc, four_class_report, validate_probabilities
from rsvt.evaluation.calibration import calibration_report, reliability_bins
from rsvt.evaluation.cross_generator import audit_logo_fold, audit_source_separation, per_generator_report, validate_rows
from rsvt.evaluation.localization import point_localization, verified_mask_metrics, voxel_xyz_to_lps
from rsvt.evaluation.thresholds import bound_threshold_report
from rsvt.experiments.phase9 import load_phase9_config, metric_demonstrations, read_f08_jobs, run_phase9
from rsvt.experiments.runner import run_phase8
from test_f08_orchestration import small_config

CONFIG = ROOT / 'configs/evaluation/phase09.toml'


def demo():
    return metric_demonstrations(load_phase9_config(CONFIG)['evaluation'])


class ClassificationAndCalibration(unittest.TestCase):
    def test_rank_auc_matches_pairwise_oracle_with_ties(self):
        rng = np.random.default_rng(113)
        for size in (2, 9, 100):
            y = np.arange(size) % 2; scores = rng.integers(0, 5, size)
            differences = scores[y == 1, None] - scores[None, y == 0]
            oracle = np.mean((differences > 0) + .5 * (differences == 0))
            self.assertAlmostEqual(binary_auroc(y, scores), oracle)
        self.assertIsNone(binary_auroc([0, 0], [.1, .9]))
        self.assertEqual(binary_auroc([0, 1], [.5, .5]), .5)

    def test_perfect_four_class_and_reversed_tamper_auc(self):
        r = four_class_report([0, 1, 2, 3], np.eye(4))
        self.assertEqual(r['macro_f1_all_four'], 1)
        self.assertEqual(r['macro_ovr_auroc_all_four'], 1)
        self.assertEqual(r['tamper_auroc'], 1)
        self.assertEqual(binary_auroc([0, 1], [.9, .1]), 0)

    def test_missing_classes_remain_explicitly_undefined(self):
        r = four_class_report([0, 0], [[1, 0, 0, 0]] * 2)
        self.assertIsNone(r['class_recall']['FB'])
        self.assertIsNone(r['macro_f1_all_four'])
        self.assertIsNone(r['macro_ovr_auroc_all_four'])
        self.assertIsNone(r['tamper_auroc'])

    def test_invalid_label_probabilities_rejected(self):
        for y, p in [([True], [[1, 0, 0, 0]]), ([1.5], [[1, 0, 0, 0]]),
                     ([4], [[1, 0, 0, 0]]), ([0], [[np.nan, 0, 0, 0]]),
                     ([0], [[1, .1, 0, 0]]), ([], np.empty((0, 4)))]:
            with self.subTest(y=y), self.assertRaises(ValueError): validate_probabilities(y, p)

    def test_bin_boundaries_empty_bins_and_exact_ece(self):
        r = reliability_bins([0, 1, 1], [0, .5, 1], bins=2)
        self.assertEqual([b['count'] for b in r['bins']], [1, 2])
        self.assertAlmostEqual(r['ece'], 1 / 6)
        self.assertEqual(sum(b['count'] for b in r['bins']), 3)
        self.assertIsNone(reliability_bins([0], [0], bins=10)['bins'][5]['observed_frequency'])
        for bins in (True, 0, 101, 2.5):
            with self.assertRaises(ValueError): reliability_bins([1], [.2], bins=bins)

    def test_perfect_and_wrong_brier_nll_toplabel(self):
        perfect = calibration_report([0, 1, 2, 3], np.eye(4))
        self.assertEqual(perfect['multiclass_brier_sum'], 0)
        self.assertEqual(perfect['tamper_brier'], 0)
        self.assertEqual(perfect['multiclass_nll'], 0)
        self.assertEqual(perfect['top_label']['ece'], 0)
        wrong = calibration_report([0], [[0, 0, 0, 1]])
        self.assertEqual(wrong['multiclass_brier_sum'], 2)
        self.assertEqual(wrong['tamper_brier'], 1)
        self.assertEqual(wrong['top_label']['ece'], 1)
        self.assertAlmostEqual(wrong['multiclass_nll'], -np.log(1e-12))

    def test_float_softmax_roundoff_keeps_reliability_probability_in_bounds(self):
        r = calibration_report([3], [[0, 0, .5, .50000001]])
        self.assertEqual(r['tamper_brier'], 0)
        self.assertEqual(sum(b['count'] for b in r['tamper_reliability']['bins']), 1)


class FrozenEndpointAndGenerators(unittest.TestCase):
    def test_target_fpr_is_observed_not_recalibrated(self):
        fold = demo()['folds'][0]; r = fold['report']['frozen_threshold']
        self.assertEqual(r['authentic_FPR'], .5)
        self.assertEqual(r['target_fpr_from_validation'], .05)
        self.assertEqual(r['FP'], 1)
        self.assertEqual(r['FB_tamper_recall'], 1)
        self.assertEqual(r['FB_classification_recall'], 0)

    def test_modified_threshold_checkpoint_and_selection_partition_rejected(self):
        f = demo()['folds'][0]; rows = f['partitions']['synthetic_holdout']; original = f['frozen_threshold']
        for key, value in [('threshold', .9), ('selection_partition', 'synthetic_holdout'), ('comparison', 'ge')]:
            c = {**original, key: value}
            with self.assertRaises(ValueError): bound_threshold_report([r['label'] for r in rows],
                [r['probabilities'] for r in rows], c, checkpoint_sha256=original['checkpoint_sha256'])
        with self.assertRaises(ValueError): bound_threshold_report([r['label'] for r in rows],
            [r['probabilities'] for r in rows], original, checkpoint_sha256='0' * 64)

    def test_shared_authentic_controls_not_counted_as_new_generator_volumes(self):
        f = demo()['folds'][0]; rows = deepcopy(f['partitions']['synthetic_holdout'])
        extra = deepcopy(rows[2:])
        for r in extra: r.update(case_id=r['case_id'] + 'extra', generator_id=2)
        r = per_generator_report(rows + extra, threshold=.2)
        self.assertEqual(r['shared_authentic_volumes'], 2)
        self.assertEqual(len(r['by_generator']), 2)
        self.assertEqual(r['worst_generator_FB_tamper_recall'], 1)

    def test_no_fb_support_worst_group_is_not_silently_dropped(self):
        rows = deepcopy(demo()['folds'][0]['partitions']['synthetic_holdout']); rows[2]['generator_id'] = 2
        r = per_generator_report(rows, threshold=.2)
        self.assertIsNone(r['by_generator']['0']['FB_tamper_recall'])
        self.assertIsNone(r['worst_generator_FB_tamper_recall'])


class FoldProvenance(unittest.TestCase):
    def test_two_generator_rotations_have_no_trained_model_claim(self):
        d = demo()
        self.assertFalse(d['model_trained'])
        self.assertEqual({f['audit']['held_out_generator'] for f in d['folds']}, {0, 1})
        self.assertTrue(all(not f['audit']['external_validation_demonstrated'] for f in d['folds']))

    def test_parent_patient_and_content_overlap_rejected(self):
        for key in ('source_group', 'patient_id', 'volume_sha256'):
            p = deepcopy(demo()['folds'][0]['partitions'])
            p['train'][0][key] = p['synthetic_holdout'][0][key] = 'same-parent'
            with self.subTest(key=key), self.assertRaises(ValueError): audit_source_separation(p)

    def test_target_generator_cannot_enter_train_or_calibration(self):
        for part in ('train', 'validation'):
            p = deepcopy(demo()['folds'][0]['partitions']); p[part][2]['generator_id'] = 0
            with self.assertRaises(ValueError): audit_logo_fold(p, held_out_generator=0)

    def test_duplicate_unknown_generator_and_authentic_attribution_rejected(self):
        rows = deepcopy(demo()['folds'][0]['partitions']['validation'])
        for mutate in (lambda r: r.append(deepcopy(r[0])),
                       lambda r: r[2].update(generator_verified=False),
                       lambda r: r[0].update(generator_id=0),
                       lambda r: r[0].update(label=True)):
            changed = deepcopy(rows); mutate(changed)
            with self.assertRaises(ValueError): validate_rows(changed, partition='validation')
        with self.assertRaises(ValueError): validate_rows(rows, partition='test')


class SpatialMetrics(unittest.TestCase):
    def test_oblique_axis_affine_and_mm_radius(self):
        a = [[0, -2, 0, 10], [1, 0, 0, -20], [0, 0, 3, 5], [0, 0, 0, 1]]
        np.testing.assert_allclose(voxel_xyz_to_lps([[1, 2, 3]], a), [[6, -19, 14]])
        r = demo()['point_localization']
        self.assertEqual(r['lesion_sensitivity'], .5)
        self.assertEqual(r['unmatched_prediction_count'], 1)
        self.assertEqual(r['minimum_target_distance_mm'], [1, 6])
        self.assertIsNone(r['dice_or_iou'])

    def test_one_to_one_matching_avoids_greedy_loss_and_duplicate_hits(self):
        r = point_localization([[0, 0, 0], [1.8, 0, 0]], [[.9, 0, 0], [0, 0, 0]],
            verified=True, radius_mm=1., coordinate_system='LPS_mm')
        self.assertEqual(r['matched_points'], 2)
        r = point_localization([[0, 0, 0]] * 2, [[0, 0, 0]], verified=True,
                               radius_mm=0., coordinate_system='LPS_mm')
        self.assertEqual(r['matched_points'], 1)
        self.assertEqual(r['unmatched_prediction_count'], 1)

    def test_empty_points_and_unverified_coordinates(self):
        r = point_localization(np.empty((0, 3)), [[0, 0, 0]], verified=True, radius_mm=5, coordinate_system='LPS_mm')
        self.assertEqual(r['lesion_sensitivity'], 0)
        self.assertEqual(r['minimum_target_distance_mm'], [None])
        for verified, system in ((False, 'LPS_mm'), (True, 'voxel_xyz')):
            with self.assertRaises(ValueError): point_localization([[0, 0, 0]], [[0, 0, 0]],
                verified=verified, radius_mm=5, coordinate_system=system)
        with self.assertRaises(ValueError): voxel_xyz_to_lps([[0, 0, 0]], np.zeros((4, 4)))

    def test_mask_counts_empty_union_and_invalid_support(self):
        r = demo()['mask_localization']; self.assertAlmostEqual(r['dice'], 16 / 17)
        self.assertAlmostEqual(r['iou'], 8 / 9)
        p = np.zeros((2, 2, 2), bool); t = p.copy(); v = np.ones_like(p); p[0, 0, 0] = True; v[0, 0, 0] = False
        r = verified_mask_metrics(p, t, v, verified=True, prediction_frame='g', reference_frame='g')
        self.assertIsNone(r['dice']); self.assertEqual(r['FP'], 0)

    def test_partial_coverage_grid_mismatch_and_unverified_mask_rejected(self):
        p = np.ones((2, 2, 2), bool)
        for kw in ({'verified': False}, {'prediction_frame': 'other'}, {'coverage': np.zeros_like(p)}):
            args = {'verified': True, 'prediction_frame': 'g', 'reference_frame': 'g', **kw}
            with self.assertRaises(ValueError): verified_mask_metrics(p, p, p, **args)


class OfflinePipeline(unittest.TestCase):
    def test_policy_has_no_test_or_medical_override(self):
        with TemporaryDirectory() as d:
            path = Path(d) / 'bad.toml'
            path.write_text(CONFIG.read_text().replace('allow_final_test = false', 'allow_final_test = true'))
            with self.assertRaises(ValueError): load_phase9_config(path)

    def test_sealed_checkpoint_opaque_evaluation_and_repeatable_csv(self):
        with TemporaryDirectory() as d:
            root = Path(d); run = root / 'f08' / 'orchestration'
            run_phase8(small_config(root), run, device='cpu')
            before = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob('*') if p.is_file()}
            with patch('torch.load', side_effect=AssertionError('No checkpoint deserialization for saved prediction metrics')):
                first = run_phase9(run, CONFIG, root / 'eval1')
                second = run_phase9(run, CONFIG, root / 'eval2')
            self.assertTrue(first['ok']); self.assertEqual(first, second)
            self.assertEqual(first['models_evaluated'], 1); self.assertEqual(first['prediction_rows'], 8)
            self.assertFalse(first['final_test_access_allowed'])
            self.assertEqual(before, {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob('*') if p.is_file()})
            with (root / 'eval1/predictions.csv').open(newline='') as stream: rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8); self.assertIn('checkpoint_sha256', rows[0])
            with self.assertRaises(FileExistsError): run_phase9(run, CONFIG, root / 'eval1')
            attempt = run / 'jobs/B1_s13/attempt_0001/predictions.json'
            attempt.write_text('[]')
            with self.assertRaises(ValueError): read_f08_jobs(run)


if __name__ == '__main__': unittest.main()
