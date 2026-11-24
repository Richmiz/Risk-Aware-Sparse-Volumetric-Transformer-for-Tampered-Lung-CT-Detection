"""Optimization regression checks on procedural inputs; no medical-image access."""

from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import importlib.util
import json

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rsvt.training.optimization import build_optimizer, build_scheduler, validate_optimization
from rsvt.training.phase4 import load_phase4_config
from rsvt.training.trainer import fit_baseline
from rsvt.training.checkpoint import load_checkpoint
from rsvt.data.splits import digest_json
from test_f04_training import ToyData, ToyModel


class OptimizationTests(unittest.TestCase):
    def options(self):
        return dict(epochs=8, learning_rate=.01, weight_decay=.1, exclude_norm_bias_decay=True,
                    lr_schedule='warmup_cosine', warmup_epochs=2, warmup_start_ratio=.1, min_lr_ratio=.1)

    def test_grouping_has_no_duplicates_or_omissions_and_protects_norm_bias(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 4))
        model.register_parameter('cls_token', nn.Parameter(torch.zeros(1, 1, 4)))
        model.add_module('position_project', nn.Linear(4, 4))
        model[0].weight.requires_grad_(False)  # must be owned for later RSVT stages
        optimizer = build_optimizer(model, self.options())
        ids = [id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        exempt = {id(p) for g in optimizer.param_groups if g['weight_decay'] == 0 for p in g['params']}
        for p in (model[0].bias, model[1].weight, model[1].bias, model.cls_token):
            self.assertIn(id(p), exempt)
        self.assertNotIn(id(model[0].weight), exempt)
        self.assertNotIn(id(model.position_project.weight), exempt)

    def test_schedule_warms_up_and_reaches_nonzero_floor(self):
        options = self.options()
        optimizer = build_optimizer(nn.Linear(2, 4), options)
        scheduler = build_scheduler(optimizer, options)
        rates = [optimizer.param_groups[0]['lr']]
        for _ in range(options['epochs']):
            optimizer.step(); scheduler.step(); rates.append(optimizer.param_groups[0]['lr'])
        self.assertLess(rates[0], rates[1]); self.assertLess(rates[1], rates[2])
        self.assertAlmostEqual(rates[2], options['learning_rate'])
        self.assertTrue(all(a >= b for a, b in zip(rates[2:], rates[3:])))
        self.assertAlmostEqual(rates[-1], .001)

    def test_invalid_schedule_rejected(self):
        for key, value in [('warmup_epochs', True), ('warmup_epochs', 8), ('warmup_epochs', -1),
                           ('warmup_start_ratio', 0), ('min_lr_ratio', float('nan')),
                           ('exclude_norm_bias_decay', 'true'), ('lr_schedule', 'unknown')]:
            options = self.options(); options[key] = value
            with self.assertRaises(ValueError): validate_optimization(options)

    def test_new_profiles_preserve_partitions_architectures_and_no_annotation_dependency(self):
        for original, optimized in [('baseline_full', 'baseline_optimized'),
                                    ('baseline_full_diagnostic', 'baseline_optimized_control')]:
            before = load_phase4_config(ROOT / f'configs/training/{original}.toml')
            after = load_phase4_config(ROOT / f'configs/training/{optimized}.toml')
            for key in ('selection', 'models', 'representation'): self.assertEqual(before[key], after[key])
            self.assertEqual(after['training']['device'], 'cuda')

    def test_new_schedule_and_grouping_resume_exactly(self):
        config = load_phase4_config(ROOT / 'configs/training/baseline_pilot.toml')['training']
        config.update(self.options(), epochs=5, device='cpu', mixed_precision=False, early_stopping=False,
                      gradient_clip=5., accumulation_steps=3)
        train, val = ToyData('train', [0, 1, 2, 3] * 2), ToyData('validation', [0, 1, 2, 3])
        contract = {'synthetic_optimization': True, 'config': config}
        torch.manual_seed(13); initial = ToyModel(.2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fit_baseline(deepcopy(initial), train, val, config, root/'full', contract)
            fit_baseline(deepcopy(initial), train, val, config, root/'resume', contract, stop_after_epoch=2)
            result = fit_baseline(deepcopy(initial), train, val, config, root/'resume', contract, resume=True)
            binding = digest_json(contract)
            full = load_checkpoint(root/'full/last.pt', binding)
            resumed = load_checkpoint(root/'resume/last.pt', binding)
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
            self.assertEqual([r['learning_rate'] for r in full['history']],
                             [r['learning_rate'] for r in resumed['history']])
            self.assertEqual(result['test_cases_opened'], 0)
            self.assertFalse(result['training_ready'])

    def test_drive_receipt_roundtrip_and_write_failure(self):
        spec = importlib.util.spec_from_file_location('optimization_job', ROOT/'scripts/run_model_optimization.py')
        job = importlib.util.module_from_spec(spec); spec.loader.exec_module(job)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'summary.json').write_text(json.dumps({'ok': True}), encoding='utf-8')
            receipt = job.verify_run(root)
            self.assertIn('summary.json', receipt['files'])
            with patch.object(Path, 'write_text', side_effect=OSError('Drive unavailable')):
                with self.assertRaises(OSError): job.verify_run(root)
            (root/'last.pt').write_bytes(b'corrupt checkpoint')
            (root/'last.pt.json').write_text(json.dumps({'sha256': '0'*64, 'bytes': 18}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Checkpoint readback mismatch'): job.verify_run(root)

    def test_colab_job_requires_gpu_and_never_starts_on_cpu(self):
        spec = importlib.util.spec_from_file_location('optimization_gpu_job', ROOT/'scripts/run_model_optimization.py')
        job = importlib.util.module_from_spec(spec); spec.loader.exec_module(job)
        with TemporaryDirectory() as directory, patch.object(job, 'PROJECT', Path(directory)), \
                patch('torch.cuda.is_available', return_value=False), patch.object(job, 'run_phase4') as trainer:
            with self.assertRaisesRegex(RuntimeError, 'GPU Colab'): job.run()
            trainer.assert_not_called()

    def test_rsvt_optimized_stages_are_finite_and_preserve_frozen_parameters(self):
        from rsvt.experiments.phase7 import run_phase7
        with TemporaryDirectory() as directory:
            result = run_phase7(ROOT/'configs/training/risk_optimized_synthetic.toml', Path(directory)/'run', device='cpu')
            self.assertTrue(result['ok'])
            for path in (Path(directory)/'run').glob('B*_history.json'):
                history = json.loads(path.read_text())
                self.assertTrue(history)
                self.assertTrue(all(row['frozen_parameters_unchanged'] for row in history))
                self.assertTrue(all(row['learning_rate'] > 0 for row in history))


if __name__ == '__main__': unittest.main()
