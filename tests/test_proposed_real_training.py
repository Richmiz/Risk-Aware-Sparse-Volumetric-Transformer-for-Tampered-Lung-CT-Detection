"""Real RSVT trainer on procedural volumes; never open a medical CT locally."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rsvt.data.splits import digest_json
from rsvt.models.risk_rsvt import RSVTRiskFoundation
from rsvt.losses.budget import ProjectedTokenDual
from rsvt.training.optimization import build_optimizer
from rsvt.training.proposed import stratified_batches, replay_step, logical_objective, fit_rsvt, _forward
from rsvt.training.proposed_run import load_proposed_config, run_proposed, implementation_fingerprint
from rsvt.training.risk_trainer import configure_stage
from rsvt.training.checkpoint import load_checkpoint
from rsvt.utils.reproducibility import seed_everything
import test_f04_training as baseline_fixtures


class ProceduralVolumes:
    def __init__(self, split, copies=1):
        self.rows = [{'split': split, 'label_index': c} for c in range(4) for _ in range(copies)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        label = self.rows[index]['label_index']
        generator = torch.Generator().manual_seed(300+index)
        image = torch.rand((1,16,16,16), generator=generator)*.4+label*.1
        return {'image': image, 'valid_mask': torch.ones_like(image, dtype=torch.bool),
                'voxel_to_lps': torch.eye(4, dtype=torch.float64), 'target': label, 'case_key': 'procedural-'+str(index)}


def tiny_config():
    c = load_proposed_config(ROOT/'configs/training/rsvt_real_ct.toml')
    c['training'].update(device='cpu', epochs=3, warmup_epochs=1, logical_batch_size=8)
    c['stages'].update(baseline_epochs=1, routing_warmup_epochs=1)
    c['model'].update(dimension=8, heads=2, layers=1, coarse_width=2, candidate_width=2,
                      router_hidden=4, max_tokens=128, sequence_cap=8, dropout=.2)
    return c


class ProposedTrainerTests(unittest.TestCase):
    def test_stratified_batches_use_all_training_cases_once_and_bound_size(self):
        labels = [0]*9+[1]*9+[2]*27+[3]*27
        batches = stratified_batches(labels,16,13)
        self.assertEqual(sorted(i for b in batches for i in b), list(range(72)))
        self.assertLessEqual(max(map(len,batches)),16)
        self.assertTrue(all(sum(labels[i]==c for i in b)>=5 for b in batches for c in (2,3)))

    def test_replay_matches_direct_logical_gradient_with_dropout(self):
        c = tiny_config(); seed_everything(13)
        model = RSVTRiskFoundation(**c['model']); direct = deepcopy(model)
        data = ProceduralVolumes('train',2); weights = torch.ones(4)
        optimizer = torch.optim.SGD(model.parameters(),lr=.01)
        reference_optimizer = torch.optim.SGD(direct.parameters(),lr=.01)
        dual = ProjectedTokenDual(**c['dual']); reference_dual = ProjectedTokenDual(**c['dual'])
        seed_everything(991)
        row = replay_step(model,optimizer,data,list(range(8)),torch.device('cpu'),weights,c['risk'],dual,
            stage='risk_finetune',gradient_clip=5.)
        seed_everything(991); configure_stage(direct,'B5_full','risk_finetune',localization_active=False)
        losses, expected = [], []
        for i in range(8):
            _, loss, count = _forward(direct,data[i],torch.device('cpu'),'soft')
            losses.append(loss); expected.append(count)
        objective, details = logical_objective(torch.stack(losses),torch.tensor([data[i]['target'] for i in range(8)]),
            torch.stack(expected),weights,c['risk'],reference_dual,8,risk_active=True,budget_active=True)
        objective.backward(); torch.nn.utils.clip_grad_norm_(direct.parameters(),5.,error_if_nonfinite=True)
        reference_optimizer.step(); reference_dual.update(details['budget']['constraint'])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value,direct.state_dict()[key],rtol=1e-5,atol=2e-6)
        self.assertTrue(row['training_is_dense']); self.assertEqual(dual.state_dict(),reference_dual.state_dict())

    def test_router_warmup_preserves_all_frozen_weights(self):
        c=tiny_config(); seed_everything(13); model=RSVTRiskFoundation(**c['model'])
        before=deepcopy(model.state_dict()); optimizer=build_optimizer(model,c['training'])
        replay_step(model,optimizer,ProceduralVolumes('train'),list(range(4)),torch.device('cpu'),torch.ones(4),
            c['risk'],ProjectedTokenDual(**c['dual']),stage='routing_warmup',gradient_clip=5.)
        for name, value in model.state_dict().items():
            if not name.startswith('router.'): torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith('router.')))

    def test_training_resume_is_identical_and_hard_inference_respects_cap(self):
        c=tiny_config(); seed_everything(13); initial=RSVTRiskFoundation(**c['model'])
        train, val = ProceduralVolumes('train'), ProceduralVolumes('validation')
        contract={'procedural':True,'config':c}; binding=digest_json(contract)
        with TemporaryDirectory() as directory:
            root=Path(directory)
            fit_rsvt(deepcopy(initial),train,val,c,root/'full',contract)
            partial=fit_rsvt(deepcopy(initial),train,val,c,root/'resume',contract,stop_after_epoch=1)
            self.assertIsNone(partial['best_epoch']); self.assertEqual(partial['checkpoint_stage'],'provisional_warmup_only')
            result=fit_rsvt(deepcopy(initial),train,val,c,root/'resume',contract,resume=True)
            full=load_checkpoint(root/'full/last.pt',binding,expected_format='rsvt_real_state_v1')
            resumed=load_checkpoint(root/'resume/last.pt',binding,expected_format='rsvt_real_state_v1')
            for key in full['model']: torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
            predictions=json.loads((root/'resume/best_validation_predictions.json').read_text())
            self.assertTrue(all(r['sequence_tokens_including_cls']<=8 for r in predictions))
            self.assertEqual(result['test_cases_opened'],0); self.assertFalse(result['training_ready'])
            self.assertFalse(result['annotation_points_used'])
            self.assertEqual(result['best_epoch'],3); self.assertEqual(result['checkpoint_stage'],'risk_finetune')
            for name in ('history.json','history.csv','epoch_003.json'):
                (root/'resume'/name).unlink()
            fit_rsvt(deepcopy(initial),train,val,c,root/'resume',contract,resume=True)
            self.assertTrue(all((root/'resume'/name).is_file() for name in ('history.json','history.csv','epoch_003.json')))

    def test_write_failure_does_not_report_training_success(self):
        c=tiny_config(); seed_everything(13); model=RSVTRiskFoundation(**c['model'])
        data=ProceduralVolumes('train'); val=ProceduralVolumes('validation')
        with TemporaryDirectory() as directory, patch('rsvt.training.proposed.save_checkpoint',side_effect=OSError('Drive unavailable')):
            with self.assertRaisesRegex(OSError,'Drive unavailable'):
                fit_rsvt(model,data,val,c,directory,{'procedural':True})
            self.assertFalse((Path(directory)/'summary.json').exists())

    def test_replay_failure_cannot_step_optimizer(self):
        c=tiny_config(); seed_everything(13); model=RSVTRiskFoundation(**c['model'])
        class ChangingInput(ProceduralVolumes):
            def __init__(self): super().__init__('train'); self.reads=0
            def __getitem__(self,index):
                sample=super().__getitem__(index); self.reads+=1
                if self.reads>4: sample['image']=1.-sample['image']
                return sample
        optimizer=build_optimizer(model,c['training']); before=deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError,'Replay differs'):
            replay_step(model,optimizer,ChangingInput(),list(range(4)),torch.device('cpu'),torch.ones(4),c['risk'],
                ProjectedTokenDual(**c['dual']),stage='risk_finetune',gradient_clip=5.)
        for key,value in model.state_dict().items(): torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_configuration_pins_existing_split_and_keeps_optional_terms_off(self):
        c=load_proposed_config(ROOT/'configs/training/rsvt_real_ct.toml')
        self.assertEqual(c['data']['shape_zyx'],[96]*3)
        self.assertFalse(c['model']['localization_head']); self.assertEqual(c['data']['train_per_class'],0)
        self.assertEqual(len(implementation_fingerprint()),64)
        for key,value in [('logical_batch_size',True),('epochs',0),('learning_rate',float('nan'))]:
            bad=deepcopy(c); bad['training'][key]=value
            with patch('rsvt.training.proposed_run.load_config',return_value=bad),self.assertRaises(ValueError):
                load_proposed_config('unused')


class ProposedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = baseline_fixtures.Phase4Integration()
        self.fixture.setUp()
        self.root, self.f03 = self.fixture.root, self.fixture.f03
    def tearDown(self): self.fixture.tearDown()
    def test_proposed_pipeline_reads_geometry_and_never_test_images(self):
        from rsvt.data.dataset import CTVolumeDataset
        c=tiny_config(); document=json.loads((self.f03/'manifest.draft.json').read_text())
        c['data'].update(expected_assignment_sha256=document['assignments_sha256'],shape_zyx=[32]*3,
                         train_per_class=1,validation_per_class=1)
        c['model'].update(sequence_cap=24,max_tokens=128)
        observed=[]; original=CTVolumeDataset.__getitem__
        def track(dataset,index):
            observed.append(dataset.rows[index]['split']); return original(dataset,index)
        manifest=self.f03/'manifest.draft.json'; before=manifest.read_bytes()
        with patch('rsvt.training.proposed_run.load_config',return_value=c), \
                patch('rsvt.data.baseline_views.CTVolumeDataset.__getitem__',track):
            result=run_proposed(self.f03,'unused',self.root/'proposed')
        self.assertTrue(result['ok']); self.assertEqual(manifest.read_bytes(),before)
        self.assertEqual(set(observed),{'train','validation'})
        self.assertTrue(result['source_manifest_unchanged'])
        self.assertFalse(result['localization_active']); self.assertFalse(result['generator_variance_active'])


if __name__=='__main__': unittest.main()
