"""F07 loss mathematics, provenance gates, staged gradients and synthetic pipeline."""

import inspect
import io
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rsvt.cli import main
from rsvt.experiments.phase7 import load_phase7_config,risk_synthetic_batch,run_phase7
from rsvt.losses.budget import ProjectedTokenDual
from rsvt.losses.combined import risk_objective
from rsvt.losses.cvar import empirical_cvar,class_conditional_cvar
from rsvt.losses.generator import generator_risk_variance
from rsvt.losses.localization import verified_voxel_mask_loss
from rsvt.models.risk_rsvt import RSVTRiskFoundation
from rsvt.training.trainer import risk_training_step

torch.set_num_threads(2)


def small(**changes):
    return RSVTRiskFoundation(**(dict(dimension=16,heads=2,layers=2,coarse_width=4,candidate_width=4,
        coarse_patch_size=16,patch_sizes=(4,8,16),sequence_cap=8)|changes))


def loss_config():return load_phase7_config(ROOT/'configs/training/risk_synthetic.toml')['loss']


def generator(losses,labels,ids,verified=None,**changes):
    return generator_risk_variance(losses,torch.tensor(labels),torch.tensor(ids),
        torch.ones(len(labels),dtype=torch.bool) if verified is None else torch.tensor(verified),
        **(dict(training_generator_ids=(0,1),partition='train')|changes))


class CVaRMathematics(unittest.TestCase):
    def test_fractional_tail_value_and_subgradient(self):
        losses=torch.tensor([1.,2.,3.,4.],requires_grad=True)
        result=empirical_cvar(losses,.625)
        self.assertAlmostEqual(float(result.detach()),11/3,places=6)
        result.backward();torch.testing.assert_close(losses.grad,torch.tensor([0.,0.,1/3,2/3]))

    def test_alpha_zero_mean_singleton_max_and_empty(self):
        self.assertEqual(float(empirical_cvar(torch.tensor([1.,2.,3.,4.]),0)),2.5)
        self.assertEqual(float(empirical_cvar(torch.tensor([1.,2.]),.9)),2.)
        self.assertEqual(float(empirical_cvar(torch.tensor([7.]),.9)),7.)
        empty=torch.tensor([],requires_grad=True);zero=empirical_cvar(empty,.9);zero.backward()
        self.assertEqual(float(zero.detach()),0.);self.assertEqual(len(empty.grad),0)

    def test_class_conditioning_fixed_denominator_and_missing_class(self):
        losses=torch.tensor([10.,1.,4.],requires_grad=True)
        result=class_conditional_cvar(losses,torch.tensor([0,2,2]),alpha=.5)
        self.assertAlmostEqual(float(result['loss'].detach()),8/3,places=6)
        self.assertTrue(result['classes']['FM']['missing']);self.assertTrue(result['classes']['FB']['low_support'])
        result['loss'].backward();torch.testing.assert_close(losses.grad,torch.tensor([0.,0.,2/3]))

    def test_no_risk_classes_is_differentiable_zero(self):
        losses=torch.tensor([1.,2.],requires_grad=True)
        result=class_conditional_cvar(losses,torch.tensor([0,1]))
        result['loss'].backward();self.assertEqual(float(result['loss'].detach()),0.)
        torch.testing.assert_close(losses.grad,torch.zeros(2))

    def test_invalid_confidence_nan_shape_and_targets_rejected(self):
        for alpha in (-.1,1.,True,float('nan')):
            with self.assertRaises(ValueError):empirical_cvar(torch.ones(2),alpha)
        for losses in (torch.ones(1,2),torch.tensor([float('nan')]),torch.tensor([1,2])):
            with self.assertRaises(ValueError):empirical_cvar(losses)
        with self.assertRaises(ValueError):class_conditional_cvar(torch.ones(2),torch.tensor([2,4]))


class GeneratorRisk(unittest.TestCase):
    def test_equal_group_population_variance_and_gradient(self):
        losses=torch.tensor([0.,2.,4.,6.],requires_grad=True)
        result=generator(losses,[2,2,2,2],[0,0,1,1])
        self.assertEqual(float(result['loss'].detach()),2.)
        self.assertEqual(result['enabled_classes'],1)
        result['loss'].backward();torch.testing.assert_close(losses.grad,torch.tensor([-.5,-.5,.5,.5]))

    def test_conditioning_removes_label_mean_difference(self):
        result=generator(torch.tensor([1.,1.,10.,10.]),[2,2,3,3],[0,1,0,1])
        self.assertEqual(float(result['loss']),0.)
        self.assertEqual(result['enabled_classes'],2)

    def test_single_group_disables_and_keeps_zero_gradient(self):
        losses=torch.ones(4,requires_grad=True)
        result=generator(losses,[2,2,3,3],[0,0,0,0],training_generator_ids=(0,))
        self.assertEqual(result['enabled_classes'],0);result['loss'].backward()
        torch.testing.assert_close(losses.grad,torch.zeros(4))

    def test_unverified_unknown_ignored_verified_heldout_rejected(self):
        result=generator(torch.ones(3),[2,2,2],[0,1,99],[True,True,False])
        self.assertEqual(result['enabled_classes'],1)
        with self.assertRaises(ValueError):generator(torch.ones(3),[2,2,2],[0,1,99])

    def test_nontraining_partition_and_insufficient_support(self):
        for partition in ('validation','test'):
            with self.assertRaises(ValueError):generator(torch.ones(2),[2,2],[0,1],partition=partition)
        result=generator(torch.ones(2),[2,2],[0,1],min_group_samples=2)
        self.assertEqual(result['enabled_classes'],0)
        with self.assertRaises(ValueError):generator(torch.ones(2),[2,2],[0,1],training_generator_ids=(0,0))


class DualConstraint(unittest.TestCase):
    def test_signed_max_constraint_gradient_and_projection(self):
        dual=ProjectedTokenDual();expected=torch.tensor([4.,20.],requires_grad=True)
        result=dual.terms(expected,10);result['loss'].backward()
        torch.testing.assert_close(expected.grad,torch.tensor([0.,.01]))
        self.assertAlmostEqual(dual.update(result['constraint']),.15)
        self.assertAlmostEqual(dual.update(-.5),.125)
        self.assertEqual(dual.update(-1e5),0.);self.assertEqual(dual.update(1e5),10.)

    def test_dual_state_roundtrip_and_invalid_values(self):
        dual=ProjectedTokenDual();dual.update(1.)
        restored=ProjectedTokenDual();restored.load_state_dict(dual.state_dict())
        self.assertEqual(restored.state_dict(),dual.state_dict())
        with self.assertRaises(ValueError):dual.update(float('nan'))
        with self.assertRaises(ValueError):ProjectedTokenDual(initial=-1)
        with self.assertRaises(ValueError):dual.terms(torch.ones(2),False)
        with self.assertRaises(ValueError):restored.load_state_dict(dual.state_dict()|{'updates':True})


class RiskModelAndObjective(unittest.TestCase):
    def setUp(self):torch.manual_seed(13)

    def test_soft_localization_has_full_valid_support_and_backward(self):
        model=small();batch=risk_synthetic_batch([[8,9,10]]*4)
        out=model(batch['image'],batch['valid_mask'],batch['voxel_to_lps'],padding_mask=batch['padding_mask'])
        self.assertTrue(torch.equal(out['spatial_valid_mask'],batch['valid_mask']))
        loc=verified_voxel_mask_loss(out['spatial_logits'],batch['localization']['targets'],out['spatial_valid_mask'],
            batch['localization']['annotation_verified'],annotation_kind='voxel_mask')
        self.assertEqual(loc['eligible_cases'],6);loc['loss'].backward()
        self.assertGreater(float(model.localizer.linear.weight.grad.norm()),0.)
        self.assertGreater(float(model.transformer.cls_token.grad.norm()),0.)

    def test_hard_prediction_mask_exact_selected_fine_support(self):
        model=small(sequence_cap=4).eval()
        for parameter in model.router.parameters():parameter.data.zero_()
        batch=risk_synthetic_batch([[8,9,10]]*4)
        with torch.inference_mode():out=model(batch['image'],batch['valid_mask'],batch['voxel_to_lps'],padding_mask=batch['padding_mask'])
        expected=torch.zeros_like(batch['valid_mask'])
        route=out['routing'];g=route['source_geometry']
        for i in range(8):
            selected=route['selected_source_mask'][i]&(g['scale_ids'][i]==1)
            for start,end in zip(g['starts_xyz'][i,selected],g['ends_xyz'][i,selected]):
                x,y,z=start.tolist();ex,ey,ez=end.tolist();expected[i,0,z:ez,y:ey,x:ex]=True
        expected&=batch['valid_mask']
        self.assertTrue(torch.equal(expected,out['spatial_valid_mask']))
        self.assertGreater(int(expected.sum()),0);self.assertLess(int(expected.sum()),int(batch['valid_mask'].sum()))
        self.assertTrue((out['spatial_logits'][~expected]==0).all())

    def test_coarse_only_has_no_fine_prediction_and_optional_head(self):
        batch=risk_synthetic_batch([[8,9,10]]*4)
        for enabled in (True,False):
            model=small(localization_head=enabled,sequence_cap=2).eval()
            with torch.inference_mode():out=model(batch['image'],batch['valid_mask'],batch['voxel_to_lps'])
            if enabled:self.assertEqual(int(out['spatial_valid_mask'].sum()),0)
            else:self.assertIsNone(out['spatial_logits'])

    def test_labels_and_generator_ids_are_not_forward_inputs(self):
        self.assertFalse(set(inspect.signature(small().forward).parameters)&{'labels','generator_ids','targets','annotations'})
        with self.assertRaises(ValueError):small(localization_head=1)

    def test_weighted_ce_and_cvar_use_raw_losses_once(self):
        logits=torch.zeros(1,4,requires_grad=True);labels=torch.tensor([2])
        result=risk_objective({'logits':logits},labels,torch.tensor([1.,1.,3.,1.]),loss_config=loss_config(),
            generator_ids=torch.tensor([0]),generator_verified=torch.tensor([True]),training_generator_ids=(0,),risk_enabled=True)
        self.assertAlmostEqual(float(result['classification'].detach()),3*math.log(4),places=6)
        self.assertAlmostEqual(float(result['cvar']['loss'].detach()),2/3*math.log(4),places=6)
        self.assertEqual(float(result['generator']['loss'].detach()),0.)
        result['loss'].backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_requested_mask_supervision_requires_head(self):
        with self.assertRaises(ValueError):risk_objective({'logits':torch.zeros(1,4)},torch.tensor([0]),torch.ones(4),
            loss_config=loss_config(),generator_ids=torch.tensor([-1]),generator_verified=torch.tensor([False]),
            training_generator_ids=(0,),risk_enabled=False,localization={})


class StageAndIntegration(unittest.TestCase):
    def setUp(self):torch.manual_seed(13)

    def test_freezing_warmup_and_risk_gradients_to_router(self):
        model=small();batch=risk_synthetic_batch([[8,9,10]]*4)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);dual=ProjectedTokenDual()
        baseline=risk_training_step(model,optimizer,batch,torch.ones(4),variant='B5_full',stage='baseline',loss_config=loss_config(),dual=dual)
        self.assertFalse(baseline['router_weight_changed'])
        before={k:v.detach().clone() for k,v in model.named_parameters()}
        warmup=risk_training_step(model,optimizer,batch,torch.ones(4),variant='B5_full',stage='routing_warmup',loss_config=loss_config(),dual=dual)
        self.assertTrue(warmup['router_weight_changed']);self.assertTrue(warmup['frozen_parameters_unchanged'])
        for name,parameter in model.named_parameters():
            if not name.startswith('router.'):self.assertTrue(torch.equal(before[name],parameter.detach()),name)
        risk=risk_training_step(model,optimizer,batch,torch.ones(4),variant='B5_full',stage='risk_finetune',loss_config=loss_config(),dual=dual)
        for value in risk['risk_gradients_to_router'].values():
            self.assertTrue(value['active']);self.assertTrue(value['present_finite']);self.assertGreater(value['l2_norm'],0.)
        self.assertEqual(dual.updates,2)

    def test_missing_annotations_and_generator_groups_disable_terms(self):
        model=small();batch=risk_synthetic_batch([[8,9,10]]*4);batch.pop('localization')
        batch['generator_verified'].fill_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
        row=risk_training_step(model,optimizer,batch,torch.ones(4),variant='B5_full',stage='risk_finetune',loss_config=loss_config(),dual=ProjectedTokenDual())
        self.assertFalse(row['gradient_groups']['localization']['active'])
        self.assertFalse(row['risk_gradients_to_router']['generator']['active'])
        self.assertEqual(row['generator']['enabled_classes'],0)

    def test_tiny_complete_B4_B5_pipeline_and_new_run_guard(self):
        with TemporaryDirectory() as temporary:
            config=Path(temporary)/'tiny.toml'
            config.write_text('extends = ['+json.dumps((ROOT/'configs/training/risk_synthetic.toml').as_posix())+']\n'
                '[model]\ndimension=16\nheads=2\ncoarse_width=4\ncandidate_width=4\ncoarse_patch_size=16\npatch_sizes=[4,8,16]\nsequence_cap=8\n'
                '[engineering]\nsynthetic_shapes_zyx=[[8,9,10],[9,10,11],[10,11,12],[12,12,12]]\n',encoding='utf-8')
            summary=run_phase7(config,Path(temporary)/'run',device='cpu')
            self.assertTrue(summary['ok'],summary['checks']);self.assertEqual(summary['logical_batch_size'],8)
            self.assertEqual(summary['training_class_counts'],[2,2,2,2])
            self.assertEqual(summary['variants']['B4_risk_only']['optimizer_steps'],3)
            self.assertEqual(summary['variants']['B5_full']['optimizer_steps'],4)
            self.assertEqual(summary['variants']['B5_full']['inference']['actual_sequence'],8)
            self.assertFalse(summary['medical_training_ready'])
            saved=json.loads((Path(temporary)/'run/summary.json').read_text())
            self.assertEqual(saved['config_sha256'],summary['config_sha256'])
            with self.assertRaises(ValueError):run_phase7(config,Path(temporary)/'run',device='cpu')

    def test_config_rejects_research_flags_and_invalid_risk_parameters(self):
        config=load_phase7_config(ROOT/'configs/training/risk_synthetic.toml')
        for change in ({'protocol':config['protocol']|{'alpha_source_validation_selected':True}},
            {'protocol':config['protocol']|{'medical_training_ready':0}}, {'loss':config['loss']|{'alpha':1}},
            {'dual':config['dual']|{'initial':-1}}, {'stages':config['stages']|{'risk_finetune_steps':0}}):
            with patch('rsvt.experiments.phase7.load_config',return_value=config|change):
                with self.assertRaises(ValueError):load_phase7_config('unused')

    def test_cli_phase7_dispatch(self):
        with patch('rsvt.experiments.phase7.run_phase7',return_value={'ok':True,'checks':{}}) as runner:
            with patch('sys.stdout',new_callable=io.StringIO):self.assertEqual(main(['phase7-risk','--output-dir','unused','--device','cpu']),0)
            runner.assert_called_once()


if __name__=='__main__':unittest.main()
