"""F06 behavioral checks: differentiability, real packing, coarse floor and costs."""

import io
import inspect
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rsvt.evaluation.efficiency import percentile,process_memory
from rsvt.experiments.phase5 import coarse_coverage,synthetic_batch
from rsvt.experiments.phase6 import calibrate_latency,load_phase6_config,profile_worker,selection_audit
from rsvt.losses.budget import expected_token_budget_loss
from rsvt.models.routing import ContextTokenRouter,hard_selection,pack_tokens
from rsvt.models.routed_rsvt import RSVTRoutingFoundation
from rsvt.models.rsvt import RSVTDenseFoundation

torch.set_num_threads(2)


def small(**changes):
    args=dict(dimension=16,heads=2,layers=2,coarse_width=4,candidate_width=4,
        coarse_patch_size=16,patch_sizes=(4,8,16),sequence_cap=16)
    return RSVTRoutingFoundation(**(args|changes))


def sample(shape=(17,19,21)):
    image=torch.rand(1,1,*shape);valid=torch.ones_like(image,dtype=torch.bool);valid[:,:,:2]=False
    return image,valid,torch.eye(4,dtype=torch.float64)[None]


class Router(unittest.TestCase):
    def test_stable_ranking_mandatory_tokens_and_variable_packing(self):
        scores=torch.zeros(2,5);valid=torch.tensor([[1,1,1,1,1],[1,0,1,0,1]],dtype=torch.bool)
        scales=torch.tensor([[0,1,0,2,3],[0,1,1,2,3]])
        indices=hard_selection(scores,valid,scales,4)
        self.assertEqual([row.tolist() for row in indices],[[0,1,2],[0,2,4]])
        indices=hard_selection(scores,valid,scales,5)
        self.assertEqual([row.tolist() for row in indices],[[0,1,2,3],[0,2,4]])
        tokens={'embeddings':torch.arange(20.).reshape(2,5,2),'valid':valid,'scale_ids':scales}
        packed,presence,source=pack_tokens(tokens,indices)
        self.assertEqual(tuple(packed['embeddings'].shape),(2,4,2))
        self.assertEqual(source.tolist(),[[0,1,2,3],[0,2,4,-1]])
        self.assertEqual(packed['scale_ids'][1,-1],-1);self.assertFalse(presence[1,-1])

    def test_floor_zero_candidates_and_bad_budgets(self):
        scores=torch.zeros(1,3);valid=torch.ones_like(scores,dtype=torch.bool);scales=torch.tensor([[0,1,2]])
        self.assertEqual(hard_selection(scores,valid,scales,2)[0].tolist(),[0])
        for cap in (0,1,False,2.5):
            with self.assertRaises(ValueError):hard_selection(scores,valid,scales,cap)
        with self.assertRaises(ValueError):hard_selection(scores+float('nan'),valid,scales,3)

    def test_router_uses_context_cues_and_valid_fraction(self):
        router=ContextTokenRouter(4,4)
        for parameter in router.parameters():parameter.data.fill_(.1)
        emb=torch.ones(1,3,4);cues=torch.zeros(1,3,4);fraction=torch.ones(1,3)
        valid=torch.ones(1,3,dtype=torch.bool);scales=torch.tensor([[0,1,2]])
        original=router(emb,cues,fraction,valid,scales)['logits']
        emb2=emb.clone();emb2[:,0]*=2
        self.assertTrue((router(emb2,cues,fraction,valid,scales)['logits'][:,1:]>original[:,1:]).all())
        self.assertTrue((router(emb,cues+1,fraction,valid,scales)['logits']>original).all())
        self.assertTrue((router(emb,cues,fraction*.5,valid,scales)['logits']<original).all())
        out=router(emb,cues,fraction,valid,torch.zeros_like(scales))
        self.assertEqual(float(out['probabilities'].detach().sum()),0.)
        self.assertEqual(float(out['expected_sequence_counts'][0].detach()),4.)

    def test_expected_budget_gradient_and_invalid_inputs(self):
        expected=torch.tensor([5.,15.],requires_grad=True)
        loss=expected_token_budget_loss(expected,10);loss.backward()
        torch.testing.assert_close(expected.grad,torch.tensor([0.,.05]))
        for value in (torch.tensor([]),torch.tensor([float('nan')]),torch.tensor([0.]),torch.ones(1,1)):
            with self.assertRaises(ValueError):expected_token_budget_loss(value,10)


class RoutedModel(unittest.TestCase):
    def setUp(self):torch.manual_seed(13)

    def test_classification_gradient_reaches_router_without_budget_loss(self):
        model=small();image,valid,affine=sample();image.requires_grad_(True)
        out=model(image,valid,affine)
        F.cross_entropy(out['logits'],torch.tensor([2])).backward()
        self.assertTrue(torch.isfinite(model.router.score[0].weight.grad).all())
        self.assertGreater(float(model.router.score[0].weight.grad.abs().sum()),0.)
        self.assertTrue((image.grad[~valid]==0).all())
        self.assertEqual(out['routing']['mode'],'soft')
        self.assertEqual(out['compute']['saved_attention_slots'],0)
        for module in (model.coarse_encoder,model.tokenizer,model.position,model.transformer,model.classifier):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_hard_pack_reduces_actual_transformer_input_and_preserves_coarse(self):
        model=small().eval();image,valid,affine=sample();seen=[]
        hook=model.transformer.register_forward_pre_hook(lambda module,args:seen.append(args[0].shape[1]))
        with torch.inference_mode():
            dense=model(image,valid,affine,routing_mode='dense')
            out=model(image,valid,affine)
        hook.remove();self.assertLess(seen[1],seen[0]);self.assertEqual(seen[1],15)
        self.assertEqual(out['compute']['actual_sequence_length_including_cls'],16)
        self.assertEqual(coarse_coverage(out,valid),[True])
        coarse=out['routing']['source_valid_mask']&(out['routing']['source_geometry']['scale_ids']==0)
        self.assertTrue(out['routing']['selected_source_mask'][coarse].all())
        self.assertGreater(out['compute']['saved_attention_slots'],0)
        self.assertEqual(out['compute']['coarse_tokens_pruned'],0)
        self.assertIsNone(out['spatial_logits'])

    def test_dense_and_all_kept_predictions_match_f05_shared_weights(self):
        model=small(sequence_cap=2048).eval()
        f05=RSVTDenseFoundation(dimension=16,heads=2,layers=2,coarse_width=4,candidate_width=4,
            coarse_patch_size=16,patch_sizes=(4,8,16),localization_head=False).eval()
        f05.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('router.')})
        image,valid,affine=sample()
        with torch.inference_mode():
            reference=f05(image,valid,affine)['logits']
            for mode in ('dense','all_valid','hard'):
                torch.testing.assert_close(reference,model(image,valid,affine,routing_mode=mode)['logits'],atol=1e-6,rtol=1e-5)

    def test_invalid_nan_and_batch_padding_invariance(self):
        model=small().eval();image,valid,affine=sample((7,9,11))
        noisy=image.clone();noisy[~valid]=float('nan')
        batch=torch.rand(2,1,17,19,21);mask=torch.ones_like(batch,dtype=torch.bool);padding=mask.clone()
        batch[0,:,:7,:9,:11]=noisy[0];mask[0]=False;mask[0,:,:7,:9,:11]=valid[0]
        padding[0]=False;padding[0,:,:7,:9,:11]=True
        with torch.inference_mode():
            solo=model(image,valid,affine);nan=model(noisy,valid,affine)
            paired=model(batch,mask,affine.repeat(2,1,1),padding_mask=padding)
        torch.testing.assert_close(solo['logits'],nan['logits'],atol=0,rtol=0)
        torch.testing.assert_close(solo['logits'][0],paired['logits'][0],atol=1e-6,rtol=1e-5)
        n=int(solo['token_presence_mask'].sum())
        self.assertEqual(solo['source_token_indices'][0].tolist(),paired['source_token_indices'][0,:n].tolist())

    def test_floor_and_training_mode_guard(self):
        model=small();image,valid,affine=sample()
        with self.assertRaisesRegex(ValueError,'inference-only'):model(image,valid,affine,routing_mode='hard')
        model.eval()
        with self.assertRaisesRegex(ValueError,'mandatory'):model(image,valid,affine,sequence_cap=1)
        with torch.inference_mode():out=model(image,valid,affine,sequence_cap=9)
        self.assertEqual(int(out['compute']['selected_candidate_tokens'][0]),0)
        self.assertEqual(coarse_coverage(out,valid),[True])

    def test_no_annotation_inputs_and_parameter_validation(self):
        signature=set(inspect.signature(small().forward).parameters)
        self.assertFalse(signature.intersection({'labels','targets','annotations','generator_id'}))
        for args in ({'temperature':0},{'router_hidden':False},{'sequence_cap':0},{'localization_head':True}):
            with self.assertRaises(ValueError):small(**args)

    def test_stage_measurements_and_scoped_flops(self):
        model=small().eval();image,valid,affine=sample()
        with torch.inference_mode():
            dense=model(image,valid,affine,routing_mode='dense',profile_stages=True)
            hard=model(image,valid,affine,profile_stages=True)
        for key in ('coarse_encoder','candidate_encoder_and_cues','controller_and_packing','transformer'):
            self.assertGreater(hard['stage_ms'][key],0.)
        a=dense['compute']['operator_flops_estimate'];b=hard['compute']['operator_flops_estimate']
        self.assertEqual(b['total'],sum(b['by_component'].values()))
        self.assertEqual(a['by_component']['candidate_encoder_and_projections'],b['by_component']['candidate_encoder_and_projections'])
        self.assertEqual(a['by_component']['coarse_encoder'],b['by_component']['coarse_encoder'])
        self.assertLess(b['by_component']['transformer'],a['by_component']['transformer'])
        self.assertGreater(b['by_component']['controller'],0)
        self.assertTrue(b['not_measured_hardware_FLOPs'])

    def test_router_state_roundtrip(self):
        model=small().eval();restored=small().eval();stream=io.BytesIO()
        torch.save(model.state_dict(),stream);stream.seek(0);restored.load_state_dict(torch.load(stream,weights_only=True))
        image,valid,affine=sample()
        with torch.inference_mode():
            first=model(image,valid,affine);second=restored(image,valid,affine)
        torch.testing.assert_close(first['logits'],second['logits'],atol=0,rtol=0)
        self.assertTrue(torch.equal(first['source_token_indices'],second['source_token_indices']))

    def test_default_96_soft_backward_and_hard_64(self):
        model=RSVTRoutingFoundation(sequence_cap=64);image,valid,affine=sample((96,96,96))
        out=model(image,valid,affine)
        self.assertEqual(out['compute']['actual_sequence_length_including_cls'],1999)
        F.cross_entropy(out['logits'],torch.tensor([3])).backward()
        self.assertGreater(float(model.router.score[0].weight.grad.abs().sum()),0.)
        del out
        model.eval()
        with torch.inference_mode():out=model(image,valid,affine)
        self.assertEqual(out['compute']['actual_sequence_length_including_cls'],64)
        self.assertEqual(coarse_coverage(out,valid),[True])


class ProfilingAndAudit(unittest.TestCase):
    def test_latency_calibration_marks_infeasible_targets(self):
        rows=[{'profile':'dense','routing_mode':'dense','p95_ms':100},
            *[dict(profile=name,routing_mode='hard',p95_ms=latency,sequence_cap=cap)
              for name,latency,cap in [('minimum',80,2),('low',70,64),('medium',90,128),('high',110,256)]]]
        result=calibrate_latency(rows,[.5,.75,1.])['proposals']
        self.assertTrue(result['low']['infeasible_among_measured_profiles'])
        self.assertEqual(result['medium']['sequence_cap'],64)
        self.assertEqual(result['high']['sequence_cap'],128)
        self.assertAlmostEqual(percentile(list(range(1,11)),.95),9.55)

    def test_selection_audit_records_unselected_candidates_and_coverage(self):
        model=small().eval();batch=synthetic_batch([[8,9,10]]*4)
        with TemporaryDirectory() as temporary:
            audit=selection_audit(model,batch,2,.01,8,temporary)
            self.assertGreater(audit['candidate_rows'],audit['candidates_never_selected'])
            self.assertGreater(audit['candidates_never_selected'],0)
            self.assertEqual(len(audit['fine_voxel_coverage_per_trial_case']),2)
            self.assertTrue((Path(temporary)/'selection_frequency.csv').is_file())

    def test_profile_worker_measures_real_forward_with_same_seed_weights(self):
        with TemporaryDirectory() as temporary:
            config=Path(temporary)/'tiny.toml'
            config.write_text('extends = ['+json.dumps((ROOT/'configs/models/rsvt_f06.toml').as_posix())+']\n'
                '[model]\ndimension=16\nheads=2\ncoarse_width=4\ncandidate_width=4\ncoarse_patch_size=16\npatch_sizes=[4,8,16]\nsequence_cap=8\n'
                '[profiling]\nshape_zyx=[8,8,8]\nwarmup=1\niterations=5\nstage_iterations=1\nlow=4\nmedium=8\nhigh=16\n',encoding='utf-8')
            dense=profile_worker(config,'cpu','dense',Path(temporary)/'dense.json')
            hard=profile_worker(config,'cpu','low',Path(temporary)/'low.json')
            self.assertEqual(dense['model_initial_weights_sha256'],hard['model_initial_weights_sha256'])
            self.assertEqual(len(hard['latency_samples_ms']),5)
            self.assertGreater(hard['p50_ms'],0.)
            self.assertLess(hard['compute']['actual_sequence_length_including_cls'],dense['compute']['actual_sequence_length_including_cls'])
            self.assertIsNone(hard['cuda_memory'])
            self.assertFalse(hard['medical_data_read'])

    def test_config_rejects_invalid_protocol_and_profiling(self):
        config=load_phase6_config(ROOT/'configs/models/rsvt_f06.toml')
        for change in ({'protocol':config['protocol']|{'medical_training_ready':0}},
            {'profiling':config['profiling']|{'iterations':0}},
            {'profiling':config['profiling']|{'medium':64}},
            {'model':config['model']|{'localization_head':True}}):
            with patch('rsvt.experiments.phase6.load_config',return_value=config|change):
                with self.assertRaises(ValueError):load_phase6_config('unused')


if __name__=='__main__':unittest.main()
