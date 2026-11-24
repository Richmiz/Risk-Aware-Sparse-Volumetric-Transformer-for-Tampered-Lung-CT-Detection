"""Behavioral checks for the dense F05 foundation, using procedural tensors only."""

import inspect
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rsvt.cli import main
from rsvt.experiments.phase5 import coarse_coverage,load_phase5_config,run_phase5
from rsvt.losses.localization import verified_voxel_mask_loss
from rsvt.models.anomaly_features import ForensicCues
from rsvt.models.geometry import patch_layout
from rsvt.models.rsvt import RSVTDenseFoundation

torch.set_num_threads(2)


def small(**changes):
    args=dict(dimension=16,heads=2,layers=2,coarse_width=4,candidate_width=4,
              coarse_patch_size=16,patch_sizes=(4,8,16))
    return RSVTDenseFoundation(**(args|changes))


def sample(shape=(9,11,13)):
    image=torch.rand(1,1,*shape)
    return image,torch.ones_like(image,dtype=torch.bool),torch.eye(4,dtype=torch.float64)[None]


class GeometryAndCues(unittest.TestCase):
    def test_clipped_patch_centers_and_lps_transform(self):
        affine=torch.tensor([[0.,-2.,0.,100.],[.5,0.,0.,-30.],[0.,0.,-3.,20.],[0.,0.,0.,1.]],dtype=torch.float64)
        layout=patch_layout((5,7,9),4,affine)
        self.assertEqual(layout["grid_shape_zyx"],(2,2,3))
        # X changes fastest in flattened Z/Y/X order; final support is 1x3x1 XYZ.
        torch.testing.assert_close(layout["starts_xyz"][1],torch.tensor([4,0,0]))
        torch.testing.assert_close(layout["centers_voxel_xyz"][-1],torch.tensor([8.,5.,4.],dtype=torch.float64))
        torch.testing.assert_close(layout["centers_lps_mm"][-1],torch.tensor([90.,-26.,8.],dtype=torch.float64))
        torch.testing.assert_close(layout["size_mm_xyz"][-1],torch.tensor([.5,6.,3.],dtype=torch.float64))
        shifted=affine.clone();shifted[:3,3]+=torch.tensor([200.,-13.,8.])
        other=patch_layout((5,7,9),4,shifted)
        torch.testing.assert_close(layout["position_features"],other["position_features"])
        self.assertFalse(torch.equal(layout["centers_lps_mm"],other["centers_lps_mm"]))

    def test_cues_constant_and_single_slice(self):
        image=torch.full((1,1,1,2,2),.5);valid=torch.ones_like(image,dtype=torch.bool)
        cues=ForensicCues()(image,valid)
        self.assertEqual(tuple(cues.shape),(1,4,1,2,2))
        torch.testing.assert_close(cues[:,:3],torch.zeros_like(cues[:,:3]),atol=1e-6,rtol=0)
        torch.testing.assert_close(cues[:,3],torch.full_like(cues[:,3],4/27))

    def test_adjacent_z_cue_has_no_wrap_or_invalid_pair(self):
        image=torch.tensor([0.,.2,.8]).reshape(1,1,3,1,1);valid=torch.ones_like(image,dtype=torch.bool)
        cue=ForensicCues()(image,valid)[0,1,:,0,0]
        torch.testing.assert_close(cue,torch.tensor([.2,.4,.6]))
        valid[:,:,1]=False
        torch.testing.assert_close(ForensicCues()(image,valid)[0,1],torch.zeros((3,1,1)))

    def test_coarse_coverage_and_multiscale_grid_geometry(self):
        image,valid,affine=sample((17,19,21));valid[:,:,:4,:5,:7]=False
        output=small()(image,valid,affine)
        self.assertEqual(output["grid_shapes_zyx"].tolist(),[[[2,2,2],[5,5,6],[3,3,3],[2,2,2]]])
        self.assertEqual(output["compute"]["constructed_candidate_tokens"].tolist(),[185])
        self.assertEqual(coarse_coverage(output,valid),[True])
        geometry=output["token_geometry"];present=output["token_presence_mask"]
        self.assertTrue(((geometry["valid_fraction"][present]>=0)&(geometry["valid_fraction"][present]<=1)).all())
        for scale in range(4):self.assertTrue((geometry["scale_ids"]==scale).any())
        self.assertFalse(output["compute"]["routing_enabled"])
        self.assertEqual(output["compute"]["saved_attention_slots"],0)
        self.assertEqual(output["compute"]["coarse_tokens_pruned"],0)


class DenseModelContract(unittest.TestCase):
    def setUp(self):torch.manual_seed(13)

    def test_forward_backward_all_components_and_invalid_gradients(self):
        model=small();image,valid,affine=sample();valid[:,:,:2]=False
        image.requires_grad_(True);output=model(image,valid,affine)
        loss=F.cross_entropy(output["logits"],torch.tensor([2]))+output["spatial_logits"][valid].square().mean()
        loss.backward()
        for module in (model.coarse_encoder,model.tokenizer,model.position,model.transformer,model.classifier,model.localizer,*model.tokenizer.projections):
            gradients=[p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0.)
        self.assertTrue((image.grad[~valid]==0).all())
        self.assertEqual(tuple(output["logits"].shape),(1,4))
        self.assertEqual(output["class_index"],{"TB":0,"TM":1,"FB":2,"FM":3})
        self.assertFalse(output["training_ready"])

    def test_invalid_noise_and_nan_do_not_change_predictions(self):
        model=small().eval();image,valid,affine=sample();valid[:,:,:3]=False
        changed=image.clone();changed[~valid]=float("nan")
        with torch.no_grad():
            first=model(image,valid,affine);second=model(changed,valid,affine)
        for key in ("logits","token_features","candidate_cues","spatial_logits"):
            torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)

    def test_right_padded_batch_matches_unpadded_case(self):
        model=small().eval();image,valid,affine=sample((7,9,11));valid[:,:,:2]=False
        batch=torch.rand(2,1,13,17,19);mask=torch.ones_like(batch,dtype=torch.bool);padding=mask.clone()
        batch[0,:,:7,:9,:11]=image[0];mask[0]=False;mask[0,:,:7,:9,:11]=valid[0]
        padding[0]=False;padding[0,:,:7,:9,:11]=True
        with torch.no_grad():
            solo=model(image,valid,affine);paired=model(batch,mask,affine.repeat(2,1,1),padding_mask=padding)
        count=int(solo["token_presence_mask"].sum())
        torch.testing.assert_close(solo["logits"][0],paired["logits"][0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(solo["token_features"][0],paired["token_features"][0,:count],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(solo["spatial_logits"][0],paired["spatial_logits"][0,:,:7,:9,:11],atol=1e-6,rtol=1e-5)
        self.assertEqual(paired["original_shape_zyx"].tolist(),[[7,9,11],[13,17,19]])
        self.assertTrue((paired["token_geometry"]["scale_ids"][0,count:]==-1).all())
        self.assertTrue((paired["token_features"][0,count:]==0).all())

    def test_localization_is_constant_on_exact_clipped_fine_support(self):
        image,valid,affine=sample();valid[:,:,:1]=False;output=small()(image,valid,affine)
        geometry=output["token_geometry"];selected=geometry["scale_ids"][0]==1
        for i,(start,end) in enumerate(zip(geometry["starts_xyz"][0,selected],geometry["ends_xyz"][0,selected])):
            x,y,z=start.tolist();ex,ey,ez=end.tolist()
            patch=output["spatial_logits"][0,0,z:ez,y:ey,x:ex]
            support=valid[0,0,z:ez,y:ey,x:ex]
            expected=output["fine_token_logits"][0,i].expand_as(patch)[support]
            torch.testing.assert_close(patch[support],expected)
        self.assertTrue((output["spatial_logits"][~valid]==0).all())

    def test_optional_localizer_and_no_ground_truth_forward_parameters(self):
        model=small(localization_head=False);image,valid,affine=sample((1,2,3))
        output=model(image,valid,affine)
        self.assertIsNone(output["spatial_logits"])
        self.assertEqual(set(inspect.signature(model.forward).parameters),{"image","valid_mask","voxel_to_lps","padding_mask"})
        with self.assertRaises(TypeError):model(image,valid,affine,labels=torch.tensor([0]))

    def test_invalid_mask_affine_padding_and_values_are_rejected(self):
        model=small();image,valid,affine=sample((5,6,7))
        invalid=[]
        singular=affine.clone();singular[:,0]=0
        nonhomogeneous=affine.clone();nonhomogeneous[:,3,0]=1
        nan=affine.clone();nan[:,0,0]=float("nan")
        bad=image.clone();bad[0,0,0,0,0]=1.01
        hole=valid.clone();hole[0,0,1,1,1]=False
        invalid.extend([(image.double(),valid,affine,None),(image,valid.float(),affine,None),
            (image,torch.zeros_like(valid),affine,None),(image,valid,singular,None),
            (image,valid,nonhomogeneous,None),(image,valid,nan,None),(bad,valid,affine,None),
            (image,hole,affine,hole),(image,valid,affine,torch.zeros_like(valid))])
        for index,args in enumerate(invalid):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):model(*args[:3],padding_mask=args[3])

    def test_constructor_validation(self):
        for arguments in ({"dimension":15},{"heads":False},{"layers":0},{"patch_sizes":(8,4,16)},
                          {"coarse_patch_size":6},{"max_tokens":4097},{"dropout":float("nan")},{"localization_head":1}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):small(**arguments)

    def test_budgets_reject_before_encoders(self):
        image,valid,affine=sample()
        for limits in ({"max_tokens":2},{"max_attention_elements":2},{"max_feature_grid_voxels":2},{"max_batch_voxels":2}):
            model=small(**limits)
            with patch.object(model.coarse_encoder,"forward",side_effect=AssertionError("should not encode")):
                with self.assertRaises(ValueError):model(image,valid,affine)

    def test_state_dict_roundtrip_and_independent_transformer_layers(self):
        model=small().eval();image,valid,affine=sample()
        self.assertFalse(torch.equal(model.transformer.blocks[0].linear1.weight,model.transformer.blocks[1].linear1.weight))
        stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
        restored=small().eval();restored.load_state_dict(torch.load(stream,weights_only=True))
        with torch.no_grad():torch.testing.assert_close(model(image,valid,affine)["logits"],restored(image,valid,affine)["logits"],atol=0,rtol=0)

    def test_default_96_cube_contract_and_backward(self):
        model=RSVTDenseFoundation();image,valid,affine=sample((96,96,96))
        output=model(image,valid,affine)
        self.assertEqual(tuple(output["token_features"].shape),(1,1998,64))
        self.assertEqual(output["compute"]["dense_sequence_length_including_cls"],1999)
        self.assertEqual(output["compute"]["constructed_coarse_tokens"].tolist(),[27])
        self.assertEqual(output["compute"]["constructed_candidate_tokens"].tolist(),[1971])
        F.cross_entropy(output["logits"],torch.tensor([3])).backward()
        self.assertTrue(torch.isfinite(model.coarse_encoder.first.weight.grad).all())


class VerifiedLocalization(unittest.TestCase):
    def test_verified_only_and_masked_nan_targets(self):
        logits=torch.zeros((2,1,2,2,2),requires_grad=True);targets=torch.full_like(logits,float("nan"))
        valid=torch.ones_like(logits,dtype=torch.bool);valid[0,0,0,0,0]=False
        targets[0][valid[0]]=1.
        result=verified_voxel_mask_loss(logits,targets,valid,torch.tensor([True,False]),annotation_kind="voxel_mask")
        self.assertEqual(result["eligible_cases"],1);result["loss"].backward()
        self.assertGreater(float(logits.grad[0].abs().sum()),0.)
        self.assertEqual(float(logits.grad[1].abs().sum()),0.)
        self.assertEqual(float(logits.grad[0,0,0,0,0]),0.)

    def test_no_verified_annotations_returns_differentiable_zero(self):
        logits=torch.full((1,1,2,2,2),float("nan"),requires_grad=True)
        result=verified_voxel_mask_loss(logits,logits.detach(),torch.ones_like(logits,dtype=torch.bool),torch.tensor([False]),annotation_kind="voxel_mask")
        self.assertEqual(float(result["loss"].detach()),0.);result["loss"].backward()
        self.assertTrue((logits.grad==0).all())

    def test_point_box_and_invalid_verified_masks_rejected(self):
        logits=torch.zeros((1,1,2,2,2));valid=torch.ones_like(logits,dtype=torch.bool)
        for kind,target in (("point",logits),("box",logits),("voxel_mask",logits+.5),("voxel_mask",logits+float("nan"))):
            with self.assertRaises(ValueError):verified_voxel_mask_loss(logits,target,valid,torch.tensor([True]),annotation_kind=kind)


class SyntheticIntegration(unittest.TestCase):
    def test_phase5_config_and_runner_write_reviewable_summary(self):
        with TemporaryDirectory() as temporary:
            summary=run_phase5(ROOT/"configs/models/rsvt_f05.toml",Path(temporary)/"run",device="cpu")
            self.assertTrue(summary["ok"],summary["checks"])
            self.assertTrue(all(summary["checks"].values()))
            self.assertFalse(summary["medical_data_read"]);self.assertFalse(summary["medical_training_ready"])
            self.assertEqual(json.loads((Path(temporary)/"run/summary.json").read_text())["config_sha256"],summary["config_sha256"])
            with self.assertRaises(ValueError):run_phase5(ROOT/"configs/models/rsvt_f05.toml",Path(temporary)/"run")

    def test_cli_and_config_rejection(self):
        config=load_phase5_config(ROOT/"configs/models/rsvt_f05.toml")
        for change in ({"protocol":{}},{"protocol":config["protocol"]|{"medical_training_ready":0}},
                       {"engineering":config["engineering"]|{"threads":0}},
                       {"model":config["model"]|{"localization_head":False}}):
            with patch("rsvt.experiments.phase5.load_config",return_value=config|change):
                with self.assertRaises(ValueError):load_phase5_config("unused")
        with patch("rsvt.experiments.phase5.run_phase5",return_value={"ok":True}) as runner:
            with patch("sys.stdout",new_callable=io.StringIO):
                self.assertEqual(main(["phase5-smoke","--output-dir","unused","--device","cpu"]),0)
            runner.assert_called_once()


if __name__=="__main__":unittest.main()
