"""F04 synthetic model, representation, optimization, validation, and resume checks."""

from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import test_f03_split_loader as fixtures
from rsvt.data.acquisition import sha256_file
from rsvt.data.baseline_views import coarse_full_fov,select_pilot_rows
from rsvt.data.dataset import CTVolumeDataset
from rsvt.data.splits import digest_json
from rsvt.evaluation.classification import classification_metrics,validation_metrics
from rsvt.losses.classification import balanced_cross_entropy,training_class_weights
from rsvt.models.baselines import DenseViT3D,ResNet50Volume
from rsvt.training.checkpoint import load_checkpoint
from rsvt.training.diagnostics import audit_model_inputs,gradient_groups
from rsvt.training.phase4 import load_phase4_config,run_phase4
from rsvt.training.trainer import evaluate_partition,fit_baseline
from rsvt.utils.config import load_config
from rsvt.utils.reproducibility import seed_everything,rng_state,restore_rng

torch.set_num_threads(2)


class ModelsAndMetrics(unittest.TestCase):
    def test_resnet50_and_vit_forward_backward_have_finite_gradients(self):
        for model in (ResNet50Volume(base_width=2),DenseViT3D((32,32,32),16,16,1,2,0.)):
            image=torch.rand(1,1,32,32,32)
            valid=torch.ones_like(image,dtype=torch.bool)
            output=model(image,valid)["logits"]
            self.assertEqual(tuple(output.shape),(1,4))
            output.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_models_ignore_changed_invalid_pixel_values(self):
        image=torch.rand(1,1,32,32,32)
        valid=torch.ones_like(image,dtype=torch.bool)
        valid[:,:,:8]=False
        changed=image.clone();changed[~valid]=10000.
        for model in (ResNet50Volume(2),DenseViT3D((32,32,32),16,16,1,2,0.)):
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(image,valid)["logits"],model(changed,valid)["logits"],rtol=0,atol=0)
            with self.assertRaises(ValueError):
                model(image,torch.zeros_like(valid))

    def test_invalid_vit_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            DenseViT3D((33,32,32),8)
        with self.assertRaises(ValueError):
            DenseViT3D(heads=0)
        with self.assertRaisesRegex(ValueError,"4096"):
            DenseViT3D((64,64,64),1)

    def test_class_weighting_remains_effective_for_batch_size_one(self):
        logits=torch.zeros((1,4),requires_grad=True)
        loss=balanced_cross_entropy(logits,torch.tensor([2]),torch.tensor([1.,1.,3.,1.]))
        self.assertAlmostEqual(float(loss.detach()),3*math.log(4),places=6)
        weights,counts=training_class_weights([0,0,1,2,3])
        torch.testing.assert_close(weights,torch.tensor([.625,1.25,1.25,1.25]))
        self.assertEqual(counts.tolist(),[2,1,1,1])
        with self.assertRaises(ValueError):
            training_class_weights([0,1,2])

    def test_perfect_validation_and_strict_threshold_ties(self):
        p=np.full((4,4),.02);np.fill_diagonal(p,.94)
        result=validation_metrics([0,1,2,3],p)
        self.assertEqual(result["accuracy"],1.)
        self.assertEqual(result["macro_f1"],1.)
        self.assertEqual(result["tamper_auroc"],1.)
        self.assertEqual(result["validation_threshold"]["empirical_fpr"],0.)
        self.assertEqual(result["validation_threshold"]["fpr_resolution"],.5)
        tied=validation_metrics([0,1,2,3],np.full((4,4),.25))
        self.assertEqual(tied["tamper_auroc"],.5)
        self.assertEqual(tied["validation_threshold"]["fb_tamper_recall"],0.)

    def test_undefined_metrics_are_not_reported_as_zero(self):
        result=validation_metrics([0],[[1.,0.,0.,0.]])
        self.assertIsNone(result["tamper_auroc"])
        self.assertIsNone(result["macro_f1"])
        self.assertIsNone(result["class_recall"]["FB"])
        with self.assertRaises(ValueError):
            validation_metrics([0],[[.1,.1,.1,.1]])

    def test_train_metrics_do_not_fit_a_validation_threshold(self):
        result=classification_metrics([0,1,2,3],np.eye(4),partition="train")
        self.assertEqual(result["scope"],"train_only")
        self.assertIsNone(result["validation_threshold"])
        with self.assertRaises(ValueError):
            classification_metrics([0,1,2,3],np.eye(4),partition="test")

    def test_constant_fb_predictions_expose_class_bias_and_false_positives(self):
        result=validation_metrics([0,0,1,1]+[2]*5+[3]*5,np.tile([.1,.1,.6,.2],(14,1)))
        self.assertEqual(result["class_recall"]["FB"],1.)
        self.assertEqual(result["predicted_class_counts"],{"TB":0,"TM":0,"FB":14,"FM":0})
        self.assertEqual(result["accuracy"],5/14)
        self.assertEqual(result["balanced_accuracy"],.25)
        self.assertTrue(result["constant_argmax"])
        self.assertEqual(result["hard_tamper_authentic_fpr"],1.)
        self.assertEqual(result["validation_threshold"]["fb_tamper_recall"],0.)

    def test_full_diagnostic_keeps_full_geometry_and_rejects_conflicting_stops(self):
        full=load_phase4_config(ROOT/"configs/training/baseline_full.toml")
        control=load_phase4_config(ROOT/"configs/training/baseline_full_diagnostic.toml")
        self.assertEqual(full["representation"],control["representation"])
        for key in full["models"]:
            if key!="dropout":self.assertEqual(full["models"][key],control["models"][key])
        self.assertEqual(control["training"]["stop_after_train_fit_epochs"],2)
        self.assertTrue(control["training"]["diagnostic_logging"])
        self.assertEqual(control["selection"]["train_per_class"],2)
        for invalid in (True,-1,31):
            bad=deepcopy(control);bad["training"]["stop_after_train_fit_epochs"]=invalid
            with patch("rsvt.training.phase4.load_config",return_value=bad),self.assertRaises(ValueError):
                load_phase4_config("unused")
        bad=deepcopy(control);bad["training"]["early_stopping"]=True
        with patch("rsvt.training.phase4.load_config",return_value=bad),self.assertRaises(ValueError):
            load_phase4_config("unused")

    def test_diagnostic_profile_preserves_subset_and_validates_criteria(self):
        pilot=load_phase4_config(ROOT/"configs/training/baseline_pilot.toml")
        diagnostic=load_phase4_config(ROOT/"configs/training/baseline_diagnostic.toml")
        self.assertEqual(pilot["selection"],diagnostic["selection"])
        self.assertEqual(pilot["representation"],diagnostic["representation"])
        self.assertFalse(diagnostic["training"]["early_stopping"])
        self.assertEqual(diagnostic["training"]["epochs"]*8//diagnostic["training"]["accumulation_steps"],320)
        with TemporaryDirectory() as directory:
            path=Path(directory)/"bad.toml"
            for invalid in (True,float("nan"),0.,1.1):
                bad=deepcopy(diagnostic);bad["training"]["train_fit_accuracy"]=invalid
                with patch("rsvt.training.phase4.load_config",return_value=bad),self.assertRaises(ValueError):
                    load_phase4_config(path)

    def test_full_fov_representation_preserves_all_support_corners(self):
        image=np.linspace(0,1,7*9*11,dtype=np.float32).reshape(1,7,9,11)
        affine=np.diag([.7,1.2,2.5,1.]);affine[:3,3]=[10,20,-5]
        sample={"image":image,"valid_mask":np.ones_like(image,dtype=bool),"voxel_to_lps":affine}
        result=coarse_full_fov(sample,(4,5,6))
        self.assertEqual(result["image"].shape,(1,4,5,6))
        for bits in range(8):
            old=[n-.5 if bits&(1<<i) else -.5 for i,n in enumerate((11,9,7))]+[1.]
            new=[n-.5 if bits&(1<<i) else -.5 for i,n in enumerate((6,5,4))]+[1.]
            np.testing.assert_allclose(affine@old,result["voxel_to_lps"]@new,atol=1e-10)
        self.assertTrue(result["valid_mask"].any())
        self.assertTrue(np.isfinite(result["image"]).all())

    def test_subset_selection_is_order_independent_and_never_selects_test(self):
        rows=[{"case_key":f"{split}-{label}-{i}","split":split,"label_candidate":label}
              for split in ("train","validation","test") for label in ("TB","TM","FB","FM") for i in range(3)]
        first=select_pilot_rows(rows,13,2,1)
        self.assertEqual(first,select_pilot_rows(list(reversed(rows)),13,2,1))
        self.assertEqual(len(first),12)
        self.assertNotIn("test",{r["split"] for r in first})


class ToyData:
    def __init__(self,split,labels):
        self.rows=[{"case_key":f"{split}-{i}","split":split,"label_index":label} for i,label in enumerate(labels)]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        image=torch.zeros(1,2,2,2);image.flatten()[row["label_index"]]=1.
        return {"image":image,"valid_mask":torch.ones_like(image,dtype=torch.bool),"target":row["label_index"],"case_key":row["case_key"]}


class ToyModel(nn.Module):
    def __init__(self,dropout=0.):
        super().__init__();self.drop=nn.Dropout(dropout);self.fc=nn.Linear(8,4)
    def forward(self,image,valid_mask):
        return {"logits":self.fc(self.drop(image.flatten(1)))}


class InputAuditTests(unittest.TestCase):
    def setUp(self):
        self.work=TemporaryDirectory(prefix="rsvt_input_audit_");self.root=Path(self.work.name)
        self.entries=[]
        for index in range(8):
            row={"case_key":f"case-{index}","split":"train" if index<4 else "validation",
                 "label_index":index%4,"shape_zyx":[4,4,4],"path":str(self.root/f"{index}.npz")}
            self.entries.append(row)
            self.save(index,np.full((1,4,4,4),(index+1)/10,dtype=np.float32))
    def tearDown(self):self.work.cleanup()
    def save(self,index,image,mask=None):
        row=self.entries[index]
        np.savez_compressed(row["path"],image=image,valid_mask=np.ones_like(image,dtype=bool) if mask is None else mask)
        row.update(sha256=sha256_file(Path(row["path"])),bytes=Path(row["path"]).stat().st_size)

    def test_unique_inputs_report_stats_without_promoting_preservation_QA(self):
        result=audit_model_inputs(self.entries)
        self.assertTrue(result["ok"])
        self.assertEqual(result["cases"],8)
        self.assertEqual(len(result["constant_valid_intensity_cases"]),8)
        self.assertEqual(result["records"][0]["valid_voxels"],64)
        self.assertAlmostEqual(result["records"][0]["mean"],.1,places=6)
        self.assertIn("not_lesion_preservation",result["scope"])

    def test_conflicting_inputs_ignore_pixels_the_model_masks_out(self):
        mask=np.ones((1,4,4,4),dtype=bool);mask[0,0,0,0]=False
        image=np.full(mask.shape,.2,dtype=np.float32)
        self.save(0,image,mask)
        changed=image.copy();changed[~mask]=123.
        self.save(1,changed,mask)
        result=audit_model_inputs(self.entries)
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["conflicting_target_groups"]),1)
        self.assertEqual(result["train_validation_overlap_groups"],[])

    def test_same_input_across_train_validation_and_out_of_range_are_flagged(self):
        self.save(4,np.full((1,4,4,4),.1,dtype=np.float32))
        result=audit_model_inputs(self.entries)
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["train_validation_overlap_groups"]),1)
        self.assertFalse(result["conflicting_target_groups"])
        self.save(4,np.full((1,4,4,4),1.1,dtype=np.float32))
        result=audit_model_inputs(self.entries)
        self.assertEqual(result["out_of_range_valid_voxels"],64)

    def test_audit_rejects_test_before_reading_any_pixels(self):
        rows=self.entries+[{"split":"test"}]
        with patch("rsvt.training.diagnostics.CachedViews",side_effect=AssertionError("must not read")),self.assertRaises(ValueError):
            audit_model_inputs(rows)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.work=TemporaryDirectory(prefix="rsvt_f04_");self.root=Path(self.work.name)
        self.config=load_config(ROOT/"configs/training/baseline_pilot.toml")["training"]
        self.config.update(device="cpu",mixed_precision=False,learning_rate=.05,weight_decay=0.,patience=30,gradient_clip=1000.)
        self.train,self.val=ToyData("train",[0,1,2,3]*2),ToyData("validation",[0,1,2,3])
        self.contract={"synthetic":True,"config":self.config}
    def tearDown(self):self.work.cleanup()
    def model(self,dropout=0.):
        seed_everything(self.config["seed"]);return ToyModel(dropout)

    def test_learns_controlled_synthetic_training_subset(self):
        self.config["epochs"]=24
        result=fit_baseline(self.model(),self.train,self.val,self.config,self.root/"learn",self.contract)
        self.assertLess(result["history"][-1]["training_weighted_loss"],result["history"][0]["training_weighted_loss"]*.5)
        self.assertEqual(result["validation_metrics"]["accuracy"],1.)
        self.assertEqual(result["training_metrics_at_best_validation"]["accuracy"],1.)
        self.assertEqual(result["history"][-1]["training_accuracy"],1.)
        self.assertEqual(result["test_cases_opened"],0)
        self.assertFalse(result["training_ready"])

    def test_train_evaluation_is_read_only_and_rng_independent(self):
        model=self.model(.5);before=deepcopy(model.state_dict());saved=rng_state()
        first,predictions=evaluate_partition(model,self.train,torch.device("cpu"),partition="train")
        second,_=evaluate_partition(model,self.train,torch.device("cpu"),partition="train")
        after=rng_state()
        self.assertEqual(first,second)
        self.assertEqual(len(predictions),len(self.train))
        self.assertEqual(saved["python"],after["python"])
        self.assertEqual(saved["numpy"],after["numpy"])
        torch.testing.assert_close(saved["torch_cpu"],after["torch_cpu"],rtol=0,atol=0)
        for key,value in before.items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        with self.assertRaises(ValueError):
            evaluate_partition(model,ToyData("test",[0,1,2,3]),torch.device("cpu"),partition="train")

    def test_fixed_diagnostic_budget_reports_last_fit_separately_from_validation_best(self):
        class ConflictingValidation(ToyData):
            def __getitem__(self,index):
                item=super().__getitem__(index)
                item["image"]=torch.roll(item["image"].flatten(),1).reshape(1,2,2,2)
                return item
        self.config.update(epochs=24,accumulation_steps=1,patience=1,early_stopping=False)
        result=fit_baseline(self.model(),self.train,ConflictingValidation("validation",[0,1,2,3]),
                            self.config,self.root/"diagnostic",self.contract)
        diagnostic=result["train_fit_diagnostic"]
        self.assertEqual(result["epochs_completed"],24)
        self.assertEqual(diagnostic["optimizer_updates_completed"],24*8)
        self.assertFalse(result["stopped_early"])
        self.assertTrue(diagnostic["last_epoch"]["passed"])
        self.assertTrue(diagnostic["ever_passed"])
        self.assertLess(result["best_epoch"],24)
        self.assertLess(result["training_metrics_at_best_validation"]["accuracy"],diagnostic["last_epoch"]["accuracy"])
        self.assertEqual(result["best_epoch"],min(result["history"],key=lambda r:r["validation_loss"])["epoch"])
        last_predictions=json.loads((self.root/"diagnostic/last_training_predictions.json").read_text())
        self.assertEqual(sum(p["predicted_class"]==p["candidate_target"] for p in last_predictions),8)

    def test_unsatisfied_train_fitting_is_reported_without_promoting_readiness(self):
        self.config.update(epochs=1,train_fit_max_loss=1e-12)
        result=fit_baseline(self.model(),self.train,self.val,self.config,self.root/"not_fit",self.contract)
        diagnostic=result["train_fit_diagnostic"]
        self.assertFalse(diagnostic["ever_passed"])
        self.assertIsNone(diagnostic["first_passing_epoch"])
        self.assertFalse(diagnostic["last_epoch"]["passed"])
        self.assertTrue(result["ok"])
        self.assertFalse(result["training_ready"])

    def test_resume_matches_uninterrupted_updates_with_dropout(self):
        self.config["diagnostic_logging"]=True
        fit_baseline(self.model(.25),self.train,self.val,self.config,self.root/"full",self.contract)
        fit_baseline(self.model(.25),self.train,self.val,self.config,self.root/"resumed",self.contract,stop_after_epoch=1)
        result=fit_baseline(self.model(.25),self.train,self.val,self.config,self.root/"resumed",self.contract,resume=True)
        digest=digest_json(self.contract)
        full=load_checkpoint(self.root/"full/last.pt",digest)
        resumed=load_checkpoint(self.root/"resumed/last.pt",digest)
        for key,value in full["model"].items():
            torch.testing.assert_close(value,resumed["model"][key],rtol=0,atol=0)
        self.assertEqual([r["training_weighted_loss"] for r in full["history"]],[r["training_weighted_loss"] for r in resumed["history"]])
        self.assertEqual([r["training_accuracy"] for r in full["history"]],[r["training_accuracy"] for r in resumed["history"]])
        self.assertEqual([r["training_eval_loss"] for r in full["history"]],[r["training_eval_loss"] for r in resumed["history"]])
        self.assertEqual(full["initial_evaluation"],resumed["initial_evaluation"])
        self.assertEqual(full["epoch_diagnostics"],resumed["epoch_diagnostics"])
        self.assertEqual(result["resume_origin"],"last_checkpoint")

    def test_train_fit_stop_and_resume_recover_diagnostics_without_new_updates(self):
        self.config.update(epochs=10,early_stopping=False,stop_after_train_fit_epochs=2,
                           train_fit_accuracy=.25,train_fit_max_loss=10.,diagnostic_logging=True)
        result=fit_baseline(self.model(),self.train,self.val,self.config,self.root/"control",self.contract)
        self.assertEqual(result["epochs_completed"],2)
        self.assertEqual(result["stop_reason"],"train_fit_control_satisfied")
        self.assertTrue(result["stopped_after_train_fit"])
        self.assertEqual(result["train_fit_diagnostic"]["last_pass_streak"],2)
        path=self.root/"control/last.pt";before=sha256_file(path)
        (self.root/"control/epoch_diagnostics.json").unlink()
        resumed=fit_baseline(self.model(),self.train,self.val,self.config,self.root/"control",self.contract,resume=True)
        self.assertEqual(sha256_file(path),before)
        self.assertEqual(resumed["epochs_completed"],2)
        self.assertEqual(len(json.loads((self.root/"control/epoch_diagnostics.json").read_text())),2)

    def test_clipping_and_module_gradients_are_logged_before_clipping(self):
        self.config.update(epochs=1,gradient_clip=1e-6,diagnostic_logging=True)
        result=fit_baseline(self.model(),self.train,self.val,self.config,self.root/"grads",self.contract)
        row=result["history"][0]
        self.assertEqual(row["clipped_gradient_steps"],2)
        self.assertEqual(row["zero_gradient_steps"],0)
        self.assertEqual(row["nonfinite_gradient_steps"],0)
        self.assertLess(row["mean_gradient_clip_scale"],1.)
        self.assertEqual(sum(row["training_predicted_"+label] for label in ("TB","TM","FB","FM")),8)
        audit=json.loads((self.root/"grads/epoch_diagnostics.json").read_text())[0]
        self.assertGreater(audit["gradient_groups_preclip_last_attempt"]["fc"]["l2_norm"],1e-6)
        self.assertEqual(audit["gradient_groups_preclip_last_attempt"]["fc"]["with_gradient"],2)
        self.assertEqual(result["initial_evaluation"]["train"]["cases"],8)

    def test_gradient_group_read_does_not_modify_gradients_or_rng(self):
        model=self.model();model.unused=nn.Parameter(torch.ones(2))
        model(torch.ones(1,1,2,2,2),torch.ones(1,1,2,2,2,dtype=torch.bool))["logits"].sum().backward()
        before={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None};rng=torch.get_rng_state()
        result=gradient_groups(model)
        self.assertIsNone(result["unused"]["l2_norm"])
        self.assertEqual(result["unused"]["with_gradient"],0)
        for k,p in model.named_parameters():
            if p.grad is not None:torch.testing.assert_close(p.grad,before[k],rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)

    def test_partial_accumulation_uses_actual_number_of_examples(self):
        train=ToyData("train",[0,1,2,3,0])
        self.config.update(epochs=1,accumulation_steps=8)
        model=self.model();expected=deepcopy(model)
        samples=[train[i] for i in range(len(train))]
        image=torch.stack([s["image"] for s in samples]);mask=torch.stack([s["valid_mask"] for s in samples])
        target=torch.tensor([s["target"] for s in samples]);weights,_=training_class_weights(target)
        balanced_cross_entropy(expected(image,mask)["logits"],target,weights).backward()
        norm=float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in expected.parameters()])))
        result=fit_baseline(model,train,self.val,self.config,self.root/"accum",self.contract)
        self.assertAlmostEqual(result["history"][0]["max_gradient_norm"],norm,places=6)
        self.assertEqual(result["history"][0]["optimizer_steps"],1)

    def test_rng_roundtrip_covers_python_numpy_and_torch(self):
        seed_everything(37);saved=rng_state()
        first=(random.random(),np.random.rand(),torch.rand(3))
        restore_rng(saved);second=(random.random(),np.random.rand(),torch.rand(3))
        self.assertEqual(first[:2],second[:2]);torch.testing.assert_close(first[2],second[2],rtol=0,atol=0)

    def test_early_stopping_tracks_validation_loss(self):
        self.config.update(epochs=5,patience=1,learning_rate=1e-12)
        result=fit_baseline(self.model(),self.train,self.val,self.config,self.root/"early",self.contract)
        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["epochs_completed"],2)
        self.assertEqual(result["best_epoch"],1)

    def test_checkpoint_corruption_and_changed_contract_are_rejected(self):
        fit_baseline(self.model(),self.train,self.val,self.config,self.root/"state",self.contract,stop_after_epoch=1)
        path=self.root/"state/last.pt"
        with self.assertRaisesRegex(ValueError,"contract"):
            load_checkpoint(path,"0"*64)
        with path.open("ab") as stream:stream.write(b"synthetic corruption")
        with self.assertRaisesRegex(ValueError,"checksum"):
            load_checkpoint(path,digest_json(self.contract))

    def test_training_requires_acknowledgement_and_rejects_test_inputs(self):
        bad={**self.config,"candidate_targets_acknowledged":False}
        with self.assertRaises(ValueError):
            fit_baseline(self.model(),self.train,self.val,bad,self.root/"no",self.contract)
        with self.assertRaises(ValueError):
            fit_baseline(self.model(),self.train,ToyData("test",[0,1,2,3]),self.config,self.root/"no",self.contract)


class Phase4Integration(fixtures.LoaderFixture):
    def setUp(self):
        super().setUp()
        f03_config=self.root/"f03.toml"
        text=(ROOT/"configs/splits/ct_gan_draft.toml").read_text().replace("train = 0.70","train = 0.34").replace("validation = 0.15","validation = 0.33").replace("test = 0.15","test = 0.33")
        f03_config.write_text(text)
        self.config=f03_config
        f03=self.build("f03")
        self.assertTrue(f03["ok"])
        self.f03=self.root/"f03"
        text=(ROOT/"configs/training/baseline_pilot.toml").read_text()
        text=text.replace("96098c998851686609b4ae9c0919bc3bdc2e3d9dd59e88a7330a95b6f61fddc6",f03["assignments_sha256"])
        for old,new in (("[64, 64, 64]","[32, 32, 32]"),("epochs = 3","epochs = 2"),("train_per_class = 2","train_per_class = 1"),
                        ('names = ["resnet50_3d", "vit3d"]','names = ["vit3d"]'),("patch_size = 8","patch_size = 16"),
                        ("dimension = 96","dimension = 16"),("layers = 2","layers = 1"),("heads = 4","heads = 2"),('device = "auto"','device = "cpu"')):
            text=text.replace(old,new)
        self.f04_config=self.root/"f04.toml";self.f04_config.write_text(text)

    def test_pipeline_preserves_draft_and_only_prepares_train_validation_pixels(self):
        source=self.f03/"manifest.draft.json";before=source.read_bytes()
        observed=[];original=CTVolumeDataset.__getitem__
        def track(dataset,index):
            observed.append(dataset.rows[index]["split"])
            return original(dataset,index)
        both=self.root/"both.toml"
        both.write_text(self.f04_config.read_text().replace('names = ["vit3d"]','names = ["resnet50_3d", "vit3d"]').replace('cnn_width = 16','cnn_width = 2'))
        with patch("rsvt.data.baseline_views.CTVolumeDataset.__getitem__",track):
            report=run_phase4(self.f03,both,self.root/"f04")
        self.assertTrue(report["ok"],report.get("reason"))
        self.assertEqual((report["train_cases"],report["validation_cases"],report["test_cases_opened"]),(4,4,0))
        self.assertEqual(set(observed),{"train","validation"})
        self.assertEqual(source.read_bytes(),before)
        self.assertEqual(set(report["models"]),{"resnet50_3d","vit3d"})
        self.assertTrue(all(not r["classification_eligible"] for r in json.loads(before)["records"]))

    def test_candidate_loader_mode_requires_explicit_acknowledgement_and_rejects_test(self):
        path=self.f03/"manifest.draft.json"
        with self.assertRaisesRegex(ValueError,"acknowledgement"):
            CTVolumeDataset(path,"validation",purpose="engineering_pilot")
        with self.assertRaisesRegex(ValueError,"train/validation only"):
            CTVolumeDataset(path,"test",purpose="engineering_pilot",allow_candidate_targets=True)
        data=CTVolumeDataset(path,"validation",purpose="engineering_pilot",allow_candidate_targets=True)
        self.assertFalse(data[0]["classification_eligible"])

    def test_input_audit_failure_is_saved_before_any_model_training(self):
        config=self.root/"audited.toml"
        config.write_text(self.f04_config.read_text().replace('diagnostic_logging = false','diagnostic_logging = true'))
        failure={"ok":False,"records":[{"case_key":"synthetic-conflict"}],"test_cases_opened":0}
        with patch("rsvt.training.phase4.audit_model_inputs",return_value=failure),patch("rsvt.training.phase4.fit_baseline") as trainer:
            result=run_phase4(self.f03,config,self.root/"failed_audit")
        trainer.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertEqual(result["models"],{})
        self.assertIn("input_audit.json",result["reason"])
        self.assertEqual(json.loads((self.root/"failed_audit/input_audit.json").read_text()),failure)
        self.assertEqual(result["test_cases_opened"],0)

    def test_resume_reuses_views_and_completed_model_without_new_updates(self):
        first=run_phase4(self.f03,self.f04_config,self.root/"f04")
        self.assertTrue(first["ok"],first.get("reason"))
        path=self.root/"f04/vit3d/last.pt";before=sha256_file(path)
        (self.root/"f04/vit3d/last_training_predictions.json").unlink()
        with patch("rsvt.data.baseline_views.CTVolumeDataset.__getitem__",side_effect=AssertionError("must reuse cached snapshot")):
            second=run_phase4(self.f03,self.f04_config,self.root/"f04",resume=True)
        self.assertTrue(second["ok"],second.get("reason"))
        self.assertEqual(sha256_file(path),before)
        self.assertEqual(len(json.loads((self.root/"f04/vit3d/last_training_predictions.json").read_text())),4)
        changed=self.root/"changed.toml";changed.write_text(self.f04_config.read_text().replace("epochs = 2","epochs = 3"))
        with self.assertRaisesRegex(ValueError,"identical"):
            run_phase4(self.f03,changed,self.root/"f04",resume=True)

    def test_cached_view_corruption_is_rejected_on_resume(self):
        first=run_phase4(self.f03,self.f04_config,self.root/"f04")
        self.assertTrue(first["ok"],first.get("reason"))
        cache=next((self.root/"f04/views").glob("*.npz"))
        with cache.open("ab") as stream:stream.write(b"changed cache")
        second=run_phase4(self.f03,self.f04_config,self.root/"f04",resume=True)
        self.assertFalse(second["ok"])
        self.assertIn("Cached coarse view",second["reason"])


if __name__ == "__main__":unittest.main()
