"""F08 experiment identities, metadata-only audits, sealed jobs and validation replay."""

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rsvt.cli import main
from rsvt.data.splits import CLASS_INDEX,digest_json
from rsvt.evaluation.thresholds import calibrate_source_threshold,evaluate_frozen_threshold
from rsvt.experiments.ablations import load_catalog
from rsvt.experiments.protocol import ASSIGNMENT_FIELDS,audit_f03_metadata,research_readiness
from rsvt.experiments.runner import (build_plan,checkpoint_load,evaluate_inference,exclusive_run,fixtures,infer_checkpoint,
    load_phase8_config,run_phase8,verify_completed)

CONFIG=ROOT/'configs/orchestration/phase08.toml'


def small_config(directory,*,ids=('B1',),steps=1):
    path=Path(directory)/'config.toml'
    path.write_text('extends = ['+json.dumps(CONFIG.as_posix())+']\n[engineering]\nexecute_ids='+json.dumps(list(ids))+
        '\nsteps='+str(steps)+'\n[catalog]\nbaselines='+json.dumps([(ROOT/f'configs/experiments/registry/B{i}.toml').as_posix() for i in range(6)])+
        '\nablations='+json.dumps([(ROOT/f'configs/ablations/E{i}.toml').as_posix() for i in range(1,8)])+'\n',encoding='utf-8')
    return path


def manifest_fixture(path,*,ready=False,leak=False):
    rows=[]
    for partition in ('train','validation','test'):
        for index,label in enumerate(CLASS_INDEX):
            key=f'{partition}:{label}'
            rows.append({'case_key':key,'source_group_candidate':key,'split':partition,'label_candidate':label,
                'source_file_set_sha256':digest_json([key,'source']),'normalized_sha256':digest_json([key,'pixels']),
                'generator_candidate':'ct_gan' if index>=2 else '', 'classification_eligible':ready,
                'identity_reconciled':ready,'label_verified':ready,'partition_status':'reviewed' if ready else 'engineering_draft',
                'volume_path':'/must_not_be_opened/medical_volume.nii.gz'})
    if leak:rows[4]['source_group_candidate']=rows[0]['source_group_candidate']
    assignments=[{key:r[key] for key in ASSIGNMENT_FIELDS} for r in sorted(rows,key=lambda r:r['case_key'])]
    document={'schema_version':'f03_candidate_v1','class_index':CLASS_INDEX,'records':rows,
        'assignments_sha256':digest_json(assignments),'training_ready':ready,'status':'reviewed' if ready else 'draft'}
    Path(path).write_text(json.dumps(document),encoding='utf-8');return document


class CatalogAndReadiness(unittest.TestCase):
    def test_shared_fixture_geometry_does_not_encode_class_identity(self):
        batches,_=fixtures(13,[32,32,32],'cpu',2)
        for batch in batches.values():
            affine=batch['voxel_to_lps']
            self.assertTrue((affine==affine[:1].expand_as(affine)).all())

    def test_thirteen_entries_and_proposal_objective_controls(self):
        config=load_phase8_config(CONFIG);catalog=load_catalog(CONFIG,config)
        self.assertEqual(len(catalog),13)
        self.assertEqual([k for k,v in catalog.items() if v['entry']['synthetic_backend_supported']],['B0','B1','B3','B4','B5'])
        self.assertFalse(catalog['B2']['entry']['synthetic_backend_supported'])
        self.assertFalse(catalog['B4']['objective']['generator_variance'])
        self.assertTrue(catalog['B5']['objective']['generator_variance'])
        self.assertTrue(all(not v['entry']['medical_backend_enabled'] for v in catalog.values()))

    def test_design_only_backend_cannot_be_requested(self):
        with TemporaryDirectory() as temporary:
            config=small_config(temporary,ids=('B2',))
            with self.assertRaisesRegex(ValueError,'No synthetic execution backend'):build_plan(config,device='cpu')

    def test_metadata_only_manifest_audit_preserves_draft_flags(self):
        with TemporaryDirectory() as temporary:
            manifest=Path(temporary)/'manifest.json';manifest_fixture(manifest)
            result=audit_f03_metadata(manifest)
            self.assertEqual(result['volume_files_read'],0);self.assertFalse(result['research_ready'])
            self.assertEqual(result['qa_true_counts']['label_verified'],0)
            self.assertEqual(result['distribution']['test']['cases'],4)
            self.assertEqual(result['generator_candidates'],['ct_gan'])
            # No files exist at volume_path; successful audit proves voxel I/O was not required.

    def test_assignment_corruption_and_partition_leakage_rejected(self):
        with TemporaryDirectory() as temporary:
            manifest=Path(temporary)/'manifest.json';doc=manifest_fixture(manifest)
            doc['records'][0]['split']='test';manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'checksum'):audit_f03_metadata(manifest)
            manifest_fixture(manifest,leak=True)
            with self.assertRaisesRegex(ValueError,'leakage'):audit_f03_metadata(manifest)

    def test_even_reviewed_flags_do_not_enable_final_test_backend(self):
        config=load_phase8_config(CONFIG);catalog=load_catalog(CONFIG,config)
        with TemporaryDirectory() as temporary:
            manifest=Path(temporary)/'manifest.json';manifest_fixture(manifest,ready=True)
            metadata=audit_f03_metadata(manifest);self.assertTrue(metadata['research_ready'])
            result=research_readiness(config['research_protocol'],metadata,catalog)
            self.assertFalse(result['research_ready']);self.assertFalse(result['final_test_access_allowed'])
            self.assertIn('medical_and_final_test_dispatch_not_implemented',result['blocking_reasons'])
            self.assertIn('target_fpr',result['pending_decisions'])

    def test_config_rejects_medical_scope_bad_limits_and_demo_FPR(self):
        config=load_phase8_config(CONFIG)
        for change in ({'engineering':config['engineering']|{'scope':'medical'}},
            {'engineering':config['engineering']|{'steps':100}}, {'engineering':config['engineering']|{'shape_zyx':[96,96,96]}},
            {'demo_policy':config['demo_policy']|{'target_fpr':True}},
            {'demo_loss':config['demo_loss']|{'alpha':1}}):
            with patch('rsvt.experiments.runner.load_config',return_value=config|change):
                with self.assertRaises(ValueError):load_phase8_config('unused')

    def test_metadata_plan_has_no_final_lock_or_test_access(self):
        plan=build_plan(CONFIG,device='cpu',forward_smoke=False)
        self.assertEqual(plan['contract_sha256'],digest_json(plan['payload']))
        self.assertEqual(plan['payload']['kind'],'engineering_draft_snapshot_not_final_protocol')
        self.assertFalse(plan['payload']['test_volume_access_allowed'])
        self.assertFalse(plan['payload']['research_readiness']['confirmatory_lock_created'])


class FrozenThresholds(unittest.TestCase):
    def setUp(self):
        self.y=np.array([0,0,1,1,2,3]);scores=np.array([.9,.8,.8,.1,.98,.95])
        self.p=np.column_stack(((1-scores)/2,(1-scores)/2,scores*.1,scores*.9))

    def calibrate(self,alpha=.25,**changes):
        return calibrate_source_threshold(self.y,self.p,**(dict(target_fpr=alpha,partition='validation',checkpoint_sha256='weights',prediction_sha256='preds')|changes))

    def test_conservative_strict_ties_at_empirical_FPR(self):
        threshold=self.calibrate();self.assertAlmostEqual(threshold['threshold'],.8)
        self.assertEqual(threshold['allowed_validation_FP'],1);self.assertEqual(threshold['observed_validation_FP'],1)
        self.assertEqual(self.calibrate(.05)['observed_validation_FP'],0)

    def test_tamper_recall_is_distinct_from_four_class_FB_recall(self):
        calibration=self.calibrate(.05)
        result=evaluate_frozen_threshold(self.y,self.p,calibration,checkpoint_sha256='weights',partition='validation')
        self.assertEqual(result['FB_tamper_recall'],1.)
        self.assertEqual(result['FB_classification_recall'],0.)

    def test_threshold_and_checkpoint_binding_reject_changes(self):
        calibration=self.calibrate()
        with self.assertRaises(ValueError):evaluate_frozen_threshold(self.y,self.p,calibration|{'threshold':.1},checkpoint_sha256='weights',partition='validation')
        with self.assertRaises(ValueError):evaluate_frozen_threshold(self.y,self.p,calibration,checkpoint_sha256='other',partition='validation')
        with self.assertRaises(ValueError):evaluate_frozen_threshold(self.y,self.p,calibration,checkpoint_sha256='weights',partition='test')

    def test_no_authentic_nonvalidation_and_invalid_probabilities_rejected(self):
        for partition in ('train','test'):
            with self.assertRaises(ValueError):self.calibrate(partition=partition)
        with self.assertRaises(ValueError):calibrate_source_threshold([2],[[0.,0.,1.,0.]],target_fpr=.05,partition='validation',checkpoint_sha256='x',prediction_sha256='y')
        with self.assertRaises(ValueError):calibrate_source_threshold(self.y,self.p*2,target_fpr=.05,partition='validation',checkpoint_sha256='x',prediction_sha256='y')


class CheckpointAndResume(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory();self.root=Path(self.temporary.name);self.config=small_config(self.root)
    def tearDown(self):self.temporary.cleanup()

    def completed(self):
        summary=run_phase8(self.config,self.root/'run',device='cpu')
        attempt=self.root/'run/jobs/B1_s13/attempt_0001'
        result=json.loads((attempt/'result.json').read_text());return summary,attempt,result

    def test_training_checkpoint_replay_and_frozen_evaluation_match(self):
        summary,attempt,result=self.completed();self.assertTrue(summary['ok'],summary['checks'])
        checkpoint=attempt/result['best_checkpoint']['path']
        replay=infer_checkpoint(checkpoint,self.root/'replay.json',device='cpu')
        original=json.loads((attempt/'predictions.json').read_text())
        self.assertEqual(replay['prediction_sha256'],digest_json(original))
        metrics=evaluate_inference(self.root/'replay.json',attempt/'threshold.json',self.root/'evaluation.json')
        self.assertEqual(metrics,result['metrics']);self.assertEqual(summary['test_cases_opened'],0)
        with self.assertRaises(ValueError):infer_checkpoint(checkpoint,self.root/'replay.json',device='cpu')

    def test_resume_skips_checksum_verified_completed_job(self):
        summary,attempt,result=self.completed();before=(attempt/'result.json').read_bytes()
        with patch('rsvt.experiments.runner.run_job',side_effect=AssertionError('must skip complete job')):
            resumed=run_phase8(self.config,self.root/'run',device='cpu',resume=True)
        self.assertEqual(resumed['jobs_skipped_on_resume'],['B1_s13']);self.assertEqual(before,(attempt/'result.json').read_bytes())
        with self.assertRaises(FileExistsError):run_phase8(self.config,self.root/'run',device='cpu')

    def test_completed_artifact_corruption_is_rejected(self):
        _,attempt,_=self.completed();(attempt/'predictions.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'artifact changed'):run_phase8(self.config,self.root/'run',device='cpu',resume=True)

    def test_checkpoint_corruption_is_rejected_before_loading(self):
        _,attempt,result=self.completed();checkpoint=attempt/result['best_checkpoint']['path']
        checkpoint.write_bytes(checkpoint.read_bytes()+b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'):checkpoint_load(checkpoint)

    def test_changed_plan_refuses_resume(self):
        self.completed();small_config(self.root,steps=2)
        with self.assertRaisesRegex(ValueError,'identical'):run_phase8(self.config,self.root/'run',device='cpu',resume=True)

    def test_interrupted_attempt_preserved_and_restarted_in_new_directory(self):
        def interrupted(entry,seed,config,binding,destination,device):
            (destination/'partial.txt').write_text('preserve evidence');raise KeyboardInterrupt()
        with patch('rsvt.experiments.runner.run_job',side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):run_phase8(self.config,self.root/'run',device='cpu')
        resumed=run_phase8(self.config,self.root/'run',device='cpu',resume=True)
        self.assertTrue(resumed['ok'])
        self.assertEqual((self.root/'run/jobs/B1_s13/attempt_0001/partial.txt').read_text(),'preserve evidence')
        self.assertTrue((self.root/'run/jobs/B1_s13/attempt_0002/result.json').is_file())

    def test_run_lock_prevents_simultaneous_writers(self):
        directory=self.root/'lock';directory.mkdir()
        with exclusive_run(directory):
            with self.assertRaises(ValueError):
                with exclusive_run(directory):pass

    def test_test_partition_inference_and_altered_prediction_partition_rejected(self):
        _,attempt,result=self.completed();checkpoint=attempt/result['best_checkpoint']['path']
        with self.assertRaises(ValueError):infer_checkpoint(checkpoint,self.root/'test.json',device='cpu',partition='test')
        prediction=infer_checkpoint(checkpoint,self.root/'replay.json',device='cpu');prediction['partition']='test'
        (self.root/'replay.json').write_text(json.dumps(prediction))
        with self.assertRaises(ValueError):evaluate_inference(self.root/'replay.json',attempt/'threshold.json',self.root/'metrics.json')

    def test_cli_plan_dispatch_writes_new_snapshot(self):
        plan=build_plan(self.config,device='cpu',forward_smoke=False)
        with patch('rsvt.experiments.runner.build_plan',return_value=plan):
            with patch('sys.stdout',new_callable=io.StringIO):
                self.assertEqual(main(['phase8-plan','--config',str(self.config),'--output',str(self.root/'plan.json')]),0)
        saved=json.loads((self.root/'plan.json').read_text());self.assertEqual(saved['contract_sha256'],plan['contract_sha256'])


if __name__=='__main__':unittest.main()
