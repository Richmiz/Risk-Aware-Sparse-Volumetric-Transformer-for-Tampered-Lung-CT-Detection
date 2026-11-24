"""Paired cluster oracles, unsupported resamples, ablation controls and replay."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from rsvt.data.splits import digest_json
from rsvt.evaluation.statistics import align_pairs, paired_cluster_bootstrap, noninferiority_status, pareto_frontier
from rsvt.evaluation.thresholds import calibrate_source_threshold
from rsvt.experiments.phase10 import load_phase10_config, paired_variants, run_phase10, analyze_phase10, read_phase10
from rsvt.experiments.runner import run_phase8
from test_f08_orchestration import small_config

CONFIG=ROOT/'configs/analysis/phase10.toml'


def pairs(groups=3, unequal=False):
    first=[]; second=[]
    for group in range(groups):
        for replica in range(2 if unequal and group==1 else 1):
            for label in range(4):
                common={'case_id':f'validation:g{group}:r{replica}:c{label}','source_group':f'parent{group}',
                    'partition':'validation','label':label,'generator_id':0 if label>=2 else -1,'generator_verified':label>=2}
                perfect=[.7 if i==label else .1 for i in range(4)]
                first.append({**common,'probabilities':perfect})
                second.append({**common,'probabilities':perfect if unequal and group==0 else [.25]*4})
    def calibration(rows, name):
        return calibrate_source_threshold([r['label'] for r in rows],[r['probabilities'] for r in rows],
            target_fpr=.05,partition='validation',checkpoint_sha256=digest_json(name),prediction_sha256=digest_json(rows))
    return first,second,calibration(first,'first'),calibration(second,'second')


class PairedClusterBootstrap(unittest.TestCase):
    def test_constant_oracles_and_pointwise_scope(self):
        r=paired_cluster_bootstrap(*pairs())
        for key,value in [('FB_tamper_recall',1),('authentic_FPR',0),('macro_f1_all_four',.9),('tamper_auroc',.5)]:
            self.assertAlmostEqual(r['metrics'][key]['delta_first_minus_second'],value)
            np.testing.assert_allclose(r['metrics'][key]['percentile_interval'],[value,value])
            self.assertEqual(r['metrics'][key]['valid_replicates'],1000)
        self.assertFalse(r['research_inference_allowed']);self.assertFalse(r['canonical_patient_independence_verified'])

    def test_identical_models_and_row_order_invariance(self):
        a,b,ca,cb=pairs(2)
        first=paired_cluster_bootstrap(a,b,ca,cb)
        self.assertEqual(first,paired_cluster_bootstrap(a[::-1],b[::-1],ca,cb))
        r=paired_cluster_bootstrap(a,a,ca,ca)
        self.assertEqual(r['metrics']['macro_f1_all_four']['percentile_interval'],[0,0])

    def test_whole_unequal_clusters_weight_volumes_and_share_draws(self):
        r=paired_cluster_bootstrap(*pairs(2,unequal=True)); self.assertEqual(r['volumes'],12)
        self.assertAlmostEqual(r['metrics']['FB_tamper_recall']['delta_first_minus_second'],2/3)
        for draw,row in zip(r['draw_indices'],r['distribution']):
            count=draw.count(1)
            expected={0:(8,0),1:(12,2/3),2:(16,1)}[count]
            self.assertEqual(row['sampled_volumes'],expected[0])
            self.assertAlmostEqual(row['delta']['FB_tamper_recall'],expected[1])

    def test_missing_resampled_classes_suppress_interval_instead_of_filtering(self):
        a,b,ca,cb=pairs(1)
        for rows in (a,b):
            for row in rows:row['source_group']='class_parent'+str(row['label'])
        r=paired_cluster_bootstrap(a,b,ca,cb)
        metric=r['metrics']['macro_f1_all_four']
        self.assertGreater(metric['valid_replicates'],0)
        self.assertGreater(metric['undefined_replicates'],0)
        self.assertIsNone(metric['percentile_interval'])
        self.assertTrue(any(d['delta']['FB_tamper_recall'] is None for d in r['distribution']))

    def test_source_or_patient_clusters_are_explicit(self):
        a,b,ca,cb=pairs(4)
        with self.assertRaises(ValueError):paired_cluster_bootstrap(a,b,ca,cb,cluster_key='patient_id')
        for rows in (a,b):
            for row in rows:row['patient_id']='patient'+str(int(row['source_group'][-1])//2)
        r=paired_cluster_bootstrap(a,b,ca,cb,cluster_key='patient_id')
        self.assertEqual(len(r['clusters']),2)
        self.assertEqual([len(v) for v in r['cluster_members'].values()],[8,8])

    def test_thresholds_are_never_reestimated_inside_bootstrap(self):
        p=pairs(2)
        with patch('rsvt.evaluation.thresholds.calibrate_source_threshold',side_effect=AssertionError('Frozen only')):
            r=paired_cluster_bootstrap(*p)
        self.assertEqual(r['threshold_bindings'],[p[2]['binding_sha256'],p[3]['binding_sha256']])

    def test_pair_labels_groups_generators_and_case_sets_must_match(self):
        a,b,_,_=pairs(2)
        for key,value in [('label',1),('source_group','other'),('generator_id',1),('volume_sha256','other')]:
            changed=deepcopy(b);changed[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):align_pairs(a,changed)
        with self.assertRaises(ValueError):align_pairs(a,b[1:])
        with self.assertRaises(ValueError):align_pairs(a,b+[deepcopy(b[0])])

    def test_repeated_folds_and_single_cluster_are_rejected(self):
        a,b,ca,cb=pairs(2)
        for rows in (a,b):
            for i,row in enumerate(rows):row['fold_id']='fold'+str(i%2)
        with self.assertRaises(ValueError):align_pairs(a,b)
        with self.assertRaises(ValueError):paired_cluster_bootstrap(*pairs(1))

    def test_seed_determinism_and_parameter_guards(self):
        p=pairs(2);a=paired_cluster_bootstrap(*p,seed=113);b=paired_cluster_bootstrap(*p,seed=114)
        self.assertNotEqual(a['draw_sha256'],b['draw_sha256'])
        for kw in ({'replicates':999},{'replicates':True},{'seed':-1},{'confidence':True},{'confidence':1.}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):paired_cluster_bootstrap(*p,**kw)


class InterpretationAndBudget(unittest.TestCase):
    def test_noninferiority_requires_margin_and_strict_boundary(self):
        self.assertEqual(noninferiority_status([0,0])['status'],'not_assessed_margin_not_locked')
        self.assertEqual(noninferiority_status(None,margin=.03)['status'],'not_assessed_interval_unavailable')
        self.assertEqual(noninferiority_status([-.02,.01],margin=.03)['status'],'noninferior_by_supplied_rule')
        self.assertEqual(noninferiority_status([-.03,.01],margin=.03)['status'],'inconclusive')
        self.assertEqual(noninferiority_status([-.1,-.04],margin=.03)['status'],'inferior_by_supplied_rule')
        with self.assertRaises(ValueError):noninferiority_status([0,1],margin=0)

    def test_pareto_uses_all_costs_and_preserves_identical_peers(self):
        rows=[{'id':'A','quality':.5,'latency_p95_ms':10.,'operator_flops':100,'matched_profile_signature':'same'},
              {'id':'B','quality':.5,'latency_p95_ms':11.,'operator_flops':110,'matched_profile_signature':'same'},
              {'id':'C','quality':.6,'latency_p95_ms':12.,'operator_flops':120,'matched_profile_signature':'same'}]
        self.assertEqual(pareto_frontier(rows)['frontier_ids'],['A','C'])
        rows.append({**rows[0],'id':'D'})
        self.assertEqual(pareto_frontier(rows)['frontier_ids'],['A','C','D'])
        rows[-1]['matched_profile_signature']='different_hardware'
        with self.assertRaises(ValueError):pareto_frontier(rows)

    def test_variants_change_only_the_declared_component(self):
        _,v=paired_variants();full=v['FULL']
        for name,key in [('WITHOUT_CVAR','cvar'),('WITHOUT_GENERATOR','generator_variance')]:
            self.assertEqual(v[name]['model'],full['model'])
            expected=deepcopy(full['objective']);expected[key]=False
            self.assertEqual(v[name]['objective'],expected)
        for cap in (128,256):
            expected={**full['model'],'sequence_cap':cap}
            self.assertEqual(v['CAP'+str(cap)]['model'],expected)
            self.assertEqual(v['CAP'+str(cap)]['objective'],full['objective'])

    def test_no_medical_final_test_margin_or_underpowered_replicate_override(self):
        for old,new in [('allow_final_test = false','allow_final_test = true'),
                        ('read_medical_volumes = false','read_medical_volumes = true'),
                        ('replicates = 1000','replicates = 999'),
                        ('noninferiority_margin = "PENDING"','noninferiority_margin = 0.03')]:
            with TemporaryDirectory() as d:
                p=Path(d)/'bad.toml';p.write_text(CONFIG.read_text().replace(old,new))
                with self.assertRaises(ValueError):load_phase10_config(p)


class SyntheticIntegration(unittest.TestCase):
    def test_five_real_checkpoint_profiles_sealed_pairing_and_recomputation(self):
        with TemporaryDirectory() as d:
            root=Path(d);f08=root/'f08/orchestration';run_phase8(small_config(root),f08,device='cpu')
            config=root/'f10.toml'
            config.write_text('extends = ['+json.dumps(CONFIG.as_posix())+']\n[engineering]\nseeds=[13]\nsteps=1\n'
                '[profiling]\nwarmup=1\niterations=3\nstage_iterations=1\n')
            result=run_phase10(f08,config,root/'run',device='cpu')
            self.assertTrue(result['ok']);self.assertEqual(result['jobs_completed'],5)
            first=json.loads((root/'run/analysis/summary.json').read_text())
            with patch('torch.load',side_effect=AssertionError('Saved analysis must not deserialize models')):
                repeat=analyze_phase10(root/'run',root/'repeat')
            self.assertEqual(first,repeat)
            self.assertEqual(len(first['paired_contrasts']),4)
            self.assertFalse(first['research_inference_allowed'])
            _,jobs=read_phase10(root/'run')
            self.assertEqual(len({j['result']['initial_weights_sha256'] for j in jobs.values()}),1)
            self.assertEqual(len({j['profile']['matched_profile_signature'] for j in jobs.values()}),1)
            self.assertTrue(all(j['profile']['iterations']==3 and j['profile']['p95_ms']>0 for j in jobs.values()))
            self.assertTrue(all(j['profile']['compute']['operator_flops_estimate']['total']>0 for j in jobs.values()))
            with self.assertRaises(FileExistsError):run_phase10(f08,config,root/'run',device='cpu')
            (root/'run/jobs/FULL_s13/profile.json').write_text('{}')
            with self.assertRaises(ValueError):analyze_phase10(root/'run',root/'bad')


if __name__=='__main__':unittest.main()
