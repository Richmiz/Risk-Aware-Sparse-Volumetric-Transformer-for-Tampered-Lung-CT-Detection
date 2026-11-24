"""Readable UI, real-loader preview and storage tests on procedural CT fixtures."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/matplotlib_test_cache'))
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0,str(ROOT/'src'))
from rsvt.training.notebook import NotebookSession
from rsvt.data.dataset import CTVolumeDataset
import test_f04_training as fixtures
from test_proposed_real_training import tiny_config


class ReadableSessionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.Phase4Integration(); self.fixture.setUp()
        self.root=self.fixture.root; self.f03=self.fixture.f03
        self.project=self.root/'project'; self.project.mkdir()
        self.config=tiny_config()
        document=json.loads((self.f03/'manifest.draft.json').read_text())
        self.config['data'].update(expected_assignment_sha256=document['assignments_sha256'],shape_zyx=[32]*3)
        self.config['model'].update(sequence_cap=24)
    def tearDown(self): self.fixture.tearDown()
    def session(self,**extra):
        return NotebookSession(self.project,ROOT,epochs=3,learning_rate=1e-4,patience=10,
            configuration=self.config,f03_path=self.f03,enforce_drive=False,display_enabled=False,**extra)

    def test_previews_use_real_loader_only_train_and_save_verified_figures(self):
        session=self.session(); observed=[]; original=CTVolumeDataset.__getitem__
        def tracked(dataset,index):
            observed.append(dataset.rows[index]['split']); return original(dataset,index)
        with patch('rsvt.training.notebook.CTVolumeDataset.__getitem__',tracked):
            previews=session.show_previews()
        self.assertEqual(len(previews),4); self.assertEqual(set(observed),{'train'})
        self.assertEqual({p['label'] for p in previews},{'TB','TM','FB','FM'})
        self.assertTrue((session.assets/'dataset_preview.png').is_file())
        receipt=session.verify_saved()
        self.assertEqual(receipt['files']['figures/dataset_preview.png']['sha256'],
                         hashlib.sha256((session.assets/'dataset_preview.png').read_bytes()).hexdigest())

    def test_two_image_preview_and_no_fake_metrics_before_training(self):
        session=self.session(preview_images=2)
        self.assertEqual(len(session.show_previews()),2)
        self.assertEqual(session.show_history(),[]); self.assertIsNone(session.show_results())
        self.assertFalse((session.root/'results_overview.json').exists())

    def test_cpu_block_is_saved_without_starting_training(self):
        session=self.session(); session.config['training']['device']='cuda'
        with patch('torch.cuda.is_available',return_value=False),patch('rsvt.training.notebook.run_proposed') as train:
            with self.assertRaisesRegex(RuntimeError,'GPU not detected'): session.train()
            train.assert_not_called()
        self.assertEqual(json.loads((session.root/'status.json').read_text())['status'],'blocked_no_gpu')

    def test_training_reports_same_checkpoint_and_reserves_test_and_resumes(self):
        session=self.session(); session.show_dataset(); session.show_configuration(); session.show_model()
        result=session.train(); self.assertTrue(result['training_finished'])
        session.show_results(); receipt=session.verify_saved()
        self.assertTrue((session.assets/'learning_curves.png').is_file())
        self.assertTrue((session.assets/'validation_confusion_matrix.png').is_file())
        overview=json.loads((session.root/'results_overview.json').read_text())
        self.assertEqual(overview['checkpoint_epoch'],3)
        self.assertEqual(overview['test']['images_opened'],0)
        self.assertEqual(overview['test']['status'],'not_evaluated_reserved')
        epoch=json.loads((session.training_dir/'epoch_003.json').read_text())
        self.assertEqual(overview['train'],epoch['training_metrics_hard'])
        resumed=self.session(resume_session=str(session.root))
        with patch('rsvt.training.notebook.run_proposed',side_effect=AssertionError('must not retrain')):
            self.assertTrue(resumed.train()['training_finished'])
        with patch('rsvt.training.notebook._json',side_effect=OSError('Drive failed')):
            with self.assertRaises(OSError): resumed.verify_saved()

    def test_resume_rejects_changed_configuration(self):
        session=self.session(); self.config['training']['dropout_unrelated']='invalid example'
        with self.assertRaisesRegex(ValueError,'Resume requires'):
            self.session(resume_session=str(session.root))

    def test_manifest_checksum_failure_blocks_previews(self):
        path=self.f03/'manifest.draft.json'; path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'pinned receipt'): self.session()


class NotebookSourceTests(unittest.TestCase):
    def test_two_notebooks_have_no_embedded_archives_and_explain_every_code_cell(self):
        paths=list((ROOT/'notebooks').glob('*.ipynb'))
        expected={'01_RSVT_TRAINING.ipynb'}
        # The review notebook is local-only; a working copy with its frozen backend must keep it.
        if (ROOT/'scripts/notebook_support/annotation_review_runtime.py').is_file():
            expected.add('02_ANNOTATION_REVIEW.ipynb')
        self.assertEqual({p.name for p in paths},expected)
        for path in paths:
            doc=json.loads(path.read_text(encoding='utf-8'))
            for i,cell in enumerate(doc['cells']):
                source=''.join(cell['source'])
                self.assertNotIn('PACKAGE_BYTES',source); self.assertNotIn('EMBEDDED_PACKAGE',source)
                if cell['cell_type']=='code':
                    self.assertEqual(doc['cells'][i-1]['cell_type'],'markdown')
                    self.assertLess(len(source),3000)
                    compile(source,str(path),'exec')
            self.assertLess(path.stat().st_size,30000)


if __name__=='__main__': unittest.main()
