"""Execute dependency/startup guards without Colab, pip, images or Drive writes."""

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import importlib.metadata

ROOT=Path(__file__).resolve().parents[1]
# The review notebook and its frozen backend stay in the local working copy only.
LOCAL_REVIEW_FILES=(ROOT/'notebooks/02_ANNOTATION_REVIEW.ipynb',
                    ROOT/'scripts/notebook_support/annotation_review_runtime.py')


@unittest.skipUnless(all(p.is_file() for p in LOCAL_REVIEW_FILES),
                     'local annotation-review notebook and frozen backend are not present')
class CompatibilityStartup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]
        document=json.loads((cls.root/'notebooks/02_ANNOTATION_REVIEW.ipynb').read_text(encoding='utf-8'))
        cls.code=[''.join(c['source']) for c in document['cells'] if c['cell_type']=='code']

    def execute_dependency(self,installed,loaded):
        state={'installed':installed}
        def version(name):
            if state['installed'] is None:raise importlib.metadata.PackageNotFoundError(name)
            return state['installed']
        def install(args,**kwargs):
            self.assertIn('ipywidgets==8.1.8',args)
            state['installed']='8.1.8'
        module=SimpleNamespace(__version__=loaded) if loaded is not None else None
        namespace={}
        with patch.dict(sys.modules,{'ipywidgets':module}),patch('importlib.metadata.version',side_effect=version),\
                patch('subprocess.run',side_effect=install) as pip,patch('builtins.print'):
            exec(self.code[1],namespace)
        return namespace,pip.call_count

    def test_compatible_kernel_does_not_install_or_require_restart(self):
        state,calls=self.execute_dependency('8.1.8','8.1.8')
        self.assertFalse(state['WIDGET_RESTART_NEEDED']);self.assertEqual(calls,0)

    def test_loaded_incompatible_kernel_skips_session_and_ui_until_restart(self):
        state,calls=self.execute_dependency('8.1.9','8.1.9')
        self.assertTrue(state['WIDGET_RESTART_NEEDED']);self.assertEqual(calls,1)
        with patch('builtins.print'):
            exec(self.code[3],state)
            exec(self.code[4],state)
        self.assertIsNone(state['SESSION'])
        self.assertNotIn('CODE_ROOT',state)
        self.assertNotIn('REVIEW_FORM',state)

    def test_unloaded_kernel_installs_compatible_version_without_restart(self):
        state,calls=self.execute_dependency('8.1.9',None)
        self.assertFalse(state['WIDGET_RESTART_NEEDED']);self.assertEqual(calls,1)

    def test_missing_widget_package_can_be_installed(self):
        state,calls=self.execute_dependency(None,None)
        self.assertFalse(state['WIDGET_RESTART_NEEDED']);self.assertEqual(calls,1)

    def test_notebook_compiles_and_preserves_executed_backend_payload(self):
        for source in self.code:ast.parse(source)
        # The exact frozen payload moved into a script at the user's request.
        # Executing with restart pending decodes code only, without opening data.
        state={'WIDGET_RESTART_NEEDED':True}
        script=self.root/'scripts/notebook_support/annotation_review_runtime.py'
        with patch('builtins.print'): exec(script.read_text(encoding='utf-8'),state)
        self.assertEqual(hashlib.sha256(state['PACKAGE_BYTES']).hexdigest(),
            '4c04973de96823a01585ce5741b8cf2ce70689ec5328567593a5204678bea224')
        self.assertNotIn('PACKAGE_BYTES','\n'.join(self.code))
        self.assertIn(hashlib.sha256(script.read_bytes()).hexdigest(),self.code[2])


if __name__=='__main__':unittest.main()
