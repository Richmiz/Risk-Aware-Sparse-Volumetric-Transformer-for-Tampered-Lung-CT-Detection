"""Load the verified RSVT coarse-to-fine runtime from Drive. No download or training here."""

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import zipfile

EXPECTED_SOURCE_SHA = '__SOURCE_SHA256__'
RELEASE = '__RELEASE__'
ARCHIVE = '__ARCHIVE__'
project = Path(globals().get('PROJECT', '/content/drive/MyDrive/Research/LUNG-CT-SCAN'))
archive_path = project/RELEASE/ARCHIVE
if not archive_path.is_file():
    raise FileNotFoundError('Mount the project Drive and copy the release folder: ' + str(archive_path))
if hashlib.sha256(archive_path.read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA:
    raise RuntimeError('Runtime source checksum mismatch. No code was loaded.')
requirements = {'torch': 'torch>=2.6,<3', 'numpy': 'numpy>=1.26,<3', 'pydicom': 'pydicom==3.0.2',
                'SimpleITK': 'SimpleITK==2.5.7', 'matplotlib': 'matplotlib>=3.8,<4', 'PIL': 'Pillow>=10'}
for module, requirement in requirements.items():
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', requirement], check=True)
CODE_ROOT = Path('/content/rsvt-notebook-runtime')/EXPECTED_SOURCE_SHA[:12]
CODE_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive_path) as package:
    for member in package.infolist():
        if not (CODE_ROOT/member.filename).resolve().is_relative_to(CODE_ROOT.resolve()):
            raise ValueError('Unsafe runtime archive path')
    package.extractall(CODE_ROOT)
    for member in package.infolist():
        if not member.is_dir() and (CODE_ROOT/member.filename).read_bytes() != package.read(member):
            raise OSError('Extracted runtime differs from its verified archive')
existing = sys.modules.get('rsvt')
if existing is not None and not Path(existing.__file__).resolve().is_relative_to(CODE_ROOT.resolve()):
    raise RuntimeError('This kernel loaded an earlier project runtime. Restart Kernel, then Run All; saved Drive sessions remain intact.')
sys.path.insert(0, str(CODE_ROOT/'src'))
from rsvt.training.m3d_notebook import create_session as _create_session


def create_session(**settings):
    return _create_session(project, CODE_ROOT, **settings)
