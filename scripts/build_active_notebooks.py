"""Build two readable entry points. User-requested rename: 17 November 2026.

17 Nov 2026 (later): 01_RSVT_TRAINING now prepares M3Dsynth and trains the
coarse-to-fine RSVT. The review notebook keeps its original release path and
frozen backend, so it rebuilds byte-identically.

24 Nov 2026: the user runs 01_RSVT_TRAINING online in Colab, opened from GitHub.
It clones the project code from GitHub at `CODE_REF` and saves the exact commit
with the session, instead of loading a code release copied to Drive.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from notebook_layout import ACTIVE_NOTEBOOKS

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'outputs/rsvt_training/release_20261117_human_readable'  # review notebook only
REPOSITORY = 'https://github.com/Richmiz/Risk-Aware-Sparse-Volumetric-Transformer-for-Tampered-Lung-CT-Detection.git'
REVIEW_ARCHIVE = ROOT/'notebooks/archive/20261117_human_readable/02_ANNOTATION_REVIEW.ipynb'
REVIEW_RUNTIME = ROOT/'scripts/notebook_support/annotation_review_runtime.py'
FROZEN_REVIEW_SHA = '4c04973de96823a01585ce5741b8cf2ce70689ec5328567593a5204678bea224'

# Clones the repository into the runtime. A token, when present, travels in git's
# environment, so it never appears in a command line, an error or an output.
CODE_CELL = (
    'import base64, importlib.util, os, subprocess, sys\n'
    'from pathlib import Path\n'
    'CODE_ROOT = Path("/content/rsvt-code")\n'
    'env = dict(os.environ, GIT_TERMINAL_PROMPT="0")\n'
    'try:\n'
    '    from google.colab import userdata\n'
    '    token = userdata.get("GITHUB_TOKEN")  # needed while the repository is private\n'
    'except Exception:\n'
    '    token = None\n'
    'if token:\n'
    '    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()\n'
    '    env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraHeader",\n'
    '               GIT_CONFIG_VALUE_0="Authorization: Basic " + basic)\n'
    '    del basic\n'
    'del token\n'
    '\n'
    'def git(*args):\n'
    '    done = subprocess.run(["git", *args], env=env, capture_output=True, text=True)\n'
    '    if done.returncode:\n'
    '        raise RuntimeError(f"git {args[0]} failed: {done.stderr.strip()[-300:]}")\n'
    '    return done.stdout.strip()\n'
    '\n'
    'if not (CODE_ROOT / ".git").exists():\n'
    '    git("clone", "--quiet", REPOSITORY, str(CODE_ROOT))\n'
    'git("-C", str(CODE_ROOT), "fetch", "--quiet", "--tags", "origin")\n'
    'commit = None\n'
    'for candidate in (f"origin/{CODE_REF}", CODE_REF):\n'
    '    try:\n'
    '        commit = git("-C", str(CODE_ROOT), "rev-parse", "--verify", "--quiet", candidate + "^{commit}")\n'
    '        break\n'
    '    except RuntimeError:\n'
    '        pass\n'
    'if commit is None:\n'
    '    raise RuntimeError(f"CODE_REF {CODE_REF!r} is not a branch, tag or commit of the repository")\n'
    'if "rsvt" in sys.modules and globals().get("RSVT_COMMIT") not in (None, commit):\n'
    '    raise RuntimeError("This session already loaded other code. Runtime > Restart session, then Run all.")\n'
    'git("-C", str(CODE_ROOT), "checkout", "--quiet", "--detach", commit)\n'
    'RSVT_COMMIT = commit\n'
    'for module, requirement in (("pydicom", "pydicom==3.0.2"), ("SimpleITK", "SimpleITK==2.5.7")):\n'
    '    if importlib.util.find_spec(module) is None:\n'
    '        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", requirement], check=True)\n'
    'if str(CODE_ROOT / "src") not in sys.path:\n'
    '    sys.path.insert(0, str(CODE_ROOT / "src"))\n'
    'print("Project code at commit", commit)\n')


def cell(kind, source, name):
    result={'cell_type':kind,'id':name,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code': result.update(execution_count=None,outputs=[])
    return result


def training_notebook():
    cells=[cell('markdown',
        '# RSVT — M3Dsynth Training and Results (coarse-to-fine)\n\n'
        'This notebook prepares the **M3Dsynth** dataset (manipulated LIDC-IDRI lung CT: nodule removal and insertion by '
        'CycleGAN, pix2pix and diffusion generators), checks it, trains the coarse-to-fine RSVT and shows its results.\n\n'
        '**Open this notebook in Google Colab from GitHub and choose a GPU runtime** (Runtime → Change runtime type). '
        'The project code is cloned from GitHub at `CODE_REF`, and the exact commit is saved with each session. '
        'While the repository is private, add a read-only GitHub token for it as the Colab secret `GITHUB_TOKEN` '
        '(key icon in the left sidebar) and allow notebook access. '
        'Every output saves automatically to project Drive and is read back against a checksum.\n\n'
        '**Data preparation downloads data only when you set `PREPARE_DATA = True`.** With the default patient caps it fetches '
        'about 260 LIDC-IDRI series from the public TCIA API (~30 GB) and streams the three author archives '
        '(cycle 171 GB, pix2pix 142 GB, diffusion 161 GB) from grip.unina.it. Only verified boxes around each edit are kept, '
        'so Drive storage stays near 15–20 GB. Streaming takes hours; it resumes where it stopped.\n\n'
        'Licences: M3Dsynth (GRIP-UNINA, non-profit research use with attribution) and LIDC-IDRI (TCIA, CC BY 3.0). '
        'Test patients are not downloaded or opened here.\n', 'introduction')]
    sections=[
        ('settings','1. Choose the run settings',
         'Why: keep the settings you may change in one place. `CODE_REF` chooses the code: a branch, a tag or a commit. '
         'To resume a session, paste its Drive folder into `RESUME_SESSION` and keep the same settings and code. '
         '`LIDC_ROOT` may point to an existing LIDC-IDRI download in Drive (NBIA folder layout); '
         'leave it empty to fetch series from TCIA.',
         'PROJECT = "/content/drive/MyDrive/Research/LUNG-CT-SCAN"\n'
         f'REPOSITORY = "{REPOSITORY}"\n'
         'CODE_REF = "main"         # branch, tag or commit; the exact commit is saved with the session\n'
         'PREPARE_DATA = False      # True = download/stream and verify M3Dsynth (resumable)\n'
         'RUN_DIAGNOSTICS = True    # resolution and shortcut checks before training\n'
         'RUN_TRAINING = True\n'
         'GENERATORS = ["cycle", "pix2pix", "diffusion"]  # remove one to hold it out\n'
         'PATIENTS_TRAIN = 200      # of 488 publisher training patients\n'
         'PATIENTS_VALIDATION = 60  # of 100 publisher validation patients\n'
         'EPOCHS = 30\nLEARNING_RATE = 0.0002\nEARLY_STOPPING_PATIENCE = 8\nSEED = 13\n'
         'LIDC_ROOT = ""            # optional existing LIDC-IDRI folder in Drive\n'
         'RESUME_SESSION = ""       # empty = a new session\n'),
        ('drive','2. Connect Google Drive',
         'Why: the dataset store and the results live here. Use the Google account containing Research/LUNG-CT-SCAN.',
         'from google.colab import drive\ndrive.mount("/content/drive", force_remount=False)\n'),
        ('code','3. Get the project code from GitHub',
         'Why: the implementation lives in the repository, not in long cells. This clones it into the runtime, checks out '
         '`CODE_REF` and installs the two missing packages. Expected output: the exact commit being run. If cloning fails '
         'while the repository is private, check the `GITHUB_TOKEN` secret and its notebook access.',
         CODE_CELL),
        ('session','4. Create a saved session',
         'Why: each run gets its own Drive folder with its settings and code commit. The M3Dsynth metadata tables are '
         'downloaded from the authors\' GitHub at a pinned commit and checked against their recorded checksums. '
         'Expected output: the session folder in Drive.',
         'from rsvt.training.m3d_notebook import create_session\n'
         'session = create_session(PROJECT, CODE_ROOT, epochs=EPOCHS, learning_rate=LEARNING_RATE,\n'
         '    patience=EARLY_STOPPING_PATIENCE, seed=SEED, patients_train=PATIENTS_TRAIN,\n'
         '    patients_validation=PATIENTS_VALIDATION, generators=tuple(GENERATORS),\n'
         '    lidc_root=LIDC_ROOT or None, resume_session=RESUME_SESSION, code_commit=RSVT_COMMIT)\n'
         'print("Session saved in Drive:", session.root)\n'),
        ('hardware','5. Check the actual runtime and GPU',
         'Why: a server label is not proof of a GPU. This checks what PyTorch sees. Data preparation runs on CPU; training needs CUDA.',
         'runtime_information = session.show_runtime()\n'),
        ('plan','6. Review the data plan before downloading',
         'Why: see patients, series, archive sizes and expected Drive storage before any transfer. Nothing is downloaded here.',
         'data_plan = session.plan_data()\n'),
        ('prepare','7. Prepare and verify M3Dsynth',
         'Why: build the dataset store. Each manipulated slice is compared with the real LIDC slice while it streams; a sample is '
         'kept only if every change lies inside its stored box and the author mask is complete. With `PREPARE_DATA=False` this '
         'cell only verifies an existing store. Expected output: progress, then a checksum-verified manifest.',
         'manifest = session.prepare_data(download=PREPARE_DATA)\n'),
        ('dataset','8. Inspect the dataset with real CT slices',
         'Why: confirm what the model will see. Expected output: counts per partition/generator/operation and training examples '
         'showing the authentic slice, the manipulated slice, their difference and the author mask. No test data is opened.',
         'if session.has_store():\n    dataset_overview = session.show_dataset()\n    preview_examples = session.show_previews()\n'),
        ('diagnostics','9. Pre-training checks: resolution and shortcuts',
         'Why: test the review findings on real data before spending GPU time. (a) A small CNN separates manipulated crops '
         'from authentic sites that look alike (removals vs healthy-tissue TB sites, insertions vs real-nodule TM sites) at '
         'native resolution and degraded to the 96³ voxel size; results are shown per operation. This tests the evidence at '
         'a known site, not whether the router can find it. (b) Archive identity outside the edit, mask coverage, and whether '
         'whole-scan statistics alone separate classes. Uses train/validation only.',
         'if RUN_DIAGNOSTICS and session.has_store():\n    diagnostics = session.run_diagnostics()\n'),
        ('configuration','10. Understand the training configuration',
         'Why: see batches, crops, loss and risk weights, early stopping and checkpoint selection before training.',
         'configuration_table = session.show_configuration()\n'),
        ('model','11. Build and inspect the coarse-to-fine RSVT',
         'Why: see parameters and compute. The 96³ view gives anatomy context and a router chooses K native-resolution crops; '
         'evidence is read only there. The table compares this with running the same encoder over the whole native volume.',
         'if session.has_store():\n    parameter_count = session.show_model()\n'),
        ('training','12. Train and watch the epoch results',
         'Why: learn from training volumes and monitor validation volumes. Expected output: batch progress, an updating epoch '
         'table and learning curves. The first epochs train without tail risk; the checkpoint is chosen among risk-stage epochs '
         'by the validation loss averaged over authentic, removal and insertion volumes. Checkpoints and logs save to Drive '
         'after every epoch; re-running resumes.',
         'if RUN_TRAINING and session.has_store():\n    training_summary = session.train()\nelse:\n'
         '    print("Training not run in this execution. No training result is claimed.")\n'),
        ('curves','13. Review the learning curves',
         'Why: compare training objective with validation loss and AUROCs. Falling training loss alone does not prove generalization.',
         'epoch_history = session.show_history()\n'),
        ('results','14. Inspect the selected model’s results',
         'Why: judge detection where it matters: removal (FB) and insertion (FM) against authentic scans, per generator, '
         'router recall (did the chosen crops contain the edit?) and mask overlap. Operating points are exploratory.',
         'selected_model_results = session.show_results()\n'),
        ('test','15. Understand the held-out test set',
         'Why: keep an independent final check. Test patients are not acquired or opened by this notebook.',
         'session.show_test_status()\n'),
        ('saved','16. Verify that the results are saved',
         'Why: confirm Drive storage by reading files back against checksums. Expected output: **Saved and verified** and the session path.',
         'persistence_receipt = session.verify_saved()\nprint("To resume this session:", session.root)\n')]
    for name,title,description,code in sections:
        cells.extend([cell('markdown','## '+title+'\n\n'+description+'\n',name+'-description'),cell('code',code,name)])
    return {'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'display_name':'Python 3 (Colab GPU)','language':'python','name':'python3'},
        'language_info':{'name':'python'}},'cells':cells}


def review_notebook():
    frozen=json.loads(REVIEW_ARCHIVE.read_text(encoding='utf-8'))
    code=[''.join(c['source']) for c in frozen['cells'] if c['cell_type']=='code']
    checksum=hashlib.sha256(REVIEW_RUNTIME.read_bytes()).hexdigest()
    wrapper=('if not WIDGET_RESTART_NEEDED:\n    from pathlib import Path\n    import hashlib\n'
        f"    script = Path('/content/drive/MyDrive/Research/LUNG-CT-SCAN/{RELEASE}/annotation_review_runtime.py')\n"
        f"    if hashlib.sha256(script.read_bytes()).hexdigest() != '{checksum}':\n"
        "        raise RuntimeError('Review script checksum mismatch; session not opened.')\n"
        "    exec(compile(script.read_text(encoding='utf-8'), str(script), 'exec'), globals())\n")
    status=("if WIDGET_RESTART_NEEDED:\n    SESSION = None\n    print('Restart Kernel, then Run All to open the existing review session.')\n"
            "else:\n    print('Existing review session:', SESSION.root)\n")
    items=[('Connect Drive','Read existing previews and save reviewer decisions in the project Drive.',code[0]),
           ('Check widget compatibility','Install the known compatible widget version. Restart only if the cell explicitly asks; saved decisions remain in Drive.',code[1]),
           ('Load the frozen review backend','The implementation is in a verified script. The original backend checksum and Drive session binding are preserved.',wrapper),
           ('Check the session','Resume the latest checksum-valid review decisions. This does not start a new review session.',status),
           ('Review and save each point','Choose a point/hypothesis, enter your reviewer name and evidence, then Save & Next. Training is in 01_RSVT_TRAINING.ipynb.',code[4])]
    document=deepcopy(frozen); document['cells']=[cell('markdown',
        '# Annotation Review — existing CT previews\n\nUse a Colab CPU kernel in VS Code. Review the 123 train/validation points; '
        '25 test points remain excluded. Your existing decisions and revision history stay in Drive. '
        'Use **01_RSVT_TRAINING.ipynb** for model training.\n','review-introduction')]
    for i,(title,description,source) in enumerate(items,1):
        document['cells'].extend([cell('markdown',f'## {i}. {title}\n\n{description}\n',f'review-{i}-description'),
                                  cell('code',source,f'review-{i}')])
    return document


def preserve(path):
    if not path.exists(): return
    folder=ROOT/'notebooks/archive'/('before_rebuild_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    folder.mkdir(parents=True)
    target=folder/path.name; target.write_bytes(path.read_bytes())
    if target.read_bytes()!=path.read_bytes(): raise OSError('Notebook archive readback failed')


def build():
    documents=(training_notebook(),review_notebook()); result={}
    for name,document in zip(ACTIVE_NOTEBOOKS,documents):
        target=ROOT/'notebooks'/name; data=(json.dumps(document,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
        if target.exists() and target.read_bytes()!=data: preserve(target)
        target.write_bytes(data)
        if target.read_bytes()!=data: raise OSError('Notebook readback failed')
        result[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    retired=ROOT/'notebooks/01_DATA_AUDIT.ipynb'
    if retired.exists():
        folder=ROOT/'notebooks/archive'/('retired_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
        folder.mkdir(parents=True); target=folder/retired.name
        if not target.resolve().is_relative_to((ROOT/'notebooks/archive').resolve()): raise ValueError('Unsafe archive target')
        retired.rename(target)
    if {p.name for p in (ROOT/'notebooks').glob('*.ipynb')}!=set(ACTIVE_NOTEBOOKS): raise ValueError('Expected exactly two active notebooks')
    return result


if __name__=='__main__':
    print(json.dumps({'notebooks':build(),'training_code_source':REPOSITORY,'review_release_unchanged':RELEASE,
                      'review_runtime_sha256':hashlib.sha256(REVIEW_RUNTIME.read_bytes()).hexdigest(),
                      'frozen_review_backend_sha256':FROZEN_REVIEW_SHA},indent=2))
