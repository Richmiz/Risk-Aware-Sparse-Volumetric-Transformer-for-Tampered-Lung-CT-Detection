"""Build two readable entry points. User-requested rename: 17 November 2026.

17 Nov 2026 (later): 01_RSVT_TRAINING now prepares M3Dsynth and trains the
coarse-to-fine RSVT. The review notebook keeps its original release path and
frozen backend, so it rebuilds byte-identically.

24 Nov 2026: the user runs 01_RSVT_TRAINING online in Colab, opened from GitHub.
It clones the project code from GitHub at `CODE_REF` and saves the exact commit
with the session, instead of loading a code release copied to Drive.

26 Nov 2026: `RUN` chooses between the crop learning check (section 12) and training,
after the first training run stayed at chance.

26 Nov 2026 (later): the check passed on residual crops, so `RUN` defaults to training
again, now with a residual crop input, AUROC-based checkpoint selection and a
validation score with the edit's crop added.

27 Nov 2026: the second training run learned nothing, so `RUN` defaults to the crop
pretraining check (section 12): pretrain the crop network, verify the transfer into the
full model, and swap-test the scan classifier before a third run.

27 Nov 2026 (later): `RECHECK_DATASET = False` (user request) skips section 7's
hour-long full read-back of the saved dataset when an earlier run already read the
same store back in full; every file a run opens is still checked when loaded.
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

# Clones the public repository into the runtime. No sign-in is needed; git never
# prompts, so a failed clone stops with its error instead of waiting for input.
CODE_CELL = (
    'import importlib.util, json, os, subprocess, sys\n'
    'from pathlib import Path\n'
    'CODE_ROOT = Path("/content/rsvt-code")\n'
    'env = dict(os.environ, GIT_TERMINAL_PROMPT="0")\n'
    '\n'
    '# A resumed run keeps the code commit it started with. Only training runs are continued.\n'
    'RESUME_FOLDER, ref = "", CODE_REF\n'
    'if RESUME_SESSION == "auto":\n'
    '    runs = Path(PROJECT, "outputs/rsvt_training").glob("c2f_*/notebook_session.json") if RUN == "train" else []\n'
    '    for record in sorted(runs, reverse=True):\n'
    '        run = record.parent\n'
    '        if (run / "training/last.pt").exists() and not (run / "training/summary.json").exists():\n'
    '            RESUME_FOLDER = str(run)\n'
    '            break\n'
    'elif RESUME_SESSION:\n'
    '    RESUME_FOLDER = RESUME_SESSION\n'
    'if RESUME_FOLDER:\n'
    '    record = json.loads(Path(RESUME_FOLDER, "notebook_session.json").read_text(encoding="utf-8"))\n'
    '    ref = record.get("code_commit") or CODE_REF\n'
    '    print("Continuing the unfinished run", RESUME_FOLDER)\n'
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
    'for candidate in (f"origin/{ref}", ref):\n'
    '    try:\n'
    '        commit = git("-C", str(CODE_ROOT), "rev-parse", "--verify", "--quiet", candidate + "^{commit}")\n'
    '        break\n'
    '    except RuntimeError:\n'
    '        pass\n'
    'if commit is None:\n'
    '    raise RuntimeError(f"{ref!r} is not a branch, tag or commit of the repository")\n'
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


def colab_url(notebook):
    """Open-in-Colab address of a notebook on the main branch of the public repository."""
    return ('https://colab.research.google.com/github/' + REPOSITORY.removeprefix('https://github.com/').removesuffix('.git')
            + '/blob/main/notebooks/' + notebook)


def training_notebook():
    cells=[cell('markdown',
        f'<a href="{colab_url("01_RSVT_TRAINING.ipynb")}" target="_parent">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>\n','colab-link'),
        cell('markdown',
        '# RSVT — M3Dsynth Training and Results (coarse-to-fine)\n\n'
        'This notebook prepares the **M3Dsynth** dataset (manipulated LIDC-IDRI lung CT: nodule removal and insertion by '
        'CycleGAN, pix2pix and diffusion generators), checks it, trains the coarse-to-fine RSVT and shows its results.\n\n'
        '**Just Run all; the defaults are ready.** Each run decides what to do from what is already saved in Drive:\n\n'
        '1. **First, on a CPU runtime** (cheapest): it downloads and verifies the dataset, runs the data checks and stops '
        'before training. This takes hours; keep the tab open.\n'
        '2. **Then on a GPU runtime** (Runtime → Change runtime type → L4): it opens the saved dataset (read back in full '
        'once; later runs skip that unless `RECHECK_DATASET = True`), reuses the checks and then does what `RUN` says. The default, `RUN = "pretrain_check"`, runs two short checks before the '
        'next training run (roughly 1–1½ hours, no training); `RUN = "train"` trains the model; `RUN = "crop_check"` '
        'repeats the crop learning check of 26 November.\n'
        '3. **After a disconnect**, Run all again: it continues the unfinished training run with the code it started with.\n\n'
        'When a run finishes and every saved file has been read back from Drive, the last cell releases the runtime so it '
        'stops using compute units; the outputs stay visible.\n\n'
        'The project code is cloned from GitHub, and the exact commit is saved with each run. '
        'Every output saves automatically to project Drive and is read back against a checksum.\n\n'
        'With the default patient caps the download is about 300 LIDC-IDRI series from the public TCIA API (~33 GB) plus '
        'one pass over the three author archives (cycle 171 GB, pix2pix 142 GB, diffusion 161 GB) from grip.unina.it. '
        'Only verified boxes around each edit are kept, so Drive storage stays near 25 GB (boxes kept from an earlier rule come on top). '
        'A dropped connection reconnects at the same byte; if the runtime restarts, finished archives and samples are kept '
        'and only the current archive is read again.\n\n'
        'Licences: M3Dsynth (GRIP-UNINA, non-profit research use with attribution) and LIDC-IDRI (TCIA, CC BY 3.0). '
        'Test patients are not downloaded or opened here.\n', 'introduction')]
    sections=[
        ('settings','1. Choose the run settings',
         'Why: everything you might change is here, and the defaults are ready to run. The notebook prepares data and '
         'checks it according to what is saved and whether a GPU is present; on a GPU, `RUN` chooses between a check '
         'before training (section 12, no training) and training (section 13). With `RUN = "train"`, '
         '`RESUME_SESSION = "auto"` continues the latest unfinished training run; changing any setting below needs a new '
         'run, so then set `RESUME_SESSION = ""`. `CODE_REF` chooses the code for a new run (a branch, tag or commit). '
         '`LIDC_ROOT` may point to an existing LIDC-IDRI download in Drive (NBIA folder layout). '
         '`RECHECK_DATASET = False` skips re-reading the whole saved dataset (about an hour) when an earlier run already '
         'read it back in full; set it to `True` to read it all again.',
         'PROJECT = "/content/drive/MyDrive/Research/LUNG-CT-SCAN"\n'
         f'REPOSITORY = "{REPOSITORY}"\n'
         'RUN = "pretrain_check"     # a check, no training: "pretrain_check" or "crop_check"; "train" = train\n'
         'CODE_REF = "main"          # code for a new run; a resumed run keeps the commit it started with\n'
         'RESUME_SESSION = "auto"    # "auto" = continue the latest unfinished run; "" = new run; or a run folder\n'
         'GENERATORS = ["cycle", "pix2pix", "diffusion"]  # remove one to hold it out\n'
         'PATIENTS_TRAIN = 200       # of 488 publisher training patients\n'
         'PATIENTS_VALIDATION = 100  # all publisher validation patients\n'
         'EPOCHS = 30\nLEARNING_RATE = 0.0002\nEARLY_STOPPING_PATIENCE = 8\nSEED = 13\n'
         'LIDC_ROOT = ""             # optional existing LIDC-IDRI folder in Drive\n'
         'RECHECK_DATASET = False    # True = re-read the whole saved dataset against its checksums (about 1 hour)\n'
         'SHUTDOWN_WHEN_DONE = True  # release the runtime after the final save check, so it stops using compute units\n'),
        ('drive','2. Connect Google Drive',
         'Why: the dataset store and the results live here. Use the Google account containing Research/LUNG-CT-SCAN.',
         'from google.colab import drive\ndrive.mount("/content/drive", force_remount=False)\n'),
        ('code','3. Get the project code from GitHub',
         'Why: the implementation lives in the repository, not in long cells. This finds the run to continue (if any), clones '
         'the repository into the runtime, checks out its code and installs the two missing packages. The repository is '
         'public, so no sign-in is needed. Expected output: the exact commit being run.',
         CODE_CELL),
        ('session','4. Open the run folder',
         'Why: each run has its own Drive folder with its settings and code commit; an unfinished run is reopened. The '
         'M3Dsynth metadata tables are downloaded from the authors\' GitHub at a pinned commit and checked against their '
         'recorded checksums. Expected output: the run folder in Drive.',
         'from rsvt.training.m3d_notebook import create_session\n'
         'session = create_session(PROJECT, CODE_ROOT, epochs=EPOCHS, learning_rate=LEARNING_RATE,\n'
         '    patience=EARLY_STOPPING_PATIENCE, seed=SEED, patients_train=PATIENTS_TRAIN,\n'
         '    patients_validation=PATIENTS_VALIDATION, generators=tuple(GENERATORS),\n'
         '    lidc_root=LIDC_ROOT or None, resume_session=RESUME_FOLDER, code_commit=RSVT_COMMIT, run=RUN,\n'
         '    recheck_dataset=RECHECK_DATASET)\n'
         'print("Run folder in Drive:", session.root)\n'),
        ('hardware','5. Check the actual runtime and GPU',
         'Why: a server label is not proof of a GPU. This checks what PyTorch sees. Data preparation and the data checks '
         'run on CPU; training needs a GPU.',
         'runtime_information = session.show_runtime()\n'),
        ('plan','6. Review the data plan',
         'Why: see patients, series, archive sizes, Drive storage, the state of the saved dataset and free space before any '
         'transfer. Nothing is downloaded here.',
         'data_plan = session.plan_data()\n'),
        ('prepare','7. Prepare and verify M3Dsynth',
         'Why: build the dataset store, or confirm the saved one. It downloads only what the saved dataset is missing (on '
         'the first run, everything). Otherwise it reads the saved store back against its checksums, unless '
         '`RECHECK_DATASET = False` and an earlier run already read this same store back in full; then it skips that hour '
         'and each file is still checked against its checksum when the run loads it. It checks free space '
         'first and stops before downloading if Google Drive or the runtime disk is too small. Each manipulated slice is '
         'compared with the real LIDC slice while it streams; a sample is kept only if every change lies inside its stored '
         'box and the author mask is complete. Expected output: progress, then a checksum-verified manifest.',
         'manifest = session.prepare_data()\n'),
        ('dataset','8. Inspect the dataset with real CT slices',
         'Why: confirm what the model will see. Expected output: counts per partition/generator/operation and training examples '
         'showing the authentic slice, the manipulated slice, their difference and the changed voxels. No test data is opened.',
         'if session.has_store():\n    dataset_overview = session.show_dataset()\n    preview_examples = session.show_previews()\n'),
        ('diagnostics','9. Pre-training checks: resolution and shortcuts',
         'Why: test the review findings on real data before spending GPU time. (a) A small CNN separates manipulated crops '
         'from authentic sites that look alike (removals vs healthy-tissue TB sites, insertions vs real-nodule TM sites) at '
         'native resolution and degraded to the 96³ voxel size; results are shown per operation. This tests the evidence at '
         'a known site, not whether the router can find it. (b) Archive identity outside the edit, mask coverage, and whether '
         'whole-scan statistics alone separate classes. Uses train/validation only. The checks run once per dataset and are '
         'reused afterwards. Training is held back if most samples failed archive verification or if whole-scan statistics '
         'alone separate the classes.',
         'diagnostics = session.run_diagnostics() if session.has_store() else None\n'),
        ('configuration','10. Understand the training configuration',
         'Why: see batches, crops, loss and risk weights, early stopping and checkpoint selection before training.',
         'configuration_table = session.show_configuration()\n'),
        ('model','11. Build and inspect the coarse-to-fine RSVT',
         'Why: see parameters and compute. The 96³ view gives anatomy context and a router chooses K native-resolution crops; '
         'evidence is read only there. The table compares this with running the same encoder over the whole native volume.',
         'if session.has_store():\n    parameter_count = session.show_model()\n'),
        ('crop_check','12. Check the crop network before training',
         'Why: the second training run (26 Nov 2026) learned nothing. Its crop network never beat a constant guess, and '
         'adding the edit\'s crop to a scan changed its score by at most 0.0007. With `RUN = "pretrain_check"` (the '
         'default) this runs two short checks before a third run. (1) It trains the model\'s crop network alone on up to '
         '384 training pairs, each an edited crop and the same crop of its unedited scan, cut exactly as the model cuts its '
         'crops. It then loads the weights into a fresh model and checks that the same validation crops (up to 192, unseen '
         'patients) get the same scores. (2) It freezes the crop network, trains the rest of the model briefly, and scores '
         'each unseen edited scan twice, with one crop swapped between edited and unedited: does the scan score rise with '
         'the edited crop? Roughly 1–1½ hours on an L4. `RUN = "crop_check"` repeats the crop learning check of 26 Nov '
         '(residual crops: AUROC 0.94 for removal, 0.75 for insertion on unseen patients). Expected output: a table, curves '
         'and a one-line conclusion. No test data is opened.',
         'check = session.run_check() if session.has_store() else None\n'),
        ('training','13. Train and watch the epoch results',
         'Why: learn from training volumes and monitor validation volumes. On a GPU runtime with `RUN = "train"` this '
         'trains, or continues the unfinished run; otherwise it stops here with a message. Expected output: batch progress, an updating epoch '
         'table and learning curves. The crop network reads each crop minus its local mean (the residual), as in section 12. '
         'The first epochs train without tail risk; the checkpoint is chosen among risk-stage epochs by the mean of removal '
         'and insertion AUROC against authentic validation scans, and only if it beats chance (0.5), so a model that gives '
         'every scan the same score is never chosen. Each validation scan is also scored with the edit\'s crop added; that '
         'column shows what crop selection loses and is not used to choose the checkpoint. Where a volume was edited is taken '
         'from its changed voxels (manipulated vs real), not the author mask, which misses most of each removal. Checkpoints '
         'and logs save to Drive after every epoch.',
         'training_summary = session.train_when_ready(diagnostics) if session.has_store() else None\n'),
        ('curves','14. Review the learning curves',
         'Why: compare training objective with validation loss and AUROCs. Falling training loss alone does not prove generalization.',
         'epoch_history = session.show_history()\n'),
        ('results','15. Inspect the selected model’s results',
         'Why: judge detection where it matters: removal (FB) and insertion (FM) against authentic scans, per generator, '
         'router recall (did the chosen crops contain the edit?), the score with the edit\'s crop added (what routing loses) '
         'and overlap with the changed voxels, then how those labels compare with the author mask. Operating points are '
         'exploratory.',
         'selected_model_results = session.show_results()\n'),
        ('test','16. Understand the held-out test set',
         'Why: keep an independent final check. Test patients are not acquired or opened by this notebook.',
         'session.show_test_status()\n'),
        ('saved','17. Verify that the results are saved',
         'Why: confirm Drive storage by reading files back against checksums. Expected output: **Saved and verified** and the session path.',
         'persistence_receipt = session.verify_saved()\nprint("Run folder:", session.root)\n'),
        ('shutdown','18. Release the runtime',
         'Why: an idle runtime keeps using compute units until Colab disconnects it. This runs only after section 17 has '
         'read every saved file back, so nothing is lost; the outputs above stay visible. If an earlier cell failed, Run all '
         'stopped there and the runtime is kept so you can see the error. Set `SHUTDOWN_WHEN_DONE = False` to keep it.',
         'if SHUTDOWN_WHEN_DONE:\n'
         '    try:\n'
         '        from google.colab import runtime\n'
         '    except ImportError:\n'
         '        print("Not running in Colab; nothing to release.")\n'
         '    else:\n'
         '        print("Everything is saved and verified in Drive. Releasing the runtime now.")\n'
         '        runtime.unassign()\n'
         'else:\n'
         '    print("Runtime kept (SHUTDOWN_WHEN_DONE = False). Disconnect it when done to stop using compute units.")\n')]
    for name,title,description,code in sections:
        cells.extend([cell('markdown','## '+title+'\n\n'+description+'\n',name+'-description'),cell('code',code,name)])
    return {'nbformat':4,'nbformat_minor':5,'metadata':{
        'colab':{'provenance':[],'include_colab_link':True},
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
