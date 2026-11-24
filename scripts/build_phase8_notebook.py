"""Generate F08 orchestration cells, retaining earlier executed notebooks."""

from copy import deepcopy
import json
from pathlib import Path
from textwrap import dedent
from notebook_layout import legacy_input, legacy_output

ROOT=Path(__file__).resolve().parents[1]


def build():
    previous=json.loads(legacy_input(ROOT,'07_phase7_colab.ipynb').read_text(encoding='utf-8'));cells=[]
    def cell(kind,source):
        item={'cell_type':kind,'metadata':{},'source':dedent(source).strip().splitlines(keepends=True)}
        if kind=='code':item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','''\
    # F08 - Orkestrasi eksperimen dan audit kesiapan protokol

    Jalankan sendiri dengan **Google Colab melalui VS Code**, CPU/GPU otomatis.
    Katalog memuat B0-B5 dan E1-E7. Lima adapter sintetis B0/B1/B3/B4/B5 dapat
    menjalankan train, checkpoint selection, inference dan evaluasi validation.
    B2 cascade serta pasangan ablation tetap diberi status belum tersedia.

    F08 membuat **engineering draft snapshot**, bukan final research protocol lock.
    Alpha, FPR, margin noninferiority, budget dan QA penelitian belum ditetapkan.
    Notebook membaca metadata manifest F03 untuk audit; volume medis dan test
    penelitian tidak dibuka. Seluruh training/checkpoint baru adalah sintetis.
    ''')
    for index in (1,2,3):
        item=deepcopy(previous['cells'][index]);item['metadata']={};item.pop('attachments',None)
        item['source']=[s.replace('F07','F08') for s in item['source']]
        if item['cell_type']=='code':item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','''\
    ## Konfigurasi, bukti F07 dan manifest metadata F03

    Akun Drive harus memiliki `MyDrive/Research/LUNG-CT-SCAN`. F07 tidak perlu
    diulang. Default membuat run baru. Jika run F08 terputus, isi
    `RESUME_RUN_OVERRIDE` dengan path root run F08 yang sama lalu ulangi Run All.
    Session log baru dibuat; completed jobs diverifikasi dan dilewati. Partial
    attempt dipertahankan dan job itu dimulai ulang dari seed pada attempt baru.
    ''')
    cell('code','''\
    from datetime import datetime, timezone
    import hashlib
    import json

    STATE_ROOT = Path("/content/drive/MyDrive/Research/LUNG-CT-SCAN")
    CODE_ARCHIVE = STATE_ROOT / "rsvt-colab-f08-code.zip"
    F07_RUN = STATE_ROOT / "outputs/engineering/f07_20261111T100301_095228Z"
    F03_MANIFEST = STATE_ROOT / "outputs/partitions/f03_20261110T013617_031412Z/manifest.draft.json"
    DEVICE = "auto"
    RESUME_RUN_OVERRIDE = None
    if DEVICE not in {"auto", "cpu", "cuda"}:
        raise ValueError("DEVICE tidak dikenal.")
    for path in (CODE_ARCHIVE, F07_RUN / "risk/summary.json", F07_RUN / "notebook_receipt.json", F03_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(f"Kode/prasyarat metadata belum ditemukan pada akun/path Drive ini: {path}")
    f07 = json.loads((F07_RUN / "risk/summary.json").read_text())
    f07_receipt = json.loads((F07_RUN / "notebook_receipt.json").read_text())
    if (f07.get("ok") is not True or len(f07.get("checks", {})) != 13 or not all(f07["checks"].values())
            or f07.get("rsvt_python_source_sha256") != "d225d58e19b1e31af26ad927b4a84db9e261dca2470473c49186b85f3e8fbb60"
            or f07_receipt.get("bundle_sha256") != "6537efdadddf577f1625eaab9eb18358e8849073b942a4cfc17959eb869ff48e"
            or f07_receipt.get("synthetic_tests_exit_code") != 0 or f07_receipt.get("risk_exit_code") != 0):
        raise RuntimeError("Bukti F07 berbeda dari run engineering yang sudah direview.")
    F03_MANIFEST_SHA = hashlib.sha256(F03_MANIFEST.read_bytes()).hexdigest()
    if F03_MANIFEST_SHA != "76b75d0235bd565b2e1a18f1099d6af1b6b181b81792621632fbe54e8d4017ff":
        raise RuntimeError("Manifest metadata F03 berbeda dari draft yang direview; jangan mengganti flag QA otomatis.")
    BUNDLE_SHA256 = hashlib.sha256(CODE_ARCHIVE.read_bytes()).hexdigest()
    CODE_ROOT = Path("/content/rsvt-f08") / BUNDLE_SHA256[:12]
    RUN_ID = datetime.now(timezone.utc).strftime("f08_%Y%m%dT%H%M%S_%fZ")
    RUN_ROOT = Path(RESUME_RUN_OVERRIDE) if RESUME_RUN_OVERRIDE is not None else STATE_ROOT / "outputs/engineering" / RUN_ID
    IS_RESUME = RESUME_RUN_OVERRIDE is not None
    SESSION_ID = datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%S_%fZ")
    SESSION_ROOT = RUN_ROOT / "sessions" / SESSION_ID
    if IS_RESUME and not (RUN_ROOT / "orchestration/plan.lock.json").is_file():
        raise FileNotFoundError("Resume memerlukan root run F08 yang memiliki orchestration/plan.lock.json.")
    print("Paket F08:", BUNDLE_SHA256)
    print("F07 direview:", F07_RUN)
    print("Manifest F03 (metadata saja):", F03_MANIFEST)
    print("Root run:", RUN_ROOT, "| Resume:", IS_RESUME)
    print("Log session baru:", SESSION_ROOT)
    ''')
    cell('markdown','''\
    ## Muat kode dan pengujian F01-F08

    Receipt, pip freeze dan log pengujian ditulis ke folder session baru agar
    log session sebelumnya tersedia saat resume. Source ZIP tidak berisi dataset
    atau checkpoint; checkpoint yang dibuat pada langkah berikut hanya fixture.
    ''')
    setup=''.join(previous['cells'][7]['source'])
    setup=setup.replace('configs/training/risk_synthetic.toml','configs/orchestration/phase08.toml')
    setup=setup.replace('rsvt.experiments.phase7 import load_phase7_config','rsvt.experiments.runner import load_phase8_config')
    setup=setup.replace('load_phase7_config','load_phase8_config').replace('"F07"','"F08"')
    setup=setup.replace('"reviewed_f06_run": str(F06_RUN)','"reviewed_f07_run": str(F07_RUN)')
    setup=setup.replace('RUN_ROOT','SESSION_ROOT')
    cell('code',setup)
    cell('markdown','''\
    ## Jalankan lima job sintetis yang tersedia

    Setiap model memakai batch 8 volume prosedural 32 cubed, 2 optimizer step,
    seed13, dan input/split yang sama. Pilih checkpoint minimum unweighted
    source-validation CE, lalu bekukan threshold dari authentic validation.
    Demo FPR0,05 hanya menguji interface: empat authentic volume tidak dapat
    menetapkan population FPR 5%. Varian generator berasal dari fixture.

    B4 pada katalog F08 memakai CVaR tanpa generator variance. B5 menambah
    routing dan generator variance. Ini penyelarasan desain proposal; hasil B4
    prototipe F07 yang memakai variance tetap dipertahankan sebagai riwayat.
    ''')
    cell('code','''\
    command = [sys.executable, "-u", str(CODE_ROOT / "scripts/project.py"), "phase8-run",
               "--config", str(CONFIG_PATH), "--output-dir", str(RUN_ROOT / "orchestration"),
               "--f03-manifest", str(F03_MANIFEST), "--device", DEVICE]
    if IS_RESUME:
        command.append("--resume")
    with (SESSION_ROOT / "phase8_console.log").open("w") as log:
        with subprocess.Popen(command, cwd=CODE_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            exit_code = process.wait()
    receipt.update(orchestration_exit_code=exit_code, f03_manifest_sha256=F03_MANIFEST_SHA)
    (SESSION_ROOT / "notebook_receipt.json").write_text(json.dumps(receipt, indent=2))
    summary_path = RUN_ROOT / "orchestration/summary.json"
    if exit_code or not summary_path.is_file():
        raise RuntimeError(f"Job belum selesai; baca {SESSION_ROOT / 'phase8_console.log'} dan ledger.json.")
    summary = json.loads(summary_path.read_text())
    if summary.get("ok") is not True:
        raise RuntimeError("Pemeriksaan orkestrasi belum lolos; periksa checks pada summary.")
    print("Checks:", json.dumps(summary["checks"], indent=2))
    ''')
    cell('markdown','''\
    ## Verifikasi job-boundary resume dan replay checkpoint

    Resume verification berikut tidak mengulang training job selesai. Semua
    artifact dicek checksum sebelum skip. Checkpoint B5 terbaik kemudian dibaca
    ulang melalui CLI inference/evaluation terpisah; prediksi serta metrik harus
    cocok dengan artifact validation yang disegel. Test medis tidak tersedia.
    ''')
    cell('code','''\
    resumed = subprocess.run(command if IS_RESUME else command + ["--resume"], cwd=CODE_ROOT,
                             capture_output=True, text=True)
    (SESSION_ROOT / "resume_verification.log").write_text(resumed.stdout + resumed.stderr)
    if resumed.returncode:
        raise RuntimeError("Verifikasi resume gagal; periksa log session.")
    summary = json.loads(summary_path.read_text())
    if len(summary["jobs_skipped_on_resume"]) != summary["jobs_completed"]:
        raise RuntimeError("Resume verification tidak melewati seluruh job selesai.")
    ledger = json.loads((RUN_ROOT / "orchestration/ledger.json").read_text())
    job_id = next(key for key in ledger["jobs"] if key.startswith("B5_"))
    record = ledger["jobs"][job_id]
    attempt = RUN_ROOT / "orchestration/jobs" / job_id / record["completed_attempt"]
    job_result = json.loads((attempt / "result.json").read_text())
    replay_predictions = SESSION_ROOT / "replay_predictions.json"
    replay_metrics = SESSION_ROOT / "replay_metrics.json"
    replay_commands = [
        [sys.executable, str(CODE_ROOT / "scripts/project.py"), "phase8-infer", "--checkpoint",
         str(attempt / job_result["best_checkpoint"]["path"]), "--output", str(replay_predictions), "--device", DEVICE],
        [sys.executable, str(CODE_ROOT / "scripts/project.py"), "phase8-evaluate", "--predictions",
         str(replay_predictions), "--threshold", str(attempt / "threshold.json"), "--output", str(replay_metrics)]]
    for replay_command in replay_commands:
        subprocess.run(replay_command, cwd=CODE_ROOT, check=True)
    replay = json.loads(replay_predictions.read_text())
    replayed_metrics = json.loads(replay_metrics.read_text())
    frozen_threshold = json.loads((attempt / "threshold.json").read_text())
    if replay["prediction_sha256"] != frozen_threshold["calibration_prediction_sha256"] or replayed_metrics != job_result["metrics"]:
        raise RuntimeError("Replay berbeda dari validation artifacts yang disegel.")
    receipt.update(resume_verification_passed=True, checkpoint_replay_passed=True)
    (SESSION_ROOT / "notebook_receipt.json").write_text(json.dumps(receipt, indent=2))
    print("Seluruh completed job dilewati; replay checkpoint dan frozen-threshold evaluation cocok.")
    ''')
    cell('markdown','''\
    ## Audit protokol dan pekerjaan penelitian yang masih terbuka

    `plan.lock.json` adalah snapshot engineering terikat checksum. Hasil
    `research_ready=false` diharapkan pada draft saat ini. Daftar keputusan yang
    belum selesai dan backend yang belum tersedia harus tetap terlihat.
    Metadata F03 diperiksa tanpa membaca volume atau mengubah flag verifikasi.
    ''')
    cell('code','''\
    from IPython.display import Markdown, display

    plan = json.loads((RUN_ROOT / "orchestration/plan.lock.json").read_text())
    table = ["| ID | Jenis | Adapter sintetis | Status |", "|---|---|---|---|"]
    for identity, data in plan["payload"]["catalog"].items():
        entry = data["entry"]
        status = "tersedia" if entry["synthetic_backend_supported"] else entry["unsupported_reason"]
        table.append(f"| {identity} | {entry['kind']} | {entry['synthetic_backend_supported']} | {status} |")
    display(Markdown("\\n".join(table)))
    print("Audit metadata F03:", json.dumps(summary["f03_metadata"], indent=2))
    print("Kesiapan penelitian:", json.dumps(summary["research_readiness"], indent=2))
    print("F08 engineering:", "LOLOS" if summary["ok"] else "PERLU DIPERIKSA")
    print("Job selesai:", summary["jobs_completed"])
    print("Final test access:", summary["research_readiness"]["final_test_access_allowed"])
    print("Hasil Drive:", RUN_ROOT)
    ''')
    cell('markdown','''\
    ## Setelah run

    `orchestration/` menyimpan plan snapshot, ledger, summary dan folder job/attempt
    berisi checkpoint sintetis, history, prediksi, threshold serta metrics.
    Session logs berada di `sessions/`. Checkpoint/RNG menyimpan provenance, tetapi
    resume F08 saat ini hanya pada batas completed job; partial job diulang pada
    attempt baru, bukan dilanjutkan dari optimizer step yang belum selesai.

    Hasil direview sebelum menyiapkan evaluasi F09. Penggunaan final test dan
    penguncian protokol konfirmatori menunggu penyelesaian QA/provenance,
    baseline, data multi-generator dan keputusan statistik/source-validation.
    ''')
    for index,item in enumerate(cells):item['id']=f'f08-{index:02d}'
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    output=legacy_output(ROOT,'08_phase8_colab.ipynb');output.write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return output


if __name__=='__main__':print(build())
