"""Generate the F09 saved-prediction notebook; preserve earlier user runs."""

from copy import deepcopy
import json
from pathlib import Path
from textwrap import dedent
from notebook_layout import legacy_input, legacy_output

ROOT = Path(__file__).resolve().parents[1]


def build():
    previous = json.loads(legacy_input(ROOT, '08_phase8_colab.ipynb').read_text(encoding='utf-8'))
    cells = []
    def cell(kind, source):
        item = {'cell_type': kind, 'metadata': {}, 'source': dedent(source).strip().splitlines(keepends=True)}
        if kind == 'code': item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell('markdown', '''\
    # F09 — Evaluasi prediksi, kalibrasi dan audit generator

    Jalankan **Run All** sendiri pada Google Colab melalui VS Code. Notebook
    membaca prediksi/checksum F08 dari Drive; F08 tidak perlu dilatih ulang.
    Tidak memerlukan dataset di komputer atau volume CT pada VM baru.

    F09 menghitung confusion matrix, macro-F1 empat kelas, AUROC OvR/tamper,
    recall FB untuk deteksi dan klasifikasi secara terpisah, FPR pada threshold
    validation yang sama, Brier, NLL, ECE serta performa per generator.
    Seluruh hasil model masih **validation sintetis F08**, bukan final test medis.

    Dua rotasi LOGO dan lokalisasi diperiksa memakai fixture probabilitas/ruang
    fisik buatan. Fixture itu tidak berasal dari model yang dilatih LOGO.
    QA, protokol, data multi-generator, profiling model terlatih dan statistik
    penelitian tetap terbuka. Tidak ada opsi untuk membuka final test di sini.
    ''')
    for index in (1, 2, 3):
        item = deepcopy(previous['cells'][index]); item['metadata'] = {}; item.pop('attachments', None)
        item['source'] = [s.replace('F08', 'F09') for s in item['source']]
        if item['cell_type'] == 'code': item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell('markdown', '''\
    ## Konfigurasi Drive dan bukti F08 yang direview

    Default menggunakan run F08 terakhir yang lolos review, termasuk resume dan
    replay. Setiap Run All membuat output F09 baru sehingga hasil lama tersimpan.
    Dataset medis tidak dibaca; checkpoint sintetis F08 dicek sebagai byte untuk
    checksum tanpa deserialisasi atau inference dengan kode yang telah berubah.
    ''')
    cell('code', '''\
    from datetime import datetime, timezone
    import hashlib
    import json

    STATE_ROOT = Path('/content/drive/MyDrive/Research/LUNG-CT-SCAN')
    CODE_ARCHIVE = STATE_ROOT / 'rsvt-colab-f09-code.zip'
    F08_RUN = STATE_ROOT / 'outputs/engineering/f08_20261111T105446_662072Z'
    F08_SESSION = F08_RUN / 'sessions/session_20261111T105446_662282Z'
    DEVICE = 'auto'
    for path in (CODE_ARCHIVE, F08_RUN / 'orchestration/summary.json',
                 F08_RUN / 'orchestration/plan.lock.json', F08_SESSION / 'notebook_receipt.json'):
        if not path.is_file():
            raise FileNotFoundError(f'Paket/bukti belum ditemukan pada akun/path Drive ini: {path}')
    f08 = json.loads((F08_RUN / 'orchestration/summary.json').read_text())
    f08_plan = json.loads((F08_RUN / 'orchestration/plan.lock.json').read_text())
    f08_receipt = json.loads((F08_SESSION / 'notebook_receipt.json').read_text())
    if (f08.get('ok') is not True or len(f08.get('checks', {})) != 6 or not all(f08['checks'].values())
            or f08.get('jobs_completed') != 5 or len(f08.get('jobs_skipped_on_resume', [])) != 5
            or f08.get('plan_sha256') != '6c190c5ee9479857e40f452815045390174322055817054455e6eaeb3010a1ae'
            or f08_plan.get('contract_sha256') != f08['plan_sha256']
            or f08_plan['payload'].get('source_sha256') != 'a8f18679a689028f54a2bb187b9d9865d2ce1d1e56a3c954185e2a5da88fdb53'
            or f08_receipt.get('bundle_sha256') != '350c362bfd93639d492481820a66f43f3d1c0003c2c175d969347478a9e78c14'
            or f08_receipt.get('synthetic_tests_exit_code') != 0 or f08_receipt.get('orchestration_exit_code') != 0
            or f08_receipt.get('resume_verification_passed') is not True or f08_receipt.get('checkpoint_replay_passed') is not True):
        raise RuntimeError('Bukti F08 berbeda dari run yang direview; periksa path/run terlebih dahulu.')
    BUNDLE_SHA256 = hashlib.sha256(CODE_ARCHIVE.read_bytes()).hexdigest()
    CODE_ROOT = Path('/content/rsvt-f09') / BUNDLE_SHA256[:12]
    RUN_ID = datetime.now(timezone.utc).strftime('f09_%Y%m%dT%H%M%S_%fZ')
    RUN_ROOT = STATE_ROOT / 'outputs/engineering' / RUN_ID
    SESSION_ROOT = RUN_ROOT / 'session'
    print('F08 direview:', F08_RUN)
    print('Paket F09:', BUNDLE_SHA256)
    print('Output baru:', RUN_ROOT)
    ''')
    cell('markdown', '''\
    ## Muat kode dan uji F01–F09

    Instal dependensi pada runtime Colab dan jalankan suite sintetis. Pengujian
    memakai fixture prosedural sementara; seluruh dataset medis tetap di Drive.
    Perhitungan metrik tersimpan memakai CPU NumPy. GPU tidak disyaratkan.
    ''')
    setup = ''.join(previous['cells'][7]['source'])
    setup = setup.replace('configs/orchestration/phase08.toml', 'configs/evaluation/phase09.toml')
    setup = setup.replace('rsvt.experiments.runner import load_phase8_config', 'rsvt.experiments.phase9 import load_phase9_config')
    setup = setup.replace('load_phase8_config', 'load_phase9_config').replace('"F08"', '"F09"')
    setup = setup.replace('"reviewed_f07_run": str(F07_RUN)', '"reviewed_f08_run": str(F08_RUN)')
    cell('code', setup)
    cell('markdown', '''\
    ## Hitung ulang prediksi F08 tanpa training atau pemilihan threshold baru

    Verifikasi ledger, artifact, assignment sumber, checkpoint dan threshold.
    Kontrol authentic yang sama dipakai untuk AUC per generator; kontrol itu
    tidak dijumlahkan ulang sebagai volume independen. Satu baris = satu volume.
    Delapan volume F08 berasal dari empat grup fixture; bukan delapan pasien.
    ''')
    cell('code', '''\
    command = [sys.executable, '-u', str(CODE_ROOT / 'scripts/project.py'), 'phase9-evaluate',
               '--f08-run', str(F08_RUN / 'orchestration'), '--config', str(CONFIG_PATH),
               '--output-dir', str(RUN_ROOT / 'evaluation')]
    with (SESSION_ROOT / 'phase9_console.log').open('w') as log:
        with subprocess.Popen(command, cwd=CODE_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
            exit_code = process.wait()
    receipt.update(evaluation_exit_code=exit_code)
    (SESSION_ROOT / 'notebook_receipt.json').write_text(json.dumps(receipt, indent=2))
    if exit_code: raise RuntimeError(f'Evaluasi gagal; baca {SESSION_ROOT / "phase9_console.log"}')
    summary = json.loads((RUN_ROOT / 'evaluation/summary.json').read_text())
    if summary.get('ok') is not True: raise RuntimeError('Checks evaluasi belum lolos.')
    print('Checks:', json.dumps(summary['checks'], indent=2))
    ''')
    cell('markdown', '''\
    ## Verifikasi reproduksi dari prediksi yang disimpan

    Evaluasi diulang ke folder baru. Summary dan checksum artifact harus identik;
    F08 tidak diubah. Ini reproduksi perhitungan metrik, bukan inference baru.
    ''')
    cell('code', '''\
    repeat = command[:-1] + [str(RUN_ROOT / 'evaluation_recomputed')]
    completed = subprocess.run(repeat, cwd=CODE_ROOT, capture_output=True, text=True)
    (SESSION_ROOT / 'recompute_verification.log').write_text(completed.stdout + completed.stderr)
    if completed.returncode: raise RuntimeError('Penghitungan ulang gagal; periksa log.')
    recomputed = json.loads((RUN_ROOT / 'evaluation_recomputed/summary.json').read_text())
    if recomputed != summary: raise RuntimeError('Metrik/checksum berbeda pada perhitungan ulang.')
    import csv
    with (RUN_ROOT / 'evaluation/predictions.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != summary['prediction_rows']: raise RuntimeError('Jumlah baris ekspor berbeda.')
    receipt.update(recompute_verification_passed=True, source_sha256=summary['source_sha256'],
                   f08_plan_sha256=summary['f08_plan_sha256'])
    (SESSION_ROOT / 'notebook_receipt.json').write_text(json.dumps(receipt, indent=2))
    print('Perhitungan ulang identik; ekspor JSON/CSV:', len(csv_rows), 'baris model-volume.')
    ''')
    cell('markdown', '''\
    ## Hasil dan batas interpretasi

    Nilai ECE menggunakan 10 bin tetap sebagai demo, bukan keputusan protokol.
    Fixture holdout sengaja menunjukkan FPR dapat melebihi target validation.
    Dice hanya untuk mask voxel verified dengan cakupan penuh. Titik memakai
    jarak LPS dalam mm dan matching satu-ke-satu; radius demo 5 mm belum menjadi
    radius penelitian. Partial fine-token coverage tidak dianggap background.
    ''')
    cell('code', '''\
    from IPython.display import Markdown, display
    table = ['| Job | Macro-F1 | FB kelas | FB deteksi | FPR | Brier 4 kelas | ECE top-label |',
             '|---|---|---|---|---|---|---|']
    for job, result in summary['results'].items():
        c, t, k = result['classification'], result['frozen_threshold'], result['calibration']
        table.append(f"| {job} | {c['macro_f1_all_four']:.4f} | {t['FB_classification_recall']:.4f} | "
                     f"{t['FB_tamper_recall']:.4f} | {t['authentic_FPR']:.4f} | "
                     f"{k['multiclass_brier_sum']:.4f} | {k['top_label']['ece']:.4f} |")
    display(Markdown('\\n'.join(table)))
    demo = json.loads((RUN_ROOT / 'evaluation/metric_demonstrations.json').read_text())
    print('Fixture spatial:', json.dumps({k: demo[k] for k in ('point_localization', 'mask_localization')}, indent=2))
    print('Actual LOGO model trained:', summary['actual_LOGO_model_trained'])
    print('Final test access:', summary['final_test_access_allowed'])
    print('Profiling:', json.dumps(summary['efficiency'], indent=2))
    print('Masih terbuka:', json.dumps(summary['pending'], indent=2))
    print('Hasil Drive:', RUN_ROOT)
    ''')
    cell('markdown', '''\
    ## Setelah run

    `evaluation/` berisi report per model, prediksi JSON/CSV, fixture metrik,
    status profiling dan checksum artifact. `evaluation_recomputed/` membuktikan
    reproduksi metrik. Log tes/runtime berada di `session/`.

    Sampaikan setelah Run All selesai. Hasil direview sebelum F10 disiapkan.
    Evaluasi medis final, actual LOGO, baseline B2, pasangan ablation dan profil
    checkpoint terlatih yang sepadan belum selesai; hasil ini belum mendukung
    klaim superioritas, noninferiority, validasi eksternal atau ketidakpastian pasien.
    ''')
    for index, item in enumerate(cells): item['id'] = f'f09-{index:02d}'
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    path = legacy_output(ROOT, '09_phase9_colab.ipynb')
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return path


if __name__ == '__main__': print(build())
