"""Generate the user-run Colab F10 notebook, preserving F08/F09 releases."""

from copy import deepcopy
import json
from pathlib import Path
from textwrap import dedent
from notebook_layout import legacy_input, legacy_output

ROOT=Path(__file__).resolve().parents[1]


def build():
    previous=json.loads(legacy_input(ROOT,'09_phase9_colab.ipynb').read_text(encoding='utf-8'));cells=[]
    def cell(kind, source):
        item={'cell_type':kind,'metadata':{},'source':dedent(source).strip().splitlines(keepends=True)}
        if kind=='code':item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','''\
    # F10 — Ablation berpasangan dan statistik berbasis grup sumber

    Jalankan sendiri dengan **Run All di Google Colab melalui VS Code**.
    Notebook membaca hasil F08 yang tersimpan dan menghitung ulang F09 sebagai
    prasyarat. Run F09 mandiri juga sudah direview: 222 tes dan replay metrik lolos.
    Tidak memerlukan dataset lokal atau volume medis pada runtime baru.

    Default: 10 job sintetis, dua seed/fixture, dua optimizer step. Setiap seed
    memasangkan FULL dengan tanpa CVaR, tanpa generator variance, cap128 dan
    cap256 (reference cap64). Input, assignment dan bobot awal tiap pasangan
    harus sama. E3/E4/E5 engineering diuji; E1/E2/E6/E7 tetap belum selesai.

    Checkpoint terpilih diprofilkan dalam worker terpisah dengan hardware/input
    sama per seed: p50/p95, controller stages, scoped FLOP estimates dan memory.
    Ini checkpoint dua-step sintetis, belum model medis atau baseline B0/B1/B2.
    Bootstrap berpasangan mengambil seluruh parent groups 1.000 kali. Hasil
    tetap exploratory; radius/FPR/margin/protokol penelitian belum ditetapkan.
    ''')
    for index in (1,2,3):
        item=deepcopy(previous['cells'][index]);item['metadata']={};item.pop('attachments',None)
        item['source']=[s.replace('F09','F10') for s in item['source']]
        if item['cell_type']=='code':item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','''\
    ## Drive dan prasyarat

    Gunakan folder proyek Drive yang sama. Tidak perlu melatih ulang F08.
    Default membuat root run F10 baru pada setiap Run All. Jika terputus, output
    parsial dipertahankan; Run All berikutnya mengulang job kecil pada root baru.
    F10 tidak menawarkan optimizer resume. Analisis completed run dapat dihitung
    ulang tanpa training melalui `phase10-analyze`.
    ''')
    configure=''.join(previous['cells'][5]['source'])
    configure=configure.replace('rsvt-colab-f09-code.zip','rsvt-colab-f10-code.zip')
    configure=configure.replace('/content/rsvt-f09','/content/rsvt-f10').replace("strftime('f09_","strftime('f10_")
    configure=configure.replace('Paket F09:','Paket F10:')
    configure+='\n'+dedent('''\
    F09_RUN = STATE_ROOT / 'outputs/engineering/f09_20261111T123343_367466Z'
    F09_SUMMARY_PATH = F09_RUN / 'evaluation/summary.json'
    if not F09_SUMMARY_PATH.is_file() or not (F09_RUN / 'session/notebook_receipt.json').is_file():
        raise FileNotFoundError('Hasil F09 yang direview belum ditemukan pada akun/path Drive ini.')
    F09_SUMMARY_SHA256 = hashlib.sha256(F09_SUMMARY_PATH.read_bytes()).hexdigest()
    f09_receipt = json.loads((F09_RUN / 'session/notebook_receipt.json').read_text())
    if (F09_SUMMARY_SHA256 != '123dc0b572c831406d6e5e113a350d8af356e01d9a5358eab609b3cef7995671'
            or f09_receipt.get('bundle_sha256') != 'd1a0df5a67f7d5fd5ab9112ff8aa931ea182fc458002298031422f0f26fd9020'
            or f09_receipt.get('synthetic_tests_exit_code') != 0 or f09_receipt.get('evaluation_exit_code') != 0
            or f09_receipt.get('recompute_verification_passed') is not True):
        raise RuntimeError('Bukti F09 berbeda dari run yang sudah direview.')
    print('F09 direview:', F09_RUN)
    ''')
    cell('code',configure)
    cell('markdown','''\
    ## Muat kode dan uji F01–F10

    Seluruh fixture kecil dibuat di runtime Colab. Suite mencakup lima checkpoint
    sintetis/profiling worker dan oracle bootstrap. CPU diperbolehkan; GPU dipakai
    untuk run utama bila tersedia. Waktu suite/profiling bergantung pada runtime.
    ''')
    setup=''.join(previous['cells'][7]['source'])
    setup=setup.replace('configs/evaluation/phase09.toml','configs/analysis/phase10.toml')
    setup=setup.replace('rsvt.experiments.phase9 import load_phase9_config','rsvt.experiments.phase10 import load_phase10_config')
    setup=setup.replace('load_phase9_config','load_phase10_config').replace('"F09"','"F10"').replace('test_f0*.py','test_f*.py')
    setup=setup.replace('"reviewed_f08_run": str(F08_RUN)',
        '"reviewed_f08_run": str(F08_RUN), "reviewed_f09_run": str(F09_RUN), "reviewed_f09_summary_sha256": F09_SUMMARY_SHA256')
    cell('code',setup)
    cell('markdown','''\
    ## Jalankan E3/E4/E5 sintetis dan analisis

    CVaR removal mempertahankan CE, routing, generator term dan dual; generator
    removal mempertahankan CE, routing, CVaR dan dual. Budget comparison melatih
    model terpisah pada cap64/128/256. Seluruh kandidat dan stem tetap dibangun;
    FLOP estimate menyatakan operator yang dikecualikan. Jangan menyamakan
    pengurangan attention dengan pengurangan seluruh biaya model.

    Threshold masing-masing model ditetapkan dari validation lalu dibekukan saat
    bootstrap. CI bersifat conditional pada model/threshold itu, tanpa mencakup
    ketidakpastian training/kalibrasi. Seed13/37 juga mengubah fixture, sehingga
    dianalisis terpisah dan belum merupakan optimizer-seed-only E7.
    ''')
    cell('code','''\
    EXPERIMENT_ROOT = RUN_ROOT / 'experiments'
    command = [sys.executable, '-u', str(CODE_ROOT / 'scripts/project.py'), 'phase10-run',
        '--f08-run', str(F08_RUN / 'orchestration'), '--config', str(CONFIG_PATH),
        '--output-dir', str(EXPERIMENT_ROOT), '--device', DEVICE]
    with (SESSION_ROOT / 'phase10_console.log').open('w') as log:
        with subprocess.Popen(command, cwd=CODE_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            exit_code=process.wait()
    receipt.update(phase10_exit_code=exit_code)
    (SESSION_ROOT / 'notebook_receipt.json').write_text(json.dumps(receipt,indent=2))
    if exit_code:raise RuntimeError(f'F10 belum selesai; periksa {SESSION_ROOT / "phase10_console.log"}')
    summary=json.loads((EXPERIMENT_ROOT / 'summary.json').read_text())
    analysis=json.loads((EXPERIMENT_ROOT / 'analysis/summary.json').read_text())
    if not summary.get('ok') or not analysis.get('ok'):raise RuntimeError('Checks F10 belum lolos.')
    print('Checks:',json.dumps(summary['checks'],indent=2))
    print('Job selesai:',summary['jobs_completed'])
    ''')
    cell('markdown','''\
    ## Reproduksi analisis tanpa inference atau training

    Semua checkpoint/profile/prediksi diverifikasi checksum sebagai artifact.
    `phase10-analyze` tidak mendeserialisasi model. Hasil dan checksum harus sama.
    ''')
    cell('code','''\
    repeat = [sys.executable, str(CODE_ROOT / 'scripts/project.py'), 'phase10-analyze',
        '--run-root', str(EXPERIMENT_ROOT), '--output-dir', str(RUN_ROOT / 'analysis_recomputed')]
    completed=subprocess.run(repeat,cwd=CODE_ROOT,capture_output=True,text=True)
    (SESSION_ROOT / 'analysis_recompute.log').write_text(completed.stdout+completed.stderr)
    if completed.returncode:raise RuntimeError('Reproduksi analisis gagal; periksa log.')
    repeated=json.loads((RUN_ROOT / 'analysis_recomputed/summary.json').read_text())
    if repeated!=analysis:raise RuntimeError('Analisis/checksum berbeda pada perhitungan ulang.')
    receipt.update(analysis_recompute_passed=True,source_sha256=summary['source_sha256'],plan_sha256=summary['plan_sha256'])
    (SESSION_ROOT / 'notebook_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('Analisis berulang identik; final test access:',summary['final_test_access_allowed'])
    ''')
    cell('markdown','''\
    ## Tabel hasil dan interpretasi

    Hanya empat parent groups validation per model. Resampling dapat menghilangkan
    kelas FB/authentic atau kelas lain. Jika ada satu replicate undefined,
    interval untuk metrik itu ditampilkan sebagai **tidak tersedia**; replicate
    invalid tidak disaring menjadi CI kondisional. Nilai point estimate tetap
    ditampilkan. Margin noninferiority masih PENDING dan tidak ada klaim dibuat.
    ''')
    cell('code','''\
    from IPython.display import Markdown,display
    table=['| Kontras | Delta FB deteksi | CI deskriptif | Replicate undefined |',
           '|---|---|---|---|']
    for name,result in analysis['paired_contrasts'].items():
        metric=result['metrics']['FB_tamper_recall'];interval=metric['percentile_interval']
        ci=str(interval) if interval is not None else 'tidak tersedia'
        table.append(f"| {name} | {metric['delta_first_minus_second']} | {ci} | {metric['undefined_replicates']} |")
    display(Markdown('\\n'.join(table)))
    cost=['| Seed | Kondisi | Cap | Macro-F1 | p95 ms | Operator FLOPs |',
          '|---|---|---|---|---|---|']
    for seed,frontier in analysis['budget_frontiers'].items():
        for row in frontier['rows']:
            cost.append(f"| {seed} | {row['id']} | {row['sequence_cap']} | {row['quality']:.4f} | "
                        f"{row['latency_p95_ms']:.3f} | {row['operator_flops']} |")
        print('Frontier deskriptif seed',seed,':',frontier['frontier_ids'])
    display(Markdown('\\n'.join(cost)))
    print('Status E1–E7:',json.dumps(analysis['ablation_status'],indent=2))
    print('Research inference allowed:',analysis['research_inference_allowed'])
    print('F09 prerequisite:',summary['prerequisite_F09'])
    print('Hasil Drive:',RUN_ROOT)
    ''')
    cell('markdown','''\
    ## Setelah Run All

    `experiments/` menyimpan plan, ledger, prasyarat F09, job/checkpoint/prediksi,
    profile worker, paired bootstrap distribution/draws, frontier dan CSV biaya.
    `analysis_recomputed/` membuktikan reproduksi analisis. `session/` menyimpan
    log tes, console, environment dan receipt.

    Hasil engineering direview sebelum langkah berikutnya. E1/E2/E6/E7, quality
    baseline, canonical QA, final protokol, actual multi-generator training,
    test medis, profil baseline terlatih yang sepadan serta analisis statistik
    pasien konfirmatori tetap terbuka. Tidak ada dataset diunduh ke komputer.
    ''')
    for i,item in enumerate(cells):item['id']=f'f10-{i:02d}'
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=legacy_output(ROOT,'10_phase10_colab.ipynb')
    path.write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');return path


if __name__=='__main__':print(build())
