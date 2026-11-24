"""Generate notebook 06 while retaining the released notebook 05 source."""

from copy import deepcopy
import json
from pathlib import Path
from textwrap import dedent
from notebook_layout import legacy_input, legacy_output

ROOT=Path(__file__).resolve().parents[1]


def build():
    previous=json.loads(legacy_input(ROOT,'05_phase5_colab.ipynb').read_text(encoding='utf-8'))
    cells=[]
    def cell(kind,source):
        item={'cell_type':kind,'metadata':{},'source':dedent(source).strip().splitlines(keepends=True)}
        if kind=='code':item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','''\
    # F06 - Routing adaptif dan profiling RSVT dengan fixture sintetis

    Jalankan sendiri dengan **Google Colab melalui VS Code**, kernel CPU/GPU.
    Paket `rsvt-colab-f06-code.zip` sudah berada di Drive proyek. F05 engineering
    telah lolos; notebook ini memeriksa B3 routing-only, tanpa CVaR/generator loss.

    Training menggunakan sigmoid attention gates yang differentiable dan tetap
    dense. Inference memakai hard top-k dan gather sebelum Transformer; coarse
    valid selalu dipertahankan. Pemeriksaan memakai tensor prosedural. Training
    medis, pretraining anatomi dan QA penelitian tetap terbuka.
    ''')
    for index in (1,2,3):
        item=deepcopy(previous['cells'][index]);item['metadata']={}
        item['source']=[line.replace('F05','F06') for line in item['source']]
        if item['cell_type']=='code':item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','''\
    ## Konfigurasi dan prasyarat engineering

    Drive harus memakai akun dengan folder `MyDrive/Research/LUNG-CT-SCAN`.
    F05 tidak perlu diulang. `DEVICE="auto"` menggunakan CUDA jika tersedia,
    atau CPU. Setiap Run All membuat run baru agar hasil sebelumnya dipertahankan.
    ''')
    cell('code','''\
    from datetime import datetime, timezone
    import hashlib
    import json

    STATE_ROOT = Path("/content/drive/MyDrive/Research/LUNG-CT-SCAN")
    CODE_ARCHIVE = STATE_ROOT / "rsvt-colab-f06-code.zip"
    DEVICE = "auto"
    F05_RUN = STATE_ROOT / "outputs/engineering/f05_20261111T082858_565780Z"
    if DEVICE not in {"auto", "cpu", "cuda"}:
        raise ValueError("DEVICE tidak dikenal.")
    for path in (CODE_ARCHIVE, F05_RUN / "smoke/summary.json", F05_RUN / "notebook_receipt.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Berkas F06/prasyarat F05 belum ditemukan pada akun/path Drive ini: {path}")
    f05 = json.loads((F05_RUN / "smoke/summary.json").read_text())
    f05_receipt = json.loads((F05_RUN / "notebook_receipt.json").read_text())
    if (f05.get("ok") is not True or not f05.get("checks") or not all(f05.get("checks", {}).values())
            or f05.get("rsvt_python_source_sha256") != "f2a85d80c260223800d38507ceebf07a2d9776622743af31a8344a64034d4d57"
            or f05_receipt.get("bundle_sha256") != "fd3c36cd4bbfece3913747f2ae88bfd4b85ed8208d38950a1582e207fc6dde96"
            or f05_receipt.get("synthetic_tests_exit_code") != 0 or f05_receipt.get("smoke_exit_code") != 0):
        raise RuntimeError("Bukti engineering F05 berbeda dari run yang sudah direview.")
    BUNDLE_SHA256 = hashlib.sha256(CODE_ARCHIVE.read_bytes()).hexdigest()
    CODE_ROOT = Path("/content/rsvt-f06") / BUNDLE_SHA256[:12]
    RUN_ID = datetime.now(timezone.utc).strftime("f06_%Y%m%dT%H%M%S_%fZ")
    RUN_ROOT = STATE_ROOT / "outputs/engineering" / RUN_ID
    print("F05 direview:", F05_RUN)
    print("Paket F06 SHA256:", BUNDLE_SHA256)
    print("Hasil F06:", RUN_ROOT)
    ''')
    cell('markdown','''\
    ## Muat source, dependensi dan suite sintetis F01-F06

    Log tes mencakup default 96 cubed soft forward/backward dan hard inference.
    Profiling dilakukan sesudah pengujian berhasil. Dependensi aktual dan receipt
    disimpan di Drive. Dataset medis tidak dibaca oleh notebook ini.
    ''')
    setup=''.join(previous['cells'][7]['source'])
    setup=setup.replace('phase5','phase6').replace('rsvt_f05','rsvt_f06').replace('"F05"','"F06"')
    setup=setup.replace('"medical_training_ready": False}',
        '"medical_training_ready": False, "reviewed_f05_run": str(F05_RUN)}')
    cell('code',setup)
    cell('markdown','''\
    ## B3 sintetis, audit seleksi dan profiling enam profil

    Satu optimizer step memakai CE + soft expected-token penalty. Gradien dari
    CE ke router diperiksa secara terpisah. Inference hard harus memenuhi cap
    sequence termasuk CLS dan mencakup setiap voxel valid melalui coarse.

    Profiling menggunakan satu volume sintetis 64 cubed, FP32, batch 1, warmup 2,
    10 sampel latency dan 3 probe komponen. Enam fresh process memakai bobot awal
    identik: dense struktural, all-valid packing, minimum coarse+CLS, cap 64,
    128 dan 256. Hasil memuat p50/p95 end-to-end, overhead per komponen, CPU RSS,
    CUDA peak memory jika tersedia, serta estimasi FLOPs operator yang dicakup.
    Pengukuran mengecualikan startup, instalasi, I/O dataset dan training.

    Audit terpisah menggunakan model sesudah satu step dan tiga variasi noise
    pada kasus sintetis yang sama. Kandidat yang selalu tidak terpilih dicatat.
    ''')
    cell('code','''\
    command = [sys.executable, "-u", str(CODE_ROOT / "scripts/project.py"), "phase6-routing",
               "--config", str(CONFIG_PATH), "--output-dir", str(RUN_ROOT / "routing"), "--device", DEVICE]
    with (RUN_ROOT / "phase6_console.log").open("w") as log:
        with subprocess.Popen(command, cwd=CODE_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            exit_code = process.wait()
    receipt.update(routing_exit_code=exit_code)
    (RUN_ROOT / "notebook_receipt.json").write_text(json.dumps(receipt, indent=2))
    summary_path = RUN_ROOT / "routing/summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Summary belum terbentuk; baca {RUN_ROOT / 'phase6_console.log'} dan routing/profiles/*.log")
    summary = json.loads(summary_path.read_text())
    print("Checks:", json.dumps(summary["checks"], indent=2))
    if exit_code or summary.get("ok") is not True:
        raise RuntimeError("Pemeriksaan F06 belum lolos; periksa checks dan log di Drive.")
    ''')
    cell('markdown','''\
    ## Membaca hasil biaya dan batas interpretasi

    `speed ratio > 1` berarti p50 lebih cepat daripada dense struktural pada
    fixture/perangkat ini; bandingkan juga all-valid untuk memisahkan efek packing
    padding. Sequence pendek dapat disertai overhead controller/stem yang dominan.
    Estimasi FLOPs mencakup Conv3d, Linear dan perkalian attention; pooling/cue,
    norm, activation, softmax, sorting/gather dan memory traffic belum dicakup.

    Kalibrasi memakai target 0,5/0,75/1,0 kali dense p95. Jika tidak ada profil
    terukur yang memenuhi target, hasil ditandai infeasible. Target latency bukan
    jaminan constraint runtime. Batas inference yang ditegakkan adalah cap token.
    Sepuluh sampel p95 hanya profiling awal; budget riset tetap perlu source validation.
    ''')
    cell('code','''\
    from IPython.display import Markdown, display

    rows = ["| Profil | Sequence | p50 ms | p95 ms | Speed ratio | Est. operator MFLOPs |",
            "|---|---:|---:|---:|---:|---:|"]
    for profile in summary["profiles"]:
        compute = profile["compute"]
        rows.append(f"| {profile['profile']} | {compute['actual_sequence_length_including_cls']} | "
                    f"{profile['p50_ms']:.3f} | {profile['p95_ms']:.3f} | "
                    f"{profile['measured_p50_speed_ratio_vs_dense']:.3f} | "
                    f"{compute['operator_flops_estimate']['total']/1e6:.1f} |")
    display(Markdown("\\n".join(rows)))
    print("Kalibrasi:", json.dumps(summary["latency_calibration"], indent=2))
    print("Audit seleksi:", json.dumps(summary["selection_audit"], indent=2))
    print("F06 engineering:", "LOLOS" if summary["ok"] else "PERLU DIPERIKSA")
    print("Hasil di Drive:", RUN_ROOT)
    print("Medical training ready:", summary["medical_training_ready"])
    ''')
    cell('markdown','''\
    ## Setelah run

    Hasil berada di `outputs/engineering/f06_<UTC>/routing/`: summary, resolved
    config, audit seleksi CSV/JSON dan enam profil JSON/log. Receipt/log tes ada
    pada root run. Semua profil memakai seed/bobot awal yang sama; bobot tersebut
    terpisah dari model smoke satu-step. Tidak ada checkpoint pretrained.

    Hasil direview sebelum F07 CVaR/generator losses. F04 optimisasi CNN, QA medis,
    pretraining dan pembanding B3 pada data penelitian tetap terbuka.
    ''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    for index,item in enumerate(cells):item['id']=f'f06-{index:02d}'
    output=legacy_output(ROOT,'06_phase6_colab.ipynb')
    output.write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return output


if __name__=='__main__':print(build())
