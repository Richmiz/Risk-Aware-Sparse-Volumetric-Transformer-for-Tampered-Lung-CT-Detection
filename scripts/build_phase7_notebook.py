"""Generate source-only F07 Colab cells without changing executed F06 artifacts."""

from copy import deepcopy
import json
from pathlib import Path
from textwrap import dedent
from notebook_layout import legacy_input, legacy_output

ROOT=Path(__file__).resolve().parents[1]


def build():
    previous=json.loads(legacy_input(ROOT,'06_phase6_colab.ipynb').read_text(encoding='utf-8'))
    cells=[]
    def cell(kind,source):
        item={'cell_type':kind,'metadata':{},'source':dedent(source).strip().splitlines(keepends=True)}
        if kind=='code':item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','''\
    # F07 - CVaR FB/FM, generator-risk dan training bertahap sintetis

    Jalankan sendiri melalui **Google Colab di VS Code**, CPU/GPU otomatis.
    Paket kode F07 sudah ada di Drive proyek. F06 engineering telah direview.
    Pemeriksaan ini menguji B4 risk-only dan B5 full pada delapan kasus prosedural.

    Dua generator ID pada fixture adalah identitas sintetis buatan. Data training
    proyek yang telah tersedia masih satu generator CT-GAN; generator penalty
    tidak diaktifkan untuk training medis sampai data/grupnya memenuhi syarat.
    Tidak ada dataset medis atau holdout penelitian yang dibaca oleh notebook ini.

    Alpha, bobot FB/FM, koefisien loss dan cap token adalah **parameter engineering**.
    Pemilihan melalui source validation, pretraining dan QA penelitian masih terbuka.
    ''')
    for index in (1,2,3):
        item=deepcopy(previous['cells'][index]);item['metadata']={}
        item.pop('attachments',None);item['source']=[s.replace('F06','F07') for s in item['source']]
        if item['cell_type']=='code':item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','''\
    ## Konfigurasi dan bukti F06

    Gunakan akun Drive dengan folder `MyDrive/Research/LUNG-CT-SCAN`.
    F06 tidak perlu dijalankan ulang. Notebook memeriksa receipt dan summary run
    yang direview. `DEVICE="auto"` memilih CUDA jika tersedia, atau CPU.
    Setiap Run All membuat folder run baru.
    ''')
    cell('code','''\
    from datetime import datetime, timezone
    import hashlib
    import json

    STATE_ROOT = Path("/content/drive/MyDrive/Research/LUNG-CT-SCAN")
    CODE_ARCHIVE = STATE_ROOT / "rsvt-colab-f07-code.zip"
    F06_RUN = STATE_ROOT / "outputs/engineering/f06_20261111T090630_527193Z"
    DEVICE = "auto"
    if DEVICE not in {"auto", "cpu", "cuda"}:
        raise ValueError("DEVICE tidak dikenal.")
    for path in (CODE_ARCHIVE, F06_RUN / "routing/summary.json", F06_RUN / "notebook_receipt.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Kode F07/prasyarat F06 belum ditemukan pada akun/path Drive ini: {path}")
    f06 = json.loads((F06_RUN / "routing/summary.json").read_text())
    f06_receipt = json.loads((F06_RUN / "notebook_receipt.json").read_text())
    if (f06.get("ok") is not True or len(f06.get("checks", {})) != 12 or not all(f06["checks"].values())
            or f06.get("config_sha256") != "3d0c3d1ca6a23a7d3a5d355a97cd20bd2f439ca18fde3e10c09814693ef9dcda"
            or f06_receipt.get("bundle_sha256") != "ad88f7c3a3da12c2585bf235183975deef51950e7dbc77deec0fc57f2c1d2c6b"
            or f06_receipt.get("synthetic_tests_exit_code") != 0 or f06_receipt.get("routing_exit_code") != 0):
        raise RuntimeError("Bukti F06 berbeda dari run engineering yang sudah direview.")
    BUNDLE_SHA256 = hashlib.sha256(CODE_ARCHIVE.read_bytes()).hexdigest()
    CODE_ROOT = Path("/content/rsvt-f07") / BUNDLE_SHA256[:12]
    RUN_ID = datetime.now(timezone.utc).strftime("f07_%Y%m%dT%H%M%S_%fZ")
    RUN_ROOT = STATE_ROOT / "outputs/engineering" / RUN_ID
    print("F06 direview:", F06_RUN)
    print("Paket F07 SHA256:", BUNDLE_SHA256)
    print("Hasil F07:", RUN_ROOT)
    ''')
    cell('markdown','''\
    ## Source, dependensi dan pengujian F01-F07

    Suite mencakup nilai/subgradien fractional CVaR, kelas hilang, generator
    tunggal/tidak terverifikasi/held-out, pembekuan bobot, risk gradient ke router,
    projected dual, head lokalisasi bersyarat dan pipeline B4/B5 kecil.
    Log tes, pip freeze serta receipt disimpan pada Drive.
    ''')
    setup=''.join(previous['cells'][7]['source'])
    setup=setup.replace('configs/models/rsvt_f06.toml','configs/training/risk_synthetic.toml')
    setup=setup.replace('phase6','phase7').replace('"F06"','"F07"')
    setup=setup.replace('"reviewed_f05_run": str(F05_RUN)','"reviewed_f06_run": str(F06_RUN)')
    cell('code',setup)
    cell('markdown','''\
    ## Jalankan tahap B4 dan B5

    Logical batch: 8 kasus, dua per kelas TB/TM/FB/FM. Enam mask voxel sintetis
    terverifikasi; point/box tidak dipakai untuk loss segmentasi. Alpha CVaR 0,9
    dengan dua contoh per kelas berarti tail estimator menjadi maksimum kelas;
    flag `low_support` dan `max_only` harus terbaca sebagai batas fixture ini.

    B4: baseline 1 step, lalu risk fine-tune 2 step; router tetap beku dan dense.
    B5: baseline 1 step, warmup router 1 step (backbone beku), lalu risk fine-tune
    2 step. Keduanya mulai dari bobot/kasus identik; jumlah update berbeda.
    Ini pemeriksaan engineering, bukan perbandingan performa yang terkontrol.

    Constraint soft memakai maksimum expected sequence count dalam batch.
    Multiplier diupdate dengan projected ascent. Training soft tetap dense;
    tidak ada jaminan konvergensi budget setelah tiga update dual. Inference B5
    wajib memenuhi cap 64 termasuk CLS. Cap ini bukan latency ms atau FLOPs total.
    ''')
    cell('code','''\
    command = [sys.executable, "-u", str(CODE_ROOT / "scripts/project.py"), "phase7-risk",
               "--config", str(CONFIG_PATH), "--output-dir", str(RUN_ROOT / "risk"), "--device", DEVICE]
    with (RUN_ROOT / "phase7_console.log").open("w") as log:
        with subprocess.Popen(command, cwd=CODE_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            exit_code = process.wait()
    receipt.update(risk_exit_code=exit_code)
    (RUN_ROOT / "notebook_receipt.json").write_text(json.dumps(receipt, indent=2))
    summary_path = RUN_ROOT / "risk/summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Summary belum terbentuk; baca {RUN_ROOT / 'phase7_console.log'} dan history JSON")
    summary = json.loads(summary_path.read_text())
    print("Checks:", json.dumps(summary["checks"], indent=2))
    if exit_code or summary.get("ok") is not True:
        raise RuntimeError("Pemeriksaan F07 belum lolos; periksa checks/history pada Drive.")
    ''')
    cell('markdown','''\
    ## Membaca komponen loss dan batas budget

    CVaR dan variance generator dihitung dari CE mentah per kasus. Class weighting
    diterapkan pada base CE; bobot FB/FM CVaR diterapkan terpisah satu kali.
    Generator variance dihitung dalam FB/FM masing-masing, dengan grup training
    terverifikasi dan allowlist. Kasus satu generator wajib menghasilkan penalty nol.

    Hasil hard localization hanya valid pada support fine token terpilih.
    Voxel lainnya berstatus unknown melalui `spatial_valid_mask=false`.
    Coarse tetap memberi konteks global. F07 menambahkan biaya head/scatter yang
    tidak termasuk estimasi operator F06; notebook ini tidak mengklaim latensi baru.
    ''')
    cell('code','''\
    from IPython.display import Markdown, display

    table = ["| Varian / tahap | Step | CE | CVaR | Generator var | Lokalisasi | Constraint g | Dual sesudah |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, variant in summary["variants"].items():
        for row in variant["history"]:
            table.append(f"| {name} / {row['stage']} | {row['stage_step']} | {row['classification']:.4f} | "
                         f"{row['cvar']['loss']:.4f} | {row['generator']['loss']:.6g} | "
                         f"{row['localization']['loss']:.4f} | {row['budget']['constraint']:.4f} | "
                         f"{row['dual_after']['multiplier']:.4f} |")
        print(name, "inference:", json.dumps(variant["inference"], indent=2))
        print(name, "CVaR terakhir:", json.dumps(variant["history"][-1]["cvar"], indent=2))
        print(name, "gradien risiko ke router:", json.dumps(variant["history"][-1]["risk_gradients_to_router"], indent=2))
    display(Markdown("\\n".join(table)))
    print("F07 engineering:", "LOLOS" if summary["ok"] else "PERLU DIPERIKSA")
    print("Medical training ready:", summary["medical_training_ready"])
    print("Project multi-generator ready:", summary["project_multi_generator_ready"])
    print("Hasil di Drive:", RUN_ROOT)
    ''')
    cell('markdown','''\
    ## Setelah run

    `outputs/engineering/f07_<UTC>/risk/` berisi `summary.json`, resolved config,
    dan history masing-masing B4/B5. Receipt, versi dependensi, log tes dan console
    berada di root run. Tidak ada checkpoint pretrained, training medis atau
    penggunaan test penelitian. Jika terputus, ulangi konfigurasi untuk run baru.

    Hasil direview sebelum F08 orkestrasi eksperimen dan penguncian protokol.
    Source-validation alpha/bobot/budget, data multi-generator aktual, pretraining,
    baseline/QA F04 dan perbandingan medis tetap terbuka.
    ''')
    for index,item in enumerate(cells):item['id']=f'f07-{index:02d}'
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    output=legacy_output(ROOT,'07_phase7_colab.ipynb')
    output.write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return output


if __name__=='__main__':print(build())
