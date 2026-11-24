"""Generate the source-only F05 Colab notebook. Does not execute remote cells."""

import json
from pathlib import Path
from textwrap import dedent
from notebook_layout import legacy_output

ROOT=Path(__file__).resolve().parents[1]


def build():
    cells=[]
    def cell(kind,source):
        item={"cell_type":kind,"metadata":{},"source":dedent(source).strip().splitlines(keepends=True)}
        if kind=="code":item.update(execution_count=None,outputs=[])
        cells.append(item)

    cell("markdown",'''\
    # F05 - Fondasi dense RSVT: pemeriksaan engineering sintetis

    Jalankan sendiri dengan **kernel Google Colab melalui VS Code**, CPU atau GPU.
    Paket kode sudah berada di folder Drive proyek. Dataset medis tidak dibaca
    pada tahap ini. Tidak perlu mengunduh dataset ke komputer atau mengulang F01-F04.

    Pemeriksaan membangun coarse encoder, kandidat token multi-skala, posisi fisik,
    dense Transformer, classification head empat kelas, dan localization head.
    Empat volume prosedural dipakai untuk satu forward/backward dan optimizer step.
    Label/mask sintetis hanya menguji kontrak. Hasil bukan metrik performa medis.

    **F06 routing belum aktif. F04 optimisasi CNN dan QA penelitian tetap terbuka.**
    ''')
    cell("code",'''\
    import importlib.util
    from pathlib import Path
    import platform
    import sys

    try:
        has_colab = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        has_colab = False
    if platform.system() != "Linux" or not Path("/content").is_dir() or not has_colab:
        raise RuntimeError("Pilih kernel Google Colab remote melalui VS Code.")
    print("Colab Python:", sys.version.split()[0])
    ''')
    cell("markdown",'''\
    ## Hubungkan Google Drive

    Selesaikan izin akun Google ketika browser/VS Code memintanya. Mount yang
    sudah aktif dipakai kembali. Folder proyek berada di
    `MyDrive/Research/LUNG-CT-SCAN` pada akun yang sama.
    ''')
    cell("code",'''\
    from google.colab import drive

    DRIVE_MOUNT = Path("/content/drive")
    MY_DRIVE = DRIVE_MOUNT / "MyDrive"
    if MY_DRIVE.is_dir():
        print("Google Drive sudah terhubung:", MY_DRIVE)
    else:
        print("Selesaikan izin Google pada browser/VS Code menggunakan akun Drive proyek.")
        try:
            drive.mount(str(DRIVE_MOUNT))
        except (ValueError, RuntimeError) as exc:
            print("Mount gagal:", type(exc).__name__, str(exc))
            print("Di VS Code: Ctrl+Shift+P > Colab: Sign Out, pilih kernel Colab dan login kembali.")
            print("Pastikan akun benar, lalu ulangi sel ini. Pemeriksaan F05 belum dimulai.")
            raise
        if not MY_DRIVE.is_dir():
            raise RuntimeError("Mount belum menghasilkan folder MyDrive yang dapat dibaca.")
        print("Drive dapat dibaca:", MY_DRIVE)
    ''')
    cell("markdown",'''\
    ## Konfigurasi dan run baru

    Default `DEVICE="auto"` memakai CUDA jika tersedia, atau CPU. GPU tidak wajib.
    Konfigurasi lengkap ada di `configs/models/rsvt_f05.toml` dalam paket F05.
    Untuk mengulang smoke check, ulangi sel konfigurasi ini agar membuat run baru.
    ''')
    cell("code",'''\
    from datetime import datetime, timezone
    import hashlib
    import json

    STATE_ROOT = Path("/content/drive/MyDrive/Research/LUNG-CT-SCAN")
    CODE_ARCHIVE = STATE_ROOT / "rsvt-colab-f05-code.zip"
    DEVICE = "auto"  # "auto", "cpu", atau "cuda"
    if DEVICE not in {"auto", "cpu", "cuda"}:
        raise ValueError("DEVICE tidak dikenal.")
    if not CODE_ARCHIVE.is_file():
        raise FileNotFoundError(f"Paket F05 tidak ditemukan pada akun/path Drive ini: {CODE_ARCHIVE}")
    BUNDLE_SHA256 = hashlib.sha256(CODE_ARCHIVE.read_bytes()).hexdigest()
    CODE_ROOT = Path("/content/rsvt-f05") / BUNDLE_SHA256[:12]
    RUN_ID = datetime.now(timezone.utc).strftime("f05_%Y%m%dT%H%M%S_%fZ")
    RUN_ROOT = STATE_ROOT / "outputs/engineering" / RUN_ID
    print("Paket kode SHA256:", BUNDLE_SHA256)
    print("Hasil pemeriksaan sintetis:", RUN_ROOT)
    ''')
    cell("markdown",'''\
    ## Muat source dan periksa implementasi

    Ekstraksi hanya memuat source pada runtime Colab. Pengujian F01-F05 menggunakan
    fixture sintetis, termasuk forward/backward grid 96 cubed. Log dan receipt
    disimpan ke Drive. Versi dependensi aktual dicatat di `pip_freeze.txt`.
    ''')
    cell("code",'''\
    import stat
    import subprocess
    import zipfile

    marker = CODE_ROOT / ".bundle_sha256"
    if CODE_ROOT.exists():
        if not marker.is_file() or marker.read_text().strip() != BUNDLE_SHA256:
            raise RuntimeError("Direktori kode tidak cocok dengan fingerprint paket.")
    else:
        with zipfile.ZipFile(CODE_ARCHIVE) as archive:
            for item in archive.infolist():
                target = (CODE_ROOT / item.filename).resolve()
                if not target.is_relative_to(CODE_ROOT.resolve()) or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Path paket kode tidak aman.")
            CODE_ROOT.mkdir(parents=True)
            archive.extractall(CODE_ROOT)
        marker.write_text(BUNDLE_SHA256 + "\\n")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", f"{CODE_ROOT}[inventory,training]"], check=True)
    CONFIG_PATH = CODE_ROOT / "configs/models/rsvt_f05.toml"
    resolved = subprocess.run([sys.executable, "-c",
        "import json,sys; from rsvt.experiments.phase5 import load_phase5_config; "
        "print(json.dumps(load_phase5_config(sys.argv[1])))", str(CONFIG_PATH)],
        cwd=CODE_ROOT, capture_output=True, text=True, check=True)
    print("Konfigurasi efektif:", json.dumps(json.loads(resolved.stdout), indent=2))
    import torch
    runtime = {"python": sys.version, "torch": str(torch.__version__),
               "cuda_available": torch.cuda.is_available(), "torch_cuda_build": torch.version.cuda,
               "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print("Perangkat aktual:", json.dumps(runtime, indent=2))
    if DEVICE == "cuda" and not runtime["cuda_available"]:
        raise RuntimeError("CUDA dipilih tetapi tidak tersedia; pilih DEVICE=auto/cpu lalu ulangi konfigurasi.")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    receipt = {"phase": "F05", "created_utc": datetime.now(timezone.utc).isoformat(),
               "bundle_sha256": BUNDLE_SHA256, "bundle_bytes": CODE_ARCHIVE.stat().st_size,
               "runtime": runtime, "requested_device": DEVICE, "medical_training_ready": False}
    (RUN_ROOT / "notebook_receipt.json").write_text(json.dumps(receipt, indent=2))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (RUN_ROOT / "pip_freeze.txt").write_text(freeze.stdout)
    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s",
                            str(CODE_ROOT / "tests"), "-p", "test_f0*.py", "-v"],
                            cwd=CODE_ROOT, capture_output=True, text=True)
    (RUN_ROOT / "synthetic_tests.log").write_text(tests.stdout + tests.stderr)
    receipt.update(synthetic_tests_exit_code=tests.returncode)
    (RUN_ROOT / "notebook_receipt.json").write_text(json.dumps(receipt, indent=2))
    print((tests.stdout + tests.stderr)[-4000:])
    if tests.returncode:
        raise RuntimeError(f"Pengujian gagal; baca {RUN_ROOT / 'synthetic_tests.log'}")
    ''')
    cell("markdown",'''\
    ## Jalankan pemeriksaan integrasi F05

    Volume default memiliki shape Z/Y/X 24/32/40, 32/40/48, 40/32/24, dan 48/48/48.
    Batch mendapat right padding; validity mask tetap terpisah. Affine meliputi
    spacing anisotropik, rotasi, flip, dan translasi. Tiga mask sintetis ditandai
    terverifikasi; target NaN pada kasus lainnya harus diabaikan oleh loss.

    Sel ini melakukan **satu optimizer step sintetis**, tanpa training/validation
    medis, checkpoint pretrained, routing sparse, atau loss risiko.
    ''')
    cell("code",'''\
    command = [sys.executable, "-u", str(CODE_ROOT / "scripts/project.py"), "phase5-smoke",
               "--config", str(CONFIG_PATH), "--output-dir", str(RUN_ROOT / "smoke"), "--device", DEVICE]
    completed = subprocess.run(command, cwd=CODE_ROOT, capture_output=True, text=True)
    (RUN_ROOT / "phase5_console.log").write_text(completed.stdout + completed.stderr)
    receipt.update(smoke_exit_code=completed.returncode)
    (RUN_ROOT / "notebook_receipt.json").write_text(json.dumps(receipt, indent=2))
    summary_path = RUN_ROOT / "smoke/summary.json"
    if not summary_path.is_file():
        print((completed.stdout + completed.stderr)[-6000:])
        raise RuntimeError(f"Summary belum terbentuk; baca {RUN_ROOT / 'phase5_console.log'}")
    summary = json.loads(summary_path.read_text())
    print(json.dumps(summary, indent=2))
    if completed.returncode or summary.get("ok") is not True:
        raise RuntimeError("Pemeriksaan F05 gagal; periksa checks dan log di Drive.")
    ''')
    cell("markdown",'''\
    ## Membaca hasil

    Sukses jika `ok=true` dan seluruh `checks` bernilai true: loss finite, semua
    kelompok modul menerima gradien finite/nonzero, coarse mencakup setiap voxel
    valid tepat satu kali, gradien image invalid nol, tiga mask eligible, dan
    bobot classifier berubah setelah optimizer step.

    `attention_pair_slots` adalah jumlah slot attention dense, bukan FLOPs atau
    pengukuran kecepatan. `saved_attention_slots=0` karena F06 belum dibuat.
    Logit lokalisasi konstan pada support patch fine; tidak mengklaim presisi
    subpatch. Fitur intensitas deterministik belum divalidasi sebagai detektor forensik.

    Hasil tersimpan pada `outputs/engineering/f05_.../` di Drive: receipt, versi
    dependensi, log tes, log smoke, `smoke/summary.json`, dan resolved config.
    Setelah selesai, hasil dapat direview sebelum implementasi F06.
    ''')
    cell("code",'''\
    print("F05 engineering:", "LOLOS" if summary["ok"] else "PERLU DIPERIKSA")
    print("Perangkat smoke:", summary["environment"]["device"])
    print("Bentuk keluaran:", summary["shapes"])
    print("Token aktif per kasus:", summary["active_token_counts"])
    print("Status medical_training_ready:", summary["medical_training_ready"])
    print("Hasil di Drive:", RUN_ROOT)
    ''')
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
    # Stable cell ids satisfy notebook schema without storing editor/session data.
    for index,item in enumerate(cells):item["id"]=f"f05-{index:02d}"
    output=legacy_output(ROOT,"05_phase5_colab.ipynb")
    output.write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return output


if __name__=="__main__":print(build())
