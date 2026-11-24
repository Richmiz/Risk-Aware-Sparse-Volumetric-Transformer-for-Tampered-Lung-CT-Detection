"""Build the user-run source audit and a pinned code/public-metadata bundle."""

from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
from textwrap import dedent
import zipfile

from build_colab_bundle import build_bundle
from notebook_layout import legacy_output

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_NAME = "data_label_verification_colab.ipynb"


def build():
    package = ROOT / "artifacts/rsvt-data-verification-code.zip"
    staging = ROOT / "artifacts/rsvt-data-verification-staging.zip"
    build_bundle(ROOT, staging)
    with zipfile.ZipFile(staging) as source, zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as destination:
        for info in source.infolist():
            if info.filename != "notebooks/" + NOTEBOOK_NAME:
                destination.writestr(info.filename, source.read(info.filename))
        for name in ("data.csv", "LIDC.csv", "sets.csv", "centers.csv", "LICENSE.md"):
            destination.write(ROOT / "data/raw/m3dsynth/metadata" / name, "reference_metadata/m3dsynth/" + name)
        destination.write(ROOT / "reports/supplementary/data_label_audit_20261111/tcia_lidc_series.json",
                          "reference_metadata/tcia_lidc_series.json")
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    embedded_package = base64.b64encode(package.read_bytes()).decode("ascii")
    package.with_suffix(package.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    cells = []

    def cell(kind, source):
        item = {"cell_type": kind, "id": "data-audit-" + str(len(cells)), "metadata": {},
                "source": dedent(source).strip().splitlines(keepends=True)}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", '''\
    # Verifikasi data, label, dan sumber scan

    Jalankan sendiri dengan **Run All di Google Colab melalui VS Code**.
    Audit ini tidak melatih model, tidak menghitung performa test, dan tidak
    mengubah data, label, checkpoint, atau partisi F03. CPU cukup.

    Seluruh 100 scan diperiksa untuk integritas dan sumber, termasuk piksel test
    untuk QA data saja. Tidak ada prediksi atau visualisasi lesi test.
    Kode dan metadata publik sudah tertanam di notebook; tidak perlu menyalin
    paket kode secara manual. Notebook dan paket rilis disimpan di Drive proyek.
    Kode berjalan pada VM ketika Anda menjalankan sel dengan kernel Colab.

    Semua laporan dan receipt ditulis otomatis ke Google Drive proyek pada
    `outputs/data_verification/`. ZIP laporan dibuat di Drive, dibaca ulang,
    dan seluruh checksum serta 100 laporan kasus diperiksa sebelum pesan sukses.
    Tidak diperlukan unduhan/upload manual untuk review. Raw CT tetap pada
    Colab/Drive; paket laporan tidak memuat citra, checkpoint, atau kredensial.

    Audit metadata sebelumnya: label 100/100 cocok dengan penerbit; pasangan
    6031 dan 7507 berbagi sumber. Ada kandidat geometri lain yang belum boleh
    dianggap identik. Blind trial memiliki 17.457 slice pada arsip proyek,
    sedangkan README menyebut 17.941. Audit ZIP membedakan kehilangan saat
    ekstraksi dari ketidaksesuaian dokumentasi/arsip. Folder Drive asli penulis
    tidak dapat diakses pada audit 11 November 2026.

    Identitas CT-GAN hanya dihubungkan ke LIDC bila UID seri publik cocok tepat.
    Identitas yang tidak cocok tetap kosong; kemiripan tidak otomatis membuktikan
    identitas pasien. Metadata M3Dsynth tidak dianggap sebagai citra tersedia.
    ''')
    cell("code", '''\
    import importlib.util, platform
    from pathlib import Path
    if platform.system() != 'Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Pilih kernel Google Colab; audit piksel tidak dijalankan pada laptop.')
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    ''')
    payload_code = (
        "# Paket kode + metadata publik. Dibuka pada VM saat Run All; tidak diunggah ke Drive.\n"
        "import base64\nEMBEDDED_PACKAGE_BASE64 = (\n"
        + "\n".join("    " + repr(embedded_package[i:i+120]) for i in range(0, len(embedded_package), 120))
        + "\n)\nCODE_PACKAGE_BYTES = base64.b64decode(EMBEDDED_PACKAGE_BASE64, validate=True)\n"
        + "del EMBEDDED_PACKAGE_BASE64\n"
    )
    cell("code", payload_code)
    cells[-1]["metadata"] = {"jupyter": {"source_hidden": True}}
    cell("code", f'''\
    import hashlib, json, sys, subprocess, zipfile
    from io import BytesIO
    from datetime import datetime, timezone
    STATE_ROOT = Path('/content/drive/MyDrive/Research/LUNG-CT-SCAN')
    EXPECTED_PACKAGE_SHA256 = '{digest}'
    if hashlib.sha256(CODE_PACKAGE_BYTES).hexdigest() != EXPECTED_PACKAGE_SHA256:
        raise RuntimeError('SHA paket audit berbeda dari rilis yang direview.')
    CODE_ROOT = Path('/content/rsvt-data-verification') / EXPECTED_PACKAGE_SHA256[:12]
    CODE_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(CODE_PACKAGE_BYTES)) as package:
        for name in package.namelist():
            target = (CODE_ROOT / name).resolve()
            if not target.is_relative_to(CODE_ROOT.resolve()):
                raise ValueError('Unsafe bundle member')
        package.extractall(CODE_ROOT)
    del CODE_PACKAGE_BYTES
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(CODE_ROOT)+'[preprocessing]'], check=True)
    sys.path.insert(0, str(CODE_ROOT / 'src'))
    F03 = STATE_ROOT / 'outputs/partitions/f03_20261110T013617_031412Z'
    F01 = STATE_ROOT / 'outputs/inventory/f01_20261109T133401_906932Z'
    F03_MANIFEST = F03 / 'manifest.draft.json'
    if hashlib.sha256(F03_MANIFEST.read_bytes()).hexdigest() != '76b75d0235bd565b2e1a18f1099d6af1b6b181b81792621632fbe54e8d4017ff':
        raise RuntimeError('Manifest historis F03 berbeda; hentikan audit dan periksa perubahan.')
    if hashlib.sha256((F01/'files/files.csv').read_bytes()).hexdigest() != 'f4d9f9801c042af25e40aeb67b2e2ded355511b84b49764abd59f8eb0232a563':
        raise RuntimeError('Inventaris F01 berbeda dari bukti yang direview.')
    RESULTS_ROOT = STATE_ROOT / 'outputs/data_verification'
    RUN = RESULTS_ROOT / datetime.now(timezone.utc).strftime('audit_%Y%m%dT%H%M%S_%fZ')
    RUN.mkdir(parents=True)
    print('Output Google Drive:', RUN)
    ''')
    cell("markdown", '''\
    ## Pemeriksaan alat dan pemulihan sumber yang sudah tersimpan

    Tes hanya memakai fixture buatan. Bila sumber CT tidak lagi ada pada VM,
    arsip **yang sudah ada di Drive** dipulihkan ke disk Colab. Tidak ada
    unduhan dataset baru. Pemulihan dapat memerlukan sekitar 20 GB ruang VM
    untuk ZIP bagian dalam dan hasil ekstraksinya; artefak lama dipertahankan.
    ''')
    cell("code", '''\
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(CODE_ROOT/'tests'), '-p', 'test_data_verification.py', '-v'], check=True)
    from rsvt.data.recovery import locate_ctgan_dataset, restore_ctgan_archive
    from rsvt.data.acquisition import sha256_file
    from rsvt.data.verification import run_metadata_verification
    from rsvt.data.source_verification import run_source_verification
    CT_ROOT = locate_ctgan_dataset(Path('/content/rsvt-data/raw/ct_gan/uci/dataset'), STATE_ROOT)
    previous_root = Path('/content/rsvt-source-restore/f02bulk_20261109T205406_040226Z/uci/dataset')
    if CT_ROOT is None:
        CT_ROOT = locate_ctgan_dataset(previous_root, STATE_ROOT)
    ACQUIRED_ARCHIVE = STATE_ROOT/'datasets/ct_gan/ct_gan_uci.zip'
    if CT_ROOT is None:
        if not ACQUIRED_ARCHIVE.is_file():
            raise FileNotFoundError('Arsip sumber yang sudah disimpan belum ditemukan; tidak memulai unduhan baru.')
        if sha256_file(ACQUIRED_ARCHIVE) != 'e88c990440e3f0ddf5e36928e42e6a5231b685ddce5c18bfccce9d513967a135':
            raise RuntimeError('Arsip berbeda dari fingerprint akuisisi yang direview.')
        restored = restore_ctgan_archive(ACQUIRED_ARCHIVE, Path('/content/rsvt-verification-source')/RUN.name, download_budget_bytes=0, progress=print)
        CT_ROOT = Path(restored['dataset_root'])
    INNER_ARCHIVE = CT_ROOT.parent/'data.zip'
    ZIP_TO_CHECK = INNER_ARCHIVE if INNER_ARCHIVE.is_file() else ACQUIRED_ARCHIVE if ACQUIRED_ARCHIVE.is_file() else None
    print('Raw CT root:', CT_ROOT)
    ''')
    cell("code", '''\
    REFERENCES = CODE_ROOT/'reference_metadata'
    metadata = run_metadata_verification(
        manifest_path=F03_MANIFEST, files_path=F01/'files/files.csv',
        series_path=F01/'files/series_candidates.csv', overlap_path=F01/'files/overlap.json',
        label_root=CT_ROOT/'Tampered Scans', readme_path=CT_ROOT/'Readme.docx',
        m3_metadata_dir=REFERENCES/'m3dsynth', tcia_catalog_path=REFERENCES/'tcia_lidc_series.json',
        policy_path=CODE_ROOT/'configs/data/ctgan_verification.json',
        output_dir=RUN/'metadata')
    print(json.dumps({'checks': metadata['checks'], 'ctgan': metadata['ctgan'], 'm3dsynth': metadata['m3dsynth']}, indent=2))
    ''')
    cell("markdown", '''\
    ## Perbandingan piksel dan sumber

    Seluruh DICOM dibandingkan SHA-nya dengan F01 lalu didekode. Slice HU native
    informatif dibandingkan melalui hash; slice konstan/padding tidak membentuk
    hubungan sumber. Tiga slice berbeda yang sama merupakan bukti sumber untuk
    review; kandidat thumbnail tetap dipisahkan dari identitas terkonfirmasi.
    Tidak ada hasil medis, skor model, mask lesi, atau tuning test dibuat.
    ''')
    cell("code", '''\
    source = run_source_verification(
        dataset_root=CT_ROOT, manifest_path=F03_MANIFEST, files_path=F01/'files/files.csv',
        catalog_path=REFERENCES/'tcia_lidc_series.json', metadata_summary_path=RUN/'metadata/summary.json',
        m3_metadata_dir=REFERENCES/'m3dsynth', output_dir=RUN/'source_pixels',
        inner_archive=ZIP_TO_CHECK, progress=print)
    print(json.dumps(source, indent=2))
    receipt = {'bundle_sha256': EXPECTED_PACKAGE_SHA256, 'F03_manifest_sha256': sha256_file(F03_MANIFEST),
               'metadata_summary_sha256': sha256_file(RUN/'metadata/summary.json'),
               'source_summary_sha256': sha256_file(RUN/'source_pixels/summary.json'),
               'metadata_checks_passed': metadata['metadata_checks_passed'],
               'all_cases_pixel_checked': source['all_cases_pixel_checked'],
               'model_loaded': False, 'training_performed': False, 'test_performance_evaluated': False,
               'training_ready': False, 'goal_verification_complete': False}
    (RUN/'notebook_receipt.json').write_text(json.dumps(receipt, indent=2)+'\\n')
    print('SELESAI EKSEKUSI AUDIT:', RUN)
    ''')
    cell("code", '''\
    from rsvt.data.source_verification import package_review_outputs
    import runpy
    REVIEW_ZIP = RESULTS_ROOT / (RUN.name + '.review.zip')
    if not RUN.resolve().is_relative_to((STATE_ROOT/'outputs/data_verification').resolve()):
        raise RuntimeError('Hasil audit wajib disimpan pada Google Drive proyek.')
    if REVIEW_ZIP.resolve().parent != (STATE_ROOT/'outputs/data_verification').resolve():
        raise RuntimeError('ZIP hasil wajib berada pada Google Drive proyek.')
    review_package = package_review_outputs(RUN, REVIEW_ZIP)
    reader = runpy.run_path(str(CODE_ROOT/'scripts/review_data_verification_return.py'))
    persisted_review = reader['review_archive'](REVIEW_ZIP, EXPECTED_PACKAGE_SHA256)
    if sha256_file(REVIEW_ZIP) != review_package['archive_sha256']:
        raise RuntimeError('ZIP yang dibaca ulang dari Drive berbeda dari paket yang ditulis.')
    persistence_receipt = {
        'run_id': RUN.name, 'results_directory': str(RUN), 'review_ZIP': str(REVIEW_ZIP),
        'review_ZIP_sha256': review_package['archive_sha256'],
        'code_bundle_sha256': EXPECTED_PACKAGE_SHA256,
        'all_report_checksums_read_back': persisted_review['artifact_integrity_verified'],
        'verified_case_count': persisted_review['reported_CT_volumes_pixel_checked'],
        'verified_DICOM_count': persisted_review['reported_DICOM_files_checked'],
        'saved_to_Google_Drive': True, 'manual_transfer_required': False,
        'training_ready': False, 'goal_verification_complete': False}
    PERSISTENCE_RECEIPT = RUN/'drive_persistence_receipt.json'
    PERSISTENCE_RECEIPT.write_text(json.dumps(persistence_receipt, indent=2)+'\\n')
    if json.loads(PERSISTENCE_RECEIPT.read_text()) != persistence_receipt:
        raise RuntimeError('Receipt penyimpanan Drive gagal dibaca ulang.')
    print('TERSIMPAN DAN TERVERIFIKASI DI GOOGLE DRIVE:', REVIEW_ZIP)
    print('SHA256 ZIP:', persistence_receipt['review_ZIP_sha256'])
    print('Receipt Drive:', PERSISTENCE_RECEIPT)
    print('Tidak perlu mengunduh atau mengunggah file secara manual untuk review.')
    ''')
    cell("markdown", '''\
    ## Review hasil

    Setelah Run All selesai, laporan, ZIP, dan `drive_persistence_receipt.json`
    sudah berada di Google Drive proyek, pada `outputs/data_verification/`.
    Cukup beri tahu run sudah selesai; hasil dapat direview langsung dari Drive.
    Unduhan lokal bersifat opsional. ZIP hanya berisi laporan metadata dan hash.
    Laporan perlu direview sebelum perubahan partisi atau kesiapan training.
    Jika ada `cross_partition_source_evidence=true`, validation lama berpotensi
    berbagi sumber dengan train/test dan tidak boleh dianggap independen.

    Identitas publik yang tidak terpetakan memerlukan pemetaan penerbit atau
    pencocokan terhadap sumber LIDC asli. Perbandingan ZIP hanya membuktikan
    kelengkapan terhadap arsip yang diperoleh, bukan kelengkapan Drive penulis.
    Konvensi titik RadiAnt/CT-GAN tetap perlu konfirmasi pada citra/anotasi.
    Karena itu keberhasilan eksekusi tidak otomatis menutup verifikasi ilmiah.
    ''')
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    target = legacy_output(ROOT, NOTEBOOK_NAME)
    if target.is_file():
        previous_bytes = target.read_bytes()
        previous_notebook = json.loads(previous_bytes)
        if any(c.get('outputs') or c.get('execution_count') is not None for c in previous_notebook['cells']):
            backup = ROOT/'reports/supplementary/data_label_audit_20261111' / (
                'executed_notebook_' + hashlib.sha256(previous_bytes).hexdigest()[:12] + '.ipynb')
            if not backup.is_file():
                backup.write_bytes(previous_bytes)
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    delivery_path = ROOT/'reports/supplementary/data_label_audit_20261111/delivery.json'
    previous = json.loads(delivery_path.read_text(encoding="utf-8")) if delivery_path.is_file() else {}
    result = {**previous, "notebook": str(target), "bundle": str(package), "bundle_sha256": digest,
              "bundle_bytes": package.stat().st_size, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_only_plus_explicit_public_metadata": True, "medical_pixels_packaged": False,
              "remote_executed": False, "drive_upload_succeeded": False,
              "drive_upload_status": "pending_current_release_upload", "user_delivery_preference": "Google_Drive_automatic",
              "cloud_upload_authorized": True, "embedded_code_bundle": True,
              "requires_code_ZIP_in_Drive": False, "audit_creates_files_on_Drive": True,
              "runtime_results_location": "/content/drive/MyDrive/Research/LUNG-CT-SCAN/outputs/data_verification",
              "results_return_method": "read_verified_persisted_Google_Drive_reports",
              "manual_transfer_required": False}
    delivery_path.write_text(json.dumps(result, indent=2)+'\n', encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
