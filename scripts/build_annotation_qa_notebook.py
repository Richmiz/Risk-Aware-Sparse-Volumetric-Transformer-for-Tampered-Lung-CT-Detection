"""Build a separate Colab QA release; preserve historical audit notebooks/runs."""

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from textwrap import dedent
import zipfile

from build_colab_bundle import build_bundle
from notebook_layout import legacy_output

ROOT = Path(__file__).resolve().parents[1]


def build():
    staging = ROOT / "artifacts/rsvt-annotation-qa-staging.zip"
    bundle = ROOT / "artifacts/rsvt-annotation-qa-code.zip"
    build_bundle(ROOT, staging)
    with zipfile.ZipFile(staging) as source, zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source.namelist():
            if not name.startswith("notebooks/"):
                archive.writestr(name, source.read(name))
        archive.write(ROOT / "reports/supplementary/data_label_audit_20261111/tcia_lidc_series.json", "reference_metadata/tcia_lidc_series.json")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    bundle.with_suffix(".zip.sha256").write_text(digest+"\n", encoding="utf-8")
    encoded = base64.b64encode(bundle.read_bytes()).decode("ascii")
    cells = []
    def cell(kind, text):
        item = {"cell_type":kind,"id":f"annotation-qa-{len(cells)}","metadata":{},"source":dedent(text).strip().splitlines(keepends=True)}
        if kind == "code": item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell("markdown", '''
    # QA identitas sumber dan koordinat anotasi — CT-GAN

    **Run All dengan kernel Google Colab melalui VS Code. CPU cukup.**
    Notebook ini melanjutkan audit terverifikasi; tidak mengulang training.
    Semua hasil otomatis ke `MyDrive/Research/LUNG-CT-SCAN/outputs/source_annotation_qa/`.
    Tidak diperlukan download/upload manual. Kode sudah tertanam.

    Pemeriksaan header memakai slice pertama, tengah, dan terakhir tiap kasus,
    dengan SHA dari F01. UID seri/studi pada referensi DICOM dicari pada katalog
    publik; kecocokan jumlah slice/SeriesNumber hanya menjadi kandidat.
    Header tidak membuktikan kesamaan piksel ataupun independensi pasien.

    Sebanyak **123 titik train/validation** diperlihatkan dengan 8 alternatif
    urutan X/Y dan indeks. **25 titik test dikecualikan**; piksel test tidak
    didekode. Koordinat diubah menjadi LPS mm mengikuti geometri DICOM native,
    tanpa pemilihan transform otomatis atau pembuatan mask lesi.

    Citra preview mengandung data CT dan tetap di Colab/Drive. Paket review
    metadata tidak mengandung citra. Hasil tetap memerlukan review manusia.
    Identitas yang belum terbukti dan selisih README 484 slice tetap terbuka.
    ''')
    cell("code", '''
    import importlib.util, platform
    from pathlib import Path
    if platform.system() != 'Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Pilih kernel Google Colab; citra CT tidak diproses di laptop.')
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    ''')
    cell("code", "import base64\nPACKAGE_BYTES=base64.b64decode(\n" + "\n".join("    " + repr(encoded[i:i+120]) for i in range(0,len(encoded),120)) + "\n, validate=True)\n")
    cells[-1]["metadata"]={"jupyter":{"source_hidden":True}}
    cell("code", f'''
    import hashlib, json, sys, subprocess, zipfile
    from io import BytesIO
    from datetime import datetime, timezone
    EXPECTED_PACKAGE_SHA256 = '{digest}'
    if hashlib.sha256(PACKAGE_BYTES).hexdigest() != EXPECTED_PACKAGE_SHA256:
        raise RuntimeError('Code package checksum mismatch.')
    CODE_ROOT = Path('/content/rsvt-annotation-qa') / EXPECTED_PACKAGE_SHA256[:12]
    CODE_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(PACKAGE_BYTES)) as archive:
        for name in archive.namelist():
            if not (CODE_ROOT/name).resolve().is_relative_to(CODE_ROOT.resolve()):
                raise ValueError('Unsafe package member')
        archive.extractall(CODE_ROOT)
    del PACKAGE_BYTES
    subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(CODE_ROOT)+'[preprocessing]'], check=True)
    sys.path.insert(0,str(CODE_ROOT/'src'))
    STATE_ROOT=Path('/content/drive/MyDrive/Research/LUNG-CT-SCAN')
    RESULTS_ROOT=STATE_ROOT/'outputs/source_annotation_qa'
    RUN=RESULTS_ROOT/datetime.now(timezone.utc).strftime('qa_%Y%m%dT%H%M%S_%fZ')
    F03=STATE_ROOT/'outputs/partitions/f03_20261110T013617_031412Z/manifest.draft.json'
    F01_FILES=STATE_ROOT/'outputs/inventory/f01_20261109T133401_906932Z/files/files.csv'
    from rsvt.data.acquisition import sha256_file
    for path,expected in ((F03,'76b75d0235bd565b2e1a18f1099d6af1b6b181b81792621632fbe54e8d4017ff'),
                          (F01_FILES,'f4d9f9801c042af25e40aeb67b2e2ded355511b84b49764abd59f8eb0232a563')):
        if sha256_file(path)!=expected:
            raise RuntimeError('Historical input changed: '+path.name)
    print('Output Drive:', RUN)
    ''')
    cell("code", '''
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(CODE_ROOT/'tests'),'-p','test_annotation_qa.py','-v'],check=True)
    from rsvt.data.recovery import locate_ctgan_dataset, restore_ctgan_archive
    CT_ROOT=locate_ctgan_dataset(Path('/content/rsvt-data/raw/ct_gan/uci/dataset'),STATE_ROOT)
    if CT_ROOT is None:
        CT_ROOT=locate_ctgan_dataset(Path('/content/rsvt-source-restore/f02bulk_20261109T205406_040226Z/uci/dataset'),STATE_ROOT)
    if CT_ROOT is None:
        ARCHIVE=STATE_ROOT/'datasets/ct_gan/ct_gan_uci.zip'
        if not ARCHIVE.is_file() or sha256_file(ARCHIVE)!='e88c990440e3f0ddf5e36928e42e6a5231b685ddce5c18bfccce9d513967a135':
            raise RuntimeError('Arsip Drive tidak tersedia atau checksum berbeda; tidak memulai unduhan baru.')
        restored=restore_ctgan_archive(ARCHIVE,Path('/content/rsvt-annotation-qa-source')/RUN.name,download_budget_bytes=0,progress=print)
        CT_ROOT=Path(restored['dataset_root'])
    if sha256_file(CT_ROOT/'Readme.docx')!='91da8903c128d9f175a4a815c497b835bb92ec5368195d8a3c691fa3e83dc671':
        raise RuntimeError('Publisher README changed.')
    print('CT source di Colab/Drive:',CT_ROOT)
    ''')
    cell("code", '''
    from rsvt.data.annotation_qa import run_annotation_qa
    summary=run_annotation_qa(dataset_root=CT_ROOT,manifest_path=F03,files_path=F01_FILES,
        catalog_path=CODE_ROOT/'reference_metadata/tcia_lidc_series.json',
        policy_path=CODE_ROOT/'configs/data/ctgan_verification.json',output_dir=RUN,
        drive_results_root=RESULTS_ROOT,code_sha256=EXPECTED_PACKAGE_SHA256,progress=print)
    # The function has read back every artifact and the persistence receipt.
    print(json.dumps(summary,indent=2))
    print('TERSIMPAN DAN DIBACA ULANG DI DRIVE:',RUN)
    print('Receipt:',RUN/'drive_persistence_receipt.json')
    print('Tidak perlu transfer manual. Cukup beri tahu run sudah selesai.')
    ''')
    cell("markdown", '''
    ## Memeriksa preview

    Pilih `POINT_INDEX` lalu jalankan sel berikut berulang untuk membandingkan
    hipotesis. Ke-123 preview telah tersimpan di Drive. Koordinat tidak otomatis
    disetujui dari satu contoh atau hanya karena berada dalam paru-paru.
    Titik FB adalah lokasi penghapusan, sehingga nodul memang dapat tidak tampak.
    Bila offset satu piksel/slice belum dapat dibedakan, catat `ambiguous`.
    Lokasi dan diagnosis klinis tidak boleh disimpulkan dari marker semata.
    ''')
    cell("code", '''
    from IPython.display import HTML, display
    POINT_INDEX=0
    points=json.loads((RUN/'points.json').read_text())
    if points:
        point=points[POINT_INDEX]
        print('point_id:',point['point_id'],'case:',point['case_key'],'split:',point['split'])
        display(HTML((RUN/point['preview']).read_text()))
    ''')
    cell("markdown", '''
    ## Catatan reviewer opsional

    Isi `DECISIONS` untuk titik yang benar-benar diperiksa. Format setiap entri:
    `point_id`, `status` (`accepted`, `ambiguous`, atau `rejected`),
    `selected_hypothesis` (nama dari panel untuk accepted; selainnya None),
    `reviewer`, dan `evidence`. Jangan menebak. Sel kosong tidak mengubah status.
    Catatan tersimpan sebagai file baru dan receipt dibaca ulang; manifest lama
    tetap dipertahankan. Review koordinat tidak membuka gate training.
    ''')
    cell("code", '''
    DECISIONS=[]
    if DECISIONS:
        from rsvt.data.annotation_qa import record_coordinate_reviews
        result=record_coordinate_reviews(RUN,DECISIONS,drive_results_root=RESULTS_ROOT)
        print(json.dumps(result,indent=2))
    else:
        print('Review visual masih PENDING; tidak ada koordinat yang disetujui otomatis.')
    ''')
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    target=legacy_output(ROOT,'source_annotation_qa_colab.ipynb')
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf-8'))
        if any(c.get('outputs') or c.get('execution_count') is not None for c in previous['cells']):
            raise RuntimeError('Preserve the executed QA notebook before rebuilding.')
    target.write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    result={"created_utc":datetime.now(timezone.utc).isoformat(),"notebook":str(target),"bundle":str(bundle),
            "bundle_sha256":digest,"notebook_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
            "medical_pixels_packaged":False,"remote_executed":False,"drive_upload_verified":False,
            "new_dataset_download_budget_bytes":0,"expected_preview_points":123,"test_points_excluded":25,
            "training_ready":False,"manual_transfer_required":False}
    delivery=ROOT/'reports/supplementary/source_annotation_qa_20261113'
    delivery.mkdir(parents=True,exist_ok=True)
    (delivery/'delivery.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':print(json.dumps(build(),indent=2))
