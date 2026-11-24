"""Build the review-only Colab notebook. No medical images or audit rerun."""

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from textwrap import dedent
import zipfile

from build_colab_bundle import build_bundle
from notebook_layout import legacy_output

ROOT=Path(__file__).resolve().parents[1]


def build():
    staging=ROOT/"artifacts/rsvt-annotation-review-staging.zip"
    bundle=ROOT/"artifacts/rsvt-annotation-review-code.zip"
    build_bundle(ROOT,staging)
    with zipfile.ZipFile(staging) as source, zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            if not name.startswith("notebooks/"):
                target.writestr(name,source.read(name))
    package_bytes=bundle.read_bytes()
    digest=hashlib.sha256(package_bytes).hexdigest()
    bundle.with_suffix(".zip.sha256").write_text(digest+"\n",encoding="utf-8")
    payload=base64.b64encode(package_bytes).decode("ascii")
    cells=[]
    def cell(kind,source):
        item={"cell_type":kind,"id":f"coordinate-review-{len(cells)}","metadata":{},
              "source":dedent(source).strip().splitlines(keepends=True)}
        if kind=="code":item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell("markdown",'''
    # Review koordinat: 123 titik yang sudah tersedia

    Buka dengan kernel **Google Colab melalui VS Code**, lalu **Run All**.
    CPU cukup. Notebook membaca preview dari run `qa_20261113T005049_219029Z`
    di Drive, tanpa mengulang audit atau mengunduh dataset baru.

    Formulir menampilkan titik dan delapan hipotesisnya. Pilih keputusan,
    masukkan nama reviewer serta bukti, lalu **Simpan & Berikutnya**.
    Gunakan `ambiguous` bila lokasi/indeks belum dapat dipastikan.
    Seluruh progres dan laporan otomatis tersimpan di Drive dan dibaca ulang.
    Buka kembali notebook dan Run All untuk melanjutkan sesi yang sama.

    Jalankan satu runtime untuk satu sesi. Dua puluh lima titik test dikecualikan.
    Penyelesaian review berarti semua titik memiliki catatan; koordinat ambigu
    tetap belum tervalidasi. Pemetaan sumber 0/100, selisih README 484 slice,
    dan kesiapan training tetap merupakan pekerjaan terbuka.
    ''')
    cell("code",'''
    import importlib.util, platform
    from pathlib import Path
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Pilih kernel Google Colab di VS Code; preview medis tetap di Colab/Drive.')
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    ''')
    cell("code","import base64\nPACKAGE_BYTES=base64.b64decode(\n"+
         "\n".join("    "+repr(payload[i:i+120]) for i in range(0,len(payload),120))+"\n, validate=True)\n")
    cells[-1]["metadata"]={"jupyter":{"source_hidden":True}}
    cell("code",f'''
    import hashlib, json, subprocess, sys, zipfile
    from io import BytesIO
    EXPECTED_PACKAGE_SHA256='{digest}'
    if hashlib.sha256(PACKAGE_BYTES).hexdigest()!=EXPECTED_PACKAGE_SHA256:
        raise RuntimeError('Checksum paket kode berbeda.')
    CODE_ROOT=Path('/content/rsvt-annotation-review')/EXPECTED_PACKAGE_SHA256[:12]
    CODE_ROOT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(BytesIO(PACKAGE_BYTES)) as archive:
        for name in archive.namelist():
            if not (CODE_ROOT/name).resolve().is_relative_to(CODE_ROOT.resolve()):
                raise ValueError('Unsafe code bundle path')
        archive.extractall(CODE_ROOT)
    del PACKAGE_BYTES
    subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(CODE_ROOT)+'[annotation_review]'],check=True)
    sys.path.insert(0,str(CODE_ROOT/'src'))
    # A preceding audit may have imported an older rsvt in this same kernel.
    for module_name in tuple(sys.modules):
        if module_name=='rsvt' or module_name.startswith('rsvt.'):
            del sys.modules[module_name]
    import rsvt
    if not Path(rsvt.__file__).resolve().is_relative_to(CODE_ROOT.resolve()):
        raise RuntimeError('Kode dari kernel lama masih aktif; restart kernel lalu jalankan notebook ini.')
    from rsvt.data.annotation_review import ReviewSource, ReviewSession, SOURCE_RUN
    STATE_ROOT=Path('/content/drive/MyDrive/Research/LUNG-CT-SCAN')
    SOURCE_ROOT=STATE_ROOT/'outputs/source_annotation_qa'/SOURCE_RUN
    RESULTS_ROOT=STATE_ROOT/'outputs/source_annotation_review'
    if not RESULTS_ROOT.resolve().is_relative_to(STATE_ROOT.resolve()):
        raise RuntimeError('Hasil wajib tersimpan pada Drive proyek.')
    SOURCE=ReviewSource(SOURCE_ROOT)
    SESSION=ReviewSession.open(SOURCE,RESULTS_ROOT,code_sha256=EXPECTED_PACKAGE_SHA256)
    print('Sesi review Drive:',SESSION.root)
    print('Sumber terverifikasi: 123 titik train/validation; 25 titik test dikecualikan.')
    ''')
    cell("markdown",'''
    ## Formulir review

    `accepted` memerlukan hipotesis yang didukung bukti. `ambiguous` dan
    `rejected` tidak memilih hipotesis. Jangan menetapkan konvensi hanya karena
    titik berada di paru-paru. Pada FB, nodul yang dihapus dapat tidak tampak.
    Catat ketidakpastian offset satu piksel/slice sebagai `ambiguous` bila
    bukti yang tersedia tidak cukup. Formulir tidak menetapkan diagnosis klinis.

    Progres diperbarui setelah penyimpanan terverifikasi. Jika penyimpanan gagal,
    titik dan isian tetap aktif. Untuk berpindah tanpa menyimpan perubahan,
    gunakan **Batalkan perubahan**.
    ''')
    cell("code",'''
    from IPython.display import display
    from rsvt.data.annotation_review_ui import AnnotationReviewForm
    if 'REVIEW_FORM' in globals():
        REVIEW_FORM.root.close()
    REVIEW_FORM=AnnotationReviewForm(SESSION)
    display(REVIEW_FORM.root)
    ''')
    cell("markdown",'''
    ## Hasil tersimpan

    Setiap penyimpanan membuat revisi dengan keputusan JSON, tabel CSV,
    ringkasan JSON, laporan Markdown, dan receipt checksum. Catatan sebelumnya
    dipertahankan. Asisten dapat membaca semuanya langsung dari Drive.
    Tidak ada langkah transfer file manual.

    Identitas sumber dan manifest historis tidak diubah. Tidak ada training
    atau evaluasi model pada notebook ini. `training_ready=false` tetap berlaku.
    ''')
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
              "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    target=legacy_output(ROOT,'review_annotation_qa_colab.ipynb')
    if target.exists():
        old=json.loads(target.read_text(encoding='utf-8'))
        if any(c.get('outputs') or c.get('execution_count') is not None for c in old['cells']):
            raise RuntimeError('Preserve the executed review notebook before rebuilding.')
    target.write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    destination=ROOT/'reports/supplementary/annotation_review_release_20261113'
    destination.mkdir(parents=True,exist_ok=True)
    result={"created_utc":datetime.now(timezone.utc).isoformat(),"notebook":str(target),"bundle":str(bundle),
            "notebook_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"bundle_sha256":digest,
            "notebook_bytes":target.stat().st_size,"bundle_bytes":bundle.stat().st_size,
            "source_run":"qa_20261113T005049_219029Z","source_points":123,"test_points_excluded":25,
            "contains_medical_pixels":False,"dataset_download_bytes":0,"remote_UI_verified":False,
            "actual_human_review_decisions_created":False,"Drive_upload_verified":False,"training_ready":False}
    (destination/'delivery.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':print(json.dumps(build(),indent=2))
