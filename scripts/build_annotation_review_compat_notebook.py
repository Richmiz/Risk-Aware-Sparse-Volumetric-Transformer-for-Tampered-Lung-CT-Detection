"""Pin the renderer-compatible widgets while keeping the executed backend/session.

Uses the verified frozen notebook/code release, not the user's executed file.
The source bundle SHA is unchanged so existing decisions can resume safely.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from textwrap import dedent, indent
from notebook_layout import legacy_output

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_NOTEBOOK_SHA256="1e2ede7f8d41db8a70dd6ccbece7490e0a657d723b959733871b5e2fcedd3a23"
BACKEND_SHA256="4c04973de96823a01585ce5741b8cf2ce70689ec5328567593a5204678bea224"


def build():
    evidence=ROOT/'reports/supplementary/annotation_review_release_20261113'
    pristine=evidence/'Drive_readback_review_annotation_qa_colab.ipynb'
    raw=pristine.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ORIGINAL_NOTEBOOK_SHA256:
        raise ValueError('Frozen source notebook checksum mismatch')
    bundle=ROOT/'artifacts/rsvt-annotation-review-code.zip'
    if hashlib.sha256(bundle.read_bytes()).hexdigest()!=BACKEND_SHA256:
        raise ValueError('Keep the executed backend bundle unchanged to resume the existing session')
    notebook=deepcopy(json.loads(raw))
    for c in notebook['cells']:
        c['metadata']={}
        if c['cell_type']=='code':c.update(execution_count=None,outputs=[])
    def source(c,text):c['source']=dedent(text).strip().splitlines(keepends=True)
    source(notebook['cells'][0],'''
    # Review koordinat — kompatibilitas VS Code

    Pilih kernel Colab di VS Code lalu **Run All**. Notebook mempertahankan
    kode backend dan sesi review yang sudah tersimpan. Tidak ada unduhan
    dataset baru atau pengulangan audit.

    Perbaikan ini memakai **ipywidgets 8.1.8** karena ada laporan resmi output
    kosong pada 8.1.9: https://github.com/microsoft/vscode-jupyter/issues/17504.
    Jika kernel masih memuat versi lama, notebook memasang versi kompatibel
    dan menampilkan instruksi **Restart Kernel, lalu Run All**. Ini cukup
    dilakukan sekali; keputusan pada Drive tetap dipertahankan.

    Setelah form terlihat, pilih keputusan, reviewer, dan bukti, lalu gunakan
    **Simpan & Berikutnya**. Semua 123 titik masih memerlukan review manusia;
    25 titik test dikecualikan dan kesiapan training tetap false.
    ''')
    dependency={'cell_type':'code','id':'coordinate-review-widget-compat','metadata':{},'execution_count':None,'outputs':[]}
    source(dependency,'''
    import sys, subprocess, importlib.metadata
    LOADED_WIDGET_VERSION=getattr(sys.modules.get('ipywidgets'),'__version__',None)
    try:
        INSTALLED_WIDGET_VERSION=importlib.metadata.version('ipywidgets')
    except importlib.metadata.PackageNotFoundError:
        INSTALLED_WIDGET_VERSION=None
    if INSTALLED_WIDGET_VERSION!='8.1.8':
        subprocess.run([sys.executable,'-m','pip','install','-q','ipywidgets==8.1.8'],check=True)
    WIDGET_RESTART_NEEDED=LOADED_WIDGET_VERSION is not None and LOADED_WIDGET_VERSION!='8.1.8'
    if WIDGET_RESTART_NEEDED:
        print('ipywidgets 8.1.8 sudah terpasang. Kernel ini masih memuat versi '+str(LOADED_WIDGET_VERSION)+'.')
        print('Klik Restart Kernel di toolbar VS Code, lalu Run All notebook ini kembali. Sesi Drive tetap tersimpan.')
    else:
        print('Versi widget kompatibel terpasang: '+importlib.metadata.version('ipywidgets'))
    ''')
    setup=notebook['cells'][3]
    text=''.join(setup['source'])
    source(setup,"if WIDGET_RESTART_NEEDED:\n    SESSION=None\n    print('Menunggu Restart Kernel untuk mengaktifkan widget yang kompatibel.')\nelse:\n"+indent(text,'    '))
    ui=notebook['cells'][5]
    text=''.join(ui['source'])
    source(ui,"if WIDGET_RESTART_NEEDED:\n    print('Formulir akan dibuka setelah Restart Kernel dan Run All.')\nelse:\n"+indent(text,'    '))
    notebook['cells'].insert(2,dependency)
    target=legacy_output(ROOT,'review_annotation_qa_compat_colab.ipynb')
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf-8'))
        if any(c.get('outputs') or c.get('execution_count') is not None for c in previous['cells']):
            raise ValueError('Preserve the executed compatibility notebook before rebuilding')
    target.write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    release=ROOT/'reports/supplementary/annotation_review_compat_20261113'
    release.mkdir(parents=True,exist_ok=True)
    delivery={'created_utc':datetime.now(timezone.utc).isoformat(),'notebook':str(target),
        'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'backend_bundle_sha256':BACKEND_SHA256,
        'backend_unchanged':True,'ipywidgets_pinned':'8.1.8','existing_session_preserved':'review_20261113T080116_765905Z',
        'remote_UI_verified':False,'dataset_download_bytes':0,'training_ready':False,'Drive_upload_verified':False}
    (release/'delivery.json').write_text(json.dumps(delivery,indent=2)+'\n',encoding='utf-8')
    return delivery


if __name__=='__main__':print(json.dumps(build(),indent=2))
