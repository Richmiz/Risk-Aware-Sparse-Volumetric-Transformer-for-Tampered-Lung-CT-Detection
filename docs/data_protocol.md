# Protokol data dan manifest

Pada runtime Colab, data asli disimpan di `/content/rsvt-data/raw/<dataset>` atau
lokasi Google Drive yang dipilih dalam notebook. Path `data/raw/<dataset>` adalah
layout logis proyek, bukan instruksi mengunduh pada laptop. Hasil preprocessing berada di
`data/interim` dan `data/processed`; definisi partisi berada di `data/splits`.
Simpan provenance dan parameter transformasi agar koordinat dapat dipetakan
kembali ke ruang volume asli. Jangan menimpa data sumber.

## Skema manifest CSV

Template tersedia di `data/manifests/manifest.template.csv`.

| Kolom | Aturan |
|---|---|
| sample_id | Unik untuk setiap sampel/turunan di manifest |
| dataset | Nama sumber dataset |
| patient_id | Identitas pasien pseudonim yang telah direkonsiliasi lintas dataset |
| source_volume_id | Identitas canonical volume sumber sebelum manipulasi/crop |
| series_uid | Identitas seri teknis bila tersedia; bukan pengganti source_volume_id |
| volume_path | Path berkas atau direktori volume, relatif terhadap root proyek |
| role | forensic atau anatomy_reference |
| label | TB/TM/FB/FM untuk forensic; kosong untuk anatomy_reference |
| generator_id | Identitas generator untuk sampel manipulasi; kosong untuk referensi autentik |
| manipulation_type | none, removal, atau insertion sesuai label |
| annotation_type | none, point, box, atau mask |
| annotation_path | Path anotasi bila annotation_type bukan none |
| split | train, validation, atau test |
| sha256 | Hash berkas volume/representasi yang didokumentasikan; opsional pada inventaris awal |

Identitas asli pasien tidak disimpan pada manifest bersama. Collision antara ID
lokal dari dataset berbeda harus diselesaikan sebelum audit. Sampel CT yang sama
di beberapa dataset memakai canonical patient/source IDs yang sama; memberi
prefix dataset tanpa pencocokan sumber dapat menyembunyikan overlap.

`check-manifest` membandingkan hash yang disediakan. `inventory-data` menghitung
SHA256 berkas secara streaming. Untuk kandidat seri, fingerprint file-set adalah
SHA256 dari digest anggota yang diurutkan, dipisah newline, dan tanpa newline akhir.
Fingerprint ini mengenali salinan berkas identik yang namanya berubah; ini bukan
hash voxel canonical dan tidak mendeteksi semua volume turunan yang dimodifikasi.

## Pemisahan data

- Pisahkan pasien dan volume sumber sebelum membuat turunan.
- Semua versi asli, manipulasi, generator, slice, crop, dan augmentasi dari sumber
  yang sama mengikuti satu partisi.
- Reconcile sumber CT-GAN, M3Dsynth, dan LIDC-IDRI sebelum penggabungan data.
- LIDC-IDRI tidak otomatis menyediakan label forensic atau kepastian keganasan.
- Pada LOGO, generator uji hanya muncul pada test. Volume sumbernya juga harus
  terpisah dari sumber training/validasi, meskipun generatornya berbeda.
- Validasi untuk threshold dan hyperparameter hanya berasal dari sumber pelatihan.

## Batas pemeriksaan otomatis

`check-manifest` memeriksa kolom, label, anotasi yang dideklarasikan, identitas
duplikat, konflik pasien/volume sumber/hash antarpartisi, dan holdout generator.
`--check-paths` menambah pemeriksaan keberadaan path. Jalankan pada manifest
gabungan untuk menemukan konflik lintas dataset; pemeriksaan file terpisah tidak
dapat membandingkan identitas antarfile.

`inventory-data` juga membaca header DICOM tanpa mendekode pixel data, mencatat
geometri, dan membentuk kandidat seri berdasarkan direktori serta UID yang
dipseudonimkan. PatientID keys bersifat dataset-scoped; UID keys lintas dataset
hanya bukti overlap untuk review, bukan canonical source IDs yang otomatis valid.
Kunci HMAC disimpan terpisah pada storage state; notebook menaruhnya di Drive.

`audit-ctgan` mempertahankan label tingkat lokasi, baris duplikat, dan koordinat
nol sebagai bukti untuk ditinjau. Label volume tidak ditentukan otomatis.
`audit-m3dsynth` mempertahankan lineage pasien/seri dan generator dari metadata
penulis; keberadaan baris metadata tidak berarti citranya telah diunduh.

Rekonsiliasi pasien, validasi koordinat/mask, kelengkapan seri, dan kemiripan voxel
masih memerlukan pemeriksaan lanjutan. Manifest kandidat sengaja memiliki canonical
IDs dan split kosong sehingga ditolak oleh audit manifest training yang ketat.
