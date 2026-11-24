# Rancangan penelitian RSVT

Proyek ini mengikuti proposal pengembangan Risk-Aware Sparse Volumetric
Transformer untuk deteksi medical deepfake pada CT paru 3D. Sistem berperan
sebagai pemeriksa integritas citra. Label forensic tidak merupakan diagnosis
kanker dan generalisasi lintas generator tidak sama dengan validasi lintas rumah sakit.

## Alur model yang direncanakan

1. Volume DICOM divalidasi, dikonversi ke HU, dan diharmonisasi secara geometris.
2. Encoder ringan mempertahankan konteks anatomi global tanpa pemangkasan token.
3. Token 3D multiskala dan fitur anomali menyediakan kandidat bukti lokal.
4. Controller memilih token beresolusi tinggi dengan batas biaya komputasi.
5. Transformer menggabungkan token terpilih dan token anatomi kasar.
6. Head menghasilkan empat kelas forensic dan peta manipulasi bila anotasi mendukung.

Optimasi menggabungkan classification loss, localization loss opsional,
class-conditional CVaR untuk FB/FM, regularisasi generator, serta biaya komputasi.
Batas numerik budget, bobot loss, alpha CVaR, dan mekanisme pemilihan token belum
ditetapkan di scaffold ini. Jalur coarse coverage perlu tetap aktif pada RSVT
utama; penghapusannya diuji secara khusus melalui ablation.

## Pengukuran utama

Endpoint utama adalah recall FB pada FPR yang ditentukan dari validasi autentik.
Evaluasi tambahan mencakup macro-F1, authentic-versus-tampered AUROC, worst-generator
AUROC, kalibrasi, lokalisasi, dan FLOPs/latency. Pembanding harus menggunakan
partisi dan preprocessing yang sama; overhead controller masuk profiling.

Target proposal, termasuk peningkatan recall FB dan efisiensi komputasi, belum
merupakan hasil. Klaim kontribusi menunggu eksperimen terkontrol dan confidence interval.
