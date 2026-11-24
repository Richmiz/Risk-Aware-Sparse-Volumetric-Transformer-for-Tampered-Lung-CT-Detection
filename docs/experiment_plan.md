# Rencana eksperimen

Matriks berikut mengikuti proposal. F04 menyediakan baseline B0/B1, F05-F07
fondasi RSVT/routing/risk sintetis, F08 katalog serta orkestrasi interface,
dan F09 evaluasi ulang prediksi sintetis yang disegel.
Konfigurasi penelitian final dan evaluasi konfirmatori masih terbuka.

## Baseline B0-B5

| Kode | Model | Peran |
|---|---|---|
| B0 | 3D-ResNet50 / 3D-DenseNet121 | Pembanding CNN volumetrik |
| B1 | Dense fixed-patch 3D-ViT | Pembanding utama |
| B2 | Local-global two-stage cascade | Pembanding deteksi manipulasi kecil |
| B3 | 3D-ViT dengan dynamic routing tanpa risk loss | Kontribusi sparsification |
| B4 | 3D-ViT dengan risk loss tanpa routing | Kontribusi optimasi risiko |
| B5 | RSVT lengkap | Kombinasi routing, risk, dan regularisasi generator |

F04 memakai `configs/training/baseline_pilot.toml` atau `baseline_full.toml`.
B0 diimplementasikan sebagai adaptasi ResNet-50 3D/GroupNorm (width 16 pada pilot,
64 pada full draft); B1 memakai dense 3D ViT. Representasi pilot 64³/full 96³ dan
target kandidat belum ditetapkan sebagai benchmark konfirmatori. B2 cascade
belum diimplementasikan. B3/B4/B5 tersedia sebagai prototipe engineering;
registry F08 ada di `configs/experiments/registry`. B4 registry memakai CVaR
tanpa generator variance; prototipe integrasi B4 F07 berbeda dan hasilnya
dipertahankan. Konfigurasi `rsvt_ct_gan_pilot.toml`
menonaktifkan regularisasi multi-generator; konfigurasi `rsvt_full.toml` memakai
M3Dsynth setelah identitas generator dan pemetaan label diperiksa.

Bandingkan `baseline_vit3d.toml` dengan pilot CT-GAN, serta
`baseline_vit3d_m3dsynth.toml` dengan RSVT pada M3Dsynth menggunakan manifest dan
partisi identik. Konfigurasi source-grouped dan LOGO memerlukan pembanding dengan
protokol yang sama; jangan menyimpulkan peningkatan dari dataset/protokol berbeda.

## Ablation E1-E7

| Kode | Kontras | Pengukuran |
|---|---|---|
| E1 | Token tetap vs adaptif/multiskala | Recall FB, macro-F1, latency |
| E2 | Dengan vs tanpa coarse anatomical coverage | Recall FB dan lokalisasi |
| E3 | Cross-entropy vs class-conditional CVaR | Recall FB/FM pada FPR tetap |
| E4 | Dengan vs tanpa regularisasi generator | Worst-generator AUROC |
| E5 | Budget rendah, sedang, tinggi | Kurva performa-biaya |
| E6 | Held-out generator dan perubahan preprocessing | Generalisasi dan kalibrasi |
| E7 | Seed/partisi berulang serta audit leakage | Confidence interval dan kestabilan |

Deklarasi E1-E7 tersedia di `configs/ablations`; paired executors belum tersedia.
F08 tidak menampilkan hasil ablation yang belum dijalankan. Gunakan
`configs/orchestration/phase08.toml` untuk pipeline sintetis dan audit readiness;
file eksperimen scaffold lama tetap berupa rancangan.

## Aturan pelaporan

Tetapkan FPR, noninferiority margin, budget numerik, threshold, dan agregasi
prediksi volume sebelum final testing. Laporkan confidence interval berbasis
resampling kelompok CT sumber, bukan menganggap patch sebagai pasien independen.
Pengulangan seed tidak menggantikan ketidakpastian akibat ukuran sampel.

Pisahkan authentic-versus-tampered AUROC dari metrik empat kelas. Untuk mask
terverifikasi gunakan Dice/IoU; untuk titik gunakan galat jarak atau hit rate
dengan radius yang ditetapkan. Ukur latency p50/p95 pada hardware yang sama,
termasuk preprocessing yang dinyatakan dan overhead controller.
