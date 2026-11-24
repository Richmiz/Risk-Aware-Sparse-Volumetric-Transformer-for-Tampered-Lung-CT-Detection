# RSVT: kode model usulan, data yang tersedia, dan teori pendukung

> **17 November 2026:** the proposed model is now the coarse-to-fine RSVT trained
> on M3Dsynth — see [rsvt_c2f_m3dsynth.md](rsvt_c2f_m3dsynth.md). The 96³-token
> design below remains as a negative control on CT-GAN; its config now enables the
> router-budget initialization, inference gate bias and max+mean pooling fixes.

RSVT = **Risk-Aware Sparse Volumetric Transformer**. Tujuannya adalah
mengklasifikasikan CT autentik atau dimanipulasi, dengan perhatian khusus pada
penghapusan nodul. Kombinasi arsitektur dan objective ini adalah hipotesis riset
proyek; manfaat deteksi dan komputasinya harus diukur pada eksperimen baru.

## Data dan cara menjalankan

Pakai CT-GAN yang sudah tersimpan di Drive: 100 volume / 22.753 DICOM.
Partisi historis tetap: **72 train, 14 validation, 14 test**. Training ini hanya
membuka 86 volume train/validation. Angka volume bukan jumlah pasien independen
yang telah diverifikasi. Pemetaan ke sumber publik masih belum terselesaikan.

Label empat kelas: TB/TM = autentik; FB = removal; FM = insertion.
Titik anotasi tidak dipakai untuk memilih ROI, melatih model, atau mengambil
keputusan inference. Lokalisasi voxel dan penalti generator tetap nonaktif.
Dataset tunggal CT-GAN tidak mendukung klaim generalisasi generator baru.

Sejak 17 November 2026 `notebooks/01_RSVT_TRAINING.ipynb` melatih RSVT
coarse-to-fine pada M3Dsynth. Desain 96³ pada CT-GAN tetap dapat dijalankan
sebagai kontrol negatif melalui `scripts/run_proposed_model.py`
(`configs/training/rsvt_real_ct.toml`).

Sumber: `outputs/partitions/f03_20261110T013617_031412Z/manifest.draft.json`.
Fingerprint assignment dikunci di `configs/training/rsvt_real_ct.toml`.
Hasil: `MyDrive/Research/LUNG-CT-SCAN/outputs/rsvt_training/rsvt_<UTC>/`.
Untuk melanjutkan sesi notebook baru yang terputus pada batas epoch, isi `RESUME_SESSION` dengan
path Drive run tersebut. Checkpoint, optimizer, scheduler, dual, RNG, config,
kode, dan manifest harus cocok. Epoch parsial diulang; tidak dilanjutkan di tengah batch.

## Hubungan teori dengan implementasi

| Dasar ilmiah | Implementasi RSVT | Batas interpretasi |
|---|---|---|
| [CT-GAN — Mirsky dkk., USENIX Security 2019](https://www.usenix.org/conference/usenixsecurity19/presentation/mirsky): bukti citra CT volumetrik dapat dimanipulasi melalui insertion/removal. | Klasifikasi forensik TB/TM/FB/FM dari volume CT yang tersedia. | Dataset/serangan ini mendefinisikan masalah; tidak membuktikan detector RSVT berhasil. |
| [UNETR — Hatamizadeh dkk., WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html): patch 3D dan Transformer untuk representasi volumetrik serta hubungan spasial global. | Encoder coarse menjaga konteks anatomi; tokenizer menghasilkan patch 8³/16³/32³, dengan posisi fisik LPS dan skala. `coarse_encoder.py`, `tokenization.py`, `transformer.py`. | RSVT mengadaptasi gagasan representasi 3D; tidak mereproduksi UNETR atau meminjam hasil segmentasinya. Pemilihan patch/LPS/coarse wajib adalah rancangan proyek. |
| [DynamicViT — Rao dkk., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html): skor kepentingan dan masking token yang bisa dilatih. | Router memakai embedding, cue lokal, fraksi valid, dan konteks coarse; soft key log-gates saat training, hard top-k dan packing sebelum attention saat inference. `routing.py`, `routed_rsvt.py`. | DynamicViT menguji ImageNet 2D. RSVT memakai routing satu tahap pada CT 3D; hasil kecepatan paper tidak berlaku otomatis. |
| [Rockafellar & Uryasev — Optimization of Conditional Value-at-Risk](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf): minimisasi ekor atas distribusi loss. | CVaR dihitung terpisah dari CE mentah untuk FB dan FM; koefisien FB lebih besar. `losses/cvar.py`, `training/proposed.py`. | Formulasi asal membahas risiko, termasuk finansial. Penerapan class-conditional untuk error removal/insertion adalah adaptasi proyek, bukan jaminan sensitivitas klinis. |

## Arsitektur dan objective yang benar-benar digunakan

```text
Volume CT + mask valid + affine XYZ→LPS
  → normalisasi yang sudah tersedia + view full-FOV 96³
  → [coarse anatomical encoder || multiscale candidate tokenizer]
  → embedding posisi fisik dan skala
  → router: soft gates saat training / hard top-k saat inference
  → Transformer volumetrik → CLS → probabilitas TB/TM/FB/FM
```

Coarse valid tidak pernah dibuang. Untuk input 96³ dengan semua voxel valid,
urutan dense berisi 1.999 token termasuk CLS; inference dibatasi maksimum
128 token, termasuk coarse dan CLS. Semua kandidat tetap dibentuk sebelum
routing. Pengurangan panjang urutan bukan pengukuran penghematan total FLOPs
atau latency. Training tetap dense, ditambah biaya replay forward kedua.

Untuk logical batch B, `ell_i = CE(logits_i,y_i)` dan bobot kelas `w` berasal
hanya dari train:

```text
L_CE = mean_i(w[y_i] * ell_i)
L_risk = (2 * CVaR_alpha(ell | FB) + CVaR_alpha(ell | FM)) / 3
g = (max_i[1 + N_coarse_valid_i + sum_j sigmoid(router_logit_ij)] - K) / K
L = L_CE + lambda * L_risk + mu * g
mu <- clip(mu + dual_lr * g, 0, mu_max)
```

CVaR adalah rerata fraksional empiris dari loss terburuk, dengan massa ekor
`n_class*(1-alpha)`. `alpha=0.5`, `lambda=0.2`, dan bobot 2:1 merupakan nilai
awal sementara, belum dipilih melalui validasi. Koefisien/massa ekor kecil
dicatat pada laporan setiap langkah. Nilai alpha lebih tinggi pada dukungan
kecil dapat membuat CVaR hanya memilih satu kasus; itu bukan estimasi kuat
risiko populasi. Lagrangian token di sini adalah surrogate nonconvex;
tidak ada klaim konvergensi global atau jaminan runtime.

Train memakai logical batch target 16 kasus, diatur agar tiap kasus dikunjungi
sekali per epoch dan kelas tersebar, tanpa oversampling. Untuk menghitung
CVaR lintas kasus tanpa menyimpan semua graph CT:

1. Ukur CE dan expected token count tiap kasus tanpa graph, simpan RNG.
2. Hitung gradien objective terhadap scalar CE/token count logical batch.
3. Replay satu volume per microbatch dengan RNG yang sama, lalu akumulasi
   gradien menurut chain rule. Parameter baru di-update setelah seluruh batch.

Tes membandingkan update replay dengan backprop logical batch langsung,
termasuk dropout dan anggaran token. Ini menguji ketepatan implementasi,
bukan akurasi medis. FP32 dipakai agar replay mudah diaudit; tidak ada klaim AMP.

Training 30 epoch: 3 epoch dense baseline, 2 epoch router warmup dengan encoder
dibekukan, lalu risk finetuning. AdamW + warmup/cosine, gradient clipping 5,
checkpoint terbaik dipilih hanya dari epoch **risk finetuning**, berdasarkan CE
validation **hard-packed**. Warmup dipantau tanpa menjadi kandidat hasil akhir.
Resume yang berhenti sebelum risk finetuning melaporkan checkpoint sementara,
bukan hasil RSVT final. Pembandingan
baseline harus melaporkan perbedaan kapasitas, jumlah update dan total biaya.

## Berkas utama dan batas penelitian

- `src/rsvt/models/proposed.py`: API `RSVT` yang menggabungkan modul model.
- `src/rsvt/training/proposed.py`: objective, replay, training, evaluasi, resume.
- `src/rsvt/training/proposed_run.py`: binding manifest, loader, geometri, config.
- `scripts/run_proposed_model.py`: pekerjaan Colab, output dan verifikasi Drive.
- `configs/training/rsvt_real_ct.toml`: arsitektur, data, loss dan hyperparameter.

Geometri affine dibaca dari cache yang diverifikasi, bukan dari titik anotasi.
Original dan assignment historis tidak diubah. View full-FOV 96³ sama dengan
baseline, tetapi dapat menghilangkan detail manipulasi kecil; masalah ini belum
selesai hanya karena kode training tersedia. Identitas sumber/leakage assurance
dan protokol endpoint tetap terbuka, sehingga `training_ready=false` dipertahankan.
Kode baru tidak berarti training CT telah dilakukan atau hasil 35,7% baseline
telah membaik. Run Colab diperlukan untuk mengukur itu.
