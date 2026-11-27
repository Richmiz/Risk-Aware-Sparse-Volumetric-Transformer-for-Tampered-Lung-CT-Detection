# Keputusan Protokol RSVT

Diperbarui: **13 November 2026**. Status keputusan mengikuti bukti saat ini.

| Keputusan | Status | Ketetapan atau pekerjaan tersisa |
|---|---|---|
| Runtime utama | Ditetapkan pengguna | Google Colab melalui VS Code; pengguna menjalankan notebook sendiri |
| Lokasi dataset | Ditetapkan pengguna | Colab atau Google Drive; tidak ada unduhan dataset lokal selanjutnya |
| Persistensi artefak | Ditetapkan pengguna 12 November | Semua hasil run, receipt, laporan, notebook, dan paket kode disimpan otomatis di Google Drive proyek dan dibaca ulang untuk verifikasi. Tidak meminta transfer manual sebagai langkah wajib; menggantikan pilihan lokal sebelumnya |
| Transfer kode | Terverifikasi | ZIP kode pada folder Drive Research/LUNG-CT-SCAN berhasil dijalankan pada Colab |
| Sumber CT-GAN | Dipilih | UCI ID 520, mempertahankan dokumentasi dan tabel label penulis |
| Sumber M3Dsynth | Dipilih | Repository dan arsip resmi penulis; metadata tersedia, citra belum tersedia |
| Sumber anatomi | Dipilih | LIDC-IDRI; perlu audit sumber bersama M3Dsynth |
| Label volume | Label penerbit diverifikasi; kesiapan penuh PENDING | Audit 11 November: target 100/100 cocok dengan definisi scan dan distribusi README; 6 duplikat hanya TB-zero, 148 titik nonnol. Verifikasi piksel, identitas dan koordinat belum ditutup; manifest F03 tetap draft |
| Koordinat nol CT-GAN | PENDING | Dikeluarkan dari target point terverifikasi sampai maknanya diperiksa |
| Duplikasi baris label | PENDING | Dipertahankan sebagai bukti; tetapkan kebijakan deduplikasi unit analisis |
| Canonical patient/source IDs | PENDING | Rekonsiliasi lintas dataset; pasangan exp1/exp2 untuk scan 6031 dan 7507 perlu berada dalam grup sumber yang sama |
| Pemilihan subset M3Dsynth | PENDING | Tetapkan sampel dan storage remote yang mendukung protokol serta generator holdout |
| Partisi final | PENDING | Draft 72/14/14 terverifikasi pada bukti sumber yang diketahui; finalisasi tetap memerlukan identitas canonical dan validasi target |
| Skor endpoint FB dan FPR | PENDING | Tetapkan skor, FPR, serta threshold rule menggunakan source validation |
| Geometri dan ukuran token | Pilot disiapkan | F02 memakai LPS, preset 1 mm dan window -1000/400 untuk QA; parameter final, token, dan agregasi volume tetap PENDING |
| Verifikasi pilot F02 | Run diverifikasi | Geometri 100/100 dan pilot 4/4 berhasil; pemeriksaan header/rescale lanjutan tercatat pada F02b |
| QA intensitas F02b | Run diverifikasi | 100 seri memenuhi kondisi HU CT Image Module; 300 slice/1.228.800 sampel cocok dengan pydicom, bukan bukti kalibrasi fisik |
| Resume F02b | Run diverifikasi | 74 cache lolos SHA/header, 26 volume baru selesai; 100/100 file-set hash cocok dengan F01; nol penolakan/penundaan |
| Pemulihan sumber F02b | Run diverifikasi | Arsip CT-GAN 6,40 GB tersimpan di Drive; ekstraksi dilakukan pada Colab dan kesesuaian sumber diperiksa pada manifest akhir |
| Penutupan QA F02 | PENDING | Cakupan teknis 100/100 selesai; semantik anotasi, dampak preprocessing, dan rekonsiliasi sumber tetap terbuka; training_ready=false |
| Draft engineering F03 | Run diverifikasi | Target 70/15/15 menghasilkan 72/14/14, seed 13, 98 grup; nol konflik bukti sumber, semua kelas terwakili; bukan partisi penelitian final |
| Target kandidat F03 | Pilot eksplisit | Unanimous location label per scan; mapping TB=0/TM=1/FB=2/FM=3; label_verified dan classification_eligible tetap false |
| Kontrak loader F03 | Run diverifikasi | DataLoader PyTorch 2.14.1+cpu, float32 B/C/Z/Y/X, dua volume train lolos; validitas/padding terpisah, augmentasi nonaktif |
| Ukuran holdout F03 | Keterbatasan dicatat | Validation/test masing-masing 4 autentik kandidat dan 5 FB/5 FM; satu FP memberi perubahan FPR empiris 25 poin persentase |
| LOGO CT-GAN F03 | Tidak berlaku | Satu generator manipulasi; utility LOGO diuji sintetis, eksekusi multi-generator menunggu citra dan rekonsiliasi sumber |
| Training engineering F04 | Pilot, diagnosis, full direview | Pilot 3 epoch dan kontrol diagnosis 8/4 dilanjutkan full 72/14 pada CPU; early stop CNN/ViT epoch 17/7, 306/126 update; flag QA F03 tetap dan test tidak dibuka |
| Representasi F04 | Pilot eksplisit | Full-FOV 64³ pada pilot / 96³ full draft; preservasi jejak manipulasi kecil belum diverifikasi |
| Baseline F04 | Pilot direview | CNN/ViT masing-masing accuracy 25% pada 4 validation; macro-F1 0,100/0,125. CNN AUROC 1,0 hanya dari 2 autentik/2 manipulasi; tidak menjadi konfirmasi performa |
| Diagnosis pembelajaran F04 | Kontrol fitting lolos | Run f04_20261110T060853_936069Z: train accuracy epoch 40 kedua model 100%; kriteria >=95% dan CE <=0,2 pertama lolos CNN epoch 13, ViT epoch 18. Best-validation checkpoint epoch 13 memberi accuracy CNN/ViT 25%/75% pada empat kasus; recall klasifikasi FB keduanya 0/1. Fitting bukan bukti generalisasi; diagnosis tanpa dropout/weight decay/early stopping tidak mengisolasi penyebab pilot lama |
| Training full F04 | Eksekusi direview; kualitas terbuka | Run f04_20261110T103203_692532Z: best epoch CNN/ViT 12/2, accuracy keduanya 35,7% (5/14), macro-F1 0,1389/0,1316 dan AUROC 0,525/0,550; prediksi FB 13/14 dan 14/14. Recall klasifikasi FB 100% disertai FPR argmax tampering 100%; pada threshold nol FP validation recall tampering FB hanya 0/5 dan 1/5. Diagnosa bias kelas/optimisasi/representasi diperlukan. Partisi tetap, REQUIRE_GPU_FOR_FULL=False, QA tetap draft |
| Kontrol arsitektur full F04 | Direview; ViT lolos/CNN parsial | Run f04_20261111T023415_243484Z: ViT lolos epoch 16–17, train 100%/CE 0,08342; CNN sampai 30 epoch, 87,5%/CE 0,47013, kasus exp1:2244 TM→TB. Input numerik lolos, sampled module gradients finite; clipping 221/240 CNN dan 125/136 ViT. Validation best CNN/ViT 25%/75% dari empat kasus, tetap eksploratori |
| Kelanjutan F05 | Engineering diperbolehkan | Implementasi komponen RSVT dengan fixture sintetis dapat dimulai berdasarkan kontrak F03/F04. F04 tidak ditutup; optimisasi CNN, QA, baseline/protokol untuk perbandingan, dan endpoint penelitian tetap terbuka. Test dipertahankan |
| Fondasi F05 | Dense/sintetis lolos Colab CPU | Run f05_20261111T082858_565780Z: 139 tes OK, semua checks integrasi lolos; source/config cocok. Coarse wajib patch 32 dan kandidat 8/16/32, dim 64, Transformer 2 layer/4 head, posisi LPS relatif FOV, empat logit kelas. Encoder acak; konfigurasi prototipe belum dipilih dari source validation. F06 sintetis dapat dimulai |
| Lokalisasi F05 | Gate eksplisit | Peta piecewise constant per fine patch; Dice/Focal hanya voxel mask biner terverifikasi. Points/boxes ditolak dan tidak dikonversi ke segmentasi; anotasi proyek belum menjadi mask eligible |
| Biaya F05 | Struktur dense dicatat | 96³ menghasilkan 27 coarse + 1.971 kandidat + CLS, sequence 1.999. Slot attention dan padded stem grids dicatat; saving=0, bukan FLOPs/latency. Sparse routing/profiling tetap F06 |
| Routing F06 | Implementasi engineering sintetis | Sigmoid log-gate attention untuk training dense; hard stable top-k dan gather saat inference. Seluruh coarse valid wajib; cap termasuk CLS. Relaksasi soft bukan estimator tak-bias untuk hard top-k; CE→router diuji tanpa budget penalty |
| B3 F06 | Sintetis; medis terbuka | Classification/routing-only, tanpa localization/CVaR/generator loss. Enam fresh process memakai bobot awal/fixture identik; kontrol dense dan all-valid memisahkan packing invalid dari seleksi adaptif |
| Budget F06 | Trial caps dan kalibrasi eksploratori | Cap 64/128/256, floor coarse+CLS; target 0,5/0,75/1,0×dense p95. Target tanpa profil feasible dilaporkan. Ini bukan jaminan constraint latency; budget final menunggu source validation dan profiling Colab |
| Profiling F06 | CPU lokal direkam | FP32, batch1 64³, 2 warmup/10 sampel; p50 dense 238,301 ms, cap64 224,006 ms. Kandidat/stem dominan; target 0,5/0,75×dense p95 infeasible. FLOPs Conv/Linear/attention merupakan estimasi dengan daftar operator yang dikecualikan; CUDA belum diuji |
| Review F06 Colab | Engineering diterima | Run f06_20261111T090630_527193Z pada CPU: 156 tes OK, semua checks dan 17 verifikasi review lolos. P50 dense/cap128 501,822/429,920 ms; p95 dense 1.332,522 ms memberi target kalibrasi yang sensitif terhadap variasi tail. Budget final belum dikunci |
| CVaR F07 | Prototipe sintetis | Fractional empirical upper tail per FB/FM; alpha confidence0,9, bobot2/1 merupakan engineering defaults. N=2/kelas memberi max-only/low-support; kelas hilang nol dengan denominator bobot tetap. Source-validation selection masih PENDING |
| Generator-risk F07 | Gate terverifikasi | Population variance mean CE generator di dalam FB/FM, >=2 verified training groups dengan allowlist. Valid/test ID tidak menjadi input model. Fixture dua grup prosedural saja; proyek CT-GAN satu generator belum eligible |
| Budget/stages F07 | Engineering diuji | Projected dual pada max expected-token surrogate; training tetap dense, hard inference cap64. B4 3 step dense, B5 4 step baseline/warmup/risk; freeze dan gradien risiko ke router diuji. Konvergensi proxy belum dibuktikan; tidak menjadi klaim biaya total atau perbandingan performa |
| Review F07 Colab | Engineering diterima | Run100301: 179 tes/13 checks/16 verifikasi review lolos; history/inference identik dengan run095111 pada CPU torch2.14.1. Alpha0,9 dan N2 per kelas tetap max-only/low-support; actual multi-generator dan QA belum selesai |
| Katalog F08 | Deklarasi dan adapter sintetis | B0/B1/B3/B4/B5 tersedia; B2 cascade serta E1-E7 paired executors belum tersedia. B4 katalog memakai CVaR tanpa generator variance, B5 menambah routing/generator variance; prototipe F07 dipertahankan |
| Snapshot F08 | Engineering draft saja | Config/catalog/source/runtime/F03 metadata/fixture fingerprints mengikat plan. Final protocol lock tidak dibuat dan dispatcher medis/final test tidak tersedia |
| Seleksi/kalibrasi F08 | Demo interface | Minimum unweighted source-validation CE dan strict score pFB+pFM, target demo0,05. Empirical FPR pada empat authentic volume bukan population guarantee; target FPR penelitian tetap PENDING |
| Resume F08 | Completed-job boundary | Completed artifacts diverifikasi checksum lalu skip; partial job diulang dari seed pada attempt baru. Lock lokal satu runtime, satu VM aktif per Drive run; bukan distributed lock |
| Evaluator F09 | Engineering saja | Membaca prediksi sintetis F08 dan checksum checkpoint sebagai byte; tidak mendeserialisasi checkpoint lama dengan source baru; final test tidak tersedia |
| Metrik F09 | Keputusan demo tetap | Brier sum empat kelas, NLL epsilon1e-12, ECE10 bin, strict frozen threshold; missing-class macro/worst-group undefined menjadi null; tidak menetapkan protokol medis |
| LOGO/lokalisasi F09 | Fixture kontrak | Dua rotasi probabilitas buatan, tidak ada model LOGO dilatih. Titik verified dalam LPS mm/radius demo5mm; Dice/IoU hanya verified voxel masks pada grid/cakupan penuh. Tidak ada point/box-to-mask |
| Efisiensi F09 | Profil terlatih belum tersedia | Bukti random-weight routing F06 tidak digunakan sebagai profil trained B0/B1/B3/B4/B5; matched hardware/input/checkpoint/timing scope masih diperlukan |
| E3/E4/E5 F10 | Synthetic paired controls | FULL versus CVaR removal, generator variance removal dan trained caps64/128/256; input/initial weights sama per seed. E1/E2/E6/E7 tetap terbuka, bukan catalog F08 execution |
| Bootstrap F10 | Deskriptif conditional | Whole-source paired draws1000, seed113, percentile95%; frozen thresholds; setiap undefined replicate membuat CI endpoint null. Tidak ada p-value/multiplicity adjustment/claim konfirmatori |
| Profil F10 | Trained synthetic checkpoints | Fresh workers, batch1/32 cubed, warmup2/samples10/stages3, hardware/input/policy signature sama per seed. CPU RSS mencakup imports/load/fixture; CUDA peaks reset setelah warmup; FLOP exclusions tetap eksplisit |
| Seed/noninferiority F10 | Keputusan penelitian belum selesai | Seed13/37 juga mengubah fixture, tidak dipool menjadi pasien independen atau E7 optimizer-seed-only. Margin tetap PENDING dan noninferiority tidak dinilai |
| Pemilihan checkpoint F04 | Eksplisit | best.pt tetap minimum unweighted source-validation CE; kontrol fitting dilaporkan pada last.pt secara terpisah. Bobot kelas dari selected train saja; threshold max-authentic strict > bersifat eksploratori |
| CVaR dan loss generator | PENDING | Alpha, bobot kelas, estimator batch, dan formulasi penalty dipilih dari source validation |
| Budget dan profiling | PENDING | Ukur hardware runtime aktual dan biaya seluruh komponen |
| Statistik dan lokalisasi | PENDING | Margin noninferiority, radius lokasi, dan unit bootstrap dikunci sebelum final testing |
| Audit data/label lanjutan | ZIP lengkap diverifikasi dan disimpan ke Drive | Run audit_20261111T144921_416065Z: 109 anggota ZIP, 100 laporan kasus, serta 22.753 path/ukuran cocok. Shared slice 6031/7507 dihitung ulang 56/272, tetap satu split. ZIP Drive dibaca ulang dan SHA cocok. UID publik 0/100, selisih README 484 slice dan koordinat tetap terbuka. Lihat reports/data_label_colab_review_20261112.md |

## Amendment 17 November 2026 — M3Dsynth training and coarse-to-fine RSVT

| Decision | Status | Rule |
|---|---|---|
| Training dataset | Set by user, 17 Nov | M3Dsynth is the training/validation dataset for RSVT. This lifts the earlier no-new-dataset-download constraint for M3Dsynth archives and their LIDC-IDRI source series only; downloads run in Colab and only when `PREPARE_DATA=True` |
| Partitions | Adopted explicitly | Publisher patient-disjoint sets (train 488 / validation 100 / test 150 patients). Patient caps select a seeded subset; every generator and site of a selected patient stays together |
| Test data | Reserved | Test patients are not acquired or opened by training, diagnostics or model selection |
| Supervision | Training only | Author voxel masks and manipulation centres supervise router, crop and mask heads during training. Inference uses router top-K crops only; masks are used in validation metrics (router recall, mask Dice), never for selection inputs |
| CT-GAN annotation points | Not used | The CT-GAN point review continues independently; its points are not used to train RSVT. The user stated no preference on point use (17 Nov) |
| Real-series conversion | Verified, not assumed | Re-implements the authors' conversion; any manipulated sample whose archive slices differ from the converted series outside the stored box is ineligible |
| CT-GAN as external test | Pending | CT-GAN derives from LIDC-IDRI with unresolved identities; it is not an independent test until reconciled with M3Dsynth/LIDC patients |
| Proposed model | Changed | Coarse-to-fine RSVT (`docs/rsvt_c2f_m3dsynth.md`). The 96³-token RSVT is kept as a negative control with router-budget initialization, inference gate bias and max+mean pooling enabled in `rsvt_real_ct.toml`; code defaults stay legacy for F05–F10 reproducibility |
| Endpoint objective | Changed | Factorized tampered / removal-vs-insertion heads; tail risk on authentic, FB and FM volume losses; generator-variance regularizer now eligible (three training generators) |
| Checkpoint selection | Set | Lowest validation balanced volume BCE among risk-stage epochs (hard top-K inference) |
| Operating points | Exploratory | Thresholds at 5%/10% FPR are set on the same validation authentic volumes; a frozen endpoint protocol remains PENDING |

## Amendment 24 November 2026 — pre-run review fixes

Made before any M3Dsynth image was downloaded or any model trained.

| Decision | Status | Rule |
|---|---|---|
| Volume-loss class weights | Corrected | Weights follow the logical-batch composition (4 authentic + 12 manipulated carry equal total weight). Dataset-level weights applied on top of the fixed batch composition had given the authentic class 78% of the volume loss |
| Router candidate cells | Corrected | Candidates come from the body-intensity test on the image only. True cells are still router-loss positives but are never added as candidates; validation previously added them from the mask |
| Checkpoint selection | Changed (supersedes 17 Nov) | Lowest validation volume BCE averaged over three equal groups (authentic, removal, insertion) among risk-stage epochs (hard top-K inference). The two-class balanced BCE is still reported |
| Resolution check | Changed | Manipulated crops at declared site centres vs authentic sites of the same appearance (removal vs TB, insertion vs TM); results per operation. Pairing with the same location compared nodule vs no nodule |

## Amendment 25 November 2026 — after the first acquisition

The first acquisition (24–25 Nov) accepted 160 of 3,594 manipulated samples and no model was trained.

| Decision | Status | Rule |
|---|---|---|
| Stored box | Changed | ±48 mm and at least 24 mm of in-plane pixels on every axis. "Cube taller than the old box" matched 448 of 453 removals that had changed voxels outside it; 26 of 26 receipts read fit the new box |
| Label slices past the scan depth | Changed | Content is counted during streaming. Empty: accepted and recorded. With mask voxels: ineligible. The first run did not record their content, so its samples are streamed again rather than relabelled |
| Whole-scan mismatches | Excluded, cause open | 98 samples (39 sites) differ from the LIDC series over more than 1 M voxels. All lie where the edit cube runs off the start of the scan; 172 other samples there do not mismatch |
| Earlier boxes and manifest | Preserved | New boxes go to `boxes_v2`; every earlier manifest is kept in `manifest_history` |

## Amendment 26 November 2026 — labels from the changed voxels

The second acquisition (25–26 Nov, CPU, no training) accepted 3,493 of 3,594 manipulated samples:
785 removals and 2,708 insertions. All 3,344 label slices past the scan depth were empty.

| Decision | Status | Rule |
|---|---|---|
| Where a volume was edited | Changed (supersedes 17 Nov) | Router targets, crop tamper labels and mask targets come from the changed voxels (manipulated ≠ real) instead of the author mask. 784 of 785 removals change voxels outside the author mask box: the mask is a ~33 mm cube, the changes span a cube with as many slices as pixels (median 126 mm deep, 3.9× the mask on 2.5 mm slices). 2,466 of 2,708 insertions stay inside their mask box. Each label's bounding box must equal the one measured while streaming |
| Router target cells | Changed | Every 12³ cell holding a changed voxel (formerly: cells overlapping the author mask box) |
| Crop tamper label | Changed | A crop is tampered when it holds at least a quarter of the changed voxels a crop of its shape can hold. Equal to the earlier 25% rule when the edit fits in a crop |
| Site crop and visible-nodule label | Unchanged | Centred on the author mask box, which lay within a voxel of the centre of the changed voxels; removal 0, insertion 1 |
| Label check | Added | Each training run saves `label_check.json`: share of the mask changed, share of the edit inside the mask, edit depth / mask depth and the share of slices changed within the edit, by operation and generator |

## Amendment 26 November 2026 (later) — first training run and the crop learning check

The first L4 training run (commit `a732b1f`) stopped after epoch 12. The volume tamper, removal and crop tamper
losses stayed at chance on the training volumes for all 600 optimizer steps, and the selected epoch 4 gives every
validation scan about the same score (tamper AUROC 0.427). Details in [rsvt_c2f_m3dsynth.md](rsvt_c2f_m3dsynth.md).

| Decision | Status | Rule |
|---|---|---|
| Next GPU run | Changed | `RUN = "crop_check"` (notebook default for now): train the fine crop network alone on paired crops that always contain the edit, and do not train the full model. `RUN = "train"` trains as before |
| Crop learning check design | Added | Manipulated crop vs the same crop of its real series (differ only in the changed voxels, checked per pair); removals and insertions balanced; arms fit (32 pairs), image and residual (up to 384 training pairs and 192 validation pairs from unseen patients); train/validation only |
| Auto-resume | Narrowed | Only `RUN = "train"` continues an unfinished training run; a crop-check run starts its own folder |
| Checkpoint selection | Open | The group-balanced validation BCE is 0.637 for the best constant prediction, so a constant model can be selected and early stopping can count from it. To be replaced before the next training run |

## Amendment 26 November 2026 (evening) — after the crop learning check

The crop learning check (L4, commit `2884b66`, 192 validation pairs from 64 unseen patients) found that a fresh
crop network learns the edit from the residual (crop AUROC removal 0.94, insertion 0.75) but not from the windowed
image (0.69 for both), which it fitted by memorising anatomy. Details in [rsvt_c2f_m3dsynth.md](rsvt_c2f_m3dsynth.md).

| Decision | Status | Rule |
|---|---|---|
| Crop network input | Changed | `fine_input = "residual"`: each crop minus its 3×3×3 local mean (the filter tested in the check) |
| Checkpoint selection | Changed (closes the open item above) | Highest mean of removal-vs-authentic and insertion-vs-authentic validation AUROC among risk-stage epochs; starts at chance (0.5) and needs a gain of `min_delta`, so a constant model is never selected; early stopping counts epochs without a new best. If no epoch beats chance, the last epoch is reported and labelled so. Checkpoint format `rsvt_c2f_state_v2` |
| Site-crop diagnostic | Added | Each validation scan is also scored with one extra crop at the author manipulation centre (manipulated) or one of its series' real TB/TM sites (authentic); reports scan AUROC with that crop and crop AUROC on it alone. Diagnostic only: never used for selection or training; router-only scores unchanged |
| Next GPU run | Changed | `RUN = "train"` is the notebook default again |

## Amendment 27 November 2026 — second training run and the crop pretraining check

The second L4 run (commit `4b585c8`, residual crops, AUROC selection) learned nothing. The volume tamper loss stayed
at chance on the training scans for all 25 epochs, the crop network matched a constant prediction, and adding the
edit's crop changed no validation scan score by more than 0.0007. Validation removal AUROC 0.626 (95% interval
0.554–0.692, resampling patients), insertion 0.506 (0.432–0.580). Details in [rsvt_c2f_m3dsynth.md](rsvt_c2f_m3dsynth.md).

| Decision | Status | Rule |
|---|---|---|
| Next GPU run | Changed | `RUN = "pretrain_check"` (notebook default): two short checks, no full training |
| Crop pretraining | Added | The crop check's recipe on the crops the full model cuts; weights saved with a checksum receipt; transfer into a fresh full model verified on the same validation crops and through the model's data path (tolerance 1e-3) |
| Scan-classifier swap test | Added | Crop network frozen; the rest trained for 300 logical batches without tail risk; each validation edit scored with one slot swapped between the edited crop and the same crop of the unedited scan (identical context, 16 crops, other edited cells excluded); paired change per operation with patient-level intervals. The unedited crop is a diagnostic reference only |
| Run 3 | Conditional | Only if both checks pass: change only the crop network's starting weights; keep the router and the checkpoint rule; track the same validation pairs and scan AUROC per operation every epoch |
| Checkpoint rule | Unchanged | A proposed 0.05 margin above chance was dropped: run 2's selected score (0.566) would have passed it, and `min_delta` also sets the improvement step between epochs. Whether a run is useful is judged separately, from effect sizes with patient-level intervals |
| Paired crop loss in joint training | Deferred | Tested only if the fixed validation pairs degrade during run 3 |
| Dataset read-back | Narrowed | `RECHECK_DATASET = False` (default, user request): section 7 skips the full read-back (58 min in run 2) when an earlier run read the same store back in full without failures; files are still checked against their manifest checksums when loaded, and the run records which full read-back it relied on |

Kelanjutan QA 13 November: run `qa_20261113T005049_219029Z` telah direview dari
Drive. Semua 103 file metadata cocok checksum; 300 hash header sampel cocok F01;
123 preview tercatat (110 train/13 validation), 25 titik test dikecualikan.
Identitas publik masih 0/100 dan semua keputusan visual masih pending.
`training_ready=false`; lihat `reports/source_annotation_qa_colab_review_20261113.md`.

Perubahan sesudah final test dibuka harus dicatat terpisah dari protokol konfirmatori.
Tahap inventaris tidak menetapkan target model atau mengklaim independensi pasien
berdasarkan jumlah baris metadata saja.
