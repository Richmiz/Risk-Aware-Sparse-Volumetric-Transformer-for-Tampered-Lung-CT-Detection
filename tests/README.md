# Rencana pengujian

Pengujian F01 tersedia di `test_f01_inventory.py`, seluruhnya menggunakan fixture
sintetis. Jalankan `python -m unittest discover -s tests -p 'test_f01*.py' -v`.
Pengujian F02 di `test_f02_preprocessing.py` memeriksa geometri, HU per slice,
padding, resampling, transformasi anotasi, grouping, dan ekspor batch sintetis.
Semua pengujian dijalankan dengan `python -m unittest discover -s tests -p 'test_f0*.py' -v`.
Pengujian F03 memeriksa grouping, draft partitions, dan kontrak loader.
F04 menambahkan forward/backward CNN/ViT, loss batch-1, preservasi support FOV,
validation metrics/ties, optimizer, early stopping, checkpoint/RNG, resume yang
setara dengan run tanpa terputus, dan pipeline cache sintetis. PyTorch CPU dipakai
untuk pemeriksaan lokal; data medis proyek tetap dijalankan pada Colab.
Tambahkan pengujian perilaku nyata saat modulnya tersedia:

- Manifest: konflik sumber/pasien/hash dan generator holdout.
- Preprocessing: konsistensi HU, orientasi, spacing, dan pemetaan anotasi.
- Model: cakupan coarse, token terpilih, gradien, dan forward pass.
- Loss: tail-risk per kelas, batch tanpa kelas tertentu, serta mask yang tidak tersedia.
- Evaluasi: threshold source-only, agregasi per volume, dan bootstrap kelompok.
- Profiling: budget aktual dan overhead controller.

Jangan menyimpan citra pasien sebagai fixture. Gunakan volume sintetis kecil
untuk perilaku yang dapat diuji tanpa dataset medis.

F05 di `test_f05_model.py` menambahkan geometri fisik dan patch tepi, cakupan
coarse, cue tanpa wrap, invariansi padding/invalid NaN, forward/backward 96³,
gradien seluruh komponen, state-dict roundtrip, budget preflight, lokalisasi
voxel-mask terverifikasi dan integrasi CLI sintetis. Tidak ada dataset medis
yang dibaca. Pemeriksaan lokal memakai CPU; dukungan CUDA menunggu run Colab.

F06 di `test_f06_routing.py` menambahkan 17 tes untuk gradien CE→router tanpa
budget penalty, gate proxy, stable ranking, hard gather sebelum attention,
coarse floor/zero-candidate, shared-weight F05/dense/all-kept equivalence,
padding/invalid invariance, save/load router, pengukuran worker, kalibrasi
infeasible dan audit kandidat. Soft backward 96³ dan hard cap 64 juga diuji.
Suite F01-F06 lokal: 156 tes lolos. Tes memakai tensor prosedural saja.

F07 menambahkan 23 tes di `test_f07_risk.py`: nilai/subgradien fractional CVaR,
kelas kosong, generator variance per kelas, gate ID/partition/support, projected
dual/state, scatter/mask fine token, objective weighting, pembekuan parameter
lintas tahap, gradien risiko terpisah ke router, dan pipeline B4/B5 kecil.
Suite F01-F07: 179 tes lolos. Kedua generator fixture adalah grup prosedural;
tidak ada training medis atau pembukaan holdout penelitian.

F08 menambahkan 21 tes di `test_f08_orchestration.py`: registry/proposal controls,
metadata-only manifest checks, source leakage/assignment corruption, research
readiness, strict empirical FPR/ties, checkpoint/prediction binding, completed-job
resume, partial-attempt preservation, writer lock dan validation replay. Suite
F01-F08: 200 tes lolos, termasuk kontrol affine yang tidak mengodekan kelas.
Final-test dispatcher tidak tersedia pada F08.

F09 menambahkan `test_f09_evaluation.py`: AUC dengan rank/ties dibanding oracle
pairwise, dukungan kelas kosong, Brier/NLL/ECE analitik, boundary bin/roundoff,
frozen threshold pada holdout, shared authentic/worst-generator, kebocoran
parent/patient/content/generator, affine LPS/mm, matching titik satu-ke-satu,
mask coverage/grid/empty union, kebijakan test gate, serta pipeline CSV/JSON
yang reproducible tanpa checkpoint deserialization atau perubahan input F08.
Suite F01-F09 memiliki 222 tes, termasuk 22 tes baru F09.

F10 di `test_f10_statistics.py` menambahkan 14 tes: analytic bootstrap oracles,
unequal whole clusters/shared draws, pairing order/provenance, missing resampled
classes, explicit patient key, fixed thresholds, seed/parameter guards,
noninferiority boundary/margin, Pareto dominance/matched signatures, exact
variant switches, config gates, dan lima checkpoint/profile workers sungguhan
dengan analisis replay identik tanpa model load. Gunakan glob `test_f*.py` agar
F10 turut diuji. Korpus dan checkpoints seluruhnya prosedural.
Suite F01-F10: 236 tes lolos pada CPU.

F11 (17 Nov 2026) `test_f11_m3dsynth.py`, 18 tests on procedural DICOM series
and LZMA TIFF archives: authors' uint16 rule (incl. corner and int16-wrap cases),
descending-z conversion and header/UID checks, streaming with spilled half-finished
samples, rejection of changes outside the stored box, exact coarse-view box delta,
crop paste and flip consistency, model shapes/selection/compute estimate, tail
risk on both threshold sides, scalar replay equal to direct backprop, resume equal
to an uninterrupted run, diagnostics, the full notebook session flow, and the
96³ router fixes. Full suite: 338 tests pass on CPU.
