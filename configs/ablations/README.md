# Konfigurasi ablation

Pemetaan E1-E7 ada di `docs/experiment_plan.md`. Deklarasi per kontras sudah
tersedia pada E1.toml sampai E7.toml dan dapat digenerasi ulang dengan
`scripts/build_phase8_catalog.py`. Semua masih design-only; paired executors
menunggu F10 serta data/protokol yang memenuhi syarat. Tambahkan konfigurasi
per kondisi setelah dataset, kapasitas model dan budget penelitian ditetapkan.
Gunakan `extends` untuk mewarisi konfigurasi eksperimen dan override hanya
komponen yang diuji. Daftar pada TOML diganti seluruhnya, bukan digabungkan.

Setiap kontras harus mempertahankan partisi data, aturan pemilihan model,
preprocessing, dan pembanding biaya komputasi yang sesuai hipotesisnya.
