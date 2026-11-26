# Ringkasan Benchmark

Benchmark model pretrained untuk klasifikasi invoice/struk dan ekstraksi isi struk, dijalankan pada 2026-11-26. Tanpa training, fine-tuning, atau ML klasik. Semua model lokal berjalan di CPU. Gemini hanya sebagai pembanding cloud.

Data uji: 100 struk CORD test, 25 invoice Sparrow (digital), dan 25 invoice RVL-CDIP (scan lama). Detail di [`data/SOURCES.md`](../data/SOURCES.md). Data milik tim (`data/own/`) **belum diisi**, jadi kolom `own`, tanggal, dan nama toko masih kosong.

Cara membaca kolom:
- `total EM`: total hasil ekstraksi sama persis dengan label. Hanya dihitung di gambar yang punya label total (95 dari 100 struk CORD).
- `F1 item`: item dianggap cocok kalau harganya sama dan namanya mirip ≥ 0.8.
- `CPU%`: 100 = satu core penuh.
- `puncak RAM`: RSS tertinggi proses, termasuk Python + torch/paddle.
- Baris `threads = 2` dijalankan dengan `taskset -c 0,2` (2 core fisik, 20 gambar per set) untuk simulasi laptop lemah.
- Waktu Gemini sudah termasuk jaringan, tetapi tidak termasuk jeda tunggu saat terkena rate limit.

<!-- tabel:mulai -->
## Mesin uji

- CPU: 12th Gen Intel(R) Core(TM) i5-12450H (8 core fisik, 12 thread)
- RAM: 15.3 GB
- OS: Linux-7.2.11-arch1-1-x86_64-with-glibc2.44
- Python 3.12.9, semua inferensi di CPU

## Klasifikasi (invoice vs struk)

| model | n | akurasi | akurasi cord | akurasi invoice | akurasi own | recall struk | recall invoice | median s/gambar | p90 s | CPU% rata-rata | puncak RAM MB | load s | threads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openai/clip-vit-base-patch32 | 150 | 0.913 | 0.87 | 1.0 | - | 0.87 | 1.0 | 0.078 | 0.136 | 335 | 1138 | 4.71 | 4 |
| victormurcia/DiT-DocumentClassifier-v1 | 40 | 1.0 | 1.0 | 1.0 | - | 1.0 | 1.0 | 0.216 | 0.249 | 186 | 888 | 2.46 | 2 |
| victormurcia/DiT-DocumentClassifier-v1 | 150 | 0.94 | 0.95 | 0.92 | - | 0.95 | 0.92 | 0.13 | 0.176 | 368 | 955 | 2.34 | 4 |

Confusion matrix:

**classify-clip** (openai/clip-vit-base-patch32, 4 thread)

| asli \ prediksi | invoice | struk |
|---|---|---|
| invoice | 50 | 0 |
| struk | 13 | 87 |

**classify-dit-2core** (victormurcia/DiT-DocumentClassifier-v1, 2 thread)

| asli \ prediksi | invoice | struk |
|---|---|---|
| invoice | 20 | 0 |
| struk | 0 | 20 |

**classify-dit** (victormurcia/DiT-DocumentClassifier-v1, 4 thread)

| asli \ prediksi | invoice | lainnya | struk |
|---|---|---|---|
| invoice | 46 | 2 | 2 |
| struk | 5 | 0 | 95 |

## Ekstraksi isi

| engine | n | error | total EM CORD | F1 item CORD | total EM own | tanggal own | toko own | median s/gambar | p90 s | CPU% rata-rata | puncak RAM MB | load s | threads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naver-clova-ix/donut-base-finetuned-cord-v2 | 20 | 0 | 0.947 | 0.784 | - | - | - | 5.722 | 6.644 | 197 | 2016 | 3.4 | 2 |
| naver-clova-ix/donut-base-finetuned-cord-v2 | 100 | 0 | 0.979 | 0.876 | - | - | - | 4.222 | 5.453 | 394 | 2131 | 3.78 | 4 |
| gemini-3.1-flash-lite | 100 | 0 | 0.989 | 0.899 | - | - | - | 4.972 | 14.967 | 1 | 385 | 0.56 | 4 |
| PP-OCRv6_small + regex | 100 | 0 | 0.758 | 0.649 | - | - | - | 0.387 | 0.726 | 690 | 1854 | 1.53 | 4 |
| PP-OCRv6_tiny + regex | 20 | 0 | 0.737 | 0.612 | - | - | - | 0.251 | 0.449 | 199 | 1176 | 1.43 | 2 |
| PP-OCRv6_tiny + regex | 100 | 0 | 0.726 | 0.612 | - | - | - | 0.154 | 0.372 | 700 | 1550 | 1.28 | 4 |
<!-- tabel:selesai -->

## Temuan

**Klasifikasi**
- DiT paling seimbang: akurasi 0.94, sekitar 0,13 s per gambar, RAM sekitar 1 GB. Semua 25 invoice Sparrow benar. Salahnya ada di 5 struk CORD (terbaca invoice) dan 4 scan RVL-CDIP (terbaca cek/struk).
- CLIP zero-shot lebih cepat dan semua invoice benar, tetapi 13% struk terbaca invoice. Hasilnya juga bergantung pada kalimat prompt.
- Skor confidence DiT bisa dipakai untuk minta konfirmasi pengguna. Dengan ambang 0.6, 28 dari 150 gambar (19%) ditandai, dan 6 dari 9 kesalahan ikut tertangkap. Ambang 0.7 menangkap 8 dari 9, tetapi menandai 37% gambar.

**Ekstraksi struk**
- Donut CORD paling akurat di antara model lokal: total EM 0.979 dan F1 item 0.876. Harganya: sekitar 4,2 s per struk di 4 thread (5,7 s di 2 core) dan RAM sekitar 2,1 GB.
- Batas Donut: model ini dilatih dengan CORD train, jadi CORD test adalah kondisi terbaiknya. Schema CORD tidak punya nama toko maupun tanggal, dan model ini tidak bisa membaca invoice.
- Gemini Flash-Lite setara dengan Donut (total EM 0.989, F1 item 0.899) dengan RAM lokal sangat kecil. Tetapi butuh internet, latensi p90-nya sekitar 15 s, ada rate limit, dan gambar struk dikirim ke server Google.
- PaddleOCR + regex paling ringan dan cepat (tiny: 0,15 s, 1,5 GB), tetapi total EM hanya 0.73 (tiny) atau 0.76 (small). Sebagian besar kegagalan (22 dari 26 di tiny) terjadi karena regex tidak menemukan total sama sekali, bukan karena salah baca angka. Di Fase 1, regex yang sama di teks ground truth mencapai 0.895, jadi batas utamanya ada di parser, bukan di OCR.
- Batas thread PaddleOCR tidak sepenuhnya dipatuhi: CPU% sekitar 700 walaupun diset 4 thread. Kalau dibatasi `taskset` ke 2 core, waktunya naik ke 0,25 s.

## Rekomendasi stack

| Langkah | Model | Alasan |
|---|---|---|
| Klasifikasi | DiT `victormurcia/DiT-DocumentClassifier-v1` | akurasi 0.94, 0,13 s; minta konfirmasi pengguna kalau skor < 0.6 |
| Struk: item, subtotal, total | Donut `donut-base-finetuned-cord-v2` | akurasi lokal tertinggi; output JSON langsung |
| Struk: tanggal, nama toko | PaddleOCR PP-OCRv6 tiny + regex | Donut tidak mengeluarkan field ini; tambahannya hanya 0,15 s |
| Invoice | PaddleOCR PP-OCRv6 tiny + regex | Donut CORD tidak cocok untuk invoice; akurasinya belum terukur karena perlu data `own/invoice` |
| Fallback (opsional) | Gemini Flash-Lite | hanya kalau total ≠ jumlah item atau confidence rendah, dan pengguna setuju datanya dikirim ke cloud |

Perlu dicek ulang setelah data tim masuk ke `data/own/`:
- akurasi DiT pada foto struk HP asli
- akurasi Donut pada struk di luar format CORD
- akurasi tanggal, nama toko, dan total invoice dari Paddle + regex

## Perkiraan spek minimum laptop

Diukur di i5-12450H. Angka untuk laptop lain adalah perkiraan dari simulasi 2 core, belum diuji di laptop sungguhan.

| | Minimum | Disarankan |
|---|---|---|
| CPU | 4 thread x86-64 dengan AVX2 (misalnya Intel gen 8+ atau Ryzen 2000+) | 4 core / 8 thread |
| RAM | 8 GB | 16 GB |
| Disk | sekitar 4 GB: venv 2 GB, model sekitar 1,1 GB (DiT 328 MB, Donut 773 MB, Paddle tiny 7 MB) | SSD |
| Waktu per struk | sekitar 6-8 s (DiT + Donut + Paddle di 2 core) | sekitar 4,5 s di 4 thread |

Dasar perkiraan RAM: kalau semua model dimuat dalam satu proses, puncaknya diperkirakan sekitar 3-3,5 GB. Angka ini berasal dari jumlah puncak per model dikurangi baseline Python/torch yang terhitung dobel, dan belum diukur langsung. Dengan OS dan browser, 8 GB masih cukup. Kalau RAM hanya 4 GB, pakai Paddle tiny + regex saja (sekitar 1,2-1,5 GB), dengan risiko akurasi turun.
