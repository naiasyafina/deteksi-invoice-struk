# Scan Struk

## Ringkasan

Aplikasi web untuk membaca foto struk belanja. Gemini mengambil nama toko, tanggal, daftar barang, diskon, pajak, dan total dari foto. Pengguna mengecek dan mengoreksi hasilnya, lalu menyimpannya ke SQLite. Tab Riwayat menampilkan rekap pengeluaran. Ini "versi kita" dari Mindee / RECEIPT-CONTROL-APP, tugas dari kating.

**Stack:** Python, Streamlit (UI), Gemini API lewat `google-genai` (model default `gemini-3.1-flash-lite`), SQLite (`sqlite3` bawaan Python), pandas, pydantic.

### Struktur file

| File | Peran |
|---|---|
| `app.py` | UI Streamlit. Tab **Scan**: upload foto, baca struk, cek dan edit hasil, cek apakah total cocok dan apakah struk sudah pernah disimpan, lalu simpan. Tab **Riwayat**: total pengeluaran, grafik per bulan, rekap per toko, unduh CSV, detail item, dan hapus struk. |
| `receipt_reader.py` | Memanggil Gemini dan mendefinisikan skema `Receipt`/`Item` (pydantic) yang dipakai sebagai format output JSON. Juga berisi `expected_total()`, yaitu jumlah harga item dikurangi diskon ditambah pajak. |
| `storage.py` | Lapisan SQLite. Tabel `receipts` dan `items` di `receipts.db`, plus fungsi simpan, hapus, cek duplikat, dan query rekap. |
| `requirements.txt` | Daftar dependency. |
| `.env.example` | Template variabel lingkungan (`GEMINI_API_KEY`, `GEMINI_MODEL`). |
| `.env` | Konfigurasi lokal, tidak di-commit. |
| `receipts.db` | Database, dibuat otomatis saat app pertama kali jalan. Tidak di-commit. |
| `benchmark.py` | Benchmark internal pembaca struk di test split CORD v2 (Hugging Face `naver-clova-ix/cord-v2`). Metode `gemini`: foto langsung ke Gemini. Metode `paddle`: OCR lewat PaddleOCR API online (`PP-OCRv5`), teksnya dikirim ke Gemini. Hasil per struk ke `benchmark_results/<metode>.csv`, ringkasan ke `benchmark_results/summary.csv`. |
| `requirements-benchmark.txt` | Dependency tambahan untuk benchmark (`datasets`, `paddleocr`). Tidak memakai `paddlepaddle` karena OCR lewat API. |
| `benchmark_local.py` | Benchmark PaddleOCR lokal di CPU (PP-OCRv6 `tiny`/`small`/`medium`, engine onnxruntime). Teks OCR dikirim ke Gemini lewat `extract_from_ocr()`. Mengukur `ocr_seconds`, `cpu_percent`, `peak_ram_mb`, `word_recall`. Hasil ke `benchmark_results/local-*.csv`, `local_resources.csv`, `local_startup.json`. |
| `requirements-local.txt` | Dependency benchmark lokal (`onnxruntime`, `psutil`, plus isi `requirements-benchmark.txt`). |

## Cara menjalankan

1. Isi `GEMINI_API_KEY` di `.env` dengan API key dari Google AI Studio.
2. Dari folder proyek:

   ```bash
   source .venv/bin/activate
   streamlit run app.py
   ```

3. Buka http://localhost:8501.

Kalau `.venv` belum ada: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## Benchmark

1. Isi `PADDLEOCR_ACCESS_TOKEN` di `.env` (dipakai metode `paddle`), selain `GEMINI_API_KEY`.
2. Install dependency tambahan:

   ```bash
   source .venv/bin/activate
   pip install -r requirements-benchmark.txt
   ```

3. Jalankan per metode, lalu ringkasan:

   ```bash
   python benchmark.py gemini --limit 20
   python benchmark.py paddle --limit 20
   python benchmark.py summary
   ```

   Opsi `--limit` (default 100) menentukan jumlah struk dan `--delay` (default 5 detik) menentukan jeda antar struk. Kalau dijalankan ulang, struk yang sudah berhasil dilewati, sedangkan struk yang error dicoba lagi. Error 429/503 dan error jaringan di-retry sekali setelah 60 detik. Kalau 429/503 masih muncul, script berhenti dan run berikutnya melanjutkan dari situ.

Metrik di `summary.csv`: `total_accuracy` (total sama dengan ground truth, hanya untuk struk yang ground truth-nya punya total), `item_price_f1` (harga item cocok persis), `item_name_f1` (nama item mirip ≥ 0.8), `self_consistent` (jumlah item − diskon + pajak = total), `median_seconds` dan `max_seconds` (waktu per struk yang berhasil).

### Hasil 2026-11-25 (20 struk pertama test split)

| method | receipts | failed | total_accuracy | item_price_f1 | item_name_f1 | self_consistent | median_seconds | max_seconds |
|---|---|---|---|---|---|---|---|---|
| gemini | 20 | 0 | 1.0 | 0.87 | 0.87 | 0.9 | 6.775 | 137.42 |
| paddle | 20 | 0 | 1.0 | 0.866 | 0.758 | 0.9 | 19.355 | 117.44 |

Akurasi total dan harga item setara. Gemini langsung lebih baik di nama item (0.87 vs 0.758) dan sekitar 3x lebih cepat (median 6.8 vs 19.4 detik). Sampel masih 20 struk, jadi belum cukup untuk keputusan final.

## Log perubahan

- 2026-11-25: Setup awal proyek. File dipindahkan dari Downloads, lalu dibuat `.gitignore`, `.env`, dan virtual environment `.venv` (Python 3.14, streamlit 1.67.0, google-genai 3.6.1). Verifikasi: `py_compile` lolos, import modul lolos, dan health check Streamlit headless membalas `ok`.
- 2026-11-25: Setup benchmark. `benchmark.py` dan `requirements-benchmark.txt` dipindahkan dari Downloads, variabel `.env` baru `PADDLEOCR_ACCESS_TOKEN`, dan terpasang paddleocr 3.7.0, paddlex 3.7.2, serta datasets 5.2.0 (tanpa paddlepaddle). Smoke test pertama metode `paddle` gagal di struk #0 karena `NetworkError` ke server PaddleOCR, lalu `benchmark.py` diganti versi baru: error jaringan ikut di-retry, struk yang error dicoba ulang saat run berikutnya, pesan error disimpan sampai 300 karakter, summary memakai baris terakhir per struk, `avg_seconds` diganti `median_seconds`/`max_seconds`, dan header CSV tidak lagi tertulis dua kali. Struk #0 berhasil saat dicoba ulang. Run 20 struk per metode selesai tanpa error dan tanpa rate limit (hasil di bagian Benchmark).
- 2026-11-25: Benchmark lokal (belum selesai). Ditambahkan `benchmark_local.py` dan `requirements-local.txt`, lalu `benchmark.py` diganti versi yang memisahkan `extract_from_ocr()` dan membuat `summary` melewati CSV tanpa kolom `image_id`. Terpasang onnxruntime 1.32.0 dan psutil 7.2.2 di Python 3.14, tanpa paddlepaddle. Model PP-OCRv6 ONNX di `~/.paddlex/official_models` totalnya 170 MB. Laptop uji: i5-12450H (4 P-core + 4 E-core, 12 thread), RAM 15 GiB, avx2 tanpa avx512. Run 20 struk `tiny`/`small`/`medium` dan `tiny-1c` (10 struk) selesai. `tiny-2c`, `medium-1c`, dan `medium-2c` belum selesai (0, 0, dan 2 struk) karena kuota Gemini habis. Simulasi 2 core memakai `taskset -c 0,2`, bukan `0,1`, karena CPU 0 dan 1 adalah hyperthread dari satu P-core. Sisa run: jalankan ulang tiga perintah itu setelah kuota pulih, lalu `python benchmark_local.py summary`. Yang perlu dicek: `peak_ram_mb` medium 4980 MB jauh di atas tiny (1686 MB) dan small (2210 MB).
