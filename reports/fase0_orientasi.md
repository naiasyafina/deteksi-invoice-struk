# Fase 0 — Orientasi

Tanggal: 2026-11-26

## Kondisi awal

Folder berisi app Streamlit + Gemini (`app.py`, `storage.py`, `receipt_reader.py`), benchmark Gemini vs PaddleOCR API vs PaddleOCR lokal (`benchmark.py`, `benchmark_local.py`, `benchmark_results/`), dokumentasi, dan venv Python 3.14. Folder ini belum jadi repo git. `receipts.db` kosong (0 struk).

## Yang dipindah ke `legacy/`

`app.py`, `storage.py`, `receipt_reader.py`, `benchmark.py`, `benchmark_local.py`, tiga `requirements*.txt` lama, `.env.example`, `dokumentasi.md`, `PANDUAN.md`, `benchmark_results/`, `benchmark-struk.zip`, `receipts.db`. Tidak ada file yang dihapus. Venv lama diganti nama menjadi `.venv-legacy` (Python 3.14, isinya paddleocr + google-genai). `.env` tetap di root dan masih terbaca oleh `load_dotenv()` dari `legacy/`.

## Yang disalin dan dirapikan (tanpa import dari legacy)

| Asal | Tujuan | Perubahan |
|---|---|---|
| `benchmark.parse_money` | `pipeline/postprocess.py` | Mengambil angka terakhir di teks (`"2 x 12.500"` → 12500), menjaga tanda minus (diskon `-5,000`), mengembalikan `None` kalau tidak ada angka. Diuji dengan 11 format, termasuk `Rp 12.500`, `12.500,00`, `12,500`, `12,500.00`, `Rp12.500,-`. |
| `benchmark.normalize` | `pipeline/postprocess.py` | sama |
| `benchmark.rows_to_text` | `pipeline/ocr.py` → `group_lines` | Mengembalikan list baris, bukan string, supaya bisa langsung dipakai regex per baris |
| `similarity`, `count_price_matches`, `count_name_matches`, `f1` | `scripts/metrics.py` | Nama dibuat generik (`count_exact_matches`, `count_fuzzy_matches`) |
| `benchmark_local.word_recall` | `scripts/metrics.py` | Parsing JSON CORD dipisah ke `cord_words()`, supaya `word_recall` bisa dipakai untuk data sintetis |
| `peak_ram_mb`, `used_cpu_seconds` | `scripts/profiling.py` | sama |

Koreksi atas laporan rencana: sebelumnya saya bilang `parse_money` lama salah membaca `"12,500"`. Itu keliru, hasilnya sudah benar (12500). Kelemahannya yang sebenarnya ada di teks yang berisi lebih dari satu angka (semua digit digabung) dan tanda minus yang hilang.

## Struktur baru

```
app/ pipeline/ data/{cord,synthetic,real_invoices,other}/ scripts/ notebooks/ models/ reports/ legacy/
```

`data/*/*` dan `models/*` di-.gitignore. Yang tetap di-commit: `.gitkeep`, `data/*/README.md`, dan nanti `data/synthetic/examples/`. `data/real_invoices/README.md` menjelaskan bahwa isinya data pribadi yang dibagi lewat Drive.

## Dependency

| File | Isi | Ukuran / waktu |
|---|---|---|
| `requirements.txt` | rapidocr_onnxruntime 1.4.4, onnxruntime 1.32.0, OpenCV 5.1.0, numpy, pillow, psutil, pandas, datasets | venv 797 MB, install 43 detik. Import RapidOCR + cv2 + onnxruntime: 0,16 detik, RSS 200 MB. Tanpa torch. |
| `requirements-train.txt` | torch, torchvision, transformers, datasets, accelerate, seqeval, scikit-learn, onnx, onnxruntime | Untuk Colab, tidak dipasang lokal |

Keputusan:
- **Python 3.12.9** dari pyenv (`~/.pyenv/versions/3.12.9`), disamakan dengan Colab sesuai permintaan.
- **`opencv-python-headless` dibatalkan.** `rapidocr_onnxruntime` mewajibkan `opencv-python`, dan kalau dua-duanya terpasang, keduanya rebutan folder `cv2/`. Cukup satu, dan yang dipakai `opencv-python` karena diwajibkan RapidOCR.
- `datasets` ikut di `requirements.txt` karena dipakai oleh script evaluasi (CORD). Paket ini membawa pyarrow (170 MB), bagian terbesar venv. Kalau nanti app perlu dipasang tanpa evaluasi, `datasets` bisa dipindah ke file terpisah.
- `scikit-learn` dan `streamlit` belum dimasukkan. Keduanya ditambahkan di fase yang memakainya (Fase 2 dan 4).

## Verifikasi

| Cek | Hasil |
|---|---|
| Self-check `pipeline.postprocess`, `pipeline.ocr`, `scripts.metrics` | ok. Satu assert sempat gagal karena ekspektasi test-nya salah (`12.500` dan `12,500` memang setara setelah dinormalisasi), lalu test-nya diperbaiki. |
| `py_compile legacy/*.py` + import modul legacy dari `legacy/` dengan `.venv-legacy` | ok (ada peringatan `sys.prefix` dari path venv relatif, tidak berbahaya) |
| `grep legacy\|receipt_reader\|google` di `pipeline/ scripts/ app/` | tidak ada |
| 27 file lama masih ada semua (di root atau `legacy/`) | ok |
| `git status` sebelum commit | `.env`, `.venv*`, `*.db`, `*.zip` tidak ikut |

Waktu/CPU/RAM untuk OCR belum diukur karena itu bagian Fase 1. Aset yang siap dipakai di Fase 1: CORD v2 sudah ada di cache Hugging Face (2,2 GB), dan model PP-OCRv6 ONNX sudah ada di `~/.paddlex` (170 MB) untuk pembanding PaddleOCR.
