# Scan Struk

Deteksi invoice/struk dengan model pretrained di CPU, lalu digitalisasi isinya ke Excel. Repo ini berisi aplikasi Streamlit dan benchmark yang dipakai untuk memilih stack.

Aturan dari kating: hanya model deep learning pretrained, tanpa training/fine-tuning, tanpa ML klasik. Inferensi lokal diutamakan, cloud (Gemini) hanya sebagai pembanding.

Hasil dan rekomendasi: [`results/SUMMARY.md`](results/SUMMARY.md). Sumber data dan lisensinya: [`data/SOURCES.md`](data/SOURCES.md).

## Setup

```bash
~/.pyenv/versions/3.12.9/bin/python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # isi GEMINI_API_KEY kalau mau ikut membandingkan Gemini
```

Torch dipasang versi CPU lewat index PyTorch yang ada di `requirements.txt`.

## Aplikasi

```bash
.venv/bin/python -m streamlit run app.py
```

Alurnya:

1. Upload satu atau beberapa foto.
2. DiT mendeteksi jenisnya secara lokal: struk atau invoice. Label cek bank dari model diabaikan.
3. Hasil deteksi langsung dipakai dan tidak bisa diganti. Kalau keyakinan di bawah 60%, muncul peringatan.
4. Tombol **Digitalisasi** mengirim gambar ke Gemini (butuh `GEMINI_API_KEY` di `.env`).
5. Hasilnya tampil sebagai tabel yang bisa diedit, lalu bisa diunduh sebagai Excel dengan sheet `Struk`, `Invoice`, dan `Item`.

Kodenya ada di `app.py`, `pipeline/classify.py`, `pipeline/gemini.py`, dan `pipeline/export.py`.

## Benchmark

Semua perintah dijalankan dari root repo.

```bash
.venv/bin/python -m scripts.prepare_data
```

Mengambil 100 struk CORD test, 25 invoice Sparrow, dan 25 invoice RVL-CDIP (streaming), lalu membuat `data/own/labels.csv`. Foto milik tim ditaruh di `data/own/` (cara isi ada di `data/SOURCES.md`).

Klasifikasi invoice vs struk:

```bash
.venv/bin/python -m scripts.bench_classify --model dit
```

```bash
.venv/bin/python -m scripts.bench_classify --model clip
```

Ekstraksi isi struk (`--engine donut|paddle-tiny|paddle-small|gemini`):

```bash
.venv/bin/python -m scripts.bench_extract --engine donut --limit 100
```

Gemini dilewati otomatis kalau `GEMINI_API_KEY` tidak ada.

Opsi bersama: `--sets` (default `cord,invoice,own` untuk klasifikasi dan `cord,own` untuk ekstraksi), `--limit` (maksimal gambar per set), dan `--threads` (default 4).

Rekap tabel ke `results/SUMMARY.md`:

```bash
.venv/bin/python -m scripts.summarize
```

Hasil mentah per gambar disimpan di `results/<tugas>-<model>.csv`, dan metadata run (waktu load, puncak RAM, spek mesin) di `.json` dengan nama yang sama.

## Struktur

| Path                                                     | Isi                                                                                |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `pipeline/`                                            | preprocess (resize/crop/deskew), OCR, ekstraksi regex (total, tanggal, toko, item) |
| `scripts/prepare_data.py`                              | menyiapkan data uji                                                                |
| `scripts/bench_classify.py`, `bench_extract.py`      | benchmark                                                                          |
| `scripts/metrics.py`, `profiling.py`, `samples.py` | metrik, pengukuran resource (psutil), loader data                                  |
| `scripts/eval_ocr.py`                                  | benchmark OCR Fase 1 (hasilnya di `reports/`)                                    |
| `data/`, `results/`                                  | tidak masuk git, kecuali `SOURCES.md` dan `SUMMARY.md`                         |

## Cek cepat

```bash
.venv/bin/python -m scripts.metrics
```

```bash
.venv/bin/python -m pipeline.extract
```
