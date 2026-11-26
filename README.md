# Scan Struk

Deteksi dan ekstraksi invoice / struk yang berjalan lokal di CPU. Alurnya: foto → klasifikasi (invoice / struk / lainnya) → ekstraksi isi → review di Streamlit → SQLite. Training dilakukan di Colab, lalu hasilnya diekspor ke ONNX dan dijalankan dengan onnxruntime.

## Struktur

| Folder | Isi |
|---|---|
| `app/` | Streamlit + akses DB |
| `pipeline/` | preprocess, ocr, classify, extract, postprocess |
| `data/` | `cord/`, `synthetic/`, `real_invoices/`, `other/` (isinya tidak masuk git) |
| `scripts/` | generator data sintetis, evaluasi, profiling |
| `notebooks/` | training di Colab |
| `models/` | file `.onnx` (tidak masuk git) |
| `reports/` | laporan per fase (`.md` + `.csv`) |
| `legacy/` | versi lama Streamlit + Gemini (lihat `legacy/dokumentasi.md`). Kode baru tidak boleh meng-import dari sini. |

## Setup

Pipeline memakai Python 3.12, sama dengan Colab:

```bash
~/.pyenv/versions/3.12.9/bin/python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` hanya untuk inferensi dan evaluasi lokal, tanpa torch. `requirements-train.txt` dipakai di notebook Colab.

Cek cepat:

```bash
.venv/bin/python -m pipeline.postprocess
.venv/bin/python -m pipeline.ocr
.venv/bin/python -m scripts.metrics
```

## Legacy

Versi Gemini memakai venv Python 3.14 terpisah (`.venv-legacy`) dan membaca `GEMINI_API_KEY` dari `.env` di root:

```bash
cd legacy && ../.venv-legacy/bin/python -m streamlit run app.py
```
