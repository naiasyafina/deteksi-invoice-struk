# Benchmark Pembaca Struk: Panduan untuk Dicoba Sendiri

Kita membandingkan beberapa cara membaca struk belanja di 20 struk pertama test split dataset CORD v2 (Hugging Face `naver-clova-ix/cord-v2`). Dataset ini sudah punya jawaban benar (ground truth).

| Metode | Cara kerja | Butuh |
|---|---|---|
| `gemini` | Foto struk langsung dikirim ke Gemini | `GEMINI_API_KEY` |
| `paddle` | OCR lewat PaddleOCR API online, teksnya dikirim ke Gemini | `GEMINI_API_KEY` + `PADDLEOCR_ACCESS_TOKEN` |
| `local-tiny/small/medium` | OCR di CPU laptop sendiri (PP-OCRv6, onnxruntime), teksnya dikirim ke Gemini | `GEMINI_API_KEY` saja |

## Isi zip

- `benchmark.py`: metode `gemini` dan `paddle`, serta `summary`.
- `benchmark_local.py`: OCR lokal, plus pengukuran CPU dan RAM.
- `receipt_reader.py`: prompt dan skema JSON Gemini. Dipakai oleh kedua script benchmark.
- `requirements.txt`, `requirements-benchmark.txt`, `requirements-local.txt`: daftar dependency.
- `.env.example`: template variabel lingkungan.
- `benchmark_results/`: hasil dari laptopku, untuk pembanding.
- `app.py`, `storage.py`, `dokumentasi.md`: aplikasi Streamlit-nya. Tidak wajib untuk benchmark.

File `.env`, `.venv`, dan `receipts.db` sengaja tidak ikut dikirim.

## Yang perlu disiapkan

1. Python 3.10 atau lebih baru. Aku memakai 3.14, dan onnxruntime 1.32 sudah jalan di versi itu.
2. **API key Gemini milikmu sendiri** dari Google AI Studio. Jangan pakai key orang lain.
3. Token PaddleOCR, hanya kalau mau mencoba metode `paddle`.
4. Ruang disk sekitar 1 GB untuk dataset, model (170 MB), dan dependency.
5. Laptop dalam keadaan dicolok charger, dengan aplikasi berat dan VM ditutup.

## Hasil di laptopku (pembanding)

Laptopku: i5-12450H (4 P-core + 4 E-core, 12 thread), RAM 15 GiB, avx2. Semua run tanpa batas core.

| Metode | Akurasi total | F1 harga item | F1 nama item | Waktu per struk (median, termasuk Gemini) |
|---|---|---|---|---|
| gemini | 1.0 | 0.87 | 0.87 | 6.8 detik |
| paddle (API) | 1.0 | 0.866 | 0.758 | 19.4 detik |
| local-tiny | 1.0 | 0.829 | 0.712 | 2.2 detik |
| local-small | 0.947 | 0.795 | 0.762 | 3.5 detik |
| local-medium | 1.0 | 0.87 | 0.762 | 4.4 detik |

Khusus OCR lokal (tanpa waktu Gemini):

| Model | Waktu OCR (median) | Puncak RAM proses | word_recall |
|---|---|---|---|
| tiny | 0.18 detik | 1.7 GB | 0.77 |
| small | 0.35 detik | 2.2 GB | 0.85 |
| medium | 2.6 detik | 5.0 GB | 0.91 |

## Cara membaca hasil

- **Akurasi total:** seberapa sering total bayar sama persis dengan ground truth.
- **F1 harga dan F1 nama item:** 1.0 berarti semua item terbaca dan tidak ada item karangan. Harga harus sama persis. Nama cukup mirip minimal 80%.
- **self_consistent:** jumlah harga item − diskon + pajak sama dengan total yang dibaca model itu sendiri.
- **word_recall:** persentase kata di struk yang berhasil ditemukan oleh OCR lokal. Ini murni kualitas OCR, tanpa Gemini.
- **cpu_percent:** 100% berarti satu core penuh, jadi 800% berarti sekitar 8 core.
- **peak_ram_mb:** puncak RAM seluruh proses Python, termasuk dataset dan client Gemini.

## Yang aku pelajari (biar kamu tidak kena juga)

- **Kuota Gemini gratis cepat habis.** Satu set lengkap (gemini, paddle, 3 model lokal, simulasi) butuh sekitar 140 panggilan, dan kuotaku habis di sekitar 120. Bagi ke dua hari, atau jalankan yang penting saja dulu. Hasil yang sudah selesai tidak diulang saat script dijalankan lagi.
- **Script memang tidak cepat.** OCR-nya kencang, tapi ada jeda 5 detik antar struk (`--delay`) ditambah waktu Gemini. Hitung sekitar 3–5 menit untuk 20 struk per metode.
- **Error jaringan atau `ServerError` itu wajar.** Script otomatis mencoba lagi setelah 60 detik.
- **Untuk simulasi laptop lemah dengan `taskset` (hanya Linux), cek dulu topologi core.** Di laptopku CPU 0 dan 1 adalah hyperthread dari satu core fisik, jadi "2 core" harus pakai `taskset -c 0,2`, bukan `0,1`. Cek dengan `cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list`.
- **Jangan install `paddlepaddle`.** Benchmark lokal memakai onnxruntime, dan metode paddle memakai API.
- **Run pertama tiap model ikut men-download model**, sehingga `load_seconds`-nya ikut menghitung waktu download. Jalankan `--limit 1` dulu per model supaya angkanya bersih.

---

## Prompt untuk AI agent

Prompt ini bisa dipakai di AI agent apa pun yang **bisa menjalankan perintah terminal dan membaca/menulis file**, misalnya Claude Code, Gemini CLI, Codex CLI, Cursor (mode Agent), Windsurf, atau Cline.

AI versi web (claude.ai, gemini.google.com, ChatGPT) tidak bisa menjalankan perintah sendiri. Kalau terpaksa pakai yang web, tempel prompt yang sama, tapi kamu sendiri yang menjalankan perintah yang ia berikan dan mengirim balik hasilnya.

Persiapan:
1. Ekstrak zip-nya.
2. Isi `.env` sendiri: salin dari `.env.example`, lalu tambahkan `PADDLEOCR_ACCESS_TOKEN=` kalau mau mencoba metode paddle.
3. Kalau mau hasilnya tercatat di Obsidian, ganti `<PATH_VAULT>` di prompt dengan path folder vault-mu, misalnya `/home/nama/Documents/Vault` atau `C:\Users\nama\Documents\Vault`. Kalau tidak pakai Obsidian, hapus langkah 10.
4. Buka folder hasil ekstrak di AI agent-mu, lalu tempel prompt di bawah ini.

```text
Kamu AI agent dengan akses terminal di folder ini. Jalankan perintahnya sendiri, jangan
cuma menyuruhku. Sesuaikan perintah dengan OS-ku (Linux, macOS, atau Windows).

Folder ini berisi benchmark pembaca struk (Gemini vs PaddleOCR) di test split CORD v2.
Baca PANDUAN-TEMAN.md dulu untuk konteks dan hasil pembanding.

Larangan:
- Jangan tampilkan, cetak, atau tulis ulang isi .env. Cek key hanya dengan perintah yang
  tidak menampilkan nilainya, misalnya: grep -c '^GEMINI_API_KEY=.' .env
- Jangan ubah benchmark.py, benchmark_local.py, receipt_reader.py, app.py, atau storage.py.
  Kalau ada error, laporkan pesan error-nya ke aku dulu, jangan ditambal sendiri.
- Jangan install paddlepaddle.
- Jangan hapus benchmark_results/. Pindahkan hasil pembanding dulu ke benchmark_results_teman/
  supaya hasilku tidak tercampur dengan hasil pembanding.
- Jangan git commit atau git push.

Langkah:
1. Pindahkan isi benchmark_results/ ke benchmark_results_teman/.
2. Catat spek laptop: model CPU, jumlah core dan thread, RAM total, dukungan avx/avx2/avx512,
   dan OS. Di Linux: lscpu, free -h, grep -o -w 'avx\|avx2\|avx512f' /proc/cpuinfo | sort -u.
3. Pastikan GEMINI_API_KEY terisi di .env. Kalau kosong, berhenti dan bilang ke aku.
4. Buat virtual environment .venv, aktifkan, lalu: pip install -r requirements-local.txt
   (file ini sudah termasuk requirements-benchmark.txt dan requirements.txt).
   Kalau onnxruntime gagal di-install untuk versi Python-ku, berhenti dan laporkan.
5. Smoke test, cek kolom error di CSV-nya harus kosong:
   - python benchmark_local.py tiny --limit 3
   - python benchmark.py gemini --limit 3
   - python benchmark.py paddle --limit 3   (hanya kalau PADDLEOCR_ACCESS_TOKEN terisi)
6. Download model dulu supaya load_seconds bersih:
   - python benchmark_local.py small --limit 1
   - python benchmark_local.py medium --limit 1
7. Run 20 struk. Jalankan di background karena bisa lebih dari 10 menit:
   - python benchmark.py gemini --limit 20
   - python benchmark.py paddle --limit 20   (kalau ada token)
   - python benchmark_local.py tiny --limit 20
   - python benchmark_local.py small --limit 20
   - python benchmark_local.py medium --limit 20
   Kalau muncul "Kuota habis", berhenti dan laporkan sampai struk ke berapa. Hasil yang
   sudah tersimpan akan dilanjutkan saat dijalankan lagi nanti.
8. Opsional, hanya Linux: simulasi laptop lemah dengan 10 struk. Cek dulu
   /sys/devices/system/cpu/cpu0/topology/thread_siblings_list. Untuk "2c", pilih dua CPU yang
   ada di core fisik BERBEDA (bukan hyperthread dari core yang sama).
   - taskset -c 0 python benchmark_local.py tiny --threads 1 --tag 1c --limit 10
   - taskset -c <2 CPU beda core> python benchmark_local.py tiny --threads 2 --tag 2c --limit 10
   - (ulangi untuk medium)
9. python benchmark_local.py summary
10. Catat hasilnya di vault Obsidian-ku: <PATH_VAULT>
   - Baca dulu struktur vault (folder dan beberapa catatan) supaya gaya penulisannya ikut
     kebiasaanku. Kalau ada folder template atau proyek, pakai itu.
   - Buat SATU catatan baru, misalnya "Benchmark Struk.md", di folder yang paling cocok.
     Kalau ragu soal lokasinya, tanya aku dulu. Jangan ubah atau hapus catatan lain.
   - Isinya: frontmatter (date, tags: [benchmark, ocr, gemini]), spek laptop, tabel hasil
     (format tabel Markdown, bukan CSV mentah), perbandingan dengan benchmark_results_teman/,
     kesimpulan singkat, dan error atau kendala yang muncul (kuota, error jaringan).
   - Jangan tulis API key, token, atau isi .env ke vault.

Laporan akhir (singkat):
- Spek laptop dari langkah 2
- Isi benchmark_results/summary.csv, local_resources.csv, dan local_startup.json apa adanya
- Struk yang gagal beserta error-nya
- Perbandingan singkat dengan benchmark_results_teman/ (lebih cepat atau lebih lambat, RAM
  lebih besar atau lebih kecil, akurasi sama atau beda)
```
