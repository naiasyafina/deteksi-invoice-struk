# Sumber Data Uji

Semua data di sini hanya untuk pengujian. Tidak ada yang dipakai untuk training. Isi `data/` (kecuali file ini) tidak masuk git. Untuk membuat ulang:

```bash
.venv/bin/python -m scripts.prepare_data
```

| Folder | Sumber | Jumlah | Lisensi | Catatan |
|---|---|---|---|---|
| `cord/` | [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2), split `test` | 100 struk | CC BY 4.0 | Struk restoran/toko Indonesia. `ground_truth.jsonl`: `items` (nama + harga), `subtotal`, `tax`, `total` dalam rupiah. Tidak ada label nama toko maupun tanggal. |
| `invoice/sparrow-*` | [katanaml-org/invoices-donut-data-v1](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1), split `test` | 25 invoice | MIT (dataset HF); data asal [Mendeley "Samples of electronic invoices"](https://data.mendeley.com/datasets/tnj49gpmtz/2) CC BY 4.0 | Invoice digital hasil generate program (bukan dokumen asli), bahasa Inggris, mata uang USD. |
| `invoice/rvl-*` | [chainyo/rvl-cdip-invoice](https://huggingface.co/datasets/chainyo/rvl-cdip-invoice), 25 baris pertama split `train`, diambil lewat streaming | 25 invoice | "other": RVL-CDIP adalah subset IIT-CDIP dari [UCSF Industry Documents Library](https://www.industrydocuments.ucsf.edu/help/copyright/) | Scan hitam-putih tahun 1980-1990an, resolusi rendah (sisi terpanjang ≤ 1000 px). Hanya untuk riset non-komersial. |
| `own/struk/`, `own/invoice/` | Foto dan scan dari tim | diisi tim | data pribadi | Tidak boleh dibagikan di luar tim. Label di `own/labels.csv`. |

## Mengisi data sendiri

Taruh gambar di `own/struk/` atau `own/invoice/`, lalu tambahkan satu baris per gambar di `own/labels.csv`:

```
file,jenis,nama_toko,tanggal,total
struk/indomaret-01.jpg,struk,Indomaret,2026-11-20,45500
invoice/inv-001.jpg,invoice,PT Contoh Jaya,2026-10-02,1250000
```

- `file`: path relatif terhadap `data/own/`
- `jenis`: `struk` atau `invoice`
- `tanggal`: format `YYYY-MM-DD`
- `total`: angka bulat tanpa titik/koma atau `Rp`

## Catatan untuk membaca hasil

- Donut `donut-base-finetuned-cord-v2` dilatih dengan CORD train. CORD test tidak ikut dilatih, jadi pengujiannya tetap adil. Tetapi hasil di struk lain yang formatnya berbeda dari CORD kemungkinan lebih rendah.
- Model DiT `victormurcia/DiT-DocumentClassifier-v1` dilatih dengan scan dokumen keuangan bahasa Inggris (data latih tidak dipublikasikan). Invoice RVL-CDIP kemungkinan mirip dengan data latihnya, jadi akurasinya di sana bisa terlalu optimis. Data `own/` adalah acuan utama.
