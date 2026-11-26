import os
from datetime import date

import pandas as pd
import streamlit as st
from google.genai import errors

import storage
from receipt_reader import Item, Receipt, expected_total, read_receipt

LABELS = {
    "id": "ID",
    "store": "Toko",
    "purchase_date": "Tanggal",
    "discount": "Diskon",
    "tax": "Pajak/Servis",
    "total": "Total",
    "name": "Barang",
    "qty": "Qty",
    "price": "Harga",
    "month": "Bulan",
    "visits": "Jumlah struk",
    "spent": "Total belanja",
}


def rupiah(value: int) -> str:
    return f"Rp{value:,}".replace(",", ".")


def parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        return date.today()


def scan(uploaded) -> None:
    try:
        with st.spinner("Membaca struk..."):
            receipt = read_receipt(uploaded.getvalue(), uploaded.type)
    except errors.APIError as error:
        if error.code == 429:
            st.error("Kuota Gemini sedang habis. Tunggu sebentar lalu coba lagi.")
        else:
            st.error(f"Gagal menghubungi Gemini: {error.message}")
        return
    if receipt is None:
        st.error("Struk tidak terbaca. Coba foto yang lebih jelas dan tidak miring.")
        return
    st.session_state.receipt = receipt
    st.session_state.source = uploaded.file_id


def review(receipt: Receipt) -> None:
    store = st.text_input("Toko", receipt.store)
    purchase_date = st.date_input("Tanggal", parse_date(receipt.purchase_date), format="DD/MM/YYYY")
    rows = st.data_editor(
        pd.DataFrame([item.model_dump() for item in receipt.items], columns=["name", "qty", "price"]),
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn(LABELS["name"], required=True),
            "qty": st.column_config.NumberColumn(LABELS["qty"], min_value=1, step=1, default=1),
            "price": st.column_config.NumberColumn(LABELS["price"], min_value=0, step=1, required=True),
        },
    ).dropna()

    discount_col, tax_col, total_col = st.columns(3)
    discount = discount_col.number_input(LABELS["discount"], min_value=0, value=receipt.discount, step=1)
    tax = tax_col.number_input(LABELS["tax"], min_value=0, value=receipt.tax, step=1)
    total = total_col.number_input("Total di struk", min_value=0, value=receipt.total, step=1)

    edited = Receipt(
        store=store.strip(),
        purchase_date=purchase_date.isoformat(),
        items=[Item(name=str(row.name), qty=int(row.qty), price=int(row.price)) for row in rows.itertuples()],
        discount=int(discount),
        tax=int(tax),
        total=int(total),
    )

    calculated = expected_total(edited)
    if calculated == edited.total:
        st.success(f"Total cocok: {rupiah(edited.total)}")
    else:
        st.warning(
            f"Hitungan item {rupiah(calculated)}, tapi total di struk {rupiah(edited.total)}. "
            f"Selisih {rupiah(abs(edited.total - calculated))}, cek item yang terlewat atau salah baca."
        )

    confirmed = True
    if storage.is_duplicate(edited):
        st.warning("Struk dengan toko, tanggal, dan total yang sama sudah pernah disimpan.")
        confirmed = st.checkbox("Tetap simpan")

    if st.button("Simpan", type="primary", disabled=not (edited.store and edited.items and confirmed)):
        storage.save_receipt(edited)
        del st.session_state.receipt
        st.session_state.saved = True
        st.rerun()


def scan_tab() -> None:
    if st.session_state.pop("saved", False):
        st.success("Struk tersimpan. Lihat rekapnya di tab Riwayat.")

    uploaded = st.file_uploader("Foto struk", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        return

    image_col, form_col = st.columns(2)
    image_col.image(uploaded)
    with form_col:
        if st.button("Baca struk"):
            scan(uploaded)
        if "receipt" in st.session_state and st.session_state.source == uploaded.file_id:
            review(st.session_state.receipt)


def history_tab() -> None:
    receipts = storage.load_receipts()
    if receipts.empty:
        st.info("Belum ada struk yang disimpan.")
        return

    total_col, count_col, average_col = st.columns(3)
    total_col.metric("Total pengeluaran", rupiah(int(receipts.total.sum())))
    count_col.metric("Jumlah struk", len(receipts))
    average_col.metric("Rata-rata per struk", rupiah(int(receipts.total.mean())))

    st.subheader("Per bulan")
    st.bar_chart(storage.monthly_totals(), x="month", y="spent", x_label=LABELS["month"], y_label=LABELS["spent"])

    st.subheader("Per toko")
    st.dataframe(storage.store_totals(), hide_index=True, column_config=LABELS)

    st.subheader("Semua struk")
    st.dataframe(receipts, hide_index=True, column_config=LABELS)
    st.download_button("Unduh CSV", receipts.to_csv(index=False), "struk.csv", "text/csv")

    selected = st.selectbox(
        "Detail struk",
        receipts.id,
        format_func=lambda receipt_id: " · ".join(
            str(value) for value in receipts.loc[receipts.id == receipt_id, ["store", "purchase_date"]].iloc[0]
        ),
    )
    st.dataframe(storage.load_items(selected), hide_index=True, column_config=LABELS)
    if st.button("Hapus struk ini"):
        storage.delete_receipt(selected)
        st.rerun()


st.set_page_config(page_title="Scan Struk", layout="wide")
st.title("Scan Struk")

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY belum diisi. Salin .env.example jadi .env lalu isi API key dari Google AI Studio.")
    st.stop()

storage.init_db()
scan_page, history_page = st.tabs(["Scan", "Riwayat"])
with scan_page:
    scan_tab()
with history_page:
    history_tab()
