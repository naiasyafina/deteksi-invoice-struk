import io

import streamlit as st
from PIL import Image

from pipeline.classify import classify, load_classifier
from pipeline.export import to_excel, to_tables
from pipeline.gemini import extract, make_client

LOW_CONFIDENCE = 0.6
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

st.set_page_config(page_title="Scan Struk & Invoice", layout="wide")


@st.cache_resource(show_spinner="Memuat model DiT...")
def classifier():
    return load_classifier()


@st.cache_resource
def gemini():
    return make_client()


@st.cache_data(show_spinner="Mendeteksi jenis dokumen...")
def predict(image: bytes) -> tuple[str, float]:
    return classify(classifier(), Image.open(io.BytesIO(image)).convert("RGB"))


st.title("Scan Struk & Invoice")
st.caption("Jenis dokumen dideteksi di laptop ini dengan model DiT. Isinya dibaca Gemini, lalu bisa dicek dan diunduh sebagai Excel.")

files = st.file_uploader("Upload foto struk atau invoice", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
if not files:
    st.stop()

results = st.session_state.setdefault("results", {})
detected = {}
for file in files:
    jenis, score = predict(file.getvalue())
    detected[file.file_id] = jenis
    left, right = st.columns([1, 2])
    left.image(file.getvalue(), width=280)
    with right:
        st.subheader(file.name)
        st.markdown(f"Terdeteksi sebagai **{jenis}** (keyakinan {score:.0%})")
        if score < LOW_CONFIDENCE:
            st.warning("Model kurang yakin dengan hasil deteksi ini.")
    st.divider()

client = gemini()
if client is None:
    st.error("GEMINI_API_KEY belum diisi di file .env, jadi digitalisasi belum bisa jalan.")
    st.stop()

pending = [file for file in files if file.file_id not in results]
if st.button(f"Digitalisasi {len(pending)} dokumen", type="primary", disabled=not pending):
    progress = st.progress(0.0, text="Membaca dokumen dengan Gemini...")
    for i, file in enumerate(pending, 1):
        jenis = detected[file.file_id]
        try:
            results[file.file_id] = {"file": file.name, "jenis": jenis, "data": extract(client, file.getvalue(), file.type, jenis)}
        except Exception as error:  # noqa: BLE001
            st.error(f"{file.name} gagal dibaca: {error}")
        progress.progress(i / len(pending), text=f"{i}/{len(pending)} dokumen")

documents = [results[file.file_id] for file in files if file.file_id in results]
if not documents:
    st.stop()

st.header("Hasil")
st.caption("Tabel bisa diedit langsung sebelum diunduh.")
version = hash(tuple((doc["file"], doc["jenis"]) for doc in documents))
tables = {}
for sheet, table in to_tables(documents).items():
    st.subheader(sheet)
    tables[sheet] = st.data_editor(table, num_rows="dynamic", width="stretch", key=f"tabel-{sheet}-{version}")

st.download_button("Download Excel", data=to_excel(tables), file_name="hasil-scan.xlsx", mime=XLSX, type="primary")
