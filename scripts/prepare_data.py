import argparse
import io
import json
from itertools import islice
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from scripts.metrics import cord_fields, cord_items

DATA = Path("data")
LABEL_HEADER = "file,jenis,nama_toko,tanggal,total\n"


def save_image(raw: bytes, path: Path) -> str:
    ext = {"JPEG": ".jpg", "PNG": ".png"}.get(Image.open(io.BytesIO(raw)).format, ".png")
    path = path.with_suffix(ext)
    path.write_bytes(raw)
    return path.name


def parquet_rows(repo: str, prefix: str) -> pd.DataFrame:
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet"))
    return pd.concat(pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset")) for f in files)


def prepare_cord(limit: int) -> None:
    out = DATA / "cord"
    out.mkdir(parents=True, exist_ok=True)
    rows = parquet_rows("naver-clova-ix/cord-v2", "data/test-").head(limit)
    with open(out / "ground_truth.jsonl", "w") as gt:
        for i, row in enumerate(rows.itertuples()):
            raw = row.ground_truth
            fields = cord_fields(raw)
            record = {"file": save_image(row.image["bytes"], out / f"{i:04d}"), "items": cord_items(raw), **fields}
            gt.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"cord: {len(rows)} gambar")


def prepare_invoice(per_source: int) -> None:
    from datasets import load_dataset

    out = DATA / "invoice"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    sparrow = parquet_rows("katanaml-org/invoices-donut-data-v1", "data/test-").head(per_source)
    for i, row in enumerate(sparrow.itertuples()):
        header = json.loads(row.ground_truth)["gt_parse"].get("header", {})
        file = save_image(row.image["bytes"], out / f"sparrow-{i:03d}")
        records.append({"file": file, "source": "katanaml-org/invoices-donut-data-v1", "vendor": header.get("seller"), "date": header.get("invoice_date")})
    stream = load_dataset("chainyo/rvl-cdip-invoice", split="train", streaming=True)
    for i, sample in enumerate(islice(stream, per_source)):
        buffer = io.BytesIO()
        sample["image"].convert("L").save(buffer, "PNG")
        records.append({"file": save_image(buffer.getvalue(), out / f"rvl-{i:03d}"), "source": "chainyo/rvl-cdip-invoice"})
    (out / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(f"invoice: {len(records)} gambar")


def prepare_own() -> None:
    for kind in ("struk", "invoice"):
        (DATA / "own" / kind).mkdir(parents=True, exist_ok=True)
    labels = DATA / "own" / "labels.csv"
    if not labels.exists():
        labels.write_text(LABEL_HEADER)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cord", type=int, default=100)
    parser.add_argument("--invoice-per-source", type=int, default=25)
    args = parser.parse_args()
    prepare_own()
    prepare_cord(args.cord)
    prepare_invoice(args.invoice_per_source)


if __name__ == "__main__":
    main()
