import csv
import json
from pathlib import Path

DATA = Path("data")
SETS = ("cord", "invoice", "own")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def to_int(value: str) -> int | None:
    digits = "".join(char for char in value or "" if char.isdigit())
    return int(digits) if digits else None


def load_samples(sets: list[str], limit: int) -> list[dict]:
    samples = []
    if "cord" in sets:
        for row in read_jsonl(DATA / "cord/ground_truth.jsonl")[:limit]:
            samples.append({"set": "cord", "jenis": "struk", "path": DATA / "cord" / row["file"], **row})
    if "invoice" in sets:
        for row in read_jsonl(DATA / "invoice/labels.jsonl")[:limit]:
            samples.append({"set": "invoice", "jenis": "invoice", "path": DATA / "invoice" / row["file"], "file": row["file"]})
    labels = DATA / "own/labels.csv"
    if "own" in sets and labels.exists():
        with open(labels, newline="") as handle:
            for row in list(csv.DictReader(handle))[:limit]:
                samples.append({
                    "set": "own",
                    "jenis": row["jenis"].strip().lower(),
                    "path": DATA / "own" / row["file"],
                    "file": row["file"],
                    "store_name": row["nama_toko"].strip() or None,
                    "date": row["tanggal"].strip() or None,
                    "total": to_int(row["total"]),
                })
    missing = [str(sample["path"]) for sample in samples if not sample["path"].exists()]
    if missing:
        raise SystemExit(f"gambar tidak ditemukan: {missing[:5]}")
    return samples


if __name__ == "__main__":
    assert to_int("Rp 45.500") == 45500 and to_int("") is None
    print("ok")
