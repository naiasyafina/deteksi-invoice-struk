import json
import re
from pathlib import Path

import pandas as pd

from scripts.profiling import RESULTS, machine

SUMMARY = RESULTS / "SUMMARY.md"
START, END = "<!-- tabel:mulai -->", "<!-- tabel:selesai -->"


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_belum ada hasil_\n"
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join("-" if value is None or value != value else str(value) for value in row.values()) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def mean(series: pd.Series, digits: int = 3):
    series = series.dropna() if series is not None else pd.Series(dtype=float)
    return round(float(series.mean()), digits) if len(series) else None


def resources(rows: pd.DataFrame, meta: dict) -> dict:
    return {
        "median s/gambar": round(rows.seconds.median(), 3),
        "p90 s": round(rows.seconds.quantile(0.9), 3),
        "CPU% rata-rata": int(rows.cpu_percent.mean()),
        "puncak RAM MB": round(max(rows.peak_rss_mb.max(), meta.get("peak_process_rss_mb", 0))),
        "load s": meta.get("load_seconds"),
        "threads": meta.get("threads"),
    }


def classify_section() -> str:
    summary, matrices = [], []
    for path in sorted(RESULTS.glob("classify-*.csv")):
        rows, meta = pd.read_csv(path), json.loads(path.with_suffix(".json").read_text())
        rows["ok"] = (rows.expected == rows.predicted).astype(int)
        entry = {"model": meta["model"], "n": len(rows), "akurasi": mean(rows.ok)}
        entry.update({f"akurasi {name}": mean(rows[rows.set == name].ok) for name in ("cord", "invoice", "own")})
        entry.update({f"recall {label}": mean(rows[rows.expected == label].ok) for label in ("struk", "invoice")})
        summary.append({**entry, **resources(rows, meta)})
        matrix = pd.crosstab(rows.expected, rows.predicted)
        matrix_rows = [{"asli \\ prediksi": index, **{column: int(matrix.loc[index, column]) for column in matrix.columns}} for index in matrix.index]
        matrices.append(f"**{path.stem}** ({meta['model']}, {meta['threads']} thread)\n\n{md_table(matrix_rows)}")
    return "## Klasifikasi (invoice vs struk)\n\n" + md_table(summary) + "\nConfusion matrix:\n\n" + "\n".join(matrices)


def extract_section() -> str:
    summary = []
    for path in sorted(RESULTS.glob("extract-*.csv")):
        rows, meta = pd.read_csv(path), json.loads(path.with_suffix(".json").read_text())
        done = rows[rows.error.isna()] if "error" in rows else rows
        cord, own = done[done.set == "cord"], done[done.set == "own"]
        entry = {
            "engine": meta["model"],
            "n": len(rows),
            "error": len(rows) - len(done),
            "total EM CORD": mean(cord.get("total_ok")),
            "F1 item CORD": mean(cord.get("item_f1")),
            "total EM own": mean(own.get("total_ok")) if len(own) else None,
            "tanggal own": mean(own.get("date_ok")) if len(own) else None,
            "toko own": mean(own.get("store_ok")) if len(own) else None,
        }
        summary.append({**entry, **(resources(done, meta) if len(done) else {})})
    return "## Ekstraksi isi\n\n" + md_table(summary)


def machine_section() -> str:
    spec = machine()
    return (
        "## Mesin uji\n\n"
        f"- CPU: {spec['cpu']} ({spec['physical_cores']} core fisik, {spec['logical_cpus']} thread)\n"
        f"- RAM: {spec['ram_gb']} GB\n- OS: {spec['os']}\n- Python {spec['python']}, semua inferensi di CPU\n"
    )


def main() -> None:
    tables = f"{START}\n{machine_section()}\n{classify_section()}\n{extract_section()}{END}"
    text = SUMMARY.read_text() if SUMMARY.exists() else f"# Ringkasan Benchmark\n\n{START}\n{END}\n"
    SUMMARY.write_text(re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: tables, text, flags=re.S))
    print(tables)


if __name__ == "__main__":
    assert md_table([{"a": 1, "b": None}]) == "| a | b |\n|---|---|\n| 1 | - |\n"
    main()
