import argparse
import csv
import json
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from google import genai

from benchmark import FIELDS, RESULTS_DIR, evaluate, extract_from_ocr, finished_ids, normalize, summarize

SIZES = ["tiny", "small", "medium"]
LOCAL_FIELDS = FIELDS + ["ocr_seconds", "cpu_seconds", "cpu_percent", "peak_ram_mb", "word_recall"]
STARTUP_PATH = RESULTS_DIR / "local_startup.json"


def peak_ram_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def used_cpu_seconds(process: psutil.Process) -> float:
    times = process.cpu_times()
    return times.user + times.system


def word_recall(raw: str, texts: list[str]) -> float | None:
    lines = json.loads(raw).get("valid_line", [])
    expected = Counter(
        word for line in lines for word in (normalize(item["text"]) for item in line.get("words", [])) if word
    )
    found = Counter(word for text in texts for word in map(normalize, text.split()) if word)
    total = sum(expected.values())
    return round(sum((expected & found).values()) / total, 3) if total else None


def load_ocr(size: str, engine: str, threads: int | None):
    from paddleocr import PaddleOCR

    options = {}
    if threads and engine == "onnxruntime":
        options["engine_config"] = {"intra_op_num_threads": threads}
    elif threads:
        options["cpu_threads"] = threads

    return PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{size}_det",
        text_recognition_model_name=f"PP-OCRv6_{size}_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine=engine,
        **options,
    )


def to_bgr(image) -> np.ndarray:
    return np.array(image.convert("RGB"))[:, :, ::-1]


def save_startup(name: str, startup: dict) -> None:
    records = json.loads(STARTUP_PATH.read_text()) if STARTUP_PATH.exists() else {}
    records[name] = startup
    STARTUP_PATH.write_text(json.dumps(records, indent=2))


def run(size: str, engine: str, threads: int | None, tag: str, limit: int, delay: float) -> None:
    from datasets import load_dataset

    dataset = load_dataset("naver-clova-ix/cord-v2", split="test")
    RESULTS_DIR.mkdir(exist_ok=True)
    name = f"local-{size}" + (f"-{tag}" if tag else "")
    path = RESULTS_DIR / f"{name}.csv"
    done = finished_ids(path)
    is_new_file = not path.exists()
    process = psutil.Process()

    load_start = time.perf_counter()
    ocr = load_ocr(size, engine, threads)
    load_seconds = time.perf_counter() - load_start
    warmup_start = time.perf_counter()
    ocr.predict(to_bgr(dataset[0]["image"]))
    save_startup(
        name,
        {
            "engine": engine,
            "threads": threads,
            "load_seconds": round(load_seconds, 2),
            "warmup_seconds": round(time.perf_counter() - warmup_start, 2),
            "ram_after_warmup_mb": peak_ram_mb(),
        },
    )

    client = genai.Client()
    stats = {}

    def read(image):
        wall_start, cpu_start = time.perf_counter(), used_cpu_seconds(process)
        result = ocr.predict(to_bgr(image))[0]
        wall = time.perf_counter() - wall_start
        cpu = used_cpu_seconds(process) - cpu_start
        stats.update(
            ocr_seconds=round(wall, 2),
            cpu_seconds=round(cpu, 2),
            cpu_percent=round(cpu / wall * 100),
            peak_ram_mb=peak_ram_mb(),
            texts=result["rec_texts"],
        )
        return extract_from_ocr(client, result["rec_texts"], result["rec_boxes"])

    count = min(limit, len(dataset))
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, LOCAL_FIELDS)
        if is_new_file:
            writer.writeheader()
        for image_id in range(count):
            if image_id in done:
                continue
            sample = dataset[image_id]
            stats.clear()
            row = evaluate(read, sample, image_id)
            if stats:
                texts = stats.pop("texts")
                row.update(stats, word_recall=word_recall(sample["ground_truth"], texts))
            writer.writerow(row)
            file.flush()
            print(
                f"[{name}] {image_id + 1}/{count} ocr={row.get('ocr_seconds', '-')}s "
                f"ram={row.get('peak_ram_mb', '-')}MB total_match={row.get('total_match', '-')} error={row['error'] or '-'}"
            )
            time.sleep(delay)


def summarize_resources() -> None:
    startup = json.loads(STARTUP_PATH.read_text()) if STARTUP_PATH.exists() else {}
    rows = []
    for path in sorted(RESULTS_DIR.glob("local-*.csv")):
        results = pd.read_csv(path).drop_duplicates("image_id", keep="last")
        parsed = results[results.error.isna()]
        info = startup.get(path.stem, {})
        rows.append(
            {
                "method": path.stem,
                "engine": info.get("engine"),
                "threads": info.get("threads"),
                "load_seconds": info.get("load_seconds"),
                "warmup_seconds": info.get("warmup_seconds"),
                "median_ocr_seconds": parsed.ocr_seconds.median(),
                "max_ocr_seconds": parsed.ocr_seconds.max(),
                "median_cpu_percent": parsed.cpu_percent.median(),
                "peak_ram_mb": parsed.peak_ram_mb.max(),
                "word_recall": parsed.word_recall.mean(),
            }
        )
    if not rows:
        return
    resources = pd.DataFrame(rows).round(3)
    resources.to_csv(RESULTS_DIR / "local_resources.csv", index=False)
    print()
    print(resources.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", choices=[*SIZES, "summary"])
    parser.add_argument("--engine", choices=["onnxruntime", "paddle"], default="onnxruntime")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--tag", default="")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--delay", type=float, default=5)
    args = parser.parse_args()

    if args.size != "summary":
        run(args.size, args.engine, args.threads, args.tag, args.limit, args.delay)
    summarize()
    summarize_resources()


if __name__ == "__main__":
    main()
