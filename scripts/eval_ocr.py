import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil

from pipeline.extract import extract_regex
from pipeline.ocr import group_lines, rapid_ocr
from pipeline.preprocess import preprocess
from scripts.metrics import cord_fields, cord_lines, cord_words, line_errors, word_recall
from scripts.profiling import PeakRss, rss_mb, used_cpu_seconds

RUNS_DIR = Path("reports/fase1")
SUMMARY_PATH = Path("reports/fase1_ocr_baseline.csv")
ENGINES = ["rapid", "rapid-cls", "paddle-tiny", "paddle-small", "gt-text"]
FIELDS = ["subtotal", "tax", "total"]


def paddle_ocr(size: str, threads: int):
    from paddleocr import PaddleOCR

    engine = PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{size}_det",
        text_recognition_model_name=f"PP-OCRv6_{size}_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="onnxruntime",
        **({"engine_config": {"intra_op_num_threads": threads}} if threads else {}),
    )

    def read(img: np.ndarray):
        result = engine.predict(img)[0]
        return list(result["rec_texts"]), [tuple(map(float, box)) for box in result["rec_boxes"]]

    return read


def load_engine(name: str, threads: int):
    if name.startswith("rapid"):
        return rapid_ocr(threads, use_cls=name == "rapid-cls")
    return paddle_ocr(name.split("-")[1], threads)


def to_bgr(image) -> np.ndarray:
    return np.ascontiguousarray(np.array(image.convert("RGB"))[:, :, ::-1])


def roi_iou(raw_ground_truth: str, quad: np.ndarray | None, size: tuple[int, int]) -> float | None:
    roi = json.loads(raw_ground_truth).get("roi")
    if not roi:
        return None
    width, height = size
    found = quad if quad is not None else np.array([[0, 0], [width, 0], [width, height], [0, height]])
    expected = np.array([[roi[f"x{i}"], roi[f"y{i}"]] for i in range(1, 5)])
    scale = 0.25
    masks = []
    for points in (expected, found):
        mask = np.zeros((int(height * scale) + 1, int(width * scale) + 1), np.uint8)
        cv2.fillPoly(mask, [np.round(np.asarray(points) * scale).astype(np.int32)], 1)
        masks.append(mask.astype(bool))
    union = (masks[0] | masks[1]).sum()
    return round(float((masks[0] & masks[1]).sum() / union), 3) if union else None


def score_fields(raw_ground_truth: str, lines: list[str]) -> dict:
    expected = cord_fields(raw_ground_truth)
    predicted = extract_regex(lines)
    row = {f"{field}_ok": (None if expected[field] is None else int(predicted[field] == expected[field])) for field in FIELDS}
    row.update(date_found=int(predicted["date"] is not None), store_found=int(predicted["store_name"] is not None))
    return row


def run(engine_name: str, threads: int, steps: tuple[str, ...], limit: int, name: str) -> None:
    from datasets import load_dataset

    if threads:
        cv2.setNumThreads(threads)
    dataset = load_dataset("naver-clova-ix/cord-v2", split="test")
    count = min(limit, len(dataset))
    process = psutil.Process()
    startup = {"engine": engine_name, "threads": threads, "steps": ",".join(steps) or "none", "rss_before_load_mb": round(rss_mb(process), 1)}
    read = None
    if engine_name != "gt-text":
        start = time.perf_counter()
        read = load_engine(engine_name, threads)
        startup["load_seconds"] = round(time.perf_counter() - start, 2)
        startup["rss_after_load_mb"] = round(rss_mb(process), 1)
        start = time.perf_counter()
        read(preprocess(to_bgr(dataset[0]["image"]), steps)[0])
        startup["warmup_seconds"] = round(time.perf_counter() - start, 2)

    rows = []
    for image_id in range(count):
        sample = dataset[image_id]
        raw = sample["ground_truth"]
        row = {"image_id": image_id, "width": sample["image"].width, "height": sample["image"].height}
        if read is None:
            lines = cord_lines(raw)
        else:
            img = to_bgr(sample["image"])
            with PeakRss(process) as peak:
                wall_start, cpu_start = time.perf_counter(), used_cpu_seconds(process)
                prepared, quad = preprocess(img, steps)
                prep_end = time.perf_counter()
                texts, boxes = read(prepared)
                wall_end, cpu_end = time.perf_counter(), used_cpu_seconds(process)
            lines = group_lines(texts, boxes)
            errors, chars = line_errors(cord_lines(raw), lines)
            row.update(
                prep_seconds=round(prep_end - wall_start, 3),
                ocr_seconds=round(wall_end - prep_end, 3),
                total_seconds=round(wall_end - wall_start, 3),
                cpu_percent=round((cpu_end - cpu_start) / (wall_end - wall_start) * 100),
                peak_rss_mb=round(peak.peak_mb, 1),
                boxes=len(texts),
                cer_errors=errors,
                cer_chars=chars,
                word_recall=word_recall(cord_words(raw), texts),
                cropped=int(quad is not None),
                crop_iou=roi_iou(raw, quad, sample["image"].size) if "crop" in steps else None,
                lines=json.dumps(lines, ensure_ascii=False),
            )
        row.update(score_fields(raw, lines))
        rows.append(row)
        print(f"[{name}] {image_id + 1}/{count} t={row.get('total_seconds', '-')}s total_ok={row['total_ok']}", flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RUNS_DIR / f"{name}.csv", index=False)
    (RUNS_DIR / f"{name}.json").write_text(json.dumps(startup, indent=2))


def summarize() -> None:
    summary = []
    for path in sorted(RUNS_DIR.glob("*.csv")):
        rows = pd.read_csv(path)
        startup = json.loads(path.with_suffix(".json").read_text())
        entry = {"run": path.stem, **{k: startup.get(k) for k in ("engine", "threads", "steps")}, "images": len(rows)}
        if "cer_errors" in rows:
            entry.update(
                line_cer=round(rows.cer_errors.sum() / rows.cer_chars.sum(), 4),
                word_recall=round(rows.word_recall.mean(), 3),
            )
        entry.update({f"{field}_acc": round(rows[f"{field}_ok"].mean(), 3) for field in FIELDS})
        entry.update({f"{field}_n": int(rows[f"{field}_ok"].notna().sum()) for field in FIELDS})
        entry.update(date_found=round(rows.date_found.mean(), 3), store_found=round(rows.store_found.mean(), 3))
        if "total_seconds" in rows:
            entry.update(
                crop_iou=round(rows.crop_iou.mean(), 3) if rows.crop_iou.notna().any() else None,
                cropped_rate=round(rows.cropped.mean(), 3),
                median_seconds=round(rows.total_seconds.median(), 3),
                p90_seconds=round(rows.total_seconds.quantile(0.9), 3),
                max_seconds=round(rows.total_seconds.max(), 3),
                median_prep_seconds=round(rows.prep_seconds.median(), 3),
                median_cpu_percent=int(rows.cpu_percent.median()),
                peak_rss_mb=rows.peak_rss_mb.max(),
                **{k: startup.get(k) for k in ("load_seconds", "warmup_seconds", "rss_after_load_mb")},
            )
        summary.append(entry)
    table = pd.DataFrame(summary)
    table.to_csv(SUMMARY_PATH, index=False)
    print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "summary"])
    parser.add_argument("--engine", choices=ENGINES, default="rapid")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--steps", default="", help="gabungan dari crop,deskew,gray; kosong = tanpa preprocessing")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--name")
    args = parser.parse_args()

    if args.mode == "run":
        steps = tuple(step for step in args.steps.split(",") if step)
        name = args.name or f"{args.engine}-{'+'.join(steps) or 'raw'}-t{args.threads}"
        run(args.engine, args.threads, steps, args.limit, name)
    summarize()


if __name__ == "__main__":
    main()
