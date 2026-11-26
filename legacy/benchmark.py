import argparse
import csv
import io
import json
import re
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

from receipt_reader import MODEL, PROMPT, Receipt, expected_total, read_receipt

RESULTS_DIR = Path("benchmark_results")
PADDLE_MODEL = "PP-OCRv5"
RETRY_STATUSES = {429, 503}
RETRY_WAIT_SECONDS = 60
NETWORK_ERRORS = {
    "NetworkError",
    "RequestTimeoutError",
    "PollTimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "RemoteProtocolError",
}
FIELDS = [
    "image_id",
    "seconds",
    "total_match",
    "items_gt",
    "items_pred",
    "price_f1",
    "name_f1",
    "self_consistent",
    "error",
]


def as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_money(text: str) -> int:
    digits = re.sub(r"\D", "", text)
    if re.search(r"[.,]\d{2}$", text.strip()):
        digits = digits[:-2]
    return int(digits) if digits else 0


def ground_truth(raw: str) -> tuple[list[tuple[str, int]], int | None]:
    parse = json.loads(raw)["gt_parse"]
    items = []
    for entry in as_list(parse.get("menu")):
        names = as_list(entry.get("nm"))
        prices = as_list(entry.get("price"))
        if names and prices:
            items.append((" ".join(map(str, names)), parse_money(str(prices[0]))))
    totals = [price for total in as_list(parse.get("total")) for price in as_list(total.get("total_price"))]
    return items, parse_money(str(totals[0])) if totals else None


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def count_price_matches(predicted: list[int], expected: list[int]) -> int:
    remaining = list(expected)
    matched = 0
    for price in predicted:
        if price in remaining:
            remaining.remove(price)
            matched += 1
    return matched


def count_name_matches(predicted: list[str], expected: list[str]) -> int:
    remaining = [normalize(name) for name in expected]
    matched = 0
    for name in map(normalize, predicted):
        best = max(remaining, key=lambda candidate: similarity(name, candidate), default=None)
        if best is not None and similarity(name, best) >= 0.8:
            remaining.remove(best)
            matched += 1
    return matched


def f1(matched: int, predicted: int, expected: int) -> float:
    if predicted == expected == 0:
        return 1.0
    return round(2 * matched / (predicted + expected), 3)


def rows_to_text(texts: list[str], boxes) -> str:
    words = sorted(zip(texts, boxes), key=lambda word: (word[1][1] + word[1][3]) / 2)
    rows = []
    for text, (left, top, _, bottom) in words:
        center = (top + bottom) / 2
        if rows and abs(center - rows[-1][0]) < (bottom - top) / 2:
            rows[-1][1].append((left, text))
        else:
            rows.append([center, [(left, text)]])
    return "\n".join(" ".join(text for _, text in sorted(row)) for _, row in rows)


def gemini_vision_reader():
    def read(image) -> Receipt | None:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        return read_receipt(buffer.getvalue(), "image/jpeg")

    return read


def extract_from_ocr(client: genai.Client, texts: list[str], boxes) -> Receipt | None:
    response = client.models.generate_content(
        model=MODEL,
        contents=f"{PROMPT}\n\nThe receipt is given as OCR text, one printed row per line:\n{rows_to_text(texts, boxes)}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Receipt,
        ),
    )
    return response.parsed


def paddle_gemini_reader():
    from paddleocr import PaddleOCRClient

    ocr = PaddleOCRClient()
    client = genai.Client()

    def read(image) -> Receipt | None:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as file:
            image.convert("RGB").save(file, format="JPEG")
            file.flush()
            result = ocr.ocr(file_path=file.name, model=PADDLE_MODEL).pages[0].pruned_result
        return extract_from_ocr(client, result["rec_texts"], result["rec_boxes"])

    return read


READERS = {
    "gemini": gemini_vision_reader,
    "paddle": paddle_gemini_reader,
}


def status_of(error: Exception) -> int | None:
    return getattr(error, "code", None) or getattr(error, "status_code", None)


def is_retryable(error: Exception) -> bool:
    return status_of(error) in RETRY_STATUSES or type(error).__name__ in NETWORK_ERRORS


def read_with_retry(read, image) -> Receipt | None:
    try:
        return read(image)
    except Exception as error:
        if not is_retryable(error):
            raise
        print(f"{type(error).__name__}, coba lagi dalam {RETRY_WAIT_SECONDS} detik...")
    time.sleep(RETRY_WAIT_SECONDS)
    try:
        return read(image)
    except Exception as error:
        if status_of(error) in RETRY_STATUSES:
            raise SystemExit("Kuota habis atau server masih sibuk. Jalankan lagi nanti, hasil yang sudah tersimpan akan dilanjutkan.")
        raise


def evaluate(read, sample, image_id: int) -> dict:
    expected_items, expected_total_price = ground_truth(sample["ground_truth"])
    start = time.perf_counter()
    try:
        receipt = read_with_retry(read, sample["image"])
        error = "" if receipt else "unparsed"
    except Exception as failure:
        receipt, error = None, f"{type(failure).__name__}: {failure}"[:300]
    row = {
        "image_id": image_id,
        "seconds": round(time.perf_counter() - start, 2),
        "items_gt": len(expected_items),
        "error": error,
    }
    if receipt is None:
        return row

    predicted_names = [item.name for item in receipt.items]
    predicted_prices = [item.price for item in receipt.items]
    expected_names = [name for name, _ in expected_items]
    expected_prices = [price for _, price in expected_items]
    row.update(
        items_pred=len(receipt.items),
        price_f1=f1(count_price_matches(predicted_prices, expected_prices), len(predicted_prices), len(expected_prices)),
        name_f1=f1(count_name_matches(predicted_names, expected_names), len(predicted_names), len(expected_names)),
        self_consistent=int(expected_total(receipt) == receipt.total),
    )
    if expected_total_price is not None:
        row["total_match"] = int(receipt.total == expected_total_price)
    return row


def finished_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as file:
        return {int(row["image_id"]) for row in csv.DictReader(file) if not row["error"]}


def run(method: str, limit: int, delay: float) -> None:
    from datasets import load_dataset

    dataset = load_dataset("naver-clova-ix/cord-v2", split="test")
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{method}.csv"
    done = finished_ids(path)
    is_new_file = not path.exists()
    read = READERS[method]()
    count = min(limit, len(dataset))

    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, FIELDS)
        if is_new_file:
            writer.writeheader()
        for image_id in range(count):
            if image_id in done:
                continue
            row = evaluate(read, dataset[image_id], image_id)
            writer.writerow(row)
            file.flush()
            print(f"[{method}] {image_id + 1}/{count} total_match={row.get('total_match', '-')} error={row['error'] or '-'}")
            time.sleep(delay)


def summarize() -> None:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        results = pd.read_csv(path)
        if "image_id" not in results:
            continue
        results = results.drop_duplicates("image_id", keep="last")
        parsed = results[results.error.isna()]
        rows.append(
            {
                "method": path.stem,
                "receipts": len(results),
                "failed": len(results) - len(parsed),
                "total_accuracy": parsed.total_match.mean(),
                "item_price_f1": parsed.price_f1.mean(),
                "item_name_f1": parsed.name_f1.mean(),
                "self_consistent": parsed.self_consistent.mean(),
                "median_seconds": parsed.seconds.median(),
                "max_seconds": parsed.seconds.max(),
            }
        )
    summary = pd.DataFrame(rows).round(3)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=[*READERS, "summary"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--delay", type=float, default=5)
    args = parser.parse_args()

    if args.method != "summary":
        run(args.method, args.limit, args.delay)
    summarize()


if __name__ == "__main__":
    main()
