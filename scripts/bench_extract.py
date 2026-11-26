import argparse
import json
import os
import re
import time

import psutil
from PIL import Image

from scripts.metrics import cord_fields, cord_items, item_f1, normalize, similarity
from scripts.profiling import measure, rss_mb, save_run
from scripts.samples import load_samples

DONUT = "naver-clova-ix/donut-base-finetuned-cord-v2"
DONUT_SAFETENSORS = "refs/pr/6"
GEMINI_PROMPT = """Read this receipt or invoice and extract its data.
- store_name: store, merchant, or vendor name as printed.
- date: transaction or invoice date as YYYY-MM-DD, empty string if not printed.
- items: every purchased line. price is the line total (qty x unit price).
- subtotal, tax: 0 if not printed.
- total: final amount to pay.
Write all money as plain integers without separators, decimals, or currency symbols."""


def from_cord_parse(parse: dict) -> dict:
    raw = json.dumps({"gt_parse": parse})
    try:
        return {"items": cord_items(raw), "store_name": None, "date": None, **cord_fields(raw)}
    except (AttributeError, TypeError, ValueError):
        return {"items": [], "store_name": None, "date": None, "subtotal": None, "tax": None, "total": None}


def load_donut():
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    processor = DonutProcessor.from_pretrained(DONUT)
    model = VisionEncoderDecoderModel.from_pretrained(DONUT, revision=DONUT_SAFETENSORS, use_safetensors=True).eval()
    tokenizer = processor.tokenizer
    prompt = tokenizer("<s_cord-v2>", add_special_tokens=False, return_tensors="pt").input_ids

    def read(path):
        pixel_values = processor(Image.open(path).convert("RGB"), return_tensors="pt").pixel_values
        with torch.inference_mode():
            output = model.generate(
                pixel_values,
                decoder_input_ids=prompt,
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=[[tokenizer.unk_token_id]],
            )
        sequence = processor.batch_decode(output)[0].replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
        parse = processor.token2json(re.sub(r"<.*?>", "", sequence, count=1).strip())
        return from_cord_parse(parse if isinstance(parse, dict) else {}), parse

    return read


def load_paddle(size: str, threads: int):
    import cv2

    from pipeline.extract import extract_regex
    from pipeline.ocr import group_lines
    from pipeline.preprocess import preprocess
    from scripts.eval_ocr import paddle_ocr

    cv2.setNumThreads(threads)
    ocr = paddle_ocr(size, threads)

    def read(path):
        lines = group_lines(*ocr(preprocess(cv2.imread(str(path)), ("resize",))[0]))
        return extract_regex(lines), lines

    return read


def load_gemini():
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        return None
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class Item(BaseModel):
        name: str
        qty: int
        price: int

    class Document(BaseModel):
        store_name: str
        date: str
        items: list[Item]
        subtotal: int
        tax: int
        total: int

    client = genai.Client()
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=Document)

    def read(path):
        mime = "image/png" if path.suffix == ".png" else "image/jpeg"
        response = client.models.generate_content(model=model, contents=[types.Part.from_bytes(data=path.read_bytes(), mime_type=mime), GEMINI_PROMPT], config=config)
        parsed = response.parsed.model_dump()
        result = {**parsed, "items": [{"name": item["name"], "price": item["price"]} for item in parsed["items"]]}
        result.update(store_name=parsed["store_name"] or None, date=parsed["date"] or None)
        return result, parsed

    read.model = model
    return read


def score(sample: dict, predicted: dict) -> dict:
    row = {"pred_total": predicted.get("total"), "gt_total": sample.get("total")}
    row["total_ok"] = None if row["gt_total"] is None else int(row["pred_total"] == row["gt_total"])
    row["item_f1"] = item_f1(predicted.get("items") or [], sample["items"]) if sample["set"] == "cord" else None
    if sample["set"] == "own":
        expected_store, store = sample.get("store_name"), predicted.get("store_name")
        row["store_ok"] = None if not expected_store else int(bool(store) and similarity(normalize(store), normalize(expected_store)) >= 0.8)
        row["date_ok"] = None if not sample.get("date") else int(predicted.get("date") == sample["date"])
    return row


def is_rate_limited(error: Exception) -> bool:
    return "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["donut", "paddle-tiny", "paddle-small", "gemini"], required=True)
    parser.add_argument("--sets", default="cord,own")
    parser.add_argument("--limit", type=int, default=100, help="maksimal gambar per set")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--name", help="nama file hasil, default <tugas>-<model>")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    samples = [sample for sample in load_samples(args.sets.split(","), args.limit) if args.engine != "donut" or sample["jenis"] == "struk"]
    process = psutil.Process()
    start = time.perf_counter()
    if args.engine == "donut":
        read, model = load_donut(), DONUT
    elif args.engine == "gemini":
        read = load_gemini()
        if read is None:
            print("GEMINI_API_KEY tidak ada di .env, benchmark Gemini dilewati")
            return
        model = read.model
    else:
        read, model = load_paddle(args.engine.split("-")[1], args.threads), f"PP-OCRv6_{args.engine.split('-')[1]} + regex"
    meta = {"engine": args.engine, "model": model, "threads": args.threads, "load_seconds": round(time.perf_counter() - start, 2), "rss_after_load_mb": round(rss_mb(process), 1)}
    if args.engine != "gemini":
        read(samples[0]["path"])

    rows = []
    for i, sample in enumerate(samples, 1):
        row = {"set": sample["set"], "file": sample["file"], "jenis": sample["jenis"]}
        for attempt in range(4):
            try:
                (predicted, raw), stats = measure(process, read, sample["path"])
                row.update(score(sample, predicted), **stats, error=None, raw=json.dumps(raw, ensure_ascii=False, default=str))
                break
            except Exception as error:  # noqa: BLE001
                if args.engine == "gemini" and is_rate_limited(error) and attempt < 3:
                    time.sleep(30)
                    continue
                row.update(error=f"{type(error).__name__}: {error}"[:300])
                break
        rows.append(row)
        print(f"[{args.engine}] {i}/{len(samples)} {sample['file']} total_ok={row.get('total_ok')} f1={row.get('item_f1')} {row.get('seconds')}s {row.get('error') or ''}", flush=True)
    save_run(args.name or f"extract-{args.engine}", rows, meta)


if __name__ == "__main__":
    main()
