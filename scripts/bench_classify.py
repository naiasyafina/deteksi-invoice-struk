import argparse
import time

import psutil
from PIL import Image

from scripts.profiling import measure, rss_mb, save_run
from scripts.samples import load_samples

DIT = "victormurcia/DiT-DocumentClassifier-v1"
CLIP = "openai/clip-vit-base-patch32"
DIT_LABELS = {"invoices": "invoice", "receipts": "struk", "bank_checks": "lainnya"}
CLIP_PROMPTS = {
    "a photo of a shopping receipt from a store or restaurant": "struk",
    "a scanned invoice or bill document": "invoice",
    "a photo of some other document": "lainnya",
}


def load_classifier(model: str):
    from transformers import pipeline

    if model == "dit":
        dit = pipeline("image-classification", model=DIT, model_kwargs={"use_safetensors": True})

        def classify(image):
            top = dit(image, top_k=1)[0]
            return DIT_LABELS.get(top["label"], "lainnya"), top["label"], top["score"]

        return classify

    clip = pipeline("zero-shot-image-classification", model=CLIP, model_kwargs={"use_safetensors": True})

    def classify(image):
        top = clip(image, candidate_labels=list(CLIP_PROMPTS))[0]
        return CLIP_PROMPTS[top["label"]], top["label"], top["score"]

    return classify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dit", "clip"], required=True)
    parser.add_argument("--sets", default="cord,invoice,own")
    parser.add_argument("--limit", type=int, default=100, help="maksimal gambar per set")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--name", help="nama file hasil, default <tugas>-<model>")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    samples = load_samples(args.sets.split(","), args.limit)
    process = psutil.Process()
    start = time.perf_counter()
    classify = load_classifier(args.model)
    meta = {"model": DIT if args.model == "dit" else CLIP, "threads": args.threads, "load_seconds": round(time.perf_counter() - start, 2), "rss_after_load_mb": round(rss_mb(process), 1)}
    classify(Image.open(samples[0]["path"]).convert("RGB"))

    rows = []
    for i, sample in enumerate(samples, 1):
        image = Image.open(sample["path"]).convert("RGB")
        (predicted, raw_label, score), stats = measure(process, classify, image)
        rows.append({"set": sample["set"], "file": sample["file"], "expected": sample["jenis"], "predicted": predicted, "raw_label": raw_label, "score": round(score, 4), **stats})
        print(f"[{args.model}] {i}/{len(samples)} {sample['file']} {sample['jenis']}->{predicted} {stats['seconds']}s", flush=True)
    save_run(args.name or f"classify-{args.model}", rows, meta)


if __name__ == "__main__":
    main()
