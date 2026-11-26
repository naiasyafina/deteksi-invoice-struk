DIT = "victormurcia/DiT-DocumentClassifier-v1"
LABELS = {"invoices": "invoice", "receipts": "struk"}


def load_classifier():
    from transformers import pipeline

    return pipeline("image-classification", model=DIT, model_kwargs={"use_safetensors": True})


def pick_label(predictions: list[dict]) -> tuple[str, float]:
    scores = {LABELS[p["label"]]: p["score"] for p in predictions if p["label"] in LABELS}
    jenis = max(scores, key=scores.get)
    return jenis, scores[jenis] / sum(scores.values())


def classify(classifier, image) -> tuple[str, float]:
    return pick_label(classifier(image, top_k=None))


if __name__ == "__main__":
    jenis, score = pick_label([{"label": "bank_checks", "score": 0.6}, {"label": "receipts", "score": 0.3}, {"label": "invoices", "score": 0.1}])
    assert jenis == "struk" and abs(score - 0.75) < 1e-9
    print("ok")
