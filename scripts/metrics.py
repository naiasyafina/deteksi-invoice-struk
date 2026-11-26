import json
from collections import Counter
from difflib import SequenceMatcher

from pipeline.postprocess import normalize


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def count_exact_matches(predicted: list, expected: list) -> int:
    remaining = list(expected)
    matched = 0
    for value in predicted:
        if value in remaining:
            remaining.remove(value)
            matched += 1
    return matched


def count_fuzzy_matches(predicted: list[str], expected: list[str], threshold: float = 0.8) -> int:
    remaining = [normalize(text) for text in expected]
    matched = 0
    for text in map(normalize, predicted):
        best = max(remaining, key=lambda candidate: similarity(text, candidate), default=None)
        if best is not None and similarity(text, best) >= threshold:
            remaining.remove(best)
            matched += 1
    return matched


def f1(matched: int, predicted: int, expected: int) -> float:
    if predicted == expected == 0:
        return 1.0
    return round(2 * matched / (predicted + expected), 3)


def cord_words(raw_ground_truth: str) -> list[str]:
    lines = json.loads(raw_ground_truth).get("valid_line", [])
    return [word["text"] for line in lines for word in line.get("words", [])]


def word_recall(expected_words: list[str], texts: list[str]) -> float | None:
    expected = Counter(word for word in map(normalize, expected_words) if word)
    found = Counter(word for text in texts for word in map(normalize, text.split()) if word)
    total = sum(expected.values())
    return round(sum((expected & found).values()) / total, 3) if total else None


if __name__ == "__main__":
    assert count_exact_matches([100, 100, 200], [100, 300]) == 1
    assert count_fuzzy_matches(["Es Teh Manis"], ["ES TEH MANIS", "Nasi"]) == 1
    assert f1(1, 2, 2) == 0.5
    assert f1(0, 0, 0) == 1.0
    assert word_recall(["Es", "Teh", "12.500"], ["es teh", "12,500"]) == 1.0
    assert word_recall(["Es", "Teh", "Manis"], ["es teh"]) == round(2 / 3, 3)
    print("ok")
