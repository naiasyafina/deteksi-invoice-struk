import json
from collections import Counter
from difflib import SequenceMatcher

from pipeline.postprocess import normalize, parse_money


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


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def line_errors(expected_lines: list[str], predicted_lines: list[str]) -> tuple[int, int]:
    predicted = [line.replace(" ", "") for line in predicted_lines] or [""]
    errors = chars = 0
    for line in expected_lines:
        line = line.replace(" ", "")
        best = len(line)
        for candidate in predicted:
            if abs(len(candidate) - len(line)) < best:
                best = min(best, edit_distance(line, candidate))
        errors += best
        chars += len(line)
    return errors, chars


def as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def cord_words(raw_ground_truth: str) -> list[str]:
    lines = json.loads(raw_ground_truth).get("valid_line", [])
    return [word["text"] for line in lines for word in line.get("words", [])]


def cord_lines(raw_ground_truth: str) -> list[str]:
    rows: dict[int, list] = {}
    for line in json.loads(raw_ground_truth).get("valid_line", []):
        for word in line.get("words", []):
            rows.setdefault(word["row_id"], []).append((word["quad"]["x1"], word["quad"]["y1"], word["text"]))
    ordered = sorted(rows.values(), key=lambda words: min(y for _, y, _ in words))
    return [" ".join(text for _, _, text in sorted(words)) for words in ordered]


def cord_fields(raw_ground_truth: str) -> dict[str, int | None]:
    parse = json.loads(raw_ground_truth)["gt_parse"]

    def first(group: str, key: str) -> int | None:
        values = [value for entry in as_list(parse.get(group)) for value in as_list(entry.get(key))]
        return abs(parse_money(str(values[0])) or 0) if values else None

    return {
        "subtotal": first("sub_total", "subtotal_price"),
        "tax": first("sub_total", "tax_price"),
        "total": first("total", "total_price"),
    }


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
    assert edit_distance("kitten", "sitting") == 3
    assert line_errors(["TOTAL 12.500", "Es Teh"], ["Es Teh", "T0TAL 12.500", "header"]) == (1, 16)
    assert line_errors(["abc"], []) == (3, 3)
    raw = json.dumps({
        "gt_parse": {"sub_total": {"subtotal_price": "35,000", "tax_price": "3.500"}, "total": [{"total_price": "38.500"}]},
        "valid_line": [
            {"words": [{"row_id": 2, "text": "38.500", "quad": {"x1": 90, "y1": 40}}, {"row_id": 2, "text": "TOTAL", "quad": {"x1": 5, "y1": 41}}]},
            {"words": [{"row_id": 1, "text": "Teh", "quad": {"x1": 5, "y1": 10}}]},
        ],
    })
    assert cord_lines(raw) == ["Teh", "TOTAL 38.500"]
    assert cord_fields(raw) == {"subtotal": 35000, "tax": 3500, "total": 38500}
    print("ok")
