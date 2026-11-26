from collections.abc import Callable

import numpy as np

Box = tuple[float, float, float, float]
OcrResult = tuple[list[str], list[Box]]


def rapid_ocr(threads: int = 1, use_cls: bool = False) -> Callable[[np.ndarray], OcrResult]:
    from rapidocr_onnxruntime import RapidOCR

    options = {"use_cls": use_cls}
    if threads:
        options.update(intra_op_num_threads=threads, inter_op_num_threads=1)
    engine = RapidOCR(**options)

    def read(img: np.ndarray) -> OcrResult:
        result, _ = engine(img)
        if not result:
            return [], []
        texts = [text for _, text, _ in result]
        boxes = [(min(x for x, _ in pts), min(y for _, y in pts), max(x for x, _ in pts), max(y for _, y in pts)) for pts, _, _ in result]
        return texts, boxes

    return read


def group_lines(texts: list[str], boxes: list[Box]) -> list[str]:
    words = sorted(zip(texts, boxes), key=lambda word: (word[1][1] + word[1][3]) / 2)
    rows: list[list] = []
    for text, (left, top, _, bottom) in words:
        center = (top + bottom) / 2
        if rows and abs(center - rows[-1][0]) < (bottom - top) / 2:
            rows[-1][1].append((left, text))
        else:
            rows.append([center, [(left, text)]])
    return [" ".join(text for _, text in sorted(row)) for _, row in rows]


if __name__ == "__main__":
    texts = ["12.500", "Es Teh", "TOTAL", "12.500"]
    boxes = [(200, 10, 260, 30), (10, 12, 80, 30), (10, 50, 60, 70), (200, 51, 260, 69)]
    assert group_lines(texts, boxes) == ["Es Teh 12.500", "TOTAL 12.500"]
    print("ok")
