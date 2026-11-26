import cv2
import numpy as np

STEPS = ("resize", "crop", "deskew", "gray")
MAX_SIDE = 2000


def find_document(img: np.ndarray) -> np.ndarray | None:
    scale = 500 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if not 0.2 < cv2.contourArea(contour) / mask.size < 0.95:
        return None
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    quad = approx.reshape(-1, 2) if len(approx) == 4 and cv2.isContourConvex(approx) else cv2.boxPoints(cv2.minAreaRect(contour))
    return order_corners(quad.astype(np.float32) / scale)


def order_corners(quad: np.ndarray) -> np.ndarray:
    total, diff = quad.sum(axis=1), np.diff(quad, axis=1).ravel()
    return np.array([quad[total.argmin()], quad[diff.argmin()], quad[total.argmax()], quad[diff.argmax()]], np.float32)


def warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, target), (width, height), borderMode=cv2.BORDER_REPLICATE)


def skew_angle(img: np.ndarray, max_angle: float = 10, step: float = 0.5) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = min(1.0, 600 / max(gray.shape))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    center = (ink.shape[1] / 2, ink.shape[0] / 2)

    def sharpness(angle: float) -> float:
        rotated = cv2.warpAffine(ink, cv2.getRotationMatrix2D(center, angle, 1), ink.shape[::-1])
        return float(np.var(rotated.sum(axis=1, dtype=np.float64)))

    return max(np.arange(-max_angle, max_angle + step, step), key=sharpness)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def preprocess(img: np.ndarray, steps: tuple[str, ...] = STEPS) -> tuple[np.ndarray, np.ndarray | None]:
    quad = None
    if "resize" in steps and max(img.shape[:2]) > MAX_SIDE:
        scale = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if "crop" in steps:
        quad = find_document(img)
        if quad is not None:
            img = warp(img, quad)
    if "deskew" in steps:
        angle = skew_angle(img)
        if abs(angle) >= 0.5:
            img = rotate(img, angle)
    if "gray" in steps:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return img, quad


if __name__ == "__main__":
    page = np.full((600, 400, 3), 255, np.uint8)
    for y in range(60, 560, 40):
        cv2.putText(page, "TOTAL 12.500", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    assert abs(skew_angle(rotate(page, 4)) + 4) <= 0.5, skew_angle(rotate(page, 4))
    canvas = np.full((900, 700, 3), 40, np.uint8)
    canvas[150:750, 150:550] = page
    quad = find_document(canvas)
    assert quad is not None and np.abs(quad - [[150, 150], [549, 150], [549, 749], [150, 749]]).max() < 6, quad
    height, width = preprocess(canvas)[0].shape[:2]
    assert abs(height - 600) < 8 and abs(width - 400) < 8, (height, width)
    print("ok")
