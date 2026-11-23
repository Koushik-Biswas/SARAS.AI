"""Image clean-up and text-line detection with OpenCV (PRD 12.1, 12.3).

The original page image is always kept for the review screen; these functions
produce a cleaned copy that only the OCR stage sees."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PageQuality:
    ink_ratio: float
    blur_score: float
    is_blank: bool
    is_poor: bool


def assess(gray: np.ndarray) -> PageQuality:
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) if gray.shape[0] > 1200 else gray
    _, inv = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = float(np.count_nonzero(inv)) / inv.size
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
    contrast = float(small.std())
    is_blank = ink < 0.002 or contrast < 4
    return PageQuality(ink, blur, is_blank, is_poor=(not is_blank and (blur < 15 or contrast < 18)))


def deskew(gray: np.ndarray, max_angle: float = 10.0) -> tuple[np.ndarray, float]:
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(inv)
    if coords is None or len(coords) < 50:
        return gray, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.3 or abs(angle) > max_angle:
        return gray, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def clean(gray: np.ndarray, variant: int = 0) -> np.ndarray:
    """Return a binarised image (black ink on white). variant=1 is the stronger
    retry used when OCR confidence is very low (PRD 18.3)."""
    den = cv2.medianBlur(gray, 3) if variant == 0 else cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0 if variant == 0 else 3.0, tileGridSize=(8, 8))
    eq = clahe.apply(den)
    block = 31 if variant == 0 else 41
    binary = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 15)
    return remove_ruled_lines(binary)


def remove_ruled_lines(binary: np.ndarray) -> np.ndarray:
    inv = 255 - binary
    width = binary.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 6), 1))
    lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.subtract(inv, lines)
    return 255 - cleaned


def detect_lines(binary: np.ndarray, min_height: int = 8) -> list[tuple[int, int, int, int]]:
    """Horizontal projection profile → text-line boxes (x, y, w, h)."""
    inv = (255 - binary) // 255
    h, w = inv.shape
    margin = int(w * 0.02)
    inv[:, :margin] = 0
    inv[:, w - margin:] = 0
    rows = inv.sum(axis=1)
    thresh = max(2, int(rows.max() * 0.02)) if rows.max() > 0 else 1
    on = rows > thresh

    spans: list[list[int]] = []
    start = None
    for y, flag in enumerate(on):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            spans.append([start, y])
            start = None
    if start is not None:
        spans.append([start, h])

    # merge spans separated by tiny gaps (descenders, i-dots)
    merged: list[list[int]] = []
    for s in spans:
        if merged and s[0] - merged[-1][1] < 6:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    boxes = []
    for y0, y1 in merged:
        if y1 - y0 < min_height:
            continue
        cols = np.where(inv[y0:y1].sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x0, x1 = int(cols[0]), int(cols[-1])
        pad = 4
        boxes.append((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad) - max(0, x0 - pad), min(h, y1 + pad) - max(0, y0 - pad)))
    return boxes
