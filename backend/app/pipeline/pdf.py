"""PDF validation and rendering (PyMuPDF)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (the bare `fitz` alias is deprecated in 2.x)
import numpy as np

from ..errors import ApiError


@dataclass
class TextLine:
    bbox: tuple[int, int, int, int]  # x, y, w, h in rendered-pixel space
    text: str


@dataclass
class RenderedPage:
    page_no: int  # 1-based
    image: np.ndarray  # grayscale uint8
    text_lines: list[TextLine] = field(default_factory=list)


# A page is rendered as one grayscale bitmap (1 byte per pixel) and copied a few times
# while it is cleaned up, all inside the API process. A tiny PDF can declare a huge page
# (14400 x 14400 pt is 900 MP at 150 dpi), so the rendered size is capped. A3 at 150 dpi
# is about 4.4 MP; no real answer sheet comes near this budget.
MAX_PAGE_PIXELS = 25_000_000
MAX_PAGE_SIDE = 10_000


def render_size(rect: fitz.Rect, dpi: float) -> tuple[int, int]:
    z = dpi / 72.0
    return max(1, round(abs(rect.width) * z)), max(1, round(abs(rect.height) * z))


def safe_dpi(rect: fitz.Rect, dpi: float) -> float:
    """`dpi`, lowered when needed so the page bitmap stays within the pixel budget."""
    w, h = render_size(rect, dpi)
    scale = min(1.0, (MAX_PAGE_PIXELS / (w * h)) ** 0.5, MAX_PAGE_SIDE / max(w, h))
    return dpi * scale


def validate_pdf(path: Path, max_pages: int, dpi: int = 150) -> int:
    """Return the page count or raise an ApiError matching PRD 16.1 error codes."""
    with open(path, "rb") as fh:
        head = fh.read(1024)
    if b"%PDF-" not in head:
        raise ApiError(415, "unsupported_media_type", "The file is not a PDF")
    try:
        doc = fitz.open(path)
    except Exception as exc:  # corrupted file
        raise ApiError(422, "unreadable_pdf", "The PDF is corrupted and cannot be opened") from exc
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise ApiError(422, "unreadable_pdf", "The PDF is password-protected")
        count = doc.page_count
        if count == 0:
            raise ApiError(422, "unreadable_pdf", "The PDF has no pages")
        if count > max_pages:
            raise ApiError(413, "file_too_large", f"The PDF has {count} pages; the limit is {max_pages}")
        for i in range(count):
            try:
                rect = doc.load_page(i).rect
            except Exception:  # a damaged page is flagged while processing (per-page isolation)
                continue
            w, h = render_size(rect, dpi)
            if w * h > MAX_PAGE_PIXELS or max(w, h) > MAX_PAGE_SIDE:
                raise ApiError(422, "unreadable_pdf",
                               f"Page {i + 1} is far larger than an answer sheet ({w}x{h} px at {dpi} dpi)")
        return count
    finally:
        doc.close()


def render_page(doc: fitz.Document, index: int, dpi: int) -> RenderedPage:
    page = doc.load_page(index)
    dpi = safe_dpi(page.rect, dpi)  # never trust the page size a PDF declares
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), colorspace=fitz.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    del pix
    scale = dpi / 72.0

    lines: list[TextLine] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            lines.append(TextLine(
                bbox=(int(x0 * scale), int(y0 * scale), max(1, int((x1 - x0) * scale)), max(1, int((y1 - y0) * scale))),
                text=text,
            ))
    lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    return RenderedPage(page_no=index + 1, image=img, text_lines=lines)
