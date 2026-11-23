"""Answer segmentation by question markers (PRD 12.2, 12.10).

Markers such as "Q1", "Q.2", "Question 3:", "Ans 4", "5.", "6)" and sub-parts
"2(a)" are found at the start of OCR lines. Text between consecutive markers is
one answer, continuing across pages until the next marker. Sub-parts are merged
into their parent question because the rubric has no sub-parts in the MVP.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ocr import OcrLine

_PREFIXED = re.compile(
    r"^\s*(?:q|qn|qs|que|ques|question|ans|answer)\s*(?:no\.?|n0\.?)?\s*[\.\-:#]?\s*"
    r"(\d{1,2})\s*(?:\(?\s*([a-h]|i{1,3}|iv|v)\s*\))?\s*[\.\):\-]?\s*(.*)$",
    re.IGNORECASE,
)
# A bare number needs trailing punctuation ("5.", "6)") so "2 apples" is not a
# marker; a parenthesised sub-part is distinctive enough on its own ("2(a) ...").
_BARE = re.compile(r"^\s*(\d{1,2})\s*(?:\(\s*([a-h])\s*\)\s*[\.\):]?|[\.\):])\s*(.*)$")


@dataclass
class Marker:
    number: int
    rest: str  # answer text on the marker line after the marker itself


def parse_marker(text: str, valid: set[int]) -> Marker | None:
    for rx in (_PREFIXED, _BARE):
        m = rx.match(text)
        if m:
            n = int(m.group(1))
            if n in valid:
                return Marker(n, m.group(3).strip())
    return None


def strip_marker(text: str, valid: set[int]) -> str:
    m = parse_marker(text, valid)
    return m.rest if m else text


@dataclass
class Region:
    page_no: int
    bbox: tuple[int, int, int, int]


@dataclass
class Segment:
    question_no: int | None  # None = text before the first marker / unassigned
    lines: list[OcrLine] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    confidence: float = 0.95
    flags: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(t for t in self.texts if t).strip()

    def regions(self, page_widths: dict[int, int]) -> list[Region]:
        by_page: dict[int, list[OcrLine]] = {}
        for ln in self.lines:
            by_page.setdefault(ln.page_no, []).append(ln)
        out = []
        for page_no, lines in sorted(by_page.items()):
            y0 = min(ln.bbox[1] for ln in lines)
            y1 = max(ln.bbox[1] + ln.bbox[3] for ln in lines)
            width = page_widths.get(page_no, max(ln.bbox[0] + ln.bbox[2] for ln in lines))
            margin = int(width * 0.03)
            pad = 6
            out.append(Region(page_no, (margin, max(0, y0 - pad), width - 2 * margin, (y1 - y0) + 2 * pad)))
        return out


@dataclass
class SegmentationResult:
    segments: dict[int, Segment]  # by question number
    unassigned: list[Segment]
    missing: list[int]
    sheet_flags: list[str]
    # Missing numbers after the last marker found: a misread last marker looks exactly
    # like a skipped question (its text runs on into the previous answer).
    missing_after_last: list[int] = field(default_factory=list)


def segment(lines: list[OcrLine], question_numbers: list[int]) -> SegmentationResult:
    valid = set(question_numbers)
    segments: dict[int, Segment] = {}
    unassigned: list[Segment] = []
    sheet_flags: list[str] = []
    current: Segment | None = None
    seen_order: list[int] = []

    for ln in lines:
        marker = parse_marker(ln.text, valid) if ln.text else None
        if marker:
            n = marker.number
            if n in segments:
                # Duplicate marker: keep it with the same question but lower confidence.
                current = segments[n]
                if "duplicate_marker" not in current.flags:
                    current.flags.append("duplicate_marker")
                current.confidence = min(current.confidence, 0.55)
            else:
                current = Segment(question_no=n)
                segments[n] = current
            if seen_order and n < seen_order[-1] and "out_of_order" not in current.flags:
                current.flags.append("out_of_order")
                current.confidence = min(current.confidence, 0.8)
            seen_order.append(n)
            current.lines.append(ln)
            current.texts.append(marker.rest)
            continue

        if current is None:
            if not unassigned or unassigned[-1].lines[-1].page_no != ln.page_no:
                unassigned.append(Segment(question_no=None, confidence=0.3, flags=["unassigned"]))
            unassigned[-1].lines.append(ln)
            unassigned[-1].texts.append(ln.text)
        else:
            current.lines.append(ln)
            current.texts.append(ln.text)

    missing = [n for n in sorted(valid) if n not in segments]
    if not segments and lines:
        sheet_flags.append("no_markers_found")
    # Text before the first marker usually means that marker was misread.
    if segments and any(s.text for s in unassigned):
        sheet_flags.append("unassigned_text")
    if any("duplicate_marker" in s.flags for s in segments.values()):
        sheet_flags.append("duplicate_marker")
    if any("out_of_order" in s.flags for s in segments.values()):
        sheet_flags.append("out_of_order")
    # A gap inside the found sequence suggests a misread marker rather than a skipped question.
    if segments and missing:
        lo, hi = min(segments), max(segments)
        if any(lo < m < hi for m in missing):
            sheet_flags.append("gap_in_sequence")
            for n, seg in segments.items():
                if n + 1 in missing or n - 1 in missing:
                    seg.confidence = min(seg.confidence, 0.7)
                    if "neighbour_missing" not in seg.flags:
                        seg.flags.append("neighbour_missing")

    # Answers that continue across pages without a marker are expected; just note it.
    for seg in segments.values():
        if len({ln.page_no for ln in seg.lines}) > 1:
            seg.flags.append("continues_across_pages")

    after_last = [m for m in missing if segments and m > max(segments)]
    return SegmentationResult(segments, unassigned, missing, sheet_flags, after_last)
