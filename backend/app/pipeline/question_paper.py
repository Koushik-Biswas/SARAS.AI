"""Reading a question paper into a draft rubric (Step 1: "upload question sheet").

The teacher uploads the exam's question paper (a PDF or a photo / scan). The result is
a *draft*: a list of questions with their number, type, marks and text. Nothing is
saved here; the Rubric screen shows the draft for the teacher to check and save.

Order of attempts:
1. PDF with a text layer -> the AI reads the text (text models, cheap) ->
   if the AI is unavailable, a rule-based parser reads "1. ... [5 marks]" style papers.
2. Scanned PDF or image -> the AI vision readers look at the pages.

A question paper holds no student data, so nothing needs anonymising.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pymupdf as fitz
from pydantic import BaseModel, Field

from ..errors import ApiError
from ..llm.router import LLMUnavailable
from .ai_ocr import encode_page
from .pdf import render_page

log = logging.getLogger(__name__)

TASK = "read_question_paper"
PROMPT_VERSION = "question-paper-v1"
MAX_PAGES = 10  # a question paper is a few pages; the AI request stays small
MIN_TEXT_CHARS = 40  # less than this on the text layer: treat the PDF as a scan
MAX_TEXT_CHARS = 12000  # sent to the AI in one request
MAX_QUESTIONS = 99
TYPES = ("short", "long", "paragraph", "maths")
IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")

INSTRUCTIONS = (
    "You read an exam QUESTION PAPER and list the questions a student must answer, for a marking rubric. "
    "Rules: (1) One entry per top-level question number (1, 2, 3 ...). Fold sub-parts (a), (b), (i), (ii) into "
    "their parent question: put all of their text in the parent's prompt and add up their marks. "
    "(2) question_no is the printed question number as an integer; if questions are unnumbered, number them "
    "1, 2, 3 in order. (3) prompt is the full question text exactly as printed (no 'Q1.' prefix, no marks "
    "label); never answer the question. (4) max_marks is the marks printed for that question; use 0 when no "
    "marks are printed. (5) type is one of: short (one-line or few-sentence answers, definitions, MCQ, "
    "fill-in), long (explain / describe answers of several sentences), paragraph (essays, letters, long "
    "discussion), maths (calculations, numericals, derivations, proofs). (6) Skip instructions, headers, "
    "school names, time allowed, section titles and 'answer any N' notes. If there are no questions, "
    "return an empty list."
)


class PaperQuestion(BaseModel):
    question_no: int = Field(description="Printed question number")
    type: str = Field(description="short | long | paragraph | maths")
    max_marks: float = Field(description="Marks printed for the question, 0 if none")
    prompt: str = Field(description="The question text exactly as printed")


class PaperQuestions(BaseModel):
    questions: list[PaperQuestion]


@dataclass
class DraftQuestion:
    question_no: int
    type: str
    max_marks: float | None
    prompt: str

    def as_dict(self) -> dict[str, Any]:
        return {"question_no": self.question_no, "type": self.type, "max_marks": self.max_marks,
                "prompt": self.prompt}


@dataclass
class Extraction:
    questions: list[DraftQuestion]
    source: str  # "ai:<provider>" | "text"
    pages: int
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"questions": [q.as_dict() for q in self.questions], "source": self.source, "pages": self.pages,
                "warnings": self.warnings}


# ---------------------------------------------------------------- input
@dataclass
class _Paper:
    text: str  # text layer (PDFs only)
    images: list[np.ndarray]  # rendered pages / the photo, grayscale
    pages: int


def _is_image(filename: str | None, content_type: str | None) -> bool:
    return (content_type or "").lower() in IMAGE_TYPES or (filename or "").lower().endswith(IMAGE_EXT)


def _load(data: bytes, filename: str | None, content_type: str | None, dpi: int) -> _Paper:
    if data[:1024].find(b"%PDF-") >= 0:
        return _load_pdf(data, dpi)
    if _is_image(filename, content_type):
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            raise ApiError(422, "unreadable_file", "The image could not be opened")
        return _Paper(text="", images=[img], pages=1)
    raise ApiError(415, "unsupported_media_type", "Upload the question paper as a PDF, PNG, JPG or WEBP file")


def _load_pdf(data: bytes, dpi: int) -> _Paper:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - corrupted file
        raise ApiError(422, "unreadable_pdf", "The PDF is corrupted and cannot be opened") from exc
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise ApiError(422, "unreadable_pdf", "The PDF is password-protected")
        if doc.page_count == 0:
            raise ApiError(422, "unreadable_pdf", "The PDF has no pages")
        if doc.page_count > MAX_PAGES:
            raise ApiError(413, "file_too_large",
                           f"The question paper has {doc.page_count} pages; the limit is {MAX_PAGES}")
        texts = []
        for i in range(doc.page_count):
            try:
                texts.append(doc.load_page(i).get_text("text"))
            except Exception:  # noqa: BLE001 - a damaged page is skipped
                continue
        text = "\n".join(texts).strip()
        images: list[np.ndarray] = []
        if len(text) < MIN_TEXT_CHARS:  # a scan: the pages are read as images
            for i in range(doc.page_count):
                try:
                    images.append(render_page(doc, i, dpi).image)  # render_page caps the page size
                except Exception:  # noqa: BLE001
                    continue
        return _Paper(text=text, images=images, pages=doc.page_count)
    finally:
        doc.close()


# ---------------------------------------------------------------- cleanup
_LEAD_NO = re.compile(r"^\s*(?:q(?:uestion)?\s*\.?\s*)?\d{1,2}\s*[.):\-]\s*", re.I)
_SPACES = re.compile(r"[ \t]+")


def _guess_type(prompt: str, marks: float | None) -> str:
    p = prompt.lower()
    if re.search(r"\b(calculate|solve|evaluate|simplify|prove|derive|compute|find the value|integrate|"
                 r"differentiate|how many|how much)\b", p) or re.search(r"\d\s*[+\-×x*/^=]\s*\d", p):
        return "maths"
    if re.search(r"\b(essay|write a (letter|paragraph|story|report)|discuss in detail|critically)\b", p):
        return "paragraph"
    if re.search(r"\b(explain|describe|discuss|compare|differentiate between|elaborate|justify)\b", p):
        return "long" if (marks or 0) < 8 else "paragraph"
    if marks is not None and marks >= 8:
        return "paragraph"
    if marks is not None and marks >= 4:
        return "long"
    return "short"


def _clean(raw: list[PaperQuestion]) -> tuple[list[DraftQuestion], list[str]]:
    """Valid, unique, numbered drafts from the AI's list (it is not trusted blindly)."""
    out: list[DraftQuestion] = []
    warnings: list[str] = []
    seen: set[int] = set()
    for q in raw:
        prompt = _SPACES.sub(" ", _LEAD_NO.sub("", (q.prompt or "").strip())).strip()
        if not prompt:
            continue
        no = q.question_no if isinstance(q.question_no, int) and 1 <= q.question_no <= MAX_QUESTIONS else None
        if no is None or no in seen:
            no = next((n for n in range(max(seen, default=0) + 1, MAX_QUESTIONS + 1) if n not in seen), None)
            if no is None:
                break
        seen.add(no)
        marks = float(q.max_marks) if q.max_marks and 0 < q.max_marks <= 1000 else None
        qtype = q.type.strip().lower() if isinstance(q.type, str) else ""
        out.append(DraftQuestion(no, qtype if qtype in TYPES else _guess_type(prompt, marks), marks, prompt[:5000]))
        if len(out) >= MAX_QUESTIONS:
            break
    missing = [q.question_no for q in out if q.max_marks is None]
    if missing:
        warnings.append(f"No marks were printed for Q{', Q'.join(map(str, missing))}: enter them before saving.")
    return sorted(out, key=lambda q: q.question_no), warnings


# ---------------------------------------------------------------- rule-based reader
_Q_START = re.compile(r"^\s*(?:q(?:uestion)?\s*\.?\s*(\d{1,2})\s*[.):\-]?|(\d{1,2})\s*[.)])\s+(\S.*)$", re.I)
_MARKS = re.compile(
    r"[\[(]\s*(\d{1,3}(?:\.\d+)?)\s*(?:marks?|mks?|m)?\s*[\])]\s*$"  # [5] (5) (5 marks) [5m]
    r"|(?:^|\s)(\d{1,3}(?:\.\d+)?)\s*marks?\s*\.?\s*$",  # ... 5 marks
    re.I,
)


def _split_marks(text: str) -> tuple[str, float | None]:
    m = _MARKS.search(text)
    if not m:
        return text, None
    value = float(m.group(1) or m.group(2))
    return text[: m.start()].rstrip(" \t-:–"), value if 0 < value <= 1000 else None


def parse_text(text: str) -> list[PaperQuestion]:
    """Questions from a typed paper: lines starting "1.", "1)", "Q1", "Question 1:".
    Lines in between belong to the question above; text before the first is skipped."""
    found: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _Q_START.match(line)
        if m:
            found.append({"no": int(m.group(1) or m.group(2)), "lines": [m.group(3)]})
        elif found:
            found[-1]["lines"].append(line)
    out = []
    for f in found:
        body, marks = _split_marks(" ".join(f["lines"]).strip())
        if marks is None:  # marks on their own line: "(5)"
            for ln in reversed(f["lines"]):
                _, marks = _split_marks(ln)
                if marks is not None:
                    break
        out.append(PaperQuestion(question_no=f["no"], type="", max_marks=marks or 0, prompt=body))
    return out


# ---------------------------------------------------------------- entry point
def extract(data: bytes, filename: str | None, content_type: str | None, settings: Any,  # noqa: ANN401
            make_router: Any, *, exam_id: str | None = None) -> Extraction:  # noqa: ANN401
    """Draft questions from a question paper. `make_router()` builds the LLM router
    (only called when the AI is needed). Raises ApiError when nothing can be read."""
    paper = _load(data, filename, content_type, int(getattr(settings, "render_dpi", 150)))
    ai_reason = ""
    if settings.ai_consent:
        router = None
        try:
            router = make_router()
            if paper.text and len(paper.text) >= MIN_TEXT_CHARS:
                res = router.generate_json(
                    TASK, f"{INSTRUCTIONS}\n\nQUESTION PAPER TEXT:\n{paper.text[:MAX_TEXT_CHARS]}", PaperQuestions,
                    prompt_version=PROMPT_VERSION, exam_id=exam_id, max_output_tokens=4096)
            elif paper.images:
                res = router.generate_json(
                    TASK, f"{INSTRUCTIONS}\n\nThe question paper pages are attached as images, in order.",
                    PaperQuestions, prompt_version=PROMPT_VERSION, exam_id=exam_id, max_output_tokens=4096,
                    images=[encode_page(img) for img in paper.images[:MAX_PAGES]],
                    providers=list(settings.ai_ocr_providers))
            else:
                res = None
            if res is not None:
                drafts, warnings = _clean(res.data.questions)
                if drafts:
                    return Extraction(drafts, "ai:" + res.provider, paper.pages, warnings)
                ai_reason = "The AI found no questions on the paper."
        except LLMUnavailable as exc:
            log.warning("AI question-paper reading unavailable (%d provider(s) tried)", len(exc.reasons))
            ai_reason = "The AI engines are busy or unavailable right now."
        except Exception as exc:  # noqa: BLE001 - fall back to the rule-based reader
            log.warning("AI question-paper reading failed (%s)", type(exc).__name__)
            ai_reason = "The AI engines could not read the paper."
        finally:
            if router is not None:
                router.close()
    else:
        ai_reason = "AI reading is off (no AI consent)."

    if paper.text:
        drafts, warnings = _clean(parse_text(paper.text))
        if drafts:
            return Extraction(drafts, "text", paper.pages,
                              [f"{ai_reason} Questions were read from the PDF text: check them.".strip(),
                               *warnings])
    if not paper.text:
        raise ApiError(422, "no_questions_found",
                       f"{ai_reason} A scanned paper or photo needs the AI reader; type the questions instead."
                       .strip())
    raise ApiError(422, "no_questions_found",
                   "No numbered questions were found in the paper. Type the questions instead.")
