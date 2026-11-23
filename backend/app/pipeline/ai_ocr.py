"""Reading scanned / handwritten pages with a vision model.

Used by the pipeline for pages that have no PDF text layer, when settings.ai_ocr is on
and AI consent is given (Gemini also needs gemini_consent; Groq's vision model is the
backup reader). The page image is downscaled, JPEG-encoded and sent through the LLM
router (retries, rate limits, model fallback, overload circuit breaker, ai_runs
logging) to the models in settings.ai_ocr_providers. The reply is one entry per
written line with a bounding box, turned into OcrLine objects so segmentation works
unchanged.

Speed (measured on the 2026-11-23 handwritten sheet: 194 s of "OCR" for 5 pages, two
pages never read): pages are now read CONCURRENTLY by :class:`PageReader` (bounded by
settings.ai_ocr_concurrency and the per-provider rate limits), a page is submitted the
moment it is rendered so reading overlaps rendering, the token reservation uses the
typical reply size instead of the 4096-token ceiling, and pages no reader could take
get one more pass at the end (after an overload cooldown has had time to pass) before
they fall back to the local engines.

Accuracy: the prompt keeps maths (every step, units), tables (" | " cells) and
diagrams ("[diagram: ...]") and marks doubtful lines ``unclear`` / unreadable words
``[illegible]``. Those lines get a lower OCR confidence, which the router turns into
the low_ocr_confidence review flag instead of the old flat 0.9 for every line.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator

from ..llm.providers import ImageInput
from ..llm.router import LLMUnavailable, parse_provider_spec
from .ocr import OcrLine

log = logging.getLogger(__name__)

TASK = "read_page"
PROMPT_VERSION = "read-page-v2"
ENGINE_PREFIX = "ai:"  # Submission.ocr_engine: "ai:gemini", "ai:groq"
REPEAT_MIN_CHARS = 12  # a repeated tail shorter than this may be a genuine repeat ("4", "Yes")
MAX_SIDE = 1600  # px: enough for handwriting, keeps the request small
JPEG_QUALITY = 82
CONFIDENCE = 0.92  # a line the model read without doubt (the model gives no per-line score)
UNCLEAR_CONFIDENCE = 0.6  # the model marked the line unclear (a guessed word)
ILLEGIBLE_CONFIDENCE = 0.35  # the line contains [illegible]: part of it was not read at all
MAX_LINE_HEIGHT = 0.12  # of the page: one written line is never taller
MAX_OUTPUT_TOKENS = 4096
EXPECTED_OUTPUT_TOKENS = 1100  # live replies: ~750 tokens for a full handwritten page
LIMITER_WAIT_SECONDS = 20.0  # an unread page costs a teacher far more than a short wait
RETRY_WAIT_CAP_SECONDS = 30.0  # longest wait for a cooldown before the final retry pass
DEFAULT_CONCURRENCY = 3
_ILLEGIBLE = re.compile(r"\[\s*illegible\s*\]", re.I)

PROMPT = (
    "This is one page of a student's handwritten exam answer sheet. Transcribe ALL the writing on it, line by "
    "line, top to bottom, exactly as written: keep the student's spelling, grammar, numbering and labels (Q1, "
    "2(a), Ans 3 ...) and punctuation; do not correct, translate, summarise, complete or answer anything.\n"
    "- Mathematics: write each expression on its line in plain text (^ for powers, / for fractions, sqrt(), "
    "and symbols such as = + - x ÷ ≤ ≥ ≠ ≈ π θ Δ Σ →). Keep every step of a calculation and every unit.\n"
    "- Tables: one line per row, cells separated by \" | \".\n"
    "- Diagrams, graphs or figures: one line \"[diagram: what is drawn, with its labels]\" where it appears.\n"
    "- A word you are not sure of: write your best reading and set unclear=true for that line. A word you "
    "cannot read at all: write [illegible] in its place and set unclear=true.\n"
    "- Leave out text the student struck through, faint show-through from the back of the page, ruled lines, "
    "margins and stamps.\n"
    "Write every line ONCE, in reading order, and stop at the last line on the page; never repeat lines. For "
    "each line give box_2d as [ymin, xmin, ymax, xmax] normalised to 0-1000. If the page has no writing, "
    "return an empty list."
)


class PageLine(BaseModel):
    text: str = Field(description="The line exactly as written")
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax] normalised to 0-1000")
    unclear: bool = Field(default=False, description="true if any word of the line was guessed or is [illegible]")

    @field_validator("text", mode="before")
    @classmethod
    def _null_is_empty(cls, v: Any) -> Any:  # noqa: ANN401
        return "" if v is None else v

    @field_validator("box_2d", mode="before")
    @classmethod
    def _box(cls, v: Any) -> Any:  # noqa: ANN401
        return [] if v is None else v

    @field_validator("unclear", mode="before")
    @classmethod
    def _null_is_false(cls, v: Any) -> Any:  # noqa: ANN401
        return False if v is None else v


class PageText(BaseModel):
    lines: list[PageLine]


def usable(settings: Any) -> bool:  # noqa: ANN401
    """AI reading is on and some reader has a key (the router also checks each provider's
    consent: Gemini needs gemini_consent)."""
    if not (getattr(settings, "ai_ocr", False) and settings.ai_consent and settings.ai_ocr_providers):
        return False
    names = {str(spec).split(":", 1)[0].strip().lower() for spec in settings.ai_ocr_providers}
    return (("gemini" in names and settings.gemini_api_key is not None and settings.gemini_consent)
            or ("groq" in names and settings.groq_api_key is not None)
            or ("openrouter" in names and settings.openrouter_api_key is not None)
            or ("ollama" in names and bool(settings.ollama_base_url)))


def encode_page(image: np.ndarray) -> ImageInput:
    h, w = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("could not encode the page image")
    return ImageInput(buf.tobytes(), "image/jpeg")


_NON_WORD = re.compile(r"[\W_]+")


def _norm(text: str) -> str:
    return _NON_WORD.sub("", text).lower()


def drop_repeated_tail(lines: list[PageLine]) -> list[PageLine]:
    """Vision models sometimes start the page over and repeat what they already wrote.
    Cut at the first point where everything after it was already read before it."""
    for k in range(1, len(lines)):
        head = "".join(_norm(ln.text) for ln in lines[:k])
        tail = "".join(_norm(ln.text) for ln in lines[k:])
        if len(tail) >= REPEAT_MIN_CHARS and tail in head:
            log.info("AI page reading: dropped %d repeated line(s)", len(lines) - k)
            return lines[:k]
    return lines


def line_confidence(line: PageLine) -> float:
    """OCR confidence of one line from the model's own doubt markers."""
    if _ILLEGIBLE.search(line.text):
        return ILLEGIBLE_CONFIDENCE
    return UNCLEAR_CONFIDENCE if line.unclear else CONFIDENCE


def to_lines(data: PageText, page_no: int, width: int, height: int) -> list[OcrLine]:
    """Model boxes are approximate (and sometimes inverted or page-tall). The pipeline
    orders a region's lines by position, so boxes are adjusted to follow the model's
    reading order: each line starts below the previous one, and is at most
    MAX_LINE_HEIGHT of the page tall."""
    out: list[OcrLine] = []
    texts = drop_repeated_tail([ln for ln in data.lines if ln.text.strip()])
    prev_y = -1
    max_h = max(1, round(height * MAX_LINE_HEIGHT))
    for i, ln in enumerate(texts):
        box = ln.box_2d
        if len(box) == 4 and all(isinstance(v, int) for v in box):
            y0, x0, y1, x1 = (min(1000, max(0, v)) for v in box)
            if y1 < y0:
                y0, y1 = y1, y0
            if x1 < x0:
                x0, x1 = x1, x0
            x, y = round(x0 * width / 1000), round(y0 * height / 1000)
            bw, bh = max(1, round((x1 - x0) * width / 1000)), max(1, round((y1 - y0) * height / 1000))
        else:  # no usable box: stack the lines evenly so their order is kept
            step = height / max(1, len(texts))
            x, y, bw, bh = 0, round(i * step), width, max(1, round(step * 0.8))
        y = min(max(y, prev_y + 1), height - 1)
        bh = max(1, min(bh, max_h, height - y))
        prev_y = y
        out.append(OcrLine(page_no, (x, y, bw, bh), ln.text.strip(), line_confidence(ln)))
    return out


def _read_encoded(router: Any, img: ImageInput, page_no: int, width: int, height: int,  # noqa: ANN401
                  settings: Any, *, exam_id: str | None, submission_id: str | None,  # noqa: ANN401
                  limiter_wait: float = LIMITER_WAIT_SECONDS) -> tuple[list[OcrLine], str] | None:
    try:
        res = router.generate_json(TASK, PROMPT, PageText, prompt_version=PROMPT_VERSION, images=[img],
                                   exam_id=exam_id, submission_id=submission_id,
                                   max_output_tokens=MAX_OUTPUT_TOKENS, expected_output_tokens=EXPECTED_OUTPUT_TOKENS,
                                   limiter_wait=limiter_wait, providers=list(settings.ai_ocr_providers))
    except LLMUnavailable as exc:
        log.warning("AI page reading unavailable for page %s (%d provider(s) tried)", page_no, len(exc.reasons))
        return None
    except Exception as exc:  # noqa: BLE001 - reading falls back to the local engines
        log.warning("AI page reading failed for page %s (%s)", page_no, type(exc).__name__)
        return None
    return to_lines(res.data, page_no, width, height), ENGINE_PREFIX + res.provider


def read_page(router: Any, image: np.ndarray, page_no: int, settings: Any, *,  # noqa: ANN401
              exam_id: str | None = None, submission_id: str | None = None) -> tuple[list[OcrLine], str] | None:
    """(the page's lines, engine name such as "ai:gemini"), or None when no vision model
    could read it (the caller falls back to the local engines)."""
    h, w = image.shape[:2]
    try:
        img = encode_page(image)
    except Exception as exc:  # noqa: BLE001
        log.warning("AI page reading: page %s could not be encoded (%s)", page_no, type(exc).__name__)
        return None
    return _read_encoded(router, img, page_no, w, h, settings, exam_id=exam_id, submission_id=submission_id)


def _cooldown_left(router: Any, settings: Any) -> float:  # noqa: ANN401
    """Shortest cooldown among the configured readers (0 when one is free now)."""
    limiter = getattr(router, "limiter", None)
    if limiter is None:
        return 0.0
    lefts = []
    for spec in settings.ai_ocr_providers:
        try:
            name, model = parse_provider_spec(spec)
        except ValueError:
            continue
        lefts.append(limiter.cooldown_left(f"{name}:{model}"))
    return min(lefts) if lefts else 0.0


class PageReader:
    """Reads pages concurrently: ``submit()`` a page as soon as it is rendered (its image
    is JPEG-encoded at once, so the full-size array can be freed), then ``results()``
    waits for every page and gives the unread ones one more sequential pass.

    `router_factory` is called once, on the first submitted page (no router for a sheet
    whose pages all have a text layer). `on_page(page_no, ok)` is called from the
    reading threads as each page finishes (used for live progress; must be thread-safe)."""

    def __init__(self, router_factory: Callable[[], Any], settings: Any, *,  # noqa: ANN401
                 exam_id: str | None = None, submission_id: str | None = None,
                 on_page: Callable[[int, bool], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._factory = router_factory
        self.settings = settings
        self.exam_id, self.submission_id = exam_id, submission_id
        self._on_page = on_page
        self._sleep = sleep
        self._router: Any = None
        self._pool: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._jobs: dict[int, tuple[Future, ImageInput | None, int, int]] = {}
        workers = int(getattr(settings, "ai_ocr_concurrency", DEFAULT_CONCURRENCY) or 1)
        self.workers = max(1, min(8, workers))

    @property
    def router(self) -> Any:  # noqa: ANN401
        with self._lock:
            if self._router is None:
                self._router = self._factory()
            return self._router

    def submit(self, page_no: int, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        try:
            img: ImageInput | None = encode_page(image)
        except Exception as exc:  # noqa: BLE001
            log.warning("AI page reading: page %s could not be encoded (%s)", page_no, type(exc).__name__)
            img = None
        router = self.router
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="saras-read")
        fut = self._pool.submit(self._read, router, img, page_no, w, h)
        self._jobs[page_no] = (fut, img, w, h)

    def _read(self, router: Any, img: ImageInput | None, page_no: int, w: int, h: int,  # noqa: ANN401
              limiter_wait: float = LIMITER_WAIT_SECONDS) -> tuple[list[OcrLine], str] | None:
        out = None
        if img is not None:
            out = _read_encoded(router, img, page_no, w, h, self.settings, exam_id=self.exam_id,
                                submission_id=self.submission_id, limiter_wait=limiter_wait)
        if self._on_page is not None:
            try:
                self._on_page(page_no, out is not None)
            except Exception:  # noqa: BLE001 - progress reporting never fails a page
                log.debug("page progress callback failed", exc_info=True)
        return out

    def results(self, tick: Callable[[], None] | None = None, on_done: Callable[[int], None] | None = None,
                tick_seconds: float = 1.0) -> dict[int, tuple[list[OcrLine], str] | None]:
        """{page_no: (lines, engine) or None} for every submitted page. Runs on the
        caller's thread: `on_done(page_no)` as each page is read (first or second pass),
        `tick()` about every `tick_seconds` while waiting (progress + job heartbeat)."""
        out: dict[int, tuple[list[OcrLine], str] | None] = {}
        try:
            pending = {fut: page_no for page_no, (fut, _i, _w, _h) in self._jobs.items()}
            while pending:
                done, _ = wait(list(pending), timeout=tick_seconds, return_when=FIRST_COMPLETED)
                for fut in done:
                    page_no = pending.pop(fut)
                    try:
                        out[page_no] = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        log.warning("AI page reading crashed for page %s (%s)", page_no, type(exc).__name__)
                        out[page_no] = None
                    if out[page_no] is not None and on_done is not None:
                        on_done(page_no)
                if tick is not None:
                    tick()
            failed = sorted(p for p, r in out.items() if r is None and self._jobs[p][1] is not None)
            if failed:
                # Every reader may be parked by an overload cooldown: give the shortest one
                # time to pass (bounded), then try the unread pages once more, one by one.
                wait_s = _cooldown_left(self._router, self.settings) if self._router is not None else 0.0
                if 0 < wait_s <= RETRY_WAIT_CAP_SECONDS:
                    self._sleep(wait_s + 0.5)
                for p in failed:
                    _fut, img, w, h = self._jobs[p]
                    log.info("AI page reading: second pass for page %s", p)
                    out[p] = self._read(self._router, img, p, w, h, limiter_wait=RETRY_WAIT_CAP_SECONDS)
                    if out[p] is not None and on_done is not None:
                        on_done(p)
                    if tick is not None:
                        tick()
        finally:
            self.close()
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        with self._lock:
            if self._router is not None:
                try:
                    self._router.close()
                except Exception:  # noqa: BLE001
                    log.debug("router close failed", exc_info=True)
                self._router = None
