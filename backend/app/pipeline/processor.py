"""End-to-end processing of one submission (PRD 9):

render → quality check → preprocess → OCR → segment → grade → route → totals.

Each page is isolated: a failing page is flagged and the rest of the sheet still
finishes (PRD 8, 18.3). Processing is idempotent — re-running a job rebuilds the
submission's lines, regions and answers from scratch.

Grading (docs/GRADING_DESIGN.md): every gradable answer is marked by the local
graders (app.grading.ensemble: the local primary and its cross-check); when AI
marking is enabled and consented, ONE AI request covers the whole sheet; then
ensemble.combine() decides per criterion and routing decides auto-accept vs review.
The AI is never in the critical path: any AI failure leaves the offline result in
place (flag ai_unavailable_fallback). Logs carry ids, statuses and scores only —
never answer text, student_ref, names or keys.
"""

from __future__ import annotations

import gc
import logging
import shutil
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import pymupdf as fitz  # the bare `fitz` alias is deprecated in PyMuPDF 2.x
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session, sessionmaker

from ..config import Settings
from ..grading import ensemble as E
from ..grading.engine import GradeResult, Thresholds
from ..models import AiRun, Answer, AnswerRegion, Job, OcrLine, Question, Submission
from ..grading import direct_ai
from . import ai_ocr, ai_segment, preprocess
from .ocr import AutoOCR, OcrLine as OcrLineDC, get_ocr_engine
from .pdf import RenderedPage, render_page
from .progress import Progress
from .routing import combine, is_review_flag, route
from .segment import segment, strip_marker

log = logging.getLogger(__name__)

SEGMENTATION_REVIEW_FLAGS = {"duplicate_marker", "neighbour_missing", "unassigned", "page_failed",
                             "next_page_failed", "page_unread"}
# Part of the answer's text is missing (an unread or failed page): its marks are never
# accepted without a teacher, even when SARAS_REVIEW_REQUIRED=false.
INCOMPLETE_TEXT_FLAGS = frozenset({"needs_transcription", "page_failed", "next_page_failed", "page_unread"})
# Only sheets without teacher-visible results are (re)built. A processed, in-review or
# approved sheet is never touched by a left-over or re-claimed job: rebuilding deletes
# its answers, and with them every review, correction and approved mark.
PROCESSABLE = ("uploaded", "processing", "failed")
TRACE_VERSION = 1


class SubmissionGone(Exception):
    """The sheet was deleted while it was being processed."""


def page_image_path(storage: Path, submission_id: str, page_no: int) -> Path:
    return storage / "submissions" / submission_id / f"page_{page_no:03d}.png"


def remove_submission_storage(storage: Path, submission_id: str) -> None:
    """Delete <storage>/submissions/<id>: used when a sheet was deleted while the
    worker still held its PDF open (Windows cannot remove an open file)."""
    root = (storage / "submissions").resolve()
    folder = (root / submission_id).resolve()
    if folder.parent != root or not folder.exists():
        return
    shutil.rmtree(folder, ignore_errors=True)
    if folder.exists():  # a PDF handle may linger until its owner is collected
        gc.collect()
        shutil.rmtree(folder, ignore_errors=True)
    if folder.exists():
        log.warning("Could not remove %s", folder)


def _exists(db: Session, submission_id: str) -> bool:
    return db.execute(select(Submission.id).where(Submission.id == submission_id)).first() is not None


def _thresholds(s: Settings, scorer_name: str = "lexical") -> Thresholds:
    """Thresholds for one scorer: the engine's per-scorer defaults plus any Settings
    value an operator changed (see ensemble.thresholds_for)."""
    return E.thresholds_for(s, scorer_name)


class _Tracker:
    """Progress of the sheet being processed plus the commit that publishes it (the
    worker). Request handlers that re-mark answers get the silent default."""

    def __init__(self, progress: Progress | None = None, flush: Callable[[], None] | None = None) -> None:
        self.p = progress or Progress()
        self._flush = flush

    def flush(self) -> None:
        if self._flush is not None:
            self._flush()


def _local_read(ocr, rp: RenderedPage, display, settings: Settings) -> tuple[list[OcrLineDC], str]:  # noqa: ANN001
    """Read one page with the local engines (text layer / TrOCR / line boxes only). The
    binarised copy is made only here: pages the vision model reads never pay for it."""
    cleaned = None if rp.text_lines else preprocess.clean(display)
    lines = ocr.read_page(rp, cleaned)
    if cleaned is not None and lines and _mean_conf(lines) < settings.low_ocr_confidence \
            and (isinstance(ocr, AutoOCR) and ocr.last_used == "trocr" or ocr.name == "trocr"):
        retry = ocr.read_page(rp, preprocess.clean(display, variant=1))
        if _mean_conf(retry) > _mean_conf(lines):
            lines = retry
    return lines, (ocr.last_used if isinstance(ocr, AutoOCR) else ocr.name)


def process_submission(db: Session, submission_id: str, settings: Settings) -> None:  # noqa: C901, PLR0912, PLR0915
    sub = db.get(Submission, submission_id)
    if sub is None:
        return
    if sub.status not in PROCESSABLE:
        log.warning("Not re-processing submission %s: it is already %s", sub.id, sub.status)
        return
    questions = sorted(sub.exam.questions, key=lambda q: q.question_no)
    timings: dict[str, float] = {"render_preprocess": 0.0, "ocr": 0.0, "segmentation": 0.0, "grading": 0.0}
    started = time.perf_counter()
    progress = Progress(pages_total=sub.page_count or 0, questions_total=len(questions))

    # Idempotent restart: clear any partial output from an earlier attempt.
    for table in (Answer, AnswerRegion, OcrLine):
        db.execute(delete(table).where(table.submission_id == sub.id))
    sub.status, sub.stage, sub.pages_done, sub.error = "processing", "ocr", 0, None
    sub.flags, sub.page_flags = [], {}
    sub.progress = progress.snapshot()
    db.commit()
    db.expire(sub)
    exam_id, submission_id = sub.exam_id, sub.id
    done_pages: set[int] = set()

    def flush() -> None:
        """Publish progress + heartbeat (worker thread only; reader threads only touch `progress`)."""
        sub.pages_done = len(done_pages)
        sub.progress = progress.snapshot()
        db.execute(update(Job).where(Job.submission_id == sub.id, Job.status == "running")
                   .values(heartbeat_at=datetime.now(timezone.utc)))
        db.commit()

    track = _Tracker(progress, flush)

    def router_factory():  # noqa: ANN202 - the router logs ai_runs through its own sessions
        return make_ai_router(settings, sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False))

    # Rubric interpretation depends only on the questions: question-only questions get
    # their AI marking scheme while the pages are still being read.
    schemes_job = _start_scheme_warmup(questions, settings, router_factory, exam_id, progress)

    ocr = get_ocr_engine(settings.ocr_engine, settings.trocr_model)
    engines_used: set[str] = set()
    page_lines: dict[int, list[OcrLineDC]] = {}
    page_widths: dict[int, int] = {}
    page_flags: dict[str, list[str]] = {}
    reader: ai_ocr.PageReader | None = None
    if ai_ocr.usable(settings) and settings.ocr_engine != "textlayer":
        def on_page(page_no: int, ok: bool) -> None:  # reader threads
            if not ok:
                progress.event(f"Page {page_no}: no vision model could read it on this attempt", "warning")

        reader = ai_ocr.PageReader(router_factory, settings, exam_id=exam_id, submission_id=submission_id,
                                   on_page=on_page)
    ai_pages: list[int] = []

    progress.start("preprocess", "Rendering pages, checking quality, straightening scans")
    progress.start("recognition", "Waiting for the first page")
    flush()
    try:
        doc = fitz.open(sub.pdf_path)
        try:
            for i in range(doc.page_count):
                page_no = i + 1
                # Deleted meanwhile: stop before writing more page images into its folder.
                if not _exists(db, sub.id):
                    raise SubmissionGone(sub.id)
                try:
                    t0 = time.perf_counter()
                    page = render_page(doc, i, settings.render_dpi)
                    quality = preprocess.assess(page.image)
                    display = page.image
                    if not page.text_lines and not quality.is_blank:
                        display, _angle = preprocess.deskew(page.image)
                    out_path = page_image_path(settings.storage_dir, sub.id, page_no)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), display)
                    page_widths[page_no] = display.shape[1]
                    timings["render_preprocess"] += time.perf_counter() - t0
                    progress.bump("pages_preprocessed")

                    if quality.is_blank and not page.text_lines:
                        page_flags[str(page_no)] = ["blank_page"]
                        progress.bump("pages_blank")
                        done_pages.add(page_no)
                        continue
                    # Image quality only matters when text is read from the image; a PDF text
                    # layer is exact however the page looks (sparse typed pages have low contrast).
                    if quality.is_poor and not page.text_lines:
                        page_flags.setdefault(str(page_no), []).append("poor_quality")

                    t1 = time.perf_counter()
                    rp = RenderedPage(page_no, display, page.text_lines)
                    if not page.text_lines and reader is not None:
                        # Scan / photo / handwriting: queued for the vision model at once, so it
                        # reads while the next pages render (the image is JPEG-encoded now).
                        reader.submit(page_no, display)
                        ai_pages.append(page_no)
                        progress.detail("recognition", f"Vision model reading {len(ai_pages)} page(s)")
                    else:
                        lines, engine = _local_read(ocr, rp, display, settings)
                        engines_used.add(engine)
                        page_lines[page_no] = lines
                        done_pages.add(page_no)
                        progress.bump("pages_text_layer" if page.text_lines else "pages_read")
                        if not page.text_lines and not any(ln.text.strip() for ln in lines):
                            page_flags.setdefault(str(page_no), []).append("page_unread")
                    timings["ocr"] += time.perf_counter() - t1
                except SubmissionGone:
                    raise
                except Exception as exc:  # per-page isolation
                    log.exception("Page %s of submission %s failed", page_no, sub.id)
                    page_flags.setdefault(str(page_no), []).append("page_failed")
                    sub.error = f"Page {page_no}: {exc.__class__.__name__}"
                    done_pages.add(page_no)
                    progress.bump("failed_operations")
                    progress.event(f"Page {page_no} could not be processed ({exc.__class__.__name__})", "error")
                finally:
                    flush()
        finally:
            doc.close()
        progress.finish("preprocess", detail=f"{progress.get('pages_preprocessed')} page(s) prepared")

        if reader is not None and ai_pages:
            t1 = time.perf_counter()
            def on_read(page_no: int) -> None:  # this thread, as each page is read
                done_pages.add(page_no)
                progress.bump("pages_read")
                progress.detail("recognition", f"{len(done_pages)} of {progress.get('pages_total')} page(s) done")

            results = reader.results(tick=flush, on_done=on_read)
            for page_no in sorted(results):
                read = results[page_no]
                if read is not None:
                    lines, engine = read
                    engines_used.add(engine)
                else:
                    # No vision model could read it: the local engines (TrOCR when installed,
                    # else line boxes for the teacher to transcribe) from the saved page image.
                    lines, engine = [], "none"
                    img = cv2.imread(str(page_image_path(settings.storage_dir, sub.id, page_no)), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        lines, engine = _local_read(ocr, RenderedPage(page_no, img, []), img, settings)
                    engines_used.add(engine)
                    if any(ln.text.strip() for ln in lines):
                        progress.bump("pages_read")
                    else:
                        page_flags.setdefault(str(page_no), []).append("page_unread")
                if read is None:
                    progress.event(f"Page {page_no}: not read by any vision model", "error")
                page_lines[page_no] = lines
                done_pages.add(page_no)
            timings["ocr"] += time.perf_counter() - t1
    finally:
        if reader is not None:
            reader.close()

    unread = sorted(int(p) for p, f in page_flags.items() if "page_unread" in f)
    failed_pages = sorted(int(p) for p, f in page_flags.items() if "page_failed" in f)
    progress.set("pages_unread", len(unread))
    readers = ", ".join(sorted(engines_used)) or "none"
    progress.finish("recognition", "warning" if unread or failed_pages else "done",
                    f"{progress.get('pages_read') + progress.get('pages_text_layer')} of "
                    f"{progress.get('pages_total') - progress.get('pages_blank')} page(s) with writing read ({readers})"
                    + (f"; page(s) {', '.join(map(str, unread))} could not be read" if unread else ""))
    all_lines = [ln for p in sorted(page_lines) for ln in page_lines[p]]
    for n, ln in enumerate(all_lines):
        db.add(OcrLine(submission_id=sub.id, page_no=ln.page_no, line_no=n, bbox=list(ln.bbox),
                       text=ln.text, confidence=float(ln.confidence)))

    # Question identification (segmentation)
    sub.stage = "segmentation"
    progress.start("identification", "Finding question numbers on the sheet")
    flush()
    t2 = time.perf_counter()
    qmap = {q.question_no: q for q in questions}
    result = segment(all_lines, list(qmap))
    if not result.segments and any(ln.text.strip() for ln in all_lines):
        # No question numbers on the sheet: one question takes everything, else the AI
        # splits the lines between the questions.
        progress.detail("identification", "No question numbers written: the AI is splitting the answers")
        flush()  # the router logs ai_runs through its own session
        alt = ai_segment.split_without_markers(
            all_lines, questions, router_factory,
            ai_enabled=E.ai_enabled(settings), exam_id=exam_id, submission_id=submission_id,
            redact_terms=[sub.student_ref] if sub.student_ref else [])
        if alt is not None:
            result = alt
    _flag_answers_cut_by_failed_pages(result, all_lines, page_flags)
    for qno, seg in result.segments.items():
        for reg in seg.regions(page_widths):
            flags = list(seg.flags) + page_flags.get(str(reg.page_no), [])
            db.add(AnswerRegion(submission_id=sub.id, page_no=reg.page_no, bbox=list(reg.bbox),
                                question_id=qmap[qno].id, source="auto",
                                segmentation_confidence=round(seg.confidence, 3), flags=flags))
    for seg in result.unassigned:
        for reg in seg.regions(page_widths):
            db.add(AnswerRegion(submission_id=sub.id, page_no=reg.page_no, bbox=list(reg.bbox), question_id=None,
                                source="auto", segmentation_confidence=0.3, flags=["unassigned"]))
    for page_key, flags in page_flags.items():
        if "page_failed" in flags or "page_unread" in flags:
            p = int(page_key)
            w = page_widths.get(p, 1000)
            has_region = any(reg.page_no == p for seg in result.segments.values() for reg in seg.regions(page_widths))
            if "page_failed" in flags or not has_region:
                db.add(AnswerRegion(submission_id=sub.id, page_no=p, bbox=[0, 0, w, int(w * 1.414)],
                                    question_id=None, source="auto", segmentation_confidence=0.0,
                                    flags=[f for f in flags if f in ("page_failed", "page_unread")]))
    sheet_flags = list(result.sheet_flags)
    if failed_pages:
        sheet_flags.append("has_failed_pages")
    if unread:
        sheet_flags.append("has_unread_pages")
    sub.flags = list(dict.fromkeys(sheet_flags))
    sub.page_flags = page_flags
    db.flush()
    db.expire(sub)
    timings["segmentation"] = time.perf_counter() - t2
    found = len(result.segments)
    progress.finish("identification", "warning" if found < len(questions) and all_lines else "done",
                    f"Answers found for {found} of {len(questions)} question(s)"
                    + (" (AI split, no numbers written)" if "ai_segmented" in result.sheet_flags else ""))

    # Answer extraction -> rubric -> analysis -> marking -> verification (compute_answers)
    sub.stage = "grading"
    flush()
    t3 = time.perf_counter()
    answers = compute_answers(db, sub, questions, settings, suspect_missing=set(result.missing_after_last),
                              commit_before_ai=True, track=track, schemes_job=schemes_job)
    timings["grading"] = time.perf_counter() - t3

    progress.start("results", "Totals, review routing, saving results")
    refresh_totals(sub)
    sub.ocr_engine = (",".join(sorted(engines_used)) or "none")[:32]
    sub.grader = _graders_label(answers)
    timings["total"] = time.perf_counter() - started
    sub.timings = {k: round(v, 3) for k, v in timings.items()}
    _ai_usage_into(progress, db, sub.id)
    progress.finish("results", detail=f"{len(answers)} question(s) marked: {sub.total_ai_marks} marks in total "
                                      "(provisional until the teacher approves)")
    progress.complete()
    sub.progress = progress.snapshot()
    sub.pages_done = sub.page_count or len(done_pages)
    sub.status, sub.stage = "processed", "done"
    db.commit()
    log_sheet_grading(sub.id, answers, timings["grading"])
    log.info("Processed submission %s in %.1fs", sub.id, timings["total"])


def _ai_usage_into(progress: Progress, db: Session, submission_id: str) -> None:
    """AI requests, cache hits, failures and tokens of this sheet (from ai_runs)."""
    try:
        rows = db.execute(
            select(AiRun.status, func.count(AiRun.id), func.coalesce(func.sum(AiRun.input_tokens), 0),
                   func.coalesce(func.sum(AiRun.output_tokens), 0))
            .where(AiRun.submission_id == submission_id).group_by(AiRun.status)).all()
    except Exception:  # noqa: BLE001 - statistics never fail a sheet
        log.debug("AI usage lookup failed", exc_info=True)
        return
    by = {r[0]: r for r in rows}
    progress.set("ai_requests", sum(int(r[1]) for s, r in by.items() if s != "cache_hit"))
    progress.set("ai_cache_hits", int(by["cache_hit"][1]) if "cache_hit" in by else 0)
    progress.set("ai_failed_requests", sum(int(r[1]) for s, r in by.items() if s not in ("ok", "cache_hit")))
    progress.set("tokens_in", sum(int(r[2] or 0) for r in by.values()))
    progress.set("tokens_out", sum(int(r[3] or 0) for r in by.values()))


def _start_scheme_warmup(questions: list[Question], settings: Settings, router_factory: Callable[[], object],
                         exam_id: str, progress: Progress) -> Future | None:
    """Derive the AI marking schemes of the question-only questions in the background
    (they depend only on the questions), so marking can start as soon as the answers are
    read. Returns a Future of {question_no: Scheme}, or None when there is nothing to do."""
    items = [_direct_item(q, "") for q in questions if direct_ai.is_direct(q)]
    if not items or not E.ai_enabled(settings) or not getattr(settings, "ai_direct_scheme", False):
        return None
    progress.start("rubric", f"AI writing the marking scheme for {len(items)} question-only question(s)")

    def work() -> dict[int, direct_ai.Scheme]:
        router = router_factory()
        try:
            if not router.usable_providers():
                return {}
            out: dict[int, direct_ai.Scheme] = {}
            with ThreadPoolExecutor(max_workers=min(3, len(items)), thread_name_prefix="saras-scheme") as pool:
                futs = {it.question_no: pool.submit(direct_ai.derive_scheme, router, it, exam_id=exam_id)
                        for it in items}
                for qno, fut in futs.items():
                    try:
                        s = fut.result()
                    except Exception:  # noqa: BLE001
                        s = None
                    if s is not None and s.criteria:
                        out[qno] = s
            return out
        finally:
            router.close()

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="saras-rubric")
    fut = pool.submit(work)
    pool.shutdown(wait=False)
    return fut


def _graders_label(answers: Iterable[Answer]) -> str | None:
    """Submission.grader (String(32)): every grader that marked an answer, sorted."""
    names: set[str] = set()
    for a in answers:
        trace = a.grading_trace if isinstance(a.grading_trace, dict) else {}
        names.update(str(n) for n in trace.get("graders_used") or [])
    return ",".join(sorted(names))[:32] or None


def log_sheet_grading(submission_id: str, answers: list[Answer], seconds: float) -> None:
    """One INFO line per sheet: graders, AI provider/model or why it fell back, counts per
    answer status and per criterion status, duration. Ids and counts only."""
    traces = [a.grading_trace for a in answers if isinstance(a.grading_trace, dict) and a.grading_trace.get("graded")]
    graders = sorted({str(n) for t in traces for n in t.get("graders_used") or []})
    ai = Counter()
    for t in traces:
        info = t.get("ai") or {}
        status = str(info.get("status") or "off")
        if status in ("used", "incomplete"):
            ai[f"{status}:{info.get('provider')}:{info.get('model')}{' (cached)' if info.get('cached') else ''}"] += 1
        elif status == "fallback":
            ai[f"fallback:{info.get('reason') or '?'}[{'; '.join(info.get('details') or [])}]"] += 1
        else:
            ai[status] += 1
    statuses = Counter(a.status for a in answers)
    kp_statuses = Counter(str(k.get("status")) for a in answers for k in (a.evidence or []) if isinstance(k, dict))
    log.info("Graded submission %s: graders=%s ai=%s answers=%s criteria=%s disputed=%d duration_ms=%d",
             submission_id, ",".join(graders) or "-",
             "; ".join(f"{k} x{v}" for k, v in sorted(ai.items())) or "-",
             dict(sorted(statuses.items())), dict(sorted(kp_statuses.items())),
             sum(1 for t in traces if "graders_disagree" in (t.get("flags") or [])), round(seconds * 1000))


def _mean_conf(lines: list[OcrLineDC]) -> float:
    return sum(ln.confidence for ln in lines) / len(lines) if lines else 0.0


def _flag_answers_cut_by_failed_pages(result, lines: list[OcrLineDC], page_flags: dict) -> None:  # noqa: ANN001
    """An answer still running when a page failed may continue on that unread page, so
    its text is truncated: flag the segment that owns the last line before each failed page."""
    failed = sorted(int(p) for p, flags in page_flags.items() if "page_failed" in flags)
    if not failed:
        return
    owner = {id(ln): seg for seg in result.segments.values() for ln in seg.lines}
    for page_no in failed:
        before = [ln for ln in lines if ln.page_no < page_no]
        seg = owner.get(id(before[-1])) if before else None
        if seg is not None and "next_page_failed" not in seg.flags:
            seg.flags.append("next_page_failed")
            seg.confidence = min(seg.confidence, 0.6)


def _inside(line_bbox: list[int], region_bbox: list[int]) -> bool:
    lx, ly, lw, lh = line_bbox
    rx, ry, rw, rh = region_bbox
    cx, cy = lx + lw / 2, ly + lh / 2
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def answer_source(sub: Submission, question: Question) -> tuple[str, float, float, list[str], bool]:
    """Collect text for a question from its regions. Returns
    (text, ocr_conf, seg_conf, region_flags, needs_transcription)."""
    regions = [r for r in sub.regions if r.question_id == question.id and not r.is_blank]
    valid = {q.question_no for q in sub.exam.questions}
    picked: list[OcrLine] = []
    seen: set[str] = set()
    for r in sorted(regions, key=lambda r: (r.page_no, r.bbox[1])):
        for ln in sub.lines:
            if ln.page_no == r.page_no and ln.id not in seen and _inside(ln.bbox, r.bbox):
                picked.append(ln)
                seen.add(ln.id)
    picked.sort(key=lambda ln: (ln.page_no, ln.bbox[1], ln.bbox[0]))
    texts = [strip_marker(ln.text, valid) for ln in picked]
    text = " ".join(t for t in texts if t).strip()
    # Confidence is weighted by text length; a line OCR could not read at all counts
    # like a typical line, not like one character, so unread lines pull it down.
    read = [len(t) for ln, t in zip(picked, texts) if ln.text.strip()]
    unread_weight = max(20, round(sum(read) / len(read))) if read else 20
    weights = [unread_weight if not ln.text.strip() else max(1, len(t)) for ln, t in zip(picked, texts)]
    ocr_conf = (sum(ln.confidence * w for ln, w in zip(picked, weights)) / sum(weights)) if picked else 0.0
    seg_conf = min((r.segmentation_confidence for r in regions), default=1.0)
    flags: list[str] = []
    for r in regions:
        flags.extend(r.flags or [])
    needs_transcription = bool(picked) and any(not ln.text.strip() for ln in picked)
    return text, ocr_conf, seg_conf, list(dict.fromkeys(flags)), needs_transcription


def _has_unassigned_text(sub: Submission) -> bool:
    """Auto-segmented text that belongs to no question (text before the first marker).
    A question without a region may well be that text under a misread marker. Once the
    teacher assigns or blanks such a region (it becomes manual) it no longer counts."""
    regions = [r for r in sub.regions if r.question_id is None and r.source == "auto" and not r.is_blank
               and "unassigned" in (r.flags or [])]
    return any(ln.page_no == r.page_no and ln.text.strip() and _inside(ln.bbox, r.bbox)
               for r in regions for ln in sub.lines)


@dataclass
class _Source:
    text: str
    ocr_conf: float
    seg_conf: float
    region_flags: list[str]
    needs_transcription: bool
    teacher_text: bool


def _source(sub: Submission, question: Question, ans: Answer | None, override_text: str | None) -> _Source:
    """The text an answer is graded on: teacher-corrected text wins over its regions' OCR."""
    text, ocr_conf, seg_conf, region_flags, needs_transcription = answer_source(sub, question)
    corrected = ans is not None and bool(ans.text_corrected)
    if override_text is not None:
        text, ocr_conf, needs_transcription = override_text.strip(), 1.0, False
    elif corrected:
        text, ocr_conf, needs_transcription = ans.ocr_text or "", 1.0, False
    return _Source(text, ocr_conf, seg_conf, region_flags, needs_transcription,
                   override_text is not None or corrected)


def _snapshot(q: Question) -> SimpleNamespace:
    """A plain copy of the rubric fields the AI marker reads: safe to use from a worker
    thread (no lazy loading through the request's / worker's session)."""
    return SimpleNamespace(id=q.id, question_no=q.question_no, type=q.type, max_marks=q.max_marks,
                           prompt=q.prompt, model_answer=q.model_answer, key_points=list(q.key_points or []))


def compute_answers(db: Session, sub: Submission, questions: Iterable[Question], settings: Settings, *,  # noqa: C901, PLR0912, PLR0915
                    suspect_missing: Collection[int] = (), commit_before_ai: bool = False,
                    track: _Tracker | None = None, schemes_job: Future | None = None) -> list[Answer]:
    """(Re)build several answers of one sheet. When AI marking is enabled the AI gets ONE
    request for all of them (split only when too big), not one per answer.

    `commit_before_ai`: commit the session before the AI request so the router can
    record its ai_runs rows in its own sessions (the worker: SQLite allows one writer at
    a time). The AI requests then run on worker threads while this thread runs the
    local checkers (network and CPU overlap). Request handlers keep their transaction
    open - their audit entries must commit together with the change - and the router
    writes through the same connection, on this thread.

    `track`: live progress (the worker); `schemes_job`: the marking schemes being derived
    in the background (process_submission)."""
    track = track or _Tracker()
    p = track.p
    questions = list(questions)
    by_question = {a.question_id: a for a in sub.answers}
    exam_id, sub_id = sub.exam_id, sub.id
    terms = [sub.student_ref] if sub.student_ref else []

    # 4. Answer extraction
    p.start("extraction", "Collecting the text of every answer from its regions")
    srcs = {q.id: _source(sub, q, by_question.get(q.id), None) for q in questions}
    pairs: list[tuple[SimpleNamespace, str]] = []
    direct_items: list[direct_ai.DirectItem] = []
    for q in questions:
        text = srcs[q.id].text
        if E.has_content(text):
            if direct_ai.is_direct(q):  # question-only rubric: the AI gives the marks itself
                direct_items.append(_direct_item(q, text))
            else:
                pairs.append((_snapshot(q), text))
    n_text = len(pairs) + len(direct_items)
    partial = sum(1 for q in questions if srcs[q.id].needs_transcription)
    p.set("answers_extracted", n_text)
    p.set("answers_blank", len(questions) - n_text)
    p.finish("extraction", "warning" if partial else "done",
             f"{n_text} answer(s) with text, {len(questions) - n_text} blank or unread"
             + (f", {partial} only partly read" if partial else ""))

    # 5. Rubric interpretation
    ai_on = E.ai_enabled(settings)
    if p.state("rubric") != "running":
        p.start("rubric")
    schemes: dict[int, direct_ai.Scheme] = {}
    if schemes_job is not None:
        try:
            schemes = schemes_job.result(timeout=max(5.0, float(settings.ai_call_deadline_seconds or 60)))
        except Exception as exc:  # noqa: BLE001 - marking derives what is missing itself
            log.warning("Marking scheme warm-up failed (%s)", type(exc).__name__)
            p.bump("failed_operations")
    kp_count = sum(1 for q in questions if not direct_ai.is_direct(q))
    parts = []
    if kp_count:
        parts.append(f"{kp_count} question(s) with the teacher's key points")
    if len(questions) - kp_count:
        parts.append(f"{len(questions) - kp_count} question-only"
                     + (f" ({len(schemes)} AI marking scheme(s) ready)" if schemes else ""))
    p.finish("rubric", detail="; ".join(parts) or "No questions")
    track.flush()

    # 6. Semantic answer analysis: AI requests (threads) || local checkers (this thread)
    parallel = commit_before_ai and ai_on and bool(pairs or direct_items)
    if commit_before_ai:
        db.commit()
        factory: Callable[[], Session] = sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False)
    else:
        factory = _joined_sessions(db) if ai_on else (lambda: Session())
    p.start("analysis", ("AI examiner judging every answer against its rubric" if ai_on and n_text
                         else "Offline checkers judging the answers") + (" while the offline checkers cross-check"
                                                                          if ai_on and pairs else ""))
    track.flush()
    local: dict[str, tuple] = {}
    outcomes: dict[int, E.AiOutcome] = {}
    direct = direct_ai.DirectResult()
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="saras-mark") if parallel else None
    try:
        f_kp = pool.submit(_ai_outcomes, factory, exam_id, sub_id, terms, pairs, settings) if pool else None
        f_direct = pool.submit(_direct_outcomes, factory, exam_id, sub_id, terms, direct_items, settings,
                               schemes) if pool else None
        for q, text in pairs:  # CPU work while the AI requests are in flight
            local[q.id] = E.grade_local(q, text, settings, primary_fn=_grade_with_fallback)
        if pool:
            outcomes, direct = f_kp.result(), f_direct.result()
        else:
            outcomes = _ai_outcomes(factory, exam_id, sub_id, terms, pairs, settings)
            direct = _direct_outcomes(factory, exam_id, sub_id, terms, direct_items, settings, schemes)
    finally:
        if pool:
            pool.shutdown(wait=True)
    ai_failed = sum(1 for o in outcomes.values() if o.info.status == "fallback") + len(direct.missing)
    p.finish("analysis", "warning" if ai_on and ai_failed else "done",
             _analysis_detail(outcomes, direct, len(pairs), ai_on, ai_failed))
    track.flush()

    # 7. Question-wise mark allocation (code computes the marks from the verdicts)
    p.start("marking", "Computing marks from the criterion verdicts")
    off = E.AiOutcome(E.AiInfo("off"))
    by_no: dict[int, Answer] = {}
    for q in questions:
        by_no[q.question_no] = compute_answer(db, sub, q, settings, suspect_missing=q.question_no in suspect_missing,
                                              ai=outcomes.get(q.question_no, off), direct=direct,
                                              local=local.get(q.id), src=srcs[q.id])
        p.bump("questions_marked")
    p.finish("marking", detail=f"{len(by_no)} question(s) marked")
    track.flush()

    # 8. Verification (selective): evidence quotes and mark limits are checked in code
    # for every answer; only doubtful AI marks get a second, more careful marking, and
    # only those answers are recomputed.
    p.start("verification", "Checking evidence, mark limits and doubtful answers")
    track.flush()
    records: dict[int, dict] = {}
    if ai_on and getattr(settings, "ai_verify", False):
        records = _verify(factory, exam_id, sub_id, terms, pairs, outcomes, direct, direct_items, local, settings)
    for q in questions:
        rec = records.get(q.question_no)
        if rec is None:
            continue
        ans = compute_answer(db, sub, q, settings, suspect_missing=q.question_no in suspect_missing,
                             ai=outcomes.get(q.question_no, off), direct=direct, local=local.get(q.id),
                             src=srcs[q.id])
        if isinstance(ans.grading_trace, dict) and isinstance(ans.grading_trace.get("ai"), dict):
            ans.grading_trace = {**ans.grading_trace, "ai": {**ans.grading_trace["ai"], "verification": rec}}
        by_no[q.question_no] = ans
    answers = [by_no[q.question_no] for q in questions]
    over = sum(1 for a, q in zip(answers, questions) if (a.ai_marks or 0) > q.max_marks + 1e-9)
    changed = sum(1 for r in records.values() if r.get("status") == "changed")
    confirmed = sum(1 for r in records.values() if r.get("status") == "confirmed")
    for a in answers:
        routing = (a.grading_trace or {}).get("routing") if isinstance(a.grading_trace, dict) else None
        if a.status == "needs_review" or (isinstance(routing, dict) and routing.get("review_waived")):
            p.bump("uncertain")
    p.set("verification_candidates", len(records))
    p.set("reevaluated", changed + confirmed)
    if over:
        p.bump("failed_operations", over)
    if not ai_on:
        detail = "Evidence and mark limits checked (AI marking is off)"
    elif records:
        detail = f"{len(records)} doubtful answer(s) re-marked: {confirmed} confirmed, {changed} changed"
    else:
        detail = "All AI marks backed by verified evidence: no second marking needed"
    p.finish("verification", "warning" if changed or over else "done",
             detail + f"; {p.get('uncertain')} answer(s) flagged for the teacher")
    return answers


def _analysis_detail(outcomes: dict[int, E.AiOutcome], direct: direct_ai.DirectResult, n_pairs: int,
                     ai_on: bool, ai_failed: int) -> str:
    if not ai_on:
        return f"{n_pairs} answer(s) judged by the offline checkers (AI marking is off)"
    used = [o for o in outcomes.values() if o.info.status == "used" and o.grade is not None]
    models = {f"{o.grade.provider}:{o.grade.model}" for o in used}
    models |= {f"{g.provider}:{g.model}" for g in direct.grades.values()}
    n = len(used) + len(direct.grades)
    text = f"{n} answer(s) judged by the AI examiner" + (f" ({', '.join(sorted(models))})" if models else "")
    if ai_failed:
        text += f"; {ai_failed} fell back (offline checker or teacher)"
    return text


def _verify(factory: Callable[[], Session], exam_id: str, sub_id: str, terms: list[str],
            pairs: list[tuple[SimpleNamespace, str]], outcomes: dict[int, E.AiOutcome],
            direct: direct_ai.DirectResult, direct_items: list[direct_ai.DirectItem], local: dict[str, tuple],
            settings: Settings) -> dict[int, dict]:
    """The selective second marking for both paths; {question_no: record}. Never raises."""
    from ..grading import llm_grader as L

    grades = L.SheetGrades()
    for qno, o in outcomes.items():
        if o.grade is not None:
            grades[qno] = o.grade
    local_marks = {}
    for q, _text in pairs:
        res = local.get(q.id)
        if res is not None and getattr(res[0], "marks", None) is not None:
            local_marks[q.question_no] = float(res[0].marks)
    needs_kp = any(L.verification_reasons(g, local_marks.get(n)) for n, g in grades.items())
    needs_direct = any(g.llm is not None and L.verification_reasons(g.llm) for g in direct.grades.values())
    if not (needs_kp or needs_direct):
        return {}
    records: dict[int, dict] = {}
    try:
        router = make_ai_router(settings, factory)
    except Exception as exc:  # noqa: BLE001
        log.warning("Verification could not start (%s)", type(exc).__name__)
        return {}
    try:
        style = E.ai_style(settings)
        if needs_kp:
            items = [L.AnswerItem.from_question(q, t) for q, t in pairs if q.question_no in grades]
            grades, recs = L.verify_sheet(router, items, grades, style, local_marks=local_marks, exam_id=exam_id,
                                          submission_id=sub_id, redact_terms=terms)
            for qno, g in grades.items():
                if qno in outcomes:
                    outcomes[qno].grade = g
            records.update(recs)
        if needs_direct:
            records.update(direct_ai.verify_scheme(router, direct_items, direct, style, exam_id=exam_id,
                                                   submission_id=sub_id, redact_terms=terms))
    except Exception as exc:  # noqa: BLE001 - verification never fails a sheet
        log.warning("Verification failed (%s)", type(exc).__name__)
    finally:
        router.close()
    return records
def compute_answer(db: Session, sub: Submission, question: Question, settings: Settings,  # noqa: C901, PLR0912, PLR0915
                   override_text: str | None = None, suspect_missing: bool = False, *,
                   ai: E.AiOutcome | None = None, direct: direct_ai.DirectResult | None = None,
                   local: tuple | None = None, src: _Source | None = None) -> Answer:
    """(Re)build one answer from its regions or from teacher-corrected text.

    `suspect_missing`: segmentation found no marker for this question after the last
    marker it did find, so a misread marker cannot be ruled out (review, never a silent 0).
    `ai`: the AI marker's outcome when the caller already asked for the whole sheet
    (compute_answers); None asks the AI now for this one answer, when AI marking is on.
    `local`: the offline checkers' results already computed for this text (compute_answers);
    `src`: the answer's text already collected from its regions (compute_answers)."""
    ans = next((a for a in sub.answers if a.question_id == question.id), None)
    if ans is None:
        ans = Answer(submission_id=sub.id, question_id=question.id)
        db.add(ans)
        sub.answers.append(ans)
    teacher_flagged = "teacher_flag" in (ans.flags or [])
    # An earlier run already doubted a missing marker: keep doubting it on regrade.
    was_suspect = {"missing_marker", "segmentation_warning"} <= set(ans.flags or [])

    if src is None or override_text is not None:
        src = _source(sub, question, ans, override_text)
    text, seg_conf, needs_transcription = src.text, src.seg_conf, src.needs_transcription
    terms = [sub.student_ref] if sub.student_ref else []

    has_region = any(r.question_id == question.id for r in sub.regions)
    flags = list(src.region_flags)
    if any(f in SEGMENTATION_REVIEW_FLAGS for f in flags):
        flags.append("segmentation_warning")

    graded = False
    ans.ocr_text = text
    ans.ocr_confidence = round(min(1.0, max(0.0, src.ocr_conf)), 4)
    ans.segmentation_confidence = round(seg_conf, 4)

    if needs_transcription and not text:
        ans.ai_marks, ans.grading_confidence, ans.confidence, ans.evidence = 0.0, 0.0, 0.0, []
        ans.flags = list(dict.fromkeys(flags + ["needs_transcription"]))
        ans.grader = None
        _not_graded(ans, "needs_transcription")
        new_status = "needs_review"
    elif not E.has_content(text):
        # Skipped or unattempted (not a single letter or digit): 0 marks, never sent to
        # the grader (PRD 12.10). A short real answer ("42", "Fe") is graded below.
        ans.ai_marks, ans.grading_confidence, ans.confidence, ans.evidence = 0.0, 1.0, 1.0, []
        ans.grader = None
        # No region is only trustworthy as "skipped" when every page was read and no
        # text is left over: with no markers found, a failed page, text before the first
        # marker, or nothing found after the last marker, the answer may be unread.
        doubtful = not has_region and not src.teacher_text and (
            bool({"no_markers_found", "has_failed_pages", "has_unread_pages"} & set(sub.flags or []))
            or suspect_missing or was_suspect or _has_unassigned_text(sub))
        if doubtful:
            ans.flags = list(dict.fromkeys(flags + ["missing_marker", "segmentation_warning"]))
            _not_graded(ans, "possibly_unread")
            new_status = "needs_review"
        else:
            ans.flags = list(dict.fromkeys(flags + (["missing_marker"] if not has_region else [])))
            _not_graded(ans, "blank")
            new_status = "blank"
    elif direct_ai.is_direct(question):
        if direct is None or (question.question_no not in direct.grades
                              and question.question_no not in direct.missing):
            direct = _direct_outcomes(lambda: _joined_sessions(db)(), sub.exam_id, sub.id, terms,
                                      [_direct_item(question, text)], settings,
                                      verify=getattr(settings, "ai_verify", False))
        new_status = _apply_direct(ans, question, text, flags, needs_transcription, direct, settings)
        graded = question.question_no in direct.grades
    else:
        if ai is None:
            ai = _ai_outcomes(lambda: _joined_sessions(db)(), sub.exam_id, sub.id, terms,
                              [(_snapshot(question), text)], settings).get(question.question_no)
        ai = ai or E.AiOutcome(E.AiInfo("off"))
        result = E.grade_ensemble(question, text, settings, llm_grade=ai.grade, ai=ai.info,
                                  primary_fn=_grade_with_fallback, local_results=local)
        ans.ai_marks = result.marks
        ans.grading_confidence = result.grading_confidence
        ans.evidence = [kp.as_dict() for kp in result.key_points]
        ans.grader = result.grader
        ans.verdict = result.verdict
        ans.feedback = result.feedback or None
        ans.confidence = combine(ans.ocr_confidence, result.grading_confidence, seg_conf,
                                 settings.calibration_temperature)
        extra: list[str] = []
        if needs_transcription:  # part of the answer could not be read: graded on partial text
            extra.append("needs_transcription")
        if len(text) < 3:  # very short answers are easy to misread and to mis-grade
            extra.append("short_answer")
        new_status, ans.flags = route(ans.confidence, ans.ocr_confidence, flags + result.flags + extra,
                                      auto_accept=settings.auto_accept_threshold,
                                      low_ocr=settings.low_ocr_confidence)
        # Disagreeing graders, contradictions, unverifiable quotes and the AI's own doubts
        # always need a teacher, whatever the routing flag lists say.
        forced = new_status == "auto_accepted" and bool(E.FORCE_REVIEW_FLAGS & set(ans.flags))
        if forced:
            new_status = "needs_review"
        graded = True
        review_flags = [f for f in ans.flags if is_review_flag(f) or f in E.FORCE_REVIEW_FLAGS]
        ans.grading_trace = {**result.trace, "routing": {
            "status": new_status, "confidence": ans.confidence, "ocr_confidence": ans.ocr_confidence,
            "segmentation_confidence": ans.segmentation_confidence,
            "auto_accept_threshold": settings.auto_accept_threshold, "review_flags": review_flags,
            "forced_review": forced}}
        if log.isEnabledFor(logging.DEBUG):
            log.debug("answer submission=%s question=%s answer=%s status=%s marks=%.2f/%.2f verdict=%s "
                      "confidence=%.3f grading_confidence=%.3f graders=%s ai=%s kps=%s flags=%s route=%s",
                      sub.id, question.question_no, ans.id, new_status, ans.ai_marks, question.max_marks,
                      ans.verdict, ans.confidence, ans.grading_confidence, ans.grader,
                      result.trace.get("ai", {}).get("status"),
                      " ".join(f"{k.index}:{k.status}{'!' if k.disputed else ''}" for k in result.key_points) or "-",
                      ",".join(ans.flags) or "-", ",".join(review_flags) or "confident")

    # Review not required: a marked answer's marks go straight to the results. Answers
    # nobody could mark (unread text, no AI reply on a question-only rubric) still wait.
    incomplete = bool(INCOMPLETE_TEXT_FLAGS & set(ans.flags or []))
    if not settings.review_required and new_status == "needs_review" and graded and not incomplete:
        new_status = "auto_accepted"
        if isinstance(ans.grading_trace, dict) and isinstance(ans.grading_trace.get("routing"), dict):
            ans.grading_trace = {**ans.grading_trace,
                                 "routing": {**ans.grading_trace["routing"], "status": new_status,
                                             "review_waived": True}}
    # A teacher's review stands; the AI suggestion underneath it is refreshed.
    if ans.status != "reviewed" or ans.final_marks is None:
        ans.status = new_status
    elif ans.final_marks > question.max_marks + 1e-9:
        # The rubric's maximum was lowered below the reviewed mark: it must be reviewed again.
        ans.status = "needs_review"
        ans.flags = list(dict.fromkeys((ans.flags or []) + ["marks_above_max"]))
    # A teacher's "look at this again" flag survives regrading until they review it.
    if teacher_flagged and ans.status != "reviewed":
        ans.flags = list(dict.fromkeys((ans.flags or []) + ["teacher_flag"]))
        ans.status = "needs_review"
    return ans


def _direct_item(question: Question, text: str) -> direct_ai.DirectItem:
    return direct_ai.DirectItem(question.question_no, question.prompt or "", question.type,
                                float(question.max_marks), text)


def _direct_outcomes(factory: Callable[[], Session], exam_id: str, sub_id: str, terms: list[str],
                     items: list[direct_ai.DirectItem], settings: Settings,
                     schemes: dict[int, direct_ai.Scheme] | None = None, verify: bool = False) -> direct_ai.DirectResult:
    """AI marks for question-only answers: the scheme-based marking (v2) when enabled,
    else the one-request direct marking (v1). Never raises. Safe on a worker thread:
    it touches no ORM object, only `factory` sessions."""
    if not items:
        return direct_ai.DirectResult()
    if not E.ai_enabled(settings):
        return direct_ai.DirectResult(missing={it.question_no: "ai_off" for it in items})
    try:
        router = make_ai_router(settings, factory)
    except Exception as exc:  # noqa: BLE001
        log.warning("Direct AI marking for submission %s could not start (%s)", sub_id, type(exc).__name__)
        return direct_ai.DirectResult(missing={it.question_no: "error" for it in items})
    try:
        if getattr(settings, "ai_direct_scheme", False):
            return direct_ai.grade_sheet_scheme(router, items, E.ai_style(settings), exam_id=exam_id,
                                                submission_id=sub_id, redact_terms=terms, verify=verify,
                                                schemes=schemes)
        return direct_ai.grade_sheet_direct(router, items, E.ai_style(settings), exam_id=exam_id,
                                            submission_id=sub_id, redact_terms=terms)
    except Exception as exc:  # noqa: BLE001 - never in the critical path
        log.warning("Direct AI marking for submission %s failed (%s)", sub_id, type(exc).__name__)
        return direct_ai.DirectResult(missing={it.question_no: "error" for it in items})
    finally:
        router.close()


# AI-side signals that the marks of a question-only answer are doubtful.
DIRECT_DOUBT_FLAGS = frozenset({"ai_unverified_evidence", "ai_contradiction", "ai_second_opinion_differs",
                                "ai_weak_evidence"})


def _apply_direct(ans: Answer, question: Question, text: str, flags: list[str], needs_transcription: bool,
                  direct: direct_ai.DirectResult, settings: Settings) -> str:
    """Fill a question-only answer from the direct AI marks; returns its routing status."""
    qno = question.question_no
    g = direct.grades.get(qno)
    out = list(flags)
    evidence: list[dict] = []
    if g is not None:
        ans.ai_marks, ans.verdict, ans.feedback = g.marks, g.verdict, g.feedback or None
        ans.grader = direct_ai.GRADER_NAME_V2 if g.scheme is not None else direct_ai.GRADER_NAME
        doubtful = g.uncertain or bool(DIRECT_DOUBT_FLAGS & set(g.flags))
        ans.grading_confidence = 0.6 if doubtful else (0.95 if "ai_verified" in g.flags else 0.9)
        if g.uncertain and "ai_uncertain" not in g.flags:
            out.append("ai_uncertain")
        out.extend(g.flags)
        evidence = [{**kp, "source": "ai"} for kp in g.key_points]
        ai_trace = g.as_trace()
    else:
        reason = direct.missing.get(qno, "unavailable")
        ans.ai_marks, ans.verdict, ans.feedback, ans.grader, ans.grading_confidence = 0.0, None, None, None, 0.0
        out.append("ai_off" if reason == "ai_off" else "ai_unavailable")
        ai_trace = {"status": "off" if reason == "ai_off" else "fallback", "mode": "direct", "reason": reason,
                    "details": list(direct.details)}
    ans.evidence = evidence
    if needs_transcription:
        out.append("needs_transcription")
    if len(text) < 3:
        out.append("short_answer")
    ans.confidence = combine(ans.ocr_confidence, ans.grading_confidence, ans.segmentation_confidence,
                             settings.calibration_temperature)
    status, ans.flags = route(ans.confidence, ans.ocr_confidence, list(dict.fromkeys(out)),
                              auto_accept=settings.auto_accept_threshold, low_ocr=settings.low_ocr_confidence)
    if g is None:
        status = "needs_review"  # nothing marked it: never a silent 0
    review_flags = [f for f in ans.flags if is_review_flag(f)]
    ans.grading_trace = {
        "version": TRACE_VERSION, "graded": g is not None, "mode": "direct_ai",
        "graders_used": [ans.grader] if g is not None else [],
        "marks": ans.ai_marks, "max_marks": question.max_marks, "verdict": ans.verdict,
        "grading_confidence": ans.grading_confidence, "feedback_source": "ai" if g is not None else None,
        "flags": list(ans.flags), "key_points": evidence, "ai": ai_trace,
        "routing": {"status": status, "confidence": ans.confidence, "ocr_confidence": ans.ocr_confidence,
                    "segmentation_confidence": ans.segmentation_confidence,
                    "auto_accept_threshold": settings.auto_accept_threshold, "review_flags": review_flags,
                    "forced_review": False},
    }
    return status


def _not_graded(ans: Answer, reason: str) -> None:
    ans.verdict, ans.feedback = None, None
    ans.grading_trace = {"version": TRACE_VERSION, "graded": False, "reason": reason}


def _grade_with_fallback(question: Question, text: str, settings: Settings) -> GradeResult:
    """The local primary grader's result (retries, then the lexical fallback); see
    ensemble.grade_local_primary. A module function so tests can intercept it."""
    return E.grade_local_primary(question, text, settings)


# ---------------------------------------------------------------- AI marker
def make_ai_router(settings: Settings, session_factory: Callable[[], Session]):  # noqa: ANN201
    """The AI router for one sheet (or one answer). Tests replace this function to plug
    in a fake provider transport; nothing else in the pipeline builds routers."""
    from ..llm.router import LLMRouter

    return LLMRouter(settings, session_factory)


def _joined_sessions(db: Session) -> Callable[[], Session]:
    """Sessions that write through `db`'s own connection and transaction: the router's
    ai_runs rows commit (or roll back) with the request, and never wait for the write
    lock `db` may already hold (SQLite has a single writer)."""
    conn = db.connection()
    return lambda: Session(bind=conn, join_transaction_mode="rollback_only", autoflush=False,
                           expire_on_commit=False)


def _ai_outcomes(factory: Callable[[], Session], exam_id: str, sub_id: str, terms: list[str],
                 pairs: list[tuple[object, str]], settings: Settings) -> dict[int, E.AiOutcome]:
    """The AI marker's outcome per question_no for these (question, text) pairs; {} when
    AI marking is off. Never raises: any failure is a fallback to the offline checker.
    Safe on a worker thread: it touches no ORM object, only `factory` sessions."""
    if not pairs or not E.ai_enabled(settings):
        return {}
    style = E.ai_style(settings)
    if not any(E.ai_applicable(q, t) for q, t in pairs):
        return {q.question_no: E.AiOutcome(E.AiInfo("not_applicable", style=style)) for q, _ in pairs}
    t0 = time.perf_counter()
    try:
        router = make_ai_router(settings, factory)
    except Exception as exc:  # noqa: BLE001 - the AI is never in the critical path
        log.warning("AI marking for submission %s could not start (%s); using the offline checker",
                    sub_id, type(exc).__name__)
        return {q.question_no: E.AiOutcome(E.AiInfo("fallback", "error", style=style)
                                           if E.ai_applicable(q, t) else E.AiInfo("not_applicable", style=style))
                for q, t in pairs}
    try:
        outcomes = E.ai_grade_items(router, pairs, settings, exam_id=exam_id, submission_id=sub_id,
                                    redact_terms=terms)
    finally:
        router.close()
    log.debug("AI marking for submission %s: %s in %d ms", sub_id,
              dict(Counter(o.info.status for o in outcomes.values())), round((time.perf_counter() - t0) * 1000))
    return outcomes


def effective_marks(ans: Answer) -> float:
    return ans.final_marks if ans.final_marks is not None else ans.ai_marks


def refresh_totals(sub: Submission) -> None:
    sub.total_ai_marks = round(sum(a.ai_marks for a in sub.answers), 2)
    sub.total_final_marks = round(sum(effective_marks(a) for a in sub.answers), 2)
