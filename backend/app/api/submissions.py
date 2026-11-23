"""Answer-sheet uploads, processing status, review/approval, page images, answer
regions (with undo), the audit log and capabilities (PRD 16.1).

Every mutation uses write ownership (admins can read but get 403 on edits) and,
where marks or regions change, refuses locked (approved) or still-processing
sheets. Every mark/text/region change is written to the append-only audit log in
the same transaction as the change itself.
"""

from __future__ import annotations

import gc
import logging
import re
import shutil
import struct
import unicodedata
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Literal

import cv2
import pymupdf as fitz  # the bare `fitz` alias is deprecated in PyMuPDF 2.x
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, Query, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..audit import audit
from ..config import Settings, get_settings
from ..db import get_db
from ..errors import ApiError, conflict, not_found, validation
from ..grading.scorers import grader_capabilities
from ..models import (
    SUBMISSION_STATUSES,
    Answer,
    AnswerRegion,
    AuditLog,
    Correction,
    Job,
    Question,
    Review,
    Submission,
    User,
)
from ..pipeline.ocr import ocr_capabilities
from ..pipeline.pdf import safe_dpi, validate_pdf
from ..pipeline.processor import compute_answer, compute_answers, effective_marks, page_image_path, refresh_totals
from ..schemas import (
    AnswerOut,
    AuditOut,
    FlagIn,
    KeyPointOut,
    Page,
    RegionCreate,
    RegionEditOut,
    RegionOut,
    RegionPatch,
    ReopenIn,
    ReviewIn,
    StatusOut,
    SubmissionOut,
    UploadOut,
)
from ..security import current_user
from ..worker import enqueue
from .deps import ensure_unlocked, owned_answer, owned_exam, owned_region, owned_submission

log = logging.getLogger(__name__)

router = APIRouter(tags=["submissions"])

CHUNK = 1024 * 1024  # uploads are copied to disk 1 MB at a time
MAX_FILES = 50
THUMB_WIDTH = 240
FLAGGED = ("needs_review", "failed")  # answers that block approval
REVIEWABLE = ("processed", "in_review")
# Segmentation doubts a teacher resolves by editing a region by hand.
MANUAL_CLEARS = frozenset({"unassigned", "duplicate_marker", "neighbour_missing", "out_of_order"})

AnswerFilter = Literal["all", "needs_review", "auto_accepted", "reviewed", "blank", "failed"]
AnswerSort = Literal["priority", "question"]
AuditEntity = Literal["answer", "submission", "exam", "region"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _r2(x: float | None) -> float | None:
    return None if x is None else round(float(x), 2)


def _page(db: Session, stmt, page: int, page_size: int) -> tuple[list, int]:  # noqa: ANN001
    total = db.scalar(select(func.count()).select_from(stmt.order_by(None).subquery())) or 0
    rows = db.scalars(stmt.offset((page - 1) * page_size).limit(page_size)).all()
    return list(rows), total


# ---------------------------------------------------------------- files
def submission_dir(storage: Path, submission_id: str) -> Path:
    return storage / "submissions" / submission_id


def delete_submission_files(sub: Submission) -> None:
    """Remove a submission's stored PDF and page images. Only ever deletes inside
    <storage_dir>/submissions, whatever pdf_path says."""
    storage = get_settings().storage_dir
    root = (storage / "submissions").resolve()
    targets = {submission_dir(storage, sub.id)}
    if sub.pdf_path:
        targets.add(Path(sub.pdf_path).parent)
    for folder in targets:
        try:
            folder = folder.resolve()
        except OSError:
            continue
        if folder == root or root not in folder.parents or not folder.exists():
            continue
        shutil.rmtree(folder, ignore_errors=True)
        if folder.exists():  # Windows: a file may still be open (e.g. by the worker)
            log.warning("Could not fully delete %s", folder)


def _remove_dir(folder: Path) -> None:
    shutil.rmtree(folder, ignore_errors=True)
    if folder.exists():  # a PDF handle may linger until its owner is collected
        gc.collect()
        shutil.rmtree(folder, ignore_errors=True)
    if folder.exists():
        log.warning("Could not remove %s", folder)


def _stream_to_disk(upload: UploadFile, dest: Path, limit: int) -> int:
    """Copy an upload to disk in 1 MB chunks, enforcing the size limit while
    copying so an oversized file never lands on disk in full."""
    if upload.size is not None and upload.size > limit:
        raise ApiError(413, "file_too_large", f"The file is larger than {limit // (1024 * 1024)} MB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = upload.file
    try:
        src.seek(0)
    except (AttributeError, OSError):
        pass
    size = 0
    with open(dest, "wb") as out:
        while True:
            chunk = src.read(CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ApiError(413, "file_too_large", f"The file is larger than {limit // (1024 * 1024)} MB")
            out.write(chunk)
    return size


def _strip_control(s: str) -> str:
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("C"))


def _display_name(filename: str | None) -> str:
    name = PureWindowsPath(filename or "").name  # handles both / and \ separators
    return _strip_control(name).strip()[:255]


FORMULA_START = "=+-@"  # a spreadsheet treats a cell starting with these as a formula


def _clean_ref(raw: str | None) -> str:
    ref = re.sub(r"\s+", " ", _strip_control(raw or "")).strip()
    # Never start with a formula character, however they are mixed with spaces ("= =1+1").
    while ref and (ref[0] in FORMULA_START or ref[0].isspace()):
        ref = ref[1:]
    return ref[:64].strip()


def _ref_from_filename(filename: str) -> str:
    stem = PureWindowsPath(filename).stem if filename else ""
    stem = re.sub(r"[^\w.\- ]+", "_", _strip_control(stem))
    return _clean_ref(stem.strip(" ._-"))


# ---------------------------------------------------------------- output builders
def _needing_review_counts(db: Session, ids: list[str]) -> dict[str, int]:
    if not ids:
        return {}
    rows = db.execute(
        select(Answer.submission_id, func.count(Answer.id))
        .where(Answer.submission_id.in_(ids), Answer.status.in_(FLAGGED))
        .group_by(Answer.submission_id)
    ).all()
    return {sid: n for sid, n in rows}


def submission_out(db: Session, sub: Submission, needing: int | None = None) -> SubmissionOut:
    out = SubmissionOut.model_validate(sub)
    out.flags = list(sub.flags or [])
    out.page_flags = dict(sub.page_flags or {})
    out.timings = dict(sub.timings or {})
    out.answers_needing_review = needing if needing is not None else _needing_review_counts(db, [sub.id]).get(sub.id, 0)
    return out


def _region_key(r: AnswerRegion) -> tuple:
    bbox = r.bbox or [0, 0, 0, 0]
    return r.page_no, bbox[1], bbox[0], r.id


def _sorted_regions(sub: Submission) -> list[AnswerRegion]:
    return sorted(sub.regions, key=_region_key)


def _region_ids_by_question(sub: Submission) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in _sorted_regions(sub):
        if r.question_id:
            out.setdefault(r.question_id, []).append(r.id)
    return out


def _kp_out(i: int, kp: dict) -> KeyPointOut:
    status = str(kp.get("status") or ("matched" if kp.get("matched") else "missed"))
    by_grader = kp.get("statuses_by_grader")
    return KeyPointOut(
        index=int(kp.get("index", i)), text=str(kp.get("text", "")), weight=float(kp.get("weight", 0.0)),
        matched=bool(kp.get("matched", status == "matched")), status=status,
        awarded=float(kp.get("awarded", 0.0)), entailment_prob=float(kp.get("entailment_prob", 0.0)),
        contradiction_prob=float(kp.get("contradiction_prob", 0.0)), evidence_sentence=kp.get("evidence_sentence"),
        reason=str(kp.get("reason") or ""), source=kp.get("source") or None, disputed=bool(kp.get("disputed", False)),
        statuses_by_grader={str(k): str(v) for k, v in by_grader.items()} if isinstance(by_grader, dict) else {},
    )


def answer_out(ans: Answer, sub: Submission | None = None,
               region_ids: dict[str, list[str]] | None = None) -> AnswerOut:
    sub = sub or ans.submission
    q = ans.question
    if region_ids is None:
        region_ids = _region_ids_by_question(sub)
    kps = [_kp_out(i, kp) for i, kp in enumerate(ans.evidence or []) if isinstance(kp, dict)]
    return AnswerOut(
        answer_id=ans.id, submission_id=ans.submission_id, question_id=ans.question_id,
        question_no=q.question_no, question_type=q.type, question_prompt=q.prompt, status=ans.status,
        ocr_text=ans.ocr_text or "", text_corrected=bool(ans.text_corrected),
        ocr_confidence=round(ans.ocr_confidence or 0.0, 4), ai_marks=_r2(ans.ai_marks) or 0.0,
        final_marks=_r2(ans.final_marks), max_marks=q.max_marks, confidence=round(ans.confidence or 0.0, 4),
        grading_confidence=round(ans.grading_confidence or 0.0, 4),
        segmentation_confidence=round(ans.segmentation_confidence or 0.0, 4), grader=ans.grader,
        key_points=kps,
        matched_points=[k.text for k in kps if k.status == "matched"],
        missed_points=[k.text for k in kps if k.status in ("missed", "contradicted")],
        flags=list(ans.flags or []), comment=ans.comment, region_ids=region_ids.get(q.id, []),
        locked=sub.status == "approved",
        verdict=ans.verdict, feedback=ans.feedback,
        grading_trace=ans.grading_trace if isinstance(ans.grading_trace, dict) else None,
    )


def _answer_state(ans: Answer) -> dict:
    return {
        "status": ans.status, "ai_marks": _r2(ans.ai_marks), "final_marks": _r2(ans.final_marks),
        "ocr_text": ans.ocr_text, "text_corrected": bool(ans.text_corrected), "comment": ans.comment,
    }


def audit_answer_changes(db: Session, user_id: str, before: dict[str, dict], answers: list[Answer],
                         action: str, reason: str | None = None) -> int:
    """One `answer` audit entry (old/new) per recomputed answer whose status or marks
    changed, e.g. after a regrade or a region edit (PRD 15.1). `before` maps
    question_id -> _answer_state() taken before recomputing. Returns the entry count."""
    if any(a.id is None for a in answers):
        db.flush()  # answers created just now need their ids
    n = 0
    for ans in answers:
        old, new = before.get(ans.question_id), _answer_state(ans)
        if old is not None and all(old[k] == new[k] for k in ("status", "ai_marks", "final_marks")):
            continue
        audit(db, user_id=user_id, entity="answer", entity_id=ans.id, action=action, old=old, new=new,
              reason=reason)
        n += 1
    return n


def _pending(sub: Submission) -> list[int]:
    """Questions that block approval: flagged answers, and marks above the question's
    maximum (possible when the rubric's maximum was lowered after review)."""
    return sorted(a.question.question_no for a in sub.answers
                  if a.status in FLAGGED or effective_marks(a) > a.question.max_marks + 1e-9)


# ---------------------------------------------------------------- uploads
@router.post("/exams/{exam_id}/submissions", response_model=UploadOut, status_code=202)
def upload_submissions(
    exam_id: str,
    files: list[UploadFile] = File(..., description="Scanned answer-sheet PDFs (1-50 per request)"),
    student_refs: list[str] | None = Form(
        None, description="One per file, same order; blank = derived from the file name"),
    idempotency_key: str | None = Header(None, max_length=128),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> UploadOut:
    """Accept answer sheets for background processing (202; poll /status every
    2-5 s). Each file is validated on its own, so one bad file never fails the
    batch; rejected files are listed in `rejected`."""
    s = get_settings()
    exam = owned_exam(db, exam_id, user, write=True)

    key = (idempotency_key or "").strip() or None
    if key:
        prior = db.scalars(
            select(Submission).where(Submission.exam_id == exam.id, Submission.idempotency_key == key)
            .order_by(Submission.created_at, Submission.student_ref)
        ).all()
        if prior:  # a retried request: return what it created the first time
            counts = _needing_review_counts(db, [p.id for p in prior])
            return UploadOut(submissions=[submission_out(db, p, counts.get(p.id, 0)) for p in prior], rejected=[])

    if not exam.questions:
        raise conflict("Add at least one question to the rubric before uploading answer sheets")
    if not files:
        raise validation("Choose at least one PDF", [{"field": "files", "message": "at least one file is required"}])
    if len(files) > MAX_FILES:
        raise validation(f"Upload at most {MAX_FILES} files at a time",
                         [{"field": "files", "message": f"{len(files)} files sent; the limit is {MAX_FILES}"}])
    refs = list(student_refs or [])
    if len(refs) > len(files):
        raise validation("More student references than files",
                         [{"field": "student_refs", "message": f"{len(refs)} refs for {len(files)} files"}])
    refs += [""] * (len(files) - len(refs))

    taken = set(db.scalars(select(Submission.student_ref).where(Submission.exam_id == exam.id)).all())
    created: list[Submission] = []
    created_dirs: list[Path] = []
    rejected: list[dict] = []
    first_status: int | None = None

    for idx, (upload, raw_ref) in enumerate(zip(files, refs), start=1):
        filename = _display_name(upload.filename) or f"file-{idx}.pdf"
        ref = _clean_ref(raw_ref) or _ref_from_filename(filename) or f"sheet-{idx}"

        def reject(status: int, code: str, message: str) -> None:
            nonlocal first_status
            rejected.append({"filename": filename, "student_ref": ref, "code": code, "message": message})
            first_status = first_status or status

        if ref in taken:
            reject(409, "conflict", f"A sheet for '{ref}' already exists in this exam")
            continue

        sub_id = str(uuid.uuid4())
        folder = submission_dir(s.storage_dir, sub_id)
        dest = folder / "original.pdf"  # random UUID folder; the client's file name is never used on disk
        problem: tuple[int, str, str] | None = None
        try:
            _stream_to_disk(upload, dest, s.max_upload_bytes)
            pages = validate_pdf(dest, s.max_pages, s.render_dpi)
        except ApiError as exc:
            problem = (exc.status, exc.code, exc.message)
        except Exception:  # per-file isolation: never fail the whole batch
            log.exception("Storing upload %s failed", filename)
            problem = (500, "upload_failed", "The file could not be stored; please try again")
        if problem is not None:
            # Cleaned up outside the except block: the exception chain can keep the
            # PDF library's handle on the file alive, which blocks deletion on Windows.
            _remove_dir(folder)
            reject(*problem)
            continue

        sub = Submission(
            id=sub_id, exam_id=exam.id, student_ref=ref, original_filename=filename, pdf_path=str(dest),
            page_count=pages, pages_done=0, status="uploaded", stage="queued", flags=[], page_flags={},
            timings={}, idempotency_key=key,
        )
        db.add(sub)
        taken.add(ref)
        created.append(sub)
        created_dirs.append(folder)

    if not created:
        first = rejected[0]
        raise ApiError(first_status or 400, first["code"], first["message"], rejected)

    try:
        # Insert the submissions before their jobs: Job has a foreign key to
        # submissions but no ORM relationship, so the unit of work cannot order them.
        db.flush()
        for sub in created:
            enqueue(db, sub.id)
            audit(db, user_id=user.id, entity="submission", entity_id=sub.id, action="upload",
                  new={"student_ref": sub.student_ref, "filename": sub.original_filename, "pages": sub.page_count})
        db.commit()
    except IntegrityError as exc:  # a concurrent upload took one of these student refs
        log.warning("Upload commit failed: %s", exc.orig)
        db.rollback()
        for folder in created_dirs:
            _remove_dir(folder)
        raise conflict("Another upload for the same students finished first; please retry") from exc
    except Exception:
        db.rollback()
        for folder in created_dirs:
            _remove_dir(folder)
        raise
    return UploadOut(submissions=[submission_out(db, x, 0) for x in created], rejected=rejected)


# ---------------------------------------------------------------- submissions
@router.get("/exams/{exam_id}/submissions", response_model=Page)
def list_submissions(
    exam_id: str,
    status: str | None = Query(None, description="Filter by submission status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict:
    exam = owned_exam(db, exam_id, user)
    stmt = select(Submission).where(Submission.exam_id == exam.id)
    if status:
        if status not in SUBMISSION_STATUSES:
            raise validation("Unknown status filter",
                             [{"field": "status", "message": f"use one of {', '.join(SUBMISSION_STATUSES)}"}])
        stmt = stmt.where(Submission.status == status)
    subs, total = _page(db, stmt.order_by(Submission.student_ref, Submission.created_at), page, page_size)
    counts = _needing_review_counts(db, [x.id for x in subs])
    return {"items": [submission_out(db, x, counts.get(x.id, 0)) for x in subs],
            "total": total, "page": page, "page_size": page_size}


@router.get("/submissions/{submission_id}", response_model=SubmissionOut)
def get_submission(submission_id: str, db: Session = Depends(get_db),
                   user: User = Depends(current_user)) -> SubmissionOut:
    return submission_out(db, owned_submission(db, submission_id, user))


@router.get("/submissions/{submission_id}/status", response_model=StatusOut)
def submission_status(submission_id: str, db: Session = Depends(get_db),
                      user: User = Depends(current_user)) -> StatusOut:
    sub = owned_submission(db, submission_id, user)
    job = db.scalars(
        select(Job).where(Job.submission_id == sub.id).order_by(Job.created_at.desc()).limit(1)
    ).first()
    errors: list[str] = []
    if sub.error:
        errors.append(sub.error)
    if job is not None and job.last_error and job.status != "done" and job.last_error not in errors:
        errors.append(job.last_error)
    for page_key, flags in sorted((sub.page_flags or {}).items(), key=lambda kv: int(kv[0])):
        if "page_failed" in flags:
            errors.append(f"Page {page_key} could not be processed")
    return StatusOut(id=sub.id, status=sub.status, stage=sub.stage, pages_done=sub.pages_done,
                     pages_total=sub.page_count, errors=list(dict.fromkeys(errors)),
                     attempts=job.attempts if job is not None else 0, progress=sub.progress)


def _has_teacher_work(db: Session, sub: Submission) -> bool:
    if any(r.source == "manual" for r in sub.regions):
        return True
    if any(a.status == "reviewed" or a.text_corrected or "teacher_flag" in (a.flags or []) for a in sub.answers):
        return True
    ids = [a.id for a in sub.answers]
    return bool(ids) and bool(db.scalar(select(func.count(Review.id)).where(Review.answer_id.in_(ids))))


@router.post("/submissions/{submission_id}/retry", response_model=SubmissionOut, status_code=202)
def retry_submission(submission_id: str, db: Session = Depends(get_db),
                     user: User = Depends(current_user)) -> SubmissionOut:
    """Re-queue processing for a failed sheet, or a processed one nobody has reviewed yet."""
    sub = owned_submission(db, submission_id, user, write=True)
    if sub.status == "processed":
        if _has_teacher_work(db, sub):
            raise conflict("This sheet has teacher reviews or region edits; re-processing would discard them")
    elif sub.status != "failed":
        raise conflict(f"Only failed or unreviewed processed sheets can be retried (status: {sub.status})")
    active = db.scalar(select(func.count(Job.id)).where(
        Job.submission_id == sub.id, Job.status.in_(("queued", "running")))) or 0
    if active:
        raise conflict("This sheet is already queued for processing")
    if not sub.pdf_path or not Path(sub.pdf_path).is_file():
        raise conflict("The original PDF is missing; delete this sheet and upload it again")
    old = {"status": sub.status, "error": sub.error}
    sub.status, sub.stage, sub.error, sub.pages_done, sub.progress = "uploaded", "queued", None, 0, None
    enqueue(db, sub.id)
    audit(db, user_id=user.id, entity="submission", entity_id=sub.id, action="retry", old=old,
          new={"status": "uploaded"})
    db.commit()
    return submission_out(db, sub)


@router.delete("/submissions/{submission_id}", status_code=204)
def delete_submission(submission_id: str, db: Session = Depends(get_db),
                      user: User = Depends(current_user)) -> Response:
    sub = owned_submission(db, submission_id, user, write=True)
    if sub.status == "approved":
        raise conflict("Approved marks are locked; reopen the sheet before deleting it")
    # The worker holds the PDF open while it runs (Windows cannot delete an open file)
    # and would keep writing page images into a folder no row points to any more.
    stale = _now() - timedelta(seconds=get_settings().job_stale_seconds)
    if db.scalar(select(func.count(Job.id)).where(Job.submission_id == sub.id, Job.status == "running",
                                                  Job.heartbeat_at >= stale)):
        raise conflict("This sheet is being processed right now; delete it once processing has finished")
    audit(db, user_id=user.id, entity="submission", entity_id=sub.id, action="delete",
          old={"student_ref": sub.student_ref, "filename": sub.original_filename, "status": sub.status,
               "total_final_marks": sub.total_final_marks})
    db.delete(sub)
    db.commit()
    delete_submission_files(sub)
    return Response(status_code=204)


# ---------------------------------------------------------------- answers & review
@router.get("/submissions/{submission_id}/answers", response_model=list[AnswerOut])
def list_answers(submission_id: str, status: AnswerFilter = "all", sort: AnswerSort = "priority",
                 db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[AnswerOut]:
    """`priority` puts answers needing review first, then lowest confidence, then question order."""
    sub = owned_submission(db, submission_id, user)
    answers = [a for a in sub.answers if status == "all" or a.status == status]
    if sort == "priority":
        answers.sort(key=lambda a: (0 if a.status in FLAGGED else 1, a.confidence or 0.0, a.question.question_no))
    else:
        answers.sort(key=lambda a: a.question.question_no)
    region_ids = _region_ids_by_question(sub)
    return [answer_out(a, sub, region_ids) for a in answers]


@router.get("/answers/{answer_id}", response_model=AnswerOut)
def get_answer(answer_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> AnswerOut:
    return answer_out(owned_answer(db, answer_id, user))


def _check_marks(value: float, max_marks: float) -> None:
    if value < 0 or value > max_marks + 1e-9:
        raise validation("Marks must be between 0 and the question's maximum",
                         [{"field": "final_marks", "message": f"must be between 0 and {max_marks}"}])
    if abs(value * 100 - round(value * 100)) > 1e-6:
        raise validation("Marks can have at most 2 decimal places",
                         [{"field": "final_marks", "message": "use at most 2 decimal places"}])


@router.patch("/answers/{answer_id}/review", response_model=AnswerOut)
def review_answer(answer_id: str, body: ReviewIn, db: Session = Depends(get_db),
                  user: User = Depends(current_user)) -> AnswerOut:
    """Correct the text (regrades), override marks (reason required when they differ
    from the AI), comment, and approve. With approve=false only the text and comment
    are saved and the answer stays in the review queue."""
    ans = owned_answer(db, answer_id, user, write=True)
    sub = ans.submission
    ensure_unlocked(sub)
    q = ans.question
    s = get_settings()
    if body.final_marks is not None:
        _check_marks(body.final_marks, q.max_marks)

    old = _answer_state(ans)
    reviewed_before = ans.status == "reviewed" and ans.final_marks is not None
    reason = (body.reason or "").strip() or None

    text_changed, original_text, new_text = False, ans.ocr_text, None
    if body.corrected_text is not None:
        new_text = body.corrected_text.strip()
        if new_text != (ans.ocr_text or "").strip():
            ans.text_corrected = True
            compute_answer(db, sub, q, s, override_text=new_text)
            text_changed = True
    if body.comment is not None:
        ans.comment = body.comment.strip() or None

    marks_override = False
    if body.approve:
        if body.final_marks is not None:
            final = round(body.final_marks, 2)
        elif reviewed_before and not text_changed:
            final = ans.final_marks  # an earlier teacher decision stands
        else:
            # Unreviewed, or the corrected text was just re-graded: the earlier mark was
            # based on text the teacher has declared wrong, so follow the new AI marks.
            final = round(ans.ai_marks, 2)
        differs_from_ai = abs(final - ans.ai_marks) > 1e-9
        differs_from_current = ans.final_marks is None or abs(final - ans.final_marks) > 1e-9
        if differs_from_ai and differs_from_current and not reason:
            raise validation("Give a reason when changing the AI's marks",
                             [{"field": "reason", "message": "required when final_marks differs from ai_marks"}])
        marks_override = differs_from_ai and differs_from_current
        ans.final_marks = final
        ans.status = "reviewed"
        ans.flags = [f for f in (ans.flags or []) if f != "teacher_flag"]
        db.add(Review(answer_id=ans.id, teacher_id=user.id, final_marks=final, comment=ans.comment, reason=reason))
    elif text_changed and old["status"] in FLAGGED and ans.status not in ("reviewed", "needs_review"):
        ans.status = "needs_review"  # saved, but not approved: keep it in the queue

    if text_changed or marks_override:  # kept for retraining (FR-19, FR-20)
        db.add(Correction(
            answer_id=ans.id, original_text=original_text if text_changed else None,
            corrected_text=new_text if text_changed else None, original_marks=old["ai_marks"],
            corrected_marks=ans.final_marks if body.approve else None,
        ))
    if sub.status == "processed":
        sub.status = "in_review"
    refresh_totals(sub)
    audit(db, user_id=user.id, entity="answer", entity_id=ans.id, action="review" if body.approve else "edit",
          old=old, new=_answer_state(ans), reason=reason)
    db.commit()
    return answer_out(ans, sub)


@router.post("/answers/{answer_id}/flag", response_model=AnswerOut)
def flag_answer(answer_id: str, body: FlagIn | None = None, db: Session = Depends(get_db),
                user: User = Depends(current_user)) -> AnswerOut:
    """Send an answer back to the review queue ("look at this again")."""
    ans = owned_answer(db, answer_id, user, write=True)
    sub = ans.submission
    ensure_unlocked(sub)
    old = {"status": ans.status, "flags": list(ans.flags or [])}
    ans.status = "needs_review"
    ans.flags = list(dict.fromkeys(list(ans.flags or []) + ["teacher_flag"]))
    reason = ((body.reason if body else None) or "").strip() or None
    audit(db, user_id=user.id, entity="answer", entity_id=ans.id, action="flag", old=old,
          new={"status": ans.status, "flags": ans.flags}, reason=reason)
    db.commit()
    return answer_out(ans, sub)


def _approve(db: Session, sub: Submission, user: User) -> None:
    old_status = sub.status
    for a in sub.answers:
        if a.final_marks is None:
            a.final_marks = round(effective_marks(a), 2)
    sub.status, sub.approved_at = "approved", _now()
    refresh_totals(sub)
    audit(db, user_id=user.id, entity="submission", entity_id=sub.id, action="approve",
          old={"status": old_status},
          new={"status": "approved", "total_final_marks": sub.total_final_marks,
               "marks": {str(a.question.question_no): a.final_marks for a in sub.answers}})


@router.post("/submissions/{submission_id}/approve", response_model=SubmissionOut)
def approve_submission(submission_id: str, db: Session = Depends(get_db),
                       user: User = Depends(current_user)) -> SubmissionOut:
    sub = owned_submission(db, submission_id, user, write=True)
    if sub.status == "approved":
        raise conflict("This sheet is already approved")
    if sub.status not in REVIEWABLE:
        raise conflict(f"Only processed sheets can be approved (status: {sub.status})")
    pending = _pending(sub)
    if pending:
        over = any(effective_marks(a) > a.question.max_marks + 1e-9 for a in sub.answers)
        raise conflict("Some marks are above their question's maximum after a rubric change; regrade the exam "
                       "or review them" if over else "Some answers still need review", {"pending": pending})
    _approve(db, sub, user)
    db.commit()
    return submission_out(db, sub, 0)


@router.post("/submissions/{submission_id}/reopen", response_model=SubmissionOut)
def reopen_submission(submission_id: str, body: ReopenIn, db: Session = Depends(get_db),
                      user: User = Depends(current_user)) -> SubmissionOut:
    sub = owned_submission(db, submission_id, user, write=True)
    if sub.status != "approved":
        raise conflict("Only approved sheets can be reopened")
    reason = body.reason.strip()
    if not reason:
        raise validation("A reason is required to reopen approved marks",
                         [{"field": "reason", "message": "required"}])
    for a in sub.answers:
        if a.status != "reviewed":  # marks frozen at approval go back to following the AI
            a.final_marks = None
    old = {"status": sub.status, "approved_at": sub.approved_at.isoformat() if sub.approved_at else None,
           "total_final_marks": sub.total_final_marks}
    sub.status, sub.approved_at = "in_review", None
    refresh_totals(sub)
    audit(db, user_id=user.id, entity="submission", entity_id=sub.id, action="reopen", old=old,
          new={"status": sub.status, "total_final_marks": sub.total_final_marks}, reason=reason)
    db.commit()
    return submission_out(db, sub)


@router.post("/exams/{exam_id}/approve-all")
def approve_all(exam_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    """Approve every processed sheet with nothing left to review; report the rest."""
    exam = owned_exam(db, exam_id, user, write=True)
    approved: list[str] = []
    blocked: list[dict] = []
    for sub in sorted(exam.submissions, key=lambda x: x.student_ref):
        if sub.status not in REVIEWABLE:
            continue
        pending = _pending(sub)
        if pending:
            blocked.append({"submission_id": sub.id, "student_ref": sub.student_ref, "pending": pending})
            continue
        _approve(db, sub, user)
        approved.append(sub.id)
    db.commit()
    return {"approved": approved, "blocked": blocked}


# ---------------------------------------------------------------- page images
def _png_size(path: Path) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)


def _pdf_page_sizes(pdf_path: str, dpi: int) -> dict[int, tuple[int, int]]:
    sizes: dict[int, tuple[int, int]] = {}
    try:
        with fitz.open(pdf_path) as doc:
            for i in range(doc.page_count):
                rect = doc.load_page(i).rect
                z = safe_dpi(rect, dpi) / 72.0
                r = (rect * fitz.Matrix(z, z)).irect
                sizes[i + 1] = (r.width, r.height)
    except Exception:
        log.warning("Could not read page sizes from %s", pdf_path)
    return sizes


def _page_size(s: Settings, sub: Submission, page_no: int) -> tuple[int, int]:
    size = _png_size(page_image_path(s.storage_dir, sub.id, page_no))
    if size is None:
        size = _pdf_page_sizes(sub.pdf_path, s.render_dpi).get(page_no)
    if size is None:
        raise not_found("Page")
    return size


def _render_preview(pdf_path: str, page_no: int, dpi: int) -> np.ndarray | None:
    """Page image for a sheet not processed yet (rendered on the fly, not cached)."""
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(page_no - 1)
            z = safe_dpi(page.rect, dpi) / 72.0  # never trust the page size a PDF declares
            pix = page.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY, alpha=False)
            return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    except Exception:
        log.warning("Could not render page %s of %s", page_no, pdf_path)
        return None


@router.get("/submissions/{submission_id}/pages")
def list_pages(submission_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    sub = owned_submission(db, submission_id, user)
    s = get_settings()
    pdf_sizes: dict[int, tuple[int, int]] | None = None
    pages = []
    for p in range(1, sub.page_count + 1):
        size = _png_size(page_image_path(s.storage_dir, sub.id, p))
        if size is None:
            if pdf_sizes is None:
                pdf_sizes = _pdf_page_sizes(sub.pdf_path, s.render_dpi)
            size = pdf_sizes.get(p, (0, 0))
        pages.append({"page_no": p, "width": size[0], "height": size[1],
                      "flags": list((sub.page_flags or {}).get(str(p), []))})
    return {"pages": pages}


@router.get("/submissions/{submission_id}/pages/{page_no}/image",
            responses={200: {"content": {"image/png": {}}}}, response_class=Response)
def page_image(submission_id: str, page_no: int, thumb: bool = Query(False, description="1 = 240 px wide"),
               db: Session = Depends(get_db), user: User = Depends(current_user)) -> Response:
    sub = owned_submission(db, submission_id, user)
    if not 1 <= page_no <= sub.page_count:
        raise not_found("Page")
    s = get_settings()
    path = page_image_path(s.storage_dir, sub.id, page_no)
    headers = {"Cache-Control": "private, no-cache"}
    if not thumb and path.is_file():
        return FileResponse(path, media_type="image/png", headers=headers)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
    if img is None:
        img = _render_preview(sub.pdf_path, page_no, 40 if thumb else s.render_dpi)
    if img is None:
        raise not_found("Page image")
    if thumb and img.shape[1] > THUMB_WIDTH:
        h = max(1, round(img.shape[0] * THUMB_WIDTH / img.shape[1]))
        img = cv2.resize(img, (THUMB_WIDTH, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ApiError(500, "internal_error", "Could not encode the page image")
    return Response(buf.tobytes(), media_type="image/png", headers=headers)


# ---------------------------------------------------------------- regions
def _region_dict(r: AnswerRegion) -> dict:
    return {"id": r.id, "page_no": r.page_no, "bbox": [int(v) for v in (r.bbox or [0, 0, 0, 0])],
            "question_id": r.question_id, "source": r.source, "is_blank": bool(r.is_blank),
            "segmentation_confidence": float(r.segmentation_confidence), "flags": list(r.flags or [])}


def _snapshot(sub: Submission) -> list[dict]:
    return [_region_dict(r) for r in _sorted_regions(sub)]


def _validate_bbox(bbox: list[int], size: tuple[int, int], field: str = "bbox") -> None:
    x, y, w, h = bbox
    pw, ph = size
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > pw or y + h > ph:
        raise validation("The box must lie inside the page image",
                         [{"field": field, "message": f"page is {pw}x{ph} px; got [x, y, w, h] = {list(bbox)}"}])


def _exam_question(db: Session, sub: Submission, question_id: str) -> Question:
    q = db.get(Question, question_id)
    if q is None or q.exam_id != sub.exam_id:
        raise validation("That question is not part of this exam",
                         [{"field": "question_id", "message": "unknown question for this exam"}])
    return q


def _mark_manual(r: AnswerRegion) -> None:
    r.source = "manual"
    r.segmentation_confidence = 1.0  # a teacher placed it
    r.flags = [f for f in (r.flags or []) if f not in MANUAL_CLEARS]


def _finish_region_edit(db: Session, sub: Submission, user: User, before: list[dict], affected: set,
                        region: AnswerRegion | None, action: str, reason: str | None,
                        audit_action: str = "regions_edit", extra: dict | None = None) -> RegionEditOut:
    """Regrade the affected questions, move the sheet into review, audit, commit."""
    s = get_settings()
    db.flush()
    questions = {q.id: q for q in sub.exam.questions}
    targets = sorted((questions[qid] for qid in affected if qid in questions), key=lambda q: q.question_no)
    before_answers = {a.question_id: _answer_state(a) for a in sub.answers if a.question_id in affected}
    answers = compute_answers(db, sub, targets, s)  # one AI request for the affected answers
    refresh_totals(sub)
    audit_answer_changes(db, user.id, before_answers, answers, audit_action, reason)
    if sub.status == "processed":
        sub.status = "in_review"
    after = _snapshot(sub)
    alive = region is not None and any(r["id"] == region.id for r in after)
    audit(db, user_id=user.id, entity="submission", entity_id=sub.id, action=audit_action,
          old={"regions": before},
          new={"regions": after, "action": action, "region_id": region.id if region is not None else None,
               **(extra or {})},
          reason=reason)
    if region is not None:
        audit(db, user_id=user.id, entity="region", entity_id=region.id, action=action,
              old=next((r for r in before if r["id"] == region.id), None),
              new=next((r for r in after if r["id"] == region.id), None), reason=reason)
    db.commit()
    region_ids = _region_ids_by_question(sub)
    return RegionEditOut(
        region=RegionOut.model_validate(_region_dict(region)) if alive else None,
        regions=[RegionOut.model_validate(r) for r in after],
        answers=[answer_out(a, sub, region_ids) for a in answers],
    )


@router.get("/submissions/{submission_id}/regions", response_model=list[RegionOut])
def list_regions(submission_id: str, db: Session = Depends(get_db),
                 user: User = Depends(current_user)) -> list[RegionOut]:
    sub = owned_submission(db, submission_id, user)
    return [RegionOut.model_validate(r) for r in _snapshot(sub)]


@router.post("/submissions/{submission_id}/regions", response_model=RegionEditOut, status_code=201)
def create_region(submission_id: str, body: RegionCreate, db: Session = Depends(get_db),
                  user: User = Depends(current_user)) -> RegionEditOut:
    """Draw a new answer region by hand (e.g. an answer segmentation missed)."""
    sub = owned_submission(db, submission_id, user, write=True)
    ensure_unlocked(sub)
    if not 1 <= body.page_no <= sub.page_count:
        raise validation("No such page", [{"field": "page_no", "message": f"must be 1-{sub.page_count}"}])
    _validate_bbox(body.bbox, _page_size(get_settings(), sub, body.page_no))
    q = _exam_question(db, sub, body.question_id) if body.question_id else None
    before = _snapshot(sub)
    region = AnswerRegion(id=str(uuid.uuid4()), submission_id=sub.id, page_no=body.page_no, bbox=list(body.bbox),
                          question_id=q.id if q else None, source="manual", is_blank=False,
                          segmentation_confidence=1.0, flags=[])
    sub.regions.append(region)
    return _finish_region_edit(db, sub, user, before, {q.id} if q else set(), region, "create", None)


@router.patch("/regions/{region_id}", response_model=RegionEditOut)
def edit_region(region_id: str, body: RegionPatch, db: Session = Depends(get_db),
                user: User = Depends(current_user)) -> RegionEditOut:
    """crop | reassign | split | merge | blank | unblank one region; the answers of
    every affected question are regraded from the stored OCR lines (no new OCR)."""
    region = owned_region(db, region_id, user, write=True)
    sub = region.submission
    ensure_unlocked(sub)
    s = get_settings()
    before = _snapshot(sub)
    affected: set = {region.question_id}
    action = body.action

    if action == "crop":
        if body.bbox is None:
            raise validation("A box is required to crop", [{"field": "bbox", "message": "required for crop"}])
        _validate_bbox(body.bbox, _page_size(s, sub, region.page_no))
        region.bbox = list(body.bbox)
    elif action == "reassign":
        q = _exam_question(db, sub, body.question_id) if body.question_id else None
        region.question_id = q.id if q else None
        affected.add(region.question_id)
    elif action == "split":
        x, y, w, h = region.bbox
        if body.split_y is None or not y < body.split_y < y + h:
            raise validation("The split line must fall inside the region",
                             [{"field": "split_y", "message": f"must be strictly between {y} and {y + h}"}])
        region.bbox = [x, y, w, body.split_y - y]
        lower = AnswerRegion(id=str(uuid.uuid4()), submission_id=sub.id, page_no=region.page_no,
                             bbox=[x, body.split_y, w, y + h - body.split_y], question_id=region.question_id,
                             source="manual", is_blank=region.is_blank, segmentation_confidence=1.0,
                             flags=list(region.flags or []))
        _mark_manual(lower)
        sub.regions.append(lower)
    elif action == "merge":
        if not body.merge_with:
            raise validation("Choose a region to merge with", [{"field": "merge_with", "message": "required"}])
        other = db.get(AnswerRegion, body.merge_with)
        if other is None or other.submission_id != sub.id or other.id == region.id:
            raise validation("Pick another region of the same sheet",
                             [{"field": "merge_with", "message": "must be a different region of this sheet"}])
        affected.add(other.question_id)
        if other.page_no == region.page_no:
            ax, ay, aw, ah = region.bbox
            bx, by, bw, bh = other.bbox
            x0, y0 = min(ax, bx), min(ay, by)
            region.bbox = [x0, y0, max(ax + aw, bx + bw) - x0, max(ay + ah, by + bh) - y0]
            region.is_blank = bool(region.is_blank and other.is_blank)
            region.flags = list(dict.fromkeys(list(region.flags or []) + list(other.flags or [])))
            sub.regions.remove(other)  # delete-orphan removes the row
        else:  # an answer continuing on another page: give that part the same question
            other.question_id = region.question_id
            _mark_manual(other)
    elif action == "blank":
        region.is_blank = True
    elif action == "unblank":
        region.is_blank = False

    _mark_manual(region)
    affected.discard(None)
    reason = (body.reason or "").strip() or None
    return _finish_region_edit(db, sub, user, before, affected, region, action, reason)


@router.post("/submissions/{submission_id}/regions/undo", response_model=RegionEditOut)
def undo_region_edit(submission_id: str, db: Session = Depends(get_db),
                     user: User = Depends(current_user)) -> RegionEditOut:
    """Restore the regions as they were before the most recent edit not yet undone
    (repeat to step further back)."""
    sub = owned_submission(db, submission_id, user, write=True)
    ensure_unlocked(sub)
    entries = db.scalars(
        select(AuditLog).where(AuditLog.entity == "submission", AuditLog.entity_id == sub.id,
                               AuditLog.action.in_(("regions_edit", "regions_undo")))
        .order_by(AuditLog.id.desc())
    ).all()
    undone = {int(e.new_value["undone"]) for e in entries
              if e.action == "regions_undo" and isinstance(e.new_value, dict) and e.new_value.get("undone") is not None}
    target = next((e for e in entries if e.action == "regions_edit" and e.id not in undone), None)
    if target is None:
        raise conflict("Nothing to undo")

    current = {r.id: r for r in sub.regions}
    expected = {r["id"] for r in (target.new_value or {}).get("regions", [])}
    if set(current) != expected:
        raise conflict("The regions have changed since that edit (for example the sheet was re-processed), "
                       "so it cannot be undone")

    before = _snapshot(sub)
    snapshot = {r["id"]: r for r in (target.old_value or {}).get("regions", [])}
    valid_q = {q.id for q in sub.exam.questions}
    affected: set = set()
    for rid, reg in list(current.items()):
        if rid not in snapshot:
            affected.add(reg.question_id)
            sub.regions.remove(reg)
    for rid, data in snapshot.items():
        qid = data.get("question_id") if data.get("question_id") in valid_q else None
        fields = {"page_no": int(data["page_no"]), "bbox": [int(v) for v in data["bbox"]], "question_id": qid,
                  "source": data.get("source", "auto"), "is_blank": bool(data.get("is_blank", False)),
                  "segmentation_confidence": float(data.get("segmentation_confidence", 1.0)),
                  "flags": list(data.get("flags") or [])}
        reg = current.get(rid)
        if reg is None:
            sub.regions.append(AnswerRegion(id=rid, submission_id=sub.id, **fields))
            affected.add(qid)
        elif {k: v for k, v in _region_dict(reg).items() if k != "id"} != fields:
            affected.update({reg.question_id, qid})
            for k, v in fields.items():
                setattr(reg, k, v)
    affected.discard(None)
    return _finish_region_edit(db, sub, user, before, affected, None, "undo", None,
                               audit_action="regions_undo", extra={"undone": target.id})


# ---------------------------------------------------------------- audit log & capabilities
def _check_entity_access(db: Session, entity: str, entity_id: str, user: User) -> None:
    if entity == "exam":
        owned_exam(db, entity_id, user)
    elif entity == "submission":
        owned_submission(db, entity_id, user)
    elif entity == "answer":
        owned_answer(db, entity_id, user)
    else:
        owned_region(db, entity_id, user)


@router.get("/audit-log", response_model=Page)
def audit_log(
    entity: AuditEntity | None = None,
    entity_id: str | None = Query(None, alias="id"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict:
    """History of one entity (entity + id), newest first. Without an id: your own
    actions (admins: everyone's), optionally filtered by entity type."""
    stmt = select(AuditLog)
    if entity_id:
        if entity is None:
            raise validation("entity is required with id", [{"field": "entity", "message": "required with id"}])
        _check_entity_access(db, entity, entity_id, user)
        stmt = stmt.where(AuditLog.entity == entity, AuditLog.entity_id == entity_id)
    else:
        if entity:
            stmt = stmt.where(AuditLog.entity == entity)
        if user.role != "admin":
            stmt = stmt.where(AuditLog.user_id == user.id)
    rows, total = _page(db, stmt.order_by(AuditLog.id.desc()), page, page_size)
    return {"items": [AuditOut.model_validate(r) for r in rows], "total": total, "page": page,
            "page_size": page_size}


@router.get("/capabilities")
def capabilities(user: User = Depends(current_user)) -> dict:
    """Which OCR engines and graders this install can run, plus the active thresholds."""
    s = get_settings()
    return {
        "ocr": ocr_capabilities(),
        "graders": grader_capabilities(),
        "settings": {
            "grader": s.grader, "ocr_engine": s.ocr_engine, "tau": s.tau, "tau_low": s.tau_low,
            "contradiction_threshold": s.contradiction_threshold,
            "auto_accept_threshold": s.auto_accept_threshold, "max_upload_mb": s.max_upload_mb,
            "max_pages": s.max_pages,
        },
    }
