from __future__ import annotations

import csv
import io
import re
import statistics
from types import SimpleNamespace
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ..audit import audit
from ..config import get_settings
from ..db import get_db
from ..errors import ApiError, conflict, validation
from ..models import Answer, Exam, Question, Submission, User
from ..pipeline.processor import compute_answers, effective_marks, refresh_totals
from ..schemas import ExamIn, ExamOut, ExamPatch, ExamStats, QuestionOut, QuestionsIn
from ..security import current_user
from .deps import owned_exam

router = APIRouter(prefix="/exams", tags=["exams"])

REVIEWABLE = ("processed", "in_review")


def _stats(db: Session, exam: Exam) -> ExamStats:
    rows = db.execute(
        select(Submission.status, func.count()).where(Submission.exam_id == exam.id).group_by(Submission.status)
    ).all()
    counts = {s: n for s, n in rows}
    needs = db.scalar(
        select(func.count(Answer.id)).join(Submission, Answer.submission_id == Submission.id)
        .where(Submission.exam_id == exam.id, Answer.status.in_(("needs_review", "failed")),
               Submission.status.in_(REVIEWABLE))
    ) or 0
    totals = db.scalars(
        select(Submission.total_final_marks).where(
            Submission.exam_id == exam.id, Submission.status.in_(("processed", "in_review", "approved")),
            Submission.total_final_marks.is_not(None))
    ).all()
    return ExamStats(
        questions=len(exam.questions), sheets=sum(counts.values()),
        processed=sum(counts.get(s, 0) for s in ("processed", "in_review", "approved")),
        processing=counts.get("uploaded", 0) + counts.get("processing", 0),
        failed=counts.get("failed", 0), approved=counts.get("approved", 0), needs_review=needs,
        average_marks=round(statistics.fmean(totals), 2) if totals else None,
    )


def exam_out(db: Session, exam: Exam, user: User) -> ExamOut:
    return ExamOut(
        id=exam.id, title=exam.title, subject=exam.subject, date=exam.exam_date, total_marks=exam.total_marks,
        created_at=exam.created_at, rubric_marks=round(sum(q.max_marks for q in exam.questions), 2),
        stats=_stats(db, exam),
        can_edit=exam.teacher_id == user.id,  # only the owner edits; admins read everyone's (PRD 22.1)
    )


@router.get("")
def list_exams(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
               db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    q = select(Exam)
    if user.role != "admin":
        q = q.where(Exam.teacher_id == user.id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    exams = db.scalars(q.order_by(Exam.created_at.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return {"items": [exam_out(db, e, user) for e in exams], "total": total, "page": page, "page_size": page_size}


@router.post("", response_model=ExamOut, status_code=201)
def create_exam(body: ExamIn, db: Session = Depends(get_db), user: User = Depends(current_user)) -> ExamOut:
    exam = Exam(teacher_id=user.id, title=body.title.strip(), subject=body.subject, exam_date=body.date,
                total_marks=body.total_marks)
    db.add(exam)
    db.flush()
    audit(db, user_id=user.id, entity="exam", entity_id=exam.id, action="create",
          new={"title": exam.title, "total_marks": exam.total_marks})
    db.commit()
    return exam_out(db, exam, user)


@router.get("/{exam_id}", response_model=ExamOut)
def get_exam(exam_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> ExamOut:
    return exam_out(db, owned_exam(db, exam_id, user), user)


@router.patch("/{exam_id}", response_model=ExamOut)
def update_exam(exam_id: str, body: ExamPatch, db: Session = Depends(get_db),
                user: User = Depends(current_user)) -> ExamOut:
    exam = owned_exam(db, exam_id, user, write=True)
    changes = body.model_dump(exclude_unset=True)
    rubric_total = round(sum(q.max_marks for q in exam.questions), 2)
    if changes.get("total_marks") is not None and changes["total_marks"] < rubric_total - 0.01:
        raise validation("Total marks cannot be less than the rubric's questions add up to",
                         [{"field": "total_marks", "message": f"must be at least {rubric_total:g} (the rubric total)"}])
    old = {k: str(getattr(exam, "exam_date" if k == "date" else k)) for k in changes}
    for k, v in changes.items():
        setattr(exam, "exam_date" if k == "date" else k, v)
    audit(db, user_id=user.id, entity="exam", entity_id=exam.id, action="update", old=old,
          new={k: str(v) for k, v in changes.items()})
    db.commit()
    return exam_out(db, exam, user)


@router.delete("/{exam_id}", status_code=204)
def delete_exam(exam_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> Response:
    exam = owned_exam(db, exam_id, user, write=True)
    from .submissions import delete_submission_files

    files = [SimpleNamespace(id=sub.id, pdf_path=sub.pdf_path) for sub in exam.submissions]
    audit(db, user_id=user.id, entity="exam", entity_id=exam.id, action="delete", old={"title": exam.title})
    db.delete(exam)
    db.commit()
    # Only after the commit: if it fails (e.g. database locked) the sheets keep their files.
    for sub in files:
        delete_submission_files(sub)
    return Response(status_code=204)


# ---------- rubric ----------
@router.get("/{exam_id}/questions", response_model=list[QuestionOut])
def get_questions(exam_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[Question]:
    return owned_exam(db, exam_id, user).questions


def _check_max_marks_change(exam: Exam, existing: dict, incoming: dict) -> None:
    """Approved marks (and the max total exported with them) never change, and a
    reviewed mark can never end up above its question's maximum (0 <= marks <= max)."""
    changed = {n: q.max_marks for n, q in incoming.items()
               if n in existing and abs(q.max_marks - existing[n].max_marks) > 1e-9}
    if not changed:
        return
    approved = sorted(sub.student_ref for sub in exam.submissions if sub.status == "approved")
    if approved:
        raise conflict("Approved sheets were marked out of the current maximums; reopen them before changing "
                       "a question's maximum marks", {"questions": sorted(changed), "approved": approved})
    over = [{"student_ref": sub.student_ref, "question_no": a.question.question_no, "final_marks": a.final_marks,
             "max_marks": changed[a.question.question_no]}
            for sub in exam.submissions for a in sub.answers
            if a.final_marks is not None and a.question.question_no in changed
            and a.final_marks > changed[a.question.question_no] + 1e-9]
    if over:
        raise conflict("Some reviewed marks are above the new maximum; lower those marks first",
                       {"over_max": sorted(over, key=lambda o: (o["student_ref"], o["question_no"]))})


@router.post("/{exam_id}/questions", response_model=list[QuestionOut], status_code=201)
@router.put("/{exam_id}/questions", response_model=list[QuestionOut])
def put_questions(exam_id: str, body: QuestionsIn, db: Session = Depends(get_db),
                  user: User = Depends(current_user)) -> list[Question]:
    """Replace the rubric. Once sheets exist, question numbers are fixed but text,
    marks and key points may still be edited (then regrade)."""
    exam = owned_exam(db, exam_id, user, write=True)
    existing = {q.question_no: q for q in exam.questions}
    incoming = {q.question_no: q for q in body.questions}
    has_sheets = bool(exam.submissions)
    if has_sheets and set(existing) != set(incoming):
        raise conflict("Sheets are already uploaded, so questions cannot be added or removed; edit them instead",
                       {"existing": sorted(existing), "incoming": sorted(incoming)})
    rubric_total = round(sum(q.max_marks for q in body.questions), 2)
    if rubric_total > exam.total_marks + 0.01:
        raise ApiError(400, "validation_error", "Questions add up to more than the exam's total marks",
                       [{"field": "questions", "message": f"{rubric_total} > {exam.total_marks}"}])
    if has_sheets:
        _check_max_marks_change(exam, existing, incoming)

    old = {n: {"max_marks": q.max_marks, "key_points": q.key_points} for n, q in existing.items()}
    for n, q in list(existing.items()):
        if n not in incoming:
            exam.questions.remove(q)
    db.flush()
    for n, qin in incoming.items():
        data = qin.model_dump()
        data["key_points"] = [dict(k) for k in data["key_points"]]
        q = existing.get(n)
        if q is None:
            exam.questions.append(Question(**data))
        else:
            for k, v in data.items():
                setattr(q, k, v)
    audit(db, user_id=user.id, entity="exam", entity_id=exam.id, action="rubric_update",
          old={"questions": old}, new={"questions": {n: {"max_marks": q.max_marks} for n, q in incoming.items()}})
    db.commit()
    db.refresh(exam)
    return exam.questions


QUESTION_PAPER_LIMIT = 15 * 1024 * 1024


@router.post("/{exam_id}/questions/extract")
def extract_questions(exam_id: str, file: UploadFile = File(..., description="Question paper: PDF, PNG, JPG or WEBP"),
                      db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    """Step 1 (question sheet upload): read the exam's question paper into a DRAFT rubric
    (question number, type, marks, text). Nothing is saved; the teacher checks the draft
    on the Rubric screen and saves it with PUT /questions."""
    from ..pipeline import processor, question_paper

    exam = owned_exam(db, exam_id, user, write=True)
    data = file.file.read(QUESTION_PAPER_LIMIT + 1)
    if len(data) > QUESTION_PAPER_LIMIT:
        raise ApiError(413, "file_too_large", f"The question paper is larger than {QUESTION_PAPER_LIMIT // 2**20} MB")
    if not data:
        raise validation("The file is empty", [{"field": "file", "message": "empty file"}])
    exam_ref = exam.id
    db.rollback()  # release the read transaction: the AI router logs its calls in its own sessions
    s = get_settings()
    factory = sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False)
    result = question_paper.extract(data, file.filename, file.content_type, s,
                                    lambda: processor.make_ai_router(s, factory), exam_id=exam_ref)
    audit(db, user_id=user.id, entity="exam", entity_id=exam_ref, action="question_paper_read",
          new={"questions": len(result.questions), "source": result.source, "pages": result.pages})
    db.commit()
    return result.as_dict()


@router.post("/{exam_id}/regrade")
def regrade(exam_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    """Re-run grading (not OCR) for every unlocked sheet after a rubric change."""
    exam = owned_exam(db, exam_id, user, write=True)
    from .submissions import _answer_state, audit_answer_changes

    s = get_settings()
    n = 0
    for sub in exam.submissions:
        if sub.status not in REVIEWABLE:
            continue
        before = {a.question_id: _answer_state(a) for a in sub.answers}
        answers = compute_answers(db, sub, exam.questions, s)  # one AI request per sheet
        refresh_totals(sub)
        # Every answer whose marks or status moved gets its own old/new entry (PRD 15.1).
        audit_answer_changes(db, user.id, before, answers, "regrade")
        n += 1
    audit(db, user_id=user.id, entity="exam", entity_id=exam.id, action="regrade", new={"sheets": n})
    db.commit()
    return {"regraded": n}


# ---------- results ----------
def _results_rows(exam: Exam, approved_only: bool) -> tuple[list[str], list[list]]:
    qs = sorted(exam.questions, key=lambda q: q.question_no)
    header = ["student_ref"] + [f"Q{q.question_no}" for q in qs] + ["total", "max_total", "status", "approved_at"]
    rows = []
    for sub in sorted(exam.submissions, key=lambda s: s.student_ref):
        if approved_only and sub.status != "approved":
            continue
        by_q = {a.question_id: a for a in sub.answers}
        marks = [round(effective_marks(by_q[q.id]), 2) if q.id in by_q else "" for q in qs]
        total = round(sum(m for m in marks if m != ""), 2)
        rows.append([sub.student_ref, *marks, total, round(sum(q.max_marks for q in qs), 2), sub.status,
                     sub.approved_at.isoformat() if sub.approved_at else ""])
    return header, rows


@router.get("/{exam_id}/export")
def export(exam_id: str, format: str = Query("csv", pattern="^csv$"), db: Session = Depends(get_db),
           user: User = Depends(current_user)) -> Response:
    """Only approved sheets are exported (screen spec 17.1, screen 8)."""
    exam = owned_exam(db, exam_id, user)
    header, rows = _results_rows(exam, approved_only=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows([[_csv_cell(v) for v in row] for row in rows])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": _content_disposition(exam.title, "_results.csv")})


def _csv_cell(value):  # noqa: ANN001, ANN202
    """A spreadsheet runs a cell starting with = + - @ (or tab / CR) as a formula: quote it."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _content_disposition(title: str, suffix: str) -> str:
    """HTTP headers are latin-1, so send an ASCII fallback name plus the real UTF-8
    name (RFC 6266 / 5987); the client prefers `filename*`."""
    ascii_name = "".join(c if (c.isascii() and c.isalnum()) or c in "-_" else "_" for c in title)
    ascii_name = ascii_name[:60].strip("_") or "exam"
    utf8_name = (re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", title).strip()[:100] or "exam") + suffix
    return f"attachment; filename=\"{ascii_name}{suffix}\"; filename*=UTF-8''{quote(utf8_name, safe='')}"


@router.get("/{exam_id}/results")
def results(exam_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    exam = owned_exam(db, exam_id, user)
    qs = sorted(exam.questions, key=lambda q: q.question_no)
    items = []
    for sub in sorted(exam.submissions, key=lambda s: s.student_ref):
        by_q = {a.question_id: a for a in sub.answers}
        cells = []
        for q in qs:
            a = by_q.get(q.id)
            cells.append(None if a is None else {
                "answer_id": a.id, "marks": round(effective_marks(a), 2), "ai_marks": a.ai_marks,
                "status": a.status, "max_marks": q.max_marks,
            })
        pending = sum(1 for a in sub.answers if a.status in ("needs_review", "failed"))
        items.append({
            "submission_id": sub.id, "student_ref": sub.student_ref, "status": sub.status, "cells": cells,
            "total": sub.total_final_marks, "total_ai": sub.total_ai_marks, "pending": pending,
        })
    return {"questions": [{"id": q.id, "question_no": q.question_no, "max_marks": q.max_marks} for q in qs],
            "max_total": round(sum(q.max_marks for q in qs), 2), "items": items}


@router.get("/{exam_id}/analytics")
def analytics(exam_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    exam = owned_exam(db, exam_id, user)
    qs = sorted(exam.questions, key=lambda q: q.question_no)
    max_total = sum(q.max_marks for q in qs) or exam.total_marks
    subs = [s for s in exam.submissions if s.status in ("processed", "in_review", "approved")]
    totals = [s.total_final_marks or 0.0 for s in subs]
    bins = [0] * 10
    for t in totals:
        bins[min(9, int((t / max_total) * 10)) if max_total else 0] += 1

    per_q = []
    all_answers = [a for s in subs for a in s.answers]
    for q in qs:
        ans = [a for a in all_answers if a.question_id == q.id]
        marks = [effective_marks(a) for a in ans]
        missed: dict[str, int] = {}
        for a in ans:
            for kp in a.evidence or []:
                if kp.get("status") in ("missed", "contradicted"):
                    missed[kp["text"]] = missed.get(kp["text"], 0) + 1
        per_q.append({
            "question_no": q.question_no, "type": q.type, "max_marks": q.max_marks,
            "average": round(statistics.fmean(marks), 2) if marks else None,
            "average_pct": round(100 * statistics.fmean(marks) / q.max_marks, 1) if marks else None,
            "blank": sum(1 for a in ans if a.status == "blank"),
            "most_missed": sorted(missed.items(), key=lambda kv: -kv[1])[:3],
        })
    graded = [a for a in all_answers if a.status != "blank"]
    auto = [a for a in graded if a.status == "auto_accepted"]
    overridden = [a for a in all_answers if a.final_marks is not None and abs(a.final_marks - a.ai_marks) > 1e-6]
    reviewed = [a for a in all_answers if a.final_marks is not None]
    flag_counts: dict[str, int] = {}
    for a in all_answers:
        for f in a.flags or []:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    return {
        "sheets": len(subs), "max_total": max_total,
        "average": round(statistics.fmean(totals), 2) if totals else None,
        "median": round(statistics.median(totals), 2) if totals else None,
        "stdev": round(statistics.pstdev(totals), 2) if len(totals) > 1 else None,
        "highest": max(totals) if totals else None, "lowest": min(totals) if totals else None,
        "distribution": [{"from_pct": i * 10, "to_pct": (i + 1) * 10, "count": c} for i, c in enumerate(bins)],
        "questions": per_q,
        "hardest": [q["question_no"] for q in sorted((q for q in per_q if q["average_pct"] is not None),
                                                     key=lambda q: q["average_pct"])[:3]],
        "auto_accepted_share": round(len(auto) / len(graded), 3) if graded else None,
        "mean_confidence": round(statistics.fmean(a.confidence for a in graded), 3) if graded else None,
        "override_rate": round(len(overridden) / len(reviewed), 3) if reviewed else None,
        "mean_abs_override": round(statistics.fmean(abs(a.final_marks - a.ai_marks) for a in overridden), 2)
        if overridden else None,
        "flags": sorted(flag_counts.items(), key=lambda kv: -kv[1]),
    }
