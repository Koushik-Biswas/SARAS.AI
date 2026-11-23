"""Ownership checks: a teacher sees only their own exams; an admin can read
everything but cannot edit marks (PRD 22.1)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..errors import ApiError, forbidden, not_found
from ..models import Answer, AnswerRegion, Exam, Submission, User


def owned_exam(db: Session, exam_id: str, user: User, *, write: bool = False) -> Exam:
    exam = db.get(Exam, exam_id)
    if exam is None:
        raise not_found("Exam")
    if exam.teacher_id != user.id and not (user.role == "admin" and not write):
        raise forbidden()
    return exam


def owned_submission(db: Session, submission_id: str, user: User, *, write: bool = False) -> Submission:
    sub = db.get(Submission, submission_id)
    if sub is None:
        raise not_found("Submission")
    owned_exam(db, sub.exam_id, user, write=write)
    return sub


def owned_answer(db: Session, answer_id: str, user: User, *, write: bool = False) -> Answer:
    ans = db.get(Answer, answer_id)
    if ans is None:
        raise not_found("Answer")
    owned_submission(db, ans.submission_id, user, write=write)
    return ans


def owned_region(db: Session, region_id: str, user: User, *, write: bool = False) -> AnswerRegion:
    reg = db.get(AnswerRegion, region_id)
    if reg is None:
        raise not_found("Region")
    owned_submission(db, reg.submission_id, user, write=write)
    return reg


def ensure_unlocked(sub: Submission) -> None:
    if sub.status == "approved":
        raise ApiError(409, "conflict", "Marks for this sheet are approved and locked")
    if sub.status in ("uploaded", "processing"):
        raise ApiError(409, "conflict", "This sheet is still being processed")
