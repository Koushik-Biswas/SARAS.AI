"""Request/response contracts (PRD 16.1). Pydantic models are the source of truth
for the OpenAPI schema served at /docs."""

from __future__ import annotations

from datetime import date, datetime

# Exam models have a field called "date"; annotate it with this alias so the
# field name never shadows the type while pydantic resolves annotations.
Date = date
from typing import Literal

import email_validator
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

# Accept intranet addresses on ".local" (school LAN mail, the dev demo account
# demo@saras.local); the library rejects them by default as a special-use name.
# This is the library's documented switch; the other special-use names stay blocked.
if "local" in email_validator.SPECIAL_USE_DOMAIN_NAMES:
    email_validator.SPECIAL_USE_DOMAIN_NAMES.remove("local")

QuestionType = Literal["short", "long", "paragraph", "maths"]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------- auth ----------
class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class RefreshIn(BaseModel):
    refresh_token: str


class UserOut(ORM):
    id: str
    name: str
    email: str
    role: str


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


# ---------- exams ----------
class ExamIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    subject: str | None = Field(default=None, max_length=120)
    date: Date | None = None
    total_marks: float = Field(gt=0, le=10000)


class ExamPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    subject: str | None = Field(default=None, max_length=120)
    date: Date | None = None
    total_marks: float | None = Field(default=None, gt=0, le=10000)


class ExamStats(BaseModel):
    questions: int = 0
    sheets: int = 0
    processed: int = 0
    processing: int = 0
    failed: int = 0
    approved: int = 0
    needs_review: int = 0
    average_marks: float | None = None


class ExamOut(BaseModel):
    id: str
    title: str
    subject: str | None
    date: Date | None
    total_marks: float
    created_at: datetime
    rubric_marks: float = 0
    stats: ExamStats = ExamStats()
    # Whether the caller may edit this exam (its rubric, sheets and marks): only its owner.
    can_edit: bool = True


class KeyPointIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    weight: float = Field(gt=0, le=1000)
    keywords: list[str] = Field(default_factory=list, max_length=30)


class QuestionIn(BaseModel):
    question_no: int = Field(ge=1, le=99)
    type: QuestionType
    max_marks: float = Field(gt=0, le=1000)
    prompt: str | None = Field(default=None, max_length=5000)
    model_answer: str | None = Field(default=None, max_length=20000)
    key_points: list[KeyPointIn] = Field(default_factory=list, max_length=40)

    @model_validator(mode="after")
    def _weights_sum(self) -> "QuestionIn":
        if self.key_points:
            total = round(sum(k.weight for k in self.key_points), 2)
            if abs(total - self.max_marks) > 0.01:
                raise ValueError(f"key-point weights sum to {total} but max_marks is {self.max_marks}")
        elif not (self.model_answer or "").strip() and not (self.prompt or "").strip():
            # Question-only rubric: the AI marks from the question text and max marks.
            raise ValueError("add the question text")
        return self


class QuestionsIn(BaseModel):
    questions: list[QuestionIn] = Field(min_length=1, max_length=99)

    @field_validator("questions")
    @classmethod
    def _unique_numbers(cls, v: list[QuestionIn]) -> list[QuestionIn]:
        nums = [q.question_no for q in v]
        dupes = sorted({n for n in nums if nums.count(n) > 1})
        if dupes:
            raise ValueError(f"question_no must be unique per exam; duplicated: {dupes}")
        return v


class QuestionOut(ORM):
    id: str
    question_no: int
    type: str
    max_marks: float
    prompt: str | None
    model_answer: str | None
    key_points: list[dict]


# ---------- submissions ----------
class SubmissionOut(ORM):
    id: str
    exam_id: str
    student_ref: str
    original_filename: str | None
    status: str
    stage: str | None
    page_count: int
    pages_done: int
    error: str | None
    total_ai_marks: float | None
    total_final_marks: float | None
    ocr_engine: str | None
    grader: str | None
    flags: list[str] = []
    page_flags: dict = {}
    timings: dict = {}
    progress: dict | None = None
    created_at: datetime
    approved_at: datetime | None
    answers_needing_review: int = 0


class StatusOut(BaseModel):
    id: str
    status: str
    stage: str | None
    pages_done: int
    pages_total: int
    errors: list[str]
    attempts: int = 0
    progress: dict | None = None


class UploadOut(BaseModel):
    submissions: list[SubmissionOut]
    rejected: list[dict] = []


class KeyPointOut(BaseModel):
    index: int
    text: str
    weight: float
    matched: bool
    status: str
    awarded: float
    entailment_prob: float
    contradiction_prob: float
    evidence_sentence: str | None
    # Additive (grading ensemble, docs/GRADING_DESIGN.md). reason: the engine's reason code
    # (entailed, no_support, role_swap, ...), the AI marker's short reason, or
    # no_verified_evidence. source: the grader whose status stands. disputed: the graders
    # disagreed on whether this criterion earns marks. statuses_by_grader: every grader's status.
    reason: str = ""
    source: str | None = None
    disputed: bool = False
    statuses_by_grader: dict[str, str] = Field(default_factory=dict)


class AnswerOut(BaseModel):
    answer_id: str
    submission_id: str
    question_id: str
    question_no: int
    question_type: str
    question_prompt: str | None
    status: str
    ocr_text: str
    text_corrected: bool
    ocr_confidence: float
    ai_marks: float
    final_marks: float | None
    max_marks: float
    confidence: float
    grading_confidence: float
    segmentation_confidence: float
    grader: str | None
    key_points: list[KeyPointOut]
    matched_points: list[str]
    missed_points: list[str]
    flags: list[str]
    comment: str | None
    region_ids: list[str]
    locked: bool
    # Additive (grading ensemble): correct | partially_correct | incorrect | ungraded (None when
    # not graded, e.g. blank); feedback from the AI marker or a summary of the criteria; and
    # the decision trace (graders used, AI status, disputes, per-criterion statuses by grader,
    # routing) - scores and codes only, never answer text.
    verdict: str | None = None
    feedback: str | None = None
    grading_trace: dict | None = None


class ReviewIn(BaseModel):
    final_marks: float | None = Field(default=None, ge=0)
    corrected_text: str | None = Field(default=None, max_length=20000)
    comment: str | None = Field(default=None, max_length=2000)
    reason: str | None = Field(default=None, max_length=1000)
    approve: bool = True  # False = save text/comment only, keep in queue


class RegionOut(ORM):
    id: str
    page_no: int
    bbox: list[int]
    question_id: str | None
    source: str
    is_blank: bool
    segmentation_confidence: float
    flags: list[str]


class RegionPatch(BaseModel):
    action: Literal["crop", "reassign", "split", "merge", "blank", "unblank"]
    bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    question_id: str | None = None
    split_y: int | None = Field(default=None, ge=1)
    merge_with: str | None = None
    reason: str | None = Field(default=None, max_length=500)


class RegionCreate(BaseModel):
    page_no: int = Field(ge=1)
    bbox: list[int] = Field(min_length=4, max_length=4)
    question_id: str | None = None


class AuditOut(ORM):
    id: int
    user_id: str | None
    entity: str
    entity_id: str
    action: str
    old_value: dict | None
    new_value: dict | None
    reason: str | None
    created_at: datetime


class Page(BaseModel):
    items: list
    total: int
    page: int
    page_size: int


class FlagIn(BaseModel):
    reason: str | None = Field(default=None, max_length=1000)


class ReopenIn(BaseModel):
    reason: str = Field(min_length=1, max_length=1000)


class RegionEditOut(BaseModel):
    region: RegionOut | None
    regions: list[RegionOut]
    answers: list[AnswerOut]
