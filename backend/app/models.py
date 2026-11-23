"""ORM models — PRD Section 15 (data model, constraints and status values)."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


UUID = String(36)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC timestamps on every backend. SQLite drops tzinfo, so
    values are normalised to UTC on the way in and re-tagged as UTC on the way
    out; the API then always emits ISO 8601 with an explicit offset."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):  # noqa: ANN001, ANN201
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(timezone.utc)
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value, dialect):  # noqa: ANN001, ANN201
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

QUESTION_TYPES = ("short", "long", "paragraph", "maths")
SUBMISSION_STATUSES = ("uploaded", "processing", "processed", "in_review", "approved", "failed")
ANSWER_STATUSES = ("auto_accepted", "needs_review", "reviewed", "blank", "failed")


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(16), default="teacher")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    __table_args__ = (CheckConstraint("role IN ('teacher','admin')", name="ck_users_role"),)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id: Mapped[str] = mapped_column(UUID, primary_key=True)  # the token's jti
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime())
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class Exam(Base):
    __tablename__ = "exams"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_id)
    teacher_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    subject: Mapped[str | None] = mapped_column(String(120))
    exam_date: Mapped[date | None] = mapped_column(Date)
    total_marks: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    questions: Mapped[list[Question]] = relationship(
        back_populates="exam", cascade="all, delete-orphan", order_by="Question.question_no"
    )
    submissions: Mapped[list[Submission]] = relationship(back_populates="exam", cascade="all, delete-orphan")

    __table_args__ = (CheckConstraint("total_marks > 0", name="ck_exams_total_marks"),)


class Question(Base):
    __tablename__ = "questions"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_id)
    exam_id: Mapped[str] = mapped_column(ForeignKey("exams.id", ondelete="CASCADE"), index=True)
    question_no: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(16))
    max_marks: Mapped[float] = mapped_column(Float)
    prompt: Mapped[str | None] = mapped_column(Text)  # question text; used to detect copied-question answers
    model_answer: Mapped[str | None] = mapped_column(Text)
    # [{"text": str, "weight": float, "keywords": [str]}]; weights sum to max_marks (checked in the API)
    key_points: Mapped[list] = mapped_column(JSON, default=list)

    exam: Mapped[Exam] = relationship(back_populates="questions")

    __table_args__ = (
        UniqueConstraint("exam_id", "question_no", name="uq_questions_exam_no"),
        CheckConstraint("max_marks > 0", name="ck_questions_max_marks"),
        CheckConstraint("type IN ('short','long','paragraph','maths')", name="ck_questions_type"),
    )


class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_id)
    exam_id: Mapped[str] = mapped_column(ForeignKey("exams.id", ondelete="CASCADE"))
    student_ref: Mapped[str] = mapped_column(String(64))
    original_filename: Mapped[str | None] = mapped_column(String(255))
    pdf_path: Mapped[str] = mapped_column(Text)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    pages_done: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="uploaded")
    stage: Mapped[str | None] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(Text)
    total_ai_marks: Mapped[float | None] = mapped_column(Float)
    total_final_marks: Mapped[float | None] = mapped_column(Float)
    ocr_engine: Mapped[str | None] = mapped_column(String(32))
    grader: Mapped[str | None] = mapped_column(String(32))
    timings: Mapped[dict] = mapped_column(JSON, default=dict)  # per-stage seconds (PRD 18.2)
    flags: Mapped[list] = mapped_column(JSON, default=list)  # sheet-level warnings
    page_flags: Mapped[dict] = mapped_column(JSON, default=dict)  # {"3": ["blank_page"]}
    # Live processing progress (app.pipeline.progress): stages, counts, events.
    progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    approved_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    exam: Mapped[Exam] = relationship(back_populates="submissions")
    answers: Mapped[list[Answer]] = relationship(back_populates="submission", cascade="all, delete-orphan")
    regions: Mapped[list[AnswerRegion]] = relationship(back_populates="submission", cascade="all, delete-orphan")
    lines: Mapped[list[OcrLine]] = relationship(back_populates="submission", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("exam_id", "student_ref", name="uq_submissions_exam_student"),
        Index("ix_submissions_exam_status", "exam_id", "status"),
        CheckConstraint(
            "status IN ('uploaded','processing','processed','in_review','approved','failed')",
            name="ck_submissions_status",
        ),
    )


class OcrLine(Base):
    """Every recognised line is kept so regions can be re-cropped or reassigned
    without running OCR again."""

    __tablename__ = "ocr_lines"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_id)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer)
    line_no: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[list] = mapped_column(JSON)  # [x, y, w, h] in page-image pixels
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)

    submission: Mapped[Submission] = relationship(back_populates="lines")


class AnswerRegion(Base):
    __tablename__ = "answer_regions"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_id)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[list] = mapped_column(JSON)  # [x, y, w, h]
    question_id: Mapped[str | None] = mapped_column(ForeignKey("questions.id", ondelete="SET NULL"))
    source: Mapped[str] = mapped_column(String(8), default="auto")
    is_blank: Mapped[bool] = mapped_column(Boolean, default=False)
    segmentation_confidence: Mapped[float] = mapped_column(Float, default=1.0)
    flags: Mapped[list] = mapped_column(JSON, default=list)

    submission: Mapped[Submission] = relationship(back_populates="regions")

    __table_args__ = (
        CheckConstraint("source IN ('auto','manual')", name="ck_regions_source"),
        CheckConstraint("segmentation_confidence BETWEEN 0 AND 1", name="ck_regions_conf"),
    )


class Answer(Base):
    __tablename__ = "answers"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_id)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id", ondelete="CASCADE"), index=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id", ondelete="CASCADE"), index=True)
    ocr_text: Mapped[str] = mapped_column(Text, default="")
    text_corrected: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    ai_marks: Mapped[float] = mapped_column(Float, default=0.0)
    final_marks: Mapped[float | None] = mapped_column(Float)
    grading_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    segmentation_confidence: Mapped[float] = mapped_column(Float, default=1.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence: Mapped[list] = mapped_column(JSON, default=list)  # per key point (PRD 12.11)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    grader: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="needs_review")
    comment: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    # Grading ensemble output (docs/GRADING_DESIGN.md). Nullable and additive: rows graded
    # before these columns existed keep NULL (db.ensure_additive_columns adds them).
    verdict: Mapped[str | None] = mapped_column(Text, nullable=True)  # correct | partially_correct | incorrect | ungraded
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)  # the AI's, or a summary of the criteria
    # Decisions and scores only: never answer text, student_ref, names or keys.
    grading_trace: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    submission: Mapped[Submission] = relationship(back_populates="answers")
    question: Mapped[Question] = relationship()
    reviews: Mapped[list[Review]] = relationship(
        back_populates="answer", cascade="all, delete-orphan", order_by="Review.created_at"
    )

    __table_args__ = (
        UniqueConstraint("submission_id", "question_id", name="uq_answers_submission_question"),
        CheckConstraint(
            "status IN ('auto_accepted','needs_review','reviewed','blank','failed')", name="ck_answers_status"
        ),
        CheckConstraint("ocr_confidence BETWEEN 0 AND 1", name="ck_answers_ocr_conf"),
        CheckConstraint("grading_confidence BETWEEN 0 AND 1", name="ck_answers_grading_conf"),
        CheckConstraint("ai_marks >= 0", name="ck_answers_ai_marks"),
    )


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_id)
    answer_id: Mapped[str] = mapped_column(ForeignKey("answers.id", ondelete="CASCADE"), index=True)
    teacher_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    final_marks: Mapped[float] = mapped_column(Float)
    comment: Mapped[str | None] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    answer: Mapped[Answer] = relationship(back_populates="reviews")


class Correction(Base):
    """Teacher corrections kept for later retraining (FR-19, FR-20)."""

    __tablename__ = "corrections"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_id)
    answer_id: Mapped[str] = mapped_column(ForeignKey("answers.id", ondelete="CASCADE"), index=True)
    original_text: Mapped[str | None] = mapped_column(Text)
    corrected_text: Mapped[str | None] = mapped_column(Text)
    original_marks: Mapped[float | None] = mapped_column(Float)
    corrected_marks: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class AuditLog(Base):
    """Append-only (enforced by triggers on SQLite, by role privileges on PostgreSQL)."""

    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    entity: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str] = mapped_column(String(36))
    action: Mapped[str] = mapped_column(String(32))
    old_value: Mapped[dict | None] = mapped_column(JSON)
    new_value: Mapped[dict | None] = mapped_column(JSON)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    __table_args__ = (Index("ix_audit_entity", "entity", "entity_id", "created_at"),)


class Job(Base):
    """Database-backed job table (low-resource profile replacement for Celery)."""

    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_id)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="process")
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued | running | done | failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    run_after: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    heartbeat_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    __table_args__ = (Index("ix_jobs_status_run_after", "status", "run_after"),)


AI_RUN_STATUSES = ("ok", "error", "invalid", "rate_limited", "cache_hit")


class AiRun(Base):
    """One row per AI (LLM) provider call or cache hit (PRD 13.5; spec section 5).

    The row is also the response cache (status 'ok', looked up by cache_key) and the
    source of the daily per-provider budgets. It stores only the anonymised prompt
    and schema: never API keys, student_ref or names."""

    __tablename__ = "ai_runs"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    task: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(32))
    # sha256 of task + prompt_version + request content (system, prompt, schema, images)
    cache_key: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(String(300))  # short, secret-free
    request_json: Mapped[dict | None] = mapped_column(JSON)  # anonymised prompt + schema
    response_json: Mapped[dict | None] = mapped_column(JSON)  # the model's JSON output
    # Deleting an exam or a sheet also removes its AI runs (deletion on request, P-7).
    exam_id: Mapped[str | None] = mapped_column(ForeignKey("exams.id", ondelete="CASCADE"), index=True)
    submission_id: Mapped[str | None] = mapped_column(ForeignKey("submissions.id", ondelete="CASCADE"), index=True)

    __table_args__ = (
        Index("ix_ai_runs_provider_model_created", "provider", "model", "created_at"),
        CheckConstraint(
            "status IN ('ok','error','invalid','rate_limited','cache_hit')", name="ck_ai_runs_status"
        ),
    )
