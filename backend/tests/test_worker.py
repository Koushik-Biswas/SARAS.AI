"""Background processing (PRD 8, 18.3, 18.6): retries with exponential backoff,
failure after 3 attempts, stale jobs re-claimed, idempotent re-processing and
per-page isolation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app import worker
from app.db import SessionLocal
from app.models import Answer, AnswerRegion, Job, OcrLine, Submission
from conftest import assert_error, drain


@pytest.fixture
def uploaded(teacher, exam, sample_bytes) -> dict:
    r = teacher.upload(exam["id"], [("ML008_continuation.pdf", sample_bytes["ML008_continuation"])])
    assert r.status_code == 202
    return r.json()["submissions"][0]


def _job(sub_id: str) -> Job:
    with SessionLocal() as db:
        return db.query(Job).filter_by(submission_id=sub_id).order_by(Job.created_at.desc()).first()


def _make_due(sub_id: str) -> None:
    with SessionLocal() as db:
        for job in db.query(Job).filter_by(submission_id=sub_id, status="queued"):
            job.run_after = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()


def test_retries_with_backoff_then_fails_after_three_attempts(teacher, uploaded, monkeypatch):
    def boom(db, submission_id, settings):  # noqa: ANN001, ANN202
        raise RuntimeError("OCR engine crashed")

    monkeypatch.setattr(worker, "process_submission", boom)
    sid = uploaded["id"]
    for attempt in (1, 2):
        before = datetime.now(timezone.utc)
        assert worker.run_one() is True
        job = _job(sid)
        assert job.status == "queued" and job.attempts == attempt
        delay = (job.run_after - before).total_seconds()
        assert 2 ** attempt - 1 <= delay <= 2 ** attempt + 1  # exponential backoff: 2 s, then 4 s
        status = teacher.get(f"/submissions/{sid}/status").json()
        assert status["status"] == "uploaded" and status["stage"] == "retrying" and status["attempts"] == attempt
        assert worker.run_one() is False  # not due yet
        _make_due(sid)

    assert worker.run_one() is True
    job = _job(sid)
    assert job.status == "failed" and job.attempts == 3
    sub = teacher.submission(sid)
    assert sub["status"] == "failed" and sub["error"] == "RuntimeError: OCR engine crashed"
    status = teacher.get(f"/submissions/{sid}/status").json()
    assert status["errors"] == ["RuntimeError: OCR engine crashed"] and status["attempts"] == 3
    assert worker.run_one() is False

    # The teacher retries once the cause is fixed.
    monkeypatch.undo()
    r = teacher.post(f"/submissions/{sid}/retry")
    assert r.status_code == 202 and r.json()["status"] == "uploaded"
    assert drain() == 1
    assert teacher.submission(sid)["status"] == "processed"
    # Once a teacher has reviewed something, re-processing would discard it: refused.
    q1 = teacher.answer_for(sid, 1)
    assert teacher.patch(f"/answers/{q1['answer_id']}/review", json={}).status_code == 200
    assert_error(teacher.post(f"/submissions/{sid}/retry"), 409, "conflict")


def test_stale_running_job_is_reclaimed(teacher, uploaded, settings):
    sid = uploaded["id"]
    with SessionLocal() as db:
        job = db.query(Job).filter_by(submission_id=sid).one()
        job.status, job.attempts = "running", 1
        job.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=10)  # alive: recent heartbeat
        db.commit()
    assert worker.run_one() is False  # a live job is never taken twice

    with SessionLocal() as db:
        job = db.query(Job).filter_by(submission_id=sid).one()
        job.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=settings.job_stale_seconds + 60)
        db.commit()
    assert worker.run_one() is True  # the crashed worker's job is re-claimed
    job = _job(sid)
    assert job.status == "done" and job.attempts == 2
    assert teacher.submission(sid)["status"] == "processed"


def _snapshot(sid: str) -> dict:
    with SessionLocal() as db:
        answers = sorted((a.question.question_no, a.status, a.ai_marks, a.ocr_text, tuple(a.flags))
                         for a in db.query(Answer).filter_by(submission_id=sid))
        regions = sorted((r.page_no, tuple(r.bbox), r.question_id) for r in
                         db.query(AnswerRegion).filter_by(submission_id=sid))
        lines = db.query(OcrLine).filter_by(submission_id=sid).count()
        sub = db.get(Submission, sid)
        return {"answers": answers, "regions": regions, "lines": lines, "total": sub.total_ai_marks,
                "flags": sub.flags, "status": sub.status}


def test_processing_is_idempotent(uploaded, settings):
    from app.pipeline.processor import process_submission

    sid = uploaded["id"]
    drain()
    first = _snapshot(sid)
    assert first["status"] == "processed" and len(first["answers"]) == 4 and first["lines"] > 0
    with SessionLocal() as db:  # a crash mid-run leaves the sheet "processing" with partial output
        db.get(Submission, sid).status = "processing"
        db.commit()
        process_submission(db, sid, settings)
    assert _snapshot(sid) == first  # rebuilt from scratch, no duplicates


def test_one_bad_page_is_flagged_while_the_rest_finishes(teacher, uploaded, monkeypatch):
    from app.pipeline import processor

    real = processor.render_page

    def flaky(doc, index, dpi):  # noqa: ANN001, ANN202
        if index == 1:
            raise ValueError("damaged page stream")
        return real(doc, index, dpi)

    monkeypatch.setattr(processor, "render_page", flaky)
    sid = uploaded["id"]
    assert drain() == 1
    sub = teacher.submission(sid)
    assert sub["status"] == "processed" and sub["pages_done"] == 2
    assert sub["page_flags"] == {"2": ["page_failed"]} and "has_failed_pages" in sub["flags"]
    status = teacher.get(f"/submissions/{sid}/status").json()
    assert "Page 2 could not be processed" in status["errors"]

    answers = {a["question_no"]: a for a in teacher.answers(sid)}
    # Page 1 was graded normally. Its route depends on the grader's confidence (lexical-only
    # answers go to review at the default threshold), not on the failed page: no page flags.
    assert answers[1]["ai_marks"] == 3 and answers[1]["grader"] and answers[1]["verdict"] == "correct"
    assert answers[1]["status"] in ("auto_accepted", "needs_review")
    assert not {"next_page_failed", "segmentation_warning", "missing_marker", "page_failed"} & set(answers[1]["flags"])
    # Q2 was still being written when page 2 failed: its text may be cut short.
    assert answers[2]["status"] == "needs_review"
    assert {"next_page_failed", "segmentation_warning"} <= set(answers[2]["flags"])
    # Q3/Q4 were on the failed page: they must be reviewed, never silently zeroed.
    for no in (3, 4):
        assert answers[no]["status"] == "needs_review" and "missing_marker" in answers[no]["flags"]
    assert_error(teacher.post(f"/submissions/{sid}/approve"), 409, "conflict")
    failed = [r for r in teacher.regions(sid) if "page_failed" in r["flags"]]
    assert len(failed) == 1 and failed[0]["page_no"] == 2


def test_deleted_submission_job_is_harmless(teacher, uploaded):
    assert teacher.delete(f"/submissions/{uploaded['id']}").status_code == 204
    assert drain() == 0  # its job went with it (cascade)
