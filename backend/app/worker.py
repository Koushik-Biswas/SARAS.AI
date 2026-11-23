"""Database-backed job worker (low-resource profile, PRD 18.6).

One background thread claims queued jobs from the `jobs` table, processes one
sheet at a time, retries with exponential backoff (up to 3 attempts) and
re-queues jobs whose heartbeat went stale after a crash. Jobs are idempotent, so
re-running one is always safe: a sheet that already has results (processed, in
review or approved) is never rebuilt, and a job that keeps crashing the process
(e.g. out of memory) fails after the same attempt limit instead of looping."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import or_, select, update

from .config import get_settings
from .db import SessionLocal
from .models import Job, Submission
from .pipeline.processor import PROCESSABLE, process_submission, remove_submission_storage

log = logging.getLogger(__name__)

CRASHED = "The worker stopped while processing this sheet (crash or out of memory)"
STEPS = ("ocr", "segmentation", "grading")  # processing steps a failure can be reported at


def _failed_stage(*candidates: str | None) -> str:
    """Keep the step a sheet stopped at (the status page names it); "failed" otherwise."""
    return next((c for c in candidates if c in STEPS), "failed")


def enqueue(db, submission_id: str) -> Job:  # noqa: ANN001
    job = Job(submission_id=submission_id)
    db.add(job)
    return job


def _fail_exhausted(db, stale: datetime, max_attempts: int) -> None:  # noqa: ANN001
    """A stale `running` job means the process died mid-job. Once that has used up every
    attempt, the job fails (and its sheet with it) instead of being re-claimed forever."""
    jobs = db.scalars(select(Job).where(Job.status == "running", Job.heartbeat_at < stale,
                                        Job.attempts >= max_attempts)).all()
    for job in jobs:
        sub = db.get(Submission, job.submission_id)
        finished = sub is not None and sub.status not in PROCESSABLE
        message = f"{CRASHED} {job.attempts} times"
        res = db.execute(
            update(Job).where(Job.id == job.id, Job.status == "running", Job.attempts == job.attempts)
            .values(status="done" if finished else "failed", last_error=None if finished else message)
        )
        if res.rowcount == 1 and sub is not None and not finished:
            # The stage is kept so the status page can say which step it stopped at.
            sub.status, sub.stage, sub.error = "failed", _failed_stage(sub.stage), message
    db.commit()


def _claim(db) -> Job | None:  # noqa: ANN001
    s = get_settings()
    now = datetime.now(timezone.utc)
    stale = now - timedelta(seconds=s.job_stale_seconds)
    _fail_exhausted(db, stale, s.job_max_attempts)
    candidate = db.execute(
        select(Job).where(or_(
            (Job.status == "queued") & (Job.run_after <= now),
            (Job.status == "running") & (Job.heartbeat_at < stale),
        )).order_by(Job.created_at).limit(1)
    ).scalar_one_or_none()
    if candidate is None:
        return None
    # Optimistic claim so two workers never take the same job.
    res = db.execute(
        update(Job).where(Job.id == candidate.id, Job.status == candidate.status,
                          Job.attempts == candidate.attempts)
        .values(status="running", attempts=Job.attempts + 1, heartbeat_at=now)
    )
    db.commit()
    if res.rowcount != 1:
        return None
    db.refresh(candidate)
    return candidate


def _stage_in_session(db, submission_id: str) -> str | None:  # noqa: ANN001
    """The stage the failed run had reached (possibly not committed), read without
    touching the database: the session may be unusable until rolled back."""
    for obj in list(db.identity_map.values()):
        if isinstance(obj, Submission):
            state = sa_inspect(obj)
            if state.identity and state.identity[0] == submission_id:
                return state.dict.get("stage")
    return None


def run_one() -> bool:
    """Process at most one job. Returns True if a job was handled."""
    s = get_settings()
    db = SessionLocal()
    try:
        job = _claim(db)
        if job is None:
            return False
        job_id, submission_id = job.id, job.submission_id
        try:
            process_submission(db, submission_id, s)
            job.status, job.last_error = "done", None
            db.commit()
        except Exception as exc:
            stage = _stage_in_session(db, submission_id)
            db.rollback()
            job = db.get(Job, job_id)  # None if the sheet (and its jobs) were deleted meanwhile
            sub = db.get(Submission, submission_id)
            if job is None or sub is None:
                # Deleted mid-processing: remove whatever the run wrote to its folder (the
                # API could not delete the PDF while this worker held it open).
                log.info("Submission %s was deleted while processing", submission_id)
                remove_submission_storage(s.storage_dir, submission_id)
                return True
            log.exception("Job %s failed (attempt %s)", job.id, job.attempts)
            job.last_error = f"{exc.__class__.__name__}: {exc}"[:500]
            if sub.status not in PROCESSABLE:
                # The sheet already finished (only the job bookkeeping failed): never
                # overwrite its results or send it back for a rebuild.
                job.status = "done"
            elif job.attempts >= s.job_max_attempts:
                job.status = "failed"
                sub.status, sub.error = "failed", job.last_error
                sub.stage = _failed_stage(stage, sub.stage)  # where it stopped, e.g. "grading"
            else:
                job.status = "queued"
                job.run_after = datetime.now(timezone.utc) + timedelta(seconds=2 ** job.attempts)
                sub.status, sub.stage, sub.error = "uploaded", "retrying", job.last_error
            db.commit()
        return True
    finally:
        db.close()


def run_pending(max_jobs: int = 1000) -> int:
    """Drain the queue synchronously (tests and CLI)."""
    n = 0
    while n < max_jobs and run_one():
        n += 1
    return n


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="saras-worker", daemon=True)
        self._halt = threading.Event()

    def run(self) -> None:
        log.info("Worker started")
        poll = get_settings().worker_poll_seconds
        while not self._halt.is_set():
            try:
                busy = run_one()
            except Exception:  # never let the thread die
                log.exception("Worker loop error")
                busy = False
            if not busy:
                self._halt.wait(poll)

    def stop(self) -> None:
        self._halt.set()
