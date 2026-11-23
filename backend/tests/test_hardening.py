"""Regression tests for verified review findings: marks integrity (stale jobs never
rebuild finished sheets, no silent zeros, max marks vs. reviewed marks, re-graded
corrections), the audit trail for regrades and region edits, CSV safety, upload and
PDF resource limits, rate limiting, refresh-token rotation and exam ownership."""

from __future__ import annotations

import copy
import csv
import io
import urllib.parse
from datetime import datetime, timedelta, timezone

import pytest

from app import worker
from app.db import SessionLocal
from app.errors import ApiError
from app.models import Job, OcrLine, Question, RefreshToken, Review, Submission
from conftest import API, PASSWORD, accept_lexical, assert_error, drain, error_fields, register
from scripts import make_samples as samples

A = samples.ANSWERS


def _process_pdf(teacher, exam_id: str, pages: list[list[str]], ref: str = "S1") -> dict:  # noqa: ANN001
    r = teacher.upload(exam_id, [(f"{ref}.pdf", samples.build_text_pdf(pages))], refs=[ref])
    assert r.status_code == 202, r.text
    drain()
    sub = teacher.submission(r.json()["submissions"][0]["id"])
    assert sub["status"] == "processed", sub
    return sub


def _make_due(sub_id: str) -> None:
    with SessionLocal() as db:
        for job in db.query(Job).filter_by(submission_id=sub_id, status="queued"):
            job.run_after = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()


# ------------------------------------------------------------------ worker (finding 0, 33)
@accept_lexical  # the sheet must be approvable after reviewing its one flagged answer (conftest)
@pytest.mark.parametrize("attempts", [1, 3])
def test_stale_job_never_rebuilds_an_approved_sheet(teacher, process, attempts):
    sid = process("ML002_partial")["ML002_partial"]["id"]
    q2 = teacher.answer_for(sid, 2)
    r = teacher.patch(f"/answers/{q2['answer_id']}/review", json={"final_marks": 4, "reason": "KP2 explained"})
    assert r.status_code == 200, r.text
    assert teacher.post(f"/submissions/{sid}/approve").status_code == 200
    before = teacher.answers(sid)
    with SessionLocal() as db:  # a job left "running" by a worker that died long ago
        db.add(Job(submission_id=sid, status="running", attempts=attempts,
                   heartbeat_at=datetime.now(timezone.utc) - timedelta(hours=1)))
        db.commit()

    worker.run_one()
    sub = teacher.submission(sid)
    assert sub["status"] == "approved" and sub["approved_at"]
    assert teacher.answers(sid) == before  # reviews and approved marks untouched
    assert teacher.answer_for(sid, 2)["final_marks"] == 4
    with SessionLocal() as db:
        assert {j.status for j in db.query(Job).filter_by(submission_id=sid)} == {"done"}
        assert db.query(Review).count() == 1
    assert worker.run_one() is False


def test_job_that_keeps_killing_the_worker_fails_instead_of_looping(teacher, exam, sample_bytes, settings):
    r = teacher.upload(exam["id"], [("a.pdf", sample_bytes["ML001_full_marks"])])
    sid = r.json()["submissions"][0]["id"]
    with SessionLocal() as db:  # the process died (e.g. out of memory) during OCR on its last attempt
        job = db.query(Job).filter_by(submission_id=sid).one()
        job.status, job.attempts = "running", settings.job_max_attempts
        job.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=settings.job_stale_seconds + 60)
        sub = db.get(Submission, sid)
        sub.status, sub.stage = "processing", "ocr"
        db.commit()

    assert worker.run_one() is False  # failed, never re-claimed
    with SessionLocal() as db:
        job = db.query(Job).filter_by(submission_id=sid).one()
        assert job.status == "failed" and "crash" in job.last_error
    sub = teacher.submission(sid)
    assert sub["status"] == "failed" and sub["stage"] == "ocr" and "crash" in sub["error"]
    assert worker.run_one() is False
    assert teacher.post(f"/submissions/{sid}/retry").status_code == 202  # the teacher can still retry
    assert drain() == 1 and teacher.submission(sid)["status"] == "processed"


def test_failed_sheet_keeps_the_step_it_stopped_at(teacher, exam, sample_bytes, monkeypatch):
    from app.pipeline import processor

    def boom(question, text, settings):  # noqa: ANN001, ANN202
        raise RuntimeError("grader crashed")

    monkeypatch.setattr(processor, "_grade_with_fallback", boom)
    sid = teacher.upload(exam["id"], [("a.pdf", sample_bytes["ML001_full_marks"])]).json()["submissions"][0]["id"]
    for _ in range(3):
        _make_due(sid)
        assert worker.run_one() is True
    sub = teacher.submission(sid)
    assert sub["status"] == "failed" and sub["stage"] == "grading"  # not the meaningless "failed"


# ------------------------------------------------------------------ silent zeros (findings 1, 2, 6, 7)
def test_segmentation_reports_unassigned_text_and_missing_last_marker():
    from app.pipeline.ocr import OcrLine as Line
    from app.pipeline.segment import segment

    texts = ["Supervised learning uses labelled data", "Q2 overfitting", "Q3 gradient"]
    res = segment([Line(1, (60, 100 + 40 * i, 900, 30), t, 0.99) for i, t in enumerate(texts)], [1, 2, 3, 4])
    assert "unassigned_text" in res.sheet_flags
    assert res.missing == [1, 4] and res.missing_after_last == [4]


@accept_lexical  # approval / routing flow: clean lexical answers must be acceptable (conftest)
def test_answer_whose_marker_was_misread_goes_to_review(teacher, exam):
    sub = _process_pdf(teacher, exam["id"], [[A["q1_full"], f"Q2. {A['q2_full']}", f"Q3. {A['q3_full']}",
                                               f"Q4. {A['q4_full']}"]])
    assert "unassigned_text" in sub["flags"]
    q1 = teacher.answer_for(sub["id"], 1)
    assert q1["status"] == "needs_review" and q1["ai_marks"] == 0 and q1["grader"] is None
    assert {"missing_marker", "segmentation_warning"} <= set(q1["flags"])
    err = assert_error(teacher.post(f"/submissions/{sub['id']}/approve"), 409, "conflict")
    assert err["details"]["pending"] == [1]

    assert teacher.post(f"/exams/{exam['id']}/regrade").status_code == 200
    assert teacher.answer_for(sub["id"], 1)["status"] == "needs_review"  # still doubted after a regrade
    stray = next(r for r in teacher.regions(sub["id"]) if r["question_id"] is None)
    r = teacher.patch(f"/regions/{stray['id']}", json={"action": "reassign", "question_id": exam["qid"][1]})
    assert r.status_code == 200, r.text
    q1 = next(a for a in r.json()["answers"] if a["question_no"] == 1)
    assert q1["ai_marks"] == 3 and q1["grader"] is not None


SHORT_RUBRIC = {"questions": [
    {"question_no": 1, "type": "short", "max_marks": 2, "prompt": "What is six times seven?", "model_answer": "42",
     "key_points": [{"text": "42", "weight": 2, "keywords": ["42"]}]},
    {"question_no": 2, "type": "maths", "max_marks": 3, "prompt": "Solve x + 3 = 10.", "model_answer": "x = 7"},
    {"question_no": 3, "type": "short", "max_marks": 2, "prompt": "Name a metal.", "model_answer": "Iron is a metal."},
]}


def test_short_answers_are_graded_or_reviewed_never_blanked(teacher):
    exam = teacher.create_exam(title="Short answers", total_marks=7)
    teacher.put_rubric(exam["id"], SHORT_RUBRIC)
    sub = _process_pdf(teacher, exam["id"], [["Q1. 42", "Q2. 7", "Q3. -"]])
    a = {x["question_no"]: x for x in teacher.answers(sub["id"])}
    assert a[1]["ocr_text"] == "42" and a[1]["grader"] is not None
    assert a[1]["status"] == "needs_review" and "short_answer" in a[1]["flags"]
    assert a[2]["ocr_text"] == "7" and a[2]["status"] == "needs_review" and "maths_manual_review" in a[2]["flags"]
    # No letter or digit at all is a skipped answer: 0 marks, never graded.
    assert a[3]["status"] == "blank" and a[3]["ai_marks"] == 0 and a[3]["grader"] is None
    err = assert_error(teacher.post(f"/submissions/{sub['id']}/approve"), 409, "conflict")
    assert err["details"]["pending"] == [1, 2]


@accept_lexical  # approval / routing flow: clean lexical answers must be acceptable (conftest)
def test_answer_with_unread_lines_is_not_auto_accepted(teacher, exam, process):
    sid = process("ML001_full_marks")["ML001_full_marks"]["id"]
    q2 = teacher.answer_for(sid, 2)
    assert q2["status"] == "auto_accepted"
    region = next(r for r in teacher.regions(sid) if r["id"] in q2["region_ids"])
    x, y, _w, _h = region["bbox"]
    with SessionLocal() as db:  # e.g. faint lines the OCR engine returned as empty text
        for i in range(10):
            db.add(OcrLine(submission_id=sid, page_no=region["page_no"], line_no=1000 + i,
                           bbox=[x + 10, y + 8 + i, 40, 2], text="", confidence=0.0))
        db.commit()
    assert teacher.post(f"/exams/{exam['id']}/regrade").status_code == 200
    q2 = teacher.answer_for(sid, 2)
    assert q2["status"] == "needs_review" and "needs_transcription" in q2["flags"]
    assert q2["ocr_confidence"] < 0.6  # the unread lines count, not as one character each


# ------------------------------------------------------------------ max marks (finding 3)
def _rubric_q1_max(max_marks: float) -> dict:
    rubric = copy.deepcopy(samples.RUBRIC)
    q1 = rubric["questions"][0]
    q1["max_marks"] = max_marks
    for kp in q1["key_points"]:
        kp["weight"] = round(max_marks / len(q1["key_points"]), 2)
    q1["key_points"][-1]["weight"] = round(max_marks - sum(k["weight"] for k in q1["key_points"][:-1]), 2)
    return rubric


@accept_lexical  # approval / routing flow: clean lexical answers must be acceptable (conftest)
def test_rubric_max_cannot_drop_below_reviewed_or_approved_marks(teacher, exam, process):
    subs = process("ML001_full_marks", "ML002_partial")
    s1, s2 = subs["ML001_full_marks"]["id"], subs["ML002_partial"]["id"]
    q1 = teacher.answer_for(s2, 1)
    r = teacher.patch(f"/answers/{q1['answer_id']}/review", json={"final_marks": 3, "reason": "all points made"})
    assert r.status_code == 200, r.text
    err = assert_error(teacher.put(f"/exams/{exam['id']}/questions", json=_rubric_q1_max(1.5)), 409, "conflict")
    assert err["details"]["over_max"] == [{"student_ref": "ML002_partial", "question_no": 1, "final_marks": 3,
                                           "max_marks": 1.5}]

    # Once a sheet is approved, no maximum may change until it is reopened.
    assert teacher.post(f"/submissions/{s1}/approve").status_code == 200
    teacher.patch(f"/answers/{q1['answer_id']}/review", json={"final_marks": 1.5, "reason": "stricter"})
    err = assert_error(teacher.put(f"/exams/{exam['id']}/questions", json=_rubric_q1_max(1.5)), 409, "conflict")
    assert err["details"]["approved"] == ["ML001_full_marks"]
    assert teacher.post(f"/submissions/{s1}/reopen", json={"reason": "rubric change"}).status_code == 200
    assert teacher.put(f"/exams/{exam['id']}/questions", json=_rubric_q1_max(1.5)).status_code == 200

    # Until it is regraded, ML001's Q1 still carries 3 AI marks: approval refuses it.
    err = assert_error(teacher.post(f"/submissions/{s1}/approve"), 409, "conflict")
    assert err["details"]["pending"] == [1] and "maximum" in err["message"]
    assert teacher.post(f"/exams/{exam['id']}/regrade").status_code == 200
    for sid in (s1, s2):
        for a in teacher.answers(sid):
            if a["status"] in ("needs_review", "failed"):
                assert teacher.patch(f"/answers/{a['answer_id']}/review", json={}).status_code == 200
        r = teacher.post(f"/submissions/{sid}/approve")
        assert r.status_code == 200, r.text
    rows = list(csv.DictReader(io.StringIO(teacher.get(f"/exams/{exam['id']}/export").text)))
    assert len(rows) == 2
    for row in rows:
        assert float(row["Q1"]) <= 1.5 and float(row["total"]) <= float(row["max_total"]) == 13.5


@accept_lexical  # approval / routing flow: clean lexical answers must be acceptable (conftest)
def test_reviewed_mark_above_a_lowered_maximum_is_reviewed_again(teacher, exam, process):
    sid = process("ML001_full_marks")["ML001_full_marks"]["id"]
    q1 = teacher.answer_for(sid, 1)
    assert teacher.patch(f"/answers/{q1['answer_id']}/review", json={}).json()["final_marks"] == 3
    with SessionLocal() as db:  # a maximum lowered before this check existed
        db.get(Question, exam["qid"][1]).max_marks = 1.5
        db.commit()
    assert assert_error(teacher.post(f"/submissions/{sid}/approve"), 409, "conflict")["details"]["pending"] == [1]
    assert teacher.post(f"/exams/{exam['id']}/regrade").status_code == 200
    q1 = teacher.answer_for(sid, 1)
    assert q1["status"] == "needs_review" and "marks_above_max" in q1["flags"]
    assert teacher.patch(f"/answers/{q1['answer_id']}/review", json={"final_marks": 1.5, "reason": "new max"}
                         ).status_code == 200
    assert teacher.post(f"/submissions/{sid}/approve").status_code == 200


# ------------------------------------------------------------------ review (finding 5)
def test_correcting_text_after_review_follows_the_regraded_marks(teacher, process):
    sid = process("ML001_full_marks")["ML001_full_marks"]["id"]
    q1 = teacher.answer_for(sid, 1)
    assert teacher.patch(f"/answers/{q1['answer_id']}/review", json={}).json()["final_marks"] == 3
    r = teacher.patch(f"/answers/{q1['answer_id']}/review",
                      json={"corrected_text": "I do not know the answer.", "approve": True})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "reviewed" and out["final_marks"] == out["ai_marks"] < 3
    # Without a text change an earlier decision still stands.
    r = teacher.patch(f"/answers/{q1['answer_id']}/review", json={"comment": "checked", "approve": True})
    assert r.json()["final_marks"] == out["final_marks"]


# ------------------------------------------------------------------ audit (finding 9)
def test_regrade_and_region_edits_audit_every_changed_answer(teacher, exam, process):
    sid = process("ML001_full_marks")["ML001_full_marks"]["id"]
    q1 = teacher.answer_for(sid, 1)
    rubric = copy.deepcopy(samples.RUBRIC)
    rubric["questions"][0]["key_points"][0] = {"text": "the kernel trick maps points to a higher dimension",
                                               "weight": 1, "keywords": ["kernel", "dimension"]}
    teacher.put_rubric(exam["id"], rubric)
    assert teacher.post(f"/exams/{exam['id']}/regrade").status_code == 200
    new = teacher.answer_for(sid, 1)
    assert new["ai_marks"] < q1["ai_marks"]
    hist = teacher.get(f"/audit-log?entity=answer&id={q1['answer_id']}").json()["items"]
    entry = next(h for h in hist if h["action"] == "regrade")
    assert entry["old_value"]["ai_marks"] == q1["ai_marks"] and entry["new_value"]["ai_marks"] == new["ai_marks"]
    assert entry["user_id"] == teacher.user["id"]
    unchanged = teacher.answer_for(sid, 4)  # Q4's marks did not move: no entry
    assert not teacher.get(f"/audit-log?entity=answer&id={unchanged['answer_id']}").json()["items"]

    region = next(r for r in teacher.regions(sid) if r["question_id"] == exam["qid"][1])
    r = teacher.patch(f"/regions/{region['id']}",
                      json={"action": "reassign", "question_id": exam["qid"][2], "reason": "wrong marker"})
    assert r.status_code == 200, r.text
    entry = teacher.get(f"/audit-log?entity=answer&id={q1['answer_id']}").json()["items"][0]
    assert entry["action"] == "regions_edit" and entry["reason"] == "wrong marker"
    assert entry["old_value"]["ai_marks"] == new["ai_marks"] and entry["new_value"]["ai_marks"] == 0


# ------------------------------------------------------------------ deleting (finding 10)
def test_sheet_cannot_be_deleted_while_the_worker_holds_it(teacher, exam, sample_bytes):
    sid = teacher.upload(exam["id"], [("a.pdf", sample_bytes["ML001_full_marks"])]).json()["submissions"][0]["id"]
    with SessionLocal() as db:
        job = db.query(Job).filter_by(submission_id=sid).one()
        job.status, job.attempts, job.heartbeat_at = "running", 1, datetime.now(timezone.utc)
        db.get(Submission, sid).status = "processing"
        db.commit()
    assert_error(teacher.delete(f"/submissions/{sid}"), 409, "conflict")


def test_sheet_deleted_mid_processing_leaves_no_files(teacher, exam, sample_bytes, settings, monkeypatch):
    from app.pipeline import processor

    sid = teacher.upload(exam["id"], [("a.pdf", sample_bytes["ML008_continuation"])]).json()["submissions"][0]["id"]
    folder = settings.storage_dir / "submissions" / sid
    real = processor._exists
    calls: list[str] = []

    def deleted_after_page_one(db, submission_id):  # noqa: ANN001, ANN202
        calls.append(submission_id)
        if len(calls) == 2:  # the row goes; its files stay behind (Windows: the PDF is still open)
            with SessionLocal() as other:
                other.delete(other.get(Submission, submission_id))
                other.commit()
        return real(db, submission_id)

    monkeypatch.setattr(processor, "_exists", deleted_after_page_one)
    assert drain() == 1
    assert len(calls) == 2 and not folder.exists()


def test_exam_files_survive_a_failed_delete(teacher, exam, sample_bytes, settings, monkeypatch):
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.orm import Session

    teacher.upload(exam["id"], [("a.pdf", sample_bytes["ML001_full_marks"])])
    stored = list((settings.storage_dir / "submissions").rglob("original.pdf"))
    assert len(stored) == 1

    def locked(self):  # noqa: ANN001, ANN202
        raise OperationalError("COMMIT", {}, Exception("database is locked"))

    monkeypatch.setattr(Session, "commit", locked)
    r = teacher.delete(f"/exams/{exam['id']}")
    monkeypatch.undo()
    assert_error(r, 503, "service_unavailable")
    assert stored[0].is_file()  # the rows survived, so must their files
    assert teacher.get(f"/exams/{exam['id']}").status_code == 200
    assert teacher.delete(f"/exams/{exam['id']}").status_code == 204
    assert not stored[0].exists()


# ------------------------------------------------------------------ CSV (findings 4, 18)
@pytest.mark.parametrize("raw,clean", [("= =1+1", "1+1"), ("+ -2+3", "2+3"), ("@ @SUM(A1)", "SUM(A1)"),
                                       (" -\t=x", "x"), ("Roll-17", "Roll-17")])
def test_student_ref_never_starts_with_a_formula_character(raw, clean):
    from app.api.submissions import _clean_ref

    assert _clean_ref(raw) == clean


@accept_lexical  # approval / routing flow: clean lexical answers must be acceptable (conftest)
def test_formula_like_refs_are_neutralised_in_the_csv(teacher, exam, sample_bytes):
    r = teacher.upload(exam["id"], [("a.pdf", sample_bytes["ML001_full_marks"])], refs=["= =1+1"])
    sub = r.json()["submissions"][0]
    assert sub["student_ref"] == "1+1"
    drain()
    with SessionLocal() as db:  # a row stored before refs were cleaned this way
        db.get(Submission, sub["id"]).student_ref = "=1+1"
        db.commit()
    assert teacher.post(f"/submissions/{sub['id']}/approve").status_code == 200
    rows = list(csv.reader(io.StringIO(teacher.get(f"/exams/{exam['id']}/export").text)))
    assert rows[1][0] == "'=1+1"


@accept_lexical  # approval / routing flow: clean lexical answers must be acceptable (conftest)
def test_export_works_for_a_non_latin_exam_title(teacher, exam, process):
    title = "गणित परीक्षा"
    assert teacher.patch(f"/exams/{exam['id']}", json={"title": title}).status_code == 200
    sid = process("ML001_full_marks")["ML001_full_marks"]["id"]
    assert teacher.post(f"/submissions/{sid}/approve").status_code == 200
    r = teacher.get(f"/exams/{exam['id']}/export?format=csv")
    assert r.status_code == 200, r.text
    disposition = r.headers["content-disposition"]
    assert 'filename="exam_results.csv"' in disposition
    assert urllib.parse.unquote(disposition.split("filename*=UTF-8''")[1]) == f"{title}_results.csv"


# ------------------------------------------------------------------ uploads and PDFs (findings 8, 13)
@pytest.fixture
def small_budget(monkeypatch):  # noqa: ANN201
    from app import uploadguard

    monkeypatch.setattr(uploadguard, "MAX_FILES_PER_REQUEST", 1)
    monkeypatch.setattr(uploadguard, "FORM_OVERHEAD", 64 * 1024)


@pytest.fixture
def parser_calls(monkeypatch) -> list[int]:
    from starlette.formparsers import MultiPartParser

    calls: list[int] = []
    real = MultiPartParser.parse

    async def spy(self):  # noqa: ANN001, ANN202
        calls.append(1)
        return await real(self)

    monkeypatch.setattr(MultiPartParser, "parse", spy)
    return calls


def _multipart(name: str, filename: str, data: bytes, boundary: str) -> bytes:
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            "Content-Type: application/pdf\r\n\r\n").encode()
    return head + data + f"\r\n--{boundary}--\r\n".encode()


@pytest.mark.settings(max_upload_mb=1)
def test_oversized_upload_is_refused_before_the_body_is_spooled(teacher, exam, sample_bytes, small_budget,
                                                                parser_calls):
    big = samples.oversized_pdf_bytes(1.5)
    err = assert_error(teacher.upload(exam["id"], [("big.pdf", big)]), 413, "file_too_large")
    assert "in total" in err["message"] and parser_calls == []  # refused on Content-Length alone

    # Without a Content-Length (chunked), the body is cut off once it passes the budget.
    boundary = "saras-test-boundary"
    body = _multipart("files", "big.pdf", big, boundary)
    chunks = (body[i:i + 65536] for i in range(0, len(body), 65536))
    r = teacher.client.post(f"{API}/exams/{exam['id']}/submissions", content=chunks,
                            headers={**teacher.headers, "Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert "in total" in assert_error(r, 413, "file_too_large")["message"]
    assert teacher.get(f"/exams/{exam['id']}/submissions").json()["total"] == 0
    assert teacher.upload(exam["id"], [("ok.pdf", sample_bytes["ML001_full_marks"])]).status_code == 202
    assert parser_calls  # a normal upload is parsed as usual


def test_unauthenticated_upload_is_refused_before_the_body_is_read(client, exam, parser_calls):
    for header in ("Bearer garbage-zz", ""):
        r = client.post(f"{API}/exams/{exam['id']}/submissions", headers={"Authorization": header},
                        files=[("files", ("a.pdf", b"%PDF-1.4 x", "application/pdf"))])
        assert_error(r, 401, "unauthorized")
    assert parser_calls == []


def test_pdf_declaring_a_huge_page_is_rejected(teacher, exam):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page(width=14400, height=14400)  # 900 MP at 150 dpi, from a file of under 1 KB
    data = doc.tobytes()
    doc.close()
    assert len(data) < 2048
    err = assert_error(teacher.upload(exam["id"], [("bomb.pdf", data)]), 422, "unreadable_pdf")
    assert "larger than an answer sheet" in err["message"]


def test_rendering_stays_within_the_pixel_budget(monkeypatch):
    import pymupdf

    from app.pipeline import pdf

    monkeypatch.setattr(pdf, "MAX_PAGE_PIXELS", 200_000)
    doc = pymupdf.open("pdf", samples.build_text_pdf([["Q1. Supervised learning uses labelled data."]]))
    try:
        page = pdf.render_page(doc, 0, 150)
    finally:
        doc.close()
    h, w = page.image.shape
    assert w * h <= 200_000 * 1.01
    assert page.text_lines and all(ln.bbox[0] + ln.bbox[2] <= w + 1 and ln.bbox[1] + ln.bbox[3] <= h + 1
                                   for ln in page.text_lines)


# ------------------------------------------------------------------ rate limiting (finding 14)
@pytest.mark.settings(rate_limit_enabled=True, rate_limit_per_minute=4)
def test_forged_authorization_headers_share_the_client_budget(client):
    api = register(client, "rl2@example.com")  # request 1 from this client
    codes = [client.get(f"{API}/exams", headers={"Authorization": f"Bearer x{i}"}).status_code for i in range(5)]
    assert codes == [401, 401, 401, 429, 429]
    assert api.get("/exams").status_code == 200  # a verified user has a budget of their own


def test_parallel_login_attempts_count_before_they_fail(app):
    from app.security import LoginLimiter

    lim = LoginLimiter()
    for _ in range(5):  # five guesses in flight at once
        lim.check("x@example.com")
    with pytest.raises(ApiError) as exc:
        lim.check("x@example.com")
    assert exc.value.status == 429
    for _ in range(5):
        lim.fail("x@example.com")
    with pytest.raises(ApiError):
        lim.check("x@example.com")  # now locked
    lim.check("y@example.com")
    lim.release("y@example.com")
    assert "y@example.com" not in lim._in_flight


def test_limiter_tables_stay_bounded(app):
    from app.ratelimit import _Window
    from app.security import LoginLimiter

    clock = [1000.0]
    w = _Window(clock=lambda: clock[0])
    w.MAX_KEYS = 50
    for i in range(500):
        assert w.allow(f"ip:{i}", 5)[0]
    assert len(w.hits) == 50  # a flood of distinct callers: only the most recent are kept
    clock[0] += 61  # every caller idle for a whole window is swept
    assert w.allow("late", 5)[0]
    assert list(w.hits) == ["late"]

    lim = LoginLimiter()
    lim.MAX_TRACKED = 20
    for i in range(200):
        lim.check(f"u{i}@example.com")
        lim.fail(f"u{i}@example.com")
    assert len(lim._failures) <= 21 and not lim._in_flight


# ------------------------------------------------------------------ refresh tokens (finding 20)
def test_replayed_refresh_token_revokes_every_session(client, monkeypatch):
    from app.api import auth

    monkeypatch.setattr(auth, "REUSE_GRACE_SECONDS", 0.0)
    api = register(client, "reuse@example.com")
    t1 = api.tokens["refresh_token"]
    t2 = client.post(f"{API}/auth/refresh", json={"refresh_token": t1}).json()["refresh_token"]
    assert_error(client.post(f"{API}/auth/refresh", json={"refresh_token": t1}), 401, "unauthorized")
    # The spent token came back (stolen?): the session chain it started is ended too.
    assert_error(client.post(f"{API}/auth/refresh", json={"refresh_token": t2}), 401, "unauthorized")


def test_concurrent_refresh_with_one_token_has_a_single_winner(client, monkeypatch):
    import jwt

    from app.api import auth

    api = register(client, "race@example.com")
    other = client.post(f"{API}/auth/login", json={"email": "race@example.com", "password": PASSWORD}).json()
    token = api.tokens["refresh_token"]
    jti = jwt.decode(token, options={"verify_signature": False})["jti"]
    real_aware = auth.aware

    def other_request_wins(dt):  # noqa: ANN001, ANN202
        with SessionLocal() as db:  # a parallel request with the same token rotates it right now
            db.query(RefreshToken).filter_by(id=jti).update({"revoked": True})
            db.commit()
        auth._remember_rotation(jti)
        return real_aware(dt)

    monkeypatch.setattr(auth, "aware", other_request_wins)
    assert_error(client.post(f"{API}/auth/refresh", json={"refresh_token": token}), 401, "unauthorized")
    monkeypatch.setattr(auth, "aware", real_aware)
    # A double submit within seconds is not treated as theft: the user's other session lives on.
    assert client.post(f"{API}/auth/refresh", json={"refresh_token": other["refresh_token"]}).status_code == 200


# ------------------------------------------------------------------ exams (findings 28, 30)
def test_exam_says_whether_the_caller_can_edit_it(teacher, admin, exam):
    assert teacher.get(f"/exams/{exam['id']}").json()["can_edit"] is True
    assert admin.get(f"/exams/{exam['id']}").json()["can_edit"] is False
    own = admin.create_exam(title="Admin's own exam")
    assert own["can_edit"] is True
    assert {e["id"]: e["can_edit"] for e in admin.get("/exams").json()["items"]} == {exam["id"]: False,
                                                                                     own["id"]: True}


def test_total_marks_cannot_drop_below_the_rubric_total(teacher, exam):
    err = assert_error(teacher.patch(f"/exams/{exam['id']}", json={"total_marks": 10}), 400, "validation_error")
    assert error_fields(err) == {"total_marks"} and "15" in err["details"][0]["message"]
    assert teacher.patch(f"/exams/{exam['id']}", json={"total_marks": 15}).json()["total_marks"] == 15
    assert teacher.patch(f"/exams/{exam['id']}", json={"title": "Renamed"}).status_code == 200
