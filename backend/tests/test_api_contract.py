"""API contract (PRD 16): OpenAPI covers every contract path, one error format
for every status code, pagination limits, health/readiness, UTC timestamps,
rate limiting and SPA serving."""

from __future__ import annotations

import re

import pytest

from conftest import API, accept_lexical, assert_error, register
from scripts import make_samples as samples

CONTRACT = {
    ("POST", "/auth/register"), ("POST", "/auth/login"), ("POST", "/auth/refresh"), ("POST", "/auth/logout"),
    ("GET", "/auth/me"),
    ("GET", "/exams"), ("POST", "/exams"), ("GET", "/exams/{}"), ("PATCH", "/exams/{}"), ("DELETE", "/exams/{}"),
    ("GET", "/exams/{}/questions"), ("PUT", "/exams/{}/questions"), ("POST", "/exams/{}/questions"),
    ("POST", "/exams/{}/regrade"), ("GET", "/exams/{}/results"), ("GET", "/exams/{}/export"),
    ("GET", "/exams/{}/analytics"),
    ("POST", "/exams/{}/submissions"), ("GET", "/exams/{}/submissions"), ("POST", "/exams/{}/approve-all"),
    ("GET", "/submissions/{}"), ("GET", "/submissions/{}/status"), ("POST", "/submissions/{}/retry"),
    ("DELETE", "/submissions/{}"), ("GET", "/submissions/{}/answers"), ("POST", "/submissions/{}/approve"),
    ("POST", "/submissions/{}/reopen"), ("GET", "/submissions/{}/pages"),
    ("GET", "/submissions/{}/pages/{}/image"), ("GET", "/submissions/{}/regions"),
    ("POST", "/submissions/{}/regions"), ("POST", "/submissions/{}/regions/undo"),
    ("GET", "/answers/{}"), ("PATCH", "/answers/{}/review"), ("POST", "/answers/{}/flag"),
    ("PATCH", "/regions/{}"),
    ("GET", "/audit-log"), ("GET", "/capabilities"), ("GET", "/health"), ("GET", "/ready"),
}


def test_openapi_lists_every_contract_path(client):
    r = client.get("/api/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["info"]["title"] == "SARAS AI API"
    found = {(method.upper(), re.sub(r"\{[^}]+\}", "{}", path.removeprefix(API)))
             for path, ops in spec["paths"].items() for method in ops}
    missing = CONTRACT - found
    assert not missing, sorted(missing)
    assert all(p.startswith(API) for p in spec["paths"])  # everything versioned under /api/v1
    # Contract schemas are published for the frontend.
    for name in ("SubmissionOut", "AnswerOut", "KeyPointOut", "ReviewIn", "RegionOut", "RegionPatch",
                 "RegionEditOut", "StatusOut", "UploadOut", "TokenOut", "ExamOut", "QuestionsIn"):
        assert name in spec["components"]["schemas"], name
    assert client.get("/api/docs").status_code == 200


def test_health_and_ready(client):
    for path in ("/health", f"{API}/health"):
        assert client.get(path).json() == {"status": "ok"}
    for path in ("/ready", f"{API}/ready"):
        assert client.get(path).json() == {"status": "ready", "database": "ok", "worker": "disabled"}


@pytest.mark.settings(worker_enabled=True, worker_poll_seconds=0.2)
def test_ready_reports_running_worker(client):
    assert client.get(f"{API}/ready").json()["worker"] == "running"


def test_ready_503_when_database_unavailable(client, monkeypatch):
    from app import main

    class Broken:
        def __enter__(self):  # noqa: ANN204
            raise RuntimeError("database down")

        def __exit__(self, *a):  # noqa: ANN002, ANN204
            return False

    monkeypatch.setattr(main, "SessionLocal", lambda: Broken())
    err = assert_error(client.get(f"{API}/ready"), 503, "service_unavailable")
    assert err["details"]["database"] == "unavailable"


def test_error_format_for_every_status(client, teacher, other_teacher, exam, sample_bytes):
    # 400 field-level validation
    err = assert_error(teacher.post("/exams", json={"title": "", "total_marks": -1}), 400, "validation_error")
    assert {"title", "total_marks"} <= {d["field"] for d in err["details"]}
    assert_error(teacher.get("/exams?page_size=abc"), 400, "validation_error")
    # 401
    assert_error(client.get(f"{API}/exams"), 401, "unauthorized")
    # 403
    assert_error(other_teacher.get(f"/exams/{exam['id']}"), 403, "forbidden")
    # 404 (resource and unknown route)
    assert_error(teacher.get("/exams/00000000-0000-0000-0000-000000000000"), 404, "not_found")
    assert_error(teacher.get("/no-such-endpoint"), 404, "not_found")
    # 405
    assert_error(teacher.put("/auth/me", json={}), 405, "method_not_allowed")
    # 409
    assert_error(client.post(f"{API}/auth/register", json={"name": "x", "email": "teacher@example.com",
                                                           "password": "password-123"}), 409, "conflict")
    # 413 / 415 / 422
    assert_error(teacher.upload(exam["id"], [("x.pdf", samples.many_pages_pdf_bytes(51))]), 413, "file_too_large")
    assert_error(teacher.upload(exam["id"], [("x.pdf", samples.not_a_pdf_bytes())]), 415, "unsupported_media_type")
    assert_error(teacher.upload(exam["id"], [("x.pdf", samples.corrupted_pdf_bytes())]), 422, "unreadable_pdf")


@pytest.mark.settings(rate_limit_enabled=True, rate_limit_per_minute=5, upload_rate_per_minute=2)
def test_rate_limit_429(client):
    api = register(client, "rl@example.com")
    codes = [api.get("/exams").status_code for _ in range(6)]
    assert codes[:5] == [200] * 5
    r = api.get("/exams")
    err = assert_error(r, 429, "rate_limited")
    assert int(r.headers["Retry-After"]) > 0 and err["details"]["retry_after_seconds"] > 0
    # Health checks are outside the API rate limit.
    assert client.get("/health").status_code == 200


@pytest.mark.settings(rate_limit_enabled=True, rate_limit_per_minute=1000, upload_rate_per_minute=2)
def test_upload_rate_limit(client, teacher, exam):
    codes = [teacher.upload(exam["id"], [("x.pdf", samples.not_a_pdf_bytes())]).status_code for _ in range(3)]
    assert codes == [415, 415, 429]


def test_pagination_defaults_and_limits(teacher):
    for i in range(23):
        teacher.create_exam(title=f"Exam {i:02d}")
    first = teacher.get("/exams").json()
    assert first["page"] == 1 and first["page_size"] == 20 and first["total"] == 23 and len(first["items"]) == 20
    second = teacher.get("/exams?page=2").json()
    assert len(second["items"]) == 3
    assert not {e["id"] for e in first["items"]} & {e["id"] for e in second["items"]}
    assert len(teacher.get("/exams?page_size=100").json()["items"]) == 23
    for bad in ("page_size=101", "page_size=0", "page=0"):
        assert_error(teacher.get(f"/exams?{bad}"), 400, "validation_error")
        assert_error(teacher.get(f"/audit-log?{bad}"), 400, "validation_error")


@accept_lexical  # counts only the flagged answers of each sheet (conftest)
def test_submission_list_pagination_and_filter(teacher, exam, process):
    process("ML001_full_marks", "ML002_partial", "ML003_paraphrased")
    page = teacher.get(f"/exams/{exam['id']}/submissions?page_size=2").json()
    assert page["total"] == 3 and [s["student_ref"] for s in page["items"]] == ["ML001_full_marks", "ML002_partial"]
    assert page["items"][1]["answers_needing_review"] == 1
    assert teacher.get(f"/exams/{exam['id']}/submissions?status=approved").json()["total"] == 0
    assert teacher.get(f"/exams/{exam['id']}/submissions?status=processed").json()["total"] == 3
    assert_error(teacher.get(f"/exams/{exam['id']}/submissions?status=bogus"), 400, "validation_error")
    assert_error(teacher.get(f"/exams/{exam['id']}/submissions?page_size=101"), 400, "validation_error")


@accept_lexical  # needs an approvable sheet (conftest)
def test_timestamps_are_utc_iso8601(teacher, exam, process):
    sub = process("ML001_full_marks")["ML001_full_marks"]
    assert teacher.post(f"/submissions/{sub['id']}/approve").status_code == 200
    sub = teacher.submission(sub["id"])
    for value in (exam["created_at"], sub["created_at"], sub["approved_at"]):
        assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(\+00:00|Z)$", value), value


def test_answer_and_submission_shapes(teacher, process):
    sub = process("ML001_full_marks")["ML001_full_marks"]
    for key in ("id", "exam_id", "student_ref", "original_filename", "status", "stage", "page_count",
                "pages_done", "error", "total_ai_marks", "total_final_marks", "ocr_engine", "grader", "flags",
                "page_flags", "timings", "created_at", "approved_at", "answers_needing_review"):
        assert key in sub, key
    assert sub["grader"] == "lexical" and sub["ocr_engine"] == "textlayer"
    a = teacher.answers(sub["id"])[0]
    for key in ("answer_id", "submission_id", "question_id", "question_no", "question_type", "question_prompt",
                "status", "ocr_text", "text_corrected", "ocr_confidence", "ai_marks", "final_marks", "max_marks",
                "confidence", "grading_confidence", "segmentation_confidence", "grader", "key_points",
                "matched_points", "missed_points", "flags", "comment", "region_ids", "locked"):
        assert key in a, key
    kp = a["key_points"][0]
    # v1 fields, plus the additive grading-ensemble fields (reason, source, disputed, statuses_by_grader).
    assert set(kp) == {"index", "text", "weight", "matched", "status", "awarded", "entailment_prob",
                       "contradiction_prob", "evidence_sentence", "reason", "source", "disputed",
                       "statuses_by_grader"}
    assert kp["source"] == "lexical" and kp["statuses_by_grader"] == {"lexical": kp["status"]}
    assert isinstance(kp["reason"], str) and kp["reason"] and kp["disputed"] is False
    # Additive answer fields: verdict, feedback and the decision trace.
    assert a["verdict"] in ("correct", "partially_correct", "incorrect") and a["feedback"]
    assert a["grading_trace"]["graders_used"] == ["lexical"] and a["grading_trace"]["routing"]["status"] == a["status"]


def test_capabilities(teacher, settings):
    caps = teacher.get("/capabilities").json()
    assert caps["ocr"]["textlayer"] is True and caps["graders"]["lexical"] is True
    assert caps["settings"] == {
        "grader": "lexical", "ocr_engine": "auto", "tau": settings.tau, "tau_low": settings.tau_low,
        "contradiction_threshold": settings.contradiction_threshold,
        "auto_accept_threshold": settings.auto_accept_threshold, "max_upload_mb": settings.max_upload_mb,
        "max_pages": settings.max_pages,
    }


@accept_lexical  # "pending" counts only the flagged answers (conftest)
def test_results_and_analytics(teacher, exam, process):
    subs = process("ML001_full_marks", "ML002_partial")
    res = teacher.get(f"/exams/{exam['id']}/results").json()
    assert res["max_total"] == 15 and [q["question_no"] for q in res["questions"]] == [1, 2, 3, 4]
    row = next(i for i in res["items"] if i["student_ref"] == "ML001_full_marks")
    assert row["total"] == 15 and all(c["marks"] == c["max_marks"] for c in row["cells"])
    assert next(i for i in res["items"] if i["student_ref"] == "ML002_partial")["pending"] == 1
    an = teacher.get(f"/exams/{exam['id']}/analytics").json()
    assert an["sheets"] == 2 and an["highest"] == 15 and an["lowest"] == subs["ML002_partial"]["total_ai_marks"]
    assert sum(b["count"] for b in an["distribution"]) == 2 and len(an["questions"]) == 4


def test_spa_served_with_fallback(tmp_path, settings, monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import set_settings
    from app.main import create_app

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>SARAS AI</title>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    try:
        app = create_app(settings.model_copy(update={"frontend_dist": dist}))
        with TestClient(app) as c:
            assert "SARAS AI" in c.get("/").text
            assert "SARAS AI" in c.get("/exams/123/review").text  # client-side route
            assert c.get("/assets/app.js").text == "console.log(1)"
            assert_error(c.get(f"{API}/nope"), 404, "not_found")
            assert_error(c.get("/assets/missing.js"), 404, "not_found")
    finally:
        set_settings(None)
