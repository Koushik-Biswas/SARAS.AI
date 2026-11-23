"""Teacher review and approval (PRD 6.2, 15.1, 20.2): approval blocked while
flagged answers are unreviewed, reasons for mark changes, bounds, text
corrections regrade, audit trail (append-only), locking, reopen, CSV export
equals approved marks, admin read-only, other teachers forbidden."""

from __future__ import annotations

import csv
import io

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from conftest import accept_lexical, assert_error, error_fields
from scripts import make_samples as samples

# These tests are about the review / approval flow, not routing: clean lexical-graded
# answers must be auto-accepted so that only the flagged ones block approval (see conftest).
pytestmark = accept_lexical


@pytest.fixture
def sheets(process) -> dict[str, dict]:
    # ML001: everything auto-accepted. ML002 / ML009: Q2 needs review.
    return process("ML001_full_marks", "ML002_partial", "ML009_flipped_fact")


def test_approve_blocked_while_answers_need_review(teacher, sheets):
    sub = sheets["ML002_partial"]
    assert sub["answers_needing_review"] == 1
    err = assert_error(teacher.post(f"/submissions/{sub['id']}/approve"), 409, "conflict")
    assert err["details"] == {"pending": [2]}
    # Reviewing the flagged answer unblocks approval.
    q2 = teacher.answer_for(sub["id"], 2)
    assert teacher.patch(f"/answers/{q2['answer_id']}/review", json={}).status_code == 200
    r = teacher.post(f"/submissions/{sub['id']}/approve")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "approved" and r.json()["approved_at"]


def test_approve_all_reports_blocked_sheets(teacher, exam, sheets):
    r = teacher.post(f"/exams/{exam['id']}/approve-all")
    assert r.status_code == 200
    body = r.json()
    assert body["approved"] == [sheets["ML001_full_marks"]["id"]]
    assert sorted(b["student_ref"] for b in body["blocked"]) == ["ML002_partial", "ML009_flipped_fact"]
    assert all(b["pending"] == [2] for b in body["blocked"])


def test_reason_required_when_marks_change(teacher, sheets):
    sub = sheets["ML002_partial"]
    q2 = teacher.answer_for(sub["id"], 2)
    new = q2["ai_marks"] + 1
    err = assert_error(teacher.patch(f"/answers/{q2['answer_id']}/review", json={"final_marks": new}),
                       400, "validation_error")
    assert error_fields(err) == {"reason"}
    r = teacher.patch(f"/answers/{q2['answer_id']}/review",
                      json={"final_marks": new, "reason": "KP2 is clearly explained", "comment": "Good"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "reviewed" and out["final_marks"] == new and out["ai_marks"] == q2["ai_marks"]
    assert out["comment"] == "Good"
    assert teacher.submission(sub["id"])["status"] == "in_review"
    # Accepting the AI marks needs no reason.
    q1 = teacher.answer_for(sub["id"], 1)
    r = teacher.patch(f"/answers/{q1['answer_id']}/review", json={"final_marks": q1["ai_marks"]})
    assert r.status_code == 200 and r.json()["final_marks"] == q1["ai_marks"]


@pytest.mark.parametrize("marks", [-1, 5.01, 6, 2.555])
def test_marks_outside_range_rejected(teacher, sheets, marks):
    q2 = teacher.answer_for(sheets["ML002_partial"]["id"], 2)
    err = assert_error(teacher.patch(f"/answers/{q2['answer_id']}/review",
                                     json={"final_marks": marks, "reason": "x"}), 400, "validation_error")
    assert error_fields(err) == {"final_marks"}
    assert teacher.answer_for(sheets["ML002_partial"]["id"], 2)["status"] == "needs_review"


def test_corrected_text_regrades_and_stores_correction(teacher, sheets):
    from app.db import SessionLocal
    from app.models import Correction

    q2 = teacher.answer_for(sheets["ML002_partial"]["id"], 2)
    assert q2["ai_marks"] < 5
    r = teacher.patch(f"/answers/{q2['answer_id']}/review",
                      json={"corrected_text": samples.ANSWERS["q2_full"], "approve": False})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["text_corrected"] and out["ocr_text"] == samples.ANSWERS["q2_full"]
    assert out["ai_marks"] == 5 and out["ocr_confidence"] == 1.0
    assert out["status"] == "needs_review"  # saved, not approved: still in the queue
    with SessionLocal() as db:
        corr = db.query(Correction).filter_by(answer_id=q2["answer_id"]).one()
    assert corr.original_text == q2["ocr_text"] and corr.corrected_text == samples.ANSWERS["q2_full"]
    # Approving now accepts the regraded marks.
    r = teacher.patch(f"/answers/{q2['answer_id']}/review", json={"approve": True})
    assert r.json()["status"] == "reviewed" and r.json()["final_marks"] == 5


def test_every_change_is_in_the_audit_log(teacher, sheets):
    sub = sheets["ML002_partial"]
    q2 = teacher.answer_for(sub["id"], 2)
    teacher.patch(f"/answers/{q2['answer_id']}/review",
                  json={"final_marks": 4, "reason": "generous on KP2", "comment": "ok"})
    teacher.post(f"/submissions/{sub['id']}/approve")
    teacher.post(f"/submissions/{sub['id']}/reopen", json={"reason": "student appeal"})

    hist = teacher.get(f"/audit-log?entity=answer&id={q2['answer_id']}").json()
    entry = hist["items"][0]
    assert entry["action"] == "review" and entry["user_id"] == teacher.user["id"]
    assert entry["reason"] == "generous on KP2"
    assert entry["old_value"]["final_marks"] is None and entry["old_value"]["ai_marks"] == q2["ai_marks"]
    assert entry["new_value"]["final_marks"] == 4 and entry["new_value"]["status"] == "reviewed"
    assert entry["created_at"].endswith(("+00:00", "Z"))

    sub_hist = teacher.get(f"/audit-log?entity=submission&id={sub['id']}").json()["items"]
    assert [h["action"] for h in sub_hist] == ["reopen", "approve", "upload"]
    assert sub_hist[0]["reason"] == "student appeal"
    assert sub_hist[1]["new_value"]["marks"]["2"] == 4


def test_audit_log_is_append_only(teacher, sheets):
    from app.db import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM audit_log")).scalar()
        assert count > 0
    for stmt in ("UPDATE audit_log SET reason = 'tampered'", "DELETE FROM audit_log"):
        with pytest.raises(DBAPIError, match="append-only"):
            with engine.begin() as conn:
                conn.execute(text(stmt))
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM audit_log")).scalar() == count
        assert conn.execute(text("SELECT COUNT(*) FROM audit_log WHERE reason = 'tampered'")).scalar() == 0


def test_approved_sheet_is_locked_and_reopen_needs_reason(teacher, sheets):
    sub = sheets["ML001_full_marks"]
    assert teacher.post(f"/submissions/{sub['id']}/approve").status_code == 200
    q1 = teacher.answer_for(sub["id"], 1)
    assert q1["locked"] is True and q1["final_marks"] == q1["ai_marks"]
    assert_error(teacher.patch(f"/answers/{q1['answer_id']}/review", json={"final_marks": 1, "reason": "x"}),
                 409, "conflict")
    assert_error(teacher.post(f"/answers/{q1['answer_id']}/flag", json={}), 409, "conflict")
    assert_error(teacher.post(f"/submissions/{sub['id']}/approve"), 409, "conflict")
    assert_error(teacher.delete(f"/submissions/{sub['id']}"), 409, "conflict")
    region = teacher.regions(sub["id"])[0]
    assert_error(teacher.patch(f"/regions/{region['id']}", json={"action": "blank"}), 409, "conflict")

    assert_error(teacher.post(f"/submissions/{sub['id']}/reopen", json={}), 400, "validation_error")
    assert_error(teacher.post(f"/submissions/{sub['id']}/reopen", json={"reason": ""}), 400, "validation_error")
    assert_error(teacher.post(f"/submissions/{sub['id']}/reopen", json={"reason": "   "}), 400, "validation_error")
    r = teacher.post(f"/submissions/{sub['id']}/reopen", json={"reason": "re-check Q1"})
    assert r.status_code == 200 and r.json()["status"] == "in_review" and r.json()["approved_at"] is None
    assert teacher.patch(f"/answers/{q1['answer_id']}/review",
                         json={"final_marks": 2, "reason": "missed KP3"}).status_code == 200


def test_flag_sends_answer_back_to_review(teacher, sheets):
    sub = sheets["ML001_full_marks"]
    q3 = teacher.answer_for(sub["id"], 3)
    assert q3["status"] == "auto_accepted"
    r = teacher.post(f"/answers/{q3['answer_id']}/flag", json={"reason": "check working"})
    assert r.status_code == 200 and r.json()["status"] == "needs_review" and "teacher_flag" in r.json()["flags"]
    err = assert_error(teacher.post(f"/submissions/{sub['id']}/approve"), 409, "conflict")
    assert err["details"]["pending"] == [3]


def test_priority_sort_and_status_filter(teacher, sheets):
    sub = sheets["ML009_flipped_fact"]
    answers = teacher.answers(sub["id"], sort="priority")
    assert answers[0]["question_no"] == 2 and answers[0]["status"] == "needs_review"
    confs = [a["confidence"] for a in answers[1:]]
    assert confs == sorted(confs)
    only = teacher.get(f"/submissions/{sub['id']}/answers?status=needs_review").json()
    assert [a["question_no"] for a in only] == [2]
    kp3 = only[0]["key_points"][2]
    assert kp3["status"] == "contradicted" and kp3["text"] in only[0]["missed_points"]


def _csv(teacher, exam_id: str) -> list[dict]:
    r = teacher.get(f"/exams/{exam_id}/export?format=csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    return list(csv.DictReader(io.StringIO(r.text)))


def test_csv_export_matches_approved_marks_exactly(teacher, exam, sheets):
    from app.db import SessionLocal
    from app.models import Submission
    from app.pipeline.processor import effective_marks

    assert _csv(teacher, exam["id"]) == []  # nothing approved yet
    ml002 = sheets["ML002_partial"]
    q2 = teacher.answer_for(ml002["id"], 2)
    teacher.patch(f"/answers/{q2['answer_id']}/review", json={"final_marks": 2.75, "reason": "half of KP2"})
    for ref in ("ML001_full_marks", "ML002_partial"):
        assert teacher.post(f"/submissions/{sheets[ref]['id']}/approve").status_code == 200

    rows = _csv(teacher, exam["id"])
    assert [r["student_ref"] for r in rows] == ["ML001_full_marks", "ML002_partial"]  # ML009 not approved
    results = {i["student_ref"]: i for i in teacher.get(f"/exams/{exam['id']}/results").json()["items"]}
    with SessionLocal() as db:
        for row in rows:
            sub = db.query(Submission).filter_by(exam_id=exam["id"], student_ref=row["student_ref"]).one()
            assert sub.status == "approved" and row["status"] == "approved"
            by_q = {a.question.question_no: a for a in sub.answers}
            for no in (1, 2, 3, 4):
                db_marks = round(effective_marks(by_q[no]), 2)
                assert by_q[no].final_marks == db_marks  # frozen at approval
                assert float(row[f"Q{no}"]) == db_marks
                assert results[row["student_ref"]]["cells"][no - 1]["marks"] == db_marks
            assert float(row["total"]) == sub.total_final_marks == results[row["student_ref"]]["total"]
            assert float(row["max_total"]) == 15
            assert row["approved_at"] == sub.approved_at.isoformat()
    assert float(next(r for r in rows if r["student_ref"] == "ML002_partial")["Q2"]) == 2.75


def test_admin_can_read_but_not_edit(teacher, admin, exam, sheets):
    sub = sheets["ML002_partial"]
    q2 = teacher.answer_for(sub["id"], 2)
    for path in (f"/exams/{exam['id']}", f"/exams/{exam['id']}/results", f"/submissions/{sub['id']}",
                 f"/submissions/{sub['id']}/answers", f"/answers/{q2['answer_id']}",
                 f"/exams/{exam['id']}/export", f"/audit-log?entity=answer&id={q2['answer_id']}"):
        assert admin.get(path).status_code == 200, path
    assert admin.get("/exams").json()["total"] == 1
    for method, path, body in (
        ("PATCH", f"/answers/{q2['answer_id']}/review", {"final_marks": 1, "reason": "x"}),
        ("POST", f"/answers/{q2['answer_id']}/flag", {}),
        ("POST", f"/submissions/{sub['id']}/approve", None),
        ("POST", f"/exams/{exam['id']}/approve-all", None),
        ("PUT", f"/exams/{exam['id']}/questions", samples.RUBRIC),
        ("POST", f"/exams/{exam['id']}/regrade", None),
        ("DELETE", f"/submissions/{sub['id']}", None),
        ("PATCH", f"/regions/{q2['region_ids'][0]}", {"action": "blank"}),
    ):
        assert_error(admin._call(method, path, json=body), 403, "forbidden")
    assert teacher.answer_for(sub["id"], 2)["final_marks"] is None


def test_other_teacher_is_forbidden(teacher, other_teacher, exam, sheets):
    sub = sheets["ML002_partial"]
    q2 = teacher.answer_for(sub["id"], 2)
    assert other_teacher.get("/exams").json()["total"] == 0
    for path in (f"/exams/{exam['id']}", f"/submissions/{sub['id']}", f"/answers/{q2['answer_id']}",
                 f"/submissions/{sub['id']}/pages/1/image", f"/exams/{exam['id']}/export",
                 f"/audit-log?entity=submission&id={sub['id']}"):
        assert_error(other_teacher.get(path), 403, "forbidden")
    assert_error(other_teacher.patch(f"/answers/{q2['answer_id']}/review", json={}), 403, "forbidden")
    assert_error(other_teacher.post(f"/submissions/{sub['id']}/approve"), 403, "forbidden")
