"""Rubric validation (PRD 6.2): key-point weights sum to max marks, unique question
numbers, rubric total within the exam total, question set fixed after upload."""

from __future__ import annotations

import copy

from conftest import assert_error, error_fields
from scripts import make_samples as samples


def _rubric() -> dict:
    return copy.deepcopy(samples.RUBRIC)


def test_rubric_round_trip_and_post_alias(teacher):
    exam = teacher.create_exam()
    r = teacher.post(f"/exams/{exam['id']}/questions", json=samples.RUBRIC)
    assert r.status_code == 201, r.text
    qs = teacher.get(f"/exams/{exam['id']}/questions").json()
    assert [q["question_no"] for q in qs] == [1, 2, 3, 4]
    assert [q["max_marks"] for q in qs] == [3, 5, 5, 2]
    assert qs[1]["key_points"][3]["text"].startswith("it is caused by")
    assert teacher.get(f"/exams/{exam['id']}").json()["rubric_marks"] == 15


def test_weights_must_sum_to_max_marks(teacher):
    exam = teacher.create_exam()
    rubric = _rubric()
    rubric["questions"][1]["key_points"][0]["weight"] = 2  # Q2 now sums to 6, max 5
    err = assert_error(teacher.put(f"/exams/{exam['id']}/questions", json=rubric), 400, "validation_error")
    assert "questions.1" in error_fields(err)
    msg = next(d["message"] for d in err["details"] if d["field"] == "questions.1")
    assert "6" in msg and "5" in msg
    assert teacher.get(f"/exams/{exam['id']}/questions").json() == []  # nothing saved


def test_question_only_rubric_needs_the_question_text(teacher):
    # No key points and no model answer: the AI marks from the question, so its text is required.
    exam = teacher.create_exam()
    rubric = _rubric()
    rubric["questions"][0]["key_points"] = []
    rubric["questions"][0]["model_answer"] = ""
    assert teacher.put(f"/exams/{exam['id']}/questions", json=rubric).status_code == 200
    rubric["questions"][0]["prompt"] = ""
    err = assert_error(teacher.put(f"/exams/{exam['id']}/questions", json=rubric), 400, "validation_error")
    assert "questions.0" in error_fields(err)


def test_duplicate_question_numbers_rejected(teacher):
    exam = teacher.create_exam()
    rubric = _rubric()
    rubric["questions"][3]["question_no"] = 2
    err = assert_error(teacher.put(f"/exams/{exam['id']}/questions", json=rubric), 400, "validation_error")
    assert "questions" in error_fields(err)
    assert "2" in err["details"][0]["message"]


def test_rubric_total_cannot_exceed_exam_total(teacher):
    exam = teacher.create_exam(total_marks=10)  # rubric adds up to 15
    err = assert_error(teacher.put(f"/exams/{exam['id']}/questions", json=samples.RUBRIC), 400, "validation_error")
    assert "questions" in error_fields(err)


def test_field_level_errors_for_bad_values(teacher):
    exam = teacher.create_exam()
    rubric = _rubric()
    rubric["questions"][0]["type"] = "essay"
    rubric["questions"][2]["max_marks"] = 0
    err = assert_error(teacher.put(f"/exams/{exam['id']}/questions", json=rubric), 400, "validation_error")
    fields = error_fields(err)
    assert "questions.0.type" in fields and "questions.2.max_marks" in fields


def test_upload_requires_a_rubric(teacher, sample_bytes):
    exam = teacher.create_exam()
    r = teacher.upload(exam["id"], [("a.pdf", sample_bytes["ML001_full_marks"])])
    assert_error(r, 409, "conflict")


def test_question_set_locked_after_upload(teacher, exam, sample_bytes):
    r = teacher.upload(exam["id"], [("a.pdf", sample_bytes["ML001_full_marks"])])
    assert r.status_code == 202
    # Adding or removing questions is refused once sheets exist...
    added = _rubric()
    added["questions"].append({"question_no": 5, "type": "short", "max_marks": 1, "model_answer": "x y z"})
    for rubric in (added, {"questions": _rubric()["questions"][:3]}):
        err = assert_error(teacher.put(f"/exams/{exam['id']}/questions", json=rubric), 409, "conflict")
        assert err["details"]["existing"] == [1, 2, 3, 4]
    # ...but the existing questions can still be edited (then regraded).
    edited = _rubric()
    edited["questions"][0]["prompt"] = "Define supervised learning in one or two sentences."
    r = teacher.put(f"/exams/{exam['id']}/questions", json=edited)
    assert r.status_code == 200, r.text
    assert r.json()[0]["prompt"].endswith("two sentences.")
    assert [q["id"] for q in r.json()] == [exam["qid"][n] for n in (1, 2, 3, 4)]  # same rows, ids kept


def test_rubric_change_is_audited(teacher, exam):
    hist = teacher.get(f"/audit-log?entity=exam&id={exam['id']}").json()["items"]
    assert [h["action"] for h in hist][:2] == ["rubric_update", "create"]
