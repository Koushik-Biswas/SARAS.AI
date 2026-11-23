"""Question-only rubrics (no model answer, no key points): the AI gives the marks
directly; AI page reading for scans; answer splitting for sheets without numbers;
and the review_required switch. Providers are faked (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from app.grading import direct_ai
from app.llm import LLMRouter, RateLimiter
from app.llm.fakes import FakeCall, FakeLLMServer
from app.pipeline import ai_ocr, ai_segment, processor
from app.pipeline.ocr import OcrLine
from scripts import make_samples as samples

GROQ = "groq:openai/gpt-oss-120b"
AI_ON = dict(ai_grading="ensemble", ai_consent=True, groq_api_key="fake-groq-key-for-tests-0001",
             ai_grading_providers=[GROQ], ai_max_retries=0, ai_direct_scheme=False, ai_verify=False)
PREFIX = "Mark these answers. Input JSON:\n"


def question_only_rubric() -> dict:
    return {"questions": [{"question_no": q["question_no"], "type": q["type"], "max_marks": q["max_marks"],
                           "prompt": q["prompt"], "model_answer": None, "key_points": []}
                          for q in samples.RUBRIC["questions"]]}


def direct_marker(fraction: float = 1.0, uncertain: tuple = ()):  # noqa: ANN201
    def respond(call: FakeCall) -> dict:
        assert call.prompt.startswith(PREFIX)
        items = json.loads(call.prompt[len(PREFIX):])
        return {"answers": [{"question_no": it["question_no"], "marks": it["max_marks"] * fraction,
                             "feedback": f"Feedback {it['question_no']}", "evidence": "",
                             "uncertain": it["question_no"] in uncertain} for it in items]}
    return respond


@pytest.fixture
def fake_ai(monkeypatch) -> FakeLLMServer:  # noqa: ANN001
    server = FakeLLMServer()

    def make(settings, session_factory):  # noqa: ANN001, ANN202
        clock = [0.0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        return LLMRouter(settings, session_factory, transport=server.transport,
                         limiter=RateLimiter(clock=lambda: clock[0]), sleep=sleep)

    monkeypatch.setattr(processor, "make_ai_router", make)
    return server


@pytest.fixture
def qo_exam(teacher) -> dict:  # noqa: ANN001
    exam = teacher.create_exam()
    teacher.put_rubric(exam["id"], question_only_rubric())
    return exam


def process_in(teacher, exam: dict, sample_bytes: dict, *refs: str) -> dict:  # noqa: ANN001
    from conftest import drain

    r = teacher.upload(exam["id"], [(f"{ref}.pdf", sample_bytes[ref]) for ref in refs])
    assert r.status_code == 202, r.text
    drain()
    return {s["student_ref"]: teacher.submission(s["id"]) for s in r.json()["submissions"]}


# ------------------------------------------------------------------ units
def test_marks_are_clamped_and_rounded() -> None:
    assert direct_ai.round_marks(2.74, 5) == 2.5
    assert direct_ai.round_marks(2.76, 5) == 3.0
    assert direct_ai.round_marks(9, 5) == 5
    assert direct_ai.round_marks(-1, 5) == 0
    assert direct_ai.round_marks(float("nan"), 5) == 0
    assert direct_ai.round_marks(0.8, 1) == 0.75
    assert direct_ai.verdict_for(5, 5) == "correct"
    assert direct_ai.verdict_for(0, 5) == "incorrect"
    assert direct_ai.verdict_for(2.5, 5) == "partially_correct"


def test_is_direct_only_without_key_points_and_model_answer() -> None:
    assert direct_ai.is_direct(NS(key_points=[], model_answer=None))
    assert direct_ai.is_direct(NS(key_points=[{"text": " "}], model_answer="  "))
    assert not direct_ai.is_direct(NS(key_points=[], model_answer="Tiger"))
    assert not direct_ai.is_direct(NS(key_points=[{"text": "Tiger", "weight": 1}], model_answer=None))


def test_repeated_tail_from_the_vision_model_is_dropped() -> None:
    L = ai_ocr.PageLine
    lines = [L(text="Question National Animal of", box_2d=[1, 1, 2, 2]), L(text="India ?", box_2d=[]),
             L(text="Ans Tiger", box_2d=[]), L(text="Question National", box_2d=[]),
             L(text="Animal of India ?", box_2d=[]), L(text="Ans Tiger", box_2d=[])]
    assert [x.text for x in ai_ocr.drop_repeated_tail(lines)] == ["Question National Animal of", "India ?",
                                                                   "Ans Tiger"]
    short = [L(text="Ans 4", box_2d=[]), L(text="4", box_2d=[])]  # a short repeat is kept
    assert len(ai_ocr.drop_repeated_tail(short)) == 2


def test_page_lines_keep_reading_order_whatever_the_boxes_say() -> None:
    data = ai_ocr.PageText(lines=[
        ai_ocr.PageLine(text="first", box_2d=[300, 10, 320, 500]),
        ai_ocr.PageLine(text="second", box_2d=[100, 600, 900, 50]),  # inverted and page-tall
        ai_ocr.PageLine(text="third", box_2d=[]),
    ])
    out = ai_ocr.to_lines(data, 1, 1000, 2000)
    assert [o.text for o in out] == ["first", "second", "third"]
    ys = [o.bbox[1] for o in out]
    assert ys == sorted(ys) and len(set(ys)) == 3
    assert all(o.bbox[3] <= 2000 * ai_ocr.MAX_LINE_HEIGHT for o in out)


def test_single_question_sheet_takes_every_line() -> None:
    lines = [OcrLine(1, (0, 0, 10, 10), "Question National Animal of India?", 0.9),
             OcrLine(1, (0, 20, 10, 10), "Ans Tiger", 0.9)]
    res = ai_segment.split_without_markers(lines, [NS(question_no=1, prompt="?")], lambda: None, ai_enabled=False)
    assert res is not None and res.segments[1].text.endswith("Ans Tiger")
    assert ai_segment.FLAG_SINGLE in res.segments[1].flags


def test_ai_ranges_become_segments() -> None:
    lines = [OcrLine(1, (0, i * 10, 10, 8), t, 0.9) for i, t in enumerate(["Name: X", "Paris", "100 C"])]
    ranges = [ai_segment.Range(question_no=1, first_line=1, last_line=1),
              ai_segment.Range(question_no=2, first_line=2, last_line=2),
              ai_segment.Range(question_no=7, first_line=0, last_line=0)]  # unknown question: ignored
    res = ai_segment._from_ranges(lines, ranges, {1, 2})
    assert res.segments[1].text == "Paris" and res.segments[2].text == "100 C"
    assert [u.text for u in res.unassigned] == ["Name: X"]


# ------------------------------------------------------------------ rubric API
def test_rubric_needs_only_the_question_text(teacher) -> None:  # noqa: ANN001
    exam = teacher.create_exam()
    ok = teacher.put(f"/exams/{exam['id']}/questions", json={"questions": [
        {"question_no": 1, "type": "short", "max_marks": 2, "prompt": "National animal of India?",
         "model_answer": None, "key_points": []}]})
    assert ok.status_code == 200, ok.text
    bad = teacher.put(f"/exams/{exam['id']}/questions", json={"questions": [
        {"question_no": 1, "type": "short", "max_marks": 2, "prompt": " ", "model_answer": None, "key_points": []}]})
    assert bad.status_code == 400 and "question text" in bad.text


# ------------------------------------------------------------------ pipeline
@pytest.mark.settings(**AI_ON)
def test_ai_gives_the_marks_for_question_only_rubrics(teacher, qo_exam, sample_bytes, fake_ai) -> None:  # noqa: ANN001
    fake_ai.responder = direct_marker(1.0)
    sub = process_in(teacher, qo_exam, sample_bytes, "ML001_full_marks")["ML001_full_marks"]
    assert len(fake_ai.calls) == 1  # one request for the whole sheet
    assert sub["status"] == "processed" and sub["total_ai_marks"] == 15
    answers = teacher.get(f"/submissions/{sub['id']}/answers").json()
    items = answers["items"] if isinstance(answers, dict) else answers
    assert {a["grader"] for a in items} == {direct_ai.GRADER_NAME}
    assert all(a["feedback"].startswith("Feedback") for a in items)


@pytest.mark.settings(**AI_ON, review_required=False)
def test_without_required_review_ai_marks_are_accepted(teacher, qo_exam, sample_bytes, fake_ai) -> None:  # noqa: ANN001
    fake_ai.responder = direct_marker(0.5, uncertain=(2,))
    sub = process_in(teacher, qo_exam, sample_bytes, "ML001_full_marks")["ML001_full_marks"]
    answers = teacher.get(f"/submissions/{sub['id']}/answers").json()
    items = answers["items"] if isinstance(answers, dict) else answers
    assert {a["status"] for a in items} == {"auto_accepted"}  # even the uncertain one
    assert sub["total_ai_marks"] == 7.5


@pytest.mark.settings(**AI_ON, review_required=False)
def test_no_ai_reply_is_never_a_silent_zero(teacher, qo_exam, sample_bytes, fake_ai) -> None:  # noqa: ANN001
    # no responder: the fake provider answers 500 to every request
    sub = process_in(teacher, qo_exam, sample_bytes, "ML001_full_marks")["ML001_full_marks"]
    answers = teacher.get(f"/submissions/{sub['id']}/answers").json()
    items = answers["items"] if isinstance(answers, dict) else answers
    assert {a["status"] for a in items} == {"needs_review"}
    assert all("ai_unavailable" in a["flags"] for a in items)


def test_ai_off_flags_question_only_answers(teacher, qo_exam, sample_bytes) -> None:  # noqa: ANN001
    sub = process_in(teacher, qo_exam, sample_bytes, "ML001_full_marks")["ML001_full_marks"]
    answers = teacher.get(f"/submissions/{sub['id']}/answers").json()
    items = answers["items"] if isinstance(answers, dict) else answers
    assert all(a["status"] == "needs_review" and "ai_off" in a["flags"] for a in items)
