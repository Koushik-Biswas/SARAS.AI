"""The grading pipeline with the ensemble and the AI marker (faked over
httpx.MockTransport; no real provider, no real key): one AI request per sheet, the AI
decides when available and the offline graders cross-check it, AI failures fall back
to the offline checker without failing the sheet, AI-side doubts force review,
nothing personal reaches the provider, and logs / traces carry no answer text."""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pytest

from app.db import SessionLocal
from app.grading import ensemble as E
from app.grading.llm_grader import USER_PREFIX
from app.llm import LLMRouter, RateLimiter
from app.llm.fakes import FakeCall, FakeLLMServer
from app.models import AiRun
from app.pipeline import processor
from conftest import drain
from scripts import make_samples as samples

GROQ = "groq:openai/gpt-oss-120b"
AI_ON = dict(ai_grading="ensemble", ai_consent=True, groq_api_key="fake-groq-key-for-tests-0001",
             ai_grading_providers=[GROQ], ai_max_retries=0, ai_verify=False)
A = samples.ANSWERS


def payload_of(call: FakeCall) -> dict:
    assert call.prompt.startswith(USER_PREFIX)
    return json.loads(call.prompt[len(USER_PREFIX):])


def marker(verdicts: dict | None = None, uncertain: tuple = ()):  # noqa: ANN201
    """A fake AI marker: every key point "matched" (quoting the answer's first words)
    unless `verdicts[(question_no, index)]` says otherwise."""

    def respond(call: FakeCall) -> dict:
        questions = []
        for q in payload_of(call)["questions"]:
            kps = []
            for kp in q["key_points"]:
                v = (verdicts or {}).get((q["question_no"], kp["index"]), "matched")
                quote = " ".join(q["answer_text"].split()[:6]) if v != "missed" else ""
                kps.append({"index": kp["index"], "verdict": v, "evidence_quote": quote, "reason": "As stated."})
            questions.append({"question_no": q["question_no"], "key_points": kps,
                              "uncertain": q["question_no"] in uncertain,
                              "feedback": f"Feedback for question {q['question_no']}."})
        return {"questions": questions}

    return respond


@pytest.fixture
def fake_ai(monkeypatch) -> FakeLLMServer:  # noqa: ANN001
    """Every router the pipeline builds talks to this fake server (fake clock: no real waits)."""
    server = FakeLLMServer()

    def make(settings, session_factory):  # noqa: ANN001, ANN202
        clock = [0.0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        return LLMRouter(settings, session_factory, transport=server.transport,
                         limiter=RateLimiter(clock=lambda: clock[0]), sleep=sleep)

    monkeypatch.setattr(processor, "make_ai_router", make)
    return server


def ai_runs(submission_id: str) -> list[AiRun]:
    with SessionLocal() as db:
        return list(db.query(AiRun).filter_by(submission_id=submission_id).order_by(AiRun.created_at))


# ------------------------------------------------------------------ AI off (the default)
def test_lexical_only_answers_go_to_review_at_the_default_threshold(teacher, process, fake_ai):
    """Routing is the point here: without a cross-check the lexical grader (prior 0.6)
    is not trusted to auto-accept anything at the production threshold (0.80)."""
    sub = process("ML001_full_marks")["ML001_full_marks"]
    assert fake_ai.calls == []  # AI off: nothing is sent
    for a in teacher.answers(sub["id"]):
        assert a["status"] == "needs_review" and "low_confidence" in a["flags"]
        assert "low_reliability_grader" in a["flags"] and a["grader"] == "lexical"
        assert a["grading_confidence"] == pytest.approx(0.6 * E.NO_SECONDARY_FACTOR)
        t = a["grading_trace"]
        assert t["graders_used"] == ["lexical"] and t["ai"] == {"status": "off"}
        assert t["agreement_factor"] == E.NO_SECONDARY_FACTOR and t["routing"]["status"] == "needs_review"
        assert a["verdict"] == "correct" and a["feedback"].startswith("Covered:")
    assert sub["total_ai_marks"] == 15 and sub["grader"] == "lexical"


# ------------------------------------------------------------------ AI on
@pytest.mark.settings(**AI_ON)
def test_one_ai_request_per_sheet_and_the_ai_decides(teacher, process, fake_ai):
    fake_ai.responder = marker()
    sub = process("ML001_full_marks")["ML001_full_marks"]
    assert len(fake_ai.calls) == 1  # four answers, one request
    assert sorted(q["question_no"] for q in payload_of(fake_ai.calls[0])["questions"]) == [1, 2, 3, 4]
    assert sub["grader"] == "lexical,llm" and sub["total_ai_marks"] == 15
    for a in teacher.answers(sub["id"]):
        assert a["grader"] == "llm+lexical" and a["verdict"] == "correct"
        assert a["feedback"] == f"Feedback for question {a['question_no']}."
        t = a["grading_trace"]
        assert t["primary"] == "llm" and t["secondary"] == "lexical"
        assert t["ai"]["status"] == "used" and t["ai"]["provider"] == "groq" and t["ai"]["cached"] is False
        assert all(k["source"] == "llm" and set(k["statuses_by_grader"]) == {"llm", "lexical"}
                   for k in a["key_points"])
        assert all(k["evidence_sentence"] and k["evidence_sentence"] in a["ocr_text"] for k in a["key_points"])
        # The AI and the checker agree on every criterion of these full answers: confident.
        assert t["disputed"] == [] and not t["marks_gap_exceeded"] and t["agreement_factor"] == 1.0
        assert a["grading_confidence"] == pytest.approx(0.9) and a["status"] == "auto_accepted"
        assert t["routing"] == {**t["routing"], "status": "auto_accepted", "review_flags": [], "forced_review": False}
    runs = ai_runs(sub["id"])  # recorded by the worker, committed independently of the sheet
    assert [r.status for r in runs] == ["ok"] and runs[0].exam_id == sub["exam_id"]


@pytest.mark.settings(**AI_ON)
def test_ai_doubts_and_disagreements_force_review(teacher, process, fake_ai):
    fake_ai.responder = marker(verdicts={(1, 0): "missed"}, uncertain=(2,))
    sub = process("ML001_full_marks")["ML001_full_marks"]
    q = {a["question_no"]: a for a in teacher.answers(sub["id"])}
    # Q1: the AI says KP1 is missing, the checker found it: the AI's status stands, disputed.
    q1 = q[1]
    assert q1["key_points"][0]["status"] == "missed" and q1["key_points"][0]["disputed"]
    assert q1["key_points"][0]["statuses_by_grader"]["lexical"] in ("matched", "partial")
    assert q1["ai_marks"] == 2 and q1["status"] == "needs_review" and "graders_disagree" in q1["flags"]
    assert q1["grading_confidence"] == pytest.approx(0.9 * E.DISAGREE_FACTOR)
    # Q2: the AI was unsure.
    assert q[2]["status"] == "needs_review" and "ai_uncertain" in q[2]["flags"]
    assert q[2]["grading_trace"]["routing"]["review_flags"]


@pytest.mark.settings(**AI_ON)
def test_ai_outage_falls_back_to_the_offline_checker(teacher, process, fake_ai):
    # no scripted reply: the fake provider answers 500 to every request
    sub = process("ML001_full_marks")["ML001_full_marks"]
    assert len(fake_ai.calls) == 1 and sub["status"] == "processed" and sub["total_ai_marks"] == 15
    for a in teacher.answers(sub["id"]):
        assert "ai_unavailable_fallback" in a["flags"] and a["grader"] == "lexical"
        t = a["grading_trace"]["ai"]
        assert t["status"] == "fallback" and t["reason"] == "unavailable"
        assert t["details"] == ["groq:openai/gpt-oss-120b: error"]
    assert [r.status for r in ai_runs(sub["id"])] == ["error"]


@pytest.mark.settings(**AI_ON)
def test_unexpected_ai_failures_never_fail_the_sheet(teacher, process, fake_ai, monkeypatch):
    def broken(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise KeyError("schema drift")

    monkeypatch.setattr(E, "grade_sheet_llm", broken)
    sub = process("ML001_full_marks")["ML001_full_marks"]
    assert sub["status"] == "processed"
    assert all(a["grading_trace"]["ai"] == {"status": "fallback", "reason": "error", "style": "moderate"}
               for a in teacher.answers(sub["id"]))

    def no_router(settings, factory):  # noqa: ANN001, ANN202
        raise OSError("no sockets")

    monkeypatch.setattr(processor, "make_ai_router", no_router)
    sub = process("ML002_partial")["ML002_partial"]
    assert sub["status"] == "processed"
    assert all("ai_unavailable_fallback" in a["flags"] for a in teacher.answers(sub["id"]))


@pytest.mark.settings(**{**AI_ON, "ai_consent": False})
def test_without_consent_nothing_is_sent(teacher, process, fake_ai):
    fake_ai.responder = marker()
    sub = process("ML001_full_marks")["ML001_full_marks"]
    assert fake_ai.calls == [] and ai_runs(sub["id"]) == []
    assert all(a["grading_trace"]["ai"] == {"status": "off"} and "ai_unavailable_fallback" not in a["flags"]
               for a in teacher.answers(sub["id"]))


@pytest.mark.settings(**{**AI_ON, "groq_api_key": None})
def test_no_usable_provider_is_a_fallback_without_a_request(teacher, process, fake_ai):
    sub = process("ML001_full_marks")["ML001_full_marks"]
    assert fake_ai.calls == []
    assert all(a["grading_trace"]["ai"]["reason"] == "no_usable_provider" for a in teacher.answers(sub["id"]))


# ------------------------------------------------------------------ requests that regrade
@pytest.mark.settings(**AI_ON)
def test_corrections_and_region_edits_regrade_with_the_ai_inside_the_request(teacher, exam, process, fake_ai):
    fake_ai.responder = marker()
    sid = process("ML002_partial")["ML002_partial"]["id"]
    assert len(fake_ai.calls) == 1
    q2 = teacher.answer_for(sid, 2)

    t0 = time.perf_counter()
    r = teacher.patch(f"/answers/{q2['answer_id']}/review", json={"corrected_text": A["q2_full"], "approve": False})
    assert r.status_code == 200, r.text
    assert len(fake_ai.calls) == 2 and [q["question_no"] for q in payload_of(fake_ai.calls[1])["questions"]] == [2]
    assert r.json()["grader"] == "llm+lexical" and r.json()["ai_marks"] == 5

    # A region edit flushes before regrading; the router writes through the request's own
    # connection, so it never waits for SQLite's write lock (the busy timeout is 30 s).
    region = next(x for x in teacher.regions(sid) if x["question_id"] == exam["qid"][3])
    assert teacher.patch(f"/regions/{region['id']}", json={"action": "blank"}).status_code == 200
    r = teacher.patch(f"/regions/{region['id']}", json={"action": "unblank"})
    assert r.status_code == 200, r.text
    assert len(fake_ai.calls) == 3  # blanking sent nothing; unblanking re-marked Q3 alone
    assert time.perf_counter() - t0 < 15
    assert [x.status for x in ai_runs(sid)] == ["ok", "ok", "ok"]  # committed with each request


@pytest.mark.settings(**AI_ON)
def test_exam_regrade_reuses_cached_ai_judgements(teacher, exam, process, fake_ai):
    fake_ai.responder = marker()
    subs = process("ML001_full_marks", "ML002_partial")
    assert len(fake_ai.calls) == 2  # one per sheet
    assert teacher.post(f"/exams/{exam['id']}/regrade").status_code == 200
    assert len(fake_ai.calls) == 2  # same sheets, same rubric: served from the ai_runs cache
    for sub in subs.values():
        assert all(a["grading_trace"]["ai"]["cached"] is True for a in teacher.answers(sub["id"]))
        assert [r.status for r in ai_runs(sub["id"])] == ["ok", "cache_hit"]


# ------------------------------------------------------------------ privacy
@pytest.mark.settings(**AI_ON)
def test_student_ids_never_reach_the_provider_and_logs_hold_no_answer_text(teacher, exam, fake_ai, caplog):
    fake_ai.responder = marker()
    ref = "21BCE1234"
    q1 = f"{A['q1_full']} My roll no is {ref}, call me at 98765 43210 or mail asha.rao@example.com."
    pdf = samples.build_text_pdf([[f"Q1. {q1}", f"Q2. {A['q2_full']}", f"Q3. {A['q3_full']}",
                                   f"Q4. {A['q4_full']}"]])
    caplog.set_level(logging.DEBUG, logger="app")
    r = teacher.upload(exam["id"], [("s.pdf", pdf)], refs=[ref])
    assert r.status_code == 202, r.text
    drain()
    prompt = fake_ai.calls[0].prompt + (fake_ai.calls[0].system or "")
    for private in (ref, "98765", "43210", "asha.rao@example.com"):
        assert private not in prompt
    sid = r.json()["submissions"][0]["id"]
    answers = teacher.answers(sid)
    assert answers[0]["ocr_text"].endswith("asha.rao@example.com.")  # stored text is the student's own
    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "Graded submission" in messages and "graders=lexical,llm" in messages
    for text in ("labelled data", "Overfitting is when", ref, "asha.rao", "98765"):
        assert text not in messages
    for a in answers:
        trace = json.dumps(a["grading_trace"])
        assert ref not in trace and a["ocr_text"][:40] not in trace
    with SessionLocal() as db:
        stored = json.dumps(db.query(AiRun).filter_by(submission_id=sid).one().request_json)
    assert ref not in stored and "98765" not in stored


# ------------------------------------------------------------------ thresholds and the local cross-check
@pytest.mark.settings(grader="nli-onnx")
def test_each_local_grader_gets_its_own_thresholds(teacher, process, monkeypatch):
    """The pipeline used to pass Settings' 0.6 / 0.4 / 0.5 to every scorer, wrong for NLI
    probabilities. A stand-in NLI scorer (no model load) records what it is given."""
    from app.grading import engine, scorers
    from app.grading.engine import Thresholds

    class StandInNli(scorers.LexicalScorer):
        name = "nli-onnx"

        def score(self, premises, hypotheses, weights=None):  # noqa: ANN001, ANN201
            e = np.full((len(hypotheses), len(premises)), 0.9)
            return e, np.zeros_like(e)

    real_get, real_grade = scorers.get_scorer, engine.grade_answer
    seen: list[tuple[str, Thresholds]] = []

    def get_scorer(kind, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        return StandInNli() if kind == "nli-onnx" else real_get(kind, *args, **kwargs)

    def grade_answer(question, text, scorer, th=None):  # noqa: ANN001, ANN202
        seen.append((scorer.name, th))
        return real_grade(question, text, scorer, th)

    monkeypatch.setattr(scorers, "get_scorer", get_scorer)
    monkeypatch.setattr(engine, "grade_answer", grade_answer)
    sub = process("ML001_full_marks")["ML001_full_marks"]
    assert {name for name, _ in seen} == {"nli-onnx", "lexical"} and len(seen) == 8
    assert all(th == Thresholds(0.5, 0.15, 0.95) for name, th in seen if name == "nli-onnx")
    assert all(th == Thresholds(0.6, 0.4, 0.5) for name, th in seen if name == "lexical")
    a = teacher.answer_for(sub["id"], 1)
    assert a["grader"] == "nli-onnx+lexical" and a["grading_trace"]["secondary"] == "lexical"
    assert sub["grader"] == "lexical,nli-onnx"
