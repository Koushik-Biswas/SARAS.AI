"""AI marker (app.grading.llm_grader): the AI only judges key points and quotes the
answer; quotes are verified and marks are computed in code. Providers are faked
(app.llm.fakes over httpx.MockTransport) — no real provider, no real key."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.db import Base
from app.grading.llm_grader import (
    PROMPT_VERSION,
    STYLES,
    USER_PREFIX,
    AnswerItem,
    find_quote,
    grade_answer_llm,
    grade_sheet_llm,
    template_feedback,
    verdict_for,
)
from app.llm import LLMRouter, LLMUnavailable, RateLimiter
from app.llm.fakes import FakeCall, FakeLLMServer, FakeReply
from app.models import AiRun

GROQ = "groq:openai/gpt-oss-120b"
GEMINI = "gemini:gemini-3.5-flash-lite"

Q1 = AnswerItem(
    question_no=1, type="long", max_marks=3, prompt="What is supervised learning?",
    model_answer="A model is trained on labelled data to map inputs to outputs and predict unseen inputs.",
    key_points=[
        {"text": "the model is trained on labelled data", "weight": 1, "keywords": ["labelled"]},
        {"text": "it learns a mapping from inputs to outputs", "weight": 1, "keywords": ["mapping"]},
        {"text": "it predicts outputs for new unseen inputs", "weight": 1, "keywords": ["predict"]},
    ],
    answer_text="Supervised learning trains a model on labelled examples.\nIt learns a function that maps "
                "inputs to outputs. Mail asha.rao@example.com or call +91 98765 43210. Roll CS2021001, STU-001, "
                "Asha Rao.",
)
Q2 = AnswerItem(
    question_no=2, type="paragraph", max_marks=2, prompt="Define overfitting.", model_answer=None,
    key_points=[{"text": "the model fits the training data too closely", "weight": 1},
                {"text": "it performs poorly on new unseen data", "weight": 1}],
    answer_text="Overfitting is when the model memorises the training data. It then does badly on unseen data.",
)


def kp(index: int, verdict: str, quote: str = "", reason: str = "Because.") -> dict:
    return {"index": index, "verdict": verdict, "evidence_quote": quote, "reason": reason}


def q(no: int, *kps: dict, uncertain: bool = False, feedback: str = "Good work.") -> dict:
    return {"question_no": no, "key_points": list(kps), "uncertain": uncertain, "feedback": feedback}


def sheet(*questions: dict, **extra) -> dict:  # noqa: ANN003
    return {"questions": list(questions), **extra}


def payload_of(call: FakeCall) -> dict:
    assert call.prompt.startswith(USER_PREFIX)
    return json.loads(call.prompt[len(USER_PREFIX):])


def all_matched(call: FakeCall) -> dict:
    """Fake marker: every key point matched, quoting the first words of the answer."""
    return sheet(*(q(qd["question_no"], *(kp(k["index"], "matched", " ".join(qd["answer_text"].split()[:3]))
                                         for k in qd["key_points"])) for qd in payload_of(call)["questions"]))


@pytest.fixture
def session_factory():  # noqa: ANN201
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[AiRun.__table__])
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def harness(tmp_path: Path, session_factory):  # noqa: ANN001, ANN201
    routers: list[LLMRouter] = []

    def make(server: FakeLLMServer | None = None, **over):  # noqa: ANN003, ANN202
        server = server or FakeLLMServer()
        base = dict(_env_file=None, env="test", database_url=f"sqlite:///{(tmp_path / 'unused.db').as_posix()}",
                    storage_dir=tmp_path / "storage", ai_grading="ensemble", ai_consent=True, gemini_consent=True,
                    groq_api_key="fake-groq-secret-0001", gemini_api_key="fake-gemini-secret-0002",
                    ai_grading_providers=[GROQ, GEMINI], ai_max_retries=1)
        base.update(over)
        clock = [0.0]  # fake time: sleeping (backoff, rate-limit waits) just advances it

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        router = LLMRouter(Settings(**base), session_factory, transport=server.transport,
                           limiter=RateLimiter(clock=lambda: clock[0]), sleep=sleep)
        routers.append(router)
        return router, server

    yield make
    for r in routers:
        r.close()


# ------------------------------------------------------------------ valid path
def test_valid_sheet_one_request_marks_in_code(harness) -> None:  # noqa: ANN001
    router, server = harness()
    server.script("groq", sheet(
        q(1, kp(0, "matched", "trains a model on labelled examples"),
          kp(1, "Matched", "It learns a function that maps inputs to outputs."), kp(2, "missed"),
          feedback="Clear definition; add how it predicts unseen inputs."),
        q(2, kp(0, "matched", "the model memorises the training data"), kp(1, "partial", "does badly on unseen data")),
    ))
    grades = grade_sheet_llm(router, [Q1, Q2], redact_terms=["Asha Rao", "STU-001"])

    assert len(server.calls) == 1  # one request per answer sheet
    g1, g2 = grades[1], grades[2]
    assert (g1.marks, g1.verdict, g1.max_marks) == (2.0, "partially_correct", 3.0)
    assert [k["status"] for k in g1.key_points] == ["matched", "matched", "missed"]
    assert [k["awarded"] for k in g1.key_points] == [1.0, 1.0, 0.0]
    # evidence is the student's own span from the original answer
    assert g1.key_points[1]["evidence_sentence"] == "It learns a function that maps inputs to outputs"
    assert g1.key_points[2]["evidence_sentence"] is None
    assert g1.feedback.startswith("Clear definition") and g1.flags == []
    assert (g2.marks, g2.verdict) == (1.5, "partially_correct")
    assert (g1.provider, g1.model, g1.cached, g1.prompt_version) == ("groq", "openai/gpt-oss-120b", False,
                                                                       PROMPT_VERSION)

    call = server.calls[0]
    sent = payload_of(call)
    q1 = sent["questions"][0]
    assert set(q1) == {"question_no", "type", "max_marks", "prompt", "model_answer", "key_points", "answer_text"}
    assert q1["key_points"][0] == {"index": 0, "text": "the model is trained on labelled data", "weight": 1.0}
    for private in ("asha.rao@example.com", "98765", "CS2021001", "STU-001", "Asha", "Rao"):
        assert private not in call.prompt and private not in (call.system or "")
    assert "[EMAIL]" in q1["answer_text"] and "[PHONE]" in q1["answer_text"] and "[ID]" in q1["answer_text"]
    assert "Do NOT compute marks" in call.system and STYLES["moderate"] in call.system
    assert call.body["response_format"]["json_schema"]["name"] == "grade_sheet"


def test_quotes_not_in_answer_are_downgraded_and_flagged(harness) -> None:  # noqa: ANN001
    router, server = harness()
    server.script("groq", sheet(q(
        1,
        kp(0, "matched", "the model generalises perfectly to any data"),  # invented
        kp(1, "contradicted", ""),  # no quote at all
        kp(2, "partial", "predicts outputs for unseen inputs"),  # paraphrase, not a quote
    )))
    g = grade_sheet_llm(router, [Q1])[1]
    assert [k["status"] for k in g.key_points] == ["missed"] * 3
    assert all(k["evidence_sentence"] is None for k in g.key_points)
    assert (g.marks, g.verdict) == (0.0, "incorrect")
    assert g.flags.count("ai_unverified_evidence") == 1
    assert g.key_points[0]["reason"].startswith("[evidence not found in the answer]")
    assert [k["evidence_check"] for k in g.key_points] == ["not_found", "not_found", "not_found"]


def test_verified_contradiction_scores_zero_and_is_flagged(harness) -> None:  # noqa: ANN001
    router, server = harness()
    server.script("groq", sheet(q(2, kp(0, "contradicted", "memorises the training data"),
                                  kp(1, "matched", "badly on unseen data"))))
    g = grade_sheet_llm(router, [Q2])[2]
    assert [k["status"] for k in g.key_points] == ["contradicted", "matched"]
    assert g.key_points[0]["evidence_sentence"] == "memorises the training data"
    assert (g.marks, g.verdict) == (1.0, "partially_correct") and "ai_contradiction" in g.flags


def test_fuzzy_quote_tolerates_ocr_noise() -> None:
    answer = "Overfiting: the model lerns the noize in the trainig data, so it fails on new data."
    assert find_quote(answer, "the model learns the noise in the training data") == \
        "the model lerns the noize in the trainig data"
    assert find_quote(answer, "  THE MODEL lerns\nthe noize  ") == "the model lerns the noize"
    assert find_quote(answer, "“so it fails on new data.”") == "so it fails on new data"
    assert find_quote(answer, "the model memorises everything") is None
    assert find_quote(answer, "...") is None and find_quote(answer, "") is None


def test_extra_model_fields_are_ignored(harness) -> None:  # noqa: ANN001
    router, server = harness()
    out = sheet(q(2, {**kp(0, "matched", "memorises the training data"), "awarded": 50, "marks": 5},
                  {**kp(1, "missed"), "awarded": 50}),
                total=999, grand_total=999)
    out["questions"][0].update(marks=99, total_marks=99, verdict="correct", score="2/2")
    server.script("groq", out)
    g = grade_sheet_llm(router, [Q2])[2]
    assert (g.marks, g.verdict) == (1.0, "partially_correct")
    assert sum(k["awarded"] for k in g.key_points) == 1.0


def test_unknown_duplicate_and_missing_indices(harness, session_factory) -> None:  # noqa: ANN001
    router, server = harness(ai_grading_providers=[GROQ])
    server.script("groq", sheet(q(
        1,
        kp(7, "matched", "trains a model on labelled examples"),  # out of range: ignored
        kp(-1, "matched", "trains a model on labelled examples"),  # out of range: ignored
        kp(0, "matched", "trains a model on labelled examples"),
        kp(0, "matched", "trains a model on labelled examples"),  # identical repeat: harmless
        kp(1, "matched", "maps inputs to outputs"),
        kp(2, "missed"),
    )))
    g = grade_sheet_llm(router, [Q1])[1]
    assert [k["status"] for k in g.key_points] == ["matched", "matched", "missed"]
    assert g.marks == 2.0 and g.flags == []

    # A key point left out, or judged twice with different verdicts: the question is not
    # graded at all (no made-up "missed" / 0) and the reply is never stored as 'ok'.
    for bad in (sheet(q(1, kp(0, "matched", "trains a model on labelled examples"), kp(1, "missed"))),
                sheet(q(1, kp(0, "missed"), kp(0, "matched", "trains a model on labelled examples"),
                        kp(1, "missed"), kp(2, "missed")))):
        server.script("groq", bad)
        grades = grade_sheet_llm(router, [Q1], style="strict")  # a new prompt: no cache hit
        assert dict(grades) == {} and grades.missing == {1: "ai_incomplete"}
    with session_factory() as db:
        rows = db.query(AiRun).filter(AiRun.status != "cache_hit").all()
    assert sorted(r.status for r in rows) == ["invalid", "invalid", "ok"]
    assert all("incomplete" in (r.error or "") for r in rows if r.status == "invalid")


def test_question_left_out_is_absent_never_a_zero(harness, session_factory) -> None:  # noqa: ANN001
    """Review finding ai-layer #5: {"questions": []} for a 2-question sheet gave both
    questions 0.0 'incorrect', was stored as 'ok' and replayed from the cache forever."""
    router, server = harness(ai_grading_providers=[GROQ])
    server.script("groq", sheet(), sheet())
    grades = grade_sheet_llm(router, [Q1, Q2])
    assert dict(grades) == {} and grades.missing == {1: "ai_incomplete", 2: "ai_incomplete"}
    again = grade_sheet_llm(router, [Q1, Q2])  # not served from the cache: asked again
    assert dict(again) == {} and len(server.calls) == 2
    with session_factory() as db:
        assert sorted(r.status for r in db.query(AiRun)) == ["invalid", "invalid"]


def test_missing_and_unknown_questions(harness, session_factory) -> None:  # noqa: ANN001
    router, server = harness()
    server.script("groq", sheet(
        q(9, kp(0, "matched", "anything")),  # not on this sheet: ignored
        q(1, kp(0, "matched", "trains a model on labelled examples"), kp(1, "missed"), kp(2, "missed")),
        q(1, kp(0, "matched", "trains a model on labelled examples"), kp(1, "missed"), kp(2, "missed")),
    ))  # Q1 twice (the same judgement: harmless); Q2 left out
    server.script("gemini", sheet(q(2, kp(0, "matched", "the model memorises the training data"),
                                    kp(1, "partial", "does badly on unseen data"))))
    grades = grade_sheet_llm(router, [Q1, Q2])
    assert grades[1].marks == 1.0 and grades[1].provider == "groq"
    # Q2 was asked again, alone, from the next provider.
    assert [c.provider for c in server.calls] == ["groq", "gemini"]
    assert [qd["question_no"] for qd in payload_of(server.calls[1])["questions"]] == [2]
    assert (grades[2].marks, grades[2].provider) == (1.5, "gemini") and grades.missing == {}
    with session_factory() as db:
        assert sorted(r.status for r in db.query(AiRun)) == ["invalid", "ok"]

    # Re-grading: the incomplete first reply is not replayed (groq is asked again), the
    # complete retry for Q2 is.
    server.script("groq", sheet(q(1, kp(0, "matched", "trains a model on labelled examples"),
                                  kp(1, "missed"), kp(2, "missed"))))
    again = grade_sheet_llm(router, [Q1, Q2])
    assert [c.provider for c in server.calls] == ["groq", "gemini", "groq"]
    assert again[2].cached and not again[1].cached


def test_retry_that_also_fails_leaves_the_question_out(harness) -> None:  # noqa: ANN001
    router, server = harness()
    server.script("groq", sheet(q(1, kp(0, "matched", "trains a model on labelled examples"),
                                  kp(1, "missed"), kp(2, "missed"))))
    # gemini: no scripted reply, the fake answers 500 to the retry of Q2
    grades = grade_sheet_llm(router, [Q1, Q2])
    assert set(grades) == {1} and grades.missing == {2: "ai_incomplete"}
    assert grades.reasons and all("gemini" in r for r in grades.reasons)


def test_marks_capped_at_max_and_verdict_rules(harness) -> None:  # noqa: ANN001
    item = AnswerItem(question_no=3, type="long", max_marks=3, answer_text="Alpha beta gamma. Delta epsilon.",
                      key_points=[{"text": "alpha beta", "weight": 2}, {"text": "delta", "weight": 2}])
    router, server = harness()
    server.script("groq", sheet(q(3, kp(0, "matched", "Alpha beta gamma"), kp(1, "matched", "Delta epsilon"))))
    g = grade_sheet_llm(router, [item])[3]
    assert (g.marks, g.verdict) == (3.0, "correct")
    assert verdict_for(0.0, 5) == "incorrect" and verdict_for(5.0, 5) == "correct"
    assert verdict_for(2.5, 5) == "partially_correct" and verdict_for(3.33, 10 / 3) == "correct"


# ------------------------------------------------------------------ what is (not) sent
def test_maths_blank_and_rubricless_questions_are_not_sent(harness) -> None:  # noqa: ANN001
    router, server = harness()
    items = [
        AnswerItem(question_no=1, type="maths", max_marks=4, answer_text="x = 2",
                   key_points=[{"text": "x=2", "weight": 4}]),
        AnswerItem(question_no=2, type="short", max_marks=2, answer_text="   ",
                   key_points=[{"text": "a", "weight": 2}]),
        AnswerItem(question_no=3, type="short", max_marks=2, answer_text="Some text.", key_points=[]),
    ]
    assert grade_sheet_llm(router, items) == {} and server.calls == []

    # No key points but a model answer: the model answer is one key point worth max_marks.
    server.responder = all_matched
    item = AnswerItem(question_no=4, type="short", max_marks=2, answer_text="Gradient descent minimises loss.",
                      model_answer="It minimises the loss iteratively.")
    g = grade_sheet_llm(router, [*items, item])[4]
    assert payload_of(server.calls[0])["questions"][0]["key_points"] == [
        {"index": 0, "text": "It minimises the loss iteratively.", "weight": 2.0}]
    assert (g.marks, g.verdict) == (2.0, "correct")
    assert grade_answer_llm(router, items[0]) is None


def test_large_sheet_is_split_under_the_token_budget(harness) -> None:  # noqa: ANN001
    long_answer = "The learning rate controls the step size of each update in gradient descent. " * 12
    items = [AnswerItem(question_no=n, type="long", max_marks=2, answer_text=long_answer,
                        key_points=[{"text": "step size", "weight": 1}, {"text": "gradient", "weight": 1}])
             for n in (1, 2, 3)]
    router, server = harness(server=FakeLLMServer(responder=all_matched))
    assert set(grade_sheet_llm(router, items)) == {1, 2, 3} and len(server.calls) == 1

    small, server2 = harness(server=FakeLLMServer(responder=all_matched), ai_max_input_tokens_per_call=1100)
    grades = grade_sheet_llm(small, items, style="strict")
    assert set(grades) == {1, 2, 3} and len(server2.calls) > 1
    sent = [qd["question_no"] for c in server2.calls for qd in payload_of(c)["questions"]]
    assert sorted(sent) == [1, 2, 3]
    assert all(g.marks == 2.0 for g in grades.values())


def test_many_key_points_are_split_to_fit_the_output_cap(harness) -> None:  # noqa: ANN001
    items = [AnswerItem(question_no=n, type="short", max_marks=4, answer_text=f"Answer {n} covers a b c d.",
                        key_points=[{"text": t, "weight": 1} for t in "abcd"]) for n in range(1, 13)]
    router, server = harness(server=FakeLLMServer(responder=all_matched), ai_max_input_tokens_per_call=10**6)
    assert set(grade_sheet_llm(router, items)) == set(range(1, 13))  # 48 key points: too many for one reply
    assert len(server.calls) == 2
    assert all(1200 <= c.body["max_tokens"] <= 4096 for c in server.calls)


def test_failed_chunk_leaves_only_its_questions_out(harness) -> None:  # noqa: ANN001
    def flaky(call: FakeCall):  # noqa: ANN202
        if [qd["question_no"] for qd in payload_of(call)["questions"]] == [2]:
            return FakeReply(status=400, body={"error": {"message": "bad request"}})
        return all_matched(call)

    items = [AnswerItem(question_no=n, type="long", max_marks=1, answer_text="Word " * 300,
                        key_points=[{"text": "word", "weight": 1}]) for n in (1, 2, 3)]
    router, server = harness(server=FakeLLMServer(responder=flaky), ai_grading_providers=[GROQ],
                             ai_max_input_tokens_per_call=900)
    grades = grade_sheet_llm(router, items)
    assert set(grades) == {1, 3} and grades.missing == {2: "ai_unavailable_fallback"}
    assert len(server.calls) == 3


def test_no_consent_raises_and_sends_nothing(harness) -> None:  # noqa: ANN001
    router, server = harness(ai_consent=False, server=FakeLLMServer(responder=all_matched))
    with pytest.raises(LLMUnavailable):
        grade_sheet_llm(router, [Q1, Q2])
    assert server.calls == []


def test_regrading_the_same_sheet_is_a_cache_hit(harness) -> None:  # noqa: ANN001
    router, server = harness(server=FakeLLMServer(responder=all_matched))
    first = grade_sheet_llm(router, [Q1, Q2])
    again = grade_sheet_llm(router, [Q1, Q2])
    assert len(server.calls) == 1
    assert all(g.cached for g in again.values()) and not any(g.cached for g in first.values())
    assert {n: g.marks for n, g in again.items()} == {n: g.marks for n, g in first.items()}
    grade_sheet_llm(router, [Q1, Q2], style="lenient")  # the style is part of the prompt and the cache key
    assert len(server.calls) == 2 and STYLES["lenient"] in server.calls[1].system


def test_inputs_styles_and_helpers(harness) -> None:  # noqa: ANN001
    router, server = harness(server=FakeLLMServer(responder=all_matched))
    with pytest.raises(ValueError):
        grade_sheet_llm(router, [Q1], style="harsh")
    with pytest.raises(ValueError):
        grade_sheet_llm(router, [Q1, Q1])
    question = SimpleNamespace(question_no=5, type="short", max_marks=1.0, prompt="p", model_answer="m",
                               key_points=[{"text": "kp", "weight": 1}])
    item = AnswerItem.from_question(question, "An answer with kp.")
    g = grade_answer_llm(router, item)
    assert g is not None and g.question_no == 5 and g.as_dict()["marks"] == 1.0
    as_dict = {"question_no": 6, "type": "short", "max_marks": 1, "answer_text": "Answer text here.",
               "key_points": [{"text": "kp", "weight": 1}], "student_ref": "never used"}
    assert grade_answer_llm(router, as_dict).question_no == 6
    assert "never used" not in server.calls[-1].prompt


# ------------------------------------------------------------------ review findings (ai-layer)
LR = AnswerItem(question_no=7, type="short", max_marks=1, prompt="Does a higher learning rate speed up training?",
                key_points=[{"text": "a higher learning rate makes training converge faster", "weight": 1}],
                answer_text="Increasing the learning rate does not make training converge faster.")


def test_misquote_that_drops_not_awards_nothing(harness) -> None:  # noqa: ANN001
    """#0: the AI said 'matched' with a quote that drops the student's 'not': it was 1/1,
    'correct', no flag, and its 'Full marks' feedback was kept."""
    router, server = harness()
    server.script("groq", sheet(q(7, kp(0, "matched", "Increasing the learning rate does make training converge faster"),
                                  feedback="Excellent: a higher learning rate speeds up training. Full marks.")))
    g = grade_sheet_llm(router, [LR])[7]
    assert (g.marks, g.verdict) == (0.0, "incorrect")
    assert (g.key_points[0]["status"], g.key_points[0]["evidence_check"]) == ("missed", "not_found")
    assert "ai_unverified_evidence" in g.flags
    # #10: the feedback describes the computed marks, not the AI's withdrawn claim
    assert g.feedback_source == "template"
    assert g.feedback == 'Missing: "a higher learning rate makes training converge faster".'

    # A true substring that leaves the negation outside the quote is not evidence either.
    server.script("groq", sheet(q(7, kp(0, "matched", "make training converge faster"))))
    g = grade_sheet_llm(router, [LR], style="lenient")[7]
    assert g.marks == 0.0 and "ai_unverified_evidence" in g.flags
    # Quoting the negation for a contradiction is fine.
    server.script("groq", sheet(q(7, kp(0, "contradicted", "does not make training converge faster"))))
    g = grade_sheet_llm(router, [LR], style="strict")[7]
    assert g.key_points[0]["status"] == "contradicted" and "ai_contradiction" in g.flags
    assert g.key_points[0]["evidence_sentence"] == "does not make training converge faster"
    assert g.key_points[0]["evidence_check"] == "exact" and "ai_unverified_evidence" not in g.flags


def test_ocr_noise_is_accepted_but_sent_to_review(harness) -> None:  # noqa: ANN001
    item = AnswerItem(question_no=8, type="short", max_marks=2,
                      answer_text="Plants need the rnitochondria and the ce11 wall to make energy.",
                      key_points=[{"text": "mitochondria release energy", "weight": 1},
                                  {"text": "the cell wall gives support", "weight": 1}])
    router, server = harness()
    server.script("groq", sheet(q(8, kp(0, "matched", "Plants need the mitochondria"),
                                  kp(1, "partial", "the cell wall to make energy"))))
    g = grade_sheet_llm(router, [item])[8]
    assert (g.marks, [k["status"] for k in g.key_points]) == (1.5, ["matched", "partial"])
    # the evidence is the student's own (noisy) text, never the AI's corrected wording
    assert [k["evidence_sentence"] for k in g.key_points] == ["Plants need the rnitochondria",
                                                              "the ce11 wall to make energy"]
    assert [k["evidence_check"] for k in g.key_points] == ["fuzzy", "fuzzy"]
    assert g.flags == ["ai_fuzzy_evidence"] and g.feedback_source == "ai"


def test_one_letter_and_one_word_quotes_award_nothing(harness) -> None:  # noqa: ANN001
    """#1: 'I do not know.' with quotes 'o' and 'know' scored 2/2, 'correct', no flag."""
    item = AnswerItem(question_no=9, type="short", max_marks=2, answer_text="I do not know.",
                      key_points=[{"text": "TCP is connection oriented", "weight": 1},
                                  {"text": "UDP is connectionless", "weight": 1}])
    router, server = harness()
    server.script("groq", sheet(q(9, kp(0, "matched", "o"), kp(1, "matched", "know"))))
    g = grade_sheet_llm(router, [item])[9]
    assert (g.marks, g.verdict) == (0.0, "incorrect")
    assert all(k["evidence_sentence"] is None and k["evidence_check"] == "not_found" for k in g.key_points)
    assert "ai_unverified_evidence" in g.flags

    # A real but too-short span (1 word, 3 letters) out of a longer answer is not evidence.
    atp = AnswerItem(question_no=10, type="short", max_marks=1, answer_text="Mitochondria make ATP for the cell.",
                     key_points=[{"text": "mitochondria make ATP", "weight": 1}])
    server.script("groq", sheet(q(10, kp(0, "matched", "ATP"))))
    g = grade_sheet_llm(router, [atp])[10]
    assert g.marks == 0.0 and g.key_points[0]["evidence_check"] == "too_short"
    assert g.key_points[0]["reason"].startswith("[evidence too short]")
    # 3 words, or 12+ letters/digits, is enough; so is half of a very short answer.
    server.script("groq", sheet(q(10, kp(0, "matched", "Mitochondria make ATP"))))
    assert grade_sheet_llm(router, [atp], style="strict")[10].marks == 1.0
    tiny = AnswerItem(question_no=11, type="short", max_marks=1, answer_text="Mitosis.",
                      key_points=[{"text": "mitosis", "weight": 1}])
    server.script("groq", sheet(q(11, kp(0, "matched", "Mitosis"))))
    assert grade_sheet_llm(router, [tiny])[11].marks == 1.0


def test_one_short_span_backing_two_key_points_is_weak_evidence(harness) -> None:  # noqa: ANN001
    item = AnswerItem(question_no=12, type="short", max_marks=2, answer_text="Mitochondria produce ATP. They sit in cells.",
                      key_points=[{"text": "mitochondria are the site of respiration", "weight": 1},
                                  {"text": "they produce ATP", "weight": 1}])
    router, server = harness()
    server.script("groq", sheet(q(12, kp(0, "matched", "Mitochondria produce ATP"),
                                  kp(1, "matched", "Mitochondria produce ATP"), feedback="Well done, full marks.")))
    g = grade_sheet_llm(router, [item])[12]
    assert g.marks == 2.0  # flagged for a teacher, not withdrawn
    assert g.flags == ["ai_weak_evidence"] and g.feedback_source == "template"
    assert [k["evidence_shared_with"] for k in g.key_points] == [[1], [0]]

    # One longer sentence can carry two criteria: recorded, not flagged.
    long_item = AnswerItem(question_no=13, type="short", max_marks=2,
                           answer_text="Mitochondria produce ATP by aerobic respiration in cells.",
                           key_points=item.key_points)
    quote = "Mitochondria produce ATP by aerobic respiration"
    server.script("groq", sheet(q(13, kp(0, "matched", quote), kp(1, "matched", quote))))
    g = grade_sheet_llm(router, [long_item])[13]
    assert g.marks == 2.0 and g.flags == [] and g.feedback_source == "ai"
    assert [k["evidence_shared_with"] for k in g.key_points] == [[1], [0]]


def test_evidence_overlap_with_the_key_point_is_recorded_not_gated(harness) -> None:  # noqa: ANN001
    item = AnswerItem(question_no=14, type="short", max_marks=1,
                      answer_text="Every training example comes with the correct answer.",
                      key_points=[{"text": "the model is trained on labelled data", "weight": 1}])
    router, server = harness()
    server.script("groq", sheet(q(14, kp(0, "matched", "Every training example comes with the correct answer"))))
    g = grade_sheet_llm(router, [item])[14]
    assert g.marks == 1.0 and g.flags == []  # a paraphrase: the AI's judgement stands
    assert 0.0 < g.key_points[0]["evidence_overlap"] < 0.5


def test_mark_claims_are_removed_from_ai_feedback(harness) -> None:  # noqa: ANN001
    router, server = harness()
    server.script("groq", sheet(q(2, kp(0, "matched", "the model memorises the training data"),
                                  kp(1, "partial", "does badly on unseen data"),
                                  feedback="Good definition. You scored 2/2 here. Full marks for the cause.")))
    g = grade_sheet_llm(router, [Q2])[2]
    assert g.marks == 1.5 and g.feedback == "Good definition." and g.feedback_source == "ai"


def test_marks_add_the_unrounded_weights(harness) -> None:  # noqa: ANN001
    item = AnswerItem(question_no=15, type="short", max_marks=1,
                      answer_text="Alpha beta gamma. Delta epsilon zeta. Eta theta iota.",
                      key_points=[{"text": t, "weight": 1 / 3} for t in ("alpha", "delta", "eta")])
    router, server = harness()
    server.script("groq", sheet(q(15, kp(0, "matched", "Alpha beta gamma"), kp(1, "matched", "Delta epsilon zeta"),
                                  kp(2, "matched", "Eta theta iota"))))
    g = grade_sheet_llm(router, [item])[15]
    assert (g.marks, g.verdict) == (1.0, "correct")  # not 0.99 from three rounded 0.33s


def test_new_ai_evidence_flags_always_go_to_review() -> None:
    from app.pipeline.routing import REVIEW_FLAGS, route

    for flag in ("ai_fuzzy_evidence", "ai_weak_evidence"):
        assert flag in REVIEW_FLAGS
        assert route(0.99, 1.0, [flag], auto_accept=0.8, low_ocr=0.5)[0] == "needs_review"


# ------------------------------------------------------------------ property test
VOCAB = ("model data training learns noise inputs outputs predicts error gradient loss rate step labelled "
         "unseen weights bias layer overfitting").split()


def test_marks_never_exceed_max_for_random_responses(harness) -> None:  # noqa: ANN001
    rng = random.Random(20261123)

    def chaos(call: FakeCall) -> dict:
        questions = []
        for qd in payload_of(call)["questions"]:
            words = qd["answer_text"].split()
            n = len(qd["key_points"])
            kps = []

            def one(index: int) -> dict:
                verdict = rng.choice(["matched", "partial", "missed", "contradicted", "MATCHED", " Partial "])
                i = rng.randrange(len(words))
                quote = rng.choice([
                    " ".join(words[i: i + rng.randint(1, 6)]),  # real span
                    " ".join(rng.choices(VOCAB, k=4)),  # invented
                    "", " ".join(words[i: i + 4]).upper(),
                ])
                return {**kp(index, verdict, quote, "r " * rng.randint(0, 40)), "marks": rng.uniform(-100, 100)}

            if rng.random() < 0.6:  # every key point once, plus junk indices
                kps = [one(i) for i in range(n)] + [one(rng.choice([-1, n, n + 5])) for _ in range(rng.randint(0, 2))]
                rng.shuffle(kps)
            else:
                for _ in range(rng.randint(0, n + 3)):
                    kps.append(one(rng.choice([rng.randrange(n), rng.randrange(n), -1, n, n + 5])))
            questions.append({**q(qd["question_no"], *kps, uncertain=rng.random() < 0.2),
                              "marks": rng.uniform(0, 1000)})
        if rng.random() < 0.2:
            questions.append(q(99, kp(0, "matched", "x")))
        rng.shuffle(questions)
        return {"questions": questions, "total": rng.uniform(0, 1e6)}

    # 150 sheets back to back would (rightly) exhaust the free-tier token budget; this test is about marks.
    router, server = harness(server=FakeLLMServer(responder=chaos), ai_grading_providers=[GROQ],
                             ai_max_retries=0, groq_tokens_per_minute=10**9, groq_tokens_per_day=10**9)
    checked = skipped = 0
    for _ in range(150):
        items = []
        for no in rng.sample(range(1, 20), rng.randint(1, 3)):
            n_kps = rng.randint(1, 5)
            items.append(AnswerItem(
                question_no=no, type=rng.choice(["short", "long", "paragraph"]),
                max_marks=rng.choice([0.5, 1, 2, 2.5, 3, 5, 10]),
                answer_text=". ".join(" ".join(rng.choices(VOCAB, k=rng.randint(3, 9))) for _ in range(3)) + ".",
                key_points=[{"text": " ".join(rng.choices(VOCAB, k=3)), "weight": rng.choice([0.5, 1, 1.5, 2, 3])}
                            for _ in range(n_kps)],
            ))
        grades = grade_sheet_llm(router, items, style=rng.choice(sorted(STYLES)))
        for item in items:
            if item.question_no not in grades:  # left out / partly judged: absent, never scored
                assert grades.missing[item.question_no] == "ai_incomplete"
                skipped += 1
                continue
            g = grades[item.question_no]
            assert item.question_no not in grades.missing
            assert 0.0 <= g.marks <= item.max_marks
            assert g.marks == round(g.marks, 2)
            assert g.verdict == verdict_for(g.marks, item.max_marks)
            assert len(g.key_points) == len(item.key_points)
            assert g.marks <= round(sum(k["awarded"] for k in g.key_points), 2)
            for k in g.key_points:
                assert k["status"] in {"matched", "partial", "missed", "contradicted"}
                assert (k["evidence_sentence"] is not None) == (k["status"] != "missed")
                if k["evidence_sentence"]:
                    assert k["evidence_sentence"].lower() in item.answer_text.lower()
                assert len(k["reason"].split()) <= 31
                if k["evidence_check"] in ("not_found", "too_short"):
                    assert k["status"] == "missed" and "ai_unverified_evidence" in g.flags
            if g.feedback_source == "ai":
                assert len(g.feedback.split()) <= 40
            else:  # code changed a verdict / weak evidence: the summary of the final statuses
                assert g.feedback == template_feedback(g.key_points) and len(g.feedback) <= 400
            checked += 1
    assert checked >= 100 and skipped >= 20 and len(server.calls) == 150
