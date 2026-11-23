"""The optimised AI evaluation engine (providers faked over httpx.MockTransport):

* handwriting reading: pages read concurrently, an overloaded model parked after one
  reply instead of retried on every page, a second pass for unread pages, per-line
  legibility confidence, and unread pages never silently marked or auto-accepted;
* question-only rubrics: one cached marking scheme per question, criterion verdicts
  with verified evidence, marks computed in code;
* selective verification: only doubtful AI marks are re-sent, once;
* router budgeting with the expected (not maximum) reply size;
* live progress driven by pipeline events.
"""

from __future__ import annotations

import base64
import json
import threading
import time

import cv2
import numpy as np
import pymupdf
import pytest

from app.grading import direct_ai, llm_grader
from app.grading.llm_grader import USER_PREFIX
from app.llm import LLMRouter, RateLimiter
from app.llm.fakes import FakeCall, FakeLLMServer, FakeReply
from app.pipeline import ai_ocr, processor
from app.pipeline.progress import STAGE_KEYS, Progress
from conftest import drain
from scripts import make_samples as samples

GROQ = "groq:openai/gpt-oss-120b"
GROQ_VISION = "groq:qwen/qwen3.8-27b"
GEMINI_VISION = "gemini:gemini-flash-lite-latest"
KEYS = dict(ai_consent=True, groq_api_key="fake-groq-key-for-tests-0001")
READ_ON = dict(**KEYS, ai_ocr=True, ai_ocr_providers=[GROQ_VISION], ai_grading="off", ai_max_retries=0)
MARK_ON = dict(**KEYS, ai_grading="ensemble", ai_grading_providers=[GROQ], ai_max_retries=0,
               ai_direct_scheme=True)  # the opt-in scheme marking
A = samples.ANSWERS
PAGE_W0 = 595  # page k of a test scan is PAGE_W0 + 10*k points wide: the fake reader tells pages apart


# ------------------------------------------------------------------ helpers
def scan_pdf(n_pages: int) -> bytes:
    """An image-only PDF (no text layer), page k a little wider than page k-1."""
    out = pymupdf.open()
    try:
        for k in range(1, n_pages + 1):
            src = pymupdf.open()
            page = src.new_page(width=PAGE_W0 + 10 * k, height=842)
            for i in range(12):
                page.insert_text((60, 90 + 40 * i), f"handwritten line {i} of page {k}", fontsize=14)
            pix = page.get_pixmap(dpi=100, colorspace=pymupdf.csGRAY, alpha=False)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=pix.tobytes("png"))
            src.close()
        return out.tobytes(garbage=3, deflate=True)
    finally:
        out.close()


def page_of(call: FakeCall) -> int:
    """Which test page a read_page request shows (from the image's aspect ratio)."""
    if call.provider == "gemini":
        data = call.body["contents"][0]["parts"][1]["inlineData"]["data"]
    else:
        data = call.body["messages"][-1]["content"][1]["image_url"]["url"].split(",", 1)[1]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    return round((w / h * 842 - PAGE_W0) / 10)


def is_read(call: FakeCall) -> bool:
    return call.prompt.startswith("This is one page")


PAGE_TEXT = {1: f"Q1. {A['q1_full']}", 2: f"Q2. {A['q2_full']}", 3: f"Q3. {A['q3_full']}", 4: f"Q4. {A['q4_full']}"}


def reader(delay: float = 0.0, fail_pages: tuple = (), track: dict | None = None,  # noqa: ANN201
           barrier: threading.Barrier | None = None):
    """Fake vision model. `barrier`: the first reads wait for each other (proof that they
    are in flight at the same time; a sequential reader would time out and fail)."""
    lock = threading.Lock()

    def respond(call: FakeCall):  # noqa: ANN202
        if not is_read(call):
            return None
        k = page_of(call)
        if barrier is not None and not barrier.broken:
            try:
                barrier.wait(timeout=20)
            except threading.BrokenBarrierError:
                if track is not None:
                    track["barrier_broken"] = True
        if track is not None:
            with lock:
                track["now"] = track.get("now", 0) + 1
                track["max"] = max(track.get("max", 0), track["now"])
        try:
            if delay:
                time.sleep(delay)
            if k in fail_pages:
                return FakeReply(status=500, body={"error": {"message": "internal"}})
            return {"lines": [{"text": PAGE_TEXT.get(k, f"page {k}"), "box_2d": [80, 60, 140, 940],
                               "unclear": False}]}
        finally:
            if track is not None:
                with lock:
                    track["now"] -= 1

    return respond


@pytest.fixture
def fake_ai(monkeypatch) -> FakeLLMServer:  # noqa: ANN001
    """Routers talk to this fake server; rate-limit waits run on a fake clock."""
    server = FakeLLMServer()
    limiter_clock = [0.0]

    def make(settings, session_factory):  # noqa: ANN001, ANN202
        def sleep(seconds: float) -> None:
            limiter_clock[0] += seconds

        return LLMRouter(settings, session_factory, transport=server.transport,
                         limiter=RateLimiter(clock=lambda: limiter_clock[0]), sleep=sleep)

    monkeypatch.setattr(processor, "make_ai_router", make)
    return server


def upload(teacher, exam: dict, files: dict[str, bytes]) -> dict:  # noqa: ANN001
    r = teacher.upload(exam["id"], list(files.items()), refs=[n.rsplit(".", 1)[0] for n in files])
    assert r.status_code == 202, r.text
    drain()
    return {s["student_ref"]: teacher.submission(s["id"]) for s in r.json()["submissions"]}


# ------------------------------------------------------------------ handwriting reading
@pytest.mark.settings(**READ_ON, ai_ocr_concurrency=3)
def test_pages_are_read_concurrently_and_every_page_is_read(teacher, exam, fake_ai) -> None:  # noqa: ANN001
    track: dict = {}
    fake_ai.responder = reader(delay=0.05, track=track, barrier=threading.Barrier(2))
    sub = upload(teacher, exam, {"scan.pdf": scan_pdf(4)})["scan"]
    reads = [c for c in fake_ai.calls if is_read(c)]
    assert sorted(page_of(c) for c in reads) == [1, 2, 3, 4]  # one request per page, no retries
    assert not track.get("barrier_broken")  # two pages were in flight at the same time
    assert sub["ocr_engine"] == "ai:groq" and sub["status"] == "processed"
    answers = teacher.answers(sub["id"])
    assert [a["question_no"] for a in answers] == [1, 2, 3, 4]
    assert all(a["ocr_text"] for a in answers)  # each page's answer reached its question
    c = sub["progress"]["counts"]
    assert c["pages_total"] == 4 and c["pages_read"] == 4 and c["pages_unread"] == 0
    assert sub["progress"]["finished"] and [s["key"] for s in sub["progress"]["stages"]] == list(STAGE_KEYS)


@pytest.mark.settings(**{**READ_ON, "gemini_consent": True, "gemini_api_key": "fake-gemini-key-for-tests-0001",
                         "ai_ocr_providers": [GEMINI_VISION, GROQ_VISION]}, ai_ocr_concurrency=1)
def test_an_overloaded_reader_is_parked_after_one_reply(teacher, exam, fake_ai) -> None:  # noqa: ANN001
    """2026-11-23: Gemini answered 503 "high demand" for an hour; every page retried every
    Gemini model three times with backoff (194 s for 5 pages). Now the first 503 parks
    the model and the other pages go straight to the next reader."""
    busy = FakeReply(status=503, body={"error": {"message": "This model is currently experiencing high demand."}})

    def respond(call: FakeCall):  # noqa: ANN202
        if call.provider == "gemini":
            return busy
        return reader()(call)

    fake_ai.responder = respond
    sub = upload(teacher, exam, {"scan.pdf": scan_pdf(4)})["scan"]
    assert len(fake_ai.calls_to("gemini")) == 1
    assert sorted(page_of(c) for c in fake_ai.calls_to("groq") if is_read(c)) == [1, 2, 3, 4]
    assert sub["progress"]["counts"]["pages_read"] == 4


@pytest.mark.settings(**READ_ON, review_required=False)
def test_an_unread_page_is_flagged_and_never_auto_accepted(teacher, exam, fake_ai) -> None:  # noqa: ANN001
    fake_ai.responder = reader(fail_pages=(2,))
    sub = upload(teacher, exam, {"scan.pdf": scan_pdf(4)})["scan"]
    page2 = [c for c in fake_ai.calls if is_read(c) and page_of(c) == 2]
    assert len(page2) == 2  # the first pass, then the second pass at the end
    assert "page_unread" in sub["page_flags"]["2"] and "has_unread_pages" in sub["flags"]
    assert sub["progress"]["counts"]["pages_unread"] == 1
    rec = {s["key"]: s for s in sub["progress"]["stages"]}["recognition"]
    assert rec["state"] == "warning" and "could not be read" in rec["detail"]
    by_q = {a["question_no"]: a for a in teacher.answers(sub["id"])}
    # Q2 was on the unread page: no silent 0, the teacher must look (even with review off).
    assert by_q[2]["status"] == "needs_review"
    # Every answer whose text may run onto the unread page waits for the teacher too.
    for a in by_q.values():
        if {"needs_transcription", "page_unread"} & set(a["flags"]):
            assert a["status"] == "needs_review"


def test_line_confidence_follows_the_models_doubt_markers() -> None:
    L = ai_ocr.PageLine
    assert ai_ocr.line_confidence(L(text="Photosynthesis needs light", box_2d=[])) == ai_ocr.CONFIDENCE
    assert ai_ocr.line_confidence(L(text="Photosynthesis needs lihgt", box_2d=[], unclear=True)) == \
        ai_ocr.UNCLEAR_CONFIDENCE
    assert ai_ocr.line_confidence(L(text="needs [illegible] energy", box_2d=[])) == ai_ocr.ILLEGIBLE_CONFIDENCE
    assert "Mathematics" in ai_ocr.PROMPT and "[diagram:" in ai_ocr.PROMPT and "[illegible]" in ai_ocr.PROMPT


# ------------------------------------------------------------------ router budgeting
def test_page_reads_reserve_the_expected_not_the_maximum_reply(tmp_path) -> None:  # noqa: ANN001
    """Groq's 6K tokens/minute bucket: reserving the 4096-token ceiling let one page a
    minute through; reserving the typical ~1.1K lets the pages of a sheet through now."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.config import Settings
    from app.db import Base
    from app.models import AiRun

    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng, tables=[AiRun.__table__])
    server = FakeLLMServer(responder=lambda call: {"lines": []})
    sleeps: list[float] = []
    clock = [0.0]
    router = LLMRouter(Settings(env="test", database_url="sqlite://", storage_dir=tmp_path, ai_consent=True,
                                groq_api_key="fake-groq-key-for-tests-0001", ai_ocr_providers=[GROQ_VISION],
                                groq_tokens_per_minute=6000),
                       sessionmaker(bind=eng), transport=server.transport,
                       limiter=RateLimiter(clock=lambda: clock[0]), sleep=sleeps.append)
    img = ai_ocr.ImageInput(b"\xff\xd8fake", "image/jpeg")
    for n in range(3):
        router.generate_json(ai_ocr.TASK, ai_ocr.PROMPT + str(n), ai_ocr.PageText, images=[img],
                             max_output_tokens=4096, expected_output_tokens=ai_ocr.EXPECTED_OUTPUT_TOKENS,
                             providers=[GROQ_VISION])
    router.close()
    assert len(server.calls) == 3 and sleeps == []


# ------------------------------------------------------------------ question-only rubrics (scheme marking)
def question_only_rubric() -> dict:
    return {"questions": [{"question_no": q["question_no"], "type": q["type"], "max_marks": q["max_marks"],
                           "prompt": q["prompt"], "model_answer": None, "key_points": []}
                          for q in samples.RUBRIC["questions"]]}


def scheme_marker(verdicts: tuple[str, ...] = ("matched", "partial"), uncertain: tuple = (),
                  second: tuple[str, ...] | None = None):  # noqa: ANN201
    """Fake examiner: writes a 2-criterion scheme (marks 3 + 1, normalised by code), then
    judges criterion i as verdicts[i] quoting the answer's first words. The second
    marking (if asked) uses `second`."""

    def respond(call: FakeCall) -> dict | None:
        if call.prompt.startswith("Write the marking scheme"):
            return {"model_answer": "Reference.", "accept_also": "",
                    "criteria": [{"text": "States the main idea", "marks": 3},
                                 {"text": "Gives a supporting detail", "marks": 1}]}
        if call.prompt.startswith(USER_PREFIX):
            again = "SECOND MARKING" in (call.system or "")
            use = second if again and second is not None else verdicts
            qs = []
            for q in json.loads(call.prompt[len(USER_PREFIX):])["questions"]:
                quote = " ".join(q["answer_text"].split()[:6])
                qs.append({"question_no": q["question_no"], "uncertain": (q["question_no"] in uncertain) and not again,
                           "feedback": "Good.", "key_points": [
                               {"index": kp["index"], "verdict": use[kp["index"]],
                                "evidence_quote": "" if use[kp["index"]] == "missed" else quote, "reason": "ok"}
                               for kp in q["key_points"]]})
            return {"questions": qs}
        return None

    return respond


@pytest.fixture
def qo_exam(teacher) -> dict:  # noqa: ANN001
    exam = teacher.create_exam()
    teacher.put_rubric(exam["id"], question_only_rubric())
    return exam


@pytest.mark.settings(**MARK_ON, ai_verify=False)
def test_question_only_answers_are_marked_criterion_by_criterion(teacher, qo_exam, sample_bytes,  # noqa: ANN001
                                                                 fake_ai) -> None:
    fake_ai.responder = scheme_marker(("matched", "partial"))
    subs = upload(teacher, qo_exam, {"a.pdf": sample_bytes["ML001_full_marks"], "b.pdf": sample_bytes["ML003_paraphrased"]})
    scheme_calls = [c for c in fake_ai.calls if c.prompt.startswith("Write the marking scheme")]
    assert len(scheme_calls) == 4  # once per question for the whole exam, not per sheet
    sheet_calls = [c for c in fake_ai.calls if c.prompt.startswith(USER_PREFIX)]
    assert len(sheet_calls) == 2  # one marking request per sheet
    for sub in subs.values():
        for a in teacher.answers(sub["id"]):
            mx = a["max_marks"]
            w = direct_ai.split_marks([3, 1], mx)
            assert a["grader"] == direct_ai.GRADER_NAME_V2
            assert a["ai_marks"] == pytest.approx(round(w[0] + w[1] / 2, 2))  # code computes the marks
            assert [k["status"] for k in a["key_points"]] == ["matched", "partial"]
            assert all(k["evidence_sentence"] and k["evidence_sentence"] in a["ocr_text"] for k in a["key_points"])
            ai = a["grading_trace"]["ai"]
            assert ai["mode"] == "scheme" and len(ai["scheme"]["criteria"]) == 2
            assert sum(c["marks"] for c in ai["scheme"]["criteria"]) == pytest.approx(mx)


@pytest.mark.settings(**MARK_ON, ai_verify=True)  # opt-in feature
def test_only_doubtful_answers_get_a_second_marking(teacher, qo_exam, sample_bytes, fake_ai) -> None:  # noqa: ANN001
    fake_ai.responder = scheme_marker(("matched", "matched"), uncertain=(2,), second=("matched", "missed"))
    sub = upload(teacher, qo_exam, {"a.pdf": sample_bytes["ML001_full_marks"]})["a"]
    second = [c for c in fake_ai.calls if "SECOND MARKING" in (c.system or "")]
    assert len(second) == 1  # one extra request for the sheet ...
    assert [q["question_no"] for q in json.loads(second[0].prompt[len(USER_PREFIX):])["questions"]] == [2]  # ... for Q2 only
    by_q = {a["question_no"]: a for a in teacher.answers(sub["id"])}
    q2 = by_q[2]
    w = direct_ai.split_marks([3, 1], q2["max_marks"])
    assert q2["ai_marks"] == pytest.approx(w[0])  # the second marking stands
    ver = q2["grading_trace"]["ai"]["verification"]
    assert ver["status"] == "changed" and ver["first_marks"] == pytest.approx(q2["max_marks"])
    assert "ai_second_opinion_differs" in q2["flags"] and q2["status"] == "needs_review"
    assert all("verification" not in a["grading_trace"]["ai"] for n, a in by_q.items() if n != 2)
    stages = {s["key"]: s for s in sub["progress"]["stages"]}
    assert sub["progress"]["counts"]["verification_candidates"] == 1
    assert "1 doubtful answer(s) re-marked" in stages["verification"]["detail"]


def test_verification_agreement_confirms_and_clears_the_doubt(tmp_path) -> None:  # noqa: ANN001
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.config import Settings
    from app.db import Base
    from app.models import AiRun

    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng, tables=[AiRun.__table__])
    server = FakeLLMServer(responder=scheme_marker(("matched", "partial"), uncertain=(1,)))
    router = LLMRouter(Settings(env="test", database_url="sqlite://", storage_dir=tmp_path, ai_consent=True,
                                groq_api_key="fake-groq-key-for-tests-0001", ai_grading_providers=[GROQ]),
                       sessionmaker(bind=eng), transport=server.transport, limiter=RateLimiter())
    items = [llm_grader.AnswerItem(question_no=n, type="short", max_marks=4, answer_text=f"answer number {n} says x y z",
                                   key_points=[{"text": "a", "weight": 3}, {"text": "b", "weight": 1}])
             for n in (1, 2)]
    grades = llm_grader.grade_sheet_llm(router, items)
    assert grades[1].uncertain and not grades[2].uncertain
    grades, records = llm_grader.verify_sheet(router, items, grades)
    router.close()
    assert list(records) == [1] and records[1]["status"] == "confirmed"
    assert "ai_verified" in grades[1].flags and "ai_uncertain" not in grades[1].flags
    assert grades[1].marks == pytest.approx(3.5)
    second = [c for c in server.calls if "SECOND MARKING" in (c.system or "")]
    assert len(second) == 1 and second[0].body.get("reasoning_effort") == "medium"


def test_split_marks_always_adds_up_to_the_maximum() -> None:
    assert direct_ai.split_marks([2, 1, 1], 5) == [2.5, 1.25, 1.25]
    assert sum(direct_ai.split_marks([7, 7, 7], 10)) == pytest.approx(10)
    assert direct_ai.split_marks([0, 0], 2) == [1.0, 1.0]
    assert sum(direct_ai.split_marks([1, 5, 2.5, 0.5], 3)) == pytest.approx(3)
    assert all(m > 0 for m in direct_ai.split_marks([100, 1, 1], 3))


# ------------------------------------------------------------------ progress
def test_progress_records_real_stage_durations_and_is_a_copy() -> None:
    t = [0.0]
    p = Progress(pages_total=3, questions_total=2, clock=lambda: t[0])
    p.start("preprocess")
    t[0] = 1.5
    p.finish("preprocess", detail="3 page(s) prepared")
    p.bump("pages_read", 2)
    snap = p.snapshot()
    st = {s["key"]: s for s in snap["stages"]}
    assert st["preprocess"] == {**st["preprocess"], "state": "done", "seconds": 1.5, "detail": "3 page(s) prepared"}
    assert st["recognition"]["state"] == "pending" and snap["counts"]["pages_read"] == 2
    assert snap["elapsed_seconds"] == 1.5
    snap["counts"]["pages_read"] = 99
    assert p.get("pages_read") == 2
