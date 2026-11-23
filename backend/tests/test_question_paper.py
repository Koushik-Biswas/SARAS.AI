"""Step 1 "upload question sheet": POST /exams/{id}/questions/extract reads a question
paper into a draft rubric (nothing saved). AI first (faked here), the rule-based text
reader when the AI is off or unavailable, a clear error when nothing can be read."""

from __future__ import annotations

import cv2
import numpy as np
import pymupdf as fitz
import pytest

from app.llm import LLMRouter, RateLimiter
from app.llm.fakes import FakeLLMServer
from app.pipeline import processor
from app.pipeline.question_paper import parse_text
from conftest import assert_error

GROQ = "groq:openai/gpt-oss-120b"
AI_ON = dict(ai_consent=True, groq_api_key="fake-groq-key-for-tests-0001", ai_grading_providers=[GROQ],
             ai_max_retries=0)

PAPER = """Delhi Public School - Unit Test
Time: 1 hour                       Max marks: 20
Answer all questions.

1. What is the national animal of India? [2]
2) Explain photosynthesis with a labelled
   diagram. (5 marks)
Q3. Solve 2x + 3 = 11.  [3 marks]
Question 4: Write an essay on climate change. 10 marks
"""


def pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 60), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def png_bytes() -> bytes:
    img = np.full((200, 300), 255, dtype=np.uint8)
    cv2.putText(img, "1. Define gravity [2]", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def extract(api, exam_id: str, name: str, body: bytes, mime: str):  # noqa: ANN001, ANN201
    return api.post(f"/exams/{exam_id}/questions/extract", files={"file": (name, body, mime)})


@pytest.fixture
def fake_ai(monkeypatch) -> FakeLLMServer:  # noqa: ANN001
    server = FakeLLMServer()

    def make(settings, session_factory):  # noqa: ANN001, ANN202
        return LLMRouter(settings, session_factory, transport=server.transport,
                         limiter=RateLimiter(clock=lambda: 0.0), sleep=lambda _s: None)

    monkeypatch.setattr(processor, "make_ai_router", make)
    return server


def test_rule_based_reader_handles_common_numbering_and_marks():
    qs = parse_text(PAPER)
    assert [q.question_no for q in qs] == [1, 2, 3, 4]
    assert [q.max_marks for q in qs] == [2, 5, 3, 10]
    assert qs[0].prompt == "What is the national animal of India?"
    assert qs[1].prompt == "Explain photosynthesis with a labelled diagram."
    assert qs[2].prompt == "Solve 2x + 3 = 11."


def test_ai_off_reads_a_typed_pdf_with_the_text_reader(teacher):
    exam = teacher.create_exam()
    r = extract(teacher, exam["id"], "paper.pdf", pdf_bytes(PAPER), "application/pdf")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "text" and body["pages"] == 1
    assert [(q["question_no"], q["max_marks"]) for q in body["questions"]] == [(1, 2), (2, 5), (3, 3), (4, 10)]
    assert [q["type"] for q in body["questions"]] == ["short", "long", "maths", "paragraph"]
    assert any("check them" in w for w in body["warnings"])
    assert teacher.get(f"/exams/{exam['id']}/questions").json() == []  # a draft: nothing saved


@pytest.mark.settings(**AI_ON)
def test_ai_reads_the_paper_and_its_draft_is_cleaned(teacher, fake_ai):
    fake_ai.script("groq", {"questions": [
        {"question_no": 1, "type": "short", "max_marks": 2, "prompt": "Q1. What is the national animal of India?"},
        {"question_no": 1, "type": "weird", "max_marks": 0, "prompt": "Explain photosynthesis."},  # dup number
        {"question_no": 3, "type": "maths", "max_marks": 3, "prompt": "   "},  # empty: dropped
    ]})
    exam = teacher.create_exam()
    r = extract(teacher, exam["id"], "paper.pdf", pdf_bytes(PAPER), "application/pdf")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "ai:groq"
    assert body["questions"] == [
        {"question_no": 1, "type": "short", "max_marks": 2, "prompt": "What is the national animal of India?"},
        {"question_no": 2, "type": "long", "max_marks": None, "prompt": "Explain photosynthesis."},
    ]
    assert any("Q2" in w for w in body["warnings"])
    assert "QUESTION PAPER TEXT" in fake_ai.calls[0].prompt and "national animal" in fake_ai.calls[0].prompt


@pytest.mark.settings(**AI_ON)
def test_ai_unavailable_falls_back_to_the_text_reader(teacher, fake_ai):
    exam = teacher.create_exam()  # no scripted reply: the fake provider answers 500
    r = extract(teacher, exam["id"], "paper.pdf", pdf_bytes(PAPER), "application/pdf")
    assert r.status_code == 200, r.text
    assert r.json()["source"] == "text" and len(r.json()["questions"]) == 4
    assert fake_ai.calls


def test_a_photo_needs_the_ai_reader(teacher):
    exam = teacher.create_exam()
    err = assert_error(extract(teacher, exam["id"], "paper.png", png_bytes(), "image/png"), 422, "no_questions_found")
    assert "AI" in err["message"]


def test_unsupported_and_foreign_files_are_rejected(teacher, other_teacher):
    exam = teacher.create_exam()
    assert_error(extract(teacher, exam["id"], "paper.txt", b"1. What? [2]", "text/plain"), 415)
    assert_error(extract(other_teacher, exam["id"], "paper.pdf", pdf_bytes(PAPER), "application/pdf"), 403)
