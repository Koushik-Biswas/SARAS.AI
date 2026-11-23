"""Answer segmentation (PRD 12.2, 12.10): marker styles, skipped questions get 0
and are never graded, duplicate / out-of-order markers flagged, answers continue
across pages, sheets without markers go to review."""

from __future__ import annotations

import pytest

from app.pipeline.ocr import OcrLine
from app.pipeline.segment import parse_marker, segment
from scripts import make_samples as samples

Q = [1, 2, 3, 4]


def lines(*items: tuple[int, str]) -> list[OcrLine]:
    """(page_no, text) pairs -> OCR lines stacked down each page."""
    out, y = [], {}
    for page, text in items:
        y[page] = y.get(page, 100) + 40
        out.append(OcrLine(page, (60, y[page], 900, 30), text, 0.99))
    return out


@pytest.mark.parametrize("text,number,rest", [
    ("Q1 Supervised learning uses labels", 1, "Supervised learning uses labels"),
    ("Q.2 Overfitting", 2, "Overfitting"),
    ("1. Supervised", 1, "Supervised"),
    ("3) Gradient", 3, "Gradient"),
    ("2(a) The model memorises", 2, "The model memorises"),
    ("2 (b). Noise", 2, "Noise"),
    ("Question 3: Gradient descent", 3, "Gradient descent"),
    ("Ans 4 tune hyperparameters", 4, "tune hyperparameters"),
    ("Q4", 4, ""),
])
def test_marker_styles(text, number, rest):
    m = parse_marker(text, set(Q))
    assert m is not None and (m.number, m.rest) == (number, rest)


@pytest.mark.parametrize("text", ["2 apples are red", "Quality 3 matters", "9. not in the rubric",
                                  "The model has 3 layers", "12) twelve"])
def test_non_markers(text):
    assert parse_marker(text, set(Q)) is None


def test_segments_follow_markers():
    res = segment(lines((1, "Q1 labelled data"), (1, "maps inputs"), (1, "2(a) overfitting"),
                        (1, "Question 3: gradient"), (1, "4. validation")), Q)
    assert {n: s.text for n, s in res.segments.items()} == {
        1: "labelled data maps inputs", 2: "overfitting", 3: "gradient", 4: "validation"}
    assert res.missing == [] and res.sheet_flags == [] and res.unassigned == []


def test_skipped_question_marker_with_no_text():
    res = segment(lines((1, "Q1 answer one"), (1, "Q2."), (1, "Q3 answer three"), (1, "Q4 four")), Q)
    assert res.segments[2].text == ""
    assert res.missing == []


def test_duplicate_and_out_of_order_markers_flagged():
    res = segment(lines((1, "Q1 a"), (1, "Q3 c"), (1, "Q2 b"), (1, "Q2 again b"), (1, "Q4 d")), Q)
    assert "duplicate_marker" in res.segments[2].flags and res.segments[2].confidence <= 0.55
    assert "out_of_order" in res.segments[2].flags
    assert {"duplicate_marker", "out_of_order"} <= set(res.sheet_flags)
    assert res.segments[2].text == "b again b"


def test_continuation_across_pages_is_joined():
    res = segment(lines((1, "Q1 a"), (1, "Q2 first part"), (2, "second part"), (2, "Q3 c")), Q)
    seg = res.segments[2]
    assert seg.text == "first part second part"
    assert "continues_across_pages" in seg.flags
    assert [r.page_no for r in seg.regions({1: 1240, 2: 1240})] == [1, 2]


def test_gap_in_sequence_flags_neighbours():
    res = segment(lines((1, "Q1 a"), (1, "Q2 b"), (1, "Q4 d")), Q)
    assert res.missing == [3] and "gap_in_sequence" in res.sheet_flags
    assert "neighbour_missing" in res.segments[2].flags and "neighbour_missing" in res.segments[4].flags


def test_no_markers_found():
    res = segment(lines((1, "Some text without any marker"), (1, "more text")), Q)
    assert res.segments == {} and res.missing == Q
    assert res.sheet_flags == ["no_markers_found"]
    assert res.unassigned and res.unassigned[0].flags == ["unassigned"]


# ------------------------------------------------------------------ through the pipeline
def test_skipped_question_scores_zero_and_is_never_graded(teacher, process, monkeypatch):
    from app.pipeline import processor

    graded: list[int] = []
    real = processor._grade_with_fallback

    def spy(question, text, settings):  # noqa: ANN001, ANN202
        graded.append(question.question_no)
        return real(question, text, settings)

    monkeypatch.setattr(processor, "_grade_with_fallback", spy)
    sub = process("ML005_skipped_question")["ML005_skipped_question"]
    q3 = teacher.answer_for(sub["id"], 3)
    assert q3["status"] == "blank" and q3["ai_marks"] == 0 and q3["grader"] is None and q3["key_points"] == []
    assert sorted(graded) == [1, 2, 4]


def test_duplicate_and_out_of_order_sheets(teacher, process):
    subs = process("ML006_duplicate_marker", "ML007_out_of_order")
    dup, ooo = subs["ML006_duplicate_marker"], subs["ML007_out_of_order"]
    assert "duplicate_marker" in dup["flags"]
    q2 = teacher.answer_for(dup["id"], 2)
    assert q2["status"] == "needs_review" and "duplicate_marker" in q2["flags"]
    assert "segmentation_warning" in q2["flags"]

    assert "out_of_order" in ooo["flags"]
    # Out-of-order is informational: each answer is still found under its own question.
    answers = {a["question_no"]: a for a in teacher.answers(ooo["id"])}
    assert answers[2]["ocr_text"].startswith("Overfitting") and answers[3]["ocr_text"].startswith("Gradient")
    assert answers[2]["ai_marks"] == 5 and answers[3]["ai_marks"] == 5


def test_answer_continuing_on_page_two(teacher, process):
    sub = process("ML008_continuation")["ML008_continuation"]
    assert sub["page_count"] == 2
    q2 = teacher.answer_for(sub["id"], 2)
    assert samples.ANSWERS["q2_continued_p2"][:40] in q2["ocr_text"]
    assert "continues_across_pages" in q2["flags"]
    regions = [r for r in teacher.regions(sub["id"]) if r["id"] in q2["region_ids"]]
    assert sorted(r["page_no"] for r in regions) == [1, 2]
    assert q2["ai_marks"] == 5


def test_sheet_without_markers_goes_to_review(teacher, exam):
    pdf = samples.build_text_pdf([["Supervised learning uses labelled data.", "Overfitting memorises noise."]])
    r = teacher.upload(exam["id"], [("nomarkers.pdf", pdf)])
    assert r.status_code == 202
    from conftest import drain

    drain()
    sub = teacher.submission(r.json()["submissions"][0]["id"])
    assert sub["status"] == "processed" and "no_markers_found" in sub["flags"]
    answers = teacher.answers(sub["id"])
    assert {a["status"] for a in answers} == {"needs_review"}
    assert all("missing_marker" in a["flags"] for a in answers)
    assert sub["answers_needing_review"] == 4
    regions = teacher.regions(sub["id"])
    assert regions and all(r["question_id"] is None and "unassigned" in r["flags"] for r in regions)
