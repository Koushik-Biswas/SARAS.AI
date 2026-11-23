"""Grading (PRD 12.4, 12.11, 20.2) with the dependency-free lexical grader:
the PRD worked example, the adversarial set, contradictions, paraphrases,
OCR noise, numeric mismatches and threshold settings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.grading.engine import Thresholds, grade_answer
from app.grading.scorers import LexicalScorer, get_scorer
from scripts import make_samples as samples

A = samples.ANSWERS
SCORER = LexicalScorer()


def question(no: int, **over) -> SimpleNamespace:  # noqa: ANN003
    q = next(q for q in samples.RUBRIC["questions"] if q["question_no"] == no)
    return SimpleNamespace(**{**q, **over})


def grade(no: int, text: str, th: Thresholds | None = None, **over):  # noqa: ANN003, ANN201
    return grade_answer(question(no, **over), text, SCORER, th or Thresholds())


def test_auto_grader_without_ml_extras_is_lexical():
    # v2: 'auto' prefers a meaning-aware NLI scorer (PyTorch, then ONNX Runtime) and
    # only falls back to lexical when no model is installed.
    from app.grading.scorers import SbertScorer, onnx_available

    if SbertScorer.available():
        pytest.skip("sentence-transformers installed; auto picks the PyTorch NLI scorer")
    assert get_scorer("auto", "x", "y").name == ("nli-onnx" if onnx_available() else "lexical")


def test_prd_worked_example_scores_four_of_five_with_kp4_missed():
    r = grade(2, A["q2_worked_example"])
    assert r.marks == 4.0 and r.max_marks == 5
    statuses = [kp.status for kp in r.key_points]
    assert statuses == ["matched", "matched", "matched", "missed", "matched"]
    assert r.key_points[3].text.startswith("it is caused by")
    assert all(kp.evidence_sentence for kp in r.key_points if kp.status == "matched")
    assert r.key_points[3].evidence_sentence is None


def test_model_answers_earn_full_marks():
    for q in samples.RUBRIC["questions"]:
        r = grade(q["question_no"], q["model_answer"])
        assert r.marks == q["max_marks"], (q["question_no"], r.marks, r.flags)
        assert "contradiction" not in r.flags


@pytest.mark.parametrize("kind", sorted(samples.ADVERSARIAL_Q2))
def test_adversarial_answers_score_at_most_twenty_percent(kind):
    r = grade(2, samples.ADVERSARIAL_Q2[kind])
    assert r.marks <= 0.2 * r.max_marks, (kind, r.marks, r.flags)


def test_adversarial_flags():
    assert "copied_question" in grade(2, A["q2_copied_question"]).flags
    assert "copied_question" in grade(1, A["q1_copied_question"]).flags
    assert "keyword_list" in grade(2, A["q2_keyword_list"]).flags
    assert grade(1, A["q1_copied_question"]).marks <= 0.2 * 3


def test_flipped_fact_is_contradicted():
    r = grade(2, A["q2_flipped"])
    kp3 = r.key_points[2]
    assert kp3.status == "contradicted" or "contradiction" in r.flags
    assert kp3.status == "contradicted" and kp3.awarded == 0 and kp3.contradiction >= 0.5
    assert "performs well" in (kp3.evidence_sentence or "")
    assert r.marks < r.max_marks


def test_clause_level_negation_does_not_contradict_other_key_points():
    # "poorly" belongs to the second clause; KP2 (noise) must still match.
    r = grade(2, "It learns noise or random patterns, so it performs poorly on new unseen data.")
    assert r.key_points[1].status == "matched" and r.key_points[2].status == "matched"
    assert "contradiction" not in r.flags


@pytest.mark.parametrize("no,key", [(1, "q1_paraphrased"), (2, "q2_paraphrased"), (3, "q3_paraphrased"),
                                    (4, "q4_paraphrased")])
def test_paraphrased_correct_answers_within_one_mark_of_max(no, key):
    r = grade(no, A[key])
    assert r.marks >= r.max_marks - 1, (key, r.marks, [kp.status for kp in r.key_points])


@pytest.mark.parametrize("no,key", [(1, "q1_noisy"), (2, "q2_noisy"), (3, "q3_noisy")])
def test_ocr_noise_tolerated(no, key):
    r = grade(no, A[key])
    assert r.marks >= r.max_marks - 1, (key, r.marks)


def test_noisy_regularisation_spelling_still_matches_remedy():
    r = grade(2, A["q2_noisy"])
    assert "regulanzation" in A["q2_noisy"] and "dota" in A["q2_noisy"]
    assert r.key_points[4].status == "matched"


def test_partial_answers_earn_partial_marks():
    full, part = grade(2, A["q2_full"]).marks, grade(2, A["q2_partial"]).marks
    assert 0 < part < full == 5


def test_numeric_mismatch_flagged():
    q = dict(question_no=9, type="long", max_marks=2, prompt="Typical learning rate?", model_answer=None,
             key_points=[{"text": "the learning rate is typically set to 0.01", "weight": 1, "keywords": []},
                         {"text": "smaller learning rates converge more slowly", "weight": 1, "keywords": []}])
    qq = SimpleNamespace(**q)
    wrong = grade_answer(qq, "The learning rate is typically set to 0.5. Smaller learning rates converge more "
                             "slowly.", SCORER, Thresholds())
    assert "numeric_mismatch" in wrong.flags
    assert wrong.key_points[0].status != "matched"
    right = grade_answer(qq, "The learning rate is typically set to 0.01. Smaller learning rates converge more "
                             "slowly.", SCORER, Thresholds())
    assert "numeric_mismatch" not in right.flags and right.key_points[0].status == "matched"


def test_thresholds_are_respected():
    default = grade(1, A["q1_paraphrased"], Thresholds(), type="long")
    strict = grade(1, A["q1_paraphrased"], Thresholds(tau=0.99, tau_low=0.98), type="long")
    lenient = grade(1, A["q1_paraphrased"], Thresholds(tau=0.3, tau_low=0.2), type="long")
    assert strict.marks < default.marks <= lenient.marks == 3
    # A contradiction threshold above 1 disables contradiction verdicts.
    assert grade(2, A["q2_flipped"], Thresholds(contradiction=1.01)).key_points[2].status != "contradicted"


def test_maths_and_empty_answers():
    maths = grade(3, A["q3_full"], type="maths")
    assert maths.marks == 0 and "maths_manual_review" in maths.flags
    empty = grade(2, "   ")
    assert empty.marks == 0 and all(kp.status == "missed" for kp in empty.key_points)


def test_marks_bounded_and_two_decimals():
    for key, text in A.items():
        r = grade(int(key[1]), text)
        assert 0 <= r.marks <= r.max_marks
        assert round(r.marks, 2) == r.marks
        assert 0 <= r.grading_confidence <= 1


# ------------------------------------------------------------------ settings in the pipeline
@pytest.mark.settings(auto_accept_threshold=1.0)
def test_auto_accept_threshold_from_settings(teacher, process):
    sub = process("ML001_full_marks")["ML001_full_marks"]
    answers = teacher.answers(sub["id"])
    assert {a["status"] for a in answers} == {"needs_review"}
    assert all("low_confidence" in a["flags"] for a in answers)
    caps = teacher.get("/capabilities").json()
    assert caps["settings"]["auto_accept_threshold"] == 1.0 and caps["settings"]["grader"] == "lexical"


@pytest.mark.settings(tau=0.99, tau_low=0.98)
def test_tau_from_settings_changes_pipeline_marks(teacher, process):
    sub = process("ML003_paraphrased")["ML003_paraphrased"]
    # Paraphrases rarely reach a 0.99 match, so the strict tau costs marks.
    assert sub["total_ai_marks"] < 13


def test_default_pipeline_marks_for_reference_sheets(teacher, process):
    subs = process("ML001_full_marks", "ML010_adversarial")
    assert subs["ML001_full_marks"]["total_ai_marks"] == 15
    adv = {a["question_no"]: a for a in teacher.answers(subs["ML010_adversarial"]["id"])}
    for no in (1, 2, 3):
        assert adv[no]["ai_marks"] <= 0.2 * adv[no]["max_marks"]
    # Q4 has no marker after Q3: 0 marks, never graded, but a teacher must confirm the skip.
    assert adv[4]["status"] == "needs_review" and adv[4]["ai_marks"] == 0 and adv[4]["grader"] is None
    assert {"missing_marker", "segmentation_warning"} <= set(adv[4]["flags"])
    assert adv[1]["status"] == "needs_review" and "copied_question" in adv[1]["flags"]
