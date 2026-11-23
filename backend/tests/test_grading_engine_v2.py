"""Grading engine v2: acceptance criteria of the grading-engine fix, one block per
criterion, on the dependency-free lexical scorer (plus a few nli-onnx checks that
skip when the model is not downloaded).

(1) equivalent answers, (2) justified partial marks, (3) no unjustified marks,
(4) contradictions, (5) missing key concepts, (6) max marks never exceeded,
(7) every awarded mark has a rubric criterion + verified evidence quote,
(8) uncertain answers flagged for review, (9) measurable (trace), (10) existing
workflows (contracts) keep working."""

from __future__ import annotations

import json
import logging
import random
from types import SimpleNamespace

import pytest

from app.grading import text as T
from app.grading.engine import (
    DEFAULT_THRESHOLDS,
    SCORER_PRIORS,
    GradeResult,
    Thresholds,
    baseline_formula_marks,
    default_thresholds,
    grade_answer,
    resolve_thresholds,
    verdict_for,
)
from app.grading.scorers import LexicalScorer, get_scorer, grader_capabilities, onnx_available
from app.pipeline.routing import REVIEW_FLAGS, combine, route
from scripts import make_samples as samples

A = samples.ANSWERS
LEX = LexicalScorer()


def sample_q(no: int, **over) -> SimpleNamespace:  # noqa: ANN003
    q = next(q for q in samples.RUBRIC["questions"] if q["question_no"] == no)
    return SimpleNamespace(**{**q, **over})


def kp(text: str, weight: float = 1, keywords: tuple[str, ...] = ()) -> dict:
    return {"text": text, "weight": weight, "keywords": list(keywords)}


def question(prompt: str, kps: list[dict], qtype: str = "long", max_marks: float | None = None,
             model_answer: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(id="q-test", question_no=1, type=qtype, prompt=prompt, model_answer=model_answer,
                           key_points=kps, max_marks=max_marks if max_marks is not None else sum(k["weight"] for k in kps))


ARTERIES_VEINS = question(
    "Differentiate between arteries and veins on the basis of the direction of blood flow.",
    [kp("arteries carry blood away from the heart"), kp("veins carry blood back to the heart")], "short")
SUPPLY_DEMAND = question(
    "State the law of demand and the law of supply.",
    [kp("when price rises, quantity demanded falls"), kp("when price rises, quantity supplied rises")])
MITOSIS_MEIOSIS = question(
    "Compare mitosis and meiosis.",
    [kp("mitosis produces two identical diploid daughter cells"),
     kp("meiosis produces four genetically different haploid daughter cells")])
NEWTON = question(
    "State Newton's second law. Calculate the force needed to give a 5 kg body an acceleration of 2 m/s².",
    [kp("The rate of change of momentum is proportional to the applied force.", 1, ("momentum",)),
     kp("For constant mass this gives F = ma.", 0.5),
     kp("Correct calculation with unit: F = 5 kg × 2 m/s² = 10 N.", 1.5, ("10",))], "short")

WORKED_EXAMPLE_BADLY = ("Overfitting happens when the model fits the training data too closely. It learns the noise "
                        "and random patterns in that data instead of the general trend. As a result it performs "
                        "badly on new data. We can reduce it with regularisation, dropout or early stopping.")
HEAVY_PARAPHRASE = ("An overfitted network memorises its examples rather than the general trend, so it gives "
                    "excellent results on what it was taught yet disappointing results on fresh inputs. This "
                    "usually happens when it has far too many parameters for a tiny dataset; halting training "
                    "early helps.")


def statuses(r: GradeResult) -> list[str]:
    return [k.status for k in r.key_points]


def assert_well_formed(r: GradeResult, answer: str) -> None:
    """(6) and (7) for any result: bounded marks, marks = sum of awards, and every
    award (and contradiction) quoted verbatim from the answer."""
    assert 0 <= r.marks <= r.max_marks
    assert round(r.marks, 2) == r.marks
    assert r.marks == round(min(r.max_marks, sum(k.awarded for k in r.key_points)), 2)
    for k in r.key_points:
        assert k.reason, k
        if k.status in ("matched", "partial", "contradicted"):
            assert k.evidence_sentence and k.evidence_sentence in answer, k
        else:
            assert k.awarded == 0 and k.evidence_sentence is None
        expected = {"matched": k.weight, "partial": k.weight / 2}.get(k.status, 0.0)
        assert k.awarded == pytest.approx(expected)
    assert r.verdict == verdict_for(r.marks, r.max_marks)
    assert 0 <= r.grading_confidence <= 1


# ------------------------------------------------------------------ (1) equivalent answers
def test_prd_worked_example_four_of_five_kp4_missed_no_contradiction():
    for text in (A["q2_worked_example"], WORKED_EXAMPLE_BADLY):
        r = grade_answer(sample_q(2), text, LEX)
        assert r.marks >= 4, (r.marks, statuses(r))
        assert r.key_points[3].status == "missed" and r.key_points[3].awarded == 0
        assert "contradiction" not in r.flags and "contradicted" not in statuses(r)
        assert_well_formed(r, text)


def test_same_side_synonyms_and_negated_opposites_do_not_contradict():
    q2 = sample_q(2)
    for text in ("As a result it performs badly on new unseen data.",
                 "It does not perform well on new unseen data.",
                 "Such a model performs poorly on data it has never seen."):
        r = grade_answer(q2, text, LEX)
        assert r.key_points[2].status != "contradicted", (text, r.key_points[2])
        assert "contradiction" not in r.flags


def test_paraphrased_model_answers_still_score():
    for no, key in [(1, "q1_paraphrased"), (2, "q2_paraphrased"), (3, "q3_paraphrased"), (4, "q4_paraphrased")]:
        r = grade_answer(sample_q(no), A[key], LEX)
        assert r.marks >= r.max_marks - 1, (key, r.marks)
        assert "contradiction" not in r.flags


def test_one_marking_path_for_all_question_types():
    # v2: short answers are no longer marked by the similarity formula rounded to 0.5.
    text = A["q2_worked_example"]
    marks = {t: grade_answer(sample_q(2, type=t), text, LEX).marks for t in ("short", "long", "paragraph")}
    assert len(set(marks.values())) == 1, marks
    assert baseline_formula_marks(sample_q(2, type="short"), text, LEX) % 0.5 == 0


# ------------------------------------------------------------------ (2) partial marks
def test_partial_answer_gets_exactly_its_matched_key_points():
    r = grade_answer(sample_q(3), A["q3_partial"], LEX)
    assert statuses(r) == ["matched", "missed", "missed", "matched", "missed"]
    assert r.marks == 2.0 and r.verdict == "partially_correct"
    assert [k.awarded for k in r.key_points] == [1, 0, 0, 1, 0]


def test_missing_teacher_keyword_caps_at_partial():
    q = question("Define supervised learning.",
                 [kp("the model is trained on labelled data", 1, ("labelled",))])
    full = grade_answer(q, "The model is trained on labelled data.", LEX)
    assert full.key_points[0].status == "matched"
    q_kw = question("Define supervised learning.",
                    [kp("the model is trained on labelled data", 1, ("annotated",))])
    capped = grade_answer(q_kw, "The model is trained on labelled data.", LEX)
    assert capped.key_points[0].status == "partial" and capped.key_points[0].reason == "keyword_missing"
    assert capped.marks == 0.5


def test_keywords_are_any_of_alternatives():
    q = question("What is a stack?", [kp("A stack follows LIFO: the element inserted last is removed first.", 1,
                                          ("LIFO", "last in first out"))])
    r = grade_answer(q, "A stack follows the last in first out rule: the element inserted last is removed first.", LEX)
    assert r.key_points[0].status == "matched"


def test_calculation_judged_on_result_and_unit():
    right = "Rate of change of momentum is proportional to the applied force. So F = ma. F = 5 × 2 = 10 N."
    r = grade_answer(NEWTON, right, LEX)
    assert statuses(r) == ["matched", "matched", "matched"] and r.marks == 3
    no_unit = grade_answer(NEWTON, "F = ma. F = 5 × 2 = 10.", LEX)
    assert no_unit.key_points[2].status == "partial" and "numeric_mismatch" in no_unit.flags
    wrong = "F = ma. F = m/a = 5/2 = 2.5 N."
    w = grade_answer(NEWTON, wrong, LEX)
    assert w.key_points[2].status == "contradicted" and w.key_points[2].evidence_sentence in wrong
    assert {"numeric_mismatch", "contradiction"} <= set(w.flags)
    assert_well_formed(w, wrong)


# ------------------------------------------------------------------ (3) no unjustified marks
@pytest.mark.parametrize("kind", sorted(samples.ADVERSARIAL_Q2))
def test_adversarial_answers_at_most_twenty_percent_with_flags(kind):
    text = samples.ADVERSARIAL_Q2[kind]
    r = grade_answer(sample_q(2), text, LEX)
    assert r.marks <= 0.2 * r.max_marks
    expected_flag = {"copied_question": "copied_question", "keyword_list": "keyword_list",
                     "off_topic": "off_topic", "empty": None}[kind]
    if expected_flag:
        assert expected_flag in r.flags, (kind, r.flags)
    else:
        assert r.flags == [] and r.grading_confidence == 1.0 and r.verdict == "incorrect"
    assert_well_formed(r, text)


def test_copied_question_padded_with_restatement_is_still_copied():
    q = question("State the law of demand and the law of supply. How is the equilibrium price determined?",
                 SUPPLY_DEMAND.key_points)
    text = ("State the law of demand and the law of supply. How is the equilibrium price determined? The law of "
            "demand and the law of supply are important laws and the equilibrium price is determined.")
    r = grade_answer(q, text, LEX)
    assert r.marks == 0 and "copied_question" in r.flags


def test_underfitting_answer_for_overfitting_question():
    text = ("Underfitting is when a model is too simple to learn the patterns in the training data. It performs "
            "poorly on both the training data and new data.")
    r = grade_answer(sample_q(2), text, LEX)
    assert r.marks <= 1, (r.marks, statuses(r))
    assert "contradiction" in r.flags
    assert_well_formed(r, text)


def test_role_swap_is_not_awarded_and_is_flagged():
    for q, text in [
        (ARTERIES_VEINS, "Veins carry blood away from the heart. Arteries carry blood back to the heart."),
        (MITOSIS_MEIOSIS, "Meiosis produces two identical diploid daughter cells. Mitosis produces four "
                          "genetically different haploid daughter cells."),
    ]:
        r = grade_answer(q, text, LEX)
        assert r.marks == 0, (text, statuses(r))
        assert "role_swap" in r.flags and "contradiction" in r.flags
        assert all(k.status == "contradicted" for k in r.key_points)
        assert_well_formed(r, text)
    swapped = "When price rises, quantity supplied falls. When price rises, quantity demanded rises."
    r = grade_answer(SUPPLY_DEMAND, swapped, LEX)
    assert r.key_points[0].status == "contradicted" and r.key_points[0].awarded == 0
    assert "contradiction" in r.flags


def test_correct_answers_naming_both_entities_are_not_role_swaps():
    for q, text in [
        (ARTERIES_VEINS, "Arteries carry blood away from the heart, whereas veins carry blood back to the heart."),
        (ARTERIES_VEINS, "Arteries carry blood away from the heart and veins bring it back to the heart."),
        (SUPPLY_DEMAND, "When price rises the quantity demanded falls, while the quantity supplied rises."),
        (MITOSIS_MEIOSIS, "Mitosis produces two identical diploid daughter cells, whereas meiosis produces four "
                          "genetically different haploid cells."),
    ]:
        r = grade_answer(q, text, LEX)
        assert r.marks == r.max_marks, (text, statuses(r), r.flags)
        assert not {"role_swap", "contradiction"} & set(r.flags)


def test_role_pairs_come_from_parallel_key_points_without_hardcoded_words():
    units = [T.content_stems(u) for u in ("arteries carry blood away from the heart",
                                          "veins carry blood back to the heart")]
    pairs = T.role_pairs(units)
    assert {(p.mine, p.theirs) for p in pairs[0]} == {("artery", "vein"), ("away", "back")}
    # unrelated key points have no role pairs
    assert T.role_pairs([T.content_stems("the model fits the training data too closely"),
                         T.content_stems("a remedy such as dropout")]) == [[], []]


# ------------------------------------------------------------------ (4) contradictions
def test_flipped_fact_is_contradicted_with_evidence():
    text = ("Overfitting is when a model fits the training data too closely. It learns noise or random patterns. "
            "An overfit model performs very well on new data.")
    r = grade_answer(sample_q(2), text, LEX)
    k3 = r.key_points[2]
    assert k3.status == "contradicted" and k3.awarded == 0 and k3.reason == "contradiction"
    assert "performs very well on new data" in k3.evidence_sentence and k3.evidence_sentence in text
    assert "contradiction" in r.flags and r.marks < r.max_marks


def test_unlabelled_data_contradicts_labelled_data():
    text = "Supervised learning finds clusters in unlabelled data without any labels."
    r = grade_answer(sample_q(1), text, LEX)
    assert r.key_points[0].status == "contradicted" and "contradiction" in r.flags
    assert r.marks == 0


def test_negation_contradicts_but_moved_negation_does_not():
    q4 = sample_q(4)
    r = grade_answer(q4, "It evaluates the model on data used for training.", LEX)
    assert r.key_points[1].status == "contradicted"
    ok = grade_answer(q4, "We keep it aside to evaluate the model on data it was not trained on.", LEX)
    assert ok.key_points[1].status == "matched"


def test_polarity_table_and_prefix_rule():
    assert T.relation("poorly", "badly") == 1 and T.relation("poorly", "well") == -1
    assert T.relation("labelled", "unlabelled") == -1 and T.relation("supervised", "unsupervised") == -1
    assert T.prefix_opposite("overfitting", "underfitting") and T.prefix_opposite("valid", "invalid")
    assert not T.prefix_opposite("increase", "crease") and not T.prefix_opposite("information", "formation")
    assert T.NEGATIONS.isdisjoint({"poorly", "unseen", "unlabelled", "fails", "lacks"})
    assert T.opposite_concept("Explain overfitting.", "Underfitting is when the model is too simple.") == \
        ("overfitting", "underfitting")
    assert T.opposite_concept("Explain overfitting.", "Unlike underfitting, overfitting memorises noise.") is None
    assert T.opposite_concept("Why is the natural greenhouse effect important?",
                              "Burning fossil fuels is an unnatural addition to it.") is None


# ------------------------------------------------------------------ (5) missing key concepts
def test_missing_key_concepts_are_reported():
    r = grade_answer(sample_q(2), A["q2_continued_p1"], LEX)
    missed = [k for k in r.key_points if k.status == "missed"]
    assert [k.index for k in missed] == [3, 4] and all(k.reason == "no_support" for k in missed)


# ------------------------------------------------------------------ (6) max marks / property test
def test_random_answers_and_rubrics_stay_within_bounds():
    rng = random.Random(20261123)
    vocab = sorted({w for t in list(A.values()) + [q["prompt"] for q in samples.RUBRIC["questions"]]
                    for w in t.replace(".", " ").replace(",", " ").split()})
    vocab += ["not", "never", "without", "poorly", "well", "unlabelled", "underfitting", "=", "10", "N", "2.5"]
    for _ in range(40):
        n_kp = rng.randint(1, 5)
        kps = [kp(" ".join(rng.choices(vocab, k=rng.randint(3, 10))), round(rng.uniform(0.1, 3), 2),
                  tuple(rng.choices(vocab, k=rng.randint(0, 2)))) for _ in range(n_kp)]
        max_marks = round(rng.choice([1.0, 0.8, 1.2]) * sum(k["weight"] for k in kps), 2) or 1.0
        q = question(" ".join(rng.choices(vocab, k=6)), kps, rng.choice(["short", "long", "paragraph"]), max_marks)
        sentences = [" ".join(rng.choices(vocab, k=rng.randint(1, 14))) for _ in range(rng.randint(0, 6))]
        answer = rng.choice([". ", "\n", "; "]).join(sentences)
        r = grade_answer(q, answer, LEX)
        assert_well_formed(r, answer)


def test_marks_capped_when_weights_exceed_max():
    q = question("Explain overfitting.", sample_q(2).key_points, max_marks=3)
    r = grade_answer(q, A["q2_full"], LEX)
    assert r.marks == 3 and r.verdict == "correct"


# ------------------------------------------------------------------ (7) evidence
def test_every_awarded_mark_has_criterion_and_verified_evidence():
    for key, text in A.items():
        r = grade_answer(sample_q(int(key[1])), text, LEX)
        assert_well_formed(r, text)
        for k in r.key_points:
            if k.awarded:
                assert k.text and k.evidence_sentence in text


def test_evidence_survives_abbreviations_and_odd_spacing():
    spans = T.sentence_spans("It performs poorly, i.e. it fails to generalise.  E.g. dropout helps.\nNext one")
    assert len(spans) == 2 + 1
    text = "Overfitting is when a model fits the training data too closely.\n\n  It learns noise or random patterns."
    r = grade_answer(sample_q(2), text, LEX)
    assert all(k.evidence_sentence in text for k in r.key_points if k.evidence_sentence)


# ------------------------------------------------------------------ (8) review routing
def test_low_overlap_paraphrase_flagged_and_not_auto_accepted():
    r = grade_answer(sample_q(2), HEAVY_PARAPHRASE, LEX)
    assert "possible_paraphrase" in r.flags
    status, _flags = route(combine(1.0, r.grading_confidence, 0.95), 1.0, r.flags, auto_accept=0.8, low_ocr=0.5)
    assert status == "needs_review"
    # even a hypothetical very confident grader would not auto-accept it
    status, _flags = route(0.99, 1.0, ["possible_paraphrase"], auto_accept=0.8, low_ocr=0.5)
    assert status == "needs_review"


def test_review_flags_in_routing():
    for flag in ("possible_paraphrase", "off_topic", "copied_question", "contradiction", "role_swap",
                 "graders_disagree", "ai_uncertain", "ai_unverified_evidence", "ai_incomplete", "ai_contradiction"):
        assert flag in REVIEW_FLAGS
        assert route(0.99, 1.0, [flag], auto_accept=0.8, low_ocr=0.5)[0] == "needs_review"
    for info in ("continues_across_pages", "ai_unavailable_fallback", "low_reliability_grader"):
        assert route(0.99, 1.0, [info], auto_accept=0.8, low_ocr=0.5)[0] == "auto_accepted"
    # the lexical grader's explanation flag does not block a deliberately lowered threshold
    assert route(0.6, 1.0, ["low_reliability_grader"], auto_accept=0.55, low_ocr=0.5)[0] == "auto_accepted"
    assert route(0.6, 1.0, ["low_reliability_grader"], auto_accept=0.8, low_ocr=0.5)[0] == "needs_review"


def test_confidence_is_reliability_not_distance_from_thresholds():
    # the old engine gave a 0.5/5 paraphrase 0.93 confidence; lexical now tops out at its prior
    r = grade_answer(sample_q(2), HEAVY_PARAPHRASE, LEX)
    assert r.grading_confidence < 0.5
    full = grade_answer(sample_q(2), A["q2_full"], LEX)
    assert full.grading_confidence <= SCORER_PRIORS["lexical"]
    assert "low_reliability_grader" in full.flags
    assert route(combine(1.0, full.grading_confidence, 0.95), 1.0, full.flags,
                 auto_accept=0.8, low_ocr=0.5)[0] == "needs_review"


# ------------------------------------------------------------------ (9) trace and logging
def test_trace_has_scores_and_reasons_but_no_answer_text(caplog):
    secret = "Zanzibarquokka"
    text = (f"Overfitting is when a model fits the training data too closely, says {secret}. It learns noise or "
            f"random patterns. An overfit model performs very well on new data.")
    with caplog.at_level(logging.DEBUG, logger="app.grading"):
        r = grade_answer(sample_q(2), text, LEX)
    dumped = json.dumps(r.trace)
    assert secret.lower() not in dumped.lower() and "very well" not in dumped
    assert r.trace["scorer"] == "lexical" and r.trace["thresholds"] == {"tau": 0.6, "tau_low": 0.4,
                                                                         "contradiction": 0.5}
    for kt in r.trace["key_points"]:
        assert {"status", "reason", "entail", "contra", "lexical_recall", "distinctive_hits"} <= set(kt)
    logged = " ".join(rec.getMessage() for rec in caplog.records if rec.name == "app.grading")
    assert "grade question=" in logged and "contradicted" in logged
    assert secret.lower() not in logged.lower() and "very well" not in logged


# ------------------------------------------------------------------ (10) contracts and registry
def test_result_field_names_are_stable():
    # the ensemble / processor code against these names
    from dataclasses import fields

    from app.grading.engine import KeyPointResult

    assert [f.name for f in fields(KeyPointResult)] == [
        "index", "text", "weight", "entailment", "contradiction", "evidence_sentence", "awarded", "status", "reason"]
    assert [f.name for f in fields(GradeResult)] == [
        "marks", "max_marks", "grading_confidence", "key_points", "flags", "grader", "verdict", "trace"]


def test_result_contract_is_additive():
    r = grade_answer(sample_q(2), A["q2_worked_example"], LEX)
    kd = r.key_points[0].as_dict()
    assert {"index", "text", "weight", "entailment_prob", "contradiction_prob", "evidence_sentence", "awarded",
            "matched", "status"} <= set(kd) and "reason" in kd
    assert r.verdict == "partially_correct" and isinstance(r.trace, dict)
    maths = grade_answer(sample_q(3, type="maths"), A["q3_full"], LEX)
    assert maths.marks == 0 and maths.flags == ["maths_manual_review"] and maths.verdict == "ungraded"


def test_thresholds_per_scorer():
    assert default_thresholds("lexical") == Thresholds(0.6, 0.4, 0.5)
    onnx = default_thresholds("nli-onnx")
    assert onnx == DEFAULT_THRESHOLDS["nli-onnx"] and onnx is not DEFAULT_THRESHOLDS["nli-onnx"]
    # unconfigured settings (the generic defaults) do not force lexical thresholds on NLI
    assert resolve_thresholds("nli-onnx", Thresholds()) == onnx
    assert resolve_thresholds("nli-onnx", Thresholds(0.7, 0.2, 0.9)) == Thresholds(0.7, 0.2, 0.9)
    assert resolve_thresholds("lexical", Thresholds(0.99, 0.98, 0.5)).tau == 0.99


def test_scorer_registry():
    assert get_scorer("lexical", "x", "y").name == "lexical"
    assert "nli-onnx" in grader_capabilities()
    with pytest.raises(ValueError):
        get_scorer("no-such-grader", "x", "y")
    if not onnx_available():
        with pytest.raises(RuntimeError):
            get_scorer("nli-onnx", "x", "y", onnx_model_dir="does/not/exist")


# ------------------------------------------------------------------ nli-onnx (skips without the model)
@pytest.fixture(scope="module")
def onnx_scorer():
    if not onnx_available():
        pytest.skip("nli-onnx model or onnxruntime not installed (python -m scripts.download_models)")
    return get_scorer("nli-onnx", "x", "y")


def test_onnx_registered_and_auto_prefers_it(onnx_scorer):
    from app.grading.scorers import NliScorer

    assert onnx_scorer.name == "nli-onnx"
    if not NliScorer.available():
        assert get_scorer("auto", "x", "y") is onnx_scorer


def test_onnx_worked_example_and_full_answer(onnx_scorer):
    r = grade_answer(sample_q(2), A["q2_worked_example"], onnx_scorer)
    assert r.marks >= 4 and r.key_points[3].status == "missed" and "contradiction" not in r.flags
    assert r.trace["thresholds"] == {"tau": 0.5, "tau_low": 0.15, "contradiction": 0.95}
    full = grade_answer(sample_q(2), A["q2_full"], onnx_scorer)
    assert full.marks == 5 and "contradiction" not in full.flags
    assert_well_formed(full, A["q2_full"])


def test_onnx_flipped_fact_and_role_swap(onnx_scorer):
    flipped = grade_answer(sample_q(2), A["q2_flipped"], onnx_scorer)
    assert flipped.key_points[2].status == "contradicted" and "contradiction" in flipped.flags
    swap = grade_answer(ARTERIES_VEINS, "Veins carry blood away from the heart. Arteries carry blood back to "
                                        "the heart.", onnx_scorer)
    assert swap.marks == 0 and "contradiction" in swap.flags
    ok = grade_answer(ARTERIES_VEINS, "Arteries carry blood away from the heart, whereas veins carry blood back "
                                      "to the heart.", onnx_scorer)
    assert ok.marks == 2 and not {"role_swap", "contradiction"} & set(ok.flags)
