"""Grading ensemble (app.grading.ensemble): roles, per-criterion disputes, verified
evidence, the partial-credit rule, confidence factors, feedback and the trace. Graders
are faked with plain objects shaped like engine.GradeResult / llm_grader.LLMGrade, so
these tests hold whatever the engine's internals do."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from app.grading import ensemble as E
from app.llm.router import LLMUnavailable

S1 = "Supervised learning trains a model on labelled examples."
S2 = "It learns a function that maps inputs to outputs."
S3 = "Overfiting happens when the model lerns the noize in the data."
ANSWER = f"{S1} {S2} {S3}"
KPS = ["the model is trained on labelled data", "it learns a mapping from inputs to outputs",
       "it predicts outputs for unseen inputs"]
STUDENT_REF = "21BCE1234"


def question(weights=(1.0, 1.0, 1.0), max_marks: float | None = None, qtype: str = "long", texts=KPS) -> NS:
    return NS(question_no=1, type=qtype, max_marks=float(max_marks if max_marks is not None else sum(weights)),
              prompt="What is supervised learning?", model_answer=None,
              key_points=[{"text": t, "weight": w, "keywords": []} for t, w in zip(texts, weights)])


def local(grader: str, *kps: tuple[str, str | None], flags=(), conf: float = 0.7, weights=None) -> NS:
    """A GradeResult-like object; kps = (status, evidence) per key point."""
    weights = weights or [1.0] * len(kps)
    credit = {"matched": 1.0, "partial": 0.5}
    points = [NS(index=i, text=KPS[i] if i < len(KPS) else f"kp{i}", weight=w,
                 entailment=0.9 if st == "matched" else 0.4 if st == "partial" else 0.05,
                 contradiction=0.97 if st == "contradicted" else 0.01, evidence_sentence=ev,
                 awarded=w * credit.get(st, 0.0), status=st,
                 reason={"matched": "entailed", "partial": "partial_support", "missed": "no_support",
                         "contradicted": "contradiction"}[st])
              for i, ((st, ev), w) in enumerate(zip(kps, weights))]
    marks = round(sum(p.awarded for p in points), 2)
    return NS(marks=marks, max_marks=sum(weights), grading_confidence=conf, key_points=points, flags=list(flags),
              grader=grader, verdict="", trace={"prior": 0.8, "thresholds": {"tau": 0.5, "tau_low": 0.15}})


def llm(*kps: tuple[str, str | None], flags=(), feedback: str = "Clear definition; say how it predicts.",
        uncertain: bool = False, weights=None) -> NS:
    """An LLMGrade-like object (the AI's reasons are free text and may quote the answer)."""
    weights = weights or [1.0] * len(kps)
    credit = {"matched": 1.0, "partial": 0.5}
    points = [{"index": i, "text": KPS[i] if i < len(KPS) else f"kp{i}", "weight": w, "status": st,
               "awarded": w * credit.get(st, 0.0), "evidence_sentence": ev,
               "reason": f"The student wrote '{ev}'." if ev else "Not mentioned."}
              for i, ((st, ev), w) in enumerate(zip(kps, weights))]
    marks = round(sum(p["awarded"] for p in points), 2)
    return NS(question_no=1, max_marks=sum(weights), marks=marks, verdict=E.verdict_for(marks, sum(weights)),
              key_points=points, feedback=feedback, uncertain=uncertain, flags=list(flags), provider="groq",
              model="openai/gpt-oss-120b", cached=False, prompt_version="grade-sheet-v1")


M, P_, X = "matched", "partial", "missed"
C = "contradicted"


# ------------------------------------------------------------------ agreement and disputes
def test_agreement_gives_full_factor_and_template_feedback():
    p = local("nli-onnx", (M, S1), (M, S2), (X, None))
    s = local("lexical", (M, S1), (M, S2), (X, None))
    r = E.combine(question(), ANSWER, p, s)
    assert (r.marks, r.max_marks, r.verdict) == (2.0, 3.0, "partially_correct")
    assert [k.status for k in r.key_points] == [M, M, X]
    assert [k.awarded for k in r.key_points] == [1.0, 1.0, 0.0]
    assert r.graders_used == ["nli-onnx", "lexical"] and r.grader == "nli-onnx+lexical"
    assert "graders_disagree" not in r.flags and not any(k.disputed for k in r.key_points)
    assert r.grading_confidence == pytest.approx(E.PRIORS["nli-onnx"] * E.AGREE_FACTOR)
    assert r.trace["agreement_factor"] == 1.0 and r.trace["marks_gap"] == 0.0
    assert r.feedback == ('Covered: "the model is trained on labelled data", "it learns a mapping from inputs to '
                          'outputs"; Missing: "it predicts outputs for unseen inputs".')
    assert r.key_points[0].evidence_sentence == S1 and r.key_points[2].evidence_sentence is None
    assert r.key_points[1].statuses_by_grader == {"nli-onnx": M, "lexical": M}


@pytest.mark.parametrize(("p_status", "s_status"), [(M, X), (P_, X), (X, M), (X, P_)])
def test_award_versus_nothing_is_a_dispute_and_the_primary_stands(p_status, s_status):
    ev = {M: S2, P_: S2, X: None}
    p = local("nli-onnx", (M, S1), (p_status, ev[p_status]), (X, None))
    s = local("lexical", (M, S1), (s_status, ev[s_status]), (X, None))
    r = E.combine(question(), ANSWER, p, s)
    assert r.key_points[1].status == p_status and r.key_points[1].disputed
    assert r.trace["disputed"] == [1] and "graders_disagree" in r.flags
    assert r.marks == 1.0 + E.CREDIT[p_status]
    assert r.grading_confidence == pytest.approx(0.8 * E.DISAGREE_FACTOR)


def test_contradiction_versus_matched_is_a_dispute_either_way():
    # the secondary finds a verified contradiction the primary missed: flag, dispute, primary stands
    p = local("nli-onnx", (M, S1), (M, S2), (X, None))
    s = local("lexical", (C, S1), (M, S2), (X, None))
    r = E.combine(question(), ANSWER, p, s)
    assert r.key_points[0].status == M and r.key_points[0].disputed
    assert {"contradiction", "graders_disagree"} <= set(r.flags) and r.marks == 2.0
    assert r.trace["key_points"][0]["contradiction_by"] == ["lexical"]

    # the primary contradicts with a verified quote while the secondary awards: 0 for it, dispute
    p = local("nli-onnx", (C, S1), (M, S2), (X, None))
    s = local("lexical", (M, S1), (M, S2), (X, None))
    r = E.combine(question(), ANSWER, p, s)
    assert r.key_points[0].status == C and r.key_points[0].awarded == 0 and r.key_points[0].disputed
    assert r.key_points[0].evidence_sentence == S1
    assert {"contradiction", "graders_disagree"} <= set(r.flags) and r.marks == 1.0


def test_contradicted_versus_missed_is_not_a_dispute_but_is_flagged():
    p = local("nli-onnx", (C, S1), (M, S2), (X, None))
    s = local("lexical", (X, None), (M, S2), (X, None))
    r = E.combine(question(), ANSWER, p, s)
    assert not r.key_points[0].disputed and "graders_disagree" not in r.flags
    assert "contradiction" in r.flags and r.grading_confidence == pytest.approx(0.8)


def test_contradiction_from_an_extra_grader_counts_only_with_a_verified_quote():
    a = llm((M, S1), (M, S2), (X, None))
    n = local("nli-onnx", (M, S1), (M, S2), (X, None))
    lex = local("lexical", (C, S1), (M, S2), (X, None))
    r = E.combine(question(), ANSWER, a, n, [lex])
    assert r.graders_used == ["llm", "nli-onnx", "lexical"]
    assert r.key_points[0].status == M and r.key_points[0].disputed
    assert {"contradiction", "graders_disagree"} <= set(r.flags)

    invented = local("lexical", (C, "The model never sees any labels at all."), (M, S2), (X, None))
    r = E.combine(question(), ANSWER, a, n, [invented])
    assert "contradiction" not in r.flags and not r.key_points[0].disputed


def test_matched_versus_partial_is_minor_and_only_traced():
    p = local("nli-onnx", (M, S1), (M, S2), (X, None))
    s = local("lexical", (M, S1), (P_, S2), (X, None))
    r = E.combine(question(), ANSWER, p, s)
    assert not any(k.disputed for k in r.key_points) and "graders_disagree" not in r.flags
    assert r.trace["minor_disagreements"] == [1] and r.trace["key_points"][1]["minor_disagreement"]
    assert r.trace["marks_gap"] == 0.5 and not r.trace["marks_gap_exceeded"]
    assert r.grading_confidence == pytest.approx(0.8)


def test_marks_gap_above_half_a_mark_is_a_disagreement():
    q = question(weights=(2.0, 1.0, 1.0))
    p = local("nli-onnx", (M, S1), (M, S2), (X, None), weights=[2.0, 1.0, 1.0])
    s = local("lexical", (P_, S1), (M, S2), (X, None), weights=[2.0, 1.0, 1.0])
    r = E.combine(q, ANSWER, p, s)
    assert not any(k.disputed for k in r.key_points)  # still "minor" per criterion
    assert r.trace["marks_gap"] == 1.0 and r.trace["marks_gap_exceeded"]
    assert "graders_disagree" in r.flags and r.trace["agreement_factor"] == E.DISAGREE_FACTOR


# ------------------------------------------------------------------ evidence
def test_award_without_a_verified_quote_is_withdrawn():
    p = local("nli-onnx", (M, "The model memorises every label perfectly."), (M, S2), (X, None))
    s = local("lexical", (X, None), (M, S2), (X, None))
    r = E.combine(question(), ANSWER, p, s)
    k = r.key_points[0]
    assert (k.status, k.awarded, k.reason, k.evidence_sentence) == (X, 0.0, "no_verified_evidence", None)
    assert "no_verified_evidence" in r.flags and r.marks == 1.0 and r.trace["downgraded"] == [0]
    assert not k.disputed  # both end at "no marks"


def test_primary_award_can_stand_on_the_secondarys_verified_quote():
    p = local("nli-onnx", (M, "a paraphrase the grader produced"), (M, S2), (X, None))
    s = local("lexical", (P_, S1), (M, S2), (X, None))
    r = E.combine(question(), ANSWER, p, s)
    assert r.key_points[0].status == M and r.key_points[0].evidence_sentence == S1
    assert r.trace["key_points"][0]["evidence_from"] == "lexical" and r.marks == 2.0
    assert "no_verified_evidence" not in r.flags


def test_unverified_contradiction_becomes_missed():
    p = local("nli-onnx", (C, "Labels are never used."), (M, S2), (X, None))
    r = E.combine(question(), ANSWER, p)
    assert r.key_points[0].status == X and "contradiction" not in r.flags
    assert "no_verified_evidence" in r.flags


def test_ocr_noisy_quote_is_verified_by_fuzzy_match():
    a = llm((M, S1), (M, S2), (P_, "the model learns the noise in the data"))
    r = E.combine(question(), ANSWER, a, local("nli-onnx", (M, S1), (M, S2), (P_, S3)))
    assert r.key_points[2].status == P_
    assert r.key_points[2].evidence_sentence == "the model lerns the noize in the data"


# ------------------------------------------------------------------ marks
def test_marks_follow_the_partial_credit_rule_and_are_capped():
    q = question(weights=(1.5, 1.25, 0.75), max_marks=3.0)
    w = [1.5, 1.25, 0.75]
    r = E.combine(q, ANSWER, local("lexical", (M, S1), (M, S2), (M, S3), weights=w))
    assert (r.marks, r.verdict) == (3.0, "correct")  # 3.5 capped at max
    r = E.combine(q, ANSWER, local("lexical", (P_, S1), (M, S2), (X, None), weights=w))
    assert (r.marks, r.verdict) == (2.0, "partially_correct")

    third = question(weights=(1 / 3, 1 / 3, 1 / 3), max_marks=1.0)
    w3 = [1 / 3] * 3
    assert E.combine(third, ANSWER, local("lexical", (M, S1), (M, S2), (P_, S3), weights=w3)).marks == 0.83
    r = E.combine(third, ANSWER, local("lexical", (M, S1), (M, S2), (M, S3), weights=w3))
    assert (r.marks, r.verdict) == (1.0, "correct")
    r = E.combine(third, ANSWER, local("lexical", (X, None), (C, S2), (X, None), weights=w3))
    assert (r.marks, r.verdict) == (0.0, "incorrect")


def test_verdict_rules():
    assert E.verdict_for(5.0, 5.0) == "correct" and E.verdict_for(0.0, 5.0) == "incorrect"
    assert E.verdict_for(2.5, 5.0) == "partially_correct" and E.verdict_for(3.33, 10 / 3) == "correct"


def test_marks_never_exceed_max_for_any_status_mix():
    import itertools

    q = question(weights=(2.0, 1.5, 1.0), max_marks=4.0)
    w = [2.0, 1.5, 1.0]
    ev = {M: S1, P_: S2, X: None, C: S3}
    for combo in itertools.product((M, P_, X, C), repeat=3):
        for other in ((M, M, M), (X, X, X), combo[::-1]):
            r = E.combine(q, ANSWER, local("nli-onnx", *[(st, ev[st]) for st in combo], weights=w),
                          local("lexical", *[(st, ev[st]) for st in other], weights=w))
            assert 0.0 <= r.marks <= 4.0 and r.marks == round(r.marks, 2)
            assert r.marks == round(min(4.0, sum(wt * E.CREDIT[k.status] for wt, k in zip(w, r.key_points))), 2)
            assert r.verdict == E.verdict_for(r.marks, 4.0)
            assert all((k.evidence_sentence is not None) == (k.status != X) for k in r.key_points)


# ------------------------------------------------------------------ the AI marker
def test_ai_is_primary_when_complete_and_the_local_graders_cross_check():
    a = llm((M, S1), (M, S2), (X, None), flags=["ai_uncertain"], uncertain=True)
    n = local("nli-onnx", (M, S1), (M, S2), (X, None))
    lex = local("lexical", (M, S1), (P_, S2), (X, None))
    p, s, x = E.assign_roles(question(), n, lex, a)
    assert (p, s, x) == (a, n, [lex])
    r = E.combine(question(), ANSWER, p, s, x, ai=E.AiInfo("used", style="moderate"))
    assert r.graders_used == ["llm", "nli-onnx", "lexical"] and r.grader == "llm+nli-onnx+lexical"
    assert r.feedback == "Clear definition; say how it predicts." and r.trace["feedback_source"] == "ai"
    assert "ai_uncertain" in r.flags and r.grading_confidence == pytest.approx(0.9)
    assert r.key_points[0].source == "llm" and r.key_points[0].reason == f"The student wrote '{S1}'."
    ai = r.trace["ai"]
    assert ai["status"] == "used" and ai["provider"] == "groq" and ai["model"] == "openai/gpt-oss-120b"
    assert ai["style"] == "moderate" and ai["cached"] is False
    assert r.key_points[0].entailment == pytest.approx(0.9)  # probabilities come from the local grader


def test_ai_feedback_is_replaced_when_an_ai_award_was_withdrawn():
    a = llm((M, "words the student never wrote"), (M, S2), (X, None))
    r = E.combine(question(), ANSWER, a, local("nli-onnx", (X, None), (M, S2), (X, None)))
    assert r.key_points[0].status == X and r.trace["feedback_source"] == "template"
    assert r.feedback.startswith('Covered: "it learns a mapping')


def test_question_missing_from_the_ai_result_falls_back_to_the_local_graders():
    n = local("nli-onnx", (M, S1), (M, S2), (X, None))
    lex = local("lexical", (M, S1), (M, S2), (X, None))
    p, s, x = E.assign_roles(question(), n, lex, None)
    assert (p, s, x) == (n, lex, [])
    # the AI replied but left this question out: offline marks stand, flagged ai_incomplete (review)
    r = E.combine(question(), ANSWER, p, s, x, ai=E.AiInfo("fallback", "not_graded"))
    assert "ai_incomplete" in r.flags and "ai_unavailable_fallback" not in r.flags
    assert r.trace["ai"] == {"status": "fallback", "reason": "not_graded"}
    assert r.graders_used == ["nli-onnx", "lexical"] and r.marks == 2.0
    down = E.combine(question(), ANSWER, p, s, x, ai=E.AiInfo("fallback", "unavailable"))
    assert "ai_unavailable_fallback" in down.flags and "ai_incomplete" not in down.flags

    off = E.combine(question(), ANSWER, p, s, x, ai=E.AiInfo("off"))
    assert "ai_unavailable_fallback" not in off.flags and off.trace["ai"] == {"status": "off"}


def test_incomplete_ai_judgement_is_an_extra_opinion_only():
    a = llm((M, S1), (X, None), (X, None), flags=["ai_incomplete", "ai_uncertain"], uncertain=True)
    a.key_points = a.key_points[:1]  # the AI left two criteria out
    n = local("nli-onnx", (M, S1), (M, S2), (X, None))
    lex = local("lexical", (M, S1), (M, S2), (X, None))
    p, s, x = E.assign_roles(question(), n, lex, a)
    assert (p, s, x) == (a, n, [lex])  # the AI decides what it judged ...
    r = E.combine(question(), ANSWER, p, s, x, ai=E.AiInfo("used"))
    # ... and the criteria it left out fall back to the offline graders (never a made-up 0)
    assert r.marks == 2.0 and r.trace["ai_fallback"] == [1, 2]
    assert [k.source for k in r.key_points] == ["llm", "nli-onnx", "nli-onnx"]
    assert {"ai_incomplete", "ai_uncertain"} <= set(r.flags) and r.trace["feedback_source"] == "template"
    assert r.key_points[1].statuses_by_grader == {"llm": "not_judged", "nli-onnx": M, "lexical": M}
    nothing = llm((X, None), (X, None), (X, None), flags=["ai_incomplete"])
    nothing.key_points = []
    assert E.assign_roles(question(), n, lex, nothing) == (n, lex, [nothing])
    r = E.combine(question(), ANSWER, n, lex, [nothing])
    assert r.trace["ai"]["status"] == "incomplete" and "ai_incomplete" in r.flags and r.marks == 2.0


def test_confidence_factors_by_primary():
    lone = E.combine(question(), ANSWER, local("lexical", (M, S1), (M, S2), (X, None)))
    assert lone.grading_confidence == pytest.approx(0.6 * E.NO_SECONDARY_FACTOR)
    assert lone.trace["agreement_factor"] == E.NO_SECONDARY_FACTOR and lone.trace["secondary"] is None
    agree = E.combine(question(), ANSWER, llm((M, S1), (M, S2), (X, None)), local("nli-onnx", (M, S1), (M, S2), (X, None)))
    assert agree.grading_confidence == pytest.approx(0.9)
    split = E.combine(question(), ANSWER, llm((M, S1), (M, S2), (X, None)), local("nli-onnx", (X, None), (M, S2), (X, None)))
    assert split.grading_confidence == pytest.approx(0.9 * E.DISAGREE_FACTOR)
    sbert = E.combine(question(), ANSWER, local("sbert", (M, S1), (M, S2), (X, None)))
    assert sbert.grading_confidence == pytest.approx(0.7 * 0.85)


# ------------------------------------------------------------------ feedback, determinism, trace
def test_feedback_template_groups_and_truncates():
    long_kp = "a criterion " + "with many words " * 10
    q = question(texts=[KPS[0], long_kp, KPS[2]])
    r = E.combine(q, ANSWER, local("lexical", (P_, S1), (C, S2), (X, None)))
    assert r.feedback.startswith('Partly: "the model is trained on labelled data"; Missing: '
                                 '"it predicts outputs for unseen inputs"; Contradicts: "a criterion with many')
    assert "…" in r.feedback and len(r.feedback) <= E.FEEDBACK_MAX_CHARS
    assert E.feedback_template([]) == ""


def test_combine_is_deterministic():
    args = (question(), ANSWER, llm((M, S1), (P_, S2), (X, None)), local("nli-onnx", (M, S1), (X, None), (X, None)),
            [local("lexical", (M, S1), (M, S2), (C, S3))])
    a = json.dumps(E.combine(*args, ai=E.AiInfo("used")).as_dict(), sort_keys=True)
    b = json.dumps(E.combine(*args, ai=E.AiInfo("used")).as_dict(), sort_keys=True)
    assert a == b


def test_trace_contains_no_answer_text():
    a = llm((M, S1), (P_, S2), (C, S3))
    r = E.combine(question(), ANSWER, a, local("nli-onnx", (M, S1), (X, None), (C, S3)),
                  [local("lexical", (M, S1), (M, S2), (X, None))], ai=E.AiInfo("used"))
    dumped = json.dumps(r.trace)
    for sentence in (S1, S2, S3):
        assert sentence not in dumped
    for word in ("Supervised", "labelled examples", "Overfiting", "lerns", "noize", "The student wrote"):
        assert word not in dumped
    assert STUDENT_REF not in dumped
    assert r.trace["key_points"][0]["reason"] == "ai_judgement"
    assert all(set(k) <= {"index", "weight", "status", "awarded", "source", "evidence_from", "reason", "disputed",
                          "minor_disagreement", "ai_fallback", "contradiction_by", "statuses_by_grader"}
               for k in r.trace["key_points"])


def test_maths_and_rubricless_results_pass_through_as_ungraded():
    maths = question(qtype="maths")
    p = NS(marks=0.0, max_marks=3.0, grading_confidence=0.0, key_points=[], flags=["maths_manual_review"],
           grader="nli-onnx", verdict="ungraded", trace={})
    r = E.combine(maths, ANSWER, p, local("lexical", (X, None), (X, None), (X, None)))
    assert (r.marks, r.verdict, r.key_points, r.grading_confidence) == (0.0, "ungraded", [], 0.0)
    assert r.flags == ["maths_manual_review"] and r.feedback == ""
    norubric = NS(question_no=2, type="short", max_marks=2.0, prompt="p", model_answer="", key_points=[])
    p2 = NS(marks=0.0, max_marks=2.0, grading_confidence=0.0, key_points=[], flags=["no_rubric"], grader="lexical",
            verdict="ungraded", trace={})
    assert E.combine(norubric, "Some text.", p2).verdict == "ungraded"


def test_real_result_types_are_accepted():
    from app.grading.engine import GradeResult, KeyPointResult
    from app.grading.llm_grader import LLMGrade

    kp = [KeyPointResult(0, KPS[0], 1.0, 0.9, 0.0, S1, 1.0, M, "entailed"),
          KeyPointResult(1, KPS[1], 1.0, 0.2, 0.0, None, 0.0, X, "no_support"),
          KeyPointResult(2, KPS[2], 1.0, 0.1, 0.0, None, 0.0, X, "no_support")]
    g = GradeResult(1.0, 3.0, 0.7, kp, ["low_reliability_grader"], "lexical", "partially_correct", {"prior": 0.6})
    a = LLMGrade(1, 3.0, 2.0, "partially_correct",
                 [{"index": 0, "text": KPS[0], "weight": 1.0, "status": M, "awarded": 1.0, "evidence_sentence": S1,
                   "reason": "ok"},
                  {"index": 1, "text": KPS[1], "weight": 1.0, "status": M, "awarded": 1.0, "evidence_sentence": S2,
                   "reason": "ok"},
                  {"index": 2, "text": KPS[2], "weight": 1.0, "status": X, "awarded": 0.0, "evidence_sentence": None,
                   "reason": ""}],
                 "Good.", False, [], "groq", "openai/gpt-oss-120b", True)
    r = E.combine(question(), ANSWER, *E.assign_roles(question(), g, None, a)[:2])
    assert r.graders_used == ["llm", "lexical"] and r.marks == 2.0 and r.key_points[1].disputed
    assert r.trace["ai"]["cached"] is True
    lone = E.combine(question(), ANSWER, g)
    assert lone.flags == ["low_reliability_grader"] and lone.marks == 1.0
    assert lone.key_points[0].as_dict()["entailment_prob"] == 0.9


# ------------------------------------------------------------------ settings and the shared helpers
@pytest.fixture
def auto_is(monkeypatch):
    from app.grading import scorers

    real = scorers.get_scorer

    def set_auto(kind: str) -> None:
        monkeypatch.setattr(scorers, "get_scorer",
                            lambda k, *a, **kw: NS(name=kind) if k == "auto" else real(k, *a, **kw))
        E.reset_caches()

    yield set_auto
    E.reset_caches()


def test_local_plan(auto_is):
    def plan(**kw):  # noqa: ANN003, ANN202
        return E.local_plan(NS(**{"grader": "auto", "local_primary": "auto", "sbert_model": "s", "nli_model": "n", **kw}))

    auto_is("nli-onnx")
    assert plan() == E.LocalPlan("nli-onnx", "lexical")
    assert plan(grader="lexical") == E.LocalPlan("lexical", None)
    assert plan(local_primary="lexical") == E.LocalPlan("lexical", "nli-onnx")
    assert plan(local_primary="lexical", grader="lexical") == E.LocalPlan("lexical", None)
    assert plan(local_primary="nli-onnx", grader="lexical") == E.LocalPlan("nli-onnx", "lexical")
    assert plan(grader="sbert") == E.LocalPlan("sbert", "lexical")
    auto_is("lexical")
    assert plan() == E.LocalPlan("lexical", None)


def test_thresholds_are_per_scorer_unless_configured():
    from app.config import Settings
    from app.grading.engine import Thresholds

    s = Settings(_env_file=None, env="test")
    assert E.thresholds_for(s, "lexical") == Thresholds(0.6, 0.4, 0.5)
    assert E.thresholds_for(s, "nli-onnx") == Thresholds(0.5, 0.15, 0.95)  # was 0.6/0.4/0.5 for every scorer
    strict = Settings(_env_file=None, env="test", tau=0.7, contradiction_threshold=0.9)
    assert E.thresholds_for(strict, "lexical") == Thresholds(0.7, 0.4, 0.9)
    assert E.thresholds_for(strict, "nli-onnx") == Thresholds(0.7, 0.15, 0.9)


def test_grade_ensemble_runs_the_primary_function_and_the_lexical_cross_check():
    from app.config import Settings

    s = Settings(_env_file=None, env="test", grader="lexical", local_primary="nli-onnx")
    calls = []

    def fake_primary(q, text, settings):  # noqa: ANN001, ANN202
        calls.append(q.question_no)
        return local("nli-onnx", (M, S1), (M, S2), (X, None))

    r = E.grade_ensemble(question(), ANSWER, s, primary_fn=fake_primary)
    assert calls == [1] and r.graders_used == ["nli-onnx", "lexical"]
    assert set(r.trace["graders"]) == {"nli-onnx", "lexical"} and r.trace["secondary"] == "lexical"
    lone = E.grade_ensemble(question(), ANSWER, Settings(_env_file=None, env="test", grader="lexical"))
    assert lone.graders_used == ["lexical"] and lone.trace["agreement_factor"] == E.NO_SECONDARY_FACTOR


class StubRouter:
    def __init__(self, usable=("groq:openai/gpt-oss-120b",), error: Exception | None = None) -> None:
        self.settings = NS(ai_max_input_tokens_per_call=3500)
        self._usable, self._error = list(usable), error
        self.calls = 0

    def usable_providers(self) -> list[str]:
        return self._usable

    def generate_json(self, *a, **k):  # noqa: ANN002, ANN003, ANN201
        self.calls += 1
        raise self._error or AssertionError("no reply scripted")


def test_ai_grade_items_never_raises_and_says_why():
    s = NS(ai_grading="ensemble", ai_consent=True, ai_grading_style="strict")
    maths = question(qtype="maths")
    pairs = [(question(), ANSWER), (NS(**{**vars(maths), "question_no": 2}), "x = 2")]

    out = E.ai_grade_items(StubRouter(usable=()), pairs, s)
    assert out[1].info == E.AiInfo("fallback", "no_usable_provider", (), "strict") and out[1].grade is None
    assert out[2].info.status == "not_applicable"

    secret_text = "groq:openai/gpt-oss-120b: rate_limited (rate_limited): HTTP 429 body says Supervised learning"
    router = StubRouter(error=LLMUnavailable([secret_text, "gemini:gemini-3.5-flash-lite: skipped: no API key "
                                                           "configured"]))
    out = E.ai_grade_items(router, pairs, s)
    assert out[1].info.status == "fallback" and out[1].info.reason == "unavailable"
    assert out[1].info.details == ("groq:openai/gpt-oss-120b: rate_limited",
                                   "gemini:gemini-3.5-flash-lite: skipped (no API key configured)")
    assert "Supervised" not in json.dumps(out[1].info.as_dict()) and router.calls == 1

    out = E.ai_grade_items(StubRouter(error=RuntimeError("boom")), pairs, s)
    assert out[1].info == E.AiInfo("fallback", "error", (), "strict")


def test_ai_settings_helpers():
    assert not E.ai_enabled(NS(ai_grading="ensemble", ai_consent=False))
    assert not E.ai_enabled(NS(ai_grading="off", ai_consent=True))
    assert E.ai_enabled(NS(ai_grading="ensemble", ai_consent=True))
    assert E.ai_style(NS(ai_grading_style="harsh")) == "moderate"
    assert E.has_content(" 42 ") and not E.has_content(" - . ")
    assert not E.ai_applicable(question(), "   ") and E.ai_applicable(question(), "text")
