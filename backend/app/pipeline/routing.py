"""Confidence combination and review routing (PRD 12.6, R3).

confidence = calibrate(ocr_confidence × grading_confidence × segmentation factor).
An answer is auto-accepted for spot-check only when that confidence clears the
threshold and no review flag is raised; everything else goes to needs_review,
which the review screen lists first."""

from __future__ import annotations

import math

# Informational flags never force a review on their own.
# * "low_reliability_grader" explains a low confidence: the lexical / SBERT grader's
#   prior is below the default auto-accept threshold, so its answers go to review
#   through "low_confidence"; a deployment that deliberately lowers the threshold
#   can still auto-accept.
# * "ai_unavailable_fallback": the AI marker was unavailable and the offline checker
#   marked the answer; its own flags and confidence decide the route.
INFO_FLAGS = frozenset({"continues_across_pages", "out_of_order", "partial_text", "low_reliability_grader",
                        "single_question_sheet", "ai_segmented",
                        "ai_unavailable_fallback", "ai_verified"})

# Grading flags that always need a teacher (importable by other modules), even if
# someone later makes them informational: the grader may have under-marked
# (possible_paraphrase), the answer may not answer the question (off_topic,
# copied_question), it states something opposite to the rubric (contradiction,
# role_swap), or the graders / the AI marker are unsure or disagree (graders_disagree,
# ai_uncertain, ai_unverified_evidence, ai_incomplete, ai_contradiction). Every other
# flag that is not informational also sends the answer to review.
REVIEW_FLAGS = frozenset({"possible_paraphrase", "off_topic", "copied_question", "contradiction", "role_swap",
                          "graders_disagree", "ai_uncertain", "ai_unverified_evidence", "ai_incomplete",
                          "ai_contradiction", "ai_fuzzy_evidence", "ai_weak_evidence",
                          "ai_second_opinion_differs", "page_unread"})
assert not REVIEW_FLAGS & INFO_FLAGS


def calibrate(p: float, temperature: float) -> float:
    """Temperature scaling on the logit. T is fitted on validation data (R3); 1.0 = identity."""
    p = min(max(p, 1e-4), 1 - 1e-4)
    if temperature == 1.0:
        return p
    logit = math.log(p / (1 - p)) / temperature
    return 1 / (1 + math.exp(-logit))


def combine(ocr_conf: float, grading_conf: float, seg_conf: float, temperature: float = 1.0) -> float:
    seg_factor = min(1.0, 0.4 + 0.6 * seg_conf / 0.95)
    return round(calibrate(ocr_conf * grading_conf * seg_factor, temperature), 4)


def is_review_flag(flag: str) -> bool:
    return flag in REVIEW_FLAGS or flag not in INFO_FLAGS


def route(confidence: float, ocr_conf: float, flags: list[str], *, auto_accept: float, low_ocr: float) -> tuple[str, list[str]]:
    flags = list(dict.fromkeys(flags))
    if ocr_conf < low_ocr and "low_ocr_confidence" not in flags:
        flags.append("low_ocr_confidence")
    if confidence < auto_accept and "low_confidence" not in flags:
        flags.append("low_confidence")
    review_flags = [f for f in flags if is_review_flag(f)]
    status = "needs_review" if review_flags else "auto_accepted"
    return status, flags
