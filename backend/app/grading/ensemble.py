"""Grading ensemble: one mark per answer from the AI marker and the offline graders
(spec G-3; design and limitations in docs/GRADING_DESIGN.md).

Candidates
* A - the AI (LLM) marker, :mod:`app.grading.llm_grader`: a verdict and a verified quote
  per key point. Used only with consent. It decides the key points it judged; a key
  point it did not judge (or a question it left out) falls back to the offline graders
  and the answer gets ``ai_incomplete`` - never a made-up 0;
* N - the semantic local grader (nli-onnx / torch NLI / SBERT through engine.grade_answer);
* L - the lexical grader (engine.grade_answer with the LexicalScorer).

Roles: primary P = A when it judged the question, else the configured local primary
(``settings.local_primary``; ``auto`` = N when one is available), else L. Secondary
S = the local primary when A is primary; the other local grader when P is local
(L when P is semantic; N when lexical was chosen as primary and N is configured);
none when only one grader ran. Any other result (L when P = A and S = N, or an AI
result that judged nothing) is an *extra*: it is shown and its contradictions count.

Per key point (:func:`combine`)
* the final status is the deciding grader's status (P's; S's where the AI did not judge);
* every non-missed status needs evidence that is a verified quote of the answer
  (llm_grader.find_quote: whitespace/case normalised, tolerant of OCR noise): the
  decider's quote, or the cross-checker's when the decider has none and both award marks.
  Otherwise the key point becomes ``missed`` with reason ``no_verified_evidence``
  (answer flag no_verified_evidence);
* dispute: decider and cross-checker disagree on the award level (one awards marks, the
  other awards nothing or finds a contradiction). The decider's status is kept, the key
  point is marked disputed and the answer gets ``graders_disagree``. matched vs partial
  is a minor disagreement: recorded in the trace only;
* a contradiction found by ANY grader with a verified quote flags the answer
  ``contradiction``; when the decider did not contradict, that key point is also disputed.

Per answer
* marks = sum(weight for matched) + sum(weight / 2 for partial), capped at max marks,
  2 dp (the partial-credit rule; there is no other marks formula). Verdict: correct
  at max, incorrect at 0, partially_correct otherwise (ungraded for maths / no rubric);
* spec G-3: when P's and S's marks differ by more than 0.5 the answer also gets
  ``graders_disagree`` (its key points stay "minor");
* grading confidence = prior(decider) x agreement factor: 1.0 when every key point was
  cross-checked and agrees, 0.85 when something was not cross-checked, 0.45 when the
  graders disagree. With mixed deciders (AI + offline fallback) the lower prior counts.
  The priors are PROVISIONAL (see PRIORS) until calibrated against teacher marks;
* feedback: the AI's only when the AI decided every key point and nothing changed its
  verdicts (no withdrawn award, no disputed key point, no marks gap, no AI-side
  downgrade); otherwise a deterministic summary of the final criteria
  ("Covered: ...; Partly: ...; Missing: ...; Contradicts: ..."), so feedback never
  contradicts the computed marks.

:func:`combine` is pure and deterministic: no I/O, clock, randomness or logging. Its
trace holds decisions and scores only - never answer text, names, student_ref or keys.

The helpers after it are what the pipeline (app.pipeline.processor) and the evaluation
harness (app.evaluation.grading) both call, so the two cannot drift: local_plan,
thresholds_for, grade_local_primary/secondary, grade_ensemble, ai_grade_items and
route_graded (confidence x OCR x segmentation, then routing). Only the engine's public
API is used: grade_answer, GradeResult/KeyPointResult fields, default_thresholds,
Thresholds, get_scorer and pipeline.routing's combine/route/is_review_flag.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .llm_grader import STYLES, AnswerItem, find_quote, grade_sheet_llm

log = logging.getLogger(__name__)

VERSION = 1
STATUSES = ("matched", "partial", "missed", "contradicted")
CREDIT = {"matched": 1.0, "partial": 0.5, "missed": 0.0, "contradicted": 0.0}
AWARD = frozenset({"matched", "partial"})
NOT_JUDGED = "not_judged"
# Statuses an AI result may use for a key point it did not judge.
_NOT_JUDGED_STATUSES = frozenset({"", "none", "null", NOT_JUDGED, "unjudged", "omitted", "not_graded", "ungraded",
                                  "skipped"})

LLM = "llm"
LEXICAL = "lexical"
SEMANTIC = frozenset({"nli", "nli-onnx", "sbert"})
LOCAL_PRIMARY_CHOICES = ("auto", "lexical", "nli-onnx")

# How far each grader's verdicts can be trusted when it decides.
# PROVISIONAL: chosen by hand from the graders' design (an LLM reads meaning, NLI
# reads entailment, SBERT topical similarity, lexical only word overlap) and to be
# recalibrated against teacher marks (temperature scaling, spec G-5). Never quote them
# as accuracy figures.
PRIORS: dict[str, float] = {LLM: 0.9, "nli": 0.8, "nli-onnx": 0.8, "sbert": 0.7, LEXICAL: 0.6}
DEFAULT_PRIOR = 0.5
AGREE_FACTOR = 1.0  # every key point cross-checked, and the graders agree
NO_SECONDARY_FACTOR = 0.85  # something was not cross-checked
DISAGREE_FACTOR = 0.45  # a disputed key point, or the marks differ by more than MARKS_GAP_LIMIT
MARKS_GAP_LIMIT = 0.5  # spec G-3: the AI and the checker agree when their marks differ by <= 0.5

# Routing inputs of a clean typed (PDF text-layer) answer: the confidence TextLayerOCR
# gives every line, and the confidence of a segment with no doubts (segment.Segment).
# A teacher-typed correction reads with OCR confidence 1.0. tests/test_pipeline_ai.py
# checks these against what the pipeline produces.
TYPED_OCR_CONFIDENCE = 0.99
TRANSCRIBED_OCR_CONFIDENCE = 1.0
CLEAN_SEGMENT_CONFIDENCE = 0.95

FEEDBACK_ITEM_CHARS = 60
FEEDBACK_MAX_CHARS = 400

FLAG_DISAGREE = "graders_disagree"
FLAG_CONTRADICTION = "contradiction"
FLAG_NO_EVIDENCE = "no_verified_evidence"
FLAG_PARTIAL = "partial_credit"
FLAG_AI_FALLBACK = "ai_unavailable_fallback"
FLAG_AI_INCOMPLETE = "ai_incomplete"
FLAG_FALLBACK = "graded_with_fallback"
AI_FLAGS = frozenset({"ai_uncertain", "ai_unverified_evidence", FLAG_AI_INCOMPLETE, "ai_contradiction",
                      "ai_second_opinion_differs"})
# Never auto-accepted, whatever routing's flag lists say (route_graded enforces it).
FORCE_REVIEW_FLAGS = AI_FLAGS | {FLAG_DISAGREE, FLAG_CONTRADICTION, FLAG_NO_EVIDENCE}
# AI flags meaning code already changed or completed the AI's verdicts: its feedback no
# longer describes the final marks.
AI_FEEDBACK_BLOCKERS = frozenset({"ai_unverified_evidence", FLAG_AI_INCOMPLETE})
# Kept from any grader: gaming guards about the answer itself, and how a local grader ran.
CARRY_FROM_ANY = frozenset({"copied_question", "keyword_list", FLAG_FALLBACK})
# Recomputed by the ensemble from the final statuses; never copied from a grader.
DERIVED_FLAGS = frozenset({FLAG_PARTIAL, FLAG_CONTRADICTION, FLAG_DISAGREE, FLAG_NO_EVIDENCE, FLAG_AI_FALLBACK})

_CODE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_HAS_CONTENT = re.compile(r"[^\W_]")  # at least one letter or digit, in any script


# ============================================================================ small helpers
def _field(obj: Any, name: str, default: Any = None) -> Any:  # noqa: ANN401
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _num(x: Any, default: float = 0.0) -> float:  # noqa: ANN401
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _code(reason: Any) -> str:  # noqa: ANN401
    """A reason code safe for the trace (engine codes are snake_case words; anything
    else - e.g. the AI's free-text reason, which may quote the answer - is not)."""
    s = str(reason or "")
    return s if _CODE.match(s) else ("other" if s else "")


def has_content(text: str | None) -> bool:
    """At least one letter or digit: anything else is a skipped answer (PRD 12.10) that
    the pipeline marks 'blank' (0 marks) without grading it."""
    return bool(_HAS_CONTENT.search(text or ""))


def verdict_for(marks: float, max_marks: float) -> str:
    if max_marks > 0 and marks >= round(max_marks, 2) - 1e-9:
        return "correct"
    if marks <= 1e-9:
        return "incorrect"
    return "partially_correct"


def question_key_points(question: Any) -> list[dict]:  # noqa: ANN401
    """The rubric's key points as graded (the same filtering as engine and llm_grader:
    key points with text, else the model answer as one key point worth max marks)."""
    kps = []
    for kp in _field(question, "key_points") or []:
        if isinstance(kp, Mapping) and str(kp.get("text", "")).strip():
            kps.append({"text": str(kp["text"]).strip(), "weight": max(0.0, _num(kp.get("weight"), 0.0))})
    if kps:
        return kps
    model_answer = str(_field(question, "model_answer") or "").strip()
    if model_answer:
        return [{"text": model_answer, "weight": max(0.0, _num(_field(question, "max_marks"), 0.0))}]
    return []


def verify_quote(answer_text: str, quote: Any) -> str | None:  # noqa: ANN401
    """The span of the answer a grader quoted, or None when it is not in the answer. The
    local graders and llm_grader already return spans of the answer (kept unchanged);
    anything else goes through llm_grader.find_quote."""
    if not quote or not answer_text:
        return None
    quote = str(quote)
    if not any(c.isalnum() for c in quote):
        return None
    if quote in answer_text:
        return quote
    return find_quote(answer_text, quote)


def _clip(text: str, limit: int) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= limit:
        return s
    cut = s[: limit - 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut).rstrip(" ,;:.") + "…"


# ============================================================================ results
@dataclass(frozen=True)
class AiInfo:
    """What happened with the AI marker for one answer. Secret- and text-free.

    status: off (AI disabled or no consent) | used | fallback (enabled but no AI
    verdict: the offline checker's marks stand) | not_applicable (maths, no rubric,
    blank: never sent). ``reason``/``details`` say why a fallback happened; reason
    "not_graded" means the AI replied but left this question out."""

    status: str = "off"
    reason: str = ""
    details: tuple[str, ...] = ()
    style: str | None = None

    def as_dict(self) -> dict:
        out: dict[str, Any] = {"status": self.status}
        if self.reason:
            out["reason"] = self.reason
        if self.details:
            out["details"] = list(self.details)
        if self.style:
            out["style"] = self.style
        return out


@dataclass
class EnsembleKeyPoint:
    index: int
    text: str
    weight: float
    status: str
    awarded: float
    evidence_sentence: str | None
    reason: str  # engine reason code, the AI's short reason, no_verified_evidence or not_graded
    source: str  # the grader whose status stands
    disputed: bool
    statuses_by_grader: dict[str, str]  # "not_judged" for a key point the AI did not judge
    entailment: float = 0.0  # from the first local grader (the AI gives no probabilities)
    contradiction: float = 0.0

    def as_dict(self) -> dict:
        """Answer.evidence entry: the engine's KeyPointResult.as_dict() keys plus the ensemble's."""
        return {
            "index": self.index, "text": self.text, "weight": self.weight,
            "entailment_prob": round(self.entailment, 4), "contradiction_prob": round(self.contradiction, 4),
            "evidence_sentence": self.evidence_sentence, "awarded": round(self.awarded, 2),
            "matched": self.status == "matched", "status": self.status, "reason": self.reason,
            "source": self.source, "disputed": self.disputed, "statuses_by_grader": dict(self.statuses_by_grader),
        }


@dataclass
class EnsembleResult:
    marks: float
    max_marks: float
    verdict: str  # correct | partially_correct | incorrect | ungraded
    key_points: list[EnsembleKeyPoint]
    flags: list[str]
    grading_confidence: float
    graders_used: list[str]  # primary first
    feedback: str
    trace: dict = field(default_factory=dict)

    @property
    def grader(self) -> str:
        """Short label for Answer.grader (a String(32) column): the graders, primary first."""
        return "+".join(self.graders_used)[:32]

    def as_dict(self) -> dict:
        return {"marks": self.marks, "max_marks": self.max_marks, "verdict": self.verdict,
                "key_points": [kp.as_dict() for kp in self.key_points], "flags": list(self.flags),
                "grading_confidence": self.grading_confidence, "graders_used": list(self.graders_used),
                "grader": self.grader, "feedback": self.feedback, "trace": self.trace}


# ============================================================================ views
def is_llm_result(result: Any) -> bool:  # noqa: ANN401
    return _field(result, "provider") is not None or _field(result, "prompt_version") is not None


@dataclass(frozen=True)
class _View:
    """One grader's result read per canonical key point."""

    name: str
    is_llm: bool
    statuses: tuple[str, ...]  # "missed" placeholder where not judged
    judged: tuple[bool, ...]
    evidence: tuple[str | None, ...]
    reasons: tuple[str, ...]
    entail: tuple[float, ...]
    contra: tuple[float, ...]
    flags: tuple[str, ...]
    marks: float | None
    confidence: float | None
    feedback: str
    meta: dict

    def shown(self, i: int) -> str:
        return self.statuses[i] if self.judged[i] else NOT_JUDGED


def _view(result: Any, n_kp: int, name: str | None = None) -> _View:  # noqa: ANN401, C901
    llm = is_llm_result(result)
    name = name or (LLM if llm else str(_field(result, "grader") or "local"))
    flags = tuple(str(f) for f in (_field(result, "flags") or []))
    by_index: dict[int, Any] = {}
    for pos, kp in enumerate(_field(result, "key_points") or []):
        try:
            idx = int(_field(kp, "index", pos))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_kp and idx not in by_index:
            by_index[idx] = kp
    statuses: list[str] = []
    judged: list[bool] = []
    evidence: list[str | None] = []
    reasons: list[str] = []
    ent: list[float] = []
    con: list[float] = []
    for i in range(n_kp):
        kp = by_index.get(i)
        st, ev, reason, e, c, ok = "missed", None, "not_graded", 0.0, 0.0, False
        if kp is not None:
            st = str(_field(kp, "status") or "").strip().lower()
            reason = str(_field(kp, "reason") or "")
            ev = _field(kp, "evidence_sentence")
            e = 0.0 if llm else _num(_field(kp, "entailment", _field(kp, "entailment_prob", 0.0)))
            c = 0.0 if llm else _num(_field(kp, "contradiction", _field(kp, "contradiction_prob", 0.0)))
            ok = True
            if _field(kp, "judged") is False or st in _NOT_JUDGED_STATUSES:
                st, ok = "missed", False
            elif llm and FLAG_AI_INCOMPLETE in flags and st == "missed" and not reason.strip() and not ev:
                ok = False  # llm_grader's placeholder for a key point the AI left out
            elif st not in STATUSES:
                st, reason = "missed", "unknown_status"
            if not ok:
                ev, reason = None, "not_graded"
        statuses.append(st)
        judged.append(ok)
        evidence.append(str(ev) if ev else None)
        reasons.append(reason)
        ent.append(e)
        con.append(c)
    marks = _field(result, "marks")
    conf = _field(result, "grading_confidence")
    if llm:
        meta = {k: _field(result, k) for k in ("provider", "model", "cached", "prompt_version", "uncertain")}
    else:
        trace = _field(result, "trace") or {}
        meta = {"prior": trace.get("prior") if isinstance(trace, Mapping) else None,
                "thresholds": _numbers(trace.get("thresholds")) if isinstance(trace, Mapping) else None}
    return _View(name=name, is_llm=llm, statuses=tuple(statuses), judged=tuple(judged), evidence=tuple(evidence),
                 reasons=tuple(reasons), entail=tuple(ent), contra=tuple(con), flags=flags,
                 marks=None if marks is None else _num(marks), confidence=None if conf is None else _num(conf),
                 feedback=str(_field(result, "feedback") or "") if llm else "", meta=meta)


def _numbers(d: Any) -> dict | None:  # noqa: ANN401
    if not isinstance(d, Mapping):
        return None
    return {str(k): v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def llm_usable(question: Any, llm: Any) -> bool:  # noqa: ANN401
    """An AI result can decide (be the primary) when it judged at least one key point of
    the question; key points it did not judge fall back to the offline graders."""
    if llm is None:
        return False
    n = len(question_key_points(question))
    return n > 0 and any(_view(llm, n).judged)


def assign_roles(question: Any, local_primary: Any, local_secondary: Any = None,  # noqa: ANN401
                 llm: Any = None) -> tuple[Any, Any, list[Any]]:
    """(primary, secondary, extras) per the module doc."""
    if llm_usable(question, llm):
        return llm, local_primary, [r for r in (local_secondary,) if r is not None]
    return local_primary, local_secondary, [r for r in (llm,) if r is not None]


# ============================================================================ combine
def _marks(statuses: Sequence[str], kps: Sequence[dict], max_marks: float) -> float:
    raw = sum(kp["weight"] * CREDIT[st] for st, kp in zip(statuses, kps))
    return round(min(max(raw, 0.0), max_marks), 2)


def _claim(v: _View | None, i: int, text: str) -> tuple[str | None, str | None]:
    """(status, verified quote) of one grader for key point i. A claim the grader cannot
    back with a quote of the answer counts as "missed"; None when there is no grader."""
    if v is None:
        return None, None
    st = v.statuses[i]
    if st == "missed" or not v.judged[i]:
        return "missed", None
    quote = verify_quote(text, v.evidence[i])
    return (st, quote) if quote else ("missed", None)


def feedback_template(key_points: Iterable[Any]) -> str:  # noqa: ANN401
    """Deterministic summary of the criteria by final status."""
    kps = list(key_points)
    parts = []
    for label, status in (("Covered", "matched"), ("Partly", "partial"), ("Missing", "missed"),
                          ("Contradicts", "contradicted")):
        items = [_clip(str(_field(kp, "text") or ""), FEEDBACK_ITEM_CHARS) for kp in kps
                 if _field(kp, "status") == status]
        if items:
            parts.append(f"{label}: " + ", ".join(f'"{t}"' for t in items))
    return _clip("; ".join(parts) + ("." if parts else ""), FEEDBACK_MAX_CHARS)


def _ai_trace(ai: AiInfo | None, views: Sequence[_View], primary: _View) -> dict:
    llm = next((v for v in views if v.is_llm), None)
    out = (ai or AiInfo("used" if llm is not None else "off")).as_dict()
    if llm is not None:
        out.update({k: v for k, v in llm.meta.items() if v is not None})
        out["judged"] = sum(llm.judged)
        if llm is not primary:
            out["status"] = "incomplete"  # judged nothing usable: the offline graders decided
    return out


def _grader_summary(v: _View) -> dict:
    out: dict[str, Any] = {"marks": v.marks, "flags": list(v.flags),
                           "statuses": [v.shown(i) for i in range(len(v.statuses))]}
    if v.is_llm:
        out.update({k: val for k, val in v.meta.items() if val is not None})
    else:
        out.update(grading_confidence=v.confidence, reasons=[_code(r) for r in v.reasons],
                   entail=[round(x, 4) for x in v.entail], contra=[round(x, 4) for x in v.contra])
        out.update({k: val for k, val in v.meta.items() if val is not None})
    return out


def combine(question: Any, answer_text: str, primary: Any, secondary: Any = None,  # noqa: ANN401, C901, PLR0912, PLR0915
            extras: Sequence[Any] = (), *, ai: AiInfo | None = None) -> EnsembleResult:
    """Combine graders' per-key-point statuses into one result (rules in the module doc).
    ``primary``/``secondary``/``extras`` are GradeResult or LLMGrade objects (or their
    as_dict() forms); ``ai`` records what happened with the AI marker."""
    text = answer_text or ""
    kps = question_key_points(question)
    n = len(kps)
    max_marks = max(0.0, _num(_field(question, "max_marks"), 0.0))
    qno = _field(question, "question_no", _field(question, "id"))

    seen: dict[str, int] = {}

    def view(r: Any) -> _View | None:  # noqa: ANN401
        if r is None:
            return None
        v = _view(r, n)
        seen[v.name] = seen.get(v.name, 0) + 1
        return v if seen[v.name] == 1 else _view(r, n, name=f"{v.name}#{seen[v.name]}")  # same grader twice

    P = view(primary)
    if P is None:
        raise ValueError("combine() needs a primary result")
    S = view(secondary)
    X = [v for v in (view(r) for r in extras) if v is not None]
    present = [v for v in (P, S, *X) if v is not None]
    graders_used = [v.name for v in present]
    ai_trace = _ai_trace(ai, present, P)
    base_trace: dict[str, Any] = {
        "version": VERSION, "question": qno, "graders_used": graders_used, "primary": P.name,
        "secondary": S.name if S is not None else None, "extras": [v.name for v in X], "ai": ai_trace,
    }

    # --- maths / no rubric: nothing to combine (always routed to the teacher by the primary's flags)
    if n == 0 or _field(question, "type") == "maths" or str(_field(primary, "verdict") or "") == "ungraded":
        flags = list(dict.fromkeys(list(P.flags) + [f for v in present[1:] for f in v.flags if f in CARRY_FROM_ANY]))
        conf = round(min(1.0, max(0.0, P.confidence if P.confidence is not None else 0.0)), 4)
        trace = {**base_trace, "graded": True, "ungraded": True, "marks": 0.0, "max_marks": max_marks,
                 "verdict": "ungraded", "grading_confidence": conf, "flags": flags, "key_points": [],
                 "graders": {v.name: _grader_summary(v) for v in present}}
        return EnsembleResult(0.0, max_marks, "ungraded", [], flags, conf, graders_used, "", trace)

    local_extra = next((v for v in X if not v.is_llm), None)
    final: list[EnsembleKeyPoint] = []
    kp_trace: list[dict] = []
    disputed: list[int] = []
    minor: list[int] = []
    downgraded: list[int] = []
    ai_fallback: list[int] = []
    uncrossed = False  # some key point had no cross-checking grader
    contradiction_found = False
    for i, kp in enumerate(kps):
        decider, checker = P, S
        if P.is_llm and not P.judged[i]:  # the AI did not judge it: the offline graders decide
            decider, checker = S, local_extra
            ai_fallback.append(i)
        uncrossed = uncrossed or checker is None

        status, evidence, ev_from, reason = "missed", None, None, "not_graded"
        if decider is not None:
            status, reason = decider.statuses[i], decider.reasons[i]
            d_ev = verify_quote(text, decider.evidence[i]) if status != "missed" else None
            c_st, c_ev = _claim(checker, i, text)
            if status != "missed":
                if d_ev:
                    evidence, ev_from = d_ev, decider.name
                elif status in AWARD and c_st in AWARD:
                    evidence, ev_from = c_ev, checker.name  # the award stands on the cross-checker's quote
                else:
                    status, reason = "missed", FLAG_NO_EVIDENCE
                    downgraded.append(i)
        else:
            c_st = None

        contradiction_by = []
        for v in present:
            if v.judged[i] and v.statuses[i] == "contradicted" and verify_quote(text, v.evidence[i]):
                contradiction_by.append(v.name)
        contradiction_found = contradiction_found or bool(contradiction_by)

        is_disputed = is_minor = False
        if c_st is not None:
            if (status in AWARD) != (c_st in AWARD):
                is_disputed = True
            elif status in AWARD and status != c_st:
                is_minor = True
        if contradiction_by and status != "contradicted":
            is_disputed = True
        if is_disputed:
            disputed.append(i)
        elif is_minor:
            minor.append(i)

        source = decider.name if decider is not None else P.name
        local = next((v for v in present if not v.is_llm), None)
        by_grader = {v.name: v.shown(i) for v in present}
        awarded = kp["weight"] * CREDIT[status]
        final.append(EnsembleKeyPoint(
            index=i, text=kp["text"], weight=kp["weight"], status=status, awarded=awarded,
            evidence_sentence=evidence if status != "missed" else None, reason=reason, source=source,
            disputed=is_disputed, statuses_by_grader=by_grader,
            entailment=local.entail[i] if local is not None else 0.0,
            contradiction=local.contra[i] if local is not None else 0.0,
        ))
        trace_reason = FLAG_NO_EVIDENCE if i in downgraded else \
            ("ai_judgement" if decider is not None and decider.is_llm else _code(reason))
        kp_trace.append({
            "index": i, "weight": kp["weight"], "status": status, "awarded": round(awarded, 2), "source": source,
            "evidence_from": ev_from if status != "missed" else None, "reason": trace_reason,
            "disputed": is_disputed, "minor_disagreement": is_minor, "ai_fallback": i in ai_fallback,
            "contradiction_by": contradiction_by, "statuses_by_grader": by_grader,
        })

    marks = _marks([k.status for k in final], kps, max_marks)
    verdict = verdict_for(marks, max_marks)
    s_marks = _marks([_claim(S, i, text)[0] for i in range(n)], kps, max_marks) if S is not None else None
    gap = round(abs(marks - s_marks), 2) if s_marks is not None else None
    gap_exceeded = gap is not None and gap > MARKS_GAP_LIMIT + 1e-9
    disagree = bool(disputed) or gap_exceeded

    flags = [f for f in P.flags if f not in DERIVED_FLAGS]
    for v in present[1:]:
        flags.extend(f for f in v.flags if f in CARRY_FROM_ANY or (v.is_llm and f in AI_FLAGS))
    ai_skipped = ai is not None and ai.status == "fallback" and ai.reason == "not_graded"
    if ai_fallback or ai_skipped or any(v.is_llm for v in X):
        flags.append(FLAG_AI_INCOMPLETE)  # the AI left this (part of the) question to the offline graders
    if downgraded:
        flags.append(FLAG_NO_EVIDENCE)
    if contradiction_found or any(k.status == "contradicted" for k in final):
        flags.append(FLAG_CONTRADICTION)
    if disagree:
        flags.append(FLAG_DISAGREE)
    if any(k.status == "partial" for k in final):
        flags.append(FLAG_PARTIAL)
    if ai is not None and ai.status == "fallback" and not ai_skipped:
        flags.append(FLAG_AI_FALLBACK)
    flags = list(dict.fromkeys(flags))

    prior = PRIORS.get(P.name, DEFAULT_PRIOR)
    if ai_fallback and S is not None:
        prior = min(prior, PRIORS.get(S.name, DEFAULT_PRIOR))  # mixed deciders: the weaker one counts
    factor = DISAGREE_FACTOR if disagree else (NO_SECONDARY_FACTOR if uncrossed else AGREE_FACTOR)
    conf = round(min(1.0, max(0.0, prior * factor)), 4)

    replaced = []
    if P.is_llm and P.feedback.strip():
        for cond, code in ((downgraded, "evidence_withdrawn"), (disputed, "disputed"), (gap_exceeded, "marks_gap"),
                           (ai_fallback, "ai_incomplete"),
                           (AI_FEEDBACK_BLOCKERS & set(P.flags), "ai_verdicts_changed")):
            if cond:
                replaced.append(code)
    use_ai_feedback = P.is_llm and bool(P.feedback.strip()) and not replaced
    feedback = P.feedback.strip() if use_ai_feedback else feedback_template(final)

    trace = {
        **base_trace, "graded": True, "prior": prior, "agreement_factor": factor, "grading_confidence": conf,
        "marks": marks, "max_marks": max_marks, "verdict": verdict,
        "secondary_marks": s_marks, "marks_gap": gap, "marks_gap_limit": MARKS_GAP_LIMIT,
        "marks_gap_exceeded": gap_exceeded, "disputed": disputed, "minor_disagreements": minor,
        "downgraded": downgraded, "ai_fallback": ai_fallback,
        "feedback_source": "ai" if use_ai_feedback else "template", "ai_feedback_replaced": replaced,
        "flags": flags, "key_points": kp_trace, "graders": {v.name: _grader_summary(v) for v in present},
    }
    return EnsembleResult(marks, max_marks, verdict, final, flags, conf, graders_used, feedback, trace)


# ============================================================================ routing (shared)
@dataclass(frozen=True)
class Routed:
    status: str  # auto_accepted | needs_review
    flags: list[str]
    confidence: float
    review_flags: list[str]
    forced: bool  # routing would have auto-accepted, but a FORCE_REVIEW_FLAGS flag is present

    def as_trace(self, ocr_confidence: float, segmentation_confidence: float, auto_accept: float) -> dict:
        return {"status": self.status, "confidence": self.confidence, "ocr_confidence": ocr_confidence,
                "segmentation_confidence": segmentation_confidence, "auto_accept_threshold": auto_accept,
                "review_flags": list(self.review_flags), "forced_review": self.forced}


def route_graded(grading_confidence: float, flags: Iterable[str], text: str, settings: Any, *,  # noqa: ANN401
                 ocr_confidence: float, segmentation_confidence: float,
                 needs_transcription: bool = False) -> Routed:
    """How the pipeline routes a graded answer (processor.compute_answer and the
    evaluation harness both call this): confidence = routing.combine(OCR, grading,
    segmentation, T); a partly unread answer and a very short one (< 3 characters) get
    their flags; routing.route decides; FORCE_REVIEW_FLAGS are never auto-accepted."""
    from ..pipeline import routing

    conf = routing.combine(ocr_confidence, grading_confidence, segmentation_confidence,
                           settings.calibration_temperature)
    extra = []
    if needs_transcription:  # part of the answer could not be read: graded on partial text
        extra.append("needs_transcription")
    if len((text or "").strip()) < 3:  # very short answers are easy to misread and to mis-grade
        extra.append("short_answer")
    status, out = routing.route(conf, ocr_confidence, [*flags, *extra], auto_accept=settings.auto_accept_threshold,
                                low_ocr=settings.low_ocr_confidence)
    forced = status == "auto_accepted" and bool(FORCE_REVIEW_FLAGS & set(out))
    if forced:
        status = "needs_review"
    review = [f for f in out if routing.is_review_flag(f) or f in FORCE_REVIEW_FLAGS]
    return Routed(status, out, conf, review, forced)


# ============================================================================ local graders (shared)
@dataclass(frozen=True)
class LocalPlan:
    primary: str  # scorer kind: lexical | nli-onnx | nli | sbert
    secondary: str | None


_semantic_cache: dict[str, str] = {}
_cache_lock = threading.Lock()
_warned: set[str] = set()


def reset_caches() -> None:
    """Forget the resolved 'auto' grader and the one-time warnings (tests)."""
    with _cache_lock:
        _semantic_cache.clear()
        _warned.clear()


def semantic_kind(settings: Any) -> str | None:  # noqa: ANN401
    """The semantic local grader selected by settings.grader (``auto`` = what
    get_scorer("auto") picks: torch NLI -> nli-onnx -> SBERT -> lexical), or None when
    only the lexical grader is available or selected."""
    kind = str(getattr(settings, "grader", "auto") or "auto").strip().lower()
    if kind == "auto":
        with _cache_lock:
            if "auto" not in _semantic_cache:
                from .scorers import get_scorer

                try:
                    _semantic_cache["auto"] = str(get_scorer("auto", settings.sbert_model, settings.nli_model).name)
                except Exception as exc:  # noqa: BLE001 - nothing semantic installed: lexical only
                    log.warning("grader=auto could not pick a semantic grader (%s); using lexical",
                                type(exc).__name__)
                    _semantic_cache["auto"] = LEXICAL
            kind = _semantic_cache["auto"]
    return kind if kind in SEMANTIC else None


def local_plan(settings: Any) -> LocalPlan:  # noqa: ANN401
    """Which local graders run, from settings.local_primary and settings.grader."""
    choice = str(getattr(settings, "local_primary", "auto") or "auto").strip().lower()
    sem = semantic_kind(settings)
    if choice == LEXICAL:
        return LocalPlan(LEXICAL, sem)
    if choice == "auto":
        return LocalPlan(sem, LEXICAL) if sem else LocalPlan(LEXICAL, None)
    return LocalPlan(choice, LEXICAL)  # an explicit semantic grader, e.g. nli-onnx


_THRESHOLD_SETTINGS = (("tau", "tau"), ("tau_low", "tau_low"), ("contradiction", "contradiction_threshold"))


def thresholds_for(settings: Any, scorer_name: str):  # noqa: ANN201, ANN401
    """Key-point thresholds for one scorer: the engine's per-scorer defaults
    (engine.default_thresholds), except the Settings values an operator changed from
    the generic defaults (SARAS_TAU, SARAS_TAU_LOW, SARAS_CONTRADICTION_THRESHOLD),
    which apply to every scorer, field by field. (Before the ensemble, the pipeline
    passed the generic 0.6 / 0.4 / 0.5 to every scorer, wrong for NLI probabilities.)"""
    from . import engine

    base = engine.Thresholds()
    fn = getattr(engine, "default_thresholds", None)
    th = fn(scorer_name) if callable(fn) else engine.Thresholds()
    for attr, key in _THRESHOLD_SETTINGS:
        value = getattr(settings, key, None)
        if value is None:
            continue
        if not callable(fn) or float(value) != getattr(base, attr):
            setattr(th, attr, float(value))
    return th


def _warn_once(key: str, msg: str, *args: Any) -> None:  # noqa: ANN401
    with _cache_lock:
        if key in _warned:
            return
        _warned.add(key)
    log.warning(msg, *args)


def grade_local_primary(question: Any, answer_text: str, settings: Any, *, attempts: int = 3,  # noqa: ANN401
                        sleep: Callable[[float], None] = time.sleep):  # noqa: ANN201
    """Grade with the local primary (scorer instances are cached per process by
    get_scorer, so a model loads once). A grader that cannot be loaded falls back to
    lexical at once; one that fails while grading is retried with backoff first
    (PRD 18.3). A fallback result carries the flag graded_with_fallback."""
    from .engine import grade_answer
    from .scorers import LexicalScorer, get_scorer

    kind = local_plan(settings).primary
    try:
        scorer = get_scorer(kind, settings.sbert_model, settings.nli_model)
    except Exception as exc:  # noqa: BLE001 - not installed / model missing: not transient
        _warn_once(f"load:{kind}", "Local grader %s unavailable (%s); grading with the lexical fallback",
                   kind, type(exc).__name__)
        scorer = None
    last: Exception | None = None
    if scorer is not None:
        for attempt in range(attempts):
            try:
                return grade_answer(question, answer_text, scorer, thresholds_for(settings, scorer.name))
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("Local grader %s failed (attempt %d of %d): %s", kind, attempt + 1, attempts,
                            type(exc).__name__)
                if attempt + 1 < attempts:
                    sleep(0.2 * (2 ** attempt))
        log.error("Local grader %s failed %d times (%s); using the lexical fallback", kind, attempts,
                  type(last).__name__ if last else "?")
    result = grade_answer(question, answer_text, LexicalScorer(), thresholds_for(settings, LEXICAL))
    if kind != LEXICAL:
        result.flags.append(FLAG_FALLBACK)
    return result


def grade_local_secondary(question: Any, answer_text: str, settings: Any, primary_result: Any):  # noqa: ANN201, ANN401
    """The local cross-check grader's result, or None (no secondary configured, the same
    grader as the primary, or it failed - a secondary never fails a sheet)."""
    from .engine import grade_answer
    from .scorers import get_scorer

    kind = local_plan(settings).secondary
    if kind is None or kind == str(_field(primary_result, "grader") or ""):
        return None
    try:
        scorer = get_scorer(kind, settings.sbert_model, settings.nli_model)
        return grade_answer(question, answer_text, scorer, thresholds_for(settings, scorer.name))
    except Exception as exc:  # noqa: BLE001
        _warn_once(f"secondary:{kind}", "Secondary grader %s failed (%s); continuing without the cross-check",
                   kind, type(exc).__name__)
        return None


def grade_local(question: Any, answer_text: str, settings: Any,  # noqa: ANN401
                primary_fn: Callable[..., Any] | None = None) -> tuple[Any, Any]:
    """(local primary result, local secondary result or None) for one answer."""
    fn = primary_fn or grade_local_primary
    local_p = fn(question, answer_text, settings)
    return local_p, grade_local_secondary(question, answer_text, settings, local_p)


def grade_ensemble(question: Any, answer_text: str, settings: Any, *, llm_grade: Any = None,  # noqa: ANN201, ANN401
                   ai: AiInfo | None = None, primary_fn: Callable[..., Any] | None = None,
                   local_results: tuple[Any, Any] | None = None) -> EnsembleResult:
    """Grade one answer the way the pipeline does: the local primary (``primary_fn``,
    default grade_local_primary) and secondary, an optional AI result, then combine().
    `local_results`: (primary, secondary) already computed by grade_local (the pipeline
    runs the local checkers while the AI request is in flight)."""
    local_p, local_s = local_results if local_results is not None else \
        grade_local(question, answer_text, settings, primary_fn)
    p, s, x = assign_roles(question, local_p, local_s, llm_grade)
    return combine(question, answer_text, p, s, x, ai=ai)


# ============================================================================ AI marker (shared)
def ai_enabled(settings: Any) -> bool:  # noqa: ANN401
    """AI marking is on: SARAS_AI_GRADING=ensemble and SARAS_AI_CONSENT=true."""
    return getattr(settings, "ai_grading", "off") == "ensemble" and bool(getattr(settings, "ai_consent", False))


def ai_style(settings: Any) -> str:  # noqa: ANN401
    style = str(getattr(settings, "ai_grading_style", "moderate") or "moderate").strip().lower()
    return style if style in STYLES else "moderate"


def ai_applicable(question: Any, answer_text: str | None) -> bool:  # noqa: ANN401
    """Whether llm_grader would send this answer: not maths, has a rubric, has content."""
    return _field(question, "type") != "maths" and bool(question_key_points(question)) \
        and has_content(answer_text)


@dataclass
class AiOutcome:
    info: AiInfo
    grade: Any = None  # llm_grader.LLMGrade when the AI judged the answer


_DETAIL_PROVIDER = re.compile(r"^[A-Za-z0-9_.:/-]{1,80}$")


def ai_failure_details(reasons: Iterable[str]) -> tuple[str, ...]:
    """'provider: status' per provider tried (router reasons are secret-free; free text
    from providers is dropped here, only our own skip reasons are kept, shortened)."""
    out = []
    for r in reasons:
        parts = str(r).split(": ", 2)
        if len(parts) < 2 or not _DETAIL_PROVIDER.match(parts[0]):
            out.append(_clip(str(r), 60) if str(r).startswith("AI consent") else "unavailable")
            continue
        word = (parts[1].split() or ["unavailable"])[0].rstrip(":")
        word = word if _CODE.match(word) else "unavailable"
        detail = f"{parts[0]}: {word}"
        if word == "skipped" and len(parts) > 2:
            detail += f" ({_clip(parts[2], 50)})"
        out.append(detail)
    return tuple(dict.fromkeys(out))


def ai_grade_items(router: Any, answers: Sequence[tuple[Any, str]], settings: Any, *,  # noqa: ANN401
                   exam_id: str | None = None, submission_id: str | None = None,
                   redact_terms: Iterable[str | None] = ()) -> dict[int, AiOutcome]:
    """AI-mark these answers of one sheet in ONE request (llm_grader splits only when the
    prompt is too big). Returns an outcome for every question_no: used, fallback (with
    the reason) or not_applicable. Never raises: the offline checker is the fallback."""
    from ..llm.router import LLMUnavailable

    style = ai_style(settings)
    out: dict[int, AiOutcome] = {}
    items: list[AnswerItem] = []
    for question, text in answers:
        qno = int(_field(question, "question_no"))
        if ai_applicable(question, text):
            items.append(AnswerItem.from_question(question, text))
        else:
            out[qno] = AiOutcome(AiInfo("not_applicable", style=style))
    if not items:
        return out

    def fallback(reason: str, details: tuple[str, ...] = ()) -> dict[int, AiOutcome]:
        for it in items:
            out[it.question_no] = AiOutcome(AiInfo("fallback", reason, details, style))
        return out

    try:
        usable = router.usable_providers()
    except Exception as exc:  # noqa: BLE001
        log.warning("AI provider check failed (%s); using the offline checker", type(exc).__name__)
        return fallback("error")
    if not usable:
        return fallback("no_usable_provider")
    try:
        grades = grade_sheet_llm(router, items, style, exam_id=exam_id, submission_id=submission_id,
                                 redact_terms=list(redact_terms))
    except LLMUnavailable as exc:
        return fallback("unavailable", ai_failure_details(exc.reasons))
    except Exception as exc:  # noqa: BLE001 - AI is never in the critical path (PRD 13.5)
        log.warning("AI marking failed unexpectedly (%s); using the offline checker", type(exc).__name__)
        return fallback("error")
    for it in items:
        grade = grades.get(it.question_no)
        out[it.question_no] = AiOutcome(AiInfo("used", style=style), grade) if grade is not None else \
            AiOutcome(AiInfo("fallback", "not_graded", style=style))  # left out of the AI's reply
    return out
