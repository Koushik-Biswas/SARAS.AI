"""Rubric grading: key-point matching, partial credit, contradictions, role swaps,
gaming guards and grading confidence (PRD 12.4, 12.6, 12.11).

One marking path for every gradable question type (short, long, paragraph):

* the answer is cut into premise *windows* - single sentences, adjacent sentence
  pairs, and the whole answer when it has three sentences - whose text always occurs
  literally in the answer, so every evidence quote can be verified;
* the scorer gives entailment / contradiction for every (window, key point). Lexical
  evidence (IDF-weighted term overlap with polarity checks) is always computed; a
  meaning-aware scorer (NLI) is combined with it as a union - NLI is precise but
  misses paraphrases it cannot entail, so it never vetoes a lexical match;
* per key point: contradicted (0) > matched (full weight) > partial (half) > missed (0);
  rubric keywords are required for "matched" (any one of them); a calculation key
  point is judged on its result and unit ("F = 5 x 2 = 10 N"), a formula key point on
  the formula; the key point's numbers must appear; and a clause that gives this key
  point's facts to a parallel key point's entity ("veins carry blood away from the
  heart") is a role swap;
* marks = sum of the awards, capped at max marks, 2 dp - no rounding to 0.5;
* every awarded mark carries its criterion (key point) and a verified evidence quote;
  an award without verifiable evidence is withdrawn;
* grading confidence = scorer prior x decision margin, and uncertain or suspicious
  answers get review flags (possible_paraphrase, off_topic, role_swap, ...).

Maths stays manual review. Nothing here logs answer text.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from . import text as T
from .scorers import LexicalScorer, PairScorer

log = logging.getLogger("app.grading")


@dataclass
class Thresholds:
    tau: float = 0.6
    tau_low: float = 0.4
    contradiction: float = 0.5


# Per-scorer default thresholds. Lexical values are the historical defaults.
# NLI values were measured on the DEV benchmark (42 answers / 148 key points,
# synthetic, not teacher-validated) by the ONNX model evaluation and are
# PROVISIONAL: tau 0.5 on the mean entailment over a key point's sub-claims; NLI
# carries no partial-credit signal, so tau_low is nominal (partial credit comes from
# sub-claim coverage and lexical evidence); an NLI contradiction counts only at
# >= 0.95, in a window that shares >= 30% of the sub-claim's words and where
# contradiction beats entailment.
DEFAULT_THRESHOLDS: dict[str, Thresholds] = {
    "lexical": Thresholds(tau=0.6, tau_low=0.4, contradiction=0.5),
    "sbert": Thresholds(tau=0.6, tau_low=0.4, contradiction=0.5),
    "nli": Thresholds(tau=0.5, tau_low=0.15, contradiction=0.95),
    "nli-onnx": Thresholds(tau=0.5, tau_low=0.15, contradiction=0.95),
}
NLI_SCORERS = frozenset({"nli", "nli-onnx"})
NLI_CONTRA_MIN_OVERLAP = 0.30  # NLI contradiction gate: word overlap of window and sub-claim
LEXICAL_THRESHOLDS = DEFAULT_THRESHOLDS["lexical"]

# How far a scorer's verdicts can be trusted at best (grading confidence = prior x
# decision margin). PROVISIONAL priors, to be calibrated against teacher marks
# (R3): word overlap cannot see meaning, so the lexical grader alone never reaches
# the default auto-accept threshold (0.80); its answers are always reviewed.
SCORER_PRIORS: dict[str, float] = {"lexical": 0.6, "sbert": 0.7, "nli": 0.8, "nli-onnx": 0.8}
DEFAULT_PRIOR = 0.5

# Multi-sentence windows are discounted for the lexical evidence: words scattered over
# two sentences are weaker evidence than one sentence stating the key point.
LEXICAL_WINDOW_DISCOUNT = 0.9
NON_PROPOSITION_FACTOR = 0.3  # keyword lists / fragments count only 0.3 as evidence (PRD 12.11)
MARGIN_SCALE = 0.2  # a score this far from every decision threshold is a clear decision
RESTATEMENT_NOVELTY = 0.25  # a sentence with fewer new words than this only restates the question
KEYWORD_PARTIAL_MIN_SUPPORT = 0.25  # lexical support needed before a keyword earns half credit

FLAG_CONTRADICTION = "contradiction"
FLAG_ROLE_SWAP = "role_swap"
FLAG_PARTIAL = "partial_credit"
FLAG_NUMERIC = "numeric_mismatch"
FLAG_KEYWORD_LIST = "keyword_list"
FLAG_COPIED = "copied_question"
FLAG_OFF_TOPIC = "off_topic"
FLAG_PARAPHRASE = "possible_paraphrase"
# The grader cannot judge meaning well enough to be trusted without a teacher (its
# prior is below the default auto-accept threshold): every answer it grades is
# reviewed, and this flag tells the review screen why.
FLAG_LOW_RELIABILITY = "low_reliability_grader"
LOW_RELIABILITY_PRIOR = 0.8

_RANK = {"missed": 0, "partial": 1, "matched": 2}
_STATUS = {v: k for k, v in _RANK.items()}


def default_thresholds(scorer_name: str) -> Thresholds:
    return replace(DEFAULT_THRESHOLDS.get(scorer_name, Thresholds()))


def resolve_thresholds(scorer_name: str, th: Thresholds | None) -> Thresholds:
    """The thresholds to grade with. ``None`` - or the generic lexical defaults
    (0.6 / 0.4 / 0.5, what an unconfigured Settings passes) for an NLI scorer, whose
    probabilities live on another scale - means the scorer's own defaults."""
    if th is None or (scorer_name in NLI_SCORERS and th == Thresholds()):
        return default_thresholds(scorer_name)
    return th


@dataclass
class KeyPointResult:
    index: int
    text: str
    weight: float
    entailment: float
    contradiction: float
    evidence_sentence: str | None
    awarded: float
    status: str  # matched | partial | missed | contradicted
    reason: str = ""  # entailed | partial_support | sub_claims | no_support | contradiction | role_swap | ...

    def as_dict(self) -> dict:
        return {
            "index": self.index, "text": self.text, "weight": self.weight,
            "entailment_prob": round(self.entailment, 4), "contradiction_prob": round(self.contradiction, 4),
            "evidence_sentence": self.evidence_sentence, "awarded": round(self.awarded, 2),
            "matched": self.status == "matched", "status": self.status, "reason": self.reason,
        }


@dataclass
class GradeResult:
    marks: float
    max_marks: float
    grading_confidence: float
    key_points: list[KeyPointResult] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    grader: str = ""
    verdict: str = ""  # correct | partially_correct | incorrect | ungraded (maths / no rubric)
    trace: dict = field(default_factory=dict)  # decision trace: scores and reasons, never answer text

    def as_dict(self) -> dict:
        return {"marks": self.marks, "max_marks": self.max_marks, "grading_confidence": self.grading_confidence,
                "key_points": [kp.as_dict() for kp in self.key_points], "flags": list(self.flags),
                "grader": self.grader, "verdict": self.verdict, "trace": self.trace}


def verdict_for(marks: float, max_marks: float) -> str:
    if max_marks > 0 and marks >= max_marks - 1e-9:
        return "correct"
    if marks <= 1e-9:
        return "incorrect"
    return "partially_correct"


def _round_half(x: float) -> float:
    return math.floor(x * 2 + 0.5) / 2


def normalise_key_points(question) -> list[dict]:  # noqa: ANN001
    kps = [kp for kp in (question.key_points or []) if str(kp.get("text", "")).strip()]
    if kps:
        return kps
    if question.model_answer and question.model_answer.strip():
        return [{"text": question.model_answer, "weight": question.max_marks, "keywords": []}]
    return []


# ------------------------------------------------------------------ windows and weights
@dataclass(frozen=True)
class Window:
    start: int
    end: int
    sentences: tuple[int, ...]
    is_proposition: bool


def answer_windows(answer_text: str) -> tuple[list[tuple[int, int]], list[Window]]:
    """Sentence spans and premise windows: single sentences, adjacent pairs and, for
    an answer of three sentences, the whole answer. Single sentences come first so
    that ties go to the shortest evidence."""
    spans = T.sentence_spans(answer_text)
    props = [T.is_proposition(answer_text[s:e]) for s, e in spans]
    wins = [Window(s, e, (k,), props[k]) for k, (s, e) in enumerate(spans)]
    for k in range(len(spans) - 1):
        wins.append(Window(spans[k][0], spans[k + 1][1], (k, k + 1), props[k] and props[k + 1]))
    if len(spans) == 3:
        wins.append(Window(spans[0][0], spans[-1][1], (0, 1, 2), all(props)))
    return spans, wins


def term_weights(kp_texts: list[str], prompt: str | None) -> dict[str, float]:
    """In-question IDF: a term used by several key points (or by the question itself)
    says less about which key point a sentence supports."""
    docs = [set(T.content_stems(t)) for t in kp_texts]
    if prompt and prompt.strip():
        docs.append(set(T.content_stems(prompt)))
    n = len(docs)
    out: dict[str, float] = {}
    for d in docs[: len(kp_texts)]:
        for s in d:
            if s not in out:
                df = sum(1 for other in docs if s in other or any(T.stems_match(s, o) for o in other))
                out[s] = 1.0 + math.log(n / max(1, df))
    return out


def _overlap(hyp: str, text: str) -> float:
    """Unweighted share of the hypothesis' content words found in ``text``."""
    hs = list(dict.fromkeys(T.content_stems(hyp)))
    if not hs:
        return 0.0
    ts = T.content_stems(text)
    return sum(1 for h in hs if any(T.stems_match(h, t) for t in ts)) / len(hs)


def _novelty(sentence: str, prompt_stems: list[str]) -> float:
    """Share of a sentence's content words that are not in the question."""
    ss = T.content_stems(sentence)
    if not ss:
        return 0.0
    return sum(1 for s in ss if not any(T.stems_match(s, p) for p in prompt_stems)) / len(ss)


def _context_text(answer_text: str, spans: list[tuple[int, int]], win: Window) -> str:
    """The window plus one sentence on each side (where rubric keywords are looked for)."""
    a = max(0, win.sentences[0] - 1)
    b = min(len(spans) - 1, win.sentences[-1] + 1)
    return answer_text[spans[a][0]:spans[b][1]]


def _margin(x: float, *edges: float) -> float:
    return min(1.0, min(abs(x - e) for e in edges) / MARGIN_SCALE)


def _status(e: float, th: Thresholds) -> str:
    return "matched" if e >= th.tau else "partial" if e >= th.tau_low else "missed"


# ------------------------------------------------------------------ scoring
@dataclass
class _Evidence:
    """Per key point and source: support score, its window and the contradiction."""
    e: np.ndarray  # [n_kp, n_win] support (after window factors)
    c: np.ndarray  # [n_kp, n_win] contradiction (gated where needed)
    coverage: np.ndarray | None = None  # [n_kp] NLI: mean over sub-claims of their best support
    sub_best: list[list[float]] | None = None  # NLI: best support of each sub-claim


def _nli_evidence(scorer: PairScorer, texts: list[str], hyps: list[str], factor: np.ndarray,
                  th: Thresholds, attributed: np.ndarray) -> tuple[_Evidence, list[list[str]]]:
    """Key points are scored as their sub-claims (asides and e.g. lists dropped) and a
    key point's support is the mean of its sub-claims' best support.

    The NLI model (trained on SNLI/MNLI) also rates *parallel* statements as
    contradictions ("A queue follows FIFO" vs "A stack follows LIFO"), so an NLI
    contradiction of a sub-claim counts only in a window that
    * shares >= 30% of the sub-claim's words and contradiction beats entailment there,
    * is about this key point more than any other (``attributed``),
    * mentions one of the sub-claim's distinctive words (not used by the other key
      points or by its sibling sub-claims), when it has any, and
    * does not entail a sibling sub-claim of the same key point ("... whereas ...")."""
    subs = [[" ".join(u.split()) for u in T.kp_units(h)] for h in hyps]
    flat = list(dict.fromkeys(u for ss in subs for u in ss))
    premises = [T.ocr_normalise(t) for t in texts]
    es, cs = scorer.score(premises, flat)
    es, cs = np.asarray(es, dtype=float), np.asarray(cs, dtype=float)
    idx = {u: k for k, u in enumerate(flat)}
    n_kp, n_win = len(hyps), len(texts)
    e = np.zeros((n_kp, n_win))
    c = np.zeros((n_kp, n_win))
    coverage = np.zeros(n_kp)
    sub_best: list[list[float]] = []
    kp_stems = [set(T.content_stems(h)) for h in hyps]
    win_stems = [T.content_stems(t) for t in texts]
    for i, ss in enumerate(subs):
        rows = np.stack([es[idx[u]] for u in ss]) * factor
        e[i] = rows.mean(axis=0)
        best = rows.max(axis=1)
        sub_best.append([float(b) for b in best])
        coverage[i] = float(best.mean())
        others = [x for j, st in enumerate(kp_stems) if j != i for x in st]
        for u in ss:
            k = idx[u]
            siblings = [v for v in ss if v != u]
            sib_stems = [x for v in siblings for x in T.content_stems(v)]
            distinct = T.distinct_terms(T.content_stems(u), others + sib_stems)
            for w in range(n_win):
                if not (cs[k, w] >= th.contradiction and cs[k, w] > es[k, w] and factor[w] >= 1.0
                        and attributed[i, w] and _overlap(u, texts[w]) >= NLI_CONTRA_MIN_OVERLAP):
                    continue
                if distinct and not any(T.stems_match(d, x) for d in distinct for x in win_stems[w]):
                    continue
                if any(es[idx[v], w] >= th.tau for v in siblings):
                    continue
                c[i, w] = max(c[i, w], float(cs[k, w]))
    return _Evidence(e, c, coverage, sub_best), subs


# ------------------------------------------------------------------ grading
def grade_answer(question, answer_text: str, scorer: PairScorer, th: Thresholds | None = None) -> GradeResult:  # noqa: ANN001, C901, PLR0912, PLR0915
    th = resolve_thresholds(scorer.name, th)
    max_marks = float(question.max_marks)
    prior = SCORER_PRIORS.get(scorer.name, DEFAULT_PRIOR)
    qid = getattr(question, "id", None) or getattr(question, "question_no", None)
    trace: dict = {"question": qid, "scorer": scorer.name, "prior": prior,
                   "thresholds": asdict(th), "key_points": []}

    def finish(result: GradeResult) -> GradeResult:
        result.trace = {**trace, "marks": result.marks, "verdict": result.verdict, "flags": list(result.flags),
                        "grading_confidence": result.grading_confidence}
        if log.isEnabledFor(logging.DEBUG):
            log.debug("grade question=%s scorer=%s marks=%.2f/%.2f verdict=%s confidence=%.3f flags=%s kps=%s",
                      qid, scorer.name, result.marks, max_marks, result.verdict, result.grading_confidence,
                      ",".join(result.flags) or "-",
                      " ".join(f"{k['index']}:{k['status']}:{k['reason']}:e{k.get('entail', 0):.2f}"
                               f":c{k.get('contra', 0):.2f}" for k in trace["key_points"]) or "-")
        return result

    if question.type == "maths":
        # Maths is optional (scope freeze 6.3): always routed to the teacher.
        return finish(GradeResult(0.0, max_marks, 0.0, [], ["maths_manual_review"], scorer.name, "ungraded"))

    kps = normalise_key_points(question)
    if not kps:
        return finish(GradeResult(0.0, max_marks, 0.0, [], ["no_rubric"], scorer.name, "ungraded"))

    answer_text = answer_text or ""
    spans, wins = answer_windows(answer_text)
    hyps = [str(kp["text"]) for kp in kps]
    kp_weights = [float(kp.get("weight", 0) or 0) for kp in kps]
    if not wins:
        results = [KeyPointResult(i, hyps[i], kp_weights[i], 0.0, 0.0, None, 0.0, "missed", "empty_answer")
                   for i in range(len(kps))]
        trace["key_points"] = [{"index": i, "status": "missed", "reason": "empty_answer"} for i in range(len(kps))]
        # Nothing written: 0 marks is certain whatever the scorer.
        return finish(GradeResult(0.0, max_marks, 1.0, results, [], scorer.name, verdict_for(0.0, max_marks)))

    flags: list[str] = []
    prompt = getattr(question, "prompt", None) or ""
    prompt_stems = T.content_stems(prompt)

    # Sentences that only restate the question are not evidence; an answer made only
    # of them is a copied question (adversarial set, PRD 20.2).
    restates = [len(prompt_stems) >= 2 and _novelty(answer_text[s:e], prompt_stems) < RESTATEMENT_NOVELTY
                for s, e in spans]
    lexical = LexicalScorer()
    copied = (bool(spans) and all(restates) and _overlap(prompt, answer_text) >= 0.5) or (
        bool(prompt.strip()) and lexical.similarity(answer_text, prompt) >= 0.75
        and len(T.words(answer_text)) <= 1.6 * max(1, len(T.words(prompt))))
    # Only longer restating sentences are dropped as evidence ("Veins have valves" is short
    # and uses the question's words, yet it is an answer).
    restates = [r and len(T.content_stems(answer_text[s:e])) >= 4 for r, (s, e) in zip(restates, spans)]
    if copied:
        flags.append(FLAG_COPIED)
    if not any(w.is_proposition for w in wins):
        flags.append(FLAG_KEYWORD_LIST)

    texts = [answer_text[w.start:w.end] for w in wins]
    base_factor = np.array([0.0 if all(restates[k] for k in w.sentences) else
                            (1.0 if w.is_proposition else NON_PROPOSITION_FACTOR) for w in wins])
    multi = np.array([len(w.sentences) > 1 for w in wins])
    weights = term_weights(hyps, prompt)

    # Lexical evidence: always computed (the only evidence for the lexical scorer).
    lex_th = th if scorer.name == "lexical" else LEXICAL_THRESHOLDS
    E_lex, C_lex = lexical.score(texts, hyps, weights=weights)
    lex = _Evidence(np.asarray(E_lex) * base_factor * np.where(multi, LEXICAL_WINDOW_DISCOUNT, 1.0),
                    np.asarray(C_lex) * (base_factor >= 1.0))
    # Which key point is each window about (either meaning)? A contradiction found in a
    # window counts only for the key point(s) the window is most about.
    kp_slots = [T.key_point_slots(h, weights) for h in hyps]
    about = np.array([[T.related_share(sl, t) for t in texts] for sl in kp_slots])
    attributed = about >= about.max(axis=0, keepdims=True) - 0.05
    lex.c = lex.c * attributed
    model: _Evidence | None = None
    subs: list[list[str]] = [[h] for h in hyps]
    if scorer.name in NLI_SCORERS:
        model, subs = _nli_evidence(scorer, texts, hyps, base_factor, th, attributed)
    elif scorer.name != "lexical":
        E_m, C_m = scorer.score(texts, hyps)
        model = _Evidence(np.asarray(E_m, dtype=float) * base_factor,
                          np.asarray(C_m, dtype=float) * (base_factor >= 1.0) * attributed)

    # Role swaps: key-point sub-claims as statements.
    units: list[tuple[int, str]] = [(i, u) for i, h in enumerate(hyps) for u in T.kp_units(h)]
    pairs = T.role_pairs([T.content_stems(u) for _, u in units])
    unit_slots = [T.key_point_slots(u, weights) for _, u in units]
    kp_stems = [T.content_stems(h) for h in hyps]
    distinct = [T.distinct_terms(kp_stems[i], [s for j, o in enumerate(kp_stems) if j != i for s in o])
                for i in range(len(kps))]
    answer_values = T.number_values(answer_text)
    given_values = T.number_values(prompt)
    known_values = given_values.union(*(T.number_values(h) for h in hyps))
    answer_eqs = [T.equations(answer_text[s:e]) for s, e in spans]

    results: list[KeyPointResult] = []
    margins: list[float] = []
    for i, kp in enumerate(kps):
        weight = kp_weights[i]
        kt: dict = {"index": i}
        # --- lexical decision
        jl = int(np.argmax(lex.e[i]))
        e_lex = float(lex.e[i, jl])
        lex_status = _status(e_lex, lex_th)
        kt["lexical_entail"] = round(e_lex, 4)
        status, j, e, src = lex_status, jl, e_lex, "lexical"
        margin = _margin(e_lex, lex_th.tau, lex_th.tau_low)
        reason = {"matched": "entailed", "partial": "partial_support", "missed": "no_support"}[status]
        # --- model decision, combined as a union with the lexical one
        if model is not None:
            jm = int(np.argmax(model.e[i]))
            if scorer.name in NLI_SCORERS:
                e_m = float(model.coverage[i])
                best = model.sub_best[i]
                m_status = "matched" if e_m >= th.tau else \
                    "partial" if len(best) > 1 and max(best) >= th.tau else "missed"
                m_reason = "entailed" if m_status == "matched" else "sub_claims"
                m_margin = _margin(e_m, th.tau)
                kt["sub_claim_entail"] = [round(b, 4) for b in best]
            else:
                e_m = float(model.e[i, jm])
                m_status = _status(e_m, th)
                m_reason = {"matched": "entailed", "partial": "partial_support", "missed": "no_support"}[m_status]
                m_margin = _margin(e_m, th.tau, th.tau_low)
            kt["model_entail"] = round(e_m, 4)
            lex_margin = margin
            if _RANK[m_status] >= _RANK[lex_status]:
                status, j, e, src, reason, margin = m_status, jm, e_m, scorer.name, m_reason, m_margin
            if m_status == lex_status:
                margin = max(m_margin, lex_margin)  # both sources agree
            else:
                margin = min(margin, 0.5)  # the sources disagree: a teacher should look
        kt["source"] = src
        ev_win = wins[j]

        # --- contradiction: strongest gated contradiction that beats the support
        c, contra_win, c_src = 0.0, None, ""
        jc = int(np.argmax(lex.c[i]))
        c_lex = float(lex.c[i, jc])
        nli_support_there = float(model.e[i, jc]) if model is not None else 0.0
        if c_lex >= lex_th.contradiction and c_lex > e and nli_support_there < th.tau:
            c, contra_win, c_src = c_lex, wins[jc], "lexical"
        if model is not None:
            jm_c = int(np.argmax(model.c[i]))
            c_m = float(model.c[i, jm_c])
            kt["model_contra"] = round(c_m, 4)
            if c_m >= th.contradiction and c_m > e and c_m >= c:
                c, contra_win, c_src = c_m, wins[jm_c], scorer.name
        kt["lexical_contra"] = round(c_lex, 4)

        # --- equations: a calculation key point is judged on its result (and unit), a
        # formula key point on the formula ("F = ma" ~ "F = m x a"); a written formula
        # without the value is partial (benchmark labelling rule)
        eq_done = False
        kp_eqs = T.equations(hyps[i])
        calc = [q for q in kp_eqs if q.result is not None
                and not any(T.same_number(q.result, g) for g in given_values)]
        if kp_eqs and contra_win is None:
            eq_status, eq_reason, eq_sent = "", "", None
            if calc:
                kq = calc[-1]
                right = [(k, a) for k, eqs in enumerate(answer_eqs) for a in eqs
                         if a.result is not None and T.same_number(a.result, kq.result)
                         and (a.lhs == kq.lhs or (kq.unit and a.unit == kq.unit))]
                wrong_eq = [(k, a) for k, eqs in enumerate(answer_eqs) for a in eqs
                            if a.result is not None and a.lhs == kq.lhs and not T.same_number(a.result, kq.result)]
                if right:
                    k, a = right[0]
                    unit_ok = not kq.unit or a.unit == kq.unit
                    eq_status, eq_reason, eq_sent = ("matched", "calculation", k) if unit_ok else \
                        ("partial", "unit_mismatch", k)
                elif wrong_eq:
                    k, _a = wrong_eq[0]
                    c, contra_win, c_src = 1.0, wins[k], "numeric"
                    eq_done = True
                elif any(T.formula_match(q, [a for eqs in answer_eqs for a in eqs]) for q in kp_eqs):
                    k = next(k for k, eqs in enumerate(answer_eqs) if any(T.formula_match(q, eqs) for q in kp_eqs))
                    eq_status, eq_reason, eq_sent = "partial", "formula_without_result", k
            else:
                for q in kp_eqs:
                    hit = next((k for k, eqs in enumerate(answer_eqs) if T.formula_match(q, eqs)), None)
                    if hit is not None:
                        st = "partial" if q.in_parens else "matched"
                        if not eq_status or _RANK[st] > _RANK[eq_status]:
                            eq_status, eq_reason, eq_sent = st, "formula", hit
            if eq_status:
                eq_done = True
                kt["equation"] = eq_reason
                if _RANK[eq_status] > _RANK[status]:
                    status, reason, j, e = eq_status, eq_reason, eq_sent, max(e, 0.0)
                    ev_win = wins[eq_sent]  # single-sentence windows come first
                    margin = min(margin, 0.5) if eq_status == "partial" else max(margin, 0.5)
            if eq_reason in ("unit_mismatch", "formula_without_result") or c_src == "numeric":
                if FLAG_NUMERIC not in flags:
                    flags.append(FLAG_NUMERIC)

        # --- numbers: a key point's own numbers (not those given in the question) must appear
        kp_values = T.number_values(hyps[i])
        missing = T.missing_numbers(T.missing_numbers(kp_values, given_values), answer_values)
        win_text = answer_text[ev_win.start:ev_win.end]
        if missing and status != "missed" and contra_win is None and not eq_done:
            wrong = T.missing_numbers(T.number_values(win_text), known_values)
            if wrong:  # another value stated where the key point's value belongs
                c, contra_win, c_src = 1.0, ev_win, "numeric"
            elif "=" in hyps[i] or any(T.number_values(str(k)) for k in (kp.get("keywords") or [])):
                status, reason = "partial", "numeric_mismatch"  # a calculation without its result
                margin = min(margin, 0.5)
            if FLAG_NUMERIC not in flags and (wrong or reason == "numeric_mismatch"):
                flags.append(FLAG_NUMERIC)

        evidence = None
        if contra_win is not None:
            status, reason = "contradicted", "numeric_mismatch" if c_src == "numeric" else "contradiction"
            evidence = answer_text[contra_win.start:contra_win.end]
            margin = 0.5 if c_src == "numeric" else min(1.0, (c - th.contradiction if c_src == scorer.name
                                                                else c - lex_th.contradiction) / MARGIN_SCALE)
            kt["contradiction_source"] = c_src
            ev_win = contra_win
        else:
            if c_lex > 0:  # an opposite statement that did not outweigh the support: less sure
                margin = min(margin, 0.3)
            if status != "missed":
                evidence = win_text

        # --- role swap: the clause aligned with this key point uses a parallel key point's
        # term (checked on the supporting window, and on a contradicting one to explain it)
        swap_text = win_text if status in ("matched", "partial") else \
            (evidence or "") if status == "contradicted" and c_src not in ("numeric", "") else ""
        if swap_text:
            for u, (ki, _) in enumerate(units):
                if ki != i:
                    continue
                span, _states = T.aligned_clause(unit_slots[u], swap_text)
                if span is None:
                    continue
                clause = swap_text[span[0]:span[1]]
                swap = T.find_role_swap(clause, pairs[u])
                if swap is not None:
                    status, reason, evidence, margin = "contradicted", "role_swap", clause, min(margin, 0.5)
                    kt["role_swap"] = {"key_point_term": swap.mine, "other_term": swap.theirs,
                                       "other_key_point": units[swap.other_unit][0]}
                    break

        # --- rubric keywords are required for a full match (any one of the alternatives)
        kws = [str(k) for k in (kp.get("keywords") or []) if str(k).strip()]
        if kws:
            ctx = _context_text(answer_text, spans, ev_win)
            hits = [k for k in kws if T.keyword_present(k, ctx)]
            kt["keyword_hits"], kt["keyword_missing"] = hits, [k for k in kws if k not in hits]
            if status == "matched" and not hits:
                status, reason = "partial", "keyword_missing"
                margin = min(margin, 0.5)
            elif status == "missed" and contra_win is None and e_lex >= KEYWORD_PARTIAL_MIN_SUPPORT:
                # The teacher's keyword stated in a sentence that is about the key point:
                # half credit (a bare keyword list never gets here - its windows are discounted).
                own = [k for k in kws if not T.keyword_present(k, prompt)]  # not just the question's words
                if any(T.keyword_present(k, answer_text[wins[jl].start:wins[jl].end]) for k in own) \
                        and wins[jl].is_proposition:
                    status, reason, ev_win = "partial", "keyword_support", wins[jl]
                    evidence = answer_text[ev_win.start:ev_win.end]
                    margin = min(margin, 0.3)

        kt["window_sentences"] = list(ev_win.sentences)
        kt["lexical_recall"] = round(T.slot_recall(T.key_point_slots(hyps[i], weights), win_text), 4)
        kt["distinctive_hits"] = sorted(d for d in distinct[i]
                                        if any(T.stems_match(d, s) for s in T.content_stems(win_text)))
        awarded = weight if status == "matched" else weight / 2 if status == "partial" else 0.0
        results.append(KeyPointResult(i, hyps[i], weight, e, c, evidence, awarded, status, reason))
        margins.append(margin)
        kt.update(entail=round(e, 4), contra=round(c, 4))
        trace["key_points"].append(kt)

    # The answer is about the opposite concept ("Underfitting is ..." for "Explain overfitting").
    inversion = T.opposite_concept(prompt, answer_text)
    if inversion is not None:
        q_term, a_term = inversion
        quote = next((answer_text[s:e] for s, e in spans if a_term in T.words(answer_text[s:e])), None)
        trace["opposite_concept"] = {"question_term": q_term}
        for k, r in enumerate(results):
            if r.status in ("matched", "partial") and quote:
                r.status, r.reason, r.awarded, r.evidence_sentence = "contradicted", "opposite_concept", 0.0, quote
                margins[k] = 0.5

    if copied:
        for r in results:
            if r.status != "missed":
                r.status, r.reason, r.awarded, r.evidence_sentence = "missed", "copied_question", 0.0, None

    # Every awarded mark (and every contradiction) needs a quote found verbatim in the answer.
    for r in results:
        if r.status != "missed" and (not r.evidence_sentence or r.evidence_sentence not in answer_text):
            r.status, r.reason, r.awarded, r.evidence_sentence = "missed", "no_evidence", 0.0, None
        if r.status == "missed":
            r.awarded, r.evidence_sentence = 0.0, None

    for r, kt in zip(results, trace["key_points"]):
        kt["status"], kt["reason"], kt["awarded"] = r.status, r.reason, round(r.awarded, 2)
    if any(r.status == "contradicted" for r in results):
        flags.append(FLAG_CONTRADICTION)
    if any(r.reason == "role_swap" for r in results):
        flags.append(FLAG_ROLE_SWAP)
    if any(r.status == "partial" for r in results):
        flags.append(FLAG_PARTIAL)

    marks = round(float(np.clip(sum(r.awarded for r in results), 0.0, max_marks)), 2)

    # Topicality: off-topic answers vs on-topic answers the scorer may have under-marked.
    vocab = set(T.content_stems(" ".join([prompt, getattr(question, "model_answer", None) or ""] + hyps
                                         + [str(k) for kp in kps for k in (kp.get("keywords") or [])])))
    ans_stems = set(T.content_stems(answer_text))
    overlap = (sum(1 for a in ans_stems if a in vocab or any(T.stems_match(a, v) for v in vocab))
               / len(ans_stems)) if ans_stems else 0.0
    best_recall = max((kt.get("lexical_recall", 0.0) for kt in trace["key_points"]), default=0.0)
    credited = [r for r in results if r.status in ("matched", "partial")]
    trace["topic"] = {"vocabulary_overlap": round(overlap, 4), "best_lexical_recall": round(best_recall, 4)}
    gaming = {FLAG_KEYWORD_LIST, FLAG_COPIED} & set(flags)
    if len(ans_stems) >= 3 and not credited and not gaming and FLAG_CONTRADICTION not in flags \
            and overlap < 0.1 and best_recall < 0.2:
        flags.append(FLAG_OFF_TOPIC)
    elif not gaming and FLAG_CONTRADICTION not in flags and marks < 0.5 * max_marks and len(ans_stems) >= 6:
        used = {k for r in credited for k in _evidence_sentences(r, answer_text, spans)}
        unexplained = [k for k in range(len(spans)) if k not in used and not restates[k]
                       and len(T.content_stems(answer_text[spans[k][0]:spans[k][1]])) >= 4]
        if unexplained:
            flags.append(FLAG_PARAPHRASE)
            trace["topic"]["unexplained_sentences"] = len(unexplained)

    if prior < LOW_RELIABILITY_PRIOR:
        flags.append(FLAG_LOW_RELIABILITY)
    total_w = sum(r.weight for r in results) or 1.0
    margin_factor = sum((0.5 + 0.5 * min(1.0, max(0.0, m))) * r.weight for m, r in zip(margins, results)) / total_w
    if FLAG_PARAPHRASE in flags:
        margin_factor *= 0.6
    if FLAG_KEYWORD_LIST in flags:
        margin_factor *= 0.8
    conf = round(float(np.clip(prior * margin_factor, 0, 1)), 4)
    trace["margin_factor"] = round(margin_factor, 4)

    return finish(GradeResult(marks, max_marks, conf, results, list(dict.fromkeys(flags)), scorer.name,
                              verdict_for(marks, max_marks)))


def _evidence_sentences(r: KeyPointResult, answer_text: str, spans: list[tuple[int, int]]) -> list[int]:
    if not r.evidence_sentence:
        return []
    pos = answer_text.find(r.evidence_sentence)
    if pos < 0:
        return []
    end = pos + len(r.evidence_sentence)
    return [k for k, (s, e) in enumerate(spans) if s < end and e > pos]


# ------------------------------------------------------------------ evaluation baseline
def baseline_formula_marks(question, answer_text: str, scorer: PairScorer) -> float:  # noqa: ANN001
    """The pre-v2 'short' formula, max x clip(0.6*similarity + 0.4*keyword coverage)
    rounded to 0.5 (PRD 12.4 starting point). It is no longer used for marking; it is
    kept only so the evaluation harness can compare against it."""
    max_marks = float(question.max_marks)
    kps = normalise_key_points(question)
    if question.type == "maths" or not kps or not (answer_text or "").strip():
        return 0.0
    hyps = [str(kp["text"]) for kp in kps]
    reference = question.model_answer or " ".join(hyps)
    sim = scorer.similarity(answer_text, reference)
    keywords = [k for kp in kps for k in (kp.get("keywords") or [])] or \
        list(dict.fromkeys(s for h in hyps for s in T.content_stems(h)))
    ans_stems = T.content_stems(answer_text)
    coverage = (sum(1 for k in keywords if any(T.fuzzy_equal(T.stem(str(k).lower()), a) for a in ans_stems))
                / len(keywords)) if keywords else 0.0
    raw = float(np.clip(0.6 * sim + 0.4 * coverage, 0, 1))
    if not any(T.is_proposition(s) for s in T.sentences(answer_text)):
        raw *= 0.3
    return round(float(np.clip(_round_half(max_marks * raw), 0, max_marks)), 2)
