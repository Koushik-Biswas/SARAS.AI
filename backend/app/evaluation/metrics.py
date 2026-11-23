"""Evaluation metrics for SARAS AI (numpy only; PRD 3.7 Q-2).

Conventions
-----------
* ``pred`` / ``gold`` are equal-length sequences. ``gold`` is the reference (teacher or
  benchmark label), ``pred`` the system output.
* Undefined values are ``float("nan")`` -- e.g. Pearson with zero variance, a precision
  with no predicted positives, kappa when chance agreement is 1. They are never replaced
  by 0 or 1; reports print them as "n/a".
* Answer-level correctness: an answer is *correct* when ``|pred - expected| <= 0.5``
  marks (``CORRECT_TOL``; half a mark is the marking granularity). ECE, Brier,
  risk-coverage and the error rate among non-flagged answers all use this definition.
* QWK bins marks to ``step``-mark ratings, ``rating = floor(marks / step + 0.5)`` (round half
  up -- the rounding the grader uses for half marks). Pooling *absolute* marks across
  questions with different maximum marks rewards knowing each question's scale rather than
  the answer: predicting max/2 for every answer gets a positive pooled QWK, and awarding
  half marks to answers that deserve full marks on 2-, 5- and 8-mark questions gets QWK
  0.44 and Pearson 1.0. The evaluation therefore uses
  - ``qwk_normalised`` (headline): marks / max_marks binned to 0.1 steps (ratings 0..10),
    pooled. Any constant fraction of the maximum scores 0. It still credits differences
    *between* questions (knowing which question is answered better), and on questions
    worth more than 5 marks two adjacent half marks can share a 0.1 bin.
  - ``qwk_per_question`` / ``qwk_per_question_mean``: QWK within one question on its own
    0.5-mark scale, for questions with >= 5 answers whose *gold* ratings are not all equal
    (the rule looks at gold only, so every grader is judged on the same questions and a
    constant predictor scores 0 rather than "undefined"); the mean is unweighted.
  - ``pearson_normalised``: Pearson on marks / max_marks.
  ``quadratic_weighted_kappa`` / ``pearson`` on pooled absolute marks remain available as
  scale-confounded references. Quarter-mark labels (e.g. 1.75) are rounded half up on the
  0.5 grid, so on that grid +0.25 counts as agreement and -0.25 as a one-step miss.
* Risk-coverage and AURC accept answers by confidence threshold. Tied confidences are
  accepted together (one curve point per distinct confidence), so the results do not
  depend on input order.
* Bootstrap CIs (``bootstrap_interval``) report how many resamples gave a defined (finite)
  value. With fewer than ``MIN_DEFINED_SHARE`` (50%) defined, the CI is nan; otherwise it is
  computed on the defined resamples and the count is reported with it, so a narrowed
  interval is never silent.
* Key-point (KP) statuses are ``matched | partial | missed | contradicted``. A KP is
  *awarded* (earns credit) when it is matched or partial.
* Confusion matrices have rows = gold, columns = pred.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

STATUSES: tuple[str, ...] = ("matched", "partial", "missed", "contradicted")
AWARDED: tuple[str, ...] = ("matched", "partial")
CONTRADICTED = "contradicted"
CORRECT_TOL = 0.5
MARK_STEP = 0.5
FRACTION_STEP = 0.1  # qwk_normalised: marks / max_marks on a 0.1 grid (ratings 0..10)
MIN_QUESTION_ANSWERS = 5  # qwk_per_question: a question needs at least this many answers
MIN_DEFINED_SHARE = 0.5  # bootstrap: CI is nan when fewer resamples than this share are defined
NAN = float("nan")
_EPS = 1e-9


def _div(num: float, den: float) -> float:
    return float(num) / float(den) if den else NAN


def _pair(pred: Sequence[float], gold: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=float).ravel()
    g = np.asarray(gold, dtype=float).ravel()
    if p.shape != g.shape:
        raise ValueError(f"pred and gold differ in length: {p.size} != {g.size}")
    return p, g


# ----------------------------------------------------------------------------- marks
def mae(pred: Sequence[float], gold: Sequence[float]) -> float:
    """Mean absolute error in marks."""
    p, g = _pair(pred, gold)
    return float(np.mean(np.abs(p - g))) if p.size else NAN


def rmse(pred: Sequence[float], gold: Sequence[float]) -> float:
    """Root mean squared error in marks."""
    p, g = _pair(pred, gold)
    return float(np.sqrt(np.mean((p - g) ** 2))) if p.size else NAN


def within(pred: Sequence[float], gold: Sequence[float], tol: float) -> float:
    """Share of items with ``|pred - gold| <= tol`` (inclusive, float-safe)."""
    p, g = _pair(pred, gold)
    return float(np.mean(np.abs(p - g) <= tol + _EPS)) if p.size else NAN


def pearson(pred: Sequence[float], gold: Sequence[float]) -> float:
    """Pearson correlation; nan when either side has zero variance or n < 2."""
    p, g = _pair(pred, gold)
    if p.size < 2:
        return NAN
    dp, dg = p - p.mean(), g - g.mean()
    den = math.sqrt(float(np.sum(dp * dp)) * float(np.sum(dg * dg)))
    return float(np.sum(dp * dg)) / den if den > _EPS else NAN


def bin_marks(marks: Sequence[float], step: float = MARK_STEP) -> np.ndarray:
    """Integer ratings on a ``step``-mark grid, rounding half up (0.25 -> 1 with step 0.5)."""
    m = np.asarray(marks, dtype=float).ravel()
    return np.floor(m / step + 0.5 + _EPS).astype(int)


def quadratic_weighted_kappa(pred: Sequence[float], gold: Sequence[float], step: float = MARK_STEP) -> float:
    """Quadratic weighted kappa on marks binned to ``step``-mark ratings (see module doc).

    ``1 - sum(W * O) / sum(W * E)`` with ``W[i, j] = (i - j)^2 / (K - 1)^2``, O the
    observed gold x pred rating matrix and E the outer product of the marginals / n.
    Computed in O(n) through the identities sum(W * O) = sum((g - p)^2) / (K - 1)^2 and
    sum(W * E) = (sum(g^2) + sum(p^2) - 2 sum(g) sum(p) / n) / (K - 1)^2 on the integer
    ratings g, p (the (K - 1)^2 cancels). nan when undefined (empty input, or every
    rating identical on both sides)."""
    p, g = _pair(pred, gold)
    if not p.size:
        return NAN
    rp, rg = bin_marks(p, step).astype(float), bin_marks(g, step).astype(float)
    num = float(np.sum((rg - rp) ** 2))
    den = float(np.sum(rg * rg) + np.sum(rp * rp) - 2.0 * rg.sum() * rp.sum() / p.size)
    return 1.0 - num / den if den > _EPS else NAN


def normalise_marks(marks: Sequence[float], max_marks: Sequence[float]) -> np.ndarray:
    """Marks as a fraction of each item's question maximum (``marks / max_marks``).
    Raises ValueError on a length mismatch or a max_marks that is not finite and > 0."""
    m = np.asarray(marks, dtype=float).ravel()
    mx = np.asarray(max_marks, dtype=float).ravel()
    if mx.shape != m.shape:
        raise ValueError(f"max_marks differs in length from marks: {mx.size} != {m.size}")
    if not np.all(np.isfinite(mx) & (mx > 0)):
        raise ValueError("max_marks must be finite and > 0")
    return m / mx


def qwk_normalised(pred: Sequence[float], gold: Sequence[float], max_marks: Sequence[float],
                   step: float = FRACTION_STEP) -> float:
    """Headline QWK: marks / max_marks binned to ``step`` (0.1 -> ratings 0..10), pooled
    across questions (see module doc). Any constant fraction of the maximum scores 0."""
    p, g = _pair(pred, gold)
    return quadratic_weighted_kappa(normalise_marks(p, max_marks), normalise_marks(g, max_marks), step)


def pearson_normalised(pred: Sequence[float], gold: Sequence[float], max_marks: Sequence[float]) -> float:
    """Pearson correlation of marks / max_marks (nan when either side is constant)."""
    p, g = _pair(pred, gold)
    return pearson(normalise_marks(p, max_marks), normalise_marks(g, max_marks))


def qwk_per_question(pred: Sequence[float], gold: Sequence[float], question_ids: Sequence[Any],
                     step: float = MARK_STEP, min_answers: int = MIN_QUESTION_ANSWERS) -> dict[Any, float]:
    """QWK within each question on its own ``step``-mark scale: {question id: qwk} in order of
    first appearance, for the questions with >= ``min_answers`` answers whose gold ratings
    are not all equal. Inclusion looks at gold only, so every grader is judged on the same
    questions, and a constant prediction scores 0 (not nan). With constant gold, QWK is 0
    for any non-identical prediction, so such questions carry no information and are left out."""
    p, g = _pair(pred, gold)
    qids = list(question_ids)
    if len(qids) != p.size:
        raise ValueError(f"question_ids differs in length from pred/gold: {len(qids)} != {p.size}")
    rows: dict[Any, list[int]] = {}
    for i, q in enumerate(qids):
        rows.setdefault(q, []).append(i)
    out: dict[Any, float] = {}
    for q, idx in rows.items():
        if len(idx) < min_answers:
            continue
        rg = bin_marks(g[idx], step)
        if np.all(rg == rg[0]):
            continue
        out[q] = quadratic_weighted_kappa(p[idx], g[idx], step)
    return out


def qwk_per_question_mean(pred: Sequence[float], gold: Sequence[float], question_ids: Sequence[Any],
                          step: float = MARK_STEP, min_answers: int = MIN_QUESTION_ANSWERS) -> float:
    """Unweighted mean of ``qwk_per_question``; nan when no question qualifies."""
    vals = list(qwk_per_question(pred, gold, question_ids, step, min_answers).values())
    return float(np.mean(vals)) if vals else NAN


# ----------------------------------------------------------------------------- labels
def confusion_matrix(gold: Sequence[Any], pred: Sequence[Any], labels: Sequence[Any] = STATUSES) -> np.ndarray:
    """Integer confusion matrix, rows = gold label, columns = pred label (in ``labels`` order).
    Raises ValueError on a label outside ``labels``."""
    if len(gold) != len(pred):
        raise ValueError(f"gold and pred differ in length: {len(gold)} != {len(pred)}")
    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(gold, pred):
        if a not in index or b not in index:
            raise ValueError(f"label outside {list(labels)}: {a if a not in index else b!r}")
        cm[index[a], index[b]] += 1
    return cm


def kappa_from_confusion(cm: np.ndarray) -> float:
    """Cohen's (unweighted) kappa from a square confusion matrix; nan when chance agreement is 1."""
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    if n <= 0:
        return NAN
    po = np.trace(cm) / n
    pe = float(np.sum(cm.sum(axis=1) * cm.sum(axis=0))) / (n * n)
    return float((po - pe) / (1.0 - pe)) if 1.0 - pe > _EPS else NAN


def cohen_kappa(a: Sequence[Any], b: Sequence[Any], labels: Sequence[Any] | None = None) -> float:
    """Cohen's unweighted kappa between two raters (e.g. expected vs predicted KP statuses)."""
    if labels is None:
        labels = sorted(set(a) | set(b), key=str)
    return kappa_from_confusion(confusion_matrix(a, b, labels))


def accuracy_from_confusion(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=float)
    return _div(np.trace(cm), cm.sum())


def prf(tp: float, fp: float, fn: float) -> dict[str, float]:
    """Precision, recall and F1 (nan where a denominator is 0)."""
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    if not (math.isfinite(p) and math.isfinite(r)):
        f1 = NAN
    else:
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def awarded_prf_from_confusion(cm: np.ndarray, labels: Sequence[str] = STATUSES,
                               positive: Sequence[str] = AWARDED) -> dict[str, float]:
    """P/R/F1 for 'awarded' KPs (gold or pred status in ``positive``) from a status confusion matrix."""
    cm = np.asarray(cm, dtype=float)
    pos = np.array([lab in positive for lab in labels])
    tp = cm[np.ix_(pos, pos)].sum()
    fp = cm[np.ix_(~pos, pos)].sum()
    fn = cm[np.ix_(pos, ~pos)].sum()
    return prf(tp, fp, fn)


def awarded_prf(gold: Sequence[str], pred: Sequence[str]) -> dict[str, float]:
    """P/R/F1 where a KP counts as positive when it is awarded (matched or partial)."""
    return awarded_prf_from_confusion(confusion_matrix(gold, pred, STATUSES))


def contradiction_from_confusion(cm: np.ndarray, labels: Sequence[str] = STATUSES,
                                 label: str = CONTRADICTED) -> dict[str, float]:
    """Contradiction precision / recall and the false-contradiction rate: the share of KPs
    whose gold status is *not* contradicted but which the grader marked contradicted."""
    cm = np.asarray(cm, dtype=float)
    c = list(labels).index(label)
    other = np.array([i != c for i in range(len(labels))])
    return {
        "precision": _div(cm[c, c], cm[:, c].sum()),
        "recall": _div(cm[c, c], cm[c, :].sum()),
        "false_contradiction_rate": _div(cm[other, c].sum(), cm[other, :].sum()),
    }


def contradiction_metrics(gold: Sequence[str], pred: Sequence[str]) -> dict[str, float]:
    return contradiction_from_confusion(confusion_matrix(gold, pred, STATUSES))


# ----------------------------------------------------------------------------- deferral
def correct_mask(pred: Sequence[float], gold: Sequence[float], tol: float = CORRECT_TOL) -> np.ndarray:
    """Boolean array: answer is correct when ``|pred - gold| <= tol``."""
    p, g = _pair(pred, gold)
    return np.abs(p - g) <= tol + _EPS


def flag_rate(flagged: Sequence[bool]) -> float:
    """Deferral rate: share of answers routed to teacher review."""
    f = np.asarray(flagged, dtype=bool).ravel()
    return float(f.mean()) if f.size else NAN


def binary_prf(gold: Sequence[bool], pred: Sequence[bool]) -> dict[str, float]:
    """P/R/F1 for a boolean decision (e.g. predicted review flag vs expected flag)."""
    g = np.asarray(gold, dtype=bool).ravel()
    p = np.asarray(pred, dtype=bool).ravel()
    if g.shape != p.shape:
        raise ValueError("gold and pred differ in length")
    return prf(float(np.sum(g & p)), float(np.sum(~g & p)), float(np.sum(g & ~p)))


def error_rate_unflagged(pred: Sequence[float], gold: Sequence[float], flagged: Sequence[bool],
                         tol: float = CORRECT_TOL) -> float:
    """Share of *incorrect* answers among those NOT flagged for review (the auto-accept
    risk). nan when every answer is flagged."""
    ok = correct_mask(pred, gold, tol)
    f = np.asarray(flagged, dtype=bool).ravel()
    if f.shape != ok.shape:
        raise ValueError("flagged differs in length from pred/gold")
    keep = ~f
    return float(np.mean(~ok[keep])) if keep.any() else NAN


# ----------------------------------------------------------------------------- calibration
def _conf_correct(conf: Sequence[float], correct: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    c = np.clip(np.asarray(conf, dtype=float).ravel(), 0.0, 1.0)
    y = np.asarray(correct, dtype=float).ravel()
    if c.shape != y.shape:
        raise ValueError("confidence and correctness differ in length")
    return c, y


def _bin_sums(c: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per equal-width bin [0, 1/B), ..., [(B-1)/B, 1] (1.0 in the last bin): count, sum of
    confidence, sum of correctness."""
    idx = np.minimum((c * n_bins + _EPS).astype(int), n_bins - 1)
    return (np.bincount(idx, minlength=n_bins), np.bincount(idx, weights=c, minlength=n_bins),
            np.bincount(idx, weights=y, minlength=n_bins))


def reliability_bins(conf: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> list[dict]:
    """Equal-width confidence bins [0, 0.1), ..., [0.9, 1.0] (1.0 goes in the last bin):
    count, mean confidence and accuracy per bin (nan for empty bins)."""
    c, y = _conf_correct(conf, correct)
    cnt, sc, sy = _bin_sums(c, y, n_bins)
    return [{"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": int(cnt[b]),
             "mean_confidence": float(sc[b] / cnt[b]) if cnt[b] else NAN,
             "accuracy": float(sy[b] / cnt[b]) if cnt[b] else NAN} for b in range(n_bins)]


def ece(conf: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """Expected calibration error: sum over bins of (n_b / n) * |accuracy_b - mean_conf_b|
    (= sum over bins of |correct_b - confidence_b sums| / n)."""
    c, y = _conf_correct(conf, correct)
    if not c.size:
        return NAN
    _, sc, sy = _bin_sums(c, y, n_bins)
    return float(np.sum(np.abs(sy - sc)) / c.size)


def brier(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """Brier score of confidence against 0/1 correctness."""
    c, y = _conf_correct(conf, correct)
    return float(np.mean((c - y) ** 2)) if c.size else NAN


def _threshold_counts(conf: Sequence[float], correct: Sequence[bool]) -> tuple[np.ndarray, np.ndarray, int]:
    """For each distinct confidence t, highest first: (number accepted with confidence >= t,
    errors among them, n). Tied answers are accepted together, so input order is irrelevant."""
    c, y = _conf_correct(conf, correct)
    if not c.size:
        return np.zeros(0, dtype=int), np.zeros(0), 0
    order = np.argsort(-c, kind="stable")
    cs = c[order]
    errors = np.cumsum(1.0 - y[order])  # sums of 0/1 values: exact
    last = np.append(np.flatnonzero(cs[1:] != cs[:-1]), cs.size - 1)  # last index of each tie block
    return last + 1, errors[last], int(c.size)


def risk_coverage(conf: Sequence[float], correct: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve at every distinct confidence threshold t, highest first: accept
    each answer with confidence >= t (ties together). Returns (coverage = accepted / n,
    risk = error rate among the accepted); one point per distinct confidence value, so
    with no ties this is k = 1..n."""
    k, err, n = _threshold_counts(conf, correct)
    if not n:
        return np.zeros(0), np.zeros(0)
    return k / n, err / k


def aurc(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage step curve, sum over thresholds of (coverage gained) x
    (risk at that threshold); lower is better. With no ties it is the mean risk over
    k = 1..n; a block of tied answers is accepted as a whole, so input order is irrelevant."""
    k, err, n = _threshold_counts(conf, correct)
    if not n:
        return NAN
    return float(np.sum(np.diff(k, prepend=0) * (err / k)) / n)


def risk_at_coverage(conf: Sequence[float], correct: Sequence[bool],
                     targets: Sequence[float]) -> list[dict[str, float]]:
    """For each target coverage, the first threshold operating point whose coverage is >=
    the target: {"coverage": target, "achieved_coverage", "risk"}. Thresholds sit at distinct
    confidences, so with ties the achieved coverage can exceed the target. Empty input
    gives nan risk and coverage."""
    cov, risk = risk_coverage(conf, correct)
    out = []
    for t in targets:
        if not cov.size:
            out.append({"coverage": float(t), "achieved_coverage": NAN, "risk": NAN})
            continue
        j = min(int(np.searchsorted(cov, float(t) - _EPS, side="left")), cov.size - 1)
        out.append({"coverage": float(t), "achieved_coverage": float(cov[j]), "risk": float(risk[j])})
    return out


# ----------------------------------------------------------------------------- bootstrap
class BootstrapCI(NamedTuple):
    """Percentile bootstrap interval plus its support: ``n_defined`` of the ``n_resamples``
    resamples gave a finite metric value (the CI is computed on those only)."""

    lo: float
    hi: float
    n_defined: int
    n_resamples: int

    @property
    def defined_share(self) -> float:
        return _div(self.n_defined, self.n_resamples)


def bootstrap_interval(metric_fn: Callable[[Any], float], data: Sequence[Any] | np.ndarray, n: int = 1000,
                       seed: int = 42, alpha: float = 0.05,
                       min_defined: float = MIN_DEFINED_SHARE) -> BootstrapCI:
    """Percentile bootstrap CI (default 95%) of ``metric_fn`` over ``data``, with the number
    of resamples on which the metric is defined.

    Each resample draws len(data) items with replacement; ``metric_fn`` receives the
    resample (a numpy array when ``data`` is one -- pass ``np.arange(n)`` and index your
    own arrays for speed -- else a list). A resample whose value is not finite (metric
    undefined, e.g. a precision with no predicted positive in the resample) is counted in
    ``n_resamples`` but not in ``n_defined``. When none, or fewer than ``min_defined`` x n,
    of the resamples are defined, the interval is (nan, nan): the estimate is too unstable
    for a CI. Otherwise the interval comes from the defined resamples only (conditional on
    the metric being defined), so callers must report ``n_defined`` with it.
    Deterministic for a given ``seed``."""
    items = data if isinstance(data, np.ndarray) else list(data)
    m = len(items)
    if m == 0 or n <= 0:
        return BootstrapCI(NAN, NAN, 0, max(int(n), 0))
    draws = np.random.default_rng(seed).integers(0, m, size=(n, m))
    vals = np.empty(n)
    for b in range(n):
        sample = items[draws[b]] if isinstance(items, np.ndarray) else [items[i] for i in draws[b]]
        vals[b] = float(metric_fn(sample))
    ok = np.isfinite(vals)
    k = int(ok.sum())
    if k == 0 or k < min_defined * n:
        return BootstrapCI(NAN, NAN, k, int(n))
    lo, hi = np.quantile(vals[ok], [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(float(lo), float(hi), k, int(n))


def bootstrap_ci(metric_fn: Callable[[Any], float], data: Sequence[Any] | np.ndarray, n: int = 1000,
                 seed: int = 42, alpha: float = 0.05, min_defined: float = MIN_DEFINED_SHARE) -> tuple[float, float]:
    """(low, high) of ``bootstrap_interval`` -- same rule: (nan, nan) when fewer than
    ``min_defined`` of the resamples are defined. Use ``bootstrap_interval`` to get the
    number of defined resamples, which must be reported whenever it is below n."""
    ci = bootstrap_interval(metric_fn, data, n=n, seed=seed, alpha=alpha, min_defined=min_defined)
    return ci.lo, ci.hi


# ----------------------------------------------------------------------------- OCR / segmentation
def levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Edit distance (insert / delete / substitute, unit cost) between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def _rate(dist: int, ref_len: int, hyp_len: int) -> float:
    if ref_len == 0:
        return 0.0 if hyp_len == 0 else 1.0
    return dist / ref_len


def cer(pred: str, ref: str) -> float:
    """Character error rate = edit distance / reference length (characters as given).
    An empty reference gives 0.0 when the prediction is empty too, else 1.0."""
    return _rate(levenshtein(pred, ref), len(ref), len(pred))


def wer(pred: str, ref: str) -> float:
    """Word error rate on whitespace tokens (same empty-reference rule as ``cer``)."""
    p, r = pred.split(), ref.split()
    return _rate(levenshtein(p, r), len(r), len(p))


def corpus_cer(preds: Sequence[str], refs: Sequence[str]) -> float:
    """Total character edits / total reference characters over a corpus."""
    if len(preds) != len(refs):
        raise ValueError("preds and refs differ in length")
    dist = sum(levenshtein(p, r) for p, r in zip(preds, refs))
    return _rate(dist, sum(len(r) for r in refs), sum(len(p) for p in preds))


def corpus_wer(preds: Sequence[str], refs: Sequence[str]) -> float:
    """Total word edits / total reference words over a corpus."""
    if len(preds) != len(refs):
        raise ValueError("preds and refs differ in length")
    ps, rs = [p.split() for p in preds], [r.split() for r in refs]
    dist = sum(levenshtein(p, r) for p, r in zip(ps, rs))
    return _rate(dist, sum(len(r) for r in rs), sum(len(p) for p in ps))


def segmentation_accuracy(pred: Mapping[Any, Any] | Sequence[Any], gold: Mapping[Any, Any] | Sequence[Any]) -> float:
    """Share of gold units assigned to the right question.

    Mappings (unit id -> question id): every gold unit counts; a unit missing from
    ``pred`` is wrong. Sequences: element-wise over equal-length lists. nan when gold is empty."""
    if isinstance(gold, Mapping):
        if not isinstance(pred, Mapping):
            raise TypeError("pred must be a mapping when gold is a mapping")
        return _div(sum(1 for k, v in gold.items() if k in pred and pred[k] == v), len(gold))
    if len(pred) != len(gold):
        raise ValueError(f"pred and gold differ in length: {len(pred)} != {len(gold)}")
    return _div(sum(1 for p, g in zip(pred, gold) if p == g), len(gold))
