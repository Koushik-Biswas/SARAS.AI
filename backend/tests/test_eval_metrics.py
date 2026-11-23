"""Evaluation metrics (app.evaluation.metrics) against hand-computed examples."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.evaluation import metrics as M


def test_mae_rmse_within_pearson_by_hand():
    pred, gold = [1, 2, 3, 4], [1, 3, 3, 2]  # |errors| = 0, 1, 0, 2
    assert M.mae(pred, gold) == pytest.approx(0.75)
    assert M.rmse(pred, gold) == pytest.approx(math.sqrt(5 / 4))
    assert M.within(pred, gold, 0.5) == pytest.approx(0.5)
    assert M.within(pred, gold, 1.0) == pytest.approx(0.75)  # tolerance is inclusive
    # deviations p: -1.5 -.5 .5 1.5 ; g: -1.25 .75 .75 -.25 -> cov 1.5, ss 5 and 2.75
    assert M.pearson(pred, gold) == pytest.approx(1.5 / math.sqrt(5 * 2.75))
    assert math.isnan(M.pearson([1, 2, 3], [2, 2, 2]))  # zero variance: undefined, not 0
    assert math.isnan(M.mae([], []))
    with pytest.raises(ValueError):
        M.mae([1, 2], [1])


def test_bin_marks_rounds_half_up_on_half_mark_grid():
    assert M.bin_marks([0, 0.24, 0.25, 0.26, 0.5, 0.74, 0.75, 3.0]).tolist() == [0, 0, 1, 1, 1, 1, 2, 6]


def test_quadratic_weighted_kappa_hand_computed():
    # ratings (0.5-mark bins) gold 0,1,2,2 and pred 0,2,2,1: sum W*O = 0.5, sum W*E = 1.375 -> 1 - 0.5/1.375 = 7/11
    assert M.quadratic_weighted_kappa([0, 1, 1, 0.5], [0, 0.5, 1, 1]) == pytest.approx(7 / 11)
    assert M.quadratic_weighted_kappa([0, 0.5, 1], [0, 0.5, 1]) == pytest.approx(1.0)
    assert M.quadratic_weighted_kappa([1, 0.5, 0], [0, 0.5, 1]) == pytest.approx(-1.0)  # full reversal
    # marks are binned first: 0.9 -> 2 (1.0 marks), 0.3 -> 1 (0.5 marks), so this is perfect agreement
    assert M.quadratic_weighted_kappa([0.9, 0.3, 0.0], [1.0, 0.5, 0.0]) == pytest.approx(1.0)
    assert math.isnan(M.quadratic_weighted_kappa([2, 2], [2, 2]))  # one rating only: undefined


def _qwk_full_matrix(pred, gold, step):
    """Textbook QWK, 1 - sum(W*O) / sum(W*E) over the full rating matrix (independent check)."""
    rp, rg = M.bin_marks(pred, step), M.bin_marks(gold, step)
    lo, k = min(rp.min(), rg.min()), max(rp.max(), rg.max()) - min(rp.min(), rg.min()) + 1
    obs = np.zeros((k, k))
    for a, b in zip(rg - lo, rp - lo):
        obs[a, b] += 1
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    return 1 - (w * obs).sum() / (w * exp).sum()


def test_pooled_absolute_qwk_is_scale_confounded_and_normalised_qwk_is_not():
    """Review finding harness-onnx #1: full marks on 2-, 5- and 8-mark questions graded as half
    marks everywhere got pooled QWK 0.436 and Pearson 1.0 on absolute marks."""
    gold, pred, mx = [2, 5, 8], [1, 2.5, 4], [2, 5, 8]
    # ratings gold 4,10,16 / pred 2,5,8: sum W*O = 93, sum W*E = 372 + 93 - 2*30*15/3 = 165
    assert M.quadratic_weighted_kappa(pred, gold) == pytest.approx(24 / 55)  # the confounded value
    assert M.pearson(pred, gold) == pytest.approx(1.0)
    # fractions: gold 1,1,1 -> bin 10; pred .5,.5,.5 -> bin 5: no agreement beyond chance
    assert M.qwk_normalised(pred, gold, mx) == 0.0
    assert math.isnan(M.pearson_normalised(pred, gold, mx))  # constant fractions: undefined, not 1
    # A predictor that knows only the scale (max/2 everywhere) gets a positive pooled QWK ...
    mx = [2, 8, 2, 8]
    gold, pred = [0, 8, 2, 0], [1, 4, 1, 4]
    # ratings gold 0,16,4,0 / pred 2,8,2,8: sum W*O = 136, sum W*E = 272 + 136 - 200 = 208
    assert M.quadratic_weighted_kappa(pred, gold) == pytest.approx(9 / 26)
    # ... and exactly 0 on the fraction-of-max scale.
    assert M.qwk_normalised(pred, gold, mx) == 0.0


def test_qwk_normalised_and_pearson_normalised_hand_computed():
    mx, gold, pred = [2, 4, 5, 10], [1, 4, 5, 0], [1, 2, 5, 1]
    # fractions gold .5,1,1,0 -> bins 5,10,10,0 ; pred .5,.5,1,.1 -> bins 5,5,10,1
    # sum W*O = 0 + 25 + 0 + 1 = 26 ; sum W*E = 225 + 151 - 2*25*21/4 = 113.5 -> 1 - 26/113.5
    assert M.qwk_normalised(pred, gold, mx) == pytest.approx(175 / 227)
    frac_p, frac_g = np.divide(pred, mx), np.divide(gold, mx)
    assert M.qwk_normalised(pred, gold, mx) == pytest.approx(_qwk_full_matrix(frac_p, frac_g, 0.1))
    # 0.1-step bins, round half up (float noise in 0.35 / 0.1 = 3.4999... is absorbed)
    assert M.bin_marks([0.05, 0.25, 0.35, 0.875, 1.0], M.FRACTION_STEP).tolist() == [1, 3, 4, 9, 10]
    # pearson on fractions gold 1,0,.5 / pred .5,.25,1: cov .125, ss .5 and 7/24 -> sqrt(3/28)
    assert M.pearson_normalised([1, 1, 8], [2, 0, 4], [2, 4, 8]) == pytest.approx(math.sqrt(3 / 28))
    assert M.normalise_marks([1, 3], [2, 4]).tolist() == [0.5, 0.75]
    with pytest.raises(ValueError):
        M.normalise_marks([1, 2], [2, 0])  # a max of 0 marks cannot be normalised
    with pytest.raises(ValueError):
        M.qwk_normalised([1, 2], [1, 2], [2])


def test_qwk_per_question_hand_computed():
    qids = ["A"] * 5 + ["B"] * 5 + ["C"] * 4 + ["D"] * 5 + ["E"] * 5
    gold = [0, .5, 1, 1.5, 2] + [4] * 5 + [0, 1, 2, 3] + [0, 1, 2, 3, 4] + [0, 1, 1, 2, 2]
    pred = [0, .5, 1, 1.5, 2] + [4, 3, 4, 4, 4] + [0, 1, 2, 3] + [2] * 5 + [0, 1, 2, 2, 1]
    per = M.qwk_per_question(pred, gold, qids)
    # A perfect; B left out (constant gold: QWK would be 0 for any prediction); C left out
    # (4 answers < 5); D constant prediction -> 0, not nan; E: ratings gold 0,2,2,4,4 / pred
    # 0,2,4,4,2: sum W*O = 8, sum W*E = 40 + 40 - 2*12*12/5 = 22.4 -> 1 - 8/22.4 = 9/14
    assert list(per) == ["A", "D", "E"]
    assert per["A"] == pytest.approx(1.0) and per["D"] == 0.0 and per["E"] == pytest.approx(9 / 14)
    assert per["E"] == pytest.approx(_qwk_full_matrix(pred[-5:], gold[-5:], M.MARK_STEP))
    assert M.qwk_per_question_mean(pred, gold, qids) == pytest.approx((1 + 0 + 9 / 14) / 3)
    # Inclusion depends on gold only: another predictor is judged on the same questions.
    assert list(M.qwk_per_question([0] * len(gold), gold, qids)) == ["A", "D", "E"]
    assert math.isnan(M.qwk_per_question_mean([1, 2], [1, 2], ["A", "A"]))  # nothing qualifies
    assert M.qwk_per_question([1, 2, 3], [1, 2, 3], ["A"] * 3, min_answers=3) == {"A": pytest.approx(1.0)}
    with pytest.raises(ValueError):
        M.qwk_per_question([1, 2], [1, 2], ["A"])


def test_cohen_kappa_textbook_and_status_example():
    # Wikipedia example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po 0.7, pe 0.5, kappa 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert M.cohen_kappa(a, b) == pytest.approx(0.4)
    gold = ["matched", "matched", "missed", "missed"]
    pred = ["matched", "missed", "missed", "missed"]  # po .75, pe (2*1 + 2*3)/16 = .5 -> kappa .5
    assert M.cohen_kappa(gold, pred, M.STATUSES) == pytest.approx(0.5)
    assert M.cohen_kappa(gold, gold) == pytest.approx(1.0)
    assert math.isnan(M.cohen_kappa(["missed"] * 3, ["missed"] * 3))


def test_confusion_awarded_prf_and_contradiction_metrics():
    gold = ["matched", "partial", "missed", "contradicted", "missed"]
    pred = ["matched", "missed", "partial", "missed", "contradicted"]
    cm = M.confusion_matrix(gold, pred)
    assert cm.tolist() == [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    # awarded = matched|partial: TP idx0, FP idx2, FN idx1
    assert M.awarded_prf(gold, pred) == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5})
    c = M.contradiction_metrics(gold, pred)
    assert c["precision"] == 0.0 and c["recall"] == 0.0
    assert c["false_contradiction_rate"] == pytest.approx(1 / 4)  # 1 of 4 non-contradicted KPs flipped
    assert M.accuracy_from_confusion(cm) == pytest.approx(1 / 5)
    nothing = M.contradiction_metrics(["missed"], ["missed"])
    assert math.isnan(nothing["precision"]) and math.isnan(nothing["recall"])
    assert nothing["false_contradiction_rate"] == 0.0
    with pytest.raises(ValueError):
        M.confusion_matrix(["matched"], ["bogus"])


def test_deferral_metrics():
    pred, gold = [1, 2, 3], [1, 3, 3]
    assert M.correct_mask([1, 1.5, 2.01], [1.5, 1, 1.5]).tolist() == [True, True, False]  # 0.5 counts as correct
    assert M.error_rate_unflagged(pred, gold, [False, False, True]) == pytest.approx(0.5)
    assert math.isnan(M.error_rate_unflagged(pred, gold, [True, True, True]))
    assert M.flag_rate([True, False, False, True]) == pytest.approx(0.5)
    assert M.binary_prf([True, True, False, False], [True, False, True, False]) == pytest.approx(
        {"precision": 0.5, "recall": 0.5, "f1": 0.5})


def test_ece_brier_reliability_by_hand():
    conf, correct = [0.9, 0.9, 0.2, 0.2], [1, 0, 0, 0]
    # bin 0.9: acc .5 vs .9 -> .4 * 2/4 ; bin 0.2: acc 0 vs .2 -> .2 * 2/4  => 0.3
    assert M.ece(conf, correct) == pytest.approx(0.3)
    assert M.brier(conf, correct) == pytest.approx((0.01 + 0.81 + 0.04 + 0.04) / 4)
    assert M.ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]) == pytest.approx(0.0)
    bins = M.reliability_bins([1.0, 0.95, 0.05], [1, 1, 0])
    assert len(bins) == 10 and bins[9]["n"] == 2 and bins[0]["n"] == 1  # 1.0 lands in the last bin
    assert bins[9]["mean_confidence"] == pytest.approx(0.975) and math.isnan(bins[5]["accuracy"])


def test_risk_coverage_and_aurc():
    cov, risk = M.risk_coverage([0.9, 0.1, 0.8], [1, 0, 1])
    assert cov.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert risk.tolist() == pytest.approx([0.0, 0.0, 1 / 3])
    assert M.aurc([0.9, 0.1, 0.8], [1, 0, 1]) == pytest.approx(1 / 9)


def test_risk_coverage_and_aurc_ignore_input_order_on_tied_confidences():
    """Review finding harness-onnx #6: four answers tied at 0.6 gave AURC 0.208 as [1,1,0,0]
    and 0.792 as [0,0,1,1] (ties were broken by input order)."""
    for correct in ([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]):
        cov, risk = M.risk_coverage([0.6] * 4, correct)
        assert cov.tolist() == [1.0] and risk.tolist() == [0.5]  # one threshold, accepted as a block
        assert M.aurc([0.6] * 4, correct) == 0.5
    # Hand-computed with a tie block: thresholds .9 (1 accepted, 0 errors), .8 (4, 2), .1 (5, 2)
    conf = [0.9, 0.8, 0.8, 0.8, 0.1]
    for correct in ([1, 1, 0, 0, 1], [1, 0, 0, 1, 1]):
        cov, risk = M.risk_coverage(conf, correct)
        assert cov.tolist() == pytest.approx([0.2, 0.8, 1.0]) and risk.tolist() == pytest.approx([0.0, 0.5, 0.4])
        assert M.aurc(conf, correct) == pytest.approx(0.2 * 0.0 + 0.6 * 0.5 + 0.2 * 0.4)
    # Any permutation of the same (confidence, correctness) pairs gives exactly the same result.
    rng = np.random.default_rng(7)
    c = rng.choice([0.6, 0.8, 0.95, 0.9999], size=40)
    y = rng.random(40) < 0.6
    ref_aurc, (ref_cov, ref_risk) = M.aurc(c, y), M.risk_coverage(c, y)
    for _ in range(20):
        perm = rng.permutation(40)
        assert M.aurc(c[perm], y[perm]) == ref_aurc
        cov, risk = M.risk_coverage(c[perm], y[perm])
        assert cov.tolist() == ref_cov.tolist() and risk.tolist() == ref_risk.tolist()
    assert len(ref_cov) == len(set(c.tolist()))  # one point per distinct confidence
    assert math.isnan(M.aurc([], []))


def test_risk_at_coverage_uses_threshold_operating_points():
    conf, correct = [0.9, 0.8, 0.8, 0.8, 0.1], [1, 1, 0, 0, 1]
    pts = M.risk_at_coverage(conf, correct, [0.1, 0.2, 0.5, 0.8, 1.0])
    assert [p["coverage"] for p in pts] == [0.1, 0.2, 0.5, 0.8, 1.0]
    # coverage 0.5 is not reachable by a threshold: the next one accepts the whole 0.8 block
    assert [p["achieved_coverage"] for p in pts] == pytest.approx([0.2, 0.2, 0.8, 0.8, 1.0])
    assert [p["risk"] for p in pts] == pytest.approx([0.0, 0.0, 0.5, 0.5, 0.4])
    # without ties it is the old "k = ceil(c * n) most confident answers"
    pts = M.risk_at_coverage([0.9, 0.1, 0.8], [1, 0, 1], [0.3, 0.5, 1.0])
    assert [p["risk"] for p in pts] == pytest.approx([0.0, 0.0, 1 / 3])
    empty = M.risk_at_coverage([], [], [0.5])[0]
    assert math.isnan(empty["risk"]) and math.isnan(empty["achieved_coverage"])


def test_bootstrap_interval_counts_defined_resamples_and_drops_unstable_cis():
    """Review finding harness-onnx #5: undefined resamples were dropped silently, e.g. one
    correct predicted contradiction among 42 answers gave contradiction precision CI
    [1.0, 1.0] from the ~64% of resamples that contain it, with nothing reported."""
    has_contradiction = np.zeros(42)
    has_contradiction[0] = 1

    def precision(i):
        predicted = has_contradiction[i].sum()
        return 1.0 if predicted else float("nan")  # the one prediction is correct

    ci = M.bootstrap_interval(precision, np.arange(42), n=1000, seed=42)
    # P(resample contains answer 0) = 1 - (41/42)^42 = 0.636: the CI rests on those only, and says so
    assert (ci.lo, ci.hi) == (1.0, 1.0) and ci.n_resamples == 1000
    assert 580 <= ci.n_defined <= 700 and ci.defined_share == ci.n_defined / 1000
    assert M.bootstrap_ci(precision, np.arange(42), n=1000, seed=42) == (ci.lo, ci.hi)

    # Fewer than 50% of the resamples defined -> no CI (nan), never an interval from the few left.
    values = iter([1.0] * 499 + [float("nan")] * 501)
    ci = M.bootstrap_interval(lambda s: next(values), [1, 2], n=1000)
    assert math.isnan(ci.lo) and math.isnan(ci.hi) and ci.n_defined == 499
    values = iter([1.0] * 499 + [float("nan")] * 501)
    assert all(math.isnan(v) for v in M.bootstrap_ci(lambda s: next(values), [1, 2], n=1000))
    # Exactly half defined is not "fewer than half": the CI is kept, with its support.
    values = iter([1.0] * 500 + [float("nan")] * 500)
    assert M.bootstrap_interval(lambda s: next(values), [1, 2], n=1000) == (1.0, 1.0, 500, 1000)
    # Fully defined metrics are unchanged: same draws, same interval as before, all defined.
    idx = np.arange(30)
    marks = np.linspace(0, 3, 30)
    full = M.bootstrap_interval(lambda i: float(marks[i].mean()), idx, n=200, seed=3)
    assert full.n_defined == 200 and full.lo < full.hi
    empty = M.bootstrap_interval(lambda s: 1.0, [], n=10)
    assert math.isnan(empty.lo) and math.isnan(empty.hi) and (empty.n_defined, empty.n_resamples) == (0, 10)


def test_runner_summary_wires_normalised_qwk_ci_support_and_tied_risk_coverage():
    """app.evaluation.grading.summarise: headline QWK on the fraction-of-max scale, the absolute
    pooled value kept (under its old name too), each CI's resample support recorded, and
    risk-coverage points at threshold operating points (findings #1, #5, #6)."""
    from app.evaluation import grading as G

    questions = [{"id": "qa", "type": "short", "max_marks": 2}, {"id": "qb", "type": "long", "max_marks": 8}]
    records = []
    for k in range(42):
        q = questions[k % 2]
        gold = [0, 0.5, 1, 1.5, 2][k % 5] * q["max_marks"] / 2
        pred = min(q["max_marks"], gold + (0.5 if k % 3 == 0 else 0.0))
        exp_kp = ["contradicted" if k == 0 else "matched"] + ["matched"] * 3
        ans = {"id": f"a{k}", "question_id": q["id"], "category": "c", "expected_kp": exp_kp,
               "expected_marks": gold, "expected_flag": False}
        pred_obj = G.Prediction(marks=pred, kp_statuses=list(exp_kp), grading_confidence=0.8,
                                confidence=0.8 if k % 4 else 0.95, review=False, grader_reported="t")
        records.append(G._record(ans, q, "t", pred_obj))
    s = G.summarise(records, n_boot=300, seed=42, detail=True)
    m = s["metrics"]
    p = [r["predicted_marks"] for r in records]
    g = [r["expected_marks"] for r in records]
    mx = [r["max_marks"] for r in records]
    qid = [r["question_id"] for r in records]
    assert m["qwk_normalised"]["value"] == pytest.approx(M.qwk_normalised(p, g, mx))
    assert m["pearson_normalised"]["value"] == pytest.approx(M.pearson_normalised(p, g, mx))
    assert m["qwk_per_question_mean"]["value"] == pytest.approx(M.qwk_per_question_mean(p, g, qid))
    assert m["qwk_per_question_mean"]["n"] == 2  # both questions qualify (21 answers, varied gold)
    assert [(x["question_id"], x["n"]) for x in s["qwk_per_question"]] == [("qa", 21), ("qb", 21)]
    assert m["qwk_absolute_pooled"]["value"] == pytest.approx(M.quadratic_weighted_kappa(p, g))
    assert m["qwk"] == m["qwk_absolute_pooled"]  # old key, same value
    # One correct predicted contradiction among 42 answers: CI [1, 1] from ~64% of the resamples,
    # and the report says so (not silently).
    cp = m["contradiction_precision"]
    assert cp["value"] == 1.0 and cp["ci95"] == [1.0, 1.0] and cp["ci95_resamples"] == 300
    assert 0.5 * 300 <= cp["ci95_defined"] < 300
    assert G._m(s, "contradiction_precision").endswith(f" d={math.floor(100 * cp['ci95_defined'] / 300)}%")
    assert m["mae"]["ci95_defined"] == m["mae"]["ci95_resamples"] == 300 and "d=" not in G._m(s, "mae")
    # Ties at confidence 0.8: the 0.1 target is reached at the 0.95 threshold, 0.5 needs the 0.8 block.
    cov = {pt["coverage"]: pt for pt in s["risk_coverage"]}
    assert cov[0.1]["achieved_coverage"] == pytest.approx(11 / 42) and cov[0.5]["achieved_coverage"] == 1.0
    assert m["aurc"]["value"] == pytest.approx(M.aurc([r["confidence"] for r in records],
                                                      [r["correct"] for r in records]))


def test_bootstrap_ci_contains_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 11, size=60) / 2
    pred = np.clip(gold + rng.normal(0, 0.7, size=60), 0, 5)
    idx = np.arange(60)
    fn = lambda i: M.mae(pred[i], gold[i])  # noqa: E731
    lo, hi = M.bootstrap_ci(fn, idx, n=500, seed=42)
    assert lo <= fn(idx) <= hi and lo < hi
    assert M.bootstrap_ci(fn, idx, n=500, seed=42) == (lo, hi)
    # list data works too and a constant metric gives a degenerate interval
    assert M.bootstrap_ci(lambda s: 1.0, [1, 2, 3], n=50) == (1.0, 1.0)
    assert all(math.isnan(v) for v in M.bootstrap_ci(fn, np.arange(0), n=10))
    assert all(math.isnan(v) for v in M.bootstrap_ci(lambda s: float("nan"), [1, 2], n=10))


def test_cer_wer_levenshtein():
    assert M.levenshtein("kitten", "sitting") == 3
    assert M.levenshtein("", "abc") == 3 and M.levenshtein("abc", "abc") == 0
    assert M.cer("kitten", "sitting") == pytest.approx(3 / 7)
    assert M.wer("the cat sit on mat", "the cat sat on the mat") == pytest.approx(2 / 6)
    assert M.cer("", "") == 0.0 and M.cer("x", "") == 1.0
    assert M.corpus_cer(["kitten", "abc"], ["sitting", "abc"]) == pytest.approx(3 / 10)
    assert M.corpus_wer(["a b", "c"], ["a c", "c"]) == pytest.approx(1 / 3)


def test_segmentation_accuracy():
    gold = {"l1": "Q1", "l2": "Q1", "l3": "Q2", "l4": "Q3"}
    pred = {"l1": "Q1", "l2": "Q2", "l3": "Q2"}  # l4 missing counts as wrong
    assert M.segmentation_accuracy(pred, gold) == pytest.approx(0.5)
    assert M.segmentation_accuracy(["Q1", "Q2", "Q2"], ["Q1", "Q2", "Q3"]) == pytest.approx(2 / 3)
    assert math.isnan(M.segmentation_accuracy([], []))
    with pytest.raises(ValueError):
        M.segmentation_accuracy(["Q1"], ["Q1", "Q2"])
