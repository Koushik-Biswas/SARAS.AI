"""Grading benchmark runner (app.evaluation.grading) on a tiny synthetic benchmark built
here (never the real benchmark files). Assertions check structure and metric wiring,
not exact grading outputs, so they hold while the engine changes underneath."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from app.evaluation import grading as G
from app.evaluation import metrics as M

BENCH = {
    "version": 1,
    "provenance": "synthetic test fixture written by hand for tests/test_eval_runner.py",
    "questions": [
        {"id": "q1", "domain": "biology", "type": "short", "max_marks": 2,
         "prompt": "What does photosynthesis produce?",
         "model_answer": "Photosynthesis produces glucose. It also releases oxygen.",
         "key_points": [{"text": "Photosynthesis produces glucose", "weight": 1, "keywords": ["glucose"]},
                        {"text": "Photosynthesis releases oxygen", "weight": 1, "keywords": ["oxygen"]}]},
        {"id": "q2", "domain": "physics", "type": "long", "max_marks": 4,
         "prompt": "Explain what happens to water when it is heated at sea level.",
         "model_answer": "Water boils at 100 degrees Celsius at sea level. Boiling turns liquid water into steam. "
                         "Heating increases the kinetic energy of the molecules.",
         "key_points": [{"text": "Water boils at 100 degrees Celsius at sea level", "weight": 2, "keywords": []},
                        {"text": "Boiling turns liquid water into steam", "weight": 1, "keywords": []},
                        {"text": "Heating increases the kinetic energy of the molecules", "weight": 1,
                         "keywords": []}]},
    ],
    "answers": [
        {"id": "a1", "question_id": "q1", "category": "correct",
         "text": "Photosynthesis produces glucose. It also releases oxygen.",
         "expected_kp": ["matched", "matched"], "expected_marks": 2, "expected_flag": False, "ambiguous": False,
         "rationale": "model answer"},
        {"id": "a2", "question_id": "q1", "category": "partial", "text": "Plants make glucose from sunlight.",
         "expected_kp": ["matched", "missed"], "expected_marks": 1, "expected_flag": False, "ambiguous": False,
         "rationale": "oxygen missing"},
        {"id": "a3", "question_id": "q1", "category": "blank", "text": "",
         "expected_kp": ["missed", "missed"], "expected_marks": 0, "expected_flag": False, "ambiguous": False,
         "rationale": "empty"},
        {"id": "a4", "question_id": "q2", "category": "correct",
         "text": "Water boils at 100 degrees Celsius at sea level. Boiling turns liquid water into steam. "
                 "Heating increases the kinetic energy of the molecules.",
         "expected_kp": ["matched", "matched", "matched"], "expected_marks": 4, "expected_flag": False,
         "ambiguous": False, "rationale": "model answer"},
        {"id": "a5", "question_id": "q2", "category": "contradiction",
         "text": "Water does not boil at 100 degrees Celsius at sea level. Boiling turns liquid water into steam.",
         "expected_kp": ["contradicted", "matched", "missed"], "expected_marks": 1, "expected_flag": True,
         "ambiguous": False, "rationale": "negated fact"},
        {"id": "a6", "question_id": "q2", "category": "partial",
         "text": "When heated the water molecules move faster and it becomes steam.",
         "expected_kp": ["missed", "partial", "partial"], "expected_marks": 1, "expected_flag": True,
         "ambiguous": True, "rationale": "loose paraphrase"},
    ],
}
N_ANSWERS = len(BENCH["answers"])
RESULT_KEYS = {"grader", "answer_id", "question_id", "category", "question_type", "ambiguous", "max_marks",
               "expected_marks", "predicted_marks", "abs_error", "correct", "expected_kp", "predicted_kp",
               "predicted_kp_reasons", "kp_comparable", "expected_flag", "review", "flags", "grading_confidence", "confidence",
               "verdict", "grader_reported", "error"}


class OracleGrader(G.Grader):
    """Returns the benchmark labels: every metric must come out perfect."""

    def __init__(self, name: str, ctx: G.RunContext) -> None:
        self.name = name
        self.by_text = {a["text"]: a for a in BENCH["answers"]}

    def grade(self, question, text):  # noqa: ANN001, ANN201
        a = self.by_text[text]
        return G.Prediction(marks=float(a["expected_marks"]), kp_statuses=list(a["expected_kp"]),
                            grading_confidence=1.0, confidence=1.0, review=a["expected_flag"],
                            grader_reported="oracle")


class BrokenGrader(G.Grader):
    def __init__(self, name: str, ctx: G.RunContext) -> None:
        self.name = name

    def grade(self, question, text):  # noqa: ANN001, ANN201
        raise RuntimeError("model exploded")


@pytest.fixture
def bench_file(tmp_path: Path) -> Path:
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(BENCH), encoding="utf-8")
    return p


@pytest.fixture
def settings():  # noqa: ANN201  (overrides the conftest fixture: no app/DB needed here)
    from app.config import Settings

    return Settings(env="test", grader="lexical")


def _registry() -> dict:
    reg = dict(G.GRADERS)
    reg["oracle"] = OracleGrader
    reg["broken"] = BrokenGrader
    reg["missing-module"] = lambda name, ctx: G.ModuleGrader(name, "app.grading._no_such_grader_module", ctx)
    return reg


def _read_results(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]


def _group(report: dict, key: str) -> dict:
    return next(g for g in report["groups"] if g["key"] == key)


def _strip_timing(report: dict) -> dict:
    return {**report, "graders": [{k: v for k, v in g.items() if k not in ("timing", "scorer_stats")}
                                  for g in report["graders"]]}


def test_runner_writes_results_and_reports(bench_file, tmp_path, settings):
    out = tmp_path / "run"
    graders = ["lexical", "baseline-formula", "oracle", "broken", "missing-module"]
    report = G.run_evaluation(bench_file, graders, out, run_label="unit test", timestamp="2026-01-01T00:00:00Z",
                              n_boot=200, settings=settings, registry=_registry())

    # --- files and per-answer lines (every grader, every answer; unavailable ones included)
    assert {p.name for p in out.iterdir()} >= {"results.jsonl", "report.json", "report.md"}
    rows = _read_results(out)
    assert len(rows) == N_ANSWERS * len(graders)
    assert all(set(r) == RESULT_KEYS for r in rows)
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == report

    # --- lexical goes through the real engine: structure only
    lex = [r for r in rows if r["grader"] == "lexical"]
    assert all(r["error"] is None for r in lex)
    for r in lex:
        assert 0 <= r["predicted_marks"] <= r["max_marks"]
        assert len(r["predicted_kp"]) == len(r["expected_kp"]) and set(r["predicted_kp"]) <= set(M.STATUSES)
        assert isinstance(r["review"], bool) and 0 <= r["confidence"] <= 1
        assert r["correct"] == (abs(r["predicted_marks"] - r["expected_marks"]) <= 0.5)

    # --- availability is reported, never faked
    info = {g["name"]: g for g in report["graders"]}
    assert info["lexical"]["status"] == "available" and info["lexical"]["n_scored"] == N_ANSWERS
    assert info["lexical"]["thresholds_mode"] == "engine" and report["config"]["thresholds_mode"] == "engine"
    assert {"tau", "tau_low", "contradiction"} <= set(info["lexical"]["thresholds"])
    assert info["missing-module"]["status"].startswith("unavailable: module app.grading._no_such_grader_module")
    assert info["broken"]["status"].startswith("failed:") and info["broken"]["n_errors"] == N_ANSWERS
    assert info["broken"]["errors"][0]["error"] == "RuntimeError: model exploded"
    for name in ("missing-module", "broken"):
        assert all(r["predicted_marks"] is None and r["error"] for r in rows if r["grader"] == name)
    bf = info["baseline-formula"]["status"]
    assert bf == "available" or bf.startswith("unavailable: ")

    # --- groups, N and metric wiring (recompute lexical MAE / QWK from results.jsonl)
    keys = [g["key"] for g in report["groups"]]
    assert keys[:2] == ["overall", "overall_unambiguous"]
    assert {"category=correct", "category=partial", "category=blank", "category=contradiction",
            "type=short", "type=long"} <= set(keys)
    overall = _group(report, "overall")
    assert overall["n_answers"] == N_ANSWERS and _group(report, "overall_unambiguous")["n_answers"] == N_ANSWERS - 1
    assert _group(report, "type=long")["n_answers"] == 3
    lx = overall["graders"]["lexical"]
    p = [r["predicted_marks"] for r in lex]
    g = [r["expected_marks"] for r in lex]
    assert lx["metrics"]["mae"]["value"] == pytest.approx(M.mae(p, g))
    assert lx["metrics"]["mae"]["n"] == N_ANSWERS
    qwk = M.quadratic_weighted_kappa(p, g)
    assert (lx["metrics"]["qwk"]["value"] is None) if math.isnan(qwk) else lx["metrics"]["qwk"]["value"] == pytest.approx(qwk)
    assert lx["n_kp"] == sum(len(r["expected_kp"]) for r in lex)
    assert np.array(lx["kp_confusion"]["matrix"]).sum() == lx["n_kp"]
    for name, m in lx["metrics"].items():
        lo, hi = m["ci95"]
        assert lo is None or hi is None or lo <= hi, name
    assert {"mae", "rmse", "within_0.5", "within_1", "qwk", "pearson", "kp_kappa", "awarded_f1",
            "false_contradiction_rate", "flag_rate", "ece", "brier"} <= set(lx["metrics"])
    assert len(lx["reliability_bins"]) == 10

    # --- the oracle is perfect on every metric the harness computes
    om = overall["graders"]["oracle"]["metrics"]
    assert om["mae"]["value"] == 0 and om["mae"]["ci95"] == [0, 0]
    assert om["qwk"]["value"] == pytest.approx(1.0) and om["kp_kappa"]["value"] == pytest.approx(1.0)
    assert om["awarded_f1"]["value"] == pytest.approx(1.0) and om["contradiction_recall"]["value"] == 1.0
    assert om["false_contradiction_rate"]["value"] == 0 and om["error_rate_unflagged"]["value"] == 0
    assert om["flag_precision"]["value"] == 1.0 and om["flag_recall"]["value"] == 1.0
    assert om["flag_rate"]["value"] == pytest.approx(2 / 6) and om["ece"]["value"] == pytest.approx(0.0)
    assert overall["graders"]["broken"]["metrics"] == {} and overall["graders"]["broken"]["n_errors"] == N_ANSWERS

    # --- report metadata and markdown
    assert report["run_label"] == "unit test" and report["timestamp"] == "2026-01-01T00:00:00Z"
    assert report["benchmark"]["provenance"] == BENCH["provenance"]
    assert report["benchmark"]["n_answers_evaluated"] == N_ANSWERS and report["benchmark"]["warnings"] == []
    boot = report["method"]["bootstrap"]
    assert (boot["resamples"], boot["seed"], boot["interval"]) == (200, 42, "95% percentile")
    # breakdown groups: CIs for the table columns, point values for the rest
    short = _group(report, "type=short")["graders"]["lexical"]["metrics"]
    assert short["mae"]["ci95"] is not None and short["rmse"]["ci95"] is None and short["rmse"]["value"] is not None
    md = (out / "report.md").read_text(encoding="utf-8")
    for needle in ("unit test", "2026-01-01T00:00:00Z", BENCH["provenance"], "unavailable: module",
                   "By category", "By question type", "QWK"):
        assert needle in md


def test_runner_is_deterministic(bench_file, tmp_path, settings):
    kw = dict(n_boot=100, settings=settings, registry=_registry(), run_label="r")
    a = G.run_evaluation(bench_file, ["lexical", "oracle"], tmp_path / "a", **kw)
    b = G.run_evaluation(bench_file, ["lexical", "oracle"], tmp_path / "b", **kw)
    assert _strip_timing(a) == _strip_timing(b)
    assert (tmp_path / "a" / "results.jsonl").read_text() == (tmp_path / "b" / "results.jsonl").read_text()
    assert a["timestamp"] is None  # no hidden clock: only what the caller passes


def test_cli_filters_and_limit(bench_file, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "GRADERS", {**G.GRADERS, "oracle": OracleGrader})
    out = tmp_path / "cli"
    code = G.main(["--data", str(bench_file), "--graders", "oracle,lexical", "--out", str(out),
                   "--categories", "correct,partial", "--limit", "3", "--bootstrap", "50",
                   "--run-label", "cli", "--timestamp", "T0", "--thresholds", "settings"])
    assert code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    lex = next(g for g in report["graders"] if g["name"] == "lexical")
    assert lex["thresholds_mode"] == "settings" and lex["thresholds"] == report["config"]["settings_thresholds"]
    assert report["benchmark"]["n_answers_evaluated"] == 3
    assert report["benchmark"]["filters"] == {"categories": ["correct", "partial"], "limit": 3}
    assert {r["category"] for r in _read_results(out)} <= {"correct", "partial"}
    assert len(_read_results(out)) == 3 * 2
    assert G.main(["--data", str(bench_file), "--graders", "nope", "--out", str(out)]) == 2
    assert G.main(["--data", str(bench_file), "--graders", "lexical", "--out", str(out),
                   "--categories", "no-such-category"]) == 2


def test_invalid_benchmark_is_rejected_and_label_inconsistency_warned(tmp_path, settings):
    bad = json.loads(json.dumps(BENCH))
    bad["answers"][0]["expected_kp"] = ["matched"]  # q1 has two key points
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(G.BenchmarkError, match="expected_kp"):
        G.load_benchmark(p)

    odd = json.loads(json.dumps(BENCH))
    odd["answers"][1]["expected_marks"] = 1.5  # rule gives 1.0 for matched + missed
    p2 = tmp_path / "odd.json"
    p2.write_text(json.dumps(odd), encoding="utf-8")
    bench = G.load_benchmark(p2)
    assert any("a2" in w and "partial-credit rule" in w for w in bench.warnings)


def test_engine_grader_rejects_a_substituted_scorer(settings):
    """If get_scorer(kind) hands back a different scorer, the grader is unavailable (not faked)."""
    ctx = G.RunContext(settings=settings, thresholds=G.thresholds_from_settings(settings), offline_scorer="lexical")

    class Ctx(G.RunContext):
        def scorer(self, kind):  # noqa: ANN001, ANN201
            return super().scorer("lexical")

    fake = Ctx(**{f: getattr(ctx, f) for f in ("settings", "thresholds", "offline_scorer")})
    reason = G.EngineGrader("nli-onnx", "nli-onnx", fake).availability()
    assert reason and "returned the 'lexical' scorer" in reason
    th = G.thresholds_from_settings(settings)
    assert (th.tau, th.tau_low, th.contradiction) == (settings.tau, settings.tau_low, settings.contradiction_threshold)
