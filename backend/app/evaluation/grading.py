"""Grading benchmark runner (PRD 3.7 Q-2).

    cd backend
    .venv/Scripts/python.exe -m app.evaluation.grading \\
        --data evaluation/data/grading_benchmark_dev.json \\
        --graders lexical,nli-onnx,baseline-formula --out evaluation/runs/dev \\
        --run-label "dev, engine v2" --timestamp 2026-11-23T18:00Z

Writes to ``--out``:
* ``results.jsonl`` -- one line per answer per grader: expected and predicted marks and
  key-point (KP) statuses, flags, confidence, review decision, verdict, error.
* ``report.json`` / ``report.md`` -- overall, overall-without-ambiguous, per-category and
  per-question-type metrics with bootstrap 95% CIs and N, grader availability, the
  caller's run label / timestamp and the benchmark provenance note.

Rules
* The harness calls the production code paths -- ``engine.grade_answer`` with
  ``scorers.get_scorer``, ``engine.baseline_formula_marks``, the llm / ensemble modules,
  and ``pipeline.routing.combine/route`` for the review decision. It never re-implements
  grading. (``CREDIT`` below is only used to sanity-check the benchmark's own labels.)
* A grader that cannot run is reported as ``unavailable: <reason>`` (its results.jsonl
  lines carry that error); per-answer exceptions are recorded as errors and excluded
  from the metrics, with the counts reported. Nothing is skipped silently or faked.
* Deterministic: fixed bootstrap seed; the report holds no wall-clock time except the
  ``--timestamp`` the caller passes. Grading durations are reported per grader under
  ``timing`` as measurements (they differ between runs).
* Review decision of a text answer: the pipeline routing with perfect OCR and
  segmentation -- ``confidence = combine(1.0, grading_confidence, 1.0, T)`` and
  ``route(confidence, 1.0, flags (+ short_answer when len(text) < 3, as in
  processor.compute_answer), auto_accept, low_ocr)``; flagged = needs_review.
* Key-point thresholds: ``--thresholds engine`` (default) passes ``th=None`` so
  grade_answer uses ``engine.default_thresholds(scorer)`` (per scorer);
  ``--thresholds settings`` passes Settings tau / tau_low / contradiction_threshold to
  every scorer, as pipeline.processor._thresholds does. The values used are recorded
  per grader. auto_accept / low_ocr / calibration T always come from ``Settings``
  (SARAS_* env / backend/.env). Tune on DEV only; run the held-out test benchmark
  once, with thresholds frozen.

Adapters for graders wired later (``llm`` -> app.grading.llm_grader, ``ensemble`` ->
app.grading.ensemble): the module must expose ``grade_answer(question, answer_text, ...)``
returning a GradeResult-like object or dict (marks, key_points[].status, flags,
grading_confidence, verdict, grader); parameters named ``th``/``thresholds``,
``settings`` or ``scorer`` are filled from the run context. An optional
``available()`` returning bool or (bool, reason) is honoured.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import inspect
import json
import logging
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import metrics as M

log = logging.getLogger(__name__)

REPORT_VERSION = 1
BENCHMARK_VERSION = 1
# The benchmark's partial-credit rule, used ONLY to sanity-check expected_marks labels.
CREDIT = {"matched": 1.0, "partial": 0.5, "missed": 0.0, "contradicted": 0.0}
DETAIL_GROUPS = ("overall", "overall_unambiguous")
# Metrics with bootstrap CIs in the per-category / per-type groups (the breakdown table columns).
BREAKDOWN_METRICS = frozenset({"mae", "within_0.5", "qwk_normalised", "kp_kappa", "flag_rate", "error_rate_unflagged"})
RISK_COVERAGE_POINTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_WARMUP = (["Water boils at 100 degrees Celsius."], ["Water boils at 100 degrees Celsius."])


class BenchmarkError(ValueError):
    """The benchmark file (or a filter on it) is invalid."""


# ============================================================================ benchmark
@dataclass
class Benchmark:
    path: Path
    sha256: str
    version: int
    provenance: str
    questions: dict[str, dict]
    answers: list[dict]
    warnings: list[str]


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_benchmark(data: Any) -> tuple[list[str], list[str]]:
    """(errors, warnings) for a parsed benchmark document."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        return ["top level must be a JSON object"], warnings
    if data.get("version") != BENCHMARK_VERSION:
        errors.append(f"version must be {BENCHMARK_VERSION}, got {data.get('version')!r}")
    qs, ans = data.get("questions"), data.get("answers")
    if not isinstance(qs, list) or not qs:
        errors.append("questions must be a non-empty list")
        qs = []
    if not isinstance(ans, list) or not ans:
        errors.append("answers must be a non-empty list")
        ans = []

    questions: dict[str, dict] = {}
    for n, q in enumerate(qs):
        where = f"questions[{n}]"
        if not isinstance(q, dict) or q.get("id") in (None, ""):
            errors.append(f"{where}: needs an id")
            continue
        qid = str(q["id"])
        where = f"question {qid}"
        if qid in questions:
            errors.append(f"{where}: duplicate id")
        if not isinstance(q.get("type"), str) or not q["type"]:
            errors.append(f"{where}: type must be a string")
        if not _is_num(q.get("max_marks")) or q["max_marks"] <= 0:
            errors.append(f"{where}: max_marks must be a positive number")
        kps = q.get("key_points")
        if not isinstance(kps, list) or not kps:
            errors.append(f"{where}: key_points must be a non-empty list")
            kps = []
        for k, kp in enumerate(kps):
            if not isinstance(kp, dict) or not str(kp.get("text", "")).strip():
                errors.append(f"{where}: key_points[{k}] needs a non-empty text")
            elif not _is_num(kp.get("weight")) or kp["weight"] < 0:
                errors.append(f"{where}: key_points[{k}].weight must be a number >= 0")
            elif kp.get("keywords") is not None and not isinstance(kp["keywords"], list):
                errors.append(f"{where}: key_points[{k}].keywords must be a list")
        if _is_num(q.get("max_marks")) and kps and all(isinstance(kp, dict) and _is_num(kp.get("weight")) for kp in kps):
            total = sum(kp["weight"] for kp in kps)
            if abs(total - q["max_marks"]) > 1e-6:
                warnings.append(f"{where}: key-point weights sum to {total:g}, max_marks is {q['max_marks']:g}")
        questions[qid] = q

    seen: set[str] = set()
    for n, a in enumerate(ans):
        if not isinstance(a, dict) or a.get("id") in (None, ""):
            errors.append(f"answers[{n}]: needs an id")
            continue
        aid = str(a["id"])
        where = f"answer {aid}"
        if aid in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(aid)
        q = questions.get(str(a.get("question_id")))
        if q is None:
            errors.append(f"{where}: unknown question_id {a.get('question_id')!r}")
            continue
        if not isinstance(a.get("category"), str) or not a["category"]:
            errors.append(f"{where}: category must be a non-empty string")
        if not isinstance(a.get("text"), str):
            errors.append(f"{where}: text must be a string")
        kp_exp = a.get("expected_kp")
        kps = q.get("key_points") or []
        if not isinstance(kp_exp, list) or len(kp_exp) != len(kps):
            errors.append(f"{where}: expected_kp must list one status per key point ({len(kps)})")
            kp_exp = None
        elif any(s not in M.STATUSES for s in kp_exp):
            errors.append(f"{where}: expected_kp statuses must be one of {list(M.STATUSES)}")
            kp_exp = None
        em = a.get("expected_marks")
        if not _is_num(em) or not -1e-9 <= em <= float(q.get("max_marks") or 0) + 1e-9:
            errors.append(f"{where}: expected_marks must be a number in [0, max_marks]")
        elif kp_exp is not None and all(isinstance(kp, dict) and _is_num(kp.get("weight")) for kp in kps):
            rule = sum(kp["weight"] * CREDIT[s] for kp, s in zip(kps, kp_exp))
            if abs(rule - em) > 1e-6:
                warnings.append(f"{where}: expected_marks {em:g} differs from the partial-credit rule "
                                f"applied to expected_kp ({rule:g})")
        if a.get("expected_flag") is not None and not isinstance(a["expected_flag"], bool):
            errors.append(f"{where}: expected_flag must be true/false")
        if a.get("ambiguous") is not None and not isinstance(a["ambiguous"], bool):
            errors.append(f"{where}: ambiguous must be true/false")
    return errors, warnings


def load_benchmark(path: str | Path) -> Benchmark:
    path = Path(path)
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"{path}: not valid UTF-8 JSON: {exc}") from exc
    errors, warnings = validate_benchmark(data)
    if errors:
        more = f"\n  ... and {len(errors) - 25} more" if len(errors) > 25 else ""
        raise BenchmarkError(f"{path}: invalid benchmark:\n  " + "\n  ".join(errors[:25]) + more)
    return Benchmark(path=path, sha256=hashlib.sha256(raw).hexdigest(), version=data["version"],
                     provenance=str(data.get("provenance") or ""),
                     questions={str(q["id"]): q for q in data["questions"]},
                     answers=list(data["answers"]), warnings=warnings)


def build_question(q: Mapping[str, Any], number: int) -> SimpleNamespace:
    """A question object with the attributes the graders read (fresh copy per call)."""
    return SimpleNamespace(
        id=q["id"], question_no=number, type=q["type"], max_marks=float(q["max_marks"]),
        prompt=q.get("prompt") or "", model_answer=q.get("model_answer") or "", domain=q.get("domain"),
        key_points=[{"text": kp["text"], "weight": float(kp["weight"]), "keywords": list(kp.get("keywords") or [])}
                    for kp in q["key_points"]],
    )


# ============================================================================ run context
def thresholds_from_settings(settings: Any) -> Any:
    """engine.Thresholds filled from Settings by field name (``x`` or ``x_threshold``),
    the same mapping app.pipeline.processor._thresholds uses."""
    from app.grading.engine import Thresholds

    th = Thresholds()
    for f in dataclasses.fields(th):
        for key in (f.name, f"{f.name}_threshold"):
            if hasattr(settings, key):
                setattr(th, f.name, getattr(settings, key))
                break
    return th


THRESHOLD_MODES = ("engine", "settings")


@dataclass
class RunContext:
    settings: Any
    thresholds: Any  # engine.Thresholds from Settings (used in "settings" mode)
    offline_scorer: str
    baseline_scorer: str = "lexical"
    onnx_model_dir: str | None = None
    thresholds_mode: str = "engine"

    def th(self) -> Any:
        """Thresholds passed to the graders: None in "engine" mode (grade_answer then uses
        engine.default_thresholds(scorer.name), per scorer), the Settings values in
        "settings" mode (what pipeline.processor._thresholds passes)."""
        return None if self.thresholds_mode == "engine" else self.thresholds

    def thresholds_used(self, scorer_name: str) -> dict | None:
        if self.thresholds_mode == "settings":
            return dataclasses.asdict(self.thresholds)
        from app.grading import engine

        fn = getattr(engine, "default_thresholds", None)
        return dataclasses.asdict(fn(scorer_name)) if callable(fn) else None

    def scorer(self, kind: str) -> Any:
        """The production scorer for ``kind`` via scorers.get_scorer (cached there)."""
        from app.grading.scorers import get_scorer

        kwargs: dict[str, Any] = {}
        if self.onnx_model_dir:
            if "onnx_model_dir" not in inspect.signature(get_scorer).parameters:
                raise RuntimeError("scorers.get_scorer() does not accept onnx_model_dir yet")
            kwargs["onnx_model_dir"] = self.onnx_model_dir
        return get_scorer(kind, self.settings.sbert_model, self.settings.nli_model, **kwargs)


def _err(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}".strip()
    return msg if len(msg) <= 300 else msg[:297] + "..."


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class Prediction:
    marks: float | None = None
    kp_statuses: list[str] | None = None
    kp_reasons: list[str] | None = None  # engine KeyPointResult.reason, when the grader gives one
    flags: list[str] = field(default_factory=list)
    grading_confidence: float | None = None
    confidence: float | None = None  # routed (calibrated) confidence, compared with auto_accept
    review: bool | None = None  # predicted deferral: routed to needs_review
    verdict: str | None = None
    grader_reported: str | None = None
    error: str | None = None


def review_decision(grading_confidence: float | None, flags: Sequence[str], text: str,
                    ctx: RunContext) -> tuple[float | None, bool | None]:
    """(confidence, flagged) from the pipeline routing for a perfectly read text answer."""
    if grading_confidence is None:
        return None, None
    from app.pipeline.routing import combine, route

    s = ctx.settings
    conf = combine(1.0, float(grading_confidence), 1.0, s.calibration_temperature)
    extra = ["short_answer"] if len(text) < 3 else []  # as processor.compute_answer
    status, _ = route(conf, 1.0, list(flags) + extra, auto_accept=s.auto_accept_threshold,
                      low_ocr=s.low_ocr_confidence)
    return conf, status == "needs_review"


def prediction_from_result(result: Any, text: str, ctx: RunContext) -> Prediction:
    """Read a GradeResult-like object (or dict) into a Prediction."""
    marks = _field(result, "marks")
    kps = _field(result, "key_points")
    gconf = _field(result, "grading_confidence")
    flags = [str(f) for f in (_field(result, "flags") or [])]
    conf, review = review_decision(gconf, flags, text, ctx)
    verdict = _field(result, "verdict")
    return Prediction(
        marks=None if marks is None else float(marks),
        kp_statuses=None if kps is None else [str(_field(kp, "status")) for kp in kps],
        kp_reasons=None if kps is None or all(_field(kp, "reason") is None for kp in kps)
        else [str(_field(kp, "reason") or "") for kp in kps],
        flags=flags, grading_confidence=None if gconf is None else float(gconf), confidence=conf, review=review,
        verdict=None if verdict is None else str(_field(verdict, "value", verdict)),
        grader_reported=None if _field(result, "grader") is None else str(_field(result, "grader")),
    )


def _context_kwargs(fn: Callable, ctx: RunContext, skip: int) -> dict[str, Any]:
    """Keyword arguments for ``fn`` from the run context, by parameter name."""
    try:
        params = list(inspect.signature(fn).parameters.values())[skip:]
    except (TypeError, ValueError):
        return {}
    kw: dict[str, Any] = {}
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in ("th", "thresholds"):
            kw[p.name] = ctx.th()
        elif p.name == "settings":
            kw[p.name] = ctx.settings
        elif p.name in ("scorer", "offline_scorer"):
            kw[p.name] = ctx.scorer(ctx.offline_scorer)
    return kw


# ============================================================================ graders
class Grader:
    """Adapter interface. ``availability()`` returns None when the grader can run, else
    the reason it cannot; ``grade()`` returns a Prediction or raises."""

    name = "base"

    def availability(self) -> str | None:
        return None

    def grade(self, question: SimpleNamespace, text: str) -> Prediction:
        raise NotImplementedError

    def info(self) -> dict:
        return {}


def _scorer_stats(scorer: Any) -> dict:
    stats = getattr(scorer, "stats", None)
    return {k: v for k, v in stats.items() if _is_num(v)} if isinstance(stats, dict) else {}


def _checked_scorer(ctx: RunContext, kind: str) -> tuple[Any, str | None]:
    """(scorer, None) when get_scorer(kind) yields a working scorer of that kind, else (None, reason)."""
    try:
        scorer = ctx.scorer(kind)
    except Exception as exc:  # noqa: BLE001
        return None, f"get_scorer({kind!r}) failed: {_err(exc)}"
    name = getattr(scorer, "name", None)
    if kind != "auto" and name != kind:
        return None, f"get_scorer({kind!r}) returned the {name!r} scorer, not {kind!r}"
    try:
        scorer.score(*_WARMUP)  # loads models now, so a broken model is "unavailable", not N errors
    except Exception as exc:  # noqa: BLE001
        return None, f"{kind} scorer failed to load/run: {_err(exc)}"
    return scorer, None


class EngineGrader(Grader):
    """engine.grade_answer with the scorer from scorers.get_scorer(kind)."""

    def __init__(self, name: str, kind: str, ctx: RunContext) -> None:
        self.name, self.kind, self.ctx = name, kind, ctx
        self._scorer: Any = None
        self._stats0: dict = {}

    def availability(self) -> str | None:
        self._scorer, reason = _checked_scorer(self.ctx, self.kind)
        if self._scorer is not None:
            self._stats0 = _scorer_stats(self._scorer)
        return reason

    def grade(self, question: SimpleNamespace, text: str) -> Prediction:
        from app.grading.engine import grade_answer

        result = grade_answer(question, text, self._scorer, self.ctx.th())
        return prediction_from_result(result, text, self.ctx)

    def info(self) -> dict:
        if self._scorer is None:
            return {}
        now = _scorer_stats(self._scorer)
        name = getattr(self._scorer, "name", self.kind)
        return {"scorer": name, "thresholds_mode": self.ctx.thresholds_mode,
                "thresholds": self.ctx.thresholds_used(name),
                "scorer_stats": {k: round(v - self._stats0.get(k, 0), 4) for k, v in now.items()}}


class BaselineFormulaGrader(Grader):
    """engine.baseline_formula_marks (the PRD 12.4 short-answer formula) for comparison."""

    def __init__(self, name: str, ctx: RunContext) -> None:
        self.name, self.ctx = name, ctx
        self._fn: Callable | None = None
        self._scorer: Any = None

    def availability(self) -> str | None:
        from app.grading import engine

        fn = getattr(engine, "baseline_formula_marks", None)
        if not callable(fn):
            return "app.grading.engine has no baseline_formula_marks() yet"
        self._scorer, reason = _checked_scorer(self.ctx, self.ctx.baseline_scorer)
        self._fn = fn
        return reason

    def grade(self, question: SimpleNamespace, text: str) -> Prediction:
        assert self._fn is not None
        out = self._fn(question, text, self._scorer)
        marks = out if _is_num(out) else _field(out, "marks")
        if not _is_num(marks):
            raise TypeError(f"baseline_formula_marks returned {type(out).__name__}, expected a number")
        return Prediction(marks=float(marks), grader_reported=f"baseline-formula/{self._scorer.name}")

    def info(self) -> dict:
        return {"scorer": self.ctx.baseline_scorer}


def _availability_reason(value: Any, where: str) -> str | None:
    if isinstance(value, tuple) and value:
        ok, reason = bool(value[0]), (str(value[1]) if len(value) > 1 and value[1] else "")
        return None if ok else (reason or f"{where} returned False")
    return None if bool(value) else f"{where} returned False"


class ModuleGrader(Grader):
    """Lazy adapter for a grader module wired by another component (see module doc)."""

    function = "grade_answer"

    def __init__(self, name: str, module: str, ctx: RunContext) -> None:
        self.name, self.module, self.ctx = name, module, ctx
        self._fn: Callable | None = None
        self._kwargs: dict[str, Any] = {}

    def availability(self) -> str | None:
        try:
            mod = importlib.import_module(self.module)
        except ModuleNotFoundError as exc:
            if exc.name and (exc.name == self.module or self.module.startswith(exc.name + ".")):
                return f"module {self.module} does not exist yet"
            return f"cannot import {self.module}: {_err(exc)}"
        except Exception as exc:  # noqa: BLE001
            return f"cannot import {self.module}: {_err(exc)}"
        fn = getattr(mod, self.function, None)
        if not callable(fn):
            found = sorted(n for n, v in vars(mod).items()
                           if callable(v) and n.startswith("grade") and getattr(v, "__module__", "") == mod.__name__)
            return (f"{self.module} has no evaluation entry point {self.function}(question, answer_text) yet"
                    + (f" (found: {', '.join(found)})" if found else ""))
        try:
            check = getattr(mod, "available", None)
            if callable(check):
                reason = _availability_reason(check(**_context_kwargs(check, self.ctx, 0)),
                                              f"{self.module}.available()")
                if reason:
                    return reason
            self._kwargs = _context_kwargs(fn, self.ctx, 2)
        except Exception as exc:  # noqa: BLE001
            return f"{self.module} availability check failed: {_err(exc)}"
        self._fn = fn
        return None

    def grade(self, question: SimpleNamespace, text: str) -> Prediction:
        assert self._fn is not None
        return prediction_from_result(self._fn(question, text, **self._kwargs), text, self.ctx)


GraderFactory = Callable[[str, RunContext], Grader]


def _engine(kind: str) -> GraderFactory:
    return lambda name, ctx: EngineGrader(name, kind, ctx)


GRADERS: dict[str, GraderFactory] = {
    "lexical": _engine("lexical"),
    "nli-onnx": _engine("nli-onnx"),
    "sbert": _engine("sbert"),
    "nli": _engine("nli"),
    "auto": _engine("auto"),
    "baseline-formula": lambda name, ctx: BaselineFormulaGrader(name, ctx),
    "llm": lambda name, ctx: ModuleGrader(name, "app.grading.llm_grader", ctx),
    "ensemble": lambda name, ctx: ModuleGrader(name, "app.grading.ensemble", ctx),
}


# ============================================================================ running
def _record(ans: dict, q: dict, grader: str, pred: Prediction) -> dict:
    exp_kp = list(ans["expected_kp"])
    comparable = (pred.kp_statuses is not None and len(pred.kp_statuses) == len(exp_kp)
                  and all(s in M.STATUSES for s in pred.kp_statuses))
    scored = pred.error is None and pred.marks is not None
    return {
        "grader": grader, "answer_id": str(ans["id"]), "question_id": str(q["id"]),
        "category": ans["category"], "question_type": q["type"], "ambiguous": bool(ans.get("ambiguous") or False),
        "max_marks": float(q["max_marks"]), "expected_marks": float(ans["expected_marks"]),
        "predicted_marks": pred.marks, "abs_error": abs(pred.marks - ans["expected_marks"]) if scored else None,
        "correct": bool(abs(pred.marks - ans["expected_marks"]) <= M.CORRECT_TOL + 1e-9) if scored else None,
        "expected_kp": exp_kp, "predicted_kp": pred.kp_statuses, "predicted_kp_reasons": pred.kp_reasons,
        "kp_comparable": comparable,
        "expected_flag": ans.get("expected_flag"), "review": pred.review, "flags": pred.flags,
        "grading_confidence": pred.grading_confidence, "confidence": pred.confidence,
        "verdict": pred.verdict, "grader_reported": pred.grader_reported, "error": pred.error,
    }


def run_graders(bench: Benchmark, answers: list[dict], names: Sequence[str], ctx: RunContext,
                registry: Mapping[str, GraderFactory]) -> tuple[list[dict], list[dict]]:
    """Run each grader over ``answers``: (records, per-grader info)."""
    numbers = {qid: n for n, qid in enumerate(bench.questions, 1)}
    records: list[dict] = []
    infos: list[dict] = []
    for name in names:
        grader = registry[name](name, ctx)
        t0 = time.perf_counter()
        try:
            reason = grader.availability()
        except Exception as exc:  # noqa: BLE001
            reason = f"availability check raised {_err(exc)}"
        load_s = time.perf_counter() - t0
        errors: Counter[str] = Counter()
        reported: Counter[str] = Counter()
        out_of_range = 0
        t1 = time.perf_counter()
        for ans in answers:
            q = bench.questions[str(ans["question_id"])]
            if reason:
                pred = Prediction(error=f"unavailable: {reason}")
            else:
                try:
                    pred = grader.grade(build_question(q, numbers[str(q["id"])]), ans["text"])
                    if pred.marks is None or not math.isfinite(pred.marks):
                        raise ValueError(f"grader returned marks={pred.marks!r}")
                except Exception as exc:  # noqa: BLE001
                    pred = Prediction(error=_err(exc))
                    errors[pred.error] += 1
            if pred.error is None:
                reported[pred.grader_reported or "?"] += 1
                if not -1e-9 <= pred.marks <= float(q["max_marks"]) + 1e-9:
                    out_of_range += 1
            records.append(_record(ans, q, name, pred))
        n_err = sum(errors.values())
        if reason:
            status = f"unavailable: {reason}"
        elif answers and n_err == len(answers):
            status = f"failed: all {n_err} answers raised errors"
        else:
            status = "available"
        infos.append({
            "name": name, "status": status, "n_answers": len(answers),
            "n_scored": 0 if reason else len(answers) - n_err, "n_errors": n_err,
            "errors": [{"error": e, "count": c} for e, c in errors.most_common(5)],
            "reported_as": dict(sorted(reported.items())), "n_marks_out_of_range": out_of_range,
            **({} if reason else grader.info()),
            "timing": {"availability_seconds": round(load_s, 3),
                       "grading_seconds": round(time.perf_counter() - t1, 3) if not reason else 0.0},
        })
        log.info("grader %s: %s", name, status)
    return records, infos


# ============================================================================ metrics
def _num(x: Any) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


class _Metrics:
    def __init__(self, n_boot: int, seed: int, ci_for: frozenset[str] | None) -> None:
        self.n_boot, self.seed, self.ci_for = n_boot, seed, ci_for
        self.out: dict[str, dict] = {}

    def add(self, name: str, fn: Callable[[np.ndarray], float], size: int, n_units: int) -> None:
        idx = np.arange(size)
        ci: list[float | None] | None = None
        boot: M.BootstrapCI | None = None
        if self.ci_for is None or name in self.ci_for:
            boot = M.bootstrap_interval(fn, idx, n=self.n_boot, seed=self.seed)
            ci = [_num(boot.lo), _num(boot.hi)]
        # ci95_defined of ci95_resamples resamples gave a defined value; the CI uses only those
        # and is null when fewer than half are defined (metrics.bootstrap_interval).
        self.out[name] = {"value": _num(fn(idx)), "ci95": ci, "n": n_units,
                          "ci95_defined": None if boot is None else boot.n_defined,
                          "ci95_resamples": None if boot is None else boot.n_resamples}


def summarise(records: list[dict], n_boot: int, seed: int, detail: bool = False) -> dict:
    """Metrics (value, bootstrap 95% CI, n) for one grader's records in one group.
    Bootstrap resamples answers; KPs are resampled together with their answer. With
    ``detail`` every metric gets a CI plus reliability bins and risk-coverage points;
    otherwise only the BREAKDOWN_METRICS shown in the breakdown tables get a CI (the
    others carry ``"ci95": null``)."""
    scored = [r for r in records if r["error"] is None and r["predicted_marks"] is not None]
    out: dict[str, Any] = {"n_answers": len(records), "n_scored": len(scored),
                           "n_errors": sum(1 for r in records if r["error"] is not None), "metrics": {}}
    if not scored:
        return out
    mt = _Metrics(n_boot, seed, None if detail else BREAKDOWN_METRICS)
    p = np.array([r["predicted_marks"] for r in scored], dtype=float)
    g = np.array([r["expected_marks"] for r in scored], dtype=float)
    mx = np.array([r["max_marks"] for r in scored], dtype=float)
    qid = np.array([str(r["question_id"]) for r in scored], dtype=object)
    n = len(scored)
    mt.add("mae", lambda i: M.mae(p[i], g[i]), n, n)
    mt.add("rmse", lambda i: M.rmse(p[i], g[i]), n, n)
    mt.add("within_0.5", lambda i: M.within(p[i], g[i], 0.5), n, n)
    mt.add("within_1", lambda i: M.within(p[i], g[i], 1.0), n, n)
    # Headline agreement is on marks / max_marks; pooling absolute marks over questions with
    # different maxima is scale-confounded (metrics module doc), kept only as a reference.
    per_q = M.qwk_per_question(p, g, qid)
    mt.add("qwk_normalised", lambda i: M.qwk_normalised(p[i], g[i], mx[i]), n, n)
    mt.add("qwk_per_question_mean", lambda i: M.qwk_per_question_mean(p[i], g[i], qid[i]), n, len(per_q))
    mt.add("qwk_absolute_pooled", lambda i: M.quadratic_weighted_kappa(p[i], g[i]), n, n)
    mt.out["qwk"] = dict(mt.out["qwk_absolute_pooled"])  # old name of qwk_absolute_pooled, kept for compatibility
    mt.add("pearson_normalised", lambda i: M.pearson_normalised(p[i], g[i], mx[i]), n, n)
    mt.add("pearson", lambda i: M.pearson(p[i], g[i]), n, n)  # absolute pooled, scale-confounded
    if detail:
        out["qwk_per_question"] = [{"question_id": q, "n": int(np.sum(qid == q)), "qwk": _num(per_q.get(q, M.NAN))}
                                   for q in dict.fromkeys(qid.tolist())]
    out["mean_expected_marks"] = _num(g.mean())
    out["mean_predicted_marks"] = _num(p.mean())

    with_kp = [r for r in scored if r["predicted_kp"] is not None]
    comp = [r for r in with_kp if r["kp_comparable"]]
    out["n_kp_answers"] = len(comp)
    out["n_kp_not_comparable"] = len(with_kp) - len(comp)
    if comp:
        cms = np.stack([M.confusion_matrix(r["expected_kp"], r["predicted_kp"]) for r in comp])
        nk, n_kp = len(comp), int(cms.sum())
        out["n_kp"] = n_kp
        mt.add("kp_kappa", lambda i: M.kappa_from_confusion(cms[i].sum(0)), nk, n_kp)
        mt.add("kp_accuracy", lambda i: M.accuracy_from_confusion(cms[i].sum(0)), nk, n_kp)
        for key in ("precision", "recall", "f1"):
            mt.add(f"awarded_{key}", lambda i, k=key: M.awarded_prf_from_confusion(cms[i].sum(0))[k], nk, n_kp)
        for key, label in (("precision", "contradiction_precision"), ("recall", "contradiction_recall"),
                           ("false_contradiction_rate", "false_contradiction_rate")):
            mt.add(label, lambda i, k=key: M.contradiction_from_confusion(cms[i].sum(0))[k], nk, n_kp)
        out["kp_confusion"] = {"labels": list(M.STATUSES), "rows": "expected", "columns": "predicted",
                               "matrix": cms.sum(0).tolist()}

    routed = [r for r in scored if r["review"] is not None]
    if routed:
        rp = np.array([r["predicted_marks"] for r in routed], dtype=float)
        rg = np.array([r["expected_marks"] for r in routed], dtype=float)
        rf = np.array([bool(r["review"]) for r in routed])
        nr = len(routed)
        mt.add("flag_rate", lambda i: M.flag_rate(rf[i]), nr, nr)
        mt.add("error_rate_unflagged", lambda i: M.error_rate_unflagged(rp[i], rg[i], rf[i]), nr, int((~rf).sum()))
        both = [r for r in routed if r["expected_flag"] is not None]
        if both:
            ef = np.array([bool(r["expected_flag"]) for r in both])
            pf = np.array([bool(r["review"]) for r in both])
            nb = len(both)
            mt.add("flag_precision", lambda i: M.binary_prf(ef[i], pf[i])["precision"], nb, int(pf.sum()))
            mt.add("flag_recall", lambda i: M.binary_prf(ef[i], pf[i])["recall"], nb, int(ef.sum()))

    calib = [r for r in scored if r["confidence"] is not None]
    if calib:
        cc = np.array([r["confidence"] for r in calib], dtype=float)
        cy = np.array([bool(r["correct"]) for r in calib])
        nc = len(calib)
        mt.add("ece", lambda i: M.ece(cc[i], cy[i]), nc, nc)
        mt.add("brier", lambda i: M.brier(cc[i], cy[i]), nc, nc)
        mt.add("aurc", lambda i: M.aurc(cc[i], cy[i]), nc, nc)
        if detail:
            out["reliability_bins"] = [{k: (_num(v) if isinstance(v, float) else v) for k, v in b.items()}
                                       for b in M.reliability_bins(cc, cy)]
            # thresholds at distinct confidences (ties accepted together): achieved_coverage >= coverage
            out["risk_coverage"] = [{k: _num(v) for k, v in pt.items()}
                                    for pt in M.risk_at_coverage(cc, cy, RISK_COVERAGE_POINTS)]
    out["metrics"] = mt.out
    return out


def build_groups(answers: list[dict], questions: Mapping[str, dict]) -> list[dict]:
    groups = [{"key": "overall", "kind": "overall", "name": "All answers",
               "ids": [str(a["id"]) for a in answers]},
              {"key": "overall_unambiguous", "kind": "overall", "name": "Excluding ambiguous answers",
               "ids": [str(a["id"]) for a in answers if not a.get("ambiguous")]}]
    for kind, key_of in (("category", lambda a: a["category"]),
                         ("type", lambda a: questions[str(a["question_id"])]["type"])):
        for value in dict.fromkeys(key_of(a) for a in answers):
            groups.append({"key": f"{kind}={value}", "kind": kind, "name": value,
                           "ids": [str(a["id"]) for a in answers if key_of(a) == value]})
    return groups


# ============================================================================ report
def _clean(x: Any) -> Any:
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def run_evaluation(data: str | Path, graders: Sequence[str], out: str | Path, *, limit: int | None = None,
                   categories: Sequence[str] | None = None, run_label: str | None = None,
                   timestamp: str | None = None, n_boot: int = 1000, seed: int = 42, settings: Any = None,
                   offline_scorer: str | None = None, baseline_scorer: str = "lexical",
                   onnx_model_dir: str | None = None, thresholds: str = "engine",
                   registry: Mapping[str, GraderFactory] | None = None) -> dict:
    """Run ``graders`` over the benchmark at ``data``; write results.jsonl, report.json and
    report.md into ``out``; return the report dict."""
    registry = dict(GRADERS if registry is None else registry)
    names = list(dict.fromkeys(g.strip() for g in graders if g and g.strip()))
    unknown = [g for g in names if g not in registry]
    if not names or unknown:
        raise ValueError(f"unknown grader(s) {unknown or names}; known: {', '.join(registry)}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if thresholds not in THRESHOLD_MODES:
        raise ValueError(f"thresholds must be one of {THRESHOLD_MODES}")

    bench = load_benchmark(data)
    answers = bench.answers
    cats = list(dict.fromkeys(c.strip() for c in (categories or []) if c.strip()))
    if cats:
        known = set(a["category"] for a in answers)
        missing = [c for c in cats if c not in known]
        if missing:
            raise BenchmarkError(f"unknown categories {missing}; the benchmark has {sorted(known)}")
        answers = [a for a in answers if a["category"] in cats]
    if limit is not None:
        answers = answers[:limit]

    if settings is None:
        from app.config import Settings

        settings = Settings()
    ctx = RunContext(settings=settings, thresholds=thresholds_from_settings(settings),
                     offline_scorer=offline_scorer or getattr(settings, "grader", "lexical"),
                     baseline_scorer=baseline_scorer, onnx_model_dir=onnx_model_dir, thresholds_mode=thresholds)

    records, infos = run_graders(bench, answers, names, ctx, registry)

    groups_out = []
    for grp in build_groups(answers, bench.questions):
        ids = set(grp["ids"])
        per = {name: summarise([r for r in records if r["grader"] == name and r["answer_id"] in ids],
                               n_boot, seed, detail=grp["key"] in DETAIL_GROUPS) for name in names}
        groups_out.append({"key": grp["key"], "kind": grp["kind"], "name": grp["name"],
                           "n_answers": len(grp["ids"]), "graders": per})

    s = ctx.settings
    report = _clean({
        "report_version": REPORT_VERSION,
        "run_label": run_label,
        "timestamp": timestamp,
        "benchmark": {"path": Path(data).as_posix(), "sha256": bench.sha256, "version": bench.version,
                      "provenance": bench.provenance, "n_questions": len(bench.questions),
                      "n_answers_total": len(bench.answers), "n_answers_evaluated": len(answers),
                      "filters": {"categories": cats or None, "limit": limit},
                      "warnings": bench.warnings},
        "config": {"thresholds_mode": thresholds,
                   "thresholds_mode_note": "engine: grade_answer(th=None) -> engine.default_thresholds(scorer); "
                                           "settings: Settings tau/tau_low/contradiction_threshold for every "
                                           "scorer (as pipeline.processor._thresholds); per-grader values "
                                           "are under graders[].thresholds",
                   "settings_thresholds": dataclasses.asdict(ctx.thresholds),
                   "auto_accept_threshold": s.auto_accept_threshold,
                   "low_ocr_confidence": s.low_ocr_confidence,
                   "calibration_temperature": s.calibration_temperature,
                   "offline_scorer_for_adapters": ctx.offline_scorer, "baseline_scorer": baseline_scorer,
                   "onnx_model_dir": onnx_model_dir},
        "method": {
            "bootstrap": {"resamples": n_boot, "seed": seed, "interval": "95% percentile",
                          "unit": "answer (key points resampled with their answer)",
                          "breakdown_groups": "category/type groups carry CIs for "
                                              f"{', '.join(sorted(BREAKDOWN_METRICS))} only",
                          "min_defined_share": M.MIN_DEFINED_SHARE},
            "ci_support": "ci95_defined of ci95_resamples bootstrap resamples gave a defined value (e.g. a "
                          "precision is undefined in a resample with no predicted positive); the CI uses only "
                          "those (shown as 'd=NN%' when below 100%) and is n/a when fewer than "
                          f"{M.MIN_DEFINED_SHARE:.0%} are defined",
            "correct": f"|predicted - expected| <= {M.CORRECT_TOL} marks",
            "qwk_normalised": "headline QWK: marks / max marks binned to 0.1 steps (ratings 0-10, round half "
                              "up), pooled across questions; a constant fraction of the maximum scores 0. It "
                              "still credits differences between questions, and on questions worth more than "
                              "5 marks two adjacent half marks can share a bin",
            "qwk_per_question_mean": f"unweighted mean of QWK within each question on its own "
                                     f"{M.MARK_STEP}-mark scale, over questions with >= {M.MIN_QUESTION_ANSWERS} "
                                     "answers whose expected marks are not all equal (gold-only rule, the same "
                                     "questions for every grader); n = number of such questions",
            "qwk_absolute_pooled": "QWK on absolute marks binned to 0.5-mark steps, pooled across questions: "
                                   "scale-confounded (predicting max/2 for every answer scores above 0); "
                                   "'qwk' is the same value under its old name. Quarter-mark labels round half "
                                   "up on the 0.5 grid",
            "pearson": "pearson_normalised is on marks / max marks; 'pearson' is on absolute pooled marks "
                       "(scale-confounded)",
            "kp_kappa": "Cohen's unweighted kappa, expected vs predicted key-point status",
            "awarded": "key point awarded = matched or partial",
            "false_contradiction_rate": "share of key points not expected 'contradicted' that were predicted 'contradicted'",
            "flag": "routed to needs_review by pipeline.routing (perfect OCR/segmentation assumed)",
            "error_rate_unflagged": "share of incorrect answers among answers not flagged for review",
            "calibration": "ECE (10 equal-width bins), Brier and AURC of the routed confidence vs correctness; "
                           "risk-coverage thresholds sit at distinct confidences (tied answers are accepted "
                           "together, so the order of answers does not matter)",
            "partial_credit_rule": "matched = full weight, partial = half, missed = 0, contradicted = 0",
            "errors": "answers whose grader raised are excluded from metrics and counted in n_errors",
        },
        "graders": infos,
        "groups": groups_out,
    })

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(_clean(r), ensure_ascii=False) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


# ============================================================================ markdown
def _f(v: Any, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _m(summary: dict, name: str, ci: bool = True) -> str:
    m = (summary.get("metrics") or {}).get(name)
    if not m or m["value"] is None:
        return "n/a"
    if not ci or m.get("ci95") is None:
        return _f(m["value"])
    lo, hi = m["ci95"]
    defined, total = m.get("ci95_defined"), m.get("ci95_resamples")
    # floor, so 999 of 1000 shows as 99%, not 100%
    support = f" d={math.floor(100 * defined / total)}%" if defined is not None and total and defined < total else ""
    return f"{_f(m['value'])} [{_f(lo)}, {_f(hi)}]{support}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    esc = [[str(c).replace("|", "\\|").replace("\n", " ") for c in row] for row in rows]
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in esc]


def _detail_section(grp: dict, names: list[str]) -> list[str]:
    lines = [f"## {grp['name']} (N = {grp['n_answers']})", "", "### Marks", ""]
    per = grp["graders"]
    lines += _table(["Grader", "Scored", "Errors", "MAE", "RMSE", "Within ±0.5", "Within ±1", "QWK (normalised)",
                     "Pearson (normalised)"],
                    [[g, str(per[g]["n_scored"]), str(per[g]["n_errors"])] +
                     [_m(per[g], k) for k in ("mae", "rmse", "within_0.5", "within_1", "qwk_normalised",
                                              "pearson_normalised")]
                     for g in names])
    lines += ["", "Other agreement scales (absolute pooled values are scale-confounded; see method):", ""]
    lines += _table(["Grader", "QWK per question, mean (questions)", "qwk_absolute_pooled (scale-confounded)",
                     "Pearson, absolute pooled (scale-confounded)"],
                    [[g, _m(per[g], "qwk_per_question_mean") + " ({})".format(
                        ((per[g].get("metrics") or {}).get("qwk_per_question_mean") or {}).get("n", 0)),
                      _m(per[g], "qwk_absolute_pooled"), _m(per[g], "pearson")] for g in names])
    lines += ["", "### Key points", ""]
    lines += _table(["Grader", "KPs (answers)", "Not comparable", "Cohen κ", "Accuracy", "Awarded P", "Awarded R",
                     "Awarded F1", "Contradiction P", "Contradiction R", "False-contradiction rate"],
                    [[g, f"{per[g].get('n_kp', 0)} ({per[g].get('n_kp_answers', 0)})",
                      str(per[g].get("n_kp_not_comparable", 0))] +
                     [_m(per[g], k) for k in ("kp_kappa", "kp_accuracy", "awarded_precision", "awarded_recall",
                                              "awarded_f1", "contradiction_precision", "contradiction_recall",
                                              "false_contradiction_rate")]
                     for g in names])
    lines += ["", "### Review routing and calibration", ""]
    rows = []
    for g in names:
        m = per[g].get("metrics") or {}
        unflagged = m.get("error_rate_unflagged", {}).get("n")
        rows.append([g, _m(per[g], "flag_rate"), _m(per[g], "flag_precision"), _m(per[g], "flag_recall"),
                     _m(per[g], "error_rate_unflagged") + ("" if unflagged is None else f" (n={unflagged})"),
                     _m(per[g], "ece"), _m(per[g], "brier"), _m(per[g], "aurc")])
    lines += _table(["Grader", "Flag rate", "Flag P vs expected", "Flag R vs expected",
                     "Error rate, non-flagged", "ECE", "Brier", "AURC"], rows)
    for g in names:
        cm = per[g].get("kp_confusion")
        if not cm:
            continue
        lines += ["", f"KP confusion, **{g}** (rows expected, columns predicted):", ""]
        lines += _table(["expected \\ predicted"] + cm["labels"],
                        [[lab] + [str(v) for v in row] for lab, row in zip(cm["labels"], cm["matrix"])])
    return lines + [""]


def _breakdown(groups: list[dict], kind: str, title: str, names: list[str]) -> list[str]:
    sel = [g for g in groups if g["kind"] == kind]
    if not sel:
        return []
    rows = []
    for grp in sel:
        for g in names:
            s = grp["graders"][g]
            rows.append([grp["name"], str(grp["n_answers"]), g, str(s["n_scored"]), _m(s, "mae"),
                         _m(s, "within_0.5"), _m(s, "qwk_normalised"), _m(s, "kp_kappa"), _m(s, "flag_rate"),
                         _m(s, "error_rate_unflagged")])
    return [f"## {title}", ""] + _table(
        [title.split()[-1].capitalize(), "N", "Grader", "Scored", "MAE", "Within ±0.5", "QWK (normalised)", "KP κ",
         "Flag rate", "Error rate, non-flagged"], rows) + [""]


def render_markdown(report: dict) -> str:
    b, c, meth = report["benchmark"], report["config"], report["method"]
    # Metric tables list graders that scored at least one answer; the Graders table lists all.
    names = [g["name"] for g in report["graders"] if g["n_scored"] > 0]
    skipped = [g["name"] for g in report["graders"] if g["n_scored"] == 0]
    th = "; ".join(f"{g['name']}: " + ", ".join(f"{k}={v}" for k, v in g["thresholds"].items())
                   for g in report["graders"] if g.get("thresholds"))
    filters = ", ".join(f"{k}={v}" for k, v in b["filters"].items() if v) or "none"
    lines = ["# SARAS AI grading evaluation", ""]
    lines += _table(["Item", "Value"], [
        ["Run label", report["run_label"] or "not given"],
        ["Timestamp (from caller)", report["timestamp"] or "not given"],
        ["Benchmark", f"`{b['path']}` (version {b['version']}, sha256 {b['sha256'][:12]})"],
        ["Provenance", b["provenance"] or "not stated"],
        ["Answers evaluated", f"{b['n_answers_evaluated']} of {b['n_answers_total']} "
                              f"({b['n_questions']} questions; filters: {filters})"],
        ["Thresholds", f"mode {c['thresholds_mode']} ({th or 'n/a'}); auto-accept {c['auto_accept_threshold']}; "
                       f"calibration T {c['calibration_temperature']}"],
        ["Bootstrap", f"{meth['bootstrap']['resamples']} resamples, seed {meth['bootstrap']['seed']}, "
                      f"{meth['bootstrap']['interval']} CI, unit: {meth['bootstrap']['unit']}"],
    ])
    lines += ["", "Values are shown as `estimate [95% CI low, high]`; n/a = undefined or not produced by the grader.",
              ""]
    lines += [f"- **{k}**: {v}" for k, v in meth.items() if isinstance(v, str)]
    lines += ["", "## Graders", ""]
    lines += _table(["Grader", "Status", "Scored", "Errors", "Reported as", "Grading time (s)"],
                    [[g["name"], g["status"], str(g["n_scored"]), str(g["n_errors"]),
                      ", ".join(f"{k} ×{v}" for k, v in g["reported_as"].items()) or "-",
                      _f(g["timing"]["grading_seconds"], 2)] for g in report["graders"]])
    errs = [(g["name"], e) for g in report["graders"] for e in g["errors"]]
    if errs:
        lines += ["", "Most frequent errors:", ""] + [f"- {n}: {e['error']} (×{e['count']})" for n, e in errs]
    if skipped:
        lines += ["", f"No scored answers (left out of the metric tables below): {', '.join(skipped)}."]
    if b["warnings"]:
        lines += ["", "## Benchmark warnings", ""] + [f"- {w}" for w in b["warnings"]]
    lines.append("")
    if not names:
        return "\n".join(lines + ["No grader produced any scored answer; there are no metrics to report."]) + "\n"
    for grp in report["groups"]:
        if grp["key"] in DETAIL_GROUPS:
            lines += _detail_section(grp, names)
    lines += _breakdown(report["groups"], "category", "By category", names)
    lines += _breakdown(report["groups"], "type", "By question type", names)
    return "\n".join(lines).rstrip() + "\n"


# ============================================================================ CLI
def _split(values: Sequence[str] | None) -> list[str]:
    return [p.strip() for v in (values or []) for p in v.split(",") if p.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m app.evaluation.grading",
                                 description="Run graders over a grading benchmark and write metrics reports.")
    ap.add_argument("--data", required=True, type=Path, help="benchmark JSON file")
    ap.add_argument("--graders", required=True, nargs="+",
                    help=f"comma/space separated grader names: {', '.join(GRADERS)}")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N answers (after filters)")
    ap.add_argument("--categories", nargs="+", default=None, help="only these answer categories (comma/space separated)")
    ap.add_argument("--run-label", default=None, help="free text recorded in the report")
    ap.add_argument("--timestamp", default=None,
                    help="recorded verbatim; the harness never puts the wall clock into the report")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples (default 1000)")
    ap.add_argument("--seed", type=int, default=42, help="bootstrap seed (default 42)")
    ap.add_argument("--offline-scorer", default=None,
                    help="scorer kind passed to llm/ensemble adapters that take one (default: settings.grader)")
    ap.add_argument("--baseline-scorer", default="lexical", help="scorer for baseline-formula (default lexical)")
    ap.add_argument("--onnx-model-dir", default=None, help="model directory passed to get_scorer(onnx_model_dir=...)")
    ap.add_argument("--thresholds", choices=THRESHOLD_MODES, default="engine",
                    help="engine (default): per-scorer engine defaults (grade_answer th=None); "
                         "settings: Settings tau/tau_low/contradiction_threshold, as the pipeline passes today")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        report = run_evaluation(args.data, _split(args.graders), args.out, limit=args.limit,
                                categories=_split(args.categories) or None, run_label=args.run_label,
                                timestamp=args.timestamp, n_boot=args.bootstrap, seed=args.seed,
                                offline_scorer=args.offline_scorer, baseline_scorer=args.baseline_scorer,
                                onnx_model_dir=args.onnx_model_dir, thresholds=args.thresholds)
    except (BenchmarkError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    overall = report["groups"][0]
    b = report["benchmark"]
    print(f"{b['n_answers_evaluated']} answers, {b['n_questions']} questions -> {args.out}")
    for g in report["graders"]:
        s = overall["graders"][g["name"]]
        print(f"  {g['name']:<17} {g['status']}")
        if s["n_scored"]:
            # ASCII only: the Windows console may be cp1252.
            print(f"    scored={s['n_scored']} errors={s['n_errors']} MAE={_m(s, 'mae')} "
                  f"QWK_norm={_m(s, 'qwk_normalised')} "
                  f"within0.5={_m(s, 'within_0.5')} KP_kappa={_m(s, 'kp_kappa')} flag={_m(s, 'flag_rate')} "
                  f"err_unflagged={_m(s, 'error_rate_unflagged')} ECE={_m(s, 'ece')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
