"""AI marking benchmark: live AI marker vs teacher-labelled marks.

Runs the AI marker the pipeline uses over a grading benchmark (evaluation/data) and
measures accuracy, consistency, latency, requests and tokens. It deliberately imports
`app` from whatever is first on PYTHONPATH, so the SAME script measures the current
engine and an older copy of it (before/after on identical data, sheets and model):

    # current engine
    python evaluation/ai_benchmark.py --data evaluation/data/grading_benchmark_dev.json \
        --mode direct --out evaluation/reports/ai-dev-direct-new
    # an older copy of backend/app (e.g. a snapshot taken before a change)
    PYTHONPATH=/path/to/snapshot python evaluation/ai_benchmark.py ... --out ...-old

Modes
  keypoint  the teacher's key points and model answer are given (llm_grader.grade_sheet_llm,
            plus the selective second marking when the engine has it and --verify on)
  direct    QUESTION-ONLY: key points and model answer are removed, as in exams whose
            rubric is just the question and its marks (direct_ai: v1 one-shot marks, or
            v2 scheme + criterion marking when the engine has it and --scheme on)

Answers are grouped into pseudo answer sheets of --sheet-size answers to different
questions (deterministic), one AI request per sheet as in the pipeline. Requests go
to ONE model (--providers) so engines are compared on the same model; the router
waits for its per-minute budget instead of falling back to another model.

API keys come from backend/.env (never printed). Every request is logged in an
ai_runs table in <out>/bench.db, which also caches replies: re-running the same
engine on the same data costs no tokens. --fresh-db ignores that cache (consistency).

Metrics: MAE, exact agreement (|pred - gold| < 0.01), within 0.5, within 1 mark,
per answer category, unmarked answers, latency per sheet, requests, tokens and an
estimated list-price cost (--price-in / --price-out, USD per 1M tokens; the free tier
actually used costs nothing).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent


def load_env_file(path: Path) -> None:
    """KEY=VALUE lines of backend/.env into os.environ (real env vars win). Values are
    never printed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def make_sheets(answers: list[dict], size: int) -> list[list[dict]]:
    """Greedy, deterministic: each answer joins the first open sheet that has no answer
    to its question yet."""
    sheets: list[list[dict]] = []
    for a in sorted(answers, key=lambda x: x["id"]):
        for sh in sheets:
            if len(sh) < size and all(x["question_id"] != a["question_id"] for x in sh):
                sh.append(a)
                break
        else:
            sheets.append([a])
    return sheets


def _round(x: float | None, n: int = 3) -> float | None:
    return None if x is None else round(x, n)


def summarise(records: list[dict]) -> dict:
    scored = [r for r in records if r["pred"] is not None]
    errs = [abs(r["pred"] - r["gold"]) for r in scored]
    out = {
        "n": len(records), "n_scored": len(scored), "n_unmarked": len(records) - len(scored),
        "mae": _round(statistics.fmean(errs)) if errs else None,
        "exact": _round(sum(e < 0.01 for e in errs) / len(errs)) if errs else None,
        "within_0.5": _round(sum(e <= 0.5 + 1e-9 for e in errs) / len(errs)) if errs else None,
        "within_1": _round(sum(e <= 1.0 + 1e-9 for e in errs) / len(errs)) if errs else None,
        "mean_signed_error": _round(statistics.fmean(r["pred"] - r["gold"] for r in scored)) if scored else None,
        "mae_normalised": _round(statistics.fmean(abs(r["pred"] - r["gold"]) / r["max"] for r in scored))
        if scored else None,
    }
    clean = [r for r in scored if not r.get("ambiguous")]
    if clean:
        out["mae_unambiguous"] = _round(statistics.fmean(abs(r["pred"] - r["gold"]) for r in clean))
    return out


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--mode", choices=("keypoint", "direct"), required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--sheet-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N answers (by id)")
    ap.add_argument("--sheets", type=int, default=None, help="only the first N sheets")
    ap.add_argument("--providers", default="groq:openai/gpt-oss-120b")
    ap.add_argument("--style", default="moderate")
    ap.add_argument("--verify", choices=("on", "off"), default="on", help="selective second marking (new engine)")
    ap.add_argument("--scheme", choices=("on", "off"), default="on", help="direct mode: scheme marking (new engine)")
    ap.add_argument("--effort", default=None, help="reasoning effort for the first marking (new engine)")
    ap.add_argument("--fresh-db", action="store_true", help="start with an empty cache (consistency runs)")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to wait between sheets (lets the per-minute budget refill, so engines whose "
                         "router skips a rate-limited model can still be compared on marking accuracy)")
    ap.add_argument("--price-in", type=float, default=0.15, help="USD per 1M input tokens (list price estimate)")
    ap.add_argument("--price-out", type=float, default=0.60, help="USD per 1M output tokens (list price estimate)")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    db_path = args.out / "bench.db"
    if args.fresh_db and db_path.exists():
        db_path.unlink()
    load_env_file(BACKEND / ".env")
    os.environ["SARAS_ENV"] = "test"
    os.environ["SARAS_DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ["SARAS_STORAGE_DIR"] = str(args.out / "storage")

    from sqlalchemy import create_engine, func, select
    from sqlalchemy.orm import sessionmaker

    import app
    from app.config import Settings
    from app.db import Base
    from app.grading import direct_ai, llm_grader
    from app.llm.router import LLMRouter
    from app.models import AiRun

    engine_root = str(Path(app.__file__).resolve().parent)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    overrides = dict(ai_grading="ensemble", ai_consent=True, ai_grading_providers=providers, ai_max_retries=2,
                     ai_call_deadline_seconds=240, ai_timeout_seconds=90)
    if args.effort and "ai_reasoning_effort" in Settings.model_fields:
        overrides["ai_reasoning_effort"] = args.effort
    settings = Settings(**overrides)
    eng = create_engine(f"sqlite:///{db_path.as_posix()}")
    Base.metadata.create_all(eng, tables=[AiRun.__table__])
    factory = sessionmaker(bind=eng, expire_on_commit=False)
    router = LLMRouter(settings, factory, max_limiter_wait=120.0)

    bench = json.loads(args.data.read_text(encoding="utf-8"))
    qs = {q["id"]: q for q in bench["questions"]}
    qno = {qid: i + 1 for i, qid in enumerate(sorted(qs))}
    answers = sorted(bench["answers"], key=lambda a: a["id"])[: args.limit]
    sheets = make_sheets(answers, args.sheet_size)[: args.sheets]

    has_v2 = hasattr(direct_ai, "grade_sheet_scheme")
    has_verify = hasattr(llm_grader, "verify_sheet")
    engine_name = {
        "direct": ("direct-v2-scheme" if has_v2 and args.scheme == "on" else "direct-v1"),
        "keypoint": f"keypoint-{llm_grader.PROMPT_VERSION}",
    }[args.mode]
    verify_on = args.verify == "on" and has_verify
    if verify_on:
        engine_name += "+verify"

    local_grade = None
    if args.mode == "keypoint" and verify_on:
        from types import SimpleNamespace

        from app.grading import ensemble as E

        def local_grade(q: dict, text: str) -> float | None:  # noqa: F811
            ns = SimpleNamespace(question_no=qno[q["id"]], type=q["type"], max_marks=q["max_marks"],
                                 prompt=q["prompt"], model_answer=q.get("model_answer"),
                                 key_points=q.get("key_points") or [])
            try:
                return float(E.grade_local_primary(ns, text, settings).marks)
            except Exception:  # noqa: BLE001
                return None

    records: list[dict] = []
    latencies: list[float] = []
    verify_records: dict[str, dict] = {}
    started = time.perf_counter()
    for si, sheet in enumerate(sheets):
        if si and args.pause > 0:
            time.sleep(args.pause)
        t0 = time.perf_counter()
        preds: dict[int, float] = {}
        if args.mode == "direct":
            items = [direct_ai.DirectItem(qno[a["question_id"]], qs[a["question_id"]]["prompt"],
                                          qs[a["question_id"]]["type"], float(qs[a["question_id"]]["max_marks"]),
                                          a["text"]) for a in sheet]
            if has_v2 and args.scheme == "on":
                res = direct_ai.grade_sheet_scheme(router, items, args.style, verify=verify_on)
            else:
                res = direct_ai.grade_sheet_direct(router, items, args.style)
            preds = {n: g.marks for n, g in res.grades.items()}
            for n, g in res.grades.items():
                if getattr(g, "verification", None):
                    verify_records[str(n) + "@" + str(si)] = g.verification
        else:
            items = [llm_grader.AnswerItem(question_no=qno[a["question_id"]], type=qs[a["question_id"]]["type"],
                                           max_marks=float(qs[a["question_id"]]["max_marks"]), answer_text=a["text"],
                                           prompt=qs[a["question_id"]]["prompt"],
                                           model_answer=qs[a["question_id"]].get("model_answer"),
                                           key_points=qs[a["question_id"]].get("key_points") or []) for a in sheet]
            try:
                grades = llm_grader.grade_sheet_llm(router, items, args.style)
            except Exception as exc:  # noqa: BLE001
                print(f"sheet {si}: unavailable ({type(exc).__name__})", file=sys.stderr)
                grades = {}
            if verify_on and grades:
                local = {qno[a["question_id"]]: local_grade(qs[a["question_id"]], a["text"]) for a in sheet}
                grades, recs = llm_grader.verify_sheet(router, items, grades, args.style,
                                                       local_marks={k: v for k, v in local.items() if v is not None})
                for n, r in recs.items():
                    verify_records[str(n) + "@" + str(si)] = r
            preds = {n: g.marks for n, g in grades.items()}
        latencies.append(time.perf_counter() - t0)
        for a in sheet:
            q = qs[a["question_id"]]
            n = qno[a["question_id"]]
            records.append({"id": a["id"], "question_id": a["question_id"], "category": a.get("category"),
                            "ambiguous": bool(a.get("ambiguous")), "max": float(q["max_marks"]),
                            "gold": float(a["expected_marks"]), "pred": preds.get(n),
                            "verification": verify_records.get(f"{n}@{si}")})
        done = sum(1 for r in records if r["pred"] is not None)
        print(f"  sheet {si + 1}/{len(sheets)}: {len(sheet)} answers, {latencies[-1]:.1f}s, marked {done}/{len(records)}",
              flush=True)
    wall = time.perf_counter() - started
    router.close()

    with factory() as db:
        rows = db.execute(select(AiRun.status, AiRun.task, func.count(AiRun.id),
                                 func.coalesce(func.sum(AiRun.input_tokens), 0),
                                 func.coalesce(func.sum(AiRun.output_tokens), 0),
                                 func.coalesce(func.avg(AiRun.latency_ms), 0))
                          .group_by(AiRun.status, AiRun.task)).all()
    usage = [{"status": r[0], "task": r[1], "rows": int(r[2]), "input_tokens": int(r[3]),
              "output_tokens": int(r[4]), "avg_latency_ms": round(float(r[5]))} for r in rows]
    sent = [u for u in usage if u["status"] != "cache_hit"]
    tok_in = sum(u["input_tokens"] for u in sent)
    tok_out = sum(u["output_tokens"] for u in sent)
    n_sheets = max(1, len(sheets))
    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in records if r["category"]}):
        by_cat[cat] = summarise([r for r in records if r["category"] == cat])
    vr = [r["verification"] for r in records if r["verification"]]
    report = {
        "label": args.label, "engine": engine_name, "engine_root": engine_root, "mode": args.mode,
        "data": str(args.data), "providers": providers, "style": args.style, "sheet_size": args.sheet_size,
        "n_sheets": len(sheets), "overall": summarise(records), "by_category": by_cat,
        "latency_s": {"wall_total": round(wall, 2), "mean_per_sheet": round(statistics.fmean(latencies), 2)
                      if latencies else None,
                      "p50_per_sheet": round(statistics.median(latencies), 2) if latencies else None,
                      "max_per_sheet": round(max(latencies), 2) if latencies else None},
        "requests": {"sent": sum(u["rows"] for u in sent), "ok": sum(u["rows"] for u in sent if u["status"] == "ok"),
                     "cache_hits": sum(u["rows"] for u in usage if u["status"] == "cache_hit"),
                     "per_sheet": round(sum(u["rows"] for u in sent) / n_sheets, 2)},
        "tokens": {"input": tok_in, "output": tok_out, "per_sheet": round((tok_in + tok_out) / n_sheets),
                   "per_answer": round((tok_in + tok_out) / max(1, len(records)))},
        "cost_usd_list_price": {"total": round((tok_in * args.price_in + tok_out * args.price_out) / 1e6, 5),
                                "per_sheet": round((tok_in * args.price_in + tok_out * args.price_out) / 1e6
                                                   / n_sheets, 6),
                                "prices_per_1m": [args.price_in, args.price_out]},
        "verification": {"answers_re_marked": len(vr),
                         "confirmed": sum(1 for r in vr if r.get("status") == "confirmed"),
                         "changed": sum(1 for r in vr if r.get("status") == "changed"),
                         "failed": sum(1 for r in vr if r.get("status") == "failed")},
        "usage_by_task": usage,
    }
    (args.out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    o = report["overall"]
    print(f"{engine_name} [{args.mode}] n={o['n']} scored={o['n_scored']} MAE={o['mae']} exact={o['exact']} "
          f"within0.5={o['within_0.5']} within1={o['within_1']} bias={o['mean_signed_error']} "
          f"| {report['requests']['sent']} requests, {tok_in}+{tok_out} tokens, "
          f"{report['latency_s']['mean_per_sheet']}s/sheet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
