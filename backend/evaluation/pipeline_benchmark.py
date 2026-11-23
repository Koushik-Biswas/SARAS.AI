"""End-to-end pipeline benchmark on a synthetic handwritten-style answer sheet.

Builds a 5-page image-only sheet (Windows script fonts Ink Free / Segoe Print, scanner
noise, slight skew; maths and a table among the answers) with KNOWN ground-truth
text, then runs the real ``app.pipeline.processor.process_submission`` on it, exactly
as the worker does, and measures wall time, per-stage time, pages read / unread, AI
requests and tokens, OCR accuracy (CER / WER against the ground truth) and the marks.

Like evaluation/ai_benchmark.py it imports `app` from PYTHONPATH, so the same script
measures the current engine and an older copy of it.

Scenarios
  overload  deterministic fault injection over httpx.MockTransport, reproducing the
            2026-11-23 incident: every Gemini model answers HTTP 503 "high demand"
            after ~2 s; Groq's vision model reads a page in ~2.5 s; the marker answers
            in ~1.5 s. No network, no tokens; real wall-clock latencies and backoff.
  live      the real providers configured in backend/.env (needs consent + keys);
            measures real OCR accuracy, latency and tokens. Synthetic data only: no
            student's work is sent.

    python evaluation/pipeline_benchmark.py --scenario overload --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
FONTS = [Path("C:/Windows/Fonts/Inkfree.ttf"), Path("C:/Windows/Fonts/segoepr.ttf")]

# One question per page. Ground truth = exactly the text drawn on the page.
PAGES = [
    ("Q1. Overfitting means the model learns the training data too closely,",
     "including its noise, so it gets very high training accuracy but",
     "performs poorly on new unseen test data. It does not generalise.",
     "We can reduce it by regularisation (L1/L2), dropout or more data."),
    ("Q2. Newton's second law: F = m x a",
     "m = 2 kg, a = 5 m/s^2",
     "F = 2 x 5 = 10 N",
     "So the net force on the body is 10 N in the direction of a."),
    ("Q3. Arteries carry blood away from the heart and have thick,",
     "elastic, muscular walls. Veins bring blood back to the heart,",
     "have thin walls and contain valves to stop backflow of blood."),
    ("Q4. Stack | LIFO | push and pop at the top",
     "Queue | FIFO | enqueue at rear, dequeue at front",
     "Example: undo in an editor uses a stack, a printer uses a queue."),
    ("Q5. The greenhouse effect: gases like CO2 and methane trap",
     "infrared heat radiated by the Earth's surface, which keeps",
     "the average temperature near 15 C instead of about -18 C."),
]
QUESTIONS = [
    (1, "short", 2.0, "What is overfitting in machine learning? Mention one way to reduce it."),
    (2, "short", 3.0, "State Newton's second law and find the force on a 2 kg body accelerating at 5 m/s^2."),
    (3, "short", 3.0, "Give three differences between arteries and veins."),
    (4, "long", 4.0, "Compare a stack and a queue, with one example of each."),
    (5, "long", 4.0, "Explain the greenhouse effect and its effect on the Earth's temperature."),
]


def build_sheet(path: Path, seed: int = 7) -> list[str]:
    """Write the image-only PDF; returns the ground-truth text per page."""
    import cv2
    import numpy as np
    import pymupdf

    rng = random.Random(seed)
    out = pymupdf.open()
    truth = []
    for i, lines in enumerate(PAGES):
        src = pymupdf.open()
        width = 595 + 12 * i  # pages differ slightly in width: the fake reader tells them apart
        page = src.new_page(width=width, height=842)
        font = FONTS[i % len(FONTS)]
        for k in range(26):  # ruled lines, like an answer booklet
            page.draw_line((40, 80 + 28 * k), (width - 40, 80 + 28 * k), color=(0.75, 0.8, 0.9), width=0.6)
        y = 100
        for ln in lines:
            page.insert_text((55 + rng.randint(-4, 6), y), ln, fontsize=15 + rng.randint(-1, 1),
                             fontname=f"F{i % 2}", fontfile=str(font), color=(0.1, 0.12, 0.35))
            y += 56
        pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).astype(np.float32)
        noise = np.random.default_rng(seed + i).normal(0, 9, img.shape)  # scanner noise
        img = np.clip(img * 0.93 + 10 + noise, 0, 255).astype(np.uint8)
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-1.6, 1.6), 1.0)  # skew
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        ok, png = cv2.imencode(".png", img)
        assert ok
        new = out.new_page(width=width, height=842)
        new.insert_image(new.rect, stream=png.tobytes())
        src.close()
        truth.append("\n".join(lines))
    out.save(str(path))
    out.close()
    return truth


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[“”]", '"', re.sub(r"[‘’]", "'", text)).split())


def _lev(a: list | str, b: list | str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    p, r = _norm(pred), _norm(ref)
    return _lev(p, r) / max(1, len(r))


def wer(pred: str, ref: str) -> float:
    p, r = _norm(pred).lower().split(), _norm(ref).lower().split()
    return _lev(p, r) / max(1, len(r))


def overload_transport(truth: list[str], pages_px: int):  # noqa: ANN201
    """FakeLLMServer reproducing the incident (real sleeps = real wall clock)."""
    import base64

    import cv2
    import numpy as np

    from app.llm.fakes import FakeLLMServer, FakeReply

    def page_index(call) -> int:  # noqa: ANN001 - page i is 595 + 12*i points wide
        data = call.body["messages"][-1]["content"][1]["image_url"]["url"].split(",", 1)[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        return min(len(truth) - 1, max(0, round((w / h * 842 - 595) / 12)))

    def respond(call):  # noqa: ANN001, ANN202
        prompt = call.prompt
        if call.provider == "gemini":
            time.sleep(2.0)
            return FakeReply(status=503, body={"error": {"message": "This model is currently experiencing high demand. "
                                                                   "Spikes in demand are usually temporary."}})
        if prompt.startswith("This is one page"):
            time.sleep(2.5)
            text = truth[page_index(call)]
            return {"lines": [{"text": t, "box_2d": [100 + 60 * j, 60, 140 + 60 * j, 940], "unclear": False}
                              for j, t in enumerate(text.split("\n"))]}
        time.sleep(1.5)
        return None  # marking requests: handled by the marker below

    return FakeLLMServer(responder=respond)


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", choices=("overload", "live"), required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    pdf = args.out / "sheet.pdf"
    truth = build_sheet(pdf)

    if args.scenario == "live":
        from ai_benchmark import load_env_file  # noqa: PLC0415 - same folder

        load_env_file(BACKEND / ".env")
    os.environ["SARAS_ENV"] = "test"
    db_path = args.out / "pipeline.db"
    os.environ["SARAS_DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ["SARAS_STORAGE_DIR"] = str(args.out / "storage")
    os.environ["SARAS_WORKER_ENABLED"] = "false"

    import app
    from sqlalchemy import func, select

    from app import db as dbm
    from app.config import Settings
    from app.models import AiRun, Answer, Exam, OcrLine, Question, Submission, User
    from app.pipeline import processor

    engine_root = str(Path(app.__file__).resolve().parent)
    if args.scenario == "overload":
        overrides = dict(ai_consent=True, gemini_consent=True, ai_ocr=True, ai_grading="ensemble",
                         gemini_api_key="fake-gemini-key-for-benchmark-0001",
                         groq_api_key="fake-groq-key-for-benchmark-0001",
                         ai_ocr_providers=["gemini:gemini-flash-lite-latest", "groq:qwen/qwen3.8-27b",
                                           "gemini:gemini-3.5-flash-lite", "gemini:gemini-3.1-flash-lite",
                                           "gemini:gemini-flash-latest"],
                         ai_grading_providers=["groq:openai/gpt-oss-120b"],
                         ai_timeout_seconds=45, ai_call_deadline_seconds=120, review_required=False)
    else:
        overrides = dict(ai_grading="ensemble", review_required=False)
    settings = Settings(**{k: v for k, v in overrides.items() if k in Settings.model_fields})
    dbm.init_engine(settings.database_url)
    dbm.create_schema()

    server = None
    if args.scenario == "overload":
        import json as _json

        from app.grading.llm_grader import USER_PREFIX
        from app.llm import LLMRouter

        server = overload_transport(truth, 0)
        base_responder = server.responder

        def respond(call):  # noqa: ANN001, ANN202
            out = base_responder(call)
            if out is not None:
                return out
            p = call.prompt
            if p.startswith("Write the marking scheme"):
                return {"model_answer": "ref", "accept_also": "",
                        "criteria": [{"text": "main idea", "marks": 1}, {"text": "detail", "marks": 1}]}
            if p.startswith(USER_PREFIX):
                qs = _json.loads(p[len(USER_PREFIX):])["questions"]
                return {"questions": [{"question_no": q["question_no"], "uncertain": False, "feedback": "ok",
                                       "key_points": [{"index": k["index"], "verdict": "matched", "reason": "ok",
                                                       "evidence_quote": " ".join(q["answer_text"].split()[:5])}
                                                      for k in q["key_points"]]} for q in qs]}
            if p.startswith("Mark these answers"):
                items = _json.loads(p.split("\n", 1)[1])
                return {"answers": [{"question_no": it["question_no"], "marks": it["max_marks"], "feedback": "ok",
                                     "evidence": "", "uncertain": False} for it in items]}
            return None

        server.responder = respond

        def make(s, session_factory):  # noqa: ANN001, ANN202
            return LLMRouter(s, session_factory, transport=server.transport)

        processor.make_ai_router = make

    with dbm.SessionLocal() as db:
        user = User(name="Bench", email="bench@example.invalid", password_hash="x")
        db.add(user)
        db.flush()
        exam = Exam(teacher_id=user.id, title="Pipeline benchmark", total_marks=sum(q[2] for q in QUESTIONS))
        db.add(exam)
        db.flush()
        for no, typ, mx, prompt in QUESTIONS:
            db.add(Question(exam_id=exam.id, question_no=no, type=typ, max_marks=mx, prompt=prompt,
                            model_answer=None, key_points=[]))
        folder = settings.storage_dir / "submissions" / "bench"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf, folder / "original.pdf")
        sub = Submission(id="00000000-0000-0000-0000-00000000beef", exam_id=exam.id, student_ref="BENCH-1",
                         original_filename="sheet.pdf", pdf_path=str(folder / "original.pdf"), page_count=len(PAGES),
                         status="uploaded", stage="queued", flags=[], page_flags={})
        db.add(sub)
        db.commit()
        sid = sub.id

    t0 = time.perf_counter()
    with dbm.SessionLocal() as db:
        processor.process_submission(db, sid, settings)
    wall = time.perf_counter() - t0

    with dbm.SessionLocal() as db:
        sub = db.get(Submission, sid)
        lines = db.scalars(select(OcrLine).where(OcrLine.submission_id == sid).order_by(OcrLine.line_no)).all()
        per_page = {p: "\n".join(ln.text for ln in lines if ln.page_no == p and ln.text.strip())
                    for p in range(1, len(PAGES) + 1)}
        answers = db.scalars(select(Answer).where(Answer.submission_id == sid)).all()
        qno = {q.id: q.question_no for q in db.scalars(select(Question).where(Question.exam_id == sub.exam_id))}
        ai = db.execute(select(AiRun.task, AiRun.status, AiRun.provider, func.count(AiRun.id),
                               func.coalesce(func.sum(AiRun.input_tokens), 0),
                               func.coalesce(func.sum(AiRun.output_tokens), 0))
                        .group_by(AiRun.task, AiRun.status, AiRun.provider)).all()
        read_pages = sum(1 for t in per_page.values() if t)
        cers = [cer(per_page[p], truth[p - 1]) for p in per_page]
        wers = [wer(per_page[p], truth[p - 1]) for p in per_page]
        report = {
            "label": args.label, "scenario": args.scenario, "engine_root": engine_root,
            "wall_seconds": round(wall, 2), "timings": sub.timings, "status": sub.status,
            "ocr_engine": sub.ocr_engine, "pages": len(PAGES), "pages_read": read_pages,
            "pages_unread": len(PAGES) - read_pages, "sheet_flags": sub.flags, "page_flags": sub.page_flags,
            "ocr": {"cer_mean": round(sum(cers) / len(cers), 4), "wer_mean": round(sum(wers) / len(wers), 4),
                    "cer_by_page": [round(x, 4) for x in cers], "wer_by_page": [round(x, 4) for x in wers]},
            "answers": sorted([{"question_no": qno[a.question_id], "status": a.status, "marks": a.ai_marks,
                                "flags": a.flags, "grader": a.grader} for a in answers], key=lambda x: x["question_no"]),
            "total_ai_marks": sub.total_ai_marks,
            "ai_runs": [{"task": r[0], "status": r[1], "provider": r[2], "n": int(r[3]), "input_tokens": int(r[4]),
                         "output_tokens": int(r[5])} for r in ai],
            "ai_requests_sent": sum(int(r[3]) for r in ai if r[1] != "cache_hit"),
            "fake_calls": len(server.calls) if server else None,
            "progress_stages": [(s["key"], s["state"], s["seconds"]) for s in (getattr(sub, "progress", None) or {})
                                .get("stages", [])] if getattr(sub, "progress", None) else None,
        }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{args.scenario}] {engine_root}\n  wall={report['wall_seconds']}s pages_read={read_pages}/{len(PAGES)} "
          f"CER={report['ocr']['cer_mean']} WER={report['ocr']['wer_mean']} ai_requests={report['ai_requests_sent']} "
          f"timings={sub.timings}")
    for a in report["answers"]:
        print(f"  Q{a['question_no']}: {a['status']:<14} marks={a['marks']} flags={','.join(a['flags'] or [])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
