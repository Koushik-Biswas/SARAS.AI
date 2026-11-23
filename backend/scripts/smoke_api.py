"""End-to-end smoke test of the SARAS AI API, in-process (no server, no network).

    cd backend
    .venv/Scripts/python.exe scripts/smoke_api.py        (Windows)
    .venv/bin/python scripts/smoke_api.py                (Linux/macOS)

Uses a temporary SQLite database and storage folder, the background worker
disabled (jobs are drained with worker.run_pending()), the lexical grader and the
PDF text-layer OCR, so it needs no ML models and little RAM. Exits non-zero on the
first failed check and prints the offending response.
"""

from __future__ import annotations

import csv
import io
import shutil
import struct
import sys
import tempfile
import time
import warnings
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
# Starlette 1.x nudges TestClient users towards httpx2; irrelevant for this script.
warnings.filterwarnings("ignore", message=r"Using `httpx` with `starlette.testclient`")

import pymupdf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings, set_settings  # noqa: E402

API = "/api/v1"
PASSED = 0


class SmokeFailure(AssertionError):
    pass


def check(cond: bool, label: str, resp=None) -> None:  # noqa: ANN001
    global PASSED
    if not cond:
        detail = ""
        if resp is not None:
            detail = f"\n    HTTP {resp.status_code}: {resp.text[:1500]}"
        raise SmokeFailure(f"FAIL: {label}{detail}")
    PASSED += 1
    print(f"  ok  {label}")


def expect(resp, status: int, label: str, code: str | None = None):  # noqa: ANN001, ANN201
    """Check the HTTP status (and, for errors, the PRD error format and code)."""
    check(resp.status_code == status, f"{label} -> {status}", resp)
    if status >= 400:
        body = resp.json()
        err = body.get("error") if isinstance(body, dict) else None
        check(isinstance(err, dict) and {"code", "message", "details"} <= set(err),
              f"{label}: error format", resp)
        if code:
            check(err["code"] == code, f"{label}: code {code}", resp)
    return resp.json() if resp.content and resp.headers.get("content-type", "").startswith("application/json") else None


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()  # A4-ish 595x842 pt
        page.insert_text((60, 80), body, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", data[16:24])


SHEET_GOOD = [
    "Q1. Photosynthesis converts light energy into chemical energy.\n"
    "Chlorophyll in the leaves absorbs sunlight to drive the process.",
    "Q2. The heart pumps blood through the body.\n"
    "Arteries carry blood away from the heart to the organs.\n"
    "I am not sure about the rest of this answer.",
]
SHEET_OTHER = [
    "Q1. Plants make food using sunlight; chlorophyll absorbs the light.\n"
    "Light energy becomes chemical energy stored as glucose.",
    "Q2. Veins carry blood back to the heart.\n"
    "The heart pumps blood and arteries carry blood away from the heart.\n"
    "Capillaries let oxygen pass into the tissues.",
]

RUBRIC = {"questions": [
    {"question_no": 1, "type": "short", "max_marks": 4, "prompt": "What is photosynthesis?",
     "model_answer": "Photosynthesis converts light energy into chemical energy using chlorophyll.",
     "key_points": [
         {"text": "photosynthesis converts light energy into chemical energy", "weight": 2,
          "keywords": ["light", "chemical", "energy"]},
         {"text": "chlorophyll absorbs sunlight", "weight": 2, "keywords": ["chlorophyll", "sunlight"]},
     ]},
    {"question_no": 2, "type": "long", "max_marks": 6, "prompt": "Describe blood circulation.",
     "model_answer": "The heart pumps blood; arteries carry it away; veins return it; capillaries exchange.",
     "key_points": [
         {"text": "the heart pumps blood", "weight": 2, "keywords": ["heart", "pumps"]},
         {"text": "arteries carry blood away from the heart", "weight": 2, "keywords": ["arteries"]},
         {"text": "veins carry blood back to the heart", "weight": 2, "keywords": ["veins"]},
     ]},
]}


def run(client: TestClient, settings: Settings) -> None:
    from app import worker
    from app.db import SessionLocal
    from app.models import User

    def H(tok: str, **extra) -> dict:  # noqa: N802, ANN003
        return {"Authorization": f"Bearer {tok}", **extra}

    print("[system]")
    expect(client.get("/health"), 200, "GET /health")
    expect(client.get(f"{API}/health"), 200, "GET /api/v1/health")
    ready = expect(client.get(f"{API}/ready"), 200, "GET /api/v1/ready")
    check(ready == {"status": "ready", "database": "ok", "worker": "disabled"}, "ready body", None)
    expect(client.get(f"{API}/nope"), 404, "unknown API path", "not_found")
    expect(client.post(f"{API}/nope"), 404, "unknown API path (POST)", "not_found")
    expect(client.get(f"{API}/exams"), 401, "no bearer token", "unauthorized")

    print("[auth]")
    def register(name: str, email: str) -> str:
        r = client.post(f"{API}/auth/register", json={"name": name, "email": email, "password": "correct-horse-1"})
        return expect(r, 201, f"register {email}")["access_token"]

    tok_a = register("Teacher A", "a@example.com")
    tok_b = register("Teacher B", "b@example.com")
    register("Admin", "admin@example.com")
    with SessionLocal() as db:  # promote to admin directly in the database
        u = db.query(User).filter_by(email="admin@example.com").one()
        u.role = "admin"
        db.commit()
    tok_admin = expect(client.post(f"{API}/auth/login", json={"email": "admin@example.com",
                                                               "password": "correct-horse-1"}),
                       200, "admin login")["access_token"]
    expect(client.post(f"{API}/auth/login", json={"email": "a@example.com", "password": "wrong-password"}),
           401, "login with wrong password", "unauthorized")
    me = expect(client.get(f"{API}/auth/me", headers=H(tok_admin)), 200, "GET /auth/me")
    check(me["role"] == "admin", "admin role", None)

    caps = expect(client.get(f"{API}/capabilities", headers=H(tok_a)), 200, "GET /capabilities")
    check(caps["graders"]["lexical"] is True and caps["ocr"]["textlayer"] is True, "capabilities flags", None)
    check(caps["settings"]["max_pages"] == settings.max_pages, "capabilities settings", None)

    print("[exam + rubric]")
    exam = expect(client.post(f"{API}/exams", headers=H(tok_a),
                              json={"title": "Biology Unit Test", "subject": "Biology",
                                    "date": "2026-11-20", "total_marks": 10}), 201, "create exam")
    eid = exam["id"]
    pdf_good = make_pdf(SHEET_GOOD)
    r = client.post(f"{API}/exams/{eid}/submissions", headers=H(tok_a),
                    files=[("files", ("s1.pdf", pdf_good, "application/pdf"))])
    expect(r, 409, "upload before rubric", "conflict")
    qs = expect(client.put(f"{API}/exams/{eid}/questions", headers=H(tok_a), json=RUBRIC), 200, "PUT questions")
    check(len(qs) == 2, "two questions stored", None)
    qs = expect(client.post(f"{API}/exams/{eid}/questions", headers=H(tok_a), json=RUBRIC), 201,
                "POST questions alias")
    qid = {q["question_no"]: q["id"] for q in qs}
    expect(client.put(f"{API}/exams/{eid}/questions", headers=H(tok_admin), json=RUBRIC), 403,
           "admin edits rubric", "forbidden")
    expect(client.get(f"{API}/exams/{eid}", headers=H(tok_b)), 403, "other teacher reads exam", "forbidden")
    expect(client.get(f"{API}/exams/{eid}", headers=H(tok_admin)), 200, "admin reads exam")

    print("[uploads]")
    key = "smoke-key-0001"
    up = expect(client.post(f"{API}/exams/{eid}/submissions", headers=H(tok_a, **{"Idempotency-Key": key}),
                            files=[("files", ("Roll 001.pdf", pdf_good, "application/pdf"))],
                            data={"student_refs": ["S001"]}), 202, "upload sheet S001")
    check(len(up["submissions"]) == 1 and up["rejected"] == [], "one submission created", None)
    s1 = up["submissions"][0]
    check(s1["student_ref"] == "S001" and s1["status"] == "uploaded" and s1["page_count"] == 2,
          "S001 uploaded with 2 pages", None)
    again = expect(client.post(f"{API}/exams/{eid}/submissions", headers=H(tok_a, **{"Idempotency-Key": key}),
                               files=[("files", ("Roll 001.pdf", pdf_good, "application/pdf"))],
                               data={"student_refs": ["S001"]}), 202, "same Idempotency-Key again")
    check([x["id"] for x in again["submissions"]] == [s1["id"]], "idempotent: same submission returned", None)
    stored = settings.storage_dir / "submissions" / s1["id"] / "original.pdf"
    check(stored.is_file() and stored.read_bytes() == pdf_good, "file stored under a UUID folder", None)

    expect(client.post(f"{API}/exams/{eid}/submissions", headers=H(tok_a),
                       files=[("files", ("notes.pdf", b"just some text, not a pdf", "application/pdf"))]),
           415, "non-PDF upload", "unsupported_media_type")
    r = client.post(f"{API}/exams/{eid}/submissions", headers=H(tok_a),
                    files=[("files", ("broken.pdf", b"%PDF-1.7\n" + b"\x00garbage" * 50, "application/pdf"))])
    err = expect(r, 422, "corrupted PDF upload", "unreadable_pdf")
    check(isinstance(r.json()["error"]["details"], list) and r.json()["error"]["details"][0]["filename"] == "broken.pdf",
          "rejection list in details", r)
    expect(client.post(f"{API}/exams/{eid}/submissions", headers=H(tok_a),
                       files=[("files", ("again.pdf", pdf_good, "application/pdf"))],
                       data={"student_refs": ["S001"]}), 409, "duplicate student_ref", "conflict")
    big = settings.max_upload_bytes + 10
    expect(client.post(f"{API}/exams/{eid}/submissions", headers=H(tok_a),
                       files=[("files", ("huge.pdf", b"%PDF-1.4\n" + b"0" * big, "application/pdf"))]),
           413, "oversized upload", "file_too_large")
    leftovers = [p for p in (settings.storage_dir / "submissions").iterdir() if p.name != s1["id"]]
    check(leftovers == [], f"rejected uploads leave no files behind {[(p.name, [c.name for c in p.iterdir()]) for p in leftovers]}", None)

    mixed = expect(client.post(f"{API}/exams/{eid}/submissions", headers=H(tok_a),
                               files=[("files", ("S002.pdf", make_pdf(SHEET_OTHER), "application/pdf")),
                                      ("files", ("bad.pdf", b"not a pdf", "application/pdf"))]),
                   202, "mixed batch (one good, one bad)")
    check(len(mixed["submissions"]) == 1 and len(mixed["rejected"]) == 1, "batch isolation", None)
    s2 = mixed["submissions"][0]
    check(s2["student_ref"] == "S002", "student_ref derived from file name", None)
    check(mixed["rejected"][0]["code"] == "unsupported_media_type", "rejection code", None)
    expect(client.post(f"{API}/exams/{eid}/submissions", headers=H(tok_admin),
                       files=[("files", ("x.pdf", pdf_good, "application/pdf"))]), 403, "admin upload", "forbidden")

    st = expect(client.get(f"{API}/submissions/{s1['id']}/status", headers=H(tok_a)), 200, "status before processing")
    check(st["status"] == "uploaded" and st["pages_total"] == 2, "status: uploaded", None)
    expect(client.get(f"{API}/submissions/{s1['id']}/pages/1/image?thumb=1", headers=H(tok_a)), 200,
           "thumbnail before processing (rendered on the fly)")

    print("[processing]")
    t0 = time.perf_counter()
    n = worker.run_pending()
    check(n == 2, f"worker processed 2 jobs in {time.perf_counter() - t0:.1f}s", None)
    st = expect(client.get(f"{API}/submissions/{s1['id']}/status", headers=H(tok_a)), 200, "status after")
    check(st["status"] == "processed" and st["pages_done"] == 2 and st["attempts"] == 1, "status: processed", None)
    page = expect(client.get(f"{API}/exams/{eid}/submissions?page=1&page_size=1", headers=H(tok_a)), 200,
                  "list submissions (page 1, size 1)")
    check(page["total"] == 2 and len(page["items"]) == 1 and page["items"][0]["student_ref"] == "S001",
          "pagination + ordering by student_ref", None)
    flt = expect(client.get(f"{API}/exams/{eid}/submissions?status=processed", headers=H(tok_a)), 200,
                 "filter by status")
    check(flt["total"] == 2, "both processed", None)
    expect(client.get(f"{API}/exams/{eid}/submissions?page_size=101", headers=H(tok_a)), 400,
           "page_size > 100", "validation_error")

    answers = expect(client.get(f"{API}/submissions/{s1['id']}/answers?sort=question", headers=H(tok_a)), 200,
                     "list answers")
    check([a["question_no"] for a in answers] == [1, 2], "one answer per question", None)
    a1, a2 = answers
    print(f"      Q1 ai={a1['ai_marks']} conf={a1['confidence']} status={a1['status']} grader={a1['grader']} flags={a1['flags']}")
    print(f"      Q2 ai={a2['ai_marks']} conf={a2['confidence']} status={a2['status']} flags={a2['flags']}")
    check("photosynthesis" in a1["ocr_text"].lower() and not a1["ocr_text"].startswith("Q1"),
          "Q1 text read from the PDF text layer, marker stripped", None)
    check(len(a1["key_points"]) == 2 and a1["region_ids"], "key points + regions on answer", None)
    check(all(k["status"] in ("matched", "partial", "missed", "contradicted") for k in a1["key_points"]),
          "key point statuses", None)
    check(a1["matched_points"] == [k["text"] for k in a1["key_points"] if k["status"] == "matched"],
          "matched_points", None)
    pri = expect(client.get(f"{API}/submissions/{s1['id']}/answers", headers=H(tok_a)), 200, "priority sort")
    keys = [(0 if a["status"] in ("needs_review", "failed") else 1, a["confidence"]) for a in pri]
    check(keys == sorted(keys), "priority order", None)
    expect(client.get(f"{API}/submissions/{s1['id']}/answers?status=bogus", headers=H(tok_a)), 400,
           "bad answer filter", "validation_error")

    pages = expect(client.get(f"{API}/submissions/{s1['id']}/pages", headers=H(tok_a)), 200, "pages")
    check(len(pages["pages"]) == 2 and pages["pages"][0]["width"] > 0, "page sizes", None)
    img = client.get(f"{API}/submissions/{s1['id']}/pages/1/image", headers=H(tok_a))
    check(img.status_code == 200 and img.headers["content-type"] == "image/png", "page image PNG", img)
    check(png_size(img.content) == (pages["pages"][0]["width"], pages["pages"][0]["height"]),
          "image size matches /pages", None)
    th = client.get(f"{API}/submissions/{s1['id']}/pages/1/image?thumb=1", headers=H(tok_a))
    check(th.status_code == 200 and png_size(th.content)[0] == 240, "thumbnail is 240 px wide", th)
    expect(client.get(f"{API}/submissions/{s1['id']}/pages/9/image", headers=H(tok_a)), 404, "missing page",
           "not_found")
    expect(client.get(f"{API}/submissions/{s1['id']}/pages/1/image", headers=H(tok_b)), 403,
           "other teacher's page image", "forbidden")

    print("[review]")
    new_marks = 1.5 if a1["ai_marks"] != 1.5 else 2.5
    expect(client.patch(f"{API}/answers/{a1['answer_id']}/review", headers=H(tok_a),
                        json={"final_marks": new_marks}), 400, "mark change without reason", "validation_error")
    r = client.patch(f"{API}/answers/{a1['answer_id']}/review", headers=H(tok_a), json={"final_marks": new_marks})
    check(r.json()["error"]["details"][0]["field"] == "reason", "error points at 'reason'", r)
    expect(client.patch(f"{API}/answers/{a1['answer_id']}/review", headers=H(tok_a),
                        json={"final_marks": 4.5, "reason": "x"}), 400, "marks above max", "validation_error")
    expect(client.patch(f"{API}/answers/{a1['answer_id']}/review", headers=H(tok_a),
                        json={"final_marks": 1.234, "reason": "x"}), 400, "more than 2 decimals", "validation_error")
    expect(client.patch(f"{API}/answers/{a1['answer_id']}/review", headers=H(tok_admin),
                        json={"final_marks": new_marks, "reason": "admin"}), 403, "admin edits marks", "forbidden")
    expect(client.patch(f"{API}/answers/{a1['answer_id']}/review", headers=H(tok_b),
                        json={"final_marks": new_marks, "reason": "b"}), 403, "other teacher edits marks", "forbidden")
    rv = expect(client.patch(f"{API}/answers/{a1['answer_id']}/review", headers=H(tok_a),
                             json={"final_marks": new_marks, "reason": "Explanation incomplete",
                                   "comment": "Good start"}), 200, "review with reason")
    check(rv["status"] == "reviewed" and rv["final_marks"] == new_marks and rv["comment"] == "Good start",
          "answer reviewed with teacher marks", None)
    sub = expect(client.get(f"{API}/submissions/{s1['id']}", headers=H(tok_a)), 200, "get submission")
    check(sub["status"] == "in_review", "sheet moved to in_review", None)

    fl = expect(client.post(f"{API}/answers/{a2['answer_id']}/flag", headers=H(tok_a),
                            json={"reason": "check handwriting"}), 200, "flag Q2")
    check(fl["status"] == "needs_review" and "teacher_flag" in fl["flags"], "flagged for review", None)
    r = client.post(f"{API}/submissions/{s1['id']}/approve", headers=H(tok_a))
    expect(r, 409, "approve with pending answers", "conflict")
    check(r.json()["error"]["details"] == {"pending": [2]}, "pending lists Q2", r)

    corrected = ("The heart pumps blood through the body. Arteries carry blood away from the heart. "
                 "Veins carry blood back to the heart.")
    ed = expect(client.patch(f"{API}/answers/{a2['answer_id']}/review", headers=H(tok_a),
                             json={"corrected_text": corrected, "approve": False}), 200, "save corrected text only")
    check(ed["text_corrected"] and ed["ocr_text"] == corrected and ed["status"] == "needs_review",
          "text saved + regraded, still in queue", None)
    check(ed["ai_marks"] >= a2["ai_marks"], f"regrade after correction ({a2['ai_marks']} -> {ed['ai_marks']})", None)
    ok2 = expect(client.patch(f"{API}/answers/{a2['answer_id']}/review", headers=H(tok_a), json={}),
                 200, "approve Q2 as graded (no reason needed)")
    check(ok2["status"] == "reviewed" and ok2["final_marks"] == ok2["ai_marks"], "Q2 final = AI marks", None)
    expect(client.post(f"{API}/submissions/{s1['id']}/approve", headers=H(tok_admin)), 403, "admin approves",
           "forbidden")
    ap = expect(client.post(f"{API}/submissions/{s1['id']}/approve", headers=H(tok_a)), 200, "approve sheet")
    expected_total = round(new_marks + ok2["final_marks"], 2)
    check(ap["status"] == "approved" and ap["approved_at"] and ap["total_final_marks"] == expected_total,
          f"approved, total {expected_total}", None)
    expect(client.patch(f"{API}/answers/{a1['answer_id']}/review", headers=H(tok_a),
                        json={"final_marks": 0, "reason": "late"}), 409, "edit after approval", "conflict")
    locked = expect(client.get(f"{API}/answers/{a1['answer_id']}", headers=H(tok_a)), 200, "get answer")
    check(locked["locked"] is True, "answer reports locked", None)

    print("[export]")
    r = client.get(f"{API}/exams/{eid}/export?format=csv", headers=H(tok_a))
    check(r.status_code == 200 and r.headers["content-type"].startswith("text/csv"), "export CSV", r)
    rows = list(csv.reader(io.StringIO(r.text)))
    check(rows[0][:4] == ["student_ref", "Q1", "Q2", "total"], "CSV header", None)
    check(len(rows) == 2 and rows[1][0] == "S001", "only the approved sheet is exported", None)
    check(float(rows[1][1]) == new_marks and float(rows[1][2]) == ok2["final_marks"]
          and float(rows[1][3]) == ap["total_final_marks"], "CSV marks match the approved marks", None)
    res = expect(client.get(f"{API}/exams/{eid}/results", headers=H(tok_a)), 200, "results grid")
    row = next(i for i in res["items"] if i["student_ref"] == "S001")
    check([c["marks"] for c in row["cells"]] == [new_marks, ok2["final_marks"]], "results match CSV", None)
    expect(client.get(f"{API}/exams/{eid}/analytics", headers=H(tok_a)), 200, "analytics")

    print("[reopen]")
    expect(client.post(f"{API}/submissions/{s1['id']}/reopen", headers=H(tok_a), json={"reason": ""}), 400,
           "reopen without reason", "validation_error")
    ro = expect(client.post(f"{API}/submissions/{s1['id']}/reopen", headers=H(tok_a),
                            json={"reason": "Student appeal"}), 200, "reopen with reason")
    check(ro["status"] == "in_review" and ro["approved_at"] is None, "back in review", None)
    expect(client.post(f"{API}/submissions/{s1['id']}/approve", headers=H(tok_a)), 200, "re-approve")

    print("[regions on S002]")
    regs = expect(client.get(f"{API}/submissions/{s2['id']}/regions", headers=H(tok_a)), 200, "list regions")
    q1_regions = [g for g in regs if g["question_id"] == qid[1]]
    q2_regions = [g for g in regs if g["question_id"] == qid[2]]
    check(len(q1_regions) == 1 and len(q2_regions) == 1, "one auto region per question", None)
    r_q2 = q2_regions[0]
    check(r_q2["page_no"] == 2 and r_q2["source"] == "auto", "Q2 region on page 2", None)
    expect(client.patch(f"{API}/regions/{r_q2['id']}", headers=H(tok_admin),
                        json={"action": "reassign", "question_id": qid[1]}), 403, "admin edits a region", "forbidden")
    expect(client.patch(f"{API}/regions/{r_q2['id']}", headers=H(tok_a),
                        json={"action": "crop", "bbox": [0, 0, 99999, 50]}), 400, "crop outside the page",
           "validation_error")
    before_answers = {a["question_no"]: a for a in expect(
        client.get(f"{API}/submissions/{s2['id']}/answers?sort=question", headers=H(tok_a)), 200, "S002 answers")}
    ed = expect(client.patch(f"{API}/regions/{r_q2['id']}", headers=H(tok_a),
                             json={"action": "reassign", "question_id": qid[1], "reason": "Q2 text belongs to Q1"}),
                200, "reassign Q2 region to Q1")
    check(ed["region"]["question_id"] == qid[1] and ed["region"]["source"] == "manual", "region reassigned", None)
    by_no = {a["question_no"]: a for a in ed["answers"]}
    check(set(by_no) == {1, 2}, "both affected answers returned", None)
    check(by_no[2]["status"] == "blank" and by_no[2]["ai_marks"] == 0 and by_no[2]["region_ids"] == [],
          "Q2 now blank with 0 marks", None)
    check(set(by_no[1]["region_ids"]) == {q1_regions[0]["id"], r_q2["id"]}, "Q1 now has both regions", None)
    check("heart" in by_no[1]["ocr_text"].lower(), "Q1 text now includes the moved lines", None)

    undo = expect(client.post(f"{API}/submissions/{s2['id']}/regions/undo", headers=H(tok_a)), 200, "undo")
    check(next(g for g in undo["regions"] if g["id"] == r_q2["id"])["question_id"] == qid[2]
          and next(g for g in undo["regions"] if g["id"] == r_q2["id"])["source"] == "auto", "region restored", None)
    after = {a["question_no"]: a for a in expect(
        client.get(f"{API}/submissions/{s2['id']}/answers?sort=question", headers=H(tok_a)), 200, "answers after undo")}
    check(after[2]["ai_marks"] == before_answers[2]["ai_marks"] and after[2]["ocr_text"] == before_answers[2]["ocr_text"]
          and after[1]["ai_marks"] == before_answers[1]["ai_marks"], "answers restored by undo", None)
    expect(client.post(f"{API}/submissions/{s2['id']}/regions/undo", headers=H(tok_a)), 409, "nothing left to undo",
           "conflict")

    x, y, w, h = q1_regions[0]["bbox"]
    sp = expect(client.patch(f"{API}/regions/{q1_regions[0]['id']}", headers=H(tok_a),
                             json={"action": "split", "split_y": y + h // 2}), 200, "split Q1 region")
    check(len([g for g in sp["regions"] if g["question_id"] == qid[1]]) == 2, "split -> two Q1 regions", None)
    lower = next(g for g in sp["regions"] if g["question_id"] == qid[1] and g["id"] != q1_regions[0]["id"])
    mg = expect(client.patch(f"{API}/regions/{q1_regions[0]['id']}", headers=H(tok_a),
                             json={"action": "merge", "merge_with": lower["id"]}), 200, "merge them back")
    check(mg["region"]["bbox"] == [x, y, w, h], "merged box equals the original", None)
    expect(client.patch(f"{API}/regions/{q1_regions[0]['id']}", headers=H(tok_a),
                        json={"action": "split", "split_y": y}), 400, "split on the edge", "validation_error")
    pw, ph = pages["pages"][0]["width"], pages["pages"][0]["height"]
    cr = expect(client.post(f"{API}/submissions/{s2['id']}/regions", headers=H(tok_a),
                            json={"page_no": 1, "bbox": [10, ph - 60, pw - 20, 50], "question_id": qid[2]}),
                201, "draw a new region")
    check(cr["region"]["source"] == "manual", "new region is manual", None)
    expect(client.post(f"{API}/submissions/{s2['id']}/regions/undo", headers=H(tok_a)), 200, "undo the new region")
    expect(client.post(f"{API}/submissions/{s2['id']}/regions/undo", headers=H(tok_a)), 200, "undo the merge")
    expect(client.post(f"{API}/submissions/{s2['id']}/regions/undo", headers=H(tok_a)), 200, "undo the split")
    regs_end = expect(client.get(f"{API}/submissions/{s2['id']}/regions", headers=H(tok_a)), 200, "regions after undos")
    check(sorted((g["id"], tuple(g["bbox"]), g["question_id"]) for g in regs_end)
          == sorted((g["id"], tuple(g["bbox"]), g["question_id"]) for g in regs), "all edits undone", None)
    expect(client.post(f"{API}/submissions/{s2['id']}/retry", headers=H(tok_a)), 409,
           "retry after teacher edits", "conflict")

    print("[audit]")
    au = expect(client.get(f"{API}/audit-log?entity=answer&id={a1['answer_id']}", headers=H(tok_a)), 200,
                "audit log for Q1 answer")
    review_entry = next((e for e in au["items"] if e["action"] == "review"), None)
    check(review_entry is not None and review_entry["reason"] == "Explanation incomplete"
          and review_entry["new_value"]["final_marks"] == new_marks and review_entry["old_value"]["final_marks"] is None,
          "review audited with old/new marks and reason", None)
    sa = expect(client.get(f"{API}/audit-log?entity=submission&id={s2['id']}&page_size=100", headers=H(tok_a)), 200,
                "audit log for S002")
    actions = [e["action"] for e in sa["items"]]
    check(actions.count("regions_edit") == 4 and actions.count("regions_undo") == 4, "region edits audited", None)
    check(sa["items"][0]["id"] > sa["items"][-1]["id"], "newest first", None)
    expect(client.get(f"{API}/audit-log?entity=answer&id={a1['answer_id']}", headers=H(tok_b)), 403,
           "other teacher's audit log", "forbidden")
    expect(client.get(f"{API}/audit-log?entity=answer&id={a1['answer_id']}", headers=H(tok_admin)), 200,
           "admin reads audit log")
    with SessionLocal() as db:
        from sqlalchemy import text
        try:
            db.execute(text("UPDATE audit_log SET reason='tampered'"))
            db.commit()
            tampered = True
        except Exception:
            db.rollback()
            tampered = False
    check(not tampered, "audit log is append-only", None)

    print("[approve-all, retry, delete]")
    aa = expect(client.post(f"{API}/exams/{eid}/approve-all", headers=H(tok_a)), 200, "approve-all")
    s2_answers = expect(client.get(f"{API}/submissions/{s2['id']}/answers", headers=H(tok_a)), 200, "S002 answers")
    pending = sorted(a["question_no"] for a in s2_answers if a["status"] in ("needs_review", "failed"))
    if pending:
        check(aa["approved"] == [] and aa["blocked"][0]["pending"] == pending, f"S002 blocked on {pending}", None)
    else:
        check(aa["approved"] == [s2["id"]], "S002 approved", None)
    up3 = expect(client.post(f"{API}/exams/{eid}/submissions", headers=H(tok_a),
                             files=[("files", ("S003.pdf", make_pdf(SHEET_OTHER), "application/pdf"))]),
                 202, "upload S003")
    s3 = up3["submissions"][0]
    worker.run_pending()
    rt = expect(client.post(f"{API}/submissions/{s3['id']}/retry", headers=H(tok_a)), 202, "retry unreviewed sheet")
    check(rt["status"] == "uploaded", "re-queued", None)
    expect(client.post(f"{API}/submissions/{s3['id']}/retry", headers=H(tok_a)), 409, "retry while queued", "conflict")
    worker.run_pending()
    st3 = expect(client.get(f"{API}/submissions/{s3['id']}/status", headers=H(tok_a)), 200, "status after retry")
    check(st3["status"] == "processed", "processed again", None)
    folder = settings.storage_dir / "submissions" / s3["id"]
    check(folder.is_dir(), "S003 files exist", None)
    expect(client.delete(f"{API}/submissions/{s3['id']}", headers=H(tok_admin)), 403, "admin deletes", "forbidden")
    r = client.delete(f"{API}/submissions/{s3['id']}", headers=H(tok_a))
    check(r.status_code == 204, "delete S003 -> 204", r)
    check(not folder.exists(), "S003 files removed", None)
    expect(client.get(f"{API}/submissions/{s3['id']}", headers=H(tok_a)), 404, "deleted sheet is gone", "not_found")
    expect(client.delete(f"{API}/submissions/{s1['id']}", headers=H(tok_a)), 409, "delete an approved sheet",
           "conflict")
    lst = expect(client.get(f"{API}/exams/{eid}/submissions", headers=H(tok_admin)), 200, "admin lists sheets")
    check(lst["total"] == 2, "two sheets remain", None)

    print("[delete exam]")
    folders = [settings.storage_dir / "submissions" / x for x in (s1["id"], s2["id"])]
    check(all(f.is_dir() for f in folders), "sheet folders exist before deleting the exam", None)
    expect(client.delete(f"{API}/exams/{eid}", headers=H(tok_admin)), 403, "admin deletes exam", "forbidden")
    r = client.delete(f"{API}/exams/{eid}", headers=H(tok_a))
    check(r.status_code == 204, "delete exam -> 204", r)
    check(not any(f.exists() for f in folders), "all sheet files removed with the exam", None)
    expect(client.get(f"{API}/submissions/{s1['id']}", headers=H(tok_a)), 404, "sheets gone with the exam", "not_found")
    ex = expect(client.get(f"{API}/exams", headers=H(tok_a)), 200, "list exams")
    check(ex["total"] == 0, "no exams left", None)
    hist = expect(client.get(f"{API}/audit-log?entity=exam", headers=H(tok_a)), 200, "own exam audit history")
    check(hist["items"][0]["action"] == "delete", "exam deletion audited", None)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="saras-smoke-"))
    settings = Settings(
        env="test",
        database_url=f"sqlite:///{(tmp / 'smoke.db').as_posix()}",
        storage_dir=tmp / "storage",
        jwt_secret="smoke-test-secret-not-for-production",
        worker_enabled=False,
        rate_limit_enabled=False,
        grader="lexical",
        ocr_engine="auto",
        frontend_dist=tmp / "no-frontend",
        log_level="WARNING",
    )
    from app.main import create_app

    app = create_app(settings)
    started = time.perf_counter()
    try:
        with TestClient(app) as client:
            run(client, settings)
    except SmokeFailure as exc:
        print(f"\n{exc}\n\n{PASSED} checks passed before the failure.")
        return 1
    finally:
        set_settings(None)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nSMOKE OK: {PASSED} checks passed in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
