"""End to end (PRD 20.2): every sample sheet uploaded in one batch, processed
with no manual step, every flagged answer reviewed (the image-only scan is
transcribed by the teacher), all sheets approved and the CSV export matching
the approved marks exactly. Per-sheet processing time stays under 60 s."""

from __future__ import annotations

import csv
import io
import time

from conftest import drain
from scripts import make_samples as samples

MAX_SECONDS_PER_SHEET = 60.0


def test_full_exam_workflow(teacher, exam, sample_bytes):
    refs = [s.student_ref for s in samples.TYPED_SHEETS] + [samples.IMAGE_ONLY_SHEET.student_ref]
    assert len(samples.TYPED_SHEETS) >= 10
    r = teacher.upload(exam["id"], [(f"{ref}.pdf", sample_bytes[ref]) for ref in refs], key="e2e-batch")
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["rejected"] == [] and len(body["submissions"]) == len(refs)
    ids = {s["student_ref"]: s["id"] for s in body["submissions"]}

    started = time.perf_counter()
    assert drain() == len(refs)
    wall = time.perf_counter() - started

    # 1. Every sheet reaches "processed" without any manual step.
    subs = {ref: teacher.submission(sid) for ref, sid in ids.items()}
    for ref, sub in subs.items():
        assert sub["status"] == "processed", (ref, sub["status"], sub["error"])
        assert sub["pages_done"] == sub["page_count"]
        assert 0 < sub["timings"]["total"] < MAX_SECONDS_PER_SHEET, (ref, sub["timings"])
    assert wall / len(refs) < MAX_SECONDS_PER_SHEET
    for spec in samples.TYPED_SHEETS:
        sub = subs[spec.student_ref]
        assert sub["ocr_engine"] in ("textlayer", "none") and sub["grader"] in ("lexical", None)
        for flag in spec.expect.get("flags", []):
            assert flag in sub["flags"], (spec.student_ref, flag)
        answers = {a["question_no"]: a for a in teacher.answers(sub["id"])}
        for no in spec.expect.get("blank", []):
            assert answers[no]["status"] == "blank" and answers[no]["ai_marks"] == 0, (spec.student_ref, no)
        for no in spec.expect.get("missing", []):  # 0 marks, but a teacher confirms it was skipped
            assert answers[no]["status"] == "needs_review" and answers[no]["ai_marks"] == 0, (spec.student_ref, no)
            assert "missing_marker" in answers[no]["flags"] and answers[no]["grader"] is None
        if "min_total" in spec.expect:
            assert sub["total_ai_marks"] >= spec.expect["min_total"]
    timings = {ref: s["timings"]["total"] for ref, s in subs.items()}
    print(f"\nper-sheet processing seconds: {timings} (wall {wall:.2f}s for {len(refs)} sheets)")

    # 2. The review queue: every flagged answer is reviewed by the teacher.
    image_ref = samples.IMAGE_ONLY_SHEET.student_ref
    transcripts = {1: samples.ANSWERS["q1_full"], 2: samples.ANSWERS["q2_full"], 3: samples.ANSWERS["q3_full"],
                   4: samples.ANSWERS["q4_full"]}
    reviewed = 0
    for ref, sid in ids.items():
        for a in teacher.answers(sid, sort="priority"):
            if a["status"] not in ("needs_review", "failed"):
                break  # priority order: flagged answers come first
            body = {"comment": "checked"}
            if ref == image_ref:
                body["corrected_text"] = transcripts[a["question_no"]]  # teacher transcribes the scan
            r = teacher.patch(f"/answers/{a['answer_id']}/review", json=body)
            assert r.status_code == 200, r.text
            assert r.json()["status"] == "reviewed"
            reviewed += 1
    assert reviewed >= 8
    image_total = sum(a["final_marks"] for a in teacher.answers(ids[image_ref]))
    assert image_total >= 14  # transcribed full-marks answers

    # 3. Approve everything in one go.
    r = teacher.post(f"/exams/{exam['id']}/approve-all")
    assert r.status_code == 200
    assert r.json()["blocked"] == [] and sorted(r.json()["approved"]) == sorted(ids.values())

    # 4. The CSV matches the approved marks exactly.
    export = teacher.get(f"/exams/{exam['id']}/export?format=csv")
    rows = {row["student_ref"]: row for row in csv.DictReader(io.StringIO(export.text))}
    assert set(rows) == set(ids)
    results = {i["student_ref"]: i for i in teacher.get(f"/exams/{exam['id']}/results").json()["items"]}
    for ref, sid in ids.items():
        answers = {a["question_no"]: a for a in teacher.answers(sid)}
        for no in (1, 2, 3, 4):
            assert answers[no]["locked"] is True
            assert float(rows[ref][f"Q{no}"]) == answers[no]["final_marks"] == results[ref]["cells"][no - 1]["marks"]
        total = round(sum(a["final_marks"] for a in answers.values()), 2)
        assert float(rows[ref]["total"]) == total == results[ref]["total"] == teacher.submission(sid)["total_final_marks"]
        assert rows[ref]["status"] == "approved"

    # 5. Analytics reflect the finished exam.
    an = teacher.get(f"/exams/{exam['id']}/analytics").json()
    assert an["sheets"] == len(ids) and an["highest"] == 15 and an["lowest"] == 0
    stats = teacher.get(f"/exams/{exam['id']}").json()["stats"]
    assert stats["approved"] == len(ids) and stats["needs_review"] == 0
