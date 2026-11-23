"""Answer-region editing (PRD 12.10, screen 6): reassign / crop / split / merge /
blank regions, answers recomputed from stored OCR lines, undo, audit, plus the
manual-transcription path for an image-only scan."""

from __future__ import annotations

import pytest

from conftest import accept_lexical, assert_error, error_fields
from scripts import make_samples as samples


@pytest.fixture
def sheet(process) -> dict:
    return process("ML001_full_marks")["ML001_full_marks"]


def _by_question(regions: list[dict], qid: str) -> list[dict]:
    return [r for r in regions if r["question_id"] == qid]


def test_regions_listed_in_page_then_y_order(teacher, exam, sheet):
    regions = teacher.regions(sheet["id"])
    assert len(regions) == 4
    assert [r["question_id"] for r in regions] == [exam["qid"][n] for n in (1, 2, 3, 4)]
    assert [r["bbox"][1] for r in regions] == sorted(r["bbox"][1] for r in regions)
    assert all(r["source"] == "auto" and not r["is_blank"] and len(r["bbox"]) == 4 for r in regions)
    pages = teacher.get(f"/submissions/{sheet['id']}/pages").json()["pages"]
    assert len(pages) == 1 and pages[0]["width"] > 1000 and pages[0]["height"] > pages[0]["width"]
    png = teacher.get(f"/submissions/{sheet['id']}/pages/1/image")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png" and png.content[:4] == b"\x89PNG"
    thumb = teacher.get(f"/submissions/{sheet['id']}/pages/1/image?thumb=1")
    assert thumb.status_code == 200 and int.from_bytes(thumb.content[16:20], "big") == 240
    assert_error(teacher.get(f"/submissions/{sheet['id']}/pages/2/image"), 404, "not_found")


def test_reassign_region_recomputes_answers(teacher, exam, sheet):
    q1_region = _by_question(teacher.regions(sheet["id"]), exam["qid"][1])[0]
    r = teacher.patch(f"/regions/{q1_region['id']}",
                      json={"action": "reassign", "question_id": exam["qid"][2], "reason": "wrong marker"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["region"]["question_id"] == exam["qid"][2] and out["region"]["source"] == "manual"
    answers = {a["question_no"]: a for a in out["answers"]}
    assert set(answers) == {1, 2}
    assert answers[1]["ocr_text"] == "" and answers[1]["ai_marks"] == 0 and answers[1]["status"] == "blank"
    assert answers[2]["ocr_text"].startswith("Supervised learning") and "Overfitting" in answers[2]["ocr_text"]
    assert q1_region["id"] in answers[2]["region_ids"]
    sub = teacher.submission(sheet["id"])
    assert sub["status"] == "in_review" and sub["total_ai_marks"] == 15 - 3

    hist = teacher.get(f"/audit-log?entity=submission&id={sheet['id']}").json()["items"][0]
    assert hist["action"] == "regions_edit" and hist["reason"] == "wrong marker"
    assert len(hist["old_value"]["regions"]) == 4
    region_hist = teacher.get(f"/audit-log?entity=region&id={q1_region['id']}").json()["items"][0]
    assert region_hist["old_value"]["question_id"] == exam["qid"][1]
    assert region_hist["new_value"]["question_id"] == exam["qid"][2]


def test_reassign_to_foreign_question_or_unassign(teacher, exam, sheet):
    other = teacher.create_exam(title="Other exam")
    other_q = teacher.put_rubric(other["id"])[0]["id"]
    region = _by_question(teacher.regions(sheet["id"]), exam["qid"][4])[0]
    err = assert_error(teacher.patch(f"/regions/{region['id']}", json={"action": "reassign", "question_id": other_q}),
                       400, "validation_error")
    assert error_fields(err) == {"question_id"}
    r = teacher.patch(f"/regions/{region['id']}", json={"action": "reassign", "question_id": None})
    assert r.status_code == 200 and r.json()["region"]["question_id"] is None
    q4 = next(a for a in r.json()["answers"] if a["question_no"] == 4)
    assert q4["status"] == "blank" and q4["ai_marks"] == 0


def test_crop_must_stay_inside_page(teacher, exam, sheet):
    page = teacher.get(f"/submissions/{sheet['id']}/pages").json()["pages"][0]
    region = _by_question(teacher.regions(sheet["id"]), exam["qid"][2])[0]
    for bbox in ([page["width"] - 10, 0, 50, 50], [0, page["height"] - 5, 10, 10], [-1, 0, 10, 10],
                 [0, 0, 0, 10]):
        err = assert_error(teacher.patch(f"/regions/{region['id']}", json={"action": "crop", "bbox": bbox}),
                           400, "validation_error")
        assert error_fields(err) == {"bbox"}
    assert_error(teacher.patch(f"/regions/{region['id']}", json={"action": "crop"}), 400, "validation_error")

    # A valid crop keeping only the first line of Q2 lowers its marks.
    x, y, w, _h = region["bbox"]
    r = teacher.patch(f"/regions/{region['id']}", json={"action": "crop", "bbox": [x, y, w, 40]})
    assert r.status_code == 200, r.text
    q2 = next(a for a in r.json()["answers"] if a["question_no"] == 2)
    assert 0 < len(q2["ocr_text"]) < len(samples.ANSWERS["q2_full"]) and q2["ai_marks"] < 5


def test_split_and_merge_same_page(teacher, exam, sheet):
    q3 = _by_question(teacher.regions(sheet["id"]), exam["qid"][3])[0]
    x, y, w, h = q3["bbox"]
    assert_error(teacher.patch(f"/regions/{q3['id']}", json={"action": "split", "split_y": y + h}),
                 400, "validation_error")
    r = teacher.patch(f"/regions/{q3['id']}", json={"action": "split", "split_y": y + h // 2})
    assert r.status_code == 200, r.text
    parts = _by_question(r.json()["regions"], exam["qid"][3])
    assert len(parts) == 2 and sum(p["bbox"][3] for p in parts) == h
    assert all(p["source"] == "manual" for p in parts)
    q3_answer = next(a for a in r.json()["answers"] if a["question_no"] == 3)
    assert q3_answer["ocr_text"] == samples.ANSWERS["q3_full"] and q3_answer["ai_marks"] == 5

    # Merge Q3's two parts back, then merge Q4 into Q3.
    upper, lower = sorted(parts, key=lambda p: p["bbox"][1])
    r = teacher.patch(f"/regions/{upper['id']}", json={"action": "merge", "merge_with": lower["id"]})
    assert r.status_code == 200, r.text
    assert [p["bbox"] for p in _by_question(r.json()["regions"], exam["qid"][3])] == [[x, y, w, h]]
    q4 = _by_question(r.json()["regions"], exam["qid"][4])[0]
    r = teacher.patch(f"/regions/{upper['id']}", json={"action": "merge", "merge_with": q4["id"]})
    assert r.status_code == 200, r.text
    answers = {a["question_no"]: a for a in r.json()["answers"]}
    assert "validation set" in answers[3]["ocr_text"] and answers[4]["status"] == "blank"
    assert len(r.json()["regions"]) == 3
    assert_error(teacher.patch(f"/regions/{upper['id']}", json={"action": "merge", "merge_with": upper["id"]}),
                 400, "validation_error")


def test_merge_across_pages_reassigns_other_part(teacher, exam, process):
    sub = process("ML008_continuation")["ML008_continuation"]
    p1, p2 = sorted(_by_question(teacher.regions(sub["id"]), exam["qid"][2]), key=lambda r: r["page_no"])
    full_text = teacher.answer_for(sub["id"], 2)["ocr_text"]
    r = teacher.patch(f"/regions/{p2['id']}", json={"action": "reassign", "question_id": None})
    short = next(a for a in r.json()["answers"] if a["question_no"] == 2)
    assert len(short["ocr_text"]) < len(full_text)
    r = teacher.patch(f"/regions/{p1['id']}", json={"action": "merge", "merge_with": p2["id"]})
    assert r.status_code == 200, r.text
    q2 = next(a for a in r.json()["answers"] if a["question_no"] == 2)
    assert q2["ocr_text"] == full_text and sorted(q2["region_ids"]) == sorted([p1["id"], p2["id"]])


@accept_lexical  # unblanking must restore the clean answer's full routing outcome (conftest)
def test_blank_region_scores_zero_and_unblank_restores(teacher, exam, sheet):
    region = _by_question(teacher.regions(sheet["id"]), exam["qid"][3])[0]
    r = teacher.patch(f"/regions/{region['id']}", json={"action": "blank"})
    assert r.status_code == 200 and r.json()["region"]["is_blank"] is True
    q3 = next(a for a in r.json()["answers"] if a["question_no"] == 3)
    assert q3["status"] == "blank" and q3["ai_marks"] == 0 and q3["ocr_text"] == ""
    r = teacher.patch(f"/regions/{region['id']}", json={"action": "unblank"})
    q3 = next(a for a in r.json()["answers"] if a["question_no"] == 3)
    assert q3["ai_marks"] == 5 and q3["status"] == "auto_accepted"


def test_undo_restores_previous_state(teacher, exam, sheet):
    assert_error(teacher.post(f"/submissions/{sheet['id']}/regions/undo"), 409, "conflict")
    before = teacher.regions(sheet["id"])
    marks_before = {a["question_no"]: a["ai_marks"] for a in teacher.answers(sheet["id"])}

    q1 = _by_question(before, exam["qid"][1])[0]
    q3 = _by_question(before, exam["qid"][3])[0]
    teacher.patch(f"/regions/{q1['id']}", json={"action": "reassign", "question_id": exam["qid"][4]})
    teacher.patch(f"/regions/{q3['id']}", json={"action": "split", "split_y": q3["bbox"][1] + 20})
    assert teacher.regions(sheet["id"]) != before

    assert teacher.post(f"/submissions/{sheet['id']}/regions/undo").status_code == 200  # undo the split
    r = teacher.post(f"/submissions/{sheet['id']}/regions/undo")  # undo the reassign
    assert r.status_code == 200, r.text
    after = teacher.regions(sheet["id"])
    strip = lambda rs: [(x["id"], x["page_no"], x["bbox"], x["question_id"], x["is_blank"]) for x in rs]  # noqa: E731
    assert strip(after) == strip(before)
    assert {a["question_no"]: a["ai_marks"] for a in teacher.answers(sheet["id"])} == marks_before
    # Both edits undone: nothing left.
    assert_error(teacher.post(f"/submissions/{sheet['id']}/regions/undo"), 409, "conflict")
    actions = [h["action"] for h in teacher.get(f"/audit-log?entity=submission&id={sheet['id']}").json()["items"]]
    assert actions[:4] == ["regions_undo", "regions_undo", "regions_edit", "regions_edit"]


def test_image_only_scan_manual_transcription(teacher, exam, process):
    sub = process("ML012_scanned_image_only")["ML012_scanned_image_only"]
    assert sub["ocr_engine"] == "none" and "no_markers_found" in sub["flags"]
    answers = teacher.answers(sub["id"])
    assert all(a["status"] == "needs_review" and a["ocr_text"] == "" for a in answers)

    # The teacher draws the Q1 region by hand: lines are found but have no text yet.
    page = teacher.get(f"/submissions/{sub['id']}/pages").json()["pages"][0]
    r = teacher.post(f"/submissions/{sub['id']}/regions",
                     json={"page_no": 1, "bbox": [0, 0, page["width"], page["height"] // 6],
                           "question_id": exam["qid"][1]})
    assert r.status_code == 201, r.text
    q1 = next(a for a in r.json()["answers"] if a["question_no"] == 1)
    assert q1["status"] == "needs_review" and "needs_transcription" in q1["flags"]
    assert r.json()["region"]["source"] == "manual"

    # ...then types what the student wrote: it is graded like any other answer.
    r = teacher.patch(f"/answers/{q1['answer_id']}/review", json={"corrected_text": samples.ANSWERS["q1_full"]})
    assert r.status_code == 200, r.text
    assert r.json()["ai_marks"] == 3 and r.json()["final_marks"] == 3 and r.json()["status"] == "reviewed"
    assert r.json()["text_corrected"] is True
