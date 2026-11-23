"""Uploads (PRD 6.2, 16.1): PDF-only with signature check, size/page limits,
password-protected and corrupted files rejected, per-file isolation in a batch,
duplicate student refs, Idempotency-Key replay, 202 + status polling."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import assert_error, drain
from scripts import make_samples as samples


@pytest.mark.parametrize("filename,data,status,code", samples.invalid_sample_files(),
                         ids=[f[0] for f in samples.invalid_sample_files()])
def test_invalid_files_rejected(teacher, exam, settings, filename, data, status, code):
    err = assert_error(teacher.upload(exam["id"], [(filename, data)]), status, code)
    assert err["details"][0]["filename"] == filename
    assert err["details"][0]["code"] == code
    # Nothing is left behind: no submission row and no stored file.
    assert teacher.get(f"/exams/{exam['id']}/submissions").json()["total"] == 0
    stored = settings.storage_dir / "submissions"
    assert not stored.exists() or not any(stored.iterdir())


@pytest.mark.settings(max_upload_mb=1)
def test_file_over_size_limit_rejected(teacher, exam):
    err = assert_error(teacher.upload(exam["id"], [("big.pdf", samples.oversized_pdf_bytes(1.5))]),
                       413, "file_too_large")
    assert "1 MB" in err["message"]


def test_upload_returns_202_and_status_can_be_polled(teacher, exam, sample_bytes, settings):
    r = teacher.upload(exam["id"], [("../../evil name.pdf", sample_bytes["ML001_full_marks"])], refs=["S-001"])
    assert r.status_code == 202, r.text
    sub = r.json()["submissions"][0]
    assert sub["status"] == "uploaded" and sub["student_ref"] == "S-001" and sub["page_count"] == 1
    assert sub["original_filename"] == "evil name.pdf"  # path parts stripped
    # Stored under a random UUID folder, never under the client's file name.
    stored = list((settings.storage_dir / "submissions").rglob("*.pdf"))
    assert [p.name for p in stored] == ["original.pdf"] and stored[0].parent.name == sub["id"]

    status = teacher.get(f"/submissions/{sub['id']}/status").json()
    assert status == {"id": sub["id"], "status": "uploaded", "stage": "queued", "pages_done": 0,
                      "pages_total": 1, "errors": [], "attempts": 0, "progress": None}
    drain()
    status = teacher.get(f"/submissions/{sub['id']}/status").json()
    assert status["status"] == "processed" and status["pages_done"] == 1 and status["attempts"] == 1
    # Live progress, from real pipeline events: every stage finished, real counts.
    prog = status["progress"]
    assert prog["finished"] is True and prog["current"] is None
    assert [s["key"] for s in prog["stages"]] == ["preprocess", "recognition", "identification", "extraction",
                                                  "rubric", "analysis", "marking", "verification", "results"]
    assert all(s["state"] in ("done", "warning") for s in prog["stages"])
    c = prog["counts"]
    assert c["pages_total"] == 1 and c["pages_preprocessed"] == 1 and c["pages_text_layer"] == 1
    assert c["questions_total"] == 4 and c["questions_marked"] == 4 and c["answers_extracted"] == 4


def test_student_ref_defaults_to_file_name_stem(teacher, exam, sample_bytes):
    r = teacher.upload(exam["id"], [("Roll 17 (Asha).pdf", sample_bytes["ML001_full_marks"])], refs=[""])
    assert r.status_code == 202, r.text
    assert r.json()["submissions"][0]["student_ref"] == "Roll 17 _Asha"


def test_batch_of_ten_with_one_corrupted(teacher, exam, sample_bytes):
    good = [(f"{s.student_ref}.pdf", sample_bytes[s.student_ref]) for s in samples.TYPED_SHEETS[:9]]
    files = good[:4] + [("corrupted.pdf", samples.corrupted_pdf_bytes())] + good[4:]
    r = teacher.upload(exam["id"], files)
    assert r.status_code == 202, r.text
    body = r.json()
    assert len(body["submissions"]) == 9
    assert body["rejected"] == [{"filename": "corrupted.pdf", "student_ref": "corrupted",
                                 "code": "unreadable_pdf", "message": body["rejected"][0]["message"]}]
    assert drain() == 9
    listing = teacher.get(f"/exams/{exam['id']}/submissions?page_size=100").json()
    assert listing["total"] == 9
    assert {s["status"] for s in listing["items"]} == {"processed"}
    assert [s["student_ref"] for s in listing["items"]] == sorted(s["student_ref"] for s in listing["items"])


def test_duplicate_student_ref_conflict(teacher, exam, sample_bytes):
    pdf = sample_bytes["ML001_full_marks"]
    r = teacher.upload(exam["id"], [("a.pdf", pdf), ("b.pdf", pdf)], refs=["S1", "S1"])
    assert r.status_code == 202, r.text
    assert [s["student_ref"] for s in r.json()["submissions"]] == ["S1"]
    assert r.json()["rejected"][0]["code"] == "conflict"
    # A later upload for the same student is refused outright (every file rejected).
    err = assert_error(teacher.upload(exam["id"], [("c.pdf", pdf)], refs=["S1"]), 409, "conflict")
    assert err["details"][0]["student_ref"] == "S1"


def test_idempotency_key_replay_creates_no_duplicates(teacher, exam, sample_bytes):
    files = [("a.pdf", sample_bytes["ML001_full_marks"]), ("b.pdf", sample_bytes["ML002_partial"])]
    first = teacher.upload(exam["id"], files, key="batch-42")
    second = teacher.upload(exam["id"], files, key="batch-42")
    assert first.status_code == second.status_code == 202
    ids1 = sorted(s["id"] for s in first.json()["submissions"])
    ids2 = sorted(s["id"] for s in second.json()["submissions"])
    assert ids1 == ids2 and len(ids1) == 2
    assert teacher.get(f"/exams/{exam['id']}/submissions").json()["total"] == 2

    from app.db import SessionLocal
    from app.models import Job

    with SessionLocal() as db:
        assert db.query(Job).count() == 2  # no second set of jobs either


def test_too_many_files_and_mismatched_refs(teacher, exam, sample_bytes):
    pdf = sample_bytes["ML001_full_marks"]
    assert_error(teacher.upload(exam["id"], [("a.pdf", pdf)], refs=["x", "y"]), 400, "validation_error")


def test_other_teacher_and_admin_cannot_upload(teacher, exam, other_teacher, admin, sample_bytes):
    pdf = [("a.pdf", sample_bytes["ML001_full_marks"])]
    assert_error(other_teacher.upload(exam["id"], pdf), 403, "forbidden")
    assert_error(admin.upload(exam["id"], pdf), 403, "forbidden")


def test_delete_submission_removes_files(teacher, exam, sample_bytes, settings):
    r = teacher.upload(exam["id"], [("a.pdf", sample_bytes["ML001_full_marks"])])
    sub_id = r.json()["submissions"][0]["id"]
    drain()
    folder = Path(settings.storage_dir) / "submissions" / sub_id
    assert (folder / "original.pdf").is_file() and (folder / "page_001.png").is_file()
    assert teacher.delete(f"/submissions/{sub_id}").status_code == 204
    assert not folder.exists()
    assert_error(teacher.get(f"/submissions/{sub_id}"), 404, "not_found")
    hist = teacher.get(f"/audit-log?entity=submission").json()["items"]
    assert hist[0]["action"] == "delete" and hist[0]["entity_id"] == sub_id
