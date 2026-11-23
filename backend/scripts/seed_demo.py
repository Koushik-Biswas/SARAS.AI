"""Seed a demo teacher, the "Machine Learning Unit Test" exam with its rubric and
every valid sample answer sheet into the database configured by the normal
settings (SARAS_DATABASE_URL / SARAS_STORAGE_DIR / backend/.env).

    cd backend
    .venv/Scripts/python -m scripts.seed_demo          (Windows)
    .venv/bin/python -m scripts.seed_demo              (Linux/macOS)

Then start the API (uvicorn app.main:app) and sign in with the printed demo
credentials. DEVELOPMENT ONLY: the demo password is public, so the script
refuses to run with SARAS_ENV=prod.

Uploads go through the real API (app factory + TestClient, in-process), so the
same validation, storage and audit logging run as for a teacher using the UI;
the job queue is then drained synchronously with worker.run_pending().
Idempotent: if the demo exam already exists nothing is uploaded again (any
still-queued jobs are processed).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
warnings.filterwarnings("ignore", message=r"Using `httpx` with `starlette.testclient`")

DEMO_EMAIL = "demo@saras.local"
DEMO_PASSWORD = "demo-password-123"  # dev only; printed on purpose
DEMO_NAME = "Demo Teacher"
API = "/api/v1"


class SeedError(RuntimeError):
    pass


def _ok(resp, *statuses: int) -> dict:  # noqa: ANN001
    if resp.status_code not in statuses:
        raise SeedError(f"{resp.request.method} {resp.request.url.path} -> HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json() if resp.content else {}


def _sample_files(samples_dir: Path) -> tuple[dict, dict, list[Path]]:
    from scripts import make_samples

    sheets_dir = samples_dir / "sheets"
    if not (samples_dir / "rubric.json").is_file() or not any(sheets_dir.glob("*.pdf")):
        print(f"Generating sample data in {samples_dir} ...")
        make_samples.write_samples(samples_dir)
    rubric = json.loads((samples_dir / "rubric.json").read_text(encoding="utf-8"))
    exam_path = samples_dir / "exam.json"
    exam = json.loads(exam_path.read_text(encoding="utf-8")) if exam_path.is_file() else dict(make_samples.EXAM)
    return exam, rubric, sorted(sheets_dir.glob("*.pdf"))


def _token(client) -> str:  # noqa: ANN001
    r = client.post(f"{API}/auth/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
    if r.status_code == 200:
        return r.json()["access_token"]
    if r.status_code != 401:
        _ok(r, 200)
    r = client.post(f"{API}/auth/register", json={"name": DEMO_NAME, "email": DEMO_EMAIL, "password": DEMO_PASSWORD})
    if r.status_code == 409:
        raise SeedError(f"{DEMO_EMAIL} exists with a different password; delete that user or the database first")
    print(f"Created demo teacher {DEMO_EMAIL}")
    return _ok(r, 201)["access_token"]


def seed(samples_dir: Path | None = None) -> dict:
    from fastapi.testclient import TestClient

    from app import worker
    from app.config import Settings, set_settings
    from app.main import create_app

    from scripts.make_samples import SAMPLES_DIR

    base = Settings()  # environment / .env, exactly as the API would use
    if base.env == "prod":
        raise SeedError("seed_demo is for development only (SARAS_ENV=prod)")
    # No background thread (the queue is drained below) and no rate limit for the bulk upload.
    settings = base.model_copy(update={"worker_enabled": False, "rate_limit_enabled": False})
    exam_meta, rubric, sheets = _sample_files(samples_dir or SAMPLES_DIR)

    app = create_app(settings)
    summary = {"exam_id": None, "uploaded": 0, "rejected": [], "processed": 0, "skipped": False}
    try:
        with TestClient(app) as client:
            headers = {"Authorization": f"Bearer {_token(client)}"}
            exams = _ok(client.get(f"{API}/exams?page_size=100", headers=headers), 200)["items"]
            exam = next((e for e in exams if e["title"] == exam_meta["title"]), None)
            if exam is not None:
                summary.update(exam_id=exam["id"], skipped=True)
                print(f"Exam '{exam['title']}' already exists ({exam['id']}); not uploading again.")
            else:
                exam = _ok(client.post(f"{API}/exams", json=exam_meta, headers=headers), 201)
                _ok(client.put(f"{API}/exams/{exam['id']}/questions", json=rubric, headers=headers), 200)
                summary["exam_id"] = exam["id"]
                print(f"Created exam '{exam['title']}' with {len(rubric['questions'])} questions")
                handles = [open(p, "rb") for p in sheets]  # noqa: SIM115 - closed below
                try:
                    files = [("files", (p.name, fh, "application/pdf")) for p, fh in zip(sheets, handles)]
                    r = client.post(f"{API}/exams/{exam['id']}/submissions", files=files,
                                    headers={**headers, "Idempotency-Key": "seed-demo-v1"})
                finally:
                    for fh in handles:
                        fh.close()
                body = _ok(r, 202)
                summary["uploaded"] = len(body["submissions"])
                summary["rejected"] = body["rejected"]
                print(f"Uploaded {summary['uploaded']} sheets ({len(body['rejected'])} rejected)")

            summary["processed"] = worker.run_pending()
            subs = _ok(client.get(f"{API}/exams/{summary['exam_id']}/submissions?page_size=100", headers=headers),
                       200)["items"]
            for s in subs:
                print(f"  {s['student_ref']:<28} {s['status']:<10} AI total {s['total_ai_marks']!s:>5}"
                      f"  needs review: {s['answers_needing_review']}")
    finally:
        set_settings(None)
    return summary


def main() -> int:
    try:
        summary = seed()
    except SeedError as exc:
        print(f"Seeding failed: {exc}", file=sys.stderr)
        return 1
    print(f"\nProcessed {summary['processed']} queued job(s).")
    print("Demo sign-in (development only):")
    print(f"  email:    {DEMO_EMAIL}")
    print(f"  password: {DEMO_PASSWORD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
