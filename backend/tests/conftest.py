"""Shared fixtures for the SARAS AI test suite.

Every test gets its own temporary SQLite database and storage folder, the
background worker disabled (jobs are drained synchronously with
worker.run_pending()), rate limiting off, the lexical grader and the
text-layer/auto OCR, so no ML models are needed.

Per-test settings overrides:  @pytest.mark.settings(max_upload_mb=1, ...)
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", message=r"Using `httpx` with `starlette.testclient`")

# Tests never use real AI providers or keys (backend/.env may hold real ones).
# Environment variables beat .env, and an empty key means "not set".
for _var in ("SARAS_GROQ_API_KEY", "SARAS_GEMINI_API_KEY", "SARAS_OPENROUTER_API_KEY", "SARAS_OLLAMA_BASE_URL"):
    os.environ[_var] = ""
os.environ.update(SARAS_AI_GRADING="off", SARAS_AI_CONSENT="false", SARAS_GEMINI_CONSENT="false",
                  SARAS_REVIEW_REQUIRED="true", SARAS_AI_OCR="false")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings, set_settings  # noqa: E402
from app.security import login_limiter  # noqa: E402
from scripts import make_samples as samples  # noqa: E402

API = "/api/v1"
PASSWORD = "correct-horse-battery-1"


# ------------------------------------------------------------------ helpers
class Api:
    """A logged-in caller: prefixes /api/v1 and sends the bearer token."""

    def __init__(self, client: TestClient, tokens: dict) -> None:
        self.client = client
        self.tokens = tokens
        self.user = tokens["user"]

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.tokens['access_token']}"}

    def _call(self, method: str, path: str, headers: dict | None = None, **kw):  # noqa: ANN003, ANN202
        # Paths are relative to /api/v1 unless already absolute API/system paths.
        url = path if path.startswith(("/api/", "/health", "/ready")) else API + path
        return self.client.request(method, url, headers={**self.headers, **(headers or {})}, **kw)

    def get(self, path: str, **kw):  # noqa: ANN003, ANN201
        return self._call("GET", path, **kw)

    def post(self, path: str, **kw):  # noqa: ANN003, ANN201
        return self._call("POST", path, **kw)

    def put(self, path: str, **kw):  # noqa: ANN003, ANN201
        return self._call("PUT", path, **kw)

    def patch(self, path: str, **kw):  # noqa: ANN003, ANN201
        return self._call("PATCH", path, **kw)

    def delete(self, path: str, **kw):  # noqa: ANN003, ANN201
        return self._call("DELETE", path, **kw)

    # -- domain shortcuts
    def create_exam(self, **over) -> dict:  # noqa: ANN003
        r = self.post("/exams", json={**samples.EXAM, **over})
        assert r.status_code == 201, r.text
        return r.json()

    def put_rubric(self, exam_id: str, rubric: dict | None = None) -> list[dict]:
        r = self.put(f"/exams/{exam_id}/questions", json=rubric or samples.RUBRIC)
        assert r.status_code == 200, r.text
        return r.json()

    def upload(self, exam_id: str, files: list[tuple[str, bytes]], refs: list[str] | None = None,
               key: str | None = None):  # noqa: ANN201
        data = {"student_refs": refs} if refs is not None else None
        headers = {"Idempotency-Key": key} if key else None
        return self.post(f"/exams/{exam_id}/submissions", headers=headers, data=data,
                         files=[("files", (name, body, "application/pdf")) for name, body in files])

    def answers(self, submission_id: str, sort: str = "question") -> list[dict]:
        r = self.get(f"/submissions/{submission_id}/answers?sort={sort}")
        assert r.status_code == 200, r.text
        return r.json()

    def answer_for(self, submission_id: str, question_no: int) -> dict:
        return next(a for a in self.answers(submission_id) if a["question_no"] == question_no)

    def submission(self, submission_id: str) -> dict:
        r = self.get(f"/submissions/{submission_id}")
        assert r.status_code == 200, r.text
        return r.json()

    def regions(self, submission_id: str) -> list[dict]:
        r = self.get(f"/submissions/{submission_id}/regions")
        assert r.status_code == 200, r.text
        return r.json()


def register(client: TestClient, email: str, name: str = "Teacher", password: str = PASSWORD) -> Api:
    r = client.post(f"{API}/auth/register", json={"name": name, "email": email, "password": password})
    assert r.status_code == 201, r.text
    return Api(client, r.json())


def make_admin(client: TestClient, email: str) -> Api:
    from app.db import SessionLocal
    from app.models import User

    api = register(client, email, name="Admin")
    with SessionLocal() as db:
        user = db.query(User).filter_by(email=email).one()
        user.role = "admin"
        db.commit()
    return api


def drain() -> int:
    from app import worker

    return worker.run_pending()


def assert_error(resp, status: int, code: str | None = None) -> dict:  # noqa: ANN001
    """PRD 16.1 error format: {"error": {"code", "message", "details"}}."""
    assert resp.status_code == status, f"expected {status}, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert set(body) == {"error"}, body
    assert {"code", "message", "details"} <= set(body["error"]), body
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    if code is not None:
        assert body["error"]["code"] == code, body
    return body["error"]


def error_fields(err: dict) -> set[str]:
    details = err.get("details") or []
    return {d.get("field", "") for d in details if isinstance(d, dict)}


# ------------------------------------------------------------------ routing in flow tests
# The suite grades with the lexical scorer only (no ML models). A lexical-only answer gets
# grading confidence 0.6 (lexical prior) x 0.85 (no second grader cross-checked it) = 0.51,
# so at the production auto-accept threshold (0.80) EVERY graded answer goes to review -
# tests/test_pipeline_ai.py asserts exactly that. Tests about approval, export, locking,
# regions or the worker, where routing is not the point, need sheets whose clean answers
# routing accepts; they opt in with this marker (a threshold just below 0.51). Answers with
# review flags (partial credit, contradictions, segmentation doubts, ...) still go to
# review. Production defaults are unchanged.
LEXICAL_AUTO_ACCEPT = 0.45
accept_lexical = pytest.mark.settings(auto_accept_threshold=LEXICAL_AUTO_ACCEPT)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="session", autouse=True)
def _cheap_argon2():
    """Real Argon2id hashes, but with test-sized cost parameters: the production
    64 MB / 3-pass setting dominates runtime on the 4 GB laptop."""
    from argon2 import PasswordHasher

    from app import security

    original = security._hasher
    security._hasher = PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1)
    yield
    security._hasher = original


@pytest.fixture(autouse=True)
def _fresh_scheme_cache():  # noqa: ANN202
    """AI marking schemes are memoised per process: no test sees another test's scheme."""
    from app.grading import direct_ai

    direct_ai.reset_scheme_cache()
    yield
    direct_ai.reset_scheme_cache()


@pytest.fixture
def settings(tmp_path: Path, request: pytest.FixtureRequest) -> Settings:
    overrides: dict = {}
    marker = request.node.get_closest_marker("settings")
    if marker is not None:
        overrides.update(marker.kwargs)
    base = dict(
        env="test",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        storage_dir=tmp_path / "storage",
        jwt_secret="test-only-secret-0123456789abcdef0123456789",
        worker_enabled=False,
        rate_limit_enabled=False,
        grader="lexical",
        ocr_engine="auto",
        frontend_dist=tmp_path / "no-frontend",
        log_level="WARNING",
        ai_grading="off", ai_consent=False, gemini_consent=False, review_required=True, ai_ocr=False,
        groq_api_key=None, gemini_api_key=None, openrouter_api_key=None, ollama_base_url=None,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def app(settings: Settings):  # noqa: ANN201
    from app.main import create_app

    login_limiter.reset()
    application = create_app(settings)
    yield application
    login_limiter.reset()
    set_settings(None)


@pytest.fixture
def client(app) -> Iterator[TestClient]:  # noqa: ANN001
    with TestClient(app) as c:
        yield c


@pytest.fixture
def teacher(client: TestClient) -> Api:
    return register(client, "teacher@example.com", name="Teacher One")


@pytest.fixture
def other_teacher(client: TestClient) -> Api:
    return register(client, "other@example.com", name="Teacher Two")


@pytest.fixture
def admin(client: TestClient) -> Api:
    return make_admin(client, "admin@example.com")


@pytest.fixture
def exam(teacher: Api) -> dict:
    """The Machine Learning Unit Test exam with its 4-question rubric."""
    e = teacher.create_exam()
    e["questions"] = teacher.put_rubric(e["id"])
    e["qid"] = {q["question_no"]: q["id"] for q in e["questions"]}
    return e


@pytest.fixture(scope="session")
def sample_bytes() -> dict[str, bytes]:
    """Sample sheet PDFs by student_ref (built once per session, in memory)."""
    out = {s.student_ref: samples.sheet_bytes(s) for s in samples.TYPED_SHEETS}
    out[samples.IMAGE_ONLY_SHEET.student_ref] = samples.build_image_only_pdf(samples.IMAGE_ONLY_SHEET.pages)
    return out


@pytest.fixture
def process(teacher: Api, exam: dict, sample_bytes: dict[str, bytes]):  # noqa: ANN201
    """Upload the named sample sheets, drain the queue and return
    {student_ref: SubmissionOut} after processing."""

    def _process(*refs: str) -> dict[str, dict]:
        r = teacher.upload(exam["id"], [(f"{ref}.pdf", sample_bytes[ref]) for ref in refs])
        assert r.status_code == 202, r.text
        assert r.json()["rejected"] == []
        drain()
        out = {}
        for sub in r.json()["submissions"]:
            out[sub["student_ref"]] = teacher.submission(sub["id"])
            assert out[sub["student_ref"]]["status"] == "processed", out[sub["student_ref"]]
        return out

    return _process
