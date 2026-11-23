"""Runtime configuration. Every value can be overridden with a SARAS_* environment
variable or a backend/.env file (never committed)."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEV_SECRET = "dev-only-insecure-secret-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_prefix="SARAS_", extra="ignore")

    env: str = "dev"  # dev | test | prod
    database_url: str = f"sqlite:///{(BACKEND_DIR / 'data' / 'saras.db').as_posix()}"
    storage_dir: Path = BACKEND_DIR / "data" / "storage"

    # Auth
    jwt_secret: str = DEV_SECRET
    access_token_minutes: int = 15
    refresh_token_days: int = 7
    login_max_failures: int = 5
    login_lock_minutes: int = 5

    # Uploads (PRD 6.2)
    max_upload_mb: int = 20
    max_pages: int = 50

    # Pipeline — low-resource profile defaults (PRD 18.6)
    render_dpi: int = 150
    ocr_engine: str = "auto"  # auto | textlayer | trocr
    trocr_model: str = "microsoft/trocr-small-handwritten"
    grader: str = "auto"  # semantic local grader: auto | lexical | sbert | nli | nli-onnx
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    # Grading ensemble (docs/GRADING_DESIGN.md): which LOCAL grader decides when the AI
    # marker does not. "auto" = the semantic grader chosen by `grader` when one is
    # available (the other local grader then cross-checks it), else lexical. Chosen from
    # the DEV benchmark measurements recorded in GRADING_DESIGN.md.
    local_primary: Literal["auto", "lexical", "nli-onnx"] = "auto"

    # Grading thresholds (PRD 12.11) — tune on validation data, freeze before test
    tau: float = 0.6
    tau_low: float = 0.4
    contradiction_threshold: float = 0.5
    auto_accept_threshold: float = 0.80
    low_ocr_confidence: float = 0.5
    calibration_temperature: float = 1.0
    # True: uncertain answers are held as needs_review and block approval until a teacher
    # checks them. False: the AI's marks go straight to the results (review stays
    # available but optional); only answers nobody could mark (unread page, every AI
    # provider down on a question-only rubric) still wait for a teacher.
    review_required: bool = True

    # Worker
    worker_enabled: bool = True
    worker_poll_seconds: float = 1.0
    job_max_attempts: int = 3
    job_stale_seconds: int = 300

    # API
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 100
    upload_rate_per_minute: int = 10
    log_level: str = "INFO"
    # Built SPA (npm run build); served at "/" when present.
    frontend_dist: Path = BACKEND_DIR.parent / "frontend" / "dist"

    # AI (LLM) grading — PRD 13.5: optional and never in the critical path; the offline
    # checker always runs. Nothing is sent to any provider unless ai_consent is true, and
    # Gemini (free tier may train on data) additionally needs gemini_consent.
    # API keys live only in backend/.env; they are never logged, returned or stored.
    ai_grading: Literal["off", "ensemble"] = "off"
    ai_consent: bool = False
    # Marking style given to the AI marker (spec 2: Strict / Moderate / Lenient).
    ai_grading_style: Literal["lenient", "moderate", "strict"] = "moderate"
    gemini_consent: bool = False
    # OpenRouter forwards prompts to upstream providers whose free endpoints may log or
    # train on them, so it needs its own consent as well (its requests also ask for
    # data_collection=deny). Ollama is used only when its URL is local / private-network.
    openrouter_consent: bool = False
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    ollama_base_url: str | None = None
    # "provider:model", tried in order (providers: groq | gemini | openrouter | ollama).
    ai_grading_providers: list[str] = [
        "groq:openai/gpt-oss-120b",
        "groq:openai/gpt-oss-20b",
        "gemini:gemini-3.5-flash-lite",
        "gemini:gemini-3.1-flash-lite",
        "gemini:gemini-flash-lite-latest",
    ]
    # Reading pages that have no PDF text layer (scans, photos, handwriting) with a
    # vision model (Gemini, then Groq's vision model). Needs ai_consent; Gemini also needs
    # gemini_consent (the page image, which may show the student's name, is sent to the
    # provider). Off, or no reader answers: TrOCR if installed, else the answer waits for
    # manual transcription.
    ai_ocr: bool = True
    ai_ocr_providers: list[str] = [
        "gemini:gemini-flash-lite-latest",
        "groq:qwen/qwen3.8-27b",  # vision model on Groq: the backup reader
        "gemini:gemini-3.5-flash-lite",
        "gemini:gemini-3.1-flash-lite",
        "gemini:gemini-flash-latest",
    ]
    # Pages of one sheet read at the same time by the vision models (each provider's
    # per-minute limits still apply). 3 keeps a 4-CPU laptop and the free tiers happy.
    ai_ocr_concurrency: int = 3
    # Reasoning effort of the AI marker (reasoning models such as gpt-oss): low is fast;
    # the selective second marking below always uses at least "medium".
    ai_reasoning_effort: Literal["low", "medium", "high"] = "low"
    # Selective verification: answers the first AI marking was unsure about (its own
    # doubt, quotes that are not in the answer, contradictions, or a large gap to the
    # offline checker) are marked a second time, more carefully, in ONE extra request per
    # sheet. Confident answers are never re-sent.
    # EXPERIMENTAL (off): measured on the dev benchmark (2026-11-24, gpt-oss-120b) it re-marked
    # 18 of 42 answers and made marks WORSE (MAE 0.27 -> 0.34; keyword-stuffed answers gained
    # marks), for +77% tokens. Kept opt-in until a better verifier is measured.
    ai_verify: bool = False
    # Question-only rubrics, EXPERIMENTAL (off): the AI first writes a marking scheme per
    # question (cached) and marks every sheet criterion by criterion against it. Measured
    # on the dev benchmark (2026-11-24, same 26 answers, gpt-oss-120b) it was LESS accurate
    # than the default one-request marking (MAE 0.47 vs 0.20), so it stays opt-in.
    ai_direct_scheme: bool = False
    ai_timeout_seconds: float = 30
    ai_max_retries: int = 2
    # Wall-clock limit for one AI request (all providers, retries and backoff together);
    # past it the remaining providers are skipped and the offline checker decides.
    ai_call_deadline_seconds: float = 60
    # Client-side budgets, kept below the free-tier limits (per model).
    groq_tokens_per_minute: int = 6000
    groq_tokens_per_day: int = 170000
    gemini_requests_per_minute: int = 10
    gemini_requests_per_day: int = 400
    ai_max_input_tokens_per_call: int = 3500

    @field_validator("groq_api_key", "gemini_api_key", "openrouter_api_key", "ollama_base_url", mode="before")
    @classmethod
    def _blank_is_none(cls, v):  # noqa: ANN001, ANN206
        """An empty value (e.g. SARAS_GROQ_API_KEY= in .env or the test session) means "not set"."""
        raw = v.get_secret_value() if isinstance(v, SecretStr) else v
        return None if isinstance(raw, str) and not raw.strip() else v

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def _prepare(s: Settings) -> Settings:
    if s.jwt_secret == DEV_SECRET:
        if s.env == "prod":
            raise RuntimeError("SARAS_JWT_SECRET must be set in production")
        if s.env != "test":
            warnings.warn("Using the development JWT secret; set SARAS_JWT_SECRET outside dev.", stacklevel=3)
    s.storage_dir.mkdir(parents=True, exist_ok=True)
    if s.database_url.startswith("sqlite:///"):
        Path(s.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return s


_override: Settings | None = None


@lru_cache
def _load_settings() -> Settings:
    return _prepare(Settings())


def get_settings() -> Settings:
    """The active settings: an explicit override (create_app(settings) / tests)
    wins over the cached environment-derived settings."""
    return _override if _override is not None else _load_settings()


def set_settings(settings: Settings | None) -> None:
    """Install (or, with None, remove) the settings every module reads via
    get_settings(). Used by create_app(settings) so tests can inject overrides."""
    global _override
    _override = _prepare(settings) if settings is not None else None
    _load_settings.cache_clear()


def _cache_clear() -> None:  # backwards-compatible: get_settings.cache_clear()
    _load_settings.cache_clear()


get_settings.cache_clear = _cache_clear  # type: ignore[attr-defined]
