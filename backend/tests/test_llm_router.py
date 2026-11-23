"""LLMRouter: consent gates, cache, retries/backoff, fallback, budgets and secret
hygiene. Every provider is faked with httpx.MockTransport (app.llm.fakes): no test
ever reaches a real provider or reads a real key."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.db import Base
from app.llm import ImageInput, LLMRouter, LLMUnavailable, RateLimiter, parse_provider_spec
from app.llm.fakes import FakeLLMServer, FakeReply
from app.llm.providers import GeminiProvider, parse_retry_after
from app.llm.router import is_local_url
from app.models import AiRun

FAKE_GROQ = "fake-groq-secret-0001"
FAKE_GEMINI = "fake-gemini-secret-0002"
FAKE_OPENROUTER = "fake-openrouter-secret-0003"
GROQ = "groq:openai/gpt-oss-120b"
GEMINI = "gemini:gemini-3.5-flash-lite"


class Echo(BaseModel):
    answer: str
    score: int


OK = {"answer": "fine", "score": 1}


def make_settings(tmp_path: Path, **over) -> Settings:  # noqa: ANN003
    base = dict(
        _env_file=None, env="test", database_url=f"sqlite:///{(tmp_path / 'unused.db').as_posix()}",
        storage_dir=tmp_path / "storage", ai_grading="ensemble", ai_consent=True, gemini_consent=True,
        groq_api_key=FAKE_GROQ, gemini_api_key=FAKE_GEMINI, openrouter_api_key=None, ollama_base_url=None,
        ai_grading_providers=[GROQ, GEMINI], ai_max_retries=2,
    )
    base.update(over)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _fresh_gemini_schema_memory():
    GeminiProvider._schema_field.clear()
    yield
    GeminiProvider._schema_field.clear()


@pytest.fixture
def session_factory():  # noqa: ANN201
    """In-memory SQLite holding just ai_runs (fast on the 4 GB laptop)."""
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[AiRun.__table__])
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def harness(tmp_path: Path, session_factory):  # noqa: ANN001, ANN201
    """make(**settings_overrides) -> (router, server, sleeps)."""
    routers: list[LLMRouter] = []

    def make(server: FakeLLMServer | None = None, limiter: RateLimiter | None = None, **over):  # noqa: ANN003, ANN202
        server = server or FakeLLMServer()
        sleeps: list[float] = []
        router = LLMRouter(make_settings(tmp_path, **over), session_factory, transport=server.transport,
                           limiter=limiter or RateLimiter(), sleep=sleeps.append)
        routers.append(router)
        return router, server, sleeps

    yield make
    for r in routers:
        r.close()


def runs(session_factory) -> list[AiRun]:  # noqa: ANN001
    with session_factory() as db:
        return list(db.scalars(select(AiRun).order_by(AiRun.created_at)))


def statuses(session_factory) -> list[str]:  # noqa: ANN001
    """Sorted: rows written within one clock tick (about 15 ms on Windows) can tie on created_at."""
    return sorted(r.status for r in runs(session_factory))


# ------------------------------------------------------------------ happy path and request shape
def test_groq_call_shape_and_ok_run(harness, session_factory) -> None:  # noqa: ANN001
    router, server, _ = harness()
    server.script("groq", OK)
    res = router.generate_json("demo", "Say fine.", Echo, system="Be brief.", prompt_version="v9")

    assert isinstance(res.data, Echo) and res.data.score == 1
    assert (res.provider, res.model, res.cached) == ("groq", "openai/gpt-oss-120b", False)
    assert res.usage == {"input_tokens": 120, "output_tokens": 60}
    [call] = server.calls
    assert call.url == "https://api.groq.com/openai/v1/chat/completions"
    assert call.headers["authorization"] == f"Bearer {FAKE_GROQ}"
    body = call.body
    assert body["temperature"] == 0 and body["seed"] == 42
    assert body["reasoning_effort"] == "low" and body["include_reasoning"] is False
    assert "reasoning_format" not in body
    assert body["max_tokens"] >= 1200
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False and schema["required"] == ["answer", "score"]
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}

    [row] = runs(session_factory)
    assert (row.status, row.provider, row.model, row.prompt_version, row.task) == \
        ("ok", "groq", "openai/gpt-oss-120b", "v9", "demo")
    assert row.request_json["prompt"] == "Say fine." and row.request_json["schema"] == schema
    assert row.response_json == OK and row.input_tokens == 120 and row.output_tokens == 60
    assert len(row.cache_key) == 64 and row.latency_ms is not None


def test_gemini_call_shape_with_images(harness) -> None:  # noqa: ANN001
    router, server, _ = harness(ai_grading_providers=[GEMINI])
    server.script("gemini", OK)
    res = router.generate_json("vision", "Read it.", Echo, system="sys",
                               images=[ImageInput(b"\x89PNG fake", "image/png")])
    assert res.provider == "gemini" and res.model == "gemini-3.5-flash-lite"
    [call] = server.calls
    assert call.url == ("https://generativelanguage.googleapis.com/v1beta/models/"
                        "gemini-3.5-flash-lite:generateContent")
    assert FAKE_GEMINI not in call.url  # the key travels only in the header
    assert call.headers["x-goog-api-key"] == FAKE_GEMINI
    cfg = call.body["generationConfig"]
    assert cfg["responseMimeType"] == "application/json" and cfg["temperature"] == 0
    assert cfg["responseJsonSchema"]["required"] == ["answer", "score"]
    assert call.body["systemInstruction"] == {"parts": [{"text": "sys"}]}
    parts = call.body["contents"][0]["parts"]
    assert parts[0] == {"text": "Read it."}
    assert parts[1]["inlineData"]["mimeType"] == "image/png" and parts[1]["inlineData"]["data"]


def test_gemini_falls_back_to_response_schema(harness) -> None:  # noqa: ANN001
    router, server, _ = harness(ai_grading_providers=[GEMINI])
    server.script("gemini", FakeReply(status=400, body={"error": {
        "code": 400, "status": "INVALID_ARGUMENT",
        "message": 'Invalid JSON payload received. Unknown name "responseJsonSchema" at \'generation_config\'.'}}), OK)
    assert router.generate_json("demo", "p", Echo).data == Echo(**OK)
    first, second = server.calls
    assert "responseJsonSchema" in first.body["generationConfig"]
    legacy = second.body["generationConfig"]["responseSchema"]
    assert legacy["type"] == "OBJECT" and legacy["propertyOrdering"] == ["answer", "score"]
    assert "additionalProperties" not in legacy


# ------------------------------------------------------------------ consent
def test_no_ai_consent_sends_nothing(harness, session_factory) -> None:  # noqa: ANN001
    router, server, _ = harness(ai_consent=False)
    server.script("groq", OK)
    with pytest.raises(LLMUnavailable) as err:
        router.generate_json("demo", "p", Echo)
    assert server.calls == [] and runs(session_factory) == []
    assert "consent" in str(err.value).lower()
    assert router.usable_providers() == []


def test_gemini_skipped_without_gemini_consent(harness) -> None:  # noqa: ANN001
    router, server, _ = harness(gemini_consent=False, ai_grading_providers=[GEMINI, GROQ])
    server.script("groq", OK).script("gemini", OK)
    assert router.generate_json("demo", "p", Echo).provider == "groq"
    assert [c.provider for c in server.calls] == ["groq"]
    assert router.usable_providers() == [GROQ]

    only_gemini, server2, _ = harness(gemini_consent=False, ai_grading_providers=[GEMINI])
    server2.script("gemini", OK)
    with pytest.raises(LLMUnavailable) as err:
        only_gemini.generate_json("demo", "another prompt", Echo)  # "p" would be a cache hit
    assert server2.calls == []
    assert "Gemini needs its own consent" in err.value.reasons[0]


def test_missing_keys_and_unknown_providers_are_skipped(harness) -> None:  # noqa: ANN001
    router, server, _ = harness(groq_api_key=None, ai_grading_providers=["xai:grok", GROQ, "openrouter:m"])
    with pytest.raises(LLMUnavailable) as err:
        router.generate_json("demo", "p", Echo)
    assert server.calls == []
    assert len(err.value.reasons) == 3
    assert "unknown provider" in err.value.reasons[0] and "no API key" in err.value.reasons[1]
    assert parse_provider_spec(" Groq : openai/gpt-oss-20b ") == ("groq", "openai/gpt-oss-20b")


# ------------------------------------------------------------------ cache
def test_cache_hit_makes_no_network_call(harness, session_factory) -> None:  # noqa: ANN001
    router, server, _ = harness()
    server.script("groq", OK)
    first = router.generate_json("demo", "same prompt", Echo, prompt_version="v1")
    assert not first.cached and len(server.calls) == 1

    again = router.generate_json("demo", "same prompt", Echo, prompt_version="v1")
    assert again.cached and again.data == first.data and (again.provider, again.model) == ("groq", first.model)
    assert len(server.calls) == 1

    # A new process (new router, new limiter) still replays from ai_runs, with no network.
    fresh, server2, _ = harness()
    assert fresh.generate_json("demo", "same prompt", Echo, prompt_version="v1").cached
    assert server2.calls == []
    assert statuses(session_factory) == ["cache_hit", "cache_hit", "ok"]

    # A different prompt version is a different cache key.
    server.script("groq", OK)
    assert not router.generate_json("demo", "same prompt", Echo, prompt_version="v2").cached
    assert len(server.calls) == 2


# ------------------------------------------------------------------ retries and fallback
def test_malformed_json_retries_then_next_provider_then_unavailable(harness, session_factory) -> None:  # noqa: ANN001
    router, server, sleeps = harness(ai_max_retries=1)
    server.script("groq", "this is not json", '{"answer": "fine", ')
    server.script("gemini", '{"unexpected": true}', json.dumps({"answer": "x", "score": "not a number"}))
    with pytest.raises(LLMUnavailable) as err:
        router.generate_json("demo", "p", Echo)
    assert [c.provider for c in server.calls] == ["groq", "groq", "gemini", "gemini"]
    assert sleeps == []  # invalid output is retried at once
    assert statuses(session_factory) == ["invalid"] * 4
    assert len(err.value.reasons) == 2 and all("invalid" in r for r in err.value.reasons)


def test_malformed_json_then_valid_on_retry(harness) -> None:  # noqa: ANN001
    router, server, _ = harness(ai_max_retries=1)
    server.script("groq", "```json\nnot json\n```", "```json\n" + json.dumps(OK) + "\n```")
    res = router.generate_json("demo", "p", Echo)
    assert res.provider == "groq" and res.data == Echo(**OK) and len(server.calls) == 2


def test_429_retry_after_backoff_then_fallback(harness, session_factory) -> None:  # noqa: ANN001
    router, server, sleeps = harness(ai_max_retries=2)
    limited = FakeReply(status=429, headers={"Retry-After": "3"},
                        body={"error": {"message": "Rate limit reached for tokens per minute"}})
    server.script("groq", limited, limited, limited).script("gemini", OK)
    res = router.generate_json("demo", "p", Echo)
    assert res.provider == "gemini"
    assert [c.provider for c in server.calls] == ["groq"] * 3 + ["gemini"]
    assert sleeps == [3.0, 3.0]
    assert statuses(session_factory) == ["ok"] + ["rate_limited"] * 3


def test_gemini_retry_info_is_honoured(harness) -> None:  # noqa: ANN001
    router, server, sleeps = harness(ai_grading_providers=[GEMINI], ai_max_retries=1)
    server.script("gemini", FakeReply(status=429, body={"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded",
        "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "4s"}]}}), OK)
    assert router.generate_json("demo", "p", Echo).provider == "gemini"
    assert sleeps == [4.0]


def test_long_retry_after_parks_the_provider(harness) -> None:  # noqa: ANN001
    router, server, sleeps = harness()
    server.script("groq", FakeReply(status=429, headers={"Retry-After": "3600"}))
    server.script("gemini", OK, OK)
    assert router.generate_json("demo", "p1", Echo).provider == "gemini"
    assert sleeps == []  # too long to wait: straight to the next provider
    assert router.generate_json("demo", "p2", Echo).provider == "gemini"
    assert [c.provider for c in server.calls] == ["groq", "gemini", "gemini"]  # groq not called again
    assert router.usable_providers() == [GEMINI]


def test_5xx_and_timeouts_back_off_exponentially(harness) -> None:  # noqa: ANN001
    router, server, sleeps = harness(ai_max_retries=2)
    server.script("groq", FakeReply(status=500, body={"error": {"message": "internal error"}}),
                  FakeReply(exc=httpx.ReadTimeout("timed out")), OK)
    assert router.generate_json("demo", "p", Echo).provider == "groq"
    assert sleeps == [1.0, 2.0] and len(server.calls) == 3


def test_overloaded_provider_is_parked_at_once_and_skipped_by_later_calls(harness) -> None:  # noqa: ANN001
    """A 503 "high demand" is not retried with backoff: the next provider answers this call
    and later calls (the next pages of the sheet) skip the overloaded model until its
    cooldown ends. Before this, every page retried every overloaded Gemini model 3 times."""
    router, server, sleeps = harness(ai_max_retries=2)
    busy = FakeReply(status=503, body={"error": {"message": "This model is currently experiencing high demand."}})
    server.script("groq", busy)
    server.script("gemini", OK, OK)
    assert router.generate_json("demo", "p1", Echo).provider == "gemini"
    assert router.generate_json("demo", "p2", Echo).provider == "gemini"
    assert [c.provider for c in server.calls] == ["groq", "gemini", "gemini"] and sleeps == []
    assert router.limiter.cooldown_left(f"groq:{router.settings.ai_grading_providers[0].split(':', 1)[1]}") > 0


def test_auth_error_is_not_retried(harness, session_factory) -> None:  # noqa: ANN001
    router, server, sleeps = harness()
    server.script("groq", FakeReply(status=401, body={"error": {"message": "Invalid API Key"}}))
    server.script("gemini", OK, OK)
    assert router.generate_json("demo", "p1", Echo).provider == "gemini"
    assert router.generate_json("demo", "p2", Echo).provider == "gemini"
    assert [c.provider for c in server.calls] == ["groq", "gemini", "gemini"] and sleeps == []
    assert [r.status for r in runs(session_factory) if r.provider == "groq"] == ["error"]


def test_no_connection_falls_back_at_once_and_parks_the_provider(harness) -> None:  # noqa: ANN001
    router, server, sleeps = harness()
    server.script("groq", FakeReply(exc=httpx.ConnectError("getaddrinfo failed")))
    server.script("gemini", FakeReply(exc=httpx.ConnectError("getaddrinfo failed")))
    with pytest.raises(LLMUnavailable) as err:  # offline laptop: the caller uses the offline checker
        router.generate_json("demo", "p1", Echo)
    assert len(server.calls) == 2 and sleeps == []
    assert all("connection failed" in r for r in err.value.reasons)
    with pytest.raises(LLMUnavailable):  # the next sheet does not even try
        router.generate_json("demo", "p2", Echo)
    assert len(server.calls) == 2 and router.usable_providers() == []


def test_groq_schema_validation_failure_is_retried(harness) -> None:  # noqa: ANN001
    router, server, _ = harness(ai_max_retries=1)
    server.script("groq", FakeReply(status=400, body={"error": {
        "message": "Generated JSON does not match the expected schema.", "type": "invalid_request_error",
        "code": "json_validate_failed", "failed_generation": "{\"answer\": 3}"}}), OK)
    assert router.generate_json("demo", "p", Echo).provider == "groq" and len(server.calls) == 2


def test_truncated_output_is_invalid(harness) -> None:  # noqa: ANN001
    router, server, _ = harness(ai_max_retries=0)
    server.script("groq", FakeReply(raw='{"answer": "fi', finish_reason="length")).script("gemini", OK)
    assert router.generate_json("demo", "p", Echo).provider == "gemini"


def test_retry_after_parsing() -> None:
    assert parse_retry_after("7") == 7.0 and parse_retry_after("1.5") == 1.5
    assert parse_retry_after(None) is None and parse_retry_after("soon") is None
    future = (datetime.now(timezone.utc) + timedelta(seconds=90)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert 80 <= parse_retry_after(future) <= 91


# ------------------------------------------------------------------ budgets and rate limits
def test_daily_budget_counts_last_24h_runs(harness, session_factory) -> None:  # noqa: ANN001
    now = datetime.now(timezone.utc)
    with session_factory() as db:
        db.add(AiRun(task="t", provider="groq", model="openai/gpt-oss-120b", prompt_version="1", cache_key="k" * 64,
                     status="ok", input_tokens=169_000, output_tokens=500, created_at=now - timedelta(hours=1)))
        db.add(AiRun(task="t", provider="gemini", model="gemini-3.5-flash-lite", prompt_version="1",
                     cache_key="k" * 64, status="ok", input_tokens=10**7, created_at=now - timedelta(hours=30)))
        db.commit()
    router, server, _ = harness()
    server.script("groq", OK).script("gemini", OK)
    res = router.generate_json("demo", "p", Echo)
    assert res.provider == "gemini"  # groq's rolling-24h token budget is used; the old gemini row is not counted
    assert [c.provider for c in server.calls] == ["gemini"]
    assert router.usage_last_24h("groq", "openai/gpt-oss-120b") == (1, 169_500)


def test_daily_request_budget(harness, session_factory) -> None:  # noqa: ANN001
    with session_factory() as db:
        for _ in range(3):
            db.add(AiRun(task="t", provider="gemini", model="gemini-3.5-flash-lite", prompt_version="1",
                         cache_key="c" * 64, status="error"))
        db.add(AiRun(task="t", provider="gemini", model="gemini-3.5-flash-lite", prompt_version="1",
                     cache_key="c" * 64, status="cache_hit"))  # cache hits cost nothing
        db.commit()
    router, server, _ = harness(ai_grading_providers=[GEMINI], gemini_requests_per_day=4)
    server.script("gemini", OK, OK)
    assert router.generate_json("demo", "p1", Echo).provider == "gemini"  # 4th request: allowed
    with pytest.raises(LLMUnavailable) as err:
        router.generate_json("demo", "p2", Echo)
    assert "daily request budget" in err.value.reasons[0] and len(server.calls) == 1


def test_token_bucket_skips_provider_until_refilled(harness) -> None:  # noqa: ANN001
    clock = [1000.0]
    limiter = RateLimiter(clock=lambda: clock[0])
    router, server, sleeps = harness(limiter=limiter, groq_tokens_per_minute=3000)
    server.script("groq", FakeReply(output=OK, input_tokens=2900, output_tokens=60), OK)
    server.script("gemini", OK)
    assert router.generate_json("demo", "p1", Echo).provider == "groq"
    assert router.generate_json("demo", "p2", Echo).provider == "gemini"  # bucket nearly empty: skipped
    assert sleeps == []
    clock[0] += 60  # a minute later the bucket is full again
    assert router.generate_json("demo", "p3", Echo).provider == "groq"
    assert [c.provider for c in server.calls] == ["groq", "gemini", "groq"]


def test_short_limiter_wait_sleeps_then_calls(harness) -> None:  # noqa: ANN001
    clock = [0.0]
    limiter = RateLimiter(clock=lambda: clock[0])
    server = FakeLLMServer()
    router, _, sleeps = harness(server=server, limiter=limiter, ai_grading_providers=[GEMINI],
                                gemini_requests_per_minute=30)  # one request every 2 s
    router._sleep = lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s))  # type: ignore[method-assign]
    for i in range(31):
        server.script("gemini", OK)
        router.generate_json("demo", f"p{i}", Echo)
    assert len(server.calls) == 31 and len(sleeps) == 1 and 1.9 <= sleeps[0] <= 2.1


# ------------------------------------------------------------------ secrets
def test_secrets_never_reach_rows_logs_or_errors(harness, session_factory, caplog) -> None:  # noqa: ANN001
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="app")
    caplog.set_level(logging.DEBUG, logger="httpx")

    def echo_keys(call) -> FakeReply:  # noqa: ANN001
        # A hostile/buggy provider that echoes the credentials back in its error body.
        leaked = f"{call.headers.get('authorization', '')} {call.headers.get('x-goog-api-key', '')}"
        status = 401 if call.provider == "groq" else 500
        return FakeReply(status=status, body={"error": {"message": f"bad key {leaked}", "code": "x",
                                                         "failed_generation": leaked}})

    router, server, _ = harness(
        server=FakeLLMServer(responder=echo_keys), ai_max_retries=1, openrouter_api_key=FAKE_OPENROUTER,
        openrouter_consent=True, ai_grading_providers=[GROQ, GEMINI, "openrouter:some/model"])
    with pytest.raises(LLMUnavailable) as err:
        router.generate_json("demo", "p", Echo)
    assert server.calls, "the fake providers were called"
    assert server.calls[0].headers["authorization"] == f"Bearer {FAKE_GROQ}"  # the key was really used

    server.responder = None
    server.script("gemini", OK)
    router.generate_json("demo", "p-ok", Echo)

    dumped = json.dumps([{c.name: getattr(r, c.name) for c in AiRun.__table__.columns}
                         for r in runs(session_factory)], default=str)
    texts = [dumped, caplog.text, str(err.value), repr(err.value), " ".join(err.value.reasons)]
    for secret in (FAKE_GROQ, FAKE_GEMINI, FAKE_OPENROUTER):
        for text in texts:
            assert secret not in text
    assert "***" in str(err.value)  # the echoed key was scrubbed, not just absent
    assert "fake-" not in repr(make_settings(Path("."), groq_api_key=FAKE_GROQ))  # SecretStr hides it


# ------------------------------------------------------------------ review findings (ai-layer)
OPENROUTER = "openrouter:meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


class Pair(BaseModel):
    items: list[int]


def need_two(data: Pair) -> str | None:
    return None if len(data.items) == 2 else f"incomplete: {len(data.items)} of 2 items"


def test_openrouter_needs_its_own_consent_and_denies_data_collection(harness) -> None:  # noqa: ANN001
    """#3: OpenRouter's free upstreams may log / train on prompts: ai_consent alone is not enough."""
    router, server, _ = harness(openrouter_api_key=FAKE_OPENROUTER, ai_grading_providers=[OPENROUTER])
    server.script("openrouter", OK)
    assert router.usable_providers() == []
    with pytest.raises(LLMUnavailable) as err:
        router.generate_json("demo", "p-or", Echo)
    assert server.calls == [] and "openrouter_consent_missing" in err.value.reasons[0]

    consented, server2, _ = harness(openrouter_api_key=FAKE_OPENROUTER, openrouter_consent=True,
                                    ai_grading_providers=[OPENROUTER])
    server2.script("openrouter", OK)
    assert consented.usable_providers() == [OPENROUTER]
    assert consented.generate_json("demo", "p-or-2", Echo).provider == "openrouter"
    [call] = server2.calls
    assert call.body["provider"] == {"data_collection": "deny"}  # no-retention upstreams only


@pytest.mark.parametrize(("url", "local"), [
    ("http://localhost:11434", True), ("http://127.0.0.1:11434/v1", True), ("http://[::1]:11434", True),
    ("http://192.168.1.20:11434", True), ("http://10.0.0.5:11434", True), ("http://ollama:11434", True),
    ("http://my-pc.local:11434", True), ("localhost:11434", True), ("http://0.0.0.0:11434", True),
    ("https://ollama.com", False), ("http://8.8.8.8:11434", False), ("https://abc.ngrok-free.app", False),
    ("", False), (None, False),
])
def test_is_local_url(url: str | None, local: bool) -> None:
    assert is_local_url(url) is local


def test_ollama_is_used_only_when_local(harness) -> None:  # noqa: ANN001
    remote, server, _ = harness(ollama_base_url="https://ollama.example.com", ai_grading_providers=["ollama:llama3.1"])
    server.script("ollama", OK)
    assert remote.usable_providers() == []
    with pytest.raises(LLMUnavailable) as err:
        remote.generate_json("demo", "p-remote", Echo)
    assert server.calls == [] and "ollama_not_local" in err.value.reasons[0]

    local, server2, _ = harness(ollama_base_url="http://localhost:11434", ai_grading_providers=["ollama:llama3.1"])
    server2.script("ollama", OK)
    assert local.generate_json("demo", "p-local", Echo).provider == "ollama"
    assert server2.calls[0].url == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in server2.calls[0].headers


def test_timeout_before_any_reply_parks_the_provider(harness, session_factory) -> None:  # noqa: ANN001
    """#4: timeouts were retried with backoff (3 x 30 s + sleeps per spec, every sheet)."""
    router, server, sleeps = harness(ai_max_retries=2)
    server.script("groq", FakeReply(exc=httpx.ReadTimeout("timed out")))
    server.script("gemini", OK, OK)
    assert router.generate_json("demo", "p1", Echo).provider == "gemini"
    assert [c.provider for c in server.calls] == ["groq", "gemini"] and sleeps == []  # no retry, no backoff
    assert router.usable_providers() == [GEMINI]
    assert router.generate_json("demo", "p2", Echo).provider == "gemini"
    assert [c.provider for c in server.calls] == ["groq", "gemini", "gemini"]  # parked: not asked again
    [row] = [r for r in runs(session_factory) if r.provider == "groq"]
    assert row.status == "error" and row.error.startswith("error (timeout)")
    assert router.usage_last_24h("groq", "openai/gpt-oss-120b") == (1, 0)  # it was sent: it counts


def test_connect_timeout_is_no_connection(harness, session_factory) -> None:  # noqa: ANN001
    router, server, sleeps = harness(ai_grading_providers=[GROQ])
    server.script("groq", FakeReply(exc=httpx.ConnectTimeout("no route to host")))
    with pytest.raises(LLMUnavailable) as err:
        router.generate_json("demo", "p", Echo)
    assert len(server.calls) == 1 and sleeps == [] and router.usable_providers() == []
    assert "no connection" in err.value.reasons[0]
    [row] = runs(session_factory)
    assert row.error.startswith("not sent: error (network)")


def test_call_deadline_bounds_every_attempt_and_skips_the_rest(tmp_path: Path, session_factory) -> None:  # noqa: ANN001
    clock = [0.0]
    seen: list[tuple[str, float | None]] = []

    def black_hole(request: httpx.Request) -> httpx.Response:
        read = request.extensions.get("timeout", {}).get("read")
        seen.append((request.url.host, read))
        clock[0] += read or 0  # the provider never answers: the whole read timeout passes
        raise httpx.ReadTimeout("timed out", request=request)

    settings = make_settings(tmp_path, ai_grading_providers=[GROQ, "groq:openai/gpt-oss-20b", GEMINI],
                             ai_timeout_seconds=30, ai_call_deadline_seconds=45, ai_max_retries=2)
    with LLMRouter(settings, session_factory, transport=httpx.MockTransport(black_hole),
                   limiter=RateLimiter(clock=lambda: clock[0]),
                   sleep=lambda s: clock.__setitem__(0, clock[0] + s)) as router:
        with pytest.raises(LLMUnavailable) as err:
            router.generate_json("demo", "p", Echo)
    # 30 s on the first spec (parked), only the 15 s left for the second, none for Gemini
    assert seen == [("api.groq.com", 30.0), ("api.groq.com", 15.0)] and clock[0] == 45.0
    assert err.value.reasons[-1] == f"{GEMINI}: skipped: deadline"


def test_replies_that_fail_the_check_are_never_ok_or_cached(harness, session_factory) -> None:  # noqa: ANN001
    """#5: a schema-valid but incomplete reply was stored as 'ok' and replayed forever."""
    router, server, _ = harness(ai_max_retries=0)
    server.script("groq", {"items": [1]}).script("gemini", {"items": [1, 2]})
    res = router.generate_json("demo", "p", Pair, check=need_two)
    assert res.provider == "gemini" and res.complete and res.data.items == [1, 2]
    [bad] = [r for r in runs(session_factory) if r.status == "invalid"]
    assert bad.provider == "groq" and bad.error == "invalid: incomplete: 1 of 2 items"
    assert bad.response_json == {"items": [1]}
    assert router.generate_json("demo", "p", Pair, check=need_two).cached  # the complete reply is

    # accept_incomplete: handed back at once (complete=False), stored 'invalid', never cached
    server.script("groq", {"items": [7]}, {"items": [7]})
    part = router.generate_json("demo", "p-part", Pair, check=need_two, accept_incomplete=True)
    assert (part.complete, part.cached, part.data.items) == (False, False, [7]) and "incomplete" in part.problem
    again = router.generate_json("demo", "p-part", Pair, check=need_two, accept_incomplete=True)
    assert not again.cached and [c.provider for c in server.calls] == ["groq", "gemini", "groq", "groq"]


def test_cache_never_replays_an_ok_row_that_fails_the_check(harness) -> None:  # noqa: ANN001
    router, server, _ = harness()
    server.script("groq", {"items": [1]}, {"items": [1, 2]})
    assert router.generate_json("demo", "p", Pair).complete  # no check: 'ok' (an old, incomplete row)
    res = router.generate_json("demo", "p", Pair, check=need_two)
    assert not res.cached and res.data.items == [1, 2] and len(server.calls) == 2


HUGE_INT = '{"answer": "x", "score": ' + "1" * 5000 + "}"


@pytest.mark.parametrize("weird", [
    lambda: httpx.Response(200, json={"choices": {"0": {"message": {"content": "{}"}}}}),  # was KeyError
    lambda: httpx.Response(200, json={"choices": [{"message": {"content": HUGE_INT},
                                                   "finish_reason": "stop"}]}),  # was ValueError
    lambda: httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=b"this is not gzip"),  # DecodingError
    lambda: httpx.Response(200, json=[1, 2, 3]),
], ids=["choices-object", "5000-digit-int", "corrupt-gzip", "list-body"])
def test_bizarre_replies_fall_back_to_the_next_provider(harness, session_factory, weird) -> None:  # noqa: ANN001
    class Weird(FakeLLMServer):
        def handle(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "api.groq.com":
                return weird()
            return super().handle(request)

    router, server, _ = harness(server=Weird(), ai_max_retries=1)
    server.script("gemini", OK)
    assert router.generate_json("demo", "p", Echo).provider == "gemini"
    groq_rows = [r for r in runs(session_factory) if r.provider == "groq"]
    assert groq_rows and all(r.status in ("error", "invalid") for r in groq_rows)


def test_usage_sent_as_a_list_is_ignored_not_fatal(harness) -> None:  # noqa: ANN001
    class ListUsage(FakeLLMServer):
        def handle(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(OK)},
                                                          "finish_reason": "stop"}], "usage": [1, 2]})

    router, _, _ = harness(server=ListUsage())
    res = router.generate_json("demo", "p", Echo)  # was AttributeError
    assert res.provider == "groq" and res.usage == {"input_tokens": None, "output_tokens": None}


def test_unexpected_exceptions_fall_back_and_never_leak_keys(harness, session_factory, caplog) -> None:  # noqa: ANN001
    """#7: only LLMUnavailable may escape; an exception carrying key material is recorded
    by type only, and the next provider is tried."""
    caplog.set_level(logging.DEBUG)
    leaky = f"Bearer {FAKE_GROQ} gsk_{'A' * 24} AIza{'B' * 30} AQ.{'C' * 30}"

    class Exploding(FakeLLMServer):
        def handle(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "api.groq.com":
                raise RuntimeError(leaky)
            return super().handle(request)

    router, server, _ = harness(server=Exploding())
    server.script("gemini", OK)
    assert router.generate_json("demo", "p", Echo).provider == "gemini"
    [row] = [r for r in runs(session_factory) if r.provider == "groq"]
    assert (row.status, row.error) == ("error", "error (unexpected): RuntimeError")

    only_groq, _, _ = harness(server=Exploding(), ai_grading_providers=[GROQ])
    with pytest.raises(LLMUnavailable) as err:
        only_groq.generate_json("demo", "p-2", Echo)
    dumped = json.dumps([{c.name: getattr(r, c.name) for c in AiRun.__table__.columns}
                         for r in runs(session_factory)], default=str)
    for text in (str(err.value), repr(err.value), " ".join(err.value.reasons), caplog.text, dumped):
        for secret in (FAKE_GROQ, "A" * 24, "B" * 30, "C" * 30):
            assert secret not in text


def test_a_key_a_header_cannot_carry_is_refused_before_sending(harness, session_factory, caplog) -> None:  # noqa: ANN001
    caplog.set_level(logging.DEBUG)
    bad_key = "gsk_" + "Z" * 20 + "​" + "Y" * 8  # a zero-width space pasted into .env
    router, server, _ = harness(groq_api_key=bad_key)
    server.script("gemini", OK)
    assert router.generate_json("demo", "p", Echo).provider == "gemini"  # was UnicodeEncodeError
    assert [c.provider for c in server.calls] == ["gemini"]
    [row] = [r for r in runs(session_factory) if r.provider == "groq"]
    assert row.status == "error" and row.error.startswith("not sent: error (auth)")
    assert router.usable_providers() == [GEMINI]  # parked like a bad key
    dumped = json.dumps([{c.name: getattr(r, c.name) for c in AiRun.__table__.columns}
                         for r in runs(session_factory)], default=str)
    assert "Z" * 20 not in dumped and "Z" * 20 not in caplog.text


def test_offline_attempts_do_not_use_the_daily_budget(harness) -> None:  # noqa: ANN001
    """#8: 45 offline minutes used up OpenRouter's 45/day budget with nothing sent."""
    clock = [0.0]
    router, server, _ = harness(limiter=RateLimiter(clock=lambda: clock[0]), openrouter_api_key=FAKE_OPENROUTER,
                                openrouter_consent=True, ai_grading_providers=[OPENROUTER])
    for i in range(50):  # one sheet a minute, past the 60 s cooldown each time
        server.script("openrouter", FakeReply(exc=httpx.ConnectError("getaddrinfo failed")))
        with pytest.raises(LLMUnavailable):
            router.generate_json("demo", f"offline {i}", Echo)
        clock[0] += 61
    assert router.usage_last_24h("openrouter", OPENROUTER_MODEL) == (0, 0)
    server.script("openrouter", OK)  # back online
    assert router.generate_json("demo", "online", Echo).provider == "openrouter"
    assert router.usage_last_24h("openrouter", OPENROUTER_MODEL) == (1, 180)


def test_safe_scrubs_key_fragments_and_key_shaped_tokens(harness) -> None:  # noqa: ANN001
    router, _, _ = harness()
    out = router._safe(f"x {FAKE_GROQ[3:15]}yz AQ.{'a' * 20} sk-{'b' * 20} gsk_{'c' * 20} plain words")
    for gone in (FAKE_GROQ[3:15], "a" * 20, "b" * 20, "c" * 20):
        assert gone not in out
    assert out.endswith("plain words")
