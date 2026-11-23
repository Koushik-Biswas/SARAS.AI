"""LLMRouter: the one entry point for AI calls (`generate_json`).

Order of checks for every call (PRD 13.5, spec section 4):

1. No AI consent  -> LLMUnavailable, nothing is sent and nothing is read from the cache.
2. Cache          -> an earlier 'ok' ai_runs row with the same cache key is replayed
                     with no network call (and logged as a 'cache_hit' row). Only
                     complete, valid replies are ever stored as 'ok'; a cached row that
                     fails the caller's `check` is not replayed.
3. Providers in order (settings.ai_grading_providers unless overridden). A provider is
   skipped when: Gemini without gemini_consent, OpenRouter without openrouter_consent
   (its upstream models may log or train on prompts), an Ollama URL that is not local
   or on a private network, no key / URL, cooling down, its daily budget (rolling
   24 h, from ai_runs) is used up, the local per-minute bucket would make the call wait
   too long, or the call's deadline has passed.
4. Each provider gets 1 + ai_max_retries attempts. 429 / 5xx back off (honouring
   Retry-After); invalid JSON, schema-invalid output and replies that fail the
   caller's `check` (e.g. an incomplete sheet) are retried at once. A Retry-After
   longer than `max_retry_wait`, an auth failure, no connection at all, or a timeout
   before the provider answered anything in this call parks the provider (in-process
   cooldown) and moves on. A timeout after an earlier reply is retried with backoff.
5. settings.ai_call_deadline_seconds bounds the whole call (attempts, retries and
   backoff): each attempt's timeout is cut to the time left, and once it is gone the
   remaining providers are skipped ("skipped: deadline").
6. Every attempt writes an ai_runs row (anonymised prompt, schema, output, tokens,
   latency). Any exception from a provider becomes an 'error' row and the next
   provider is tried: only LLMUnavailable(reasons) ever escapes, and the caller falls
   back to the offline checker. Attempts that never reached the provider (no
   connection, a refused key) are marked "not sent: ..." and do not count toward the
   daily budgets.

API keys never reach logs, ai_runs rows or exception messages: every stored or raised
message is scrubbed of the configured keys (also fragments of them) and key-shaped
tokens.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import math
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ValidationError
from sqlalchemy import func, not_, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import Settings
from ..models import AiRun, new_id
from .limiter import Limits, RateLimiter, default_limiter
from .providers import (
    GROQ_BASE_URL,
    OPENROUTER_BASE_URL,
    GeminiProvider,
    ImageInput,
    OpenAICompatProvider,
    Provider,
    ProviderError,
    ProviderRequest,
)
from .schema import canonical_json, estimate_tokens, strict_json_schema

log = logging.getLogger(__name__)

PROVIDER_NAMES = ("groq", "gemini", "openrouter", "ollama")
# Request limits not exposed as settings, kept below the published free tiers.
GROQ_REQUESTS_PER_MINUTE = 25  # free tier: 30
GROQ_REQUESTS_PER_DAY = 900  # free tier: 1000
OPENROUTER_REQUESTS_PER_MINUTE = 15  # free models: 20
OPENROUTER_REQUESTS_PER_DAY = 45  # free models without credits: 50
AUTH_COOLDOWN_SECONDS = 600.0
NETWORK_COOLDOWN_SECONDS = 60.0  # no connection, or a timeout before any reply
# A provider that answered "overloaded" (HTTP 503 / 529, "high demand") is parked this
# long: retrying it page after page (each with backoff) cost minutes per sheet while the
# next provider in the list was answering in ~2.5 s (measured, ai_runs 2026-11-23).
OVERLOAD_COOLDOWN_SECONDS = 45.0
_OVERLOAD_TEXT = re.compile(r"overload|high demand|capacity", re.I)
MIN_ATTEMPT_SECONDS = 1.0  # an attempt needs at least this much of the deadline left
DEFAULT_DEADLINE_SECONDS = 60.0
TOKENS_PER_IMAGE = 1100
DEFAULT_OUTPUT_TOKENS = 1200
# ai_runs.error prefix of attempts that never reached the provider (not budgeted).
NOT_SENT_PREFIX = "not sent: "

_KEY_PATTERNS = re.compile(
    r"gsk_[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_\-]{20,}|AQ\.[0-9A-Za-z_\-.]{16,}|sk-[A-Za-z0-9_\-]{16,}"
    r"|(?i:bearer)\s+\S+|(?i:x-goog-api-key)\W+\S+")
_SECRET_GRAM = 12  # a run of this many characters of a configured key counts as the key
_TOKENISH = re.compile(r"[A-Za-z0-9_\-.+/=]{%d,}" % _SECRET_GRAM)
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)
_LOCAL_SUFFIXES = (".local", ".lan", ".internal", ".home.arpa", ".localdomain", ".localhost")


class LLMUnavailable(Exception):
    """No provider produced a valid answer. `reasons` lists why, one per provider tried
    (secret-free). Callers fall back to the offline checker."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("No AI provider is available: " + ("; ".join(self.reasons) or "none configured"))


@dataclass
class LLMResult:
    data: Any  # the validated pydantic model instance
    provider: str
    model: str
    cached: bool
    latency_ms: int
    usage: dict[str, int | None] = field(default_factory=dict)
    run_id: str | None = None
    # False only with generate_json(accept_incomplete=True): the reply passed the schema
    # but failed the caller's check (`problem` says why). Stored as 'invalid', never cached.
    complete: bool = True
    problem: str | None = None


class _Skip(Exception):
    pass


def parse_provider_spec(spec: str) -> tuple[str, str]:
    """'groq:openai/gpt-oss-120b' -> ('groq', 'openai/gpt-oss-120b')."""
    name, _, model = spec.partition(":")
    name, model = name.strip().lower(), model.strip()
    if name not in PROVIDER_NAMES or not model:
        raise ValueError(f"unknown provider spec {spec!r} (expected <{'|'.join(PROVIDER_NAMES)}>:<model>)")
    return name, model


def is_local_url(url: str | None) -> bool:
    """True when `url` points at this machine or a private network: localhost, a
    loopback / private / link-local IP, a single-label host name ("ollama", a PC's
    name) or a local-network suffix (.local, .lan, .internal, .home.arpa). Decided from
    the URL alone (no DNS lookup)."""
    if not url or not url.strip():
        return False
    raw = url.strip()
    try:
        host = (urlsplit(raw if "://" in raw else "http://" + raw).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost" or host.endswith(_LOCAL_SUFFIXES):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host  # a single-label name only resolves on the local network
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _overloaded(exc: ProviderError) -> bool:
    """HTTP 503 / 529 or an "overloaded / high demand" message: capacity, not a bug."""
    return exc.status_code in (503, 529) or bool(_OVERLOAD_TEXT.search(str(exc)))


def _strip_fences(text: str) -> str:
    m = _FENCE.match(text)
    return m.group(1) if m else text


def _secret_forms(secret: str) -> list[str]:
    forms = {secret, secret.strip(), repr(secret)[1:-1], repr(secret.strip())[1:-1]}
    return sorted((f for f in forms if len(f) >= 4), key=len, reverse=True)


class LLMRouter:
    def __init__(self, settings: Settings, session_factory: Callable[[], Session], *,
                 transport: httpx.BaseTransport | None = None, limiter: RateLimiter | None = None,
                 sleep: Callable[[float], None] = time.sleep, max_retry_wait: float = 30.0,
                 max_limiter_wait: float = 5.0, clock: Callable[[], float] | None = None) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.limiter = limiter or default_limiter()
        self.max_retry_wait = max_retry_wait
        self.max_limiter_wait = max_limiter_wait
        self._sleep = sleep
        # The deadline clock: the limiter's by default, so a test's fake clock and fake
        # sleep (which advances it) also drive the deadline.
        self._clock = clock or getattr(self.limiter, "clock", time.monotonic)
        # Pages of one sheet are read concurrently through the same router.
        self._lock = threading.Lock()
        self._client = httpx.Client(transport=transport, timeout=httpx.Timeout(settings.ai_timeout_seconds),
                                    follow_redirects=False)
        self._providers: dict[str, Provider] = {}
        raw = [k.get_secret_value() for k in (settings.groq_api_key, settings.gemini_api_key,
                                              settings.openrouter_api_key) if k is not None and k.get_secret_value()]
        self._secrets = sorted({f for s in raw for f in _secret_forms(s)}, key=len, reverse=True)
        self._grams = {s[i: i + _SECRET_GRAM] for s in (r.strip() for r in raw) if len(s) >= _SECRET_GRAM
                       for i in range(len(s) - _SECRET_GRAM + 1)}

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMRouter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers
    def _safe(self, text: Any, limit: int = 300) -> str:  # noqa: ANN401
        """Short, single-line and with every configured key (also a 12+ character
        fragment of one) and every key-shaped token removed."""
        out = " ".join(str(text or "").split())
        for secret in self._secrets:
            out = out.replace(secret, "***")
        out = _KEY_PATTERNS.sub("***", out)
        if self._grams:
            def scrub(m: re.Match[str]) -> str:
                tok = m.group(0)
                hit = any(tok[i: i + _SECRET_GRAM] in self._grams for i in range(len(tok) - _SECRET_GRAM + 1))
                return "***" if hit else tok

            out = _TOKENISH.sub(scrub, out)
        return out if len(out) <= limit else out[: limit - 1] + "…"

    def _provider(self, spec: str) -> Provider:
        try:
            name, model = parse_provider_spec(spec)
        except ValueError as exc:
            raise _Skip(str(exc)) from None
        s = self.settings
        if name == "gemini" and not s.gemini_consent:
            raise _Skip("skipped: Gemini needs its own consent (free tier may use data for training)")
        if name == "openrouter" and not getattr(s, "openrouter_consent", False):
            raise _Skip("skipped: openrouter_consent_missing (its upstream models may log or train on prompts)")
        key = f"{name}:{model}"
        with self._lock:
            if key in self._providers:
                return self._providers[key]
        timeout = s.ai_timeout_seconds
        if name == "groq":
            if s.groq_api_key is None:
                raise _Skip("skipped: no API key configured")
            p: Provider = OpenAICompatProvider("groq", GROQ_BASE_URL, model, self._client, timeout, s.groq_api_key)
        elif name == "gemini":
            if s.gemini_api_key is None:
                raise _Skip("skipped: no API key configured")
            p = GeminiProvider(model, self._client, timeout, s.gemini_api_key)
        elif name == "openrouter":
            if s.openrouter_api_key is None:
                raise _Skip("skipped: no API key configured")
            p = OpenAICompatProvider("openrouter", OPENROUTER_BASE_URL, model, self._client, timeout,
                                     s.openrouter_api_key)
        else:
            if not s.ollama_base_url:
                raise _Skip("skipped: no Ollama URL configured")
            if not is_local_url(s.ollama_base_url):
                # A remote "Ollama" (e.g. a cloud service) is a third party with its own
                # data policy: only a server on this machine or the local network counts.
                raise _Skip("skipped: ollama_not_local (only a local or private-network Ollama is used)")
            base = s.ollama_base_url.strip().rstrip("/")
            p = OpenAICompatProvider("ollama", base if base.endswith("/v1") else base + "/v1", model,
                                     self._client, timeout)
        with self._lock:
            return self._providers.setdefault(key, p)

    def limits_for(self, provider: str) -> Limits:
        s = self.settings
        if provider == "groq":
            return Limits(rpm=GROQ_REQUESTS_PER_MINUTE, tpm=s.groq_tokens_per_minute,
                          rpd=GROQ_REQUESTS_PER_DAY, tpd=s.groq_tokens_per_day)
        if provider == "gemini":
            return Limits(rpm=s.gemini_requests_per_minute, rpd=s.gemini_requests_per_day)
        if provider == "openrouter":
            return Limits(rpm=OPENROUTER_REQUESTS_PER_MINUTE, rpd=OPENROUTER_REQUESTS_PER_DAY)
        return Limits()  # local Ollama

    def usage_last_24h(self, provider: str, model: str) -> tuple[int, int]:
        """(requests, tokens) sent to provider:model in the last 24 hours. Cache hits and
        attempts that never reached the provider (no connection) are not counted."""
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        tokens = func.coalesce(AiRun.input_tokens, 0) + func.coalesce(AiRun.output_tokens, 0)
        reached = or_(AiRun.error.is_(None), not_(AiRun.error.startswith(NOT_SENT_PREFIX)))
        with self.session_factory() as db:
            requests, used = db.execute(
                select(func.count(AiRun.id), func.coalesce(func.sum(tokens), 0)).where(
                    AiRun.provider == provider, AiRun.model == model,
                    AiRun.created_at >= since, AiRun.status != "cache_hit", reached,
                )
            ).one()
        return int(requests or 0), int(used or 0)

    def usable_providers(self) -> list[str]:
        """Provider specs that could be tried right now (consent, keys, local Ollama, cooldown)."""
        if not self.settings.ai_consent:
            return []
        out = []
        for spec in self.settings.ai_grading_providers:
            try:
                p = self._provider(spec)
            except _Skip:
                continue
            if self.limiter.cooldown_left(p.key) <= 0:
                out.append(p.key)
        return out

    def _record(self, **fields: Any) -> str | None:  # noqa: ANN401
        if fields.get("error"):
            fields["error"] = self._safe(fields["error"])
        row = AiRun(id=new_id(), **fields)
        try:
            with self.session_factory() as db:
                db.add(row)
                db.commit()
            return row.id
        except SQLAlchemyError as exc:
            log.warning("Could not record an AI run (%s)", type(exc).__name__)
            return None

    def _deadline(self) -> float | None:
        seconds = getattr(self.settings, "ai_call_deadline_seconds", DEFAULT_DEADLINE_SECONDS)
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            seconds = DEFAULT_DEADLINE_SECONDS
        if not math.isfinite(seconds) or seconds <= 0:
            return None  # no overall limit
        return self._clock() + seconds

    def _left(self, deadline: float | None) -> float:
        return math.inf if deadline is None else deadline - self._clock()

    # ------------------------------------------------------------------ public API
    def generate_json(self, task: str, prompt: str, schema: type[BaseModel], *, system: str | None = None,
                      prompt_version: str = "1", images: Sequence[ImageInput] | None = None,
                      exam_id: str | None = None, submission_id: str | None = None,
                      max_output_tokens: int | None = None, providers: Sequence[str] | None = None,
                      check: Callable[[Any], str | None] | None = None,
                      accept_incomplete: bool = False, expected_output_tokens: int | None = None,
                      limiter_wait: float | None = None, use_cache: bool = True,
                      reasoning_effort: str | None = None) -> LLMResult:
        """Ask the configured providers for JSON matching `schema` (a pydantic model,
        also used to validate the reply). `prompt`/`system` must already be anonymised.

        `check(data)` is a semantic check run on every schema-valid reply (and on cached
        ones): it returns None when the reply is usable, else a short, text-free problem
        ("incomplete: ..."). Such a reply is stored as 'invalid' (never 'ok', never
        replayed from the cache) and retried like invalid output - unless
        `accept_incomplete`, in which case it is returned at once with complete=False so
        the caller can keep the usable part and re-ask for the rest.

        `expected_output_tokens`: the typical reply size, used for the per-minute token
        reservation instead of the `max_output_tokens` ceiling (a page read is capped at
        4096 tokens but uses ~750: reserving the cap let only one page a minute through
        Groq's 6K-token bucket). `limiter_wait`: longest wait for the local per-minute
        bucket before the provider is skipped (default: the router's max_limiter_wait).
        `use_cache=False` always asks a provider (benchmark consistency runs).
        `reasoning_effort`: low | medium | high, for reasoning models that take it."""
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError("schema must be a pydantic BaseModel subclass")
        if not self.settings.ai_consent:
            log.info("AI %s not sent: AI consent is off", task)
            raise LLMUnavailable(["AI consent not given: nothing was sent"])

        images = list(images or [])
        json_schema = strict_json_schema(schema)
        request_json = {
            "task": task, "prompt_version": prompt_version, "system": system, "prompt": prompt,
            "schema": json_schema,
            "images": [{"sha256": img.digest(), "mime_type": img.mime_type} for img in images],
        }
        cache_key = hashlib.sha256(canonical_json(request_json).encode("utf-8")).hexdigest()
        base = {"task": task, "prompt_version": prompt_version, "cache_key": cache_key,
                "exam_id": exam_id, "submission_id": submission_id}

        if use_cache:
            hit = self._from_cache(schema, base, check)
            if hit is not None:
                return hit

        req = ProviderRequest(prompt=prompt, json_schema=json_schema, system=system, images=images,
                              schema_name=re.sub(r"[^A-Za-z0-9_-]", "_", task)[:64] or "output",
                              max_output_tokens=max_output_tokens, reasoning_effort=reasoning_effort)
        out_estimate = expected_output_tokens or max_output_tokens or DEFAULT_OUTPUT_TOKENS
        if max_output_tokens:
            out_estimate = min(out_estimate, max_output_tokens)
        estimate = (estimate_tokens((system or "") + prompt + canonical_json(json_schema))
                    + TOKENS_PER_IMAGE * len(images) + out_estimate)
        deadline = self._deadline()
        reasons: list[str] = []
        for spec in (providers if providers is not None else self.settings.ai_grading_providers):
            if self._left(deadline) < MIN_ATTEMPT_SECONDS:
                reasons.append(f"{spec}: skipped: deadline")
                continue
            try:
                provider = self._provider(spec)
            except _Skip as exc:
                reasons.append(f"{spec}: {exc}")
                continue
            try:
                outcome = self._try_provider(provider, req, schema, estimate, base, request_json,
                                             deadline=deadline, check=check, accept_incomplete=accept_incomplete,
                                             limiter_wait=limiter_wait)
            except Exception as exc:  # noqa: BLE001 - AI is never in the critical path: next provider
                log.warning("AI %s via %s failed unexpectedly (%s)", task, provider.key, type(exc).__name__)
                outcome = f"error (unexpected): {type(exc).__name__}"
            if isinstance(outcome, LLMResult):
                return outcome
            reasons.append(f"{provider.key}: {outcome}")
        reasons = [self._safe(r) for r in reasons]
        log.warning("AI %s unavailable after %d provider(s); using the offline checker", task, len(reasons))
        raise LLMUnavailable(reasons)

    # ------------------------------------------------------------------ internals
    def _from_cache(self, schema: type[BaseModel], base: dict[str, Any],
                    check: Callable[[Any], str | None] | None = None) -> LLMResult | None:
        t0 = time.perf_counter()
        try:
            with self.session_factory() as db:
                rows = db.scalars(
                    select(AiRun).where(AiRun.cache_key == base["cache_key"], AiRun.status == "ok")
                    .order_by(AiRun.created_at.desc()).limit(3)
                ).all()
        except SQLAlchemyError as exc:
            log.warning("AI cache lookup failed (%s)", type(exc).__name__)
            return None
        for row in rows:
            if row.task != base["task"] or row.prompt_version != base["prompt_version"]:
                continue
            try:
                data = schema.model_validate(row.response_json)
                if check is not None and check(data):
                    continue  # e.g. an incomplete reply cached as 'ok' before the check existed
            except Exception:  # noqa: BLE001, S112 - an unreadable cached row is just not replayed
                continue
            latency = int((time.perf_counter() - t0) * 1000)
            run_id = self._record(**base, provider=row.provider, model=row.model, status="cache_hit",
                                  latency_ms=latency, input_tokens=0, output_tokens=0)
            log.info("AI %s served from cache (%s:%s)", base["task"], row.provider, row.model)
            return LLMResult(data, row.provider, row.model, True, latency,
                             {"input_tokens": 0, "output_tokens": 0}, run_id)
        return None

    def _try_provider(self, p: Provider, req: ProviderRequest, schema: type[BaseModel], estimate: int,  # noqa: C901, PLR0912, PLR0915
                      base: dict[str, Any], request_json: dict[str, Any], *, deadline: float | None = None,
                      check: Callable[[Any], str | None] | None = None,
                      accept_incomplete: bool = False, limiter_wait: float | None = None) -> LLMResult | str:
        key = p.key
        max_wait = self.max_limiter_wait if limiter_wait is None else max(0.0, float(limiter_wait))
        left = self.limiter.cooldown_left(key)
        if left > 0:
            return f"skipped: cooling down for {left:.0f}s"
        limits = self.limits_for(p.name)
        if limits.rpd or limits.tpd:
            try:
                n_requests, n_tokens = self.usage_last_24h(p.name, p.model)
            except SQLAlchemyError as exc:
                log.warning("AI budget lookup failed (%s)", type(exc).__name__)
                n_requests, n_tokens = 0, 0
            if limits.rpd and n_requests + 1 > limits.rpd:
                return f"skipped: daily request budget used ({n_requests}/{limits.rpd})"
            if limits.tpd and n_tokens + estimate > limits.tpd:
                return f"skipped: daily token budget used ({n_tokens}/{limits.tpd})"

        reserve = min(estimate, limits.tpm) if limits.tpm else estimate
        last = "no attempt made"
        attempts = 1 + max(0, self.settings.ai_max_retries)
        responded = False  # the provider answered something (any HTTP reply) during this call
        for attempt in range(attempts):
            remaining = self._left(deadline)
            if remaining < MIN_ATTEMPT_SECONDS:
                return "skipped: deadline" if attempt == 0 else f"{last} (deadline reached)"
            wait = self.limiter.reserve(key, limits, reserve)
            waited = 0.0
            while wait > 0:
                # Re-checked after every sleep: a timer that wakes a little early, or another
                # thread (pages are read concurrently) that took the tokens first, means a
                # short extra wait - not a skipped provider and an unread page.
                if waited + wait > max_wait or wait > remaining - MIN_ATTEMPT_SECONDS:
                    return f"skipped: local rate limit (free again in {wait:.0f}s)"
                step = max(wait, 0.05)
                self._sleep(step)
                waited += step
                remaining = self._left(deadline)
                wait = self.limiter.reserve(key, limits, reserve)
            timeout = None if deadline is None else max(MIN_ATTEMPT_SECONDS,
                                                         min(float(self.settings.ai_timeout_seconds), remaining))
            row = {**base, "provider": p.name, "model": p.model, "request_json": request_json}
            t0 = time.perf_counter()
            try:
                resp = p.generate(replace(req, timeout=timeout))
            except ProviderError as exc:
                latency = int((time.perf_counter() - t0) * 1000)
                # Any HTTP reply (an error status, unusable output) shows the provider is alive.
                responded = responded or exc.status_code is not None or \
                    (exc.sent and exc.kind not in ("timeout", "network"))
                used = None
                if exc.input_tokens is not None or exc.output_tokens is not None:
                    used = (exc.input_tokens or 0) + (exc.output_tokens or 0)
                elif exc.kind in ("rate_limited", "auth", "client", "network") or not exc.sent:
                    used = 0  # rejected before generation: give the reservation back
                self.limiter.settle(key, limits, reserve, used)
                status = {"rate_limited": "rate_limited", "invalid": "invalid"}.get(exc.kind, "error")
                last = self._safe(f"{status} ({exc.kind}): {exc}")
                self._record(**row, status=status, latency_ms=latency, input_tokens=exc.input_tokens,
                             output_tokens=exc.output_tokens, error=last if exc.sent else NOT_SENT_PREFIX + last,
                             response_json={"raw": exc.raw_text[:4000]} if exc.raw_text else None)
                log.info("AI %s via %s: %s in %d ms (attempt %d)", base["task"], key, status, latency, attempt + 1)
                if exc.kind == "server" and _overloaded(exc):
                    # "Model overloaded / high demand": retrying it now rarely helps and costs
                    # seconds per attempt. Park it so the next provider answers this call and
                    # every later page / sheet skips it until the cooldown ends.
                    self.limiter.set_cooldown(key, OVERLOAD_COOLDOWN_SECONDS)
                    return last
                if exc.kind in ("auth", "network") or (exc.kind == "timeout" and not responded):
                    # Bad key, no connection (offline laptop), or a provider that has not
                    # answered anything in this call: park it so this and the next sheets
                    # go straight to the fallback instead of waiting again.
                    self.limiter.set_cooldown(key, AUTH_COOLDOWN_SECONDS if exc.kind == "auth"
                                              else NETWORK_COOLDOWN_SECONDS)
                    return last
                if not exc.retryable or attempt + 1 >= attempts:
                    if exc.retry_after and exc.retry_after > self.max_retry_wait:
                        self.limiter.set_cooldown(key, exc.retry_after)
                    return last
                if exc.kind == "invalid":
                    delay = 0.0
                elif exc.retry_after is not None:
                    delay = exc.retry_after
                else:
                    delay = min(2.0 ** attempt, 8.0)
                if delay > self.max_retry_wait:
                    self.limiter.set_cooldown(key, delay)
                    return f"{last} (retry after {delay:.0f}s)"
                if delay > self._left(deadline) - MIN_ATTEMPT_SECONDS:
                    return f"{last} (deadline reached)"
                if delay > 0:
                    self._sleep(delay)
                continue
            except Exception as exc:  # noqa: BLE001 - a provider bug or a bizarre reply: next provider
                latency = int((time.perf_counter() - t0) * 1000)
                self.limiter.settle(key, limits, reserve, None)
                last = self._safe(f"error (unexpected): {type(exc).__name__}")
                self._record(**row, status="error", latency_ms=latency, error=last)
                log.warning("AI %s via %s: unexpected %s (attempt %d)", base["task"], key, type(exc).__name__,
                            attempt + 1)
                return last

            responded = True
            latency = int((time.perf_counter() - t0) * 1000)
            used = None
            if resp.input_tokens is not None or resp.output_tokens is not None:
                used = (resp.input_tokens or 0) + (resp.output_tokens or 0)
            self.limiter.settle(key, limits, reserve, used)
            usage = {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}
            payload: Any = None
            try:
                payload = json.loads(_strip_fences(resp.text))
                data = schema.model_validate(payload)
            except json.JSONDecodeError as exc:
                last = f"invalid: output is not JSON (line {exc.lineno} col {exc.colno})"
            except ValidationError as exc:
                first = exc.errors()[0] if exc.errors() else {}
                loc = ".".join(str(x) for x in first.get("loc", ()))
                last = f"invalid: output does not match the schema ({exc.error_count()} error(s), first at '{loc}')"
            except Exception as exc:  # noqa: BLE001 - e.g. a 5000-digit number, nesting too deep
                last = f"invalid: output could not be read ({type(exc).__name__})"
            else:
                problem: str | None = None
                retry_only = False
                if check is not None:
                    try:
                        problem = check(data)
                    except Exception as exc:  # noqa: BLE001
                        problem, retry_only = f"check failed ({type(exc).__name__})", True
                if not problem:
                    run_id = self._record(**row, status="ok", latency_ms=latency, response_json=payload, **usage)
                    log.info("AI %s via %s: ok in %d ms", base["task"], key, latency)
                    return LLMResult(data, p.name, p.model, False, latency, usage, run_id)
                last = self._safe(f"invalid: {problem}")
                stored = payload if isinstance(payload, dict) else {"raw": resp.text[:4000]}
                run_id = self._record(**row, status="invalid", latency_ms=latency, response_json=stored, error=last,
                                      **usage)
                log.info("AI %s via %s: unusable reply in %d ms (attempt %d)", base["task"], key, latency,
                         attempt + 1)
                if accept_incomplete and not retry_only:
                    return LLMResult(data, p.name, p.model, False, latency, usage, run_id,
                                     complete=False, problem=last)
                continue
            stored = payload if isinstance(payload, dict) else {"raw": resp.text[:4000]}
            self._record(**row, status="invalid", latency_ms=latency, response_json=stored, error=last, **usage)
            log.info("AI %s via %s: invalid output in %d ms (attempt %d)", base["task"], key, latency, attempt + 1)
        return last
