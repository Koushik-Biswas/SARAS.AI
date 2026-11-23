"""Provider adapters: one small interface over plain httpx (no vendor SDKs).

* GeminiProvider        Google `generateContent` REST (x-goog-api-key header,
                        responseMimeType application/json + responseJsonSchema, with a
                        one-time fallback to the older OpenAPI `responseSchema`).
* OpenAICompatProvider  `/chat/completions` for Groq, OpenRouter and Ollama (Bearer
                        token, response_format json_schema strict). OpenRouter requests
                        always carry provider.data_collection = "deny", so OpenRouter only
                        routes them to upstream endpoints that do not store or train on
                        prompts.

Adapters only translate requests and classify failures (ProviderError.kind, and
ProviderError.sent: whether the request reached the provider); the retry, fallback,
caching and budget logic lives in router.py. API keys are only ever placed in request
headers, never in URLs, messages or exceptions. A key that an HTTP header cannot carry
(non-ASCII, e.g. a zero-width space pasted into .env, or inner whitespace) is refused
before anything is sent, with a message that does not contain it.
"""

from __future__ import annotations

import base64
import email.utils
import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import SecretStr

from .schema import to_openapi_schema

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
SEED = 42
GPT_OSS_MIN_OUTPUT_TOKENS = 1200
# Extra max_tokens for the hidden reasoning of gpt-oss models, by reasoning effort.
REASONING_HEADROOM = {"low": 0, "medium": 2048, "high": 6144}
# Connecting is quick when the network works; a black-holed network should not cost the
# full read timeout per attempt.
CONNECT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class ImageInput:
    """An inline image for vision tasks (e.g. a cropped answer region)."""

    data: bytes
    mime_type: str = "image/png"

    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass
class ProviderRequest:
    prompt: str
    json_schema: dict[str, Any]
    schema_name: str = "output"
    system: str | None = None
    images: Sequence[ImageInput] = field(default_factory=tuple)
    max_output_tokens: int | None = None
    timeout: float | None = None  # seconds for this attempt (the router's deadline); None = the provider's
    reasoning_effort: str | None = None  # low | medium | high (reasoning models); None = the adapter's default


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class ProviderError(Exception):
    """kind: rate_limited | server | timeout | network | invalid (bad output) |
    auth | client. The message is short and never contains a key. "network"
    (no connection at all) is not retried: the router parks the provider instead.
    `sent` is False when the request never reached the provider (no connection, a
    refused key): such attempts do not count toward the provider's daily budget."""

    RETRYABLE = frozenset({"rate_limited", "server", "timeout", "invalid"})

    def __init__(self, kind: str, message: str, *, status_code: int | None = None,
                 retry_after: float | None = None, raw_text: str | None = None,
                 input_tokens: int | None = None, output_tokens: int | None = None, sent: bool = True) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.retry_after = retry_after
        self.raw_text = raw_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.sent = sent

    @property
    def retryable(self) -> bool:
        return self.kind in self.RETRYABLE


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After as delta-seconds or an HTTP date; None when absent or unreadable."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return max(0.0, seconds) if math.isfinite(seconds) else None
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _json_of(resp: httpx.Response) -> Any:  # noqa: ANN401
    """The response body as JSON; ValueError when it is not (bad JSON, bad encoding, too deep)."""
    try:
        return resp.json()
    except RecursionError:
        raise ValueError("response nested too deeply") from None


def _error_body(resp: httpx.Response) -> dict[str, Any]:
    try:
        body = _json_of(resp)
    except ValueError:
        return {}
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return err
    return {"message": err} if isinstance(err, str) else {}


def _short(text: Any, limit: int = 160) -> str:  # noqa: ANN401
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _count(value: Any) -> int | None:  # noqa: ANN401
    """A token count from a provider envelope, or None when it is missing or not a count."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value >= 0:
        return int(value)
    return None


def _dict(value: Any) -> dict[str, Any]:  # noqa: ANN401
    return value if isinstance(value, dict) else {}


class Provider(ABC):
    name: str = ""

    def __init__(self, model: str, client: httpx.Client, timeout: float, api_key: SecretStr | None = None) -> None:
        self.model = model
        self.client = client
        self.timeout = timeout
        self._api_key = api_key

    @property
    def key(self) -> str:
        return f"{self.name}:{self.model}"

    def _secret(self) -> str | None:
        """The configured key as written (used to scrub messages)."""
        return self._api_key.get_secret_value() if self._api_key is not None else None

    def _header_secret(self) -> str | None:
        """The key for the request header: surrounding whitespace removed; refused before
        sending when a header cannot carry it (the message never contains the key)."""
        raw = self._secret()
        if raw is None:
            return None
        key = raw.strip()
        if not key:
            return None
        if not (key.isascii() and key.isprintable()) or any(c.isspace() for c in key):
            raise ProviderError("auth", "the API key contains characters an HTTP header cannot carry "
                                        "(copy it into backend/.env again)", sent=False)
        return key

    def _clean(self, text: Any) -> str:  # noqa: ANN401
        out = _short(text)
        secret = self._secret()
        if secret:
            out = out.replace(secret, "***")
            if secret.strip():
                out = out.replace(secret.strip(), "***")
        return out

    def _post(self, url: str, headers: dict[str, str], body: dict[str, Any],
              timeout: float | None = None) -> httpx.Response:
        t = float(timeout) if timeout else float(self.timeout)
        try:
            return self.client.post(url, headers=headers, json=body,
                                    timeout=httpx.Timeout(t, connect=min(t, CONNECT_TIMEOUT_SECONDS)))
        except (httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteTimeout) as exc:
            # The request never (fully) reached the provider: treated like no connection.
            raise ProviderError("network", f"no connection within {t:g}s ({type(exc).__name__})",
                                sent=False) from None
        except httpx.TimeoutException:
            # ReadTimeout: the request was sent, the reply did not come in time.
            raise ProviderError("timeout", f"no response within {t:g}s") from None
        except (httpx.ConnectError, httpx.ProxyError, httpx.UnsupportedProtocol, httpx.LocalProtocolError,
                httpx.WriteError) as exc:
            raise ProviderError("network", f"connection failed ({type(exc).__name__})", sent=False) from None
        except httpx.TransportError as exc:  # ReadError, RemoteProtocolError, CloseError: sent, then lost
            raise ProviderError("network", f"connection failed ({type(exc).__name__})") from None
        except httpx.DecodingError:
            raise ProviderError("server", "response body could not be decoded") from None
        except httpx.InvalidURL:
            raise ProviderError("client", "invalid provider URL", sent=False) from None
        except (httpx.HTTPError, httpx.StreamError) as exc:
            raise ProviderError("client", f"request failed ({type(exc).__name__})") from None

    def _http_error(self, resp: httpx.Response, message: Any,  # noqa: ANN401
                    retry_after: float | None = None) -> ProviderError:
        code = resp.status_code
        if code == 429:
            kind = "rate_limited"
        elif code in (408, 504):
            kind = "timeout"
        elif code >= 500:
            kind = "server"
        elif code in (401, 403):
            kind = "auth"
        else:
            kind = "client"
        if retry_after is None:
            retry_after = parse_retry_after(resp.headers.get("retry-after"))
        return ProviderError(kind, f"HTTP {code}: {self._clean(message) or resp.reason_phrase}",
                             status_code=code, retry_after=retry_after)

    @abstractmethod
    def generate(self, req: ProviderRequest) -> ProviderResponse: ...


class OpenAICompatProvider(Provider):
    """Groq, OpenRouter, Ollama (and other OpenAI-compatible chat/completions APIs)."""

    def __init__(self, name: str, base_url: str, model: str, client: httpx.Client, timeout: float,
                 api_key: SecretStr | None = None) -> None:
        super().__init__(model, client, timeout, api_key)
        self.name = name
        self.base_url = base_url.rstrip("/")

    def build_body(self, req: ProviderRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        content: Any = req.prompt
        if req.images:
            content = [{"type": "text", "text": req.prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:{img.mime_type};base64,{img.b64()}"}}
                for img in req.images
            ]
        messages.append({"role": "user", "content": content})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "seed": SEED,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": req.schema_name, "strict": True, "schema": req.json_schema},
            },
        }
        if self.name == "openrouter":
            # Only upstream endpoints that neither store nor train on prompts (PRD 13.5).
            body["provider"] = {"data_collection": "deny"}
        max_tokens = req.max_output_tokens
        if self.name == "groq" and "gpt-oss" in self.model:
            # Reasoning models: keep reasoning short and out of the reply; leave room for it.
            effort = req.reasoning_effort if req.reasoning_effort in REASONING_HEADROOM else "low"
            body["reasoning_effort"] = effort
            body["include_reasoning"] = False
            # Reasoning tokens count against max_tokens: without headroom a longer think
            # truncates the JSON reply (Groq then rejects it as failing the schema).
            max_tokens = max(max_tokens or 0, GPT_OSS_MIN_OUTPUT_TOKENS) + REASONING_HEADROOM[effort]
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        return body

    def generate(self, req: ProviderRequest) -> ProviderResponse:
        headers = {"Content-Type": "application/json"}
        secret = self._header_secret()
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        resp = self._post(f"{self.base_url}/chat/completions", headers, self.build_body(req), req.timeout)
        if resp.status_code != 200:
            err = _error_body(resp)
            if resp.status_code == 400 and err.get("code") == "json_validate_failed":
                # Groq strict mode: the model's output failed the schema (worth a retry).
                raise ProviderError("invalid", "output failed the JSON schema", status_code=400,
                                    raw_text=str(err.get("failed_generation") or "")[:4000])
            raise self._http_error(resp, err.get("message"))
        try:
            data = _json_of(resp)
        except ValueError:
            raise ProviderError("server", "response was not JSON", status_code=resp.status_code) from None
        if not isinstance(data, dict):
            raise ProviderError("server", "unexpected response shape", status_code=resp.status_code)
        choices = data.get("choices")
        choice = _dict(choices[0]) if isinstance(choices, list) and choices else {}
        content = _dict(choice.get("message")).get("content")
        if isinstance(content, list):  # content parts
            content = "".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
        if not isinstance(content, str):
            content = None
        usage = _dict(data.get("usage"))
        in_t, out_t = _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens"))
        finish = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
        if finish == "length":
            raise ProviderError("invalid", "output truncated at the token limit", raw_text=content,
                                input_tokens=in_t, output_tokens=out_t)
        if not content or not content.strip():
            raise ProviderError("invalid", "empty output", input_tokens=in_t, output_tokens=out_t)
        return ProviderResponse(content, in_t, out_t, finish)


_GEMINI_BLOCKED = {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "LANGUAGE"}
_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s*$")


class GeminiProvider(Provider):
    name = "gemini"
    # model -> schema field that the API accepted ("responseJsonSchema" | "responseSchema")
    _schema_field: dict[str, str] = {}

    def __init__(self, model: str, client: httpx.Client, timeout: float, api_key: SecretStr | None,
                 base_url: str = GEMINI_BASE_URL) -> None:
        super().__init__(model, client, timeout, api_key)
        self.base_url = base_url.rstrip("/")

    def build_body(self, req: ProviderRequest, schema_field: str = "responseJsonSchema") -> dict[str, Any]:
        parts: list[dict[str, Any]] = [{"text": req.prompt}]
        parts += [{"inlineData": {"mimeType": img.mime_type, "data": img.b64()}} for img in req.images]
        config: dict[str, Any] = {"temperature": 0, "responseMimeType": "application/json"}
        if schema_field == "responseJsonSchema":
            config["responseJsonSchema"] = req.json_schema
        else:
            config["responseSchema"] = to_openapi_schema(req.json_schema)
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}
        if req.system:
            body["systemInstruction"] = {"parts": [{"text": req.system}]}
        return body

    def _retry_delay(self, resp: httpx.Response, err: dict[str, Any]) -> float | None:
        details = err.get("details")
        for detail in details if isinstance(details, list) else []:
            if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
                m = _DURATION.match(str(detail.get("retryDelay", "")))
                if m:
                    return float(m.group(1))
        return parse_retry_after(resp.headers.get("retry-after"))

    def generate(self, req: ProviderRequest) -> ProviderResponse:
        secret = self._header_secret()
        headers = {"Content-Type": "application/json", "x-goog-api-key": secret or ""}
        url = f"{self.base_url}/models/{self.model}:generateContent"
        schema_field = self._schema_field.get(self.model, "responseJsonSchema")
        resp = self._post(url, headers, self.build_body(req, schema_field), req.timeout)
        if resp.status_code == 400 and schema_field == "responseJsonSchema":
            msg = str(_error_body(resp).get("message", ""))
            if re.search(r"response_?json_?schema|Unknown name|Invalid JSON payload", msg, re.I):
                schema_field = "responseSchema"
                resp = self._post(url, headers, self.build_body(req, schema_field), req.timeout)
        if resp.status_code != 200:
            err = _error_body(resp)
            raise self._http_error(resp, err.get("message"), self._retry_delay(resp, err))
        self._schema_field[self.model] = schema_field
        try:
            data = _json_of(resp)
        except ValueError:
            raise ProviderError("server", "response was not JSON", status_code=resp.status_code) from None
        if not isinstance(data, dict):
            raise ProviderError("server", "unexpected response shape", status_code=resp.status_code)
        usage = _dict(data.get("usageMetadata"))
        in_t = _count(usage.get("promptTokenCount"))
        out_t = ((_count(usage.get("candidatesTokenCount")) or 0) + (_count(usage.get("thoughtsTokenCount")) or 0)) \
            or None
        candidates = data.get("candidates")
        cand = _dict(candidates[0]) if isinstance(candidates, list) and candidates else {}
        if not cand:
            blocked = _dict(data.get("promptFeedback")).get("blockReason")
            blocked = _short(blocked, 40) if isinstance(blocked, str) else None
            raise ProviderError("client" if blocked else "invalid", f"no candidates (block reason: {blocked})",
                                input_tokens=in_t, output_tokens=out_t)
        finish = cand.get("finishReason") if isinstance(cand.get("finishReason"), str) else None
        parts = _dict(cand.get("content")).get("parts")
        text = "".join(str(p.get("text") or "") for p in (parts if isinstance(parts, list) else [])
                       if isinstance(p, dict) and not p.get("thought"))
        if finish == "MAX_TOKENS":
            raise ProviderError("invalid", "output truncated at the token limit", raw_text=text,
                                input_tokens=in_t, output_tokens=out_t)
        if finish in _GEMINI_BLOCKED:
            raise ProviderError("client", f"output blocked ({finish})", input_tokens=in_t, output_tokens=out_t)
        if not text.strip():
            raise ProviderError("invalid", f"empty output (finish reason: {_short(finish, 40)})",
                                input_tokens=in_t, output_tokens=out_t)
        return ProviderResponse(text, in_t, out_t, finish)
