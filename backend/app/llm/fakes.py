"""Offline fake AI providers for tests and demos (spec section 4: tests never call
real providers). Plug `FakeLLMServer().transport` into LLMRouter(transport=...):
it answers Groq / OpenRouter / Ollama chat/completions and Gemini generateContent
requests with the same envelopes the real APIs use, and records every call.

    server = FakeLLMServer()
    server.script("groq", {"questions": [...]})                 # 200 with this JSON output
    server.script("gemini", FakeReply(status=429, headers={"Retry-After": "2"}))
    server.script("groq", "not json")                           # malformed output
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote

import httpx


@dataclass
class FakeCall:
    provider: str  # groq | gemini | openrouter | ollama
    model: str
    url: str
    headers: dict[str, str]
    body: dict[str, Any]
    system: str | None
    prompt: str


@dataclass
class FakeReply:
    output: Any = None  # JSON-able model output, wrapped in a 200 envelope
    raw: str | None = None  # raw model text (e.g. malformed JSON), wrapped in a 200 envelope
    status: int = 200  # non-200: `body` is sent as-is
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None
    exc: Exception | None = None  # raised instead of answering (e.g. httpx.ReadTimeout)
    finish_reason: str | None = None
    input_tokens: int = 120
    output_tokens: int = 60


Responder = Callable[[FakeCall], "FakeReply | dict | str"]


def _provider_of(request: httpx.Request) -> str:
    host = request.url.host
    if host.endswith("groq.com"):
        return "groq"
    if host.endswith("googleapis.com"):
        return "gemini"
    if host.endswith("openrouter.ai"):
        return "openrouter"
    return "ollama"


def _text_of(content: Any) -> str:  # noqa: ANN401
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class FakeLLMServer:
    def __init__(self, responder: Responder | None = None) -> None:
        self.calls: list[FakeCall] = []
        self.responder = responder
        self._scripts: dict[str, list[Any]] = {}

    def script(self, target: str, *replies: FakeReply | dict | str) -> FakeLLMServer:
        """Queue replies for 'groq', 'gemini', ... or one 'provider:model'. Each request
        consumes one; once a queue is empty the responder (or a 500) answers."""
        self._scripts.setdefault(target, []).extend(replies)
        return self

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def calls_to(self, provider: str) -> list[FakeCall]:
        return [c for c in self.calls if c.provider == provider or f"{c.provider}:{c.model}" == provider]

    def handle(self, request: httpx.Request) -> httpx.Response:
        provider = _provider_of(request)
        body = json.loads(request.content or b"{}")
        if provider == "gemini":
            path = unquote(request.url.path)
            model = path.split("/models/", 1)[-1].split(":", 1)[0]
            system = _text_of(((body.get("systemInstruction") or {}).get("parts")) or [])
            prompt = "".join(_text_of(c.get("parts")) for c in body.get("contents", []))
        else:
            model = body.get("model", "")
            msgs = body.get("messages", [])
            system = "".join(_text_of(m.get("content")) for m in msgs if m.get("role") == "system")
            prompt = "".join(_text_of(m.get("content")) for m in msgs if m.get("role") == "user")
        call = FakeCall(provider, model, str(request.url), dict(request.headers), body, system or None, prompt)
        self.calls.append(call)

        reply: Any = None
        for target in (f"{provider}:{model}", provider):
            if self._scripts.get(target):
                reply = self._scripts[target].pop(0)
                break
        if reply is None and self.responder is not None:
            reply = self.responder(call)
        if reply is None:
            return httpx.Response(500, json={"error": {"message": "fake: no scripted reply"}})
        if not isinstance(reply, FakeReply):
            reply = FakeReply(raw=reply) if isinstance(reply, str) else FakeReply(output=reply)
        if reply.exc is not None:
            raise reply.exc
        if reply.status != 200:
            if isinstance(reply.body, (dict, list)):
                return httpx.Response(reply.status, json=reply.body, headers=reply.headers)
            return httpx.Response(reply.status, text=str(reply.body or ""), headers=reply.headers)
        text = reply.raw if reply.raw is not None else json.dumps(reply.output)
        if provider == "gemini":
            envelope = {
                "candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                                "finishReason": reply.finish_reason or "STOP"}],
                "usageMetadata": {"promptTokenCount": reply.input_tokens,
                                  "candidatesTokenCount": reply.output_tokens,
                                  "totalTokenCount": reply.input_tokens + reply.output_tokens},
                "modelVersion": model,
            }
        else:
            envelope = {
                "id": "fake-completion", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": reply.finish_reason or "stop"}],
                "usage": {"prompt_tokens": reply.input_tokens, "completion_tokens": reply.output_tokens,
                          "total_tokens": reply.input_tokens + reply.output_tokens},
            }
        return httpx.Response(200, json=envelope, headers=reply.headers)
