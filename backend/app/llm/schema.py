"""JSON-schema helpers shared by the provider adapters.

One pydantic model is the single source of truth for an AI task's output: it is
turned into a strict JSON schema for the request (Groq/OpenAI `strict: true`, Gemini
`responseJsonSchema`) and used again to validate whatever comes back.
"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel

_DROP = {"title", "default", "$defs", "$schema", "examples"}


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for English)."""
    return math.ceil(len(text) / 4) if text else 0


def canonical_json(obj: Any) -> str:  # noqa: ANN401
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _inline(node: Any, defs: dict[str, Any]) -> Any:  # noqa: ANN401
    if isinstance(node, list):
        return [_inline(x, defs) for x in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        target = _inline(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
        extra = {k: _inline(v, defs) for k, v in node.items() if k != "$ref" and k not in _DROP}
        return {**target, **extra}
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROP:
            continue
        if key == "properties":
            out[key] = {name: _inline(sub, defs) for name, sub in value.items()}
        else:
            out[key] = _inline(value, defs)
    if out.get("type") == "object" and "properties" in out:
        # Strict structured output: every property required, nothing else allowed.
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    return out


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A self-contained strict JSON schema for `model` ($refs inlined, titles removed)."""
    raw = model.model_json_schema()
    return _inline(raw, raw.get("$defs", {}))


_OPENAPI_KEYS = {"type", "format", "description", "nullable", "enum", "items", "properties", "required",
                 "minItems", "maxItems", "minimum", "maximum", "anyOf", "propertyOrdering"}


def to_openapi_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a strict JSON schema to Gemini's older `responseSchema` (OpenAPI subset):
    upper-case types, no additionalProperties, explicit property ordering."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _OPENAPI_KEYS:
            continue
        if key == "type" and isinstance(value, str):
            out[key] = value.upper()
        elif key == "properties":
            out[key] = {name: to_openapi_schema(sub) for name, sub in value.items()}
            out["propertyOrdering"] = list(value)
        elif key == "items" and isinstance(value, dict):
            out[key] = to_openapi_schema(value)
        elif key == "anyOf":
            out[key] = [to_openapi_schema(v) for v in value]
        else:
            out[key] = value
    return out
