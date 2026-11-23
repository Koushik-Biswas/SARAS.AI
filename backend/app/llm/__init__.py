"""Optional AI (LLM) layer — PRD 13.5, spec section 4.

Never in the critical path: every call can fail with LLMUnavailable and the caller
then uses the offline checker. Nothing is sent without settings.ai_consent (and
settings.gemini_consent for Gemini); prompts must be anonymised (see redact.py).
"""

from .limiter import Limits, RateLimiter, default_limiter
from .providers import GeminiProvider, ImageInput, OpenAICompatProvider, ProviderError
from .redact import redact
from .router import LLMResult, LLMRouter, LLMUnavailable, parse_provider_spec

__all__ = [
    "GeminiProvider", "ImageInput", "LLMResult", "LLMRouter", "LLMUnavailable", "Limits",
    "OpenAICompatProvider", "ProviderError", "RateLimiter", "default_limiter", "parse_provider_spec",
    "redact",
]
