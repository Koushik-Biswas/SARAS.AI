"""Per provider:model rate limiting for free-tier AI APIs (spec section 4).

Short-term limits (requests and tokens per minute) are token buckets held in memory
and shared by every router in the process. Daily budgets are counted from the
ai_runs table by the router, so they survive restarts. A cooldown parks a provider
after an auth failure, no connection, a timeout before it answered anything, or a long
Retry-After, so later sheets skip it without a call.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    rpm: int | None = None  # requests per minute
    tpm: int | None = None  # tokens per minute (input + output)
    rpd: int | None = None  # requests per rolling 24 h
    tpd: int | None = None  # tokens per rolling 24 h


class TokenBucket:
    """Classic token bucket refilled continuously to `capacity` per `period` seconds.
    The level may go negative when a call used more tokens than reserved."""

    def __init__(self, capacity: float, period: float, clock: Callable[[], float]) -> None:
        self.capacity = float(capacity)
        self.rate = self.capacity / period
        self.level = self.capacity
        self._clock = clock
        self._t = clock()

    def _refill(self) -> None:
        now = self._clock()
        self.level = min(self.capacity, self.level + (now - self._t) * self.rate)
        self._t = now

    def wait_time(self, n: float) -> float:
        self._refill()
        n = min(n, self.capacity)  # a request larger than the bucket waits for a full bucket
        # The epsilon absorbs float error after sleeping exactly the advertised wait.
        return 0.0 if self.level + 1e-6 >= n else (n - self.level) / self.rate

    def take(self, n: float) -> None:
        self._refill()
        self.level -= n

    def give(self, n: float) -> None:
        self._refill()
        self.level = min(self.capacity, self.level + n)


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, str], tuple[float, TokenBucket]] = {}
        self._cooldown: dict[str, float] = {}

    def _bucket(self, key: str, kind: str, capacity: int) -> TokenBucket:
        entry = self._buckets.get((key, kind))
        if entry is None or entry[0] != capacity:  # (re)created when the configured limit changes
            entry = (capacity, TokenBucket(capacity, 60.0, self.clock))
            self._buckets[(key, kind)] = entry
        return entry[1]

    def reserve(self, key: str, limits: Limits, tokens: int) -> float:
        """Take one request and `tokens` tokens if both are available now and return 0;
        otherwise take nothing and return the seconds until they would be."""
        with self._lock:
            buckets = []
            if limits.rpm:
                buckets.append((self._bucket(key, "rpm", limits.rpm), 1))
            if limits.tpm:
                buckets.append((self._bucket(key, "tpm", limits.tpm), tokens))
            wait = max((b.wait_time(n) for b, n in buckets), default=0.0)
            if wait > 0:
                return wait
            for b, n in buckets:
                b.take(n)
            return 0.0

    def settle(self, key: str, limits: Limits, reserved: int, actual: int | None) -> None:
        """Correct the token bucket once the provider reports the real usage."""
        if not limits.tpm or actual is None:
            return
        with self._lock:
            bucket = self._bucket(key, "tpm", limits.tpm)
            if actual < reserved:
                bucket.give(reserved - actual)
            elif actual > reserved:
                bucket.take(actual - reserved)

    def cooldown_left(self, key: str) -> float:
        with self._lock:
            return max(0.0, self._cooldown.get(key, 0.0) - self.clock())

    def set_cooldown(self, key: str, seconds: float) -> None:
        with self._lock:
            self._cooldown[key] = max(self._cooldown.get(key, 0.0), self.clock() + seconds)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
            self._cooldown.clear()


_default = RateLimiter()


def default_limiter() -> RateLimiter:
    """The process-wide limiter shared by every LLMRouter that is not given its own."""
    return _default
