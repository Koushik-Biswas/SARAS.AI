"""In-process sliding-window rate limiter: 100 requests/minute per caller and
10 uploads/minute (PRD 16.2). Fine for a single API process; the extended
profile would move this to Redis.

A caller is the verified user of a valid access token, otherwise the client IP:
an arbitrary or forged Authorization header never earns a fresh allowance. The key
table is swept and capped, so a flood of distinct callers cannot grow memory."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .config import get_settings
from .errors import ApiError


class _Window:
    MAX_KEYS = 20_000

    def __init__(self, clock=time.monotonic) -> None:  # noqa: ANN001
        self.clock = clock
        self.hits: OrderedDict[str, deque[float]] = OrderedDict()
        self.lock = threading.Lock()
        self._last_sweep = clock()

    def _sweep(self, now: float, period: float) -> None:
        for key in [k for k, q in self.hits.items() if not q or now - q[-1] > period]:
            del self.hits[key]
        self._last_sweep = now

    def allow(self, key: str, limit: int, period: float = 60.0) -> tuple[bool, int]:
        now = self.clock()
        with self.lock:
            if now - self._last_sweep > period:  # drop callers idle for a whole window
                self._sweep(now, period)
            q = self.hits.get(key)
            if q is None:
                q = self.hits[key] = deque()
            else:
                self.hits.move_to_end(key)
            while q and now - q[0] > period:
                q.popleft()
            if len(q) >= limit:
                return False, int(period - (now - q[0])) + 1
            q.append(now)
            while len(self.hits) > self.MAX_KEYS:  # still too many callers: forget the least recent
                self.hits.popitem(last=False)
            return True, 0


def caller_identity(request: Request) -> str:
    from .security import decode_token  # imported lazily: security pulls in the database layer

    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token:
        try:
            payload = decode_token(token.strip(), "access")
        except ApiError:
            payload = None
        if payload and payload.get("sub"):
            return f"user:{payload['sub']}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:  # noqa: ANN001
        super().__init__(app)
        self.window = _Window()

    async def dispatch(self, request: Request, call_next):
        s = get_settings()
        path = request.url.path
        if not s.rate_limit_enabled or not path.startswith("/api/") or request.method == "OPTIONS":
            return await call_next(request)

        ident = caller_identity(request)
        checks = [(f"all:{ident}", s.rate_limit_per_minute)]
        if request.method == "POST" and path.endswith("/submissions"):
            checks.append((f"upload:{ident}", s.upload_rate_per_minute))

        for key, limit in checks:
            ok, retry = self.window.allow(key, limit)
            if not ok:
                return JSONResponse(
                    {"error": {"code": "rate_limited", "message": "Too many requests", "details": {"retry_after_seconds": retry}}},
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                )
        return await call_next(request)
