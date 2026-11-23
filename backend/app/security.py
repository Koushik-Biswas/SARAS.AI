"""Password hashing (Argon2), JWT access/refresh tokens, login lockout and the
current-user dependency."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, Request
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .errors import ApiError
from .models import RefreshToken, User

_hasher = PasswordHasher()
ALGORITHM = "HS256"


def aware(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; treat them as UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def _encode(payload: dict) -> str:
    return jwt.encode(payload, get_settings().jwt_secret, algorithm=ALGORITHM)


def create_access_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=get_settings().access_token_minutes)
    return _encode({"sub": user.id, "role": user.role, "type": "access", "iat": now, "exp": exp})


def create_refresh_token(db: Session, user: User) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(days=get_settings().refresh_token_days)
    jti = str(uuid.uuid4())
    db.add(RefreshToken(id=jti, user_id=user.id, expires_at=exp))
    return _encode({"sub": user.id, "type": "refresh", "jti": jti, "iat": now, "exp": exp})


def decode_token(token: str, expected_type: str) -> dict:
    try:
        payload = jwt.decode(token, get_settings().jwt_secret, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise ApiError(401, "unauthorized", "Token expired") from exc
    except jwt.PyJWTError as exc:
        raise ApiError(401, "unauthorized", "Invalid token") from exc
    if payload.get("type") != expected_type:
        raise ApiError(401, "unauthorized", "Wrong token type")
    return payload


def _bearer(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ApiError(401, "unauthorized", "Missing bearer token")
    return token


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    payload = decode_token(_bearer(request), "access")
    user = db.get(User, payload["sub"])
    if user is None:
        raise ApiError(401, "unauthorized", "User no longer exists")
    request.state.user_id = user.id
    return user


class LoginLimiter:
    """Locks an email for N minutes after M consecutive failures (screen spec 17.1).

    check() also reserves the attempt: attempts still being verified count against the
    limit, so a burst of parallel guesses cannot all pass check() before the first
    failure is recorded. Every check() must be followed by fail(), succeed() or release().
    Old entries are swept, so a flood of distinct emails cannot grow memory without bound."""

    MAX_TRACKED = 10_000

    def __init__(self) -> None:
        self._failures: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}
        self._in_flight: dict[str, int] = {}
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def _sweep(self, now: float, window: float) -> None:
        if now - self._last_sweep < 60 and len(self._failures) + len(self._locked_until) <= self.MAX_TRACKED:
            return
        self._last_sweep = now
        for email in [e for e, ts in self._failures.items() if not ts or now - ts[-1] >= window]:
            del self._failures[email]
        for email in [e for e, until in self._locked_until.items() if until <= now]:
            del self._locked_until[email]
        for table in (self._failures, self._locked_until):  # still too many: forget the oldest
            while len(table) > self.MAX_TRACKED:
                del table[next(iter(table))]

    def _release(self, email: str) -> None:
        n = self._in_flight.get(email, 0) - 1
        if n > 0:
            self._in_flight[email] = n
        else:
            self._in_flight.pop(email, None)

    def check(self, email: str) -> None:
        s = get_settings()
        window = s.login_lock_minutes * 60
        with self._lock:
            now = time.time()
            self._sweep(now, window)
            until = self._locked_until.get(email, 0)
            if until > now:
                raise ApiError(
                    429, "rate_limited", "Too many failed attempts. Try again in a few minutes.",
                    {"retry_after_seconds": int(until - now) + 1},
                )
            recent = [t for t in self._failures.get(email, []) if now - t < window]
            busy = self._in_flight.get(email, 0)
            if len(recent) + busy >= s.login_max_failures:
                raise ApiError(429, "rate_limited", "Too many sign-in attempts at once. Try again in a moment.",
                               {"retry_after_seconds": 2})
            self._in_flight[email] = busy + 1

    def release(self, email: str) -> None:
        """Give back a reserved attempt that was neither a failure nor a success (server error)."""
        with self._lock:
            self._release(email)

    def fail(self, email: str) -> None:
        s = get_settings()
        with self._lock:
            self._release(email)
            now = time.time()
            window = [t for t in self._failures.get(email, []) if now - t < s.login_lock_minutes * 60] + [now]
            self._failures[email] = window
            if len(window) >= s.login_max_failures:
                self._locked_until[email] = now + s.login_lock_minutes * 60
                self._failures.pop(email, None)

    def succeed(self, email: str) -> None:
        with self._lock:
            self._release(email)
            self._failures.pop(email, None)
            self._locked_until.pop(email, None)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()
            self._locked_until.clear()
            self._in_flight.clear()


login_limiter = LoginLimiter()
