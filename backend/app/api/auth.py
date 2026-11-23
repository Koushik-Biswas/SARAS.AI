from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..errors import ApiError
from ..models import RefreshToken, User
from ..schemas import LoginIn, RefreshIn, RegisterIn, TokenOut, UserOut
from ..security import (
    aware,
    create_access_token,
    create_refresh_token,
    current_user,
    decode_token,
    hash_password,
    login_limiter,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger(__name__)

# A used refresh token presented again means it was copied: every session of that user
# is ended. Within a few seconds of its rotation, though, it is far more likely the same
# client sending it twice (two tabs restoring at once), which only gets a 401.
REUSE_GRACE_SECONDS = 10.0
_rotated: dict[str, float] = {}  # jti -> monotonic time it was rotated (this process)
_rotated_lock = threading.Lock()


def _remember_rotation(jti: str) -> None:
    now = time.monotonic()
    with _rotated_lock:
        for old in [k for k, t in _rotated.items() if now - t > REUSE_GRACE_SECONDS]:
            del _rotated[old]
        _rotated[jti] = now


def _rotated_recently(jti: str) -> bool:
    with _rotated_lock:
        t = _rotated.get(jti)
    return t is not None and time.monotonic() - t <= REUSE_GRACE_SECONDS


def _tokens(db: Session, user: User) -> TokenOut:
    refresh = create_refresh_token(db, user)
    db.commit()
    return TokenOut(
        access_token=create_access_token(user), refresh_token=refresh,
        expires_in=get_settings().access_token_minutes * 60, user=UserOut.model_validate(user),
    )


@router.post("/register", response_model=TokenOut, status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)) -> TokenOut:
    email = body.email.lower()
    if db.scalar(select(User).where(func.lower(User.email) == email)):
        raise ApiError(409, "conflict", "An account with this email already exists")
    user = User(name=body.name.strip(), email=email, password_hash=hash_password(body.password), role="teacher")
    db.add(user)
    db.flush()
    return _tokens(db, user)


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)) -> TokenOut:
    email = body.email.lower()
    login_limiter.check(email)  # also reserves this attempt, so parallel guesses count at once
    try:
        user = db.scalar(select(User).where(func.lower(User.email) == email))
        ok = user is not None and verify_password(body.password, user.password_hash)
    except Exception:
        login_limiter.release(email)
        raise
    if not ok:
        login_limiter.fail(email)
        raise ApiError(401, "unauthorized", "Incorrect email or password")  # one generic error
    login_limiter.succeed(email)
    return _tokens(db, user)


@router.post("/refresh", response_model=TokenOut)
def refresh(body: RefreshIn, db: Session = Depends(get_db)) -> TokenOut:
    payload = decode_token(body.refresh_token, "refresh")
    jti = str(payload.get("jti", ""))
    row = db.get(RefreshToken, jti)
    if row is None or row.user_id != payload.get("sub") or aware(row.expires_at) < datetime.now(timezone.utc):
        raise ApiError(401, "unauthorized", "Refresh token is no longer valid")
    user = db.get(User, payload["sub"])
    if user is None:
        raise ApiError(401, "unauthorized", "User no longer exists")
    # Rotate: each refresh token works once. The claim is one conditional UPDATE, so of
    # two concurrent requests with the same token exactly one wins.
    claimed = db.execute(
        update(RefreshToken).where(RefreshToken.id == jti, RefreshToken.revoked.is_(False)).values(revoked=True)
    ).rowcount == 1
    if not claimed:
        db.rollback()
        if not _rotated_recently(jti):
            # Reuse of a spent token: assume theft and end every session of this user.
            db.execute(update(RefreshToken).where(RefreshToken.user_id == user.id, RefreshToken.revoked.is_(False))
                       .values(revoked=True))
            db.commit()
            log.warning("Refresh token reuse for user %s: all of their sessions were revoked", user.id)
        raise ApiError(401, "unauthorized", "Refresh token is no longer valid")
    _remember_rotation(jti)
    return _tokens(db, user)


@router.post("/logout", status_code=204)
def logout(body: RefreshIn, db: Session = Depends(get_db)) -> None:
    try:
        payload = decode_token(body.refresh_token, "refresh")
    except ApiError:
        return None
    row = db.get(RefreshToken, payload.get("jti", ""))
    if row:
        row.revoked = True
        db.commit()
    return None


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> User:
    return user
