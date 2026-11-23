"""Request-level guard for answer-sheet uploads (PRD 6.2, 16.1).

The multipart parser spools every file part to a temporary file before the upload
handler runs, so per-file checks in the handler come too late to protect the disk.
This middleware runs first, for POST .../submissions only:

1. A request without a valid access token is refused before a byte of its body is read.
2. A declared Content-Length above the request budget (50 files x the per-file limit,
   plus form overhead) is refused at once with 413.
3. Otherwise the body is counted as it streams in (also chunked bodies with no
   Content-Length) and the request is cut off with 413 as soon as it exceeds the budget.
"""

from __future__ import annotations

import json

from .config import get_settings
from .errors import ApiError

MAX_FILES_PER_REQUEST = 50
FORM_OVERHEAD = 1024 * 1024  # multipart boundaries, part headers and the student_refs fields


def upload_budget(max_upload_bytes: int) -> int:
    return MAX_FILES_PER_REQUEST * max_upload_bytes + FORM_OVERHEAD


def _is_upload(scope: dict) -> bool:
    path = scope.get("path", "")
    return (scope["type"] == "http" and scope.get("method") == "POST" and path.startswith("/api/")
            and path.rstrip("/").endswith("/submissions"))


def _valid_token(header: str) -> bool:
    from .security import decode_token  # imported lazily: security pulls in the database layer

    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return False
    try:
        decode_token(token.strip(), "access")
    except ApiError:
        return False
    return True


async def _reject(send, status: int, code: str, message: str) -> None:  # noqa: ANN001
    body = json.dumps({"error": {"code": code, "message": message, "details": None}}).encode()
    await send({"type": "http.response.start", "status": status, "headers": [
        (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
        (b"connection", b"close"),
    ]})
    await send({"type": "http.response.body", "body": body})


class _TooLarge(Exception):
    pass


class UploadGuardMiddleware:
    def __init__(self, app) -> None:  # noqa: ANN001
        self.app = app

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if not _is_upload(scope):
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        if not _valid_token(headers.get(b"authorization", b"").decode("latin-1")):
            await _reject(send, 401, "unauthorized", "Missing or invalid bearer token")
            return

        limit = upload_budget(get_settings().max_upload_bytes)
        too_large = f"The upload is larger than {limit // (1024 * 1024)} MB in total; send fewer files at a time"
        try:
            declared = int(headers.get(b"content-length", b"-1"))
        except ValueError:
            declared = -1
        if declared > limit:
            await _reject(send, 413, "file_too_large", too_large)
            return

        received = 0
        exceeded = False
        started = False

        async def guarded_receive() -> dict:
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _TooLarge()
            return message

        async def guarded_send(message: dict) -> None:
            nonlocal started
            if exceeded:
                return  # the app's own error for the aborted body is replaced by our 413
            started = True
            await send(message)

        try:
            await self.app(scope, guarded_receive, guarded_send)
        except _TooLarge:
            pass
        except Exception:
            if not exceeded:
                raise
        if exceeded and not started:
            await _reject(send, 413, "file_too_large", too_large)
