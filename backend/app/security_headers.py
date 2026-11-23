"""Security response headers (OWASP secure-headers baseline).

Every response: X-Content-Type-Options, X-Frame-Options, Referrer-Policy,
Permissions-Policy, Cross-Origin-Opener-Policy. HTTPS requests in production also get
Strict-Transport-Security (the scheme comes from the proxy's X-Forwarded-Proto when
uvicorn runs with --proxy-headers).

The SPA and the API also get a Content-Security-Policy. Scripts are limited to this
origin plus the exact SHA-256 hashes of the inline scripts in the built index.html (the
no-flash theme script), read once at startup - so no 'unsafe-inline' for scripts.
Styles allow 'unsafe-inline' because React renders style attributes (progress bars,
canvas overlays). The interactive API docs (/api/docs) load Swagger UI from a CDN and
are left without a CSP.

Pure ASGI middleware: it only adds headers, it never buffers or alters bodies (uploads
and downloads stream through unchanged).
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

_INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)
CSP_EXEMPT_PREFIXES = ("/api/docs", "/api/openapi.json")


def inline_script_hashes(index_html: Path) -> list[str]:
    """CSP source expressions ('sha256-...') for the inline scripts of the built SPA."""
    try:
        html = index_html.read_text(encoding="utf-8")
    except OSError:
        return []
    return ["'sha256-" + base64.b64encode(hashlib.sha256(m.group(1).encode("utf-8")).digest()).decode() + "'"
            for m in _INLINE_SCRIPT.finditer(html)]


def build_csp(script_hashes: list[str]) -> str:
    return "; ".join([
        "default-src 'self'",
        "script-src 'self' " + " ".join(script_hashes) if script_hashes else "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "font-src 'self' data:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])


class SecurityHeadersMiddleware:
    def __init__(self, app, frontend_dist: Path, hsts: bool) -> None:  # noqa: ANN001
        self.app = app
        self.hsts = hsts
        self.csp = build_csp(inline_script_hashes(Path(frontend_dist) / "index.html")).encode()
        self.common = [
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"strict-origin-when-cross-origin"),
            (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()"),
            (b"cross-origin-opener-policy", b"same-origin"),
        ]

    async def __call__(self, scope, receive, send):  # noqa: ANN001, ANN204
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        https = scope.get("scheme") == "https"

        async def send_with_headers(message):  # noqa: ANN001, ANN202
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {k.lower() for k, _ in headers}
                extra = [h for h in self.common if h[0] not in present]
                if not path.startswith(CSP_EXEMPT_PREFIXES) and b"content-security-policy" not in present:
                    extra.append((b"content-security-policy", self.csp))
                if self.hsts and https:
                    extra.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                message = {**message, "headers": headers + extra}
            await send(message)

        await self.app(scope, receive, send_with_headers)
