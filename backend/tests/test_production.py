"""Production hardening: security headers on every response, a script-hash CSP for the
SPA, HSTS only on HTTPS in prod, the API docs left usable, and online backups."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pytest

from app.security_headers import build_csp, inline_script_hashes

SPA = """<!doctype html><html><head>
<script>
  (function () { document.documentElement.setAttribute("data-theme", "dark"); })();
</script>
<script type="module" src="/assets/index.js"></script>
</head><body><div id="root"></div></body></html>"""


@pytest.fixture
def spa_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(SPA, encoding="utf-8")
    (dist / "assets" / "index.js").write_text("console.log(1)", encoding="utf-8")
    return dist


def test_csp_allows_exactly_the_inline_theme_script(spa_dist: Path) -> None:
    hashes = inline_script_hashes(spa_dist / "index.html")
    assert len(hashes) == 1 and hashes[0].startswith("'sha256-")  # the module script (src=) is not hashed
    csp = build_csp(hashes)
    assert f"script-src 'self' {hashes[0]}" in csp and "unsafe-inline' ;" not in csp
    assert "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
    assert "'unsafe-inline'" not in csp.split("script-src", 1)[1].split(";", 1)[0]


def test_every_response_carries_security_headers(client, settings) -> None:  # noqa: ANN001
    for path in ("/health", "/api/v1/health", "/api/v1/exams", "/no-such-page"):
        r = client.get(path)
        h = r.headers
        assert h["x-content-type-options"] == "nosniff", path
        assert h["x-frame-options"] == "DENY" and h["referrer-policy"] == "strict-origin-when-cross-origin"
        assert "camera=()" in h["permissions-policy"] and "content-security-policy" in h
        assert "strict-transport-security" not in h  # plain HTTP / not prod: no HSTS
    docs = client.get("/api/docs")
    assert docs.status_code == 200 and "content-security-policy" not in docs.headers  # Swagger UI uses a CDN
    assert docs.headers["x-frame-options"] == "DENY"


def test_hsts_only_over_https_in_production(spa_dist: Path) -> None:
    import asyncio

    from app.security_headers import SecurityHeadersMiddleware

    async def app(scope, receive, send):  # noqa: ANN001, ANN202
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    def headers_for(scheme: str, hsts: bool) -> dict:
        sent: list[dict] = []

        async def send(msg):  # noqa: ANN001, ANN202
            sent.append(msg)

        mw = SecurityHeadersMiddleware(app, spa_dist, hsts=hsts)
        asyncio.run(mw({"type": "http", "path": "/", "scheme": scheme}, None, send))
        return {k.decode(): v.decode() for k, v in sent[0]["headers"]}

    assert "strict-transport-security" in headers_for("https", hsts=True)
    assert "strict-transport-security" not in headers_for("http", hsts=True)
    assert "strict-transport-security" not in headers_for("https", hsts=False)
    assert "'sha256-" in headers_for("http", hsts=False)["content-security-policy"]


def test_backup_is_a_consistent_copy_of_database_and_uploads(client, teacher, exam, settings, tmp_path) -> None:  # noqa: ANN001
    from scripts import backup

    upload = settings.storage_dir / "submissions" / "x" / "original.pdf"
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_bytes(b"%PDF-1.4 test")
    out = tmp_path / "backups"
    first = backup.backup(out)
    with zipfile.ZipFile(first) as zf:
        names = zf.namelist()
        assert "saras.db" in names and "storage/submissions/x/original.pdf" in names
        zf.extract("saras.db", tmp_path / "restore")
    con = sqlite3.connect(tmp_path / "restore" / "saras.db")
    try:
        assert con.execute("select count(*) from exams").fetchone()[0] == 1  # the exam made above
    finally:
        con.close()
    import time

    for _ in range(2):
        time.sleep(1.1)  # distinct second-resolution names
        backup.backup(out, keep=2)
    assert len(list(out.glob("saras-*.zip"))) == 2
