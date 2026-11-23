"""Authentication (PRD 6.2 / 20.2): generic login errors, bearer tokens on every
protected endpoint, Argon2 password hashes, lockout, refresh rotation, logout."""

from __future__ import annotations

from conftest import API, PASSWORD, assert_error, register


def test_register_login_and_me(client):
    api = register(client, "Alice@Example.com", name="Alice")
    assert api.user["email"] == "alice@example.com"  # stored lower-case
    assert api.user["role"] == "teacher"
    assert api.tokens["token_type"] == "bearer" and api.tokens["expires_in"] > 0

    r = client.post(f"{API}/auth/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert r.status_code == 200, r.text
    me = client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"})
    assert me.status_code == 200 and me.json()["email"] == "alice@example.com"


def test_duplicate_registration_conflicts(client):
    register(client, "dup@example.com")
    r = client.post(f"{API}/auth/register", json={"name": "X", "email": "DUP@example.com", "password": PASSWORD})
    assert_error(r, 409, "conflict")


def test_short_password_rejected_with_field_details(client):
    r = client.post(f"{API}/auth/register", json={"name": "X", "email": "x@example.com", "password": "short"})
    err = assert_error(r, 400, "validation_error")
    assert any(d["field"] == "password" for d in err["details"])


def test_wrong_credentials_always_rejected_with_one_generic_error(client):
    register(client, "bob@example.com")
    wrong_pw = client.post(f"{API}/auth/login", json={"email": "bob@example.com", "password": "not-the-password"})
    unknown = client.post(f"{API}/auth/login", json={"email": "nobody@example.com", "password": PASSWORD})
    e1 = assert_error(wrong_pw, 401, "unauthorized")
    e2 = assert_error(unknown, 401, "unauthorized")
    assert e1["message"] == e2["message"]  # no account enumeration
    assert "access_token" not in wrong_pw.text


PROTECTED = [
    ("GET", "/auth/me"), ("GET", "/exams"), ("POST", "/exams"), ("GET", "/exams/x"), ("PATCH", "/exams/x"),
    ("DELETE", "/exams/x"), ("GET", "/exams/x/questions"), ("PUT", "/exams/x/questions"),
    ("POST", "/exams/x/regrade"), ("GET", "/exams/x/results"), ("GET", "/exams/x/export"),
    ("GET", "/exams/x/analytics"), ("POST", "/exams/x/submissions"), ("GET", "/exams/x/submissions"),
    ("POST", "/exams/x/approve-all"), ("GET", "/submissions/x"), ("GET", "/submissions/x/status"),
    ("POST", "/submissions/x/retry"), ("DELETE", "/submissions/x"), ("GET", "/submissions/x/answers"),
    ("GET", "/answers/x"), ("PATCH", "/answers/x/review"), ("POST", "/answers/x/flag"),
    ("POST", "/submissions/x/approve"), ("POST", "/submissions/x/reopen"), ("GET", "/submissions/x/pages"),
    ("GET", "/submissions/x/pages/1/image"), ("GET", "/submissions/x/regions"),
    ("POST", "/submissions/x/regions"), ("PATCH", "/regions/x"), ("POST", "/submissions/x/regions/undo"),
    ("GET", "/audit-log"), ("GET", "/capabilities"),
]


def test_protected_endpoints_require_a_bearer_token(client):
    # One app for all endpoints (a fresh app per case is slow on the 4 GB laptop).
    for method, path in PROTECTED:
        r = client.request(method, API + path)
        assert r.status_code == 401, f"{method} {path} -> {r.status_code}"
        assert_error(r, 401, "unauthorized")


def test_invalid_and_wrong_type_tokens_rejected(client):
    api = register(client, "carol@example.com")
    for header in ("Bearer not-a-jwt", "Basic abc", f"Bearer {api.tokens['refresh_token']}", "Bearer"):
        assert_error(client.get(f"{API}/exams", headers={"Authorization": header}), 401, "unauthorized")


def test_public_endpoints_need_no_token(client):
    assert client.get(f"{API}/health").status_code == 200
    assert client.get("/health").status_code == 200


def test_password_is_stored_as_argon2_hash(client):
    from app.db import SessionLocal
    from app.models import User

    register(client, "dave@example.com")
    with SessionLocal() as db:
        user = db.query(User).filter_by(email="dave@example.com").one()
    assert user.password_hash.startswith("$argon2")
    assert PASSWORD not in user.password_hash


def test_lockout_after_five_failures(client):
    register(client, "eve@example.com")
    for _ in range(5):
        r = client.post(f"{API}/auth/login", json={"email": "eve@example.com", "password": "wrong-password"})
        assert_error(r, 401, "unauthorized")
    # Locked: even the right password is refused for a few minutes.
    locked = client.post(f"{API}/auth/login", json={"email": "eve@example.com", "password": PASSWORD})
    err = assert_error(locked, 429, "rate_limited")
    assert err["details"]["retry_after_seconds"] > 0
    # Another account is unaffected.
    register(client, "frank@example.com")
    ok = client.post(f"{API}/auth/login", json={"email": "frank@example.com", "password": PASSWORD})
    assert ok.status_code == 200


def test_refresh_rotates_and_old_token_is_rejected(client):
    api = register(client, "grace@example.com")
    old = api.tokens["refresh_token"]
    r = client.post(f"{API}/auth/refresh", json={"refresh_token": old})
    assert r.status_code == 200, r.text
    new = r.json()
    assert new["refresh_token"] != old
    assert client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {new['access_token']}"}).status_code == 200
    # The used refresh token is single-use.
    assert_error(client.post(f"{API}/auth/refresh", json={"refresh_token": old}), 401, "unauthorized")
    # The rotated one still works (once).
    assert client.post(f"{API}/auth/refresh", json={"refresh_token": new["refresh_token"]}).status_code == 200


def test_access_token_cannot_be_used_as_refresh_token(client):
    api = register(client, "heidi@example.com")
    r = client.post(f"{API}/auth/refresh", json={"refresh_token": api.tokens["access_token"]})
    assert_error(r, 401, "unauthorized")


def test_logout_revokes_refresh_token(client):
    api = register(client, "ivan@example.com")
    r = client.post(f"{API}/auth/logout", json={"refresh_token": api.tokens["refresh_token"]})
    assert r.status_code == 204
    again = client.post(f"{API}/auth/refresh", json={"refresh_token": api.tokens["refresh_token"]})
    assert_error(again, 401, "unauthorized")
