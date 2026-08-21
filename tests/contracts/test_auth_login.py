"""契约 P19d：运营台登录（账号密码 → 签名 token → 受保护接口）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from agent_base.api.main import create_app
from agent_base.auth import create_token, hash_password, verify_password, verify_token


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=True)


def test_password_hash_roundtrip():
    """bcrypt 哈希：正确密码通过、错误密码拒绝。"""
    h = hash_password("abc123")
    assert verify_password("abc123", h)
    assert not verify_password("wrong-pass", h)


def test_token_roundtrip():
    """签名 token：签发可解析、伪造拒绝。"""
    token = create_token("admin")
    assert verify_token(token) == "admin"
    assert verify_token("forged-token") is None


def test_token_expired(monkeypatch):
    """过期 token 拒绝（monkeypatch max_age 为负模拟过期）。"""
    import agent_base.auth as auth_mod

    monkeypatch.setattr(auth_mod, "TOKEN_MAX_AGE", -1)
    token = auth_mod.create_token("admin")
    assert auth_mod.verify_token(token) is None


def test_login_success_and_protected_access():
    """登录成功 → 用 Bearer token 访问受保护接口。"""
    client = _client()
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("token")
    assert data.get("username") == "admin"

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {data['token']}"})
    assert me.status_code == 200, me.text
    assert me.json()["username"] == "admin"


def test_login_wrong_password():
    """错误密码返回 401。"""
    client = _client()
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "wrong-pass"})
    assert resp.status_code == 401


def test_protected_endpoint_requires_token():
    """未携带 token 访问受保护接口被拒。"""
    client = _client()
    resp = client.get("/api/auth/me")
    assert resp.status_code in (401, 403)
