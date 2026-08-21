"""认证接口：登录 / 鉴权 / 退出凭据清理。"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_login_success_returns_token(client: TestClient):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    body = r.json()
    assert body["token"]
    assert body["username"] == "admin"
    assert "role" in body


def test_login_wrong_password(client: TestClient):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code in (400, 401)
    assert "token" not in r.json()


def test_admin_endpoint_rejects_without_token(client: TestClient):
    """未带 X-Admin-Token 访问运营台接口 → 403。"""
    r = client.get("/api/documents")
    assert r.status_code == 403


def test_admin_endpoint_rejects_bad_token(client: TestClient):
    r = client.get("/api/documents", headers={"X-Admin-Token": "not-a-real-token"})
    assert r.status_code == 403


def test_admin_endpoint_accepts_valid_token(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/documents", headers=headers)
    assert r.status_code == 200


def test_admin_endpoint_rejects_non_admin_role(
    client: TestClient,
    agent_headers: dict[str, str],
    user_headers: dict[str, str],
):
    """BUG-7：客服/买家 token 访问管理接口 → 403（角色校验）。"""
    r1 = client.get("/api/documents", headers=agent_headers)
    assert r1.status_code == 403
    r2 = client.get("/api/handoff/stats", headers=user_headers)
    assert r2.status_code == 403


def test_auth_me(client: TestClient, headers: dict[str, str]):
    r = client.get("/api/auth/me", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body.get("username") == "admin"


def test_auth_logout(client: TestClient, headers: dict[str, str]):
    r = client.post("/api/auth/logout", headers=headers)
    assert r.status_code == 200
