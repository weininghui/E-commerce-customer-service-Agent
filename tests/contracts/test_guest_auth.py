"""契约测试：v0.53 游客直接咨询（免登录一键进聊天）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from agent_base.api.main import create_app


def test_guest_issues_token_and_user():
    with TestClient(create_app()) as c:
        r = c.post("/api/auth/guest", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["token"]
    assert body["role"] == "user"
    assert body["guest"] is True
    assert body["username"].startswith("guest_")
