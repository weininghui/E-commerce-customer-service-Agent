"""契约测试：v0.53 手机号注册/登录（验证码）——校验 / 发码 / 自动注册 / 登录。"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from agent_base.api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_sms_send_rejects_bad_phone():
    with _client() as c:
        r = c.post("/api/auth/sms/send", json={"phone": "123"})
    assert r.status_code == 400
    assert "手机号" in r.json()["detail"]


def test_sms_send_ok_and_code_returned():
    with patch("agent_base.sms.send_code", return_value={"ok": True, "sent": False, "code": "123456", "message": "dev"}):
        with _client() as c:
            r = c.post("/api/auth/sms/send", json={"phone": "13800138000"})
    assert r.status_code == 200
    assert r.json()["code"] == "123456"


def test_sms_login_new_user_auto_registers():
    """首次手机号登录 = 自动注册（is_new=True）并签发 token。"""
    fake_user = {"username": "phone_13800138000", "display_name": "用户8000", "role": "user", "is_new": True}
    with patch("agent_base.sms.verify_code", return_value=True), \
         patch("agent_base.auth.find_or_create_user_by_phone", return_value=fake_user):
        with _client() as c:
            r = c.post("/api/auth/sms/login", json={"phone": "13800138000", "code": "123456"})
    assert r.status_code == 200
    body = r.json()
    assert body["token"] and body["role"] == "user"
    assert body["is_new"] is True
    assert body["username"] == "phone_13800138000"


def test_sms_login_existing_user_not_new():
    fake_user = {"username": "phone_13800138000", "display_name": "用户8000", "role": "user", "is_new": False}
    with patch("agent_base.sms.verify_code", return_value=True), \
         patch("agent_base.auth.find_or_create_user_by_phone", return_value=fake_user):
        with _client() as c:
            r = c.post("/api/auth/sms/login", json={"phone": "13800138000", "code": "123456"})
    assert r.status_code == 200
    assert r.json()["is_new"] is False


def test_sms_login_rejects_wrong_code():
    with patch("agent_base.sms.verify_code", return_value=False):
        with _client() as c:
            r = c.post("/api/auth/sms/login", json={"phone": "13800138000", "code": "000000"})
    assert r.status_code == 401
