"""契约测试：v0.53 游客升级迁移 + 孤儿清理。"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from agent_base.api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_sms_login_migrates_guest():
    """手机号登录带 guest_uid → 迁移游客会话/记忆 → 返回 migrated=True。"""
    fake_user = {"username": "phone_13800138000", "display_name": "用户8000", "role": "user", "is_new": True}
    with patch("agent_base.sms.verify_code", return_value=True), \
         patch("agent_base.auth.find_or_create_user_by_phone", return_value=fake_user), \
         patch("agent_base.storage.pg.migrate_guest_to_user", return_value={"ok": True, "sessions": 2, "memories": 1}) as mock_migrate:
        with _client() as c:
            r = c.post("/api/auth/sms/login", json={"phone": "13800138000", "code": "123456", "guest_uid": "guest_abc123"})
    assert r.status_code == 200
    assert r.json()["migrated"] is True
    mock_migrate.assert_called_once_with("guest_abc123", "phone_13800138000")


def test_sms_login_without_guest_no_migration():
    fake_user = {"username": "phone_13800138000", "display_name": "用户8000", "role": "user", "is_new": False}
    with patch("agent_base.sms.verify_code", return_value=True), \
         patch("agent_base.auth.find_or_create_user_by_phone", return_value=fake_user), \
         patch("agent_base.storage.pg.migrate_guest_to_user") as mock_migrate:
        with _client() as c:
            r = c.post("/api/auth/sms/login", json={"phone": "13800138000", "code": "123456"})
    assert r.status_code == 200
    assert r.json()["migrated"] is False
    mock_migrate.assert_not_called()


def test_migrate_guest_to_user_rejects_non_guest():
    from agent_base.storage.pg import migrate_guest_to_user

    with patch("agent_base.storage.pg._conn", side_effect=Exception("no db")):
        result = migrate_guest_to_user("phone_123", "phone_456")
    assert result["ok"] is False  # 非 guest_ 前缀直接拒绝（不碰数据库）


def test_cleanup_guests_swallows_db_errors():
    from agent_base.storage.pg import cleanup_guests

    with patch("agent_base.storage.pg._conn", side_effect=Exception("no db")):
        result = cleanup_guests(30)
    assert result["users"] == 0 and result["sessions"] == 0
