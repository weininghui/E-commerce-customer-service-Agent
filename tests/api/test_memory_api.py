"""记忆接口：会话状态 / 长期记忆 / 提炼 / 清空短期。"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient
from agent_base.storage.chat_memory import _get_client


def test_session_memory_state(client: TestClient, headers: dict[str, str]):
    sid = f"test-api-{uuid.uuid4().hex[:8]}"
    r = client.get(f"/api/memory/session/{sid}", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert "history_count" in body and "history_chars" in body
    assert "history_budget" in body and body["history_budget"] > 0
    assert "compaction" in body


def test_long_term_memory_get(client: TestClient, headers: dict[str, str]):
    user = f"api_test_user_{uuid.uuid4().hex[:6]}"
    r = client.get(f"/api/memory/{user}", headers=headers)
    assert r.status_code == 200
    assert "memories" in r.json()


def test_clear_short_memory(client: TestClient, headers: dict[str, str]):
    redis = _get_client()
    if redis is None:
        return
    sid = f"test-api-clear-{uuid.uuid4().hex[:8]}"
    redis.set(f"chat:memory:{sid}", '[]', ex=600)
    r = client.delete(f"/api/memory/session/{sid}", headers=headers)
    assert r.status_code == 200
    assert r.json().get("cleared") is True
    redis.delete(f"chat:memory:{sid}")


def test_summarize_empty_session(client: TestClient, headers: dict[str, str]):
    """空会话提炼 → 返回 count=0 且不报错。"""
    sid = f"test-api-empty-{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/api/memory/summarize",
        json={"session_id": sid, "user_id": "api_test"},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json().get("count", 0) == 0
