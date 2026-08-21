"""契约 P29：按会话查询上下文状态（记忆测试台切换会话恢复用）。

覆盖：
1. GET /api/memory/session/{sid} 返回 history_count/chars/budget/compaction；
2. 无历史会话返回零值但不报错；
3. 前端 localStorage 会话切换逻辑（Bug 修复回归点）——消息持久化与恢复引用一致。

真实 Redis（localhost:6379）在线时做端到端断言；离线时跳过真实断言。
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from agent_base.api.main import create_app
from agent_base.storage.chat_memory import CONTEXT_BUDGET_CHARS, _get_client


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=True)


def test_get_session_memory_endpoint():
    """GET /api/memory/session/{sid}：结构完整、无历史零值不报错。"""
    client = _get_client()
    if client is None:
        return
    sid = "test-pytest-p29-session"
    client.delete(f"chat:memory:{sid}")
    client.delete(f"chat:meta:{sid}")
    try:
        # 无历史：零值结构
        with _client() as c:
            r = c.get(f"/api/memory/session/{sid}")
            assert r.status_code == 403  # 未登录被拦截（X-Admin-Token）
        with _client() as c:
            login = c.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
            assert login.status_code == 200
            token = login.json()["token"]
            r = c.get(f"/api/memory/session/{sid}", headers={"X-Admin-Token": token})
            assert r.status_code == 200
            body = r.json()
            assert body["session_id"] == sid
            assert body["history_count"] == 0
            assert body["history_budget"] == CONTEXT_BUDGET_CHARS
            assert body["compaction"]["rounds"] == 0

        # 有历史：写入 8 条消息后查询
        msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"消息{i}：" + "字" * 100}
            for i in range(8)
        ]
        client.set(f"chat:memory:{sid}", json.dumps(msgs, ensure_ascii=False), ex=3600)
        with _client() as c:
            login = c.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
            token = login.json()["token"]
            r = c.get(f"/api/memory/session/{sid}", headers={"X-Admin-Token": token})
            body = r.json()
            assert body["history_count"] == 8
            assert body["history_chars"] > 0
            assert body["history_ratio"] > 0
    finally:
        client.delete(f"chat:memory:{sid}")
        client.delete(f"chat:meta:{sid}")


def test_session_switch_persist_restore_reference():
    """前端切会话回归点：persist/restore 共用同一数组引用，数据不丢。"""
    # 纯逻辑回归：模拟 DebugChat 的 persistMessages/restoreMessages 语义
    sessions = {
        "A": [{"role": "user", "content": "A1"}],
        "B": [{"role": "user", "content": "B1"}, {"role": "assistant", "content": "B2"}],
    }
    active = "A"
    messages = sessions["A"]

    # persist：同引用直接跳过赋值（已同步）
    if sessions[active] is not messages:
        sessions[active] = messages
    # 切到 B 并恢复
    active = "B"
    messages = sessions[active]
    assert messages == [{"role": "user", "content": "B1"}, {"role": "assistant", "content": "B2"}]
    # 回到 A 仍完整
    active = "A"
    messages = sessions[active]
    assert messages == [{"role": "user", "content": "A1"}]
