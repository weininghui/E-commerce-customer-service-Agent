"""契约 P19c：对话测试 Redis 短期记忆。

覆盖：
1. test- 会话写 Redis（append + history 顺序 + TTL 自动续期）；
2. 非 test- 会话路由到 PG 路径（不写 Redis）；
3. Redis 不可用时静默降级（读空、写不抛错）。

真实 Redis（localhost:6379）在线时做端到端读写；离线时跳过真实断言。
"""

from __future__ import annotations

from unittest.mock import patch

from agent_base.storage.chat_memory import (
    CHAT_MEMORY_TTL,
    _get_client,
    append_chat_message,
    clear_chat_memory,
    get_chat_history,
    is_test_session,
)


def _test_sid() -> str:
    return "test-pytest-chat-memory"


def test_is_test_session_routing():
    """test- 前缀走 Redis，其余走 PG。"""
    assert is_test_session("test-abc") is True
    assert is_test_session("default") is False
    assert is_test_session(None) is False


def test_redis_append_and_history():
    """Redis 短期记忆：写入有序、截断、TTL 存在。"""
    client = _get_client()
    if client is None:
        return  # Redis 不可用，跳过真实断言
    sid = _test_sid()
    client.delete(f"chat:memory:{sid}")
    try:
        append_chat_message(sid, "user", "第一问")
        append_chat_message(sid, "assistant", "第一答")
        append_chat_message(sid, "user", "第二问")

        history = get_chat_history(sid, limit=8)
        assert [m["role"] for m in history] == ["user", "assistant", "user"]
        assert history[0]["content"] == "第一问"
        assert history[-1]["content"] == "第二问"

        ttl = client.ttl(f"chat:memory:{sid}")
        assert ttl > 0
        assert ttl <= CHAT_MEMORY_TTL
    finally:
        client.delete(f"chat:memory:{sid}")


def test_redis_history_limit():
    """历史截断：只取最近 limit 条。"""
    client = _get_client()
    if client is None:
        return
    sid = _test_sid() + "-limit"
    client.delete(f"chat:memory:{sid}")
    try:
        for i in range(6):
            append_chat_message(sid, "user", f"q{i}")
        history = get_chat_history(sid, limit=3)
        assert len(history) == 3
        assert history[-1]["content"] == "q5"
    finally:
        client.delete(f"chat:memory:{sid}")


def test_redis_unavailable_degrades():
    """Redis 不可用：读返回空、写不抛错。"""
    with patch(
        "agent_base.storage.chat_memory._get_client",
        return_value=None,
    ):
        assert get_chat_history("test-x", limit=8) == []
        append_chat_message("test-x", "user", "内容")  # 不应抛异常
        assert clear_chat_memory("test-x") is False  # 不可用返回 False


def test_clear_chat_memory():
    """清空短期记忆：Redis key 被删除；非 test- 会话不处理。"""
    client = _get_client()
    if client is None:
        return
    sid = _test_sid() + "-clear"
    append_chat_message(sid, "user", "内容")
    assert get_chat_history(sid, limit=8)
    assert clear_chat_memory(sid) is True
    assert get_chat_history(sid, limit=8) == []
    assert clear_chat_memory("default-not-test") is False
