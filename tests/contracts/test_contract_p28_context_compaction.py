"""契约 P28：上下文压缩（记忆测试台进度条 / 压缩状态数据源）。

覆盖：
1. 历史字符未超预算阈值 → 不压缩（返回 None）；
2. 超阈值 → LLM 摘要旧消息、保留最近 N 条原文、写压缩元数据；
3. 摘要失败 → 静默降级，历史不变；
4. 非 test- 会话 / Redis 不可用 → 不处理；
5. get_history_meta 默认值与持久化。

真实 Redis（localhost:6379）在线时做端到端断言；离线时跳过真实断言。
"""

from __future__ import annotations

import json
from unittest.mock import patch

from agent_base.storage.chat_memory import (
    COMPACT_KEEP_RECENT,
    CONTEXT_BUDGET_CHARS,
    _get_client,
    compact_chat_history,
    get_chat_history,
    get_history_meta,
)


def _test_sid() -> str:
    return "test-pytest-p28-compact"


def _seed_history(sid: str, messages: list[dict]) -> None:
    client = _get_client()
    assert client is not None
    client.set(
        f"chat:memory:{sid}",
        json.dumps(messages, ensure_ascii=False),
        ex=3600,
    )


def _make_messages(n: int = 8, chars: int = 700) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"第{i}轮内容：" + "字" * chars}
        for i in range(n)
    ]


def test_compact_below_threshold_noop():
    """未超预算阈值：不触发压缩，历史与元数据不变。"""
    client = _get_client()
    if client is None:
        return
    sid = _test_sid() + "-below"
    msgs = _make_messages(n=6, chars=50)  # 300 chars << 4800
    _seed_history(sid, msgs)
    try:
        assert compact_chat_history(sid, llm_cfg={"provider": "none"}) is None
        assert get_chat_history(sid, limit=16) == msgs
        assert get_history_meta(sid)["rounds"] == 0
    finally:
        client.delete(f"chat:memory:{sid}")
        client.delete(f"chat:meta:{sid}")


def test_compact_triggers_and_keeps_recent():
    """超阈值：旧消息被摘要替换，保留最近 N 条原文，压缩记录落 meta。"""
    client = _get_client()
    if client is None:
        return
    sid = _test_sid() + "-hit"
    # T9：按常量推导超阈值长度；条数需 > COMPACT_KEEP_RECENT+2（现 10）
    msgs = _make_messages(n=12, chars=int(CONTEXT_BUDGET_CHARS * 0.8 / 12) + 200)
    _seed_history(sid, msgs)
    try:
        with patch(
            "agent_base.storage.chat_memory._summarize_messages",
            return_value="用户咨询了精华与退货政策，偏好敏感肌适用产品。",
        ):
            meta = compact_chat_history(sid, llm_cfg={"provider": "none"})
        assert meta is not None
        assert meta["rounds"] == 1
        assert meta["before_chars"] >= CONTEXT_BUDGET_CHARS * 0.8
        assert meta["after_chars"] < meta["before_chars"]
        assert meta["history"][-1]["before"] == meta["before_chars"]

        history = get_chat_history(sid, limit=16)
        assert history[0]["role"] == "system"
        assert "【对话摘要】" in history[0]["content"]
        assert len(history) == COMPACT_KEEP_RECENT + 1
        assert history[-1] == msgs[-1]

        # 二次压缩轮次递增（超阈值场景）
        big = [
            {"role": "user", "content": "补长内容" + "长" * (int(CONTEXT_BUDGET_CHARS * 0.8 / 12) + 100)}
            for _ in range(12)
        ]
        _seed_history(sid, big)
        with patch(
            "agent_base.storage.chat_memory._summarize_messages",
            return_value="第二轮摘要。",
        ):
            meta2 = compact_chat_history(sid, llm_cfg={"provider": "none"})
        assert meta2["rounds"] == 2
        assert len(meta2["history"]) == 2
    finally:
        client.delete(f"chat:memory:{sid}")
        client.delete(f"chat:meta:{sid}")


def test_compact_summary_failure_degrades():
    """摘要失败：返回 None，历史保持原样。"""
    client = _get_client()
    if client is None:
        return
    sid = _test_sid() + "-fail"
    msgs = _make_messages(n=8, chars=700)
    _seed_history(sid, msgs)
    try:
        with patch(
            "agent_base.storage.chat_memory._summarize_messages",
            return_value="",
        ):
            assert compact_chat_history(sid, llm_cfg={"provider": "none"}) is None
        assert get_chat_history(sid, limit=16) == msgs
        assert get_history_meta(sid)["rounds"] == 0
    finally:
        client.delete(f"chat:memory:{sid}")
        client.delete(f"chat:meta:{sid}")


def test_compact_non_test_session_and_redis_down():
    """非 test- 会话与 Redis 不可用：均不处理、不抛错。"""
    assert compact_chat_history("default-session", llm_cfg={"provider": "none"}) is None
    with patch("agent_base.storage.chat_memory._get_client", return_value=None):
        assert compact_chat_history("test-offline", llm_cfg={"provider": "none"}) is None
        assert get_history_meta("test-offline")["rounds"] == 0
