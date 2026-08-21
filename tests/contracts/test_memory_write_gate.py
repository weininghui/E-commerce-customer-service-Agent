"""契约 P13-05：长期记忆写门控 + 信任层级 + 归属校验。

覆盖：
1. 写门控三层：低置信丢弃 / 锚点保护 / 冲突延迟 / 高置信覆盖；
2. 信任层级：user_statement > tool_result > agent_inference（推断默认低于门槛）；
3. 归属校验：save_memory_tool 忽略工具参数 user_id，以 config 注入为准；
4. 防抖：maybe_async_extract 轮数不足 / 冷却期 / 空历史跳过。
"""

from __future__ import annotations

from unittest.mock import patch

from agent_base.storage.memory import (
    trust_confidence,
    upsert_memory_guarded,
)


def _cfg() -> dict:
    return {
        "min_confidence": 0.6,
        "anchor_confidence": 0.9,
        "conflict_confidence": 0.7,
        "conflict_boost": 0.1,
        "trust_base_confidence": {
            "user_statement": 0.9,
            "tool_result": 0.8,
            "agent_inference": 0.55,
            "conflict_confirmed": 0.95,
        },
    }


def test_low_confidence_rejected():
    """层 1：新值置信度低于门槛 → 丢弃。"""
    with patch("agent_base.storage.memory.get_memory_config", return_value=_cfg()):
        r = upsert_memory_guarded("u", "skin_type", "干皮", confidence=0.5)
    assert r["written"] is False
    assert r["reason"] == "low_confidence"


def test_anchor_protected():
    """层 2：旧值高置信（≥0.9）且新值不更高 → 不覆盖。"""
    with patch("agent_base.storage.memory.get_memory_config", return_value=_cfg()):
        with patch(
            "agent_base.storage.memory.retrieve_memory",
            return_value=[{"confidence": 0.92, "value": "干皮"}],
        ):
            r = upsert_memory_guarded("u", "skin_type", "干皮", confidence=0.8)
    assert r["written"] is False
    assert r["reason"] == "anchor_protected"


def test_conflict_deferred():
    """层 3：值方向冲突且新值不够高 → 延迟，不静默覆盖。"""
    with patch("agent_base.storage.memory.get_memory_config", return_value=_cfg()):
        with patch(
            "agent_base.storage.memory.retrieve_memory",
            return_value=[{"confidence": 0.8, "value": "干皮"}],
        ):
            with patch("agent_base.storage.memory._record_conflict") as rec:
                r = upsert_memory_guarded("u", "skin_type", "油皮", confidence=0.8)
    assert r["written"] is False
    assert r["reason"] == "conflict_deferred"
    rec.assert_called_once()


def test_confirmed_overwrite():
    """用户改口确认（高置信）→ 覆盖旧值。"""
    with patch("agent_base.storage.memory.get_memory_config", return_value=_cfg()):
        with patch(
            "agent_base.storage.memory.retrieve_memory",
            return_value=[{"confidence": 0.8, "value": "干皮"}],
        ):
            with patch("agent_base.storage.memory.save_memory") as save:
                r = upsert_memory_guarded("u", "skin_type", "油皮", confidence=0.95)
    assert r["written"] is True
    assert r["reason"] == "written"
    save.assert_called_once()


def test_trust_hierarchy():
    """信任层级：用户陈述 0.9 > 工具返回 0.8 > 推断 0.55（低于门槛）。"""
    assert trust_confidence("user_statement") == 0.9
    assert trust_confidence("tool_result") == 0.8
    assert trust_confidence("agent_inference") == 0.55
    assert trust_confidence("conflict_confirmed") == 0.95
    assert trust_confidence("user_statement") > trust_confidence("tool_result")
    assert trust_confidence("tool_result") > trust_confidence("agent_inference")
    # Agent 推断默认低于写入门槛 0.6 → 天然被写门控拦截
    assert trust_confidence("agent_inference") < 0.6


def test_save_tool_ownership_injected():
    """归属校验：工具忽略参数 user_id，以 config 注入为准。"""
    from agent_base.agents.tools_memory import make_save_memory_tool

    with patch("agent_base.storage.memory.save_memory") as save:
        tool = make_save_memory_tool()
        result = tool.invoke(
            {"user_id": "attacker", "key": "skin_type", "value": "油皮", "confidence": 0.9},
            config={"configurable": {"user_id": "real-user"}},
        )
    assert "real-user" in result
    save.assert_called_once()
    args = save.call_args[0]
    assert args[0] == "real-user"  # 写入归属 = config 注入值


def test_save_tool_rejects_missing_ownership():
    """归属缺失时拒绝写入。"""
    from agent_base.agents.tools_memory import make_save_memory_tool

    with patch("agent_base.storage.memory.save_memory") as save:
        tool = make_save_memory_tool()
        result = tool.invoke(
            {"user_id": "attacker", "key": "skin_type", "value": "油皮", "confidence": 0.9},
            config={"configurable": {}},
        )
    assert "拒绝" in result
    save.assert_not_called()


def test_async_extract_cooldown():
    """防抖：轮数不足 / 冷却期跳过，force 绕过。"""
    from agent_base.agents.tools_memory import maybe_async_extract

    with patch("agent_base.storage.memory.get_memory_config", return_value={**_cfg(), "async_extract_enabled": True}):
        with patch("agent_base.agents.tools_memory._last_extract_at", return_value={"history_count": 50}):
            with patch("agent_base.storage.chat_memory.get_chat_history", return_value=[]):
                r = maybe_async_extract("test-1", "u", history_count=5)
                assert r["triggered"] is False
                assert r["skipped_reason"] == "too_few_rounds"

                r = maybe_async_extract("test-1", "u", history_count=55)
                assert r["triggered"] is False
                assert r["skipped_reason"] == "cooldown"

                r = maybe_async_extract("test-1", "u", history_count=0, force=True)
                assert r["triggered"] is False
                assert r["skipped_reason"] == "empty_history"
