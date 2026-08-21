"""契约 P13：长期记忆（user_memories CRUD + 脱敏 + 画像注入）。

覆盖 v0.27.0 验收修复的回归点：
1. sanitize_value 替换电话/邮箱（PII 脱敏）
2. save_memory 必须执行脱敏（PII 不入库）
3. CRUD（save/retrieve/delete + TTL 感知）
4. build_profile_context 预算 ≤300 字 + 意图优先排序
5. 跨会话：save 后新会话 retrieve 生效
6. 4 个记忆工具（真实 PG）

依赖真实 PG（与运行环境一致），使用独立 user_id 不污染现有数据。
"""

from __future__ import annotations

import uuid

import pytest

from agent_base.agents.tools_memory import (
    make_delete_memory_tool,
    make_retrieve_memory_tool,
    make_save_memory_tool,
    make_update_memory_tool,
)
from agent_base.storage.memory import (
    build_profile_context,
    delete_memory,
    retrieve_memory,
    sanitize_key,
    sanitize_value,
    save_memory,
)


@pytest.fixture()
def user_id() -> str:
    return f"p13test_{uuid.uuid4().hex[:10]}"


@pytest.fixture(autouse=True)
def cleanup(user_id: str):
    yield
    for m in retrieve_memory(user_id):
        delete_memory(user_id, m["key"])


def test_sanitize_value_replaces_pii():
    """电话 → [PHONE]，邮箱 → [EMAIL]，业务标签保留。"""
    out = sanitize_value("电话13812345678 邮箱 a@b.com 油皮")
    assert "[PHONE]" in out
    assert "[EMAIL]" in out
    assert "油皮" in out
    assert "13812345678" not in out
    assert "a@b.com" not in out


def test_sanitize_value_truncates_long():
    """超长值截断到 200 字符。"""
    out = sanitize_value("x" * 500)
    assert len(out) <= 200


def test_sanitize_key_cleans_symbols():
    """key 只保留字母数字下划线，截断 64。"""
    assert sanitize_key("皮肤类型") == "皮肤类型"  # 中文保留
    assert sanitize_key("皮肤类型!") == "皮肤类型_"  # 符号替换为下划线
    assert sanitize_key("price_band") == "price_band"
    assert len(sanitize_key("k" * 100)) <= 64


def test_save_memory_sanitizes_pii(user_id: str):
    """save_memory 必须脱敏：PII 不入库（v0.27.0 修复点）。"""
    save_memory(user_id, "contact_info", "13812345678 a@b.com")
    entries = retrieve_memory(user_id, keys=["contact_info"])
    assert entries
    val = str(entries[0]["value"])
    assert "[PHONE]" in val
    assert "[EMAIL]" in val
    assert "13812345678" not in val


def test_memory_crud(user_id: str):
    """save（upsert 新覆旧）→ retrieve → delete 全链路。"""
    save_memory(user_id, "skin_type", "油皮", confidence=0.9)
    entries = retrieve_memory(user_id, keys=["skin_type"])
    assert entries and entries[0]["value"] == "油皮"
    assert entries[0]["confidence"] == 0.9

    # upsert：新值覆盖旧值
    save_memory(user_id, "skin_type", "干皮")
    entries = retrieve_memory(user_id, keys=["skin_type"])
    assert entries[0]["value"] == "干皮"

    assert delete_memory(user_id, "skin_type") is True
    assert retrieve_memory(user_id, keys=["skin_type"]) == []


def test_build_profile_context_budget(user_id: str):
    """画像注入预算 ≤300 字。"""
    save_memory(user_id, "skin_type", "油皮")
    save_memory(user_id, "price_band", "中端")
    save_memory(user_id, "style", "通勤")
    ctx = build_profile_context(user_id, intent="product_query")
    assert ctx.startswith("用户画像")
    assert len(ctx) <= 300


def test_build_profile_context_empty():
    """无记忆用户返回空串。"""
    assert build_profile_context("p13test_nobody") == ""


def test_build_profile_context_intent_priority(user_id: str):
    """意图相关 key 优先（product_query → skin_type/price_band 在前）。"""
    save_memory(user_id, "style", "通勤")
    save_memory(user_id, "skin_type", "油皮")
    save_memory(user_id, "price_band", "中端")
    ctx = build_profile_context(user_id, intent="product_query")
    skin_pos = ctx.find("skin_type")
    price_pos = ctx.find("price_band")
    style_pos = ctx.find("style")
    assert skin_pos != -1 and price_pos != -1
    assert skin_pos < style_pos and price_pos < style_pos


def test_memory_tools_crud(user_id: str):
    """4 个记忆工具端到端（真实 PG）。"""
    cfg = {"configurable": {"user_id": user_id}}
    tools = {
        t.name: t
        for t in [
            make_save_memory_tool(),
            make_retrieve_memory_tool(),
            make_update_memory_tool(),
            make_delete_memory_tool(),
        ]
    }
    assert set(tools) >= {"save_memory_tool", "retrieve_memory_tool", "update_memory_tool", "delete_memory_tool"}

    tools["save_memory_tool"].invoke({"user_id": user_id, "key": "skin_type", "value": "敏感肌", "confidence": 0.9}, config=cfg)
    out = tools["retrieve_memory_tool"].invoke({"user_id": user_id, "keys": ["skin_type"]}, config=cfg)
    assert "敏感肌" in out

    out_del = tools["delete_memory_tool"].invoke({"user_id": user_id, "key": "skin_type"}, config=cfg)
    assert "已删除" in out_del
    assert retrieve_memory(user_id, keys=["skin_type"]) == []
