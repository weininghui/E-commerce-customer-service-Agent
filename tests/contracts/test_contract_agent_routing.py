"""契约 P2-01（电商版）：should_use_agent 触发路由三条件。"""

from agent_base.agents.routing import (
    _looks_like_comparison,
    _looks_like_multi_entity,
    should_use_agent,
)


def test_low_confidence_triggers_agent():
    assert should_use_agent({"question": "有什么推荐", "route": {"confidence": 0.3}}) == "agent"


def test_high_confidence_fixed_intent_returns_retrieve():
    assert should_use_agent({
        "question": "这件白T恤怎么洗？",
        "route": {"confidence": 1.0, "intent": "product_query", "sections": ["商品参数"]},
    }) == "retrieve"


def test_comparison_triggers_agent():
    for q in [
        "A和B哪个好？",
        "玻尿酸精华和烟酰胺精华有什么区别",
        "这两件衣服对比一下",
        "防晒衣和防晒霜怎么选",
    ]:
        assert should_use_agent({"question": q, "route": {"confidence": 1.0}}) == "agent"


def test_multi_entity_triggers_agent():
    # ≥2 个商品/类目信号 → 走 agent
    assert should_use_agent({
        "question": "这两件衣服分别适合什么场合？",
        "route": {"confidence": 1.0},
    }) == "agent"


def test_normal_question_returns_retrieve():
    assert should_use_agent({
        "question": "这款玻尿酸精华适合敏感肌吗？",
        "route": {"confidence": 1.0, "intent": "product_query", "sections": ["商品参数"]},
    }) == "retrieve"


def test_comparison_detection():
    assert _looks_like_comparison("A和B哪个好？")
    assert _looks_like_comparison("和XX比有什么优势")
    assert _looks_like_comparison("这件和那件有什么区别")
    assert not _looks_like_comparison("这款面霜适合干皮吗？")


def test_multi_entity_detection():
    # 问题中出现两个商品/类目信号 → 触发
    assert _looks_like_multi_entity("这两件衣服分别适合什么场合")
    # 只有单个商品 → 不触发
    assert not _looks_like_multi_entity("这款面霜适合干皮吗")
    # "两件/两款"等信号词 → 触发
    assert _looks_like_multi_entity("帮我对比这两款精华")
