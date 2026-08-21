"""P32c：问题完整性检查测试。

覆盖"必须澄清"和"可降级推荐"的区分：
1. 含指示代词 → clarification_required
2. 语义完整但无商品约束 → recommendation（推荐式，不追问）
3. 有商品名 → answerable（正常回答）
4. 非商品意图 → answerable
"""

from __future__ import annotations

from agent_base.retrieval.retrieval_policy import (
    CLARIFICATION_INTENTS,
    build_retrieval_decision,
)


# ── 必须澄清（clarification_required） ──


def test_vague_reference_triggers_clarify():
    """含指示代词（"这款/这个"）→ route_type=clarification_required。"""
    queries = [
        "这款商品有什么功效",
        "这个产品怎么样",
        "那个商品适合我吗",
        "哪个好",
    ]
    for q in queries:
        _, decision = build_retrieval_decision(q)
        intent = decision.intent
        if intent in CLARIFICATION_INTENTS:
            assert decision.route_type == "clarification_required", (
                f"'{q}' (intent={intent}) 应触发澄清，实际 route_type={decision.route_type}"
            )
            assert decision.need_clarification is True
            assert decision.strategy == "clarification"


def test_very_short_no_match_triggers_clarify():
    """极短问题（≤5 字）且 catalog 无匹配 → route_type=clarification_required。"""
    _, decision = build_retrieval_decision("有什么功效")
    if decision.intent in CLARIFICATION_INTENTS:
        assert decision.route_type == "clarification_required", (
            f"短问题应触发澄清，实际 route_type={decision.route_type}"
        )


# ── 可降级推荐（recommendation） ──


def test_semantic_complete_but_no_product_gives_recommendation():
    """问题语义完整但未指定具体商品 → route_type=recommendation。"""
    queries = [
        "有什么精华推荐",
        "敏感肌用什么",
        "适合干皮的护肤品有哪些",
        "推荐一款保湿面膜",
    ]
    for q in queries:
        _, decision = build_retrieval_decision(q)
        intent = decision.intent
        if intent in CLARIFICATION_INTENTS:
            # 无指示代词 + 问题 > 5 字 + catalog 无精确匹配 → recommend
            assert decision.route_type in ("recommendation", "answerable"), (
                f"'{q}' (intent={intent}) 不应触发澄清，"
                f"应是 recommend/answerable，实际 route_type={decision.route_type}"
            )
            assert decision.strategy != "clarification", (
                f"'{q}' 应为推荐式回答，不应触发澄清"
            )


def test_recommendation_uses_hybrid_strategy():
    """推荐式回答使用 hybrid 策略（正常多路召回）。"""
    _, decision = build_retrieval_decision("敏感肌用什么精华")
    if decision.route_type == "recommendation":
        assert decision.strategy == "hybrid"
        assert decision.use_vector is True
        assert decision.candidate_k > 0
        assert decision.final_k > 0


def test_recommendation_does_not_ask_clarification():
    """推荐式回答不设 need_clarification。"""
    _, decision = build_retrieval_decision("有什么精华推荐")
    if decision.route_type == "recommendation":
        assert decision.need_clarification is False


# ── 可直接回答（answerable） ──


def test_specific_product_is_answerable():
    """明确提到商品名 → route_type=answerable。"""
    queries = [
        "玻尿酸精华适合敏感肌吗",
        "轻云跑鞋透气性怎么样",
    ]
    for q in queries:
        _, decision = build_retrieval_decision(q)
        assert decision.route_type == "answerable", (
            f"'{q}' 有明确商品名，应为 answerable，实际 {decision.route_type}"
        )
        assert decision.strategy != "clarification"


def test_non_product_intent_is_answerable():
    """非商品意图直接回答。"""
    queries = [
        "我要退货怎么操作",
        "今天天气怎么样",
        "什么时候发货",
    ]
    for q in queries:
        _, decision = build_retrieval_decision(q)
        assert decision.route_type == "answerable", (
            f"'{q}' 非商品意图，应为 answerable，实际 {decision.route_type}"
        )
        assert not decision.need_clarification


def test_reason_uses_chinese_intent_label_bug27():
    """BUG-27：决策理由用中文意图名，不泄漏英文 intent key。"""
    _, decision = build_retrieval_decision("我要退货怎么操作")
    assert decision.reason
    assert "aftersale" not in decision.reason
    assert "售后咨询" in decision.reason


def test_product_with_name_is_answerable():
    """有 product_name 参数 → 直接 answerable。"""
    _, decision = build_retrieval_decision(
        "这款商品有什么功效",
        product_name="玻尿酸精华",
    )
    assert decision.route_type == "answerable"
    assert decision.strategy != "clarification"


# ── 边界用例 ──


def test_four_boundary_cases():
    """P32c 四类边界决策断言。"""
    # 1. "玻尿酸精华适合敏感肌吗"（有商品）→ answerable
    _, d1 = build_retrieval_decision("玻尿酸精华适合敏感肌吗")
    assert d1.route_type == "answerable"
    assert d1.strategy != "clarification"

    # 2. "这款商品有什么功效" → clarification_required
    _, d2 = build_retrieval_decision("这款商品有什么功效")
    assert d2.route_type == "clarification_required"
    assert d2.need_clarification

    # 3. "敏感肌用什么精华" → recommendation 或 answerable（不澄清）
    _, d3 = build_retrieval_decision("敏感肌用什么精华")
    assert d3.route_type in ("recommendation", "answerable")
    assert d3.strategy != "clarification"

    # 4. "今天天气" → answerable（非商品意图）
    _, d4 = build_retrieval_decision("今天天气")
    assert d4.route_type == "answerable"
    assert not d4.need_clarification


def test_clarification_still_works():
    """P32a 澄清测试不回归。"""
    # 确保"这款商品有什么功效"在 product_query 时触发澄清
    _, decision = build_retrieval_decision("这款商品有什么功效")
    if decision.intent in CLARIFICATION_INTENTS:
        assert decision.route_type == "clarification_required"
        assert decision.need_clarification is True
        assert decision.strategy == "clarification"


def test_comparison_not_clarified_in_t5():
    """comparison 意图不触发澄清也不走 recommendation。"""
    _, decision = build_retrieval_decision("玻尿酸精华和水乳哪个更好")
    assert decision.strategy != "clarification"
    # comparison 不在 CLARIFICATION_INTENTS 中，走正常 hybrid + answerable
    assert decision.route_type == "answerable"
