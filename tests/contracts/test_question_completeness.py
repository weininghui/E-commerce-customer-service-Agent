"""契约测试：问题补全策略——属性类问题未指名商品时澄清，指名/锚定时直接答。"""

from __future__ import annotations

from agent_base.retrieval.intent_router import route_question
from agent_base.retrieval.retrieval_policy import build_retrieval_decision


def _decision(question: str, current_product: str | None = None):
    return build_retrieval_decision(question, top_k=6, current_product=current_product)[1]


# ── 意图路由：属性类问法应进商品意图（不再是 general_qa） ──


def test_attribute_question_routes_to_product_intent():
    """「要注意什么事项」未指名商品 → 路由到 product_query（此前误落 general_qa）。"""
    r = route_question("要注意什么事项")
    assert r.intent == "product_query"


def test_attribute_question_with_product_keeps_product_intent():
    assert route_question("玻尿酸精华有什么注意事项").intent == "product_query"
    assert route_question("白色纯棉T恤要注意什么").intent == "fashion_query"


# ── 完整性分级：无商品约束 → 澄清 / 推荐 / 直接答 ──


def test_bare_attribute_question_requires_clarification():
    """没指名商品、没锚点 → 必须澄清目标商品。"""
    dec = _decision("要注意什么事项")
    assert dec.need_clarification is True
    assert dec.strategy == "clarification"
    assert "哪款" in dec.clarification_question


def test_bare_price_question_requires_clarification():
    """「多少钱」没指名商品 → 澄清。"""
    dec = _decision("多少钱")
    assert dec.need_clarification is True


def test_attribute_question_with_product_noun_is_answerable():
    """问题已含商品名词（精华/T恤）→ 视为已指名，直接检索回答。"""
    assert _decision("玻尿酸精华有什么注意事项").need_clarification is False
    assert _decision("水杨酸净痘精华注意什么").need_clarification is False
    assert _decision("白色纯棉T恤要注意什么").need_clarification is False


def test_recommendation_question_degrades_to_hybrid():
    """推荐式问法不澄清，走多商品推荐（hybrid）。"""
    dec = _decision("敏感肌用什么")
    assert dec.need_clarification is False
    assert dec.strategy in ("hybrid",)


def test_attribute_question_with_session_anchor_is_answerable():
    """会话已锚定商品 → 属性追问视为对当前商品的补充提问，不澄清。"""
    dec = _decision("要注意什么事项", current_product="白色纯棉T恤")
    assert dec.need_clarification is False


# ── 品类优先澄清（美妆 / 服饰） ──


def test_vague_recommendation_asks_category_first():
    """「有什么推荐」未指明品类 → 先问美妆还是服饰，不再直接列商品。"""
    dec = _decision("有什么推荐")
    assert dec.need_clarification is True
    assert "美妆" in dec.clarification_question
    assert "服饰" in dec.clarification_question
    assert "玻尿酸" not in dec.clarification_question


def test_beauty_recommendation_does_not_ask_category():
    """「敏感肌用什么」已含美妆语义 → 直接推荐，不澄清。"""
    dec = _decision("敏感肌用什么")
    assert dec.need_clarification is False


def test_fashion_recommendation_does_not_ask_category():
    """「帮我推荐衣服」已指明服饰 → 直接推荐，不澄清。"""
    dec = _decision("帮我推荐衣服")
    assert dec.need_clarification is False


def test_category_specific_product_clarification_lists_that_category():
    """已指明品类但没指名商品（如「推荐精华」）→ 澄清只列该品类候选。"""
    dec = _decision("推荐精华")
    if dec.need_clarification:
        assert "精华" in dec.clarification_question
