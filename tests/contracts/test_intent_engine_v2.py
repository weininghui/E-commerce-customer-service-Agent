"""契约测试：会话级意图识别 v2。

覆盖：
- QueryRoute 扩展字段（sub_intent / buying_signal / objection_type / missing_info）；
- 子意图规则检测（闲聊/砍价/退换/物流/过敏/推荐）；
- 向量语义兜底（bge-m3 复用，测试用确定性桩验证选择与阈值）；
- LLM 多字段结构化输出与失败回退。
"""

from __future__ import annotations

from agent_base.retrieval.intent_router import (
    IntentRule,
    QueryRoute,
    _try_semantic_layer,
    route_question,
)
from agent_base.retrieval.llm_intent_classifier import route_question_with_llm


class _FeatureEmbeddings:
    """确定性向量桩：按关键词给特征向量，用于语义层单测（无外部服务依赖）。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        if "退" in text:
            return [1.0, 0.0, 0.0]
        if "搭" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def _fallback_route() -> QueryRoute:
    return QueryRoute(
        intent="general_qa",
        sections=[],
        metadata_filter={},
        matched_keywords=[],
        confidence=0.0,
        scores={},
        source="rule",
        fallback_reason="no_keyword_match",
    )


# ── 扩展字段 ──


def test_route_has_extended_fields_with_defaults():
    r = route_question("这款玻尿酸精华适合油皮吗")
    assert hasattr(r, "sub_intent")
    assert hasattr(r, "buying_signal")
    assert hasattr(r, "objection_type")
    assert hasattr(r, "missing_info")
    assert isinstance(r.missing_info, list)
    assert r.buying_signal in {"buying", "objection", "normal"}
    assert r.objection_type in {"price", "hesitant", "risk", "none"}


def test_route_populates_buying_signal():
    r = route_question("这个精华适合我吗，有点想买")
    assert r.buying_signal == "buying"


def test_route_populates_price_objection():
    r = route_question("99还是有点贵")
    assert r.buying_signal == "objection"
    assert r.objection_type == "price"


# ── 子意图 ──


def test_sub_intent_chat():
    assert route_question("你好呀").sub_intent == "chat"


def test_sub_intent_price_negotiation():
    assert route_question("能便宜点吗").sub_intent == "price_negotiation"


def test_sub_intent_return_exchange():
    r = route_question("拆封了还能退货吗")
    assert r.intent == "aftersale"
    assert r.sub_intent == "return_exchange"


def test_sub_intent_logistics():
    assert route_question("什么时候发货").sub_intent == "logistics"


def test_sub_intent_allergy():
    assert route_question("敏感肌能用吗").sub_intent == "allergy"


def test_sub_intent_recommend_request():
    assert route_question("帮我推荐一款精华").sub_intent == "recommend_request"
    assert route_question("还有什么适合我的精华吗").sub_intent == "recommend_request"


def test_sub_intent_media_request():
    for q in (
        "有图片吗",
        "看看实物",
        "有视频吗",
        "上身效果怎么样",
        "看看这件商品的图片",
        "发个视频看看",
    ):
        assert route_question(q).sub_intent == "media_request", q


def test_sub_intent_not_media_for_non_visual():
    for q in ("看看有什么", "先看看", "帮我推荐衣服"):
        assert route_question(q).sub_intent != "media_request", q


def test_route_category_dim():
    assert route_question("帮我推荐衣服").category_dim == "fashion"
    assert route_question("推荐一件连衣裙").category_dim == "fashion"
    assert route_question("推荐精华").category_dim == "beauty"
    assert route_question("敏感肌用什么面霜").category_dim == "beauty"
    assert route_question("有什么推荐").category_dim == ""


# ── 缺失信息（需求挖掘） ──


def test_missing_info_for_recommendation():
    r = route_question("帮我推荐一款精华")
    assert "skin_type" in r.missing_info


def test_missing_info_skin_type_present():
    r = route_question("敏感肌用什么面霜")
    assert "skin_type" not in r.missing_info


def test_missing_info_resolved_by_profile():
    """画像已含肤质/价位时，意图层不再标记为缺失（避免重复挖需）。"""
    r = route_question(
        "这个精华适合我吗，有点想买",
        profile={"skin_type": "干皮", "price_band": "中端"},
    )
    assert r.missing_info == []


def test_retrieval_decision_resolves_missing_with_profile():
    from agent_base.retrieval.retrieval_policy import build_retrieval_decision

    rewrite, _ = build_retrieval_decision(
        "这个精华适合我吗，有点想买",
        top_k=6,
        profile={"skin_type": "干皮"},
    )
    assert "skin_type" not in rewrite.route.missing_info


def test_recommend_phrasing_prefers_recommendation_intent():
    assert route_question("帮我推荐一款精华").intent == "recommendation"
    assert route_question("敏感肌泛红用什么面霜").intent == "recommendation"
    assert route_question("干皮秋冬用什么面霜").intent == "recommendation"


# ── 向量语义兜底 ──


def test_semantic_layer_matches_intent_example():
    rules = [
        IntentRule("aftersale", [], [], examples=["拆封了还能退货吗"]),
        IntentRule("fashion_query", [], [], examples=["白色T恤怎么搭配"]),
    ]
    route = _try_semantic_layer(
        "用过了想退掉",
        _fallback_route(),
        rules,
        embeddings=_FeatureEmbeddings(),
    )
    assert route is not None
    assert route.source == "semantic"
    assert route.intent == "aftersale"


def test_semantic_layer_rejects_unrelated_question():
    rules = [
        IntentRule("aftersale", [], [], examples=["拆封了还能退货吗"]),
        IntentRule("fashion_query", [], [], examples=["白色T恤怎么搭配"]),
    ]
    assert (
        _try_semantic_layer(
            "今天天气真不错",
            _fallback_route(),
            rules,
            embeddings=_FeatureEmbeddings(),
        )
        is None
    )


def test_route_question_without_embeddings_does_not_crash():
    r = route_question("今天天气真不错")
    assert r.intent == "general_qa"


def test_route_semantic_hit_when_enabled_with_embeddings():
    r = route_question(
        "用过了想退掉",
        intent_classifier_cfg={"semantic": {"enabled": True, "threshold": 0.5}},
        embeddings=_FeatureEmbeddings(),
    )
    assert r.source == "semantic"
    assert r.intent == "aftersale"


def test_retrieval_decision_route_carries_v2_fields():
    from agent_base.retrieval.retrieval_policy import build_retrieval_decision

    rewrite, decision = build_retrieval_decision("这个精华适合我吗，有点想买", top_k=6)
    route = rewrite.route
    assert route.buying_signal == "buying"
    assert hasattr(route, "sub_intent")
    assert "sub_intent" in route.to_dict()


# ── LLM 多字段结构化输出 ──


class _FakeStructuredModel:
    """返回带扩展字段的结构化结果的假模型。"""

    intent = "price_query"
    confidence = 0.92
    reasoning = "价格咨询"
    sub_intent = "price_negotiation"
    buying_signal = "objection"
    objection_type = "price"
    missing_info = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return self


class _FailingModel:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        raise RuntimeError("model unavailable")


def test_llm_route_carries_extended_fields(monkeypatch):
    from agent_base.retrieval import llm_intent_classifier

    monkeypatch.setattr(
        llm_intent_classifier,
        "build_chat_model",
        lambda **kwargs: _FakeStructuredModel(),
    )
    rule_route = _fallback_route()
    llm_route = route_question_with_llm("99还是有点贵", rule_route)
    assert llm_route.source == "llm"
    assert llm_route.intent == "price_query"
    assert llm_route.sub_intent == "price_negotiation"
    assert llm_route.buying_signal == "objection"
    assert llm_route.objection_type == "price"


def test_llm_route_falls_back_on_failure(monkeypatch):
    from agent_base.retrieval import llm_intent_classifier

    monkeypatch.setattr(
        llm_intent_classifier,
        "build_chat_model",
        lambda **kwargs: _FailingModel(),
    )
    rule_route = _fallback_route()
    assert route_question_with_llm("99还是有点贵", rule_route) is rule_route
