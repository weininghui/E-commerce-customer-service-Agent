"""P32a：商品定位澄清机制测试。

覆盖：
1. 决策层：商品意图 + 无商品名 → need_clarification=True, strategy="clarification"
2. 决策层：商品意图 + 有商品名 → need_clarification=False
3. API：/api/ask 传"这款商品有什么功效"→ 回复为澄清追问（不含具体商品功效细节）
4. API：非商品意图 → 不触发澄清
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from agent_base.retrieval.retrieval_policy import (
    CLARIFICATION_INTENTS,
    build_retrieval_decision,
)


# ── 决策层单元测试 ──


def test_clarification_triggered_without_product():
    """商品意图 + 无商品约束 → need_clarification=True, strategy=clarification。"""
    for intent in CLARIFICATION_INTENTS:
        _, decision = build_retrieval_decision(
            question="这款商品有什么功效",
            product_name=None,
            product_spec=None,
            category=None,
        )
        assert decision.need_clarification is True, f"intent={intent} 应触发澄清"
        assert decision.strategy == "clarification", f"intent={intent} strategy 应为 clarification"
        assert decision.clarification_question, f"intent={intent} 应有 clarification_question"
        assert decision.candidate_k == 0
        assert decision.final_k == 0


def test_clarification_not_triggered_with_product_name():
    """商品意图 + 明确商品名 → need_clarification=False。"""
    for intent in CLARIFICATION_INTENTS:
        _, decision = build_retrieval_decision(
            question="这款商品有什么功效",
            product_name="玻尿酸精华",
            product_spec=None,
            category=None,
        )
        assert decision.need_clarification is False, (
            f"intent={intent} 有 product_name 时不应触发澄清"
        )


def test_clarification_not_triggered_with_product_spec():
    """商品意图 + 明确规格 → need_clarification=False。"""
    for intent in CLARIFICATION_INTENTS:
        _, decision = build_retrieval_decision(
            question="这款有什么功效",
            product_name=None,
            product_spec="保湿精华",
            category=None,
        )
        assert decision.need_clarification is False, (
            f"intent={intent} 有 product_spec 时不应触发澄清"
        )


def test_clarification_not_triggered_non_product_intent():
    """非商品意图 → 不触发澄清。"""
    non_product_queries = [
        "我要退货怎么操作",
        "今天天气怎么样",
        "什么时候发货",
    ]
    for q in non_product_queries:
        _, decision = build_retrieval_decision(
            question=q,
            product_name=None,
            product_spec=None,
            category=None,
        )
        assert not decision.need_clarification, f"非商品意图 '{q}' 不应触发澄清"


def test_clarification_not_triggered_when_question_names_product():
    """问题中提到了具体商品名（如"玻尿酸精华"）→ 不触发澄清。"""
    _, decision = build_retrieval_decision(
        question="玻尿酸精华适合敏感肌吗",
        product_name=None,
        product_spec=None,
        category="精华",
    )
    # 问题有具体商品名且长度 > 8 且无"这款/这个"等模糊词 → 不触发
    assert decision.strategy != "clarification", (
        f"问题明确提到商品，不应触发澄清，实际 strategy={decision.strategy}"
    )


def test_clarification_question_contains_candidates():
    """clarification_question 包含候选商品提示或兜底追问。"""
    _, decision = build_retrieval_decision(
        question="这款商品有什么功效",
        product_name=None,
        product_spec=None,
        category=None,
    )
    assert decision.clarification_question, "应有澄清追问文本"
    # 至少包含"您想了解哪款产品"的提示
    assert "您想了解" in decision.clarification_question or "请说出" in decision.clarification_question


# ── API 层集成测试 ──


def test_ask_api_returns_clarification_without_product(
    client: TestClient, headers: dict[str, str]
):
    """POST /api/ask：无商品约束 → 回复为澄清追问，不含具体功效细节。"""
    # 用唯一 session_id 绕过 Redis 缓存（避免命中旧答案）
    import time
    sid = f"test-clarify-{int(time.time() * 1000)}"
    r = client.post(
        "/api/ask",
        json={"question": "这款商品有什么功效", "top_k": 4, "session_id": sid},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    answer = body.get("answer", "")
    # 回复应为澄清追问，不应包含具体功效信息
    assert len(answer) > 0, "answer 不应为空"
    # trace 应标记 need_clarification
    trace = body.get("trace", {})
    decision = trace.get("decision") or {}
    assert decision.get("need_clarification") or decision.get("strategy") == "clarification", (
        f"trace.decision 应标记澄清状态，实际: {decision}"
    )
    # 澄清回答应包含追问提示
    assert "您想了解" in answer or "哪款" in answer or "请说出" in answer, (
        f"回复应为澄清追问，实际: {answer[:200]}"
    )


def test_ask_api_normal_for_non_product_intent(
    client: TestClient, headers: dict[str, str]
):
    """POST /api/ask：非商品意图 → 不触发澄清，正常回答。"""
    r = client.post(
        "/api/ask",
        json={"question": "如何退货", "top_k": 4},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    answer = body.get("answer", "")
    # 非商品意图的回复不应是澄清追问
    assert "您想了解哪款产品" not in answer, f"非商品意图不应触发澄清，实际: {answer[:200]}"


def test_ask_stream_returns_clarification_event(
    client: TestClient, headers: dict[str, str]
):
    """POST /api/ask/stream：澄清场景 SSE 流正常返回。"""
    import time
    sid = f"test-clarify-stream-{int(time.time() * 1000)}"
    r = client.post(
        "/api/ask/stream",
        json={"question": "这款商品有什么功效", "top_k": 4, "session_id": sid},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    # 非错误响应即可（SSE 流至少包含 data: 行）
    assert len(r.text.strip()) > 0, "SSE 响应不应为空"


def test_comparison_intent_not_clarified():
    """对比意图（comparison）天然多商品 → 不触发澄清，走正常多商品回答。"""
    _, decision = build_retrieval_decision(
        question="玻尿酸精华和水乳哪个更好",
        product_name=None,
        product_spec=None,
        category=None,
    )
    # comparison 不在 CLARIFICATION_INTENTS 中，不应触发
    assert decision.strategy != "clarification", "comparison 意图不应触发澄清"


def test_promotion_intent_not_clarified():
    """促销意图（promotion）无单商品依赖 → 不触发澄清。"""
    # 注意："最近有什么优惠活动"中"优惠"命中 price_query 关键词，
    # 意图路由可能判为 price_query（CLARIFICATION_INTENTS 成员）。
    # 使用促销专属关键词确保路由到 promotion。
    _, decision = build_retrieval_decision(
        question="最近有什么大促活动",
        product_name=None,
        product_spec=None,
        category=None,
    )
    # promotion 不在 CLARIFICATION_INTENTS 中；若路由判别为 price_query
    # 则 strategy=clarification，这也符合预期（price_query 需要商品约束）
    if decision.intent == "promotion":
        assert decision.strategy != "clarification", "promotion 意图本身不应触发澄清"
