"""LLM 意图分类增强（电商域，可选）。

规则路由结果置信度不足时，可用 LLM 二次分类（配置 intent_classifier.enabled=true）。
LLM 不可用/输出非法时回退规则路由，不阻塞主链路。
"""

from __future__ import annotations


from agent_base.llms import build_chat_model
from agent_base.retrieval.intent_router import QueryRoute


INTENT_CATEGORIES = [
    "product_query",
    "fashion_query",
    "price_query",
    "aftersale",
    "recommendation",
    "comparison",
    "size_recommendation",
    "promotion",
    "general_qa",
]

INTENT_SECTIONS = {
    "product_query": ["商品参数", "卖点"],
    "fashion_query": ["商品参数", "卖点", "搭配建议"],
    "price_query": ["价格"],
    "aftersale": ["售后FAQ"],
    "recommendation": ["商品参数", "卖点", "评价"],
    "comparison": ["商品参数", "卖点", "评价"],
    "size_recommendation": ["商品参数", "卖点", "搭配建议"],
    "promotion": ["价格", "售后FAQ"],
    "general_qa": [],
}

def route_question_with_llm(
    question: str,
    rule_route: QueryRoute,
    provider: str = "langchain",
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
    temperature: float = 0.0,
) -> QueryRoute:
    """使用 LLM 对问题进行意图分类，增强规则路由的结果。

    Args:
        question: 用户问题。
        rule_route: 规则路由的初步结果。
        provider: LLM 提供商。
        model: 模型名称。
        base_url: API 基础 URL。
        api_key_env: API 密钥环境变量名。
        temperature: 生成温度。

    Returns:
        更新后的 QueryRoute；LLM 不可用或输出非法时返回 rule_route。
    """
    chat_model = build_chat_model(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        temperature=temperature,
    )
    if chat_model is None:
        return rule_route

    try:
        from agent_base.structured import IntentResult, parse_json_or_none
        from agent_base.prompts import get_prompt
        system_prompt = get_prompt("intent", "system")
        user_template = get_prompt("intent", "user_template")
        user_prompt = user_template.format(
            question=question,
            rule_intent=rule_route.intent,
            rule_sections=", ".join(rule_route.sections),
            rule_keywords=", ".join(rule_route.matched_keywords),
        )
        # LCEL 官方链：直接以消息列表调用（system 提示词含 JSON 花括号，
        # 不用 ChatPromptTemplate 解析，避免被当成模板变量）
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        # P26b：优先 with_structured_output，失败回退 JSON 解析
        structured = None
        try:
            structured = chat_model.with_structured_output(IntentResult).invoke(messages)
        except Exception:
            structured = None
        if structured is not None:
            llm_result = {
                "intent": structured.intent,
                "confidence": structured.confidence,
                "reasoning": structured.reasoning,
                "sub_intent": structured.sub_intent,
                "buying_signal": structured.buying_signal,
                "objection_type": structured.objection_type,
                "missing_info": list(structured.missing_info or []),
            }
        else:
            try:
                resp = chat_model.invoke(messages)
                raw = str(getattr(resp, "content", "") or resp or "")
            except Exception:
                raw = ""
            llm_result = parse_json_or_none(raw)
        if llm_result and llm_result.get("intent") in INTENT_CATEGORIES:
            llm_intent = llm_result["intent"]
            llm_confidence = llm_result.get("confidence", 0.8)
            sections = INTENT_SECTIONS.get(llm_intent, []) or rule_route.sections
            from agent_base.retrieval.intent_router import _section_filter
            llm_route = QueryRoute(
                intent=llm_intent,
                sections=sections,
                metadata_filter=_section_filter(sections),
                matched_keywords=rule_route.matched_keywords,
                confidence=llm_confidence,
                scores={**rule_route.scores, "llm": llm_confidence},
                source="llm",
                sub_intent=str(llm_result.get("sub_intent") or ""),
                buying_signal=str(llm_result.get("buying_signal") or "normal"),
                objection_type=str(llm_result.get("objection_type") or "none"),
                missing_info=list(llm_result.get("missing_info") or []),
            )
            from agent_base.retrieval.intent_router import _enrich_route

            _enrich_route(llm_route, question)
            return llm_route
    except Exception as e:
        print(f"LLM 意图分类失败，回退到规则路由: {e}")

    return rule_route
