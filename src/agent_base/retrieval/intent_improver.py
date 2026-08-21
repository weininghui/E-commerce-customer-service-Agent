"""意图配置 AI 优化服务（P21）：LLM 基于现有配置生成改进建议。

从 api/main.py 抽出（v0.30.0 分层：业务逻辑归检索域）。
"""

from __future__ import annotations

from typing import Any


def improve_intent_config(intent: dict[str, Any]) -> dict[str, Any]:
    """AI 优化单个意图的关键词/章节/示例。

    Args:
        intent: intent_get 返回的意图配置（intent/keywords/sections/examples）。

    Returns:
        {intent, suggested: {keywords, sections, examples}, reasoning}。

    Raises:
        RuntimeError: LLM 调用或解析失败。
    """
    from agent_base.llms import build_chat_model

    from agent_base.prompts import get_prompt

    intent_name = intent["intent"]
    _DEFAULT_IMPROVER = (
        "你是电商客服意图配置优化专家。基于现有配置优化这个意图的关键词、章节、示例，"
        "让非技术运营也能理解并提升识别准确率。\n"
        "意图：{intent_name}\n"
        "当前关键词：{keywords}\n"
        "当前章节：{sections}\n"
        "当前示例：{examples}\n"
        "要求：\n"
        "1. 关键词：补充同义表达和常见问法用词，去重，每项不超过 12 字，8-25 项\n"
        "2. 章节：保持不变（检索匹配由系统内置管理，不修改）\n"
        "3. 示例：保留现有并补充 3-5 条真实用户问法，共 6-10 条\n"
        '只输出 JSON：{"keywords": [...], "sections": [...], "examples": [...], "reasoning": "改动理由"}'
    )
    prompt = (
        get_prompt("improve_intent", "system", _DEFAULT_IMPROVER)
        .replace("{intent_name}", intent_name)
        .replace("{keywords}", str(intent.get("keywords", [])))
        .replace("{sections}", str(intent.get("sections", [])))
        .replace("{examples}", str(intent.get("examples", [])))
    )
    model = build_chat_model(
        provider="langchain", model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key_env="ANTHROPIC_AUTH_TOKEN", temperature=0.2,
    )
    # LCEL 官方链：ChatPromptTemplate | model | StrOutputParser
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    chain = ChatPromptTemplate.from_messages([("user", "{prompt}")]) | model | StrOutputParser()
    text = chain.invoke({"prompt": prompt}).strip()
    import json as _json
    import re
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("AI 返回格式异常")
    result = _json.loads(cleaned[start:end + 1])
    return {
        "intent": intent_name,
        "suggested": {
            "keywords": result.get("keywords", []),
            "sections": result.get("sections", []),
            "examples": result.get("examples", []),
        },
        "reasoning": result.get("reasoning", ""),
    }
