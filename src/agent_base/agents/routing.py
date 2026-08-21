"""Agent 触发路由（契约 P2-01，电商版）。

在 P1 图 route 节点后判断是否需要走 Agent 路径。
注意：不 import RagState（避免 graphs→agents 循环导入），state 参数用 dict。
"""

from __future__ import annotations

import re
from typing import Any


_COMPARISON_PATTERNS = [
    r"哪个好", r"和.*比", r"对比", r"有什么区别", r"区别",
    r"哪个更", r"怎么选", r"选哪个", r"哪个牌子", r"哪款",
    r"两者", r"分别", r"还是.*好",
]


def should_use_agent(state: dict[str, Any]) -> str:
    """条件边路由：返回 "agent"（走 ReAct）或 "retrieve"（走确定性 Workflow）。

    触发条件（满足任一即走 agent）：
    1. 规则路由置信度 < 0.5
    2. 问题包含比较/多跳句式（多商品、跨类目对比）
    3. 问题中出现跨商品/跨类目信号（≥2 个商品提示词，如"两件""分别""对比"）
    """
    route = state.get("route") or {}
    question = state.get("question", "")

    # 条件 1：低置信度
    confidence = route.get("confidence", 1.0)
    if isinstance(confidence, (int, float)) and confidence < 0.5:
        return "agent"

    # 条件 2：比较/多跳句式
    if _looks_like_comparison(question):
        return "agent"

    # 条件 3：跨商品/跨类目信号
    if _looks_like_multi_entity(question):
        return "agent"

    return "retrieve"


def _looks_like_comparison(question: str) -> bool:
    """规则检测比较/多跳句式（中文关键词）。"""
    for pattern in _COMPARISON_PATTERNS:
        if re.search(pattern, question):
            return True
    return False


def _looks_like_multi_entity(question: str) -> bool:
    """检测跨商品/跨类目对比信号。

    当前实现：问题中出现 ≥2 个商品/类目提示词（"两件""两款""两个""分别""对比"），
    或"和/与/、"连接的并列实体 + 选择句式。
    后续可接入 NER/商品实体识别，P12 意图生产化时升级。
    """
    if any(kw in question for kw in ["两件", "两款", "两个", "分别", "对比"]):
        return True
    if re.search(r"[和与、]\S{1,20}(?:哪个|还是|怎么选)", question):
        return True
    return False
