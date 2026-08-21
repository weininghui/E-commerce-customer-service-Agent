"""查询改写（电商域）。

把用户问题 + 商品约束 + 意图章节组合成检索查询锚点，
保留商品名/类目/肤质/尺码/功效等关键信息。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agent_base.retrieval.intent_router import QueryRoute, route_question


INTENT_ANCHORS = {
    "product_query": ["商品参数", "成分", "功效", "适合肤质", "规格"],
    "fashion_query": ["穿搭", "搭配", "版型", "尺码", "面料", "季节", "风格", "场合"],
    "price_query": ["价格", "优惠", "折扣", "到手价"],
    "aftersale": ["退换货", "物流", "发票", "售后", "保质期", "快递"],
    "recommendation": ["推荐", "适合肤质", "使用场景", "评价"],
    "comparison": ["对比", "区别", "差异", "选购"],
    "size_recommendation": ["尺码", "身高", "体重", "合身", "偏大", "偏小"],
    "promotion": ["满减", "优惠券", "折扣", "秒杀", "会员", "积分"],
    "general_qa": [],
}


@dataclass(slots=True)
class QueryRewrite:
    """改写后的检索查询：原始问题 + 锚点拼接 + 意图路由结果。"""

    original_question: str
    rewritten_question: str
    route: QueryRoute
    anchors: list[str]

    def to_dict(self) -> dict[str, Any]:
        """转为 dict，route 会展开为嵌套字典。

        Returns:
            可 JSON 序列化的查询改写结果。
        """
        data = asdict(self)
        data["route"] = self.route.to_dict()
        return data


def build_query_rewrite(
    question: str,
    product_name: str | None = None,
    product_spec: str | None = None,
    intent_classifier: dict[str, Any] | None = None,
    domain: Any | None = None,
    profile: dict[str, Any] | None = None,
) -> QueryRewrite:
    """构造改写后的检索查询（电商版）。

    Args:
        question: 用户问题。
        product_name: 商品名约束。
        product_spec: 商品规格/别名约束。
        intent_classifier: LLM 意图增强配置（可选）。
        domain: DomainAdapter 实例（意图路由用）。
        profile: 用户画像字典（意图层消解缺失需求用）。

    Returns:
        QueryRewrite。
    """
    route = route_question(
        question,
        domain=domain,
        intent_classifier_cfg=intent_classifier,
        profile=profile,
    )
    # P12: LLM 增强层已内置在 route_question 三层识别中（仅在规则置信度 < 阈值时触发），
    # 不再在 query_rewriter 中重复调用 route_question_with_llm。
    domain_anchors: dict[str, list[str]] = getattr(domain, "intent_anchors", {}) or {}
    anchors: list[str] = []
    if product_name and product_name != "unknown":
        anchors.append(product_name)
    if product_spec and product_spec != "unknown":
        anchors.append(product_spec)
    anchors.extend(route.sections)
    if route.intent in domain_anchors:
        anchors.extend(domain_anchors.get(route.intent, []))
    else:
        anchors.extend(INTENT_ANCHORS.get(route.intent, []))
    anchors.extend(route.matched_keywords)
    anchors.append(question)
    unique_anchors = list(dict.fromkeys(anchor for anchor in anchors if anchor))
    return QueryRewrite(
        original_question=question,
        rewritten_question=" ".join(unique_anchors),
        route=route,
        anchors=unique_anchors,
    )
