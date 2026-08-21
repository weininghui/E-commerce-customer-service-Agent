"""专家级销售策略（导购模式）：购买信号识别 → 异议分类 → 话术策略注入。

设计：
- 不改变检索链路，只在受控 LLM 生成层注入"导购策略"系统提示词；
- 触发条件：商品类意图（product_query/fashion_query/price_query/recommendation/
  comparison/size_recommendation）+ 检测到购买信号（想买/下单/值不值/纠结/嫌贵...）；
- 异议分类：price（嫌贵）/ hesitant（犹豫）/ risk（怕不合适），策略块内给出
  对应话术打法；合规红线（不编造数值、不承诺功效、不绝对化、不贬低竞品）保留。
"""

from __future__ import annotations

from functools import lru_cache

# 商品类意图：只有这类问题才值得切导购模式（售后/物流/通用问答不硬推销）
# 商品类意图：导购模式优先覆盖范围（历史语义保留，现由 NON_SALES_INTENTS 黑名单决定）
SALES_INTENTS = {
    "product_query",
    "fashion_query",
    "price_query",
    "recommendation",
    "comparison",
    "size_recommendation",
}

# 不切导购模式的意图：售后/物流是处理问题，不是卖货
NON_SALES_INTENTS = {"aftersale"}

# 购买信号：强购买意向（提到买/下单/纠结/值不值/适合我吗）
_BUYING_PATTERNS = [
    "想买", "想入手", "下单", "入手", "种草", "值得买", "值得入", "要买",
    "适合我吗", "适合我", "合适吗", "合适我", "值不值", "要不要买", "纠结买",
    "买哪个", "买哪款", "有点想", "心动了", "想试试", "考虑买", "准备买",
    "多少钱", "什么价", "价位",
    "想淘", "淘点", "淘货", "好东西", "看看有什么", "有什么好", "推荐点", "来点好",
    "配什么", "搭配什么", "还缺", "需要配",
]

# 价格异议
_PRICE_OBJECTION_PATTERNS = [
    "太贵", "贵了", "有点贵", "贵吗", "便宜", "优惠", "打折", "性价比",
    "降价", "划不划算", "值这个价", "能便宜",
]

# 犹豫
_HESITANT_PATTERNS = [
    "再想想", "考虑考虑", "纠结", "犹豫", "等等再说", "先看看", "再看看",
    "还是犹豫", "拿不定主意", "不确定要不要", "要不要现在买",
]

# 风险顾虑（怕不合适/怕踩雷）
_RISK_PATTERNS = [
    "怕不合适", "怕踩雷", "万一", "不合适怎么办", "能退吗", "退货麻烦",
    "怕买错", "怕不好用", "怕过敏", "怕用了不好", "不合适能换吗",
]

# 促销/优惠咨询（区别于嫌贵异议）：问"有优惠吗/满减/活动"是购买兴趣，不是异议
_DISCOUNT_ASK_PATTERNS = (
    "优惠吗", "有优惠", "打折吗", "满减", "活动吗", "赠品", "优惠券", "折扣吗", "折扣",
)


def detect_sales_signal(question: str) -> dict[str, str]:
    """检测购买信号与异议类型。

    Args:
        question: 用户当前问题。

    Returns:
        {mode: buying|objection|normal, objection_type: price|hesitant|risk|none}。
    """
    if not question:
        return {"mode": "normal", "objection_type": "none"}
    q = question.strip()
    # 促销咨询优先判定为购买兴趣（避免被"优惠/打折"误判为嫌贵异议）
    if any(p in q for p in _DISCOUNT_ASK_PATTERNS):
        return {"mode": "buying", "objection_type": "none"}
    # 异议优先级：价格 > 犹豫 > 风险（同时出现取最紧迫的）
    for pat, typ in (
        (_PRICE_OBJECTION_PATTERNS, "price"),
        (_HESITANT_PATTERNS, "hesitant"),
        (_RISK_PATTERNS, "risk"),
    ):
        if any(p in q for p in pat):
            return {"mode": "objection", "objection_type": typ}
    if any(p in q for p in _BUYING_PATTERNS):
        return {"mode": "buying", "objection_type": "none"}
    return {"mode": "normal", "objection_type": "none"}


@lru_cache(maxsize=1)
def _load_sales_strategy() -> str:
    """从 prompts_ecommerce.yaml 加载专家级销售策略块（缺失回退内置兜底）。"""
    try:
        from agent_base.config import load_yaml

        prompts = load_yaml("configs/prompts_ecommerce.yaml") or {}
        strategy = ((prompts.get("sales") or {}).get("strategy") or "").strip()
        if strategy:
            return strategy
    except Exception:
        pass
    return (
        "【导购模式】用户表现出购买意向/异议时，像顶级电商导购一样："
        "先确认需求（肤质/尺码/预算/场景），讲透卖点与适合场景，"
        "异议用价值重述+售后保障回应，需求匹配后给明确行动建议，自然带出搭配推荐。"
    )


def build_sales_strategy(question: str, intent: str = "") -> str:
    """构建注入系统提示词的销售策略块（无购买信号返回空串，不影响普通问答）。

    Args:
        question: 用户问题。
        intent: 路由意图（非商品类意图不切导购模式）。

    Returns:
        导购策略提示词块；无购买信号/非商品意图返回 ""。
    """
    # 仅售后/物流类问题不切导购（处理退换货，不是卖货）；
    # 其余意图只要检测到购买/异议信号就注入导购策略（含模糊购物问题落 general_qa 的场景）
    signal = detect_sales_signal(question)
    if signal["mode"] == "normal":
        return ""
    if intent in NON_SALES_INTENTS:
        return ""
    return _load_sales_strategy()
