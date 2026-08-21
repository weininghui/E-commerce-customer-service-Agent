"""模型单价表与 Token 成本估算（运营看板成本展示）。"""

from __future__ import annotations

from typing import Any


# 模型单价（元 / 1M tokens）：(输入单价, 输出单价)
# 值为常见商用价位的近似估算，用于看板展示量级，不做财务结算。
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "deepseek-chat": (1.0, 2.0),
    "deepseek-reasoner": (2.0, 8.0),
    "deepseek-v4-flash": (1.0, 2.0),
    "deepseek-v4-pro": (2.0, 8.0),
    "doubao": (0.8, 2.0),
    "gpt-4o-mini": (1.5, 4.5),
    "gpt-4o": (18.0, 54.0),
    "claude": (18.0, 54.0),
    "qwen": (1.0, 2.0),
    "glm": (1.0, 2.0),
    "moonshot": (1.0, 2.0),
}

# 未匹配到单价时的兜底（元 / 1M tokens）
DEFAULT_PRICE: tuple[float, float] = (1.0, 2.0)


def model_price(model: str) -> tuple[float, float]:
    """按模型名（前缀匹配，忽略大小写）返回 (输入单价, 输出单价)。"""
    name = (model or "").lower()
    for key, price in MODEL_PRICES.items():
        if key in name:
            return price
    return DEFAULT_PRICE


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """估算一次调用的成本（元），按输入/输出 token 分别计价。"""
    price_in, price_out = model_price(model)
    cost = (
        int(prompt_tokens or 0) * price_in
        + int(completion_tokens or 0) * price_out
    ) / 1_000_000
    return round(cost, 6)


def row_cost(row: dict[str, Any]) -> float:
    """按 token_usage 行数据估算成本。"""
    return estimate_cost(
        str(row.get("model") or ""),
        int(row.get("prompt_tokens") or 0),
        int(row.get("completion_tokens") or 0),
    )
