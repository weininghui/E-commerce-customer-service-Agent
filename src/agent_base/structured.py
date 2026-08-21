"""结构化输出辅助（P26b）。

用 ``with_structured_output``（langchain-core）替代"只返回 JSON"裸约束，
让意图分类 / 记忆提炼 / 文档预审 / 问法变体四个环节输出由 Pydantic 校验，
失败回退现有 JSON 解析 / 规则模板，不阻塞链路。
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, Field


# ── Pydantic 输出模型 ───────────────────────────────────────────────────────


class IntentResult(BaseModel):
    """意图分类结果。"""

    intent: str = Field(description="意图类别")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度")
    reasoning: str = Field(default="", description="分类理由")
    sub_intent: str = Field(
        default="",
        description="子意图：chat/price_negotiation/price_inquiry/return_exchange/"
        "logistics/allergy/recommend_request/compare/usage/review 等",
    )
    buying_signal: str = Field(
        default="normal",
        description="购买信号：normal/buying/objection",
    )
    objection_type: str = Field(
        default="none",
        description="异议类型：price/hesitant/risk/none",
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="缺失需求信息（skin_type/budget/scene/product/size）",
    )


class MemoryItem(BaseModel):
    """记忆提炼单条画像。"""

    key: str = Field(description="画像键（skin_type/price_band/category/style/size/season/intent）")
    value: str = Field(description="画像值")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度")
    tier: str = Field(
        default="user_statement",
        description="信任层级：user_statement（用户直接陈述）/ tool_result（工具返回）/ "
        "agent_inference（Agent 推断）/ conflict_confirmed（用户改口确认）",
    )


class PreReviewResult(BaseModel):
    """文档预审决策包。"""

    type: str = Field(description="doc_type")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度")
    reasoning: str = Field(default="", description="评审理由")
    suggest_action: str = Field(default="review", description="approve/review/reject")
    reject_hint: str = Field(default="", description="打回原因提示")
    risk_flags: list[str] = Field(default_factory=list, description="风险标记")


class TaskGoal(BaseModel):
    """主 agent 任务编排：单个子目标（SUPERVISOR-TOTAL-PLAN v2）。"""

    action: str = Field(
        description="动作类型：chat(闲聊直答，无需检索)/query(单商品查询)/compare(多实体对比)/recommend(推荐决策)/"
        "combine(搭配方案)/review(评价口碑)/"
        "usage(使用效果方法)/order(订单物流售后)/price(价格优惠)",
    )
    targets: list[str] = Field(default_factory=list, description="涉及实体（商品名/概念）")
    constraints: list[str] = Field(default_factory=list, description="约束（干皮/秋冬/预算等）")


class TaskPlan(BaseModel):
    """主 agent 任务编排意图：LLM 理解任务后输出的执行计划。

    区别于检索意图 intent（管「查什么怎么查」）：
    TaskPlan 管「任务怎么拆、派给谁」（编排决策）。两者各管一条链路，不混。
    """

    goals: list[TaskGoal] = Field(description="拆出的子任务列表（≥1）")
    strategy: str = Field(
        default="delegate",
        description="执行策略：direct(主 agent 直答)/delegate(分配子 agent)/clarify(缺信息先澄清)",
    )
    complexity: int = Field(default=1, ge=1, le=5, description="任务复杂度 1-5")
    requires_reflection: bool = Field(default=False, description="是否需要执行后反思补漏")
    missing_info: list[str] = Field(default_factory=list, description="缺失信息（触发 clarify）")




T = TypeVar("T", bound=BaseModel)


def try_structured(
    model: Any,
    schema: type[T],
    messages: list[dict[str, str]],
) -> T | None:
    """尝试用 with_structured_output 获取结构化结果；失败返回 None。

    Args:
        model: LangChain chat 模型。
        schema: Pydantic 输出模型。
        messages: [{role, content}] 消息列表。

    Returns:
        Pydantic 实例；构造/调用失败返回 None。
    """
    if model is None:
        return None
    try:
        chain = model.with_structured_output(schema)
        result = chain.invoke(messages)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
    except Exception:
        return None
    return None


def parse_json_or_none(text: str) -> dict[str, Any] | list[Any] | None:
    """健壮 JSON 解析：支持裸 JSON / markdown 代码块 / 正则提取。"""
    text = (text or "").strip()
    if not text:
        return None
    # 去 markdown 代码块围栏
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 提取第一个 JSON 对象/数组
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return None
