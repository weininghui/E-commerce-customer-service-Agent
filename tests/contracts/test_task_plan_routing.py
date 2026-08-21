"""契约 P33b-v2：TaskPlan 路由机制确定性测试（不依赖 LLM 输出）。

端到端测试受 LLM 决策波动影响（同问题可能判 direct 或 delegate），
机制层用固定 TaskPlan 输入验证路由/分发/反思逻辑本身，保证确定性。
"""

from __future__ import annotations

from types import SimpleNamespace

from agent_base.agents.graph_supervisor import (
    _reflect,
    _route_after_supervise,
    _supervise,
)
from agent_base.structured import TaskPlan


def _state(plan: TaskPlan, **extra) -> dict:
    # P1: Claude Code 引入 RemainingSteps 路由条件，fixture 需带足够步数
    return {"task_plan": plan.model_dump(), "clarify": False, "remaining_steps": 30, **extra}


def _fake_runtime() -> SimpleNamespace:
    """契约测试直接调用节点函数：按官方签名注入 fake Runtime（context 为空，节点不 emit）。"""
    return SimpleNamespace(context={})


def test_route_direct():
    """direct + 单 goal + 低复杂度 → retrieve（零并行）。"""
    plan = TaskPlan(
        goals=[{"action": "query", "targets": ["玻尿酸精华"], "constraints": []}],
        strategy="direct",
        complexity=1,
    )
    assert _route_after_supervise(_state(plan)) == "retrieve"


def test_route_direct_chat_skips_retrieval():
    """direct + chat goal（闲聊/寒暄）→ generate 直答，跳过检索。"""
    plan = TaskPlan(
        goals=[{"action": "chat", "targets": [], "constraints": []}],
        strategy="direct",
        complexity=1,
    )
    assert _route_after_supervise(_state(plan)) == "generate"


def test_route_direct_query_still_retrieves():
    """direct + query goal（简单事实问题）→ 仍需单检索（幻觉红线）。"""
    plan = TaskPlan(
        goals=[{"action": "query", "targets": ["玻尿酸精华"], "constraints": []}],
        strategy="direct",
        complexity=1,
    )
    assert _route_after_supervise(_state(plan)) == "retrieve"


def test_route_delegate_send():
    """delegate + 多 goal → Send 列表（并行分发）。"""
    plan = TaskPlan(
        goals=[
            {"action": "combine", "targets": ["玻尿酸精华", "面霜"], "constraints": []},
            {"action": "review", "targets": ["玻尿酸精华"], "constraints": []},
        ],
        strategy="delegate",
        complexity=4,
    )
    result = _route_after_supervise(_state(plan))
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(getattr(s, "node", None) == "worker" for s in result)


def test_route_clarify_end():
    """clarify 命中 → END（不复用检索）。"""
    plan = TaskPlan(
        goals=[{"action": "query", "targets": [], "constraints": []}],
        strategy="delegate",
        complexity=1,
    )
    assert _route_after_supervise(_state(plan, clarify=True)) == "__end__"


def test_route_taskplan_clarify_strategy():
    """TaskPlan 输出 strategy=clarify 或 missing_info → END（不走 worker）。"""
    plan = TaskPlan(
        goals=[{"action": "query", "targets": [], "constraints": []}],
        strategy="clarify",
        complexity=1,
        missing_info=["商品名称"],
    )
    assert _route_after_supervise(_state(plan)) == "__end__"

    plan2 = TaskPlan(
        goals=[{"action": "query", "targets": [], "constraints": []}],
        strategy="delegate",
        complexity=1,
        missing_info=["价格"],
    )
    assert _route_after_supervise(_state(plan2)) == "__end__"


def test_route_empty_goals_fallback():
    """空 goals → 兜底单检索 Send（不崩）。"""
    plan = TaskPlan(goals=[], strategy="delegate", complexity=3)
    result = _route_after_supervise(_state(plan))
    assert isinstance(result, list)
    assert len(result) == 1


def test_reflect_no_uncovered():
    """全部 goal 有证据 → reflect_ok=True，不补检索。"""
    plan = TaskPlan(
        goals=[{"action": "query", "targets": ["玻尿酸精华"], "constraints": []}],
        strategy="delegate",
        complexity=3,
    )
    state = _state(
        plan,
        goal_evidences=[{
            "goal": {"action": "query", "targets": ["玻尿酸精华"], "constraints": []},
            "sources": [{"section": "商品", "content": "玻尿酸精华 适合敏感肌"}],
            "evidence": "玻尿酸精华 适合敏感肌",
        }],
        reflect_rounds=0,
        runtime={},
    )
    extra = _reflect(state, _fake_runtime())
    assert extra["reflect_ok"] is True


def test_reflect_rounds_cap():
    """反思轮数达上限 → 直接放行（不无限循环）。"""
    plan = TaskPlan(
        goals=[{"action": "query", "targets": ["玻尿酸精华"], "constraints": []}],
        strategy="delegate",
        complexity=3,
    )
    state = _state(
        plan,
        goal_evidences=[],
        reflect_rounds=2,
        runtime={},
    )
    extra = _reflect(state, _fake_runtime())
    assert extra["reflect_ok"] is True


def test_supervise_clarify_plan_sets_question(monkeypatch):
    """TaskPlan 判 clarify（缺信息）→ supervise 置 clarify=True + 生成追问。

    回归 BUG：此前 LLM 判 strategy=clarify 时只改 mode，graph END 后
    clarify=False / clarification='' → 前端收到空回复。
    """
    from agent_base.agents import graph_supervisor as gs
    from langgraph.runtime import Runtime

    plan = {
        "goals": [{"action": "order", "targets": [], "constraints": []}],
        "strategy": "clarify",
        "complexity": 1,
        "requires_reflection": False,
        "missing_info": ["商品名称", "规格"],
    }
    monkeypatch.setattr(gs, "_parse_task_plan", lambda *a, **k: plan)
    # 用推荐类问题（不含订单/物流关键词）验证 clarify 分支——订单词会命中
    # _looks_like_order 兜底被强制转 order，clarify plan 被覆盖
    state = {"question": "帮我推荐一款适合我的精华", "enriched_q": "帮我推荐一款适合我的精华", "intent": "general_qa", "clarification": ""}
    extra = _supervise(state, Runtime(context={"on_agent": None}))
    assert extra["clarify"] is True
    assert extra["mode"] == "clarify"
    assert "商品名称" in extra["clarification"]
    assert "规格" in extra["clarification"]


def test_supervise_missing_info_only(monkeypatch):
    """strategy=delegate 但 missing_info 非空 → 同样置 clarify（路由 END 前必须产出追问）。"""
    from agent_base.agents import graph_supervisor as gs
    from langgraph.runtime import Runtime

    plan = {
        "goals": [{"action": "query", "targets": [], "constraints": []}],
        "strategy": "delegate",
        "complexity": 2,
        "requires_reflection": False,
        "missing_info": ["商品名称"],
    }
    monkeypatch.setattr(gs, "_parse_task_plan", lambda *a, **k: plan)
    state = {"question": "这款商品有什么功效", "enriched_q": "这款商品有什么功效", "intent": "product_query", "clarification": ""}
    extra = _supervise(state, Runtime(context={"on_agent": None}))
    assert extra["clarify"] is True
    assert extra["mode"] == "clarify"
    assert "商品名称" in extra["clarification"]


def test_supervise_clarify_fallback_question(monkeypatch):
    """strategy=clarify 且无 missing_info → 复用前置 clarify 追问；为空则通用兜底。"""
    from agent_base.agents import graph_supervisor as gs
    from langgraph.runtime import Runtime

    plan = {
        "goals": [{"action": "query", "targets": [], "constraints": []}],
        "strategy": "clarify",
        "complexity": 1,
        "requires_reflection": False,
        "missing_info": [],
    }
    monkeypatch.setattr(gs, "_parse_task_plan", lambda *a, **k: plan)
    state = {"question": "推荐一下", "enriched_q": "推荐一下", "intent": "recommendation", "clarification": "您想了解哪款产品呢？"}
    extra = _supervise(state, Runtime(context={"on_agent": None}))
    assert extra["clarify"] is True
    assert extra["clarification"] == "您想了解哪款产品呢？"

    state2 = {**state, "clarification": ""}
    extra2 = _supervise(state2, Runtime(context={"on_agent": None}))
    assert extra2["clarify"] is True
    assert extra2["clarification"]


def test_task_plan_json_fallback():
    """TaskPlan 解析 JSON 兜底：文本 JSON → model_validate。"""

    from agent_base.structured import parse_json_or_none

    raw = '[{"action":"review","targets":["面霜"],"constraints":[]}]'
    parsed = parse_json_or_none(raw)
    assert parsed is not None
    assert parsed[0]["action"] == "review"

    plan = TaskPlan.model_validate({
        "goals": parsed,
        "strategy": "delegate",
        "complexity": 3,
    })
    assert plan.goals[0].action == "review"
    assert plan.strategy == "delegate"
