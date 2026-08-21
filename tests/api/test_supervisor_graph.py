"""v0.47：LangGraph 多 Agent 编排测试——StateGraph 五模式路由 + 执行链 + 工具 Agent。"""

from __future__ import annotations

from agent_base.agents.graph_supervisor import _get_graph, run_supervisor_graph


def _run(runtime, question: str) -> tuple[dict, list[tuple[str, str]]]:
    events: list[tuple[str, str]] = []

    def on_agent(name, status, duration_ms=None, data=None):
        events.append((name, status))

    plan = run_supervisor_graph(
        question,
        runtime,
        {"top_k": 6, "candidate_k": None},
        on_agent=on_agent,
    )
    return plan, events


def _done_chain(events: list[tuple[str, str]]) -> list[str]:
    return [name for name, status in events if status == "done"]


def test_graph_nodes_complete():
    """StateGraph 节点齐全（编排 + worker/反思 + finalize）。"""
    nodes = set(_get_graph().get_graph().nodes.keys())
    for n in ("memory", "enrich", "intent", "clarify", "supervise", "worker", "reflect", "retrieve", "finalize", "generate"):
        assert n in nodes, f"missing node: {n}"


def test_direct_mode(runtime_supervisor):
    plan, events = _run(runtime_supervisor, "玻尿酸精华适合敏感肌吗")
    assert plan["clarify"] is False
    # P33b-v2：direct（单检索）或 delegate（LLM 认为需多步）都是合法编排
    assert plan["mode"] in ("direct", "delegate")
    assert plan["intent"] == "product_query"
    # direct → retrieve；delegate → worker（内部检索），两种路径都产出 sources
    assert "retrieve" in _done_chain(events) or "worker" in _done_chain(events)
    # sources 非空属检索测试职责（测试环境检索可能波动）；编排测试验证链路完整
    assert plan.get("answer") or plan.get("clarification")


def test_clarify_mode(runtime_supervisor):
    plan, events = _run(runtime_supervisor, "这款商品有什么功效")
    assert plan["clarify"] is True
    assert plan.get("clarification")
    chain = _done_chain(events)
    assert "retrieve" not in chain  # 澄清命中不检索
    assert len(plan.get("sources") or []) == 0


def test_self_ask_mode(runtime_supervisor):
    plan, events = _run(runtime_supervisor, "玻尿酸精华和水乳哪个好")
    # P33b-v2：对比问题 → TaskPlan 判 direct（单检索由生成层对比）、delegate（worker 拆解）、
    # 或 clarify（LLM 非确定性判缺商品名，合理波动）
    strategy = (plan.get("task_plan") or {}).get("strategy", "")
    assert plan["mode"] in ("direct", "delegate") or strategy == "clarify"
    assert "supervise" in _done_chain(events)
    # answer/clarification/strategy 有一即可（LLM 判 clarify 时 graph 提前 END）
    assert plan.get("answer") or plan.get("clarification") or strategy == "clarify"


def test_pae_mode(runtime_supervisor):
    plan, events = _run(runtime_supervisor, "怎么下单购买")
    # P33b-v2：下单引导 → TaskPlan 可判 order goal（worker+tool）、direct（FAQ 单检索）、
    # 或 clarify（LLM 判定缺商品名→mode=clarify 走澄清追问）。
    # 验证"supervise 决策 + 有结果"即可（LLM 判断波动不绑死路径）
    assert plan["mode"] in ("delegate", "direct", "clarify")
    chain = _done_chain(events)
    assert "supervise" in chain
    # LLM 非确定性：偶尔判 clarify → 图在 supervise 后 END（无 worker/retrieve），
    # 此时 task_plan.strategy 为 clarify，必须产出澄清追问（回归：曾为空回复）
    strategy = (plan.get("task_plan") or {}).get("strategy", "")
    if strategy == "clarify":
        assert plan["mode"] == "clarify"
        assert plan.get("clarification")
    else:
        assert "worker" in chain or "tool" in chain or "retrieve" in chain
        assert plan.get("answer") or plan.get("clarification")


def test_tool_mode(runtime_supervisor):
    plan, events = _run(runtime_supervisor, "订单 ORD123 的物流到哪里了")
    assert plan["intent"] == "aftersale"
    chain = _done_chain(events)
    # P33b-v2：订单问题 → order goal（worker+tool）或 direct（FAQ 检索），均合法
    assert "supervise" in chain
    assert "worker" in chain or "tool" in chain or "retrieve" in chain
    tool_result = plan.get("tool_result") or ""
    assert tool_result or plan.get("answer")
    dispatch = plan.get("dispatch") or []
    sv_entry = next((d for d in dispatch if d.get("agent") == "supervise"), None)
    assert sv_entry is not None
    assert "duration_ms" in sv_entry.get("meta", {})


def test_dispatch_shape(runtime_supervisor):
    plan, _ = _run(runtime_supervisor, "玻尿酸精华适合敏感肌吗")
    dispatch = plan.get("dispatch") or []
    assert len(dispatch) >= 5  # memory/enrich/intent/clarify/retrieve
    for d in dispatch:
        assert d.get("agent")
        assert "meta" in d and "duration_ms" in d.get("meta", {})
