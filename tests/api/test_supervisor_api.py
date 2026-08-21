"""P33a 主 Agent（LangGraph 版）编排测试（v0.49 全 LangGraph 化）。"""

from __future__ import annotations

import time

from agent_base.agents.graph_supervisor import run_supervisor_graph
from agent_base.agents.supervisor import AgentResult, enrich_agent


def _run(runtime, question: str, sid: str) -> dict:
    return run_supervisor_graph(
        question,
        runtime,
        {"product_name": None, "product_spec": None, "category": None, "catalog_resolution": {}},
        session_id=sid,
    )


def _ended_at_clarify(result: dict) -> bool:
    """LLM 决策波动下，graph 是否在澄清处合法收尾（clarify 或 missing_info → END，不派工）。"""
    tp = result.get("task_plan") or {}
    return bool(result.get("clarify") or tp.get("strategy") == "clarify" or tp.get("missing_info"))


def test_agent_result_contract():
    """统一契约：status/data/sources/confidence/meta 五字段。"""
    r = AgentResult(status="ok", data={"k": "v"}, sources=[{"doc_name": "x"}], confidence=0.8)
    d = r.to_dict()
    assert set(d.keys()) == {"status", "data", "sources", "confidence", "meta"}
    assert d["status"] == "ok"
    assert d["data"]["k"] == "v"
    assert d["confidence"] == 0.8


def test_enrich_alias_expansion():
    """完善 Agent：别名「玻尿酸精华」→ 标准商品名。"""
    r = enrich_agent("玻尿酸精华适合敏感肌吗")
    assert "玻尿酸保湿精华液" in r.data["rewritten"]


def test_supervisor_orchestration_latency_budget(runtime_supervisor):
    """性能回归：澄清分支（纯规则/DB，无 LLM）编排耗时必须 < 5s。"""
    t0 = time.perf_counter()
    result = _run(runtime_supervisor, "这款商品有什么功效", "test-sv-perf")
    elapsed = time.perf_counter() - t0
    assert result["clarify"] is True
    assert elapsed < 5.0, f"编排层耗时 {elapsed:.2f}s 超出预算"


def test_supervisor_clarify_branch(runtime_supervisor):
    """未指定商品 → clarify 分支返回澄清追问，不做检索。"""
    result = _run(runtime_supervisor, "这款商品有什么功效", "test-sv-clarify")
    assert result["clarify"] is True
    assert "您想了解哪款产品" in result.get("clarification", "")
    agents = [d["agent"] for d in result["dispatch"]]
    assert "clarify" in agents
    assert "retrieve" not in agents


def test_supervisor_direct_branch(runtime_supervisor):
    """有商品名 → TaskPlan 编排（retrieve 或 worker），正常回答（澄清路径也合法）。"""
    result = _run(runtime_supervisor, "玻尿酸精华适合敏感肌吗", "test-sv-direct")
    if _ended_at_clarify(result):
        assert result.get("clarification")
        return
    assert result["answer"]
    agents = [d["agent"] for d in result["dispatch"]]
    # P33b-v2：direct → retrieve；delegate → worker（内部检索），均正常
    assert "supervise" in agents
    assert "retrieve" in agents or "worker" in agents
    assert "generate" in agents


def test_supervisor_tool_branch(runtime_supervisor):
    """订单查询 → TaskPlan order goal → worker 内部调工具（澄清/缺信息路径也合法）。"""
    result = _run(runtime_supervisor, "订单 ORD123 的物流到哪里了", "test-sv-tool")
    if _ended_at_clarify(result):
        assert result.get("clarification")
        return
    agents = [d["agent"] for d in result["dispatch"]]
    # P33b-v2：worker 内部调 tool（事件透传），dispatch 含 supervise/worker
    assert "supervise" in agents
    assert "worker" in agents or "tool" in agents
    assert result.get("tool_result") or result.get("answer")


def test_supervisor_self_ask_mode(runtime_supervisor):
    """P33b-v2：比较问题 → TaskPlan compare/query goal → worker（澄清路径也合法）。"""
    result = _run(runtime_supervisor, "玻尿酸精华和水乳哪个好", "test-sv-ask")
    if _ended_at_clarify(result):
        assert result.get("clarification")
        return
    assert result["mode"] in ("direct", "delegate", "clarify")
    agents = [d["agent"] for d in result["dispatch"]]
    assert "supervise" in agents
    assert "worker" in agents or "self_ask" in agents or "retrieve" in agents
    assert result["answer"]


def test_supervisor_pae_mode(runtime_supervisor):
    """P33b-v2：下单引导 → TaskPlan order goal → worker 调工具（澄清/缺信息路径也合法）。"""
    result = _run(runtime_supervisor, "怎么下单购买", "test-sv-pae")
    if _ended_at_clarify(result):
        assert result.get("clarification")
        return
    assert result["mode"] in ("direct", "delegate", "clarify")
    agents = [d["agent"] for d in result["dispatch"]]
    assert "supervise" in agents
    assert "worker" in agents or "tool" in agents


def test_supervisor_per_mode(runtime_supervisor):
    """P33b-v2：推荐问题 → TaskPlan 决策 + worker（反思节点已入图；澄清/缺信息路径也合法）。"""
    result = _run(runtime_supervisor, "敏感肌适合哪款精华，帮我推荐一下", "test-sv-per")
    if _ended_at_clarify(result):
        assert result.get("clarification")
        return
    assert result["mode"] in ("direct", "delegate", "clarify")
    agents = [d["agent"] for d in result["dispatch"]]
    assert "supervise" in agents
    assert "worker" in agents or "retrieve" in agents
