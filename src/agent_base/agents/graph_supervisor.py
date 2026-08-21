"""LangGraph 版多 Agent 编排（v0.47）。

边界：
- 多 Agent 编排用 LangGraph 框架：StateGraph 建图 / 条件边 / create_react_agent 工具 Agent / ToolNode。
- 意图识别、检索、澄清、Enrich、记忆等确定性 RAG 逻辑保持自研（复用 supervisor.py 子 Agent 函数）。

接口与自研 ``run_supervisor_plan`` 完全一致（dispatch/trace/sources/evidence...），
streaming.py 与前端执行链零破坏。
"""

from __future__ import annotations

import operator
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.runtime import Runtime
from langgraph.types import Send

from agent_base.agents.supervisor import (
    AgentResult,
    _detect_mode,
    _extract_current_product,
    clarify_agent,
    enrich_agent,
    intent_agent,
    memory_agent,
    retrieve_agent,
    self_ask_agent,
    tool_agent,
)

# LangGraph 官方运行上下文（context_schema + Runtime[Context] 注入节点），
# 替代手写 ContextVar——官方 reference：StateGraph(context_schema=...) + graph.invoke(context=...)
class RuntimeContext(TypedDict):
    """运行期不可变上下文：on_agent 事件回调 + 基础设施配置。

    不入 state（保持 state 可 msgpack 序列化——PostgresSaver checkpoint 依赖），
    Send 子任务（worker）共享同一 context。
    """
    on_agent: Callable[..., None] | None
    runtime: dict[str, Any]
    rerank_cfg: dict[str, Any]
    intent_classifier_cfg: dict[str, Any] | None
    llm_cfg: dict[str, Any]


_graph: Any = None  # 编译后的 StateGraph 缓存
_pg_ctx: Any = None  # PostgresSaver.from_conn_string 生成器（模块级持有防 GC，保连接存活）


# P1: Multi Schema — 外部输入/输出收窄，内部 SupervisorState 为完整 GlobalState
class InputState(TypedDict, total=False):
    """外部调用者只能传这些字段。"""
    question: str
    session_id: str | None
    user_id: str | None


class OutputState(TypedDict, total=False):
    """外部调用者只能读取这些字段（run_supervisor_graph 返回 + streaming.py 消费）。"""
    answer: str
    sources: list[dict[str, Any]]
    clarification: str
    task_plan: dict[str, Any]
    tool_result: str
    trace: dict[str, Any]
    # 以下为 streaming.py / 前端执行链依赖字段
    clarify: bool
    dispatch: list[dict[str, Any]]
    mode: str
    goal_evidences: list[dict[str, Any]]
    reflect_rounds: int
    reflect_ok: bool
    intent: str
    evidence: str
    docs: list[Any]
    history: list[dict[str, Any]]
    profile: str


class SupervisorState(TypedDict, total=False):
    """LangGraph 全局状态：编排全链路共享的数据仓库。"""
    question: str
    session_id: str | None
    user_id: str | None
    constraints: dict[str, Any]
    history: list[dict[str, Any]]
    profile: str
    current_product: str | None
    enriched_q: str
    intent: str
    mode: str
    task_plan: dict[str, Any]                      # P33b-v2: 主 agent 任务编排意图
    goal_evidences: Annotated[list[dict[str, Any]], operator.add]  # worker 并行写合并
    reflect_rounds: int                            # 反思轮数（护栏上限）
    reflect_ok: bool                               # 反思是否通过
    steps: list[str]
    tool_result: str
    sources: list[dict[str, Any]]
    trace: dict[str, Any]
    docs: list[Any]
    evidence: str
    answer: str
    clarify: bool
    clarification: str
    dispatch: Annotated[list[dict[str, Any]], operator.add]
    remaining_steps: RemainingSteps  # P1: LangGraph 托管字段，自动扣减，路由提前退出


def _infra(runtime: Runtime[RuntimeContext]) -> dict[str, Any]:
    """从运行上下文读取基础设施（runtime/rerank/llm 配置），保持 state 可序列化。"""
    ctx = runtime.context or {}
    return {
        "runtime": ctx.get("runtime") or {},
        "rerank_cfg": ctx.get("rerank_cfg") or {},
        "intent_classifier_cfg": ctx.get("intent_classifier_cfg"),
        "llm_cfg": ctx.get("llm_cfg") or {},
    }


def _emit(
    runtime: Runtime[RuntimeContext],
    agent_name: str,
    status: str,
    duration_ms: int | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    cb = runtime.context.get("on_agent")
    if cb is not None:
        try:
            cb(agent_name, status, duration_ms, data)
        except Exception:
            pass


def _fallback(exc: Exception, name: str = "") -> AgentResult:
    """节点异常兜底：构造 fallback AgentResult（不阻断图执行）。"""
    try:
        from agent_base.monitoring.logger import log_event
        log_event("ERROR", "graph_supervisor", "node_error", {"node": name, "error": str(exc)[:200]})
    except Exception:
        pass
    return AgentResult(status="fallback", meta={"error": str(exc)[:120]})


def _finish_node(
    runtime: Runtime[RuntimeContext],
    name: str,
    res: AgentResult,
    t0: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """节点收尾：写耗时 + on_agent done 事件 + 组装 dispatch 与状态增量。

    统一返回 LangGraph 官方 Partial<State>（节点签名 State -> Partial<State>）。
    """
    ms = int((time.perf_counter() - t0) * 1000)
    res.meta["duration_ms"] = ms
    data = None
    if name == "tool" and res.data.get("used"):
        data = {"tool": res.data.get("tool"), "result": res.data.get("result")}
    _emit(runtime, name, "done", ms, data=data)
    return {"dispatch": [{"agent": name, **res.to_dict()}], **extra}


# ---------- 节点：确定性 RAG 逻辑（自研，复用 supervisor 子 Agent） ----------


def _memory(state: SupervisorState, runtime: Runtime[RuntimeContext]) -> dict[str, Any]:
    """记忆节点：会话历史 + 用户画像 + 当前商品锚定（官方签名 State -> Partial<State>）。"""
    _emit(runtime, "memory", "running")
    t0 = time.perf_counter()
    try:
        res = memory_agent(state.get("session_id"), state.get("user_id"))
    except Exception as exc:  # noqa: BLE001
        res = _fallback(exc, "memory")
    history = res.data.get("history") or []
    profile = str(res.data.get("profile") or "")
    return _finish_node(runtime, "memory", res, t0, {
        "history": history,
        "profile": profile,
        "current_product": _extract_current_product(history),
    })


def _enrich(state: SupervisorState, runtime: Runtime[RuntimeContext]) -> dict[str, Any]:
    """改写节点：别名扩展 + 多轮指代补全（官方签名 State -> Partial<State>）。"""
    _emit(runtime, "enrich", "running")
    t0 = time.perf_counter()
    try:
        res = enrich_agent(state["question"], current_product=state.get("current_product"))
    except Exception as exc:  # noqa: BLE001
        res = _fallback(exc, "enrich")
    return _finish_node(runtime, "enrich", res, t0, {
        "enriched_q": str(res.data.get("rewritten") or state["question"]),
    })


def _intent(state: SupervisorState, runtime: Runtime[RuntimeContext]) -> dict[str, Any]:
    """意图节点：检索意图判定（自研白名单，官方签名 State -> Partial<State>）。"""
    _emit(runtime, "intent", "running")
    t0 = time.perf_counter()
    try:
        res = intent_agent(state["question"])
    except Exception as exc:  # noqa: BLE001
        res = _fallback(exc, "intent")
    intent = str(res.data.get("intent") or "general_qa")
    try:
        from agent_base.monitoring.logger import log_event
        log_event("INFO", "graph_supervisor", "intent_decided", {
            "intent": intent,
            "source": res.meta.get("source", "rule") if res.meta else "rule",
        })
    except Exception:
        pass
    return _finish_node(runtime, "intent", res, t0, {"intent": intent})


def _clarify(state: SupervisorState, runtime: Runtime[RuntimeContext]) -> dict[str, Any]:
    """澄清节点：完整性检查——必须澄清 / 可降级推荐 / 可直接回答（官方签名 State -> Partial<State>）。"""
    _emit(runtime, "clarify", "running")
    t0 = time.perf_counter()
    constraints = state.get("constraints") or {}
    try:
        res = clarify_agent(
            state["question"],
            product_name=constraints.get("product_name"),
            product_spec=constraints.get("product_spec"),
        )
    except Exception as exc:  # noqa: BLE001
        res = _fallback(exc, "clarify")
    need = res.status == "clarify"
    mode = _detect_mode(state["question"], state.get("intent", "general_qa"))
    return _finish_node(runtime, "clarify", res, t0, {
        "clarify": need,
        "clarification": str(res.data.get("clarification_question") or ""),
        "mode": mode,
    })


# P2: 多源知识库检索（worker 按 action 分发）


def _search_reviews(product_id: str) -> list[dict[str, Any]]:
    """查评价库（wrapper，失败静默返回 []）。"""
    try:
        from agent_base.retrieval.multi_source import search_reviews
        return search_reviews(product_id, top_k=5)
    except Exception:
        return []


def _search_combos(scenario: str | None) -> list[dict[str, Any]]:
    """查搭配方案库（wrapper，失败静默返回 []）。"""
    try:
        from agent_base.retrieval.multi_source import search_combos
        return search_combos(scenario, top_k=5)
    except Exception:
        return []


def _search_cases(product_id: str, skin_type: str | None = None) -> list[dict[str, Any]]:
    """查客户案例库（wrapper，失败静默返回 []）。"""
    try:
        from agent_base.retrieval.multi_source import search_cases
        return search_cases(product_id, skin_type=skin_type, top_k=5)
    except Exception:
        return []


# ---------- 主 agent 任务编排（P33b-v2：TaskPlan + Send 动态分发） ----------


def _build_task_plan_model():
    """构建 TaskPlan 结构化输出模型（bind_tools，DeepSeek 不支持 with_structured_output）。

    Returns:
        bound model（bind_tools([TaskPlan])）；LLM 不可用返回 None。
    """
    try:
        from agent_base.config import deep_get, load_yaml
        from agent_base.llms import build_chat_model
        from agent_base.structured import TaskPlan

        cfg = load_yaml("configs/app.yaml") or {}
        llm_cfg = deep_get(cfg, "llm", {}) or {}
        provider = llm_cfg.get("provider", "none")
        if provider in {"none", "off", "false"}:
            return None
        model = build_chat_model(
            provider=provider,
            model=llm_cfg.get("model"),
            base_url=llm_cfg.get("base_url"),
            api_key_env=llm_cfg.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"),
            temperature=0.0,
        )
        if model is None:
            return None
        return model.bind_tools([TaskPlan])
    except Exception:
        return None


def _parse_task_plan(question: str, intent: str, llm_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """三层兜底解析 TaskPlan：bind_tools → 文本 JSON → 正则构造。

    Args:
        question: 用户问题。
        intent: 检索意图（已有 intent 节点产出）。
        llm_cfg: LLM 配置（可选，默认读 app.yaml）。

    Returns:
        TaskPlan dict；任何层失败都返回可用的兜底 plan（单 goal + delegate）。
    """
    from agent_base.structured import TaskPlan, parse_json_or_none

    # 层 1：bind_tools 强制 function calling（已实测 DeepSeek 可用）
    try:
        bound = _build_task_plan_model()
        if bound is not None:
            from agent_base.prompts import get_prompt

            system = get_prompt("supervisor", "system") if _has_prompt("supervisor") else (
                "你是电商客服任务编排助手。把用户问题拆成结构化任务计划，"
                "只输出 JSON：{\"goals\":[{\"action\":\"query\",\"targets\":[\"商品名\"]}],"
                "\"strategy\":\"delegate\",\"complexity\":3}"
            )
            resp = bound.invoke([
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ])
            tool_calls = getattr(resp, "tool_calls", None)
            if tool_calls:
                plan = TaskPlan.model_validate(tool_calls[0]["args"])
                return plan.model_dump()
    except Exception:
        pass

    # 层 2：文本 JSON 解析（复用项目 parse_json_or_none）
    try:
        from agent_base.llms import build_chat_model
        from agent_base.prompts import get_prompt

        cfg = llm_cfg or {}
        provider = cfg.get("provider", "langchain")
        model = build_chat_model(
            provider=provider,
            model=cfg.get("model", "deepseek-v4-flash"),
            base_url=cfg.get("base_url", "https://api.deepseek.com"),
            api_key_env=cfg.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"),
            temperature=0.0,
        )
        if model is not None:
            system = get_prompt("supervisor", "system") if _has_prompt("supervisor") else (
                "你是电商客服任务编排助手。把用户问题拆成结构化任务计划，"
                "只输出 JSON：{\"goals\":[{\"action\":\"query\",\"targets\":[\"商品名\"]}],"
                "\"strategy\":\"delegate\",\"complexity\":3}"
            )
            # LCEL 官方链：直接以消息列表调用（system 含 JSON 花括号，避免模板变量解析）
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [SystemMessage(content=system), HumanMessage(content=question)]
            try:
                resp = model.invoke(messages)
                resp_text = str(getattr(resp, "content", "") or resp or "")
            except Exception:
                resp_text = ""
            data = parse_json_or_none(resp_text)
            if isinstance(data, dict):
                plan = TaskPlan.model_validate(data)
                return plan.model_dump()
    except Exception:
        pass

    # 层 3：正则构造兜底（复用 _detect_mode 语义，保证任何情况都有可用 plan）
    mode = _detect_mode(question, intent)
    action = "compare" if mode == "self_ask" else ("order" if intent == "aftersale" else "query")
    return {
        "goals": [{"action": action, "targets": [], "constraints": []}],
        "strategy": "delegate",
        "complexity": 2,
        "requires_reflection": False,
        "missing_info": [],
    }


def _has_prompt(name: str) -> bool:
    """检查 prompts 配置是否存在该键。"""
    try:
        from agent_base.config import load_yaml

        prompts = load_yaml("configs/prompts_ecommerce.yaml") or {}
        return name in prompts
    except Exception:
        return False


def _supervise(state: SupervisorState, runtime: Runtime[RuntimeContext]) -> dict[str, Any]:
    """主 agent 任务理解节点：产出 TaskPlan（三层兜底，替代 _detect_mode 正则路由）。

    官方节点签名：State -> Partial<State>。
    """
    _emit(runtime, "supervise", "running")
    t0 = time.perf_counter()
    try:
        plan = _parse_task_plan(
            # 用改写后的问题（enrich 已做指代补全/别名扩展），TaskPlan 才能正确识别实体
            state.get("enriched_q") or state["question"],
            state.get("intent", "general_qa"),
            _infra(runtime)["llm_cfg"],
        )
        # 规则兜底：闲聊/寒暄 → 强制 direct+chat（LLM 判定不可靠时保底）
        from agent_base.agents.emotion import looks_like_chat

        if looks_like_chat(state["question"]):
            plan = {
                "goals": [{"action": "chat", "targets": [], "constraints": []}],
                "strategy": "direct",
                "complexity": 1,
                "requires_reflection": False,
                "missing_info": [],
            }
        # 规则兜底：订单/物流/库存类问题 → 强制 order goal（LLM 易把工具类问题
        # 判成 direct 单检索，绕过 worker 的 order 工具分支，导致工具不调用）
        elif _looks_like_order(state["question"]):
            plan = {
                "goals": [{"action": "order", "targets": [], "constraints": []}],
                "strategy": "delegate",
                "complexity": 2,
                "requires_reflection": False,
                "missing_info": [],
            }
        # 兼容前端执行链：mode 字段从 TaskPlan 推导
        mode = "direct" if plan.get("strategy") == "direct" else "delegate"
        try:
            from agent_base.monitoring.logger import log_event
            log_event("INFO", "graph_supervisor", "task_plan_decided", {
                "strategy": plan.get("strategy"),
                "complexity": plan.get("complexity"),
                "goals_count": len(plan.get("goals") or []),
                "requires_reflection": bool(plan.get("requires_reflection")),
            })
        except Exception:
            pass
        # P1: RemainingSteps 由 LangGraph 托管自动扣减，节点无需手动递减
        # TaskPlan 判 clarify（缺信息）→ 走澄清：置位 clarify + 生成追问。
        # （此前只改 mode 不置 clarify，graph END 后 clarification 为空 → 用户收到空回复）
        if plan.get("strategy") == "clarify" or plan.get("missing_info"):
            missing = plan.get("missing_info") or []
            if missing:
                clarification = "为了更准确地帮您解答，请补充：" + "、".join(str(m) for m in missing) + "。"
            else:
                # 复用前置 clarify 节点已生成的追问；没有则通用兜底
                clarification = state.get("clarification") or "您能再补充一些信息吗？我好更准确地为您解答。"
            return _finish_node(runtime, "supervise", AgentResult(
                data={"task_plan": plan, "goals": plan.get("goals", [])},
                confidence=1.0,
                meta={"agent": "supervise", "mode": "clarify", "complexity": plan.get("complexity")},
            ), t0, {
                "task_plan": plan,
                "mode": "clarify",
                "clarify": True,
                "clarification": clarification,
                "reflect_rounds": 0,
                "reflect_ok": False,
            })
        return _finish_node(runtime, "supervise", AgentResult(
            data={"task_plan": plan, "goals": plan.get("goals", [])},
            confidence=1.0,
            meta={"agent": "supervise", "mode": mode, "complexity": plan.get("complexity")},
        ), t0, {
            "task_plan": plan,
            "mode": mode,
            "reflect_rounds": 0,
            "reflect_ok": False,
        })
    except Exception as exc:  # noqa: BLE001
        try:
            from agent_base.monitoring.logger import log_event
            log_event("ERROR", "graph_supervisor", "task_plan_failed", {"error": str(exc)[:200]})
        except Exception:
            pass
        return _finish_node(runtime, "supervise", AgentResult(
            status="fallback", data={"task_plan": {}}, meta={"error": str(exc)[:120]},
        ), t0, {
            "mode": "direct",
            "task_plan": {"goals": [], "strategy": "direct", "complexity": 1, "requires_reflection": False, "missing_info": []},
            "reflect_rounds": 0,
            "reflect_ok": False,
        })


def _looks_like_order(question: str) -> bool:
    """工具类意图启发式：订单/物流/库存类问题强制走 order 工具分支。

    LLM TaskPlan 易把这类问题判成 direct 单检索，绕过 worker 的 order 分支，
    导致 get_order/check_stock 等工具不被调用（supervisor 编排路径下订单查询失效）。
    """
    q = str(question or "")
    return any(k in q for k in (
        "订单", "下单", "物流", "发货", "快递", "到货", "签收",
        "库存", "有货", "缺货", "补货", "查单",
    )) or "ORD" in q.upper()


def _route_after_supervise(state: SupervisorState) -> str | Sequence[Send]:
    """supervise 出口条件边：direct → retrieve；delegate → Send 动态分发 worker。

    支持返回 str（走 path_map）或 Sequence[Send]（动态并行），
    一个条件边同时处理两条路径（Orchestrator-worker 模式）。
    """
    # P1: RemainingSteps 托管字段——剩余不足 3 步时优雅退出，避免 GraphRecursionError
    _rs = state.get("remaining_steps", 30)
    if _rs is not None and int(_rs) < 3:
        return END
    if state.get("clarify"):
        return END
    plan = state.get("task_plan") or {}
    strategy = plan.get("strategy", "delegate")
    # TaskPlan 判 clarify（缺信息）→ 走澄清（复用前置 clarify 节点的 clarification）
    if strategy == "clarify" or plan.get("missing_info"):
        return END
    complexity = int(plan.get("complexity", 2) or 2)
    goals = plan.get("goals") or []
    # direct：单 goal + 低复杂度 → 纯对话（chat goal）直接生成；简单事实问题走单检索
    try:
        from agent_base.config import deep_get, load_yaml

        _oc = load_yaml("configs/app.yaml") or {}
        _direct_max = int(deep_get(_oc, "orchestration.direct_max_complexity", 2))
    except Exception:
        _direct_max = 2
    if strategy == "direct" and len(goals) <= 1 and complexity <= _direct_max:
        action = (goals[0] or {}).get("action", "query") if goals else "query"
        # chat（闲聊/寒暄，与商品资料无关）→ 跳过检索直接生成，零成本
        return "generate" if action == "chat" else "retrieve"
    # delegate：Send 动态分发（任务数量运行时才知道）
    common = {
        "question": state.get("question", ""),
        "enriched_q": state.get("enriched_q") or state.get("question", ""),
        "intent": state.get("intent", "general_qa"),
        "constraints": state.get("constraints") or {},
        "current_product": state.get("current_product"),
    }
    sends = [Send("worker", {**common, "goal": g}) for g in goals if isinstance(g, dict)]
    if not sends:
        # 兜底：空 goals → 单检索 worker
        sends = [Send("worker", {**common, "goal": {"action": "query", "targets": [], "constraints": []}})]
    return sends


def _worker(state: SupervisorState, runtime: Runtime[RuntimeContext]) -> dict[str, Any]:
    """通用 worker 节点：按 goal.action 分发到现有子 agent（Orchestrator-worker）。

    官方节点签名：State -> Partial<State>。
    过渡期复用现有确定性函数（retrieve_agent / self_ask_agent / tool_agent），
    后续可逐个升级为 create_agent（LLM + 专属工具）。
    """
    _emit(runtime, "worker", "running")
    t0 = time.perf_counter()
    try:
        goal = state.get("goal") or {}
        action = str(goal.get("action") or "query")
        question = state.get("enriched_q") or state.get("question", "")
        constraints = state.get("constraints") or {}
        rt = _infra(runtime)["runtime"]

        # compare → self_ask（拆子问题分别检索）；order → 工具 + 检索；其余 → 单检索
        if action == "compare":
            _emit(runtime, "self_ask", "running")
            t1 = time.perf_counter()
            try:
                res = self_ask_agent(
                    question,
                    rt.get("vector_store"),
                    k=max(4, constraints.get("top_k", 6)),
                )
            finally:
                _emit(runtime, "self_ask", "done", int((time.perf_counter() - t1) * 1000))
            trace = res.data.get("trace") or {}
            return _finish_node(runtime, "worker", res, t0, {
                "goal_evidences": [{
                    "goal": goal,
                    "sources": res.sources,
                    "trace": trace,
                    "evidence": "\n\n".join(
                        f"[{s.get('section', '商品')}] {str(s.get('content', ''))[:400]}" for s in res.sources[:6]
                    ),
                }],
            })

        if action == "order":
            _emit(runtime, "tool", "running")
            t1 = time.perf_counter()
            tool_result = ""
            try:
                # 官方路线：create_react_agent（LLM 自主决策调工具 + PII/HITL/摘要中间件）
                from agent_base.agents.official_agents import build_official_worker_agent

                official = build_official_worker_agent(_infra(runtime)["llm_cfg"])
                if official is not None:
                    resp = official.invoke({"messages": [{"role": "user", "content": question}]})
                    # HITL 中断：工具均为只读查询（订单/库存/物流），无需人工确认，
                    # 直接降级到手写工具路由获取结果（官方链路能力已展示：LLM 已自主选择工具）
                    if isinstance(resp, dict) and resp.get("__interrupt__"):
                        raise RuntimeError("HITL interrupt -> fallback to deterministic tool routing")
                    msgs = resp.get("messages", []) if isinstance(resp, dict) else []
                    if msgs:
                        last = msgs[-1]
                        content = getattr(last, "content", "") if not isinstance(last, dict) else last.get("content", "")
                        if content:
                            tool_result = str(content)[:800]
            except Exception:
                tool_result = ""
            try:
                if not tool_result:
                    # 兜底：手写正则工具路由（官方 agent 不可用时保底）
                    tool_res = tool_agent(question, "aftersale")
                    tool_result = str(tool_res.data.get("result") or "") if tool_res.data.get("used") else ""
            finally:
                _emit(runtime, "tool", "done", int((time.perf_counter() - t1) * 1000))
            ret = retrieve_agent(
                question,
                rt.get("vector_store"),
                rt.get("summary_store"),
                constraints,
                _infra(runtime)["rerank_cfg"],
                _infra(runtime)["intent_classifier_cfg"],
                sparse_store=rt.get("sparse_store"),
            )
            trace = ret.data.get("trace") or {}
            return _finish_node(runtime, "worker", ret, t0, {
                "goal_evidences": [{
                    "goal": goal,
                    "sources": ret.sources,
                    "trace": trace,
                    "tool_result": tool_result,
                    "evidence": "\n\n".join(
                        f"[{s.get('section', '商品')}] {str(s.get('content', ''))[:400]}" for s in ret.sources[:6]
                    ),
                }],
            })

        # P2: 多源知识库——按 goal.action 分发到专属数据源
        _multi_sources: list[dict[str, Any]] = []
        _source_type = ""
        if action == "review":
            _product_id = (goal.get("targets") or [None])[0] or constraints.get("product_name", "")
            _multi_sources = _search_reviews(_product_id)
            _source_type = "reviews"
        elif action == "combine":
            _scenario = (goal.get("constraints") or [None])[0]  # 干皮/油皮/通勤等
            _multi_sources = _search_combos(_scenario)
            _source_type = "combos"
        elif action == "usage":
            _product_id = (goal.get("targets") or [None])[0] or constraints.get("product_name", "")
            _skin_type = None
            for c in (goal.get("constraints") or []):
                if c in ("干皮", "油皮", "敏感肌", "混合皮"):
                    _skin_type = c
                    break
            _multi_sources = _search_cases(_product_id, skin_type=_skin_type)
            _source_type = "cases"

        # price/query：先精确查商品工具（catalog + 商品长文），
        # 避免价格/属性类问题被向量检索张冠李戴；结果注入 evidence 前置。
        tool_result = ""
        if action in ("price", "query"):
            _emit(runtime, "tool", "running")
            t1 = time.perf_counter()
            _tool_ok = False
            _tool_err = ""
            try:
                from agent_base.agents.tools_ecommerce import get_product_info

                _q = (goal.get("targets") or [None])[0] or question
                tool_result = get_product_info.invoke({"product_query": _q})
                _tool_ok = True
            except Exception as _exc:
                tool_result = ""
                _tool_err = str(_exc)[:500]
            finally:
                try:
                    from agent_base.monitoring.usage import record_tool_call

                    record_tool_call(
                        agent="supervisor_worker",
                        tool_name="get_product_info",
                        params={"product_query": (goal.get("targets") or [None])[0] or question},
                        ok=_tool_ok,
                        error=_tool_err,
                        latency_ms=int((time.perf_counter() - t1) * 1000),
                        result_preview=str(tool_result)[:500],
                    )
                except Exception:
                    pass
                _emit(runtime, "tool", "done", int((time.perf_counter() - t1) * 1000))

        # 商品库兜底检索（所有路径都需要——多源补充 + 商品库保底）
        ret = retrieve_agent(
            question,
            rt.get("vector_store"),
            rt.get("summary_store"),
            constraints,
            _infra(runtime)["rerank_cfg"],
            _infra(runtime)["intent_classifier_cfg"],
            sparse_store=rt.get("sparse_store"),
        )
        trace = ret.data.get("trace") or {}
        # 多源结果前置 + 商品库兜底去重
        _all_sources = _multi_sources + [s for s in ret.sources if s.get("section") not in {"", "商品"}][:6]
        _evidence_parts = ([f"[商品工具] {tool_result}"] if tool_result else []) + [
            f"[{s.get('section', '商品')}] {str(s.get('content', ''))[:400]}" for s in _all_sources[:6]
        ]
        return _finish_node(runtime, "worker", ret, t0, {
            "goal_evidences": [{
                "goal": goal,
                "sources": _all_sources[:6],
                "trace": trace,
                "source_type": _source_type,
                "tool_result": tool_result,
                "evidence": "\n\n".join(_evidence_parts),
            }],
        })
    except Exception as exc:  # noqa: BLE001
        return _finish_node(runtime, "worker", _fallback(exc, "worker"), t0, {})


def _reflect(state: SupervisorState, runtime: Runtime[RuntimeContext]) -> dict[str, Any]:
    """反思节点（Evaluator-optimizer）：检查每个 goal 是否有证据覆盖。

    官方节点签名：State -> Partial<State>。
    缺证据的 goal → 补检索（复用 _worker 单 goal 逻辑），上限 reflect_rounds ≤ 2。
    """
    _emit(runtime, "reflect", "running")
    t0 = time.perf_counter()
    try:
        plan = state.get("task_plan") or {}
        goals = plan.get("goals") or []
        evidences = state.get("goal_evidences") or []
        rounds = int(state.get("reflect_rounds", 0) or 0)

        try:
            from agent_base.config import deep_get, load_yaml

            _oc = load_yaml("configs/app.yaml") or {}
            _max_rounds = int(deep_get(_oc, "orchestration.max_reflect_rounds", 2))
        except Exception:
            _max_rounds = 2
        if rounds >= _max_rounds or not goals:
            return _finish_node(runtime, "reflect", AgentResult(
                data={"reflect_ok": True}, confidence=1.0, meta={"agent": "reflect"},
            ), t0, {"reflect_ok": True})

        # 检查：每个 goal 是否已有 worker/补检索产出（sources 非空即视为覆盖）。
        # 不依赖 target 字符串在 evidence 文本中的子串匹配——商品全名常不出现在
        # 资料正文（资料用简称/卖点表述），子串匹配误判率高，会把已覆盖 goal
        # 判为 uncovered 触发无意义补检索（还容易补进无关商品内容）。
        covered_keys: set[tuple[str, tuple[str, ...]]] = set()
        for e in evidences:
            gl = e.get("goal") or {}
            if e.get("sources"):
                covered_keys.add(
                    (str(gl.get("action")), tuple(str(t) for t in (gl.get("targets") or [])))
                )
        uncovered = [
            g for g in goals
            if (str(g.get("action")), tuple(str(t) for t in (g.get("targets") or [])))
            not in covered_keys
        ]

        if not uncovered:
            return _finish_node(runtime, "reflect", AgentResult(
                data={"reflect_ok": True, "checked": len(goals)}, confidence=1.0, meta={"agent": "reflect"},
            ), t0, {"reflect_ok": True})

        # 缺证据 → 补检索（记录一轮，最多 2 轮）
        new_rounds = rounds + 1
        question = state.get("enriched_q") or state.get("question", "")
        constraints = state.get("constraints") or {}
        rt = _infra(runtime)["runtime"]
        added: list[dict[str, Any]] = []
        for g in uncovered[:2]:
            try:
                ret = retrieve_agent(
                    " ".join(g.get("targets", [])) or question,
                    rt.get("vector_store"),
                    rt.get("summary_store"),
                    constraints,
                    _infra(runtime)["rerank_cfg"],
                    _infra(runtime)["intent_classifier_cfg"],
                    sparse_store=rt.get("sparse_store"),
                )
                if ret.sources:
                    added.append({
                        "goal": g,
                        "sources": ret.sources,
                        "evidence": "\n\n".join(
                            f"[{s.get('section', '商品')}] {str(s.get('content', ''))[:400]}" for s in ret.sources[:6]
                        ),
                    })
            except Exception:
                continue

        return _finish_node(runtime, "reflect", AgentResult(
            data={"reflect_ok": False, "uncovered": [g.get("action") for g in uncovered], "rounds": new_rounds},
            confidence=0.8,
            meta={"agent": "reflect"},
        ), t0, {
            "reflect_rounds": new_rounds,
            "goal_evidences": added,  # reducer 自动追加，只返回新增
            "reflect_ok": False,
        })
    except Exception as exc:  # noqa: BLE001
        return _finish_node(runtime, "reflect", _fallback(exc, "reflect"), t0, {})


def _route_after_reflect(state: SupervisorState) -> str:
    """反思后条件边：有缺漏且未超上限 → 再补一轮；否则收尾。"""
    try:
        from agent_base.config import deep_get, load_yaml

        _oc = load_yaml("configs/app.yaml") or {}
        _max_rounds = int(deep_get(_oc, "orchestration.max_reflect_rounds", 2))
    except Exception:
        _max_rounds = 2
    if not state.get("reflect_ok") and int(state.get("reflect_rounds", 0) or 0) < _max_rounds:
        return "reflect"
    return "finalize"


def _retrieve(state: SupervisorState, runtime: Runtime[RuntimeContext]) -> dict[str, Any]:
    """单检索节点：direct 路径的确定性检索（官方签名 State -> Partial<State>）。"""
    _emit(runtime, "retrieve", "running")
    t0 = time.perf_counter()
    constraints = state.get("constraints") or {}
    rt = _infra(runtime)["runtime"]
    try:
        res = retrieve_agent(
            state["enriched_q"],
            rt["vector_store"],
            rt.get("summary_store"),
            constraints,
            _infra(runtime)["rerank_cfg"],
            _infra(runtime)["intent_classifier_cfg"],
            sparse_store=rt.get("sparse_store"),
        )
    except Exception as exc:  # noqa: BLE001
        res = _fallback(exc, "retrieve")
    trace = res.data.get("trace") or {}
    try:
        from agent_base.monitoring.logger import log_event
        docs_count = len(trace.get("results") or res.sources)
        strategy = (trace.get("decision") or {}).get("strategy", "unknown") if isinstance(trace, dict) else "unknown"
        log_event("INFO", "graph_supervisor", "retrieve_done", {
            "strategy": str(strategy),
            "docs_count": docs_count,
        })
    except Exception:
        pass
    return _finish_node(runtime, "retrieve", res, t0, {
        "trace": trace,
        "docs": trace.get("results") or [],
        "sources": res.sources,
    })


def _finalize(state: SupervisorState) -> dict[str, Any]:
    # 优先从 goal_evidences 聚合（worker 并行产物）；否则回退直接 sources
    evidences = state.get("goal_evidences") or []
    if evidences:
        sources = []
        tool_result = ""
        seen: set[str] = set()
        for ev in evidences:
            if ev.get("tool_result") and not tool_result:
                tool_result = str(ev.get("tool_result"))
            for s in (ev.get("sources") or []):
                key = str(s.get("section", "")) + str(s.get("content", ""))[:80]
                if key not in seen:
                    seen.add(key)
                    sources.append(s)
        sources = sources[:6]
        evidence = "\n\n".join(
            f"[{s.get('section', '商品')}] {str(s.get('content', ''))[:400]}" for s in sources[:6]
        )
        ret_state: dict[str, Any] = {"evidence": evidence, "sources": sources}
        if tool_result:
            ret_state["tool_result"] = tool_result
        return ret_state
    else:
        sources = state.get("sources") or []
    evidence = "\n\n".join(
        f"[{s.get('section', '商品')}] {str(s.get('content', ''))[:400]}" for s in sources[:6]
    )
    # 写回 sources（前端执行链展示用）；worker 路径下 sources 从 goal_evidences 聚合
    return {"evidence": evidence, "sources": sources}


def _generate(state: SupervisorState, runtime: Runtime[RuntimeContext]) -> dict[str, Any]:
    """生成节点：LLM 生成最终回答（官方 create_agent 优先，LCEL 模板兜底；动态温度）。"""
    from agent_base.agents.supervisor import build_generate_context, compute_generation_temperature, generate_agent

    _emit(runtime, "generate", "running")
    t0 = time.perf_counter()
    # 动态温度（P0-2）：意图基底 + 情绪调节；情绪用规则通道检测（不调 LLM，成本为零）
    try:
        from agent_base.agents.emotion import detect_emotion

        _emotion = detect_emotion(state["question"]).get("label", "neutral")
    except Exception:
        _emotion = "neutral"
    _temperature = compute_generation_temperature(
        intent=state.get("intent", "general_qa"),
        emotion=_emotion,
    )
    # 温度注入 llm_cfg（官方 create_agent 与 LCEL generate_agent 均从 llm_cfg 读 temperature）
    _llm_cfg = dict(_infra(runtime)["llm_cfg"])
    _llm_cfg["temperature"] = _temperature
    res = None
    try:
        # 官方路线：create_agent 受控生成（PII 中间件保护输出层）
        from agent_base.agents.official_agents import build_official_generate_agent

        official = build_official_generate_agent(_llm_cfg)
        if official is not None:
            context_block = build_generate_context(
                state["question"],
                state.get("evidence") or "",
                history=state.get("history"),
                profile=state.get("profile") or "",
                tool_result=state.get("tool_result") or "",
            )
            resp = official.invoke({"messages": [{"role": "user", "content": context_block}]})
            msgs = resp.get("messages", []) if isinstance(resp, dict) else []
            if msgs:
                last = msgs[-1]
                content = getattr(last, "content", "") if not isinstance(last, dict) else last.get("content", "")
                if content:
                    res = AgentResult(
                        data={"answer": str(content)},
                        confidence=1.0,
                        meta={"agent": "generate", "mode": "official_agent"},
                    )
    except Exception:
        res = None
    if res is None:
        # 兜底：LCEL generate_agent（LLM 不可用 → 模板摘要）
        try:
            res = generate_agent(
                state["question"],
                state.get("evidence") or "",
                history=state.get("history"),
                profile=state.get("profile") or "",
                tool_result=state.get("tool_result") or "",
                llm_cfg=_llm_cfg,
            )
        except Exception as exc:  # noqa: BLE001
            res = _fallback(exc, "generate")
    answer = str(res.data.get("answer") or "")
    try:
        from agent_base.monitoring.logger import log_event
        log_event("INFO", "graph_supervisor", "generate_done", {
            "answer_len": len(answer),
        })
    except Exception:
        pass
    return _finish_node(runtime, "generate", res, t0, {"answer": answer})


# ---------- 条件路由（LangGraph 条件边） ----------


def _route_after_clarify(state: SupervisorState) -> str:
    if state.get("clarify"):
        return END
    mode = state.get("mode")
    if mode == "self_ask":
        return "self_ask"
    if mode in ("pae", "per"):
        return "plan"
    if state.get("intent") == "aftersale":
        return "tool"
    return "retrieve"


# ---------- 建图 ----------


def _get_graph() -> Any:
    global _graph, _pg_ctx
    if _graph is not None:
        return _graph

    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.types import RetryPolicy

    # P1: 读取配置
    try:
        from agent_base.config import load_yaml
        _cfg = load_yaml("configs/app.yaml") or {}
        _oc = _cfg.get("orchestration") or {}
    except Exception:
        _oc = {}

    _retry_max = int(_oc.get("retry_max_attempts", 2))
    _timeout_s = int(_oc.get("node_timeout_seconds", 15))
    _max_steps = int(_oc.get("max_steps", 30))
    _pg_url_raw = _oc.get("checkpoint_db_url", "")
    _pg_url = "" if _pg_url_raw == "none" else (_pg_url_raw or _get_pg_url())

    g = StateGraph(
        state_schema=SupervisorState,
        context_schema=RuntimeContext,
        input_schema=InputState,
        output_schema=OutputState,
    )

    # P1: 关键 LLM 节点加 RetryPolicy（LLM 临时故障自动重试）
    _retry = RetryPolicy(max_attempts=_retry_max, initial_interval=2.0, backoff_factor=2.0)
    g.add_node("memory", _memory)
    g.add_node("enrich", _enrich)
    g.add_node("intent", _intent)
    g.add_node("clarify", _clarify)
    g.add_node("supervise", _supervise, retry_policy=_retry)
    g.add_node("worker", _worker, retry_policy=_retry)
    g.add_node("reflect", _reflect, retry_policy=_retry)
    g.add_node("retrieve", _retrieve)
    g.add_node("finalize", _finalize)
    g.add_node("generate", _generate)

    g.add_edge(START, "memory")
    g.add_edge("memory", "enrich")
    g.add_edge("enrich", "intent")
    g.add_edge("intent", "clarify")
    # P33b-v2：clarify 命中（缺信息）直接 END，不浪费 supervise LLM 调用
    g.add_conditional_edges(
        "clarify",
        lambda s: END if s.get("clarify") else "supervise",
        {"supervise": "supervise", END: END},
    )
    g.add_conditional_edges(
        "supervise",
        _route_after_supervise,
        {
            "retrieve": "retrieve",
            "worker": "worker",
            "generate": "generate",
            END: END,
        },
    )
    # worker（Send 并行）→ 全部完成后 reflect 反思
    g.add_edge("worker", "reflect")
    g.add_conditional_edges("reflect", _route_after_reflect, {"reflect": "reflect", "finalize": "finalize"})
    g.add_edge("retrieve", "finalize")
    g.add_edge("finalize", "generate")
    g.add_edge("generate", END)

    # P1: PostgresSaver 持久化（checkpoint 表自动创建）
    # from_conn_string 返回 context manager 生成器：必须模块级持有引用，
    # 否则函数返回后 GC 触发 with 退出关闭 psycopg 连接（the connection is closed）。
    checkpointer = None
    if _pg_url:
        try:
            _pg_ctx = PostgresSaver.from_conn_string(_pg_url)
            checkpointer = _pg_ctx.__enter__()
            checkpointer.setup()
        except Exception:
            pass  # checkpoint 不可用静默降级

    _graph = g.compile(
        checkpointer=checkpointer,
    )
    return _graph


def _get_pg_url() -> str:
    """从现有 PG 配置提取连接串（复用 ragdb）。"""
    try:
        from agent_base.storage.pg import _conn

        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT current_database()")
            db = cur.fetchone()[0]
            cur.execute("SHOW port")
            port = cur.fetchone()[0]
            cur.execute("SELECT current_user")
            user = cur.fetchone()[0]
        return f"postgresql://{user}:postgres@localhost:{port}/{db}"
    except Exception:
        return "postgresql://postgres:postgres@localhost:5432/ragdb"


def run_supervisor_graph(
    question: str,
    runtime: dict[str, Any],
    constraints: dict[str, Any],
    session_id: str | None = None,
    user_id: str | None = None,
    on_agent=None,
) -> dict[str, Any]:
    """LangGraph 版编排入口，接口与自研 ``run_supervisor_plan`` 完全一致。"""
    # v0.48: 转人工状态判断——人工接管中（active）不生成，排队中（pending）提示等待
    if session_id:
        try:
            from agent_base.config import deep_get, load_yaml
            from agent_base.storage.pg import handoff_check

            _cfg = load_yaml("configs/app.yaml") or {}
            handoff = handoff_check(
                session_id,
                pending_timeout=int(deep_get(_cfg, "handoff.pending_timeout_min", 15)) * 60,
                idle_timeout=int(deep_get(_cfg, "handoff.idle_timeout_min", 20)) * 60,
            )
        except Exception:
            handoff = None
        if handoff and handoff.get("status") in ("active", "pending"):
            return {
                "clarify": False,
                "clarification": "",
                "dispatch": [],
                "mode": "handoff",
                "intent": "handoff",
                "trace": {},
                "evidence": "",
                "answer": "",
                "sources": [],
                "docs": [],
                "tool_result": "",
                "history": [],
                "profile": "",
                "handoff": handoff,
            }
    # P1: 读取 RemainingSteps 上限
    try:
        from agent_base.config import load_yaml as _ly
        _oc = (_ly("configs/app.yaml") or {}).get("orchestration") or {}
        _max_steps = int(_oc.get("max_steps", 30))
    except Exception:
        _max_steps = 30

    graph = _get_graph()
    initial: dict[str, Any] = {
        "question": question,
        "session_id": session_id,
        "user_id": user_id,
        "constraints": constraints,
        "clarify": False,
        "dispatch": [],
        "goal_evidences": [],
        "reflect_rounds": 0,
        "reflect_ok": False,
        "sources": [],
        "trace": {},
        "docs": [],
        "tool_result": "",
        "evidence": "",
        "answer": "",
        "history": [],
        "profile": "",
    }
    # P1: 传 recursion_limit + thread_id 给 checkpointer
    _config: dict[str, Any] = {"recursion_limit": _max_steps}
    # 启用 PostgresSaver 后 checkpointer 必须带 thread_id：
    # 有 session 用会话 ID（跨轮恢复），无 session（如单测/匿名调用）生成匿名线程 ID。
    _thread_id = session_id or f"anon-{uuid.uuid4().hex[:12]}"
    _config["configurable"] = {"thread_id": _thread_id}
    out = graph.invoke(
        initial,
        config=_config if _config else None,
        # 官方 Runtime 上下文：on_agent 经 context 注入（替代旧 ContextVar，跨 Send 子任务透传）
        context={
            "on_agent": on_agent,
            "runtime": runtime,
            "rerank_cfg": runtime.get("rerank_config") or {},
            "intent_classifier_cfg": runtime.get("intent_classifier_config"),
            "llm_cfg": runtime.get("llm_config") or {},
        },
    )
    return {
        "clarify": bool(out.get("clarify")),
        "clarification": out.get("clarification") or "",
        "dispatch": out.get("dispatch") or [],
        "mode": out.get("mode", "supervisor"),
        "task_plan": out.get("task_plan") or {},
        "goal_evidences": out.get("goal_evidences") or [],
        "reflect_rounds": out.get("reflect_rounds", 0),
        "reflect_ok": out.get("reflect_ok", False),
        "intent": out.get("intent", "general_qa"),
        "trace": out.get("trace") or {},
        "evidence": out.get("evidence") or "",
        "answer": out.get("answer") or "",
        "sources": out.get("sources") or [],
        "docs": out.get("docs") or [],
        "tool_result": out.get("tool_result") or "",
        "history": out.get("history") or [],
        "profile": out.get("profile") or "",
    }
