"""LangGraph 节点函数（契约 P1-02）。

基础设施（vector_store/summary_store/配置）通过工厂闭包注入，
不放入 RagState（避免 Chroma 对象被 checkpoint msgpack 序列化）。
"""

from __future__ import annotations

from typing import Any

from agent_base.chains.safety_chain import SafetyAssessment, SafetyFinding, assess_safety
from agent_base.graphs.state import RagState
from agent_base.retrieval.advanced_retriever import retrieve_advanced
from agent_base.retrieval.retrieval_config import RetrievalConfig
from agent_base.retrieval.retrieval_policy import RetrievalDecision, build_retrieval_decision


# ── route（无外部依赖） ──

def route_node(state: RagState) -> dict[str, Any]:
    """意图路由 + 查询改写 + 检索策略决策。

    Args:
        state: 当前图状态（至少含 question 与可选商品约束）。

    Returns:
        写入 route / rewritten_query / metadata_filter / decision 的字段子集。
    """
    rewrite, decision = build_retrieval_decision(
        question=state["question"],
        product_name=state.get("product_name"),
        product_spec=state.get("product_spec"),
        category=state.get("category"),
    )
    return {
        "route": rewrite.route.to_dict(),
        "rewritten_query": decision.rewritten_query,
        "metadata_filter": decision.metadata_filter,
        "decision": decision.to_dict(),
    }


def make_sales_node():
    """会话级销售决策节点（确定性规则 + 阶段持久化）。

    输入：route（含 v2 扩展字段）+ conversation（可选 session_id）。
    输出：sales 上下文（stage/action/sales_strategy/guide 等），
    供 generate 节点注入受控生成层、写 trace。
    """

    def _sales(state: RagState) -> dict[str, Any]:
        route = state.get("route") or {}
        conversation = state.get("conversation") or {}
        session_id = conversation.get("session_id")
        try:
            from agent_base.agents.sales_stage import build_sales_context

            ctx = build_sales_context(
                dict(route),
                session_id,
                str(state.get("question") or ""),
            )
            return {"sales": ctx}
        except Exception as exc:
            return {
                "sales": {
                    "stage": "none",
                    "action": "answer",
                    "reason": f"sales_node: {type(exc).__name__}: {exc}"[:160],
                }
            }

    return _sales


# ── retrieve（工厂：闭包注入 vector_store / summary_store） ──

def make_retrieve_node(vector_store: Any, summary_store: Any | None = None, sparse_store: Any | None = None):
    """创建 retrieve_node（闭包注入向量库引用）。

    Args:
        vector_store: 原文 chunk 向量库。
        summary_store: 摘要向量库（可选）。
        sparse_store: BM25 稀疏向量库（可选，混合检索补充通道）。

    Returns:
        LangGraph 节点函数 (state) -> Partial[RagState]。
    """
    def _retrieve(state: RagState) -> dict[str, Any]:
        decision_dict = state.get("decision")
        if not decision_dict:
            return {"docs": [], "errors": _append(state, "缺少 decision，跳过检索")}
        decision = RetrievalDecision(**decision_dict)
        try:
            cfg = RetrievalConfig(
                top_k=decision.final_k,
                candidate_k=decision.candidate_k,
                rerank="none",
                product_name=state.get("product_name"),
                product_spec=state.get("product_spec"),
                category=state.get("category"),
                preserve_preferred_sections=False,
                fallback_without_filter=False,
            )
            trace = retrieve_advanced(
                vector_store,
                state["question"],
                cfg,
                summary_store=summary_store,
                sparse_store=sparse_store,
                decision=decision,
            )
            return {
                "trace": trace.to_dict(include_preview=True),
                "docs": list(trace.docs),
                "errors": _append(state, *trace.errors) if trace.errors else {},
            }
        except Exception as exc:
            return {"docs": [], "errors": _append(state, f"retrieve: {type(exc).__name__}: {exc}")}
    return _retrieve


def make_retrieve_fallback_node(vector_store: Any, summary_store: Any | None = None, sparse_store: Any | None = None):
    """创建 retrieve_fallback_node（去过滤兜底检索）。"""
    def _fallback(state: RagState) -> dict[str, Any]:
        decision_dict = state.get("decision")
        if not decision_dict:
            return {"docs": [], "errors": _append(state, "retrieve_fallback: 缺少 decision")}
        decision = RetrievalDecision(**decision_dict)
        decision.metadata_filter = {}
        try:
            cfg = RetrievalConfig(
                top_k=decision.final_k,
                candidate_k=decision.candidate_k,
                rerank="none",
                product_name=state.get("product_name"),
                product_spec=state.get("product_spec"),
                category=state.get("category"),
                preserve_preferred_sections=False,
                fallback_without_filter=True,
            )
            trace = retrieve_advanced(
                vector_store,
                state["question"],
                cfg,
                summary_store=summary_store,
                sparse_store=sparse_store,
                decision=decision,
            )
            return {
                "trace": trace.to_dict(include_preview=True),
                "docs": list(trace.docs),
                "errors": _append(state, *trace.errors) if trace.errors else {},
            }
        except Exception as exc:
            return {"docs": [], "errors": _append(state, f"retrieve_fallback: {type(exc).__name__}: {exc}")}
    return _fallback


# ── rerank（工厂：闭包注入重排配置） ──

def make_rerank_node(rerank_cfg: dict[str, Any] | None = None):
    """创建 rerank_node。"""
    cfg = rerank_cfg or {}
    def _rerank(state: RagState) -> dict[str, Any]:
        from agent_base.retrieval.advanced_retriever import _dedupe
        from agent_base.retrieval.reranker import rerank_documents
        docs = list(state.get("docs") or [])
        decision = state.get("decision") or {}
        route = state.get("route") or {}
        if not docs:
            return {}
        candidates = _dedupe(docs)
        strategy = _resolve_rerank(decision.get("strategy", ""), cfg)
        rerank_errors: list[str] = []
        selected = rerank_documents(
            state.get("rewritten_query", state.get("question", "")),
            candidates,
            strategy=strategy,
            top_k=decision.get("final_k", 6),
            preferred_sections=route.get("sections") or [],
            model_provider=cfg.get("provider", "none"),
            model_name=cfg.get("model", "gte-rerank-v2"),
            model_endpoint=cfg.get("endpoint"),
            model_api_key_env=cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
            model_timeout=int(cfg.get("timeout", 30)),
            preserve_preferred_sections=True,
            errors=rerank_errors,
        )
        result: dict[str, Any] = {"docs": selected}
        if rerank_errors:
            result["errors"] = _append(state, *rerank_errors)
        return result
    return _rerank


# ── safety（无外部依赖） ──

def safety_node(state: RagState) -> dict[str, Any]:
    """安全/合规门禁——永远在图里，不在 agent 自觉里。

    Args:
        state: 当前图状态（question / docs / route）。

    Returns:
        写入 safety（风险等级 / 标签 / 警告）的字段子集。
    """
    question = state.get("question", "")
    docs = state.get("docs") or []
    route = state.get("route") or {}
    route_sections = route.get("sections") or []
    if route_sections and docs:
        context = "\n\n".join(
            getattr(d, "page_content", str(d)) for d in docs
            if (getattr(d, "metadata", {}) or {}).get("section") in route_sections
        )
    else:
        context = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
    assessment = assess_safety(question, context)
    return {"safety": assessment.to_dict()}


# ── generate（工厂：闭包注入 LLM 配置） ──

def make_generate_node(llm_cfg: dict[str, Any] | None = None, prompts_path: str | None = None):
    """创建 generate_node（受控生成或模板兜底）。

    Args:
        llm_cfg: LLM 配置；provider 为 none/off/false 时走模板兜底。
        prompts_path: prompts.yaml 路径。

    Returns:
        LangGraph 节点函数 (state) -> {"answer": str}。
    """
    cfg = llm_cfg or {}
    def _generate(state: RagState) -> dict[str, Any]:
        from agent_base.chains.qa_chain import (
            _answer_docs, _build_conclusion, _build_guidance,
            _compact_evidence, _controlled_llm_answer, _format_answer,
            _format_sources, _build_llm_evidence, _structured_adverse_frequency_parts,
        )
        question = state.get("question", "")
        docs = state.get("docs") or []
        route = state.get("route") or {}
        safety_dict = state.get("safety") or {}
        route_sections = route.get("sections") or []

        assessment = SafetyAssessment(
            risk_level=safety_dict.get("risk_level", "low"),
            findings=[SafetyFinding(**f) for f in safety_dict.get("findings", [])],
            warnings=safety_dict.get("warnings", []),
            must_consult=safety_dict.get("must_consult", False),
            emergency=safety_dict.get("emergency", False),
        )
        answer_docs = _answer_docs(docs, route_sections)
        if not answer_docs:
            return {"answer": _format_answer(
                conclusion="当前商品资料中未检索到足够信息，不能据此给出明确建议。",
                evidence="未找到相关商品/FAQ 片段。",
                guidance="请核对商品名称、规格和使用场景后重新提问，或直接联系在线客服。",
                safety=assessment, sources="无")}

        structured = _structured_adverse_frequency_parts(question, answer_docs)
        conclusion = structured["conclusion"] if structured else _build_conclusion(question, answer_docs, assessment)
        evidence = structured["evidence"] if structured else _compact_evidence(answer_docs)

        provider = cfg.get("provider", "none")
        if provider not in {"none", "off", "false"}:
            ev_cfg = cfg.get("evidence", {})
            llm_evidence = structured["evidence"] if structured else _build_llm_evidence(
                question=question, docs=answer_docs, route_sections=route_sections,
                max_chars_per_doc=int(ev_cfg.get("max_chars_per_doc", 1200)),
                max_total_chars=int(ev_cfg.get("max_total_chars", 6000)),
            )
            sales = state.get("sales") or {}
            sales_block = "\n\n".join(
                part
                for part in (sales.get("sales_strategy") or "", sales.get("guide") or "")
                if part
            )
            answer = _controlled_llm_answer(
                question=question, conclusion=conclusion, evidence=llm_evidence,
                guidance=_build_guidance(question, answer_docs, assessment),
                safety=assessment, sources=_format_sources(answer_docs),
                llm_provider=provider, llm_model=cfg.get("model"),
                llm_base_url=cfg.get("base_url"),
                llm_api_key_env=cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
                llm_temperature=float(cfg.get("temperature", 0.1)),
                prompts_path=prompts_path,
                sales_strategy=sales_block,
            )
        else:
            answer = _format_answer(conclusion=conclusion, evidence=evidence,
                guidance=_build_guidance(question, answer_docs, assessment),
                safety=assessment, sources=_format_sources(answer_docs))
        return {"answer": answer}
    return _generate


# ── refusal（无外部依赖） ──

def make_agent_node(
    vector_store: Any,
    summary_store: Any | None = None,
    rerank_cfg: dict[str, Any] | None = None,
    llm_cfg: dict[str, Any] | None = None,
    prompts_path: str | None = None,
):
    """创建 agent_node（闭包注入基础设施）。

    Agent 不可用时返回空 answer + errors，图自动降级到 generate。
    Agent 输出必须经过 safety 节点——门禁永远在图里。
    """
    def _agent(state: RagState) -> dict[str, Any]:
        from agent_base.agents import build_ecommerce_agent

        agent_graph = build_ecommerce_agent(
            vector_store=vector_store,
            summary_store=summary_store,
            llm_cfg=llm_cfg,
            rerank_cfg=rerank_cfg,
        )
        if agent_graph is None:
            return {
                "answer": "",
                "docs": [],
                "errors": _append(state, "agent: LLM 不可用，降级到确定性 Workflow"),
            }

        question = state.get("question", "")
        category = state.get("category") or ""
        conversation = state.get("conversation") or {}
        context_lines = [f"用户问题：{question}"]
        if category:
            context_lines.append(f"商品类目：{category}")
        if conversation.get("current_product"):
            context_lines.append(f"当前浏览商品：{conversation['current_product']}")
        agent_input = {
            "messages": [{"role": "user", "content": "\n".join(context_lines)}],
        }

        try:
            result = agent_graph.invoke(
                agent_input,
                {"configurable": {"thread_id": f"agent_{hash(question)}"}, "recursion_limit": 40},
            )
            # 从 agent 消息中提取最终回答
            messages = result.get("messages", [])
            final_answer = ""
            if messages:
                last_msg = messages[-1]
                final_answer = getattr(last_msg, "content", str(last_msg))
            return {
                "answer": final_answer,
                "docs": [],
                "agent_trace": {"iterations": len([m for m in messages if getattr(m, "type", "") == "ai"])},
            }
        except Exception as exc:
            # 生产安全：异常详情只进 trace/日志，不回显给买家
            try:
                from agent_base.monitoring.logger import log_event

                log_event("ERROR", "graphs", "agent_node_failed", {
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                })
            except Exception:
                pass
            return {
                "answer": "抱歉，智能助手暂时不可用，请稍后再试或转人工客服。",
                "docs": [],
                "errors": _append(state, f"agent: {type(exc).__name__}: {exc}"),
            }

    return _agent


def refusal_node(state: RagState) -> dict[str, Any]:
    """安全门禁拦截：生成拒答文本后到 END（电商客服口径）。

    Args:
        state: 当前图状态（safety）。

    Returns:
        {"answer": 拒答文本}。
    """
    s = state.get("safety") or {}
    warnings = s.get("warnings", [])
    wt = "\n".join(f"- {w}" for w in warnings) if warnings else "- 请联系在线人工客服"
    return {"answer": (
        "\u26a0\ufe0f 系统检测到需要人工介入的高风险场景，暂时无法自动回答。\n\n"
        f"安全等级：{s.get('risk_level', 'high')}\n\n"
        f"相关安全提示：\n{wt}\n\n"
        "建议：请联系在线人工客服，我们会尽快为您处理。"
    )}


# ── 条件边路由 ──

def should_fallback_retrieve(state: RagState) -> str:
    """retrieve 后条件边：过滤检索无结果时去过滤兜底。

    Args:
        state: 当前图状态。

    Returns:
        "retrieve_fallback"（去过滤重试）或 "rerank"。
    """
    docs = state.get("docs") or []
    metadata_filter = state.get("metadata_filter") or {}
    if not docs and metadata_filter:
        return "retrieve_fallback"
    return "rerank"


def should_block(state: RagState) -> str:
    """safety 后条件边：高风险 + 紧急场景拦截拒答。

    Args:
        state: 当前图状态（safety）。

    Returns:
        "__end__"（进入 refusal）或 "generate"。
    """
    safety = state.get("safety") or {}
    if safety.get("risk_level") == "high" and safety.get("emergency"):
        return "__end__"
    return "generate"


# ── 辅助函数 ──

def _append(state: RagState, *items: str) -> dict[str, list[str]]:
    """把错误信息追加进 state.errors。

    Args:
        state: 当前图状态。
        items: 要追加的错误信息。

    Returns:
        {"errors": 合并后的错误列表}。
    """
    existing: list[str] = list(state.get("errors") or [])
    existing.extend(items)
    return {"errors": existing}


def _resolve_rerank(strategy: str, cfg: dict[str, Any]) -> str:
    """按检索策略与重排配置决定实际重排策略。

    Args:
        strategy: 检索策略名（如 safety_hybrid）。
        cfg: 重排配置。

    Returns:
        "model"（策略命中启用列表且 provider 可用）或 "keyword"。
    """
    provider = (cfg.get("provider", "none") or "none").lower()
    if provider in {"none", "off", "false"}:
        return "keyword"
    enabled = set(cfg.get("use_for_strategies") or ["safety_hybrid", "summary_guided_hybrid", "hybrid"])
    return "model" if strategy in enabled else "keyword"
