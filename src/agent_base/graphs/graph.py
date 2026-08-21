"""LangGraph StateGraph 工厂（契约 P2-04 图拓扑）。

P2 拓扑：
  START → route → should_use_agent?
    ├─ "agent" → agent → safety → should_block? → generate | refusal
    └─ "retrieve" → retrieve → [fallback?] → rerank → safety → should_block? → generate | refusal
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_base.graphs.nodes import (
    make_agent_node,
    make_generate_node,
    make_rerank_node,
    make_retrieve_fallback_node,
    make_retrieve_node,
    make_sales_node,
    refusal_node,
    route_node,
    safety_node,
    should_block,
    should_fallback_retrieve,
)
from agent_base.graphs.state import RagState


def build_rag_graph(
    vector_store: Any,
    summary_store: Any | None = None,
    sparse_store: Any | None = None,
    rerank_cfg: dict[str, Any] | None = None,
    llm_cfg: dict[str, Any] | None = None,
    prompts_path: str | None = None,
) -> CompiledStateGraph:
    """构建并编译 RAG 状态图（确定性 Workflow + Agent 分支）。

    图拓扑：
    START → route → should_use_agent?
      ├─ "agent" → agent → safety → should_block? → generate | refusal
      └─ "retrieve" → retrieve → [fallback?] → rerank → safety → should_block? → generate | refusal

    基础设施（vector_store / summary_store / 配置）通过闭包注入节点，
    不放入 RagState——避免 Chroma 对象被 checkpoint msgpack 序列化。
    model 不可用时 agent_node 内部降级——不阻塞图执行。

    Args:
        vector_store: 原文 chunk 向量库。
        summary_store: 摘要向量库（可选）。
        sparse_store: BM25 稀疏向量库（可选，混合检索补充通道）。
        rerank_cfg: 重排配置（provider / model / use_for_strategies）。
        llm_cfg: LLM 配置（provider / model / base_url / api_key_env / temperature）。
        prompts_path: prompts.yaml 路径。

    Returns:
        编译后的 CompiledStateGraph，可 `invoke(state, config)` 执行。
    """
    builder = StateGraph(RagState)

    # ── 确定性 Workflow 节点 ──
    builder.add_node("route", route_node)
    builder.add_node("sales", make_sales_node())
    builder.add_node("retrieve", make_retrieve_node(vector_store, summary_store, sparse_store))
    builder.add_node("retrieve_fallback", make_retrieve_fallback_node(vector_store, summary_store, sparse_store))
    builder.add_node("rerank", make_rerank_node(rerank_cfg))
    builder.add_node("safety", safety_node)
    builder.add_node("generate", make_generate_node(llm_cfg, prompts_path))
    builder.add_node("refusal", refusal_node)

    # ── Agent 节点 ──
    builder.add_node("agent", make_agent_node(
        vector_store, summary_store, rerank_cfg, llm_cfg, prompts_path,
    ))

    # ── 连线 ──
    builder.add_edge(START, "route")
    builder.add_edge("route", "sales")

    # P2 新增：Agent 触发路由（延迟导入避免循环）
    from agent_base.agents.routing import should_use_agent as _should_use_agent

    # agent 不可用（无 LLM）时禁止路由进 agent 分支，否则比较类问题会被
    # 路由进空 agent → 降级为空答案且无 trace。这是防御性设计。
    agent_available = (llm_cfg or {}).get("provider", "none") not in {"none", "off", "false"}

    def _route_agent_or_retrieve(state: RagState) -> str:
        if not agent_available:
            return "retrieve"
        return _should_use_agent(state)

    builder.add_conditional_edges(
        "sales",
        _route_agent_or_retrieve,
        {"agent": "agent", "retrieve": "retrieve"},
    )

    # Agent 分支：agent → safety → should_block → generate | refusal
    builder.add_edge("agent", "safety")

    # 确定性检索分支：retrieve → [fallback?] → rerank → safety → ...
    builder.add_conditional_edges(
        "retrieve",
        should_fallback_retrieve,
        {"retrieve_fallback": "retrieve_fallback", "rerank": "rerank"},
    )
    builder.add_edge("retrieve_fallback", "rerank")
    builder.add_edge("rerank", "safety")

    # 安全门禁（两个分支汇聚于此）
    builder.add_conditional_edges(
        "safety",
        should_block,
        {"generate": "generate", "__end__": "refusal"},
    )
    builder.add_edge("refusal", END)
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=MemorySaver())
