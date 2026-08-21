"""RagState — LangGraph 共享状态（契约 P1-01）。

每个节点签名 (state: RagState) → Partial[RagState]，只返回自身改动的字段。
"""

from __future__ import annotations

from typing import Any, TypedDict


class RagState(TypedDict, total=False):
    """RAG 全链路共享状态。

    total=False 表示所有字段可选——节点只返回自己改动的子集，
    LangGraph 自动合并入全局状态。
    """

    # ── 输入 ──
    question: str
    """原始用户问题。"""

    product_name: str | None
    product_spec: str | None
    category: str | None
    """领域约束（商品名/规格/分类）。"""

    # ── 路由层（route_node 产出） ──
    route: dict[str, Any]
    """意图路由：intent / sections / matched_keywords / confidence / scores / source。"""

    rewritten_query: str
    """改写后查询。"""

    metadata_filter: dict[str, Any]
    """Chroma metadata 过滤条件。"""

    decision: dict[str, Any]
    """检索策略决策：strategy / reason / candidate_k / final_k / route_type / need_clarification 等。"""

    sales: dict[str, Any]
    """会话级销售决策（v2）：stage / action / reason / sales_strategy / guide /
    buying_signal / objection_type / missing_info。由 sales 节点产出。"""

    # ── 检索层（retrieve_node 产出） ──
    trace: dict[str, Any]
    """检索全过程 Trace（AdvancedRetrievalTrace.to_dict() 结构）。
    包含 route / rewrite / metadata_filter / decision / stage_counts / errors / results / docs。"""

    docs: list[Any]
    """最终证据文档列表（LangChain Document）。"""

    # ── 安全层（safety_node 产出） ──
    safety: dict[str, Any]
    """安全评估：risk_level / findings / warnings / must_consult / emergency。"""

    # ── 生成层（generate_node 产出） ──
    answer: str
    """最终答案文本。"""

    # ── 会话记忆（P6-03） ──
    conversation: dict[str, Any]
    """工作记忆：{current_product, pending_request, emotion, turns, thread_id}。
    agent 多轮对话时注入上下文，实现跨轮记忆延续。"""

    # ── 容错 ──
    errors: list[str]
    """节点容错记录。各节点追加而非覆盖。"""
