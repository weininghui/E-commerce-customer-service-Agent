"""Self-Ask 多跳拆解（契约 P8-03，保留自研）。

跨商品比较类问题（"A 和 B 哪个好"）拆成两个子查询分别检索，再经 RRF
融合汇总。P8 采用规则拆解（可解释、零成本），LLM 拆解作为后续可选增强。

为什么保留自研而不是 LangChain：``langchain.retrievers.SelfAskRetriever``
（含 llm_chain 版）在 langchain 1.x 已彻底移除，官方无现代等价物；当前
规则拆解（比较句式检测 + 实体切分）零 LLM 依赖、可解释，符合"能用官方
就用官方、没有才自研"的原则。
"""

from __future__ import annotations

import re
from typing import Any

from agent_base.retrieval.fusion import rrf_fusion
from agent_base.retrieval.metadata_retriever import _similarity_search


_COMPARISON_PATTERNS = [
    r"哪个好",
    r"哪个更",
    r"和.*比",
    r"与.*比",
    r"对比",
    r"有什么区别",
    r"区别",
    r"还是.*好",
    r"哪个适合",
]


def _looks_like_comparison(question: str) -> bool:
    """规则检测比较/多跳句式。"""
    return any(re.search(pattern, question) for pattern in _COMPARISON_PATTERNS)


def decompose_question(question: str) -> list[str] | None:
    """拆解比较句式为子问题。

    规则版（MVP）：对"A 和 B 哪个好/哪个适合"类句式，把两个对比实体
    分别作为子查询（检索按商品名召回）；非比较问题返回 None。

    Args:
        question: 用户问题。

    Returns:
        子查询列表；非比较句式返回 None。
    """
    if not _looks_like_comparison(question):
        return None
    # 尝试拆出两个对比实体：A 和 B 哪个好 / A还是B好 / A与B对比
    m = re.match(r"^(.*?)(?:和|与|、)(.*?)(?:哪个|还是|谁|更|对比).*$", question)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        if left and right and left != right:
            return [left, right]
    return [question]


def self_ask_retrieve(
    question: str,
    vector_store: Any,
    k: int = 6,
    metadata_filter: dict[str, Any] | None = None,
) -> list[Any]:
    """Self-Ask 检索：拆子问题 → 各检索 → RRF 融合 → top_k。

    Args:
        question: 用户问题。
        vector_store: 向量库。
        k: 最终返回条数。
        metadata_filter: 检索过滤条件。

    Returns:
        融合排序后的文档列表（最多 k 条）；非比较问题等价于单查询检索。
    """
    sub_queries = decompose_question(question) or [question]
    ranked_lists: list[list[Any]] = []
    for q in sub_queries:
        try:
            docs = [
                doc
                for doc, _ in _similarity_search(
                    vector_store, q, k=k, metadata_filter=metadata_filter
                )
            ]
        except Exception:
            docs = []
        if docs:
            ranked_lists.append(docs)
    fused = rrf_fusion(ranked_lists)
    return fused[:k]
