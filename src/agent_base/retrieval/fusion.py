"""RRF 融合（RAG-Fusion 核心，契约 P8-02，算法用官方实现）。

RRF（Reciprocal Rank Fusion）：多路召回结果按"排名"融合，不依赖分数绝对值——
向量相似度与 BM25 等异构分数不可比，但"各自排第几"是通用的。

    score(d) = Σ 1 / (k + rank_i(d))     # k 通常取 60

官方/自研分工（v0.22.6 调整）：
- **RRF 融合算法用官方** ``EnsembleRetriever.weighted_reciprocal_rank``——
  加权公式 `score = Σ weight_i / (rank + c)` 与自研完全一致，且官方支持
  每路权重 + 去重 + 稳定排序，没必要重复实现。
- **多查询变体编排保留自研**：本模块用法是"同一个向量库 + 多个查询变体
  （原问题/关键词/加 domain anchor）各自召回 → RRF 融合"，查询串不同；
  ``EnsembleRetriever`` 只对"多个检索器 × 同一查询"场景做整体封装，不支持
  按子检索器改写查询，因此这里保留"变体生成 + 各路检索"编排，仅把融合
  一步委托官方。
- P15 引入 BM25（sparse）与向量（dense）双路时，直接用
  ``EnsembleRetriever`` 整体封装（多检索器同查询语义正好匹配）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.retrievers import BaseRetriever


class _NoopRetriever(BaseRetriever):
    """占位检索器：仅用于满足 EnsembleRetriever 的校验器。

    ``weighted_reciprocal_rank`` 只依赖 weights/c 实例属性，不触发子检索器；
    但构造校验要求 retrievers 数量 == weights 数量，故按路数生成占位实例。
    """

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Any]:
        return []



def _doc_key(doc: Any) -> str:
    """提取文档唯一标识（chunk_id 优先，回退内容前 120 字符）。"""
    metadata = getattr(doc, "metadata", {}) or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    text = getattr(doc, "page_content", str(doc))[:120]
    return str(text)


def rrf_fusion(
    ranked_lists: list[list[Any]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[Any]:
    """RRF 融合多路排名结果（官方 weighted_reciprocal_rank），返回按融合分降序的文档列表。

    Args:
        ranked_lists: 多路排名列表（每路已按相关度排序）。
        k: RRF 常数，默认 60。
        weights: 每路权重（与 ranked_lists 等长）；None 时各路等权 1.0。
            dense/sparse 双路融合时可用 [dense_weight, sparse_weight]
            表达"稠密语义优先、稀疏关键词补充"的业务偏好。

    Returns:
        融合排序后的文档列表（同一文档在多路出现只保留首次出现的实例，
        顺序按融合分降序）。

    Raises:
        ValueError: k 非正数、ranked_lists 为空或 weights 长度不匹配。
    """
    if k <= 0:
        raise ValueError(f"k 必须为正数，收到 {k}")
    n = len(ranked_lists)
    if n == 0:
        return []
    if weights is not None:
        if len(weights) != n:
            raise ValueError(f"weights 长度 {len(weights)} 与 ranked_lists 路数 {n} 不一致")
        weights = [float(w) for w in weights]
    else:
        weights = [1.0] * n
    from langchain_classic.retrievers import EnsembleRetriever

    ensemble = EnsembleRetriever(
        retrievers=[_NoopRetriever() for _ in range(n)],
        weights=weights,
        c=k,
    )
    return ensemble.weighted_reciprocal_rank(ranked_lists)
