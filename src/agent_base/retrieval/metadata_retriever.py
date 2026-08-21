"""检索结果展示视图工具（Chroma/Qdrant 兼容）。

对外入口：
- document_to_retrieval_item：将 Document 转为 RetrievalItem 展示视图
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any



@dataclass(slots=True)
class RetrievalItem:
    """单条检索结果的展示视图：排名、多阶段得分、来源定位与摘要。"""

    rank: int
    score: float | None
    vector_score: float | None
    rerank_score: float | None
    retrieval_stage: str
    chunk_id: str
    section: str
    product_name: str
    product_spec: str
    source_file: str
    page_start: int | str
    page_end: int | str
    preview: str
    doc_name: str = ""

def _similarity_search(vector_store: Any, query: str, k: int, metadata_filter: dict[str, Any]) -> list[tuple[Any, float | None]]:
    """向量相似度检索（自动适配 Chroma / Qdrant filter 语法）。

    Args:
        vector_store: Chroma 或 Qdrant VectorStore 实例。
        query: 检索查询文本。
        k: 返回数量。
        metadata_filter: Chroma 语法的 where filter。Qdrant 时自动转换。

    Returns:
        [(Document, score), ...] 列表。
    """
    kwargs = {"k": k}
    if metadata_filter:
        actual_filter = metadata_filter
        # P9-02：检测 Qdrant store 并转换 filter 语法
        if _is_qdrant(vector_store):
            from agent_base.retrieval.filter_adapter import chroma_to_qdrant_filter
            qdrant_filter = chroma_to_qdrant_filter(metadata_filter)
            if qdrant_filter:
                actual_filter = qdrant_filter
            else:
                actual_filter = {}
        kwargs["filter"] = actual_filter
    if hasattr(vector_store, "similarity_search_with_score"):
        return vector_store.similarity_search_with_score(query, **kwargs)
    docs = vector_store.similarity_search(query, **kwargs)
    return [(doc, None) for doc in docs]


def _is_qdrant(store: Any) -> bool:
    """检测向量库是否为 Qdrant（有 qdrant_client 属性且无 Chroma 的 _collection）。

    Args:
        store: VectorStore 实例。

    Returns:
        True 表示 Qdrant。
    """
    return hasattr(store, "client") and not hasattr(store, "_collection")


def _to_item(rank: int, doc: Any, score: float | None) -> RetrievalItem:
    metadata = getattr(doc, "metadata", {}) or {}
    text = getattr(doc, "page_content", str(doc)).replace("\n", " ")
    preview = text[:240].rstrip() + ("..." if len(text) > 240 else "")
    return RetrievalItem(
        rank=rank,
        score=round(float(score), 6) if score is not None else None,
        vector_score=_round_optional(metadata.get("vector_score", score)),
        rerank_score=_round_optional(metadata.get("rerank_score")),
        retrieval_stage=str(metadata.get("retrieval_stage", metadata.get("summary_type", "chunk"))),
        chunk_id=str(metadata.get("chunk_id", "")),
        section=str(metadata.get("section", "")),
        doc_name=str(metadata.get("doc_name", "")),
        product_name=str(metadata.get("product_name", metadata.get("drug_name", ""))),
        product_spec=str(metadata.get("product_spec", metadata.get("generic_name", ""))),
        source_file=str(metadata.get("source_file", "")),
        page_start=metadata.get("page_start", "?"),
        page_end=metadata.get("page_end", metadata.get("page_start", "?")),
        preview=preview,
    )


def document_to_retrieval_item(rank: int, doc: Any, score: float | None = None) -> RetrievalItem:
    """将任意 Document 转换为 RetrievalItem 视图（供 trace 使用）。

    Args:
        rank: 排名（从 1 开始）。
        doc: 带 metadata 的 Document 对象。
        score: 可选的相似度得分。

    Returns:
        RetrievalItem。
    """
    return _to_item(rank, doc, score)


def _round_optional(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None
