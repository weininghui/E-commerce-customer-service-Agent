"""Multi-Query 多查询改写（契约 P8-01，LangChain 官方封装版）。

一个问法召回不全：用 LLM 把原始问题改写成多个查询变体，分别检索后合并去重。
实现直接复用 LangChain 官方 ``MultiQueryRetriever``（langchain-classic），
不再自研"改写 → RRF 融合"链路。LLM 不可用/失败时回退单查询，不阻塞链路。

说明：官方 MultiQueryRetriever 的合并策略是 unique union（按内容去重，保留
首次出现顺序），与自研 RRF 的"按排名加权"语义等价但更轻——当前 P8 场景
（同库多查询变体）不需要跨异构检索器融合，RRF 仅保留给 fusion.py 的
多路变体召回使用。
"""

from __future__ import annotations

from typing import Any

from agent_base.retrieval.filter_adapter import chroma_to_qdrant_filter
from agent_base.retrieval.metadata_retriever import _is_qdrant, _similarity_search

# Multi-Query 改写提示词：一个问法 → 3 个覆盖同义词/口语/实体的检索变体
_MULTI_QUERY_PROMPT = (
    "你是电商客服检索助手。请把用户问题改写成 3 个不同角度的检索查询，"
    "覆盖同义词、口语表达与关键商品实体，每行一个查询，不要编号、不要解释。"
)


def _build_langchain_retriever(
    vector_store: Any,
    k: int,
    metadata_filter: dict[str, Any] | None,
) -> Any:
    """把项目向量库包装成 LangChain BaseRetriever（自动适配 Qdrant filter）。

    Args:
        vector_store: 向量库（Chroma / Qdrant VectorStore 实例）。
        k: 单路返回条数。
        metadata_filter: Chroma 语法 where filter；Qdrant 时自动转换。

    Returns:
        vector_store.as_retriever(...) 实例。
    """
    search_kwargs: dict[str, Any] = {"k": k}
    if metadata_filter:
        actual_filter: dict[str, Any] = metadata_filter
        if _is_qdrant(vector_store):
            qdrant_filter = chroma_to_qdrant_filter(metadata_filter)
            actual_filter = qdrant_filter or {}
        search_kwargs["filter"] = actual_filter
    return vector_store.as_retriever(search_kwargs=search_kwargs)


def multi_query_retrieve(
    question: str,
    vector_store: Any,
    llm: Any | None = None,
    k: int = 6,
    metadata_filter: dict[str, Any] | None = None,
) -> list[Any]:
    """多查询检索：LangChain MultiQueryRetriever 改写 → 多路检索 → 去重合并。

    Args:
        question: 用户问题。
        vector_store: 向量库（支持 as_retriever / similarity_search_with_score）。
        llm: LangChain chat 模型（ChatOpenAI 等）；None 或失败时退化为单查询检索。
        k: 最终返回条数。
        metadata_filter: 检索过滤条件（Chroma 语法，Qdrant 时自动转换）。

    Returns:
        去重合并后的文档列表（最多 k 条）。
    """
    if llm is None:
        return [doc for doc, _ in _similarity_search(vector_store, question, k=k, metadata_filter=metadata_filter)]

    try:
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever.from_llm(
            retriever=_build_langchain_retriever(vector_store, k=k, metadata_filter=metadata_filter),
            llm=llm,
            prompt=_MULTI_QUERY_PROMPT,
            include_original=True,
        )
        return retriever.invoke(question)[:k]
    except Exception:
        # LLM/封装失败：退化为单查询，保证链路不中断
        return [doc for doc, _ in _similarity_search(vector_store, question, k=k, metadata_filter=metadata_filter)]
