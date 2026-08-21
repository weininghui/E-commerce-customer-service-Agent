"""检索后重排模块。

支持三种策略：
- none：不重排，原序截断
- keyword：本地关键词打分（无需外部服务，兜底可用）
- model：调用本地 TEI（bge-reranker-v2-m3），失败自动降级到 keyword
"""

from __future__ import annotations

from typing import Any


def rerank_documents(
    query: str,
    docs: list[Any],
    strategy: str = "none",
    top_k: int = 5,
    preferred_sections: list[str] | None = None,
    model_provider: str = "none",
    model_name: str = "gte-rerank-v2",
    model_endpoint: str | None = None,
    model_api_key_env: str = "DASHSCOPE_API_KEY",
    model_timeout: int = 30,
    preserve_preferred_sections: bool = True,
    errors: list[str] | None = None,
) -> list[Any]:
    """按策略对候选项重排并截断。

    Args:
        query: 用户查询文本。
        docs: 候选 Document 列表。
        strategy: none / keyword / model。
        top_k: 最终保留数量。
        preferred_sections: 优先保留的章节列表（model 策略下生效）。
        model_provider: 模型重排提供方（local_tei / tei）。
        model_name: 模型名（记录用）。
        model_endpoint: TEI /rerank 地址。
        model_api_key_env: 环境变量名（保留兼容）。
        model_timeout: 模型请求超时秒数。
        preserve_preferred_sections: 是否保留偏好章节。
        errors: 可选错误收集列表。

    Returns:
        重排后的 Document 列表，每条带 rerank_strategy/score/rank 元数据。

    Raises:
        ValueError: 不支持的 strategy。
    """
    strategy = (strategy or "none").lower()
    if strategy == "none":
        for rank, doc in enumerate(docs[:top_k], start=1):
            _set_rerank_metadata(doc, strategy=strategy, score=None, rank=rank)
        return docs[:top_k]

    if strategy == "model":
        try:
            selected = _model_rerank_official(
                query=query,
                docs=docs,
                top_k=top_k,
                endpoint=model_endpoint,
                timeout=model_timeout,
            )
            if preserve_preferred_sections:
                selected = _preserve_preferred_section_docs(selected, docs, preferred_sections, top_k)
            for rank, doc in enumerate(selected, start=1):
                metadata = getattr(doc, "metadata", {}) or {}
                _set_rerank_metadata(
                    doc,
                    strategy="model",
                    score=_optional_float(metadata.get("relevance_score")),
                    rank=rank,
                )
            return selected[:top_k]
        except Exception as exc:
            if errors is not None:
                errors.append(f"rerank_model: {type(exc).__name__}: {exc}")
            # 模型重排是增强层，不能因为外部服务不可用导致检索链路失败。
            # P19b：降级不再用关键词顺序整体覆盖向量语义序，改为
            # “关键词信号 + 原始向量信号”融合兜底（无 vector_score 时退化为纯关键词）。
            from agent_base.retrieval.keyword_ranker import hybrid_fallback_rank

            return hybrid_fallback_rank(
                query,
                docs,
                top_k=top_k,
                preferred_sections=preferred_sections,
            )

    if strategy == "keyword":
        scored = [(keyword_score(query, doc, preferred_sections=preferred_sections), doc) for doc in docs]
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        for rank, (score, doc) in enumerate(scored[:top_k], start=1):
            _set_rerank_metadata(doc, strategy=strategy, score=score, rank=rank)
            selected.append(doc)
        return selected
    raise ValueError(f"Unsupported rerank strategy: {strategy}")


def keyword_score(question: str, doc: Any, preferred_sections: list[str] | None = None) -> float:
    """升级版关键词得分（0-1，P19b）：电商词典分词 + TF-IDF + 长度归一化。

    委托 keyword_ranker.keyword_signal，保持函数签名兼容：

    Args:
        question: 查询文本。
        doc: 待打分 Document。
        preferred_sections: 偏好章节列表。

    Returns:
        0-1 归一化得分（越高越相关）。
    """
    from agent_base.retrieval.keyword_ranker import keyword_signal

    return keyword_signal(question, doc, preferred_sections=preferred_sections)


def _model_rerank_official(
    query: str,
    docs: list[Any],
    top_k: int,
    endpoint: str | None,
    timeout: int,
) -> list[Any]:
    """P17：官方 CrossEncoderReranker 包装本地 TEI。

    替代自研 ``_model_rerank_documents``：使用
    ``langchain_classic.retrievers.document_compressors.CrossEncoderReranker``，
    内部调用 ``TEICrossEncoder.score(text_pairs)``。

    Args:
        query: 用户查询。
        docs: 候选 Document。
        top_k: Number of top results to keep.
        endpoint: TEI /rerank URL.
        timeout: HTTP timeout in seconds.

    Returns:
        Top-k Documents sorted by TEI relevance score.

    Raises:
        RuntimeError: TEI unavailable / invalid response.
    """
    if not docs:
        return []
    if not endpoint:
        raise RuntimeError(
            "model rerank requires endpoint (example: http://localhost:8081/rerank)"
        )

    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from agent_base.retrieval.tei_encoder import TEICrossEncoder

    encoder = TEICrossEncoder(endpoint=endpoint, timeout=timeout)
    reranker = CrossEncoderReranker(model=encoder, top_n=top_k)
    selected = list(reranker.compress_documents(documents=docs, query=query))
    # P17 修复：官方 CrossEncoderReranker 排序后不把分数写进 metadata，
    # 补一次 score 调用回填 relevance_score，保证 trace 的 rerank_score 有值
    try:
        scores = encoder.score([(query, d.page_content) for d in selected])
        for doc, score in zip(selected, scores):
            md = dict(getattr(doc, "metadata", {}) or {})
            md["relevance_score"] = float(score)
            doc.metadata = md
    except Exception:
        pass
    return selected




def _preserve_preferred_section_docs(
    selected: list[Any],
    candidates: list[Any],
    preferred_sections: list[str] | None,
    top_k: int,
) -> list[Any]:
    if not preferred_sections or not selected:
        return selected[:top_k]
    selected_keys = {_doc_key(doc) for doc in selected}
    output = list(selected[:top_k])
    for section in preferred_sections:
        if any((getattr(doc, "metadata", {}) or {}).get("section") == section for doc in output):
            continue
        replacement = next(
            (
                doc
                for doc in candidates
                if (getattr(doc, "metadata", {}) or {}).get("section") == section
                and _doc_key(doc) not in selected_keys
            ),
            None,
        )
        if replacement is None:
            continue
        if len(output) < top_k:
            output.append(replacement)
        else:
            output[-1] = replacement
        selected_keys.add(_doc_key(replacement))
    return output[:top_k]


def _doc_key(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("chunk_id") or f"{metadata.get('doc_id')}:{metadata.get('section')}" or id(doc))


def _doc_text(doc: Any) -> str:
    return getattr(doc, "page_content", str(doc)).replace("\n", " ").strip()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _set_rerank_metadata(doc: Any, strategy: str, score: float | None, rank: int) -> None:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    metadata["rerank_strategy"] = strategy
    metadata["rerank_score"] = score
    metadata["rerank_rank"] = rank
    try:
        doc.metadata = metadata
    except Exception:
        pass
