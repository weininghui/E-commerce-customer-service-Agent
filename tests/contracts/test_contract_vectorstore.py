"""契约 V-01：向量库工厂 chroma/qdrant 可切换，qdrant 未配置时报明确错误。"""

from __future__ import annotations

import uuid

import pytest

from agent_base.embeddings import build_embeddings
from agent_base.indexing.vector_index import _qdrant_point_id
from agent_base.vectorstore import build_vector_store, load_vector_store


def _hash_embeddings():
    return build_embeddings(provider="hash")


def test_chroma_default_delegates_to_legacy(tmp_path):
    # 不传 embedding_function 时委托原实现（默认 hash embedding），行为不变
    store = load_vector_store(provider="chroma", persist_dir=tmp_path, collection="ecommerce_chunks")
    assert store.similarity_search("query", k=2) == []


def test_chroma_empty_store_returns_empty_list(tmp_path):
    store = load_vector_store(
        provider="chroma",
        persist_dir=tmp_path,
        collection="empty",
        embedding_function=_hash_embeddings(),
    )
    assert store.similarity_search("anything") == []


def test_chroma_build_add_search_delete(tmp_path):
    store = build_vector_store(
        provider="chroma",
        persist_dir=tmp_path,
        collection="build",
        embedding_function=_hash_embeddings(),
    )
    store.add_texts(["坎地沙坦酯片的用法用量"], ids=["a"])
    results = store.similarity_search("用法用量", k=3)
    assert len(results) == 1
    store.delete(ids=["a"])
    assert store.similarity_search("用法用量", k=3) == []


def test_qdrant_without_url_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="url"):
        load_vector_store(
            provider="qdrant",
            collection="c",
            embedding_function=_hash_embeddings(),
        )


def test_qdrant_without_embedding_raises_clear_error():
    with pytest.raises(ValueError, match="embedding_function"):
        load_vector_store(provider="qdrant", collection="c", url="http://localhost:6333")


def test_qdrant_unreachable_raises_clear_error(tmp_path):
    # 本机一个必然不可达的端口：连接失败要抛明确错误，不静默失败
    with pytest.raises(RuntimeError, match="无法连接 Qdrant"):
        load_vector_store(
            provider="qdrant",
            collection="c",
            embedding_function=_hash_embeddings(),
            url="http://127.0.0.1:1",
        )


def test_unsupported_provider_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported vectorstore provider"):
        load_vector_store(provider="milvus", persist_dir=tmp_path, collection="c")


def test_qdrant_point_id_mapping():
    """业务 chunk_id -> Qdrant point ID 必须是合法 UUID 且确定性映射（v0.21.2 修复）。"""
    cid = "abc123:000:001"
    pid = _qdrant_point_id(cid)
    parsed = uuid.UUID(pid)  # 非法 UUID 会抛 ValueError
    assert str(parsed) == pid
    assert _qdrant_point_id(cid) == pid  # 确定性：同一 chunk_id 同一 point ID
    assert _qdrant_point_id("other:000:001") != pid
