"""向量索引构建：从 chunk JSONL 构建/加载 Chroma 或 Qdrant 索引。"""

from __future__ import annotations

import uuid
from pathlib import Path

from agent_base.embeddings import build_embeddings



def _chroma_class():
    try:
        from langchain_chroma import Chroma

        return Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma

            return Chroma
        except ImportError as exc:
            raise RuntimeError(
                "Missing Chroma dependencies. Install with `pip install -r requirements.txt`."
            ) from exc


def _qdrant_point_id(chunk_id: str) -> str:
    """把业务 chunk_id 确定性映射为 Qdrant point ID（UUID）。

    Qdrant 只接受 unsigned int 或 UUID 作为 point ID；上传/解析链路的 chunk_id
    是 `{doc_id}:{section}:{chunk}` 格式，不能直接使用。用 uuid5 做确定性映射，
    保证写入与删除使用同一 ID。
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag:{chunk_id}"))


def load_vector_store(
    persist_dir: str | Path = "data/chroma",
    collection: str = "ecommerce_chunks",
    embedding_provider: str = "hash",
    embedding_model: str | None = None,
    dimensions: int | None = 512,
    embedding_base_url: str | None = None,
    embedding_api_key_env: str = "DASHSCOPE_API_KEY",
    embedding_keep_alive: str | None = None,
):
    """按配置加载一个 Chroma collection，供检索或删除旧向量使用。"""
    embeddings = build_embeddings(
        provider=embedding_provider,
        model=embedding_model,
        dimensions=dimensions,
        base_url=embedding_base_url,
        api_key_env=embedding_api_key_env,
        keep_alive=embedding_keep_alive,
    )
    Chroma = _chroma_class()
    return Chroma(
        collection_name=collection,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def _delete_existing(vector_store, ids: list[str]) -> None:
    """写入前删除同 ID 文档；Chroma 中不存在这些 ID 时直接忽略。"""
    try:
        vector_store.delete(ids=ids)
    except Exception:
        pass
