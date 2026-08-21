"""向量库工厂：按 provider 创建/加载 Chroma 或 Qdrant VectorStore。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_base.indexing import vector_index


def build_vector_store(
    provider: str = "chroma",
    persist_dir: str | Path = "data/chroma",
    collection: str = "ecommerce_chunks",
    embedding_function: Any = None,
    url: str | None = None,
    api_key: str | None = None,
):
    """按 provider 创建向量库连接（可写入，用于入库/重建）。

    Args:
        provider: 向量库类型（chroma / qdrant）。
        persist_dir: Chroma 持久化目录。
        collection: collection 名。
        embedding_function: LangChain Embeddings 实例；qdrant 必填。
        url: Qdrant 服务地址（如 http://localhost:6333）。
        api_key: Qdrant API key（可选）。

    Returns:
        支持 similarity_search / add_texts / delete 的 LangChain VectorStore。

    Raises:
        ValueError: provider 不支持、qdrant 缺 url 或 embedding。
        RuntimeError: 缺少依赖包或 Qdrant 连接失败。
    """
    return _create(
        provider=provider,
        persist_dir=persist_dir,
        collection=collection,
        embedding_function=embedding_function,
        url=url,
        api_key=api_key,
    )


def load_vector_store(
    provider: str = "chroma",
    persist_dir: str | Path = "data/chroma",
    collection: str = "ecommerce_chunks",
    embedding_function: Any = None,
    url: str | None = None,
    api_key: str | None = None,
):
    """按 provider 加载向量库（供检索）。

    Args:
        provider: 向量库类型（chroma / qdrant）。
        persist_dir: Chroma 持久化目录。
        collection: collection 名。
        embedding_function: LangChain Embeddings 实例；qdrant 必填。
        url: Qdrant 服务地址（如 http://localhost:6333）。
        api_key: Qdrant API key（可选）。

    Returns:
        支持 similarity_search / similarity_search_with_score / add_texts / delete
        的 LangChain VectorStore。

    Raises:
        ValueError: provider 不支持、qdrant 缺 url 或 embedding。
        RuntimeError: 缺少依赖包或 Qdrant 连接失败。
    """
    return _create(
        provider=provider,
        persist_dir=persist_dir,
        collection=collection,
        embedding_function=embedding_function,
        url=url,
        api_key=api_key,
    )


def _create(
    provider: str,
    persist_dir: str | Path,
    collection: str,
    embedding_function: Any,
    url: str | None,
    api_key: str | None,
):
    """按 provider 分发到对应实现。

    Args:
        provider: 向量库类型。
        persist_dir: Chroma 持久化目录。
        collection: collection 名。
        embedding_function: Embeddings 实例（qdrant 必填）。
        url: Qdrant 服务地址。
        api_key: Qdrant API key。

    Returns:
        对应 provider 的 LangChain VectorStore。

    Raises:
        ValueError: 不支持的 provider。
    """
    provider = (provider or "chroma").lower()
    if provider in {"chroma", "chromadb"}:
        return _create_chroma(persist_dir, collection, embedding_function)
    if provider == "qdrant":
        return _create_qdrant(collection, embedding_function, url, api_key)
    raise ValueError(f"Unsupported vectorstore provider: {provider}")


def _create_chroma(persist_dir: str | Path, collection: str, embedding_function: Any):
    """创建或加载 Chroma VectorStore。

    Args:
        persist_dir: Chroma 持久化目录。
        collection: collection 名。
        embedding_function: Embeddings 实例；None 时委托原实现使用默认行为。

    Returns:
        langchain_chroma.Chroma 实例。
    """
    if embedding_function is None:
        # 未显式传 embedding 时，委托原实现保持默认行为（hash embedding 兜底），
        # 保证现有调用方行为不变。
        return vector_index.load_vector_store(persist_dir=persist_dir, collection=collection)
    Chroma = vector_index._chroma_class()
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection,
        persist_directory=str(persist_dir),
        embedding_function=embedding_function,
    )


def _create_qdrant(collection: str, embedding_function: Any, url: str | None, api_key: str | None):
    """创建 Qdrant VectorStore，并在启动时校验连接。

    Args:
        collection: collection 名。
        embedding_function: Embeddings 实例（必填）。
        url: Qdrant 服务地址。
        api_key: Qdrant API key（可选）。

    Returns:
        langchain_qdrant.QdrantVectorStore 实例。

    Raises:
        ValueError: url 或 embedding_function 缺失。
        RuntimeError: 缺少依赖包或 Qdrant 连接失败。
    """
    if not url:
        raise ValueError(
            "vectorstore provider=qdrant 必须提供 url（例如 http://localhost:6333）。"
            "检查 configs/app.yaml 的 vectorstore.url 配置。"
        )
    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: langchain-qdrant. Install with `pip install langchain-qdrant`."
        ) from exc
    if embedding_function is None:
        raise ValueError("vectorstore provider=qdrant 必须提供 embedding_function，不能为 None。")
    client = QdrantClient(url=url, api_key=api_key)
    # 启动即验证连接：Qdrant 不可达时抛明确错误，不静默失败，方便上线时快速定位。
    try:
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(f"无法连接 Qdrant（{url}）：{type(exc).__name__}: {exc}") from exc
    # P9 集成：auto-create collection if it doesn't exist (langchain-qdrant 1.x no longer auto-creates)
    try:
        client.get_collection(collection)
    except Exception:
        from qdrant_client.models import Distance, VectorParams
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embedding_function,
        validate_collection_config=False,
    )
