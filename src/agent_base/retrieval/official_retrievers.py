"""Official LangChain retrievers wrapper (P14).

Wraps langchain_classic official implementations:
  - MultiVectorRetriever (summary index: hit summary -> return full doc)
  - ParentDocumentRetriever (parent-child: hit child -> return parent)
  - SelfQueryRetriever (LLM metadata filtering, complementary to rule-based)

All imports use langchain_classic (v1 migration path).

v0.23.1 修复（Codex 接手）：
1. docstore 必须用 ``langchain_core.stores.InMemoryStore``（BaseStore 接口，
   mset/mget）——langgraph.store 的 InMemoryStore 同名不同类，Pydantic 校验失败
2. ParentDocumentRetriever 的 child_splitter 是必填字段，不传构造即失败
3. SelfQueryRetriever 必须传自研 translator（内置 translator 走 langchain_community
   弃用红线 + 0.4.2 ImportError），见 retrieval/self_query_visitors.py
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent_base.retrieval.metadata_retriever import _is_qdrant
from agent_base.retrieval.self_query_visitors import ChromaStyleVisitor, QdrantStyleVisitor


def _make_docstore(docstore: Any | None) -> Any:
    """构造 MultiVector/ParentDocument 兼容的 docstore。

    Args:
        docstore: 传入的 docstore；None 时用 langchain_core.stores.InMemoryStore。

    Returns:
        BaseStore[str, Document] 实例。
    """
    return docstore or InMemoryStore()


def _make_child_splitter(child_splitter: Any | None) -> Any:
    """构造父文档检索的子块切分器（ParentDocumentRetriever 必填）。

    Args:
        child_splitter: 传入的 splitter；None 时用 300/30 的默认递归切分。

    Returns:
        TextSplitter 实例。
    """
    return child_splitter or RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)


def build_multi_vector_retriever(
    vectorstore: Any,
    docstore: Any | None = None,
    id_key: str = "doc_id",
    search_kwargs: dict[str, Any] | None = None,
) -> Any:
    """构建官方 MultiVectorRetriever（摘要 → 父文档检索）。

    Summary embeddings live in vectorstore; full parent docs in docstore.
    A summary hit looks up the parent doc by id_key in docstore.

    Args:
        vectorstore: 摘要向量库（小向量指向完整文档）。
        docstore: InMemoryStore or PostgresStore for full parent documents.
        id_key: Metadata key on summary docs that points to the parent doc ID.
        search_kwargs: Additional search kwargs (e.g. {"k": 5}).

    Returns:
        MultiVectorRetriever instance.
    """
    from langchain_classic.retrievers import MultiVectorRetriever

    store = _make_docstore(docstore)
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs=search_kwargs or {"k": 5},
    )


def build_parent_document_retriever(
    vectorstore: Any,
    docstore: Any | None = None,
    child_splitter: Any | None = None,
    parent_splitter: Any | None = None,
    search_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build official ParentDocumentRetriever for child-parent retrieval.

    Child chunks live in vectorstore; parent docs in docstore.
    A child chunk hit automatically returns its parent document.

    Args:
        vectorstore: Vector store containing child chunks.
        docstore: InMemoryStore or PostgresStore for parent documents.
        child_splitter: 子块切分器（必填字段；None 时默认 300/30 递归切分）。
        parent_splitter: 父文档切分器（可选；None 时整文档为父）。
        search_kwargs: Additional search kwargs.

    Returns:
        ParentDocumentRetriever instance.
    """
    from langchain_classic.retrievers import ParentDocumentRetriever

    store = _make_docstore(docstore)
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=_make_child_splitter(child_splitter),
        parent_splitter=parent_splitter,
        search_kwargs=search_kwargs or {"k": 6},
    )


def build_self_query_retriever(
    llm: Any,
    vectorstore: Any,
    metadata_field_info: list[Any] | None = None,
    document_contents: str = "E-commerce product and FAQ documents",
    search_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build official SelfQueryRetriever for LLM-based metadata filtering.

    Complements the rule-based metadata filter: LLM extracts structured queries
    from natural language (e.g. "under 200 yuan for oily skin" ->
    price_band="入门" AND skin_types="油皮").

    Args:
        llm: Chat model instance (deepseek-v4-flash recommended).
        vectorstore: Vector store with metadata fields.
        metadata_field_info: List of AttributeInfo objects.
        document_contents: Description of what the documents contain.
        search_kwargs: Additional search kwargs.

    Returns:
        SelfQueryRetriever instance.
    """
    from langchain_classic.retrievers import SelfQueryRetriever
    from langchain_classic.chains.query_constructor.schema import AttributeInfo

    if metadata_field_info is None:
        metadata_field_info = [
            AttributeInfo(
                name="category",
                description="Product category (e.g. 精华, 面霜, T恤, 连衣裙)",
                type="string",
            ),
            AttributeInfo(
                name="price_band",
                description="Price tier (入门/中端/高端)",
                type="string",
            ),
            AttributeInfo(
                name="section",
                description="Document section (商品参数/卖点/搭配建议/售后FAQ)",
                type="string",
            ),
            AttributeInfo(
                name="brand",
                description="Brand name",
                type="string",
            ),
        ]

    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_contents,
        metadata_field_info=metadata_field_info,
        structured_query_translator=QdrantStyleVisitor() if _is_qdrant(vectorstore) else ChromaStyleVisitor(),
        search_kwargs=search_kwargs or {"k": 6},
    )


def build_pg_docstore(table_name: str = "docstore_summary") -> Any:
    """P16b：创建 PG 持久化 docstore 供检索使用。

    Args:
        table_name: PG 表名（docstore_summary / docstore_parent）。

    Returns:
        PGDocStore 实例（实现 BaseStore 接口）。
    """
    from agent_base.storage.docstore import PGDocStore
    return PGDocStore(table_name=table_name)


def _add_parents_to_docstore(
    docstore: Any,
    parent_docs: list[Document],
) -> None:
    """为 MultiVector/ParentDocument 检索器填充父文档 docstore。

    Each parent doc must have a unique ID in its metadata (used as the store key).

    Args:
        docstore: InMemoryStore 或 PostgresStore 实例。
        parent_docs: List of LangChain Documents (each must have a unique id in metadata).
    """
    if hasattr(docstore, "mset"):
# InMemoryStore / PostgresStore 均通过 mset([(key, doc), ...]) 批量写入
        items = []
        for doc in parent_docs:
            key = doc.metadata.get("doc_id") or doc.metadata.get("chunk_id") or str(hash(doc.page_content))
            items.append((key, doc))
        docstore.mset(items)
    elif hasattr(docstore, "add_documents"):
        docstore.add_documents(parent_docs)
    else:
# 兜底：可用时尝试 add_texts
        for doc in parent_docs:
            key = doc.metadata.get("doc_id") or doc.metadata.get("chunk_id") or str(hash(doc.page_content))
            try:
                docstore.add_texts(texts=[doc.page_content], metadatas=[doc.metadata], ids=[key])
            except Exception:
                pass
