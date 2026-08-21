"""契约 P14：官方检索器封装（MultiVector / ParentDocument / SelfQuery）。

覆盖 v0.23.1 修复的回归点：
1. docstore 必须是 langchain_core.stores.BaseStore（不是 langgraph 的同名类）
2. ParentDocumentRetriever 必须传 child_splitter（必填）
3. SelfQueryRetriever 必须传自研 translator（零弃用警告）
4. _add_parents_to_docstore 真实写入
"""

from __future__ import annotations

import warnings

from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.stores import InMemoryStore
from langchain_core.structured_query import Comparison, Comparator, Operation, Operator, StructuredQuery
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent_base.embeddings import build_embeddings
from agent_base.retrieval.official_retrievers import (
    _add_parents_to_docstore,
    build_multi_vector_retriever,
    build_parent_document_retriever,
    build_self_query_retriever,
)
from agent_base.retrieval.self_query_visitors import ChromaStyleVisitor, QdrantStyleVisitor
from agent_base.vectorstore import build_vector_store


def _store(tmp_path, collection: str = "official"):
    return build_vector_store(
        provider="chroma",
        persist_dir=str(tmp_path),
        collection=collection,
        embedding_function=build_embeddings(provider="hash"),
    )


def test_multi_vector_retriever_builds_with_core_docstore(tmp_path):
    """MultiVectorRetriever 构造必须通过（docstore 为 langchain_core BaseStore）。"""
    store = _store(tmp_path, "mvv")
    docstore = InMemoryStore()
    retriever = build_multi_vector_retriever(vectorstore=store, docstore=docstore, id_key="doc_id")
    assert retriever.vectorstore is store
    assert retriever.id_key == "doc_id"


def test_multi_vector_retriever_summary_backtrack(tmp_path):
    """摘要向量命中 -> 按 doc_id 回溯原文全文。"""
    store = _store(tmp_path, "mvback")
    docstore = InMemoryStore()
    retriever = build_multi_vector_retriever(vectorstore=store, docstore=docstore, id_key="doc_id")
    store.add_texts(["白T恤 纯棉 圆领 基础款"], ids=["sum-1"], metadatas=[{"doc_id": "doc-1"}])
    docstore.mset([("doc-1", Document(page_content="白T恤完整详情：纯棉 圆领 基础款 夏季", metadata={"doc_id": "doc-1"}))])
    hits = retriever.invoke("纯棉白T恤")
    assert hits and "完整详情" in hits[0].page_content


def test_parent_document_retriever_builds_with_child_splitter(tmp_path):
    """ParentDocumentRetriever 构造必须通过（child_splitter 必填已修复）。"""
    store = _store(tmp_path, "pdd")
    retriever = build_parent_document_retriever(
        vectorstore=store,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=5),
    )
    assert retriever.child_splitter is not None


def test_parent_document_retriever_child_to_parent(tmp_path):
    """子块检索 -> 自动回溯父文档。"""
    store = _store(tmp_path, "pdback")
    retriever = build_parent_document_retriever(
        vectorstore=store,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=5),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0),
    )
    retriever.add_documents([
        Document(page_content="商品名称：防晒衣。功能：UPF50+ 防晒，轻薄透气，适合户外通勤。面料：聚酯纤维。洗涤：冷水手洗。",
                 metadata={"product_id": "P018"}),
    ])
    hits = retriever.invoke("防晒衣 轻薄")
    assert hits and "防晒" in hits[0].page_content


def test_self_query_retriever_builds_without_deprecation(tmp_path):
    """SelfQueryRetriever 构造零弃用警告（自研 translator，绕开 langchain_community）。"""
    store = _store(tmp_path, "sqq")
    fake_llm = GenericFakeChatModel(messages=iter(['```json\n{"query": "x", "filter": null}\n```']))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        retriever = build_self_query_retriever(llm=fake_llm, vectorstore=store)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations == [], f"SelfQuery 构造触发弃用警告: {deprecations}"
    assert retriever.structured_query_translator is not None


def test_self_query_retriever_filters_by_metadata(tmp_path):
    """SelfQuery 完整链路：LLM 结构化查询 -> filter -> 命中正确商品。"""
    store = _store(tmp_path, "sqfilter")
    store.add_texts(
        ["氨基酸洁面 温和 敏感肌适用", "玻尿酸精华 补水保湿"],
        metadatas=[{"category": "洁面", "skin_type": "敏感肌"}, {"category": "精华", "skin_type": "干皮"}],
        ids=["sq1", "sq2"],
    )
    fake_llm = GenericFakeChatModel(messages=iter([
        '```json\n{"query": "敏感肌 洁面", "filter": "and(eq(\'category\', \'洁面\'), eq(\'skin_type\', \'敏感肌\'))"}\n```',
    ]))
    retriever = build_self_query_retriever(
        llm=fake_llm,
        vectorstore=store,
        metadata_field_info=[
            AttributeInfo(name="category", description="商品类目", type="string"),
            AttributeInfo(name="skin_type", description="适用肤质", type="string"),
        ],
    )
    hits = retriever.invoke("油皮敏感肌用哪个洁面")
    assert hits and hits[0].metadata.get("category") == "洁面"


def test_add_parents_to_docstore_writes(tmp_path):
    """_add_parents_to_docstore 必须真实写入 docstore（v0.23.1 修复静默 0 条）。"""
    docstore = InMemoryStore()
    _add_parents_to_docstore(docstore, [Document(page_content="父文档", metadata={"doc_id": "d1"})])
    assert docstore.mget(["d1"])[0] is not None


def test_chroma_visitor_translates_ir():
    """ChromaStyleVisitor：IR -> Chroma where（$and/$or）。"""
    sq = StructuredQuery(
        query="油皮面霜",
        filter=Operation(operator=Operator.AND, arguments=[
            Comparison(comparator=Comparator.EQ, attribute="category", value="面霜"),
            Operation(operator=Operator.OR, arguments=[
                Comparison(comparator=Comparator.EQ, attribute="skin_type", value="油皮"),
                Comparison(comparator=Comparator.IN, attribute="price_band", value=["中端", "高端"]),
            ]),
        ]),
    )
    query, kwargs = ChromaStyleVisitor().visit_structured_query(sq)
    assert query == "油皮面霜"
    assert kwargs["filter"] == {
        "$and": [
            {"category": {"$eq": "面霜"}},
            {"$or": [{"skin_type": {"$eq": "油皮"}}, {"price_band": {"$in": ["中端", "高端"]}}]},
        ]
    }


def test_qdrant_visitor_translates_ir():
    """QdrantStyleVisitor：IR -> Qdrant payload Filter（must/should）。"""
    sq = StructuredQuery(
        query="油皮面霜",
        filter=Operation(operator=Operator.AND, arguments=[
            Comparison(comparator=Comparator.EQ, attribute="category", value="面霜"),
            Comparison(comparator=Comparator.NE, attribute="brand", value="杂牌"),
        ]),
    )
    query, kwargs = QdrantStyleVisitor().visit_structured_query(sq)
    assert query == "油皮面霜"
    assert kwargs["filter"] == {
        "must": [
            {"key": "category", "match": {"value": "面霜"}},
            {"key": "brand", "match": {"except": ["杂牌"]}},
        ]
    }
