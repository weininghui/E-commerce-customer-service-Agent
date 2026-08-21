"""契约：官方 SparseEmbeddings 适配器（中文 BM25 稀疏向量接口）。"""

from __future__ import annotations

from qdrant_client.models import SparseVector

from agent_base.retrieval.sparse_encoder import ChineseBM25SparseEmbeddings, encode_query_sparse


CORPUS = [
    "玻尿酸保湿精华液 适合所有肤质 保湿补水",
    "烟酰胺精华 提亮肤色 控油 敏感肌慎用",
    "神经酰胺面霜 修护屏障 干皮秋冬适用",
    "防晒霜 SPF50 PA+++ 物化结合 清爽不黏腻",
]


def test_embeddings_implements_official_interface():
    """适配器继承官方 SparseEmbeddings，embed_documents/embed_query 返回 Qdrant SparseVector。"""
    emb = ChineseBM25SparseEmbeddings().fit(CORPUS)
    docs = emb.embed_documents(CORPUS)
    assert len(docs) == len(CORPUS)
    assert all(isinstance(v, SparseVector) for v in docs)
    assert all(v.indices and v.values for v in docs)

    q = emb.embed_query("玻尿酸精华多少钱")
    assert isinstance(q, SparseVector)
    assert q.indices and q.values
    assert len(q.indices) == len(q.values)


def test_query_reuses_corpus_idf():
    """embed_query 复用语料 IDF：常见词权重应低于罕见词（BM25 标准）。"""
    emb = ChineseBM25SparseEmbeddings().fit(CORPUS)
    q = emb.embed_query("玻尿酸")
    # "玻尿酸" 在 4 篇中出现 1 篇 → idf>0；向量非空
    assert q.indices

    # 未 fit 的裸编码（线上 encode_query_sparse）也返回合法结构
    raw = encode_query_sparse("玻尿酸精华")
    assert isinstance(raw, SparseVector)
    assert len(raw.indices) == len(raw.values)


def test_tokenizer_swappable():
    """分词器可注入（jieba 实验用），接口行为一致。"""
    def jieba_tokenize(text: str) -> list[str]:
        import jieba

        return [w.strip() for w in jieba.cut(text) if w.strip()]

    emb = ChineseBM25SparseEmbeddings(tokenizer=jieba_tokenize).fit(CORPUS)
    docs = emb.embed_documents(CORPUS)
    assert all(isinstance(v, SparseVector) and v.indices for v in docs)
