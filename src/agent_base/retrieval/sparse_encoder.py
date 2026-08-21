"""BM25 稀疏向量编码器（P15-01）。

默认用 jieba 分词（真实语料对比实验 recall@1 94.7% vs n-gram 89.5%，见
tests/tokenizer_compare.py），构建语料 BM25 索引并输出 Qdrant ``SparseVector``。
底层 ``bm25s`` 向量化建索引，万级语料比 rank_bm25 快一个量级。
"""

from __future__ import annotations

import hashlib as _hashlib
import math
import re
from collections import Counter
from typing import Any

from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from qdrant_client.models import SparseVector


def _term_id(term: str) -> int:
    """生成确定性词条 ID（sha256，跨进程稳定）。

    Python 内置 ``hash()`` 按进程随机化（PYTHONHASHSEED），
    这里改用 sha256（v0.27.3：md5 → sha256）保证跨进程一致。

    Args:
        term: 词条文本。

    Returns:
        0~2^31-1 的整数词条 ID。
    """
    return int(_hashlib.sha256(term.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def tokenize_chinese(text: str) -> list[str]:
    """中文分词：按词生成 unigram + bigram。

    Args:
        text: 输入文本（支持中文及任意 CJK 文本）。

    Returns:
        unigram 与 bigram 组成的词条列表。
    """
    tokens: list[str] = []
    # 按空白分词，词内生成 unigram + bigram（不跨词，避免"衣轻"这类噪声）
    for word in re.split(r"\s+", text.strip()):
        if not word:
            continue
        chars = [ch for ch in word if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"]
        if not chars:
            continue
        tokens.extend(chars)
        tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return tokens


def tokenize_jieba(text: str) -> list[str]:
    """jieba 分词（精确模式 + HMM），词粒度稀疏检索锚点。

    对比实验（tests/tokenizer_compare.py，真实 19 组查询）：
    jieba recall@1=94.7% vs n-gram 89.5%，recall@5/10 均 100%。

    Args:
        text: 输入文本（中文电商语料）。

    Returns:
        jieba 词条列表。
    """
    import jieba

    return [w.strip() for w in jieba.cut(text) if w.strip()]


class BM25SparseEncoder:
    """构建语料的 BM25 加权稀疏向量索引。

    用法示例::

        encoder = BM25SparseEncoder()
        encoder.fit(corpus_texts)
        sparse_vecs = [encoder.encode_single(t) for t in corpus_texts]
        # 每个 sparse_vec 都是 Qdrant SparseVector
    """

    def __init__(self):
        """初始化：空索引 + 默认 jieba 分词器。"""
        self._index: Any = None
        self._tokenizer = tokenize_jieba

    def fit(self, texts: list[str]) -> "BM25SparseEncoder":
        """在语料上构建 BM25 索引（官方 bm25s，向量化建索引）。

        Args:
            texts: 文档文本列表。

        Returns:
            自身（支持链式调用）。

        Notes:
            IDF 公式与旧 rank_bm25 BM25Okapi 一致（log((N-df+0.5)/(df+0.5))）；
            稀疏向量输出保持不变，``_index`` 保留 bm25s 模型供大规模 get_scores 复用。
        """
        from bm25s import BM25

        tokenized = [self._tokenizer(t) for t in texts]
        self._index = BM25(method="lucene")
        self._index.index(tokenized, show_progress=False)
        self._corpus_size = len(tokenized)
        # 文档频率（df）直接按 doc 级集合统计，与 rank_bm25 的 nd 语义一致
        doc_freq: Counter[str] = Counter()
        for doc in tokenized:
            doc_freq.update(set(doc))
        self._idf = {
            term: math.log(self._corpus_size - freq + 0.5) - math.log(freq + 0.5)
            for term, freq in doc_freq.items()
        }
        return self

    def encode_single(self, text: str) -> SparseVector:
        """将单个文本编码为 Qdrant 稀疏向量。

        Args:
            text: 输入文本。

        Returns:
            含词条索引与 BM25 权重的 Qdrant SparseVector。
        """
        tokens = self._tokenizer(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        # 计算该文档自身的 BM25 权重
        # 用 TF × IDF 作为稀疏特征：常见词（如"防晒/怎么样"）降权，
        # 避免纯 TF 归一化让短文本常见词主导检索
        tf = Counter(tokens)
        total = len(tokens) or 1
        indices: list[int] = []
        values: list[float] = []
        for term, count in tf.items():
            idf = self._idf.get(term, 1.0) if hasattr(self, "_idf") else 1.0
            if idf <= 0:
                # 高频词（过半文档出现）BM25 idf 为负，负权重会破坏 Qdrant 相似度，丢弃
                continue
            term_id = _term_id(term)  # Deterministic sha256-based hash
            indices.append(term_id)
            values.append((count / total) * idf)
        return SparseVector(indices=indices, values=values)


class ChineseBM25SparseEmbeddings(SparseEmbeddings):
    """中文 BM25 稀疏向量适配器（官方 ``langchain_qdrant.SparseEmbeddings`` 接口）。

    包装 ``BM25SparseEncoder``：``fit(corpus)`` 建 BM25 索引后，
    ``embed_documents/embed_query`` 输出 Qdrant ``SparseVector``，
    可直接作为 ``QdrantVectorStore(sparse_embedding=...)`` 的稀疏通道使用。

    Args:
        tokenizer: 中文分词器（默认 jieba；n-gram 保留可选，见 tests/tokenizer_compare.py）。
    """

    def __init__(self, tokenizer=tokenize_jieba):
        """初始化：包装 BM25SparseEncoder 并注入分词器。"""
        self._encoder = BM25SparseEncoder()
        self._encoder._tokenizer = tokenizer

    def fit(self, texts: list[str]) -> "ChineseBM25SparseEmbeddings":
        """对语料建 BM25 索引（IDF 归一，供 doc 向量与 query 向量共用）。"""
        self._encoder.fit(texts)
        return self

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """批量编码文档稀疏向量（TF×IDF，与线上索引口径一致）。"""
        return [self._encoder.encode_single(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        """编码查询稀疏向量（复用语料 IDF，BM25 标准做法）。"""
        return self._encoder.encode_single(text)


def encode_query_sparse(query: str) -> SparseVector:
    """将查询编码为稀疏向量，供混合检索使用。

    与文档编码共用 unigram+bigram 分词，权重取 TF（未 fit 时无 IDF）。

    Args:
        query: 搜索查询文本。

    Returns:
        Qdrant SparseVector。
    """
    encoder = BM25SparseEncoder()
    return encoder.encode_single(query)
