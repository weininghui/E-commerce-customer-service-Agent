"""TEI cross-encoder wrapper for local bge-reranker-v2-m3 (P17-01).

Implements ``BaseCrossEncoder.score()`` by POST-ing to the local TEI
``/rerank`` endpoint.  Used with ``CrossEncoderReranker`` from
``langchain_classic.retrievers.document_compressors``.

Data never leaves the host — the endpoint must be localhost.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from langchain_core.cross_encoders import BaseCrossEncoder


class TEICrossEncoder(BaseCrossEncoder):
    """本地 TEI /rerank 服务驱动的 Cross-encoder（bge-reranker-v2-m3）。

    ``score(text_pairs)`` calls ``POST {endpoint}`` with
    ``{"query": query, "texts": documents}`` and returns the raw
    relevance scores in text-pair order.

    Args:
        endpoint: TEI /rerank 完整 URL（默认 http://localhost:8081/rerank）。
        timeout: HTTP 超时秒数。
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8081/rerank",
        timeout: int = 30,
    ):
        """初始化：保存 TEI 端点与超时配置。"""
        super().__init__()
        self.endpoint = endpoint
        self.timeout = timeout

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        """通过本地 TEI 对查询-文档对做交叉编码。

        文本对格式为 ``[(query, doc_text), ...]``；TEI /rerank 每次只接收
        单个 query + 文档数组，多 query 时按唯一 query 逐个请求后拼接。

        Args:
            text_pairs: (query, document_text) 元组列表。

        Returns:
            每个文本对一个分数（顺序一致）。

        Raises:
            RuntimeError: HTTP 错误、超时或响应非法。
        """
        if not text_pairs:
            return []

        # 按 query 分组（典型场景：单个 query、多个文档）
        groups: dict[str, list[int]] = {}
        for idx, (query, _doc_text) in enumerate(text_pairs):
            groups.setdefault(query, []).append(idx)

        scores = [0.0] * len(text_pairs)

        for query, indices in groups.items():
            docs = [text_pairs[i][1] for i in indices]
            batch_scores = self._call_tei(query, docs)
            for i, s in zip(indices, batch_scores):
                scores[i] = s

        return scores

    def _call_tei(self, query: str, documents: list[str]) -> list[float]:
        """向 TEI /rerank 发送单个 query + 文档批次。

        Args:
            query: 查询文本。
            documents: 文档文本。

        Returns:
            每个文档一个相关性分数（顺序一致）。

        Raises:
            RuntimeError: TEI 不可用或响应非法。
        """
        payload = {"query": query, "texts": documents}
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"TEI rerank failed: HTTP {exc.code} {detail}") from exc

        data = json.loads(body)
        if not isinstance(data, list):
            raise RuntimeError(f"TEI rerank unexpected response: {str(data)[:200]}")

        # 构建索引 → 分数映射，再按文档顺序返回
        index_score = {int(item["index"]): float(item["score"]) for item in data}
        return [index_score.get(i, 0.0) for i in range(len(documents))]
