"""契约 P17：模型重排官方化（CrossEncoderReranker + TEICrossEncoder）。

覆盖 v0.26.0 验收修复的回归点：
1. TEICrossEncoder 实现 BaseCrossEncoder.score（按 query 分组、按 index 归位）
2. _model_rerank_official 走官方 CrossEncoderReranker（构造断言 + 排序验证）
3. 分数回填：官方排序后 relevance_score 写入 metadata，trace 的 rerank_score 有值
4. TEI 不可用自动降级 keyword，链路不中断
5. keyword / none 分支行为不变

全部 mock TEI 响应，不依赖本地 reranker 容器在线。
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.documents import Document

from agent_base.retrieval.reranker import _model_rerank_official, rerank_documents
from agent_base.retrieval.tei_encoder import TEICrossEncoder


def _make_docs():
    return [
    Document(page_content="花颜集轻透防晒乳 SPF50+ 轻薄不泛白", metadata={"section": "商品参数", "source_file": "legacy-source#P004"}),
    Document(page_content="玻尿酸保湿精华液 补水保湿 敏感肌可用", metadata={"section": "商品参数", "source_file": "legacy-source#P001"}),
        Document(page_content="防晒衣搭配 内搭吊带或修身T恤 下装阔腿裤", metadata={"section": "穿搭建议", "source_file": "穿搭指南_通勤.md"}),
    ]


def _fake_tei_scores(query: str, documents: list[str]) -> list[float]:
    """模拟 TEI 打分：防晒乳最高、玻尿酸次之、穿搭最低。"""
    scores = {
        "花颜集轻透防晒乳 SPF50+ 轻薄不泛白": 0.95,
        "玻尿酸保湿精华液 补水保湿 敏感肌可用": 0.80,
        "防晒衣搭配 内搭吊带或修身T恤 下装阔腿裤": 0.30,
    }
    return [scores.get(d, 0.1) for d in documents]


def test_tei_cross_encoder_score_groups_and_stitches():
    """TEICrossEncoder.score：按 query 分组 POST，按 index 归位，顺序一致。"""
    encoder = TEICrossEncoder(endpoint="http://fake/rerank")
    pairs = [
        ("q1", "doc-a"),
        ("q2", "doc-b"),
        ("q1", "doc-c"),
    ]
    with patch.object(
        encoder,
        "_call_tei",
        side_effect=lambda q, docs: [0.1 * (i + 1) for i in range(len(docs))],
    ) as mock_call:
        scores = encoder.score(pairs)
    assert scores == [0.1, 0.1, 0.2]
    # q1 调用一次（2 docs），q2 调用一次（1 doc）
    assert mock_call.call_count == 2


def test_model_rerank_official_uses_reranker_and_scores():
    """官方 CrossEncoderReranker 排序 + 分数回填 relevance_score。"""
    docs = _make_docs()
    with patch(
        "agent_base.retrieval.tei_encoder.TEICrossEncoder._call_tei",
        side_effect=_fake_tei_scores,
    ):
        out = _model_rerank_official(
            "有什么防晒推荐？要轻薄透气的", docs, 2, "http://localhost:8081/rerank", 30,
        )
    assert len(out) == 2
    # 防晒乳应排第一（TEI 模拟分数最高）
    assert out[0].metadata.get("source_file") == "legacy-source#P004"
    # 分数已回填到 metadata
    assert out[0].metadata.get("relevance_score") == 0.95
    assert out[1].metadata.get("relevance_score") == 0.80


def test_model_rerank_sets_trace_fields():
    """rerank_documents(strategy=model) 写入 rerank_strategy/score/rank。"""
    docs = _make_docs()
    with patch(
        "agent_base.retrieval.tei_encoder.TEICrossEncoder._call_tei",
        side_effect=_fake_tei_scores,
    ):
        out = rerank_documents(
            "有什么防晒推荐？要轻薄透气的",
            docs,
            strategy="model",
            top_k=2,
            model_provider="local_tei",
            model_endpoint="http://localhost:8081/rerank",
        )
    assert out[0].metadata.get("rerank_strategy") == "model"
    assert out[0].metadata.get("rerank_rank") == 1
    assert out[0].metadata.get("rerank_score") == 0.95


def test_model_rerank_falls_back_to_keyword():
    """TEI 不可用时自动降级 keyword，链路不中断。"""
    docs = _make_docs()
    out = rerank_documents(
        "防晒乳 SPF50",
        docs,
        strategy="model",
        top_k=2,
        model_provider="local_tei",
        model_endpoint="http://127.0.0.1:9/rerank",  # 必然失败的端口，触发降级
        errors=[],
    )
    assert len(out) == 2
    assert all(d.metadata.get("rerank_strategy") == "keyword" for d in out)


def test_keyword_and_none_branches_unchanged():
    """keyword / none 分支行为不变。"""
    docs = _make_docs()
    out_none = rerank_documents("防晒", docs, strategy="none", top_k=2)
    assert len(out_none) == 2
    assert out_none[0].metadata.get("rerank_strategy") == "none"

    out_kw = rerank_documents("防晒乳", docs, strategy="keyword", top_k=2)
    assert len(out_kw) == 2
    assert out_kw[0].metadata.get("rerank_strategy") == "keyword"
    assert out_kw[0].metadata.get("rerank_score") is not None
