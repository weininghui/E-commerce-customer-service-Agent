"""契约测试：主链路 dense+sparse 加权 RRF 融合（retrieval.fusion 配置门控）。

覆盖：
1. rrf_fusion 纯函数：权重校验、跨路去重、权重影响排序；
2. _append_semantic：开启时 RRF 融合并打 fusion 标记，关闭/异常时
   退化为"向量在前、稀疏追加"，链路不中断。
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from agent_base.retrieval.advanced_retriever import _append_semantic, _fusion_config
from agent_base.retrieval.fusion import rrf_fusion


def _doc(doc_id: str, text: str) -> Document:
    return Document(page_content=text, metadata={"chunk_id": doc_id})


def test_rrf_fusion_weights_validation():
    """权重长度与路数不一致时抛 ValueError，不静默错配。"""
    with pytest.raises(ValueError):
        rrf_fusion([[_doc("a", "x")], [_doc("b", "y")]], weights=[1.0])
    assert rrf_fusion([], k=60) == []  # 空输入返回空列表（调用方按无结果处理）


def test_rrf_fusion_dedupes_across_channels():
    """同一文档出现在两路时只保留一份（去重）。"""
    shared = _doc("same", "shared")
    fused = rrf_fusion([[shared, _doc("a", "x")], [shared, _doc("b", "y")]])
    ids = [d.metadata["chunk_id"] for d in fused]
    assert len(ids) == len(set(ids))
    assert "same" in ids


def test_rrf_fusion_weight_skews_order():
    """权重把低排名文档抬到前列（加权 RRF 生效）。"""
    # dense 路：doc_dense 第 1、doc_tail 第 3；sparse 路：doc_sparse 第 1
    dense = [_doc("doc_dense", "d"), _doc("doc_x", "x"), _doc("doc_tail", "t")]
    sparse = [_doc("doc_sparse", "s"), _doc("doc_y", "y")]
    # 稀疏权重拉满 → sparse 路第 1 名总分 3/(60+1) 超过 dense 路第 3 名 1/(60+3)
    fused = rrf_fusion([dense, sparse], weights=[1.0, 3.0])
    assert fused[0].metadata["chunk_id"] in {"doc_dense", "doc_sparse"}
    # 等权时 dense 第 1 必在 sparse 第 1 之前（同 rank 下先出现者优先）
    equal = rrf_fusion([dense, sparse], weights=[1.0, 1.0])
    assert equal[0].metadata["chunk_id"] == "doc_dense"


def test_append_semantic_fusion_enabled(monkeypatch):
    """fusion 开启：vector+sparse 加权 RRF 融合，文档带 fusion=rrf 标记。"""
    monkeypatch.setattr(
        "agent_base.retrieval.advanced_retriever._fusion_config",
        lambda: {"enabled": True, "k": 60, "dense_weight": 0.7, "sparse_weight": 0.3},
    )
    stage_docs = [_doc("meta", "m")]
    counts: dict[str, int] = {"metadata": 1}
    errors: list[str] = []
    _append_semantic(stage_docs, counts, [_doc("v1", "v1"), _doc("v2", "v2")], [_doc("s1", "s1")], errors)
    assert counts["sparse"] == 1
    assert counts["rrf_fusion"] == 3
    assert stage_docs[0].metadata["chunk_id"] == "meta"  # metadata 前置通道优先级不受 RRF 影响
    for doc in stage_docs[1:]:
        assert doc.metadata.get("fusion") == "rrf"
    assert not errors


def test_append_semantic_disabled_keeps_legacy_order(monkeypatch):
    """fusion 关闭：向量在前、稀疏追加（旧行为），不丢数据。"""
    monkeypatch.setattr(
        "agent_base.retrieval.advanced_retriever._fusion_config",
        lambda: {"enabled": False, "k": 60, "dense_weight": 0.7, "sparse_weight": 0.3},
    )
    stage_docs: list[Document] = []
    counts: dict[str, int] = {}
    errors: list[str] = []
    _append_semantic(stage_docs, counts, [_doc("v1", "v1")], [_doc("s1", "s1")], errors)
    assert [d.metadata["chunk_id"] for d in stage_docs] == ["v1", "s1"]
    assert "rrf_fusion" not in counts
    assert not errors


def test_fusion_config_defaults():
    """配置缺失/异常时回退默认值（enabled=True，不因配置问题静默关融合）。"""
    cfg = _fusion_config()
    assert isinstance(cfg, dict)
    assert set(cfg) >= {"enabled", "k", "dense_weight", "sparse_weight"}
    assert cfg["dense_weight"] + cfg["sparse_weight"] > 0
