"""契约测试：RAGAS 四指标打分（_ragas_scores 的门控与容错）。

覆盖：
1. 空回答 → 直接返回 {}（不触网）；
2. ragas 包不可用 → 返回 {}（回退 LCEL 判官，不抛异常）；
3. ragas 可用 → 四指标打分；无 reference 时跳过 context_recall。
"""

from __future__ import annotations

from agent_base.retrieval.eval_chain import _ragas_scores


class _FakeMetric:
    """假 ragas 指标：构造签名兼容，固定返回 0.9。"""

    def __init__(self, llm=None, embeddings=None, max_retries=1, **kwargs):
        pass

    def single_turn_score(self, sample):
        return 0.9


def _mock_ragas(monkeypatch):
    monkeypatch.setattr("agent_base.llms.build_chat_model", lambda **kwargs: object())
    monkeypatch.setattr("agent_base.embeddings.build_embeddings", lambda **kwargs: object())
    monkeypatch.setattr("ragas.llms.LangchainLLMWrapper", lambda model: model)
    monkeypatch.setattr("ragas.embeddings.LangchainEmbeddingsWrapper", lambda emb: emb)
    monkeypatch.setattr("ragas.metrics.Faithfulness", _FakeMetric)
    monkeypatch.setattr("ragas.metrics.AnswerRelevancy", _FakeMetric)
    monkeypatch.setattr("ragas.metrics.ContextPrecision", _FakeMetric)
    monkeypatch.setattr("ragas.metrics.ContextRecall", _FakeMetric)


def test_ragas_scores_empty_answer():
    """空回答：不做任何模型调用，直接空结果。"""
    assert _ragas_scores("q", "", "ctx") == {}


def test_ragas_scores_missing_package(monkeypatch):
    """ragas 不可用：返回 {} 且不抛异常（调用方回退 LCEL）。"""
    import sys

    monkeypatch.setitem(sys.modules, "ragas", None)
    assert _ragas_scores("q", "a", "ctx") == {}


def test_ragas_scores_with_reference(monkeypatch):
    """ragas 可用 + reference：四指标全部返回。"""
    _mock_ragas(monkeypatch)
    scores = _ragas_scores("玻尿酸精华适合油皮吗", "适合", "证据片段一\n\n证据片段二", reference="适合油皮")
    assert scores == {
        "faithfulness": 0.9,
        "relevancy": 0.9,
        "context_precision": 0.9,
        "context_recall": 0.9,
    }


def test_ragas_scores_without_reference_skips_recall(monkeypatch):
    """无 reference：context_recall 需要参考回答，跳过该指标。"""
    _mock_ragas(monkeypatch)
    scores = _ragas_scores("问题", "回答", "证据")
    assert "context_recall" not in scores
    assert scores["faithfulness"] == 0.9
    assert scores["relevancy"] == 0.9
    assert scores["context_precision"] == 0.9
