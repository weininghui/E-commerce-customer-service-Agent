"""预审置信度打分单元测试（_heuristic_confidence / _heuristic_classify_with_hits）。

覆盖打分逻辑的边界与单调性，不依赖真实文档内容（避免 flaky）。
"""

from __future__ import annotations

from agent_base.knowledge_factory import (
    _heuristic_classify_with_hits,
    _heuristic_confidence,
)


def test_confidence_zero_hit_is_low():
    """0 命中 → 0.0（无信号 → 无置信度，避免 30% 误导）。"""
    assert _heuristic_confidence(0, 11) == 0.0
    assert _heuristic_confidence(0, 8) == 0.0


def test_confidence_monotonic_increasing():
    """命中越多置信度越高（单调递增）。"""
    total = 11
    confs = [_heuristic_confidence(h, total) for h in range(0, total + 1)]
    for i in range(1, len(confs)):
        assert confs[i] >= confs[i - 1], f"非单调: {confs}"
    # 至少有几个不同档位（不是全同分）
    assert len(set(confs)) >= 3


def test_confidence_capped_at_08():
    """全命中 → 0.8 封顶（规则兜底不超过 LLM 下限）。"""
    assert _heuristic_confidence(11, 11) == 0.8
    assert _heuristic_confidence(8, 8) == 0.8


def test_confidence_normalized_by_total():
    """归一化：相同命中数、不同信号总数，分值不同（比例更高者更高）。"""
    c_few = _heuristic_confidence(4, 8)   # 50%
    c_many = _heuristic_confidence(4, 11)  # 36%
    assert c_few > c_many
    assert c_few == round(0.3 + 0.5 * (4 / 8), 2)


def test_confidence_in_range():
    """所有合法输入都在 0.0-0.8。"""
    for total in (5, 8, 11):
        for hits in range(0, total + 1):
            c = _heuristic_confidence(hits, total)
            assert 0.0 <= c <= 0.8


def test_classify_selects_highest_ratio():
    """多类型命中时按命中率胜出（FAQ 命中 5/11 高于商品 2/11）。"""
    # 用 ASCII 信号（Q:/A: 属于 faq；无商品信号）
    content = "Q: can return A: refund Q: shipping A: fast Q: fee A: free"
    doc_type, hits, total = _heuristic_classify_with_hits(content, "faq.md")
    assert doc_type == "faq"
    assert hits >= 2 and total >= 5


def test_classify_no_signal_falls_back_product():
    """无任何特征命中 → 默认 product_detail，hits=0（置信度 0，提示需人工细看）。"""
    doc_type, hits, total = _heuristic_classify_with_hits(
        "zzz no matching signal content", "unknown.md"
    )
    assert doc_type == "product_detail"
    assert hits == 0
    assert _heuristic_confidence(hits, total) == 0.0
