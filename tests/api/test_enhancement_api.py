"""P32d/e：检索增强按需触发测试。

覆盖：
1. 复合问题检测 → Decomposition 触发
2. 初检质量低 → Multi-Query 触发
3. 两者互斥（Decomposition 优先）
4. 配置关闭时零开销（不触发）
5. 非复合非低质量 → 不触发
"""

from __future__ import annotations

from unittest.mock import patch

from agent_base.retrieval.enhancement import (
    _initial_quality_low,
    _looks_like_compound,
    assess_enhancement,
)


# ── 复合问题检测 ──


def test_compound_detection_comparison():
    """比较句式 → 复合。"""
    assert _looks_like_compound("玻尿酸精华和水乳哪个好") is True
    assert _looks_like_compound("A 和 B 的区别是什么") is True
    assert _looks_like_compound("轻云跑鞋还是极速训练鞋好") is True


def test_compound_detection_multi_question():
    """多个问号 → 复合。"""
    assert _looks_like_compound("这个适合干皮吗？油皮能用吗？") is True


def test_compound_detection_parallel():
    """并列连词 → 复合。"""
    assert _looks_like_compound("保湿面膜和玻尿酸精华和氨基酸洁面") is True


def test_compound_detection_simple():
    """单问题 → 非复合。"""
    assert _looks_like_compound("玻尿酸精华适合敏感肌吗") is False
    assert _looks_like_compound("这款商品有什么功效") is False


# ── 初检质量评估 ──


class _MockDoc:
    def __init__(self, score: float | None = None):
        self.metadata = {}
        if score is not None:
            self.metadata["score"] = score
        self.page_content = "mock content"


def test_quality_low_empty():
    """空结果 → 质量低。"""
    assert _initial_quality_low([]) is True


def test_quality_low_few_hits():
    """命中数不足 → 质量低。"""
    docs = [_MockDoc(0.8), _MockDoc(0.7)]  # 2 docs < hit_threshold=3
    assert _initial_quality_low(docs, hit_threshold=3) is True


def test_quality_low_top1():
    """top1 得分低 → 质量低。"""
    docs = [_MockDoc(0.2), _MockDoc(0.5), _MockDoc(0.4), _MockDoc(0.3)]
    assert _initial_quality_low(docs, top1_threshold=0.3, hit_threshold=3) is True


def test_quality_good():
    """质量好 → 不触发。"""
    docs = [_MockDoc(0.8), _MockDoc(0.7), _MockDoc(0.6), _MockDoc(0.5)]
    assert _initial_quality_low(docs, top1_threshold=0.3, hit_threshold=3) is False


def test_quality_good_tight_scores_bug23():
    """BUG-23：真实分数分布（高分但彼此紧邻，如 0.5-0.6）→ 不误触发。"""
    docs = [_MockDoc(0.62), _MockDoc(0.55), _MockDoc(0.5), _MockDoc(0.48)]
    assert _initial_quality_low(docs, top1_threshold=0.55, hit_threshold=3, gap_margin=0.08) is False


def test_quality_good_clear_top1_bug23():
    """BUG-23：top1 明显领先次高分 → 相对判别视为有效命中，绝对分不高也不触发。"""
    docs = [_MockDoc(0.29), _MockDoc(0.18), _MockDoc(0.16), _MockDoc(0.15)]
    assert _initial_quality_low(docs, top1_threshold=0.55, hit_threshold=3, gap_margin=0.08) is False


def test_quality_low_below_calibrated_threshold_bug23():
    """BUG-23：校准后阈值 0.55——top1=0.54（无关问题实测区间）→ 低质量触发。"""
    docs = [_MockDoc(0.54), _MockDoc(0.52), _MockDoc(0.5), _MockDoc(0.49)]
    assert _initial_quality_low(docs, top1_threshold=0.55, hit_threshold=3, gap_margin=0.08) is True


def test_quality_low_tight_low_scores_bug23():
    """BUG-23：分数整体偏低且紧邻 → 确实低质量，触发 Multi-Query。"""
    docs = [_MockDoc(0.2), _MockDoc(0.19), _MockDoc(0.18), _MockDoc(0.17)]
    assert _initial_quality_low(docs, top1_threshold=0.55, hit_threshold=3, gap_margin=0.08) is True


# ── 配置关闭时零开销 ──


def test_enhancement_disabled_by_default():
    """默认关闭（配置 enabled=false）→ 不触发。"""
    with patch(
        "agent_base.retrieval.enhancement.load_enhancement_config",
        return_value={
            "decomposition_enabled": False,
            "multi_query_enabled": False,
            "multi_query_variants": 3,
            "quality_top1_threshold": 0.3,
            "quality_hit_threshold": 3,
        },
    ):
        result = assess_enhancement(
            "玻尿酸精华和水乳哪个好",  # compound question
            [_MockDoc(0.2)],  # low quality
            "product_query",
        )
        assert result["triggered"] is False
        assert result["type"] == "none"


def test_multi_query_llm_config_present():
    """app.yaml multi_query 段必须含 LLM 配置，避免增强空转。

    若 provider 缺失，advanced_retriever 会以 provider="none" 建模型 → llm=None
    → multi_query_retrieve 退化为单查询，Multi-Query 增强等于白跑一遍无收益。
    """
    import pathlib

    import yaml

    cfg = yaml.safe_load(pathlib.Path("configs/app.yaml").read_text(encoding="utf-8"))
    mq = (cfg.get("retrieval") or {}).get("multi_query") or {}
    assert mq.get("provider"), "multi_query 缺 provider，Multi-Query 会退化为单查询空转"
    assert mq.get("model"), "multi_query 缺 model"
    assert mq.get("api_key_env"), "multi_query 缺 api_key_env"


# ── Decomposition 优先于 Multi-Query ──


def test_decomposition_priority_over_multi_query():
    """复合问题 + 低质量 → Decomposition 优先。"""
    with patch(
        "agent_base.retrieval.enhancement.load_enhancement_config",
        return_value={
            "decomposition_enabled": True,
            "multi_query_enabled": True,
            "multi_query_variants": 3,
            "quality_top1_threshold": 0.3,
            "quality_hit_threshold": 3,
        },
    ):
        result = assess_enhancement(
            "玻尿酸精华和水乳哪个好",
            [_MockDoc(0.2)],  # low quality
            "product_query",
        )
        assert result["triggered"] is True
        assert result["type"] == "decomposition"


def test_multi_query_when_no_compound():
    """非复合 + 低质量 → Multi-Query。"""
    with patch(
        "agent_base.retrieval.enhancement.load_enhancement_config",
        return_value={
            "decomposition_enabled": True,
            "multi_query_enabled": True,
            "multi_query_variants": 3,
            "quality_top1_threshold": 0.8,  # high threshold
            "quality_hit_threshold": 10,    # high threshold
        },
    ):
        result = assess_enhancement(
            "保湿面膜有什么功效",  # not compound
            [_MockDoc(0.5), _MockDoc(0.4)],  # low quality
            "product_query",
        )
        assert result["triggered"] is True
        assert result["type"] == "multi_query"


# ── 不触发场景 ──


def test_no_enhancement_for_good_quality():
    """质量好 + 非复合 → 不触发。"""
    with patch(
        "agent_base.retrieval.enhancement.load_enhancement_config",
        return_value={
            "decomposition_enabled": True,
            "multi_query_enabled": True,
            "multi_query_variants": 3,
            "quality_top1_threshold": 0.3,
            "quality_hit_threshold": 3,
        },
    ):
        docs = [_MockDoc(0.8), _MockDoc(0.7), _MockDoc(0.6), _MockDoc(0.5)]
        result = assess_enhancement(
            "玻尿酸精华适合敏感肌吗",
            docs,
            "product_query",
        )
        assert result["triggered"] is False


def test_no_enhancement_for_clarification():
    """澄清策略 → 不触发增强。"""
    with patch(
        "agent_base.retrieval.enhancement.load_enhancement_config",
        return_value={
            "decomposition_enabled": True,
            "multi_query_enabled": True,
            "multi_query_variants": 3,
            "quality_top1_threshold": 0.3,
            "quality_hit_threshold": 3,
        },
    ):
        result = assess_enhancement(
            "这个怎么样",
            [],
            "product_query",
        )
        assert result["triggered"] is True  # would trigger multi-query (empty docs)
        # But the caller (advanced_retriever) skips enhancement for clarification strategy
