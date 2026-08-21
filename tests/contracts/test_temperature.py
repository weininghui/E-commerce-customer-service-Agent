"""契约 P0-2：动态温度纯函数。

覆盖：
1. 所有意图 × 情绪组合输出在 [0.05, 0.5] 区间；
2. 情绪调节方向：anger 比 neutral 低、positive 比 neutral 高；
3. 意图基底：recommendation 最高、product_query 最低；
4. 决策类温度恒 0.0（由调用方保证，此处验证生成类不上界）。
"""

from __future__ import annotations

from agent_base.agents.supervisor import compute_generation_temperature


INTENTS = ["product_query", "price_query", "comparison", "recommendation", "aftersale", "general_qa", "unknown"]
EMOTIONS = ["anger", "anxiety", "positive", "neutral", "unknown"]


def test_range_all_combinations():
    """所有意图×情绪组合温度在 [0.05, 0.5]。"""
    for intent in INTENTS:
        for emotion in EMOTIONS:
            t = compute_generation_temperature(intent, emotion)
            assert 0.05 <= t <= 0.5, f"{intent}/{emotion}: {t}"


def test_emotion_direction():
    """情绪调节方向：anger 最低，positive 最高。"""
    for intent in INTENTS:
        t_anger = compute_generation_temperature(intent, "anger")
        t_neutral = compute_generation_temperature(intent, "neutral")
        t_positive = compute_generation_temperature(intent, "positive")
        assert t_anger < t_neutral, f"{intent}: anger 应低于 neutral"
        assert t_positive > t_neutral, f"{intent}: positive 应高于 neutral"


def test_intent_base_order():
    """意图基底：推荐 > 对比 > 售后 > 商品查询。"""
    assert compute_generation_temperature("recommendation") > compute_generation_temperature("comparison")
    assert compute_generation_temperature("comparison") > compute_generation_temperature("aftersale")
    assert compute_generation_temperature("aftersale") >= compute_generation_temperature("product_query")


def test_unknown_intent_fallback():
    """未知意图回退默认基底（不崩）。"""
    t = compute_generation_temperature("not_a_real_intent", "neutral")
    assert 0.05 <= t <= 0.5


def test_clamp_boundaries():
    """边界钳制：推荐+积极不超 0.5，售后+愤怒不低于 0.05。"""
    assert compute_generation_temperature("recommendation", "positive") <= 0.5
    assert compute_generation_temperature("aftersale", "anger") >= 0.05
