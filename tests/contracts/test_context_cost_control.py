"""契约 P28-2：上下文成本控制四级降级。

覆盖：
1. 档 1（<25%）全量注入，不裁剪消息条数；
2. 档 2（25-50%）滑动窗口：条数 + 每条长度受窗口参数约束；
3. 档 3（50-80%）规则压缩：摘要复用 + 首条锚点 + 最近 N 条，0 LLM；
4. 档 4（>=80%）needs_compact=True，由调用方触发 LLM 摘要；
5. 信息保全：商品锚点（玻尿酸精华）与最近轮次原文在降级后仍保留；
6. 成本指标完整：tier/ratio/raw/inject/saved/est_tokens。

纯内存函数测试，不依赖 Redis / LLM / PG。
"""

from __future__ import annotations

from agent_base.storage.chat_memory import (
    _trim_for_summarize,
    build_injectable_history,
    get_context_config,
)


def _cfg() -> dict:
    """固定档位阈值，避免依赖 configs/app.yaml 的当前值。"""
    return {
        "budget_chars": 60000,
        "tier_window_ratio": 0.25,
        "tier_rule_ratio": 0.50,
        "compact_trigger_ratio": 0.80,
        "compact_keep_recent": 8,
        "summary_max_chars": 400,
        "summarize_trim_chars": 8000,
        "window_recent_msgs": 16,
        "window_msg_chars": 800,
        "rule_first_msgs": 1,
        "rule_recent_msgs": 12,
        "llm_compact_model": "flash",
    }


def _msgs(n: int, chars: int = 200, prefix: str = "q") -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{prefix}{i}：" + "内" * chars}
        for i in range(n)
    ]


def test_tier1_full_inject():
    """档 1：用量 <25%，全量注入，指标正确。"""
    history = _msgs(5, chars=100)  # 5*~200 = ~1000 字符
    injectable, metrics = build_injectable_history(history, _cfg())
    assert metrics["tier"] == 1
    assert metrics["needs_compact"] is False
    assert len(injectable) == len(history)
    assert metrics["inject_msgs"] == len(history)
    assert metrics["saved_chars"] == 0
    assert "est_tokens" in metrics and metrics["est_tokens"] > 0


def test_tier2_window():
    """档 2：用量 25-50%，滑动窗口保留最近 N 条。"""
    history = _msgs(30, chars=700)  # 30*~700 = ~2.1 万 → ratio ~0.35
    injectable, metrics = build_injectable_history(history, _cfg())
    assert metrics["tier"] == 2
    assert len(injectable) <= 16  # window_recent_msgs
    assert metrics["inject_msgs"] <= 16
    assert metrics["saved_chars"] > 0
    # 最近一条必须保留
    assert injectable[-1]["content"].startswith("q29")


def test_tier3_rule_compress_anchor():
    """档 3：50-80%，规则压缩保留首条锚点 + 最近 N 条，商品锚点不丢。"""
    history = [
        {"role": "user", "content": "玻尿酸精华适合敏感肌吗？"},
        {"role": "assistant", "content": "适合，配方不含酒精。"},
        *[{"role": "user", "content": f"追问 {i}：" + "内" * 1500} for i in range(22)],
        {"role": "user", "content": "那这款适合干皮还是油皮？"},
    ]
    injectable, metrics = build_injectable_history(history, _cfg())
    assert metrics["tier"] == 3
    assert metrics["needs_compact"] is False
    text = "".join(m["content"] for m in injectable)
    assert "玻尿酸精华" in text  # 首条锚点
    assert "那这款" in text  # 最近轮次
    assert "追问 19" in text  # 最近 N 条


def test_tier3_summary_reuse():
    """档 3：已有摘要时直接复用，不重复调 LLM。"""
    history = [
        {"role": "system", "content": "【对话摘要】用户干皮敏感肌，咨询过玻尿酸精华。"},
        {"role": "user", "content": "玻尿酸精华适合敏感肌吗？"},
        *[{"role": "user", "content": f"追问 {i}：" + "内" * 1500} for i in range(22)],
    ]
    injectable, metrics = build_injectable_history(history, _cfg())
    assert metrics["tier"] == 3
    assert metrics["summary_reused"] is True
    assert any("【对话摘要】" in m["content"] for m in injectable)


def test_tier4_needs_compact():
    """档 4：>=80%，needs_compact=True，提示调用方触发 LLM 摘要。"""
    history = _msgs(40, chars=1500)  # 40*~1500 = ~6 万 → ratio ~1.0
    injectable, metrics = build_injectable_history(history, _cfg())
    assert metrics["tier"] == 4
    assert metrics["needs_compact"] is True
    assert metrics["saved_pct"] > 0


def test_empty_history():
    """空历史：返回空注入 + 全 0 指标。"""
    injectable, metrics = build_injectable_history([], _cfg())
    assert injectable == []
    assert metrics["tier"] == 0
    assert metrics["raw_chars"] == 0


def test_max_msgs_cap():
    """max_msgs 限制注入条数（supervisor 轻量上下文用）。"""
    history = _msgs(20, chars=100)
    injectable, metrics = build_injectable_history(history, _cfg(), max_msgs=6)
    assert len(injectable) <= 6
    assert metrics["raw_msgs"] == 6


def test_trim_for_summarize():
    """摘要输入裁剪：保留首条锚点 + 最近内容，总量不超上限。"""
    history = _msgs(20, chars=500)
    trimmed = _trim_for_summarize(history, max_chars=2000)
    total = sum(len(m["content"]) for m in trimmed)
    assert total <= 2000
    assert trimmed[0]["content"] == history[0]["content"]  # 首条锚点保留
    assert trimmed[-1]["content"] == history[-1]["content"]  # 最近内容保留


def test_default_config_readable():
    """默认配置可从代码读取（不依赖 yaml）。"""
    cfg = get_context_config()
    assert cfg["budget_chars"] > 0
    assert 0 < cfg["tier_window_ratio"] < cfg["tier_rule_ratio"] < cfg["compact_trigger_ratio"] <= 1.0
