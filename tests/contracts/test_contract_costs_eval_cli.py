"""契约测试：Token 成本估算 + 一键评测 CLI 报告。"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from agent_base.eval_cli import _load_cases, build_markdown_report
from agent_base.monitoring.costs import estimate_cost


def test_estimate_cost_known_model():
    # deepseek-v4-flash：(1.0, 2.0) 元 / 1M
    cost = estimate_cost("deepseek-v4-flash", 1_000_000, 500_000)
    assert round(cost, 4) == round(1.0 + 1.0, 4)


def test_estimate_cost_default_fallback():
    cost = estimate_cost("some-unknown-model", 1_000_000, 0)
    assert round(cost, 6) == 1.0


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._cursor = _FakeCursor(rows)

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_token_usage_stats_includes_cost():
    from agent_base.storage.pg import token_usage_stats

    ts = datetime(2026, 8, 16, tzinfo=timezone.utc)
    rows = [
        (ts, "agent_a", "deepseek-v4-flash", 1_000_000, 500_000, 1_500_000, 120),
        (ts, "agent_b", "unknown-model", 1_000_000, 0, 1_000_000, 80),
    ]
    with patch("agent_base.storage.pg._conn", return_value=_FakeConn(rows)):
        stats = token_usage_stats(days=7, group_by="day")
    assert stats["rows"]
    assert stats["total_cost"] > 0
    assert all("cost" in r for r in stats["rows"])


def test_eval_cli_load_cases_limit():
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "cases.json"
        p.write_text(
            json.dumps(
                [
                    {"question": "q1", "expected_intent": "i1"},
                    {"question": "q2", "expected_intent": "i2"},
                    {"question": "q3", "expected_intent": "i3"},
                ]
            ),
            encoding="utf-8",
        )
        cases = _load_cases(str(p), limit=2)
        assert [c["question"] for c in cases] == ["q1", "q2"]


def test_eval_cli_build_report():
    result = {
        "run_id": 1,
        "name": "回归",
        "generation": "custom",
        "total_cases": 2,
        "intent_acc": 0.5,
        "recall_acc": 1.0,
        "fact_acc": 1.0,
        "compliance_acc": 1.0,
        "faithfulness_acc": 0.8,
        "relevancy_acc": 0.9,
        "overall": 0.8,
        "cases": [
            {
                "question": "面膜能天天用吗",
                "expected_intent": "usage",
                "actual_intent": "usage",
                "intent_hit": True,
                "recall_hit": True,
                "fact_hits": 1,
                "fact_total": 1,
                "compliance_ok": True,
                "faithfulness": 0.8,
                "relevancy": 0.9,
            },
            {
                "question": "有没有药妆",
                "expected_intent": "ingredient",
                "actual_intent": "other",
                "intent_hit": False,
                "recall_hit": False,
                "fact_hits": 0,
                "fact_total": 1,
                "compliance_ok": True,
                "faithfulness": 0.0,
                "relevancy": 0.2,
                "error": "",
            },
        ],
    }
    md = build_markdown_report(result)
    assert "综合得分" in md
    assert "失败归因" in md
    assert "用例明细" in md
    assert "intent_miss" in md
