"""契约测试：销售评测框架（离线评分 + 合规扫描 + 报告）。"""

from __future__ import annotations

from agent_base.sales_eval import _compliance_scan, build_report, eval_signal_strategy, load_cases, run_offline


def test_load_cases_default():
    cases = load_cases()
    assert len(cases) >= 15
    ids = {c.get("id") for c in cases}
    assert "buying_01" in ids and "compliance_01" in ids


def test_eval_signal_strategy_hit():
    row = eval_signal_strategy("这个玻尿酸精华适合我吗，有点想买", {"intent": "product_query", "signal": {"mode": "buying"}, "strategy": True})
    assert row["signal_hit"] is True
    assert row["strategy_hit"] is True
    assert row["intent_hit"] is True


def test_eval_signal_strategy_normal_not_triggered():
    row = eval_signal_strategy("你们几点发货", {"signal": {"mode": "normal"}, "strategy": False})
    assert row["signal_hit"] is True
    assert row["strategy_hit"] is True


def test_offline_run_aggregates():
    cases = load_cases(limit=5)
    result = run_offline(cases)
    assert result["mode"] == "offline"
    assert result["total_cases"] == 5
    assert 0 <= result["signal_acc"] <= 1
    assert len(result["details"]) == 5


def test_compliance_scan_flags_absolute_claims():
    hits = _compliance_scan("这个效果绝对是最好的，100%有效")
    assert any("绝对" in h for h in hits)
    assert any("100%" in h for h in hits)


def test_compliance_scan_clean():
    assert _compliance_scan("这款质地清爽，成分温和，适合多数肤质") == []


def test_build_report_contains_sections():
    cases = load_cases(limit=5)
    report = build_report(run_offline(cases))
    assert "销售评测报告" in report
    assert "失败归因" in report
    assert "优化建议" in report
