"""契约测试：回答输出格式（标准 Markdown、可读、无内部字段名）。"""

from __future__ import annotations

from agent_base.chains.qa_chain import _format_answer
from agent_base.chains.safety_chain import SafetyAssessment


def _safety() -> SafetyAssessment:
    return SafetyAssessment(
        risk_level="low",
        warnings=["敏感肌使用前建议局部测试"],
        findings=[],
        must_consult=False,
        emergency=False,
    )


def test_fallback_answer_is_readable_markdown():
    """兜底模板：标准 Markdown 结构，不出现内部字段名。"""
    out = _format_answer(
        conclusion="这款精华适合油皮，质地清爽。",
        evidence="- 质地清爽不粘腻\n- 不含酒精、不含香精",
        guidance="- 洁面后取 2-3 滴按压上脸",
        safety=_safety(),
        sources="玻尿酸精华.md",
    )
    assert "**商品资料要点**" in out
    assert "安全等级" not in out
    assert "风险标签" not in out
    assert "- " in out


def test_qa_system_prompt_allows_markdown_and_bans_fluff():
    """生成规范：允许标准 Markdown，禁止章节字段名与废话。"""
    from agent_base.prompts import get_prompt

    system = get_prompt("qa", "system")
    assert "Markdown" in system or "## " in system
    # 内部字段名以"禁止出现"的禁令形式存在（约束输出，而非作为输出标签）
    assert "禁止出现" in system
    assert "内部字段名" in system
    assert "禁废话" in system
