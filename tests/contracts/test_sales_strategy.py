"""契约测试：专家级销售策略（导购模式）——信号识别 / 策略注入。"""

from __future__ import annotations

from unittest.mock import patch

from agent_base.agents.sales import build_sales_strategy, detect_sales_signal


# ── 购买信号识别 ──


def test_detect_buying_signal():
    assert detect_sales_signal("这个精华适合我吗，有点想买")["mode"] == "buying"
    assert detect_sales_signal("想下单了")["mode"] == "buying"
    assert detect_sales_signal("白色纯棉T恤多少钱")["mode"] == "buying"


def test_detect_price_objection():
    assert detect_sales_signal("99还是有点贵")["mode"] == "objection"
    assert detect_sales_signal("太贵了，能便宜点吗")["objection_type"] == "price"


def test_detect_hesitant():
    assert detect_sales_signal("我再想想吧")["mode"] == "objection"
    assert detect_sales_signal("有点纠结要不要买")["objection_type"] == "hesitant"


def test_detect_risk_concern():
    assert detect_sales_signal("怕买错尺码不合适")["objection_type"] == "risk"
    assert detect_sales_signal("万一不好用怎么办")["mode"] == "objection"


def test_detect_normal_question():
    assert detect_sales_signal("你们几点发货")["mode"] == "normal"
    assert detect_sales_signal("介绍一下这款精华的成分")["mode"] == "normal"


def test_discount_ask_is_not_price_objection():
    """「有优惠吗/满减活动」是促销咨询，不是嫌贵异议。"""
    assert detect_sales_signal("双十一有优惠吗")["mode"] != "objection"
    assert detect_sales_signal("现在有满减活动吗")["mode"] != "objection"


# ── 策略注入 ──


def test_build_strategy_for_buying_product_question():
    s = build_sales_strategy("这个精华适合我吗，有点想买", intent="product_query")
    assert "导购" in s


def test_build_strategy_for_objection():
    s = build_sales_strategy("99还是有点贵", intent="price_query")
    assert "导购" in s


def test_no_strategy_for_normal_question():
    assert build_sales_strategy("你们几点发货", intent="product_query") == ""


def test_no_strategy_for_non_sales_intent():
    assert build_sales_strategy("想买退货方便的", intent="aftersale") == ""


def test_controlled_llm_answer_appends_sales_strategy():
    """受控生成层：有购买信号时系统提示词追加导购策略块。"""
    from agent_base.chains.qa_chain import _controlled_llm_answer
    from agent_base.chains.safety_chain import SafetyAssessment

    safety = SafetyAssessment(risk_level="low", warnings=[], findings=[], must_consult=False, emergency=False)

    class _Fake:
        """LCEL 可调用模型：记录 system 提示词后返回固定回答。"""

        def __init__(self, captured):
            self._captured = captured

        def __call__(self, messages):
            # LCEL 传入 ChatPromptValue，取消息列表
            msgs = getattr(messages, "to_messages", lambda: messages)()
            self._captured["system"] = msgs[0].content
            return "好的，亲～"

    captured = {}
    fake = _Fake(captured)
    with patch("agent_base.chains.qa_chain.build_chat_model", return_value=fake):
        _controlled_llm_answer(
            question="这个精华适合我吗，有点想买",
            conclusion="结论",
            evidence="证据",
            guidance="建议",
            safety=safety,
            sources="来源",
            llm_provider="langchain",
            llm_model=None,
            llm_base_url=None,
            llm_api_key_env="X",
            llm_temperature=0.1,
            prompts_path="configs/prompts_ecommerce.yaml",
            sales_strategy=build_sales_strategy("这个精华适合我吗，有点想买", intent="product_query"),
        )
    assert "导购模式" in captured["system"]
