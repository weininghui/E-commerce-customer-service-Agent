"""契约测试：会话级导购状态机。

覆盖：进入边界（首轮不推销 / 售后不推销 / 闲聊不推销）、阶段推进
（挖需 → 推荐 → 异议 → 促单 → 连带 → 售后）、阶段持久化。
"""

from __future__ import annotations

from agent_base.agents.sales_stage import (
    STAGE_AFTER,
    STAGE_CLOSE,
    STAGE_CONSULT,
    STAGE_EVALUATE,
    STAGE_HESITATE,
    STAGE_NONE,
    build_sales_context,
    build_stage_guide,
    decide_sales_step,
    load_stage,
    load_media_offer,
    reset_media_offer,
    reset_stage,
    save_stage,
    save_media_offer,
)


def _route(**overrides) -> dict:
    base = {
        "intent": "product_query",
        "sub_intent": "",
        "buying_signal": "normal",
        "objection_type": "none",
        "missing_info": [],
        "emotion": "neutral",
    }
    base.update(overrides)
    return base


# ── 进入边界：首轮不硬推销 ──


def test_normal_product_question_does_not_enter_sales():
    d = decide_sales_step(_route(), STAGE_NONE, question="这款精华的成分是什么")
    assert d.stage == STAGE_NONE
    assert d.action == "answer"


def test_buying_signal_enters_consult_and_asks_requirements():
    d = decide_sales_step(
        _route(buying_signal="buying", missing_info=["skin_type"]),
        STAGE_NONE,
        question="这个精华适合我吗，有点想买",
    )
    assert d.stage == STAGE_CONSULT
    assert d.action == "clarify_requirements"


def test_recommendation_enters_consult():
    d = decide_sales_step(
        _route(intent="recommendation", missing_info=["budget"]),
        STAGE_NONE,
        question="敏感肌用什么面霜",
    )
    assert d.stage == STAGE_CONSULT
    assert d.action == "clarify_requirements"


def test_first_turn_objection_handled():
    d = decide_sales_step(
        _route(buying_signal="objection", objection_type="price"),
        STAGE_NONE,
        question="99还是有点贵",
    )
    assert d.stage == STAGE_CONSULT
    assert d.action == "objection_handle"


# ── 售后 / 闲聊 / 促销永不推销 ──


def test_aftersale_never_sells_from_active_stage():
    d = decide_sales_step(
        _route(intent="aftersale"),
        STAGE_EVALUATE,
        question="拆封了还能退货吗",
    )
    assert d.stage == STAGE_AFTER
    assert d.action == "answer"


def test_chat_never_sells():
    d = decide_sales_step(
        _route(sub_intent="chat", intent="general_qa"),
        STAGE_CLOSE,
        question="好的，谢谢",
    )
    assert d.action == "answer"


def test_promotion_inquiry_is_answer_only():
    d = decide_sales_step(
        _route(intent="promotion", buying_signal="objection", objection_type="price"),
        STAGE_NONE,
        question="双十一有优惠吗",
    )
    assert d.action == "answer"


def test_anger_triggers_handoff():
    d = decide_sales_step(
        _route(emotion="anger", intent="aftersale"),
        STAGE_EVALUATE,
        question="你们太过分了，我要投诉",
    )
    assert d.action == "handoff"


# ── 阶段推进 ──


def test_consult_advances_to_recommend_when_requirements_met():
    d = decide_sales_step(
        _route(buying_signal="buying", missing_info=[]),
        STAGE_CONSULT,
        question="我是油皮，预算三百以内",
    )
    assert d.stage == STAGE_EVALUATE
    assert d.action == "recommend"


def test_consult_advances_when_user_provides_requirements():
    d = decide_sales_step(
        _route(buying_signal="normal", missing_info=["skin_type", "budget"]),
        STAGE_CONSULT,
        question="我是干皮，预算三百左右",
    )
    assert d.stage == STAGE_EVALUATE
    assert d.action == "recommend"


def test_consult_objection_moves_to_hesitate():
    d = decide_sales_step(
        _route(buying_signal="objection", objection_type="price"),
        STAGE_CONSULT,
        question="感觉还是有点贵",
    )
    assert d.stage == STAGE_HESITATE
    assert d.action == "objection_handle"


def test_evaluate_buying_signal_moves_to_close():
    d = decide_sales_step(
        _route(buying_signal="buying"),
        STAGE_EVALUATE,
        question="这个可以，下单吧",
    )
    assert d.stage == STAGE_CLOSE
    assert d.action == "close_attempt"


def test_evaluate_recommend_request_keeps_recommending():
    d = decide_sales_step(
        _route(sub_intent="recommend_request"),
        STAGE_EVALUATE,
        question="还有什么适合我的精华吗",
    )
    assert d.stage == STAGE_EVALUATE
    assert d.action == "recommend"


def test_evaluate_objection_moves_to_hesitate():
    d = decide_sales_step(
        _route(buying_signal="objection", objection_type="risk"),
        STAGE_EVALUATE,
        question="万一不适合能退吗",
    )
    assert d.stage == STAGE_HESITATE
    assert d.action == "objection_handle"


def test_hesitate_confirm_moves_to_close():
    d = decide_sales_step(
        _route(buying_signal="buying"),
        STAGE_HESITATE,
        question="好吧，那就它了",
    )
    assert d.stage == STAGE_CLOSE
    assert d.action == "close_attempt"


def test_hesitate_give_up_resets_stage():
    d = decide_sales_step(
        _route(buying_signal="normal"),
        STAGE_HESITATE,
        question="算了，先不买了",
    )
    assert d.stage == STAGE_NONE
    assert d.action == "answer"


def test_close_confirm_moves_to_after():
    d = decide_sales_step(
        _route(buying_signal="buying"),
        STAGE_CLOSE,
        question="已经下单了",
    )
    assert d.stage == STAGE_AFTER
    assert d.action == "answer"


def test_after_restarts_on_new_buying_signal():
    """售后阶段后用户又开始问新品价格/想买 → 重新进入导购。"""
    d = decide_sales_step(
        _route(buying_signal="buying", sub_intent="price_inquiry"),
        STAGE_AFTER,
        question="这款白色纯棉T恤多少钱",
    )
    assert d.stage == STAGE_CONSULT
    assert d.action == "answer"


def test_close_without_objection_cross_sells():
    d = decide_sales_step(
        _route(buying_signal="normal"),
        STAGE_CLOSE,
        question="这个搭配什么好",
    )
    assert d.stage == STAGE_CLOSE
    assert d.action == "cross_sell"


# ── 阶段持久化 ──


def test_stage_persist_roundtrip():
    sid = "sales-stage-test-roundtrip"
    reset_stage(sid)
    try:
        assert load_stage(sid) == STAGE_NONE
        save_stage(sid, STAGE_CONSULT)
        assert load_stage(sid) == STAGE_CONSULT
    finally:
        reset_stage(sid)
    assert load_stage(sid) == STAGE_NONE


def test_stage_guide_for_clarify():
    guide = build_stage_guide(
        decide_sales_step(
            _route(buying_signal="buying", missing_info=["skin_type"]),
            STAGE_NONE,
            question="这个精华适合我吗",
        )
    )
    assert "1-2 个问题" in guide


def test_stage_guide_empty_for_answer():
    d = decide_sales_step(_route(), STAGE_NONE, question="成分是什么")
    assert build_stage_guide(d) == ""


# ── 会话理解上下文（接线层） ──


def test_build_sales_context_buying_saves_stage():
    sid = "sales-ctx-test-buying"
    reset_stage(sid)
    try:
        ctx = build_sales_context(
            _route(buying_signal="buying", missing_info=["skin_type"]),
            sid,
            question="这个精华适合我吗，有点想买",
        )
        assert ctx["stage"] == STAGE_CONSULT
        assert ctx["action"] == "clarify_requirements"
        assert "导购" in ctx["sales_strategy"]
        assert ctx["guide"] != ""
        assert load_stage(sid) == STAGE_CONSULT
    finally:
        reset_stage(sid)


def test_build_sales_context_aftersale_no_strategy():
    sid = "sales-ctx-test-after"
    reset_stage(sid)
    try:
        ctx = build_sales_context(
            _route(intent="aftersale"),
            sid,
            question="拆封了还能退货吗",
        )
        assert ctx["action"] == "answer"
        assert ctx["sales_strategy"] == ""
    finally:
        reset_stage(sid)


def test_build_sales_context_normal_first_turn_no_sales():
    sid = "sales-ctx-test-normal"
    reset_stage(sid)
    try:
        ctx = build_sales_context(
            _route(),
            sid,
            question="这款精华的成分是什么",
        )
        assert ctx["stage"] == STAGE_NONE
        assert ctx["action"] == "answer"
        assert ctx["sales_strategy"] == ""
    finally:
        reset_stage(sid)


def test_profile_resolves_missing_requirements():
    """画像已含肤质/价位时，不再重复挖需，直接进入推荐。"""
    d = decide_sales_step(
        _route(buying_signal="buying", missing_info=["skin_type", "budget"]),
        STAGE_NONE,
        question="这个精华适合我吗，有点想买",
        profile={"skin_type": "干皮", "price_band": "中端"},
    )
    assert d.stage == STAGE_CONSULT
    assert d.action == "recommend"


def test_profile_partial_keeps_unresolved_missing():
    """画像只覆盖肤质时，预算仍需要确认。"""
    d = decide_sales_step(
        _route(buying_signal="buying", missing_info=["skin_type", "budget"]),
        STAGE_NONE,
        question="这个精华适合我吗，有点想买",
        profile={"skin_type": "干皮"},
    )
    assert d.stage == STAGE_CONSULT
    assert d.action == "clarify_requirements"


def test_build_sales_context_accepts_profile():
    sid = "sales-ctx-test-profile"
    reset_stage(sid)
    try:
        ctx = build_sales_context(
            _route(buying_signal="buying", missing_info=["skin_type"]),
            sid,
            question="这个精华适合我吗，有点想买",
            profile={"skin_type": "干皮"},
        )
        assert ctx["action"] == "recommend"
        assert ctx["profile"] == {"skin_type": "干皮"}
    finally:
        reset_stage(sid)


def test_media_offer_flag_set_once():
    sid = "sales-offer-test-once"
    reset_stage(sid)
    reset_media_offer(sid)
    try:
        ctx1 = build_sales_context(
            _route(buying_signal="buying"),
            sid,
            question="这个精华适合我吗，有点想买",
        )
        assert ctx1["offer_media"] is True
        ctx2 = build_sales_context(
            _route(buying_signal="buying"),
            sid,
            question="这个精华适合我吗，有点想买",
        )
        assert ctx2["offer_media"] is False
        assert load_media_offer(sid) is True
    finally:
        reset_stage(sid)
        reset_media_offer(sid)


def test_media_request_does_not_offer_media():
    sid = "sales-offer-test-media"
    reset_media_offer(sid)
    try:
        ctx = build_sales_context(
            _route(sub_intent="media_request", buying_signal="buying"),
            sid,
            question="看看这件商品的图片",
        )
        assert ctx["offer_media"] is False
        assert load_media_offer(sid) is False
    finally:
        reset_media_offer(sid)


def test_media_offer_store_roundtrip():
    sid = "sales-offer-test-store"
    reset_media_offer(sid)
    try:
        assert load_media_offer(sid) is False
        save_media_offer(sid)
        assert load_media_offer(sid) is True
    finally:
        reset_media_offer(sid)
    assert load_media_offer(sid) is False
