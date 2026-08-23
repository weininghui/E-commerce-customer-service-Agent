"""会话级意图识别 + 导购状态机离线评测（v2）。

用法：
    python scripts/eval_intent_stage.py
    python scripts/eval_intent_stage.py --out reports/intent_stage_eval.md

评测维度：
- 主意图 Top-1 准确率（规则层，离线确定性，不调 LLM/embedding）；
- 子意图准确率；
- 购买信号 / 异议类型准确率；
- 阶段 / 动作决策准确率（单轮 + 多轮场景流）；
- 不推销红线：闲聊 / 售后 / 促销 / 首轮普通咨询 → 动作必须为 answer 且销售策略为空。

输出：reports/intent_stage_eval.json 与 .md 报告。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_base.agents.sales_stage import (  # noqa: E402
    STAGE_NONE,
    build_sales_context,
    decide_sales_step,
    reset_stage,
)
from agent_base.domain import load_domain  # noqa: E402
from agent_base.retrieval.intent_router import route_question  # noqa: E402


# ── 单轮用例（人工标注期望值，作为规则回归基线） ─────────────────────────────

CASES: list[dict[str, Any]] = [
    # 商品咨询
    {"id": "pq_01", "q": "玻尿酸精华适合油皮吗", "intent": "product_query", "sub": "", "signal": "normal", "obj": "none"},
    {"id": "pq_02", "q": "这款面霜的成分表有防腐剂吗", "intent": "product_query", "sub": "", "signal": "normal", "obj": "none"},
    {"id": "pq_03", "q": "烟酰胺控油精华有什么功效", "intent": "product_query", "sub": "", "signal": "normal", "obj": "none"},
    {"id": "pq_04", "q": "氨基酸洁面乳含酒精吗", "intent": "product_query", "sub": "", "signal": "normal", "obj": "none"},
    {"id": "pq_05", "q": "敏感肌能用含水杨酸的产品吗", "intent": "product_query", "sub": "allergy", "signal": "normal", "obj": "none"},
    {"id": "pq_06", "q": "这款精华孕妇可以用吗", "intent": "product_query", "sub": "allergy", "signal": "normal", "obj": "none"},
    {"id": "pq_07", "q": "胜肽紧致眼霜适合什么年龄段", "intent": "product_query", "sub": "", "signal": "normal", "obj": "none"},
    # 穿搭服饰
    {"id": "fq_01", "q": "白色T恤怎么搭配通勤", "intent": "fashion_query", "sub": "", "signal": "normal", "obj": "none"},
    {"id": "fq_02", "q": "醋酸衬衫适合通勤吗", "intent": "fashion_query", "sub": "", "signal": "normal", "obj": "none"},
    {"id": "fq_03", "q": "这件针织开衫会缩水吗", "intent": "fashion_query", "sub": "", "signal": "normal", "obj": "none"},
    {"id": "fq_04", "q": "白色衬衫会不会透", "intent": "fashion_query", "sub": "", "signal": "normal", "obj": "none"},
    {"id": "fq_05", "q": "高腰阔腿裤怎么搭配", "intent": "fashion_query", "sub": "", "signal": "normal", "obj": "none"},
    # 价格
    {"id": "pr_01", "q": "这款精华多少钱", "intent": "price_query", "sub": "price_inquiry", "signal": "buying", "obj": "none"},
    {"id": "pr_02", "q": "这个能便宜点吗", "intent": "price_query", "sub": "price_negotiation", "signal": "objection", "obj": "price"},
    {"id": "pr_03", "q": "99还是有点贵", "intent": "price_query", "sub": "price_negotiation", "signal": "objection", "obj": "price"},
    {"id": "pr_04", "q": "这个价位划不划算", "intent": "price_query", "sub": "price_negotiation", "signal": "objection", "obj": "price"},
    # 售后物流
    {"id": "as_01", "q": "拆封了还能退货吗", "intent": "aftersale", "sub": "return_exchange", "signal": "normal", "obj": "none"},
    {"id": "as_02", "q": "退款多久到账", "intent": "aftersale", "sub": "return_exchange", "signal": "normal", "obj": "none"},
    {"id": "as_03", "q": "什么时候发货", "intent": "aftersale", "sub": "logistics", "signal": "normal", "obj": "none"},
    {"id": "as_04", "q": "支持七天无理由退货吗", "intent": "aftersale", "sub": "return_exchange", "signal": "normal", "obj": "none"},
    {"id": "as_05", "q": "用了过敏可以退吗", "intent": "aftersale", "sub": "return_exchange", "signal": "normal", "obj": "none"},
    {"id": "as_06", "q": "订单显示已发货但查不到物流", "intent": "aftersale", "sub": "logistics", "signal": "normal", "obj": "none"},
    # 推荐
    {"id": "rc_01", "q": "敏感肌泛红用什么面霜", "intent": "recommendation", "sub": "recommend_request", "signal": "normal", "obj": "none"},
    {"id": "rc_02", "q": "帮我推荐一款精华", "intent": "recommendation", "sub": "recommend_request", "signal": "normal", "obj": "none"},
    {"id": "rc_03", "q": "干皮秋冬用什么面霜", "intent": "recommendation", "sub": "recommend_request", "signal": "normal", "obj": "none"},
    {"id": "rc_04", "q": "送女朋友什么护肤品好", "intent": "recommendation", "sub": "recommend_request", "signal": "normal", "obj": "none"},
    {"id": "rc_05", "q": "预算三百以内推荐什么精华", "intent": "recommendation", "sub": "recommend_request", "signal": "normal", "obj": "none"},
    # 对比
    {"id": "cp_01", "q": "氨基酸洁面和皂基洁面有什么区别", "intent": "comparison", "sub": "compare", "signal": "normal", "obj": "none"},
    {"id": "cp_02", "q": "这款和那个哪个好", "intent": "comparison", "sub": "compare", "signal": "normal", "obj": "none"},
    {"id": "cp_03", "q": "玻尿酸精华和烟酰胺精华怎么选", "intent": "comparison", "sub": "compare", "signal": "normal", "obj": "none"},
    # 尺码
    {"id": "sz_01", "q": "我165怎么选尺码", "intent": "size_recommendation", "sub": "recommend_request", "signal": "normal", "obj": "none"},
    {"id": "sz_02", "q": "身高160体重50穿什么码", "intent": "size_recommendation", "sub": "recommend_request", "signal": "normal", "obj": "none"},
    {"id": "sz_03", "q": "这件T恤偏大吗", "intent": "size_recommendation", "sub": "", "signal": "normal", "obj": "none"},
    # 促销
    {"id": "pm_01", "q": "现在有什么满减活动吗", "intent": "promotion", "sub": "", "signal": "buying", "obj": "none"},
    {"id": "pm_02", "q": "双十一有优惠吗", "intent": "promotion", "sub": "", "signal": "buying", "obj": "none"},
    {"id": "pm_03", "q": "新人首单有什么折扣", "intent": "promotion", "sub": "", "signal": "buying", "obj": "none"},
    # 闲聊/泛问答
    {"id": "ch_01", "q": "你好呀", "intent": "general_qa", "sub": "chat", "signal": "normal", "obj": "none"},
    {"id": "ch_02", "q": "谢谢", "intent": "general_qa", "sub": "chat", "signal": "normal", "obj": "none"},
    {"id": "ch_03", "q": "在吗", "intent": "general_qa", "sub": "chat", "signal": "normal", "obj": "none"},
    {"id": "ch_04", "q": "今天天气不错", "intent": "general_qa", "sub": "", "signal": "normal", "obj": "none"},
    # 用法/评价子意图
    {"id": "us_01", "q": "这个精华怎么用", "intent": "product_query", "sub": "usage", "signal": "normal", "obj": "none"},
    {"id": "rv_01", "q": "这个好用吗，有没有人用过", "intent": "product_query", "sub": "review", "signal": "normal", "obj": "none"},
    # 媒体请求（按需展示）
    {"id": "md_01", "q": "有图片吗", "intent": "general_qa", "sub": "media_request", "signal": "normal", "obj": "none"},
    {"id": "md_02", "q": "看看实物", "intent": "general_qa", "sub": "media_request", "signal": "normal", "obj": "none"},
    {"id": "md_03", "q": "有视频吗", "intent": "general_qa", "sub": "media_request", "signal": "normal", "obj": "none"},
    # 口语化/否定/指代
    {"id": "sl_01", "q": "这个精华适合我吗，有点想买", "intent": "product_query", "sub": "", "signal": "buying", "obj": "none"},
    {"id": "sl_02", "q": "它多少钱", "intent": "price_query", "sub": "price_inquiry", "signal": "buying", "obj": "none"},
    {"id": "sl_03", "q": "不含酒精吧？敏感肌能用不", "intent": "product_query", "sub": "allergy", "signal": "normal", "obj": "none"},
    {"id": "sl_04", "q": "有点纠结要不要买", "intent": "product_query", "sub": "", "signal": "objection", "obj": "hesitant"},
    {"id": "sl_05", "q": "万一不适合能退吗", "intent": "aftersale", "sub": "return_exchange", "signal": "objection", "obj": "risk"},
]


# ── 单轮阶段决策用例（current_stage 为输入） ─────────────────────────────────

STAGE_CASES: list[dict[str, Any]] = [
    {"id": "st_01", "route": {"intent": "product_query"}, "stage": STAGE_NONE, "q": "成分是什么", "exp_stage": STAGE_NONE, "exp_action": "answer"},
    {"id": "st_02", "route": {"intent": "product_query", "buying_signal": "buying", "missing_info": ["skin_type"]}, "stage": STAGE_NONE, "q": "适合我吗，想买", "exp_stage": "consult", "exp_action": "clarify_requirements"},
    {"id": "st_03", "route": {"intent": "recommendation", "missing_info": ["budget"]}, "stage": STAGE_NONE, "q": "敏感肌用什么面霜", "exp_stage": "consult", "exp_action": "clarify_requirements"},
    {"id": "st_04", "route": {"intent": "price_query", "buying_signal": "objection", "objection_type": "price"}, "stage": STAGE_NONE, "q": "99还是有点贵", "exp_stage": "consult", "exp_action": "objection_handle"},
    {"id": "st_05", "route": {"intent": "product_query", "buying_signal": "buying"}, "stage": "consult", "q": "我是干皮，预算三百", "exp_stage": "evaluate", "exp_action": "recommend"},
    {"id": "st_06", "route": {"intent": "product_query", "buying_signal": "objection", "objection_type": "price"}, "stage": "consult", "q": "还是有点贵", "exp_stage": "hesitate", "exp_action": "objection_handle"},
    {"id": "st_07", "route": {"intent": "product_query", "buying_signal": "buying"}, "stage": "evaluate", "q": "这个可以，下单吧", "exp_stage": "close", "exp_action": "close_attempt"},
    {"id": "st_08", "route": {"intent": "product_query", "buying_signal": "objection", "objection_type": "risk"}, "stage": "evaluate", "q": "怕不合适", "exp_stage": "hesitate", "exp_action": "objection_handle"},
    {"id": "st_09", "route": {"intent": "product_query", "buying_signal": "objection", "objection_type": "price"}, "stage": "hesitate", "q": "还是觉得贵", "exp_stage": "hesitate", "exp_action": "objection_handle"},
    {"id": "st_10", "route": {"intent": "product_query", "buying_signal": "buying"}, "stage": "hesitate", "q": "好吧，就它了", "exp_stage": "close", "exp_action": "close_attempt"},
    {"id": "st_11", "route": {"intent": "product_query"}, "stage": "hesitate", "q": "算了，先不买了", "exp_stage": STAGE_NONE, "exp_action": "answer"},
    {"id": "st_12", "route": {"intent": "product_query", "buying_signal": "buying"}, "stage": "close", "q": "已经下单了", "exp_stage": "after", "exp_action": "answer"},
    {"id": "st_13", "route": {"intent": "product_query"}, "stage": "close", "q": "这个搭配什么好", "exp_stage": "close", "exp_action": "cross_sell"},
    {"id": "st_14", "route": {"intent": "product_query"}, "stage": "after", "q": "怎么清洗", "exp_stage": "after", "exp_action": "answer"},
    {"id": "st_15", "route": {"intent": "aftersale"}, "stage": "evaluate", "q": "拆封了还能退吗", "exp_stage": "after", "exp_action": "answer"},
    {"id": "st_16", "route": {"intent": "general_qa", "sub_intent": "chat"}, "stage": "close", "q": "好的，谢谢", "exp_stage": "close", "exp_action": "answer"},
    {"id": "st_17", "route": {"intent": "promotion", "buying_signal": "buying"}, "stage": STAGE_NONE, "q": "双十一有优惠吗", "exp_stage": STAGE_NONE, "exp_action": "answer"},
    {"id": "st_18", "route": {"intent": "aftersale", "emotion": "anger"}, "stage": "evaluate", "q": "我要投诉", "exp_stage": "evaluate", "exp_action": "handoff"},
]


# ── 多轮场景流（会话级阶段推进） ─────────────────────────────────────────────

FLOWS: list[dict[str, Any]] = [
    {
        "id": "flow_buy_ok",
        "name": "顺利成交：挖需 → 推荐 → 异议 → 促单 → 已购",
        "turns": [
            {"q": "这个精华适合我吗，有点想买", "exp_stage": "consult", "exp_action": "clarify_requirements"},
            {"q": "我是干皮，预算三百左右", "exp_stage": "evaluate", "exp_action": "recommend"},
            {"q": "这款和那款哪个好", "exp_stage": "evaluate", "exp_action": "answer"},
            {"q": "有点纠结，怕不适合我", "exp_stage": "hesitate", "exp_action": "objection_handle"},
            {"q": "好吧，就它了", "exp_stage": "close", "exp_action": "close_attempt"},
            {"q": "已经下单了", "exp_stage": "after", "exp_action": "answer"},
        ],
    },
    {
        "id": "flow_price_objection",
        "name": "价格异议：询价 → 嫌贵 → 价值重述 → 促单",
        "turns": [
            {"q": "这款精华多少钱", "exp_stage": "consult", "exp_action": "answer"},
            {"q": "太贵了，能便宜点吗", "exp_stage": "hesitate", "exp_action": "objection_handle"},
            {"q": "行吧，那就入手", "exp_stage": "close", "exp_action": "close_attempt"},
        ],
    },
    {
        "id": "flow_hesitate_give_up",
        "name": "犹豫放弃：尊重用户，不留缠",
        "turns": [
            {"q": "这个适合我吗", "exp_stage": "consult", "exp_action": "clarify_requirements"},
            {"q": "我再想想吧", "exp_stage": "hesitate", "exp_action": "objection_handle"},
            {"q": "算了，先不买了", "exp_stage": STAGE_NONE, "exp_action": "answer"},
        ],
    },
    {
        "id": "flow_aftersale_pivot",
        "name": "售后切入：销售中转向售后，不再推销",
        "turns": [
            {"q": "这个精华适合我吗", "exp_stage": "consult", "exp_action": "clarify_requirements"},
            {"q": "其实我想问退货怎么弄", "exp_stage": "after", "exp_action": "answer"},
        ],
    },
    {
        "id": "flow_chat_pivot",
        "name": "闲聊切入：销售中闲聊，阶段保留但不推销",
        "turns": [
            {"q": "这个精华适合我吗", "exp_stage": "consult", "exp_action": "clarify_requirements"},
            {"q": "好的，谢谢", "exp_stage": "consult", "exp_action": "answer"},
        ],
    },
    {
        "id": "flow_recommend_ask",
        "name": "求推荐：挖需两次 → 给方案 → 连带推荐",
        "turns": [
            {"q": "帮我推荐一款精华", "exp_stage": "consult", "exp_action": "clarify_requirements"},
            {"q": "我是油皮，预算两百", "exp_stage": "evaluate", "exp_action": "recommend"},
            {"q": "这个搭配什么好", "exp_stage": "evaluate", "exp_action": "recommend"},
        ],
    },
    {
        "id": "flow_normal_no_sell",
        "name": "首轮普通咨询：连续咨询不推销",
        "turns": [
            {"q": "这款精华的成分是什么", "exp_stage": STAGE_NONE, "exp_action": "answer"},
            {"q": "适合油皮吗", "exp_stage": STAGE_NONE, "exp_action": "answer"},
        ],
    },
    {
        "id": "flow_promotion_never_sell",
        "name": "促销咨询：只答活动，不催单",
        "turns": [
            {"q": "双十一有优惠吗", "exp_stage": STAGE_NONE, "exp_action": "answer"},
            {"q": "满减能叠加吗", "exp_stage": STAGE_NONE, "exp_action": "answer"},
        ],
    },
    {
        "id": "flow_size_consult",
        "name": "尺码咨询：确认尺码后推荐",
        "turns": [
            {"q": "我165怎么选尺码", "exp_stage": "consult", "exp_action": "clarify_requirements"},
            {"q": "身高165体重50，喜欢宽松", "exp_stage": "evaluate", "exp_action": "recommend"},
        ],
    },
]


def _run_single(case: dict[str, Any], domain: Any) -> dict[str, Any]:
    """单轮用例：路由 + 阶段决策，返回逐字段命中情况。"""
    route = route_question(case["q"], domain=domain)
    return {
        "id": case["id"],
        "q": case["q"],
        "intent_ok": route.intent == case["intent"],
        "sub_ok": route.sub_intent == case["sub"],
        "signal_ok": route.buying_signal == case["signal"],
        "obj_ok": route.objection_type == case["obj"],
        "got": {
            "intent": route.intent,
            "sub": route.sub_intent,
            "signal": route.buying_signal,
            "obj": route.objection_type,
        },
        "exp": {
            "intent": case["intent"],
            "sub": case["sub"],
            "signal": case["signal"],
            "obj": case["obj"],
        },
    }


def _run_stage_case(case: dict[str, Any]) -> dict[str, Any]:
    decision = decide_sales_step(case["route"], case["stage"], question=case["q"])
    return {
        "id": case["id"],
        "q": case["q"],
        "stage_ok": decision.stage == case["exp_stage"],
        "action_ok": decision.action == case["exp_action"],
        "got": {"stage": decision.stage, "action": decision.action},
        "exp": {"stage": case["exp_stage"], "action": case["exp_action"]},
    }


def _run_flow(flow: dict[str, Any], domain: Any) -> dict[str, Any]:
    """多轮场景流：模拟会话阶段推进，校验每轮阶段/动作。"""
    session_id = f"eval-flow-{flow['id']}"
    reset_stage(session_id)
    turn_results: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for idx, turn in enumerate(flow["turns"]):
        route = route_question(turn["q"], domain=domain)
        ctx = build_sales_context(route.to_dict(), session_id, turn["q"], history=history)
        history.append({"role": "user", "content": turn["q"]})
        history.append({"role": "assistant", "content": "ok"})
        no_sell = (
            turn["exp_action"] == "answer"
            and ctx["sales_strategy"] == ""
            and ctx["guide"] == ""
        )
        turn_results.append(
            {
                "turn": idx + 1,
                "q": turn["q"],
                "stage_ok": ctx["stage"] == turn["exp_stage"],
                "action_ok": ctx["action"] == turn["exp_action"],
                "no_sell": no_sell,
                "got": {"stage": ctx["stage"], "action": ctx["action"]},
                "exp": {"stage": turn["exp_stage"], "action": turn["exp_action"]},
            }
        )
    return {"id": flow["id"], "name": flow["name"], "turns": turn_results}


def _rate(ok_flags: list[bool]) -> float:
    return round(100.0 * sum(ok_flags) / len(ok_flags), 1) if ok_flags else 0.0


def run_eval(domain: Any) -> dict[str, Any]:
    """执行全部评测用例并聚合指标。"""
    single = [_run_single(c, domain) for c in CASES]
    stages = [_run_stage_case(c) for c in STAGE_CASES]
    flows = [_run_flow(f, domain) for f in FLOWS]

    intent_ok = [s["intent_ok"] for s in single]
    sub_ok = [s["sub_ok"] for s in single]
    signal_ok = [s["signal_ok"] for s in single]
    obj_ok = [s["obj_ok"] for s in single]
    stage_ok = [s["stage_ok"] for s in stages] + [t["stage_ok"] for f in flows for t in f["turns"]]
    action_ok = [s["action_ok"] for s in stages] + [t["action_ok"] for f in flows for t in f["turns"]]

    no_sell_ok: list[bool] = []
    for flow in flows:
        for turn in flow["turns"]:
            if turn["exp"]["action"] == "answer":
                no_sell_ok.append(turn["no_sell"])

    return {
        "case_count": len(CASES),
        "stage_case_count": len(STAGE_CASES),
        "flow_count": len(FLOWS),
        "flow_turn_count": sum(len(f["turns"]) for f in flows),
        "total_checks": len(CASES) + len(STAGE_CASES) + sum(len(f["turns"]) for f in flows),
        "metrics": {
            "intent_acc": _rate(intent_ok),
            "sub_intent_acc": _rate(sub_ok),
            "signal_acc": _rate(signal_ok),
            "objection_acc": _rate(obj_ok),
            "stage_acc": _rate(stage_ok),
            "action_acc": _rate(action_ok),
            "no_sell_acc": _rate(no_sell_ok),
        },
        "single": single,
        "stages": stages,
        "flows": flows,
    }


def build_report(result: dict[str, Any]) -> str:
    """生成 Markdown 评测报告。"""
    m = result["metrics"]
    lines = [
        "# 会话级意图识别 + 导购状态机评测报告",
        "",
        f"- 用例数：单轮 {result['case_count']} 条 + 阶段决策 {result['stage_case_count']} 条 + "
        f"多轮场景流 {result['flow_count']} 条（共 {result['flow_turn_count']} 轮），合计检查 {result['total_checks']} 项",
        "- 评测模式：离线规则（不调 LLM / embedding，确定性）",
        "",
        "## 指标",
        "",
        "| 指标 | 得分 | 目标 |",
        "| --- | --- | --- |",
        f"| 主意图 Top-1 | {m['intent_acc']}% | ≥95% |",
        f"| 子意图 | {m['sub_intent_acc']}% | ≥90% |",
        f"| 购买信号 | {m['signal_acc']}% | ≥90% |",
        f"| 异议类型 | {m['objection_acc']}% | ≥90% |",
        f"| 阶段识别 | {m['stage_acc']}% | ≥90% |",
        f"| 动作决策 | {m['action_acc']}% | ≥90% |",
        f"| 不推销红线 | {m['no_sell_acc']}% | 100% |",
        "",
    ]
    lines.append("## 单轮意图用例失败明细")
    lines.append("")
    fails = [s for s in result["single"] if not all((s["intent_ok"], s["sub_ok"], s["signal_ok"], s["obj_ok"]))]
    if not fails:
        lines.append("- 无失败用例 ✓")
    for s in fails:
        lines.append(f"- `{s['id']}` {s['q']}：got={s['got']} exp={s['exp']}")
    lines.append("")
    lines.append("## 阶段/动作决策失败明细")
    lines.append("")
    s_fails = [s for s in result["stages"] if not (s["stage_ok"] and s["action_ok"])]
    f_fails = [
        (f["id"], t) for f in result["flows"] for t in f["turns"]
        if not (t["stage_ok"] and t["action_ok"] and t["no_sell"])
    ]
    if not s_fails and not f_fails:
        lines.append("- 无失败用例 ✓")
    for s in s_fails:
        lines.append(f"- `{s['id']}` {s['q']}：got={s['got']} exp={s['exp']}")
    for fid, t in f_fails:
        lines.append(f"- `{fid}` 第{t['turn']}轮 {t['q']}：got={t['got']} exp={t['exp']} no_sell={t['no_sell']}")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("达标后接入 CI：`python scripts/eval_intent_stage.py` 失败即阻断。")
    return "\n".join(lines)


TARGETS = {
    "intent_acc": 95.0,
    "sub_intent_acc": 90.0,
    "signal_acc": 90.0,
    "objection_acc": 90.0,
    "stage_acc": 90.0,
    "action_acc": 90.0,
    "no_sell_acc": 100.0,
}


def _check_targets(metrics: dict[str, float]) -> bool:
    """全部指标达到目标才返回 True；否则打印未达标项。"""
    ok = True
    for key, target in TARGETS.items():
        if metrics.get(key, 0.0) < target:
            ok = False
            print(f"  [未达标] {key}={metrics.get(key)}% < {target}%")
    return ok


def main() -> None:
    """评测入口：运行、写报告、校验目标并给出退出码。"""
    parser = argparse.ArgumentParser(description="会话级意图识别 + 导购状态机离线评测")
    parser.add_argument("--out", default="reports/intent_stage_eval.md", help="Markdown 报告输出路径")
    args = parser.parse_args()

    domain = load_domain("ecommerce")
    result = run_eval(domain)
    out_json = Path(args.out).with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out).write_text(build_report(result), encoding="utf-8")

    m = result["metrics"]
    print(
        f"intent={m['intent_acc']}% sub={m['sub_intent_acc']}% signal={m['signal_acc']}% "
        f"obj={m['objection_acc']}% stage={m['stage_acc']}% action={m['action_acc']}% no_sell={m['no_sell_acc']}%"
    )
    print(f"report: {args.out}")
    if not _check_targets(m):
        raise SystemExit("评测未达标，阻断 CI")


if __name__ == "__main__":
    main()
