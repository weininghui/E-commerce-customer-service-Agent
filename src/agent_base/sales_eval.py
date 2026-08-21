"""销售评测（基于销售测试驱动技术优化）。

目标：把「卖得好不好」变成可量化、可回归的评测闭环。

评测维度：
- 信号识别准确率：detect_sales_signal 是否命中预期（购买/异议/正常 + 异议类型）；
- 策略触发准确率：build_sales_strategy 是否按预期触发（该触发时不漏、不该触发时不误伤）；
- 意图路由准确率：route_question 是否命中预期商品类意图（导购模式前置）；
- 话术质量（--judge）：真实回答由 LLM 判官按 judge_guide 打分 1-5；
- 合规红线（--judge）：规则扫描回答中的绝对化用语/医疗用语/贬低竞品。

用法：
    python -m agent_base.sales_eval                # 离线：信号/策略/意图（零成本）
    python -m agent_base.sales_eval --judge        # 全链路：真实回答 + LLM 判官 + 合规
    python -m agent_base.sales_eval --cases xxx.yaml --limit 5

输出：终端摘要 + reports/sales_eval_<时间戳>.md/.json（含失败归因与优化建议）。

每次修改销售提示词/策略/意图后重跑，用分数变化验证优化效果（数据飞轮）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 合规红线词（规则扫描）
# 绝对化用语（正则：排除"第一次/第一个/第一时间/第一步"等安全组合）
_ABSOLUTE_RE = [
    r"绝对",
    r"最佳",
    r"100%",
    r"百分百",
    r"包治",
    r"根治",
    r"永不",
    r"最有效",
    r"全网最低",
    r"第一(?![次个时间步款名位场线件])",
    r"(最好|最棒|最强|第一名)",
]
_MEDICAL_WORDS = ["治疗", "治愈", "药妆", "疗效", "根治"]
_DISPARAGE_PATTERNS = ["智商税", "垃圾货", "不如别家", "别家都不行", "比别家差"]

CASES_DEFAULT = "configs/sales_eval_cases.yaml"


def _fmt_ratio(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "n/a"


# ── 用例加载 ──


def load_cases(path: str = CASES_DEFAULT, limit: int = 0) -> list[dict[str, Any]]:
    """加载销售评测用例（YAML）；文件缺失/解析失败返回 []。"""
    import yaml as _yaml

    p = Path(path)
    if not p.exists():
        return []
    data = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cases = list(data.get("cases") or [])
    if limit > 0:
        cases = cases[:limit]
    return cases


# ── 离线评测（信号/策略/意图） ──


def eval_signal_strategy(question: str, expected: dict[str, Any]) -> dict[str, Any]:
    """离线评测单例：信号识别 + 策略触发 + 意图路由。

    Args:
        question: 用户问题。
        expected: 期望 {intent?, signal:{mode, objection_type?}, strategy}。

    Returns:
        {signal_hit, strategy_hit, intent_hit, actual...} 明细。
    """
    from agent_base.agents.sales import build_sales_strategy, detect_sales_signal
    from agent_base.retrieval.intent_router import route_question

    exp_signal = expected.get("signal") or {}
    act_signal = detect_sales_signal(question)
    signal_hit = act_signal["mode"] == exp_signal.get("mode", "normal")
    if signal_hit and exp_signal.get("objection_type"):
        signal_hit = signal_hit and act_signal["objection_type"] == exp_signal["objection_type"]

    try:
        intent = route_question(question).intent
    except Exception:
        intent = ""
    intent_hit = intent == expected.get("intent") if expected.get("intent") else None

    strategy = build_sales_strategy(question, intent)
    strategy_hit = bool(strategy) == bool(expected.get("strategy"))

    return {
        "signal_hit": bool(signal_hit),
        "signal_actual": act_signal,
        "strategy_hit": bool(strategy_hit),
        "strategy_triggered": bool(strategy),
        "intent_hit": intent_hit,
        "intent_actual": intent,
    }


def run_offline(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """跑全部用例的离线评测，聚合得分与失败归因。"""
    details: list[dict[str, Any]] = []
    for case in cases:
        row = eval_signal_strategy(case.get("question", ""), case.get("expected") or {})
        row["id"] = case.get("id", "")
        row["scenario"] = case.get("scenario", "")
        row["question"] = case.get("question", "")
        row["dimension"] = str(case.get("dimension") or "销售")
        # 归因：哪个环节挂了
        reasons = []
        if not row["signal_hit"]:
            reasons.append("信号识别")
        if not row["strategy_hit"]:
            reasons.append("策略触发")
        if row["intent_hit"] is False:
            reasons.append("意图路由")
        row["fail_reasons"] = reasons
        details.append(row)

    n = max(1, len(details))
    agg = {
        "signal_acc": sum(1 for d in details if d["signal_hit"]) / n,
        "strategy_acc": sum(1 for d in details if d["strategy_hit"]) / n,
        "intent_acc": (
            sum(1 for d in details if d["intent_hit"] is True)
            / max(1, sum(1 for d in details if d["intent_hit"] is not None))
        ),
        "overall": (
            sum(
                1
                for d in details
                if d["signal_hit"] and d["strategy_hit"] and d["intent_hit"] is not False
            )
            / n
        ),
    }
    return {"mode": "offline", "total_cases": len(details), **agg, "details": details}


# ── 全链路（--judge）：真实回答 + LLM 判官 + 合规 ──


def _compliance_scan(answer: str) -> list[str]:
    """规则扫描回答中的合规违规（绝对化/医疗用语/贬低竞品）。

    Args:
        answer: AI 回答文本。

    Returns:
        违规描述列表；无违规返回 []。
    """
    hits: list[str] = []
    if not answer:
        return hits
    import re as _re

    for pat in _ABSOLUTE_RE:
        m = _re.search(pat, answer)
        if m:
            hits.append(f"绝对化用语「{m.group(0)}」")
    for w in _MEDICAL_WORDS:
        if w in answer:
            hits.append(f"医疗用语「{w}」")
    for p in _DISPARAGE_PATTERNS:
        if p in answer:
            hits.append(f"贬低竞品「{p}」")
    return hits


def _judge_answer(question: str, answer: str, guide: str) -> dict[str, Any]:
    """LLM 判官：按 judge_guide 给话术质量打分 1-5 并简述理由。

    Args:
        question: 用户问题。
        answer: AI 回答。
        guide: 期望行为说明（来自用例 judge_guide）。

    Returns:
        {score, reason}；LLM 不可用返回 {score: None, reason: "判官不可用"}。
    """
    try:
        from agent_base.llms import build_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        model = build_chat_model(provider="langchain", model="deepseek-v4-flash",
                                 api_key_env="ANTHROPIC_AUTH_TOKEN", temperature=0.0, timeout=30)
        if model is None:
            return {"score": None, "reason": "判官不可用（LLM 未配置）"}
        sys_prompt = (
            "你是电商销售话术评审专家。根据期望行为，给 AI 客服的回答打分 1-5 分"
            "（5=专家级销售话术；3=合格但平淡；1=答非所问或话术错误）。"
            "只输出 JSON：{“score”: 1-5, “reason”: “一句话理由”}"
        )
        user_prompt = (
            f"问题：{question}"
            + "\n\n期望行为："
            + f"{guide}\n\nAI 回答：{answer[:1500]}"
        )
        resp = model.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        import json as _json
        import re as _re

        text = str(getattr(resp, "content", resp))
        m = _re.search(r"{.*}", text, _re.S)
        obj = _json.loads(m.group(0)) if m else {}
        score = int(obj.get("score") or 0)
        return {"score": max(1, min(5, score)), "reason": str(obj.get("reason") or "")[:200]}
    except Exception:
        return {"score": None, "reason": "判官调用失败"}


def run_judge(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """全链路评测：真实回答 → 合规扫描 → LLM 判官打分。"""
    from agent_base.api.main import get_runtime
    from agent_base.chains import answer_question_with_trace
    from agent_base.retrieval.retrieval_config import RetrievalConfig

    runtime = get_runtime()
    vector_store = runtime["vector_store"]
    cfg = RetrievalConfig.from_runtime(runtime)
    cfg.llm.use_llm = True

    details: list[dict[str, Any]] = []
    for case in cases:
        question = case.get("question", "")
        row = eval_signal_strategy(question, case.get("expected") or {})
        row["id"] = case.get("id", "")
        row["scenario"] = case.get("scenario", "")
        row["question"] = question
        row["dimension"] = str(case.get("dimension") or "销售")
        try:
            res = answer_question_with_trace(
                question, vector_store, cfg,
                summary_store=runtime.get("summary_store"),
                sparse_store=runtime.get("sparse_store"),
            )
            answer = str(res.answer or "")
        except Exception as exc:
            answer = ""
            row["answer_error"] = str(exc)[:150]
        row["answer"] = answer[:800]
        row["compliance_violations"] = _compliance_scan(answer)
        row["compliance_pass"] = not row["compliance_violations"]
        judged = _judge_answer(question, answer, str(case.get("judge_guide") or ""))
        row["judge_score"] = judged["score"]
        row["judge_reason"] = judged["reason"]
        reasons = list(row.get("fail_reasons", []))
        if row.get("compliance_violations"):
            reasons.append("合规")
        if row.get("judge_score") is not None and row["judge_score"] < 3:
            reasons.append("话术质量")
        row["fail_reasons"] = reasons
        details.append(row)

    n = max(1, len(details))
    judged = [d for d in details if d.get("judge_score") is not None]
    agg = {
        "mode": "judge",
        "total_cases": len(details),
        "signal_acc": sum(1 for d in details if d["signal_hit"]) / n,
        "strategy_acc": sum(1 for d in details if d["strategy_hit"]) / n,
        "intent_acc": (
            sum(1 for d in details if d["intent_hit"] is True)
            / max(1, sum(1 for d in details if d["intent_hit"] is not None))
        ),
        "compliance_acc": sum(1 for d in details if d["compliance_pass"]) / n,
        "avg_judge_score": (sum(d["judge_score"] for d in judged) / len(judged)) if judged else None,
        "overall": (
            sum(
                1
                for d in details
                if d["signal_hit"] and d["strategy_hit"] and d["intent_hit"] is not False and d["compliance_pass"]
            )
            / n
        ),
        "details": details,
    }
    return agg


# ── 报告 ──


def build_report(result: dict[str, Any]) -> str:
    """构建 Markdown 报告（含失败归因与优化建议）。"""
    lines: list[str] = []
    lines.append("# 销售评测报告")
    lines.append("")
    lines.append(f"- 模式：{'全链路（真实回答 + LLM 判官）' if result.get('mode') == 'judge' else '离线（信号/策略/意图）'}")
    lines.append(f"- 用例数：{result.get('total_cases', 0)}")
    lines.append(f"- 综合得分：{_fmt_ratio(result.get('overall', 0))}")
    lines.append(f"- 信号识别：{_fmt_ratio(result.get('signal_acc', 0))}")
    lines.append(f"- 策略触发：{_fmt_ratio(result.get('strategy_acc', 0))}")
    lines.append(f"- 意图路由：{_fmt_ratio(result.get('intent_acc', 0))}")
    if result.get("mode") == "judge":
        lines.append(f"- 合规通过：{_fmt_ratio(result.get('compliance_acc', 0))}")
        avg = result.get("avg_judge_score")
        lines.append(f"- 话术质量均分：{'n/a' if avg is None else f'{avg:.1f} / 5'}")
    lines.append("")

    # 维度汇总
    dims: dict[str, dict[str, float]] = {}
    for d in result.get("details", []):
        dim = d.get("dimension", "销售")
        entry = dims.setdefault(dim, {"total": 0, "pass": 0, "judge_sum": 0, "judge_n": 0})
        entry["total"] += 1
        ok = d.get("signal_hit") and d.get("strategy_hit") and d.get("intent_hit") is not False
        if d.get("compliance_violations"):
            ok = False
        if ok:
            entry["pass"] += 1
        if d.get("judge_score") is not None:
            entry["judge_sum"] += d["judge_score"]
            entry["judge_n"] += 1
    lines.append("## 维度汇总")
    lines.append("")
    lines.append("| 维度 | 用例 | 通过 | 通过率 | 话术均分 |")
    lines.append("|---|---|---|---|---|")
    for dim, e in dims.items():
        javg = f"{e['judge_sum'] / e['judge_n']:.1f}" if e["judge_n"] else "—"
        lines.append(f"| {dim} | {e['total']} | {e['pass']} | {_fmt_ratio(e['pass'] / max(1, e['total']))} | {javg} |")
    lines.append("")

    # 失败归因
    fails = [d for d in result.get("details", []) if d.get("fail_reasons")]
    lines.append(f"## 失败归因（{len(fails)} 例）")
    lines.append("")
    reason_counter: dict[str, int] = {}
    for d in fails:
        for r in d.get("fail_reasons", []):
            reason_counter[r] = reason_counter.get(r, 0) + 1
    for r, c in sorted(reason_counter.items(), key=lambda kv: -kv[1]):
        lines.append(f"- **{r}**：{c} 例")
    if not fails:
        lines.append("- 无失败用例 🎉")
    lines.append("")

    # 明细
    lines.append("## 用例明细")
    lines.append("")
    lines.append("| ID | 场景 | 信号 | 策略 | 意图 | 合规 | 话术 | 归因 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for d in result.get("details", []):
        sig = "✅" if d["signal_hit"] else "❌"
        stg = "✅" if d["strategy_hit"] else "❌"
        itn = "✅" if d.get("intent_hit") is True else ("⬜" if d.get("intent_hit") is None else "❌")
        comp = "✅" if d.get("compliance_pass") else ("❌" if d.get("compliance_violations") else "—")
        js = d.get("judge_score")
        judge = "—" if js is None else f"{js}/5"
        reasons = "、".join(d.get("fail_reasons", [])) or "—"
        lines.append(f"| {d.get('id', '')} | {d.get('scenario', '')} | {sig} | {stg} | {itn} | {comp} | {judge} | {reasons} |")
    lines.append("")

    # 优化建议（按归因自动生成）
    lines.append("## 优化建议")
    lines.append("")
    hints: dict[str, str] = {
        "信号识别": "检查 agent_base/agents/sales.py 的 _BUYING/_PRICE/_HESITANT/_RISK 模式词："
                    "漏检→补词；误检→去掉过宽的子串（如单字）。",
        "策略触发": "检查 sales.py build_sales_strategy 的 SALES_INTENTS 与信号联动；"
                    "漏触发→确认意图已命中商品类；误触发→确认普通问题不在 SALES_INTENTS。",
        "意图路由": "检查 configs/domain/ecommerce.yaml 意图关键词与优先级；"
                    "该问题应归到哪个商品意图，就补对应关键词。",
        "话术质量": "优化 prompts_ecommerce.yaml 的 sales.strategy（价值呈现/异议处理/促单/连带）；"
                    "按 judge_reason 逐条改进。",
        "合规": "检查 qa.system 与 sales.strategy 的红线约束是否被绕开；"
                "违规回答需加强提示词或加后处理过滤。",
    }
    seen: set[str] = set()
    for d in fails:
        for r in d.get("fail_reasons", []):
            if r in hints and r not in seen:
                seen.add(r)
                lines.append(f"- **{r}**：{hints[r]}")
    if not seen:
        lines.append("- 无失败项，可尝试加大难度（新增异议/连带/合规边界用例）。")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：加载用例 → 评测 → 输出终端摘要与报告文件。"""
    parser = argparse.ArgumentParser(description="销售评测（信号/策略/意图 + 可选话术判官与合规）")
    parser.add_argument("--cases", default=CASES_DEFAULT, help="用例 YAML 路径")
    parser.add_argument("--limit", type=int, default=0, help="最多执行用例数（0=不限）")
    parser.add_argument("--judge", action="store_true", help="全链路模式：真实回答 + LLM 判官 + 合规扫描（需后端运行时与 LLM 密钥）")
    parser.add_argument("--output", default="", help="Markdown 报告路径（默认 reports/sales_eval_<时间戳>.md）")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases, args.limit)
    if not cases:
        print(f"未加载到用例：{args.cases}", file=sys.stderr)
        return 1
    print(f"销售评测开始：{len(cases)} 个用例（{'全链路' if args.judge else '离线'}）", flush=True)
    result = run_judge(cases) if args.judge else run_offline(cases)

    print(f"综合：{_fmt_ratio(result.get('overall', 0))}")
    print(f"  信号识别 {_fmt_ratio(result.get('signal_acc', 0))} / "
          f"策略触发 {_fmt_ratio(result.get('strategy_acc', 0))} / "
          f"意图路由 {_fmt_ratio(result.get('intent_acc', 0))}")
    if args.judge:
        avg = result.get("avg_judge_score")
        print(f"  合规 {_fmt_ratio(result.get('compliance_acc', 0))} / 话术均分 "
              f"{'n/a' if avg is None else f'{avg:.1f}/5'}")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = Path(args.output) if args.output else Path("reports") / f"sales_eval_{ts}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_report(result), encoding="utf-8")
    json_path = md_path.with_suffix(".json")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"报告已生成：{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
