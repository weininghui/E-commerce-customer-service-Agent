"""电商客服「全维度长对话」模拟器。

按场景脚本逐轮调用 /api/ask/stream（同一 session_id 模拟多轮对话），
解析 SSE 事件并逐轮断言，输出统一评测报告（含 token 用量与估算成本）。

用法：
    python scripts/longchat/eval_long_conversation.py                 # 跑全部场景
    python scripts/longchat/eval_long_conversation.py --scenario s2   # 只跑 s2
    python scripts/longchat/eval_long_conversation.py --max-turns 20  # 每场景最多 20 轮（冒烟）
    python scripts/longchat/eval_long_conversation.py --base-url http://116.62.69.48:8000 --token <admin-token>
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
DEFAULT_OUT = ROOT / "reports" / "longchat"

# DeepSeek 估算单价（元 / 1M tokens，输入按未命中缓存计）
COST_PER_1M_IN = 2.0
COST_PER_1M_OUT = 8.0

# 每轮请求超时（秒）：真实模型长生成 + 检索，给足余量
TURN_TIMEOUT_S = 120

# 主动追问标记：画像已提供后不应再出现（no_repeat_ask 断言）
REPEAT_ASK_MARKERS = (
    "什么肤质", "肤质是", "偏油还是偏干", "油皮还是干皮", "什么类型",
    "预算多少", "多少预算", "什么尺码", "身高体重", "平时用什么",
)

# 合规红线词（s8 断言）
BANNED_WORDS = ("绝对", "最佳", "第一", "100%", "根治", "永不复发", "特效", "治疗", "治愈")
NEGATION_MARKERS = ("不能", "没有", "不敢", "不是", "别当", "别把", "不承诺", "拒绝", "做不到", "无法", "没法", "不保证")

INTERNAL_LABELS = ("依据商品/FAQ资料", "安全等级", "风险标签", "购买/使用建议", "来源：", "结论：")


def load_scenarios(max_turns: int | None = None) -> list[dict[str, Any]]:
    """加载全部场景 YAML；支持 --max-turns 截断每场景轮数。"""
    import yaml

    scenarios: list[dict[str, Any]] = []
    for path in sorted(SCENARIOS_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not data.get("turns"):
            continue
        if max_turns:
            data["turns"] = data["turns"][:max_turns]
        scenarios.append(data)
    return scenarios


def parse_sse(line: str) -> dict[str, Any] | None:
    """解析一行 SSE（data: {json}）；非 data 行返回 None。"""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def stream_turn(
    base_url: str,
    question: str,
    session_id: str,
    token: str | None = None,
) -> dict[str, Any]:
    """调用一次流式问答，返回结构化事件摘要。"""
    import httpx

    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Admin-Token"] = token
    body = {
        "question": question,
        "top_k": 6,
        "rerank": "none",
        "use_cache": False,
        "session_id": session_id,
    }
    summary: dict[str, Any] = {
        "events": [],
        "sources_count": 0,
        "media": [],
        "trace": {},
        "memory": {},
        "answer": "",
        "status": 0,
        "error": "",
    }
    try:
        with httpx.Client(timeout=TURN_TIMEOUT_S) as client:
            with client.stream("POST", f"{base_url}/api/ask/stream", json=body, headers=headers) as resp:
                summary["status"] = resp.status_code
                if resp.status_code != 200:
                    summary["error"] = resp.read().decode("utf-8", "ignore")[:300]
                    return summary
                for line in resp.iter_lines():
                    event = parse_sse(line)
                    if event is None:
                        continue
                    etype = event.get("type", "")
                    summary["events"].append(etype)
                    if etype == "sources":
                        summary["sources_count"] = len(event.get("sources") or [])
                    elif etype == "media":
                        summary["media"] = list(event.get("media") or [])
                    elif etype == "trace":
                        summary["trace"] = event.get("trace") or {}
                    elif etype == "memory":
                        summary["memory"] = event.get("memory") or {}
                    elif etype == "done":
                        summary["answer"] = str(event.get("answer") or "")
                        if event.get("error"):
                            summary["error"] = str(event["error"])[:200]
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return summary


def _has(event_types: list[str], name: str) -> bool:
    return name in event_types


def _banned_hits(answer: str) -> list[str]:
    """违规词命中（排除“拒绝语境”下的引用，如“不能保证100%”“不能治疗”）。"""
    hits: list[str] = []
    for sentence in (answer or "").replace("\n", "。").split("。"):
        for word in BANNED_WORDS:
            if word == "第一" and "第一次" in sentence:
                continue
            if f'"{word}' in sentence or f"“{word}" in sentence:
                continue  # 引号内引用（如“绝对不过敏”）是拒绝语境，不是承诺
            if word in sentence and not any(m in sentence for m in NEGATION_MARKERS):
                hits.append(word)
    return sorted(set(hits))


def run_asserts(question: str, summary: dict[str, Any], asserts: dict[str, Any]) -> list[str]:
    """逐条执行断言，返回失败原因列表（空 = 全过）。"""
    failures: list[str] = []
    events = summary["events"]
    trace = summary["trace"] or {}
    route = trace.get("route") or {}
    sales = trace.get("sales") or {}
    answer = summary["answer"] or ""

    if asserts.get("answer_nonempty", True) and not answer.strip():
        failures.append("answer 为空")
    if asserts.get("event_complete", True):
        if not _has(events, "sources") and not asserts.get("handoff"):
            failures.append("缺少 sources 事件")
        if not _has(events, "done"):
            failures.append("缺少 done 事件")
    if asserts.get("media") is True and not summary["media"]:
        failures.append("期望 media 事件，实际无媒体")
    if asserts.get("media") is False and summary["media"]:
        failures.append(f"期望无媒体，实际返回 {len(summary['media'])} 条")
    if asserts.get("intent") and route.get("intent") != asserts["intent"]:
        failures.append(f"意图期望 {asserts['intent']}，实际 {route.get('intent')}")
    if asserts.get("stage") and sales.get("stage") != asserts["stage"]:
        failures.append(f"阶段期望 {asserts['stage']}，实际 {sales.get('stage')}")
    if asserts.get("action") and sales.get("action") != asserts["action"]:
        failures.append(f"动作期望 {asserts['action']}，实际 {sales.get('action')}")
    if asserts.get("no_repeat_ask") and any(m in answer for m in REPEAT_ASK_MARKERS):
        failures.append(f"仍在重复追问需求信息（回答含 {[m for m in REPEAT_ASK_MARKERS if m in answer]}）")
    if asserts.get("no_clarify"):
        decision = trace.get("decision") or {}
        if decision.get("need_clarification"):
            failures.append("触发了澄清追问（期望已锚定直接回答）")
    if asserts.get("anchor"):
        anchor = asserts["anchor"]
        hit = anchor in answer
        if not hit:
            for src in (trace.get("sources") or trace.get("results") or []):
                if isinstance(src, dict) and anchor in str(src.get("doc_name") or ""):
                    hit = True
                    break
        if not hit:
            failures.append(f"未锚定商品「{anchor}」")
    if asserts.get("no_banned"):
        banned_hits = _banned_hits(answer)
        if banned_hits:
            failures.append(f"回答含违规词 {banned_hits}")
    if asserts.get("no_internal_labels") and any(w in answer for w in INTERNAL_LABELS):
        failures.append(f"回答泄露内部字段 {[w for w in INTERNAL_LABELS if w in answer]}")
    if asserts.get("handoff"):
        if not any(k in answer for k in ("人工", "转接", "稍候")):
            failures.append("期望转人工回复，实际未命中")
    if summary["status"] not in (0, 200):
        failures.append(f"HTTP {summary['status']}: {summary['error'][:100]}")
    if summary.get("error") and summary["status"] != 0:
        failures.append(summary["error"][:150])
    return failures


def est_tokens(summary: dict[str, Any]) -> tuple[int, int]:
    """估算单轮输入/输出 token（输入取 memory 事件 est_tokens，输出按回答字数估算）。"""
    memory = summary.get("memory") or {}
    in_tokens = int(memory.get("context_est_tokens") or 0)
    out_tokens = int(len(summary.get("answer") or "") / 2)
    return in_tokens, out_tokens


def run_scenario(
    scenario: dict[str, Any],
    base_url: str,
    token: str | None,
    delay: float = 0.0,
) -> dict[str, Any]:
    """跑一条场景，返回逐轮结果与汇总。"""
    sid = f"longchat-{scenario['id']}-{uuid.uuid4().hex[:8]}"
    turns: list[dict[str, Any]] = []
    total_in = total_out = 0
    passed = failed = infra_failed = 0
    for idx, turn in enumerate(scenario["turns"], start=1):
        question = turn["q"]
        asserts = turn.get("assert") or {}
        started = time.time()
        summary = stream_turn(base_url, question, sid, token=token)
        # 限流 429：退避重试（最多 3 次）
        for _ in range(2):
            if summary["status"] != 429:
                break
            time.sleep(3.0)
            summary = stream_turn(base_url, question, sid, token=token)
        # LLM 偶发空回答：状态码正常但无答案时重试一次
        if not summary.get("answer") and summary.get("status") == 200:
            summary = stream_turn(base_url, question, sid, token=token)
        duration = round(time.time() - started, 1)
        failures = run_asserts(question, summary, asserts)
        infra = any(
            "402" in f
            or "Insufficient Balance" in f
            or "timed out" in f.lower()
            or f.startswith("HTTP 5")
            for f in failures
        )
        in_tok, out_tok = est_tokens(summary)
        total_in += in_tok
        total_out += out_tok
        ok = not failures
        passed += ok
        failed += (not ok)
        infra_failed += (infra and not ok)
        turns.append(
            {
                "turn": idx,
                "q": question,
                "ok": ok,
                "failures": failures,
                "duration_s": duration,
                "status": summary["status"],
                "events": summary["events"],
                "media_count": len(summary["media"]),
                "stage": (summary["trace"] or {}).get("sales", {}).get("stage"),
                "action": (summary["trace"] or {}).get("sales", {}).get("action"),
                "context_tier": (summary["memory"] or {}).get("context_tier"),
                "infra": infra,
                "est_in_tokens": in_tok,
                "est_out_tokens": out_tok,
                "answer_snippet": (summary["answer"] or "")[:120],
            }
        )
        if delay > 0 and idx < len(scenario["turns"]):
            time.sleep(delay)
    cost = (total_in / 1_000_000 * COST_PER_1M_IN) + (total_out / 1_000_000 * COST_PER_1M_OUT)
    return {
        "id": scenario["id"],
        "name": scenario.get("name", scenario["id"]),
        "session_id": sid,
        "total_turns": len(turns),
        "passed": passed,
        "failed": failed,
        "infra_failed": infra_failed,
        "assertion_failed": failed - infra_failed,
        "pass_rate": round(100.0 * passed / len(turns), 1) if turns else 0.0,
        "effective_rate": (
            round(100.0 * passed / max(1, len(turns) - infra_failed), 1)
            if turns
            else 0.0
        ),
        "total_in_tokens": total_in,
        "total_out_tokens": total_out,
        "est_cost_rmb": round(cost, 3),
        "turns": turns,
    }


def build_report(results: list[dict[str, Any]]) -> str:
    """生成 Markdown 汇总报告。"""
    lines = [
        "# 全维度长对话测试报告",
        "",
        f"- 场景数：{len(results)}",
        f"- 总计轮数：{sum(r['total_turns'] for r in results)}，通过 {sum(r['passed'] for r in results)}，"
        f"外部失败（余额/超时等）{sum(r['infra_failed'] for r in results)}，"
        f"断言失败 {sum(r['assertion_failed'] for r in results)}",
        f"- 估算成本：¥{sum(r['est_cost_rmb'] for r in results):.3f} "
        f"（输入 {sum(r['total_in_tokens'] for r in results)} tokens / "
        f"输出 {sum(r['total_out_tokens'] for r in results)} tokens）",
        "",
        "## 场景汇总",
        "",
        "| 场景 | 轮数 | 通过率 | 外部失败 | 成本(¥) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} {r['name']} | {r['total_turns']} | {r['pass_rate']}% "
            f"（有效 {r['effective_rate']}%）| {r['infra_failed']} | {r['est_cost_rmb']:.3f} |"
        )
    lines.append("")
    for r in results:
        fails = [t for t in r["turns"] if not t["ok"]]
        assert_fails = [t for t in fails if not t.get("infra")]
        lines.append(
            f"## {r['id']} {r['name']}（失败 {len(fails)} 轮，其中断言失败 {len(assert_fails)}）"
        )
        lines.append("")
        if not assert_fails and fails:
            lines.append("- 失败均为外部原因（余额不足/超时等）✓")
        elif not fails:
            lines.append("- 无失败 ✓")
        for t in assert_fails[:20]:
            lines.append(f"- 第{t['turn']}轮：{t['q']} → {t['failures']} | {t['answer_snippet']}")
        for t in fails[:5]:
            if t.get("infra"):
                lines.append(f"- 第{t['turn']}轮（外部失败）：{t['failures'][0][:80]}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """模拟器入口：跑场景、写逐场景 JSON 与汇总报告。"""
    parser = argparse.ArgumentParser(description="电商客服全维度长对话模拟器")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API 地址")
    parser.add_argument("--scenario", default="", help="只跑指定场景 id（如 s2）")
    parser.add_argument("--max-turns", type=int, default=0, help="每场景最多轮数（冒烟用）")
    parser.add_argument("--token", default="", help="管理端 token（服务器部署需要）")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="报告输出目录")
    parser.add_argument("--delay", type=float, default=1.2, help="轮间间隔秒数（默认 1.2s，规避 60 次/分钟限流）")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="不跑场景，仅聚合 out-dir 下已有的 <id>.json 生成汇总报告",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.aggregate:
        results = []
        for path in sorted(out_dir.glob("s*.json")):
            if path.name == "eval_report.json":
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if "turns" in data:
                # 兼容旧结果：按失败文本回补外部失败（余额/超时）分类
                if "infra_failed" not in data:
                    infra_count = 0
                    for turn in data["turns"]:
                        infra = any(
                            "402" in f
                            or "Insufficient Balance" in f
                            or "timed out" in f.lower()
                            or f.startswith("HTTP 5")
                            for f in turn.get("failures", [])
                        )
                        turn["infra"] = infra
                        infra_count += infra and not turn.get("ok")
                    data["infra_failed"] = infra_count
                    data["assertion_failed"] = data.get("failed", 0) - infra_count
                    total = len(data["turns"])
                    data["effective_rate"] = (
                        round(100.0 * data.get("passed", 0) / max(1, total - infra_count), 1)
                        if total
                        else 0.0
                    )
                results.append(data)
        if not results:
            raise SystemExit("没有可聚合的场景 JSON")
        (out_dir / "eval_report.md").write_text(build_report(results), encoding="utf-8")
        (out_dir / "eval_report.json").write_text(
            json.dumps({"results": [{k: v for k, v in r.items() if k != "turns"} for r in results]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"聚合报告已输出：{out_dir / 'eval_report.md'}（{len(results)} 个场景）")
        return

    scenarios = load_scenarios(args.max_turns or None)
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
    if not scenarios:
        raise SystemExit("没有可执行的场景")

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        print(f"[{scenario['id']}] {scenario.get('name')} 开始，{len(scenario['turns'])} 轮")
        result = run_scenario(scenario, args.base_url, args.token or None, delay=args.delay)
        results.append(result)
        (out_dir / f"{result['id']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[{result['id']}] 完成：{result['passed']}/{result['total_turns']} 通过 "
            f"({result['pass_rate']}%)，成本 ¥{result['est_cost_rmb']:.3f}"
        )
        if args.delay > 0:
            time.sleep(args.delay)

    (out_dir / "eval_report.md").write_text(build_report(results), encoding="utf-8")
    (out_dir / "eval_report.json").write_text(
        json.dumps(
            {
                "results": [
                    {k: v for k, v in r.items() if k != "turns"} for r in results
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"报告已输出：{out_dir / 'eval_report.md'}")


if __name__ == "__main__":
    main()
