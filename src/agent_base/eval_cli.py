"""一键全链路评测命令行：python -m agent_base.eval_cli。

用法示例：
    python -m agent_base.eval_cli                                   # AI 动态生成用例跑全量评测
    python -m agent_base.eval_cli --cases cases.json --name "回归"   # 使用自定义用例集
    python -m agent_base.eval_cli --output reports/eval.md --no-json

输出：终端摘要 + Markdown/JSON 报告（默认 reports/eval_report_<时间戳>.md/.json）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_cases(path: str | None, limit: int = 0) -> list[dict[str, Any]] | None:
    """读取自定义用例 JSON（列表），可选截断条数。"""
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"用例文件必须是 JSON 数组：{path}")
    cases = [
        {
            "question": str(c.get("question", "")).strip(),
            "expected_intent": str(c.get("expected_intent", "")),
            "expected_source": str(c.get("expected_source", "")),
            "expected_facts": [str(f) for f in (c.get("expected_facts") or [])],
        }
        for c in data
        if str(c.get("question", "")).strip()
    ]
    if limit > 0:
        cases = cases[:limit]
    return cases


def _failure_distribution(cases: list[dict[str, Any]]) -> dict[str, int]:
    """按失败归因统计分布。"""
    from agent_base.retrieval.eval_chain import classify_failure

    dist: dict[str, int] = {}
    for c in cases:
        ftype, _ = classify_failure(c)
        if ftype:
            dist[ftype] = dist.get(ftype, 0) + 1
    return dist


def _fmt_ratio(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "-"


def build_markdown_report(result: dict[str, Any]) -> str:
    """把评测结果渲染为 Markdown 报告。"""
    from agent_base.retrieval.eval_chain import classify_failure

    dims = [
        ("意图命中", result.get("intent_acc", 0)),
        ("召回命中", result.get("recall_acc", 0)),
        ("事实命中", result.get("fact_acc", 0)),
        ("合规", result.get("compliance_acc", 0)),
        ("忠实度", result.get("faithfulness_acc", 0)),
        ("相关性", result.get("relevancy_acc", 0)),
    ]
    lines = [
        f"# 全链路评测报告 · {result.get('name', '全链路评测')}",
        "",
        f"- 批次 ID：`{result.get('run_id', 0)}`",
        f"- 用例数：{result.get('total_cases', 0)}",
        f"- 综合得分：**{_fmt_ratio(result.get('overall', 0))}**",
        f"- 生成方式：{result.get('generation', '-')}",
        "",
        "## 各维度得分",
        "",
        "| 维度 | 得分 |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {_fmt_ratio(v)} |" for k, v in dims]

    ragas = result.get("ragas") or {}
    if ragas:
        lines += ["", "## RAGAS 指标（标准公开口径）", "", "| 指标 | 得分 |", "| --- | --- |"]
        for key, label in (
            ("faithfulness_acc", "忠实度"),
            ("relevancy_acc", "答案相关性"),
            ("context_precision_acc", "上下文精确率"),
            ("context_recall_acc", "上下文召回"),
        ):
            if key in ragas:
                lines += [f"| {label} | {_fmt_ratio(ragas.get(key, 0))} |"]

    dist = _failure_distribution(result.get("cases", []))
    lines += ["", "## 失败归因分布", ""]
    if dist:
        lines += ["| 归因 | 数量 |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in sorted(dist.items(), key=lambda x: -x[1])]
    else:
        lines += ["本次无失败用例。"]

    lines += ["", "## 用例明细", "", "| 问题 | 预期意图 | 实际意图 | 意图 | 召回 | 事实 | 归因 | 归因详情 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in result.get("cases", []):
        ftype, detail = classify_failure(c)
        lines.append(
            "| {q} | {ei} | {ai} | {ih} | {rh} | {fh}/{ft} | {ftype} | {detail} |".format(
                q=(c.get("question") or "")[:40].replace("|", "/"),
                ei=(c.get("expected_intent") or "")[:20],
                ai=(c.get("actual_intent") or "")[:20],
                ih="✓" if c.get("intent_hit") else "✗",
                rh="✓" if c.get("recall_hit") else "✗",
                fh=c.get("fact_hits", 0),
                ft=c.get("fact_total", 0),
                ftype=ftype or "-",
                detail=(detail or "-")[:80].replace("|", "/"),
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """执行一键评测：加载用例 → 跑全链路 → 输出终端摘要与报告文件。"""
    parser = argparse.ArgumentParser(description="一键全链路评测（意图/召回/事实/合规/忠实度/相关性）")
    parser.add_argument("--cases", help="自定义用例 JSON 文件路径（缺省则 AI 动态生成）")
    parser.add_argument("--name", default="", help="批次名称")
    parser.add_argument("--limit", type=int, default=0, help="最多执行用例数（0 为不限）")
    parser.add_argument("--output", default="", help="Markdown 报告输出路径（默认 reports/eval_report_<时间戳>.md）")
    parser.add_argument("--json-out", default="", help="JSON 报告输出路径（默认与 MD 同名的 .json）")
    parser.add_argument("--no-json", action="store_true", help="不输出 JSON 报告")
    parser.add_argument("--ragas", action="store_true", help="附加 RAGAS 四指标评测（成本更高：每例多次 LLM + embedding）")
    args = parser.parse_args(argv)

    cases = _load_cases(args.cases, args.limit)
    source = args.cases or "AI 动态生成"

    def _progress(done: int, total: int, phase: str) -> None:
        print(f"[{done}/{total}] {phase}...", file=sys.stderr)

    from agent_base.retrieval.eval_chain import run_chain_eval

    print(f"评测开始：{source}", flush=True)
    result = run_chain_eval(custom_cases=cases, name=args.name, progress_cb=_progress, use_ragas=args.ragas)
    if result.get("cancelled"):
        print("已取消", file=sys.stderr)
        return 130

    total = int(result.get("total_cases", 0))
    print(f"完成：{total} 个用例，综合得分 {_fmt_ratio(result.get('overall', 0))}")
    for key, label in (
        ("intent_acc", "意图"),
        ("recall_acc", "召回"),
        ("fact_acc", "事实"),
        ("compliance_acc", "合规"),
        ("faithfulness_acc", "忠实度"),
        ("relevancy_acc", "相关性"),
    ):
        print(f"  {label}: {_fmt_ratio(result.get(key, 0))}")
    ragas = result.get("ragas") or {}
    if ragas:
        print("  [RAGAS]")
        for key, label in (
            ("faithfulness_acc", "RAGAS 忠实度"),
            ("relevancy_acc", "RAGAS 相关性"),
            ("context_precision_acc", "RAGAS 上下文精确率"),
            ("context_recall_acc", "RAGAS 上下文召回"),
        ):
            if key in ragas:
                print(f"  {label}: {_fmt_ratio(ragas.get(key, 0))}")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = Path(args.output) if args.output else Path("reports") / f"eval_report_{ts}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_markdown_report(result), encoding="utf-8")
    print(f"报告已生成：{md_path}")

    if not args.no_json:
        json_path = Path(args.json_out) if args.json_out else md_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"JSON：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
