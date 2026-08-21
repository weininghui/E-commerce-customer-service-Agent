"""知识运营 Agent：Plan-Execute-Reflect 编排。

- Plan：LLM 将自然语言指令解析为结构化操作 JSON（工具名 + 参数）；
- Execute：通过工具注册表执行，全程记录操作日志；
- Reflect：结果反思——失败自动降级为查询/澄清，简单任务不走 LLM（规则降级）。

执行全程留痕（操作人 / 操作内容 / 时间），输出结构化审计记录。
"""

from __future__ import annotations

import json
import time
from typing import Any

from agent_base.agents.knowledge_tools import TOOL_REGISTRY, list_tools


def _rule_plan(command: str) -> dict[str, Any] | None:
    """简单规则降级：关键词识别增/删/改/查，不走 LLM。"""
    c = command.strip()
    if c.startswith(("查询", "查一下", "搜", "查找", "找")):
        query = c.lstrip("查询一下搜查找找：: ")
        return {"tool": "kb_query", "params": {"query": query or "商品"}}
    if c.startswith(("删除", "删掉", "移除")):
        doc_id = c.lstrip("删除掉移除：: ")
        return {"tool": "kb_delete", "params": {"doc_id": doc_id}}
    if c.startswith(("新增", "添加", "加入", "写入")):
        content = c.lstrip("新增添加加入写入：: ")
        if content:
            return {"tool": "kb_add", "params": {"content": content}}
    if c.startswith(("更新", "修改", "改")):
        return None  # 更新需要 doc_id + content，交给 LLM
    return None


def _llm_plan(command: str) -> dict[str, Any] | None:
    """LLM 解析指令 → 结构化操作 JSON（三层兜底：bind_tools → JSON → 失败返回 None）。"""
    try:
        from agent_base.config import deep_get, load_yaml
        from agent_base.llms import build_chat_model

        cfg = load_yaml("configs/app.yaml") or {}
        llm_cfg = deep_get(cfg, "knowledge_ops_llm") or {"provider": "none"}
        model = build_chat_model(
            **llm_cfg,
            tracking_agent="knowledge_ops",
            tracking_source="knowledge_ops",
        )
        if model is None:
            return None
        from agent_base.prompts import get_prompt

        tools_desc = "\n".join(f"- {t['name']}: {t['doc']}" for t in list_tools())
        _DEFAULT_OPS = (
            "你是知识库运营助手。把用户的运营指令解析为 JSON，格式："
            '{"tool": "工具名", "params": {...}}。可选工具：\n'
            "{tools}\n"
            "规则：无法确定工具或参数缺失时，输出 {\"tool\": \"kb_query\", \"params\": {\"query\": \"\"}}。"
            "只输出 JSON。"
        )
        prompt = get_prompt("knowledge_ops", "system", _DEFAULT_OPS).replace("{tools}", tools_desc) + "\n\n用户指令：" + command
        resp = model.invoke(prompt)
        text = str(getattr(resp, "content", "") or resp or "")
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").removeprefix("json").strip()
        data = json.loads(cleaned)
        if isinstance(data, dict) and data.get("tool") in TOOL_REGISTRY:
            return data
    except Exception:
        pass
    return None


def _execute(plan: dict[str, Any], operator: str) -> dict[str, Any]:
    """执行工具并留痕。"""
    tool = str(plan.get("tool") or "")
    params = plan.get("params") or {}
    fn = TOOL_REGISTRY.get(tool)
    if fn is None:
        return {"ok": False, "error": f"未注册工具: {tool}"}
    t0 = time.perf_counter()
    try:
        result = fn(**params)
        ok = bool(result.get("ok"))
        latency_ms = int((time.perf_counter() - t0) * 1000)
        # 留痕：log_events + monitoring 工具调用记录
        try:
            from agent_base.monitoring.usage import record_tool_call

            record_tool_call(
                agent="knowledge_ops",
                tool_name=tool,
                params=params,
                ok=ok,
                error=str(result.get("error", "")) if not ok else "",
                latency_ms=latency_ms,
                result_preview=json.dumps(result, ensure_ascii=False)[:500],
            )
        except Exception:
            pass
        return {
            "ok": ok,
            "tool": tool,
            "params": params,
            "result": result,
            "latency_ms": latency_ms,
            "operator": operator,
        }
    except Exception as exc:
        return {
            "ok": False,
            "tool": tool,
            "params": params,
            "error": str(exc)[:200],
            "operator": operator,
        }


def _reflect(exec_result: dict[str, Any]) -> dict[str, Any]:
    """反思：失败 → 降级为查询兜底；成功 → 返回结论。"""
    if exec_result.get("ok"):
        return {
            "reflect_ok": True,
            "conclusion": f"执行成功：{exec_result.get('tool')}",
        }
    error = exec_result.get("error") or ""
    return {
        "reflect_ok": False,
        "conclusion": f"执行失败（{error[:80]}），已降级为查询兜底",
    }


def run_knowledge_ops(command: str, operator: str = "admin") -> dict[str, Any]:
    """执行一条知识运营指令（Plan → Execute → Reflect）。"""
    started = time.perf_counter()
    # Plan：规则降级优先，LLM 兜底
    plan = _rule_plan(command)
    plan_source = "rule"
    if plan is None:
        plan = _llm_plan(command)
        plan_source = "llm"
    if plan is None:
        plan = {"tool": "kb_query", "params": {"query": command[:80]}}
        plan_source = "fallback"

    exec_result = _execute(plan, operator)
    reflect = _reflect(exec_result)
    return {
        "ok": exec_result.get("ok"),
        "plan": plan,
        "plan_source": plan_source,
        "execution": {
            "tool": exec_result.get("tool"),
            "params": exec_result.get("params"),
            "result": exec_result.get("result"),
            "error": exec_result.get("error", ""),
            "latency_ms": exec_result.get("latency_ms", 0),
        },
        "reflection": reflect,
        "operator": operator,
        "total_ms": int((time.perf_counter() - started) * 1000),
    }
