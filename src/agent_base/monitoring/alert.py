"""轻量监控告警：基于 PG log_events 冷层统计 ERROR 告警 + 群机器人推送。

设计：
- 复用日志冷层（log_events）统计窗口内 ERROR/WARNING，零额外依赖；
- 配置 alerting.webhook_url（钉钉/企微自定义机器人）后，超阈值自动推送
  markdown 告警，带去重（Redis 键 TTL），避免告警风暴刷屏；
- 供 /api/admin/alert 查询 + 后台 _alert_loop 定时巡检（main.py lifespan）。
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any


def error_stats(minutes: int = 5) -> dict[str, Any]:
    """统计最近 N 分钟 ERROR/WARNING 日志数（按模块聚合）。

    Args:
        minutes: 统计窗口（分钟）。

    Returns:
        {total, error, warning, by_module: {module: count}}。
    """
    try:
        from agent_base.storage.pg import _conn

        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT level, module, COUNT(*) FROM log_events "
                "WHERE ts >= %s GROUP BY level, module",
                (since,),
            )
            rows = cur.fetchall()
        total = sum(r[2] for r in rows)
        by_level: dict[str, int] = {}
        by_module: dict[str, int] = {}
        for level, module, cnt in rows:
            by_level[level] = by_level.get(level, 0) + cnt
            by_module[str(module)] = by_module.get(str(module), 0) + cnt
        return {
            "window_minutes": minutes,
            "total": total,
            "error": by_level.get("ERROR", 0),
            "warning": by_level.get("WARNING", 0),
            "by_module": by_module,
        }
    except Exception:
        return {"window_minutes": minutes, "total": 0, "error": 0, "warning": 0, "by_module": {}}


def check_alert(error_threshold: int = 5, minutes: int = 5) -> dict[str, Any]:
    """告警检查：最近窗口 ERROR 数超阈值返回 alert 状态。

    Args:
        error_threshold: ERROR 告警阈值（默认 5）。
        minutes: 统计窗口（分钟）。

    Returns:
        含 level（ok/warning/alert）与统计详情的字典。
    """
    stats = error_stats(minutes)
    if stats["error"] >= error_threshold:
        return {"level": "alert", "message": f"最近 {minutes} 分钟 ERROR ≥ {error_threshold}", **stats}
    if stats["error"] > 0:
        return {"level": "warning", "message": f"最近 {minutes} 分钟有 ERROR 日志", **stats}
    return {"level": "ok", "message": "无 ERROR 日志", **stats}


def alert_config() -> dict[str, Any]:
    """告警配置：环境变量优先，回退 configs/app.yaml alerting 段。"""
    from agent_base.config import deep_get, load_yaml

    cfg = load_yaml("configs/app.yaml") or {}
    a = deep_get(cfg, "alerting", {}) or {}
    return {
        "enabled": bool(a.get("enabled", False)),
        "webhook_url": str(os.getenv("ALERT_WEBHOOK_URL", a.get("webhook_url", ""))).strip(),
        "style": str(os.getenv("ALERT_WEBHOOK_STYLE", a.get("webhook_style", "dingtalk"))).strip(),
        "error_threshold": int(a.get("error_threshold", 5)),
        "window_minutes": int(a.get("window_minutes", 5)),
        "dedup_minutes": int(a.get("dedup_minutes", 30)),
    }


def send_webhook(text: str, title: str = "AI 客服告警", style: str = "dingtalk", url: str = "") -> bool:
    """推送到群机器人（钉钉/企微 markdown）；失败返回 False 不抛异常。

    Args:
        text: markdown 正文。
        title: 告警标题。
        style: dingtalk | wecom（机器人消息格式差异）。
        url: 机器人 webhook 地址。

    Returns:
        是否推送成功。
    """
    if not url:
        return False
    try:
        if style == "wecom":
            payload = {"msgtype": "markdown", "markdown": {"content": f"### {title}\n{text}"}}
        else:  # 钉钉
            payload = {"msgtype": "markdown", "markdown": {"title": title, "text": f"### {title}\n{text}"}}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("errcode") in (0, None)
    except Exception:
        return False


def maybe_notify(force: bool = False) -> dict[str, Any]:
    """定时巡检：超阈值自动推送 webhook（Redis 键去重，避免告警风暴）。

    Args:
        force: 强制推送（忽略阈值与去重，人工测试用）。

    Returns:
        检查结果 + notified（是否实际推送）+ dedup（是否被去重拦截）。
    """
    cfg = alert_config()
    result = check_alert(error_threshold=cfg["error_threshold"], minutes=cfg["window_minutes"])
    result["notified"] = False
    result["dedup"] = False
    if force:
        result["level"] = "alert"
    if not cfg.get("enabled") or not cfg.get("webhook_url"):
        return result
    if result["level"] not in ("alert", "warning") and not force:
        return result
    # 去重：Redis 键（level 维度）带 TTL；Redis 不可用时直接推送（宁重复不漏报）
    dedup_ok = True
    try:
        from agent_base.storage.cache import _get_client

        client = _get_client()
        if client is not None:
            key = f"alert:notify:{result['level']}"
            dedup_ok = bool(client.set(key, "1", nx=True, ex=int(cfg["dedup_minutes"]) * 60))
    except Exception:
        dedup_ok = True
    if not dedup_ok:
        result["dedup"] = True
        return result
    top = sorted(result.get("by_module", {}).items(), key=lambda kv: -kv[1])[:5]
    top_text = "\n".join(f"- {m}: {c}" for m, c in top) or "- （无）"
    text = (
        f"> 时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"> 状态：{result['message']}\n\n"
        f"**ERROR**：{result['error']} 条 / {result['window_minutes']} 分钟\n"
        f"**WARNING**：{result['warning']} 条\n"
        f"**模块 TOP5**：\n{top_text}"
    )
    result["notified"] = send_webhook(
        text,
        title="AI 客服告警：" + result["level"],
        style=cfg["style"],
        url=cfg["webhook_url"],
    )
    return result
