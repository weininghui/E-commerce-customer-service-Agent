"""结构化 logger：JSON 行 → stdout + Redis 缓冲（app:logs List，LIFO 最近 1000 条）。

- ``get_logger(module)`` 返回带 request_id 的 LoggerAdapter
- ``log_to_redis(level, module, event, data)`` 写入 Redis（静默降级）
- 不新增第三方依赖；纯标准 logging + 已有 redis
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# ── ContextVar：request_id 跨异步链路传递 ──

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# ── Redis 配置 ──

_LOG_KEY = "app:logs"
_DEFAULT_MAX = 1000
_DEFAULT_TTL = 7 * 24 * 3600  # 7 天
_pg_counter = 0  # 惰性清理计数（每 100 次 ERROR/WARNING 清理一次冷层）


def _redis_client():
    """获取 Redis 客户端（复用现有连接工厂）。"""
    try:
        from agent_base.storage.chat_memory import _get_client

        return _get_client()
    except Exception:
        return None


def _log_cfg() -> dict[str, Any]:
    """读取 app.yaml logging 段（缺失回退默认）。"""
    try:
        from agent_base.config import load_yaml

        return (load_yaml("configs/app.yaml") or {}).get("logging") or {}
    except Exception:
        return {}


# ── JSON 格式化器 ──


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "request_id": request_id_var.get("-"),
            "module": record.name,
            "event": getattr(record, "event", ""),
            "data": getattr(record, "data", {}),
        }
        if record.exc_info and record.exc_info[1]:
            obj["exc"] = str(record.exc_info[1])
        return json.dumps(obj, ensure_ascii=False, default=str)


# ── Redis 处理器 ──


class _RedisHandler(logging.Handler):
    """将日志行 push 到 Redis List（LIFO，最近 N 条）。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            global _pg_counter
            cfg = _log_cfg()
            max_entries = int(cfg.get("redis_max_entries", _DEFAULT_MAX))
            ttl_days = int(cfg.get("redis_ttl_days", 7))
            client = _redis_client()
            if client is None:
                return
            entry = self.format(record)
            client.lpush(_LOG_KEY, entry)
            client.ltrim(_LOG_KEY, 0, max_entries - 1)
            client.expire(_LOG_KEY, ttl_days * 24 * 3600)

            # 冷层：ERROR/WARNING 落 PG log_events（审计级，90 天），失败静默
            if record.levelno >= logging.WARNING and cfg.get("pg_enabled", True):
                try:
                    from agent_base.storage.pg import insert_log_event, purge_expired_logs

                    data = dict(getattr(record, "data", {}) or {})
                    if record.exc_info and record.exc_info[1]:
                        # 生产口径：异常原因必须随事件落库，否则面板只有次数没有原因
                        data.setdefault(
                            "exc", f"{type(record.exc_info[1]).__name__}: {record.exc_info[1]}"
                        )
                    insert_log_event(
                        level=record.levelname,
                        module=record.name,
                        event=getattr(record, "event", "") or "",
                        request_id=request_id_var.get("-"),
                        data=data,
                    )
                    # 惰性清理：每 100 次写触发一次超期删除
                    _pg_counter += 1
                    if _pg_counter % 100 == 0:
                        purge_expired_logs(int(cfg.get("pg_retention_days", 90)))
                except Exception:
                    pass
        except Exception:
            pass  # 日志不阻塞业务


# ── 初始化 ──

def _setup_root_logger() -> None:
    """配置根 logger：JSON → stdout + Redis。"""
    root = logging.getLogger("agent_base")
    if root.handlers:  # 幂等
        return
    root.setLevel(logging.DEBUG)

# 标准输出
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(_JsonFormatter())
    root.addHandler(sh)

# Redis 通道
    rh = _RedisHandler()
    rh.setLevel(logging.INFO)
    rh.setFormatter(_JsonFormatter())
    root.addHandler(rh)

    # uvicorn/其他不传播
    root.propagate = False


_setup_root_logger()


# ── 公开 API ──


def get_logger(module: str) -> logging.LoggerAdapter:
    """获取带 request_id 注入的结构化 logger。

    Usage::

        from agent_base.monitoring.logger import get_logger
        logger = get_logger(__name__)
        logger.info("intent_decided", extra={"data": {"intent": "product_query"}})
    """
    base = logging.getLogger(f"agent_base.{module.lstrip('agent_base.')}")
    adapter = logging.LoggerAdapter(base, {"request_id": "-"})

    # 拦截：每次 emit 前注入当前 request_id + 自定义字段
    original_process = adapter.process

    def _process(msg, kwargs):
        kwargs = original_process(msg, kwargs) if callable(original_process) else kwargs
        kwargs["extra"] = {
            **(kwargs.get("extra") or {}),
            "request_id": request_id_var.get("-"),
        }
        return msg, kwargs

    adapter.process = _process
    return adapter


def log_event(level: str, module: str, event: str, data: dict[str, Any] | None = None) -> None:
    """直接写日志（无需先 get_logger）。

    Args:
        level: DEBUG / INFO / WARNING / ERROR
        module: 模块名（如 "graph_supervisor"）
        event: 事件名（如 "intent_decided"）
        data: 附加数据
    """
    logger = logging.getLogger(f"agent_base.{module}")
    extra = {
        "request_id": request_id_var.get("-"),
        "event": event,
        "data": data or {},
    }
    getattr(logger, level.lower())("", extra=extra)
