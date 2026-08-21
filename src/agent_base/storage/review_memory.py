"""评审短期记忆（P27）：Redis TTL + PG 双写。

打回时写：Redis ``review:memory:{doc_id}:{round}``（TTL 7 天）+ PG reject_reason。
重提时读：优先 Redis，缺失回退 PG reject_reason（不阻塞）。

设计（P27 契约 §4）：记忆要释放（TTL 自动过期），审计要保留（PG audit）。
"""

from __future__ import annotations

import json
import os
from typing import Any

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MEMORY_TTL = 7 * 24 * 3600  # 7 天

_client: redis.Redis | None = None


def _get_client() -> redis.Redis | None:
    """Lazy Redis client（复用 cache.py 的连接容错模式）。"""
    global _client
    if _client is not None:
        return _client
    try:
        _client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2,
                              socket_timeout=2, decode_responses=True)
        _client.ping()
    except Exception:
        _client = None
    return _client


def memory_key(doc_id: str, round_no: int) -> str:
    """评审记忆 Redis key。"""
    return f"review:memory:{doc_id}:{round_no}"


def save_memory(doc_id: str, round_no: int, reject_reason: str, decision: dict[str, Any] | None = None) -> None:
    """打回时写短期记忆（Redis TTL 7 天；Redis 不可用静默跳过）。"""
    client = _get_client()
    if client is None:
        return
    try:
        payload = {
            "reject_reason": reject_reason,
            "decision": decision or {},
        }
        client.set(memory_key(doc_id, round_no), json.dumps(payload, ensure_ascii=False), ex=MEMORY_TTL)
    except Exception:
        pass


def load_memory(doc_id: str, round_no: int) -> dict[str, Any] | None:
    """重提时读短期记忆（优先 Redis，缺失回退 PG reject_reason）。"""
    client = _get_client()
    if client is not None:
        try:
            raw = client.get(memory_key(doc_id, round_no))
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    # 回退 PG（主数据兜底）
    try:
        from agent_base.storage.pg import strategy_get
        tag = strategy_get(doc_id)
        if tag and tag.get("reject_reason"):
            return {
                "reject_reason": tag.get("reject_reason", ""),
                "decision": tag.get("first_review") or {},
            }
    except Exception:
        pass
    return None


def clear_memory(doc_id: str, round_no: int) -> None:
    """评审通过后清除短期记忆（记忆释放）。"""
    client = _get_client()
    if client is None:
        return
    try:
        client.delete(memory_key(doc_id, round_no))
    except Exception:
        pass
