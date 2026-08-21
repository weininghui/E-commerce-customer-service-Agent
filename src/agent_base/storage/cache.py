"""Redis 检索缓存（P9-04）。

缓存策略：question + 约束 hash → 检索结果 JSON，TTL 1800s。
缓存命中/未命中计数用于压测报告的缓存命中率统计。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = 1800  # 30 min
DATA_VERSION = "2026-08-05-v2"  # v0.21.0 电商化 + 轻量 payload 后递升：旧缓存（医药文案）全局失效

_client: redis.Redis | None = None
_cache_hits = 0
_cache_misses = 0
# P19d: 命中统计改为 Redis 原子计数（跨进程/跨重启准确），键前缀
_HITS_KEY = "rag:cache:stats:hits"
_MISSES_KEY = "rag:cache:stats:misses"


def _get_client() -> redis.Redis | None:
    """懒加载 Redis 客户端（连接失败可容忍）。"""
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


def cache_key(question: str, constraints: dict[str, Any] | None = None) -> str:
    """生成确定性缓存键。

    Args:
        question: 用户问题。
        constraints: 可选商品/类目约束字典。

    Returns:
        形如 "rag:cache:abc123def" 的缓存键。
    """
    payload = question
    if constraints:
        payload += json.dumps(constraints, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"rag:cache:{digest}"


def get_cached(key: str) -> dict[str, Any] | None:
    """读取缓存结果；未命中或 Redis 不可用时返回 None。"""
    client = _get_client()
    if client is None:
        _record_miss()
        return None
    try:
        data = client.get(key)
        if data:
            _record_hit()
            return json.loads(data)
        _record_miss()
        return None
    except Exception:
        _record_miss()
        return None


def _record_hit() -> None:
    """命中计数：优先 Redis 原子 INCR，Redis 不可用回退内存。"""
    global _cache_hits
    client = _get_client()
    if client is not None:
        try:
            client.incr(_HITS_KEY)
            return
        except Exception:
            pass
    _cache_hits += 1


def _record_miss() -> None:
    """未命中计数：优先 Redis 原子 INCR，Redis 不可用回退内存。"""
    global _cache_misses
    client = _get_client()
    if client is not None:
        try:
            client.incr(_MISSES_KEY)
            return
        except Exception:
            pass
    _cache_misses += 1


def set_cache(key: str, value: dict[str, Any], ttl: int = CACHE_TTL) -> None:
    """带 TTL 写入缓存；Redis 异常静默忽略。"""
    client = _get_client()
    if client is None:
        return
    try:
        client.setex(key, ttl, json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        pass


def cache_stats() -> dict[str, Any]:
    """返回缓存使用统计。"""
    hits = _cache_hits
    misses = _cache_misses
    client = _get_client()
    if client is not None:
        try:
            r_hits = client.get(_HITS_KEY)
            r_misses = client.get(_MISSES_KEY)
            if r_hits is not None:
                hits = int(r_hits)
            if r_misses is not None:
                misses = int(r_misses)
        except Exception:
            pass
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 3) if total > 0 else 0.0,
        "total": total,
    }


def invalidate_pattern(pattern: str = "rag:cache:*") -> int:
    """删除匹配 pattern 的缓存键，返回删除数量。

    Args:
        pattern: Redis key 模式（默认全部 rag 缓存键）。
    """
    client = _get_client()
    if client is None:
        return 0
    try:
        keys = list(client.scan_iter(match=pattern, count=100))
        if keys:
            return client.delete(*keys)
    except Exception:
        pass
    return 0
