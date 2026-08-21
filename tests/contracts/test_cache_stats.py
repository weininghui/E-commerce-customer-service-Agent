"""契约 P19d：缓存命中统计（Redis 原子计数，跨进程/跨重启）。"""

from __future__ import annotations

from agent_base.storage import cache as cache_mod


def _cleanup():
    client = cache_mod._get_client()
    if client is not None:
        try:
            client.delete(cache_mod._HITS_KEY, cache_mod._MISSES_KEY)
            client.delete("rag:cache:stats-test-key")
        except Exception:
            pass


def test_cache_stats_redis_counters():
    """命中/未命中写入 Redis 统计，cache_stats 读取准确。"""
    _cleanup()
    try:
        cache_mod.get_cached("rag:cache:stats-test-miss-1")
        cache_mod.get_cached("rag:cache:stats-test-miss-2")
        cache_mod.set_cache("rag:cache:stats-test-key", {"answer": "x"})
        cache_mod.get_cached("rag:cache:stats-test-key")

        stats = cache_mod.cache_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 2
        assert stats["total"] == stats["hits"] + stats["misses"]
        assert 0.0 <= stats["hit_rate"] <= 1.0

        # 统计持久化在 Redis（内存计数与 Redis 一致）
        client = cache_mod._get_client()
        if client is not None:
            assert int(client.get(cache_mod._HITS_KEY) or 0) == stats["hits"]
            assert int(client.get(cache_mod._MISSES_KEY) or 0) == stats["misses"]
    finally:
        _cleanup()
