"""契约测试：生产级安全组件（限流器 + 默认凭据检查）。

覆盖：
1. RateLimiter：窗口内配额用尽即拒绝，窗口滑动后恢复，并发安全；
2. 严格模式开关：SECURITY_STRICT 环境变量优先级高于 YAML；
3. 生产安全检查：默认 token/开发密钥/默认密码时返回风险清单。
"""

from __future__ import annotations

import threading

from agent_base.security import (
    RateLimiter,
    check_production_security,
    is_strict_mode,
    reject_default_token,
)


def test_rate_limiter_blocks_over_quota():
    """窗口内超过配额即拒绝；不同 key 互不影响。"""
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        assert limiter.allow("ip-a") is True
    assert limiter.allow("ip-a") is False
    assert limiter.allow("ip-b") is True


def test_rate_limiter_window_slides(monkeypatch):
    """窗口过期后配额恢复（用 monkeypatch 拨时间，基准取真实时钟避免回拨）。"""
    import time as _time

    base = _time.monotonic()
    fake_now = [base]
    monkeypatch.setattr(_time, "monotonic", lambda: fake_now[0])
    # limiter 在打补丁后构造，内部 _last_cleanup 使用假时钟
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow("ip-c") and limiter.allow("ip-c")
    assert limiter.allow("ip-c") is False

    # 拨快 61 秒：过期条目被剔除，配额恢复
    fake_now[0] = base + 61.0
    assert limiter.allow("ip-c") is True


def test_rate_limiter_thread_safe():
    """并发请求不超过配额（原子计数）。"""
    limiter = RateLimiter(max_requests=50, window_seconds=60)
    results: list[bool] = []
    lock = threading.Lock()

    def worker():
        ok = limiter.allow("shared-ip")
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 50
    assert results.count(False) == 50


def test_strict_mode_env_priority(monkeypatch):
    """SECURITY_STRICT=1 开启严格模式；默认 token 在严格模式下被拒绝。"""
    monkeypatch.setenv("SECURITY_STRICT", "1")
    assert is_strict_mode() is True
    assert reject_default_token("admin_token") is True
    assert reject_default_token("platform_token") is True
    monkeypatch.setenv("SECURITY_STRICT", "0")
    assert is_strict_mode() is False
    assert reject_default_token("admin_token") is False


def test_production_check_reports_defaults():
    """默认凭据在用时返回风险清单（不抛异常）。"""
    issues = check_production_security(logger=None)
    assert isinstance(issues, list)
    joined = "；".join(issues)
    # 开发默认场景下至少命中：默认 token / 开发密钥 / 默认密码 中的一项
    assert any(k in joined for k in ("默认", "ADMIN_SECRET", "admin123"))
