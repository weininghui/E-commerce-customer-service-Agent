"""生产级安全组件：接口限流 + 默认凭据检查（零新依赖）。

- RateLimiter：进程内滑动窗口限流（按客户端 IP），单 worker 足够；
  多 worker 部署时按 IP 哈希均摊，仍可提供有效保护，Redis 化留作后续。
- 默认凭据检查：SECURITY_STRICT=1 时拒绝 configs/app.yaml 里随仓库分发的
  默认 admin_token / platform_token；启动时对默认密码/开发密钥打 WARNING 日志。
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Any

from agent_base.config import load_yaml


# 随仓库分发的默认 token（SECURITY_STRICT=1 时一律拒绝）
DEFAULT_TOKENS = {
    "admin_token": "admin-dev-token-2026",
    "platform_token": "platform-dev-token-2026",
}

DEFAULT_PASSWORDS = {"admin": "admin123", "agent": "agent123"}
DEV_SECRET_KEY = "rag-admin-dev-secret-change-me"


def _project_root() -> str:
    return str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def security_config() -> dict[str, Any]:
    """读取 configs/app.yaml security 段（带默认值）。"""
    try:
        cfg = load_yaml(f"{_project_root()}/configs/app.yaml") or {}
        return dict(cfg.get("security") or {})
    except Exception:
        return {}


def _as_bool(value: Any) -> bool:
    """配置值 → bool（YAML 插值后可能是字符串 "false"/"true"）。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def is_strict_mode() -> bool:
    """SECURITY_STRICT=1 或 security.strict=true 时启用严格模式。"""
    env = os.getenv("SECURITY_STRICT", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    return _as_bool(security_config().get("strict", False))


def reject_default_token(token_name: str) -> bool:
    """严格模式下，是否应拒绝该 token 名对应的默认值（调用方先比对再判断）。"""
    return is_strict_mode() and token_name in DEFAULT_TOKENS


def check_production_security(logger: Any | None = None) -> list[str]:
    """启动期生产安全检查：返回风险清单，并写 WARNING 日志（logger 可注入）。

    不抛出异常——检查只提示，不阻断启动（严格模式由 token 校验处强制）。
    """
    def _warn(message: str) -> None:
        if logger is not None:
            try:
                logger("WARNING", "security", "production_check", {"detail": message})
            except Exception:
                pass

    issues: list[str] = []
    sec = security_config()

    # 1) 默认 token 仍在使用
    if not is_strict_mode():
        for name, default in DEFAULT_TOKENS.items():
            configured = str(sec.get(name) or os.getenv(name.upper(), "")).strip()
            if configured == default:
                issues.append(f"默认 {name} 仍在生效（SECURITY_STRICT=1 可强制禁用）")

    # 2) token 签名密钥回退开发默认
    admin_secret = os.getenv("ADMIN_SECRET", "").strip()
    if not admin_secret and str(sec.get("secret_key", "")).strip() == "":
        issues.append("ADMIN_SECRET 未配置，登录 token 使用开发默认签名密钥")

    # 3) 种子账号默认密码
    for username, default_pw in DEFAULT_PASSWORDS.items():
        env_name = "ADMIN_INITIAL_PASSWORD" if username == "admin" else "AGENT_INITIAL_PASSWORD"
        if os.getenv(env_name, "").strip() in {"", default_pw}:
            issues.append(f"{username} 账号可能使用默认密码（{default_pw}），请尽快修改")

    for issue in issues:
        _warn(issue)
    return issues


class RateLimiter:
    """进程内滑动窗口限流器（线程安全，零依赖）。

    按客户端 IP 计数：窗口内超过 max_requests 次返回 False（拒绝）。
    定期清理过期桶，防止内存无限增长。
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        """初始化限流器。

        Args:
            max_requests: 窗口内最大请求数（按 key 计数）。
            window_seconds: 滑动窗口长度（秒）。
        """
        self.max_requests = max(1, int(max_requests))
        self.window = max(1.0, float(window_seconds))
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def allow(self, key: str) -> bool:
        """key（如客户端 IP）在窗口内是否还有配额。"""
        now = time.monotonic()
        with self._lock:
            if now - self._last_cleanup > self.window:
                self._cleanup(now)
            bucket = self._buckets[key]
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            return True

    def _cleanup(self, now: float) -> None:
        stale = [k for k, bucket in self._buckets.items() if not bucket or now - bucket[-1] > self.window]
        for key in stale:
            self._buckets.pop(key, None)
        self._last_cleanup = now


def build_rate_limiter() -> RateLimiter | None:
    """按配置构建限流器；security.rate_limit.enabled=false 时返回 None（关闭）。"""
    sec = security_config()
    rl = dict(sec.get("rate_limit") or {})
    if rl.get("enabled") is False:
        return None
    return RateLimiter(
        max_requests=int(rl.get("requests_per_minute", 60)),
        window_seconds=float(rl.get("window_seconds", 60.0)),
    )
