"""配置加载工具：YAML 解析、环境变量插值与点路径取值。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# 整串匹配 ${VAR} / ${VAR:-default}，仅对完整值插值，避免污染提示词模板
_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$")


# 整串匹配 ${VAR} / ${VAR:-default}，仅对完整值插值，避免污染提示词模板
_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$")


def load_env_file(path: Path | str) -> dict[str, str]:
    """解析 .env 文件为 {KEY: VALUE}（纯函数，无副作用；缺失/损坏返回空 dict）。

    零依赖替代 python-dotenv：支持注释、空行、引号包裹值、值内 =。
    """
    result: dict[str, str] = {}
    try:
        for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\x27", "\x22"):
                value = value[1:-1]
            result[key] = value
    except Exception:
        pass
    return result


def _load_env_file_once() -> None:
    """模块导入时兜底加载项目根 .env（幂等：已存在的环境变量优先）。

    docker 走 env_file 不受影响；本兜底解决「任意 shell 直接启动 uvicorn 时
    拿不到 .env 密钥」的问题。pytest 运行时不加载（避免测试误触真实 API）。
    """
    try:
        import sys

        if os.environ.get("AGENT_BASE_SKIP_ENV_FILE") or os.environ.get("PYTEST_CURRENT_TEST"):
            return
        if any("pytest" in str(arg) for arg in sys.argv):
            return
        env_file = Path(__file__).resolve().parents[2] / ".env"
        for key, value in load_env_file(env_file).items():
            os.environ.setdefault(key, value)
    except Exception:
        pass


_load_env_file_once()


def _interpolate(value: Any) -> Any:
    """递归替换 `${VAR}` / `${VAR:-default}` 为环境变量值。

    仅整串匹配（value 本身就是 `${...}`）才插值；default 缺省且环境变量
    未设置时返回空字符串，让调用方走自身兜底逻辑。
    """
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if isinstance(value, str):
        m = _ENV_PATTERN.match(value.strip())
        if m:
            name, default = m.group(1), m.group(2)
            env_val = os.getenv(name)
            # shell 语义：环境变量未设置或为空串时都回退默认值（:-），
            # 避免 docker compose 显式传空串把默认值顶掉
            if env_val:
                return env_val
            return default if default is not None else ""
        return value
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """加载 YAML 配置文件。

    支持环境变量插值：``${VAR}`` 或 ``${VAR:-默认值}``，仅整串匹配。

    Args:
        path: YAML 文件路径。

    Returns:
        配置字典。

    Raises:
        RuntimeError: 缺少 pyyaml 依赖。
        FileNotFoundError: 文件不存在。
        ValueError: YAML 根节点不是映射。
    """
    path = Path(path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from exc

    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return _interpolate(data)


def deep_get(data: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """按点路径安全取值，例如 deep_get(cfg, "embedding.model")。

    Args:
        data: 配置字典。
        dotted_key: 用点分隔的键路径。
        default: 缺失时返回的默认值。

    Returns:
        路径对应的值，缺失时返回 default。
    """
    current: Any = data
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
