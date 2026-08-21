"""ragas 0.4.3 与当前依赖栈的兼容 shim（两处修复，均幂等）。

1. langchain-community 0.4 移除了 chat_models.vertexai 模块（迁移到独立的
   langchain-google-vertexai 包），而 ragas 0.4.3 的 llms.base 仍然 import 它，
   导致 import ragas 直接 ModuleNotFoundError -> 注册占位模块（stub）。
2. ragas 0.4.3 的 async_utils.run 默认 apply_nest_asyncio，在 Windows
   （ProactorEventLoop + Python 3.12）上会死锁，评测卡死 -> 替换为
   apply_nest_asyncio = lambda: False（禁用补丁后实测正常到达模型 API）。

本模块在导入 ragas 前把缺失的模块路径注册进 sys.modules（幂等）。
项目使用显式 LangchainLLMWrapper 实例，不经过 ragas 的 provider 名分发，
stub 不会被实例化；安装了 langchain-google-vertexai 时则透传真实类。
"""

from __future__ import annotations

import sys
import types


def ensure_vertexai_compat() -> None:
    """注册 langchain_community.chat_models.vertexai 兼容模块（幂等）。"""
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    try:
        from langchain_google_vertexai import ChatVertexAI  # noqa: F401

        mod.ChatVertexAI = ChatVertexAI
    except Exception:
        class _ChatVertexAIStub:
            """占位：仅满足 ragas 的 import 契约，运行时不会被实例化。"""

        mod.ChatVertexAI = _ChatVertexAIStub
    sys.modules[name] = mod


def disable_nest_asyncio_patch() -> None:
    """禁用 ragas 的 nest_asyncio 补丁（Windows 死锁修复，幂等）。

    必须在 ragas 已导入之后调用；被替换的 apply_nest_asyncio 返回 False，
    ragas.async_utils.run 走普通 asyncio.run 路径。
    """
    try:
        import ragas.async_utils as async_utils

        if getattr(async_utils.apply_nest_asyncio, "_ragas_compat_noop", False):
            return

        def _noop() -> bool:
            return False

        _noop._ragas_compat_noop = True  # type: ignore[attr-defined]
        async_utils.apply_nest_asyncio = _noop
    except Exception:
        pass


ensure_vertexai_compat()
