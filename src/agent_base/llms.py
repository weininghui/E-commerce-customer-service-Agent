"""Chat 模型工厂：统一创建 LangChain OpenAI 兼容客户端。"""

from __future__ import annotations

import os


def build_chat_model(
    provider: str = "none",
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
    temperature: float = 0.1,
    timeout: float | None = None,
    max_retries: int = 2,
    tracking: bool = True,
    tracking_agent: str = "",
    tracking_source: str = "chat",
):
    """按配置创建 chat 模型（工厂，统一 LangChain）。

    支持的 provider：
    - none / off / false：返回 None，问答链走模板兜底。
    - langchain / lc_openai：langchain-openai 的 ChatOpenAI（OpenAI 兼容，
      默认 DeepSeek 网关；base_url/api_key_env 由配置提供）。

    Args:
        provider: chat provider 名。
        model: 模型名；None 时使用默认模型。
        base_url: OpenAI 兼容服务地址；None 时使用默认地址。
        api_key_env: 存放 API key 的环境变量名。
        temperature: 采样温度。
        timeout: 单次请求超时秒数（None 时用生产默认 60s，避免请求挂死阻塞链路）。
        max_retries: 请求失败自动重试次数（SDK 原生重试）。
        tracking: 是否接入用量/失败埋点（监控层，默认开启）。
        tracking_agent: 埋点归属 Agent 标识（如 supervisor / worker / pre_review）。
        tracking_source: 埋点来源场景（chat / eval / pipeline 等）。

    Returns:
        langchain_openai.ChatOpenAI；provider=none 时返回 None。

    Raises:
        RuntimeError: 缺少依赖包。
        ValueError: 不支持的 provider（legacy 手写客户端已移除，见 CHANGELOG）。
    """
    provider = (provider or "none").lower()
    if provider in {"none", "off", "false"}:
        return None
    if provider in {"langchain", "lc_openai"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Missing dependency: langchain-openai.") from exc
        # 与 embeddings 同理：langchain-openai 要求 api_key 非空才能构造对象。
        api_key = os.getenv(api_key_env) or f"missing-{api_key_env}"
        model_obj = ChatOpenAI(
            model=model or "deepseek-v4-pro",
            base_url=base_url or "https://api.deepseek.com",
            api_key=api_key,
            temperature=temperature,
            timeout=timeout if timeout is not None else 60,
            max_retries=max(0, int(max_retries)),
        )
        if tracking:
            from agent_base.monitoring.usage import wrap_chat_model

            return wrap_chat_model(
                model_obj,
                agent=tracking_agent,
                source=tracking_source,
            )
        return model_obj
    raise ValueError(f"Unsupported LLM provider: {provider}")
