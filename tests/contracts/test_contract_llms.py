"""契约 L-01：LLM 工厂新增 langchain provider，none 仍返回 None。"""

from __future__ import annotations

import pytest

from agent_base.llms import build_chat_model


def test_langchain_returns_chat_openai(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    model = build_chat_model(provider="langchain", model="qwen3-max")
    # 观测底座：工厂返回统一埋点包装器，内部代理真实 ChatOpenAI
    from agent_base.monitoring.usage import _TrackingModel

    assert isinstance(model, _TrackingModel)
    assert model._inner.__class__.__module__ == "langchain_openai.chat_models.base"
    assert model.model_name == "qwen3-max"


def test_langchain_alias():
    model = build_chat_model(provider="lc_openai")
    from agent_base.monitoring.usage import _TrackingModel

    assert isinstance(model, _TrackingModel)
    assert model._inner.__class__.__module__ == "langchain_openai.chat_models.base"


def test_langchain_default_model_and_url():
    model = build_chat_model(provider="langchain")
    assert model.model_name == "deepseek-v4-pro"
    assert "api.deepseek.com" in str(model.openai_api_base)


def test_langchain_timeout_and_max_retries(monkeypatch):
    # v0.28.1：P18 摘要生成依赖 timeout/max_retries 透传，缺参即 TypeError
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    model = build_chat_model(provider="langchain", timeout=30, max_retries=3)
    # langchain-openai 1.4.x：构造别名 timeout → 字段 request_timeout
    assert model.request_timeout == 30
    assert model.max_retries == 3


def test_none_provider_returns_none():
    assert build_chat_model(provider="none") is None
    assert build_chat_model(provider="off") is None
    assert build_chat_model(provider="false") is None


def test_legacy_handwritten_provider_removed():
    # v0.18.0：手写 OpenAI 兼容客户端已移除，统一 LangChain（ChatOpenAI）
    with pytest.raises(ValueError):
        build_chat_model(provider="dashscope_compatible")
    with pytest.raises(ValueError):
        build_chat_model(provider="openai_compatible")


def test_unsupported_provider_raises():
    with pytest.raises(ValueError):
        build_chat_model(provider="unknown_provider")
