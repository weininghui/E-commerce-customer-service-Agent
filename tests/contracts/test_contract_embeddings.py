"""契约 E-01：embeddings 工厂新增 langchain_openai provider，现有行为不变。"""

from __future__ import annotations

import pytest

from agent_base.embeddings import HashEmbeddings, build_embeddings


def test_langchain_openai_returns_openai_embeddings(monkeypatch):
    # 无真实 API key 也能构造对象（契约验收：不发起真实请求）
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    embeddings = build_embeddings(
        provider="langchain_openai",
        model="text-embedding-v3",
        dimensions=1024,
    )
    assert type(embeddings).__module__ == "langchain_openai.embeddings.base"
    assert embeddings.model == "text-embedding-v3"


def test_langchain_openai_alias():
    embeddings = build_embeddings(provider="lc_openai")
    assert type(embeddings).__module__ == "langchain_openai.embeddings.base"


def test_langchain_openai_default_model_and_url():
    embeddings = build_embeddings(provider="langchain_openai")
    assert embeddings.model == "text-embedding-v3"
    assert "dashscope.aliyuncs.com" in embeddings.openai_api_base


def test_hash_provider_behavior_unchanged():
    hash_a = build_embeddings(provider="hash")
    assert isinstance(hash_a, HashEmbeddings)
    assert hash_a.dimensions == 512
    # 确定性：同一文本两次嵌入结果一致，且已归一化
    vector = hash_a.embed_query("坎地沙坦酯片一天吃几次？")
    assert hash_a.embed_query("坎地沙坦酯片一天吃几次？") == vector
    norm = sum(v * v for v in vector) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_ollama_returns_ollama_embeddings():
    # E-01 v0.1.1：ollama provider（本地 bge-m3），构造不发起真实请求
    embeddings = build_embeddings(provider="ollama")
    assert type(embeddings).__module__ == "langchain_ollama.embeddings"
    assert embeddings.model == "bge-m3"
    assert "localhost:11434" in embeddings.base_url


def test_ollama_custom_model_and_url():
    embeddings = build_embeddings(
        provider="ollama",
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11435",
    )
    assert embeddings.model == "nomic-embed-text"
    assert "127.0.0.1:11435" in embeddings.base_url


def test_ollama_requires_no_api_key():
    # ollama 不读取 api_key_env，任何 key 环境下都能构造
    embeddings = build_embeddings(provider="ollama")
    assert type(embeddings).__module__ == "langchain_ollama.embeddings"


def test_legacy_openai_provider_unchanged(monkeypatch):
    # provider="openai" 走 OPENAI_API_KEY（旧行为不变）；假 key 只用于构造对象
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embeddings = build_embeddings(provider="openai")
    assert type(embeddings).__module__ == "langchain_openai.embeddings.base"
    # v0.48 统一 OpenAIEmbeddings 后默认模型为 text-embedding-v3（DashScope 兼容端点）
    assert embeddings.model == "text-embedding-v3"


def test_unsupported_provider_raises():
    with pytest.raises(ValueError):
        build_embeddings(provider="unknown_provider")
