"""契约 C-01：app.yaml 新增 framework + vectorstore 段，缺省时默认值生效。"""

from __future__ import annotations

from pathlib import Path

from agent_base.config import deep_get, load_yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "app.yaml"


def test_config_file_exists():
    assert CONFIG_PATH.exists()


def test_framework_section_readable():
    config = load_yaml(CONFIG_PATH)
    assert deep_get(config, "framework.orchestrator", "classic") == "classic"
    assert deep_get(config, "framework.agent.enabled", False) is False


def test_vectorstore_section_readable():
    config = load_yaml(CONFIG_PATH)
    # P10: 默认切电商 — provider=qdrant, collection=ecommerce_*
    assert deep_get(config, "vectorstore.provider", "chroma") in {"chroma", "qdrant"}
    assert deep_get(config, "vectorstore.collection", "ecommerce_chunks") in {"ecommerce_chunks", "ecommerce_chunks"}
    assert deep_get(config, "vectorstore.summary_collection", "ecommerce_summaries") in {"ecommerce_summaries", "ecommerce_summaries"}


def test_interpolate_empty_env_falls_back_to_default(monkeypatch):
    """docker compose 显式传空串时，${VAR:-default} 应回退默认值（shell 语义）。"""
    from agent_base.config import _interpolate

    monkeypatch.setenv("EMPTY_VAR", "")
    assert _interpolate("${EMPTY_VAR:-admin-dev-token-2026}") == "admin-dev-token-2026"
    monkeypatch.setenv("EMPTY_VAR", "real-value")
    assert _interpolate("${EMPTY_VAR:-admin-dev-token-2026}") == "real-value"
    monkeypatch.delenv("EMPTY_VAR", raising=False)
    assert _interpolate("${EMPTY_VAR:-admin-dev-token-2026}") == "admin-dev-token-2026"


def test_defaults_apply_when_key_missing():
    config = load_yaml(CONFIG_PATH)
    assert deep_get(config, "framework.orchestrator", "classic") is not None
    # P10：url 可能为 null 或实际 Qdrant URL
    url_val = deep_get(config, "vectorstore.url", None)
    assert url_val is None or url_val == "null" or "localhost" in str(url_val) or "qdrant" in str(url_val)
    assert deep_get(config, "vectorstore.nonexistent_key", "default_value") == "default_value"


def test_existing_sections_unchanged():
    config = load_yaml(CONFIG_PATH)
    assert deep_get(config, "paths.chroma_dir", "data/chroma") == "data/chroma"
    # P10 默认域切电商
    assert deep_get(config, "index.chunk_collection", "ecommerce_chunks") in {"ecommerce_chunks", "ecommerce_chunks"}
    assert deep_get(config, "llm.model") is not None
    assert deep_get(config, "embedding.provider") is not None


def test_load_env_file_parses(tmp_path):
    """零依赖 .env 解析：注释/引号/值内等号/空行/坏行。"""
    from agent_base.config import load_env_file

    p = tmp_path / ".env"
    p.write_text(
        "# 注释\nA=1\nB=\"quoted value\"\nC='single'\nD=has=equals\n\nE=\nBADLINE\n",
        encoding="utf-8",
    )
    result = load_env_file(p)
    assert result["A"] == "1"
    assert result["B"] == "quoted value"
    assert result["C"] == "single"
    assert result["D"] == "has=equals"
    assert result["E"] == ""
    assert "BADLINE" not in result

    assert load_env_file(tmp_path / "missing.env") == {}
