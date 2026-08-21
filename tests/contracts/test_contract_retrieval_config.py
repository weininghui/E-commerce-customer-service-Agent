"""契约：RetrievalConfig 默认值 == 旧函数签名默认值（参数对象迁移不悄悄改行为）。"""

from __future__ import annotations

from agent_base.retrieval.retrieval_config import GenerateConfig, RerankConfig, RetrievalConfig


def test_retrieval_defaults_match_old_signature():
    """retrieve_advanced 旧签名默认值逐项对齐。"""
    cfg = RetrievalConfig()
    assert cfg.top_k == 6
    assert cfg.candidate_k is None
    assert cfg.rerank == "auto"
    assert cfg.preserve_preferred_sections is True
    assert cfg.use_rewrite is True
    assert cfg.fallback_without_filter is True
    assert cfg.product_name is None
    assert cfg.product_spec is None
    assert cfg.category is None


def test_rerank_defaults_match_old_signature():
    """rerank_model_* 旧签名默认值逐项对齐。"""
    r = RerankConfig()
    assert r.provider == "none"
    assert r.model == "gte-rerank-v2"
    assert r.endpoint is None
    assert r.api_key_env == "DASHSCOPE_API_KEY"
    assert r.timeout == 30
    assert r.strategies is None
    assert r.preserve_preferred_sections is True


def test_generate_defaults_match_old_signature():
    """llm_* 旧签名默认值逐项对齐。"""
    g = GenerateConfig()
    assert g.provider == "none"
    assert g.model is None
    assert g.base_url is None
    assert g.api_key_env == "DASHSCOPE_API_KEY"
    assert g.temperature == 0.1
    assert g.use_llm is False
    assert g.evidence_max_chars_per_doc == 1200
    assert g.evidence_max_total_chars == 6000
    assert g.evidence_max_chars_per_product == 1800
    assert g.evidence_max_chars_per_product_doc == 900


def test_from_runtime_maps_config_sections():
    """from_runtime 对齐 get_runtime() 的配置键。"""
    runtime = {
        "llm_config": {
            "provider": "langchain",
            "model": "deepseek-v4-pro",
            "base_url": "https://api.deepseek.com",
            "api_key_env": "ANTHROPIC_AUTH_TOKEN",
            "temperature": 0.2,
            "evidence": {"max_chars_per_doc": 800, "max_total_chars": 3000},
        },
        "rerank_config": {
            "provider": "local_tei",
            "model": "bge-reranker-v2-m3",
            "endpoint": "http://localhost:8081/rerank",
            "timeout": 15,
        },
        "intent_classifier_config": {"enabled": True},
        "prompts_path": "configs/prompts_ecommerce.yaml",
    }
    cfg = RetrievalConfig.from_runtime(runtime)
    assert cfg.llm.provider == "langchain"
    assert cfg.llm.model == "deepseek-v4-pro"
    assert cfg.llm.api_key_env == "ANTHROPIC_AUTH_TOKEN"
    assert cfg.llm.temperature == 0.2
    assert cfg.llm.use_llm is True  # provider 非 none
    assert cfg.llm.evidence_max_chars_per_doc == 800
    assert cfg.llm.evidence_max_total_chars == 3000
    assert cfg.rerank_model.provider == "local_tei"
    assert cfg.rerank_model.endpoint == "http://localhost:8081/rerank"
    assert cfg.rerank_model.timeout == 15
    assert cfg.intent_classifier == {"enabled": True}
    assert cfg.prompts_path == "configs/prompts_ecommerce.yaml"
