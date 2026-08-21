"""检索/生成配置对象（参数对象模式）。

收敛 retrieve_advanced / answer_question* 的长签名：配置项集中在 dataclass，
函数签名只保留核心数据 + 一个 cfg 参数。默认值全在这里管理，加新配置项不改签名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RerankConfig:
    """重排模型配置（对应 configs/app.yaml rerank 段）。"""

    provider: str = "none"
    model: str = "gte-rerank-v2"
    endpoint: str | None = None
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout: int = 30
    strategies: list[str] | None = None
    preserve_preferred_sections: bool = True


@dataclass(slots=True)
class GenerateConfig:
    """LLM 生成 + 证据预算配置（对应 configs/app.yaml llm 段）。"""

    provider: str = "none"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str = "DASHSCOPE_API_KEY"
    temperature: float = 0.1
    use_llm: bool = False
    evidence_max_chars_per_doc: int = 1200
    evidence_max_total_chars: int = 6000
    evidence_max_chars_per_product: int = 1800
    evidence_max_chars_per_product_doc: int = 900


@dataclass(slots=True)
class RetrievalConfig:
    """检索全链路配置（retrieve_advanced / answer_question* 共用）。"""

    top_k: int = 6
    candidate_k: int | None = None
    rerank: str = "auto"
    preserve_preferred_sections: bool = True
    intent_classifier: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
    product_name: str | None = None
    product_spec: str | None = None
    category: str | None = None
    use_rewrite: bool = True
    fallback_without_filter: bool = True
    prompts_path: str | None = None
    rerank_model: RerankConfig = field(default_factory=RerankConfig)
    llm: GenerateConfig = field(default_factory=GenerateConfig)

    @classmethod
    def from_runtime(cls, runtime: dict[str, Any]) -> "RetrievalConfig":
        """从 get_runtime() 的运行时 dict 构造配置（对齐调用方现有读取键）。"""
        rerank_cfg = runtime.get("rerank_config") or {}
        llm_cfg = runtime.get("llm_config") or {}
        evidence_cfg = llm_cfg.get("evidence") or {}
        llm_provider = str(llm_cfg.get("provider", "none"))
        return cls(
            intent_classifier=runtime.get("intent_classifier_config"),
            prompts_path=runtime.get("prompts_path"),
            rerank_model=RerankConfig(
                provider=str(rerank_cfg.get("provider", "none")),
                model=str(rerank_cfg.get("model", "gte-rerank-v2")),
                endpoint=rerank_cfg.get("endpoint"),
                api_key_env=str(rerank_cfg.get("api_key_env", "DASHSCOPE_API_KEY")),
                timeout=int(rerank_cfg.get("timeout", 30)),
                strategies=rerank_cfg.get("use_for_strategies"),
                preserve_preferred_sections=bool(rerank_cfg.get("preserve_preferred_sections", True)),
            ),
            llm=GenerateConfig(
                provider=llm_provider,
                model=llm_cfg.get("model"),
                base_url=llm_cfg.get("base_url"),
                api_key_env=str(llm_cfg.get("api_key_env", "DASHSCOPE_API_KEY")),
                temperature=float(llm_cfg.get("temperature", 0.1)),
                use_llm=llm_provider not in {"none", "off", "false"},
                evidence_max_chars_per_doc=int(evidence_cfg.get("max_chars_per_doc", 1200)),
                evidence_max_total_chars=int(evidence_cfg.get("max_total_chars", 6000)),
                evidence_max_chars_per_product=int(evidence_cfg.get("max_chars_per_product", 1800)),
                evidence_max_chars_per_product_doc=int(evidence_cfg.get("max_chars_per_product_doc", 900)),
            ),
        )
