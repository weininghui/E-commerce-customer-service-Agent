"""领域适配器（P5-01）。

DomainAdapter 将"换领域只改配置"具象化：
  - intent schema / retrieval schema / prompt / compliance 全部从 YAML 加载。
  - 主项目默认电商域，领域行为由配置驱动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_base.config import load_yaml


@dataclass
class DomainAdapter:
    """领域配置容器。

    Attributes:
        name: 领域名（ecommerce）。
        intents: 意图定义 -> {keywords, sections}。
        intent_anchors: 意图改写锚点 -> list[str]（问题改写用，电商版）。
        retrieval: 检索配置（collection 名 / filter 字段）。
        prompt: 提示词配置（system prompt）。
        compliance: 合规配置（违禁词列表等）。
    """

    name: str
    intents: dict[str, dict[str, Any]] = field(default_factory=dict)
    intent_anchors: dict[str, list[str]] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    prompt: dict[str, str] = field(default_factory=dict)
    compliance: dict[str, Any] = field(default_factory=dict)


def load_domain(name: str = "ecommerce") -> DomainAdapter:
    """从 YAML 加载领域配置（P12：意图优先读 PG，YAML 降级）。

    意图加载优先级：
    1. PG intent_rules 表（生产运维即时生效）
    2. YAML 文件兜底（PG 不可用/为空）

    Args:
        name: 领域名，对应 configs/domain/{name}.yaml。

    Returns:
        填充好的 DomainAdapter 实例。
    """
    config_path = Path(f"configs/domain/{name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"领域配置不存在: {config_path}")
    data = load_yaml(config_path)
    intents = data.get("intents", {})

    # P12-02: 优先读 PG 意图（运行时修改即时生效）
    try:
        from agent_base.storage.pg import intent_to_domain_dict
        pg_intents = intent_to_domain_dict()
        if pg_intents:
            intents = pg_intents
    except Exception:
        pass

    return DomainAdapter(
        name=data.get("domain", name),
        intents=intents,
        intent_anchors=data.get("intent_anchors", {}),
        retrieval=data.get("retrieval", {}),
        prompt=data.get("prompt", {}),
        compliance=data.get("compliance", {}),
    )
