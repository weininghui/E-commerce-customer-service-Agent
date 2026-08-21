"""把 v2 意图规则（关键词/章节/示例/优先级）同步到 PG intent_rules。

configs/domain/ecommerce.yaml 是规则唯一真相源；PG 表生产运维即时生效
（DomainAdapter 优先读 PG）。本脚本幂等：每次调用为每个意图生成新版本，
旧版本自动归档，管理端可回滚。

用法：
    python scripts/seed/seed_intent_rules_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_base.config import load_yaml  # noqa: E402
from agent_base.storage.pg import intent_upsert  # noqa: E402


def main() -> None:
    """把 domain yaml 的意图规则同步到 PG intent_rules（幂等，旧版归档）。"""
    data = load_yaml(str(ROOT / "configs/domain/ecommerce.yaml")) or {}
    intents = data.get("intents") or {}
    if not intents:
        raise SystemExit("domain yaml 缺少 intents 配置")
    versions = {}
    for name, cfg in intents.items():
        versions[name] = intent_upsert(
            intent=name,
            keywords=list(cfg.get("keywords") or []),
            sections=list(cfg.get("sections") or []),
            examples=list(cfg.get("examples") or []),
            priority=1.5 if name in {"recommendation", "size_recommendation", "comparison"} else 1.0,
        )
    print("synced intents:", len(versions))
    for name, version in versions.items():
        print(f"  {name}: v{version}")


if __name__ == "__main__":
    main()
