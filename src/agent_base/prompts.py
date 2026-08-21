"""Prompt 加载工具（P26 单一口径）。

所有生成/分类/提炼 prompt 从 ``configs/prompts_ecommerce.yaml`` 加载，
代码只保留兜底常量。各模块通过 :func:`get_prompt` 读取对应段的 system /
user_template，避免重复内联副本。

用法::

    from agent_base.prompts import get_prompt, prompts_path

    sys_prompt = get_prompt("intent", "system")
    user_tpl = get_prompt("intent", "user_template")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from agent_base.config import deep_get, load_yaml


def prompts_path() -> str:
    """返回配置的 prompts YAML 路径（app.yaml prompts.path，默认 prompts_ecommerce.yaml）。"""
    try:
        cfg = load_yaml("configs/app.yaml") or {}
        return str(deep_get(cfg, "prompts.path", "configs/prompts_ecommerce.yaml"))
    except Exception:
        return "configs/prompts_ecommerce.yaml"


@lru_cache(maxsize=8)
def _load_prompts() -> dict[str, Any]:
    """读取 prompts YAML（安全，失败返回 {}）。"""
    path = Path(prompts_path())
    if not path.exists():
        return {}
    try:
        return load_yaml(path) or {}
    except Exception:
        return {}


def get_prompt(section: str, key: str = "system", default: str = "") -> str:
    """读取某段 prompt 的指定键（如 intent.system），缺失回退 default。"""
    cfg = _load_prompts()
    seg = cfg.get(section) or {}
    value = seg.get(key) if isinstance(seg, dict) else None
    if value is None or not str(value).strip():
        return default
    return str(value).strip()

# ── 内置提示词目录（只读展示；管理端「提示词库」页使用） ──

PROMPT_SECTIONS: list[dict[str, str]] = [
    {"section": "qa", "name_zh": "客服问答", "name_en": "QA Persona",
     "desc_zh": "买家对话的主人格「小满」与回复格式铁律", "desc_en": "The “Xiaoman” persona and reply-format rules for buyer chat"},
    {"section": "supervisor", "name_zh": "任务编排", "name_en": "Supervisor Planning",
     "desc_zh": "把用户问题拆解为结构化任务计划", "desc_en": "Decompose a user question into a structured task plan"},
    {"section": "intent", "name_zh": "意图识别", "name_en": "Intent Classification",
     "desc_zh": "规则路由失败时的 LLM 兜底分类", "desc_en": "LLM fallback classification when rule routing misses"},
    {"section": "memory", "name_zh": "记忆提炼", "name_en": "Memory Extraction",
     "desc_zh": "从对话中提取用户画像标签（含隐私红线）", "desc_en": "Extract structured profile tags from dialogue (privacy-safe)"},
    {"section": "rewrite", "name_zh": "查询改写", "name_en": "Query Rewrite",
     "desc_zh": "把用户问题改写成多角度检索查询", "desc_en": "Rewrite the user question into multi-angle retrieval queries"},
    {"section": "variant", "name_zh": "问题变体", "name_en": "Query Variants",
     "desc_zh": "生成语义等价的问法变体提升召回", "desc_en": "Generate semantically equivalent question variants to boost recall"},
    {"section": "pre_review", "name_zh": "入库评审", "name_en": "Intake Review",
     "desc_zh": "文档四维评审（完整/合规/分类/路由）", "desc_en": "Four-dimension document review (completeness / compliance / type / routing)"},
    {"section": "summary", "name_zh": "检索摘要", "name_en": "Retrieval Summary",
     "desc_zh": "生成面向检索定位的压缩摘要", "desc_en": "Generate compressed, retrieval-oriented summaries"},
    {"section": "polish", "name_zh": "格式整理", "name_en": "Format Polish",
     "desc_zh": "文件清洗页「AI 整理格式」使用的规范", "desc_en": "Rules used by “AI format” on the file-cleaning page"},
    {"section": "knowledge_ops", "name_zh": "知识运营", "name_en": "Knowledge Ops",
     "desc_zh": "把运营大白话指令解析为知识库工具调用", "desc_en": "Parse plain-language ops commands into knowledge-base tool calls"},
    {"section": "improve_intent", "name_zh": "意图优化", "name_en": "Intent Improvement",
     "desc_zh": "优化意图关键词、章节与示例", "desc_en": "Optimize intent keywords, sections and examples"},
]


def prompt_catalog() -> list[dict[str, Any]]:
    """返回全部内置提示词（只读目录，供管理端展示）。

    以 YAML 为唯一真相源；元信息（名称/描述）来自 PROMPT_SECTIONS。
    """
    cfg = _load_prompts()
    meta = {m["section"]: m for m in PROMPT_SECTIONS}
    items: list[dict[str, Any]] = []
    for section, seg in cfg.items():
        if not isinstance(seg, dict):
            continue
        for key, content in seg.items():
            if not str(content).strip():
                continue
            m = meta.get(section, {})
            items.append({
                "section": section,
                "key": key,
                "name_zh": m.get("name_zh", section),
                "name_en": m.get("name_en", section),
                "desc_zh": m.get("desc_zh", ""),
                "desc_en": m.get("desc_en", ""),
                "content": str(content).strip(),
            })
    # 稳定顺序：按 PROMPT_SECTIONS 定义顺序，未定义的排最后
    order = {m["section"]: i for i, m in enumerate(PROMPT_SECTIONS)}
    items.sort(key=lambda it: (order.get(it["section"], 999), it["key"]))
    return items

