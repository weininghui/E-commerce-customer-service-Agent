"""P32d/e 检索增强按需触发：Decomposition + Multi-Query 分配逻辑。

- Decomposition：复合问题（比较句式 / 多个问号 / 并列连词）→ 拆子问题 RRF 融合
- Multi-Query：单意图 + 初检质量低 → LLM 改写多查询变体合并去重
- 两者互斥，Decomposition 优先；都关闭时链路零开销
"""

from __future__ import annotations

import re
from typing import Any

# ── 复合问题检测 ──

_COMPOUND_PATTERNS = [
    r"哪个好",
    r"哪个更",
    r"和.*比",
    r"与.*比",
    r"对比",
    r"有什么区别",
    r"区别.*什么",
    r"还是.*好",
    r"哪个适合",
    r"还有",
    r"以及",
]


def _looks_like_compound(question: str) -> bool:
    """检测复合/比较句式（规则版）。"""
    if any(re.search(pattern, question) for pattern in _COMPOUND_PATTERNS):
        return True
    # 多个问号
    if question.count("?") + question.count("？") >= 2:
        return True
    # 并列连词
    if re.search(r"(?:和|与|、).*(?:和|与|、)", question):
        return True
    return False


# ── 初检质量评估 ──


def _initial_quality_low(
    stage_docs: list[Any],
    top1_threshold: float = 0.55,
    hit_threshold: int = 3,
    gap_margin: float = 0.08,
) -> bool:
    """评估初检质量是否偏低（需 Multi-Query 增强）。

    Args:
        stage_docs: 初检召回文档列表。
        top1_threshold: top1 得分低于此值认为质量低。
        hit_threshold: 命中数低于此值认为质量低。
        gap_margin: top1 与次高分的领先差距，超过则认为有明确最佳命中
            （相对判别，避免绝对阈值随 embedding/向量库分数尺度漂移）。

    Returns:
        True 表示初检质量偏低。
    """
    if not stage_docs:
        return True
    if len(stage_docs) < hit_threshold:
        return True
    top1_score = _top1_score(stage_docs)
    if top1_score is None:
        return False
    # BUG-23：相对判别优先——top1 明显领先次高分时视为有效命中，
    # 不再依赖绝对阈值（bge-m3/Qdrant 余弦分数尺度与 0.3 不匹配会误触发）
    top2_score = _nth_score(stage_docs, 1)
    if top2_score is not None and (top1_score - top2_score) >= gap_margin:
        return False
    return top1_score < top1_threshold


def _top1_score(docs: list[Any]) -> float | None:
    """获取首篇文档的相似度得分。"""
    return _nth_score(docs, 0)


def _nth_score(docs: list[Any], index: int) -> float | None:
    """获取第 index 篇文档的相似度得分（0 起）。"""
    if not docs:
        return None
    if index >= len(docs):
        return None
    doc = docs[index]
    metadata = getattr(doc, "metadata", {}) or {}
# Qdrant 分数 / Chroma 分数
    score = metadata.get("score") or metadata.get("vector_score") or metadata.get("rerank_score")
    if score is not None:
        return float(score)
    return None


# ── 增强配置读取 ──


def load_enhancement_config() -> dict[str, Any]:
    """从 configs/app.yaml 读取增强配置。"""
    try:
        from agent_base.config import deep_get, load_yaml

        cfg = load_yaml("configs/app.yaml") or {}
        return {
            "decomposition_enabled": bool(deep_get(cfg, "retrieval.decomposition.enabled", False)),
            "multi_query_enabled": bool(deep_get(cfg, "retrieval.multi_query.enabled", False)),
            "multi_query_variants": int(deep_get(cfg, "retrieval.multi_query.variants", 3)),
            "quality_top1_threshold": float(deep_get(cfg, "retrieval.enhancement.quality_top1_threshold", 0.3)),
            "quality_hit_threshold": int(deep_get(cfg, "retrieval.enhancement.quality_hit_threshold", 3)),
            "quality_gap_margin": float(deep_get(cfg, "retrieval.enhancement.quality_gap_margin", 0.08)),
        }
    except Exception:
        return {}


def assess_enhancement(
    question: str,
    stage_docs: list[Any],
    intent: str,
) -> dict[str, Any]:
    """评估是否触发检索增强及触发哪种。

    Args:
        question: 用户问题。
        stage_docs: 初检召回文档（用于质量评估）。
        intent: 意图路由结果。

    Returns:
        {"triggered": bool, "type": "decomposition"|"multi_query"|"none",
         "reason": str, "variants": int}
    """
    config = load_enhancement_config()
    decomposition_enabled = config.get("decomposition_enabled", False)
    multi_query_enabled = config.get("multi_query_enabled", False)

    # 互斥分配：Decomposition 优先
    if decomposition_enabled and _looks_like_compound(question):
        return {
            "triggered": True,
            "type": "decomposition",
            "reason": "检测到复合/比较句式，拆子问题分别检索后 RRF 融合",
            "variants": 0,  # 运行时决定
        }

    if multi_query_enabled and _initial_quality_low(
        stage_docs,
        top1_threshold=config.get("quality_top1_threshold", 0.55),
        hit_threshold=config.get("quality_hit_threshold", 3),
        gap_margin=config.get("quality_gap_margin", 0.08),
    ):
        return {
            "triggered": True,
            "type": "multi_query",
            "reason": "初检质量偏低（top1 得分或命中数不足），多查询改写增强召回",
            "variants": config.get("multi_query_variants", 3),
        }

    return {"triggered": False, "type": "none", "reason": "", "variants": 0}


def run_decomposition_enhancement(
    question: str,
    vector_store: Any,
    k: int,
    metadata_filter: dict[str, Any] | None,
) -> list[Any]:
    """执行 Decomposition 增强：拆子问题 → 分别检索 → RRF 融合。

    Args:
        question: 用户问题。
        vector_store: 向量库。
        k: 每路返回条数。
        metadata_filter: 检索过滤条件。

    Returns:
        融合后的文档列表。
    """
    from agent_base.retrieval.decomposition import self_ask_retrieve

    return self_ask_retrieve(question, vector_store, k=k, metadata_filter=metadata_filter)


def run_multi_query_enhancement(
    question: str,
    vector_store: Any,
    llm: Any | None,
    k: int,
    metadata_filter: dict[str, Any] | None,
) -> list[Any]:
    """执行 Multi-Query 增强：LLM 改写多查询 → 合并去重。

    Args:
        question: 用户问题。
        vector_store: 向量库。
        llm: LLM 模型（None 时退化为单查询）。
        k: 最终返回条数。
        metadata_filter: 检索过滤条件。

    Returns:
        去重合并后的文档列表。
    """
    from agent_base.retrieval.multi_query import multi_query_retrieve

    return multi_query_retrieve(question, vector_store, llm=llm, k=k, metadata_filter=metadata_filter)
