"""Chroma where -> Qdrant payload Filter adapter (P9-02).

Converts Chroma metadata filter syntax to Qdrant payload Filter,
so the existing retrieval layer works with both vector stores unchanged.

Qdrant Filter mapping:
  {"key": "value"}          -> {"must": [{"key": "key","match":{"value":"value"}}]}
  {"key": {"$in": [...]}}   -> {"must": [{"key": "key","match":{"any":[...]}}]}
  {"key": {"$ne": "v"}}     -> {"must": [{"key": "key","match":{"except":["v"]}}]}
  {"$and": [...]}           -> {"must": [...]}
  {"$or": [...]}            -> {"should": [...]}
"""

from __future__ import annotations

from typing import Any


def chroma_to_qdrant_filter(chroma_filter: dict[str, Any] | None) -> dict[str, Any] | None:
    """将 Chroma where 过滤条件转换为 Qdrant payload Filter。

    Args:
        chroma_filter: Chroma 风格 metadata 过滤字典；空或 None 返回 None。

    Returns:
        供 QdrantVectorStore.similarity_search 使用的 Qdrant Filter 字典。
    """
    if not chroma_filter:
        return None

    if "$and" in chroma_filter:
        children = [_convert_condition(c) for c in chroma_filter["$and"]]
        return {"must": [c for c in children if c]} if any(children) else None

    if "$or" in chroma_filter:
        children = [_convert_condition(c) for c in chroma_filter["$or"]]
        return {"should": [c for c in children if c]} if any(children) else None

    conditions = _convert_top_level(chroma_filter)
    if not conditions:
        return None
    return {"must": conditions}


def _convert_top_level(chroma_filter: dict[str, Any]) -> list[dict[str, Any]]:
    """转换顶层键值对（无 $and/$or 包装）。"""
    conditions: list[dict[str, Any]] = []
    for key, value in chroma_filter.items():
        cond = _build_kv_condition(key, value)
        if cond:
            conditions.append(cond)
    return conditions


def _convert_condition(cond: dict[str, Any]) -> dict[str, Any] | None:
    """转换 $and/$or 内部的子条件（递归）。"""
    if not cond:
        return None
    if "$and" in cond:
        sub = [_convert_condition(c) for c in cond["$and"]]
        sub = [c for c in sub if c]
        return {"must": sub} if sub else None
    if "$or" in cond:
        sub = [_convert_condition(c) for c in cond["$or"]]
        sub = [c for c in sub if c]
        return {"should": sub} if sub else None
    kv_conditions = _convert_top_level(cond)
    if len(kv_conditions) == 1:
        return kv_conditions[0]
    return {"must": kv_conditions} if kv_conditions else None


def _build_kv_condition(key: str, value: Any) -> dict[str, Any] | None:
    """由 Chroma 键值对构建单个 Qdrant 条件。

    Qdrant 把文档 metadata 存为嵌套 payload：{"metadata": {...}}；
    扁平键如 "section" 必须转为 "metadata.section" 过滤才生效。
    """
    qdrant_key = f"metadata.{key}" if "." not in key else key
    if value is None:
        return {"key": qdrant_key, "match": {"value": None}}
    if isinstance(value, dict):
        if "$in" in value:
            items = value["$in"]
            if not isinstance(items, list) or not items:
                return None
            return {"key": qdrant_key, "match": {"any": items}}
        if "$ne" in value:
            return {"key": qdrant_key, "match": {"except": [value["$ne"]]}}
        return {"key": qdrant_key, "match": {"value": str(value)}}
    return {"key": qdrant_key, "match": {"value": value}}
