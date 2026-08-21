"""契约：P2 多源知识库（评价/搭配/案例）检索 + 商品名解析。"""

from __future__ import annotations

from agent_base.retrieval.multi_source import (
    _resolve_product_ids,
    search_cases,
    search_combos,
    search_reviews,
)


def test_resolve_product_ids_alias_and_token():
    """简称解析：别名/分词 token 匹配 catalog ID。"""
    assert _resolve_product_ids("玻尿酸精华") == ["P001"]
    assert _resolve_product_ids("烟酰胺精华") == ["P005"]
    assert _resolve_product_ids("P001") == ["P001"]


def test_search_reviews_returns_source_type():
    """评价检索：中文商品名命中，返回带 source_type 的结构。"""
    rows = search_reviews("玻尿酸精华")
    assert rows
    assert rows[0]["source_type"] == "reviews"
    assert "content" in rows[0] and "score" in rows[0]


def test_search_combos_by_scenario():
    """搭配检索：按场景（干皮/油皮等）命中。"""
    rows = search_combos("干皮")
    assert rows
    assert rows[0]["source_type"] == "combos"
    assert rows[0].get("product_ids")


def test_search_cases_with_skin_type():
    """案例检索：商品 + 肤质过滤（无匹配时返回空而非报错）。"""
    rows = search_cases("玻尿酸", skin_type="干皮")
    assert rows
    assert rows[0]["source_type"] == "cases"
    # 无匹配场景优雅降级为空
    assert search_cases("烟酰胺精华", skin_type="敏感肌") == []
