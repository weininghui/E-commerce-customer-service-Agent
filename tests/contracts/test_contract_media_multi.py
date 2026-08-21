"""契约测试：Phase 1 多商品推荐带图（SSE media 批量解析）。"""

from __future__ import annotations

from unittest.mock import patch

from agent_base.chains.streaming import _extract_all_product_names, _media_for_stream


def _fake_media(ids, limit=8):
    return [
        {
            "id": i + 1,
            "product_id": pid,
            "media_type": "image",
            "url": f"/media/products/{pid.lower()}.svg",
            "title": "商品主图",
        }
        for i, pid in enumerate(ids)
    ]


def _fake_catalog():
    return {
        "products": {
            "P001": {"name": "玻尿酸保湿精华液", "category": "精华"},
            "P005": {"name": "烟酰胺焕亮精华", "category": "精华"},
            "P008": {"name": "积雪草舒缓精华", "category": "精华"},
            "P009": {"name": "水杨酸净痘精华", "category": "精华"},
            "P002": {"name": "神经酰胺修护面霜", "category": "面霜"},
        }
    }


def test_single_product_not_expanded_by_category():
    """单商品守卫：命中具体商品时，不得按类目展开（「精华」是「精华液」的子串）。"""
    constraints = {
        "catalog_resolution": {
            "matched_products": [{"id": "P001"}],
            "matched_categories": ["精华"],
        }
    }
    with patch("agent_base.storage.pg.media_for_product_ids", side_effect=_fake_media) as mock_media, \
         patch("agent_base.retrieval.retrieval_policy._load_catalog", return_value=_fake_catalog()):
        items = _media_for_stream(constraints, None, "玻尿酸保湿精华液适合油皮吗")
    assert [m["product_id"] for m in items] == ["P001"]
    assert mock_media.call_count == 1
    assert mock_media.call_args.kwargs["limit"] == 8


def test_category_expands_to_multiple_products():
    """类目推荐：「推荐几款精华」→ 精华类全部商品一次出多图。"""
    constraints = {
        "catalog_resolution": {
            "matched_products": [],
            "matched_categories": ["精华"],
            "category": "精华",
        }
    }
    with patch("agent_base.storage.pg.media_for_product_ids", side_effect=_fake_media), \
         patch("agent_base.retrieval.retrieval_policy._load_catalog", return_value=_fake_catalog()):
        items = _media_for_stream(constraints, None, "推荐几款精华")
    pids = [m["product_id"] for m in items]
    assert pids == ["P001", "P005", "P008", "P009"]
    # 标题被补成商品名（多图时可区分商品）
    assert items[0]["title"] == "玻尿酸保湿精华液"


def test_history_product_names_fed_as_candidates():
    """多轮指代：历史商品名作为候选（本问题无商品/类目命中时）。"""
    constraints = {"catalog_resolution": {"matched_products": [], "matched_categories": []}}
    with patch("agent_base.storage.pg.media_for_product_ids", side_effect=_fake_media) as mock_media, \
         patch("agent_base.retrieval.retrieval_policy._load_catalog", return_value=_fake_catalog()), \
         patch("agent_base.retrieval.multi_source._resolve_product_ids", return_value=["P002"]):
        items = _media_for_stream(constraints, None, "那这个呢", history_products=["神经酰胺修护面霜"])
    assert [m["product_id"] for m in items] == ["P002"]
    assert mock_media.call_args.kwargs["limit"] == 8


def test_no_candidates_returns_empty():
    """无任何候选（无商品/类目/历史/P###）→ 不下发 media。"""
    constraints = {"catalog_resolution": {}}
    with patch("agent_base.storage.pg.media_for_product_ids") as mock_media, \
         patch("agent_base.retrieval.retrieval_policy._load_catalog", return_value=_fake_catalog()):
        items = _media_for_stream(constraints, None, "今天天气怎么样")
    assert items == []
    mock_media.assert_not_called()


def test_pid_regex_fallback():
    """问题里直接出现 P### 编号的兜底。"""
    constraints = {"catalog_resolution": {}}
    with patch("agent_base.storage.pg.media_for_product_ids", side_effect=_fake_media), \
         patch("agent_base.retrieval.retrieval_policy._load_catalog", return_value=_fake_catalog()):
        items = _media_for_stream(constraints, None, "P013 会透吗")
    assert [m["product_id"] for m in items] == ["P013"]


def test_extract_all_product_names_from_history():
    """历史多商品名全部提取（按出现顺序去重）。"""
    history = [
        {"role": "user", "content": "玻尿酸保湿精华液适合油皮吗"},
        {"role": "assistant", "content": "适合的"},
        {"role": "user", "content": "那烟酰胺焕亮精华呢"},
    ]
    with patch("agent_base.retrieval.retrieval_policy._load_catalog", return_value=_fake_catalog()), \
         patch("agent_base.retrieval.enrich.load_aliases", return_value={}), \
         patch("agent_base.retrieval.enrich.expand_aliases", side_effect=lambda text, aliases: text):
        names = _extract_all_product_names(history)
    assert names == ["玻尿酸保湿精华液", "烟酰胺焕亮精华"]


def test_media_for_stream_swallows_errors():
    """解析异常静默返回空（不阻塞主链路）。"""
    with patch("agent_base.retrieval.retrieval_policy._load_catalog", side_effect=Exception("db down")):
        items = _media_for_stream({"catalog_resolution": {"matched_categories": ["精华"]}}, None, "推荐几款精华")
    assert items == []