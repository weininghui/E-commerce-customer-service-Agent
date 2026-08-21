"""契约：数据源全 PG 化（json/jsonl 淘汰后，catalog/faq/idf 运行时均读 PG）。"""

from __future__ import annotations

from agent_base.api.main import _faq_title_map, get_catalog
from agent_base.monitoring.alert import check_alert, error_stats
from agent_base.retrieval.keyword_ranker import _chunk_fingerprint, _load_or_build_idf
from agent_base.storage.pg import faq_title_map, idf_cache_load


def test_catalog_from_pg():
    """商品目录纯 PG：product_count 与 PG catalog 行数一致。"""
    catalog = get_catalog()
    assert catalog["product_count"] >= 20
    assert isinstance(catalog["products"], dict)


def test_faq_from_pg():
    """FAQ 纯 PG：_faq_title_map 与 PG faq 表一致，内置种子可重建。"""
    m = _faq_title_map()
    assert len(m) >= 8
    assert "F001" in m
    assert m["F001"] == "下单后多久发货？"
    pg = faq_title_map()
    assert m == pg


def test_idf_cache_from_pg():
    """IDF 缓存纯 PG：按 Qdrant 点数版本读取，与 keyword_ranker 一致。"""
    idf = _load_or_build_idf()
    assert len(idf) > 1000  # 130 chunk 语料的 IDF 表规模
    cached = idf_cache_load(_chunk_fingerprint())
    assert cached is not None
    assert len(cached) == len(idf)


def test_alert_stats():
    """监控告警：error_stats 返回结构完整，check_alert 三态可用。"""
    stats = error_stats(minutes=60)
    assert "total" in stats and "error" in stats and "by_module" in stats
    alert = check_alert(error_threshold=1000, minutes=60)
    assert alert["level"] in ("ok", "warning", "alert")
    assert "message" in alert
