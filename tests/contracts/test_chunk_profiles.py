"""切分参数覆盖层契约测试（需本地 PG；无 PG 时自动跳过）。"""
from __future__ import annotations

import pytest

DT = "test_override_type"


def _pg_available() -> bool:
    try:
        from agent_base.storage import pg as _pg

        _pg.init_db()
        with _pg._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pg_available(), reason="本地 PG 未启动，跳过")


def test_override_changes_profile_and_split():
    from agent_base.ingest.splitter import get_profile, split_markdown_by_type
    from agent_base.storage.pg import chunk_override_delete, chunk_override_upsert

    chunk_override_delete(DT)
    # chunk_size 必须小于正文长度，递归切分器才会真正启用自定义分隔符
    # （section 内正文 ≤ chunk_size 时整段保留，分隔符不生效——这是主链路语义）
    assert chunk_override_upsert(DT, 8, 0, ["\n\n", "。", ""], "tester")
    p = get_profile(DT)
    assert p["chunk_size"] == 8
    assert p["chunk_overlap"] == 0
    assert p["separators"] == ["\n\n", "。", ""]
    docs = split_markdown_by_type(DT, "# A\n\n句子一。\n\n句子二。")
    # 自定义分隔符生效：按 \n\n 段落拆成 ≥2 块（而非整段保留）
    assert len(docs) >= 2
    chunk_override_delete(DT)


def test_delete_restores_default():
    from agent_base.ingest.splitter import get_profile
    from agent_base.storage.pg import chunk_override_delete, chunk_override_upsert

    chunk_override_upsert(DT, 200, 10, ["\n\n"], "tester")
    chunk_override_delete(DT)
    p = get_profile(DT)
    assert p["chunk_size"] == 900  # 未知类型回退 DEFAULT_PROFILE
    assert p["separators"] == ["\n\n", "\n", "。", "；", "，", ""]
