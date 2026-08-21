"""契约测试：图片知识库（Phase 2）——校验 / 落盘 / 入库 / 解析任务。"""

from __future__ import annotations

import base64
from unittest.mock import patch

import pytest

from agent_base.media_library import (
    MAX_MEDIA_BYTES,
    _safe_stored_name,
    handle_media_delete,
    handle_media_upload,
    media_file_path,
    validate_media_upload,
)

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def test_validate_rejects_bad_type():
    with pytest.raises(ValueError, match="不支持"):
        validate_media_upload("x.exe", b"MZ")
    with pytest.raises(ValueError, match="不支持"):
        validate_media_upload("x.svg", b"<svg/>")


def test_validate_rejects_empty_and_oversize():
    with pytest.raises(ValueError, match="为空"):
        validate_media_upload("a.png", b"")
    with pytest.raises(ValueError, match="8MB"):
        validate_media_upload("a.png", b"x" * (MAX_MEDIA_BYTES + 1))


def test_validate_ok_and_safe_name():
    name = validate_media_upload("商品主图.PNG", PNG_1PX)
    assert name.endswith(".png")
    assert "商品主图" in name  # 中文文件名保留（不再退化成下划线）
    # 路径穿越被中和
    evil = _safe_stored_name("../../etc/passwd.png")
    assert ".." not in evil and "/" not in evil


def test_upload_flow_with_mocked_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BASE_MEDIA_DIR", str(tmp_path / "media" / "uploads"))
    with patch("agent_base.storage.pg.media_document_create", return_value=7):
        payload = handle_media_upload("主图.png", PNG_1PX, description="测试图")
    assert payload["ok"] is True and payload["id"] == 7
    assert payload["url"].startswith("/media/uploads/")
    path = media_file_path(payload["url"])
    assert path is not None and path.exists()


def test_delete_flow_removes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BASE_MEDIA_DIR", str(tmp_path / "media" / "uploads"))
    with patch("agent_base.storage.pg.media_document_create", return_value=9):
        payload = handle_media_upload("a.png", PNG_1PX)
    path = media_file_path(payload["url"])
    assert path and path.exists()
    with patch("agent_base.storage.pg.media_document_delete", return_value={"id": 9, "url": payload["url"]}):
        result = handle_media_delete(9)
    assert result["ok"] is True
    assert not path.exists()


def test_delete_missing_row_returns_not_ok():
    with patch("agent_base.storage.pg.media_document_delete", return_value=None):
        result = handle_media_delete(999)
    assert result["ok"] is False


def test_pg_media_functions_swallow_db_errors():
    from agent_base.storage.pg import (
        media_document_bind,
        media_document_create,
        media_document_delete,
        media_document_get,
        media_document_list,
        media_document_update_parse,
    )

    with patch("agent_base.storage.pg._conn", side_effect=Exception("no db")):
        assert media_document_create(original_name="a.png", url="/media/uploads/a.png") == 0
        assert media_document_list() == []
        assert media_document_get(1) is None
        assert media_document_bind(1, "P001") is False
        assert media_document_update_parse(1, description="d") is False
        assert media_document_delete(1) is None


def test_analyze_media_image_mock_without_key(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    from agent_base.multimodal import analyze_media_image

    result = analyze_media_image(image_path="/no/such/file.png")
    assert result["ok"] is True and result["engine"] == "mock"
    assert result["ocr_text"] == ""


def test_analyze_media_image_mineru_fallback(monkeypatch, tmp_path):
    """视觉模型不可用时，图片走 MinerU OCR 兜底（引擎=mineru，文字入库）。"""
    from agent_base.multimodal import analyze_media_image

    monkeypatch.delenv("ARK_API_KEY", raising=False)
    img = tmp_path / "t.png"
    img.write_bytes(b"fake-png-bytes")
    with patch("agent_base.ingest.mineru_parser.parse_document", return_value={"text": "神经酰胺修护面霜 屏障修护", "engine": "mineru", "ok": True}):
        result = analyze_media_image(image_path=str(img))
    assert result["engine"] == "mineru"
    assert result["ocr_text"] == "神经酰胺修护面霜 屏障修护"


def test_analyze_media_image_mock_when_mineru_mock(monkeypatch, tmp_path):
    """MinerU 也无密钥时保持 mock 空解析（不把二进制当文本）。"""
    from agent_base.multimodal import analyze_media_image

    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("MINERU_API_KEY", raising=False)
    img = tmp_path / "t.png"
    img.write_bytes(b"fake-png-bytes")
    with patch("agent_base.ingest.mineru_parser.parse_document", return_value={"text": b"\x89PNG".decode("utf-8", "ignore"), "engine": "mock", "ok": True}):
        result = analyze_media_image(image_path=str(img))
    assert result["engine"] == "mock"
    assert result["ocr_text"] == ""


def test_analyze_media_image_ark_wins_no_mineru(monkeypatch, tmp_path):
    """ARK 真实出结果时不触发 MinerU（避免重复调用）。"""
    from agent_base.multimodal import analyze_media_image

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    img = tmp_path / "t.png"
    img.write_bytes(b"fake-png-bytes")
    with patch("agent_base.multimodal._call_vision", return_value={"ok": True, "engine": "ark", "description": "d", "ocr_text": "t"}), \
         patch("agent_base.ingest.mineru_parser.parse_document") as mock_mineru:
        result = analyze_media_image(image_path=str(img))
    assert result["engine"] == "ark"
    mock_mineru.assert_not_called()


def test_parse_vision_content_json_and_fallback():
    from agent_base.multimodal import _parse_vision_content

    r1 = _parse_vision_content(
        '{"product_name":"神经酰胺修护面霜","text":"屏障修护","tags":["面霜","干敏"]}'
    )
    assert r1["description"] == "神经酰胺修护面霜 面霜 干敏"
    assert r1["ocr_text"] == "屏障修护"
    r2 = _parse_vision_content("不是JSON的纯文本输出")
    assert r2["ocr_text"] == "不是JSON的纯文本输出"


def test_media_parse_task_handler():
    from agent_base.async_tasks import execute_task

    with patch("agent_base.storage.pg.media_document_get", return_value={"id": 5, "url": "/media/uploads/x.png", "description": ""}), \
         patch("agent_base.storage.pg.media_document_update_parse", return_value=True) as mock_update, \
         patch("agent_base.multimodal.analyze_media_image", return_value={"ok": True, "engine": "mock", "description": "d", "ocr_text": "t"}), \
         patch("agent_base.media_library.media_file_path", return_value=None), \
         patch("agent_base.async_tasks._index_media_document", return_value=True) as mock_index:
        result = execute_task({"task_type": "media_parse", "payload": {"media_id": 5}})
    assert result["ok"] is True
    assert result["result"]["engine"] == "mock"
    assert result["result"]["indexed"] is True
    mock_update.assert_called_once_with(5, description="d", ocr_text="t")
    mock_index.assert_called_once()


def test_media_parse_skips_index_on_empty_text():
    from agent_base.async_tasks import execute_task

    with patch("agent_base.storage.pg.media_document_get", return_value={"id": 6, "url": "/media/uploads/x.png", "description": "人工描述"}), \
         patch("agent_base.storage.pg.media_document_update_parse", return_value=True) as mock_update, \
         patch("agent_base.multimodal.analyze_media_image", return_value={"ok": True, "engine": "mock", "description": "", "ocr_text": ""}), \
         patch("agent_base.media_library.media_file_path", return_value=None), \
         patch("agent_base.async_tasks._index_media_document") as mock_index:
        result = execute_task({"task_type": "media_parse", "payload": {"media_id": 6}})
    assert result["result"]["indexed"] is False
    mock_index.assert_not_called()
    # mock 空解析不覆盖人工描述
    mock_update.assert_called_once_with(6, description="人工描述", ocr_text="")


def test_index_media_document_ingests_with_metadata():
    from agent_base.async_tasks import _index_media_document

    row = {"id": 7, "url": "/media/uploads/a.png", "product_id": "P002", "source_type": "upload"}
    with patch("agent_base.storage.documents.ingest_document_from_chunks", return_value={"status": "ingested"}) as mock_ingest, \
         patch("agent_base.api.main.get_runtime", return_value={"vector_store": object()}):
        ok = _index_media_document(row, "神经酰胺修护面霜", "屏障修护 干敏友好")
    assert ok is True
    kwargs = mock_ingest.call_args.kwargs
    assert kwargs["doc_id"] == "media-7"
    assert kwargs["skip_tag_check"] is True
    chunk = kwargs["chunks"][0]
    assert chunk["metadata"]["media_id"] == 7
    assert chunk["metadata"]["image_urls"] == ["/media/uploads/a.png"]
    assert chunk["metadata"]["product_id"] == "P002"
    assert "屏障修护" in chunk["text"]


def test_index_media_document_empty_text_returns_false():
    from agent_base.async_tasks import _index_media_document

    row = {"id": 8, "url": "/media/uploads/a.png"}
    with patch("agent_base.storage.documents.ingest_document_from_chunks") as mock_ingest:
        ok = _index_media_document(row, "", "  ")
    assert ok is False
    mock_ingest.assert_not_called()


def test_doc_images_from_sources():
    from agent_base.chains.streaming import _doc_images_from_sources

    sources = [
        {"doc_name": "图A", "image_urls": ["/media/uploads/a.png"]},
        {"doc_name": "图B", "image_urls": "/media/uploads/b.png"},
        {"doc_name": "图A2", "image_urls": ["/media/uploads/a.png"]},
        {"doc_name": "无图", "image_urls": []},
    ]
    items = _doc_images_from_sources(sources)
    assert [x["url"] for x in items] == ["/media/uploads/a.png", "/media/uploads/b.png"]
    assert items[0]["title"] == "知识库图片：图A"
    assert items[0]["media_type"] == "image"
    assert _doc_images_from_sources([]) == []


def test_combined_media_items_dedup_and_cap():
    from agent_base.chains.streaming import _combined_media_items

    products = [{"url": "/p/1", "media_type": "image"}, {"url": "/p/2", "media_type": "image"}]
    sources = [{"doc_name": "d", "image_urls": ["/p/1", "/p/3"]}]
    combined = _combined_media_items(products, sources)
    assert [x["url"] for x in combined] == ["/p/1", "/p/2", "/p/3"]
    many = [{"url": f"/p/{i}", "media_type": "image"} for i in range(20)]
    capped = _combined_media_items(many, None)
    assert len(capped) == 8