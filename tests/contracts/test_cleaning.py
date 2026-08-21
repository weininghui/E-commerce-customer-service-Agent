"""契约测试：文件清洗工作台（两段式入库第一段）。"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent_base.cleaning import (
    handle_clean_push,
    handle_clean_upload,
    parse_clean_text,
    validate_clean_upload,
)


def test_validate_clean_upload():
    with pytest.raises(ValueError, match="不支持"):
        validate_clean_upload("x.exe", b"MZ")
    with pytest.raises(ValueError, match="为空"):
        validate_clean_upload("a.md", b"")
    assert validate_clean_upload("a.docx", b"x") == ".docx"


def test_parse_clean_text_md_direct():
    text, engine = parse_clean_text("说明.md", "# 标题\n内容".encode("utf-8"))
    assert "标题" in text and engine == "direct"
    text2, _ = parse_clean_text("说明.txt", "中文内容".encode("gb18030"))
    assert "中文内容" in text2


def test_parse_clean_text_docx_via_mineru():
    with patch("agent_base.ingest.mineru_parser.parse_document", return_value={"text": "清洗后文本", "engine": "mineru", "ok": True}):
        text, engine = parse_clean_text("a.docx", b"fake")
    assert text == "清洗后文本" and engine == "mineru"


def test_handle_clean_upload_creates_draft_only():
    with patch("agent_base.storage.pg.clean_draft_create", return_value=11):
        payload = handle_clean_upload("说明.md", "# 标题\n正文".encode("utf-8"))
    assert payload["ok"] is True
    assert payload["id"] == 11
    assert payload["engine"] == "direct"
    assert "正文" in payload["text"]


def test_handle_clean_upload_empty_text_raises():
    with patch("agent_base.ingest.mineru_parser.parse_document", return_value={"text": "", "engine": "mock", "ok": True}), \
         patch("agent_base.storage.pg.clean_draft_create") as mock_create:
        with pytest.raises(ValueError, match="可清洗文本"):
            handle_clean_upload("a.pdf", b"%PDF fake")
    mock_create.assert_not_called()


def test_handle_clean_push_uses_cleaned_text():
    draft = {
        "id": 3,
        "original_name": "说明.md",
        "raw_text": "原文",
        "cleaned_text": "人工清洗后的文本",
        "status": "pending",
    }
    with patch("agent_base.storage.pg.clean_draft_get", return_value=draft), \
         patch("agent_base.storage.pg.clean_draft_set_status", return_value=True) as mock_status, \
         patch("agent_base.storage.staging.stage_uploaded_document", return_value={"doc_id": "d1", "status": "staged"}) as mock_stage:
        payload = handle_clean_push(3, category="商品说明")
    assert payload["clean_id"] == 3
    assert payload["doc_id"] == "d1"
    mock_stage.assert_called_once()
    assert "人工清洗后的文本" == mock_stage.call_args.kwargs["content"]
    assert "商品说明" == mock_stage.call_args.kwargs["category"]
    mock_status.assert_called_once_with(3, "pushed")


def test_handle_clean_push_missing_draft_raises():
    with patch("agent_base.storage.pg.clean_draft_get", return_value=None):
        with pytest.raises(ValueError, match="不存在"):
            handle_clean_push(999)


def test_pg_clean_functions_swallow_db_errors():
    from agent_base.storage.pg import (
        clean_draft_create,
        clean_draft_delete,
        clean_draft_get,
        clean_draft_list,
        clean_draft_set_status,
        clean_draft_update,
    )

    with patch("agent_base.storage.pg._conn", side_effect=Exception("no db")):
        assert clean_draft_create(original_name="a.md", raw_text="t") == 0
        assert clean_draft_get(1) is None
        assert clean_draft_list() == []
        assert clean_draft_update(1, "t") is False
        assert clean_draft_set_status(1, "pushed") is False
        assert clean_draft_delete(1) is False


def test_polish_clean_text_with_mocked_llm():
    from agent_base.cleaning import polish_clean_text

    class _FakeResp:
        content = "# 标题\n\n## 使用方法\n- 每日涂抹"

    class _FakeModel:
        def invoke(self, *a, **k):
            return _FakeResp()

    with patch("agent_base.llms.build_chat_model", return_value=_FakeModel()) as mock_build:
        out = polish_clean_text("每日涂抹")
    assert "## 使用方法" in out
    mock_build.assert_called_once()


def test_polish_clean_text_no_llm_raises():
    from agent_base.cleaning import polish_clean_text

    with patch("agent_base.llms.build_chat_model", return_value=None):
        with pytest.raises(RuntimeError, match="LLM"):
            polish_clean_text("x")


def test_handle_clean_polish_writes_back():
    from agent_base.cleaning import handle_clean_polish

    draft = {"id": 5, "raw_text": "原始文本", "cleaned_text": ""}
    with patch("agent_base.storage.pg.clean_draft_get", return_value=draft), \
         patch("agent_base.cleaning.polish_clean_text", return_value="# 整理后"), \
         patch("agent_base.storage.pg.clean_draft_update", return_value=True) as mock_update:
        payload = handle_clean_polish(5)
    assert payload["polished"] == "# 整理后"
    mock_update.assert_called_once_with(5, "# 整理后")


def test_handle_clean_polish_missing_draft():
    from agent_base.cleaning import handle_clean_polish

    with patch("agent_base.storage.pg.clean_draft_get", return_value=None):
        with pytest.raises(ValueError, match="不存在"):
            handle_clean_polish(999)