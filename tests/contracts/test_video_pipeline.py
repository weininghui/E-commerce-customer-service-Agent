"""契约测试：视频知识管线（Phase 3 视频版）——抽帧 / 视觉理解 / 入库 / 任务分发。"""

from __future__ import annotations

from unittest.mock import patch


def test_extract_frames_degrades_without_ffmpeg():
    """无 ffmpeg 时抽帧返回空列表（调用方降级，不抛异常）。"""
    from agent_base.multimodal import video

    with patch("agent_base.multimodal.video.has_ffmpeg", return_value=False):
        assert video.extract_frames("/no/such/video.mp4") == []


def test_analyze_video_mock_without_key(monkeypatch):
    """无 ARK 密钥时视频解析降级为 mock（仅登记元数据，不抽帧不调视觉）。"""
    from agent_base.multimodal import video

    monkeypatch.delenv("ARK_API_KEY", raising=False)
    result = video.analyze_video("/no/such/video.mp4")
    assert result["ok"] is True
    assert result["engine"] == "mock"
    assert result["scenes_text"] == ""
    assert "ARK_API_KEY" in result["note"]


def test_analyze_video_ark_with_frames(monkeypatch, tmp_path):
    """ARK 密钥就绪且抽帧成功时，逐帧视觉理解组装描述与分镜文本。"""
    from agent_base.multimodal import video

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    # 造两个假帧文件（_frame_to_data_uri 会真实读取）
    f1 = tmp_path / "frame_000.png"
    f2 = tmp_path / "frame_001.png"
    f1.write_bytes(b"fake-png-1")
    f2.write_bytes(b"fake-png-2")

    def _fake_vision(target: str, cfg, idx: int):
        return {"product_name": "白色纯棉T恤", "text": f"第{idx}帧：面料标签 100%棉", "tags": ["T恤", "纯棉"]}

    with patch("agent_base.multimodal.video.extract_frames", return_value=[str(f1), str(f2)]), \
         patch("agent_base.multimodal.video._call_vision_frame", side_effect=_fake_vision):
        result = video.analyze_video("/no/such/video.mp4")
    assert result["engine"] == "ark"
    assert result["frame_count"] == 2
    assert "白色纯棉T恤" in result["description"]
    assert "面料标签" in result["scenes_text"]
    assert "T恤" in result["description"]


def test_index_video_document_ingests_chunks():
    """视频文本入库：概述 + 分镜 chunk，metadata 带 video_urls/poster/duration。"""
    from agent_base.multimodal import video

    row = {
        "id": 3,
        "url": "/media/uploads/a.mp4",
        "poster_url": "/media/uploads/a.poster.jpg",
        "product_id": "P001",
        "source_type": "video",
        "video_urls": ["/media/uploads/a.mp4"],
    }
    with patch("agent_base.storage.documents.ingest_document_from_chunks", return_value={"status": "ingested"}) as mock_ingest, \
         patch("agent_base.api.main.get_runtime", return_value={"vector_store": object()}):
        ok = video.index_video_document(row, "白色纯棉T恤 T恤 纯棉", "[0s-15s] 面料标签 100%棉\n[15s-30s] 水洗不变形", 30)
    assert ok is True
    kwargs = mock_ingest.call_args.kwargs
    assert kwargs["doc_id"] == "video-3"
    assert kwargs["skip_tag_check"] is True
    first = kwargs["chunks"][0]
    assert first["metadata"]["video_urls"] == ["/media/uploads/a.mp4"]
    assert first["metadata"]["poster_url"] == "/media/uploads/a.poster.jpg"
    assert first["metadata"]["duration_sec"] == 30
    assert first["metadata"]["source_type"] == "video"
    assert "纯棉" in first["text"]
    # 分镜超过单 chunk 上限时拆分多个 chunk
    assert len(kwargs["chunks"]) >= 2


def test_index_video_document_empty_returns_false():
    """无有效文本时不入库。"""
    from agent_base.multimodal import video

    row = {"id": 4, "url": "/media/uploads/b.mp4"}
    with patch("agent_base.storage.documents.ingest_document_from_chunks") as mock_ingest:
        ok = video.index_video_document(row, "", "  ", 0)
    assert ok is False
    mock_ingest.assert_not_called()


def test_media_parse_video_task_handler():
    """media_parse_video 任务：解析 → 更新记录 → 入库。"""
    from agent_base.async_tasks import execute_task

    row = {
        "id": 5,
        "url": "/media/uploads/x.mp4",
        "description": "",
        "poster_url": "",
        "product_id": "",
        "source_type": "video",
        "mime_type": "video/mp4",
        "parse_type": "video",
    }
    with patch("agent_base.storage.pg.media_document_get", return_value=row), \
         patch("agent_base.storage.pg.media_document_update_parse", return_value=True) as mock_update, \
         patch("agent_base.media_library.media_file_path", return_value=None), \
         patch("agent_base.multimodal.video.analyze_video", return_value={
             "ok": True, "engine": "ark", "description": "白色纯棉T恤", "scenes_text": "[0s-30s] 100%棉",
             "frame_count": 2, "duration_sec": 30, "note": "",
         }), \
         patch("agent_base.multimodal.video.index_video_document", return_value=True) as mock_index:
        result = execute_task({"task_type": "media_parse_video", "payload": {"media_id": 5}})
    assert result["ok"] is True
    assert result["result"]["engine"] == "ark"
    assert result["result"]["indexed"] is True
    mock_update.assert_called_once()
    kwargs = mock_update.call_args
    assert kwargs.args[0] == 5
    assert kwargs.kwargs["duration_sec"] == 30
    assert kwargs.kwargs["parse_type"] == "video"
    assert mock_index.call_count == 1


def test_media_parse_video_skips_index_on_empty_scenes():
    """视频无有效分镜文本时不入库（mock 降级场景）。"""
    from agent_base.async_tasks import execute_task

    row = {"id": 6, "url": "/media/uploads/y.mp4", "description": "人工描述", "poster_url": "",
           "product_id": "", "source_type": "video", "mime_type": "video/mp4", "parse_type": "video"}
    with patch("agent_base.storage.pg.media_document_get", return_value=row), \
         patch("agent_base.storage.pg.media_document_update_parse", return_value=True), \
         patch("agent_base.media_library.media_file_path", return_value=None), \
         patch("agent_base.multimodal.video.analyze_video", return_value={
             "ok": True, "engine": "mock", "description": "", "scenes_text": "",
             "frame_count": 0, "duration_sec": 0, "note": "ffmpeg 不可用",
         }), \
         patch("agent_base.multimodal.video.index_video_document") as mock_index:
        result = execute_task({"task_type": "media_parse_video", "payload": {"media_id": 6}})
    assert result["result"]["indexed"] is False
    mock_index.assert_not_called()


def test_doc_media_extracts_videos_from_sources():
    """检索命中的视频源（video_urls）转成 media 项（media_type=video，带封面/时长）。"""
    from agent_base.chains.streaming import _doc_images_from_sources

    sources = [
        {
            "doc_name": "视频A",
            "video_urls": ["/media/uploads/a.mp4"],
            "poster_url": "/media/uploads/a.poster.jpg",
            "duration_sec": 30,
        },
        {"doc_name": "图B", "image_urls": ["/media/uploads/b.png"]},
    ]
    items = _doc_images_from_sources(sources)
    video_items = [i for i in items if i["media_type"] == "video"]
    image_items = [i for i in items if i["media_type"] == "image"]
    assert len(video_items) == 1
    assert video_items[0]["url"] == "/media/uploads/a.mp4"
    assert video_items[0]["poster"] == "/media/uploads/a.poster.jpg"
    assert video_items[0]["duration_sec"] == 30
    assert video_items[0]["title"] == "知识库视频：视频A"
    assert len(image_items) == 1
