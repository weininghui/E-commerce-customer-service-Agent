"""契约测试：聊天媒体护栏——视频数量限制 + 封面卡片元数据。"""

from __future__ import annotations

from agent_base.chains.streaming import _combined_media_items, _doc_images_from_sources


def test_doc_media_extracts_videos_and_images():
    sources = [
        {"doc_name": "视频A", "video_urls": ["/v/a.mp4"], "poster_url": "/v/a.jpg", "duration_sec": 30},
        {"doc_name": "视频B", "video_urls": ["/v/b.mp4"]},
        {"doc_name": "图C", "image_urls": ["/i/c.png"]},
    ]
    items = _doc_images_from_sources(sources)
    assert sum(1 for i in items if i["media_type"] == "video") == 2
    assert sum(1 for i in items if i["media_type"] == "image") == 1


def test_combined_media_caps_videos_at_two():
    """8 个视频源 → 合并后视频 ≤2、总量 ≤8（防止对话窗口被视频撑长）。"""
    sources = [{"doc_name": f"v{i}", "video_urls": [f"/v/{i}.mp4"]} for i in range(8)]
    combined = _combined_media_items([], sources)
    videos = [m for m in combined if m["media_type"] == "video"]
    assert len(videos) <= 2
    assert len(combined) <= 8


def test_combined_media_prefers_images_over_extra_videos():
    """总量封顶时图片优先，多余视频被丢弃。"""
    sources = [
        {"doc_name": "图", "image_urls": [f"/i/{i}.png"]} for i in range(7)
    ] + [{"doc_name": "视频", "video_urls": [f"/v/{i}.mp4"]} for i in range(4)]
    combined = _combined_media_items([], sources)
    videos = [m for m in combined if m["media_type"] == "video"]
    images = [m for m in combined if m["media_type"] == "image"]
    assert len(combined) <= 8
    assert len(videos) <= 2
    assert len(images) >= 6
