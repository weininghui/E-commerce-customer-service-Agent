"""契约测试：媒体按需门控（先讲解，用户要看图/视频才返回）。"""

from __future__ import annotations

from agent_base.chains.streaming import _media_requested


def test_media_requested_true_for_visual_requests():
    assert _media_requested("看看这件商品的图片", "media_request") is True
    assert _media_requested("有视频吗", "") is True
    assert _media_requested("看看实物", "product_query") is True
    assert _media_requested("发个视频看看", "") is True
    assert _media_requested("上身效果怎么样", "fashion_query") is True


def test_media_requested_false_for_normal_questions():
    assert _media_requested("帮我推荐衣服", "") is False
    assert _media_requested("看看有什么", "") is False
    assert _media_requested("这件衣服多少钱", "price_inquiry") is False
    assert _media_requested("这款精华适合油皮吗", "product_query") is False
