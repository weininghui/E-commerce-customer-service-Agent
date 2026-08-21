"""契约测试：多模态商品图（文生图 + 图生图 mock 降级）。

无 ARK_API_KEY 时全部走 mock 分支（不触网），验证：
1. 文生图返回 mock 占位图；
2. 图生图有参考图 → 原样返回参考图（engine=mock + note 说明）；
3. 图生图无参考图 → 退化为文生图占位；
4. 输入归一化：裸 base64 自动补 data URI 前缀。
"""

from __future__ import annotations

import pytest

from agent_base.multimodal import (
    _normalize_image_input,
    edit_product_image,
    generate_product_image,
)

FAKE_PNG_B64 = "iVBORw0KGgo="
FAKE_DATA_URI = "data:image/png;base64," + FAKE_PNG_B64


@pytest.fixture(autouse=True)
def _no_ark_key(monkeypatch):
    """强制无密钥：mock 分支确定性（不触网）。"""
    monkeypatch.setenv("ARK_API_KEY", "")


def test_text_to_image_mock():
    """文生图 mock：无密钥返回 SVG 占位。"""
    result = generate_product_image("氨基酸洁面乳 白管包装")
    assert result["ok"] is True
    assert result["engine"] == "mock"
    assert str(result["image"]).startswith("data:image/svg+xml;base64,")


def test_image_to_image_mock_returns_input():
    """图生图 mock：有参考图 → 原样返回参考图，附 note 说明。"""
    result = edit_product_image("包装换蓝色", image=FAKE_DATA_URI)
    assert result["ok"] is True
    assert result["engine"] == "mock"
    assert result["image"] == FAKE_DATA_URI
    assert "note" in result


def test_image_to_image_mock_without_image_falls_back():
    """图生图 mock：无参考图 → 退化为文生图占位。"""
    result = edit_product_image("包装换蓝色")
    assert result["ok"] is True
    assert result["engine"] == "mock"
    assert str(result["image"]).startswith("data:image/svg+xml;base64,")


def test_normalize_image_input():
    """输入归一化：data URI / URL 直通，裸 base64 补前缀。"""
    assert _normalize_image_input(FAKE_DATA_URI, None) == FAKE_DATA_URI
    assert _normalize_image_input("https://example.com/a.png", None) == "https://example.com/a.png"
    assert _normalize_image_input(FAKE_PNG_B64, None) == FAKE_DATA_URI
    assert _normalize_image_input(None, "https://example.com/b.png") == "https://example.com/b.png"
    assert _normalize_image_input(None, None) == ""
