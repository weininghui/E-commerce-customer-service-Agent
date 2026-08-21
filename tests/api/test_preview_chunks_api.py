"""分割预览接口契约测试（与入库同一切分器）。"""
import sys

sys.path.insert(0, "src")
from fastapi.testclient import TestClient

from agent_base.api.main import app

client = TestClient(app)
H = {"X-Admin-Token": "admin-dev-token-2026"}


def test_preview_chunks_with_text_and_type():
    md = "# 产品信息\n\n这是成分说明，包含烟酰胺与玻尿酸。\n\n## 使用方法\n\n早晚各一次，避开眼周。"
    r = client.post("/api/documents/preview-chunks", json={"text": md, "doc_type": "product_detail"}, headers=H)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["doc_type"] == "product_detail"
    assert d["profile"]["chunk_size"] == 900
    assert d["profile"]["chunk_overlap"] == 120
    assert d["profile"]["mode"] == "section"
    assert d["total_chunks"] >= 2
    assert any(s["raw"] == "\n\n" for s in d["profile"]["separators"])
    for c in d["chunks"]:
        assert c["text"].strip()
        assert c["chars"] == len(c["text"])


def test_preview_chunks_requires_type():
    r = client.post("/api/documents/preview-chunks", json={"text": "x"}, headers=H)
    assert r.status_code == 400


def test_preview_chunks_faq_profile():
    md = "Q: 支持退货吗\nA: 支持七天无理由退货。\n\nQ: 发货多久\nA: 48 小时内发货。"
    r = client.post("/api/documents/preview-chunks", json={"text": md, "doc_type": "faq"}, headers=H)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["profile"]["chunk_size"] == 512
    assert d["total_chunks"] >= 1
