"""契约 P31：opencode 回归 12 项 BUG 修复的固化测试。

覆盖（后端行为 + 前端契约文件校验）：
1. 文档列表 active/archived 不重叠（doc_list 按 doc_id 取最新版本再按 status 过滤）；
2. 归档/恢复接口往返（POST /archive → archived，POST /activate → active）；
3. /api/ask/stream 请求体必须用 question（query → 422 回归保护）；
4. intent_improver 返回 suggested.keywords 结构（前端读字段对齐）；
5. 切分器保留真实章节名（来源卡不再显示"上传文档"占位）；
6. 前端 doc_type 映射含 material → 面料（防英文泄漏回归）。

依赖真实 PG（与运行环境一致）；归档测试用临时文档，测后恢复原状态。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent_base.api.main import create_app
from agent_base.ingest.splitter import split_markdown_by_type
from agent_base.retrieval.intent_improver import improve_intent_config
from agent_base.storage.pg import doc_set_status

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_PAGES = PROJECT_ROOT / "frontend-react" / "src" / "pages"


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=True)


def _admin_token(c: TestClient) -> str:
    r = c.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    return r.json()["token"]


def _headers(token: str) -> dict[str, str]:
    return {"X-Admin-Token": token}


def test_doc_list_active_archived_no_overlap():
    """P1-04 回归：active / archived 两个查询不出现同一 doc_id。"""
    with _client() as c:
        token = _admin_token(c)
        act = c.get("/api/documents?status=active", headers=_headers(token))
        arc = c.get("/api/documents?status=archived", headers=_headers(token))
        assert act.status_code == 200 and arc.status_code == 200
        active_ids = {d["doc_id"] for d in act.json().get("documents", [])}
        archived_ids = {d["doc_id"] for d in arc.json().get("documents", [])}
        assert not (active_ids & archived_ids), "同一 doc_id 同时出现在 active 与 archived"


def test_archive_activate_roundtrip():
    """P1-05 回归：归档 → 移入 archived；恢复 → 回到 active（往返后还原）。"""
    with _client() as c:
        token = _admin_token(c)
        r = c.get("/api/documents?status=active", headers=_headers(token))
        docs = r.json().get("documents", [])
        if not docs:
            return  # 无文档可测，跳过
        doc_id = docs[0]["doc_id"]
        try:
            r1 = c.post(f"/api/documents/{doc_id}/archive", headers=_headers(token))
            assert r1.status_code == 200 and r1.json()["status"] == "archived"
            arc = c.get("/api/documents?status=archived", headers=_headers(token)).json()
            assert doc_id in {d["doc_id"] for d in arc.get("documents", [])}

            r2 = c.post(f"/api/documents/{doc_id}/activate", headers=_headers(token))
            assert r2.status_code == 200 and r2.json()["status"] == "active"
            act = c.get("/api/documents?status=active", headers=_headers(token)).json()
            assert doc_id in {d["doc_id"] for d in act.get("documents", [])}
        finally:
            doc_set_status(doc_id, "active")  # 还原


def test_ask_stream_requires_question_field():
    """P0/P1 回归：/api/ask/stream 必须用 question 字段（query → 422）。"""
    with _client() as c:
        r = c.post("/api/ask/stream", json={"query": "测试", "top_k": 4, "rerank": "none"})
        assert r.status_code == 422  # query 不是合法字段

        r2 = c.post("/api/ask/stream", json={"question": "", "top_k": 4, "rerank": "none"})
        assert r2.status_code == 422  # question 为空也拒绝


def test_intent_improver_returns_suggested_keywords():
    """P1-06 回归：AI 优化返回 suggested.keywords（前端读取字段对齐）。"""
    intent = {"intent": "aftersale", "keywords": ["退货"], "sections": ["售后FAQ"], "examples": ["能退货吗"]}
    # LCEL 官方链兼容的 mock：RunnableLambda（可被 | 组合）
    from langchain_core.runnables import RunnableLambda

    fake_model = RunnableLambda(
        lambda _prompt: json.dumps(
            {
                "keywords": ["退货", "退款", "换货"],
                "sections": ["售后FAQ"],
                "examples": ["能退货吗"],
                "reasoning": "补充同义表达",
            },
            ensure_ascii=False,
        )
    )
    with patch("agent_base.llms.build_chat_model", return_value=fake_model):
        result = improve_intent_config(intent)
    assert result["intent"] == "aftersale"
    assert "keywords" in result["suggested"]
    assert "换货" in result["suggested"]["keywords"]
    assert "reasoning" in result


def test_splitter_preserves_real_section():
    """P2-08 回归：切分器保留真实章节名（来源卡不显示占位）。"""
    md = "# 商品介绍\n\n## 功效说明\n\n玻尿酸精华主打保湿补水。\n\n## 使用禁忌\n\n孕妇慎用。"
    docs = split_markdown_by_type("product_detail", md)
    sections = [str(d.metadata.get("section") or "") for d in docs]
    assert any("功效说明" in s for s in sections)
    assert any("使用禁忌" in s for s in sections)
    assert not all(s == "" for s in sections)


def test_frontend_doc_type_mapping_has_material():
    """P2-09 回归：前端 doc_type 映射含 material（i18n key + 中文文案，防英文泄漏）。"""
    doc_page = FRONTEND_PAGES / "DocumentsPage" / "DocumentsPage.tsx"
    assert doc_page.exists(), f"找不到 {doc_page}"
    content = doc_page.read_text(encoding="utf-8")
    # 代码已迁移到 i18n：映射值指向 i18n key，中文文案在页面内 i18n 字典里
    assert "material: 'documents.docType.material'" in content, "DocumentsPage 缺少 material → i18n key 映射"
    assert "面料材质" in content, "DocumentsPage 缺少 material 中文文案（面料材质）"
