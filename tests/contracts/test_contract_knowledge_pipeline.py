"""契约测试：四 Agent 知识生产流水线 + MinerU 解析 + 知识运营 + 商品图 + 归因。"""

from __future__ import annotations

from agent_base.ingest.mineru_parser import parse_document


def test_mineru_mock_md_roundtrip():
    r = parse_document("faq.md", "# FAQ\nQ: 怎么退货？\nA: 7 天无理由。".encode("utf-8"))
    assert r["ok"] is True
    assert r["engine"] == "mock"
    assert "退货" in r["text"]


def test_mineru_mock_pdf_fallback():
    r = parse_document("doc.pdf", b"%PDF-1.4 fake bytes")
    assert r["ok"] is True
    assert r["engine"] == "mock"
    assert "MinerU" in r["text"] or "mock" in r["text"].lower()


def test_knowledge_pipeline_runs_end_to_end():
    from agent_base.knowledge_pipeline import run_knowledge_pipeline

    r = run_knowledge_pipeline(
        filename="guide.md",
        content="# 敏感肌选购攻略\n敏感肌换季应该选择温和洁面，避免皂基成分，注意保湿修护。",
        category="测试",
    )
    # 链路必须完整走通：无论质检通过与否，都有终态（published/returned/failed）
    assert r.get("status") in {"published", "returned", "published_staged", "failed"}
    assert r.get("parser_engine") == "mock"


def test_knowledge_ops_rule_plan():
    from agent_base.agents.knowledge_ops import _rule_plan

    plan = _rule_plan("查询防晒相关文档")
    assert plan is not None and plan["tool"] == "kb_query"
    plan = _rule_plan("删除 doc123")
    assert plan is not None and plan["tool"] == "kb_delete"
    plan = _rule_plan("新增：面膜使用说明")
    assert plan is not None and plan["tool"] == "kb_add"


def test_knowledge_tools_registry():
    from agent_base.agents.knowledge_tools import TOOL_REGISTRY, list_tools, register_tool

    assert {"kb_query", "kb_add", "kb_update", "kb_delete"} <= set(TOOL_REGISTRY)

    def _dummy(**kwargs):
        return {"ok": True}

    register_tool("kb_test", _dummy)
    assert "kb_test" in TOOL_REGISTRY
    names = {t["name"] for t in list_tools()}
    assert "kb_test" in names


def test_knowledge_ops_unknown_tool_fallback():
    from agent_base.agents.knowledge_ops import run_knowledge_ops

    r = run_knowledge_ops("查询", operator="test")
    assert r["ok"] in (True, False)  # 不抛异常
    assert r["plan"]["tool"] == "kb_query"


def test_image_gen_mock_without_key():
    import os

    os.environ.pop("ARK_API_KEY", None)
    from agent_base.multimodal import generate_product_image

    r = generate_product_image("test product")
    assert r["ok"] is True
    assert r["engine"] == "mock"
    assert r["image"].startswith("data:image/")


def test_failure_attribution():
    from agent_base.retrieval.eval_chain import classify_failure

    assert classify_failure({"error": "boom"})[0] == "system_error"
    assert classify_failure({"expected_intent": "a", "intent_hit": False, "actual_intent": "b"})[0] == "intent_miss"
    assert classify_failure({"expected_source": "x", "recall_hit": False})[0] == "retrieval_fail"
    assert classify_failure({"expected_facts": ["x"], "fact_hits": 1, "fact_total": 2})[0] == "generation_fail"
    assert classify_failure({"expected_source": "", "expected_facts": [], "recall_hit": False, "fact_total": 0})[0] == "knowledge_gap"


def test_mineru_mock_docx_extraction():
    """mock：docx（zip）本地抽取正文文本，不再输出二进制垃圾。"""
    import io
    import zipfile

    from agent_base.ingest.mineru_parser import parse_document

    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:t>神经酰胺修护面霜</w:t></w:p>"
        "<w:p><w:t>屏障修护 干敏肌友好</w:t></w:p></w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", xml)
    r = parse_document("面霜说明.docx", buf.getvalue())
    assert "神经酰胺修护面霜" in r["text"]
    assert "屏障修护 干敏肌友好" in r["text"]
    assert r["engine"] == "mock"


def test_mineru_mock_pptx_extraction():
    """mock：pptx（zip）抽取每页 slide 文本。"""
    import io
    import zipfile

    from agent_base.ingest.mineru_parser import parse_document

    xml = (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:t>使用要点</a:t></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ppt/slides/slide1.xml", xml)
    r = parse_document("培训.pptx", buf.getvalue())
    assert "使用要点" in r["text"]


def test_mineru_mock_html_strip():
    """mock：HTML 去脚本/样式后剥标签。"""
    from agent_base.ingest.mineru_parser import _strip_html, parse_document

    html = "<html><body><h1>使用要点</h1><script>bad()</script><p>避免入眼</p></body></html>"
    text = _strip_html(html)
    assert "使用要点" in text and "避免入眼" in text
    assert "bad()" not in text
    r = parse_document("notes.htm", html.encode("utf-8"))
    assert "使用要点" in r["text"]
    assert r["engine"] == "mock"


def test_mineru_mock_unknown_format_placeholder():
    """mock：未知格式返回明确占位说明，不把二进制当文本。"""
    from agent_base.ingest.mineru_parser import parse_document

    r = parse_document("data.bin", b"\x00\x01\x02")
    assert "MINERU_API_KEY" in r["text"]
    assert r["engine"] == "mock"
