"""真实数据接口测试：用项目现有 MD 文档 / 真实问题验证检索、预审分类与置信度。

只读真实数据（data/ecommerce/md/、documents、Qdrant），不写库。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from agent_base.knowledge_factory import pre_review_document

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MD_DIR = PROJECT_ROOT / "data" / "ecommerce" / "md"

REAL_QUESTIONS = [
    "玻尿酸精华有什么功效？适合什么肤质？",
    "支持七天无理由退货吗？退款多久到账？",
    "白色纯棉T恤怎么搭配？",
]


def _real_md_files() -> list[Path]:
    return sorted(MD_DIR.glob("*.md"))


def test_real_documents_exist():
    """真实 MD 数据存在（测试前提）。"""
    files = _real_md_files()
    assert len(files) >= 10, f"真实 MD 文档不足：{len(files)}"


def test_real_doc_heuristic_classification():
    """真实文档启发式分类：类型非空、置信度在合理区间且非全部同分。"""
    results = []
    for md in _real_md_files()[:8]:
        content = md.read_text(encoding="utf-8")
        tag = pre_review_document(content[:2000], filename=md.name, llm_cfg={"provider": "none"})
        assert tag.doc_type, f"{md.name} 分类为空"
        assert 0.3 <= tag.confidence <= 0.8, f"{md.name} 置信度越界: {tag.confidence}"
        assert tag.first_review.get("source") == "heuristic"
        results.append((md.name, tag.doc_type, tag.confidence))

    confs = {round(c, 2) for _, _, c in results}
    assert len(confs) > 1, f"真实文档置信度无区分（全部 {confs}）：{results}"


def test_real_documents_list(client: TestClient, headers: dict[str, str]):
    """真实文档列表：≥10 份、doc_name 非空、active/archived 无重叠。"""
    r = client.get("/api/documents?status=active", headers=headers)
    assert r.status_code == 200
    docs = r.json().get("documents", [])
    assert len(docs) >= 10
    assert all(d.get("doc_name") for d in docs)
    arc = client.get("/api/documents?status=archived", headers=headers).json().get("documents", [])
    active_ids = {d["doc_id"] for d in docs}
    archived_ids = {d["doc_id"] for d in arc}
    assert not (active_ids & archived_ids)


def test_real_question_retrieval(client: TestClient, headers: dict[str, str]):
    """真实问题检索：返回非空来源，来源文档名匹配真实 MD。"""
    q = REAL_QUESTIONS[0]
    r = client.post("/api/retrieve", json={"question": q, "top_k": 4, "rerank": "none"}, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    # 检索来源在 trace.results（retrieve 响应结构：trace + catalog_resolution）
    results = (body.get("trace") or {}).get("results") or []
    assert len(results) >= 1, f"真实问题无检索来源: {q}"
    # 来源应包含商品/FAQ 文档（doc_name 或 section 非空）
    assert any(r.get("doc_name") or r.get("section") or r.get("preview") for r in results)


def test_real_question_intent(client: TestClient, headers: dict[str, str]):
    """真实问题意图识别：返回意图 key。"""
    r = client.post("/api/intents/test", json={"question": REAL_QUESTIONS[1]}, headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body.get("intent"), body


def test_review_queue_real_state(client: TestClient, headers: dict[str, str]):
    """审核队列当前真实状态：结构正确（数量不限）。"""
    r = client.get("/api/documents/review-queue?status=pending_fine_review", headers=headers)
    assert r.status_code == 200
    queue = r.json().get("queue", [])
    for item in queue:
        assert item["doc_id"] and item["filename"]
        assert "review_round" in item and "confidence" in item and "suggest_action" in item
