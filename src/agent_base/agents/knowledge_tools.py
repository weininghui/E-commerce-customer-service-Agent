"""知识运营工具集 + 可插拔工具注册表。

- 工具：知识库增 / 删 / 改 / 查（复用 PG 存储层）；
- 注册表：``TOOL_REGISTRY`` 名称 → 可调用对象，支持动态注册扩展；
- 所有工具失败返回错误信息（不抛异常），供运营 Agent 反思降级。
"""

from __future__ import annotations

from typing import Any, Callable


def kb_query(query: str, limit: int = 5) -> dict[str, Any]:
    """查询知识库文档（按 doc_id/文件名/内容模糊匹配）。"""
    try:
        from agent_base.storage.pg import _conn

        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT doc_id, metadata->>'doc_name', content, status "
                "FROM documents WHERE status='active' AND deleted_at IS NULL "
                "AND (doc_id ILIKE %s OR COALESCE(metadata->>'doc_name','') ILIKE %s "
                "     OR content ILIKE %s) "
                "ORDER BY updated_at DESC LIMIT %s",
                (f"%{query}%", f"%{query}%", f"%{query}%", int(limit)),
            )
            rows = cur.fetchall()
        return {
            "ok": True,
            "items": [
                {
                    "doc_id": r[0],
                    "doc_name": r[1] or "",
                    "content": str(r[2] or "")[:200],
                    "status": r[3],
                }
                for r in rows
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


def kb_add(content: str, filename: str = "", category: str = "运营指令") -> dict[str, Any]:
    """新增知识文档（先进入暂存，标注运营来源）。"""
    try:
        from agent_base.storage.pg import staging_upsert

        import hashlib

        doc_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
        staging_upsert(
            doc_id=doc_id,
            filename=filename or f"ops-{doc_id[:8]}.md",
            content=content,
            category=category,
            status="pending",
            first_review={"source": "knowledge_ops", "type": ""},
        )
        return {"ok": True, "doc_id": doc_id, "action": "add"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


def kb_update(doc_id: str, content: str) -> dict[str, Any]:
    """更新知识文档（新增版本并置为 active）。"""
    try:
        from agent_base.storage.pg import _conn

        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COALESCE(MAX(version),0) FROM documents WHERE doc_id=%s",
                (doc_id,),
            )
            version = cur.fetchone()[0] + 1
            cur.execute(
                "INSERT INTO documents (doc_id, version, content, status, updated_at) "
                "VALUES (%s,%s,%s,'active',NOW()) "
                "ON CONFLICT (doc_id, version) DO NOTHING",
                (doc_id, version, content),
            )
            cur.execute(
                "UPDATE documents SET status='archived', updated_at=NOW() "
                "WHERE doc_id=%s AND version<%s",
                (doc_id, version),
            )
            conn.commit()
        return {"ok": True, "doc_id": doc_id, "version": version, "action": "update"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


def kb_delete(doc_id: str) -> dict[str, Any]:
    """删除知识文档（软删除）。"""
    try:
        from agent_base.storage.pg import doc_set_status

        ok = doc_set_status(doc_id, "deleted")
        if not ok:
            return {"ok": False, "error": f"文档 {doc_id} 不存在"}
        return {"ok": True, "doc_id": doc_id, "action": "delete"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "kb_query": kb_query,
    "kb_add": kb_add,
    "kb_update": kb_update,
    "kb_delete": kb_delete,
}


def register_tool(name: str, fn: Callable[..., Any]) -> None:
    """动态注册知识运营工具（可插拔扩展）。"""
    TOOL_REGISTRY[name] = fn


def list_tools() -> list[dict[str, Any]]:
    """列出已注册工具（名称 + 说明）。"""
    return [
        {
            "name": name,
            "doc": (getattr(fn, "__doc__", "") or "").strip().splitlines()[0]
            if getattr(fn, "__doc__", "")
            else "",
        }
        for name, fn in TOOL_REGISTRY.items()
    ]
