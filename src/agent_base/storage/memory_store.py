"""官方 LangGraph BaseStore 适配器：user_memories 表 → 官方 Store API。

面试点：
- ``langgraph.store.base.BaseStore`` 是官方长期记忆存储接口（跨会话/跨线程）。
- 本适配器把现有 PG ``user_memories`` 表包装成官方 Store API，
  ``create_agent(store=...)`` 即可获得官方长期记忆能力。
- 写门控（confidence/冲突检测/脱敏）保留自研（memory.py），Store 只做存取。

namespace 约定：(user_id,)；key = memory_key；value 为记忆记录 dict。
"""

from __future__ import annotations

from typing import Any, Iterable

from langgraph.store.base import BaseStore, Item

from agent_base.storage.pg import _conn


class UserMemoryStore(BaseStore):
    """PG-backed BaseStore：namespace=(user_id,) → user_memories 表。

    实现官方 Store 抽象方法：get / put / delete / search / list_namespaces。
    语义与 langgraph InMemoryStore 对齐：put 覆盖写、get 单键读、search 前缀扫。
    """

    def _key(self, namespace: tuple[str, ...], key: str) -> str:
        return f"{'/'.join(str(n) for n in namespace)}::{key}"

    def get(self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None) -> Item | None:
        """按命名空间 + 键读取记忆条目（官方 Store 接口）。"""
        user_id = namespace[0]
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT memory_key, value, confidence, source, updated_at "
                "FROM user_memories WHERE user_id=%s AND memory_key=%s",
                (user_id, key),
            )
            row = cur.fetchone()
        if not row:
            return None
        return Item(
            namespace=namespace,
            key=row[0],
            value={
                "value": row[1],
                "confidence": row[2],
                "source": row[3],
                "updated_at": row[4].isoformat() if row[4] else None,
            },
            created_at=row[4],
            updated_at=row[4],
        )

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: list[str] | None = None,
        *,
        ttl: float | None = None,
    ) -> None:
        """写入记忆条目（upsert，值做业务标签清洗，官方 Store 接口）。"""
        user_id = namespace[0]
        from psycopg2.extras import Json
        from agent_base.storage.memory import sanitize_key, sanitize_value

        safe_key = sanitize_key(key)
        safe_value = sanitize_value(value.get("value", value))
        confidence = float(value.get("confidence", 0.5) or 0.5)
        source = str(value.get("source", "conversation"))
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO user_memories (user_id, memory_key, value, source, confidence, updated_at)
                   VALUES (%s,%s,%s,%s,%s,NOW())
                   ON CONFLICT (user_id, memory_key) DO UPDATE SET
                     value=EXCLUDED.value, source=EXCLUDED.source,
                     confidence=EXCLUDED.confidence, updated_at=NOW()""",
                (user_id, safe_key, Json(safe_value), source, confidence),
            )

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """删除指定记忆条目（官方 Store 接口）。"""
        user_id = namespace[0]
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM user_memories WHERE user_id=%s AND memory_key=%s",
                (user_id, key),
            )

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Item]:
        """按命名空间前缀搜索记忆条目（官方 Store 接口）。"""
        user_id = namespace_prefix[0]
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT memory_key, value, confidence, source, updated_at "
                "FROM user_memories WHERE user_id=%s ORDER BY updated_at DESC LIMIT %s OFFSET %s",
                (user_id, limit, offset),
            )
            rows = cur.fetchall()
        items = []
        for row in rows:
            items.append(
                Item(
                    namespace=namespace_prefix,
                    key=row[0],
                    value={
                        "value": row[1],
                        "confidence": row[2],
                        "source": row[3],
                        "updated_at": row[4].isoformat() if row[4] else None,
                    },
                    created_at=row[4],
                    updated_at=row[4],
                )
            )
        return items

    def list_namespaces(self, *, prefix: tuple[str, ...] | None = None, limit: int = 100, offset: int = 0) -> list[tuple[str, ...]]:
        """列出全部用户命名空间（官方 Store 接口）。"""
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT user_id FROM user_memories ORDER BY user_id LIMIT %s OFFSET %s", (limit, offset))
            return [(row[0],) for row in cur.fetchall()]

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """内部批量操作：逐个分发到 get/put/delete/search（语义等价官方 Store）。"""
        results: list[Any] = []
        for op in ops:
            kind = op[0]
            args = op[1] if len(op) > 1 else {}
            kwargs = op[2] if len(op) > 2 else {}
            if kind == "get":
                results.append(self.get(*args, **kwargs))
            elif kind == "put":
                self.put(*args, **kwargs)
                results.append(None)
            elif kind == "delete":
                self.delete(*args, **kwargs)
                results.append(None)
            elif kind == "search":
                results.append(self.search(*args, **kwargs))
            else:
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        """异步批量：同步实现代理（PG 连接为同步，官方允许简单代理）。"""
        return self.batch(ops)
