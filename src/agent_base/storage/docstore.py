"""PG-backed BaseStore for MultiVector/ParentDocument docstore persistence (P16-05).

Implements ``langchain_core.stores.BaseStore[str, bytes]`` using a Postgres table.
Survives restarts — the official retrievers can read parent docs back after ingest.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Sequence

import pickle

from langchain_core.stores import BaseStore

logger = logging.getLogger(__name__)


class PGDocStore(BaseStore[str, Any]):
    """Persist docstore entries in a PG table (Document semantics in/out).

    Table ``docstore_kv`` is created idempotently on first use.  Values are
    pickled on write and unpickled on read, so ``mget`` returns the same
    objects that were stored (LangChain Document by default) — matching
    ``InMemoryStore`` semantics required by ``MultiVectorRetriever`` /
    ``ParentDocumentRetriever`` (v0.28.2 修复：此前 mget 返回原始 BYTEA
    memoryview，官方检索器把字节当 Document 返回导致"命中摘要回原块"失效).

    Usage::

        store = PGDocStore()
        # 入库：存储父文档
        store.mset([("doc_1", doc1), ("doc_2", doc2)])  # Document 或 bytes 均可
        # 检索：MultiVectorRetriever 读回父文档
        parents = store.mget(["doc_1", "doc_2"])  # -> [Document, Document]
    """

    def __init__(self, table_name: str = "docstore_kv"):
        """初始化 PG 文档存储。

        Args:
            table_name: KV 表名（默认 docstore_kv）。
        """
        self._table = table_name
        self._ensure_table()

    def _ensure_table(self):
        """创建 KV 表（不存在时）。"""
        try:
            from agent_base.storage.pg import _conn
            with _conn() as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        key TEXT PRIMARY KEY,
                        value BYTEA NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
        except Exception as exc:  # pragma: no cover - 依赖 PG，失败不应静默
            logger.warning("PGDocStore mset failed: %s", exc)

    def mget(self, keys: Sequence[str]) -> list[Any | None]:
        """按 key 批量读取值（反序列化）。

        Args:
            keys: 文档 key 列表。

        Returns:
            存储对象列表（缺失为 None），顺序与 keys 一致。
        """
        if not keys:
            return []
        try:
            from agent_base.storage.pg import _conn
            with _conn() as conn:
                cur = conn.cursor()
# 安全构造 IN 子句
                placeholders = ",".join(["%s"] * len(keys))
                cur.execute(
                    f"SELECT key, value FROM {self._table} WHERE key IN ({placeholders})",
                    list(keys),
                )
                found = {row[0]: row[1] for row in cur.fetchall()}
            out: list[Any | None] = []
            for k in keys:
                raw = found.get(k)
                if raw is None:
                    out.append(None)
                    continue
                data = bytes(raw) if isinstance(raw, (memoryview, bytearray)) else raw
                out.append(deserialize_doc(data))
            return out
        except Exception:
            return [None] * len(keys)

    def mset(self, key_value_pairs: Sequence[tuple[str, Any]]) -> None:
        """批量写入 key-value（upsert，写入时 pickle）。

        Args:
            key_value_pairs: (key, Document | bytes) 元组列表。
        """
        if not key_value_pairs:
            return
        try:
            from agent_base.storage.pg import _conn
            with _conn() as conn:
                cur = conn.cursor()
                for key, value in key_value_pairs:
                    payload = value if isinstance(value, bytes) else serialize_doc(value)
                    cur.execute(
                        f"INSERT INTO {self._table} (key, value, updated_at) VALUES (%s,%s,NOW()) "
                        f"ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW()",
                        (key, payload),
                    )
        except Exception as exc:  # pragma: no cover - 依赖 PG，失败不应静默
            logger.warning("PGDocStore mdelete failed: %s", exc)

    def mdelete(self, keys: Sequence[str]) -> None:
        """按 key 批量删除。

        Args:
            keys: 待删除的 key 列表。
        """
        if not keys:
            return
        try:
            from agent_base.storage.pg import _conn
            with _conn() as conn:
                cur = conn.cursor()
                placeholders = ",".join(["%s"] * len(keys))
                cur.execute(
                    f"DELETE FROM {self._table} WHERE key IN ({placeholders})",
                    list(keys),
                )
        except Exception:
            pass

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        """Yield all stored keys, optionally filtered by prefix.

        BaseStore 抽象方法实现（MultiVectorRetriever / 文档生命周期检查依赖）。

        Args:
            prefix: Optional key prefix filter.

        Yields:
            Key strings.
        """
        try:
            from agent_base.storage.pg import _conn
            with _conn() as conn:
                cur = conn.cursor()
                if prefix:
                    cur.execute(
                        f"SELECT key FROM {self._table} WHERE key LIKE %s",
                        (f"{prefix}%",),
                    )
                else:
                    cur.execute(f"SELECT key FROM {self._table}")
                for row in cur.fetchall():
                    yield row[0]
        except Exception:
            return



def serialize_doc(doc: Any) -> bytes:
    """将 LangChain Document pickle 为字节，供 PG 存储。

    Args:
        doc: LangChain Document 实例。

    Returns:
        pickle 后的字节。
    """
    return pickle.dumps(doc)


def deserialize_doc(data: bytes) -> Any:
    """将字节反序列化为 LangChain Document。

    Args:
        data: pickle 字节。

    Returns:
        LangChain Document 实例。
    """
    if data is None:
        return None
    return pickle.loads(data)


def populate_parent_docstore(
    docstore: Any,
    parent_docs: list[Any],
) -> int:
    """填充父子 docstore：key=parent_id，value=父 Document。

    Args:
        docstore: BaseStore 实例。
        parent_docs: 父 Document 列表。

    Returns:
        写入条目数。
    """
    items = []
    for doc in parent_docs:
        pid = doc.metadata.get("chunk_id") or doc.metadata.get("doc_id") or str(hash(doc.page_content))
        items.append((pid, serialize_doc(doc)))

    if items:
        docstore.mset(items)
    return len(items)
