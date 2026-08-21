"""索引构建包：向量库加载 + catalog 解析（JSONL 管道已淘汰）。"""

from agent_base.indexing.vector_index import load_vector_store
from agent_base.indexing.metadata_index import find_products, resolve_query_constraints

__all__ = ["load_vector_store", "find_products", "resolve_query_constraints"]
