"""SelfQueryRetriever 结构化查询翻译器（自研，P14-04）。

官方 ``SelfQueryRetriever.from_llm`` 不传 ``structured_query_translator`` 时，
内部 ``_get_builtin_translator`` 会 import ``langchain_community``（sunset 弃用
警告 + 0.4.2 内部兼容性问题，实测 ImportError）——违反项目弃用红线。

这里自研两个最小 Visitor（继承 ``langchain_core.structured_query.Visitor``），
把 LLM 产出的结构化查询（IR）翻译成对应向量库的 filter 语法：

- ``ChromaStyleVisitor``：Chroma where（``{"key": {"$eq": v}}`` / ``$and`` / ``$or``）
- ``QdrantStyleVisitor``：Qdrant payload Filter（``must`` / ``should`` / ``match``）

``visit_structured_query`` 必须返回 ``(query, search_kwargs)`` 二元组
（``SelfQueryRetriever._prepare_query`` 的约定）。
"""

from __future__ import annotations

from langchain_core.structured_query import Comparator, Operator, Visitor


class ChromaStyleVisitor(Visitor):
    """结构化查询 -> Chroma where filter。"""

    allowed_comparators = (Comparator.EQ, Comparator.IN, Comparator.NE)
    allowed_operators = (Operator.AND, Operator.OR)

    def visit_operation(self, operation) -> dict:
        """递归翻译 AND/OR 逻辑运算为 Chroma where 结构。

        Args:
            operation: StructuredQuery 的 Operation 节点。

        Returns:
            {"$and": [...]} 或 {"$or": [...]} 字典。
        """
        args = [arg.accept(self) for arg in operation.arguments]
        if operation.operator == Operator.AND:
            return {"$and": args}
        return {"$or": args}

    def visit_comparison(self, comparison) -> dict:
        """翻译单个比较条件为 Chroma where 子句。

        Args:
            comparison: StructuredQuery 的 Comparison 节点。

        Returns:
            形如 {"key": {"$eq": value}} 的过滤子句。

        Raises:
            ValueError: 不支持的比较符。
        """
        if comparison.comparator == Comparator.EQ:
            return {comparison.attribute: {"$eq": comparison.value}}
        if comparison.comparator == Comparator.IN:
            return {comparison.attribute: {"$in": comparison.value}}
        if comparison.comparator == Comparator.NE:
            return {comparison.attribute: {"$ne": comparison.value}}
        raise ValueError(f"unsupported comparator: {comparison.comparator}")

    def visit_structured_query(self, structured_query) -> tuple:
        """翻译完整结构化查询，返回 SelfQueryRetriever 约定的二元组。

        Args:
            structured_query: StructuredQuery 对象。

        Returns:
            (query, {"filter": ...})，无过滤时 filter 为空 dict。
        """
        if structured_query.filter is None:
            return structured_query.query, {}
        return structured_query.query, {"filter": structured_query.filter.accept(self)}


class QdrantStyleVisitor(Visitor):
    """结构化查询 -> Qdrant payload Filter。"""

    allowed_comparators = (Comparator.EQ, Comparator.IN, Comparator.NE)
    allowed_operators = (Operator.AND, Operator.OR)

    def visit_operation(self, operation) -> dict:
        """递归翻译 AND/OR 逻辑运算为 Qdrant payload Filter 结构。

        Args:
            operation: StructuredQuery 的 Operation 节点。

        Returns:
            {"must": [...]} 或 {"should": [...]} 字典。
        """
        args = [arg.accept(self) for arg in operation.arguments]
        if operation.operator == Operator.AND:
            return {"must": args}
        return {"should": args}

    def visit_comparison(self, comparison) -> dict:
        """翻译单个比较条件为 Qdrant payload Filter 子句。

        Args:
            comparison: StructuredQuery 的 Comparison 节点。

        Returns:
            形如 {"key": ..., "match": {"value": ...}} 的过滤子句。

        Raises:
            ValueError: 不支持的比较符。
        """
        if comparison.comparator == Comparator.EQ:
            return {"key": comparison.attribute, "match": {"value": comparison.value}}
        if comparison.comparator == Comparator.IN:
            return {"key": comparison.attribute, "match": {"any": comparison.value}}
        if comparison.comparator == Comparator.NE:
            return {"key": comparison.attribute, "match": {"except": [comparison.value]}}
        raise ValueError(f"unsupported comparator: {comparison.comparator}")

    def visit_structured_query(self, structured_query) -> tuple:
        """翻译完整结构化查询，返回 SelfQueryRetriever 约定的二元组。

        Args:
            structured_query: StructuredQuery 对象。

        Returns:
            (query, {"filter": ...})，无过滤时 filter 为空 dict。
        """
        if structured_query.filter is None:
            return structured_query.query, {}
        return structured_query.query, {"filter": structured_query.filter.accept(self)}
