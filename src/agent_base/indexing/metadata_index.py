"""商品元数据目录（catalog）——电商版。

catalog 不是向量索引，而是结构化目录：记录商品名、规格、类目、品牌、
来源文件、章节列表等。问答前用它在用户问题中识别商品/类目约束。
兼容两种结构：电商 catalog（`products` 为 {id: {...}} 字典或列表）与
legacy 医药 catalog（`drugs` 列表，仅归档数据使用）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

@dataclass(slots=True)
class CatalogResolution:
    """从用户问题中解析出的商品约束结果。

    唯一命中一个商品时给出 product_name/product_spec/category；
    命中多个商品或类目时标记 ambiguous，避免把多商品结论混成单一答案。
    """

    product_name: str | None = None
    product_spec: str | None = None
    category: str | None = None
    matched_products: list[dict[str, Any]] | None = None
    matched_categories: list[str] | None = None
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转为 dict，便于写入 trace。"""
        return asdict(self)
def find_products(
    catalog: dict[str, Any],
    product_name: str | None = None,
    product_spec: str | None = None,
    category: str | None = None,
    section: str | None = None,
) -> list[dict[str, Any]]:
    """按商品名、规格、类目或章节在 catalog 中查商品。"""
    results = []
    for item in _iter_catalog_items(catalog):
        item_product_name = item.get("product_name") or item.get("drug_name")
        item_product_spec = item.get("product_spec") or item.get("generic_name")
        if product_name and product_name not in item_product_name:
            continue
        if product_spec and product_spec not in item_product_spec:
            continue
        if category and category != item.get("category"):
            continue
        if section and section not in item.get("sections", []):
            continue
        results.append(item)
    return results


def resolve_query_constraints(catalog: dict[str, Any], question: str) -> CatalogResolution:
    """从用户问题中解析商品/类目约束。

    唯一命中某个商品时返回 product_name/product_spec/category；
    只命中一个类目时返回 category；命中多个商品时标记 ambiguous，
    避免把多个商品的结论混成一个答案。
    """
    normalized_question = _normalize(question)
    matched_products: list[dict[str, Any]] = []
    for item in _iter_catalog_items(catalog):
        names = _product_match_names(item)
        if any(name and name in normalized_question for name in names):
            matched_products.append(item)

    matched_categories = [
        category
        for category in _iter_catalog_categories(catalog)
        if category and _normalize(category) in normalized_question
    ]

    unique_doc_ids = {item.get("doc_id") or item.get("id") for item in matched_products}
    if len(unique_doc_ids) == 1 and matched_products:
        item = matched_products[0]
        return CatalogResolution(
            product_name=item.get("product_name") or item.get("name"),
            product_spec=item.get("product_spec"),
            category=item.get("category"),
            matched_products=matched_products,
            matched_categories=matched_categories,
            ambiguous=False,
        )

    if not matched_products and len(matched_categories) == 1:
        return CatalogResolution(
            category=matched_categories[0],
            matched_products=[],
            matched_categories=matched_categories,
            ambiguous=False,
        )

    return CatalogResolution(
        matched_products=matched_products,
        matched_categories=matched_categories,
        ambiguous=len(unique_doc_ids) > 1 or len(matched_categories) > 1,
    )


def _iter_catalog_items(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """统一 catalog 商品条目：兼容 products 字典/列表与 legacy drugs 列表。"""
    products = catalog.get("products") or catalog.get("drugs") or []
    if isinstance(products, dict):
        items = []
        for pid, entry in products.items():
            if isinstance(entry, dict):
                item = dict(entry)
                item.setdefault("id", pid)
                item.setdefault("doc_id", pid)
                item.setdefault("sections", [])
                items.append(item)
        return items
    return list(products)


def _iter_catalog_categories(catalog: dict[str, Any]) -> list[str]:
    categories = catalog.get("categories") or []
    return list(categories)


def _product_match_names(item: dict[str, Any]) -> list[str]:
    names = [
        item.get("product_name", ""),
        item.get("name", ""),
        item.get("product_spec", ""),
        item.get("brand", ""),
        Path(item.get("source_file", "")).stem,
    ]
    return [_normalize(name) for name in names if name and name != "unknown"]


def _normalize(text: str) -> str:
    return (
        str(text)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("®", "")
        .replace("（", "(")
        .replace("）", ")")
    )
