"""P2 多源知识库检索：评价 / 搭配方案 / 客户案例。

三个专属 PG 数据源，worker 按 goal.action 分发：
- review  → search_reviews
- combine → search_combos
- usage   → search_cases

均返回带 source_type 字段的 dict 列表，与 goal_evidences 兼容。
"""

from __future__ import annotations

from typing import Any


def _resolve_product_ids(product: str) -> list[str]:
    """商品名/ID → catalog ID 列表（中文名模糊匹配，ID 精确匹配）。

    worker 的 goal.targets 是商品名（如"玻尿酸精华"），而评价/案例表用
    catalog ID（P001）关联，需先解析。简称先走 alias_rules 换成全称再匹配
    （"玻尿酸精华" → "玻尿酸保湿精华液"），避免子串不连续导致解析不到。

    Args:
        product: 商品名或 catalog ID。

    Returns:
        catalog ID 列表；解析失败回退按原值查询。
    """
    if not product:
        return []
    try:
        from agent_base.storage.pg import _conn
        from agent_base.retrieval.enrich import load_aliases

        # 1. 别名解析：简称 → 全称（alias_rules 运行时数据源）
        aliases = load_aliases()
        names = [product]
        names.extend(aliases.get(product.lower(), []) or [])

        with _conn() as conn:
            cur = conn.cursor()
            ids: list[str] = []
            # 1) ILIKE 精确/子串匹配（含别名全称）
            for name in names:
                cur.execute(
                    "SELECT id FROM catalog WHERE name ILIKE %s OR id = %s LIMIT 20",
                    (f"%{name}%", name),
                )
                for r in cur.fetchall():
                    if str(r[0]) not in ids:
                        ids.append(str(r[0]))
            if not ids:
                # 2) token 子集匹配：product 的分词 token 全部出现在 catalog.name 中
                #    （"烟酰胺精华" → tokens{烟酰胺,精华} ⊆ "烟酰胺焕亮精华"）
                try:
                    import jieba

                    query_tokens = {w for w in jieba.cut(product) if len(w) >= 2}
                    if query_tokens:
                        cur.execute("SELECT id, name FROM catalog")
                        for rid, name in cur.fetchall():
                            name_tokens = {w for w in jieba.cut(str(name)) if len(w) >= 2}
                            if query_tokens.issubset(name_tokens):
                                if str(rid) not in ids:
                                    ids.append(str(rid))
                except Exception:
                    pass
        return ids or [product]
    except Exception:
        return [product]


def search_reviews(product_id: str, top_k: int = 5) -> list[dict[str, Any]]:
    """按商品查评价库（PG ILIKE 模糊匹配，评分降序）。

    Args:
        product_id: catalog 商品 ID。
        top_k: 返回条数。

    Returns:
        [{"source_type": "reviews", "section": "用户评价", "content": ..., "score": rating/5, ...}, ...]
    """
    try:
        from agent_base.storage.pg import review_list

        rows = []
        seen: set[int] = set()
        for pid in _resolve_product_ids(product_id):
            for r in review_list(product_id=pid, limit=top_k):
                if r.get("id") in seen:
                    continue
                seen.add(r.get("id"))
                rows.append(r)
                if len(rows) >= top_k:
                    break
            if len(rows) >= top_k:
                break
        return [
            {
                "source_type": "reviews",
                "section": f"用户评价（{r.get('sentiment', 'positive')}）",
                "content": str(r.get("content", ""))[:400],
                "score": float(r.get("rating", 5)) / 5.0,
                "product_id": str(r.get("product_id", "")),
                "doc_name": f"评价 #{r.get('id', '')}",
                "chapter": "用户评价",
                "relevance": int(float(r.get("rating", 5)) / 5.0 * 100),
                "preview": str(r.get("content", ""))[:200],
            }
            for r in rows
        ]
    except Exception:
        return []


def search_combos(scenario: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
    """按场景查搭配方案库（PG ILIKE 模糊匹配）。

    Args:
        scenario: 场景关键词（干皮/油皮/通勤/约会等），None 返回全部。
        top_k: 返回条数。

    Returns:
        [{"source_type": "combos", "section": "搭配方案", "content": ..., ...}, ...]
    """
    try:
        from agent_base.storage.pg import combo_list

        rows = combo_list(scenario=scenario, limit=top_k)
        return [
            {
                "source_type": "combos",
                "section": f"搭配方案 · {r.get('scenario', '通用')}",
                "content": f"{r.get('name', '')}：{r.get('description', '')}"[:400],
                "score": 0.85,
                "doc_name": f"搭配 #{r.get('id', '')}",
                "chapter": r.get("scenario", ""),
                "relevance": 85,
                "preview": str(r.get("description", ""))[:200],
                "product_ids": r.get("product_ids", []),
            }
            for r in rows
        ]
    except Exception:
        return []


def search_cases(product_id: str, skin_type: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
    """按商品+肤质查客户案例库（PG 精确匹配）。

    Args:
        product_id: catalog 商品 ID。
        skin_type: 肤质过滤（干皮/油皮/敏感肌等），None 不过滤。
        top_k: 返回条数。

    Returns:
        [{"source_type": "cases", "section": "客户案例", "content": ..., ...}, ...]
    """
    try:
        from agent_base.storage.pg import case_list

        rows = []
        for pid in _resolve_product_ids(product_id):
            rows.extend(case_list(product_id=pid, skin_type=skin_type, limit=top_k))
            if len(rows) >= top_k:
                break
        return [
            {
                "source_type": "cases",
                "section": f"客户案例 · {r.get('skin_type', '')}",
                "content": f"肤质{r.get('skin_type', '')}，{r.get('scenario', '')}，使用{r.get('duration', '')}：{r.get('result', '')}"[:400],
                "score": 0.8,
                "doc_name": f"案例 #{r.get('id', '')}",
                "chapter": r.get("scenario", ""),
                "relevance": 80,
                "preview": str(r.get("result", ""))[:200],
            }
            for r in rows
        ]
    except Exception:
        return []
