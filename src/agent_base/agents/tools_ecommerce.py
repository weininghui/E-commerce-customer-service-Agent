"""电商客服工具链（P6-02）。

8 工具：3 真实 + 5 mock（标注"真实系统替换点"）。
所有工具 @tool + Pydantic schema + 失败容错。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

# ── mock 数据表 ──

# 真实系统替换点：替换为 ERP/OMS 实时查询
_MOCK_STOCK: dict[str, dict[str, Any]] = {
    "P001": {"status": "有货", "quantity": 200},
    "P002": {"status": "有货", "quantity": 150},
    "P003": {"status": "有货", "quantity": 300},
    "P009": {"status": "预售", "eta": "7-10 天"},
    "P020": {"status": "缺货", "restock": "预计 8 月底"},
}
_MOCK_DISCOUNTS: dict[str, dict[str, float]] = {
    "normal": {"discount": 1.0},
    "vip": {"discount": 0.88},
    "new_user": {"discount": 0.85},
}
_MOCK_ORDERS: dict[str, dict[str, Any]] = {
    "ORD001": {"status": "已发货", "logistics_id": "SF1234567890", "items": ["P001"], "created": "2026-08-01"},
    "ORD002": {"status": "待付款", "logistics_id": None, "items": ["P002", "P003"], "created": "2026-08-04"},
}
_MOCK_LOGISTICS: dict[str, dict[str, Any]] = {
    "SF1234567890": {"carrier": "顺丰", "status": "运输中", "eta": "2026-08-07", "nodes": ["已揽件", "到达中转仓", "派送中"]},
}


# ── 真实工具 ──


# ── Mock 工具（真实系统替换点） ──

# P11-02：已升级为真实 PG 表读取（接口不变，实现换 real table）
@tool
def check_stock(product_id: str) -> str:
    """查询商品库存状态——真实系统替换点：已接 PG inventory 表。

    Args:
        product_id: 商品 ID（如 P001）。
    """
    try:
        from agent_base.storage.pg import _conn
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT quantity, reserved, status FROM inventory WHERE product_id=%s", (product_id,))
            row = cur.fetchone()
            if not row:
                return f"商品 {product_id} 未找到库存信息"
            available = row[0] - row[1]
            return f"商品 {product_id}：{row[2]}，可用库存 {available} 件（总 {row[0]}，预占 {row[1]}）"
    except Exception as e:
        return f"check_stock 失败: {e}"


# P11-02：已升级为真实 PG 表读取（含明细+状态流）
@tool
def get_order(order_id: str) -> str:
    """查询订单状态——已接 PG orders/order_items/order_status_log 表。

    Args:
        order_id: 订单号（如 ORD001）。
    """
    try:
        from agent_base.storage.pg import _conn
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT order_no,status,total_amount,pay_amount,created_at FROM orders WHERE order_id=%s", (order_id,))
            row = cur.fetchone()
            if not row:
                return f"订单 {order_id} 未找到"
            cur.execute("SELECT product_name,price,quantity FROM order_items WHERE order_id=%s", (order_id,))
            items = cur.fetchall()
            item_str = "; ".join(f"{it[0]} x{it[2]} @{it[1]}" for it in items)
            cur.execute("SELECT to_status,note,created_at FROM order_status_log WHERE order_id=%s ORDER BY id", (order_id,))
            logs = cur.fetchall()
            log_str = " → ".join(f"{lg[0]}({lg[1]})" for lg in logs[-3:])
            return f"订单 {row[0]}：{row[1]}，金额 {row[2]}/实付 {row[3]}，商品: {item_str}，物流: {log_str}"
    except Exception as e:
        return f"get_order 失败: {e}"


# 真实系统替换点：替换为物流追踪 TMS 接口
@tool
def get_logistics(order_id: str) -> str:
    """查询物流轨迹——真实系统替换点：对接 TMS 物流系统。

    Args:
        order_id: 订单号（如 ORD001）。
    """
    order = _MOCK_ORDERS.get(order_id)
    if not order:
        return f"订单 {order_id} 未找到，无法查询物流。"
    lid = order.get("logistics_id")
    if not lid:
        return f"订单 {order_id} 暂无物流信息（状态：{order['status']}）。"
    info = _MOCK_LOGISTICS.get(lid)
    if not info:
        return f"物流单号 {lid} 未找到轨迹。"
    return (
        f"物流 {info['carrier']} {lid}：状态 {info['status']}，"
        f"预计 {info['eta']} 送达。轨迹：{' → '.join(info['nodes'])}"
    )


@tool
def get_product_info(product_query: str) -> str:
    """查询商品信息（价格/规格/功效/肤质/FAQ）——精确查 catalog + 商品长文文档。

    用于价格、属性类问题：先按商品名/ID 精确命中 catalog，再取商品长文中的
    规格与价格段落，避免向量检索把其他商品的资料张冠李戴。

    Args:
        product_query: 商品名称或商品 ID（如"视黄醇眼霜"或"P007"）。
    """
    try:
        from agent_base.retrieval.multi_source import _resolve_product_ids
        from agent_base.storage.pg import _conn

        pids = _resolve_product_ids(product_query)
        if not pids or pids == [product_query]:
            # 解析失败：按原值模糊查 catalog
            pids = [product_query]
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, brand, category, price_band, metadata FROM catalog "
                "WHERE id = ANY(%s) OR name ILIKE %s LIMIT 3",
                (pids, f"%{product_query}%"),
            )
            rows = cur.fetchall()
            # 商品长文文档（doc_id 可能是 sha256，doc_name 在 metadata 里；
            # 名称常用简称，用字符子序列匹配商品名）
            cur.execute(
                "SELECT DISTINCT ON (metadata->>'doc_name') metadata->>'doc_name', content "
                "FROM documents "
                "WHERE metadata ? 'doc_name' AND metadata->>'doc_name' LIKE '商品长文_%' "
                "AND deleted_at IS NULL"
            )
            product_docs = [
                (d or "", c)
                for d, c in cur.fetchall()
                if d
            ]
        if not rows:
            return f"未找到商品「{product_query}」，请确认商品名称。"

        parts: list[str] = []
        for pid, name, brand, category, price_band, meta in rows:
            m = meta if isinstance(meta, dict) else {}
            effects = "、".join(str(x) for x in (m.get("effects") or []))
            faq = ""
            if isinstance(m.get("faq"), list):
                faq = "；".join(
                    f"{f.get('q', '')}→{f.get('a', '')}"
                    for f in m["faq"][:2] if isinstance(f, dict)
                )
            parts.append(
                f"商品 {pid} {name}（{brand}，{category}，{price_band or '中端'}）："
                f"功效 {effects or '见文档'}。{faq}"
            )
            # 取商品长文文档中的规格与价格段落（doc_id 常用简称，如
            # 「商品长文_视黄醇眼霜.md」对应 catalog「视黄醇抗皱眼霜」，
            # 用互相包含关系匹配，避免全名 LIKE 落空）
            try:
                for doc_name_full, content in product_docs:
                    doc_name = str(doc_name_full).removeprefix("商品长文_").removesuffix(".md")
                    # doc_id 常用简称（「视黄醇眼霜」对应 catalog「视黄醇抗皱眼霜」），
                    # 非连续子串，用字符子序列判断（允许中间插入"抗皱/保湿"等修饰词）
                    def _is_subseq(sub: str, s: str) -> bool:
                        it = iter(s)
                        return all(ch in it for ch in sub)

                    if not doc_name or not _is_subseq(doc_name, name):
                        continue
                    i = str(content).find("## 产品规格与价格")
                    j = str(content).find("## ", i + 1)
                    if i >= 0:
                        price_section = str(content)[i:j if j > i else i + 400]
                        for line in price_section.splitlines():
                            line = line.strip()
                            if line.startswith("- ") and ("规格" in line or "价格" in line or "保质期" in line):
                                parts.append(f"{name} {line[2:]}")
                    break
            except Exception:
                pass
        return "\n".join(parts)
    except Exception as e:
        return f"get_product_info 失败: {e}"
