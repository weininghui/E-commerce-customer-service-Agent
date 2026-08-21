"""Query enrichment: alias expansion + anaphora resolution (P15-02 / P15-03).

Two complementary layers that operate before retrieval:
  1. **Alias expansion**: maps user-friendly aliases to canonical product names.
  2. **Anaphora resolution**: resolves "it" / "this" etc. in multi-turn
     conversations using the ``current_product`` from session memory.

Both are controlled by config switches (default false).
"""

from __future__ import annotations



# ── 别名扩展（P15-02）───────────────────────────────────────────────────────


def load_aliases() -> dict[str, list[str]]:
    """加载别名映射（纯 PG 运行时数据源，json 文件已淘汰）。

    Returns:
        ``{alias_lower: [canonical_name, ...]}`` 字典；PG 不可用返回空。
    """
    # P32c: 运行时数据源 = PG alias_rules（纯数据库，内置种子，不依赖文件）
    try:
        from agent_base.storage.pg import alias_list, alias_seed_from_json

        pg_aliases = alias_list()
        if pg_aliases:
            return pg_aliases
        # PG 空（首次/迁移前）→ 用内置种子初始化一次
        try:
            alias_seed_from_json()
            pg_aliases = alias_list()
            if pg_aliases:
                return pg_aliases
        except Exception:
            pass
    except Exception:
        pass
    return {}


def expand_aliases(
    question: str,
    aliases: dict[str, list[str]] | None = None,
) -> str:
    """扩展用户问题中的别名。

    If a known alias appears in the question, append the canonical product
    name(s) to improve dense/sparse retrieval recall.

    Args:
        question: 原始用户问题。
        aliases: 预加载的别名字典（None 时从 PG 加载）。

    Returns:
        Enriched question string (with canonical names appended), or the
        original question if no aliases matched.
    """
    if aliases is None:
        aliases = load_aliases()
    if not aliases:
        return question

    matched_canonical: list[str] = []
    q_lower = question.lower()
    for alias, canonicals in aliases.items():
        if alias in q_lower:
            # P15 修复：问题已含该别名的某个 canonical（用户已指名商品），
            # 不扩展，避免追加无关候选稀释查询
            if any(canonical in q_lower for canonical in canonicals):
                continue
            matched_canonical.extend(canonicals)

    if not matched_canonical:
        return question

# 去重
    unique = list(dict.fromkeys(matched_canonical))
    # P15 修复：问题中已包含的 canonical 不需要追加；
    # 剩余候选中只追加与问题重叠度最高的一个，避免多个候选稀释检索查询
    # （如"法式碎花连衣裙适合什么身材"不应追加"莫代尔连衣裙"）
    q_set = set(question.lower())
    candidates = [name for name in unique if name not in question.lower()]
    if not candidates:
        return question
    best = max(candidates, key=lambda name: sum(1 for ch in name if ch in q_set))
    suffix = best
    return f"{question} ({suffix})"


# ── 指代消解（P15-03）───────────────────────────────────────────────────────


# 常见中文指代模式
_REFERENCE_PATTERNS = ["它", "这个", "那个", "这款", "那款", "这件", "那件", "这", "那"]

# 视为独立问题的最小长度（过短不消解）
_MIN_STANDALONE_LEN = 4


def resolve_question(
    question: str,
    current_product: str | None = None,
) -> str:
    """消解多轮问题中的指代引用。

    If the question is short AND contains a referential pattern AND we have
    a ``current_product`` from the session, prepend the product name.
    Otherwise return the question unchanged.

    Args:
        question: 原始用户问题（当前轮）。
        current_product: Product name from previous turn (None if unavailable).

    Returns:
        Resolved question, or the original question.
    """
    if not current_product:
        return question

    q_stripped = question.strip()
    has_ref = any(p in q_stripped for p in _REFERENCE_PATTERNS)
    if not has_ref and len(q_stripped) >= _MIN_STANDALONE_LEN:
        return question

# 短问题或含指代 → 前置商品名
    return f"{current_product} {q_stripped}"
