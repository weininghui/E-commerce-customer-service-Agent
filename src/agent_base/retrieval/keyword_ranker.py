"""升级版关键词重排（keyword_v2，兜底专用，零外部依赖）。

旧版 keyword_score 的硬伤：
- 中文按 {1,2} 切出单字/碎块，区分度差；
- 没有同义词处理（“油皮”匹配不上“油性肌肤”）；
- 词频/稀有度无权重，长文档天然占优；
- 作为 model 兜底时用“词面顺序整体覆盖”向量语义序，可能排坏结果。

升级点：
1. 电商词典最长匹配分词 + 1-2 字 n-gram 兜底；
2. 同义词/别名扩展（复用 PG alias_rules）；
3. TF-IDF 加权（IDF 从语料离线预计算并缓存，稀有词权重大）；
4. 文档长度归一化，避免长文占优；
5. 兜底排序改为“关键词信号 + 原始向量信号”融合，不再整体覆盖语义序。

本模块保持确定性、无网络、无外部进程依赖，可离线评测与复现。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


# ── 电商领域词典（成分/品类/场景/售后）──────────────────────────────────────
BUILTIN_TERMS = [
    # 成分
    "玻尿酸", "神经酰胺", "烟酰胺", "氨基酸", "水杨酸", "角鲨烷", "胜肽",
    "维C", "vc", "维生素C", "透明质酸", "尿囊素", "积雪草",
    # 品类
    "精华", "精华液", "面霜", "洁面乳", "洗面奶", "防晒霜", "防晒乳",
    "防晒衣", "面膜", "眼霜", "凝胶", "祛痘", "修护霜", "T恤", "白T恤",
    "衬衫", "阔腿裤", "直筒裤", "半身裙", "乐福鞋", "帆布鞋", "运动鞋",
    "打底", "针织开衫", "防晒袖套",
    # 肤质/场景
    "油皮", "油性皮肤", "干皮", "干性皮肤", "敏感肌", "敏感皮肤", "混油",
    "泛红", "保湿", "补水", "控油", "美白", "淡斑", "通勤", "约会",
    "孕妇", "男士", "夏季", "冬季", "春秋",
    # 售后/服务
    "退换货", "退货", "退款", "换货", "七天无理由", "物流", "发货",
    "快递", "发票", "客服", "转人工", "签收", "到货", "补货", "库存",
    "断货", "缺货", "有货", "价格", "多少钱", "优惠", "折扣", "满减",
    "优惠券", "秒杀", "包邮", "活动",
    # 基础
    "成分", "配方", "功效", "质地", "规格", "保质期", "适合", "怎么用",
    "用量", "使用方法", "尺码", "版型", "面料", "克重", "透", "缩水",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_aliases() -> dict[str, list[str]]:
    """加载电商别名表（PG alias_rules 优先，key 统一小写）。"""
    try:
        from agent_base.retrieval.enrich import load_aliases

        return {str(k).lower(): [str(v) for v in vals] for k, vals in load_aliases().items()}
    except Exception:
        return {}


_ALIAS_CACHE: dict[str, list[str]] | None = None


def alias_map() -> dict[str, list[str]]:
    """别名表（带缓存）。"""
    global _ALIAS_CACHE
    if _ALIAS_CACHE is None:
        _ALIAS_CACHE = _load_aliases()
    return _ALIAS_CACHE


def _vocabulary() -> list[str]:
    """词典 = 内置词 + 别名表的全部词（按长度降序，保证最长匹配优先）。"""
    terms = list(BUILTIN_TERMS)
    aliases = alias_map()
    for key, vals in aliases.items():
        terms.append(key)
        terms.extend(vals)
    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        t = term.strip().lower()
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    ordered.sort(key=len, reverse=True)
    return ordered


_VOCAB_CACHE: list[str] | None = None


def vocabulary() -> list[str]:
    """词典（带缓存）。"""
    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        _VOCAB_CACHE = _vocabulary()
    return _VOCAB_CACHE


_EN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str, with_aliases: bool = True) -> list[str]:
    """电商词典最长匹配 + 中文 2-gram 兜底 + 英文整词。

    Args:
        text: 待分词文本。
        with_aliases: 是否对分词结果做同义词/别名扩展（仅查询词启用）。

    Returns:
        词列表（可含重复，用于 TF 计数；查询侧调用方自行去重）。
    """
    if not text:
        return []
    vocab = vocabulary()
    tokens: list[str] = []
    rest = str(text).lower()
    pos = 0
    n = len(rest)
    while pos < n:
        ch = rest[pos]
        # 1) 词典最长匹配（优先，保证“T恤”“维C”等中英混合词不被英文分支拆散）
        matched = False
        for term in vocab:
            tlen = len(term)
            if pos + tlen <= n and rest[pos:pos + tlen] == term:
                tokens.append(term)
                pos += tlen
                matched = True
                break
        if matched:
            continue
        # 2) 英文/数字整词
        m = _EN_RE.match(rest, pos)
        if m:
            tokens.append(m.group(0))
            pos = m.end()
            continue
        # 3) 中文兜底：重叠 2-gram（滑窗），保证“玻尿酸”→“玻尿”“尿酸”
        if "\u4e00" <= ch <= "\u9fff":
            if pos + 2 <= n:
                tokens.append(rest[pos:pos + 2])
                pos += 1
            else:
                tokens.append(rest[pos:pos + 1])
                pos += 1
            continue
        # 4) 其他字符（标点/空格）跳过
        pos += 1

    if with_aliases:
        tokens = _expand_aliases(tokens)
    return tokens


def _expand_aliases(tokens: list[str]) -> list[str]:
    """对查询词做别名扩展：命中别名 key 时追加其标准全称词。"""
    aliases = alias_map()
    extra: list[str] = []
    for tok in tokens:
        for full in aliases.get(tok, []):
            extra.extend(tokenize(full, with_aliases=False))
    return tokens + extra


# ── IDF 表（从 Qdrant 权威语料预计算，PG 缓存）─────────────────────────────
_IDF_CACHE: dict[str, float] | None = None
_IDF_SOURCE_VERSION: str | None = None


def _load_chunk_texts() -> list[str]:
    """从 Qdrant 权威向量库读取全部 chunk 文本（IDF 统计用）。

    数据源为入库后的 ecommerce_chunks（页内容即权威文本），
    不再依赖切分 jsonl 文件（P32c 同款：PG/Qdrant 为真相源，文件仅缓存）。
    """
    texts: list[str] = []
    try:
        from qdrant_client import QdrantClient
        from agent_base.config import load_yaml

        cfg = load_yaml("configs/app.yaml") or {}
        vs = cfg.get("vectorstore", {}) or {}
        client = QdrantClient(url=vs.get("url") or "http://localhost:6333")
        collection = vs.get("collection", "ecommerce_chunks")
    except Exception:
        return texts
    offset = None
    while True:
        try:
            batch, offset = client.scroll(
                collection, limit=256, offset=offset,
                with_payload=["page_content"], with_vectors=False,
            )
        except Exception:
            break
        for p in batch:
            text = (p.payload or {}).get("page_content") or ""
            if text:
                texts.append(str(text))
        if offset is None:
            break
    return texts


def _chunk_count() -> int:
    """Qdrant 当前 chunk 总数（兼容保留）。"""
    try:
        from qdrant_client import QdrantClient
        from agent_base.config import load_yaml

        cfg = load_yaml("configs/app.yaml") or {}
        vs = cfg.get("vectorstore", {}) or {}
        client = QdrantClient(url=vs.get("url") or "http://localhost:6333")
        return client.count(vs.get("collection", "ecommerce_chunks"), exact=True).count
    except Exception:
        return 0


def _chunk_fingerprint() -> str:
    """Qdrant chunk 内容指纹（chunk_id 排序 hash），IDF 缓存版本键。

    点数相同但内容变化时指纹不同，保证缓存正确失效重建。
    """
    import hashlib

    try:
        from qdrant_client import QdrantClient
        from agent_base.config import load_yaml

        cfg = load_yaml("configs/app.yaml") or {}
        vs = cfg.get("vectorstore", {}) or {}
        client = QdrantClient(url=vs.get("url") or "http://localhost:6333")
        collection = vs.get("collection", "ecommerce_chunks")
    except Exception:
        return ""
    ids: list[str] = []
    offset = None
    while True:
        try:
            batch, offset = client.scroll(
                collection, limit=512, offset=offset, with_payload=False, with_vectors=False,
            )
        except Exception:
            break
        ids.extend(str(p.id) for p in batch)
        if offset is None:
            break
    ids.sort()
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]


def _build_idf(texts: list[str]) -> dict[str, float]:
    """df → idf = ln((N+1)/(df+1)) + 1；未登录词 idf 取 ln(N+1)+1。"""
    n = len(texts)
    df: dict[str, int] = {}
    for text in texts:
        seen: set[str] = set()
        for tok in tokenize(text, with_aliases=False):
            seen.add(tok)
        for tok in seen:
            df[tok] = df.get(tok, 0) + 1
    table = {tok: math.log((n + 1) / (cnt + 1)) + 1.0 for tok, cnt in df.items()}
    table["__unseen__"] = math.log(n + 1) + 1.0 if n else 1.0
    return table


def _load_or_build_idf() -> dict[str, float]:
    """加载 IDF 表；Qdrant 语料内容指纹变化时自动重建，PG 缓存，失败时回退均匀权重。"""
    global _IDF_CACHE, _IDF_SOURCE_VERSION
    version = _chunk_fingerprint()
    if not version:
        return {"__unseen__": 1.0}
    if _IDF_CACHE is not None and _IDF_SOURCE_VERSION == version:
        return _IDF_CACHE

    try:
        from agent_base.storage.pg import idf_cache_load

        cached = idf_cache_load(version)
        if cached is not None:
            _IDF_CACHE = cached
            _IDF_SOURCE_VERSION = version
            return _IDF_CACHE
    except Exception:
        pass

    texts = _load_chunk_texts()
    table = _build_idf(texts) if texts else {"__unseen__": 1.0}
    try:
        from agent_base.storage.pg import idf_cache_save

        idf_cache_save(version, table)
    except Exception:
        pass
    _IDF_CACHE = table
    _IDF_SOURCE_VERSION = version
    return table


def idf_table() -> dict[str, float]:
    """IDF 表（带缓存）。"""
    return _load_or_build_idf()


# ── 打分与融合 ───────────────────────────────────────────────────────────────
def keyword_signal(
    query: str,
    doc: Any,
    preferred_sections: list[str] | None = None,
    idf: dict[str, float] | None = None,
) -> float:
    """升级版关键词得分（0-1）。

    得分 = Σ tf * idf(查询词) + 章节命中加分 + 偏好章节加分，再做长度归一化。

    Args:
        query: 用户查询。
        doc: LangChain Document（page_content / metadata.section）。
        preferred_sections: 偏好章节列表（命中额外加分）。
        idf: 可选 IDF 表（不传则加载缓存表）。

    Returns:
        0-1 归一化得分（tanh 压缩）。
    """
    idf = idf or idf_table()
    q_terms = list(dict.fromkeys(tokenize(query, with_aliases=True)))
    text = getattr(doc, "page_content", str(doc)) or ""
    text_lower = text.lower()
    metadata = getattr(doc, "metadata", {}) or {}
    section = str(metadata.get("section", "")).lower()

    raw = 0.0
    for term in q_terms:
        weight = idf.get(term, idf.get("__unseen__", 1.0))
        tf = text_lower.count(term)
        if tf:
            raw += tf * weight
        # 章节命中（词出现在章节名里，信号更强）
        if term and term in section:
            raw += 2.0 * weight

    if preferred_sections and section in {str(s).lower() for s in preferred_sections}:
        raw += 5.0

    # 长度归一化：长文档按 log 长度摊薄，避免“词多即赢”
    denom = 1.0 + math.log(max(len(text), 1) / 20.0 + 1.0)
    norm = raw / denom
    # tanh 压缩到 0-1，区分度集中在低分区间
    return math.tanh(norm / 4.0)


def _min_max(scores: list[float | None]) -> list[float]:
    """把可空分数列表 min-max 归一化到 0-1（全空返回全 0.5）。"""
    vals = [s for s in scores if s is not None and math.isfinite(s)]
    if not vals:
        return [0.5] * len(scores)
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 1e-9:
        return [1.0 if s == hi else 0.5 for s in scores]
    return [((s - lo) / span) if s is not None else 0.0 for s in scores]


def hybrid_fallback_rank(
    query: str,
    docs: list[Any],
    top_k: int = 5,
    preferred_sections: list[str] | None = None,
    keyword_weight: float = 0.45,
    vector_weight: float = 0.55,
) -> list[Any]:
    """模型重排不可用时的融合兜底：关键词信号 + 原始向量信号。

    不再用关键词顺序“整体覆盖”向量语义序：候选的最终分 = 关键词分与
    向量分（min-max 归一化）加权融合，语义序为主、关键词微调为辅。
    vector_score 缺失时退化为纯关键词排序。
    """
    if not docs:
        return []
    idf = idf_table()
    kw_scores = [keyword_signal(query, doc, preferred_sections=preferred_sections, idf=idf) for doc in docs]
    vec_scores = [(getattr(doc, "metadata", {}) or {}).get("vector_score") for doc in docs]
    vec_norm = _min_max(vec_scores)
    final_scores = [
        keyword_weight * kw + vector_weight * vn
        for kw, vn in zip(kw_scores, vec_norm)
    ]
    ranked = sorted(zip(final_scores, docs), key=lambda item: item[0], reverse=True)
    selected = []
    for rank, (score, doc) in enumerate(ranked[:top_k], start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        metadata["rerank_strategy"] = "keyword"
        metadata["rerank_score"] = round(float(score), 6)
        metadata["keyword_score"] = round(float(kw_scores[docs.index(doc)]), 6)
        metadata["rerank_rank"] = rank
        selected.append(doc)
    return selected
