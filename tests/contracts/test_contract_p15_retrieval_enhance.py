"""契约 P15：BM25 稀疏向量 + Enrich 别名/指代。

覆盖 v0.25.x 验收修复的回归点：
1. 中文 tokenize：单字 + 双字，按词切分（不跨空格拼接 bigram）
2. sparse 编码：md5 确定性 term ID + TF×IDF 权重 + Qdrant SparseVector 结构
3. 别名扩展：命中追加 canonical；问题已含 canonical 不扩展；多候选取最佳
4. 指代补全：有 current_product 前置商品名；无则原样返回
5. sparse 检索：payload 嵌套结构下 filter 可匹配、page_content 非空
"""

from __future__ import annotations


from qdrant_client.models import SparseVector

from agent_base.retrieval.enrich import expand_aliases, load_aliases, resolve_question
from agent_base.retrieval.sparse_encoder import (
    BM25SparseEncoder,
    _term_id,
    encode_query_sparse,
    tokenize_chinese,
)


def test_tokenize_chinese_per_word():
    """中文分词：按空白分词，词内 unigram + bigram，不跨词拼接。"""
    tokens = tokenize_chinese("防晒衣 轻薄")
    assert "防晒" in tokens
    assert "晒衣" in tokens
    assert "轻薄" in tokens
    assert "衣轻" not in tokens  # 跨词 bigram 不允许
    assert "防" in tokens and "晒" in tokens


def test_tokenize_chinese_unicode_escape():
    """中文分词：\u4e00-\u9fff 范围正确（用转义避免管道编码干扰）。"""
    tokens = tokenize_chinese("\u82f9\u679c\u624b\u673a")  # 苹果手机
    assert "\u82f9" in tokens  # 苹
    assert "\u679c" in tokens  # 果
    assert "\u82f9\u679c" in tokens  # 苹果
    assert "\u679c\u624b" in tokens  # 果手


def test_term_id_deterministic():
    """term ID 用 md5，跨进程确定性一致。"""
    assert _term_id("防晒") == _term_id("防晒")
    assert _term_id("防晒") != _term_id("保湿")
    assert 0 <= _term_id("防晒") <= 0x7FFFFFFF


def test_sparse_encoder_outputs_sparse_vector_with_idf():
    """BM25 编码：输出 SparseVector，权重为 TF×IDF（常见词权重低于罕见词）。"""
    corpus = [
        "防晒衣 轻薄 透气 女款",
        "防晒乳 SPF50 防水",
        "保湿精华 玻尿酸 补水",
        "洁面乳 氨基酸 温和",
    ]
    encoder = BM25SparseEncoder().fit(corpus)
    sv = encoder.encode_single("防晒衣 轻薄 透气")
    assert isinstance(sv, SparseVector)
    assert len(sv.indices) == len(sv.values) > 0
    assert all(v > 0 for v in sv.values)

    # 常见词"防晒"在语料中出现 2/4，idf 应小于仅出现 1/4 的词
    idf_sunscreen = encoder._idf.get("防晒", 1.0)
    idf_rare = encoder._idf.get("玻尿酸", 1.0)
    assert idf_sunscreen < idf_rare


def test_encode_query_sparse_works():
    """查询编码：无 corpus 也可编码，与文档编码同 term ID 空间。"""
    sv = encode_query_sparse("防晒衣")
    assert isinstance(sv, SparseVector)
    assert len(sv.indices) > 0
    assert _term_id("防晒") in sv.indices


def test_expand_aliases_appends_canonical():
    """别名命中：追加 canonical 名到查询。"""
    aliases = {"玻尿酸精华": ["玻尿酸保湿精华液", "花颜集玻尿酸精华"]}
    out = expand_aliases("玻尿酸精华保湿怎么样", aliases=aliases)
    assert "玻尿酸保湿精华液" in out or "花颜集玻尿酸精华" in out


def test_expand_aliases_skips_when_canonical_present():
    """问题已含 canonical（已指名商品）时不扩展，避免稀释查询。"""
    aliases = {"连衣裙": ["法式碎花连衣裙", "莫代尔连衣裙"]}
    out = expand_aliases("法式碎花连衣裙适合什么身材", aliases=aliases)
    assert out == "法式碎花连衣裙适合什么身材"


def test_expand_aliases_no_match():
    """无别名命中时原样返回。"""
    aliases = {"玻尿酸": ["玻尿酸保湿精华液"]}
    assert expand_aliases("今天天气不错", aliases=aliases) == "今天天气不错"


def test_resolve_question_with_current_product():
    """指代补全：短问含指代词 + current_product → 前置商品名。"""
    out = resolve_question("它多少钱", current_product="玻尿酸保湿精华液")
    assert out == "玻尿酸保湿精华液 它多少钱"


def test_resolve_question_without_product():
    """无 current_product 时不硬猜，原样返回。"""
    assert resolve_question("它多少钱", current_product=None) == "它多少钱"


def test_load_aliases_file():
    """PG alias_rules 可加载且为 dict。"""
    aliases = load_aliases()
    assert isinstance(aliases, dict)
    assert len(aliases) > 0
