"""P24a 契约测试：官方切分器 + doc_type 档位映射（官方 MarkdownHeaderTextSplitter / RecursiveCharacterTextSplitter）。"""

from __future__ import annotations

from agent_base.ingest.splitter import (
    CHUNK_PROFILES,
    DEFAULT_PROFILE,
    build_md_section_splitter,
    get_profile,
    split_markdown_by_type,
    split_plain_text_by_type,
)


def test_build_md_section_splitter_is_official():
    """MarkdownHeaderTextSplitter 实例且支持中文标题层级。"""
    splitter = build_md_section_splitter()
    assert type(splitter).__name__ == "MarkdownHeaderTextSplitter"
    docs = splitter.split_text(
        "## 玻尿酸是什么\n玻尿酸是保湿成分。\n\n## 适用肤质\n各肤质均适用。"
    )
    assert len(docs) == 2
    assert docs[0].metadata.get("H2") == "玻尿酸是什么"
    assert docs[1].metadata.get("H2") == "适用肤质"


def test_split_markdown_by_type_section_metadata_and_title_prefix():
    """MD 切分：章节名进 metadata，标题拼回正文（参与向量化）。"""
    text = (
        "> doc_type: ingredient\n"
        "\n"
        "## 玻尿酸是什么\n"
        "玻尿酸（透明质酸）是皮肤天然存在的保湿成分，可结合自身重量 1000 倍的水分。\n"
        "\n"
        "## 适用肤质\n"
        "各肤质均适用，尤其适合干燥缺水的肌肤。"
    )
    docs = split_markdown_by_type("ingredient", text)
    assert len(docs) == 2
    assert docs[0].metadata["section"] == "玻尿酸是什么"
    assert docs[0].page_content.startswith("玻尿酸是什么\n")
    assert "> doc_type" not in docs[0].page_content  # front matter 已剥离


def test_split_markdown_long_section_recursive():
    """超长章节按档位递归切分（保留章节名前缀）。"""
    text = "## 长文\n" + "玻尿酸具有保湿作用。" * 200
    docs = split_markdown_by_type("product_detail", text)
    assert len(docs) > 1
    assert all(d.metadata["section"] == "长文" for d in docs)
    assert all(d.page_content.startswith("长文\n") for d in docs)


def test_split_plain_text_paragraph_priority():
    """纯文本：段落优先，多段各成一块；单段超长才递归切。"""
    chunks = split_plain_text_by_type("faq", "第一段。\n\n第二段。\n\n第三段。")
    assert len(chunks) == 3
    long_para = "玻尿酸具有保湿锁水作用。" * 300
    chunks_long = split_plain_text_by_type("faq", long_para)
    assert len(chunks_long) > 1


def test_profile_mapping_covers_md_types():
    """现有 MD 的 doc_type 全部有档位（P24a 修静默兜底）。"""
    for t in ("faq", "origin_cert", "product_longdoc", "guide", "ingredient",
              "policy", "fashion_guide", "material", "metadata_doc"):
        assert t in CHUNK_PROFILES, f"missing profile: {t}"
        p = get_profile(t)
        assert p["chunk_size"] > 0
        assert p["chunk_overlap"] < p["chunk_size"]


def test_unknown_type_falls_back_to_default():
    """未知 doc_type 回退默认档位，行为不劣化。"""
    assert get_profile("not_a_real_type") == DEFAULT_PROFILE
