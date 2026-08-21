"""提示词库只读目录契约测试。"""
from agent_base.prompts import prompt_catalog, get_prompt


def test_catalog_covers_all_sections():
    items = prompt_catalog()
    sections = {it["section"] for it in items}
    for s in ["qa", "supervisor", "intent", "memory", "summary", "polish", "knowledge_ops", "improve_intent"]:
        assert s in sections, f"缺 section: {s}"


def test_catalog_items_have_content_and_meta():
    items = prompt_catalog()
    assert items, "目录为空"
    for it in items:
        assert it["content"].strip(), f"{it['section']}.{it['key']} 内容为空"
        assert it["name_zh"] and it["name_en"]


def test_get_prompt_falls_back_to_default():
    assert get_prompt("qa", "system") != ""
    assert get_prompt("nonexistent_xyz", "system", "D") == "D"


def test_polish_and_ops_prompts_load_from_yaml():
    assert "Markdown" in get_prompt("polish", "system")
    assert "{tools}" in get_prompt("knowledge_ops", "system")
    assert "{intent_name}" in get_prompt("improve_intent", "system")
