"""文件清洗工作台（两段式知识入库流水线 · 第一段）。

上传外部文件（MD/PDF/Word/PPT/Excel/HTML/TXT/EPUB）→ 解析清洗
（MinerU 真调 / 本地兜底）→ 草稿供人工查看修改 → 显式推送到知识入库
（staging 精审队列）。清洗结果绝不自动入库。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CLEAN_EXTENSIONS = {".md", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".txt", ".epub"}
MAX_CLEAN_BYTES = 10 * 1024 * 1024


def validate_clean_upload(original_name: str, content: bytes) -> str:
    """校验上传文件：格式白名单 + 非空 + 大小上限；通过返回后缀。

    Raises:
        ValueError: 校验失败（调用方转 4xx）。
    """
    suffix = Path(original_name or "").suffix.lower()
    if suffix not in CLEAN_EXTENSIONS:
        supported = ", ".join(sorted(CLEAN_EXTENSIONS))
        raise ValueError(f"暂不支持该文件类型，请上传：{supported}")
    if not content:
        raise ValueError("上传文件为空。")
    if len(content) > MAX_CLEAN_BYTES:
        raise ValueError("文件超过 10MB 限制。")
    return suffix


def parse_clean_text(original_name: str, content: bytes) -> tuple[str, str]:
    """按格式分流解析：MD/TXT 直接解码；其余走 MinerU 解析清洗。返回 (text, engine)。"""
    suffix = Path(original_name or "").suffix.lower()
    if suffix in {".md", ".txt"}:
        for enc in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return content.decode(enc), "direct"
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="ignore"), "direct"
    from agent_base.ingest.mineru_parser import parse_document

    parsed = parse_document(original_name, content)
    return parsed["text"], parsed.get("engine", "mock")


def handle_clean_upload(original_name: str, content: bytes) -> dict[str, Any]:
    """上传 → 解析清洗 → 存草稿（不入库）。"""
    from agent_base.storage.pg import clean_draft_create

    validate_clean_upload(original_name, content)
    text, engine = parse_clean_text(original_name, content)
    if not text.strip():
        raise ValueError("文件已解析，但没有可清洗文本。")
    draft_id = clean_draft_create(
        original_name=original_name,
        engine=engine,
        raw_text=text,
    )
    if not draft_id:
        raise ValueError("清洗草稿写入失败（数据库不可用）")
    return {
        "ok": True,
        "id": draft_id,
        "filename": original_name,
        "engine": engine,
        "text_len": len(text),
        "text": text,
        "status": "pending",
    }


def handle_clean_update(draft_id: int, text: str) -> bool:
    """保存人工清洗后的文本。"""
    from agent_base.storage.pg import clean_draft_update

    return clean_draft_update(draft_id, text[:200000])


def polish_clean_text(text: str) -> str:
    """AI 格式整理：把清洗文本整理成规范的知识库 Markdown。

    Raises:
        RuntimeError: LLM 未配置/返回为空。
    """
    from agent_base.config import deep_get, load_yaml
    from agent_base.llms import build_chat_model

    cfg = load_yaml("configs/app.yaml") or {}
    llm_cfg = deep_get(cfg, "llm") or {}
    # 只透传 build_chat_model 接受的参数（配置里可能混有 temperature_by_intent 等扩展键）
    allowed = {"provider", "model", "base_url", "api_key_env", "temperature", "timeout", "max_retries"}
    model_kwargs = {k: v for k, v in llm_cfg.items() if k in allowed}
    model = build_chat_model(**model_kwargs, tracking_agent="clean_polish", tracking_source="file_cleaning")
    if model is None:
        raise RuntimeError("LLM 未配置，无法整理格式")
    from agent_base.prompts import get_prompt

    _DEFAULT_POLISH = (
        "你是电商客服知识库的文档整理助手。请把下面的清洗文本整理成规范的 Markdown 知识文档：\n"
        "要求：\n"
        "1) 用 # 标题和 ## 小节组织（参考：## 产品信息 / ## 使用方法 / ## 注意事项 / ## 成分）；\n"
        "2) 保留全部事实信息，不得增删、不得编造；\n"
        "3) 条目用列表或表格表达；\n"
        "4) 清除 OCR 噪音、多余空行与无意义字符；\n"
        "5) 只输出整理后的 Markdown，不要任何解释。"
    )
    prompt = get_prompt("polish", "system", _DEFAULT_POLISH) + f"\n\n清洗文本：\n{text[:8000]}"
    resp = model.invoke(prompt)
    out = str(getattr(resp, "content", "") or resp or "").strip()
    if not out:
        raise RuntimeError("LLM 返回为空")
    return out


def handle_clean_polish(draft_id: int) -> dict[str, Any]:
    """AI 整理草稿格式：整理结果写回 cleaned_text，返回新文本。"""
    from agent_base.storage.pg import clean_draft_get, clean_draft_update

    draft = clean_draft_get(draft_id)
    if not draft:
        raise ValueError("清洗草稿不存在")
    source = (draft.get("cleaned_text") or draft.get("raw_text") or "").strip()
    if not source:
        raise ValueError("草稿内容为空，无法整理")
    polished = polish_clean_text(source)
    clean_draft_update(draft_id, polished)
    return {"ok": True, "id": draft_id, "polished": polished, "polished_len": len(polished)}


def handle_clean_push(draft_id: int, category: str = "上传文档") -> dict[str, Any]:
    """把清洗后的文本推送到知识入库（staging 精审队列）。"""
    from agent_base.storage.pg import clean_draft_get, clean_draft_set_status

    draft = clean_draft_get(draft_id)
    if not draft:
        raise ValueError("清洗草稿不存在")
    text = (draft.get("cleaned_text") or draft.get("raw_text") or "").strip()
    if not text:
        raise ValueError("草稿内容为空，无法推送")
    from agent_base.storage.staging import stage_uploaded_document

    payload = stage_uploaded_document(
        filename=draft.get("original_name") or "cleaned-upload",
        content=text,
        category=(category or "上传文档").strip() or "上传文档",
    )
    clean_draft_set_status(draft_id, "pushed")
    payload["clean_id"] = draft_id
    return payload