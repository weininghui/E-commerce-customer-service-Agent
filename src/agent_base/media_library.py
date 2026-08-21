"""图片知识库业务层（Phase 2）：上传校验 / 落盘 / 入库 / 绑定 / 删除。

- 图片文件存本地 data/media/uploads/（可被 AGENT_BASE_MEDIA_DIR 环境变量覆盖）；
  后续换对象存储时只改 URL 生成逻辑，业务接口不变。
- 元数据入 PG media_documents（独立于商品素材表 product_media）。
- 全部接口仅管理端可用（鉴权在 api/main.py 层）。
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

# 图片类型白名单 + 大小上限（8MB）；视频白名单 + 上限（50MB）
MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
ALL_MEDIA_EXTENSIONS = MEDIA_EXTENSIONS | VIDEO_EXTENSIONS
MAX_MEDIA_BYTES = 8 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024


def media_upload_dir() -> Path:
    """图片上传目录（默认 <项目根>/data/media/uploads）。"""
    root = Path(os.getenv("AGENT_BASE_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
    return Path(os.getenv("AGENT_BASE_MEDIA_DIR", str(root / "data" / "media" / "uploads")))


def _safe_stored_name(original_name: str) -> str:
    """生成安全的落盘文件名：uuid 前缀 + 安全化原名 + 合法扩展名，杜绝路径穿越。"""
    stem = Path(original_name or "image").stem
    # UNICODE 词字符（含中文）保留，只替换路径分隔符/控制符等危险字符
    safe = re.sub(r"[^\w.\-]", "_", stem, flags=re.UNICODE)[:64] or "image"
    suffix = Path(original_name or "").suffix.lower()
    ext = suffix if suffix in ALL_MEDIA_EXTENSIONS else ".png"
    return f"{uuid.uuid4().hex}_{safe}{ext}"


def _media_kind(original_name: str) -> tuple[str, str]:
    """按扩展名判定媒体类型；未知返回 ("", "")。

    Returns:
        (mime_type, parse_type)：如 ("image/png", "image") / ("video/mp4", "video")。
    """
    suffix = Path(original_name or "").suffix.lower()
    if suffix in MEDIA_EXTENSIONS:
        return f"image/{suffix.lstrip('.')}", "image"
    if suffix in VIDEO_EXTENSIONS:
        return f"video/{suffix.lstrip('.')}", "video"
    return "", ""


def validate_media_upload(original_name: str, content: bytes) -> str:
    """校验上传图片/视频：类型白名单 + 非空 + 大小上限；通过返回存储名。

    Raises:
        ValueError: 校验失败（调用方转 4xx）。
    """
    suffix = Path(original_name or "").suffix.lower()
    if suffix not in ALL_MEDIA_EXTENSIONS:
        supported = ", ".join(sorted(ALL_MEDIA_EXTENSIONS))
        raise ValueError(f"暂不支持该媒体类型，请上传：{supported}")
    if not content:
        raise ValueError("上传文件为空。")
    limit = MAX_VIDEO_BYTES if suffix in VIDEO_EXTENSIONS else MAX_MEDIA_BYTES
    if len(content) > limit:
        raise ValueError(
            f"文件超过 {limit // (1024 * 1024)}MB 限制。"
        )
    return _safe_stored_name(original_name)


def save_media_bytes(stored_name: str, content: bytes) -> str:
    """落盘图片字节，返回访问 URL（/media/uploads/<stored_name>）。"""
    target = media_upload_dir() / stored_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return f"/media/uploads/{stored_name}"


def handle_media_upload(
    original_name: str, content: bytes, *, description: str = "", source_type: str = ""
) -> dict[str, Any]:
    """上传图片/视频全流程：校验 → 落盘 → 建记录（parse_type 按扩展名判定）。

    Args:
        original_name: 原始文件名（决定媒体类型）。
        content: 文件字节。
        description: 人工描述（可选）。
        source_type: 来源标记（upload/collect 等；视频默认 "video"）。

    Returns:
        {ok, id, url, original_name, size_bytes, status, parse_type}。
    """
    from agent_base.storage.pg import media_document_create

    stored_name = validate_media_upload(original_name, content)
    mime_type, parse_type = _media_kind(original_name)
    url = save_media_bytes(stored_name, content)
    if not source_type:
        source_type = "video" if parse_type == "video" else "upload"
    media_id = media_document_create(
        original_name=original_name,
        url=url,
        mime_type=mime_type,
        size_bytes=len(content),
        source_type=source_type,
        description=description,
        parse_type=parse_type,
    )
    if not media_id:
        # 入库失败回收磁盘文件，避免孤儿文件
        try:
            (media_upload_dir() / stored_name).unlink(missing_ok=True)
        except Exception:
            pass
        raise ValueError("媒体记录写入失败（数据库不可用）")
    return {
        "ok": True,
        "id": media_id,
        "url": url,
        "original_name": original_name,
        "size_bytes": len(content),
        "status": "pending",
        "parse_type": parse_type,
    }


def handle_media_delete(media_id: int) -> dict[str, Any]:
    """删除图片：先删记录，再清理磁盘文件（仅限本目录内的相对文件名）。"""
    from agent_base.storage.pg import media_document_delete

    row = media_document_delete(int(media_id))
    if not row:
        return {"ok": False, "id": int(media_id)}
    url = str(row.get("url") or "")
    prefix = "/media/uploads/"
    if url.startswith(prefix):
        stored = url[len(prefix):]
        # 防路径穿越：只允许单段文件名
        if stored and "/" not in stored and "\\" not in stored:
            try:
                (media_upload_dir() / stored).unlink(missing_ok=True)
            except Exception:
                pass
    # Phase 3：同步清理该媒体的索引文档（media-<id> / video-<id>），避免幽灵检索结果
    try:
        from agent_base.api.main import _delete_document_core

        _delete_document_core(f"media-{int(media_id)}")
        _delete_document_core(f"video-{int(media_id)}")
    except Exception:
        pass
    return {"ok": True, "id": row.get("id"), "url": url}


def media_file_path(url: str) -> Path | None:
    """把 /media/uploads/<name> 转成本地绝对路径（解析任务读取文件用）。"""
    prefix = "/media/uploads/"
    if not url.startswith(prefix):
        return None
    stored = url[len(prefix):]
    if not stored or "/" in stored or "\\" in stored:
        return None
    return media_upload_dir() / stored
