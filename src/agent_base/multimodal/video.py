"""视频知识管线：抽帧 → 视觉理解 → 文本入库（Phase 3 视频版）。

设计：
- ffmpeg 抽帧（均匀采样 ≤N 帧，缩放 ≤512px 控制视觉 token 成本）；
- 每帧走豆包视觉模型理解，输出商品名/文字/标签 JSON；
- 结果组装为「视频概述 chunk + 逐帧分镜 chunk」直入向量库，
  metadata 带 video_urls / poster_url / duration_sec，检索命中时前端可播放；
- 无 ffmpeg / 无密钥 / 解析失败均优雅降级（不阻塞主链路），
  返回 note 说明降级原因，与图片管线（multimodal/__init__.py）行为一致。
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any

# 抽帧与缩放参数（成本护栏：视频帧数 × 视觉调用数 = 单视频 LLM 成本）
MAX_FRAMES = 8          # 单视频最多抽帧数（均匀采样）
FRAME_MAX_SIDE = 512    # 帧图最长边缩放上限（px），控视觉 token


def has_ffmpeg() -> bool:
    """ffmpeg 是否可用（缺失时视频管线降级为仅标题/描述入库）。"""
    return shutil.which("ffmpeg") is not None


def _probe_duration(video_path: str) -> float:
    """ffprobe 读取视频时长（秒）；失败返回 0.0。"""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            return max(0.0, float(proc.stdout.strip() or 0))
    except Exception:
        pass
    return 0.0


def extract_frames(video_path: str, max_frames: int = MAX_FRAMES) -> list[str]:
    """ffmpeg 均匀抽帧，返回临时 PNG 路径列表（空 = 抽帧失败/无 ffmpeg）。

    Args:
        video_path: 本地视频文件路径。
        max_frames: 最多抽帧数（均匀采样）。

    Returns:
        抽帧 PNG 的绝对路径列表；失败返回空列表（调用方降级）。
    """
    if not has_ffmpeg():
        return []
    duration = _probe_duration(video_path)
    if duration <= 0:
        duration = 30.0  # 探测失败时按 30s 估算采样
    tmp_dir = tempfile.mkdtemp(prefix="video_frames_")
    n = min(max_frames, max(1, int(duration) + 1))
    frame_paths: list[str] = []
    try:
        for i in range(n):
            t = i * duration / n
            out = os.path.join(tmp_dir, f"frame_{i:03d}.png")
            # 均匀取时间点抽 1 帧，缩放最长边 ≤ FRAME_MAX_SIDE
            proc = subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video_path,
                 "-frames:v", "1", "-vf",
                 f"scale='min({FRAME_MAX_SIDE},iw)':'-2'", "-q:v", "2", out],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
                frame_paths.append(out)
    except Exception:
        pass
    return frame_paths


def _frame_to_data_uri(frame_path: str) -> str:
    """本地 PNG → base64 data URI（视觉接口输入格式）。"""
    with open(frame_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return "data:image/png;base64," + b64


def _vision_cfg() -> dict[str, Any]:
    """视觉理解模型配置（与图片管线共用 ARK 网关）。"""
    return {
        "api_key": os.getenv("ARK_API_KEY", "").strip(),
        "base_url": os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip(),
        "model": os.getenv("ARK_VISION_MODEL", "doubao-seed-2-0-lite-260428").strip(),
    }


def _call_vision_frame(target: str, cfg: dict[str, Any], frame_index: int) -> dict[str, Any]:
    """单帧视觉理解：识别商品信息，输出 JSON（OpenAI 兼容 chat/completions）。

    Args:
        target: 帧图 data URI。
        cfg: 视觉模型配置。
        frame_index: 帧序号（用于提示模型关注画面内容）。

    Returns:
        {product_name, text, tags}；失败返回空结构（单帧失败不影响其余帧）。
    """
    import urllib.request

    url = cfg["base_url"] + "/chat/completions"
    instruction = (
        "这是商品视频的第" + str(frame_index + 1) + "帧画面。"
        "识别画面中的商品与文字，输出 JSON（不要输出其他内容）："
        '{"product_name":"商品名","text":"画面中出现的全部文字与商品信息","tags":["标签1","标签2"]}'
    )
    payload = json.dumps(
        {
            "model": cfg["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": target}},
                    ],
                }
            ],
            "max_tokens": 1024,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + cfg["api_key"]},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = (data.get("choices") or []) if isinstance(data, dict) else []
    content = ""
    if choices:
        msg = choices[0].get("message") or {}
        content = str(msg.get("content") or "")
    return _parse_frame_content(content)


def _parse_frame_content(content: str) -> dict[str, Any]:
    """解析单帧模型输出（容错：非 JSON 整体作为文字）。"""
    import re as _re

    product_name, text, tags = "", "", []
    try:
        match = _re.search(r"\{.*\}", content, _re.S)
        if match:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                product_name = str(obj.get("product_name") or "").strip()
                text = str(obj.get("text") or "").strip()
                raw_tags = obj.get("tags") or []
                tags = [str(t).strip() for t in raw_tags if str(t).strip()] if isinstance(raw_tags, list) else []
    except Exception:
        pass
    if not text:
        text = content.strip()[:2000]
    return {"product_name": product_name, "text": text, "tags": tags}


def analyze_video(
    video_path: str | None,
    video_url: str | None = None,
    max_frames: int = MAX_FRAMES,
) -> dict[str, Any]:
    """视频解析入口：抽帧 → 逐帧视觉理解 → 组装描述/分镜文本。

    Args:
        video_path: 本地视频路径（优先）。
        video_url: 视频 URL（本地路径缺失时兜底展示用，不抽帧）。
        max_frames: 最大抽帧数。

    Returns:
        {ok, engine, description, scenes_text, frame_count, duration_sec,
         note, product_names, tags}；任何环节失败均降级返回，不抛异常。
    """
    cfg = _vision_cfg()
    duration = _probe_duration(video_path) if video_path else 0.0
    frames = extract_frames(video_path, max_frames) if video_path else []
    if not cfg["api_key"]:
        return {
            "ok": True,
            "engine": "mock",
            "description": "",
            "scenes_text": "",
            "frame_count": 0,
            "duration_sec": int(duration),
            "note": "未配置 ARK_API_KEY，视频未做视觉理解（仅登记元数据）",
        }
    if not frames:
        return {
            "ok": True,
            "engine": "mock",
            "description": "",
            "scenes_text": "",
            "frame_count": 0,
            "duration_sec": int(duration),
            "note": "ffmpeg 不可用或抽帧失败，视频未做视觉理解（仅登记元数据）",
        }
    scene_lines: list[str] = []
    product_names: list[str] = []
    all_tags: list[str] = []
    failed = 0
    for i, fp in enumerate(frames):
        try:
            result = _call_vision_frame(_frame_to_data_uri(fp), cfg, i)
        except Exception:
            failed += 1
            continue
        if not result.get("text"):
            failed += 1
            continue
        seg_sec = duration / max(1, len(frames))
        t0, t1 = i * seg_sec, min(duration, (i + 1) * seg_sec)
        scene_lines.append(f"[{t0:.0f}s-{t1:.0f}s] " + result["text"])
        if result.get("product_name"):
            product_names.append(result["product_name"])
        all_tags.extend(result.get("tags") or [])
    if not scene_lines:
        return {
            "ok": True,
            "engine": "mock",
            "description": "",
            "scenes_text": "",
            "frame_count": len(frames),
            "duration_sec": int(duration),
            "note": "视觉理解全部失败（密钥无效/额度不足），视频未解析",
        }
    description = " ".join(
        [p for p in [sorted(set(product_names))[0] if product_names else "",
                     " ".join(sorted(set(all_tags)))] if p]
    ).strip()
    return {
        "ok": True,
        "engine": "ark",
        "description": description or "商品视频",
        "scenes_text": "\n".join(scene_lines),
        "frame_count": len(frames),
        "duration_sec": int(duration),
        "note": "" if failed == 0 else f"{failed} 帧解析失败已跳过",
    }


def index_video_document(
    row: dict[str, Any],
    description: str,
    scenes_text: str,
    duration_sec: int,
) -> bool:
    """视频文本入库（Phase 3 视频版）：概述 + 分镜 chunk 直入向量库。

    Args:
        row: media_documents 行（id/url/product_id/source_type 等）。
        description: 视频概述（商品名 + 标签）。
        scenes_text: 逐帧分镜文本（多行，换行分隔）。
        duration_sec: 视频时长（秒）。

    Returns:
        是否入库成功（失败静默返回 False，不阻塞任务）。
    """
    try:
        import hashlib

        from agent_base.storage.documents import ingest_document_from_chunks

        media_id = int(row.get("id") or 0)
        url = str(row.get("url") or "")
        poster = str(row.get("poster_url") or "")
        video_urls = list(row.get("video_urls") or []) or ([url] if url else [])
        doc_id = f"video-{media_id}"
        chunks: list[dict[str, Any]] = []

        def _mk_chunk(idx: int, text: str) -> dict[str, Any]:
            chunk_id = hashlib.sha256((doc_id + f":chunk{idx}").encode("utf-8")).hexdigest()[:16]
            return {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "doc_id": doc_id,
                    "media_id": media_id,
                    "video_urls": video_urls,
                    "poster_url": poster,
                    "duration_sec": int(duration_sec or 0),
                    "product_id": str(row.get("product_id") or ""),
                    "source_type": "video",
                    "category": "视频知识库",
                    "image_urls": [poster] if poster else [],
                },
            }

        overview = " ".join([p for p in [description, scenes_text] if p]).strip()
        if overview:
            chunks.append(_mk_chunk(0, overview[:4000]))
        # 分镜过长时按帧段拆分，保证单 chunk 长度可控
        scenes = [s for s in (scenes_text or "").split("\n") if s.strip()]
        if len(scenes) > 1:
            buf: list[str] = []
            size = 0
            for s in scenes:
                if size + len(s) > 1800 and buf:
                    chunks.append(_mk_chunk(len(chunks), "\n".join(buf)))
                    buf, size = [], 0
                buf.append(s)
                size += len(s)
            if buf:
                chunks.append(_mk_chunk(len(chunks), "\n".join(buf)))
        if not chunks:
            return False
        from agent_base.api.main import get_runtime

        runtime = get_runtime()
        ingest_document_from_chunks(
            doc_id=doc_id,
            content="\n".join(c["text"] for c in chunks),
            chunks=chunks,
            vector_store=runtime["vector_store"],
            category="视频知识库",
            metadata=chunks[0]["metadata"],
            skip_tag_check=True,
        )
        return True
    except Exception:
        return False
