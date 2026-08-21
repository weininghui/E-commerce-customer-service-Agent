"""多模态封装：ARK 商品图生成（配置驱动 + mock 降级）。"""

from __future__ import annotations

import base64
import os
from typing import Any


def _ark_cfg() -> dict[str, Any]:
    return {
        "api_key": os.getenv("ARK_API_KEY", "").strip(),
        "base_url": os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip(),
        "model": os.getenv("ARK_MODEL", "doubao-seed-2-0-lite-260428").strip(),
    }


def _mock_image(prompt: str) -> dict[str, Any]:
    """无密钥时的 mock 占位图（SVG data URI）。"""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">'
        '<rect width="512" height="512" fill="#F3F4F6"/>'
        '<text x="256" y="240" font-size="20" text-anchor="middle" fill="#6B7280" '
        'font-family="sans-serif">商品图（mock）</text>'
        '<text x="256" y="280" font-size="14" text-anchor="middle" fill="#9CA3AF" '
        'font-family="sans-serif">请配置 ARK_API_KEY 生成真实图片</text>'
        "</svg>"
    )
    data_uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return {
        "ok": True,
        "engine": "mock",
        "image": data_uri,
        "prompt": prompt,
    }


def _call_ark(prompt: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """调用火山方舟生图接口（OpenAI 兼容 images/generations）。"""
    import json
    import urllib.request

    url = f"{cfg['base_url']}/images/generations"
    payload = json.dumps(
        {"model": cfg["model"], "prompt": prompt, "size": "1024x1024", "n": 1}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    images = (data.get("data") or []) if isinstance(data, dict) else []
    if not images:
        raise RuntimeError(f"ARK image response empty: {data}")
    first = images[0]
    # 返回 b64_json 或 url
    b64 = first.get("b64_json")
    if b64:
        return {
            "ok": True,
            "engine": "ark",
            "image": "data:image/png;base64," + b64,
            "prompt": prompt,
        }
    image_url = first.get("url") or first.get("image_url") or ""
    if image_url:
        return {"ok": True, "engine": "ark", "image": image_url, "prompt": prompt}
    raise RuntimeError(f"ARK image no content: {first}")


def generate_product_image(prompt: str) -> dict[str, Any]:
    """生成商品图：配置密钥则真调 ARK，否则 mock 占位。"""
    cfg = _ark_cfg()
    if cfg["api_key"]:
        try:
            return _call_ark(prompt, cfg)
        except Exception as exc:
            result = _mock_image(prompt)
            result["error"] = str(exc)[:200]
            return result
    return _mock_image(prompt)


def _normalize_image_input(image: str | None, image_url: str | None) -> str:
    """把图生图输入统一为 data URI（裸 base64 自动补前缀）。"""
    if image:
        if image.startswith("data:image/"):
            return image
        if image.startswith("http://") or image.startswith("https://"):
            return image
        return "data:image/png;base64," + image
    return (image_url or "").strip()


def _mock_image_edit(prompt: str, image: str | None, image_url: str | None) -> dict[str, Any]:
    """无密钥时的图生图 mock：有参考图则原样返回（保留输入），无参考图退化占位图。"""
    if image or image_url:
        return {
            "ok": True,
            "engine": "mock",
            "image": _normalize_image_input(image, image_url),
            "prompt": prompt,
            "note": "未配置 ARK_API_KEY，参考图原样返回（未执行编辑）",
        }
    result = _mock_image(prompt)
    result["note"] = "未配置 ARK_API_KEY，且未提供参考图（退化为文生图占位）"
    return result


def _call_ark_edit(prompt: str, image: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """调用火山方舟图生图接口（OpenAI 兼容 images/edits）。"""
    import json
    import urllib.request

    url = f"{cfg['base_url']}/images/edits"
    payload = json.dumps(
        {"model": cfg["model"], "prompt": prompt, "image": image, "size": "1024x1024", "n": 1}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    images = (data.get("data") or []) if isinstance(data, dict) else []
    if not images:
        raise RuntimeError(f"ARK image edit response empty: {data}")
    first = images[0]
    b64 = first.get("b64_json")
    if b64:
        return {
            "ok": True,
            "engine": "ark",
            "image": "data:image/png;base64," + b64,
            "prompt": prompt,
        }
    image_out = first.get("url") or first.get("image_url") or ""
    if image_out:
        return {"ok": True, "engine": "ark", "image": image_out, "prompt": prompt}
    raise RuntimeError(f"ARK image edit no content: {first}")


def edit_product_image(
    prompt: str,
    image: str | None = None,
    image_url: str | None = None,
) -> dict[str, Any]:
    """图生图编辑商品图：在参考图基础上按 prompt 修改（ARK images/edits）。

    Args:
        prompt: 编辑指令（如"把包装换成蓝色，背景换成纯白电商主图"）。
        image: 参考图 data URI / 裸 base64 / URL（优先级高于 image_url）。
        image_url: 参考图 URL（image 未提供时使用）。

    Returns:
        同 generate_product_image 结构；无密钥时原样返回参考图（mock），
        既无密钥又无参考图时退化为文生图占位。
    """
    cfg = _ark_cfg()
    normalized = _normalize_image_input(image, image_url)
    if not normalized:
        return _mock_image_edit(prompt, None, None)
    if cfg["api_key"]:
        try:
            return _call_ark_edit(prompt, normalized, cfg)
        except Exception as exc:
            result = _mock_image_edit(prompt, image, image_url)
            result["error"] = str(exc)[:200]
            return result
    return _mock_image_edit(prompt, image, image_url)


def _vision_cfg() -> dict[str, Any]:
    """视觉理解模型配置（与生图共用 ARK 网关，模型名独立）。"""
    return {
        "api_key": os.getenv("ARK_API_KEY", "").strip(),
        "base_url": os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip(),
        "model": os.getenv("ARK_VISION_MODEL", "doubao-seed-2-0-lite-260428").strip(),
    }


def _mock_analyze() -> dict[str, Any]:
    """视觉理解 mock：无密钥/调用失败时返回空解析（不阻塞主链路）。"""
    return {
        "ok": True,
        "engine": "mock",
        "description": "",
        "ocr_text": "",
        "note": "未配置视觉模型或调用失败，图片文字内容未解析",
    }


def _parse_vision_content(content: str) -> dict[str, Any]:
    """从模型输出中抽取结构化字段（容错：非 JSON 输出整体作为文字）。"""
    import json as _json
    import re as _re

    product_name, text, tags = "", "", []
    try:
        match = _re.search(r"\{.*\}", content, _re.S)
        if match:
            obj = _json.loads(match.group(0))
            if isinstance(obj, dict):
                product_name = str(obj.get("product_name") or "").strip()
                text = str(obj.get("text") or "").strip()
                raw_tags = obj.get("tags") or []
                tags = [str(t).strip() for t in raw_tags if str(t).strip()] if isinstance(raw_tags, list) else []
    except Exception:
        pass
    if not text:
        text = content.strip()[:2000]
    description = " ".join([p for p in [product_name, " ".join(tags)] if p]).strip()
    return {"ok": True, "engine": "ark", "description": description, "ocr_text": text}


def _call_vision(target: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """调用豆包视觉模型做图片理解（OpenAI 兼容 chat/completions + image_url）。

    target 支持 http(s) URL 或 data URI（本地图片走 base64 data URI）。
    """
    import json
    import urllib.request

    url = cfg["base_url"] + "/chat/completions"
    instruction = "识别这张商品图，输出 JSON（不要输出其他内容）：{\"product_name\":\"商品名\",\"text\":\"图片中出现的全部文字\",\"tags\":[\"标签1\",\"标签2\"]}"
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
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + cfg["api_key"],
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = (data.get("choices") or []) if isinstance(data, dict) else []
    content = ""
    if choices:
        msg = choices[0].get("message") or {}
        content = str(msg.get("content") or "")
    return _parse_vision_content(content)


def analyze_media_image(
    *,
    image: str | None = None,
    image_url: str | None = None,
    image_path: str | None = None,
) -> dict[str, Any]:
    """图片理解（OCR/视觉结构化）。

    Args:
        image: 图片 data URI / 裸 base64（优先级最高）。
        image_url: 图片 http(s) URL。
        image_path: 本地文件路径（读取后转 base64 data URI，供云端模型识别本地图）。

    Returns:
        {"ok", "engine", "description", "ocr_text"}；无密钥/调用失败 → mock 空解析。
    """
    from pathlib import Path as _Path

    cfg = _vision_cfg()
    if image is None and image_path:
        try:
            raw = _Path(image_path).read_bytes()
            suffix = _Path(image_path).suffix.lower().lstrip(".") or "png"
            if suffix == "jpg":
                suffix = "jpeg"
            image = f"data:image/{suffix};base64," + base64.b64encode(raw).decode("ascii")
        except Exception:
            image = None
    target = image or image_url
    if not target:
        return _mock_analyze()
    result: dict[str, Any] | None = None
    if cfg["api_key"]:
        try:
            result = _call_vision(target, cfg)
        except Exception as exc:
            result = _mock_analyze()
            result["error"] = str(exc)[:200]
    if result is None:
        result = _mock_analyze()
    # MinerU 兜底：视觉模型未开通/失败时，图片走 MinerU OCR 提取文字，
    # 保证文字入库可检索（引擎记为 mineru，mock 时保持原样）
    if result.get("engine") != "ark" and image_path and not (result.get("ocr_text") or "").strip():
        try:
            from agent_base.ingest.mineru_parser import parse_document

            p = _Path(image_path)
            mr = parse_document(p.name, p.read_bytes())
            if mr.get("engine") == "mineru" and (mr.get("text") or "").strip():
                result["ocr_text"] = str(mr["text"]).strip()
                result["engine"] = "mineru"
                result.pop("note", None)
                result.pop("error", None)
        except Exception:
            pass
    return result
