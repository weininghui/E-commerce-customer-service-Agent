"""MinerU 文档解析器：PDF/扫描件 → Markdown。

配置驱动 + mock 降级：
- 设置 ``MINERU_API_KEY`` 时调用真实解析服务（接口契约固定）；
- 未设置时走 mock：MD 原样返回 / PDF 抽取文本（若有 pypdf）或返回占位说明，
  保证无密钥也能跑通全链路。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _mineru_cfg() -> dict[str, Any]:
    return {
        "api_key": os.getenv("MINERU_API_KEY", "").strip(),
        "base_url": os.getenv(
            "MINERU_BASE_URL", "https://mineru.net/api/v4"
        ).strip(),
    }


def _mock_parse(filename: str, content: bytes) -> str:
    """无密钥时的 mock 解析：MD/TXT 原样；PDF 抽文本；Office/HTML 本地抽取；其余占位说明。"""
    suffix = Path(filename).suffix.lower()
    if suffix in {".md", ".txt"}:
        return content.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            pages = []
            for page in reader.pages[:50]:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    continue
            text = "\n\n".join(p for p in pages if p.strip())
            if text.strip():
                return text
        except Exception:
            pass
        return (
            f"# {Path(filename).stem}\n\n"
            "> 文档由 MinerU 解析（mock 模式）：当前未配置 MINERU_API_KEY，"
            "PDF 文本抽取不可用，请补充解析结果后入库。\n"
        )
    if suffix == ".docx":
        return _extract_docx_text(content) or _mock_placeholder(filename, suffix)
    if suffix == ".pptx":
        return _extract_pptx_text(content) or _mock_placeholder(filename, suffix)
    if suffix == ".xlsx":
        return _extract_xlsx_text(content) or _mock_placeholder(filename, suffix)
    if suffix in {".html", ".htm"}:
        text = _strip_html(content.decode("utf-8", errors="ignore"))
        return text or _mock_placeholder(filename, suffix)
    return _mock_placeholder(filename, suffix)


def _mock_placeholder(filename: str, suffix: str) -> str:
    """mock 无法抽取时的明确占位（不把二进制当文本）。"""
    return (
        f"# {Path(filename).stem}\n\n"
        "> 文档由 MinerU 解析（mock 模式）：当前未配置 MINERU_API_KEY，"
        f"该格式（{suffix or "未知"}）本地无法抽取文本，请配置密钥后重新上传。\n"
    )


def _extract_docx_text(content: bytes) -> str:
    """mock：docx 是 zip，抽取 word/document.xml 文本。"""
    try:
        import io
        import re
        import zipfile

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        xml = re.sub(r"</w:p>", "\n", xml)
        xml = re.sub(r"</w:tc>", "\t", xml)
        text = re.sub(r"<[^>]+>", "", xml)
        return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())
    except Exception:
        return ""


def _extract_pptx_text(content: bytes) -> str:
    """mock：pptx 是 zip，抽取每页 slide XML 文本。"""
    try:
        import io
        import re
        import zipfile

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = sorted(n for n in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml", n))
            parts = []
            for n in names:
                xml = zf.read(n).decode("utf-8", errors="ignore")
                xml = re.sub(r"</a:p>", "\n", xml)
                parts.append(re.sub(r"<[^>]+>", "", xml))
            text = "\n".join(parts)
            return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())
    except Exception:
        return ""


def _extract_xlsx_text(content: bytes) -> str:
    """mock：xlsx 是 zip，抽取 sharedStrings 文本。"""
    try:
        import io
        import re
        import zipfile

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            if "xl/sharedStrings.xml" not in zf.namelist():
                return ""
            xml = zf.read("xl/sharedStrings.xml").decode("utf-8", errors="ignore")
        cells = re.findall(r"<t[^>]*>(.*?)</t>", xml, re.S)
        return " ".join(c.strip() for c in cells if c.strip())
    except Exception:
        return ""


def _strip_html(text: str) -> str:
    """mock：HTML 去脚本/样式后剥标签。"""
    try:
        import re

        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())
    except Exception:
        return text


def _call_mineru_api(filename: str, content: bytes, cfg: dict[str, Any]) -> str:
    """调用真实 MinerU 解析 API（官方流程：申请上传链接 → PUT 上传 → 轮询 → 下载 zip 取 full.md）。"""
    import json
    import time
    import urllib.request

    base_url = cfg["base_url"].rstrip("/")
    auth = {"Authorization": f"Bearer {cfg['api_key']}"}

    # 1. 申请文件上传链接（batch_id + file_urls）
    batch_url = f"{base_url}/file-urls/batch"
    body = json.dumps(
        {"files": [{"name": filename, "data_id": f"kb-{int(time.time())}"}], "model_version": "vlm"}
    ).encode("utf-8")
    req = urllib.request.Request(
        batch_url,
        data=body,
        headers={**auth, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        batch_resp = json.loads(resp.read().decode("utf-8"))
    if batch_resp.get("code") != 0:
        raise RuntimeError(f"MinerU apply upload url failed: {batch_resp}")
    data = batch_resp.get("data") or {}
    batch_id = str(data.get("batch_id") or "")
    file_urls = data.get("file_urls") or []
    if not batch_id or not file_urls:
        raise RuntimeError(f"MinerU no upload url: {batch_resp}")

    # 2. PUT 上传文件（不设置 Content-Type）
    upload_url = str(file_urls[0])
    # OSS 预签名链接要求 PUT 时不携带 Content-Type（urllib 会自动添加，
    # 会导致签名校验失败返回 403），因此改用 http.client 裸 PUT。
    import http.client
    from urllib.parse import urlsplit

    parsed = urlsplit(upload_url)
    conn_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_cls(parsed.netloc, timeout=120)
    try:
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        conn.putrequest("PUT", path, skip_accept_encoding=True)
        conn.putheader("Host", parsed.netloc)
        conn.putheader("Content-Length", str(len(content)))
        conn.endheaders()
        conn.send(content)
        put_resp = conn.getresponse()
        put_resp.read()
        if put_resp.status != 200:
            raise RuntimeError(f"MinerU upload failed: HTTP {put_resp.status}")
    finally:
        conn.close()

    # 3. 轮询批量解析结果
    poll_url = f"{base_url}/extract-results/batch/{batch_id}"
    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(5)
        poll_req = urllib.request.Request(poll_url, headers=auth, method="GET")
        try:
            with urllib.request.urlopen(poll_req, timeout=60) as resp2:
                state = json.loads(resp2.read().decode("utf-8"))
        except Exception:
            continue
        if state.get("code") != 0:
            continue
        sdata = state.get("data") or {}
        results = sdata.get("extract_result") or []
        item = results[0] if results else {}
        status = str(item.get("state") or "")
        if status == "done":
            zip_url = str(item.get("full_zip_url") or "")
            if not zip_url:
                # 部分响应直接内联 markdown
                return str(item.get("markdown") or item.get("content") or "")
            return _download_zip_markdown(zip_url)
        if status == "failed":
            raise RuntimeError(f"MinerU parse failed: {item.get('err_msg') or state}")
    raise RuntimeError("MinerU parse timeout")


def _download_zip_markdown(zip_url: str) -> str:
    """下载结果 zip 并提取 full.md。"""
    import io
    import urllib.request
    import zipfile

    with urllib.request.urlopen(zip_url, timeout=120) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        md_names = [n for n in zf.namelist() if n.endswith(".md")]
        if not md_names:
            return ""
        # 优先 full.md，其次第一个 md
        pick = "full.md" if "full.md" in md_names else md_names[0]
        return zf.read(pick).decode("utf-8", errors="ignore")


def parse_document(filename: str, content: bytes) -> dict[str, Any]:
    """解析上传文档为 Markdown 文本。

    Returns:
        {"text": str, "engine": "mineru" | "mock", "ok": bool}
    """
    cfg = _mineru_cfg()
    if cfg["api_key"]:
        try:
            text = _call_mineru_api(filename, content, cfg)
            if text.strip():
                return {"text": text, "engine": "mineru", "ok": True}
        except Exception:
            pass
    text = _mock_parse(filename, content)
    return {"text": text, "engine": "mock", "ok": True}
