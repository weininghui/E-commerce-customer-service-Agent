"""批量导入媒体文件到媒体知识库（图片 + 视频），并自动入队解析任务。

用法（在项目根目录，配置好 .env 后）：
    python scripts/ingest/ingest_media_batch.py --folder data/collect/xhs_pics --product-id P001
    python scripts/ingest/ingest_media_batch.py --csv data/collect/media_manifest.csv
    python scripts/ingest/ingest_media_batch.py --folder data/collect/xhs --dry-run

CSV 格式（表头可选）：
    path,product_id,description,source_type
    data/collect/a.mp4,P001,白色纯棉T恤 视频,collect

说明：
- 图片走 OCR/视觉管线（media_parse），视频走抽帧+视觉管线（media_parse_video）；
- 同名文件已入库则跳过（幂等），避免重复导入；
- 解析任务异步执行（PG 任务队列 + worker 自动消费），本脚本只负责入库+入队；
- --dry-run 只打印将导入的文件清单，不落盘不入库。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def _resolve_root() -> Path:
    """项目根目录（scripts/ingest/ 的上上级）。"""
    return Path(__file__).resolve().parents[2]


def _load_env(root: Path) -> None:
    """加载 .env（若存在），与 docker-compose 行为一致。"""
    from dotenv import load_dotenv

    load_dotenv(root / ".env")


def _existing_names() -> set[str]:
    """已入库的 original_name 集合（幂等去重用）。"""
    from agent_base.storage.pg import media_document_list

    names: set[str] = set()
    page = media_document_list(limit=500)
    names.update(str(x.get("original_name") or "") for x in page)
    return names


def _iter_manifest(args: argparse.Namespace):
    """遍历待导入清单：--folder 扫目录 / --csv 读清单，产出 (path, product_id, description, source_type)。"""
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise SystemExit(f"CSV 不存在：{csv_path}")
        with open(csv_path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
            if not rows:
                raise SystemExit("CSV 为空（或缺少表头）")
            for r in rows:
                p = (r.get("path") or "").strip()
                if not p:
                    continue
                yield (
                    Path(p),
                    (r.get("product_id") or "").strip() or args.product_id,
                    (r.get("description") or "").strip(),
                    (r.get("source_type") or "").strip() or args.source_type,
                )
        return
    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"目录不存在：{folder}")
    exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov", ".webm", ".mkv", ".avi"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield (p, args.product_id, args.description, args.source_type)


def main() -> int:
    parser = argparse.ArgumentParser(description="批量导入媒体文件（图片/视频）到媒体知识库并入队解析")
    parser.add_argument("--folder", default="", help="媒体文件目录（扫描支持的图片/视频扩展名）")
    parser.add_argument("--csv", default="", help="媒体清单 CSV（path,product_id,description,source_type）")
    parser.add_argument("--product-id", default="", help="默认绑定商品 ID（CSV 未指定时使用）")
    parser.add_argument("--description", default="", help="默认描述（CSV 未指定时使用）")
    parser.add_argument("--source-type", default="collect", help="来源标记（默认 collect，区别于手动 upload）")
    parser.add_argument("--dry-run", action="store_true", help="只打印清单不落盘不入库")
    parser.add_argument("--enqueue", action="store_true", help="入库后入队解析任务（默认开启，--no-enqueue 关闭）")
    parser.add_argument("--no-enqueue", dest="enqueue", action="store_false", help="入库但不入队解析")
    parser.set_defaults(enqueue=True)
    args = parser.parse_args()
    if not args.folder and not args.csv:
        parser.error("必须提供 --folder 或 --csv")

    root = _resolve_root()
    sys.path.insert(0, str(root / "src"))
    _load_env(root)

    manifest = list(_iter_manifest(args))
    if not manifest:
        print("未发现可导入的媒体文件。")
        return 0
    print(f"待导入 {len(manifest)} 个文件" + ("（dry-run，不落盘）" if args.dry_run else ""))

    if args.dry_run:
        for p, pid, desc, src in manifest:
            print(f"  [DRY] {p.name}  product_id={pid or '-'}  source_type={src}")
        return 0

    from agent_base.media_library import handle_media_upload
    from agent_base.storage.pg import task_enqueue

    existing = _existing_names()
    imported, skipped, failed = 0, 0, 0
    for p, pid, desc, src in manifest:
        if p.name in existing:
            print(f"  [SKIP] {p.name}（已入库）")
            skipped += 1
            continue
        try:
            content = p.read_bytes()
            payload = handle_media_upload(
                p.name, content, description=desc or "", source_type=src or "collect"
            )
        except ValueError as exc:
            print(f"  [FAIL] {p.name}: {exc}")
            failed += 1
            continue
        media_id = int(payload.get("id") or 0)
        parse_type = str(payload.get("parse_type") or "image")
        if pid:
            from agent_base.storage.pg import media_document_bind

            media_document_bind(media_id, pid)
        if args.enqueue:
            task_type = "media_parse_video" if parse_type == "video" else "media_parse"
            task_id = task_enqueue(task_type, {"media_id": media_id}, owner="batch")
            if not task_id:
                print(f"  [WARN] {p.name}: 入库成功但任务入队失败（worker 可后续手动触发）")
        imported += 1
        print(f"  [OK] {p.name}  id={media_id}  {parse_type}  → {task_type if args.enqueue else '未入队'}")
        existing.add(p.name)
    print(f"\n完成：导入 {imported}，跳过 {skipped}，失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
