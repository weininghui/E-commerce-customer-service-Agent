"""批量入库 md 文档（RAG 数据扩充）。

扫描目录下所有 .md → ingest_document（PG 真相源 + Qdrant 向量 + 摘要索引），
幂等：doc_id（文件名）已存在则跳过，重复执行安全。

用法：
    python scripts/ingest/ingest_md_batch.py [目录] [--doc-type product_longdoc]

默认目录 data/ecommerce/md（其他 AI 生成的文档放这里即可批量入库）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="批量入库 md 文档到 RAG")
    parser.add_argument("dir", nargs="?", default="data/ecommerce/md", help="md 文档目录")
    parser.add_argument("--doc-type", default="product_longdoc", help="文档类型（决定切分档位）")
    parser.add_argument("--category", default="商品长文", help="文档分类")
    args = parser.parse_args()

    from agent_base.api.main import get_runtime
    from agent_base.storage.documents import ingest_document
    from agent_base.storage.pg import _conn

    target = Path(args.dir)
    if not target.is_dir():
        print(f"目录不存在: {target}")
        sys.exit(1)

    runtime = get_runtime()
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT doc_id FROM documents WHERE deleted_at IS NULL")
        existing = {str(r[0]) for r in cur.fetchall()}
        # 原始 14 份文档 doc_id 是 hash、doc_name 才是文件名——按 doc_name 也判重，
        # 避免同一文件重复入库两份
        cur.execute(
            "SELECT DISTINCT metadata->>'doc_name' FROM documents "
            "WHERE deleted_at IS NULL AND metadata ? 'doc_name'"
        )
        existing_names = {str(r[0]) for r in cur.fetchall()}

    files = sorted(target.glob("*.md"))
    if not files:
        print(f"目录下没有 md 文件: {target}")
        return

    ok = skip = fail = 0
    for i, f in enumerate(files, 1):
        doc_id = f.name
        if doc_id in existing or f.name in existing_names:
            skip += 1
            continue
        try:
            result = ingest_document(
                doc_id=doc_id,
                content=f.read_text(encoding="utf-8"),
                vector_store=runtime["vector_store"],
                category=args.category,
                summary_store=runtime.get("summary_store"),
                skip_tag_check=True,  # 批量导入豁免精审（与 seed 脚本一致）
                doc_type=args.doc_type,
                filename=f.name,
            )
            ok += 1
            existing.add(doc_id)
            print(f"[{i}/{len(files)}] +{result.get('chunk_count', '?')} chunks  {doc_id}")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[{i}/{len(files)}] FAIL {doc_id}: {exc}")

    print(f"\n完成：新增 {ok}，跳过 {skip}（已存在），失败 {fail}，共 {len(files)}")


if __name__ == "__main__":
    main()
