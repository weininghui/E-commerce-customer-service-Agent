"""Per-chunk LLM summary generator (P18-01, productionised P18b).

ThreadPoolExecutor-based concurrent summarization with per-item
official retry (``Runnable.with_retry``), config-driven concurrency,
and progress logging.  Never uses ``chain.batch`` — a single failure
must not abort the batch.

Config (``configs/app.yaml``):
  ``retrieval.summary_index.enabled`` (default false)
  ``retrieval.summary_index.summary_concurrency`` (default 50, max 500)
  ``retrieval.summary_index.summary_timeout`` (default 60, seconds)
  ``retrieval.summary_index.summary_max_retries`` (default 2)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

# P18 摘要 system prompt——完整版以 configs/prompts_ecommerce.yaml summary.system 为准（单一真相源），
# 此处仅为 YAML 缺失时的降级兜底（不重复完整 prompt，避免双份维护）。
SUMMARY_SYSTEM = "你是电商商品资料的检索摘要生成器，输出精炼摘要。"


def _load_summary_system_prompt() -> str:
    """加载摘要 system prompt。

    P26a 接线：优先读 configs/prompts_ecommerce.yaml 的 summary.system（D6 迁出决策），
    缺失时回退代码兜底 SUMMARY_SYSTEM。
    """
    try:
        from agent_base.prompts import get_prompt

        return get_prompt("summary", "system", default=SUMMARY_SYSTEM)
    except Exception:
        return SUMMARY_SYSTEM


# ── 辅助函数 ────────────────────────────────────────────────────────────────


def _load_summary_cfg() -> dict[str, Any]:
    """从 app.yaml 加载摘要配置（失败时安全返回默认值）。"""
    try:
        from agent_base.config import load_yaml, deep_get

        cfg = load_yaml("configs/app.yaml")
        retrieval = cfg.get("retrieval", {})
        return {
            "concurrency": int(deep_get(retrieval, "summary_index.summary_concurrency", 50)),
            "timeout": int(deep_get(retrieval, "summary_index.summary_timeout", 60)),
            "max_retries": int(deep_get(retrieval, "summary_index.summary_max_retries", 2)),
        }
    except Exception:
        return {"concurrency": 50, "timeout": 60, "max_retries": 2}


def _build_chat_model_with_timeout(cfg: dict[str, Any] | None = None, timeout: int = 60):
    """构建带显式 timeout + max_retries 的 ChatOpenAI。

    Args:
        cfg: LLM 配置字典。
        timeout: 请求超时秒数。

    Returns:
        ChatModel 实例。
    """
    from agent_base.llms import build_chat_model

    c = cfg or {
        "provider": "langchain",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "temperature": 0.1,
    }
    return build_chat_model(
        provider=c.get("provider", "langchain"),
        model=c.get("model", "deepseek-v4-flash"),
        base_url=c.get("base_url", "https://api.deepseek.com"),
        api_key_env=c.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"),
        temperature=float(c.get("temperature", 0.1)),
        timeout=timeout,
        max_retries=c.get("max_retries", 2),
    )


def _summarize_one(text: str, chain) -> str:
    """为单个 chunk 生成摘要。

    重试由链内官方 ``Runnable.with_retry``（指数退避 + 抖动）负责，
    此处只做空文本/失败兜底——截断原文，保证批量任务不中断。

    Args:
        text: Chunk text.
        chain: 已带官方 with_retry 的 LCEL summary chain。

    Returns:
        Summary string (≤120 chars)。
    """
    if not text.strip():
        return "(empty chunk)"
    try:
        summary = str(chain.invoke({"doc": text})).strip()
        if summary:
            return summary[:120]
    except Exception as exc:
        logger.warning("Summary generation failed after with_retry exhaustion: %s", exc)
    return text[:60]


# ── 公开 API ────────────────────────────────────────────────────────────────


def make_summary_chain(llm_cfg: dict[str, Any] | None = None, timeout: int = 60):
    """构建带 timeout 的 LCEL 摘要链。

    Args:
        llm_cfg: LLM 配置字典（默认 DeepSeek-flash）。
        timeout: 请求超时秒数。

    Returns:
        接收 ``{"doc": "<text>"}`` 并返回摘要字符串的 Runnable。
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda

    llm = _build_chat_model_with_timeout(llm_cfg, timeout=timeout)

    def _extract_doc(input_dict: dict) -> dict:
        if isinstance(input_dict, dict) and "doc" in input_dict:
            return {"doc": str(input_dict["doc"])[:1500]}
        return {"doc": str(input_dict)[:1500]}

    system_prompt = _load_summary_system_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "商品块文本：\n{doc}"),
        ]
    )

    # 官方重试：指数退避 + 抖动（替代手写 backoff 循环；模型层另有 SDK max_retries）
    from openai import APIError as _OpenAIError

    base_chain = RunnableLambda(_extract_doc) | prompt | llm | StrOutputParser()
    retries = int(_load_summary_cfg().get("max_retries", 2))
    return base_chain.with_retry(
        retry_if_exception_type=(_OpenAIError,),
        wait_exponential_jitter=True,
        stop_after_attempt=retries + 1,
    )


def generate_summaries(
    chunks: list[dict[str, Any]],
    llm_cfg: dict[str, Any] | None = None,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """并发为每个 chunk 生成摘要。

    Args:
        chunks: ``{chunk_id, text, metadata}`` 字典列表。
        llm_cfg: LLM 配置（默认 DeepSeek-flash）。
        max_workers: 线程池大小（None 时读 ``summary_index.summary_concurrency``，
            默认 50，上限 500）。

    Returns:
        List of ``{summary_id, summary, chunk_id, doc_id, metadata}``, one per chunk.
    """
    from agent_base.indexing.vector_index import _qdrant_point_id

    s_cfg = _load_summary_cfg()
    if max_workers is None:
        max_workers = min(s_cfg["concurrency"], 500)
    timeout = s_cfg["timeout"]
    max_retries = s_cfg["max_retries"]

    chain = make_summary_chain(llm_cfg, timeout=timeout)
    n = len(chunks)
    workers = min(max_workers, n)
    results: list[dict[str, Any]] = [{} for _ in range(n)]

    logger.info(
        "Generating summaries for %d chunks (workers=%d, timeout=%ds, retries=%d)",
        n,
        workers,
        timeout,
        max_retries,
    )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures_map = {}
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            future = ex.submit(_summarize_one, text, chain)
            futures_map[future] = (i, chunk)

        completed = 0
        for future in as_completed(futures_map):
            idx, chunk = futures_map[future]
            try:
                summary_text = future.result()
            except Exception:
                summary_text = chunk.get("text", "")[:60]

            chunk_id = chunk.get("chunk_id", "")
            doc_id = chunk.get("metadata", {}).get("doc_id", "")

            results[idx] = {
                "summary_id": _qdrant_point_id(f"summary:{chunk_id}"),
                "summary": summary_text,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "metadata": {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "summary_type": "llm_per_chunk",
                    "section": chunk.get("metadata", {}).get("section", ""),
                },
            }

            completed += 1
            if completed % 20 == 0:
                logger.info("  summaries: %d/%d", completed, n)

    logger.info("Summaries complete: %d/%d", completed, n)
    return results
