"""Embedding 工厂：统一 hash / ollama / openai 兼容三种编码器。"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass


@dataclass(slots=True)
class HashEmbeddings:
    """无外部依赖的 hash embedding 兜底实现。

    用于教学演示和冒烟测试：对文本做 token 化后映射到固定维度向量，
    不具备真实语义能力。生产环境请使用 langchain_openai 或 ollama provider。

    Attributes:
        dimensions: 输出向量维度，默认 512。
    """

    dimensions: int = 512

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档文本。

        Args:
            texts: 待编码的文本列表。

        Returns:
            每个文本对应的归一化向量列表。
        """
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """编码单个查询文本。

        Args:
            text: 查询文本。

        Returns:
            归一化查询向量。
        """
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        """将单个文本映射为归一化 hash 向量。

        Args:
            text: 输入文本。

        Returns:
            归一化后的固定维度向量。
        """
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def build_embeddings(
    provider: str = "hash",
    model: str | None = None,
    dimensions: int | None = 512,
    base_url: str | None = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
    keep_alive: int | None = None,
):
    """按配置创建 embedding 对象（工厂）。

    支持的 provider：
    - hash：无外部依赖的教学/测试兜底（512 维）。
    - langchain_openai / lc_openai：OpenAI 兼容接口（如 DashScope、DeepSeek 网关）。
    - ollama：本地 embedding（bge-m3，1024 维）。
    - openai：langchain-openai 的官方 OpenAI 客户端。

    注意：上传入库和运行时检索必须使用同一个 embedding 配置（模型/维度一致），
    否则向量空间不一致会导致检索结果不可信。

    Args:
        provider: embedding provider 名。
        model: 模型名；None 时使用各 provider 的默认模型。
        dimensions: 输出向量维度；None 表示由服务端决定。
        base_url: OpenAI 兼容服务地址；None 时使用 provider 默认地址。
        api_key_env: 存放 API key 的环境变量名。
        keep_alive: Ollama 模型驻留时长（秒），仅 ollama provider 使用。

    Returns:
        实现了 embed_documents / embed_query 接口的 embedding 对象。

    Raises:
        RuntimeError: 缺少对应依赖包。
        ValueError: 不支持的 provider。
    """
    provider = (provider or "hash").lower()
    if provider == "hash":
        return HashEmbeddings(dimensions=dimensions or 512)

    if provider in {"langchain_openai", "lc_openai", "openai"}:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError("Missing dependency: langchain-openai.") from exc
        # langchain-openai 要求 api_key 非空才能构造对象。生产环境从 api_key_env
        # 读取真实 key；缺 key 时用占位符保证对象可构造（契约验收：无 key 也能建对象），
        # 真正发请求时会以清晰的认证错误失败。
        api_key = os.getenv(api_key_env) or f"missing-{api_key_env}"
        return OpenAIEmbeddings(
            model=model or "text-embedding-v3",
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            dimensions=dimensions,
            check_embedding_ctx_length=False,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise RuntimeError("Missing dependency: langchain-ollama.") from exc
        # 本地向量化（E-01 v0.1.1）：bge-m3 固定 1024 维，不依赖任何 api_key。
        # 与配置的映射：base_url 默认本地 ollama 端点，model 默认 bge-m3。
        return OllamaEmbeddings(
            model=model or "bge-m3",
            base_url=base_url or "http://localhost:11434",
            keep_alive=keep_alive or 1800,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


def _tokenize(text: str) -> list[str]:
    """对中文/英文混合文本做轻量 token 化（hash embedding 用）。

    Args:
        text: 输入文本。

    Returns:
        token 列表：英文词 + 单个汉字 + 相邻双字组合（最多 800 个）。
    """
    text = text.lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    bigrams = [text[i : i + 2] for i in range(max(0, len(text) - 1)) if text[i : i + 2].strip()]
    return words + bigrams[:800]
