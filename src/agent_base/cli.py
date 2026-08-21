"""命令行入口（ecommerce-rag）。

子命令覆盖完整链路：
parse / batch-parse / batch-build / catalog / catalog-search /
route / retrieve / ingest / ask
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from agent_base.chains import answer_question
from agent_base.graphs import build_rag_graph
from agent_base.config import deep_get, load_yaml
from agent_base.indexing import load_vector_store
from agent_base.retrieval import build_metadata_filter, build_query_rewrite, retrieve_advanced
from agent_base.retrieval.retrieval_config import GenerateConfig, RerankConfig, RetrievalConfig
from agent_base.retrieval.advanced_retriever import RERANK_STRATEGIES


def main() -> None:
    """解析命令行参数并分发到对应子命令。

    支持解析 PDF、批量建索引、意图路由、检索与问答等操作。
    """
    _configure_stdout()
    parser = argparse.ArgumentParser(prog="ecommerce-rag")
    parser.add_argument("--config", default="configs/app.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    route_cmd = subparsers.add_parser("route", help="Show intent routing, query rewrite, and metadata filter.")
    route_cmd.add_argument("--question", required=True)
    route_cmd.add_argument("--product-name", default=None)
    route_cmd.add_argument("--product-spec", default=None)
    route_cmd.add_argument("--category", default=None)
    route_cmd.add_argument("--catalog", default=None)
    route_cmd.add_argument("--json", action="store_true")

    retrieve_cmd = subparsers.add_parser("retrieve", help="Run observable metadata-aware retrieval.")
    retrieve_cmd.add_argument("--question", required=True)
    retrieve_cmd.add_argument("--persist-dir", default=None)
    retrieve_cmd.add_argument("--collection", default=None)
    retrieve_cmd.add_argument("--summary-collection", default=None)
    retrieve_cmd.add_argument("--top-k", type=int, default=6)
    retrieve_cmd.add_argument("--candidate-k", type=int, default=None)
    retrieve_cmd.add_argument("--final-k", type=int, default=None)
    retrieve_cmd.add_argument("--rerank", choices=RERANK_STRATEGIES, default="auto")
    retrieve_cmd.add_argument("--product-name", default=None)
    retrieve_cmd.add_argument("--product-spec", default=None)
    retrieve_cmd.add_argument("--category", default=None)
    retrieve_cmd.add_argument("--catalog", default=None)
    retrieve_cmd.add_argument("--json", action="store_true")

    ask_cmd = subparsers.add_parser("ask", help="Ask a question against the Chroma index.")
    ask_cmd.add_argument("--question", required=True)
    ask_cmd.add_argument("--framework", choices=["classic", "langgraph", "graph", "agent"], default=None, help="编排方式（默认按 configs/app.yaml framework.orchestrator）")
    ask_cmd.add_argument("--persist-dir", default=None)
    ask_cmd.add_argument("--collection", default=None)
    ask_cmd.add_argument("--summary-collection", default=None)
    ask_cmd.add_argument("--top-k", type=int, default=6)
    ask_cmd.add_argument("--rerank", choices=RERANK_STRATEGIES, default="auto")
    ask_cmd.add_argument("--product-name", default=None)
    ask_cmd.add_argument("--product-spec", default=None)
    ask_cmd.add_argument("--category", default=None)
    ask_cmd.add_argument("--catalog", default=None)

    args = parser.parse_args()

    if args.command == "route":
        config = _load_config_if_exists(args.config)
        intent_classifier_cfg = config.get("intent_classifier", {})
        product_name, product_spec, category, resolution = _resolve_catalog_args(args)
        rewrite = build_query_rewrite(
            args.question,
            product_name=product_name,
            product_spec=product_spec,
            intent_classifier=intent_classifier_cfg,
        )
        metadata_filter = build_metadata_filter(
            rewrite.route,
            product_name=product_name,
            product_spec=product_spec,
            category=category,
        )
        payload = {
            "question": args.question,
            "route": rewrite.route.to_dict(),
            "rewrite": rewrite.to_dict(),
            "metadata_filter": metadata_filter,
            "catalog_resolution": resolution.to_dict() if resolution else None,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_route(payload)
        return

    if args.command == "retrieve":
        config = _load_config_if_exists(args.config)
        persist_dir = args.persist_dir or deep_get(config, "paths.chroma_dir", "data/chroma")
        collection = args.collection or deep_get(config, "index.chunk_collection", "ecommerce_chunks")
        summary_collection = args.summary_collection or deep_get(config, "index.summary_collection", "ecommerce_summaries")
        embedding_cfg = config.get("embedding", {})
        llm_cfg = config.get("llm", {})
        intent_classifier_cfg = config.get("intent_classifier", {})
        rerank_cfg = config.get("rerank", {})
        prompts_path = deep_get(config, "prompts.path", "configs/prompts.yaml")
        vector_store = load_vector_store(
            persist_dir=persist_dir,
            collection=collection,
            embedding_provider=embedding_cfg.get("provider", "hash"),
            embedding_model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            embedding_base_url=embedding_cfg.get("base_url"),
            embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
        )
        summary_store = load_vector_store(
            persist_dir=persist_dir,
            collection=summary_collection,
            embedding_provider=embedding_cfg.get("provider", "hash"),
            embedding_model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            embedding_base_url=embedding_cfg.get("base_url"),
            embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
        )
        from types import SimpleNamespace
        from qdrant_client import QdrantClient

        vectorstore_cfg = config.get("vectorstore", {}) or {}
        sparse_store = SimpleNamespace(
            client=QdrantClient(url=vectorstore_cfg.get("url") or "http://localhost:6333"),
            collection_name=vectorstore_cfg.get("sparse_collection", "ecommerce_chunks_sparse"),
        )
        product_name, product_spec, category, resolution = _resolve_catalog_args(args)
        cfg = RetrievalConfig(
            top_k=args.final_k or args.top_k,
            candidate_k=args.candidate_k,
            rerank=args.rerank,
            product_name=product_name,
            product_spec=product_spec,
            category=category,
            intent_classifier=intent_classifier_cfg,
            rerank_model=RerankConfig(
                provider=rerank_cfg.get("provider", "none"),
                model=rerank_cfg.get("model", "gte-rerank-v2"),
                endpoint=rerank_cfg.get("endpoint"),
                api_key_env=rerank_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
                timeout=int(rerank_cfg.get("timeout", 30)),
                strategies=rerank_cfg.get("use_for_strategies"),
                preserve_preferred_sections=bool(rerank_cfg.get("preserve_preferred_sections", True)),
            ),
        )
        trace = retrieve_advanced(
            vector_store, args.question, cfg,
            summary_store=summary_store, sparse_store=sparse_store,
        )
        if args.json:
            print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_retrieval_trace(trace.to_dict())
        return

    if args.command == "ask":
        config = _load_config_if_exists(args.config)
        persist_dir = args.persist_dir or deep_get(config, "paths.chroma_dir", "data/chroma")
        collection = args.collection or deep_get(config, "index.chunk_collection", "ecommerce_chunks")
        summary_collection = args.summary_collection or deep_get(config, "index.summary_collection", "ecommerce_summaries")
        embedding_cfg = config.get("embedding", {})
        llm_cfg = config.get("llm", {})
        intent_classifier_cfg = config.get("intent_classifier", {})
        prompts_path = deep_get(config, "prompts.path", "configs/prompts.yaml")
        vector_store = load_vector_store(
            persist_dir=persist_dir,
            collection=collection,
            embedding_provider=embedding_cfg.get("provider", "hash"),
            embedding_model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            embedding_base_url=embedding_cfg.get("base_url"),
            embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
        )
        summary_store = load_vector_store(
            persist_dir=persist_dir,
            collection=summary_collection,
            embedding_provider=embedding_cfg.get("provider", "hash"),
            embedding_model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            embedding_base_url=embedding_cfg.get("base_url"),
            embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
        )
        product_name, product_spec, category, _ = _resolve_catalog_args(args)
        llm_evidence_cfg = llm_cfg.get("evidence", {})
        rerank_cfg = config.get("rerank", {})
        # ── framework 解析 ──
        framework = args.framework or deep_get(config, "framework.orchestrator", "classic")
        if framework in {"langgraph", "graph", "agent"}:
            import uuid
            # agent 模式：从 agent 配置段构造 llm_cfg
            agent_cfg = config.get("agent", {}) or deep_get(config, "framework.agent", {}) or {}
            if framework == "agent" and agent_cfg:
                graph_llm_cfg = {
                    "provider": "langchain",
                    "model": agent_cfg.get("model", "deepseek-v4-pro"),
                    "base_url": agent_cfg.get("base_url", "https://api.deepseek.com"),
                    "api_key_env": agent_cfg.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"),
                    "temperature": float(agent_cfg.get("temperature", 0.1)),
                }
            else:
                graph_llm_cfg = llm_cfg
            graph = build_rag_graph(
                vector_store=vector_store,
                summary_store=summary_store,
                sparse_store=sparse_store,
                rerank_cfg=rerank_cfg,
                llm_cfg=graph_llm_cfg,
                prompts_path=prompts_path,
            )
            state = {
                "question": args.question,
                "product_name": product_name,
                "product_spec": product_spec,
                "category": category,
                "errors": [],
            }
            result = graph.invoke(state, {"configurable": {"thread_id": str(uuid.uuid4())}})
            print(result.get("answer", ""))
            errors = result.get("errors") or []
            if errors:
                print(f"\n── 错误 ({len(errors)}) ──")
                for e in errors:
                    print(f"  {e[:200]}")
            return
        # classic 模式（默认）
        cfg = RetrievalConfig(
            top_k=args.top_k,
            rerank=args.rerank,
            product_name=product_name,
            product_spec=product_spec,
            category=category,
            prompts_path=prompts_path,
            intent_classifier=intent_classifier_cfg,
            rerank_model=RerankConfig(
                provider=rerank_cfg.get("provider", "none"),
                model=rerank_cfg.get("model", "gte-rerank-v2"),
                endpoint=rerank_cfg.get("endpoint"),
                api_key_env=rerank_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
                timeout=int(rerank_cfg.get("timeout", 30)),
                strategies=rerank_cfg.get("use_for_strategies"),
                preserve_preferred_sections=bool(rerank_cfg.get("preserve_preferred_sections", True)),
            ),
            llm=GenerateConfig(
                provider=llm_cfg.get("provider", "none"),
                model=llm_cfg.get("model"),
                base_url=llm_cfg.get("base_url"),
                api_key_env=llm_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
                temperature=float(llm_cfg.get("temperature", 0.1)),
                use_llm=llm_cfg.get("provider", "none") not in {"none", "off", "false"},
                evidence_max_chars_per_doc=int(llm_evidence_cfg.get("max_chars_per_doc", 1200)),
                evidence_max_total_chars=int(llm_evidence_cfg.get("max_total_chars", 6000)),
                evidence_max_chars_per_product=int(llm_evidence_cfg.get("max_chars_per_product", 1800)),
                evidence_max_chars_per_product_doc=int(llm_evidence_cfg.get("max_chars_per_product_doc", 900)),
            ),
        )
        print(answer_question(args.question, vector_store, cfg, summary_store=summary_store))
        return


def _load_config_if_exists(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        return load_yaml(config_path)
    except RuntimeError as exc:
        print(f"Warning: {exc} Falling back to CLI defaults.", file=sys.stderr)
        return {}


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _print_route(payload: dict) -> None:
    route = payload["route"]
    print(f"intent={route['intent']}")
    print(f"confidence={route['confidence']}")
    print(f"matched_keywords={', '.join(route['matched_keywords']) or '-'}")
    print(f"sections={', '.join(route['sections']) or '-'}")
    print(f"rewritten_question={payload['rewrite']['rewritten_question']}")
    print(f"metadata_filter={json.dumps(payload['metadata_filter'], ensure_ascii=False)}")
    if payload.get("catalog_resolution"):
        print(f"catalog_resolution={json.dumps(payload['catalog_resolution'], ensure_ascii=False)}")


def _print_retrieval_trace(payload: dict) -> None:
    route = payload["route"]
    print(f"mode={payload.get('mode', 'metadata')} rerank={payload.get('rerank', 'none')}")
    print(f"intent={route['intent']} confidence={route['confidence']}")
    if payload.get("decision"):
        decision = payload["decision"]
        print(f"strategy={decision.get('strategy')} reason={decision.get('reason')}")
    print(f"search_query={payload['search_query']}")
    print(f"metadata_filter={json.dumps(payload['metadata_filter'], ensure_ascii=False)}")
    if "stage_counts" in payload:
        print(f"stage_counts={json.dumps(payload['stage_counts'], ensure_ascii=False)}")
    print(f"fallback_used={payload['fallback_used']}")
    if payload.get("error"):
        print(f"error={payload['error']}")
    if payload.get("errors"):
        print(f"errors={json.dumps(payload['errors'], ensure_ascii=False)}")
    for item in payload["results"]:
        print(
            f"{item['rank']}. stage={item.get('retrieval_stage', '-')} "
            f"section={item['section']} score={item['score']} "
            f"rerank_score={item.get('rerank_score')} page={item['page_start']}-{item['page_end']} "
            f"chunk_id={item['chunk_id']}"
        )
        print(f"   {item['preview']}")


def _resolve_catalog_args(args) -> tuple[str | None, str | None, str | None, object | None]:
    product_name = getattr(args, "product_name", None)
    product_spec = getattr(args, "product_spec", None)
    category = getattr(args, "category", None)
    resolution = None
    question = getattr(args, "question", None)
    if question:
        # catalog 纯 PG 运行时数据源（JSON 文件已淘汰）
        from agent_base.api.main import get_catalog
        from agent_base.indexing.metadata_index import resolve_query_constraints

        catalog = get_catalog()
        resolution = resolve_query_constraints(catalog, question)
        product_name = product_name or resolution.product_name
        product_spec = product_spec or resolution.product_spec
        category = category or resolution.category
    return product_name, product_spec, category, resolution


def _print_catalog_search(payload: dict) -> None:
    if "matches" in payload:
        matches = payload["matches"]
    else:
        matches = payload.get("matched_products") or []
        print(f"category={payload.get('category') or '-'} ambiguous={payload.get('ambiguous')}")
    if not matches:
        print("No catalog matches.")
        return
    for item in matches:
        print(
            f"{item.get('product_name')} | {item.get('product_spec')} | "
            f"category={item.get('category')} | chunks={item.get('chunk_count')} | "
            f"source={item.get('source_file')}"
        )


def _optional_int(value) -> int | None:
    if value in {None, "", "none", "null"}:
        return None
    return int(value)


if __name__ == "__main__":
    main()
