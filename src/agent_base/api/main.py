"""FastAPI 应用：问答、检索、上传入库、文档管理与管理端接口。"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator

import secrets
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent_base.chains import answer_question_with_trace
from agent_base.chains.classic_ask import light_results, run_classic_ask
from agent_base.chains.streaming import stream_ask
from agent_base.config import deep_get, load_yaml
from agent_base.indexing import load_vector_store
from agent_base.indexing.metadata_index import resolve_query_constraints
from agent_base.retrieval import retrieve_advanced
from agent_base.retrieval.advanced_retriever import RERANK_STRATEGIES
from agent_base.retrieval.retrieval_config import RetrievalConfig

_PROJECT_ROOT_ENV = Path(os.getenv("AGENT_BASE_PROJECT_ROOT", "") or "").resolve()
PROJECT_ROOT = _PROJECT_ROOT_ENV if str(_PROJECT_ROOT_ENV) not in {".", ""} and _PROJECT_ROOT_ENV.exists() else Path(__file__).resolve().parents[3]
UPLOAD_DIR = Path(os.getenv("AGENT_BASE_UPLOAD_DIR", PROJECT_ROOT / "data" / "uploads"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# v0.30.6：数据中台统一清洗为 MD，主项目上传仅支持 MD
# 知识入库直接上传仅收 Markdown；外部格式（PDF/Word/PPT/Excel 等）先走「文件清洗」面板
SUPPORTED_UPLOAD_EXTENSIONS = {".md"}


class RagRequest(BaseModel):
    """问答/检索请求体：问题、检索参数与可选约束。"""

    question: str = Field(..., min_length=1, max_length=1000)
    rerank: str = Field("model")  # 项目默认模型重排（本地 TEI），不可用时自动降级关键词
    top_k: int = Field(6, ge=1, le=20)
    candidate_k: int | None = Field(None, ge=1, le=50)
    product_name: str | None = None
    product_spec: str | None = None
    category: str | None = None
    use_catalog: bool = True
    stream: bool = False
    session_id: str | None = None  # P11-01: 会话 ID（多轮记忆；agent 模式必传，classic 可选）
    user_id: str | None = None  # P19c: 用户 ID（长期记忆 user_memories 画像注入；留空不注入）
    framework: str = Field("classic", pattern=r"^(classic|langgraph|graph|agent)$")


class MemorySummarizeRequest(BaseModel):
    """记忆测试台：会话提炼请求。"""

    session_id: str
    user_id: str


class LoginRequest(BaseModel):
    """运营台登录请求。"""

    username: str
    password: str


class AccountCreateRequest(BaseModel):
    """账号管理：创建账号请求。"""

    username: str
    password: str
    display_name: str = ""
    role: str = "agent"


class AccountUpdateRequest(BaseModel):
    """账号管理：更新账号请求（中文名 / 密码）。"""

    display_name: str | None = None
    password: str | None = None


class UploadResponse(BaseModel):
    """上传入库响应：文件、doc_id、分块/摘要数量与商品信息。"""

    filename: str
    saved_path: str
    doc_id: str
    product_name: str
    product_spec: str
    category: str
    chunk_count: int
    summary_count: int
    catalog_product_count: int
    supported_extensions: list[str]
    action: str = "indexed"
    message: str = ""


class AppSettings(BaseModel):
    """应用运行时设置：配置路径、collection 与 catalog 路径。"""

    config_path: str = "configs/app.yaml"
    persist_dir: str = "data/chroma"
    collection: str = "ecommerce_chunks"
    summary_collection: str = "ecommerce_summaries"
    retention_days: int = 30
    handoff_pending_timeout: int = 900
    handoff_idle_timeout: int = 1200


def create_app() -> FastAPI:
    """创建 FastAPI 应用，并把前端页面、问答接口和上传入库接口挂载起来。"""
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # v0.46: 回收站后台清理（每小时物理清除过期软删记录，失败不影响主服务）
        task = asyncio.create_task(_purge_loop())
        # v0.48: OpenAIEmbeddings 无 keep_alive → 每 4 分钟预热保持 Ollama 模型加载
        ka_task = asyncio.create_task(_embedding_keepalive_loop())
        # 生产安全检查：默认凭据/开发密钥在用时打 WARNING 日志（不阻断启动）
        try:
            from agent_base.monitoring.logger import log_event
            from agent_base.security import check_production_security

            _issues = check_production_security(log_event)
            if _issues:
                log_event("WARNING", "security", "production_issues_found", {"count": len(_issues)})
        except Exception:
            pass
        # 异步任务队列 worker（知识流水线/图片生成等耗时操作，PG 持久化队列）
        worker_task = None
        try:
            from agent_base.config import deep_get, load_yaml

            _tcfg = load_yaml("configs/app.yaml") or {}
            if deep_get(_tcfg, "tasks.worker_enabled", True):
                from agent_base.async_tasks import start_task_worker

                worker_task, _stop_event = start_task_worker(
                    poll_interval=float(deep_get(_tcfg, "tasks.worker_poll_interval", 2.0)),
                    concurrency=int(deep_get(_tcfg, "tasks.worker_concurrency", 2)),
                    timeout_s=float(deep_get(_tcfg, "tasks.worker_timeout_s", 300)),
                )
        except Exception:
            worker_task = None
        # v0.46: 预热 embedding（Ollama 模型常驻，避免首个用户等待模型加载 10s+）
        try:
            await asyncio.to_thread(_warm_embedding)
        except Exception:
            pass
        # 生产告警巡检：每 60s 检查 ERROR 阈值，超阈值推送群机器人（alerting.enabled 门控）
        alert_task = None
        try:
            from agent_base.config import deep_get, load_yaml as _load_yaml

            _acfg = _load_yaml("configs/app.yaml") or {}
            if deep_get(_acfg, "alerting.enabled", False):

                async def _alert_loop():
                    while True:
                        try:
                            from agent_base.monitoring.alert import maybe_notify

                            await asyncio.to_thread(maybe_notify)
                        except Exception:
                            pass
                        await asyncio.sleep(60)

                alert_task = asyncio.create_task(_alert_loop())
        except Exception:
            alert_task = None
        yield
        task.cancel()
        ka_task.cancel()
        if worker_task is not None:
            worker_task.cancel()
        if alert_task is not None:
            alert_task.cancel()

    app = FastAPI(title="E-commerce RAG Platform", version="0.1.0", lifespan=_lifespan)

    # ── 限流中间件（对话接口按客户端 IP 滑动窗口限流，security.rate_limit 配置门控） ──
    _rate_limiter = None
    try:
        from agent_base.security import build_rate_limiter

        _rate_limiter = build_rate_limiter()
    except Exception:
        _rate_limiter = None

    @app.middleware("http")
    async def _rate_limit_middleware(request: Request, call_next):
        if _rate_limiter is not None and request.url.path.startswith("/api/ask"):
            forwarded = request.headers.get("X-Forwarded-For", "")
            client_ip = (forwarded.split(",")[0].strip() if forwarded else "") or (
                request.client.host if request.client else "unknown"
            )
            if not _rate_limiter.allow(client_ip):
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=429,
                    content={"detail": "请求过于频繁，请稍后再试", "retry_after": int(_rate_limiter.window)},
                )
        return await call_next(request)

    # ── request_id 中间件（日志 MVP） ──
    @app.middleware("http")
    async def _request_id_middleware(request: Request, call_next):
        from uuid import uuid4

        from agent_base.monitoring.logger import request_id_var

        rid = request.headers.get("X-Request-ID") or str(uuid4())
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_var.reset(token)

    # 前端构建产物目录：默认豆包 React 前端（frontend-react/dist），
    # 可用环境变量 FRONTEND_DIST_DIR 覆盖（旧 Vue 前端已移除）。
    _FRONTEND_DIST = Path(
        os.environ.get("FRONTEND_DIST_DIR", str(PROJECT_ROOT / "frontend-react" / "dist"))
    )
    if (_FRONTEND_DIST / "assets").exists():
        # P20: 新前端（Vite 构建产物）优先
        app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")

    # Phase 2：图片知识库上传目录静态服务（/media/uploads/*）
    _MEDIA_UPLOAD_DIR = Path(os.getenv("AGENT_BASE_MEDIA_DIR", str(PROJECT_ROOT / "data" / "media" / "uploads")))
    _MEDIA_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/media/uploads", StaticFiles(directory=str(_MEDIA_UPLOAD_DIR)), name="media_uploads")

    @app.get("/media/products/{product_id}.svg")
    def product_media_svg(product_id: str, style: str = "main"):
        """动态生成预置商品主图，避免依赖生图模型。"""
        import html

        pid = product_id.upper()
        products = get_catalog().get("products", {})
        item = products.get(pid) or {}
        name = str(item.get("name") or pid)
        brand = str(item.get("brand") or "星禾甄选")
        category = str(item.get("category") or "商品")
        price = ""
        if item.get("price_band"):
            price = str(item["price_band"])
        style_meta = {
            "main": ("商品主图", "#fdf8f2", "#f1e7db"),
            "detail": ("商品详情图", "#f4f7fb", "#e2ebf5"),
            "scene": ("使用场景图", "#f4fbf7", "#e0f2e8"),
        }.get(style, ("商品图", "#fdf8f2", "#f1e7db"))
        badge, bg_start, bg_end = style_meta
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800" viewBox="0 0 800 800">'
            '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0%" stop-color="{bg_start}"/><stop offset="100%" stop-color="{bg_end}"/>'
            '</linearGradient></defs>'
            '<rect width="800" height="800" rx="42" fill="url(#bg)"/>'
            '<circle cx="400" cy="330" r="160" fill="#fff" stroke="#d9c5a8" stroke-width="3"/>'
            '<text x="400" y="330" font-size="120" text-anchor="middle" dominant-baseline="middle" fill="#c9a87e">'
            "●</text>"
            f'<text x="400" y="555" font-size="42" text-anchor="middle" fill="#4a4038" font-family="sans-serif">'
            f"{html.escape(name)}</text>"
            f'<text x="400" y="610" font-size="24" text-anchor="middle" fill="#8b8177" font-family="sans-serif">'
            f"{html.escape(brand)} · {html.escape(category)}"
            + (" · " + html.escape(price) if price else "") +
            "</text>"
            '<text x="400" y="680" font-size="20" text-anchor="middle" fill="#b4a99f" font-family="sans-serif">'
            f"{html.escape(badge)}</text>"
            "</svg>"
        )
        return Response(content=svg, media_type="image/svg+xml")

    # P19 D2: 存量文档批量 approved 迁移（幂等）
    _seed_legacy_tags_once()

    # P19d: 运营台账号表 + 种子管理员（幂等）
    try:
        from agent_base.auth import init_admin_table, seed_admin, seed_agent, seed_user

        init_admin_table()
        seed_admin()
        seed_agent()
        seed_user()
    except Exception:
        pass

# P9-05：管理端鉴权依赖
    def _verify_admin(request: Request):
        # BUG-7 修复：校验 token 有效且账号 role=admin（与 _verify_agent 一致）
        auth_header = request.headers.get("Authorization", "")
        username: str | None = None
        if auth_header.startswith("Bearer "):
            from agent_base.auth import get_user_role, verify_token

            username = verify_token(auth_header[7:].strip())
        else:
            legacy = request.headers.get("X-Admin-Token", "")
            if legacy:
                from agent_base.auth import get_user_role, verify_token

                username = verify_token(legacy)
        if username:
            if get_user_role(username) != "admin":
                raise HTTPException(403, "Forbidden: admin role required")
            return
        # 兼容开发默认 X-Admin-Token（静态 admin_token 配置）
        config: dict[str, Any] = {}
        try:
            config = load_yaml(_project_path("configs/app.yaml"))
        except Exception:
            pass
        admin_token = (config.get("security", {}) or {}).get("admin_token") or os.getenv("ADMIN_TOKEN", "")
        if not admin_token:
            raise HTTPException(500, "Admin token not configured")
        from agent_base.security import reject_default_token

        if reject_default_token("admin_token") and admin_token == "admin-dev-token-2026":
            raise HTTPException(403, "默认 admin token 已在严格模式下禁用，请配置 ADMIN_TOKEN 环境变量")
        if not secrets.compare_digest(admin_token, request.headers.get("X-Admin-Token", "")):
            raise HTTPException(403, "Forbidden: valid X-Admin-Token required")

    def _verify_agent(request: Request):
        """v0.49: 客服角色鉴权——token 有效且账号 role=agent（人工端接口专用）。"""
        auth_header = request.headers.get("Authorization", "")
        username: str | None = None
        if auth_header.startswith("Bearer "):
            from agent_base.auth import get_user_role, verify_token

            username = verify_token(auth_header[7:].strip())
        else:
            legacy = request.headers.get("X-Admin-Token", "")
            if legacy:
                from agent_base.auth import get_user_role, verify_token

                username = verify_token(legacy)
        if not username:
            raise HTTPException(403, "Forbidden: valid token required")
        if get_user_role(username) != "agent":
            raise HTTPException(403, "Forbidden: agent role required")

    def _verify_user(request: Request):
        """v0.51: 任意登录用户鉴权——token 有效即可（买家/客服/管理员），防未登录枚举会话。"""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            from agent_base.auth import verify_token

            if verify_token(auth_header[7:].strip()):
                return
        legacy = request.headers.get("X-Admin-Token", "")
        if legacy:
            from agent_base.auth import verify_token

            if verify_token(legacy):
                return
        raise HTTPException(403, "Forbidden: valid token required")

    def _verify_platform(request: Request):
        """数据中台对接鉴权（X-Platform-Token），与运营台 token 分离。"""
        config: dict[str, Any] = {}
        try:
            config = load_yaml(_project_path("configs/app.yaml"))
        except Exception:
            pass
        platform_token = (
            (config.get("security", {}) or {}).get("platform_token")
            or os.getenv("PLATFORM_TOKEN", "")
        )
        if not platform_token:
            raise HTTPException(500, "Platform token not configured")
        from agent_base.security import reject_default_token

        if reject_default_token("platform_token") and platform_token == "platform-dev-token-2026":
            raise HTTPException(403, "默认 platform token 已在严格模式下禁用，请配置 PLATFORM_TOKEN 环境变量")
        provided = request.headers.get("X-Platform-Token", "")
        if not secrets.compare_digest(platform_token, provided):
            raise HTTPException(403, "Forbidden: valid X-Platform-Token required")

    # ── P19d: 运营台账号登录 ──
    @app.post("/api/auth/login")
    def auth_login(payload: LoginRequest) -> dict[str, Any]:
        """账号密码登录 → 签发签名 token（12h 过期）。"""
        from agent_base.auth import authenticate, create_token

        user = authenticate(payload.username, payload.password)
        if not user:
            raise HTTPException(401, "用户名或密码错误")
        return {
            "token": create_token(user["username"]),
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
        }
    # ── v0.53: 手机号验证码登录（mock 开发模式 / 阿里云 SMS 生产模式） ──
    @app.post("/api/auth/sms/send")
    def auth_sms_send(payload: dict[str, Any]) -> dict[str, Any]:
        """发送登录验证码（频率护栏；mock 模式响应带 code 便于联调，aliyun 模式真实发送）。"""
        import re as _re

        phone = str((payload or {}).get("phone") or "").strip()
        if not _re.fullmatch(r"1[3-9]\d{9}", phone):
            raise HTTPException(400, "手机号格式不正确")
        from agent_base.sms import send_code

        result = send_code(phone)
        if not result.get("ok"):
            raise HTTPException(429, result.get("message", "发送失败"))
        return result

    @app.post("/api/auth/sms/login")
    def auth_sms_login(payload: dict[str, Any]) -> dict[str, Any]:
        """手机号 + 验证码登录：校验 → 自动注册 → 签发签名 token（12h）。"""
        import re as _re

        phone = str((payload or {}).get("phone") or "").strip()
        code = str((payload or {}).get("code") or "").strip()
        if not _re.fullmatch(r"1[3-9]\d{9}", phone) or not code:
            raise HTTPException(400, "手机号或验证码不正确")
        from agent_base.sms import verify_code

        if not verify_code(phone, code):
            raise HTTPException(401, "验证码错误或已过期")
        from agent_base.auth import create_token, find_or_create_user_by_phone

        user = find_or_create_user_by_phone(phone)
        if not user:
            raise HTTPException(500, "用户创建失败")
        # 游客升级：手机号登录时把游客会话/记忆并入新账号（历史不丢）
        migrated = False
        guest_uid = str((payload or {}).get("guest_uid") or "").strip()
        if guest_uid:
            from agent_base.storage.pg import migrate_guest_to_user

            migrated = bool(migrate_guest_to_user(guest_uid, user["username"]).get("ok"))
        return {
            "token": create_token(user["username"]),
            "username": user["username"],
            "display_name": user["display_name"],
            "role": "user",
            "is_new": bool(user.get("is_new")),
            "migrated": migrated,
        }

    
    # ── v0.53: 游客直接咨询（免注册，一键进聊天；guest 账号自动建档） ──
    @app.post("/api/auth/guest")
    def auth_guest() -> dict[str, Any]:
        """游客咨询：生成一次性 guest 账号并签发 token（免登录直接体验）。

        每次进入生成独立 guest 账号（user_id = guest_<随机>），与真实登录共用
        同一鉴权/记忆体系；账号行入库失败不阻塞（token 鉴权不依赖用户行）。
        """
        import uuid

        from agent_base.auth import GUEST_TOKEN_MAX_AGE, create_token

        user_id = "guest_" + uuid.uuid4().hex[:10]
        display = "游客" + user_id[-4:]
        try:
            from agent_base.storage.pg import _conn

            with _conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO users (user_id, nickname) VALUES (%s,%s) ON CONFLICT (user_id) DO NOTHING",
                    (user_id, display),
                )
                conn.commit()
        except Exception:
            pass
        return {
            "token": create_token(user_id, max_age=GUEST_TOKEN_MAX_AGE),
            "username": user_id,
            "display_name": display,
            "role": "user",
            "guest": True,
        }

    @app.get("/api/auth/me", dependencies=[Depends(_verify_admin)])
    def auth_me(request: Request) -> dict[str, Any]:
        """返回当前登录账号信息（开发默认 token 时返回 admin）。"""
        from agent_base.auth import verify_token

        auth_header = request.headers.get("Authorization", "")
        token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else request.headers.get("X-Admin-Token", "")
        username = verify_token(token) or "admin"
        return {"username": username, "display_name": username, "role": "admin"}

    @app.post("/api/auth/logout", dependencies=[Depends(_verify_admin)])
    def auth_logout() -> dict[str, Any]:
        """退出登录（无状态 token，前端清本地即可，接口用于语义完整）。"""
        return {"ok": True}

    @app.get("/api/admin/accounts", dependencies=[Depends(_verify_admin)])
    def admin_list_accounts() -> dict[str, Any]:
        """账号列表（客服/管理员/买家）。"""
        from agent_base.auth import list_accounts

        return {"accounts": list_accounts()}

    @app.post("/api/admin/accounts", dependencies=[Depends(_verify_admin)])
    def admin_create_account(payload: AccountCreateRequest) -> dict[str, Any]:
        """创建账号（默认客服角色，可建管理员）。"""
        from agent_base.auth import create_account

        try:
            account = create_account(
                payload.username.strip(),
                payload.password,
                payload.display_name.strip(),
                payload.role,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        if account is None:
            raise HTTPException(409, "用户名已存在")
        return {"ok": True, "account": account}

    @app.put("/api/admin/accounts/{username}", dependencies=[Depends(_verify_admin)])
    def admin_update_account(username: str, payload: AccountUpdateRequest) -> dict[str, Any]:
        """更新账号中文名 / 重置密码。"""
        from agent_base.auth import update_account

        if username == "admin":
            raise HTTPException(403, "内置管理员账号不可修改")
        ok = update_account(
            username,
            display_name=payload.display_name.strip() if payload.display_name is not None else None,
            password=payload.password,
        )
        if not ok:
            raise HTTPException(404, "账号不存在")
        return {"ok": True}

    @app.delete("/api/admin/accounts/{username}", dependencies=[Depends(_verify_admin)])
    def admin_delete_account(username: str) -> dict[str, Any]:
        """删除账号（内置种子账号不可删）。"""
        from agent_base.auth import delete_account

        if not delete_account(username):
            raise HTTPException(400, "内置账号不可删除或账号不存在")
        return {"ok": True}

    @app.get("/")
    def index():
        """P20: 新前端对话端（Vite 构建产物）。"""
        index_path = _FRONTEND_DIST / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=503, detail="前端未构建：请先运行 cd frontend && npx vite build")
        return FileResponse(str(index_path))

    @app.get("/debug")
    def debug():
        """P20: /debug 作为运营台别名（新前端路由重定向 /admin）。"""
        index_path = _FRONTEND_DIST / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=503, detail="前端未构建：请先运行 cd frontend && npx vite build")
        return FileResponse(str(index_path))

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """健康检查：返回运行状态与关键配置摘要。"""
        settings = get_settings()
        return {
            "status": "ok",
            "persist_dir": settings.persist_dir,
            "collection": settings.collection,
            "summary_collection": settings.summary_collection,
            "upload_limit_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "supported_upload_extensions": sorted(SUPPORTED_UPLOAD_EXTENSIONS),
        }

    @app.get("/api/admin/alert", dependencies=[Depends(_verify_admin)])
    def admin_alert(minutes: int = 5) -> dict[str, Any]:
        """监控告警：最近窗口 ERROR 日志统计与阈值检查（复用 log_events 冷层）。"""
        from agent_base.monitoring.alert import check_alert

        return check_alert(minutes=min(minutes, 60))

    @app.get("/api/catalog", dependencies=[Depends(_verify_admin)])
    def catalog() -> dict[str, Any]:
        """返回商品元数据目录（管理端用）。"""
        return get_catalog()

    @app.get("/api/catalog/resolve", dependencies=[Depends(_verify_admin)])
    def resolve_catalog(query: str = Query(..., min_length=1)) -> dict[str, Any]:
        """从查询文本中解析商品/类目约束。

        Args:
            query: 用户查询文本。

        Returns:
            CatalogResolution 的 dict 形式。
        """
        catalog_data = get_catalog()
        return resolve_query_constraints(catalog_data, query).to_dict()

    @app.post("/api/retrieve", dependencies=[Depends(_verify_admin)])
    def retrieve(request: RagRequest) -> dict[str, Any]:
        """只执行检索链路并返回 Trace，用于观察路由、改写、filter 和召回结果。"""
        # 校验前端传来的参数是否合法，比如
        # top_k、检索模式、rerank
        # 选项等，避免非法配置进入后面的检索流程。
        _validate_options(request)
        # 解析用户问题里的商品约束和分类约束。
        # 例如用户问“玻尿酸精华适合油皮吗”，这里会从
        # 里识别出
        # product_name = 玻尿酸保湿精华液，后面检索时就能只查这个商品相关chunk。
        constraints = _resolve_constraints(request)
        # 获取运行时对象，包括：
# Chroma 向量库
        # 向量库、摘要索引、LLM
        # 配置、embedding
        # 配置、prompt
        # 配置等。
        # 这是为了避免每次请求都重新加载数据库和模型配置。
        runtime = get_runtime()

        cfg = RetrievalConfig.from_runtime(runtime)
        cfg.top_k = request.top_k
        cfg.candidate_k = request.candidate_k
        cfg.rerank = request.rerank
        cfg.product_name = constraints["product_name"]
        cfg.product_spec = constraints["product_spec"]
        cfg.category = constraints["category"]
        trace = retrieve_advanced(
            runtime["vector_store"],
            request.question,
            cfg,
            summary_store=runtime["summary_store"],
            sparse_store=runtime["sparse_store"],
        )
        return {
            "trace": trace.to_dict(),
            "catalog_resolution": constraints["catalog_resolution"],
        }

    @app.post("/api/ask")
    def ask(request: RagRequest):
        """完整问答入口；stream=true 时返回 SSE 事件流。"""
        _validate_options(request)
        constraints = _resolve_constraints(request)
        runtime = get_runtime()

        if request.stream:
            return StreamingResponse(
                _sse_event_stream(request, constraints, runtime),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # ── 非流式（原有逻辑） ──
        if request.framework in {"langgraph", "graph", "agent"}:
            return _run_graph_ask(request, constraints, runtime)
        return run_classic_ask(request, constraints, runtime)

    @app.post("/api/ask/stream")
    def ask_stream(request: RagRequest, http: Request):
        """P20 流式问答（CONTRACT-P20 §2.5）。

        sources → trace → memory → thinking → delta → done；思考过程用
        deepseek-reasoner 透传 reasoning_content（langchain-openai 官方不提取，
        走 SDK 原生流式）。
        """
        _validate_options(request)
        constraints = _resolve_constraints(request)
        runtime = get_runtime()
        # SEC-2: 解析登录账号作为会话归属（前端 X-Admin-Token / Bearer）
        from agent_base.auth import verify_token

        _auth = http.headers.get("Authorization", "")
        _token = _auth[7:].strip() if _auth.startswith("Bearer ") else http.headers.get("X-Admin-Token", "")
        owner = ""
        if _token:
            owner = verify_token(_token) or ""
        return StreamingResponse(
            stream_ask(request, constraints, runtime, owner=owner),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/memory/context-stats", dependencies=[Depends(_verify_admin)])
    def get_context_cost_stats_api(limit: int = 200) -> dict[str, Any]:
        """P28-2: 上下文成本控制聚合统计（档位分布/节省/token/摘要复用）。

        记忆测试台「成本控制」面板数据源；仅管理员可访问。
        定义在 /api/memory/{user_id} 之前，避免被路径参数捕获。
        """
        from agent_base.storage.chat_memory import get_context_cost_stats

        return get_context_cost_stats(limit=limit)

    @app.get("/api/memory/{user_id}", dependencies=[Depends(_verify_admin)])
    def get_user_memory(user_id: str) -> dict[str, Any]:
        """P19c: 长期记忆查询（user_memories，记忆测试台用）。"""
        try:
            from agent_base.storage.memory import retrieve_memory

            return {"user_id": user_id, "memories": retrieve_memory(user_id, top_k=20)}
        except Exception as exc:
            return {"user_id": user_id, "memories": [], "error": str(exc)}

    @app.post("/api/memory/summarize", dependencies=[Depends(_verify_admin)])
    def summarize_memory(payload: MemorySummarizeRequest) -> dict[str, Any]:
        """P19c: 会话提炼 → 写入长期记忆（summarize_session → save_memory）。"""
        from agent_base.storage.chat_memory import get_chat_history

        history = get_chat_history(payload.session_id, limit=16)
        if not history:
            return {"saved": [], "count": 0, "note": "会话历史为空，无内容可提炼"}
        try:
            from agent_base.agents.tools_memory import summarize_session
            from agent_base.storage.memory import upsert_memory_guarded

            runtime = get_runtime()
            items = summarize_session(
                history,
                user_id=payload.user_id,
                model_cfg=runtime["llm_config"],
            )
            saved: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            for item in items:
                guard = upsert_memory_guarded(
                    payload.user_id,
                    item["key"],
                    item["value"],
                    source="conversation",
                    confidence=float(item.get("confidence", 0.5)),
                )
                if guard.get("written"):
                    saved.append(item)
                else:
                    rejected.append({**item, "reason": guard.get("reason")})
            return {
                "saved": saved,
                "count": len(saved),
                "rejected": rejected,
                "rejected_count": len(rejected),
            }
        except Exception as exc:
            return {"saved": [], "count": 0, "error": str(exc)}

    @app.delete("/api/memory/session/{session_id}", dependencies=[Depends(_verify_admin)])
    def clear_session_memory(session_id: str) -> dict[str, Any]:
        """P19c: 清空测试会话短期记忆（Redis TTL 键；仅 test- 会话）。"""
        from agent_base.storage.chat_memory import clear_chat_memory

        cleared = clear_chat_memory(session_id)
        return {"session_id": session_id, "cleared": cleared}

    @app.get("/api/memory/session/{session_id}", dependencies=[Depends(_verify_admin)])
    def get_session_memory(session_id: str) -> dict[str, Any]:
        """P29: 按会话查询上下文状态（历史条数/字数/压缩状态）。

        切换会话后前端据此恢复记忆测试台展示，无需等下一轮问答的
        memory 事件。返回结构与 /api/ask/stream 的 memory 事件一致。
        """
        from agent_base.storage.chat_memory import (
            CONTEXT_BUDGET_CHARS,
            get_chat_history,
            get_context_config,
            get_history_meta,
        )

        history = get_chat_history(session_id, limit=64)
        history_chars = sum(len(m.get("content", "")) for m in history)
        compaction = get_history_meta(session_id)
        _ctx_cfg = get_context_config()
        _budget = int(_ctx_cfg.get("budget_chars", CONTEXT_BUDGET_CHARS))
        _threshold = int(float(_ctx_cfg.get("compact_trigger_ratio", 0.8)) * 100)
        return {
            "session_id": session_id,
            "user_id": None,
            "storage": "redis" if session_id.startswith("test-") else "pg",
            "history_count": len(history),
            "history_chars": history_chars,
            "history_budget": _budget,
            "history_ratio": round(history_chars / _budget, 3),
            "context_threshold": _threshold,
            "profile_chars": 0,
            "profile_budget": 1000,
            "profile_snippet": "",
            "compaction": {
                "compacted": bool(compaction and compaction.get("rounds")),
                "rounds": int((compaction or {}).get("rounds", 0)),
                "last_compacted_at": (compaction or {}).get("last_compacted_at"),
                "before_chars": int((compaction or {}).get("before_chars", 0)),
                "after_chars": int((compaction or {}).get("after_chars", 0)),
                "history": (compaction or {}).get("history", []),
            },
        }

    @app.delete("/api/memory/{user_id}/{memory_key}", dependencies=[Depends(_verify_admin)])
    def delete_user_memory(user_id: str, memory_key: str) -> dict[str, Any]:
        """删除单条长期记忆（user_memories 画像条目，管理员可维护）。

        Args:
            user_id: 用户标识。
            memory_key: 记忆键（如 skin_type / intent）。
        """
        from agent_base.storage.memory import delete_memory

        ok = delete_memory(user_id, memory_key)
        if not ok:
            raise HTTPException(404, f"记忆条目 {user_id}/{memory_key} 不存在")
        return {"status": "deleted", "user_id": user_id, "key": memory_key}

    async def _sse_event_stream(
        request: RagRequest,
        constraints: dict[str, Any],
        runtime: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """生成 SSE 事件流（P4-01）。

        Yields:
            SSE 格式的事件字符串。
        """

        def _emit(event_type: str, **kwargs: Any) -> str:
            payload = json.dumps({"type": event_type, **kwargs}, ensure_ascii=False)
            return f"data: {payload}\n\n"

        # Stage 1: 检索
        yield _emit("stage", stage="retrieving", message="正在检索商品资料...")

        if request.framework in {"langgraph", "graph", "agent"}:
            # graph/agent 模式：P4 先整段输出
            result = _run_graph_ask_internal(request, constraints, runtime)
            answer = result["answer"]
            yield _emit("stage", stage="generating")
            if answer:
                yield _emit("answer_chunk", content=answer)
            yield _emit("done", payload=result)
            return

        # classic 模式：完整流式
        llm_config = runtime["llm_config"]
        use_llm = llm_config.get("provider", "none") not in {"none", "off", "false"}

        # 执行检索（同步，快）
        cfg = RetrievalConfig.from_runtime(runtime)
        cfg.top_k = request.top_k
        cfg.candidate_k = request.candidate_k
        cfg.rerank = request.rerank
        cfg.product_name = constraints["product_name"]
        cfg.product_spec = constraints["product_spec"]
        cfg.category = constraints["category"]
        result = answer_question_with_trace(
            request.question,
            runtime["vector_store"],
            cfg,
            summary_store=runtime["summary_store"],
            sparse_store=runtime["sparse_store"],
        )
        payload = result.to_dict()
        payload["catalog_resolution"] = constraints["catalog_resolution"]

        yield _emit("stage", stage="generating")

        if use_llm:
            # 用 ChatOpenAI.stream() 实现逐 token 流式
            try:
                from agent_base.llms import build_chat_model
                model = build_chat_model(
                    provider=llm_config.get("provider", "none"),
                    model=llm_config.get("model"),
                    base_url=llm_config.get("base_url"),
                    api_key_env=llm_config.get("api_key_env", "DASHSCOPE_API_KEY"),
                    temperature=float(llm_config.get("temperature", 0.1)),
                )
                if model and hasattr(model, "stream"):
                    # 重新用 LLM 生成（流式版），跳过模板兜底
                    from agent_base.chains.qa_chain import (
                        _answer_docs,
                    )

                    # 用相同证据做流式生成
                    docs = payload.get("trace", {}).get("docs", [])
                    if not docs:
                        question_docs = result.trace.docs
                    else:
                        question_docs = docs
                    answer_docs = _answer_docs(list(question_docs) if question_docs else [],
                                               payload.get("trace", {}).get("route", {}).get("sections", []) or [])
                    # 简单的流式 prompt
                    evidence_text = "\n".join(
                        getattr(d, "page_content", str(d))[:800] for d in (answer_docs or [])[:5]
                    )
                    # LCEL 官方链：ChatPromptTemplate | model.stream（管理端对话测试）
                    from langchain_core.prompts import ChatPromptTemplate

                    chain = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "你是电商客服问答系统。只依据给定资料回答，"
                                '不输出"安全等级"或"风险标签"。',
                            ),
                            (
                                "user",
                                "问题：{question}\n\n证据：\n{evidence}\n\n"
                                "请给出：结论、依据商品/FAQ资料、购买/使用建议、来源。",
                            ),
                        ]
                    )
                    for chunk in chain.stream(
                        {"question": request.question, "evidence": evidence_text}
                    ):
                        token = getattr(chunk, "content", "") or ""
                        if token:
                            yield _emit("answer_chunk", content=token)
                            await asyncio.sleep(0)  # 让出事件循环
                else:
                    # 无 stream 能力 → 整段输出
                    yield _emit("answer_chunk", content=payload.get("answer", ""))
            except Exception:
                yield _emit("answer_chunk", content=payload.get("answer", ""))
        else:
            yield _emit("answer_chunk", content=payload.get("answer", ""))

        yield _emit("done", payload=payload)

    @app.post("/api/upload", dependencies=[Depends(_verify_admin)])
    async def upload_document(
        file: UploadFile = File(...),
        category: str = Form("上传文档"),
        product_name: str | None = Form(None),
        product_spec: str | None = Form(None),
    ) -> dict[str, Any]:
        """文件上传入口：保存文件、解析 chunk、写入 Chroma、更新摘要索引和 catalog。"""  #  ctrl+鼠标
        payload = await _process_upload(
            file=file,
            category=category,
            product_name=product_name,
            product_spec=product_spec,
        )
        return payload

# ── P16：文档管理 API（PG 真相源 → Qdrant 投影）──────────────────────────

    @app.delete("/api/documents/{doc_id}", dependencies=[Depends(_verify_admin)])
    def delete_document(doc_id: str):
        """删除文档（软删）：删 Qdrant 向量 → PG 全部版本 deleted → 删打标标签 → 缓存失效。"""
        deleted_vectors = _delete_document_core(doc_id)
        return {"status": "deleted", "doc_id": doc_id, "deleted_vectors": deleted_vectors}

    @app.post("/api/documents/{doc_id}/archive", dependencies=[Depends(_verify_admin)])
    def archive_document(doc_id: str):
        """归档文档：最新版本状态改 archived（保留向量与版本历史，可在已归档恢复）。"""
        from agent_base.storage.pg import doc_set_status

        if not doc_set_status(doc_id, "archived"):
            raise HTTPException(404, f"文档 {doc_id} 不存在")
        return {"status": "archived", "doc_id": doc_id}

    @app.post("/api/documents/{doc_id}/activate", dependencies=[Depends(_verify_admin)])
    def activate_document(doc_id: str):
        """恢复文档：最新版本状态改回 active（已归档 → 正常文档）。"""
        from agent_base.storage.pg import doc_set_status

        if not doc_set_status(doc_id, "active"):
            raise HTTPException(404, f"文档 {doc_id} 不存在")
        return {"status": "active", "doc_id": doc_id}

    @app.post("/api/documents/batch-delete", dependencies=[Depends(_verify_admin)])
    def batch_delete_documents(body: dict[str, Any]):
        """v0.30.6: 批量删除文档（多选清空场景）。逐篇软删，失败不阻断其余。"""
        doc_ids = body.get("doc_ids", [])
        if not isinstance(doc_ids, list) or not doc_ids:
            raise HTTPException(400, "doc_ids 必填（非空数组）")
        if len(doc_ids) > 500:
            raise HTTPException(400, "单次最多删除 500 篇，请分批操作")
        deleted = 0
        errors: list[dict[str, str]] = []
        for did in doc_ids:
            try:
                _delete_document_core(str(did))
                deleted += 1
            except Exception as exc:
                errors.append({"doc_id": str(did), "error": str(exc)[:120]})
        return {"status": "batch_deleted", "deleted": deleted, "errors": errors}

    @app.get("/api/documents/trash", dependencies=[Depends(_verify_admin)])
    def trash_list():
        """v0.46: 回收站列表：软删文档（含删除时间、剩余天数、版本数）。"""
        try:
            from agent_base.storage.pg import doc_trash_list

            retention = get_settings().retention_days
            rows = doc_trash_list(retention_days=retention)
            for d in rows:
                d["doc_name"] = _doc_display_name(
                    str(d.get("metadata", {}).get("doc_name") or d.get("doc_id", ""))
                )
            return {"documents": rows}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"读取回收站列表失败: {e}")

    @app.post("/api/documents/{doc_id}/recover", dependencies=[Depends(_verify_admin)])
    def recover_document(doc_id: str):
        """v0.46: 从回收站恢复：最新版本转 active + 重新向量化；历史版本转 archived。"""
        try:
            from agent_base.storage.pg import doc_delete, doc_recover, doc_set_chunk_ids
            from agent_base.storage.cache import invalidate_pattern

            result = doc_recover(doc_id)
            if result is None:
                raise HTTPException(404, f"文档 {doc_id} 不在回收站")
            content = result["content"]
            category = str(result.get("metadata", {}).get("category", ""))
            chunks = list({c["chunk_id"]: c for c in _parse_content_to_chunks(doc_id, content, category)}.values())
            chunk_ids = [c["chunk_id"] for c in chunks]

            runtime = get_runtime()
            vs = runtime["vector_store"]
            from agent_base.indexing.vector_index import _qdrant_point_id

            point_ids = [_qdrant_point_id(cid) for cid in chunk_ids]
            try:
                vs.add_texts(
                    texts=[c["text"] for c in chunks],
                    metadatas=[c.get("metadata", {}) for c in chunks],
                    ids=point_ids,
                )
            except Exception as exc:
                try:
                    vs.delete(ids=point_ids)
                except Exception:
                    pass
                doc_delete(doc_id)  # 写向量失败 → 退回回收站，用户可重试
                raise HTTPException(500, f"恢复索引写入失败，文档已退回回收站: {exc}")

            doc_set_chunk_ids(doc_id, result["version"], chunk_ids)
            invalidate_pattern("rag:cache:*")
            return {"ok": True, "doc_id": doc_id, "version": result["version"], "chunk_count": len(chunk_ids)}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"恢复文档失败: {e}")

    @app.post("/api/documents/batch-recover", dependencies=[Depends(_verify_admin)])
    def batch_recover_documents(body: dict[str, Any]):
        """v0.46: 批量恢复回收站文档。逐篇串行，失败不阻断其余。"""
        doc_ids = body.get("doc_ids", [])
        if not isinstance(doc_ids, list) or not doc_ids:
            raise HTTPException(400, "doc_ids 必填（非空数组）")
        results: list[dict[str, Any]] = []
        recovered = 0
        for did in doc_ids:
            try:
                recover_document(str(did))
                recovered += 1
                results.append({"doc_id": str(did), "ok": True})
            except HTTPException as exc:
                results.append({"doc_id": str(did), "ok": False, "error": exc.detail})
            except Exception as exc:
                results.append({"doc_id": str(did), "ok": False, "error": str(exc)[:120]})
        return {"status": "batch_recovered", "recovered": recovered, "results": results}

    @app.delete("/api/documents/{doc_id}/purge", dependencies=[Depends(_verify_admin)])
    def purge_document(doc_id: str):
        """v0.46: 彻底删除回收站文档（物理清除全部版本 + 标签，不可恢复）。"""
        try:
            from agent_base.storage.pg import doc_purge
            from agent_base.storage.cache import invalidate_pattern

            ok = doc_purge(doc_id)
            if not ok:
                raise HTTPException(404, f"文档 {doc_id} 不存在")
            invalidate_pattern("rag:cache:*")
            return {"ok": True, "doc_id": doc_id}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"彻底删除失败: {e}")

    @app.post("/api/documents/batch-purge", dependencies=[Depends(_verify_admin)])
    def batch_purge_documents(body: dict[str, Any]):
        """v0.46: 批量彻底删除回收站文档。"""
        doc_ids = body.get("doc_ids", [])
        if not isinstance(doc_ids, list) or not doc_ids:
            raise HTTPException(400, "doc_ids 必填（非空数组）")
        results: list[dict[str, Any]] = []
        purged = 0
        for did in doc_ids:
            try:
                purge_document(str(did))
                purged += 1
                results.append({"doc_id": str(did), "ok": True})
            except HTTPException as exc:
                results.append({"doc_id": str(did), "ok": False, "error": exc.detail})
            except Exception as exc:
                results.append({"doc_id": str(did), "ok": False, "error": str(exc)[:120]})
        return {"status": "batch_purged", "purged": purged, "results": results}

    @app.post("/api/handoff/polish", dependencies=[Depends(_verify_agent)])
    def handoff_polish_api(body: dict[str, Any]):
        """客服回复润色：只改语气不改原意，不检索、不落库、不代答。

        请求体：{text, style}，style ∈ polite(更礼貌) | concise(更简洁) | professional(更专业)。
        返回：{ok: true, text} 或 {ok: false, error}（LLM 不可用时降级，前端保留原文本）。
        """
        text = str(body.get("text", "")).strip()
        style = str(body.get("style", "polite")).strip()
        if not text:
            return {"ok": False, "error": "内容不能为空"}
        if style not in {"polite", "concise", "professional"}:
            style = "polite"
        style_desc = {
            "polite": "更礼貌、更有温度，多使用敬语和安抚性措辞",
            "concise": "更简洁精炼，去掉冗余客套，突出要点",
            "professional": "更专业严谨，术语准确、条理清晰",
        }[style]
        try:
            from agent_base.config import load_yaml as _ly
            from agent_base.llms import build_chat_model
            from langchain_core.prompts import ChatPromptTemplate

            _app = _ly(_project_path("configs/app.yaml")) or {}
            _lc = _app.get("llm") or {}
            model = build_chat_model(
                provider=_lc.get("provider", "none"),
                model=_lc.get("model"),
                base_url=_lc.get("base_url"),
                api_key_env=_lc.get("api_key_env", "DASHSCOPE_API_KEY"),
                temperature=float(_lc.get("temperature", 0.1)),
                timeout=float(_lc.get("timeout", 30)) if _lc.get("timeout") else 30,
            )
            if model is None:
                return {"ok": False, "error": "LLM 未配置（provider=none），无法润色"}
            chain = (
                ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "你是电商人工客服的文本润色助手。只改写语气和表达，"
                            "严格保持原意与事实，不新增信息、不检索资料、不代替客服回答。"
                            f"目标风格：{style_desc}。直接输出润色后的文本，不要任何解释或引号。",
                        ),
                        ("user", "{text}"),
                    ]
                )
                | model
            )
            result = chain.invoke({"text": text})
            out = str(getattr(result, "content", result) or "").strip()
            if not out:
                return {"ok": False, "error": "润色结果为空"}
            return {"ok": True, "text": out, "style": style}
        except Exception as e:
            return {"ok": False, "error": f"润色失败: {e}"}

    @app.post("/api/handoff/{session_id}", dependencies=[Depends(_verify_user)])
    def trigger_handoff(session_id: str, request: Request, body: dict[str, Any] | None = None):
        """v0.48: 触发转人工（用户端按钮/检测命中）→ 进入待接入队列。"""
        try:
            # BUG-15: 空会话（未发消息）转人工时绑定 owner，保证买家可读自己会话
            from agent_base.auth import verify_token
            from agent_base.storage.pg import ensure_session_owner

            _auth = request.headers.get("Authorization", "")
            _token = _auth[7:].strip() if _auth.startswith("Bearer ") else request.headers.get("X-Admin-Token", "")
            if _token:
                _u = verify_token(_token) or ""
                if _u:
                    ensure_session_owner(session_id, _u)

            from agent_base.storage.pg import handoff_trigger

            reason = str((body or {}).get("reason", ""))[:200]
            return handoff_trigger(session_id, reason)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"触发转人工失败: {e}")

    @app.get("/api/handoff/queue", dependencies=[Depends(_verify_agent)])
    def handoff_queue_api():
        """v0.48: 待接入队列（pending + active，含等待时长/原因/用户最后消息）。"""
        try:
            from agent_base.storage.pg import handoff_queue

            s = get_settings()
            rows = handoff_queue(s.handoff_pending_timeout, s.handoff_idle_timeout)
            # v0.50: 客服端用户上下文——情绪/意图（实时规则计算，数据不落库）
            try:
                from agent_base.agents.emotion import detect_emotion
                from agent_base.retrieval.intent_router import route_question

                for r in rows:
                    msg = r.get("last_user_message") or ""
                    if msg:
                        emo = detect_emotion(msg)
                        r["emotion"] = emo.get("label", "neutral")
                        r["emotion_intensity"] = round(float(emo.get("intensity", 0)), 2)
                        try:
                            _route = route_question(msg)
                            r["intent"] = _route.intent
                            r["intent_confidence"] = round(float(_route.confidence), 2)
                        except Exception:
                            r["intent"] = ""
                            r["intent_confidence"] = 0.0
                    else:
                        r["emotion"] = "neutral"
                        r["emotion_intensity"] = 0.0
                        r["intent"] = ""
                        r["intent_confidence"] = 0.0
            except Exception:
                pass
            return {"handoffs": rows}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"读取转人工队列失败: {e}")

    @app.post("/api/handoff/{session_id}/reply", dependencies=[Depends(_verify_agent)])
    def handoff_reply_api(session_id: str, body: dict[str, Any], request: Request):
        """v0.48: 人工回复 → 写入会话（role=agent）+ 置 active。"""
        content = str(body.get("content", "")).strip()
        if not content:
            raise HTTPException(400, "回复内容不能为空")
        try:
            from agent_base.storage.pg import handoff_reply
            from agent_base.storage.cache import invalidate_pattern

            # BUG-19: 从 token 解析客服身份写入 agent_name（by_agent 统计依赖），
            # 优先服务端解析，body.agent_name 仅作兜底
            from agent_base.auth import get_user_role, verify_token

            _auth = request.headers.get("Authorization", "")
            _token = _auth[7:].strip() if _auth.startswith("Bearer ") else request.headers.get("X-Admin-Token", "")
            agent_name = ""
            if _token:
                _u = verify_token(_token) or ""
                if _u and get_user_role(_u) == "agent":
                    agent_name = _u
            if not agent_name:
                agent_name = str(body.get("agent_name", ""))
            result = handoff_reply(session_id, content, agent_name=agent_name)
            invalidate_pattern("rag:cache:*")
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"人工回复失败: {e}")

    @app.post("/api/handoff/{session_id}/resolve", dependencies=[Depends(_verify_agent)])
    def handoff_resolve_api(session_id: str, body: dict[str, Any]):
        """v0.48: 转回 AI（mode=ai）或关闭会话（mode=close）。"""
        try:
            from agent_base.storage.pg import handoff_resolve

            mode = str(body.get("mode", "ai"))
            return handoff_resolve(session_id, mode)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"转回 AI 失败: {e}")

    @app.get("/api/handoff/stats", dependencies=[Depends(_verify_admin)])
    def handoff_stats():
        """v0.49: 人工转接状态统计（管理员只读，无聊天内容）。"""
        try:
            from agent_base.storage.pg import _conn

            with _conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT status, COUNT(*) FROM chat_handoffs GROUP BY status")
                rows = dict(cur.fetchall())
                cur.execute(
                    "SELECT COALESCE(AVG(EXTRACT(EPOCH FROM (NOW() - created_at))), 0) "
                    "FROM chat_handoffs WHERE status IN ('pending','active')"
                )
                avg_wait = float(cur.fetchone()[0])
                # 近 7 天趋势（按天统计转人工数）
                cur.execute(
                    "SELECT DATE(created_at) AS d, COUNT(*) FROM chat_handoffs "
                    "WHERE created_at >= NOW() - INTERVAL '7 days' GROUP BY DATE(created_at) ORDER BY d"
                )
                daily = [
                    {"date": r[0].isoformat() if r[0] else None, "count": int(r[1])}
                    for r in cur.fetchall()
                ]
                # 总会话数（转人工率分母）
                cur.execute("SELECT COUNT(DISTINCT session_id) FROM chat_messages")
                total_sessions = int(cur.fetchone()[0])
                # 平均解决时长（resolved/closed 的 创建→解决 耗时）
                cur.execute(
                    "SELECT AVG(EXTRACT(EPOCH FROM (resolved_at - created_at))) "
                    "FROM chat_handoffs WHERE resolved_at IS NOT NULL AND status='resolved'"
                )
                avg_resolve = float(cur.fetchone()[0] or 0)
                # 客服处理量（按 agent 统计）
                cur.execute(
                    "SELECT h.agent_name, COALESCE(u.display_name, h.agent_name), COUNT(*) "
                    "FROM chat_handoffs h "
                    "LEFT JOIN admin_users u ON u.username = h.agent_name "
                    "WHERE h.agent_name IS NOT NULL AND h.agent_name != '' "
                    "GROUP BY h.agent_name, u.display_name ORDER BY 3 DESC"
                )
                by_agent = [
                    {"agent": r[1], "username": r[0], "count": int(r[2])}
                    for r in cur.fetchall()
                ]
                # 转人工原因分布（情绪 / 主动请求 / 其他）
                cur.execute("SELECT reason FROM chat_handoffs WHERE reason IS NOT NULL AND reason != ''")
                reason_dist = {"emotion": 0, "manual": 0, "other": 0}
                for (rr,) in cur.fetchall():
                    if "情绪" in str(rr):
                        reason_dist["emotion"] += 1
                    elif "主动" in str(rr):
                        reason_dist["manual"] += 1
                    else:
                        reason_dist["other"] += 1
            total = sum(int(rows.get(s, 0)) for s in ("pending", "active", "expired", "closed", "resolved"))
            return {
                "stats": {
                    "total": total,
                    "pending": int(rows.get("pending", 0)),
                    "active": int(rows.get("active", 0)),
                    "expired": int(rows.get("expired", 0)),
                    "closed": int(rows.get("closed", 0)),
                    "resolved": int(rows.get("resolved", 0)),
                    "avg_waiting_secs": int(avg_wait),
                    "daily": daily,
                    "total_sessions": total_sessions,
                    "handoff_rate": round(total / total_sessions, 4) if total_sessions else 0,
                    "avg_resolve_secs": int(avg_resolve),
                    "expired_rate": round(int(rows.get("expired", 0)) / max(1, total), 4),
                    "by_agent": by_agent,
                    "reason_dist": reason_dist,
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"读取转人工统计失败: {e}")

    @app.get("/api/handoff/recent", dependencies=[Depends(_verify_admin)])
    def handoff_recent():
        """v0.50: 最近转人工记录（管理员只读，最近 20 条，无聊天内容）。"""
        try:
            from agent_base.storage.pg import _conn

            with _conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT session_id, status, reason, created_at FROM chat_handoffs "
                    "ORDER BY created_at DESC LIMIT 20"
                )
                rows = cur.fetchall()
            return {
                "handoffs": [
                    {
                        "session_id": r[0],
                        "status": r[1],
                        "reason": r[2] or "",
                        "created_at": r[3].isoformat() if r[3] else None,
                    }
                    for r in rows
                ]
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"读取转人工记录失败: {e}")

    @app.get("/api/handoff/{session_id}")
    def handoff_status_api(session_id: str):
        """v0.49.1: 转人工状态查询（买家端轮询用，免鉴权）；无记录返回 idle。"""
        try:
            from agent_base.config import deep_get as _dg
            from agent_base.config import load_yaml as _ly
            from agent_base.storage.pg import handoff_check, handoff_rating_get

            _cfg = _ly("configs/app.yaml") or {}
            ho = handoff_check(
                session_id,
                pending_timeout=int(_dg(_cfg, "handoff.pending_timeout_min", 15)) * 60,
                idle_timeout=int(_dg(_cfg, "handoff.idle_timeout_min", 20)) * 60,
            )
            if ho is None:
                return {"session_id": session_id, "status": "idle"}
            rating = handoff_rating_get(session_id)
            return {
                "session_id": session_id,
                "status": ho["status"],
                "reason": ho["reason"],
                "rating": rating.get("rating", 0),
                "rating_comment": rating.get("comment", ""),
                "rated_at": rating.get("rated_at"),
            }
        except Exception as e:
            raise HTTPException(500, f"查询转人工状态失败: {e}")

    @app.post("/api/handoff/{session_id}/rating", dependencies=[Depends(_verify_user)])
    def handoff_rating_api(session_id: str, body: dict[str, Any]):
        """人工客服结束后，由买家端提交 1-5 星服务评价。"""
        rating = int(body.get("rating") or 0)
        if rating < 1 or rating > 5:
            raise HTTPException(400, "评分必须是 1-5 星")
        comment = str(body.get("comment") or "").strip()
        from agent_base.storage.pg import handoff_rate

        return handoff_rate(session_id, rating, comment)

    @app.delete("/api/documents/{doc_id}/versions/{version}", dependencies=[Depends(_verify_admin)])
    def delete_doc_version(doc_id: str, version: int):
        """v0.30.5: 删除单个历史版本（软删；active 版本不可删）。"""
        from agent_base.storage.pg import doc_delete_version
        from agent_base.storage.cache import invalidate_pattern
        ok = doc_delete_version(doc_id, version)
        if not ok:
            raise HTTPException(400, "版本不存在或为当前生效版本（不可删除）")
        invalidate_pattern("rag:cache:*")
        return {"status": "version_deleted", "doc_id": doc_id, "version": version}

    @app.post("/api/documents/update", dependencies=[Depends(_verify_admin)])
    def update_document(body: dict[str, Any]):
        """更新文档（P16 编排）：传 content → 重分块 → 向量化 → 写 Qdrant → 回填 chunk_ids → 删旧向量。

        请求体：
        - doc_id: str（必填）
        - content: str（必填，新全文内容）
        - category: str（可选）
        """
        if "chunk_ids" in body:
            raise HTTPException(400, "update 不接受 chunk_ids，索引由服务端自动管理")
        doc_id = body.get("doc_id", "")
        content = body.get("content", "")
        if not doc_id or not content:
            raise HTTPException(400, "doc_id 和 content 必填")

        # P19 D1: 精审硬约束——非 approved 禁止更新
        try:
            from agent_base.knowledge_factory import get_tag
            _tag = get_tag(doc_id)
            if _tag is None or _tag.status != "approved":
                raise HTTPException(
                    403,
                    f"文档 {doc_id} 未通过精审（status={_tag.status if _tag else '无标签'}），禁止更新",
                )
        except HTTPException:
            raise
        except Exception:
            pass

        from agent_base.storage.pg import doc_upsert, doc_versions
        from agent_base.storage.cache import invalidate_pattern

        # 1. 读旧版本 chunk_ids
        all_versions = doc_versions(doc_id)
        old_chunk_ids: list[str] = []
        for v in all_versions:
            old_chunk_ids.extend(v.get("chunk_ids", []))

        # 2. 解析新内容为 chunks
        try:
            new_chunks = _parse_content_to_chunks(doc_id, content, body.get("category", ""))
        except Exception as e:
            raise HTTPException(400, f"内容解析失败: {e}")

        # 3. version+1 + 写 PG（含 content）
        # v0.27.3：chunk 去重——相同段落（sha256 相同）只索引一次，
        # 避免 PG chunk_ids 数组重复与 Qdrant 不一致
        new_chunks = list({c["chunk_id"]: c for c in new_chunks}.values())
        new_chunk_ids = [c["chunk_id"] for c in new_chunks]
        new_version = doc_upsert(doc_id, new_chunk_ids, metadata={"category": body.get("category", "")}, content=content)

        # 4. 向量化 + 写 Qdrant
        try:
            runtime = get_runtime()
            vs = runtime["vector_store"]
            from agent_base.indexing.vector_index import _qdrant_point_id
            texts = [c["text"] for c in new_chunks]
            metas = [c["metadata"] for c in new_chunks]
            point_ids = [_qdrant_point_id(cid) for cid in new_chunk_ids]
            vs.add_texts(texts=texts, metadatas=metas, ids=point_ids)
        except Exception as e:
            # 向量写失败 → 回滚 PG
            from agent_base.storage.pg import _conn
            try:
                vs.delete(ids=point_ids)  # 补偿清理可能已写入的部分向量
            except Exception:
                pass
            try:
                with _conn() as conn:
                    cur = conn.cursor()
                    cur.execute("DELETE FROM documents WHERE doc_id=%s AND version=%s", (doc_id, new_version))
            except Exception:
                pass
            raise HTTPException(500, f"向量化写入失败，PG 已回滚: {e}")

        # 5. 删旧向量（仅删不在新版本中的旧 chunk，避免误删同 ID 的新写入向量）
        new_id_set = set(new_chunk_ids)
        old_to_delete = [cid for cid in old_chunk_ids if cid not in new_id_set]
        if old_to_delete:
            try:
                from agent_base.indexing.vector_index import _qdrant_point_id
                old_point_ids = [_qdrant_point_id(cid) for cid in old_to_delete]
                vs.delete(ids=old_point_ids)
            except Exception:
                pass

        # 5.5 P18: 摘要同步（开关开启时生效，失败不阻断主链路）
        try:
            summary_store = get_runtime().get("summary_store")
            if summary_store is not None:
                from agent_base.storage.documents import sync_document_summaries
                sync_document_summaries(doc_id, new_chunks, summary_store)
        except Exception:
            pass

        # 6. 缓存失效
        invalidate_pattern("rag:cache:*")
        return {"status": "updated", "doc_id": doc_id, "version": new_version, "chunk_count": len(new_chunk_ids), "deleted_old_vectors": len(old_to_delete)}

    @app.get("/api/documents/{doc_id}/versions", dependencies=[Depends(_verify_admin)])
    def list_versions(doc_id: str):
        """列出文档的版本历史。

        Args:
            doc_id: 文档 ID。
        """
        from agent_base.storage.pg import doc_versions
        return {"doc_id": doc_id, "versions": doc_versions(doc_id)}

    @app.post("/api/documents/{doc_id}/restore/{version}", dependencies=[Depends(_verify_admin)])
    def restore_version(doc_id: str, version: int):
        """回滚文档：读版本 content+chunk_ids → 重写 Qdrant 向量 → 版本激活。"""
        from agent_base.storage.pg import doc_restore_version
        from agent_base.storage.cache import invalidate_pattern

        restored = doc_restore_version(doc_id, version)
        if restored is None:
            raise HTTPException(404, f"Version {version} not found for doc '{doc_id}'")

        # P19 D1: 精审硬约束——非 approved 禁止恢复索引
        try:
            from agent_base.knowledge_factory import get_tag
            _tag = get_tag(doc_id)
            if _tag is None or _tag.status != "approved":
                raise HTTPException(
                    403,
                    f"文档 {doc_id} 未通过精审（status={_tag.status if _tag else '无标签'}），禁止恢复",
                )
        except HTTPException:
            raise
        except Exception:
            pass

        restored_content = restored["content"]

        # 读当前 active 版本的 chunk_ids，用于 restore 后清理不在目标版本中的向量
        from agent_base.storage.pg import doc_versions as _doc_versions
        active_chunk_ids: list[str] = []
        for v in _doc_versions(doc_id):
            if v.get("status") == "active":
                active_chunk_ids.extend(v.get("chunk_ids", []))

        # 如果该版本有 content，重新分块并重建向量（P16-修复：
        # 不要求 chunk_id 与历史版本一致——导入器文档的 chunk_id 是
        # build_chunk_id 体系，与 _parse_content_to_chunks 的 sha256 体系不同，
        # 按目标版本 content 重建索引才能保证向量可恢复）
        vector_count = 0
        new_chunk_ids: list[str] = []
        if restored_content:
            try:
                _doc_meta = restored.get("metadata") or {}
                chunks = _parse_content_to_chunks(
                    doc_id,
                    restored_content,
                    "",
                    doc_name=str(_doc_meta.get("doc_name") or ""),
                )
                runtime = get_runtime()
                vs = runtime["vector_store"]
                from agent_base.indexing.vector_index import _qdrant_point_id
                # v0.27.3：去重（相同段落只索引一次）
                chunks = list({c["chunk_id"]: c for c in chunks}.values())
                new_chunk_ids = [c["chunk_id"] for c in chunks]
                if new_chunk_ids:
                    vs.add_texts(
                        texts=[c["text"] for c in chunks],
                        metadatas=[c["metadata"] for c in chunks],
                        ids=[_qdrant_point_id(cid) for cid in new_chunk_ids],
                    )
                    vector_count = len(new_chunk_ids)
            except Exception as e:
                raise HTTPException(500, f"向量恢复失败: {e}")

        # 删除 active 版本中不属于新索引的向量（避免 restore 后残留孤儿向量）
        new_set = set(new_chunk_ids)
        removed_ids = [cid for cid in active_chunk_ids if cid not in new_set]
        if removed_ids:
            try:
                vs = get_runtime()["vector_store"]
                from agent_base.indexing.vector_index import _qdrant_point_id
                vs.delete(ids=[_qdrant_point_id(cid) for cid in removed_ids])
            except Exception:
                pass

        # P18: 摘要同步（开关开启时生效，失败不阻断主链路）
        if restored_content:
            try:
                summary_store = get_runtime().get("summary_store")
                if summary_store is not None:
                    from agent_base.storage.documents import sync_document_summaries
                    sync_document_summaries(doc_id, chunks, summary_store)
            except Exception:
                pass

        # v0.30.2：切换 active 版本指针（不生成新版本，版本号不递增）
        from agent_base.storage.pg import doc_switch_version
        doc_switch_version(doc_id, version, new_chunk_ids)
        invalidate_pattern("rag:cache:*")
        return {"status": "restored", "doc_id": doc_id, "version": version, "chunk_ids": new_chunk_ids, "vectors_synced": vector_count, "removed_vectors": len(removed_ids)}

    @app.post("/api/documents/ingest", dependencies=[Depends(_verify_admin)])
    def ingest_file(body: dict[str, Any]):
        """P16 入库端点：文件内容 → PG documents + Qdrant 向量 + 回填 chunk_ids。

        请求体：
        - doc_id: str（必填）
        - content: str（必填，文档全文）
        - category: str（可选）
        - doc_type: str（可选，决定切分档位，P24a）
        """
        doc_id = body.get("doc_id", "")
        content = body.get("content", "")
        if not doc_id or not content:
            raise HTTPException(400, "doc_id 和 content 必填")

        from agent_base.storage.documents import TagNotApprovedError, ingest_document
        try:
            result = ingest_document(
                doc_id=doc_id,
                content=content,
                vector_store=get_runtime()["vector_store"],
                category=body.get("category", ""),
                summary_store=get_runtime().get("summary_store"),
                doc_type=body.get("doc_type", ""),
            )
            return result
        except TagNotApprovedError as e:
            raise HTTPException(403, str(e))
        except RuntimeError as e:
            raise HTTPException(500, str(e))
        except Exception as e:
            raise HTTPException(400, f"入库失败: {e}")

# P9-07：缓存统计
    @app.get("/api/stats/cache", dependencies=[Depends(_verify_admin)])
    def cache_stats():
        """返回 Redis 缓存命中率统计。"""
        from agent_base.storage.cache import cache_stats as _stats
        return _stats()

    @app.get("/api/stats/vector", dependencies=[Depends(_verify_admin)])
    def vector_stats():
        """P21: 向量库点数统计（dense/sparse/summaries 三 collection）。"""
        try:
            from qdrant_client import QdrantClient
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            client = QdrantClient(url=url)
            names = ["ecommerce_chunks", "ecommerce_chunks_sparse", "ecommerce_summaries"]
            counts = {}
            for name in names:
                try:
                    counts[name] = client.get_collection(name).points_count
                except Exception:
                    counts[name] = None
            return {"qdrant_url": url, "collections": counts}
        except Exception as exc:
            return {"qdrant_url": None, "collections": {}, "error": str(exc)}

    # 观测底座：运营看板统计接口
    @app.get("/api/stats/token", dependencies=[Depends(_verify_admin)])
    def token_stats(days: int = 7, group_by: str = "day"):
        from agent_base.storage.pg import token_usage_stats

        return token_usage_stats(days=max(1, min(90, days)), group_by=group_by)

    @app.get("/api/stats/tool-calls", dependencies=[Depends(_verify_admin)])
    def tool_calls_stats_api(days: int = 7):
        from agent_base.storage.pg import tool_calls_stats as _tool_stats

        return _tool_stats(days=max(1, min(90, days)))

    @app.get("/api/stats/failures", dependencies=[Depends(_verify_admin)])
    def failures_stats_api(days: int = 7):
        from agent_base.storage.pg import failure_stats as _fail_stats

        return _fail_stats(days=max(1, min(90, days)))

    @app.get("/api/stats/failure-events", dependencies=[Depends(_verify_admin)])
    def failure_events_api(days: int = 7, limit: int = 50):
        from agent_base.storage.pg import recent_failure_events

        return recent_failure_events(
            days=max(1, min(90, days)), limit=max(1, min(200, limit))
        )

    @app.get("/api/stats/overview", dependencies=[Depends(_verify_admin)])
    def stats_overview_api(days: int = 7):
        from agent_base.storage.pg import (
            eval_feedback_list,
            failure_stats,
            token_usage_stats,
            tool_calls_stats,
        )

        days = max(1, min(90, days))
        tok = token_usage_stats(days=days, group_by="day")
        tool = tool_calls_stats(days=days)
        fail = failure_stats(days=days)
        pending = len(eval_feedback_list(status="pending", limit=500))
        tool_total = sum(r["calls"] for r in tool["rows"])
        tool_ok = sum(r["success"] for r in tool["rows"])
        err_total = sum(r["count"] for r in fail["by_module"])
        return {
            "days": days,
            "total_tokens": tok["total_tokens"],
            "total_cost": tok.get("total_cost", 0),
            "llm_calls": sum(r["calls"] for r in tok["rows"]),
            "tool_calls": tool_total,
            "tool_success_rate": round(tool_ok / max(1, tool_total), 4),
            "error_events": err_total,
            "pending_feedback": pending,
        }

    # 知识运营 Agent：自然语言指令 → Plan-Execute-Reflect → 留痕
    @app.post("/api/knowledge-ops/execute", dependencies=[Depends(_verify_admin)])
    def knowledge_ops_execute(body: dict[str, Any]):
        command = str(body.get("command") or "").strip()
        if not command:
            raise HTTPException(status_code=400, detail="运营指令不能为空")
        from agent_base.agents.knowledge_ops import run_knowledge_ops

        result = run_knowledge_ops(command, operator="admin")
        return result

    @app.get("/api/knowledge-ops/tools", dependencies=[Depends(_verify_admin)])
    def knowledge_ops_tools():
        from agent_base.agents.knowledge_tools import list_tools

        return {"tools": list_tools()}

    # 评测反馈：失败归因 + 数据飞轮状态流转
    @app.get("/api/feedback", dependencies=[Depends(_verify_admin)])
    def feedback_list(status: str | None = None, limit: int = 100):
        from agent_base.storage.pg import eval_feedback_list

        return {"items": eval_feedback_list(status=status, limit=max(1, min(500, limit)))}

    @app.post("/api/feedback/{feedback_id}/status", dependencies=[Depends(_verify_admin)])
    def feedback_update_status(feedback_id: int, body: dict[str, Any]):
        status = str(body.get("status") or "").strip()
        regression = float(body.get("regression", 0) or 0)
        if status not in {"pending", "supplemented", "regressed"}:
            raise HTTPException(status_code=400, detail="非法状态")
        from agent_base.storage.pg import update_eval_feedback_status

        ok = update_eval_feedback_status(feedback_id, status, regression)
        if not ok:
            raise HTTPException(status_code=404, detail="反馈记录不存在")
        return {"ok": True, "id": feedback_id, "status": status, "regression": regression}

    # ARK 商品图生成（配置驱动 + mock 降级）
    @app.post("/api/content/image-gen", dependencies=[Depends(_verify_admin)])
    def content_image_gen(body: dict[str, Any]):
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="商品描述不能为空")
        from agent_base.multimodal import generate_product_image

        return generate_product_image(prompt)

    # ARK 商品图编辑（图生图：参考图 + 编辑指令，配置驱动 + mock 降级）
    @app.post("/api/content/image-edit", dependencies=[Depends(_verify_admin)])
    def content_image_edit(body: dict[str, Any]):
        prompt = str(body.get("prompt") or "").strip()
        image = str(body.get("image") or "").strip()
        image_url = str(body.get("image_url") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="编辑指令不能为空")
        if not image and not image_url:
            raise HTTPException(status_code=400, detail="必须提供参考图（image 或 image_url）")
        from agent_base.multimodal import edit_product_image

        return edit_product_image(prompt, image=image or None, image_url=image_url or None)

    # ── 图片知识库（Phase 2）：上传 / 列表 / 绑定 / 解析 / 删除（仅管理端） ──

    @app.post("/api/media/upload", dependencies=[Depends(_verify_admin)])
    async def media_upload(file: UploadFile = File(...), description: str = Form("")):
        """上传图片/视频到媒体知识库：校验类型/大小 → 落盘 → 建 media_documents 记录（parse_type 自动判定）。"""
        from agent_base.media_library import handle_media_upload

        content = await file.read()
        try:
            payload = handle_media_upload(
                file.filename or "image", content, description=str(description or "").strip()
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return payload

    @app.get("/api/media/list", dependencies=[Depends(_verify_admin)])
    def media_list(status: str | None = None, product_id: str | None = None, limit: int = 200):
        """图片知识库列表（可按审核状态 / 绑定商品过滤）。"""
        from agent_base.storage.pg import media_document_list

        return {
            "items": media_document_list(
                status=status, product_id=product_id, limit=max(1, min(500, limit))
            )
        }

    @app.post("/api/media/{media_id}/bind", dependencies=[Depends(_verify_admin)])
    def media_bind(media_id: int, body: dict[str, Any]):
        """绑定/解绑图片到商品（product_id 为空串表示解绑）。"""
        product_id = str(body.get("product_id") or "").strip()
        from agent_base.storage.pg import media_document_bind

        if not media_document_bind(media_id, product_id):
            raise HTTPException(status_code=404, detail="图片记录不存在")
        return {"ok": True, "id": media_id, "product_id": product_id}

    @app.post("/api/media/{media_id}/parse", dependencies=[Depends(_verify_admin)])
    def media_parse(media_id: int):
        """触发媒体解析（图片 OCR/视觉理解；视频抽帧+视觉理解；入 PG 任务队列异步执行）。"""
        from agent_base.storage.pg import media_document_get, task_enqueue

        row = media_document_get(media_id)
        if not row:
            raise HTTPException(status_code=404, detail="媒体记录不存在")
        # 按类型分发任务：视频走抽帧管线，图片走 OCR/视觉管线
        is_video = str(row.get("mime_type") or "").startswith("video/") or row.get("parse_type") == "video"
        task_type = "media_parse_video" if is_video else "media_parse"
        task_id = task_enqueue(task_type, {"media_id": int(media_id)}, owner="admin")
        if not task_id:
            raise HTTPException(status_code=503, detail="任务入队失败（数据库不可用）")
        return {"ok": True, "id": media_id, "task_id": task_id, "status": "pending", "task_type": task_type}

    @app.post("/api/media/{media_id}/status", dependencies=[Depends(_verify_admin)])
    def media_set_status(media_id: int, body: dict[str, Any]):
        """审核状态流转：pending / approved / rejected。"""
        status = str(body.get("status") or "").strip()
        if status not in {"pending", "approved", "rejected"}:
            raise HTTPException(status_code=400, detail="非法状态")
        from agent_base.storage.pg import media_document_set_status

        if not media_document_set_status(media_id, status):
            raise HTTPException(status_code=404, detail="图片记录不存在")
        return {"ok": True, "id": media_id, "status": status}

    @app.delete("/api/media/{media_id}", dependencies=[Depends(_verify_admin)])
    def media_delete(media_id: int):
        """删除图片：先删记录，再清理磁盘文件。"""
        from agent_base.media_library import handle_media_delete

        payload = handle_media_delete(media_id)
        if not payload.get("ok"):
            raise HTTPException(status_code=404, detail="图片记录不存在")
        return payload

    # ── 文件清洗工作台（两段式入库第一段）：上传解析 → 人工清洗 → 推送到知识入库 ──

    @app.post("/api/clean/upload", dependencies=[Depends(_verify_admin)])
    async def clean_upload(file: UploadFile = File(...)):
        """上传文件 → 解析清洗 → 存草稿（不自动入库，等待人工确认推送）。"""
        from agent_base.cleaning import handle_clean_upload

        content = await file.read()
        try:
            return handle_clean_upload(file.filename or "upload", content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            # 任何未知异常都转成可读 JSON（避免前端收到 HTML 500 解析失败）
            from agent_base.monitoring.logger import log_event

            try:
                log_event("ERROR", "cleaning", "clean_upload_failed", {"error": str(exc)[:300]})
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"文件解析失败：{str(exc)[:200]}")

    @app.get("/api/clean/list", dependencies=[Depends(_verify_admin)])
    def clean_list(limit: int = 100):
        """清洗草稿列表。"""
        from agent_base.storage.pg import clean_draft_list

        return {"items": clean_draft_list(limit=max(1, min(500, limit)))}

    @app.get("/api/clean/{draft_id}", dependencies=[Depends(_verify_admin)])
    def clean_get(draft_id: int):
        """单条草稿（含原文与清洗后文本）。"""
        from agent_base.storage.pg import clean_draft_get

        draft = clean_draft_get(draft_id)
        if not draft:
            raise HTTPException(status_code=404, detail="清洗草稿不存在")
        return draft

    @app.put("/api/clean/{draft_id}", dependencies=[Depends(_verify_admin)])
    def clean_update(draft_id: int, body: dict[str, Any]):
        """保存人工清洗后的文本。"""
        from agent_base.storage.pg import clean_draft_update

        text = str(body.get("text") or "")
        if not clean_draft_update(draft_id, text):
            raise HTTPException(status_code=404, detail="清洗草稿不存在")
        return {"ok": True, "id": draft_id}

    @app.post("/api/clean/{draft_id}/polish", dependencies=[Depends(_verify_admin)])
    def clean_polish(draft_id: int):
        """AI 辅助整理格式：清洗文本 → 规范知识库 Markdown（写回 cleaned_text）。"""
        from agent_base.cleaning import handle_clean_polish

        try:
            return handle_clean_polish(draft_id)
        except ValueError as exc:
            code = 404 if "不存在" in str(exc) else 400
            raise HTTPException(status_code=code, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=f"AI 整理失败：{exc}")

    @app.post("/api/clean/{draft_id}/push", dependencies=[Depends(_verify_admin)])
    def clean_push(draft_id: int, body: dict[str, Any]):
        """把清洗后的文本推送到知识入库（staging 精审队列）。"""
        from agent_base.cleaning import handle_clean_push

        try:
            payload = handle_clean_push(draft_id, category=str(body.get("category") or ""))
        except ValueError as exc:
            code = 404 if "不存在" in str(exc) else 400
            raise HTTPException(status_code=code, detail=str(exc))
        return payload

    @app.delete("/api/clean/{draft_id}", dependencies=[Depends(_verify_admin)])
    def clean_delete(draft_id: int):
        """丢弃清洗草稿。"""
        from agent_base.storage.pg import clean_draft_delete

        if not clean_draft_delete(draft_id):
            raise HTTPException(status_code=404, detail="清洗草稿不存在")
        return {"ok": True, "id": draft_id}

    # ── 异步任务队列（PG 持久化：入队 / 查询 / 列表） ──

    @app.post("/api/tasks", dependencies=[Depends(_verify_admin)])
    def tasks_enqueue(body: dict[str, Any]):
        task_type = str(body.get("task_type") or "").strip()
        if not task_type:
            raise HTTPException(status_code=400, detail="task_type 不能为空")
        payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
        from agent_base.async_tasks import TASK_HANDLERS
        from agent_base.storage.pg import task_enqueue

        if task_type not in TASK_HANDLERS:
            raise HTTPException(
                status_code=400,
                detail=f"未注册的任务类型: {task_type}（可用：{', '.join(sorted(TASK_HANDLERS))}）",
            )
        task_id = task_enqueue(task_type, payload, owner=str(body.get("owner") or ""))
        if not task_id:
            raise HTTPException(status_code=503, detail="任务入队失败（数据库不可用）")
        return {"ok": True, "id": task_id, "task_type": task_type, "status": "pending"}

    @app.get("/api/tasks/{task_id}", dependencies=[Depends(_verify_admin)])
    def tasks_get(task_id: int):
        from agent_base.storage.pg import task_get

        task = task_get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return task

    @app.get("/api/tasks", dependencies=[Depends(_verify_admin)])
    def tasks_list(status: str | None = None, limit: int = 50):
        from agent_base.storage.pg import task_list

        return {"items": task_list(status=status, limit=max(1, min(200, limit)))}

    @app.post("/api/tasks/{task_id}/cancel", dependencies=[Depends(_verify_admin)])
    def tasks_cancel(task_id: int):
        """停止 pending / running 任务。"""
        from agent_base.storage.pg import task_cancel

        if not task_cancel(task_id):
            raise HTTPException(status_code=404, detail="任务不存在或不可停止")
        return {"ok": True, "id": task_id}

    @app.delete("/api/tasks/{task_id}", dependencies=[Depends(_verify_admin)])
    def tasks_delete(task_id: int):
        """删除任务记录。"""
        from agent_base.storage.pg import task_delete

        if not task_delete(task_id):
            raise HTTPException(status_code=404, detail="任务不存在")
        return {"ok": True, "id": task_id}

    # ── P14-01: 文档打标 API（全部 X-Admin-Token 鉴权） ──


    @app.get("/api/documents/tags", dependencies=[Depends(_verify_admin)])
    def list_doc_tags(doc_id: str | None = None):
        """列出文档打标记录。"""
        try:
            from agent_base.knowledge_factory import load_tags
            tags = load_tags(doc_id=doc_id)
            # 附加当前版本号（documents 表 active 版本；tags 本身无 version）
            try:
                from agent_base.storage.pg import _conn
                with _conn() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT doc_id, version FROM documents WHERE status='active'")
                    ver_map = dict(cur.fetchall())
                    # v0.46: 回收站（deleted）文档的标签不展示
                    cur.execute(
                        "SELECT DISTINCT doc_id FROM documents WHERE status IN ('active','archived')"
                    )
                    valid_ids = {r[0] for r in cur.fetchall()}
            except Exception:
                ver_map = {}
                valid_ids = set()
            items = []
            for t in tags:
                if valid_ids and t.doc_id not in valid_ids:
                    continue
                d = t.to_dict()
                d["doc_name"] = _doc_display_name(d.get("doc_id", ""))
                d["current_version"] = ver_map.get(d.get("doc_id"))
                items.append(d)
            return {"tags": items}
        except Exception as e:
            raise HTTPException(500, f"读取打标列表失败: {e}")

    @app.get("/api/documents", dependencies=[Depends(_verify_admin)])
    def document_list(status: str | None = None):
        """列出已入库文档（documents 表，RAG 数据核心）。

        P27 职责边界：文档管理只管理已入库文档（active/archived），
        不混入审核台状态（pending/returned 走知识入库页）。

        Args:
            status: 可选过滤 active / archived。

        Returns:
            {"documents": [...]}。
        """
        try:
            from agent_base.storage.pg import doc_list
            allowed = {"active", "archived"}
            if status is not None and status not in allowed:
                raise HTTPException(400, "status 仅支持 active / archived")
            rows = doc_list(status=status)
            for d in rows:
                d["doc_name"] = _doc_display_name(
                    str(d.get("metadata", {}).get("doc_name") or d.get("doc_id", ""))
                )
                d["current_version"] = d.get("version")
                # doc_type 从 document_strategy 取（已入库文档应有标签）
                try:
                    from agent_base.storage.pg import strategy_get
                    tag = strategy_get(d.get("doc_id", ""))
                    d["doc_type"] = (tag or {}).get("doc_type", "")
                except Exception:
                    d["doc_type"] = ""
            return {"documents": rows}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"读取文档列表失败: {e}")

    @app.post("/api/documents/pre-review", dependencies=[Depends(_verify_admin)])
    def pre_review(body: dict[str, Any]):
        """预审文档：返回 agent（启发式+LLM）建议的 doc_type 和 strategies。

        请求体：
        - content: str（文档内容前 2000 字符）
        - filename: str（可选，用于启发式回退）

        P19 D5：LLM 预审启用，配置读 ``configs/app.yaml`` pre_review_llm 段；agent 只建议
        （pending_fine_review），人工精审后才生效。
        """
        content = body.get("content", "")
        if not content:
            raise HTTPException(400, "content 不能为空")
        try:
            from agent_base.knowledge_factory import pre_review_document
            tag = pre_review_document(
                content_snippet=content,
                filename=body.get("filename", ""),
        llm_cfg=None,  # None → 读 app.yaml pre_review_llm 段（flash）
                prev_reject_reason=body.get("prev_reject_reason", ""),
            )
            return tag.to_dict()
        except Exception as e:
            raise HTTPException(500, f"预审失败: {e}")

    @app.post("/api/documents/batch-pre-review", dependencies=[Depends(_verify_admin)])
    def batch_pre_review(body: dict[str, Any]):
        """AI 批量审核：对全部待审文档跑 LLM 预审，刷新决策包（不直接入库）。

        上传时仅启发式预审（快、低置信）；此接口用 LLM 重新评审全部待审，
        更新 first_review 的 doc_type/confidence/reasoning/suggest_action/
        reject_hint/risk_flags。人工仍需确认/打回（P19 approved 门不变）。
        """
        try:
            from agent_base.knowledge_factory import pre_review_document
            from agent_base.storage.pg import staging_list, staging_upsert

            doc_ids = body.get("doc_ids") or []
            if not isinstance(doc_ids, list) or not doc_ids:
                raise HTTPException(400, "doc_ids 必填（非空数组）")
            if len(doc_ids) > 100:
                raise HTTPException(400, "单次最多 100 篇")

            reviewed: list[dict[str, Any]] = []
            for doc_id in doc_ids:
                st = next((r for r in staging_list(status="pending") if r["doc_id"] == doc_id), None)
                if st is None:
                    continue
                tag = pre_review_document(
                    (st.get("content") or "")[:2000],
                    filename=st.get("filename", ""),
        llm_cfg=None,  # None → 读 app.yaml pre_review_llm 段（flash）
                    prev_reject_reason=((st.get("first_review") or {}).get("prev_reject_reason") or ""),
                )
                first_review = tag.first_review or {}
                staging_upsert(
                    doc_id=doc_id,
                    content=st.get("content", ""),
                    filename=st.get("filename", ""),
                    category=st.get("category", ""),
                    status="pending",
                    review_round=int(st.get("review_round") or 1),
                    first_review=first_review,
                    reject_reason=st.get("reject_reason", ""),
                )
                reviewed.append({
                    "doc_id": doc_id,
                    "doc_type": first_review.get("type", ""),
                    "confidence": float(first_review.get("confidence", 0.0)),
                })
            get_runtime.cache_clear()
            return {"status": "batch_pre_reviewed", "reviewed": reviewed, "count": len(reviewed)}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"批量预审失败: {e}")

    @app.post("/api/documents/tags/apply", dependencies=[Depends(_verify_admin)])
    def apply_doc_tag(body: dict[str, Any]):
        """确认/修改打标并提交入库。

        请求体：
        - doc_id: str
        - doc_type: str
        - strategy: list[str]（可选，不传则按 doc_type 自动决定）
        - reviewer: str（默认 admin）
        """
        doc_id = body.get("doc_id", "")
        doc_type = body.get("doc_type", "")
        if not doc_id or not doc_type:
            raise HTTPException(400, "doc_id 和 doc_type 必填")
        try:
            from agent_base.knowledge_factory import DocTag, apply_tag, persist_tag, STRATEGY_MAP
            strategy = body.get("strategy") or STRATEGY_MAP.get(doc_type, ["default_vector"])
            # P20：暂存文档精审确认 → approved + 自动入库（CONTRACT-P20 §2.3）
            from agent_base.storage.pg import staging_get
            if staging_get(doc_id) is not None:
                from agent_base.storage.staging import approve_and_ingest
                runtime = get_runtime()
                result = approve_and_ingest(
                    doc_id=doc_id,
                    doc_type=doc_type,
                    strategy=strategy,
                    reviewer=body.get("reviewer", "admin"),
                    vector_store=runtime["vector_store"],
                    summary_store=runtime.get("summary_store"),
                )
                get_runtime.cache_clear()
                return result
            # 非暂存文档（历史数据/直接打标）：仅打标
            tag = DocTag(doc_id=doc_id, doc_type=doc_type, strategy=strategy)
            tag = apply_tag(doc_id, tag, reviewer=body.get("reviewer", "admin"))
            persist_tag(tag)
            return tag.to_dict()
        except Exception as e:
            raise HTTPException(500, f"打标失败: {e}")

    @app.post("/api/documents/tags/reject", dependencies=[Depends(_verify_admin)])
    def reject_doc_tag(body: dict[str, Any]):
        """打回文档（status → returned，原路打回不入库）。

        请求体：
        - doc_id: str
        - reason: str（可选）
        - reviewer: str（默认 admin）
        """
        doc_id = body.get("doc_id", "")
        if not doc_id:
            raise HTTPException(400, "doc_id 必填")
        try:
            from agent_base.knowledge_factory import DocTag, reject_tag, persist_tag
            tag = DocTag(doc_id=doc_id)
            tag = reject_tag(doc_id, tag, reviewer=body.get("reviewer", "admin"),
                           reason=body.get("reason", ""))
            persist_tag(tag)
            # P27：打回写短期记忆（Redis TTL + PG reject_reason 双写）
            try:
                from agent_base.storage.review_memory import save_memory
                from agent_base.storage.pg import staging_get as _stg_get
                _st = _stg_get(doc_id)
                _round = int((_st or {}).get("review_round") or 1)
                _decision = (_st or {}).get("first_review") or {}
                save_memory(doc_id, _round, body.get("reason", ""), _decision)
            except Exception:
                pass
            # P20: 同步暂存状态（打回 → returned + reason，同名重传时 round+1）
            try:
                from agent_base.storage.pg import staging_get, staging_upsert
                st = staging_get(doc_id)
                if st is not None:
                    staging_upsert(
                        doc_id=doc_id,
                        content=st["content"],
                        filename=st["filename"],
                        category=st["category"],
                        status="returned",
                        review_round=int(st.get("review_round") or 1),
                        first_review=st.get("first_review") or {},
                        reject_reason=body.get("reason", ""),
                    )
            except Exception:
                pass
            return tag.to_dict()
        except Exception as e:
            raise HTTPException(500, f"打回失败: {e}")

    # ── 待审队列批量操作（生产审核工作台标配：批量确认 / 批量打回 / 批量丢弃） ──

    def _approve_doc(
        doc_id: str,
        doc_type: str,
        strategy: list[str] | None,
        reviewer: str,
    ) -> dict[str, Any]:
        """单篇确认：暂存文档 → approved + 自动入库；非暂存 → 仅打标。"""
        from agent_base.knowledge_factory import STRATEGY_MAP, DocTag, apply_tag, persist_tag
        from agent_base.storage.pg import staging_get

        strategy = strategy or STRATEGY_MAP.get(doc_type, ["default_vector"])
        staging = staging_get(doc_id)
        if staging is not None:
            from agent_base.storage.staging import approve_and_ingest
            runtime = get_runtime()
            result = approve_and_ingest(
                doc_id=doc_id,
                doc_type=doc_type,
                strategy=strategy,
                reviewer=reviewer,
                vector_store=runtime["vector_store"],
                summary_store=runtime.get("summary_store"),
            )
            return {"doc_id": doc_id, "ok": True, "message": result.get("message", "已入库")}
        tag = DocTag(doc_id=doc_id, doc_type=doc_type, strategy=strategy)
        tag = apply_tag(doc_id, tag, reviewer=reviewer)
        persist_tag(tag)
        return {"doc_id": doc_id, "ok": True, "message": "已确认打标"}

    def _reject_doc(
        doc_id: str,
        reason: str,
        reviewer: str,
    ) -> dict[str, Any]:
        """单篇打回：status → returned + 原因；同步暂存状态。"""
        from agent_base.knowledge_factory import DocTag, persist_tag, reject_tag
        from agent_base.storage.pg import staging_get, staging_upsert

        tag = DocTag(doc_id=doc_id)
        tag = reject_tag(doc_id, tag, reviewer=reviewer, reason=reason)
        persist_tag(tag)
        st = staging_get(doc_id)
        if st is not None:
            staging_upsert(
                doc_id=doc_id,
                content=st["content"],
                filename=st.get("filename", ""),
                category=st.get("category", ""),
                status="returned",
                review_round=int(st.get("review_round") or 1),
                first_review=st.get("first_review") or {},
                reject_reason=reason,
            )
            # BUG-25: 批量打回也写短期记忆（tags/reject 单条路径已写），
            # 重提时读取注入 prev_reject_reason，精审抽屉可展示上轮原因
            try:
                from agent_base.storage.review_memory import save_memory

                save_memory(
                    doc_id,
                    int(st.get("review_round") or 1),
                    reason,
                    st.get("first_review") or {},
                )
            except Exception:
                pass
        return {"doc_id": doc_id, "ok": True, "message": "已打回"}

    @app.post("/api/documents/batch-approve", dependencies=[Depends(_verify_admin)])
    def batch_approve_documents(body: dict[str, Any]):
        """批量确认入库（≤100/次）。

        P27b：高置信才允许批量（AI 自动路由），缺类型或 confidence<0.75 跳过并提示逐条精审。
        """
        doc_ids = body.get("doc_ids", [])
        if not isinstance(doc_ids, list) or not doc_ids:
            raise HTTPException(400, "doc_ids 必填（非空数组）")
        if len(doc_ids) > 100:
            raise HTTPException(400, "单次最多处理 100 篇，请分批")
        reviewer = body.get("reviewer", "admin")
        from agent_base.config import deep_get, load_yaml
        try:
            _cfg = load_yaml("configs/app.yaml") or {}
            threshold = float(deep_get(_cfg, "review.confidence_batch_threshold", 0.75))
        except Exception:
            threshold = 0.75
        results: list[dict[str, Any]] = []
        for raw_id in doc_ids:
            doc_id = str(raw_id)
            try:
                from agent_base.storage.pg import staging_get, strategy_get
                st = staging_get(doc_id)
                first_review = (st or {}).get("first_review") or {}
                doc_type = first_review.get("type") or ""
                confidence = float(first_review.get("confidence", 0.0))
                if not doc_type:
                    tag = strategy_get(doc_id)
                    doc_type = (tag or {}).get("doc_type", "")
                if not doc_type:
                    results.append({"doc_id": doc_id, "ok": False, "message": "缺少建议类型，请逐条精审"})
                    continue
                if confidence < threshold:
                    results.append({
                        "doc_id": doc_id,
                        "ok": False,
                        "message": f"置信度 {confidence:.2f} 低于 {threshold:.2f}，需逐条精审",
                    })
                    continue
                strategy = first_review.get("strategy") or None
                results.append(_approve_doc(doc_id, doc_type, strategy, reviewer))
            except Exception as exc:
                results.append({"doc_id": doc_id, "ok": False, "message": f"失败: {exc}"})
        get_runtime.cache_clear()
        return {
            "status": "batch_approved",
            "approved": sum(1 for r in results if r["ok"]),
            "failed": [r for r in results if not r["ok"]],
            "results": results,
        }

    @app.post("/api/documents/batch-reject", dependencies=[Depends(_verify_admin)])
    def batch_reject_documents(body: dict[str, Any]):
        """批量打回（≤100/次）：结构化原因模板 + 自定义说明，必填。"""
        doc_ids = body.get("doc_ids", [])
        if not isinstance(doc_ids, list) or not doc_ids:
            raise HTTPException(400, "doc_ids 必填（非空数组）")
        if len(doc_ids) > 100:
            raise HTTPException(400, "单次最多处理 100 篇，请分批")
        reason_label = str(body.get("reason_code", "")).strip()
        reason_text = str(body.get("reason", "")).strip()
        if not reason_label and not reason_text:
            raise HTTPException(400, "打回原因必填（请选择原因并填写说明）")
        full_reason = (
            f"{reason_label}：{reason_text}" if reason_label and reason_text
            else reason_label or reason_text
        )
        reviewer = body.get("reviewer", "admin")
        results: list[dict[str, Any]] = []
        for raw_id in doc_ids:
            try:
                results.append(_reject_doc(str(raw_id), full_reason, reviewer))
            except Exception as exc:
                results.append({"doc_id": str(raw_id), "ok": False, "message": f"失败: {exc}"})
        return {
            "status": "batch_rejected",
            "rejected": sum(1 for r in results if r["ok"]),
            "failed": [r for r in results if not r["ok"]],
            "results": results,
        }

    @app.post("/api/documents/batch-discard", dependencies=[Depends(_verify_admin)])
    def batch_discard_documents(body: dict[str, Any]):
        """批量丢弃（从队列移除，不产生版本历史；慎用，建议优先打回）。"""
        doc_ids = body.get("doc_ids", [])
        if not isinstance(doc_ids, list) or not doc_ids:
            raise HTTPException(400, "doc_ids 必填（非空数组）")
        if len(doc_ids) > 500:
            raise HTTPException(400, "单次最多丢弃 500 篇，请分批")
        from agent_base.storage.pg import staging_delete, strategy_delete
        for raw_id in doc_ids:
            doc_id = str(raw_id)
            staging_delete(doc_id)
            strategy_delete(doc_id)
        return {"status": "batch_discarded", "discarded": len(doc_ids)}

    # ── 数据中台推回闭环：平台侧查询推送文档的审核状态与打回原因 ──

    @app.get("/api/platform/documents/{doc_id}", dependencies=[Depends(_verify_platform)])
    def platform_doc_status(doc_id: str):
        """数据中台查询推送文档的审核结果（X-Platform-Token 鉴权）。

        status 语义：pending=待审核，returned=已打回（reject_reason 为原因），
        approved 且已入库后暂存记录会移除，可继续查询 document_strategy 状态。
        """
        from agent_base.storage.pg import staging_get, strategy_get

        st = staging_get(doc_id)
        if st is not None:
            first_review = st.get("first_review") or {}
            return {
                "doc_id": doc_id,
                "filename": st.get("filename", ""),
                "status": st.get("status", ""),
                "review_round": int(st.get("review_round") or 1),
                "doc_type": first_review.get("type", ""),
                "confidence": float(first_review.get("confidence", 0.0)),
                "reject_reason": st.get("reject_reason", ""),
                "updated_at": st.get("updated_at"),
            }
        tag = strategy_get(doc_id)
        if tag is not None:
            return {
                "doc_id": doc_id,
                "filename": doc_id,
                "status": tag.get("status", ""),
                "review_round": int(tag.get("review_round") or 1),
                "doc_type": tag.get("doc_type", ""),
                "confidence": 0.0,
                "reject_reason": tag.get("reject_reason", ""),
                "updated_at": tag.get("updated_at"),
            }
        raise HTTPException(404, "文档不存在")

    @app.get("/api/documents/review-queue", dependencies=[Depends(_verify_admin)])
    def review_queue(status: str = "pending_fine_review"):
        """P19/P20: 审核队列 = 暂存队列 + 打标队列（可按状态筛选）。

        status 映射：pending_fine_review（暂存 pending）| returned。
        """
        try:
            from agent_base.storage.pg import staging_list, strategy_list

            rows: list[dict[str, Any]] = []
            # 暂存队列（P20 上传=暂存）
            st_status = "pending" if status == "pending_fine_review" else status
            for r in staging_list(status=st_status):
                first_review = r.get("first_review") or {}
                rows.append({
                    "doc_id": r["doc_id"],
                    "filename": r.get("filename", ""),
                    "doc_type": first_review.get("type", ""),
                    "strategy": first_review.get("strategy") or [],
                    "reasoning": first_review.get("reasoning", ""),
                    "suggest_action": first_review.get("suggest_action", ""),
                    "reject_hint": first_review.get("reject_hint", ""),
                    "risk_flags": first_review.get("risk_flags") or [],
                    "prev_reject_reason": first_review.get("prev_reject_reason", ""),
                    "status": "pending_fine_review" if r["status"] == "pending" else r["status"],
                    "review_round": r.get("review_round", 1),
                    "confidence": float(first_review.get("confidence", 0.0)),
                    "review_source": first_review.get("source", ""),
                    "reject_reason": r.get("reject_reason", ""),
                    "updated_at": r.get("updated_at"),
                    "source": "staging",
                    "content_preview": (r.get("content") or "")[:2000],
                })
            # 打标队列（P19 历史/直接打标）
            for r in strategy_list():
                if r.get("status") == status:
                    rows.append({
                        "doc_id": r["doc_id"],
                        "filename": r["doc_id"],
                        "doc_type": r.get("doc_type", ""),
                        "strategy": r.get("strategy", []),
                        "reasoning": (r.get("first_review") or {}).get("reasoning", ""),
                        "suggest_action": (r.get("first_review") or {}).get("suggest_action", ""),
                        "reject_hint": (r.get("first_review") or {}).get("reject_hint", ""),
                        "risk_flags": (r.get("first_review") or {}).get("risk_flags") or [],
                        "prev_reject_reason": (r.get("first_review") or {}).get("prev_reject_reason", ""),
                        "status": r.get("status", ""),
                        "review_round": r.get("review_round", 1),
                        "confidence": 0.0,
                        "review_source": (r.get("first_review") or {}).get("source", ""),
                        "reject_reason": r.get("reject_reason", ""),
                        "updated_at": r.get("updated_at"),
                        "source": "strategy",
                    })
            # P27 v0.31.12：按 doc_id 去重——同一文档在 staging（上传暂存）
            # 和 strategy（打标）都可能存在，双来源合并会显示两行。
            # staging 优先（有 filename/内容预览），strategy 补缺字段。
            merged: dict[str, dict[str, Any]] = {}
            for row in rows:
                doc_id = row.get("doc_id", "")
                if not doc_id:
                    continue
                if doc_id in merged:
                    prev = merged[doc_id]
                    # 已有 staging（source=staging）则 strategy 只补缺失字段
                    for k, v in row.items():
                        if not prev.get(k) and v:
                            prev[k] = v
                    continue
                merged[doc_id] = row
            rows = list(merged.values())
            rows.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
            return {"queue": rows}
        except Exception as e:
            raise HTTPException(500, f"读取审核队列失败: {e}")

    @app.post("/api/documents/preview-chunks", dependencies=[Depends(_verify_admin)])
    def preview_chunks(body: dict[str, Any]):
        """切分效果预览：与入库完全同一切分器/参数，返回块列表 + 分隔符配置（只读，无副作用）。"""
        doc_id = str(body.get("doc_id", "")).strip()
        text_override = str(body.get("text", "") or "")
        from agent_base.storage.pg import staging_get, strategy_get

        content = text_override
        doc_type = str(body.get("doc_type", "")).strip()
        if doc_id and not content:
            st = staging_get(doc_id)
            if st is not None:
                content = str(st.get("content") or "")
                doc_type = str(((st.get("first_review") or {}).get("type")) or "")
        if not content:
            raise HTTPException(400, "文档内容不存在（可能尚未解析或已入库）")
        if not doc_type:
            tag = strategy_get(doc_id) if doc_id else None
            doc_type = str((tag or {}).get("doc_type") or "")
        if not doc_type:
            raise HTTPException(400, "缺少文档类型，无法确定切分档位（请先完成 AI 预审）")

        from agent_base.ingest.splitter import get_profile, split_markdown_by_type

        profile = dict(get_profile(doc_type))
        # 未保存的试验参数（预览用）：前端传了就按试验值切，不落库
        _trial_seps = body.get("separators")
        if isinstance(_trial_seps, list):
            _trial = [str(s) for s in _trial_seps if s is not None]
            if _trial:
                profile["separators"] = _trial
        if body.get("chunk_size"):
            profile["chunk_size"] = max(64, min(int(body["chunk_size"]), 4000))
        if body.get("chunk_overlap") is not None:
            profile["chunk_overlap"] = max(0, min(int(body["chunk_overlap"]), int(profile["chunk_size"]) - 1))
        preview_text = content[:80000]
        truncated = len(content) > 80000
        docs = split_markdown_by_type(doc_type, preview_text, profile=profile)

        _sep_labels = {
            "\n\n": "空行 (⏎⏎)",
            "\n": "换行 (⏎)",
            "Q:": "Q:",
            "A:": "A:",
            "|": "竖线 (|)",
            "。": "句号 (。)",
            "；": "分号 (；)",
            "，": "逗号 (，)",
            "": "逐字（兜底）",
        }
        chunks = [
            {
                "index": i + 1,
                "section": str(d.metadata.get("section", "")),
                "text": d.page_content,
                "chars": len(d.page_content),
            }
            for i, d in enumerate(docs)
        ]
        _customized = False
        try:
            from agent_base.storage.pg import chunk_override_get
            _customized = chunk_override_get(doc_type) is not None
        except Exception:
            pass
        return {
            "doc_type": doc_type,
            "customized": _customized,
            "profile": {
                "chunk_size": int(profile["chunk_size"]),
                "chunk_overlap": int(profile["chunk_overlap"]),
                "mode": str(profile.get("mode", "section")),
                "separators": [
                    {"raw": str(s), "label": _sep_labels.get(str(s), str(s) or "空")}
                    for s in profile["separators"]
                ],
            },
            "total_chunks": len(chunks),
            "total_chars": len(content),
            "truncated": truncated,
            "shown_chunks": min(len(chunks), 40),
            "chunks": chunks[:40],
        }

    @app.get("/api/chunk-profiles", dependencies=[Depends(_verify_admin)])
    def chunk_profiles():
        """全部 doc_type 的切分档位（含自定义覆盖标记与代码默认值）。"""
        from agent_base.ingest.splitter import CHUNK_PROFILES, DEFAULT_PROFILE, get_profile
        from agent_base.storage.pg import chunk_override_list

        overs = {r["doc_type"]: r for r in chunk_override_list()}
        doc_types: list[str] = list(CHUNK_PROFILES.keys())
        for dt in overs:
            if dt not in doc_types:
                doc_types.append(dt)
        items = []
        for dt in doc_types:
            eff = get_profile(dt)
            ov = overs.get(dt)
            items.append({
                "doc_type": dt,
                "chunk_size": eff.get("chunk_size"),
                "chunk_overlap": eff.get("chunk_overlap"),
                "separators": eff.get("separators"),
                "mode": eff.get("mode", "section"),
                "customized": bool(ov),
                "updated_by": (ov or {}).get("updated_by", ""),
                "updated_at": (ov or {}).get("updated_at", ""),
                "default": {
                    "chunk_size": CHUNK_PROFILES.get(dt, DEFAULT_PROFILE).get("chunk_size"),
                    "chunk_overlap": CHUNK_PROFILES.get(dt, DEFAULT_PROFILE).get("chunk_overlap"),
                    "separators": CHUNK_PROFILES.get(dt, DEFAULT_PROFILE).get("separators"),
                },
            })
        return {"profiles": items}

    @app.put("/api/chunk-profiles/{doc_type}", dependencies=[Depends(_verify_admin)])
    def upsert_chunk_profile(doc_type: str, body: dict[str, Any]):
        """保存某 doc_type 的切分参数覆盖（分隔符/块大小/重叠）。"""
        from agent_base.ingest.splitter import CHUNK_PROFILES, DEFAULT_PROFILE
        from agent_base.storage.pg import chunk_override_upsert

        base = CHUNK_PROFILES.get(doc_type, DEFAULT_PROFILE)
        chunk_size = int(body.get("chunk_size", base.get("chunk_size") or 900))
        chunk_overlap = int(body.get("chunk_overlap", base.get("chunk_overlap") or 120))
        separators = body.get("separators")
        if not isinstance(separators, list) or not separators:
            raise HTTPException(400, "separators 必填（非空数组）")
        # 空字符串是合法的"逐字兜底"分隔符，允许保留
        separators = [str(s) for s in separators if s is not None]
        if not separators:
            raise HTTPException(400, "separators 至少包含一个分隔符")
        if len(separators) > 12:
            raise HTTPException(400, "分隔符最多 12 个")
        for s in separators:
            if len(s) > 32:
                raise HTTPException(400, f"分隔符过长: {s[:16]}...")
        if not (64 <= chunk_size <= 4000):
            raise HTTPException(400, "chunk_size 需在 64-4000 之间")
        if not (0 <= chunk_overlap < chunk_size):
            raise HTTPException(400, "chunk_overlap 需 ≥0 且小于 chunk_size")
        reviewer = str(body.get("reviewer", "admin"))
        ok = chunk_override_upsert(doc_type, chunk_size, chunk_overlap, separators, reviewer)
        if not ok:
            raise HTTPException(500, "保存切分参数失败")
        return {"status": "saved", "doc_type": doc_type}

    @app.delete("/api/chunk-profiles/{doc_type}", dependencies=[Depends(_verify_admin)])
    def delete_chunk_profile(doc_type: str):
        """删除覆盖，回到代码默认档位。"""
        from agent_base.storage.pg import chunk_override_delete

        chunk_override_delete(doc_type)
        return {"status": "reset", "doc_type": doc_type}

    @app.post("/api/documents/submit", dependencies=[Depends(_verify_admin)])
    def submit_doc(body: dict[str, Any]):
        """P19: 提交/重提文档进入待精审（数据中台 re-submit 契约入口）。

        returned 文档重提 → review_round + 1 → 重新预审。
        注意：主项目工作台不再展示"重新提交"按钮——重提是数据中台职责，
        数据中台通过本接口（或 platform 推送更新）重新送审，主项目只审核。
        """
        doc_id = body.get("doc_id", "")
        if not doc_id:
            raise HTTPException(400, "doc_id 必填")
        try:
            from agent_base.knowledge_factory import submit_document_for_review
            tag = submit_document_for_review(
                doc_id=doc_id,
                doc_type=body.get("doc_type", ""),
                strategy=body.get("strategy"),
                reviewer=body.get("reviewer", "data-platform"),
                content=body.get("content", ""),
            )
            return tag.to_dict()
        except Exception as e:
            raise HTTPException(500, f"提交失败: {e}")

    @app.post("/api/documents/returned/clear", dependencies=[Depends(_verify_admin)])
    def clear_returned_documents(body: dict[str, Any]):
        """清空/删除已打回记录（删除 returned 暂存 + 打标，不可恢复）。

        主项目只审核：打回记录在数据中台拿到回执后即完成闭环，
        历史打回不再长期堆积；数据中台如需重新送审走 submit / platform 推送。
        body.doc_ids 可选：传了只删指定记录，缺省删全部 returned。
        """
        try:
            from agent_base.storage.pg import (
                staging_delete,
                staging_list,
                strategy_delete,
                strategy_list,
            )
            requested = body.get("doc_ids") or []
            if not isinstance(requested, list):
                raise HTTPException(400, "doc_ids 必须为数组")
            if len(requested) > 500:
                raise HTTPException(400, "单次最多删除 500 篇，请分批")

            cleared = 0
            if requested:
                doc_ids = {str(x) for x in requested}
            else:
                # 已打回列表 = 暂存队列（document_staging）+ 打标队列（document_strategy）
                # 两个来源都要清，否则列表仍会展示残留（v0.31.3 修复）
                doc_ids = {r["doc_id"] for r in staging_list(status="returned")}
                doc_ids.update(
                    r["doc_id"] for r in strategy_list() if r.get("status") == "returned"
                )
            # P31 BUG-1 修复：删除前重新确认——staging 与 strategy 都只删
            # 当前仍是 returned 的记录。打回后重提的文档：staging 可能仍
            # returned（旧暂存），而 strategy 已变 pending_fine_review（round+1，
            # 待审记录所在）；无差别删除会误删待审核记录。
            staging_returned_ids = {r["doc_id"] for r in staging_list(status="returned")}
            strategy_returned_ids = {
                r["doc_id"] for r in strategy_list() if r.get("status") == "returned"
            }
            for doc_id in sorted(doc_ids):
                if doc_id in staging_returned_ids:
                    staging_delete(doc_id)
                if doc_id in strategy_returned_ids:
                    strategy_delete(doc_id)
                cleared += 1
            return {"status": "cleared", "cleared": cleared}
        except Exception as e:
            raise HTTPException(500, f"清空已打回失败: {e}")

    @app.post("/api/platform/documents", dependencies=[Depends(_verify_platform)])
    def platform_push_document(body: dict[str, Any]):
        """数据中台推送/更新文档（预留对接入口，X-Platform-Token 鉴权）。

        数据中台在上游完成清洗、评估与内容修改后，通过此接口送审：
          - 内容进暂存队列（pending_fine_review），主项目精审一次后入库；
          - 可携带 doc_type/strategy/confidence/reasoning 评估建议，
            精审抽屉直接预填（source=data_platform）；
          - 幂等：内容 sha256 去重（同内容 → skipped）。

        请求体：
        - filename: str（必填）
        - content: str（必填，清洗后全文）
        - category: str（可选）
        - doc_type: str（可选，评估建议）
        - strategy: list[str]（可选，评估建议）
        - confidence: float（可选，默认 0.8）
        - reasoning: str（可选，评估说明）
        """
        filename = str(body.get("filename", "")).strip()
        content = body.get("content", "")
        if not filename or not content:
            raise HTTPException(400, "filename 和 content 必填")
        if len(content.encode("utf-8")) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "文档超过 10MB 限制")

        suggestion = None
        doc_type = str(body.get("doc_type") or "").strip()
        if doc_type:
            suggestion = {
                "type": doc_type,
                "confidence": body.get("confidence"),
                "reasoning": body.get("reasoning"),
                "strategy": body.get("strategy") or [],
            }

        from agent_base.storage.staging import stage_uploaded_document
        payload = stage_uploaded_document(
            filename=filename,
            content=content,
            category=str(body.get("category") or "").strip() or "数据中台",
            suggestion=suggestion,
        )
        get_runtime.cache_clear()
        return payload

    @app.get("/api/sessions", dependencies=[Depends(_verify_admin)])
    def list_sessions():
        """P20: 会话列表（查 chat_messages 表，标题取首条用户消息）。"""
        try:
            from agent_base.storage.pg import chat_sessions
            sessions = chat_sessions()
        except Exception:
            sessions = []
        return {"sessions": sessions}

    @app.get("/api/user/sessions", dependencies=[Depends(_verify_user)])
    def list_user_sessions(request: Request):
        """用户端侧边栏：读取当前登录用户自己的历史会话。"""
        from agent_base.auth import verify_token

        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else request.headers.get("X-Admin-Token", "")
        username = verify_token(token) if token else ""
        if not username:
            return {"sessions": []}
        from agent_base.storage.pg import chat_sessions_for_owner

        return {"sessions": chat_sessions_for_owner(username)}

    @app.get("/api/sessions/{session_id}/messages", dependencies=[Depends(_verify_user)])
    def get_session_messages(session_id: str, request: Request):
        """v0.48: 获取指定会话的完整消息历史（人工客服台用）。

        BUG-8 修复：管理员/客服可读任意会话；普通买家仅可读自己（owner）的会话。
        """
        from agent_base.auth import get_user_role, verify_token

        _auth = request.headers.get("Authorization", "")
        _token = _auth[7:].strip() if _auth.startswith("Bearer ") else request.headers.get("X-Admin-Token", "")
        _username = ""
        if _token:
            _username = verify_token(_token) or ""
        _role = get_user_role(_username) if _username else ""
        if _role not in ("admin", "agent"):
            from agent_base.storage.pg import session_owner

            owner = session_owner(session_id)
            if owner and owner != _username:
                raise HTTPException(403, "Forbidden: session not owned by current user")
            if not owner:
                # 会话尚未在后端落库（买家新建的本地会话）：返回空历史，避免控制台 403 噪音
                return {"messages": []}
        try:
            from agent_base.storage.pg import chat_history
            messages = chat_history(session_id, limit=200)
            return {"messages": messages}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"获取会话消息失败: {e}")

    @app.delete("/api/sessions/{session_id}", dependencies=[Depends(_verify_user)])
    def delete_session(session_id: str, request: Request):
        """v0.52: 删除会话（P1.5-7a 删会话清短期记忆）。

        清理范围：chat_messages 消息原文、chat_sessions 归属、chat_handoffs 转人工记录，
        以及 Redis 短期记忆（chat:memory / chat:meta）；长期记忆 user_memories 保留。
        权限：管理员/客服可删任意会话；普通买家仅可删自己（owner）的会话。
        """
        from agent_base.auth import get_user_role, verify_token

        _auth = request.headers.get("Authorization", "")
        _token = _auth[7:].strip() if _auth.startswith("Bearer ") else request.headers.get("X-Admin-Token", "")
        _username = ""
        if _token:
            _username = verify_token(_token) or ""
        _role = get_user_role(_username) if _username else ""
        if _role not in ("admin", "agent"):
            from agent_base.storage.pg import session_owner

            owner = session_owner(session_id)
            if owner and owner != _username:
                raise HTTPException(403, "Forbidden: session not owned by current user")
            if not owner:
                # 幂等删除：会话不存在视为已删除（买家删除未落库的本地新会话）
                return {"status": "deleted", "session_id": session_id, "cleared": False}
        try:
            from agent_base.storage.chat_memory import clear_chat_memory
            from agent_base.storage.pg import delete_chat_session

            cleared = delete_chat_session(session_id)
            clear_chat_memory(session_id)
            return {"status": "deleted", "session_id": session_id, "cleared": cleared}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"删除会话失败: {e}")

    # ── 提示词库：只读展示内置 Agent 提示词（内置不可修改） ──

    @app.get("/api/prompts", dependencies=[Depends(_verify_admin)])
    def list_prompts():
        """返回全部内置提示词目录（只读，YAML 为唯一真相源）。"""
        try:
            from agent_base.prompts import prompt_catalog
            items = prompt_catalog()
            return {"items": items, "total": len(items), "editable": False}
        except Exception as e:
            raise HTTPException(500, f"读取提示词失败: {e}")

    # ── P12-02: 意图管理 API（全部 X-Admin-Token 鉴权） ──

    @app.get("/api/intents", dependencies=[Depends(_verify_admin)])
    def list_intents():
        """列出所有意图（keywords/sections/examples/版本/状态）。"""
        try:
            from agent_base.storage.pg import intent_list
            return {"intents": intent_list(include_archived=False)}
        except Exception as e:
            raise HTTPException(500, f"读取意图列表失败: {e}")

    @app.get("/api/intents/{intent_name}", dependencies=[Depends(_verify_admin)])
    def get_intent(intent_name: str):
        """获取单个意图详情。"""
        try:
            from agent_base.storage.pg import intent_get
            result = intent_get(intent_name)
            if result is None:
                raise HTTPException(404, f"意图 '{intent_name}' 不存在")
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"读取意图失败: {e}")

    @app.put("/api/intents/{intent_name}", dependencies=[Depends(_verify_admin)])
    def update_intent(intent_name: str, body: dict[str, Any]):
        """更新意图（version+1，保留历史）。

        请求体字段（全部可选）：
        - keywords: list[str]
        - sections: list[str]
        - examples: list[str]
        - priority: float
        """
        try:
            # P21: 格式校验（管理端可编辑，后端兜底校验）
            keywords = body.get("keywords")
            # 章节锁定：关系检索匹配，用户不可修改（保留现值，忽略请求值）
            from agent_base.storage.pg import intent_get
            _current = intent_get(intent_name)
            if _current is None:
                raise HTTPException(404, f"意图 '{intent_name}' 不存在")
            sections = _current.get("sections") or []
            # 示例：界面隐藏，AI 优化可更新；未传时保留现值
            examples = body.get("examples")
            if examples is None:
                examples = _current.get("examples") or []
            errors: list[str] = []
            if not isinstance(keywords, list) or not keywords:
                errors.append("关键词不能为空")
            else:
                clean_kw = [str(k).strip() for k in keywords if str(k).strip()]
                if not clean_kw:
                    errors.append("关键词不能为空")
                if len(set(clean_kw)) != len(clean_kw):
                    errors.append("关键词存在重复项")
                if any(len(k) > 20 for k in clean_kw):
                    errors.append("关键词单项不能超过 20 字")
                keywords = clean_kw
            for field, val in (("示例", examples),):
                if val is not None:
                    if not isinstance(val, list):
                        errors.append(f"{field}必须是数组")
                    else:
                        clean = [str(x).strip() for x in val if str(x).strip()]
                        if len(set(clean)) != len(clean):
                            errors.append(f"{field}存在重复项")
            try:
                priority = float(body.get("priority", 1.0))
                if not (0.1 <= priority <= 10):
                    errors.append("优先级必须在 0.1-10 之间")
            except (TypeError, ValueError):
                errors.append("优先级必须是数字")
            if errors:
                raise HTTPException(400, "格式校验失败：" + "；".join(errors))

            from agent_base.storage.pg import intent_upsert
            new_version = intent_upsert(
                intent=intent_name,
                keywords=keywords,
                sections=sections,
                examples=examples,
                priority=priority,
            )
            return {"status": "updated", "intent": intent_name, "version": new_version}
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(500, f"更新意图失败: {e}")

    @app.post("/api/intents/{intent_name}/ai-improve", dependencies=[Depends(_verify_admin)])
    def ai_improve_intent(intent_name: str):
        """P21: AI 优化意图配置（关键词/章节/示例），返回建议由前端确认后保存。"""
        from agent_base.storage.pg import intent_get
        intent = intent_get(intent_name)
        if intent is None:
            raise HTTPException(404, f"意图 '{intent_name}' 不存在")
        try:
            from agent_base.retrieval.intent_improver import improve_intent_config
            return improve_intent_config(intent)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"AI 优化失败: {e}")

    @app.post("/api/intents/{intent_name}/examples", dependencies=[Depends(_verify_admin)])
    def add_intent_examples(intent_name: str, body: dict[str, Any]):
        """追加 few-shot 示例（version+1）。

        请求体：
        - examples: list[str]（新增示例列表）
        """
        try:
            from agent_base.storage.pg import intent_add_examples
            new_version = intent_add_examples(intent_name, body.get("examples", []))
            if new_version == 0:
                raise HTTPException(404, f"意图 '{intent_name}' 不存在")
            return {"status": "updated", "intent": intent_name, "version": new_version}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"追加示例失败: {e}")

    @app.get("/api/intents/{intent_name}/versions", dependencies=[Depends(_verify_admin)])
    def list_intent_versions(intent_name: str):
        """列出意图的版本历史。"""
        try:
            from agent_base.storage.pg import intent_versions
            return {"intent": intent_name, "versions": intent_versions(intent_name)}
        except Exception as e:
            raise HTTPException(500, f"读取版本历史失败: {e}")

    @app.post("/api/intents/{intent_name}/restore/{version}", dependencies=[Depends(_verify_admin)])
    def restore_intent(intent_name: str, version: int):
        """回滚意图到指定版本。"""
        try:
            from agent_base.storage.pg import intent_restore
            result = intent_restore(intent_name, version)
            if result is None:
                raise HTTPException(404, f"意图 '{intent_name}' 版本 {version} 不存在")
            return {"status": "restored", "intent": intent_name, "version": version, "data": result}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"回滚意图失败: {e}")

    @app.post("/api/intents/test", dependencies=[Depends(_verify_admin)])
    def test_intent(body: dict[str, Any]):
        """实时意图测试：输入问题 → 显示路由意图、来源、置信度、命中关键词。

        请求体：
        - question: str
        """
        question = body.get("question", "")
        if not question:
            raise HTTPException(400, "question 不能为空")
        from agent_base.retrieval.intent_router import route_question
        route = route_question(question)
        return {
            "question": question,
            "intent": route.intent,
            "source": route.source,
            "confidence": route.confidence,
            "matched_keywords": route.matched_keywords,
            "sections": route.sections,
            "fallback_reason": route.fallback_reason,
            "scores": route.scores,
        }

    @app.post("/api/intents/eval-chain", dependencies=[Depends(_verify_admin)])
    async def eval_chain_stream(request: Request, body: dict[str, Any]):
        """全链路评测（SSE）：意图→检索→生成→打分→落库。

        请求体（可选）：
        - cases: [{question, expected_intent, expected_source?, expected_facts?}] 自定义用例
        - name: 批次名称
        - ragas: 是否附加 RAGAS 四指标（默认读 configs/app.yaml eval.ragas_enabled）
        事件：progress(done,total) / done(result) / error。
        """
        import asyncio

        from agent_base.retrieval.eval_chain import run_chain_eval

        queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        cancelled = {"flag": False}
        ragas_flag = body.get("ragas")
        use_ragas = bool(ragas_flag) if isinstance(ragas_flag, bool) else None

        async def _run() -> None:
            try:
                result = await asyncio.to_thread(
                    run_chain_eval,
                    body.get("cases"),
                    name=str(body.get("name", "") or ""),
                    progress_cb=lambda done, total, phase: queue.put_nowait(
                        {"type": "progress", "done": done, "total": total, "phase": phase}
                    ),
                    is_cancelled=lambda: cancelled["flag"],
                    use_ragas=use_ragas,
                )
                await queue.put({"type": "done", "result": result})
            except Exception as exc:  # noqa: BLE001
                await queue.put({"type": "error", "message": str(exc)})

        task = asyncio.create_task(_run())

        async def _event_gen():
            try:
                while True:
                    if await request.is_disconnected():
                        cancelled["flag"] = True
                        break
                    try:
                        evt = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
                    if evt["type"] in ("done", "error"):
                        break
            finally:
                cancelled["flag"] = True
                task.cancel()

        return StreamingResponse(
            _event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/intents/eval/runs", dependencies=[Depends(_verify_admin)])
    def eval_runs_list():
        """全链路评测历史批次列表（最新在前）。"""
        from agent_base.storage.pg import eval_run_list

        return {"runs": eval_run_list(limit=50)}

    @app.get("/api/intents/eval/runs/{run_id}", dependencies=[Depends(_verify_admin)])
    def eval_runs_detail(run_id: int):
        """全链路评测批次明细（含每用例打分）。"""
        from agent_base.storage.pg import eval_run_get

        run = eval_run_get(run_id)
        if run is None:
            raise HTTPException(404, f"评测批次 {run_id} 不存在")
        return run

    # P20: 新前端 SPA fallback（history 路由；非 /api 路径回 index.html）
    if (_FRONTEND_DIST / "index.html").exists():
        @app.get("/{full_path:path}", include_in_schema=False)
        def _frontend_spa(full_path: str):
            """SPA fallback：前端 history 路由（/admin/...）返回 index.html。"""
            if full_path.startswith("api/"):
                raise HTTPException(404, "Not Found")
            from fastapi.responses import FileResponse
            return FileResponse(str(_FRONTEND_DIST / "index.html"))

    return app


def _seed_legacy_tags_once() -> None:
    """P19 D2: 启动时对存量文档批量 approved 迁移（幂等，失败静默）。"""
    try:
        from agent_base.knowledge_factory import seed_legacy_tags
        seeded = seed_legacy_tags()
        if seeded:
            print(f"P19: legacy tags seeded: {seeded}")
    except Exception:
        pass

def _run_graph_ask(request: RagRequest, constraints: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """graph/agent 模式非流式回答。"""
    return _run_graph_ask_internal(request, constraints, runtime)


def _run_graph_ask_internal(request: RagRequest, constraints: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """graph/agent 模式执行体（流式和非流式共用）。"""
    import uuid as _uuid
    from agent_base.graphs import build_rag_graph

    agent_cfg = runtime.get("agent_config") or {}
    if request.framework == "agent" and agent_cfg:
        g_llm = {
            "provider": "langchain",
            "model": agent_cfg.get("model", "deepseek-v4-pro"),
            "base_url": agent_cfg.get("base_url"),
            "api_key_env": agent_cfg.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"),
            "temperature": float(agent_cfg.get("temperature", 0.1)),
        }
    else:
        g_llm = runtime["llm_config"]
    graph = build_rag_graph(
        vector_store=runtime["vector_store"],
        summary_store=runtime["summary_store"],
        sparse_store=runtime["sparse_store"],
        rerank_cfg=runtime["rerank_config"],
        llm_cfg=g_llm,
        prompts_path=runtime["prompts_path"],
    )
    state = {
        "question": request.question,
        "product_name": constraints["product_name"],
        "product_spec": constraints["product_spec"],
        "category": constraints["category"],
        "errors": [],
    }
    # P11-03: 缓存 graph/agent 路径
    from agent_base.storage.cache import cache_key, get_cached, set_cache, DATA_VERSION
    gk = cache_key(request.question, {"v": DATA_VERSION, "session": request.session_id or "", "fw": request.framework})
    gc = get_cached(gk)
    if gc:
        return gc

    result = graph.invoke(state, {"configurable": {"thread_id": str(_uuid.uuid4())}})
    payload = {
        "answer": result.get("answer", ""),
        "trace": result.get("trace") or {"question": request.question, "mode": request.framework},
        "safety": result.get("safety") or {"risk_level": "low", "findings": [], "warnings": [], "must_consult": False, "emergency": False},
        "catalog_resolution": constraints["catalog_resolution"],
    }
    if payload.get("answer"):
        light = {"answer": payload["answer"], "safety": payload.get("safety", {}),
                 "catalog_resolution": payload.get("catalog_resolution"),
                 "trace": {"results": light_results(payload.get("trace", {}).get("results", []))}}
        set_cache(gk, light)
    return payload



@lru_cache(maxsize=1)
def _faq_title_map() -> dict[str, str]:
    """FAQ id → 问题标题映射（PG 运行时数据源，内置种子，json 文件已淘汰）。"""
    try:
        from agent_base.storage.pg import faq_seed, faq_title_map

        m = faq_title_map()
        if not m:
            # 表空（首次/迁移前）→ 内置种子导入一次
            try:
                faq_seed()
                m = faq_title_map()
            except Exception:
                pass
        return m
    except Exception:
        return {}


def _doc_display_name(doc_id: str) -> str:
    """P21: 文档 ID → 友好名称（商品名 / FAQ 问题 / 文件名去扩展名）。

    Args:
        doc_id: 文档标识（P001 / F001 / 尺码指南.md / 内容哈希）。

    Returns:
        界面展示用中文名称。
    """
    if not doc_id:
        return doc_id
    # 1. 商品：catalog name
    try:
        products = get_catalog().get("products", {})
        item = products.get(doc_id)
        if item and item.get("name"):
            return item["name"]
    except Exception:
        pass
    # 2. FAQ：PG faq 表的 question
    faq_title = _faq_title_map().get(doc_id)
    if faq_title:
        return faq_title
    # 2.5 标准命名：把知识库内部文件名转成更适合文档管理页展示的名称
    stripped = doc_id
    for ext in (".md", ".txt", ".pdf", ".docx"):
        if stripped.lower().endswith(ext):
            stripped = stripped[: -len(ext)]
            break
    sales_match = re.match(r"sales_p(\d{3})", stripped, re.IGNORECASE)
    if sales_match:
        pid = "P" + sales_match.group(1)
        item = products.get(pid) if products else {}
        name = str(item.get("name") or "").strip() if isinstance(item, dict) else ""
        return f"{name} · 销售话术" if name else stripped
    video_match = re.match(r"video_p(\d{3})", stripped, re.IGNORECASE)
    if video_match:
        pid = "P" + video_match.group(1)
        item = products.get(pid) if products else {}
        name = str(item.get("name") or "").strip() if isinstance(item, dict) else ""
        return f"{name} · 视频知识" if name else stripped
    standard_prefixes = [
        ("商品详情_", "商品详情"),
        ("商品长文_", "商品长文"),
        ("FAQ_", "FAQ"),
        ("参数_", "参数"),
        ("品牌_", "品牌"),
        ("搭配指南_", "搭配指南"),
        ("对比_", "对比"),
        ("成分_", "成分"),
        ("政策_", "政策"),
        ("材质_", "材质"),
        ("案例_", "案例"),
    ]
    for prefix, label in standard_prefixes:
        if stripped.startswith(prefix):
            title = stripped[len(prefix):]
            return f"{title} · {label}" if title else stripped
    # 3. 文件类：去扩展名
    low = doc_id.lower()
    for ext in (".md", ".txt", ".pdf", ".docx"):
        if low.endswith(ext):
            return doc_id[: -len(ext)]
    # 4. 长哈希：上传文档
    if len(doc_id) > 24:
        return f"上传文档（{doc_id[:8]}…）"
    return doc_id


def _delete_document_core(doc_id: str) -> int:
    """删除文档核心（单删/批删复用）：删向量 + 软删版本 + 删标签 + 缓存失效。

    Args:
        doc_id: Document ID.

    Returns:
        删除的向量数。
    """
    from agent_base.storage.pg import doc_delete, doc_versions, staging_delete
    from agent_base.storage.cache import invalidate_pattern

    all_versions = doc_versions(doc_id)
    all_chunk_ids: list[str] = []
    for v in all_versions:
        all_chunk_ids.extend(v.get("chunk_ids", []))
    if all_chunk_ids:
        try:
            runtime = get_runtime()
            vs = runtime["vector_store"]
            ss = runtime["summary_store"]
            from agent_base.indexing.vector_index import _qdrant_point_id
            point_ids = [_qdrant_point_id(cid) for cid in all_chunk_ids]
            try:
                vs.delete(ids=point_ids)
            except Exception:
                pass
            try:
                ss.delete(ids=point_ids)
            except Exception:
                pass
        except Exception:
            pass
    doc_delete(doc_id)
    # v0.46: 删除保留 strategy 标签（回收站恢复时原样还原；物理清理在 purge/定时任务时执行）
    # P31: 删除时同步清理暂存残留（approved 记录会误导"内容已存在"去重，
    # 且待审队列不再展示已处理文档）
    try:
        staging_delete(doc_id)
    except Exception:
        pass
    invalidate_pattern("rag:cache:*")
    return len(all_chunk_ids)


def _validate_options(request: RagRequest) -> None:
    """校验前端传入的检索模式和 rerank 策略是否在白名单中。"""
    if request.rerank not in RERANK_STRATEGIES:
        raise HTTPException(status_code=422, detail=f"Unsupported rerank strategy: {request.rerank}")


def _resolve_constraints(request: RagRequest) -> dict[str, Any]:
    """用 catalog 从用户问题中识别商品名、规格和分类，作为检索 metadata 约束。"""
    product_name = request.product_name
    product_spec = request.product_spec
    category = request.category
    resolution_payload = None
    if request.use_catalog:
        # catalog 纯 PG 运行时数据源（json 文件已淘汰）
        resolution = resolve_query_constraints(get_catalog(), request.question)
        resolution_payload = resolution.to_dict()
        product_name = product_name or resolution.product_name
        product_spec = product_spec or resolution.product_spec
        category = category or resolution.category
    return {
        "product_name": product_name,
        "product_spec": product_spec,
        "category": category,
        "catalog_resolution": resolution_payload,
    }


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """读取应用配置；环境变量优先，YAML 配置作为默认值。"""
    config_path = _project_path(os.getenv("AGENT_BASE_CONFIG", "configs/app.yaml"))
    config = load_yaml(config_path) if config_path.exists() else {}
    return AppSettings(
        config_path=str(config_path),
        persist_dir=str(_project_path(os.getenv("AGENT_BASE_CHROMA_DIR", deep_get(config, "paths.chroma_dir", "data/chroma")))),
        collection=os.getenv("AGENT_BASE_COLLECTION", deep_get(config, "index.chunk_collection", "ecommerce_chunks")),
        summary_collection=os.getenv(
            "AGENT_BASE_SUMMARY_COLLECTION",
            deep_get(config, "index.summary_collection", "ecommerce_summaries"),
        ),
        retention_days=int(os.getenv("AGENT_BASE_RETENTION_DAYS", deep_get(config, "documents.retention_days", 30))),
        handoff_pending_timeout=int(deep_get(config, "handoff.pending_timeout_min", 15)) * 60,
        handoff_idle_timeout=int(deep_get(config, "handoff.idle_timeout_min", 20)) * 60,
    )


def _project_path(path: str | Path) -> Path:
    """把配置中的相对路径解析到项目根目录。

    FastAPI 在不同启动方式下工作目录可能不同。如果直接使用 data/chroma 这类相对路径，
    可能会落到 src/agent_base/api/data/chroma，导致前端仍读取旧 catalog/旧 Chroma。
    """
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_runtime() -> dict[str, Any]:
    """加载运行期对象，包括原文向量库、摘要向量库、LLM 和 prompt 配置。"""
    settings = get_settings()
    config = load_yaml(settings.config_path) if Path(settings.config_path).exists() else {}
    embedding_cfg = config.get("embedding", {})
    llm_cfg = config.get("llm", {})
    intent_classifier_cfg = config.get("intent_classifier", {})
    rerank_cfg = config.get("rerank", {})
    vectorstore_cfg = config.get("vectorstore", {})
    provider = (vectorstore_cfg.get("provider") or "chroma").lower()
    if provider == "qdrant":
        # 生产默认：Qdrant（configs/app.yaml vectorstore.provider=qdrant）。
        # 此前 runtime 一直走 Chroma（data/chroma 已归档），导致 /api/ask 查空库。
        from agent_base.embeddings import build_embeddings
        from agent_base.vectorstore import build_vector_store

        embeddings = build_embeddings(
            provider=embedding_cfg.get("provider", "hash"),
            model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            base_url=embedding_cfg.get("base_url"),
            api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
            keep_alive=int(embedding_cfg.get("keep_alive", 1800)),
        )
        vector_store = build_vector_store(
            provider="qdrant",
            url=vectorstore_cfg.get("url"),
            collection=settings.collection,
            embedding_function=embeddings,
        )
        summary_store = build_vector_store(
            provider="qdrant",
            url=vectorstore_cfg.get("url"),
            collection=settings.summary_collection,
            embedding_function=embeddings,
        )
    else:
        vector_store = load_vector_store(
            persist_dir=settings.persist_dir,
            collection=settings.collection,
            embedding_provider=embedding_cfg.get("provider", "hash"),
            embedding_model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            embedding_base_url=embedding_cfg.get("base_url"),
            embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
            embedding_keep_alive=int(embedding_cfg.get("keep_alive", 1800)),
        )
        summary_store = load_vector_store(
            persist_dir=settings.persist_dir,
            collection=settings.summary_collection,
            embedding_provider=embedding_cfg.get("provider", "hash"),
            embedding_model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            embedding_base_url=embedding_cfg.get("base_url"),
            embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
            embedding_keep_alive=int(embedding_cfg.get("keep_alive", 1800)),
        )
    prompts_path = deep_get(config, "prompts.path", "configs/prompts.yaml")
    # 稀疏检索通道（BM25 稀疏向量，jieba 分词）：与 _search_sparse_stage 的
    # client + collection_name 接口对齐，作为混合检索的补充召回通道。
    from types import SimpleNamespace
    from qdrant_client import QdrantClient

    sparse_store = SimpleNamespace(
        client=QdrantClient(url=vectorstore_cfg.get("url") or "http://localhost:6333"),
        collection_name=vectorstore_cfg.get("sparse_collection", "ecommerce_chunks_sparse"),
    )
    return {
        "vector_store": vector_store,
        "summary_store": summary_store,
        "sparse_store": sparse_store,
        "llm_config": llm_cfg,
        "intent_classifier_config": intent_classifier_cfg,
        "rerank_config": rerank_cfg,
        "prompts_path": prompts_path,
    }

# least recently used 最近最少使用 maxsize 缓存中的个数
def _parse_content_to_chunks(
    doc_id: str,
    content: str,
    category: str = "",
    doc_name: str = "",
) -> list[dict[str, Any]]:
    """P16: Parse raw document content into chunk dicts for vector indexing.

    Uses simple paragraph splitting (double newline or 500-char sliding window).
    Each chunk gets a deterministic UUID chunk_id via _qdrant_point_id.

    Args:
        doc_id: Document identifier.
        content: Full document text.
        category: Optional category tag.
        doc_name: 原始文件名（写入 chunk metadata.doc_name，来源卡/评测展示用）。

    Returns:
        List of chunk dicts with text/chunk_id/metadata.
    """
    import hashlib as _hashlib
    from agent_base.indexing.vector_index import _qdrant_point_id

# 按段落（双换行）或滑动窗口切分
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
# 滑动窗口：400 字符，200 重叠
        paragraphs = []
        ws, overlap = 400, 200
        for start in range(0, len(content), ws - overlap):
            chunk = content[start:start + ws].strip()
            if chunk:
                paragraphs.append(chunk)

    # T6b：段落切分时跟踪 Markdown 标题，恢复真实章节名到 metadata.section
    # （此前简单切分所有 chunk 都带"文档"章节，导致章节过滤失效、来源卡章节失真）
    import re as _re

    units: list[tuple[str, str]] = []
    current_section = category or "概述"
    for para in paragraphs:
        # MD 标题常与正文同段（无空行分隔），取段落首行判断标题
        first_line = para.split("\n", 1)[0].strip()
        heading = _re.match(r"^#{1,6}\s+(.+?)\s*$", first_line)
        if heading:
            current_section = heading.group(1).strip()
        units.append((para, current_section))
    if not units and paragraphs:
        units = [(p, current_section) for p in paragraphs]

    chunks: list[dict[str, Any]] = []
    for para, section in units:
        # P16-修复：chunk_id 内容化（段落 sha256 完整 64 hex），同内容幂等、
        # 内容变 ID 变，避免更新时复用旧序号 ID 导致"删旧向量误删新向量"。
        # v0.27.3：截断 [:16]（64bit）在千万级 chunk 下有生日碰撞风险，
        # 升级为完整 64 hex，杜绝不同段落共用 chunk_id 导致更新差集误判。
        digest = _hashlib.sha256(para.encode("utf-8")).hexdigest()
        chunk_id = _qdrant_point_id(f"{doc_id}:chunk:{digest}")
        chunks.append({
            "chunk_id": chunk_id,
            "text": para,
            "metadata": {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "section": section,
                "source_file": f"api://{doc_id}",
                "chunk_type": "ingested",
                **({"doc_name": doc_name} if doc_name else {}),
            },
        })
    return chunks


@lru_cache(maxsize=1)
def get_catalog() -> dict[str, Any]:
    """读取商品 catalog（纯 PG，JSON 文件已淘汰）。"""
    try:
        from agent_base.storage.pg import _conn

        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, brand, category, price_band, metadata FROM catalog")
            rows = cur.fetchall()
        if rows:
            products: dict[str, Any] = {}
            for rid, name, brand, category, price_band, md in rows:
                base: dict[str, Any] = {
                    "name": name,
                    "brand": brand or "",
                    "category": category or "",
                    "price_band": price_band or "",
                }
                if md and isinstance(md, dict):
                    for k, v in md.items():
                        if k not in base:
                            base[k] = v
                products[str(rid)] = base
            category_counts: dict[str, int] = {}
            for p in products.values():
                cat = p.get("category") or ""
                if cat:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            return {
                "product_count": len(products),
                "categories": sorted(category_counts.keys()),
                "category_counts": category_counts,
                "products": products,
            }
    except Exception:
        pass
    return {"product_count": 0, "categories": [], "category_counts": {}, "products": {}}


async def _process_upload(
    file: UploadFile,
    category: str,
    product_name: str | None = None,
    product_spec: str | None = None,
) -> UploadResponse:
    """执行上传入库主流程。

    流程：
    1. 校验文件类型和大小。
    2. 保存上传文件字节副本（data/uploads 留痕，真相源在 PG）。
    3. 暂存 + 自动预审（P20：精审 approved 后才真正入库到 PG + Qdrant）。
    """
    original_name = file.filename or "uploaded-document"
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"暂不支持该文件类型，请上传：{supported}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件超过 10MB 限制。")

    # 知识入库直接上传只收 Markdown：直接解码文本，进入分块→质检→入库流程。
    # 外部格式（PDF/Word/PPT/Excel/HTML/EPUB 等）请先到「文件清洗」面板解析清洗后推送。
    text = _decode_text(content)
    _parsed_engine = "direct"
    if not text.strip():
        raise HTTPException(status_code=422, detail="文件已上传，但没有解析出可入库文本。")

    _save_upload_bytes(original_name, content)
    from agent_base.storage.staging import stage_uploaded_document
    # 上传预审为启发式（llm_cfg={"provider":"none"}，不调 LLM）；
    # LLM 预审走「AI 审核」按钮的 batch-pre-review。
    # stage_uploaded_document 含 DB 写入，丢线程池避免阻塞事件循环。
    payload = await asyncio.to_thread(
        stage_uploaded_document,
        filename=original_name,
        content=text,
        category=category.strip() or "上传文档",
    )
    if isinstance(payload, dict):
        payload["parser"] = _parsed_engine
    get_runtime.cache_clear()
    return payload


def _save_upload_bytes(original_name: str, content: bytes) -> Path:
    path = _upload_target_path(original_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _upload_target_path(original_name: str) -> Path:
    return UPLOAD_DIR / _safe_filename(original_name)


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(filename).name).strip(" .")
    if not cleaned:
        cleaned = "uploaded-document"
    if len(cleaned) > 120:
        stem = Path(cleaned).stem[:80]
        suffix = Path(cleaned).suffix[:12]
        cleaned = f"{stem}{suffix}"
    return cleaned


def _optional_int(value: Any) -> int | None:
    if value in {None, "", "none", "null"}:
        return None
    return int(value)


async def _purge_loop() -> None:
    """v0.46/0.52: 后台清理循环——每小时检查一次。

    - 回收站：物理清除超过保留期的软删文档；
    - PG 短期记忆：清除超过保留期的会话消息（chat_messages）及归属孤儿。
    """
    from agent_base.storage.pg import purge_deleted_documents, purge_old_chat_messages

    while True:
        try:
            days = get_settings().retention_days
            result = await asyncio.to_thread(purge_deleted_documents, days)
            if result.get("documents") or result.get("strategy"):
                print(f"[purge] cleaned deleted docs: {result}")
        except Exception as exc:
            print(f"[purge] cleanup failed (retry next cycle): {exc}")
        try:
            from agent_base.config import deep_get as _dg
            from agent_base.config import load_yaml as _ly

            _app_cfg = _ly("configs/app.yaml") or {}
            chat_days = int(_dg(_app_cfg, "memory.pg_retention_days", 30))
            result2 = await asyncio.to_thread(purge_old_chat_messages, chat_days)
            if result2.get("messages"):
                print(f"[purge] cleaned old chat messages: {result2}")
        except Exception as exc:
            print(f"[purge] chat cleanup failed (retry next cycle): {exc}")
        # v0.53: 游客孤儿数据回收（保留期内游客不删）
        try:
            from agent_base.storage.pg import cleanup_guests

            _gdays = int(os.getenv("GUEST_RETENTION_DAYS", "30"))
            gres = await asyncio.to_thread(cleanup_guests, _gdays)
            if gres.get("users"):
                print(f"[purge] cleaned guests: {gres}")
        except Exception as exc:
            print(f"[purge] guest cleanup failed (retry next cycle): {exc}")
        await asyncio.sleep(3600)


def _warm_embedding() -> None:
    """预热 embedding 服务（Ollama bge-m3 首次加载约 10s+，预热后常驻）。"""
    try:
        store = get_runtime().get("vector_store")
        if store is None:
            return
        fn = getattr(store, "embedding_function", None) or getattr(store, "embeddings", None)
        if fn is not None and hasattr(fn, "embed_query"):
            fn.embed_query("预热")
    except Exception:
        pass


async def _embedding_keepalive_loop() -> None:
    """v0.48: OpenAIEmbeddings 无 keep_alive 参数——每 4 分钟预热一次保持 Ollama 模型加载。"""
    while True:
        await asyncio.sleep(240)
        try:
            await asyncio.to_thread(_warm_embedding)
        except Exception:
            pass


app = create_app()
# 功能类似
#   启动命令：python -m uvicorn agent_base.api.main:app --host 127.0.0.1 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_base.api.main:app", host="127.0.0.1", port=8000, reload=True)
