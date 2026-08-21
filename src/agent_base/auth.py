"""运营台账号认证（登录页 + Bearer token）。

设计（零新依赖）：
- 账号表 admin_users（PG，幂等建表 + 种子管理员）；
- 密码用 bcrypt 哈希（项目已装 bcrypt 5.0.0）；
- token 用 itsdangerous.URLSafeTimedSerializer（HMAC 签名 + 12h 过期），
  不引入 JWT 库；SECRET_KEY 默认取环境变量 ADMIN_SECRET / config，生产必须配置。
"""

from __future__ import annotations

import os
from typing import Any

import bcrypt
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from agent_base.config import load_yaml
from agent_base.storage.pg import _conn


TOKEN_MAX_AGE = 12 * 3600           # 真实账号 token 有效期（12 小时）
GUEST_TOKEN_MAX_AGE = 30 * 24 * 3600  # 游客 token 有效期（30 天，与游客清理周期一致）
TOKEN_SALT = "rag-admin-auth"


def _project_root() -> str:
    return str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _secret_key() -> str:
    """签名密钥：优先环境变量 ADMIN_SECRET，其次 configs/app.yaml security.secret_key。"""
    key = os.getenv("ADMIN_SECRET", "")
    if key:
        return key
    try:
        cfg = load_yaml(f"{_project_root()}/configs/app.yaml")
        key = (cfg.get("security", {}) or {}).get("secret_key", "")
    except Exception:
        key = ""
    if not key:
        # 开发默认（生产必须通过 ADMIN_SECRET 配置）
        return "rag-admin-dev-secret-change-me"
    return key


def init_admin_table() -> None:
    """幂等创建 admin_users 表。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT DEFAULT '',
                role TEXT DEFAULT 'admin',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )


def seed_admin() -> None:
    """幂等写入种子管理员（admin / ADMIN_INITIAL_PASSWORD，默认 admin123）。"""
    username = os.getenv("ADMIN_INITIAL_USERNAME", "admin")
    password = os.getenv("ADMIN_INITIAL_PASSWORD", "admin123")
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM admin_users WHERE username=%s", (username,))
        if cur.fetchone():
            return
        cur.execute(
            """INSERT INTO admin_users (username, password_hash, display_name, role)
               VALUES (%s, %s, %s, %s)""",
            (username, hash_password(password), "管理员", "admin"),
        )


def hash_password(password: str) -> str:
    """对密码做 bcrypt 哈希。"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def seed_agent() -> None:
    """幂等写入客服种子账号（agent / AGENT_INITIAL_PASSWORD，默认 agent123，role=agent）。"""
    username = os.getenv("AGENT_INITIAL_USERNAME", "agent")
    password = os.getenv("AGENT_INITIAL_PASSWORD", "agent123")
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM admin_users WHERE username=%s", (username,))
        if cur.fetchone():
            return
        cur.execute(
            """INSERT INTO admin_users (username, password_hash, display_name, role)
               VALUES (%s, %s, %s, %s)""",
            (username, hash_password(password), "人工客服", "agent"),
        )


def get_user_role(username: str) -> str:
    """查询账号角色（admin / agent / user）；不存在返回空串。

    运营账号查 admin_users（admin/agent），买家查 users（user）——分表设计。
    """
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT role FROM admin_users WHERE username=%s AND role IN ('admin','agent')",
                (username,),
            )
            row = cur.fetchone()
            if row:
                return str(row[0])
            cur.execute("SELECT 1 FROM users WHERE user_id=%s", (username,))
            return "user" if cur.fetchone() else ""
    except Exception:
        return ""


def list_accounts() -> list[dict[str, Any]]:
    """列出运营台账号（仅 admin/agent；买家用户不在运营账号体系内）。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT username, display_name, role, created_at, updated_at
               FROM admin_users WHERE role IN ('admin', 'agent') ORDER BY
               CASE role WHEN 'admin' THEN 0 WHEN 'agent' THEN 1 ELSE 2 END,
               created_at ASC"""
        )
        return [
            {
                "username": r[0],
                "display_name": r[1] or r[0],
                "role": r[2],
                "created_at": r[3].isoformat() if r[3] else None,
                "updated_at": r[4].isoformat() if r[4] else None,
            }
            for r in cur.fetchall()
        ]


def create_account(username: str, password: str, display_name: str = "", role: str = "agent") -> dict[str, Any]:
    """创建运营台账号（用户名唯一；重复返回 None）。"""
    if not username or not password:
        raise ValueError("用户名和密码不能为空")
    # 运营账号体系只允许 admin/agent；买家用户走 C 端用户体系，不在此创建
    if role not in ("admin", "agent"):
        raise ValueError("运营账号角色仅支持 admin / agent（买家用户请走 C 端注册）")
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM admin_users WHERE username=%s", (username,))
        if cur.fetchone():
            return None
        cur.execute(
            """INSERT INTO admin_users (username, password_hash, display_name, role)
               VALUES (%s, %s, %s, %s)""",
            (username, hash_password(password), display_name or username, role),
        )
        conn.commit()
    return {
        "username": username,
        "display_name": display_name or username,
        "role": role,
    }


def update_account(username: str, *, display_name: str | None = None, password: str | None = None) -> bool:
    """更新账号中文名 / 重置密码。返回是否找到该账号。"""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM admin_users WHERE username=%s AND role IN ('admin','agent')",
            (username,),
        )
        if not cur.fetchone():
            return False
        if password:
            cur.execute(
                "UPDATE admin_users SET password_hash=%s, updated_at=NOW() WHERE username=%s",
                (hash_password(password), username),
            )
        if display_name is not None:
            cur.execute(
                "UPDATE admin_users SET display_name=%s, updated_at=NOW() WHERE username=%s",
                (display_name, username),
            )
        conn.commit()
    return True


def delete_account(username: str) -> bool:
    """删除运营账号（保护内置种子账号 admin / agent；买家账号不在可删范围）。"""
    if username in ("admin", "agent", "user"):
        return False
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM admin_users WHERE username=%s AND role IN ('admin','agent')",
            (username,),
        )
        deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def seed_user() -> None:
    """幂等写入买家种子账号到 users 表（user / USER_INITIAL_PASSWORD，默认 user123）。

    v0.52: 买家账号与运营账号分表——买家写 users，admin_users 只保留 admin/agent。
    同时清理旧版误写入 admin_users 的 role=user 账号。
    """
    username = os.getenv("USER_INITIAL_USERNAME", "user")
    password = os.getenv("USER_INITIAL_PASSWORD", "user123")
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO users (user_id, nickname, level, password_hash)
               VALUES (%s, %s, 'normal', %s)
               ON CONFLICT (user_id) DO UPDATE SET password_hash=EXCLUDED.password_hash""",
            (username, "买家用户", hash_password(password)),
        )
        # 清理旧版误放 admin_users 的买家账号（role=user）
        cur.execute("DELETE FROM admin_users WHERE role='user'")
        conn.commit()


def verify_password(password: str, password_hash: str) -> bool:
    """校验明文密码与哈希是否匹配。"""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def create_token(username: str, max_age: int = TOKEN_MAX_AGE) -> str:
    """为用户生成签名令牌（payload 带过期时间，游客可用更长有效期）。

    Args:
        username: 用户名。
        max_age: 有效期秒数（默认 12h；游客 30d）。
    """
    import time as _time

    serializer = URLSafeTimedSerializer(_secret_key(), salt=TOKEN_SALT)
    return serializer.dumps({"u": username, "e": int(_time.time()) + int(max_age)})


def verify_token(token: str) -> str | None:
    """校验 token，返回 username；无效/过期返回 None。

    兼容两种格式：新版 payload 带显式过期时间 e；旧版（无 e）按 12h 判活。
    """
    import time as _time

    serializer = URLSafeTimedSerializer(_secret_key(), salt=TOKEN_SALT)
    try:
        # 外层用游客上限做防篡改检查（itsdangerous 的 iat 时效），精确过期看 e 字段
        data = serializer.loads(token, max_age=GUEST_TOKEN_MAX_AGE)
        if not isinstance(data, dict) or not data.get("u"):
            return None
        exp = data.get("e")
        if exp is None:
            # 旧版 token：按默认 12h 判活
            iat = data.get("iat") or 0
            if _time.time() - float(iat) > TOKEN_MAX_AGE:
                return None
        elif _time.time() > int(exp):
            return None
        return str(data["u"])
    except (BadSignature, SignatureExpired):
        return None
    return None


def find_or_create_user_by_phone(phone: str) -> dict[str, Any] | None:
    """手机号注册/登录：已注册则取回，未注册自动建档（user_id = phone_<手机号>）。

    Args:
        phone: 11 位手机号（调用方已校验格式）。

    Returns:
        {username, display_name, role: "user", is_new}；is_new=True 表示本次自动注册；
        失败返回 None。
    """
    import re

    if not re.fullmatch(r"1[3-9]\d{9}", phone or ""):
        return None
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT user_id, nickname FROM users WHERE phone=%s", (phone,))
            row = cur.fetchone()
            if row:
                return {
                    "username": row[0],
                    "display_name": row[1] or row[0],
                    "role": "user",
                    "is_new": False,
                }
            user_id = f"phone_{phone}"
            display = f"用户{phone[-4:]}"
            cur.execute(
                "INSERT INTO users (user_id, nickname, phone) VALUES (%s,%s,%s) "
                "ON CONFLICT (user_id) DO UPDATE SET phone=EXCLUDED.phone",
                (user_id, display, phone),
            )
            conn.commit()
            return {
                "username": user_id,
                "display_name": display,
                "role": "user",
                "is_new": True,
            }
    except Exception:
        return None


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    """校验账号密码，成功返回用户信息，失败返回 None。

    分表：先查运营账号 admin_users（admin/agent），再查买家 users（user）。
    """
    if not username or not password:
        return None
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT username, password_hash, display_name, role
               FROM admin_users WHERE username=%s AND role IN ('admin','agent')""",
            (username,),
        )
        row = cur.fetchone()
        if row:
            if not verify_password(password, row[1]):
                return None
            return {
                "username": row[0],
                "display_name": row[2] or row[0],
                "role": row[3],
            }
        # 买家账号：users 表
        cur.execute(
            "SELECT user_id, password_hash, nickname FROM users WHERE user_id=%s",
            (username,),
        )
        urow = cur.fetchone()
        if not urow or not verify_password(password, urow[1]):
            return None
        return {
            "username": urow[0],
            "display_name": urow[2] or urow[0],
            "role": "user",
        }


