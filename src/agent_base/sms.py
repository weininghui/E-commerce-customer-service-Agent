"""手机验证码登录（短信通道：mock 开发模式 / 阿里云 SMS 生产模式）。

设计（零强依赖）：
- 验证码 6 位数字，存 Redis（key sms:code:{phone}，5 分钟有效，最多重试 5 次）；
- 发送频率护栏：同一手机号 60 秒内只发一次（Redis 计数），防短信轰炸；
- provider=mock（默认，未配置阿里云密钥时）：验证码直接随响应返回/打日志，
  便于本地与演示联调；provider=aliyun：调阿里云短信 API 真实发送；
- 发送失败不泄露内部错误（统一返回 ok），日志留痕。
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

CODE_TTL_SEC = 300          # 验证码有效期 5 分钟
MAX_ATTEMPTS = 5            # 单手机号最多校验次数（防爆破）
RESEND_INTERVAL_SEC = 60    # 同手机号重发间隔（防轰炸）


def _redis() -> Any:
    """Redis 客户端（连接失败返回 None，降级内存 map）。"""
    from agent_base.storage.cache import _get_client

    return _get_client()


# Redis 不可用时的进程内兜底（单实例可用，多实例不共享——生产应确保 Redis 在线）
_fallback: dict[str, str] = {}
_fallback_sent: dict[str, float] = {}


def _sms_cfg() -> dict[str, Any]:
    """短信配置：provider=mock（默认）| aliyun。"""
    return {
        "provider": os.getenv("SMS_PROVIDER", "mock").strip().lower() or "mock",
        "access_key": os.getenv("ALIYUN_SMS_ACCESS_KEY", "").strip(),
        "access_secret": os.getenv("ALIYUN_SMS_ACCESS_SECRET", "").strip(),
        "sign_name": os.getenv("ALIYUN_SMS_SIGN_NAME", "").strip(),
        "template_code": os.getenv("ALIYUN_SMS_TEMPLATE_CODE", "").strip(),
    }


def send_code(phone: str) -> dict[str, Any]:
    """给手机号发送验证码（含频率护栏）。

    Args:
        phone: 11 位手机号。

    Returns:
        {ok, sent, code?, message}；mock 模式 code 随响应返回（联调用），
        aliyun 模式不发回 code（真实发送）。
    """
    import time

    client = _redis()
    now = time.time()
    if client is not None:
        last = client.get(f"sms:sent:{phone}")
        if last and now - float(last) < RESEND_INTERVAL_SEC:
            return {"ok": False, "sent": False, "message": "发送太频繁，请稍后再试"}
    else:
        if _fallback_sent.get(phone) and now - _fallback_sent[phone] < RESEND_INTERVAL_SEC:
            return {"ok": False, "sent": False, "message": "发送太频繁，请稍后再试"}

    code = f"{random.randint(0, 999999):06d}"
    if client is not None:
        client.set(f"sms:code:{phone}", code, ex=CODE_TTL_SEC)
        client.set(f"sms:attempts:{phone}", 0, ex=CODE_TTL_SEC)
        client.set(f"sms:sent:{phone}", str(now), ex=RESEND_INTERVAL_SEC)
    else:
        _fallback[f"sms:code:{phone}"] = code
        _fallback_sent[phone] = now

    cfg = _sms_cfg()
    if cfg["provider"] == "aliyun" and cfg["access_key"] and cfg["sign_name"] and cfg["template_code"]:
        try:
            _send_aliyun_sms(phone, code, cfg)
            return {"ok": True, "sent": True, "message": "验证码已发送"}
        except Exception:
            return {"ok": False, "sent": False, "message": "短信发送失败，请稍后再试"}
    # mock 开发模式：验证码随响应返回（仅联调/演示；生产务必配 SMS_PROVIDER=aliyun）
    return {"ok": True, "sent": False, "code": code, "message": "开发模式：验证码见响应（配置阿里云 SMS 后真实发送）"}


def verify_code(phone: str, code: str) -> bool:
    """校验验证码（一次性消费 + 次数上限，防爆破）。

    Args:
        phone: 手机号。
        code: 用户输入的 6 位验证码。

    Returns:
        是否正确且未过期；校验通过后立即作废。
    """
    client = _redis()
    if client is not None:
        key = f"sms:code:{phone}"
        saved = client.get(key)
        if not saved:
            return False
        attempts = int(client.get(f"sms:attempts:{phone}") or 0)
        if attempts >= MAX_ATTEMPTS:
            client.delete(key)
            return False
        if saved != code:
            client.incr(f"sms:attempts:{phone}")
            return False
        client.delete(key)
        client.delete(f"sms:attempts:{phone}")
        return True
    saved = _fallback.get(f"sms:code:{phone}")
    if not saved:
        return False
    if saved != code:
        return False
    _fallback.pop(f"sms:code:{phone}", None)
    return True


def _send_aliyun_sms(phone: str, code: str, cfg: dict[str, Any]) -> None:
    """调用阿里云短信发送（OpenAPI RPC 风格，仅依赖 urllib + hmac）。

    Args:
        phone: 接收手机号。
        code: 验证码。
        cfg: aliyun 配置（access_key/access_secret/sign_name/template_code）。

    Raises:
        RuntimeError: 调用失败（上层统一兜底文案）。
    """
    import datetime
    import urllib.parse

    params = {
        "AccessKeyId": cfg["access_key"],
        "Action": "SendSms",
        "Format": "JSON",
        "PhoneNumbers": phone,
        "RegionId": "cn-hangzhou",
        "SignatureMethod": "HMAC-SHA1",
        "SignatureNonce": str(random.randint(10**15, 10**16 - 1)),
        "SignatureVersion": "1.0",
        "SignName": cfg["sign_name"],
        "TemplateCode": cfg["template_code"],
        "TemplateParam": json.dumps({"code": code}, ensure_ascii=False),
        "Timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Version": "2017-05-25",
    }
    # 排序 + 规范化查询串（阿里云 RPC 签名）
    keys = sorted(params)
    query = "&".join(
        f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(str(params[k]), safe='')}" for k in keys
    )
    string_to_sign = "GET&%2F&" + urllib.parse.quote(query, safe="")
    signature = base64_hmac_sha1(cfg["access_secret"] + "&", string_to_sign)
    params["Signature"] = signature
    url = "https://dysmsapi.aliyuncs.com/?" + "&".join(
        f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(str(params[k]), safe='')}" for k in params
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8", "ignore"))
    if data.get("Code") != "OK":
        raise RuntimeError(f"aliyun sms error: {data.get('Code')} {data.get('Message')}")


def base64_hmac_sha1(secret: str, message: str) -> str:
    """HMAC-SHA1 签名 → base64（阿里云 RPC 签名算法）。"""
    import base64 as _b64
    import hashlib as _hl
    import hmac as _hm

    return _b64.b64encode(_hm.new(secret.encode("utf-8"), message.encode("utf-8"), _hl.sha1).digest()).decode("ascii")
