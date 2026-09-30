"""Auth-path rate limit for external API key validation.

``validate_api_key`` looks up candidates by a short plaintext head and then
runs Scrypt (n=2**14) per candidate. Failed attempts with a colliding head
therefore burn CPU/memory even though they always return 401. Post-auth
route quotas (see open PR #14545) never run on that path.

This module puts a fixed-window Redis counter **in front of** Scrypt, keyed
by client IP and key head, so abusive bursts get HTTP 429 before unbounded
crypto work. Successful authentications under normal rates still pass.

Availability: Redis trouble **fails open** with a short deadline (same
trade-off as ``credits_rate_limit`` / search QPS) — a global Redis blip must
not take the external API offline. When Redis is healthy the budget bounds
scrypt burn.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import Any, cast

import fastapi
from fastapi import Request
from redis.exceptions import RedisClusterException, RedisError

from autogpt_libs.api_key.keysmith import APIKeySmith
from backend.data.redis_client import get_redis_async
from backend.monitoring.instrumentation import record_rate_limit_hit

logger = logging.getLogger(__name__)

# Per (IP, key-head) attempts that may run Scrypt. Generous for legitimate
# clients (well above steady polling) while stopping a concurrent hammer.
AUTH_VALIDATE_WINDOW_SECONDS = 60
AUTH_VALIDATE_MAX_REQUESTS = 60

# Per-IP ceiling across all heads so an enumerator cannot rotate heads to
# multiply the per-head budget indefinitely from one source.
AUTH_VALIDATE_IP_MAX_REQUESTS = 120

AUTH_VALIDATE_REDIS_TIMEOUT_SECONDS = 0.25

_INCR_OPEN_WINDOW = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""

# Heads are PREFIX + urlsafe chars; keep redis keys printable and bounded.
_SAFE_HEAD = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_SAFE_IP = re.compile(r"^[0-9A-Fa-f.:]{1,64}$")


def client_ip_from_request(request: Request) -> str:
    """Best-effort client IP for rate-limit bucketing.

    Prefer the first ``X-Forwarded-For`` hop when present (typical behind a
    reverse proxy), else the ASGI client host. Spoofable without a trusted
    proxy — still useful as one dimension alongside the key head.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",", 1)[0].strip()
        if first and _SAFE_IP.match(first):
            return first
    if request.client and request.client.host:
        host = request.client.host
        if _SAFE_IP.match(host):
            return host
        return "unknown"
    return "unknown"


def key_head_from_plaintext(plaintext_key: str) -> str | None:
    """Return the lookup head when the key is well-formed enough to hit Scrypt.

    Malformed / wrong-prefix keys are rejected cheaply in ``validate_api_key``
    without Scrypt, so they do not consume this budget.
    """
    if not plaintext_key.startswith(APIKeySmith.PREFIX):
        return None
    if len(plaintext_key) < APIKeySmith.HEAD_LENGTH:
        return None
    head = plaintext_key[: APIKeySmith.HEAD_LENGTH]
    if not _SAFE_HEAD.match(head):
        return None
    return head


def _window_bucket(now: datetime, window_seconds: int) -> int:
    return int(now.timestamp()) // window_seconds


def _head_key(ip: str, head: str, *, now: datetime) -> str:
    bucket = _window_bucket(now, AUTH_VALIDATE_WINDOW_SECONDS)
    return f"external_api:auth_validate:rl:{ip}:{head}:{bucket}"


def _ip_key(ip: str, *, now: datetime) -> str:
    bucket = _window_bucket(now, AUTH_VALIDATE_WINDOW_SECONDS)
    return f"external_api:auth_validate:ip:{ip}:{bucket}"


async def _incr_window(key: str, window_seconds: int) -> int:
    redis = await get_redis_async()
    return await cast(
        Any,
        redis.eval(_INCR_OPEN_WINDOW, 1, key, str(window_seconds)),
    )


async def _incr_with_deadline(key: str, window_seconds: int) -> int | None:
    """Increment or return ``None`` when Redis cannot answer in time / at all."""
    try:
        return await asyncio.wait_for(
            _incr_window(key, window_seconds),
            timeout=AUTH_VALIDATE_REDIS_TIMEOUT_SECONDS,
        )
    except (
        RedisError,
        RedisClusterException,
        ConnectionError,
        OSError,
        asyncio.TimeoutError,
        ValueError,
    ) as e:
        logger.warning(
            "API-key auth rate-limit check failed open for key %s: %s",
            key,
            e,
        )
        return None


def _retry_after_seconds(now: datetime, window_seconds: int) -> int:
    return window_seconds - (int(now.timestamp()) % window_seconds)


def _raise_429(*, now: datetime, detail: str) -> None:
    retry_after = _retry_after_seconds(now, AUTH_VALIDATE_WINDOW_SECONDS)
    record_rate_limit_hit("external_api:auth_validate", "anonymous")
    raise fastapi.HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(retry_after)},
    )


async def enforce_api_key_validate_rate_limit(
    request: Request,
    plaintext_key: str,
) -> None:
    """Raise HTTP 429 when this IP (+ key head) exceeds the auth-path budget.

    Called **before** ``validate_api_key`` so Scrypt is not run once the
    fixed window is exhausted. Keys that cannot reach Scrypt (bad prefix)
    skip the limiter.
    """
    head = key_head_from_plaintext(plaintext_key)
    if head is None:
        return

    ip = client_ip_from_request(request)
    now = datetime.now(UTC)

    ip_count = await _incr_with_deadline(
        _ip_key(ip, now=now),
        AUTH_VALIDATE_WINDOW_SECONDS,
    )
    if ip_count is not None and ip_count > AUTH_VALIDATE_IP_MAX_REQUESTS:
        logger.info(
            "API-key auth IP rate limit hit for %s (count=%s)",
            ip,
            ip_count,
        )
        _raise_429(
            now=now,
            detail=(
                f"API key authentication rate limit exceeded "
                f"({AUTH_VALIDATE_IP_MAX_REQUESTS} attempts per "
                f"{AUTH_VALIDATE_WINDOW_SECONDS}s from this address). "
                f"Try again shortly."
            ),
        )

    head_count = await _incr_with_deadline(
        _head_key(ip, head, now=now),
        AUTH_VALIDATE_WINDOW_SECONDS,
    )
    if head_count is not None and head_count > AUTH_VALIDATE_MAX_REQUESTS:
        logger.info(
            "API-key auth head rate limit hit for %s head=%s (count=%s)",
            ip,
            head,
            head_count,
        )
        _raise_429(
            now=now,
            detail=(
                f"API key authentication rate limit exceeded "
                f"({AUTH_VALIDATE_MAX_REQUESTS} attempts per "
                f"{AUTH_VALIDATE_WINDOW_SECONDS}s for this key prefix). "
                f"Try again shortly."
            ),
        )
