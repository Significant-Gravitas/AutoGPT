"""Reserve a pre-Scrypt budget for failed and in-flight API-key validations.

Successful validations refund their reservation; credentials are never cached.
Single-key Lua scripts work across Redis cluster slots. The whole admission
check fails open within 250ms when Redis is unavailable.
"""

import asyncio
import logging
import re
from collections.abc import Awaitable
from datetime import UTC, datetime
from ipaddress import IPv6Address, ip_address, ip_network
from typing import cast

from autogpt_libs.api_key.keysmith import APIKeySmith
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field
from redis.exceptions import RedisClusterException, RedisError

from backend.data.redis_client import get_redis_async
from backend.monitoring.instrumentation import record_rate_limit_hit

logger = logging.getLogger(__name__)

AUTH_VALIDATE_WINDOW_SECONDS = 60
AUTH_VALIDATE_PER_IP_HEAD_MAX_REQUESTS = 60
AUTH_VALIDATE_REDIS_TIMEOUT_SECONDS = 0.25

_INCR_OPEN_WINDOW = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""
_REFUND_WINDOW = """
local count = tonumber(redis.call('GET', KEYS[1]) or '0')
if count > 0 then
    return redis.call('DECR', KEYS[1])
end
return 0
"""
_SAFE_HEAD = re.compile(r"[A-Za-z0-9_-]+")
_REDIS_FAILURES = (RedisError, RedisClusterException, TimeoutError)


class APIKeyValidationBudget(BaseModel):
    head_key: str = Field(repr=False)


async def enforce_api_key_validate_rate_limit(
    request: Request, plaintext_key: str
) -> APIKeyValidationBudget | None:
    """Reserve before validation, or raise 429 without running Scrypt."""
    head = key_head_from_plaintext(plaintext_key)
    ip = client_ip_from_request(request)
    if head is None or ip is None:
        return None

    now = datetime.now(UTC)
    window = int(now.timestamp()) // AUTH_VALIDATE_WINDOW_SECONDS
    budget = APIKeyValidationBudget(
        head_key=f"external_api:auth_validate:rl:{ip}:{head}:{window}",
    )
    try:
        await asyncio.wait_for(
            _reserve(budget, now), timeout=AUTH_VALIDATE_REDIS_TIMEOUT_SECONDS
        )
    except _REDIS_FAILURES as error:
        logger.warning(
            f"API-key auth rate-limit admission failed open: {type(error).__name__}"
        )
        return None
    return budget


async def refund_api_key_validate_rate_limit(
    budget: APIKeyValidationBudget | None,
) -> None:
    """Refund a valid key against its original window, without recreating expiry."""
    if budget is None:
        return
    try:
        await asyncio.wait_for(
            _refund(budget), timeout=AUTH_VALIDATE_REDIS_TIMEOUT_SECONDS
        )
    except _REDIS_FAILURES as error:
        logger.warning(f"API-key auth rate-limit refund failed: {type(error).__name__}")


def client_ip_from_request(request: Request) -> str | None:
    """Use the server's trusted-proxy-resolved peer, never raw forwarding headers.

    IPv6 clients share a /64 budget; IPv4-mapped addresses use their IPv4 budget.
    Missing peers skip IP limiting rather than sharing a global anonymous bucket.
    Deployments must supply a client address and configure trusted proxy handling
    at the ASGI server boundary to enforce these per-client budgets.
    """
    if request.client is None:
        return None
    try:
        address = ip_address(request.client.host)
    except ValueError:
        return None
    if address.version == 6:
        ipv6 = IPv6Address(address)
        if ipv6.ipv4_mapped is not None:
            return ipv6.ipv4_mapped.compressed
        return ip_network(f"{ipv6}/64", strict=False).with_prefixlen
    return address.compressed


def key_head_from_plaintext(plaintext_key: str) -> str | None:
    """Skip cheap rejects whose prefix cannot match a generated API key."""
    if not plaintext_key.startswith(APIKeySmith.PREFIX):
        return None
    head = plaintext_key[: APIKeySmith.HEAD_LENGTH]
    if len(head) != APIKeySmith.HEAD_LENGTH or not _SAFE_HEAD.fullmatch(head):
        return None
    return head


async def _reserve(budget: APIKeyValidationBudget, now: datetime) -> None:
    count = await _eval(
        _INCR_OPEN_WINDOW, budget.head_key, str(AUTH_VALIDATE_WINDOW_SECONDS)
    )
    if count > AUTH_VALIDATE_PER_IP_HEAD_MAX_REQUESTS:
        record_rate_limit_hit("external_api:auth_validate", "anonymous")
        raise HTTPException(
            status_code=429,
            detail="Too many API key authentication attempts. Try again shortly.",
            headers={
                "Retry-After": str(
                    AUTH_VALIDATE_WINDOW_SECONDS
                    - int(now.timestamp()) % AUTH_VALIDATE_WINDOW_SECONDS
                )
            },
        )


async def _refund(budget: APIKeyValidationBudget) -> None:
    await _eval(_REFUND_WINDOW, budget.head_key)


async def _eval(script: str, key: str, *args: str) -> int:
    try:
        redis = await get_redis_async()
        return await cast(Awaitable[int], redis.eval(script, 1, key, *args))
    except (OSError, ValueError) as error:
        raise RedisError("Rate-limit Redis operation failed") from error
