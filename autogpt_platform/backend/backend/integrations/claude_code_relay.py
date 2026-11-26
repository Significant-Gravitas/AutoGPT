"""Per-run tokens for the platform's Anthropic relay route: a Claude Code
sandbox holds one in place of the user's key, and the route adds the key."""

import hashlib
import ipaddress
import logging
import secrets
from urllib.parse import urlsplit

from pydantic import BaseModel

from backend.data.redis_client import get_redis_async
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

RELAY_PATH = "/api/anthropic-relay"
RELAY_UNREACHABLE = (
    "Claude Code reaches Anthropic through this AutoGPT server, so the server "
    "must be reachable from E2B's cloud: set PLATFORM_BASE_URL to its public "
    "address, or set CLAUDE_CODE_KEY_IN_SANDBOX=true to give the sandbox the "
    "Anthropic key itself."
)

_TOKEN_KEY_PREFIX = "anthropic-relay:token:"


class RelayGrant(BaseModel):
    """What a relay token stands for: one user's one Anthropic credential."""

    user_id: str
    credentials_id: str


class RelayUnreachableError(ValueError):
    """The relay route is not at an address E2B's cloud can reach."""


def relay_base_url() -> str:
    """The ``ANTHROPIC_BASE_URL`` a sandbox's Claude Code reaches the relay at."""
    base = Settings().config.platform_base_url.rstrip("/")
    if not _reachable_from_outside(base):
        raise RelayUnreachableError(RELAY_UNREACHABLE)
    return f"{base}{RELAY_PATH}"


async def mint_relay_token(user_id: str, credentials_id: str, ttl_seconds: int) -> str:
    """A token the relay accepts for *credentials_id* until it is revoked or
    *ttl_seconds* pass. Only its digest is stored."""
    token = secrets.token_urlsafe(32)
    grant = RelayGrant(user_id=user_id, credentials_id=credentials_id)
    redis = await get_redis_async()
    await redis.set(_token_key(token), grant.model_dump_json(), ex=ttl_seconds)
    return token


async def relay_grant(token: str) -> RelayGrant | None:
    """What *token* stands for, or None once it is revoked or expired."""
    redis = await get_redis_async()
    stored = await redis.get(_token_key(token))
    return RelayGrant.model_validate_json(stored) if stored else None


async def revoke_relay_token(token: str) -> None:
    """End *token* now. Never raises: a token it cannot reach ends at its TTL."""
    try:
        redis = await get_redis_async()
        await redis.delete(_token_key(token))
    except Exception as error:
        logger.warning(f"Could not revoke an Anthropic relay token: {error!r}")


def _token_key(token: str) -> str:
    return _TOKEN_KEY_PREFIX + hashlib.sha256(token.encode()).hexdigest()


def _reachable_from_outside(base_url: str) -> bool:
    """Whether *base_url* can name a server outside this machine's network. A
    host name is taken at its word; a loopback or private address is not."""
    host = urlsplit(base_url).hostname
    if not host or host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return True
