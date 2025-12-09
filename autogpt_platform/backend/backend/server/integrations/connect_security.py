"""
Security utilities for the integration connect popup flow.

Handles state management, nonce validation, and origin verification
for the OAuth-style popup flow when connecting integrations.
"""

import hashlib
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

from prisma.models import OAuthClient
from pydantic import BaseModel

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# State expiration time
STATE_EXPIRATION_SECONDS = 600  # 10 minutes
NONCE_EXPIRATION_SECONDS = 3600  # 1 hour (nonces valid for longer to prevent races)


class ConnectState(BaseModel):
    """Pydantic model for connect state stored in Redis."""

    user_id: str
    client_id: str
    provider: str
    requested_scopes: list[str]
    redirect_origin: str
    nonce: str
    credential_id: Optional[str] = None
    created_at: str
    expires_at: str


class ConnectContinuationState(BaseModel):
    """
    State for continuing the connect flow after OAuth completes.

    When a user chooses to "connect new" during the connect flow,
    we store this state so we can complete the grant creation after
    the OAuth callback.
    """

    user_id: str
    client_id: str  # Public client ID
    client_db_id: str  # Database UUID of the OAuth client
    provider: str
    requested_scopes: list[str]  # Integration scopes (e.g., "google:gmail.readonly")
    redirect_origin: str
    nonce: str
    created_at: str


# Continuation state expiration (same as regular state)
CONTINUATION_EXPIRATION_SECONDS = 600  # 10 minutes


async def store_connect_continuation(
    user_id: str,
    client_id: str,
    client_db_id: str,
    provider: str,
    requested_scopes: list[str],
    redirect_origin: str,
    nonce: str,
) -> str:
    """
    Store continuation state for completing connect flow after OAuth.

    Args:
        user_id: User initiating the connection
        client_id: Public OAuth client ID
        client_db_id: Database UUID of the OAuth client
        provider: Integration provider name
        requested_scopes: Requested integration scopes
        redirect_origin: Origin to send postMessage to
        nonce: Client-provided nonce for replay protection

    Returns:
        Continuation token to be stored in OAuth state metadata
    """
    token = generate_connect_token()
    now = datetime.now(timezone.utc)

    state = ConnectContinuationState(
        user_id=user_id,
        client_id=client_id,
        client_db_id=client_db_id,
        provider=provider,
        requested_scopes=requested_scopes,
        redirect_origin=redirect_origin,
        nonce=nonce,
        created_at=now.isoformat(),
    )

    redis = await get_redis_async()
    key = f"connect_continuation:{token}"
    await redis.setex(key, CONTINUATION_EXPIRATION_SECONDS, state.model_dump_json())

    logger.debug(f"Stored connect continuation state for token {token[:8]}...")
    return token


async def get_connect_continuation(token: str) -> Optional[ConnectContinuationState]:
    """
    Get continuation state without consuming it.

    Args:
        token: Continuation token

    Returns:
        ConnectContinuationState or None if not found/expired
    """
    redis = await get_redis_async()
    key = f"connect_continuation:{token}"
    data = await redis.get(key)

    if not data:
        return None

    return ConnectContinuationState.model_validate_json(data)


async def consume_connect_continuation(
    token: str,
) -> Optional[ConnectContinuationState]:
    """
    Get and consume (delete) continuation state.

    This ensures the token can only be used once.

    Args:
        token: Continuation token

    Returns:
        ConnectContinuationState or None if not found/expired
    """
    redis = await get_redis_async()
    key = f"connect_continuation:{token}"

    # Atomic get-and-delete to prevent race conditions
    data = await redis.getdel(key)
    if not data:
        return None

    state = ConnectContinuationState.model_validate_json(data)
    logger.debug(f"Consumed connect continuation state for token {token[:8]}...")

    return state


def generate_connect_token() -> str:
    """Generate a secure random token for connect state."""
    return secrets.token_urlsafe(32)


async def store_connect_state(
    user_id: str,
    client_id: str,
    provider: str,
    requested_scopes: list[str],
    redirect_origin: str,
    nonce: str,
    credential_id: Optional[str] = None,
) -> str:
    """
    Store connect state in Redis and return a state token.

    Args:
        user_id: User initiating the connection
        client_id: OAuth client ID (public identifier)
        provider: Integration provider name
        requested_scopes: Requested integration scopes
        redirect_origin: Origin to send postMessage to
        nonce: Client-provided nonce for replay protection
        credential_id: Optional existing credential to grant access to

    Returns:
        State token to be used in the connect flow
    """
    token = generate_connect_token()
    now = datetime.now(timezone.utc)
    expires_at = now.timestamp() + STATE_EXPIRATION_SECONDS

    state = ConnectState(
        user_id=user_id,
        client_id=client_id,
        provider=provider,
        requested_scopes=requested_scopes,
        redirect_origin=redirect_origin,
        nonce=nonce,
        credential_id=credential_id,
        created_at=now.isoformat(),
        expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
    )

    redis = await get_redis_async()
    key = f"connect_state:{token}"
    await redis.setex(key, STATE_EXPIRATION_SECONDS, state.model_dump_json())

    logger.debug(f"Stored connect state for token {token[:8]}...")
    return token


async def get_connect_state(token: str) -> Optional[ConnectState]:
    """
    Get connect state without consuming it.

    Args:
        token: State token

    Returns:
        ConnectState or None if not found/expired
    """
    redis = await get_redis_async()
    key = f"connect_state:{token}"
    data = await redis.get(key)

    if not data:
        return None

    return ConnectState.model_validate_json(data)


async def consume_connect_state(token: str) -> Optional[ConnectState]:
    """
    Get and consume (delete) connect state.

    This ensures the token can only be used once.

    Args:
        token: State token

    Returns:
        ConnectState or None if not found/expired
    """
    redis = await get_redis_async()
    key = f"connect_state:{token}"

    # Atomic get-and-delete to prevent race conditions
    data = await redis.getdel(key)
    if not data:
        return None

    state = ConnectState.model_validate_json(data)
    logger.debug(f"Consumed connect state for token {token[:8]}...")

    return state


async def validate_nonce(client_id: str, nonce: str) -> bool:
    """
    Validate that a nonce hasn't been used before (replay protection).

    Uses atomic SET NX EX for check-and-set with automatic TTL expiry.

    Args:
        client_id: OAuth client ID
        nonce: Client-provided nonce

    Returns:
        True if nonce is valid (not replayed)
    """
    redis = await get_redis_async()

    # Create a hash of the nonce for storage
    nonce_hash = hashlib.sha256(nonce.encode()).hexdigest()
    key = f"nonce:{client_id}:{nonce_hash}"

    # Atomic set-if-not-exists with expiration (prevents race condition)
    was_set = await redis.set(key, "1", nx=True, ex=NONCE_EXPIRATION_SECONDS)
    if was_set:
        return True

    logger.warning(f"Nonce replay detected for client {client_id}")
    return False


def validate_redirect_origin(origin: str, client: OAuthClient) -> bool:
    """
    Validate that a redirect origin is allowed for the client.

    The origin must match one of the client's registered redirect URIs
    or webhook domains.

    Args:
        origin: Origin URL to validate
        client: OAuth client to check against

    Returns:
        True if origin is allowed
    """
    from backend.util.url import hostname_matches_any_domain

    try:
        parsed_origin = urlparse(origin)
        origin_host = parsed_origin.netloc.lower()

        # Check against redirect URIs
        for redirect_uri in client.redirectUris:
            parsed_redirect = urlparse(redirect_uri)
            if parsed_redirect.netloc.lower() == origin_host:
                return True

        # Check against webhook domains
        if hostname_matches_any_domain(origin_host, client.webhookDomains):
            return True

        return False

    except Exception:
        return False


def create_post_message_data(
    success: bool,
    grant_id: Optional[str] = None,
    credential_id: Optional[str] = None,
    provider: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    nonce: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create the postMessage data to send back to the opener.

    Args:
        success: Whether the operation succeeded
        grant_id: ID of the created grant (if successful)
        credential_id: ID of the credential (if successful)
        provider: Provider name
        error: Error code (if failed)
        error_description: Human-readable error description
        nonce: Original nonce for correlation

    Returns:
        Dictionary to be sent via postMessage
    """
    data: dict[str, Any] = {
        "type": "autogpt_connect_result",
        "success": success,
    }

    if nonce:
        data["nonce"] = nonce

    if success:
        data["grant_id"] = grant_id
        data["credential_id"] = credential_id
        data["provider"] = provider
    else:
        data["error"] = error
        data["error_description"] = error_description

    return data
