import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from autogpt_libs.utils.synchronize import AsyncRedisKeyedMutex
from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.data.model import Credentials, OAuth2Credentials
from backend.integrations.credentials_store import (
    IntegrationCredentialsStore,
    provider_matches,
)
from backend.integrations.oauth import CREDENTIALS_BY_PROVIDER, HANDLERS_BY_NAME
from backend.integrations.providers import ProviderName
from backend.util.exceptions import MissingConfigError
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.integrations.oauth import BaseOAuthHandler

logger = logging.getLogger(__name__)
settings = Settings()


_on_creds_changed: Callable[[str, str], None] | None = None


def register_creds_changed_hook(hook: Callable[[str, str], None]) -> None:
    """Register a callback invoked after any credential is created/updated/deleted.

    The callback receives ``(user_id, provider)`` and should be idempotent.
    Only one hook can be registered at a time.  Intended to be called once at
    application startup (e.g. by the copilot module) without creating an
    import cycle.

    Raises:
        RuntimeError: If a hook is already registered.  Call
            :func:`unregister_creds_changed_hook` first if replacement is needed.
    """
    global _on_creds_changed
    if _on_creds_changed is not None:
        raise RuntimeError(
            "A creds_changed hook is already registered. "
            "Call unregister_creds_changed_hook() before registering a new one."
        )
    _on_creds_changed = hook


def unregister_creds_changed_hook() -> None:
    """Remove the currently registered creds-changed hook (if any).

    Primarily useful in tests to reset global state between test cases.
    """
    global _on_creds_changed
    _on_creds_changed = None


def _invoke_creds_changed_hook(user_id: str, provider: str) -> None:
    """Invoke the registered creds-changed hook (if any)."""
    if _on_creds_changed is not None:
        try:
            _on_creds_changed(user_id, provider)
        except Exception:
            logger.warning(
                "Credential-change hook failed for user=%s provider=%s",
                user_id,
                provider,
                exc_info=True,
            )


class IntegrationCredentialsManager:
    """
    Handles the lifecycle of integration credentials.
    - Automatically refreshes requested credentials if needed.
    - Uses locking mechanisms to ensure system-wide consistency and
      prevent invalidation of in-use tokens.

    ### ⚠️ Gotcha
    With `acquire(..)`, credentials can only be in use in one place at a time (e.g. one
    block execution).

    ### Locking mechanism
    - Because *getting* credentials can result in a refresh (= *invalidation* +
      *replacement*) of the stored credentials, *getting* is an operation that
      potentially requires read/write access.
    - Checking whether a token has to be refreshed is subject to an additional `refresh`
      scoped lock to prevent unnecessary sequential refreshes when multiple executions
      try to access the same credentials simultaneously.
    - We MUST lock credentials while in use to prevent them from being invalidated while
      they are in use, e.g. because they are being refreshed by a different part
      of the system.
    - The `!time_sensitive` lock in `acquire(..)` is part of a two-tier locking
      mechanism in which *updating* gets priority over *getting* credentials.
      This is to prevent a long queue of waiting *get* requests from blocking essential
      credential refreshes or user-initiated updates.

    It is possible to implement a reader/writer locking system where either multiple
    readers or a single writer can have simultaneous access, but this would add a lot of
    complexity to the mechanism. I don't expect the current ("simple") mechanism to
    cause so much latency that it's worth implementing.
    """

    def __init__(self):
        self.store = IntegrationCredentialsStore()

    async def locks(self) -> AsyncRedisKeyedMutex:
        # Delegate to store's @thread_cached locks.  Manager uses these for
        # fine-grained per-credential locking (refresh, acquire); the store
        # uses its own for coarse per-user integrations locking.  Same mutex
        # type, different key spaces — no collision.
        return await self.store.locks()

    async def create(self, user_id: str, credentials: Credentials) -> None:
        result = await self.store.add_creds(user_id, credentials)
        # Notify listeners so downstream caches are invalidated immediately.
        _invoke_creds_changed_hook(user_id, credentials.provider)
        return result

    async def exists(self, user_id: str, credentials_id: str) -> bool:
        return (await self.store.get_creds_by_id(user_id, credentials_id)) is not None

    async def get(
        self, user_id: str, credentials_id: str, lock: bool = True
    ) -> Credentials | None:
        credentials = await self.store.get_creds_by_id(user_id, credentials_id)
        if not credentials:
            return None

        # Refresh OAuth credentials if needed
        if credentials.type == "oauth2" and credentials.access_token_expires_at:
            logger.debug(
                f"Credentials #{credentials.id} expire at "
                f"{datetime.fromtimestamp(credentials.access_token_expires_at)}; "
                f"current time is {datetime.now()}"
            )
            credentials = await self.refresh_if_needed(user_id, credentials, lock)
        else:
            logger.debug(f"Credentials #{credentials.id} never expire")

        return credentials

    async def acquire(
        self, user_id: str, credentials_id: str
    ) -> tuple[Credentials, AsyncRedisLock]:
        """
        ⚠️ WARNING: this locks credentials system-wide and blocks both acquiring
        and updating them elsewhere until the lock is released.
        See the class docstring for more info.
        """
        # Use a low-priority (!time_sensitive) locking queue on top of the general lock
        # to allow priority access for refreshing/updating the tokens.
        async with self._locked(user_id, credentials_id, "!time_sensitive"):
            lock = await self._acquire_lock(user_id, credentials_id)
        credentials = await self.get(user_id, credentials_id, lock=False)
        if not credentials:
            raise ValueError(
                f"Credentials #{credentials_id} for user #{user_id} not found"
            )
        return credentials, lock

    def cached_getter(
        self, user_id: str
    ) -> Callable[[str], "Coroutine[Any, Any, Credentials | None]"]:
        all_credentials = None

        async def get_credentials(creds_id: str) -> "Credentials | None":
            nonlocal all_credentials
            if not all_credentials:
                # Fetch credentials on first necessity
                all_credentials = await self.store.get_all_creds(user_id)

            credential = next((c for c in all_credentials if c.id == creds_id), None)
            if not credential:
                return None
            if credential.type != "oauth2" or not credential.access_token_expires_at:
                # Credential doesn't expire
                return credential

            # Credential is OAuth2 credential and has expiration timestamp
            return await self.refresh_if_needed(user_id, credential)

        return get_credentials

    async def refresh_if_needed(
        self, user_id: str, credentials: OAuth2Credentials, lock: bool = True
    ) -> OAuth2Credentials:
        # When lock=False, skip ALL Redis locking (both the outer "refresh" scope
        # lock and the inner credential lock).  This is used by the copilot's
        # integration_creds module which runs across multiple threads with separate
        # event loops; acquiring a Redis lock whose asyncio.Lock() was created on
        # a different loop raises "Future attached to a different loop".
        if lock:
            return await self._refresh_locked(user_id, credentials)
        return await self._refresh_unlocked(user_id, credentials)

    async def _get_oauth_handler(
        self, credentials: OAuth2Credentials
    ) -> "BaseOAuthHandler":
        """Resolve the appropriate OAuth handler for the given credentials."""
        if provider_matches(credentials.provider, ProviderName.MCP.value):
            return create_mcp_oauth_handler(credentials)
        return await _get_provider_oauth_handler(credentials.provider)

    async def _refresh_locked(
        self, user_id: str, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        async with self._locked(user_id, credentials.id, "refresh"):
            oauth_handler = await self._get_oauth_handler(credentials)
            if oauth_handler.needs_refresh(credentials):
                logger.debug(
                    "Refreshing '%s' credentials #%s",
                    credentials.provider,
                    credentials.id,
                )
                # Wait until the credentials are no longer in use anywhere
                _lock = await self._acquire_lock(user_id, credentials.id)
                try:
                    fresh_credentials = await oauth_handler.refresh_tokens(credentials)
                    await self.store.update_creds(user_id, fresh_credentials)
                    _invoke_creds_changed_hook(user_id, fresh_credentials.provider)
                    credentials = fresh_credentials
                finally:
                    if (await _lock.locked()) and (await _lock.owned()):
                        try:
                            await _lock.release()
                        except Exception:
                            logger.warning(
                                "Failed to release OAuth refresh lock",
                                exc_info=True,
                            )
        return credentials

    async def _refresh_unlocked(
        self, user_id: str, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        """Best-effort token refresh without any Redis locking.

        Safe for use from multi-threaded contexts (e.g. copilot workers) where
        each thread has its own event loop and sharing Redis-backed asyncio locks
        is not possible.  Concurrent refreshes are tolerated: the last writer
        wins, and stale tokens are overwritten.
        """
        oauth_handler = await self._get_oauth_handler(credentials)
        if oauth_handler.needs_refresh(credentials):
            logger.debug(
                "Refreshing '%s' credentials #%s (lock-free)",
                credentials.provider,
                credentials.id,
            )
            fresh_credentials = await oauth_handler.refresh_tokens(credentials)
            await self.store.update_creds(user_id, fresh_credentials)
            _invoke_creds_changed_hook(user_id, fresh_credentials.provider)
            credentials = fresh_credentials
        return credentials

    async def update(self, user_id: str, updated: Credentials) -> None:
        async with self._locked(user_id, updated.id):
            await self.store.update_creds(user_id, updated)
        # Notify listeners so the updated credential is picked up immediately.
        _invoke_creds_changed_hook(user_id, updated.provider)

    async def delete(self, user_id: str, credentials_id: str) -> None:
        async with self._locked(user_id, credentials_id):
            # Read inside the lock to avoid TOCTOU — another coroutine could
            # delete the same credential between the read and the delete.
            creds = await self.store.get_creds_by_id(user_id, credentials_id)
            await self.store.delete_creds_by_id(user_id, credentials_id)
        if creds:
            _invoke_creds_changed_hook(user_id, creds.provider)

    # -- Locking utilities -- #

    async def _acquire_lock(
        self, user_id: str, credentials_id: str, *args: str
    ) -> AsyncRedisLock:
        key = (
            f"user:{user_id}",
            f"credentials:{credentials_id}",
            *args,
        )
        locks = await self.locks()
        return await locks.acquire(key)

    @asynccontextmanager
    async def _locked(self, user_id: str, credentials_id: str, *args: str):
        lock = await self._acquire_lock(user_id, credentials_id, *args)
        try:
            yield
        finally:
            if (await lock.locked()) and (await lock.owned()):
                try:
                    await lock.release()
                except Exception:
                    logger.warning(
                        "Failed to release credentials lock",
                        exc_info=True,
                    )

    async def release_all_locks(self):
        """Call this on process termination to ensure all locks are released"""
        await (await self.store.locks()).release_all_locks()


async def _get_provider_oauth_handler(provider_name_str: str) -> "BaseOAuthHandler":
    provider_name = ProviderName(provider_name_str)
    if provider_name not in HANDLERS_BY_NAME:
        raise KeyError(f"Unknown provider '{provider_name}'")

    provider_creds = CREDENTIALS_BY_PROVIDER[provider_name]
    if not provider_creds.use_secrets:
        # This is safe to do as we check that the env vars exist in the registry
        client_id = (
            os.getenv(provider_creds.client_id_env_var)
            if provider_creds.client_id_env_var
            else None
        )
        client_secret = (
            os.getenv(provider_creds.client_secret_env_var)
            if provider_creds.client_secret_env_var
            else None
        )
    else:
        client_id = getattr(settings.secrets, f"{provider_name.value}_client_id")
        client_secret = getattr(
            settings.secrets, f"{provider_name.value}_client_secret"
        )

    if not (client_id and client_secret):
        raise MissingConfigError(
            f"Integration with provider '{provider_name}' is not configured",
        )

    handler_class = HANDLERS_BY_NAME[provider_name]
    frontend_base_url = (
        settings.config.frontend_base_url or settings.config.platform_base_url
    )
    return handler_class(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=f"{frontend_base_url}/auth/integrations/oauth_callback",
    )


def create_mcp_oauth_handler(
    credentials: OAuth2Credentials,
) -> "BaseOAuthHandler":
    """Create an MCPOAuthHandler from credential metadata for token refresh.

    MCP OAuth handlers have dynamic endpoints discovered per-server, so they
    can't be registered as singletons in HANDLERS_BY_NAME. Instead, the handler
    is reconstructed from metadata stored on the credential during initial auth.
    """
    from backend.blocks.mcp.oauth import MCPOAuthHandler

    meta = credentials.metadata or {}
    token_url = meta.get("mcp_token_url", "")
    if not token_url:
        raise ValueError(
            f"MCP credential {credentials.id} is missing 'mcp_token_url' metadata; "
            "cannot refresh tokens"
        )
    return MCPOAuthHandler(
        client_id=meta.get("mcp_client_id", ""),
        client_secret=meta.get("mcp_client_secret", ""),
        redirect_uri="",  # Not needed for token refresh
        authorize_url="",  # Not needed for token refresh
        token_url=token_url,
        resource_url=meta.get("mcp_resource_url"),
    )
