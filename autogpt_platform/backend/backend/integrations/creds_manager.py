import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from autogpt_libs.utils.synchronize import AsyncRedisKeyedMutex
from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.data.model import Credentials, OAuth2Credentials
from backend.data.redis_client import get_redis_async
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.integrations.oauth import CREDENTIALS_BY_PROVIDER, HANDLERS_BY_NAME
from backend.integrations.providers import ProviderName
from backend.util.exceptions import MissingConfigError
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.integrations.oauth import BaseOAuthHandler

logger = logging.getLogger(__name__)
settings = Settings()


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
        self._locks = None

    async def locks(self) -> AsyncRedisKeyedMutex:
        if self._locks:
            return self._locks

        self._locks = AsyncRedisKeyedMutex(await get_redis_async())
        return self._locks

    async def create(self, user_id: str, credentials: Credentials) -> None:
        return await self.store.add_creds(user_id, credentials)

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
        async with self._locked(user_id, credentials.id, "refresh"):
            oauth_handler = await _get_provider_oauth_handler(credentials.provider)
            if oauth_handler.needs_refresh(credentials):
                logger.debug(
                    f"Refreshing '{credentials.provider}' "
                    f"credentials #{credentials.id}"
                )
                _lock = None
                if lock:
                    # Wait until the credentials are no longer in use anywhere
                    _lock = await self._acquire_lock(user_id, credentials.id)

                fresh_credentials = await oauth_handler.refresh_tokens(credentials)
                await self.store.update_creds(user_id, fresh_credentials)
                if _lock and (await _lock.locked()) and (await _lock.owned()):
                    try:
                        await _lock.release()
                    except Exception as e:
                        logger.warning(f"Failed to release OAuth refresh lock: {e}")

                credentials = fresh_credentials
        return credentials

    async def update(self, user_id: str, updated: Credentials) -> None:
        async with self._locked(user_id, updated.id):
            await self.store.update_creds(user_id, updated)

    async def delete(self, user_id: str, credentials_id: str) -> None:
        async with self._locked(user_id, credentials_id):
            await self.store.delete_creds_by_id(user_id, credentials_id)

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
                except Exception as e:
                    logger.warning(f"Failed to release credentials lock: {e}")

    async def release_all_locks(self):
        """Call this on process termination to ensure all locks are released"""
        await (await self.locks()).release_all_locks()
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
