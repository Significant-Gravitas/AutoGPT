import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Coroutine
from uuid import uuid4

from autogpt_libs.utils.synchronize import AsyncRedisKeyedMutex
from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.data.model import Credentials, OAuth2Credentials
from backend.integrations.credential_lease import CredentialLease
from backend.integrations.credential_refresh import (
    REFRESH_CLEANUP_SECONDS,
    REFRESH_LEASE_SECONDS,
    REFRESH_PENDING,
    REFRESH_REVISION,
    CredentialRefreshRequired,
    admitted_refresh,
)
from backend.integrations.credentials_store import (
    IntegrationCredentialsStore,
    provider_matches,
)
from backend.integrations.creds_events import publish_creds_changed
from backend.integrations.oauth import (
    CREDENTIALS_BY_PROVIDER,
    DEVICE_HANDLERS_BY_NAME,
    HANDLERS_BY_NAME,
)
from backend.integrations.providers import ProviderName
from backend.util.exceptions import MissingConfigError
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.integrations.oauth.base import BaseOAuthHandler
    from backend.integrations.oauth.device_base import BaseDeviceAuthHandler

logger = logging.getLogger(__name__)
settings = Settings()
_REFRESH_LOCK_WAIT_SECONDS = 5.0


_on_creds_changed: Callable[[str, str], None] | None = None


def register_creds_changed_hook(hook: Callable[[str, str], None]) -> None:
    """Register a callback invoked after any credential is created/updated/deleted.

    The callback receives ``(user_id, provider)`` and should be idempotent.
    Only one hook can be registered at a time.  Intended to be called once at
    application startup (e.g. by the copilot module) without creating an
    import cycle.

    In-process only.  Cross-process invalidation rides on
    :func:`~backend.integrations.creds_events.publish_creds_changed`, which
    :func:`_invoke_creds_changed_hook` calls alongside this hook.

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


async def _invoke_creds_changed_hook(user_id: str, provider: str) -> None:
    """Notify this process's hook, then every other process over Redis.

    The hook alone is not enough: it evicts only the writing process's caches,
    and the copilot executor holds its own in a different process.
    """
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
    await publish_creds_changed(user_id, provider)


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
        await _invoke_creds_changed_hook(user_id, credentials.provider)
        return result

    async def exists(self, user_id: str, credentials_id: str) -> bool:
        return (await self.store.get_creds_by_id(user_id, credentials_id)) is not None

    async def get(
        self, user_id: str, credentials_id: str, lock: bool = True
    ) -> Credentials | None:
        credentials = await self.store.get_creds_by_id(user_id, credentials_id)
        if not credentials:
            return None

        if (
            credentials.type == "oauth2"
            and credentials.refresh_strategy == "provider_runtime"
        ):
            logger.debug(
                "Skipping generic refresh for provider-managed credentials #%s",
                credentials.id,
            )
        elif credentials.type == "oauth2" and credentials.access_token_expires_at:
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
        # Refresh before taking the long-lived lease lock. A rotating OAuth
        # provider refreshes under that same credential lock; calling get()
        # after acquiring it would make acquire() wait on its own lock.
        refreshed = await self.get(user_id, credentials_id)
        if not refreshed:
            raise ValueError(
                f"Credentials #{credentials_id} for user #{user_id} not found"
            )

        # Use a low-priority (!time_sensitive) locking queue on top of the
        # general lock to allow priority access for refreshing/updating the
        # tokens.
        async with self._locked(user_id, credentials_id, "!time_sensitive"):
            lock = await self._acquire_lock(user_id, credentials_id)
        try:
            # Reload under the lease lock. Another writer may have replaced or
            # deleted the row between the pre-refresh and this acquisition;
            # the lease must expose the stored winner, never the stale snapshot.
            credentials = await self.store.get_creds_by_id(user_id, credentials_id)
            if not credentials:
                raise ValueError(
                    f"Credentials #{credentials_id} for user #{user_id} not found"
                )
            if credentials.metadata.get(REFRESH_PENDING):
                raise CredentialRefreshRequired()
        except BaseException:
            await self._release_owned_lock(lock)
            raise
        return credentials, lock

    async def acquire_lease(self, user_id: str, credentials_id: str) -> CredentialLease:
        credentials, lock = await self.acquire(user_id, credentials_id)
        checkpoint = partial(self.update_acquired, user_id)
        delete = partial(self.delete_acquired, user_id)
        lease = CredentialLease(credentials, lock, checkpoint, delete)
        lease.start_heartbeat()
        return lease

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
            if (
                credential.type != "oauth2"
                or credential.refresh_strategy == "provider_runtime"
                or not credential.access_token_expires_at
            ):
                # Credential doesn't expire
                return credential

            # Credential is OAuth2 credential and has expiration timestamp
            return await self.refresh_if_needed(user_id, credential)

        return get_credentials

    async def refresh_idle_credentials(self) -> None:
        """Maintain unused refresh grants after 14 days of access-token expiry."""
        refresh_before = int(time.time()) - 14 * 24 * 60 * 60
        candidates = await self.store.db_manager.get_near_expiry_oauth_credentials(
            refresh_before
        )

        for user_id, credentials in candidates:
            try:
                await self.refresh_if_needed(user_id, credentials)
            except Exception as error:
                logger.warning(
                    f"Idle refresh failed for credential #{credentials.id} "
                    f"belonging to user {user_id[:12]} ({type(error).__name__})"
                )

    async def refresh_if_needed(
        self, user_id: str, credentials: OAuth2Credentials, lock: bool = True
    ) -> OAuth2Credentials:
        if credentials.refresh_strategy == "provider_runtime":
            return credentials

        # Copilot skips refresh locks for reusable tokens. Rotating tokens must
        # serialize refreshes and reload storage, even for a lock-free caller.
        handler = await self._get_oauth_handler(credentials)
        if lock:
            return await self._refresh_locked(user_id, credentials, handler=handler)
        # Lock-free callers pay for the lock only when a refresh will happen.
        if not handler.needs_refresh(credentials):
            return credentials
        if handler.ROTATES_REFRESH_TOKEN:
            # A provider that rotates its refresh token cannot take the
            # unlocked path: "last writer wins" is safe when the refresh token
            # is static, but two workers replaying the same rotated token can
            # have the grant revoked by a provider that treats reuse as
            # compromise, and the loser can persist a token already
            # invalidated upstream.
            return await self._refresh_locked(
                user_id,
                credentials,
                handler=handler,
                lock_timeout=_REFRESH_LOCK_WAIT_SECONDS,
            )
        return await self._refresh_unlocked(user_id, credentials, handler=handler)

    async def _get_oauth_handler(
        self, credentials: OAuth2Credentials
    ) -> "BaseOAuthHandler | BaseDeviceAuthHandler":
        """Resolve the appropriate OAuth handler for the given credentials."""
        if provider_matches(credentials.provider, ProviderName.MCP.value):
            return create_mcp_oauth_handler(credentials)

        # Try device handlers first (they don't need client_id/secret lookup)
        # `provider` is typed `str` but carries a ProviderName at runtime; prefer the
        # enum value so the registry lookup matches either form.
        provider_key = getattr(credentials.provider, "value", None) or str(
            credentials.provider
        )
        if provider_key in DEVICE_HANDLERS_BY_NAME:
            handler_class = DEVICE_HANDLERS_BY_NAME[provider_key]
            return handler_class()

        return await _get_provider_oauth_handler(credentials.provider)

    async def _refresh_locked(
        self,
        user_id: str,
        credentials: OAuth2Credentials,
        handler: "BaseOAuthHandler | BaseDeviceAuthHandler | None" = None,
        *,
        lock_timeout: float | None = None,
    ) -> OAuth2Credentials:
        oauth_handler = handler or await self._get_oauth_handler(credentials)
        if oauth_handler.ROTATES_REFRESH_TOKEN is True:
            return await self._refresh_rotating(
                user_id, credentials, oauth_handler, lock_timeout=lock_timeout
            )
        # Bound acquisition only: cancelling a refresh can lose a rotated token.
        async with self._locked(
            user_id, credentials.id, "refresh", blocking_timeout=lock_timeout
        ):
            credentials = await self._reload_oauth_credentials(user_id, credentials)
            if credentials.refresh_strategy == "provider_runtime":
                return credentials
            oauth_handler = handler or await self._get_oauth_handler(credentials)
            if oauth_handler.needs_refresh(credentials):
                logger.debug(
                    "Refreshing '%s' credentials #%s",
                    credentials.provider,
                    credentials.id,
                )
                # Wait until the credentials are no longer in use anywhere
                _lock = await self._acquire_lock(
                    user_id, credentials.id, blocking_timeout=lock_timeout
                )
                try:
                    credentials = await self._reload_oauth_credentials(
                        user_id, credentials
                    )
                    if (
                        credentials.refresh_strategy == "provider_runtime"
                        or not oauth_handler.needs_refresh(credentials)
                    ):
                        return credentials
                    fresh_credentials = await oauth_handler.refresh_tokens(credentials)
                    await self.store.update_creds(user_id, fresh_credentials)
                    await _invoke_creds_changed_hook(
                        user_id, fresh_credentials.provider
                    )
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

    async def _reload_oauth_credentials(
        self, user_id: str, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        """Re-read after either lock wait; a lease holder may replace or delete it."""
        current = await self.store.get_creds_by_id(user_id, credentials.id)
        if current is None:
            raise ValueError(
                f"Credentials #{credentials.id} for user #{user_id} not found"
            )
        if current.type != "oauth2":
            raise TypeError(
                f"Credentials #{credentials.id} for user #{user_id} changed "
                f"type to {current.type!r} during OAuth refresh"
            )
        if current.provider != credentials.provider:
            raise RuntimeError("Credential provider changed during refresh")
        return current

    async def _fresh_refresh_lock(
        self, key: tuple[str, ...], *, blocking_timeout: float | None = None
    ) -> AsyncRedisLock:
        mutex = await self.locks()
        lock = mutex.redis.lock(str(key), REFRESH_LEASE_SECONDS, thread_local=False)
        if not await lock.acquire(blocking_timeout=blocking_timeout):
            raise TimeoutError("Timed out waiting for a Redis lock")
        return lock

    async def _renew_refresh_locks(self, locks: list[AsyncRedisLock]) -> None:
        for lock in locks:
            if not await lock.extend(REFRESH_LEASE_SECONDS, replace_ttl=True):
                raise CredentialRefreshRequired()
        for lock in locks:
            if not await lock.owned():
                raise CredentialRefreshRequired()

    async def _refresh_rotating(
        self,
        user_id: str,
        credentials: OAuth2Credentials,
        handler: "BaseOAuthHandler | BaseDeviceAuthHandler",
        *,
        lock_timeout: float | None = None,
    ) -> OAuth2Credentials:
        locks: list[AsyncRedisLock] = []
        transferred = False
        try:
            locks.append(
                await self._fresh_refresh_lock(
                    self._credentials_lock_key(user_id, credentials.id, "refresh"),
                    blocking_timeout=lock_timeout,
                )
            )
            current = await self.store.get_creds_by_id(user_id, credentials.id)
            if current is not None and current.type != "oauth2":
                raise TypeError(
                    f"Credential changed type to {current.type!r} during OAuth refresh"
                )
            if current is None or current.provider != credentials.provider:
                raise CredentialRefreshRequired()
            if current.metadata.get(REFRESH_PENDING):
                raise CredentialRefreshRequired()
            if (
                current.refresh_strategy == "provider_runtime"
                or not handler.needs_refresh(current)
            ):
                return current
            locks.append(
                await self._fresh_refresh_lock(
                    self._credentials_lock_key(user_id, credentials.id),
                    blocking_timeout=lock_timeout,
                )
            )
            locks.append(
                await self._fresh_refresh_lock(
                    (f"user:{user_id}", "integrations"),
                    blocking_timeout=lock_timeout,
                )
            )
            persisted = await self.store._get_persisted_user_creds_unlocked(user_id)
            current = next(
                (item for item in persisted if item.id == credentials.id), None
            )
            if (
                current is None
                or current.type != "oauth2"
                or current.provider != credentials.provider
            ):
                raise CredentialRefreshRequired()
            if current.metadata.get(REFRESH_PENDING):
                raise CredentialRefreshRequired()
            if (
                current.refresh_strategy == "provider_runtime"
                or not handler.needs_refresh(current)
            ):
                return current
            await self._renew_refresh_locks(locks)

            def operation(deadline: float):
                nonlocal transferred
                transferred = True
                return self._exchange_rotating(
                    user_id, current, handler, locks, deadline
                )

            return await admitted_refresh(operation)
        finally:
            if not transferred:

                async def cleanup(deadline: float):
                    async with asyncio.timeout_at(deadline + REFRESH_CLEANUP_SECONDS):
                        for lock in reversed(locks):
                            await self._release_owned_lock(lock)

                await admitted_refresh(cleanup, cleanup=True)

    async def _exchange_rotating(
        self,
        user_id: str,
        credentials: OAuth2Credentials,
        handler: "BaseOAuthHandler | BaseDeviceAuthHandler",
        locks: list[AsyncRedisLock],
        deadline: float,
    ) -> OAuth2Credentials:
        loop = asyncio.get_running_loop()
        database_deadline = time.time() + max(0.0, deadline - loop.time())
        checkpoint: str | None = None
        exchange_started = False
        committed: OAuth2Credentials | None = None
        phase = "prepare"
        nonce = uuid4().hex
        try:
            async with asyncio.timeout_at(deadline):
                await self._renew_refresh_locks(locks)
                checkpoint = await self.store.begin_refresh(
                    user_id, credentials, nonce, database_deadline
                )
                if checkpoint is None:
                    raise CredentialRefreshRequired()
                await self._renew_refresh_locks(locks)
                if loop.time() >= deadline:
                    raise TimeoutError()
                phase = "exchange"
                exchange_started = True
                fresh = await handler.refresh_tokens(deepcopy(credentials))
                if (
                    type(fresh) is not type(credentials)
                    or fresh.id != credentials.id
                    or fresh.provider != credentials.provider
                    or not set(fresh.scopes).issuperset(credentials.scopes)
                    or fresh.metadata.get(REFRESH_PENDING)
                ):
                    raise CredentialRefreshRequired()
                phase = "checkpoint"
                fresh = fresh.model_copy(
                    update={
                        "metadata": {
                            **fresh.metadata,
                            REFRESH_REVISION: nonce + ":complete",
                        },
                    }
                )
                if loop.time() >= deadline:
                    raise TimeoutError()
                await self._renew_refresh_locks(locks)
                if not await self.store.complete_refresh(
                    user_id, fresh, checkpoint, database_deadline
                ):
                    raise CredentialRefreshRequired()
                committed = fresh
                phase = "notify"
                try:
                    await _invoke_creds_changed_hook(user_id, fresh.provider)
                except Exception as error:  # noqa: BLE001 - Already committed.
                    logger.warning(
                        "Credential refresh notification failed: %s",
                        type(error).__name__,
                    )
                return fresh
        except BaseException as error:  # noqa: BLE001 - Never replay uncertain tokens.
            if committed is not None:
                return committed
            logger.warning(
                "Credential refresh failed during %s: %s", phase, type(error).__name__
            )
            if not exchange_started and checkpoint is not None:
                try:
                    async with asyncio.timeout_at(deadline + REFRESH_CLEANUP_SECONDS):
                        await self.store.complete_refresh(
                            user_id,
                            credentials,
                            checkpoint,
                            database_deadline + REFRESH_CLEANUP_SECONDS,
                        )
                except BaseException:  # noqa: BLE001 - Rollback may be cancelled.
                    logger.warning(
                        "Credential refresh prepare rollback did not complete"
                    )
            raise CredentialRefreshRequired() from None
        finally:
            try:
                async with asyncio.timeout_at(deadline + REFRESH_CLEANUP_SECONDS):
                    for lock in reversed(locks):
                        await self._release_owned_lock(lock)
            except BaseException:  # noqa: BLE001 - Keep outcome if cleanup fails.
                logger.warning("Credential refresh lock cleanup deadline expired")

    async def _refresh_unlocked(
        self,
        user_id: str,
        credentials: OAuth2Credentials,
        handler: "BaseOAuthHandler | BaseDeviceAuthHandler | None" = None,
    ) -> OAuth2Credentials:
        """Best-effort token refresh without any Redis locking.

        Concurrent refreshes are tolerated only for reusable refresh tokens;
        rotating providers must use the locked path and reload stored credentials.
        """
        oauth_handler = handler or await self._get_oauth_handler(credentials)
        if oauth_handler.needs_refresh(credentials):
            logger.debug(
                "Refreshing '%s' credentials #%s (lock-free)",
                credentials.provider,
                credentials.id,
            )
            fresh_credentials = await oauth_handler.refresh_tokens(credentials)
            await self.store.update_creds(user_id, fresh_credentials)
            await _invoke_creds_changed_hook(user_id, fresh_credentials.provider)
            credentials = fresh_credentials
        return credentials

    async def update(
        self,
        user_id: str,
        updated: Credentials,
        *,
        reconnect_expected: OAuth2Credentials | None = None,
    ) -> None:
        async with self._locked(user_id, updated.id):
            if reconnect_expected is None:
                await self.store.update_creds(user_id, updated)
            else:
                await self.store.update_creds(
                    user_id,
                    updated,
                    reconnect_expected=reconnect_expected,
                )
        # Notify listeners so the updated credential is picked up immediately.
        await _invoke_creds_changed_hook(user_id, updated.provider)

    async def upsert_single_provider(
        self,
        user_id: str,
        credentials: Credentials,
    ) -> Credentials:
        async with self.locked_provider_credentials(user_id, credentials.provider):
            return await self.upsert_single_provider_locked(user_id, credentials)

    @asynccontextmanager
    async def locked_provider_credentials(
        self,
        user_id: str,
        provider: str,
    ) -> AsyncIterator[None]:
        matching = await self.store.get_creds_by_provider(user_id, provider)
        locks: list[AsyncRedisLock] = []
        try:
            for credential_id in sorted({credential.id for credential in matching}):
                locks.append(await self._acquire_lock(user_id, credential_id))
            yield
        finally:
            for lock in reversed(locks):
                await self._release_owned_lock(lock)

    async def upsert_single_provider_locked(
        self,
        user_id: str,
        credentials: Credentials,
    ) -> Credentials:
        stored = await self.store.upsert_single_provider_creds(user_id, credentials)
        await _invoke_creds_changed_hook(user_id, stored.provider)
        return stored

    async def update_acquired(
        self,
        user_id: str,
        updated: Credentials,
        lock: AsyncRedisLock,
    ) -> None:
        expected_lock_name = str(self._credentials_lock_key(user_id, updated.id))
        if (
            lock.name != expected_lock_name
            or not (await lock.locked())
            or not (await lock.owned())
        ):
            raise RuntimeError(
                f"Cannot update credentials #{updated.id} without its owned lock"
            )
        provider_runtime_checkpoint = (
            updated.type == "oauth2" and updated.refresh_strategy == "provider_runtime"
        )
        codex_http_migration = (
            updated.type == "oauth2"
            and updated.provider == "codex"
            and updated.refresh_strategy == "oauth_handler"
        )
        if not (provider_runtime_checkpoint or codex_http_migration):
            raise RuntimeError("Acquired updates require provider-runtime credentials")

        current = await self.store.get_creds_by_id(user_id, updated.id)
        if (
            current is None
            or current.type != "oauth2"
            or current.refresh_strategy != "provider_runtime"
            or current.id != updated.id
            or current.provider != updated.provider
            or type(current) is not type(updated)
        ):
            raise RuntimeError("Provider-runtime credential identity changed")
        await self.store.update_creds(user_id, updated)
        await _invoke_creds_changed_hook(user_id, updated.provider)

    async def delete_acquired(
        self,
        user_id: str,
        credentials: Credentials,
        lock: AsyncRedisLock,
    ) -> None:
        expected_lock_name = str(self._credentials_lock_key(user_id, credentials.id))
        if (
            lock.name != expected_lock_name
            or not (await lock.locked())
            or not (await lock.owned())
        ):
            raise RuntimeError(
                f"Cannot delete credentials #{credentials.id} without its owned lock"
            )

        current = await self.store.get_creds_by_id(user_id, credentials.id)
        if (
            current is None
            or current.id != credentials.id
            or current.provider != credentials.provider
            or type(current) is not type(credentials)
        ):
            raise RuntimeError("Credential identity changed before deletion")
        await self.store.delete_creds_by_id(user_id, credentials.id)
        await _invoke_creds_changed_hook(user_id, credentials.provider)

    async def delete(self, user_id: str, credentials_id: str) -> None:
        async with self._locked(user_id, credentials_id):
            # Read inside the lock to avoid TOCTOU — another coroutine could
            # delete the same credential between the read and the delete.
            creds = await self.store.get_creds_by_id(user_id, credentials_id)
            await self.store.delete_creds_by_id(user_id, credentials_id)
        if creds:
            await _invoke_creds_changed_hook(user_id, creds.provider)

    async def delete_pending_refresh(
        self, user_id: str, credentials_id: str, provider: str
    ) -> bool:
        async with self._locked(user_id, credentials_id):
            current = await self.store.get_creds_by_id(user_id, credentials_id)
            if (
                current is None
                or current.provider != provider
                or not current.metadata.get(REFRESH_PENDING)
            ):
                return False
            await self.store.delete_creds_by_id(user_id, credentials_id)
        await _invoke_creds_changed_hook(user_id, provider)
        return True

    # -- Locking utilities -- #

    async def _acquire_lock(
        self,
        user_id: str,
        credentials_id: str,
        *args: str,
        blocking_timeout: float | None = None,
    ) -> AsyncRedisLock:
        key = self._credentials_lock_key(user_id, credentials_id, *args)
        locks = await self.locks()
        return await locks.acquire(key, blocking_timeout=blocking_timeout)

    @staticmethod
    def _credentials_lock_key(
        user_id: str, credentials_id: str, *args: str
    ) -> tuple[str, ...]:
        return (
            f"user:{user_id}",
            f"credentials:{credentials_id}",
            *args,
        )

    async def _release_owned_lock(self, lock: AsyncRedisLock) -> None:
        if (await lock.locked()) and (await lock.owned()):
            try:
                await lock.release()
            except Exception:
                logger.warning("Failed to release credentials lock", exc_info=True)

    @asynccontextmanager
    async def _locked(
        self,
        user_id: str,
        credentials_id: str,
        *args: str,
        blocking_timeout: float | None = None,
    ):
        lock = await self._acquire_lock(
            user_id, credentials_id, *args, blocking_timeout=blocking_timeout
        )
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
        revoke_url=meta.get("mcp_revoke_url"),
        resource_url=meta.get("mcp_resource_url"),
        token_endpoint_auth_method=meta.get("mcp_token_endpoint_auth_method"),
    )
