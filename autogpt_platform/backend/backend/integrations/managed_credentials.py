"""Generic infrastructure for system-provided, per-user managed credentials.

Managed credentials are provisioned automatically by the platform (e.g. an
AgentMail pod-scoped API key) and stored alongside regular user credentials
with ``is_managed=True``.  Users cannot update or delete them.

New integrations register a :class:`ManagedCredentialProvider` at import time;
the two entry-points consumed by the rest of the application are:

* :func:`ensure_managed_credentials` – fired as a background task from the
  credential-listing endpoints (non-blocking).
* :func:`cleanup_managed_credentials` – called during account deletion to
  revoke external resources (API keys, pods, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from cachetools import TTLCache

if TYPE_CHECKING:
    from backend.data.model import Credentials
    from backend.integrations.credentials_store import IntegrationCredentialsStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------


class ManagedCredentialProvider(ABC):
    """Base class for integrations that auto-provision per-user credentials."""

    provider_name: str
    """Must match the ``provider`` field on the resulting credential."""

    auto_provision: bool = True
    """Whether :func:`ensure_managed_credentials` should provision this on
    credential-list load.

    Default ``True`` matches the AgentMail contract: cheap provisioning that
    is safe to run for every user on first visit.  Set to ``False`` when
    provisioning has per-user upstream cost (e.g. Ayrshare's profile quota);
    such providers still register here so account-deletion cleanup works,
    but only run via an explicit :func:`ensure_managed_credential` call
    from a user-triggered endpoint.
    """

    @abstractmethod
    async def is_available(self) -> bool:
        """Return ``True`` when the org-level configuration is present.

        Used by :func:`ensure_managed_credentials` to skip providers whose
        config is missing (e.g. missing env vars).  Independent of
        :attr:`auto_provision` — a provider can be available yet opt out
        of the startup sweep.
        """

    @abstractmethod
    async def provision(
        self, user_id: str, store: IntegrationCredentialsStore
    ) -> Credentials:
        """Create external resources and return a credential.

        The returned credential **must** have ``is_managed=True``.  The
        caller-supplied *store* is the same instance the framework will use
        for :meth:`post_provision` and the credential upsert; subclasses
        should thread it through when they need to read per-user state
        (e.g. Ayrshare's legacy migration read).
        """

    @abstractmethod
    async def deprovision(self, user_id: str, credential: Credentials) -> None:
        """Revoke external resources during account deletion."""

    async def post_provision(
        self,
        user_id: str,
        store: IntegrationCredentialsStore,
        credential: Credentials,
    ) -> None:
        """Optional cleanup hook run *after* the credential is durably stored.

        Runs inside the provision lock, immediately after
        ``add_managed_credential`` returns.  Subclasses can safely mutate
        other per-user state (e.g. clear a legacy migration field) knowing
        the new managed credential is already durable.

        **Must be idempotent and retry-safe.** The framework swallows any
        exception raised here and only logs a warning — the managed
        credential is already persisted, so subsequent provision calls
        short-circuit on ``has_managed_credential`` and this hook never
        runs again for that credential.  If a subclass needs the hook to
        retry on failure, it must drive that retry explicitly (e.g. a
        scheduled job), not rely on the provision path.

        Default: no-op.
        """
        _ = user_id, store, credential


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, ManagedCredentialProvider] = {}

# Users whose managed credentials have already been verified recently.
# Avoids redundant DB checks on every GET /credentials call.
# maxsize caps memory; TTL re-checks periodically (e.g. when new providers
# are added).  ~100K entries ≈ 4-8 MB.
_provisioned_users: TTLCache[str, bool] = TTLCache(maxsize=100_000, ttl=3600)


def register_managed_provider(provider: ManagedCredentialProvider) -> None:
    _PROVIDERS[provider.provider_name] = provider


def get_managed_provider(name: str) -> ManagedCredentialProvider | None:
    return _PROVIDERS.get(name)


def get_managed_providers() -> dict[str, ManagedCredentialProvider]:
    return dict(_PROVIDERS)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


async def _ensure_one(
    user_id: str,
    store: IntegrationCredentialsStore,
    name: str,
    provider: ManagedCredentialProvider,
) -> bool:
    """Provision a single managed credential under a distributed Redis lock.

    Returns ``True`` if the credential already exists or was successfully
    provisioned, ``False`` on transient failure so the caller knows not to
    cache the user as fully provisioned.
    """
    try:
        if not provider.auto_provision:
            # Registered for cleanup lookup, but opts out of the sweep.
            # Callers use `ensure_managed_credential(...)` directly.
            return True
        if not await provider.is_available():
            return True
        return await _provision_under_lock(user_id, store, name, provider)
    except Exception:
        logger.warning(
            "Failed to provision managed credential for provider=%s user=%s",
            name,
            user_id,
            exc_info=True,
        )
        return False


async def _provision_under_lock(
    user_id: str,
    store: IntegrationCredentialsStore,
    name: str,
    provider: ManagedCredentialProvider,
) -> bool:
    """Provision a credential under a distributed Redis lock (double-check).

    Separated from :func:`_ensure_one` so on-demand callers can invoke it
    via :func:`ensure_managed_credential` without re-entering the
    ``is_available`` gate — that gate is what the ``ensure_managed_credentials``
    sweep uses to skip opt-out providers.
    """
    # Use a distributed Redis lock so the check-then-provision operation
    # is atomic across all workers, preventing duplicate external
    # resource provisioning (e.g. AgentMail API keys).
    locks = await store.locks()
    key = (f"user:{user_id}", f"managed-provision:{name}")
    async with locks.locked(key):
        # Re-check under lock to avoid duplicate provisioning.
        if await store.has_managed_credential(user_id, name):
            return True
        credential = await provider.provision(user_id, store)
        await store.add_managed_credential(user_id, credential)
        # Run the post-provision cleanup hook only after the managed
        # credential is durably stored.  If it raises, the managed
        # credential is still in place and future provision calls
        # short-circuit on has_managed_credential — no duplicate
        # upstream resource, no data loss on migration paths.
        try:
            await provider.post_provision(user_id, store, credential)
        except Exception:
            logger.warning(
                "post_provision hook failed for provider=%s user=%s; "
                "managed credential is persisted so retry is safe",
                name,
                user_id,
                exc_info=True,
            )
        logger.info(
            "Provisioned managed credential for provider=%s user=%s",
            name,
            user_id,
        )
        return True


async def ensure_managed_credential(
    user_id: str,
    store: IntegrationCredentialsStore,
    provider: ManagedCredentialProvider,
) -> bool:
    """Provision *provider*'s managed credential for *user_id* on demand.

    Bypasses the provider's ``is_available()`` gate — callers are expected to
    have validated org-level config themselves (e.g. the Ayrshare SSO-URL
    endpoint checks its secrets before invoking this).  Use for providers
    that opt out of the ``ensure_managed_credentials`` startup sweep because
    provisioning has per-user cost or quota implications.

    Returns ``True`` on success, ``False`` on transient failure.
    """
    try:
        return await _provision_under_lock(
            user_id, store, provider.provider_name, provider
        )
    except Exception:
        logger.warning(
            "Failed to provision managed credential for provider=%s user=%s",
            provider.provider_name,
            user_id,
            exc_info=True,
        )
        return False


async def ensure_managed_credentials(
    user_id: str,
    store: IntegrationCredentialsStore,
) -> None:
    """Provision missing managed credentials for *user_id*.

    Fired as a non-blocking background task from the credential-listing
    endpoints.  Failures are logged but never propagated — the user simply
    will not see the managed credential until the next page load.

    Skips entirely if this user has already been checked during the current
    process lifetime (in-memory cache).  Resets on restart — just a
    performance optimisation, not a correctness guarantee.

    Providers are checked concurrently via ``asyncio.gather``.
    """
    if user_id in _provisioned_users:
        return

    results = await asyncio.gather(
        *(_ensure_one(user_id, store, n, p) for n, p in _PROVIDERS.items())
    )

    # Only cache the user as provisioned when every provider succeeded or
    # was already present.  A transient failure (network timeout, Redis
    # blip) returns False, so the next page load will retry.
    if all(results):
        _provisioned_users[user_id] = True


async def cleanup_managed_credentials(
    user_id: str,
    store: IntegrationCredentialsStore,
) -> None:
    """Revoke all external managed resources for a user being deleted."""
    all_creds = await store.get_all_creds(user_id)
    managed = [c for c in all_creds if c.is_managed]
    for cred in managed:
        provider = _PROVIDERS.get(cred.provider)
        if not provider:
            logger.warning(
                "No managed provider registered for %s — skipping cleanup",
                cred.provider,
            )
            continue
        try:
            await provider.deprovision(user_id, cred)
            logger.info(
                "Deprovisioned managed credential for provider=%s user=%s",
                cred.provider,
                user_id,
            )
        except Exception:
            logger.error(
                "Failed to deprovision %s for user %s",
                cred.provider,
                user_id,
                exc_info=True,
            )
