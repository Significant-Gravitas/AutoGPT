"""Disconnect every app a user connected, as part of closing their account."""

import asyncio
import logging

from backend.api.features.integrations.router import (
    creds_manager,
    delete_and_revoke_credentials,
)
from backend.data.model import Credentials, is_sdk_default
from backend.data.user import cleanup_user_managed_credentials
from backend.integrations.credentials_store import (
    canonical_provider,
    is_system_credential,
)
from backend.integrations.providers import ProviderName

logger = logging.getLogger(__name__)

# Per credential: the provider calls have no timeout of their own, and the
# deletion request must answer before the frontend proxy gives up on it.
REVOKE_TIMEOUT_SECONDS = 10.0
MAX_CONCURRENT_REVOKES = 10


async def revoke_connected_apps(user_id: str) -> None:
    """Delete each credential the user connected and revoke it at its provider.

    Uses the same delete-then-revoke path as the "disconnect" button, with
    `force=True` so webhooks still in use by an agent are unlinked too, and a
    webhook the provider refuses to deregister does not keep the credential.
    AutoGPT-managed, system and SDK-default credentials are not the user's
    own, so they are skipped here; managed ones are deprovisioned afterwards
    by `cleanup_user_managed_credentials`.

    The credentials are handled together (at most `MAX_CONCURRENT_REVOKES`
    at once), each bounded by
    `REVOKE_TIMEOUT_SECONDS`. One failing or timing out is logged and does not
    stop the rest: a retry would meet the same refusal, and the erasure
    (SECRT-2854) deletes whatever is still stored. Revoking at the provider is
    best effort the same way, and some providers have no revoke endpoint at
    all. An error listing the credentials or from the managed cleanup
    propagates, so the account stays open.
    """
    credentials = await creds_manager.store.get_all_creds(user_id)
    # At most a few provider calls at once: someone with many connected apps
    # shouldn't burst the HTTP pool or trip a provider's rate limit.
    slots = asyncio.Semaphore(MAX_CONCURRENT_REVOKES)

    async def revoke(creds: Credentials) -> None:
        async with slots:
            await _revoke_one(user_id, creds)

    await asyncio.gather(*(revoke(c) for c in credentials if _is_user_connected(c)))
    await cleanup_user_managed_credentials(user_id)


def _is_user_connected(creds: Credentials) -> bool:
    return not (
        creds.is_managed or is_system_credential(creds.id) or is_sdk_default(creds.id)
    )


async def _revoke_one(user_id: str, creds: Credentials) -> None:
    provider = ProviderName(canonical_provider(creds.provider))
    try:
        revoked = await asyncio.wait_for(
            delete_and_revoke_credentials(
                user_id, creds, provider, force=True, skip_failed_prunes=True
            ),
            timeout=REVOKE_TIMEOUT_SECONDS,
        )
    except Exception as e:
        # Only the id, provider and error type: a provider error message can
        # echo the token it was given.
        logger.error(
            f"Account deletion could not revoke credential #{creds.id} "
            f"(provider {provider.value}): {type(e).__name__}"
        )
        return
    # None means there was nothing to revoke (an API key); False means the
    # provider refused or has no revoke endpoint, so the grant may be live.
    if revoked is False:
        logger.warning(
            f"Account deletion deleted credential #{creds.id} but provider "
            f"{provider.value} did not revoke it"
        )
