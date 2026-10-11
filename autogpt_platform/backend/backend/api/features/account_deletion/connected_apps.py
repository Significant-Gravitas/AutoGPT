"""Disconnect every app a user connected, as part of closing their account."""

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


async def revoke_connected_apps(user_id: str) -> None:
    """Delete each credential the user connected and revoke it at its provider.

    Uses the same delete-then-revoke path as the "disconnect" button, with
    `force=True` so webhooks still in use by an agent are unlinked too.
    AutoGPT-managed, system and SDK-default credentials are not the user's
    own, so they are skipped here; managed ones are deprovisioned afterwards
    by `cleanup_user_managed_credentials`.

    One credential failing is logged and does not stop the rest, and the
    managed cleanup runs even when every revoke failed.
    """
    credentials = await creds_manager.store.get_all_creds(user_id)
    for creds in credentials:
        if _is_user_connected(creds):
            await _revoke_one(user_id, creds)
    await cleanup_user_managed_credentials(user_id)


def _is_user_connected(creds: Credentials) -> bool:
    return not (
        creds.is_managed or is_system_credential(creds.id) or is_sdk_default(creds.id)
    )


async def _revoke_one(user_id: str, creds: Credentials) -> None:
    provider = ProviderName(canonical_provider(creds.provider))
    try:
        await delete_and_revoke_credentials(user_id, creds, provider, force=True)
    except Exception as e:
        # Only the id, provider and error type: a provider error message can
        # echo the token it was given.
        logger.error(
            f"Account deletion could not revoke credential #{creds.id} "
            f"(provider {provider.value}): {type(e).__name__}"
        )
