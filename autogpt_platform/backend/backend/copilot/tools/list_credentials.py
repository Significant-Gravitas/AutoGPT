"""List the user's connected integration credentials (metadata only, no secrets)."""

import asyncio
import logging
from typing import Any

from backend.api.features.integrations.router import (
    CredentialsMetaResponse,
    to_meta_response,
)
from backend.copilot.model import ChatSession
from backend.data.model import is_sdk_default
from backend.integrations.credentials_store import SYSTEM_CREDENTIAL_IDS
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.managed_credentials import (
    ensure_managed_credentials,
    get_managed_provider,
)

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase
from .utils import get_user_credentials

logger = logging.getLogger(__name__)

# Mirrors the bound the integrations router puts on its first-time managed
# credential sweep, so a slow upstream can't hang the tool call.
_MANAGED_PROVISION_TIMEOUT_S = 10.0


class CredentialListResponse(ToolResponseBase):
    """Response listing the user's connected credentials."""

    type: ResponseType = ResponseType.CREDENTIAL_LIST
    credentials: list[CredentialsMetaResponse] = []
    providers: list[str] = []
    count: int = 0
    # False when the managed-credential provisioning sweep timed out or
    # failed, meaning platform-managed integrations may be missing below.
    provisioning_complete: bool = True


class ListUserCredentialsTool(BaseTool):
    """Lists the integrations the user has already connected (never secrets)."""

    @property
    def name(self) -> str:
        return "list_user_credentials"

    @property
    def description(self) -> str:
        return (
            "List the integrations the user has already connected (metadata "
            "only, never secrets). Call before asking the user to sign in or "
            "connect an integration, so you only surface sign-in for "
            "integrations that are genuinely missing."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": (
                        "Optional provider slug to filter by "
                        "(e.g. 'github', 'google', 'notion')."
                    ),
                },
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        provider: str | None = None,
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id

        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="missing_user_id",
                session_id=session_id,
            )

        # The managed-credential sweep only affects managed providers, and it
        # sits on the hot path (the model calls this before every sign-in
        # prompt). Skip it when filtering to a provider that is never managed.
        wanted_provider = provider.strip().lower() if provider else ""
        if wanted_provider and get_managed_provider(wanted_provider) is None:
            provisioning_complete = True
        else:
            provisioning_complete = await _ensure_managed_credentials_bounded(user_id)

        try:
            all_creds = await get_user_credentials(user_id)
        except Exception:
            logger.exception("Failed to list credentials for user %s", user_id)
            return ErrorResponse(
                message="Could not retrieve the user's connected credentials.",
                error="credential_lookup_failed",
                session_id=session_id,
            )

        metas, wanted = _serialize_connected_credentials(all_creds, provider)
        providers = sorted({m.provider for m in metas})

        return CredentialListResponse(
            message=_build_inventory_message(
                metas, providers, wanted, provisioning_complete
            ),
            credentials=metas,
            providers=providers,
            count=len(metas),
            provisioning_complete=provisioning_complete,
            session_id=session_id,
        )


def _serialize_connected_credentials(
    all_creds: list[Any], provider: str | None
) -> tuple[list[CredentialsMetaResponse], str]:
    """Strip secrets and drop non-user credentials, then apply the provider filter."""
    wanted = provider.strip().lower() if provider else ""

    # System credentials (platform-provided API keys) and SDK defaults are
    # not user-connected integrations, so they'd mislead the model here. Filter
    # on the raw credentials (including the provider filter) before serializing,
    # so to_meta_response only runs on the retained set.
    metas = [
        to_meta_response(cred)
        for cred in all_creds
        if not is_sdk_default(cred.id)
        and cred.id not in SYSTEM_CREDENTIAL_IDS
        and (not wanted or cred.provider.lower() == wanted)
    ]

    return metas, wanted


def _build_inventory_message(
    metas: list[CredentialsMetaResponse],
    providers: list[str],
    wanted: str,
    provisioning_complete: bool,
) -> str:
    """Compose the model-facing summary of the connected-credential inventory."""
    if metas:
        message = (
            f"The user has {len(metas)} connected credential(s) across "
            f"{len(providers)} provider(s): {', '.join(providers)}."
        )
    elif wanted:
        message = (
            f"The user has no connected credentials for provider "
            f"'{wanted}'. Use connect_integration to surface a "
            "sign-in card if this integration is needed."
        )
    else:
        message = (
            "The user has not connected any integrations yet. Use "
            "connect_integration to surface a sign-in card if one is needed."
        )

    if not provisioning_complete:
        message += (
            " Note: platform-managed credential provisioning did not "
            "complete, so managed integrations may be missing from this "
            "list — do not treat their absence as authoritative."
        )

    return message


async def _ensure_managed_credentials_bounded(user_id: str) -> bool:
    """Run the managed-credential sweep; return False on timeout or failure."""
    try:
        await asyncio.wait_for(
            ensure_managed_credentials(user_id, IntegrationCredentialsManager().store),
            timeout=_MANAGED_PROVISION_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Managed credential sweep exceeded %.1fs for user=%s; "
            "listing without it",
            _MANAGED_PROVISION_TIMEOUT_S,
            user_id,
        )
        return False
    except Exception:
        logger.exception(
            "Managed credential provisioning failed for user %s; listing without it",
            user_id,
        )
        return False
    return True
