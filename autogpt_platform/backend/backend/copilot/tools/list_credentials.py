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
from backend.integrations.managed_credentials import ensure_managed_credentials

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase
from .utils import get_user_credentials

logger = logging.getLogger(__name__)

# Mirrors the bound the integrations router puts on its first-time managed
# credential sweep, so a slow upstream can't hang the tool call.
_MANAGED_PROVISION_TIMEOUT_S = 10.0


async def _ensure_managed_credentials_bounded(user_id: str) -> None:
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
    except Exception:
        logger.exception(
            "Managed credential provisioning failed for user %s; listing without it",
            user_id,
        )


class CredentialListResponse(ToolResponseBase):
    """Response listing the user's connected credentials."""

    type: ResponseType = ResponseType.CREDENTIAL_LIST
    credentials: list[CredentialsMetaResponse] = []
    providers: list[str] = []
    count: int = 0


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

        await _ensure_managed_credentials_bounded(user_id)

        try:
            all_creds = await get_user_credentials(user_id)
        except Exception:
            logger.exception("Failed to list credentials for user %s", user_id)
            return ErrorResponse(
                message="Could not retrieve the user's connected credentials.",
                error="credential_lookup_failed",
                session_id=session_id,
            )

        # System credentials (platform-provided API keys) and SDK defaults are
        # not user-connected integrations, so they'd mislead the model here.
        metas = [
            to_meta_response(cred)
            for cred in all_creds
            if not is_sdk_default(cred.id) and cred.id not in SYSTEM_CREDENTIAL_IDS
        ]

        if provider:
            wanted = provider.strip().lower()
            metas = [m for m in metas if m.provider.lower() == wanted]

        providers = sorted({m.provider for m in metas})

        if metas:
            message = (
                f"The user has {len(metas)} connected credential(s) across "
                f"{len(providers)} provider(s): {', '.join(providers)}."
            )
        elif provider:
            message = (
                f"The user has no connected credentials for provider "
                f"'{provider}'. Use connect_integration to surface a "
                "sign-in card if this integration is needed."
            )
        else:
            message = (
                "The user has not connected any integrations yet. Use "
                "connect_integration to surface a sign-in card if one is needed."
            )

        return CredentialListResponse(
            message=message,
            credentials=metas,
            providers=providers,
            count=len(metas),
            session_id=session_id,
        )
