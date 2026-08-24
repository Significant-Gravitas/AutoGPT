"""Resolve the model credential used by partner-embedded chat sessions."""

import os

from fastapi import HTTPException, status

from backend.copilot.model import CopilotLlmAuthProvider
from backend.integrations.codex.credential_codec import (
    CodexAuthBundleError,
    bundle_from_credentials,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager

_CREDENTIAL_ENV = "PARTNER_EMBED_CODEX_CREDENTIAL_ID"


async def resolve_embed_llm_route(
    user_id: str,
) -> tuple[CopilotLlmAuthProvider, str | None]:
    """Use a user-owned Codex credential when the local PoC config selects one."""
    credential_id = os.getenv(_CREDENTIAL_ENV, "").strip()
    if not credential_id:
        return "platform", None

    credentials = await IntegrationCredentialsManager().get(user_id, credential_id)
    if (
        credentials is None
        or credentials.provider != "codex"
        or credentials.type != "oauth2"
    ):
        raise _credential_unavailable()
    try:
        bundle_from_credentials(credentials)
    except CodexAuthBundleError as error:
        raise _credential_unavailable() from error
    return "codex", credential_id


def _credential_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="partner_chat_credential_unavailable",
    )
