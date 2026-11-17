"""Revoke what AutoGPT issued on a user's behalf, as part of closing their account."""

import logging

from backend.data.auth.api_key import revoke_user_api_keys
from backend.data.auth.oauth import revoke_user_oauth_tokens

logger = logging.getLogger(__name__)


async def revoke_issued_access(user_id: str) -> None:
    """Revoke the user's AutoGPT API keys and every OAuth grant AutoGPT's own
    OAuth provider issued for them (access tokens, refresh tokens and unused
    authorization codes, for every third-party app).

    The external API checks keys and access tokens against the database on
    every request, with no cache in between, so they stop working as soon as
    this returns. Both steps are idempotent.

    OAuth apps the user registered as a developer are not touched here.

    Errors propagate: the account must not close while a key still works.
    """
    await revoke_user_api_keys(user_id)
    await revoke_user_oauth_tokens(user_id)
    logger.info(f"Account deletion revoked the API keys and OAuth grants of #{user_id}")
