"""Per-workspace OAuth install credentials for multi-tenant webhook bots.

Each installed workspace (Slack team, and later Teams etc.) gets one row holding
that workspace's bot token, ENCRYPTED at rest via ``JSONCryptor`` — the same
mechanism ``User.integrations`` uses — keyed by the platform's workspace/team ID.

The OAuth install callback writes these; the adapter reads them to call the
workspace's API with its own token. Accessed in-process by the webhook routes
mounted on the main backend API, so it talks to Prisma directly.

Privacy/security: the ``credentials`` column is always an encrypted blob — a raw
token is never persisted.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from prisma.models import BotInstall
from pydantic import BaseModel

from backend.platform_linking.models import Platform
from backend.util.encryption import JSONCryptor

logger = logging.getLogger(__name__)


class BotInstallCredentials(BaseModel):
    """Decrypted, ready-to-use credentials for one installed workspace."""

    team_id: str
    bot_token: str
    bot_user_id: Optional[str] = None
    app_id: Optional[str] = None
    name: Optional[str] = None


def _key(platform: Platform, team_id: str) -> dict:
    return {
        "platform_platformServerId": {
            "platform": platform.value,
            "platformServerId": team_id,
        }
    }


async def upsert_bot_install(
    *,
    platform: Platform,
    team_id: str,
    bot_token: str,
    bot_user_id: Optional[str] = None,
    app_id: Optional[str] = None,
    name: Optional[str] = None,
) -> None:
    """Store (or refresh) a workspace's encrypted bot token. Clears ``revokedAt``
    so a re-install after an uninstall resurrects the row."""
    fields = {
        "credentials": JSONCryptor().encrypt({"bot_token": bot_token}),
        "botUserId": bot_user_id,
        "appId": app_id,
        "name": name,
        "revokedAt": None,
    }
    await BotInstall.prisma().upsert(
        where=_key(platform, team_id),
        data={
            "create": {
                "platform": platform.value,
                "platformServerId": team_id,
                **fields,
            },
            "update": fields,
        },
    )


async def get_bot_install(
    platform: Platform, team_id: str
) -> Optional[BotInstallCredentials]:
    """Return decrypted credentials for a live install, or ``None`` if the
    workspace isn't installed, is revoked, or the token can't be decrypted."""
    row = await BotInstall.prisma().find_unique(where=_key(platform, team_id))
    if row is None or row.revokedAt is not None:
        return None
    token = JSONCryptor().decrypt(row.credentials).get("bot_token")
    if not token:
        logger.warning(
            "BotInstall for %s/%s has no decryptable token", platform.value, team_id
        )
        return None
    return BotInstallCredentials(
        team_id=team_id,
        bot_token=token,
        bot_user_id=row.botUserId,
        app_id=row.appId,
        name=row.name,
    )


async def is_install_revoked(platform: Platform, team_id: str) -> bool:
    """Whether this workspace explicitly uninstalled the app / revoked its
    token. Distinguishes "revoked" from "never installed" — a revoked
    workspace must NOT fall back to any other credential."""
    row = await BotInstall.prisma().find_unique(where=_key(platform, team_id))
    return row is not None and row.revokedAt is not None


async def revoke_bot_install(platform: Platform, team_id: str) -> None:
    """Mark a workspace's install revoked (app_uninstalled / tokens_revoked)."""
    await BotInstall.prisma().update_many(
        where={
            "platform": platform.value,
            "platformServerId": team_id,
            "revokedAt": None,
        },
        data={"revokedAt": datetime.now(timezone.utc)},
    )
