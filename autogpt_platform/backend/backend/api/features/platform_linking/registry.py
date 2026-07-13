"""Per-platform metadata for the Bots settings page.

API-facing source of truth for which chat-bot platforms exist, which are
enabled on this deployment, and how to build their "Add bot to your server"
invite URL. New platforms slot in by adding a row + their config check below.
"""

from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict

from backend.copilot.bot.adapters.discord import config as discord_config
from backend.copilot.bot.adapters.slack import config as slack_config


class PlatformMeta(BaseModel):
    """Static + runtime metadata for one chat-bot platform."""

    model_config = ConfigDict(frozen=True)

    platform: str  # canonical key, matches PlatformLinkInfo.platform (uppercase)
    display_name: str
    icon: str  # filename under /public/integrations/<icon>
    enabled: bool
    add_bot_url: str | None  # null when the platform has no invite URL we can build


def enabled_platforms() -> list[PlatformMeta]:
    """Return metadata for every platform currently enabled on this deployment.

    Platforms whose adapter isn't configured (missing credentials) are
    omitted entirely so the Bots page hides them.
    """
    all_platforms = [_discord_meta(), _slack_meta()]
    return [platform for platform in all_platforms if platform.enabled]


def _discord_meta() -> PlatformMeta:
    enabled = bool(discord_config.get_bot_token())
    return PlatformMeta(
        platform="DISCORD",
        display_name="Discord",
        icon="discord.png",
        enabled=enabled,
        add_bot_url=_discord_invite_url() if enabled else None,
    )


def _slack_meta() -> PlatformMeta:
    # Slack has no Discord-style one-click invite for a self-hosted app — it's
    # installed per-workspace via the app manifest, then linked with /setup — so
    # there's no add_bot_url. Enabled once both the token and signing secret are
    # set (the same gate the webhook adapter mounts on).
    enabled = bool(slack_config.get_bot_token() and slack_config.get_signing_secret())
    return PlatformMeta(
        platform="SLACK",
        display_name="Slack",
        icon="slack.png",
        enabled=enabled,
        add_bot_url=None,
    )


def _discord_invite_url() -> str | None:
    client_id = discord_config.get_client_id()
    if not client_id:
        return None
    params = urlencode(
        {
            "client_id": client_id,
            "scope": "bot applications.commands",
            "permissions": discord_config.get_invite_permissions(),
        }
    )
    return f"https://discord.com/oauth2/authorize?{params}"
