"""Per-platform metadata for the Bots settings page.

API-facing source of truth for which chat-bot platforms exist, which are
enabled on this deployment, and how to build their "Add bot to your server"
invite URL. New platforms slot in by adding a row + their config check below.
"""

from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict

from backend.copilot.bot.adapters.discord import config as discord_config
from backend.copilot.bot.adapters.slack import config as slack_config
from backend.copilot.bot.adapters.telegram import config as telegram_config
from backend.util.settings import Settings

# Backend route that starts the Slack "Add to Slack" OAuth install (kept in sync
# with slack.oauth.INSTALL_PATH; hardcoded here to avoid importing the heavier
# oauth module — which pulls in the Slack SDK + Prisma — into this metadata file).
_SLACK_INSTALL_PATH = "/api/copilot-webhooks/slack/install"

# The platform's own word for a server: Slack has workspaces, Telegram
# groups, Teams teams. Declared once so every surface (settings page, link
# confirmation, bot copy) reads natively instead of saying "server".
_SERVER_NOUNS = {
    "DISCORD": "server",
    "SLACK": "workspace",
    "TELEGRAM": "group",
}
_DEFAULT_SERVER_NOUN = "server"


class PlatformMeta(BaseModel):
    """Static + runtime metadata for one chat-bot platform."""

    model_config = ConfigDict(frozen=True)

    platform: str  # canonical key, matches PlatformLinkInfo.platform (uppercase)
    display_name: str
    icon: str  # filename under /public/integrations/<icon>
    server_noun: str  # see _SERVER_NOUNS
    enabled: bool
    add_bot_url: str | None  # null when the platform has no invite URL we can build


def enabled_platforms() -> list[PlatformMeta]:
    """Return metadata for every platform currently enabled on this deployment.

    Platforms whose adapter isn't configured (missing credentials) are
    omitted entirely so the Bots page hides them.
    """
    all_platforms = [_discord_meta(), _slack_meta(), _telegram_meta()]
    return [platform for platform in all_platforms if platform.enabled]


def server_noun_for(platform: str) -> str:
    """The platform's word for a server, for surfaces that only know its key.

    Reads the map rather than the metas: building those hits config getters
    and URL builders, which is a lot of work for a constant string. Unknown
    platforms get the safe generic wording.
    """
    return _SERVER_NOUNS.get(platform.upper(), _DEFAULT_SERVER_NOUN)


def _discord_meta() -> PlatformMeta:
    enabled = bool(discord_config.get_bot_token())
    return PlatformMeta(
        platform="DISCORD",
        display_name="Discord",
        icon="discord.png",
        server_noun=_SERVER_NOUNS["DISCORD"],
        enabled=enabled,
        add_bot_url=_discord_invite_url() if enabled else None,
    )


def _slack_meta() -> PlatformMeta:
    # Enabled once the signing secret plus credentials are set — the same gate
    # the webhook adapter mounts on. "Add to Slack" needs the OAuth app creds
    # (multi-workspace install); with only a static token there's no install URL
    # (single-workspace mode), so the button is hidden.
    oauth_ready = bool(
        slack_config.get_client_id() and slack_config.get_client_secret()
    )
    has_credentials = oauth_ready or bool(slack_config.get_bot_token())
    enabled = bool(slack_config.get_signing_secret() and has_credentials)
    return PlatformMeta(
        platform="SLACK",
        display_name="Slack",
        icon="slack.png",
        server_noun=_SERVER_NOUNS["SLACK"],
        enabled=enabled,
        add_bot_url=_slack_install_url() if (enabled and oauth_ready) else None,
    )


def _telegram_meta() -> PlatformMeta:
    # Enabled on the same gate the webhook adapter mounts on. The t.me
    # startgroup deep link opens Telegram's own "add to group" picker, so the
    # button only renders when the bot's public username is configured.
    enabled = bool(
        telegram_config.get_bot_token() and telegram_config.get_webhook_secret()
    )
    username = telegram_config.get_bot_username().lstrip("@")
    return PlatformMeta(
        platform="TELEGRAM",
        display_name="Telegram",
        icon="telegram.png",
        server_noun=_SERVER_NOUNS["TELEGRAM"],
        enabled=enabled,
        add_bot_url=(
            f"https://t.me/{username}?startgroup=true"
            if (enabled and username)
            else None
        ),
    )


def _slack_install_url() -> str | None:
    base = Settings().config.platform_base_url.rstrip("/")
    if not base:
        return None
    return f"{base}{_SLACK_INSTALL_PATH}"


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
