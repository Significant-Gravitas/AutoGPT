"""Discord-specific configuration."""

from backend.util.settings import Settings


def get_bot_token() -> str:
    return Settings().secrets.autopilot_bot_discord_token


def get_client_id() -> str:
    return Settings().secrets.autopilot_bot_discord_client_id


def get_invite_permissions() -> str:
    return (
        Settings().secrets.autopilot_bot_discord_permissions
        or DEFAULT_INVITE_PERMISSIONS
    )


# Discord message content limit (hard platform cap)
MAX_MESSAGE_LENGTH = 2000

# Flush the streaming buffer at 1900 — leaves 100-char headroom under the
# 2000 cap so the boundary-splitter has room to reach a natural break point.
CHUNK_FLUSH_AT = 1900

# Discord's hard per-attachment cap for non-Nitro bot uploads. Bots can't have
# Nitro and Discord shrank this from 50 MB to 25 MB years ago, so it's
# hardcoded rather than configurable. Over-cap artifacts fall back to a
# link-to-chat button.
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024

# Sensible default Discord permissions for the "Add to server" invite URL —
# covers send messages, embed links, attach files, read history, add
# reactions, use external emoji, slash commands, public threads, and sending
# in threads. Override per-deployment via AUTOPILOT_BOT_DISCORD_PERMISSIONS.
DEFAULT_INVITE_PERMISSIONS = "377957124928"
