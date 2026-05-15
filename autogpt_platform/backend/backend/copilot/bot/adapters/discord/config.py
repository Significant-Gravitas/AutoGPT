"""Discord-specific configuration."""

from backend.util.settings import Settings


def get_bot_token() -> str:
    return Settings().secrets.autopilot_bot_discord_token


# Discord message content limit (hard platform cap)
MAX_MESSAGE_LENGTH = 2000

# Flush the streaming buffer at 1900 — leaves 100-char headroom under the
# 2000 cap so the boundary-splitter has room to reach a natural break point.
CHUNK_FLUSH_AT = 1900
