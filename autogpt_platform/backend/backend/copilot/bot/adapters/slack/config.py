"""Slack-specific configuration."""

from backend.util.settings import Settings


def get_bot_token() -> str:
    return Settings().secrets.autopilot_bot_slack_token


def get_signing_secret() -> str:
    return Settings().secrets.autopilot_bot_slack_signing_secret


# Slack's hard cap is 40000 chars per message; 4000 is the practical ceiling
# for readability.
MAX_MESSAGE_LENGTH = 4000

# Flush at 3800 — leaves 200-char headroom under the cap for the boundary
# splitter to reach a natural break point.
CHUNK_FLUSH_AT = 3800
