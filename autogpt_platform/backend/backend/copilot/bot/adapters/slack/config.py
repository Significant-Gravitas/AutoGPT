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

# Cap on a single inbound/outbound file. Slack allows far larger, but the bot
# only shuttles chat artifacts; over-cap outbound artifacts fall back to a
# link-to-chat button, and over-cap inbound files are skipped with a note.
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024

# Slack threads have no name (they're implicit off a parent message), so this
# is unused — create_thread encodes channel|ts and rename_thread is a no-op.
# Present only to satisfy the adapter contract.
MAX_THREAD_NAME_LENGTH = 100

# Slack bot apps expose no typing indicator, so start_typing is a no-op and
# this cadence is unused; present only to satisfy the adapter contract.
TYPING_REFRESH_SECONDS = 5.0
