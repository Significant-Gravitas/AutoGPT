"""Telegram-specific configuration.

One bot (from @BotFather) serves every chat — Telegram has no per-workspace
install, so a single token + webhook secret is the whole credential story.

Bot setup checklist (once per bot, via @BotFather):
- ``/setprivacy`` -> Disable — group privacy mode ON (the default) means the
  bot never receives plain group messages, so @mentions and reply-to-bot go
  unseen. After changing it, remove + re-add the bot to existing groups.
- The command menu registers itself on startup (setMyCommands).

The webhook itself is registered once, out of band, with the secret Telegram
will echo back in the ``X-Telegram-Bot-Api-Secret-Token`` header:

    curl "https://api.telegram.org/bot<TOKEN>/setWebhook" \
      -d "url=<PLATFORM_BASE_URL>/api/copilot-webhooks/telegram/updates" \
      -d "secret_token=<WEBHOOK_SECRET>"       -d "allowed_updates=[\"message\",\"my_chat_member\"]"
"""

from backend.util.settings import Settings


def get_bot_token() -> str:
    return Settings().secrets.autopilot_bot_telegram_token


def get_webhook_secret() -> str:
    return Settings().secrets.autopilot_bot_telegram_webhook_secret


def get_bot_username() -> str:
    """The bot's public @username (without the @) — used for the t.me
    add-to-group link; mention detection resolves the authoritative username
    from getMe at runtime."""
    return Settings().secrets.autopilot_bot_telegram_username


# Telegram's hard cap is 4096 chars per message.
MAX_MESSAGE_LENGTH = 4096

# Flush at 3800 — leaves headroom under the cap for the boundary splitter to
# reach a natural break point.
CHUNK_FLUSH_AT = 3800

# Bots can only download files up to 20MB via getFile.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

# Forum topic names cap at 128 chars; also a sane cap for our title strings.
MAX_THREAD_NAME_LENGTH = 128

# sendChatAction's "typing" indicator lasts ~5s — refresh just under that.
TYPING_REFRESH_SECONDS = 4.0
