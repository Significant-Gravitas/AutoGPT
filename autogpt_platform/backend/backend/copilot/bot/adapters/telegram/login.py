"""Verification of Telegram login payloads (login_url buttons / Login Widget).

When a user taps a ``login_url`` button in the bot chat, Telegram appends a
signed identity (id, name, username, auth_date, hash) to the opened URL. The
hash is HMAC-SHA256 over the sorted key=value lines, keyed with
SHA256(bot_token) — verifying it proves Telegram itself vouched for the
user. Docs: https://core.telegram.org/widgets/login#checking-authorization
"""

import hashlib
import hmac
import time
from typing import Mapping, Optional

# Telegram-signed payloads expire — a day matches Telegram's own guidance.
MAX_AUTH_AGE_SECONDS = 24 * 60 * 60

# Every key Telegram may include in the signed payload; anything else in the
# query string (e.g. our own params) must not enter the data-check string.
_SIGNED_KEYS = ("auth_date", "first_name", "id", "last_name", "photo_url", "username")


def verify_login(params: Mapping[str, str], bot_token: str) -> Optional[str]:
    """Return the verified Telegram user id, or None if the payload doesn't
    check out (bad hash, stale auth_date, missing fields)."""
    received_hash = params.get("hash", "")
    auth_date = params.get("auth_date", "")
    user_id = params.get("id", "")
    if not received_hash or not auth_date.isdigit() or not user_id or not bot_token:
        return None
    if time.time() - int(auth_date) > MAX_AUTH_AGE_SECONDS:
        return None
    # Telegram signs the key=value lines in alphabetical order — sort here so
    # verification can't silently break if _SIGNED_KEYS is ever reordered.
    data_check_string = "\n".join(
        f"{key}={params[key]}" for key in sorted(_SIGNED_KEYS) if key in params
    )
    secret_key = hashlib.sha256(bot_token.encode()).digest()
    expected = hmac.new(
        secret_key, data_check_string.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, received_hash):
        return None
    return user_id
