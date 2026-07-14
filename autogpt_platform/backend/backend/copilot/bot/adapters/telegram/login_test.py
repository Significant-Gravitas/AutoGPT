"""Tests for Telegram login payload verification."""

import hashlib
import hmac
import time
from unittest.mock import patch

from .login import verify_login

_TOKEN = "123:test-bot-token"


def _signed(params: dict[str, str], token: str = _TOKEN) -> dict[str, str]:
    check = "\n".join(f"{k}={v}" for k, v in sorted(params.items()))
    secret = hashlib.sha256(token.encode()).digest()
    params["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return params


def _fresh_payload(**overrides) -> dict[str, str]:
    params = {
        "id": "424242",
        "first_name": "Bently",
        "username": "bently",
        "auth_date": str(int(time.time())),
    }
    params.update(overrides)
    return _signed(params)


def test_valid_payload_returns_the_telegram_user_id():
    assert verify_login(_fresh_payload(), _TOKEN) == "424242"


def test_tampered_payload_is_rejected():
    payload = _fresh_payload()
    payload["id"] = "999999"  # swap identity after signing
    assert verify_login(payload, _TOKEN) is None


def test_wrong_bot_token_is_rejected():
    assert verify_login(_fresh_payload(), "456:other-token") is None


def test_stale_auth_date_is_rejected():
    old = str(int(time.time()) - 2 * 24 * 60 * 60)
    assert verify_login(_fresh_payload(auth_date=old), _TOKEN) is None


def test_extra_unsigned_params_do_not_break_verification():
    payload = _fresh_payload()
    payload["utm_source"] = "telegram"  # our own query params ride along
    assert verify_login(payload, _TOKEN) == "424242"


def test_missing_hash_or_token_rejected():
    payload = _fresh_payload()
    payload.pop("hash")
    assert verify_login(payload, _TOKEN) is None
    assert verify_login(_fresh_payload(), "") is None


def test_verification_survives_signed_keys_reordering():
    # Telegram signs key=value lines alphabetically; verification must sort
    # rather than depend on _SIGNED_KEYS happening to be declared in order.
    scrambled = ("username", "id", "photo_url", "last_name", "first_name", "auth_date")
    with patch("backend.copilot.bot.adapters.telegram.login._SIGNED_KEYS", scrambled):
        assert verify_login(_fresh_payload(), _TOKEN) == "424242"
