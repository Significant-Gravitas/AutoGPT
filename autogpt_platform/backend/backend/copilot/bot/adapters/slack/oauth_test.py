"""Tests for the Slack OAuth v2 install flow."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot.adapters.slack import oauth

_O = "backend.copilot.bot.adapters.slack.oauth"


def _creds():
    return (
        patch(f"{_O}.config.get_client_id", return_value="cid"),
        patch(f"{_O}.config.get_client_secret", return_value="csecret"),
    )


def _req(params: dict) -> MagicMock:
    r = MagicMock()
    r.query_params = params
    return r


def test_is_enabled_requires_both_credentials():
    with (
        patch(f"{_O}.config.get_client_id", return_value="cid"),
        patch(f"{_O}.config.get_client_secret", return_value=""),
    ):
        assert oauth.is_enabled() is False
    cid, secret = _creds()
    with cid, secret:
        assert oauth.is_enabled() is True


def test_state_roundtrip_and_tamper_rejected():
    with patch(f"{_O}.config.get_client_secret", return_value="csecret"):
        state = oauth._make_state()
        assert oauth._verify_state(state) is True
        assert oauth._verify_state(state + "x") is False
        assert oauth._verify_state("garbage") is False


def test_expired_state_rejected():
    with patch(f"{_O}.config.get_client_secret", return_value="csecret"):
        old_ts = int(time.time()) - oauth._STATE_TTL_SECONDS - 5
        payload = f"nonce.{old_ts}"
        stale = f"{payload}.{oauth._sign(payload)}"
        assert oauth._verify_state(stale) is False


@pytest.mark.asyncio
async def test_callback_exchanges_code_and_stores_install():
    cid, secret = _creds()
    with (
        cid,
        secret,
        patch(f"{_O}.AsyncWebClient") as web_client,
        patch(f"{_O}.upsert_bot_install", new=AsyncMock()) as upsert,
        patch(f"{_O}.record_guild_joined", new=AsyncMock()) as roster,
        patch(f"{_O}.Settings") as settings,
    ):
        settings.return_value.config.platform_base_url = "https://b.example"
        settings.return_value.config.frontend_base_url = ""
        web_client.return_value.oauth_v2_access = AsyncMock(
            return_value={
                "ok": True,
                "access_token": "xoxb-workspace",
                "team": {"id": "T1", "name": "Acme"},
                "bot_user_id": "UBOT",
                "app_id": "A1",
            }
        )
        resp = await oauth._handle_callback(
            _req({"state": oauth._make_state(), "code": "auth-code"})
        )

    upsert.assert_awaited_once()
    kwargs = upsert.await_args.kwargs
    assert kwargs["team_id"] == "T1"
    assert kwargs["bot_token"] == "xoxb-workspace"
    assert kwargs["bot_user_id"] == "UBOT"
    roster.assert_awaited_once()
    assert resp.status_code == 200  # plain success page (no frontend URL configured)


@pytest.mark.asyncio
async def test_callback_fires_on_installed_so_stale_clients_evict():
    cid, secret = _creds()
    evicted: list[str] = []
    with (
        cid,
        secret,
        patch(f"{_O}.AsyncWebClient") as web_client,
        patch(f"{_O}.upsert_bot_install", new=AsyncMock()),
        patch(f"{_O}.record_guild_joined", new=AsyncMock()),
        patch(f"{_O}.Settings") as settings,
    ):
        settings.return_value.config.platform_base_url = "https://b.example"
        settings.return_value.config.frontend_base_url = ""
        web_client.return_value.oauth_v2_access = AsyncMock(
            return_value={
                "ok": True,
                "access_token": "xoxb-new",
                "team": {"id": "T1", "name": "Acme"},
                "bot_user_id": "UBOT",
                "app_id": "A1",
            }
        )
        await oauth._handle_callback(
            _req({"state": oauth._make_state(), "code": "c"}),
            on_installed=evicted.append,
        )
    assert evicted == ["T1"]


@pytest.mark.asyncio
async def test_callback_rejects_invalid_state():
    with patch(f"{_O}.config.get_client_secret", return_value="csecret"):
        resp = await oauth._handle_callback(_req({"state": "forged", "code": "c"}))
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_callback_handles_user_denied():
    with patch(f"{_O}.Settings") as settings:
        settings.return_value.config.frontend_base_url = ""
        resp = await oauth._handle_callback(_req({"error": "access_denied"}))
    assert resp.status_code == 400
