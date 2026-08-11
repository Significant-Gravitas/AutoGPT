"""Tests for the Slack OAuth v2 install flow."""

import time
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

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
        assert oauth._verify_state(state) == ""  # anonymous install
        assert oauth._verify_state(state + "x") is None
        assert oauth._verify_state("garbage") is None


def test_state_carries_the_installing_user():
    with patch(f"{_O}.config.get_client_secret", return_value="csecret"):
        state = oauth._make_state("user-123")
        assert oauth._verify_state(state) == "user-123"


def test_state_survives_a_user_id_containing_dots():
    # The user id is the last field, so parsing has to stop after the two
    # fixed ones or a dotted id silently loses its tail.
    with patch(f"{_O}.config.get_client_secret", return_value="csecret"):
        state = oauth._make_state("user.with.dots")
        assert oauth._verify_state(state) == "user.with.dots"


def test_expired_state_rejected():
    with patch(f"{_O}.config.get_client_secret", return_value="csecret"):
        old_ts = int(time.time()) - oauth._STATE_TTL_SECONDS - 5
        payload = f"nonce.{old_ts}.user-123"
        stale = f"{payload}.{oauth._sign(payload)}"
        assert oauth._verify_state(stale) is None


def test_user_param_roundtrip_tamper_and_expiry():
    with patch(f"{_O}.config.get_client_secret", return_value="csecret"):
        param = oauth.make_install_user_param("user-123")
        assert oauth._verify_user_param(param) == "user-123"
        assert oauth._verify_user_param(param + "x") == ""
        assert oauth._verify_user_param("") == ""
        old_ts = int(time.time()) - oauth._USER_PARAM_TTL_SECONDS - 5
        payload = f"user-123.{old_ts}"
        stale = f"{payload}.{oauth._sign(payload)}"
        assert oauth._verify_user_param(stale) == ""


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
        settings.return_value.config.frontend_base_url = "https://f.example"
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
    # Success hands off to our own page, which opens the bot DM and returns
    # the tab to settings — never a dead-end browser tab.
    assert resp.status_code == 302
    assert resp.headers["location"] == (
        "https://f.example/link/slack/installed?team=T1&app=A1&bot=UBOT"
    )


@pytest.mark.asyncio
async def test_callback_with_user_state_records_pending_install():
    cid, secret = _creds()
    with (
        cid,
        secret,
        patch(f"{_O}.AsyncWebClient") as web_client,
        patch(f"{_O}.upsert_bot_install", new=AsyncMock()),
        patch(f"{_O}.record_guild_joined", new=AsyncMock()),
        patch(f"{_O}.mark_pending", new=AsyncMock()) as pending,
        patch(f"{_O}.Settings") as settings,
    ):
        settings.return_value.config.frontend_base_url = "https://f.example"
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
            _req({"state": oauth._make_state("user-123"), "code": "auth-code"})
        )
    pending.assert_awaited_once()
    user_id, install = pending.await_args.args
    assert user_id == "user-123"
    assert install.team_id == "T1"
    assert install.team_name == "Acme"
    assert install.app_id == "A1"
    # Pin the target: without Settings patched both branches return 302 and
    # the assertion passes either way.
    assert resp.status_code == 302
    assert resp.headers["location"] == (
        "https://f.example/link/slack/installed?team=T1&app=A1&bot=UBOT"
    )


@pytest.mark.asyncio
async def test_callback_without_app_id_falls_back_to_settings_page():
    cid, secret = _creds()
    with (
        cid,
        secret,
        patch(f"{_O}.AsyncWebClient") as web_client,
        patch(f"{_O}.upsert_bot_install", new=AsyncMock()),
        patch(f"{_O}.record_guild_joined", new=AsyncMock()),
        patch(f"{_O}.Settings") as settings,
    ):
        settings.return_value.config.frontend_base_url = "https://f.example"
        web_client.return_value.oauth_v2_access = AsyncMock(
            return_value={
                "ok": True,
                "access_token": "xoxb-workspace",
                "team": {"id": "T1", "name": "Acme"},
                "bot_user_id": "UBOT",
            }
        )
        resp = await oauth._handle_callback(
            _req({"state": oauth._make_state(), "code": "auth-code"})
        )
    assert resp.status_code == 302
    assert (
        resp.headers["location"] == "https://f.example/settings/bots?slack_installed=1"
    )


@pytest.mark.asyncio
async def test_install_folds_a_verified_user_param_into_the_state():
    cid, secret = _creds()
    with cid, secret, patch(f"{_O}.Settings") as settings:
        settings.return_value.config.platform_base_url = "https://b.example"
        param = oauth.make_install_user_param("user-123")
        resp = await oauth._handle_install(_req({"u": param}))
    location = resp.headers["location"]
    state = parse_qs(urlparse(location).query)["state"][0]
    with patch(f"{_O}.config.get_client_secret", return_value="csecret"):
        assert oauth._verify_state(state) == "user-123"


@pytest.mark.asyncio
async def test_install_ignores_a_forged_user_param():
    cid, secret = _creds()
    with cid, secret, patch(f"{_O}.Settings") as settings:
        settings.return_value.config.platform_base_url = "https://b.example"
        resp = await oauth._handle_install(_req({"u": "user-666.123.badsig"}))
    location = resp.headers["location"]
    state = parse_qs(urlparse(location).query)["state"][0]
    with patch(f"{_O}.config.get_client_secret", return_value="csecret"):
        assert oauth._verify_state(state) == ""


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


@pytest.mark.asyncio
async def test_callback_falls_back_to_slack_redirect_without_a_frontend_url():
    # Self-hosted deployments may run the API without our frontend; the Slack
    # redirect still gets the user to the bot rather than a blank page.
    cid, secret = _creds()
    with (
        cid,
        secret,
        patch(f"{_O}.AsyncWebClient") as web_client,
        patch(f"{_O}.upsert_bot_install", new=AsyncMock()),
        patch(f"{_O}.record_guild_joined", new=AsyncMock()),
        patch(f"{_O}.Settings") as settings,
    ):
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
    assert resp.status_code == 302
    assert resp.headers["location"] == "https://slack.com/app_redirect?app=A1&team=T1"
