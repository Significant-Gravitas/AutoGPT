"""Tests for Slack slash-command handlers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.exceptions import LinkAlreadyExistsError

from . import commands
from .adapter import SlackAdapter

_SETUP = {"command": "/setup", "team_id": "T1", "user_id": "U1"}


def _body(resp) -> dict:
    return json.loads(bytes(resp.body).decode())


def _api() -> MagicMock:
    api = MagicMock()
    api.create_link_token = AsyncMock(
        return_value=MagicMock(link_url="https://example.com/link/abc")
    )
    return api


def _client_for(
    user: dict | None = None,
    *,
    error: Exception | None = None,
    installed: bool = True,
) -> AsyncMock:
    """The adapter's per-workspace client lookup, whose users.info answers
    with ``user`` (or raises ``error``)."""
    slack = MagicMock()
    slack.users_info = AsyncMock(side_effect=error, return_value={"user": user or {}})
    return AsyncMock(return_value=slack if installed else None)


def _admin(**flags) -> AsyncMock:
    return _client_for({"id": "U1", **(flags or {"is_admin": True})})


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["is_admin", "is_owner", "is_primary_owner"])
async def test_setup_by_a_workspace_admin_returns_a_link_only_they_see(flag):
    api = _api()
    client_for = _admin(**{flag: True})
    resp = await commands.handle(api, _SETUP, client_for)
    body = _body(resp)
    # Ephemeral: the link reaches the admin who ran /setup, never the channel.
    assert body["response_type"] == "ephemeral"
    assert body["blocks"][1]["elements"][0]["url"] == "https://example.com/link/abc"
    client_for.assert_awaited_once_with("T1")
    client_for.return_value.users_info.assert_awaited_once_with(user="U1")
    api.create_link_token.assert_awaited_once()
    assert api.create_link_token.await_args.kwargs["platform"] == "slack"
    assert api.create_link_token.await_args.kwargs["platform_user_id"] == "U1"


@pytest.mark.asyncio
async def test_setup_by_a_member_is_refused_without_a_link():
    api = _api()
    member = _client_for({"id": "U1", "is_admin": False, "is_owner": False})
    resp = await commands.handle(api, _SETUP, member)
    body = _body(resp)
    assert body["response_type"] == "ephemeral"
    assert "blocks" not in body
    assert body["text"].startswith(
        "Only a workspace admin or owner can link this workspace"
    )
    api.create_link_token.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_for",
    [
        _client_for(error=RuntimeError("users.info failed")),
        _client_for(installed=False),
    ],
    ids=["users-info-fails", "no-workspace-token"],
)
async def test_setup_is_refused_when_slack_cannot_say_who_ran_it(client_for):
    api = _api()
    resp = await commands.handle(api, _SETUP, client_for)
    body = _body(resp)
    assert "blocks" not in body
    assert "couldn't confirm that you're a workspace admin or owner" in body["text"]
    api.create_link_token.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_admin", [True, False])
async def test_the_command_route_asks_the_workspaces_own_client(is_admin):
    # The signed slash-command route hands /setup the adapter's per-workspace
    # client, so the role comes from the workspace the command came from.
    api = _api()
    adapter = SlackAdapter(api)
    slack = MagicMock()
    slack.users_info = AsyncMock(return_value={"user": {"is_admin": is_admin}})
    adapter._clients["T1"] = slack
    request = MagicMock()
    request.body = AsyncMock(return_value=b"command=%2Fsetup&team_id=T1&user_id=U1")
    request.headers = {}
    request.form = AsyncMock(return_value=_SETUP)
    with patch(
        "backend.copilot.bot.adapters.slack.adapter.signing.verify",
        return_value=True,
    ):
        resp = await adapter._handle_command_request(request)

    slack.users_info.assert_awaited_once_with(user="U1")
    assert ("blocks" in _body(resp)) is is_admin
    assert api.create_link_token.await_count == int(is_admin)


@pytest.mark.asyncio
async def test_setup_without_team_is_rejected():
    api = _api()
    client_for = _admin()
    resp = await commands.handle(
        api, {"command": "/setup", "user_id": "U1"}, client_for
    )
    assert "workspace/user info" in _body(resp)["text"]
    api.create_link_token.assert_not_awaited()
    client_for.assert_not_awaited()


@pytest.mark.asyncio
async def test_setup_already_linked_gets_friendly_message():
    api = _api()
    api.create_link_token = AsyncMock(side_effect=LinkAlreadyExistsError("linked"))
    resp = await commands.handle(api, _SETUP, _admin())
    assert "already linked" in _body(resp)["text"]


@pytest.mark.asyncio
async def test_setup_link_failure_is_graceful():
    api = _api()
    api.create_link_token = AsyncMock(side_effect=RuntimeError("boom"))
    resp = await commands.handle(api, _SETUP, _admin())
    assert "went wrong" in _body(resp)["text"]


@pytest.mark.asyncio
async def test_help_is_ephemeral():
    resp = await commands.handle(MagicMock(), {"command": "/help"}, _admin())
    assert _body(resp)["response_type"] == "ephemeral"
    assert "/setup" in _body(resp)["text"]
    assert "workspace admins and owners only" in _body(resp)["text"]


@pytest.mark.asyncio
async def test_unlink_points_at_settings():
    settings = MagicMock()
    settings.config.frontend_base_url = "https://app.example"
    settings.config.platform_base_url = ""
    with patch("backend.copilot.bot.command_core.Settings", return_value=settings):
        resp = await commands.handle(MagicMock(), {"command": "/unlink"}, _admin())
    assert (
        _body(resp)["blocks"][1]["elements"][0]["url"]
        == "https://app.example/settings/bots"
    )


@pytest.mark.asyncio
async def test_unlink_without_base_url_returns_error():
    settings = MagicMock()
    settings.config.frontend_base_url = ""
    settings.config.platform_base_url = ""
    with patch("backend.copilot.bot.command_core.Settings", return_value=settings):
        resp = await commands.handle(MagicMock(), {"command": "/unlink"}, _admin())
    assert "Settings → Bots" in _body(resp)["text"]


@pytest.mark.asyncio
async def test_unknown_command():
    resp = await commands.handle(MagicMock(), {"command": "/wat"}, _admin())
    assert "Unknown command" in _body(resp)["text"]
