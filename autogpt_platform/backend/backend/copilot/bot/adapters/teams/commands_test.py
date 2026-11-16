"""Tests for Teams /setup: only a team owner may link a team, and the link
goes to that owner in a private chat, never into the channel."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.util.exceptions import LinkAlreadyExistsError

from . import commands
from .graph import TeamsGraphError

_SERVICE_URL = "https://smba.trafficmanager.net/teams/"
_TEAM_ID = "19:team@thread.tacv2"
_CHANNEL = "19:room@thread.tacv2"
_GROUP_ID = "6f1c2a3b-0000-4000-8000-000000000001"
_OWNER = {
    "id": "29:owner",
    "name": "Ada",
    "aadObjectId": "0b6c3c1e-0000-4000-8000-000000000002",
}
_LINK = "https://link.example/t"
_DM = "a:dm-with-ada"


def _activity(*, team: dict | None = None, sender: dict | None = None) -> dict:
    return {
        "type": "message",
        "id": "activity-1",
        "serviceUrl": _SERVICE_URL,
        "text": "/setup",
        "from": sender or _OWNER,
        "conversation": {"id": _CHANNEL, "conversationType": "channel"},
        "channelData": {
            "team": (
                team
                if team is not None
                else {"id": _TEAM_ID, "name": "Eng", "aadGroupId": _GROUP_ID}
            )
        },
    }


def _adapter(
    *, owner: bool | Exception = True, private_chat: str | None = _DM
) -> MagicMock:
    adapter = MagicMock()
    adapter.send_message = AsyncMock()
    adapter.send_link = AsyncMock()
    adapter.open_dm_channel = AsyncMock(return_value=private_chat)
    adapter.client.get_team_details = AsyncMock(return_value={})
    if isinstance(owner, Exception):
        adapter.graph.is_team_owner = AsyncMock(side_effect=owner)
    else:
        adapter.graph.is_team_owner = AsyncMock(return_value=owner)
    return adapter


def _api() -> MagicMock:
    api = MagicMock()
    api.create_link_token = AsyncMock(return_value=MagicMock(link_url=_LINK))
    return api


def _channel_text(adapter: MagicMock) -> str:
    conversation_id, text = adapter.send_message.await_args.args
    assert conversation_id == _CHANNEL
    return text


@pytest.mark.asyncio
async def test_setup_by_a_team_owner_sends_the_link_to_a_private_chat():
    api, adapter = _api(), _adapter()
    await commands.handle(api, adapter, _activity(), "setup")

    adapter.graph.is_team_owner.assert_awaited_once_with(
        _GROUP_ID, _OWNER["aadObjectId"]
    )
    kwargs = api.create_link_token.await_args.kwargs
    assert kwargs["platform"] == "teams"
    assert kwargs["platform_server_id"] == _TEAM_ID
    assert kwargs["platform_user_id"] == "29:owner"
    assert kwargs["server_name"] == "Eng"
    adapter.open_dm_channel.assert_awaited_once_with("29:owner")
    # The card goes to the owner's 1:1 chat only; the channel, which every
    # member of the team reads, gets a pointer without the link.
    adapter.send_link.assert_awaited_once()
    destination, _text, label, url = adapter.send_link.await_args.args
    assert (destination, label, url) == (_DM, "Link Team", _LINK)
    channel_text = _channel_text(adapter)
    assert "private message" in channel_text
    assert _LINK not in channel_text


@pytest.mark.asyncio
async def test_setup_by_a_member_is_refused_in_the_channel():
    api, adapter = _api(), _adapter(owner=False)
    await commands.handle(api, adapter, _activity(), "setup")

    api.create_link_token.assert_not_awaited()
    adapter.open_dm_channel.assert_not_awaited()
    adapter.send_link.assert_not_awaited()
    assert _channel_text(adapter).startswith("Only a team owner can link this team")


@pytest.mark.asyncio
async def test_setup_is_refused_when_graph_cannot_say_who_owns_the_team():
    api = _api()
    adapter = _adapter(owner=TeamsGraphError("team member lookup failed (403)"))
    await commands.handle(api, adapter, _activity(), "setup")

    api.create_link_token.assert_not_awaited()
    adapter.send_link.assert_not_awaited()
    text = _channel_text(adapter)
    assert "couldn't confirm that you're a team owner" in text
    assert "ask your Teams admin to update the AutoGPT app" in text


@pytest.mark.asyncio
async def test_setup_from_a_sender_without_an_entra_id_is_refused():
    api, adapter = _api(), _adapter()
    sender = {"id": "29:someone", "name": "Someone"}
    await commands.handle(api, adapter, _activity(sender=sender), "setup")

    adapter.graph.is_team_owner.assert_not_awaited()
    api.create_link_token.assert_not_awaited()
    assert "couldn't confirm" in _channel_text(adapter)


@pytest.mark.asyncio
async def test_setup_asks_the_connector_for_a_missing_group_id_and_name():
    # Teams stamps these onto install activities but not onto the message
    # carrying /setup: without the lookup the owner check can't run and the
    # link is stored nameless.
    api, adapter = _api(), _adapter()
    adapter.client.get_team_details = AsyncMock(
        return_value={"name": "Engineering", "aadGroupId": _GROUP_ID}
    )
    await commands.handle(api, adapter, _activity(team={"id": _TEAM_ID}), "setup")

    adapter.client.get_team_details.assert_awaited_once_with(_SERVICE_URL, _TEAM_ID)
    adapter.graph.is_team_owner.assert_awaited_once_with(
        _GROUP_ID, _OWNER["aadObjectId"]
    )
    assert api.create_link_token.await_args.kwargs["server_name"] == "Engineering"


@pytest.mark.asyncio
async def test_setup_does_not_call_the_connector_when_the_activity_names_the_team():
    api, adapter = _api(), _adapter()
    await commands.handle(api, adapter, _activity(), "setup")

    adapter.client.get_team_details.assert_not_awaited()
    assert api.create_link_token.await_args.kwargs["server_name"] == "Eng"


@pytest.mark.asyncio
async def test_setup_survives_a_failed_name_lookup():
    # A nameless link still works; the settings page falls back to the id.
    api, adapter = _api(), _adapter()
    adapter.client.get_team_details = AsyncMock(side_effect=RuntimeError("connector"))
    team = {"id": _TEAM_ID, "aadGroupId": _GROUP_ID}
    await commands.handle(api, adapter, _activity(team=team), "setup")

    assert api.create_link_token.await_args.kwargs["server_name"] == ""
    adapter.send_link.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failed_lookup_that_leaves_no_group_id_refuses():
    api, adapter = _api(), _adapter()
    adapter.client.get_team_details = AsyncMock(side_effect=RuntimeError("connector"))
    await commands.handle(api, adapter, _activity(team={"id": _TEAM_ID}), "setup")

    adapter.graph.is_team_owner.assert_not_awaited()
    api.create_link_token.assert_not_awaited()
    assert "couldn't confirm" in _channel_text(adapter)


@pytest.mark.asyncio
async def test_setup_mints_nothing_when_no_private_chat_can_be_opened():
    api, adapter = _api(), _adapter(private_chat=None)
    await commands.handle(api, adapter, _activity(), "setup")

    api.create_link_token.assert_not_awaited()
    adapter.send_link.assert_not_awaited()
    assert "Start a chat with me first" in _channel_text(adapter)


@pytest.mark.asyncio
async def test_a_failed_private_send_never_falls_back_to_the_channel():
    api, adapter = _api(), _adapter()
    adapter.send_link = AsyncMock(side_effect=RuntimeError("403 from Connector"))
    await commands.handle(api, adapter, _activity(), "setup")

    adapter.send_link.assert_awaited_once()
    assert adapter.send_link.await_args.args[0] == _DM
    channel_text = _channel_text(adapter)
    assert "couldn't send you the setup link in a private chat" in channel_text
    assert _LINK not in channel_text


@pytest.mark.asyncio
async def test_setup_of_an_already_linked_team_answers_in_the_channel():
    api, adapter = _api(), _adapter()
    api.create_link_token = AsyncMock(side_effect=LinkAlreadyExistsError("linked"))
    await commands.handle(api, adapter, _activity(), "setup")

    adapter.send_link.assert_not_awaited()
    assert "already linked" in _channel_text(adapter)


@pytest.mark.asyncio
async def test_help_names_who_can_link_a_team():
    adapter = _adapter()
    await commands.handle(MagicMock(), adapter, _activity(), "help")
    assert "team owners only" in _channel_text(adapter)
