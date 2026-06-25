"""Tests for proactive-output authorization + channel resolution."""

from unittest.mock import AsyncMock

import pytest

from backend.copilot.bot import outbound
from backend.copilot.bot.adapters.base import ChannelInfo, PostedRef


def _api(server_ids: list[str]) -> AsyncMock:
    api = AsyncMock()
    api.list_linked_server_ids.return_value = server_ids
    return api


def _adapter(
    *,
    channels: list[ChannelInfo] | None = None,
    channel_server: str | None = None,
    posted: PostedRef | None = None,
    thread: PostedRef | None = None,
) -> AsyncMock:
    adapter = AsyncMock()
    adapter.list_text_channels.return_value = channels or []
    adapter.get_channel_server_id.return_value = channel_server
    adapter.post_channel_message.return_value = posted
    adapter.create_channel_thread.return_value = thread
    return adapter


@pytest.mark.asyncio
async def test_deliver_message_resolves_name_and_posts():
    adapter = _adapter(
        channels=[ChannelInfo(id="42", name="announcements", server_id="g1")],
        posted=PostedRef(id="100", url="https://discord.com/x"),
    )
    result = await outbound.deliver_message(
        adapter, _api(["g1"]), "discord", "user-1", "#announcements", "hi"
    )
    assert result.ok is True
    assert result.kind == "message"
    assert result.channel_id == "42"
    assert result.ref_id == "100"
    assert result.url == "https://discord.com/x"
    adapter.post_channel_message.assert_awaited_once_with("42", "hi")


@pytest.mark.asyncio
async def test_deliver_message_by_authorized_id():
    adapter = _adapter(channel_server="g1", posted=PostedRef(id="100"))
    result = await outbound.deliver_message(
        adapter, _api(["g1", "g2"]), "discord", "user-1", "999888777666555444", "hi"
    )
    assert result.ok is True
    assert result.channel_id == "999888777666555444"
    # ID path must not enumerate channels.
    adapter.list_text_channels.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_message_id_in_unlinked_server_is_rejected():
    adapter = _adapter(channel_server="other-guild", posted=PostedRef(id="100"))
    result = await outbound.deliver_message(
        adapter, _api(["g1"]), "discord", "user-1", "999888777666555444", "hi"
    )
    assert result.ok is False
    assert result.error == "not_authorized"
    adapter.post_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_message_unknown_id_is_not_found():
    adapter = _adapter(channel_server=None)
    result = await outbound.deliver_message(
        adapter, _api(["g1"]), "discord", "user-1", "999888777666555444", "hi"
    )
    assert result.ok is False
    assert result.error == "channel_not_found"


@pytest.mark.asyncio
async def test_deliver_message_name_not_found():
    adapter = _adapter(channels=[ChannelInfo(id="42", name="general", server_id="g1")])
    result = await outbound.deliver_message(
        adapter, _api(["g1"]), "discord", "user-1", "#announcements", "hi"
    )
    assert result.ok is False
    assert result.error == "channel_not_found"


@pytest.mark.asyncio
async def test_deliver_message_ambiguous_name():
    adapter = _adapter(
        channels=[
            ChannelInfo(id="42", name="general", server_id="g1"),
            ChannelInfo(id="43", name="general", server_id="g2"),
        ]
    )
    result = await outbound.deliver_message(
        adapter, _api(["g1", "g2"]), "discord", "user-1", "general", "hi"
    )
    assert result.ok is False
    assert result.error == "ambiguous_channel"


@pytest.mark.asyncio
async def test_no_linked_servers_short_circuits():
    adapter = _adapter()
    result = await outbound.deliver_message(
        adapter, _api([]), "discord", "user-1", "#announcements", "hi"
    )
    assert result.ok is False
    assert result.error == "no_linked_servers"
    adapter.list_text_channels.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_message_send_failure():
    adapter = _adapter(
        channels=[ChannelInfo(id="42", name="announcements", server_id="g1")],
        posted=None,
    )
    result = await outbound.deliver_message(
        adapter, _api(["g1"]), "discord", "user-1", "#announcements", "hi"
    )
    assert result.ok is False
    assert result.error == "send_failed"
    assert result.channel_id == "42"


@pytest.mark.asyncio
async def test_deliver_message_empty_content_is_distinct_error():
    adapter = _adapter()
    result = await outbound.deliver_message(
        adapter, _api(["g1"]), "discord", "user-1", "#x", "   "
    )
    assert result.ok is False
    assert result.error == "empty_content"
    # Nothing should be resolved or sent for empty content.
    adapter.list_text_channels.assert_not_awaited()
    adapter.post_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_thread_empty_content_is_distinct_error():
    adapter = _adapter()
    result = await outbound.create_thread(
        adapter, _api(["g1"]), "discord", "user-1", "#x", "Monday", ""
    )
    assert result.ok is False
    assert result.error == "empty_content"
    adapter.create_channel_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_thread_happy_path():
    adapter = _adapter(
        channels=[ChannelInfo(id="42", name="announcements", server_id="g1")],
        thread=PostedRef(id="t1", url="https://discord.com/t"),
    )
    result = await outbound.create_thread(
        adapter, _api(["g1"]), "discord", "user-1", "#announcements", "Monday", "body"
    )
    assert result.ok is True
    assert result.kind == "thread"
    assert result.ref_id == "t1"
    adapter.create_channel_thread.assert_awaited_once_with("42", "Monday", "body")


@pytest.mark.asyncio
async def test_create_thread_failure():
    adapter = _adapter(
        channels=[ChannelInfo(id="42", name="announcements", server_id="g1")],
        thread=None,
    )
    result = await outbound.create_thread(
        adapter, _api(["g1"]), "discord", "user-1", "#announcements", "Monday", "body"
    )
    assert result.ok is False
    assert result.error == "thread_failed"


@pytest.mark.asyncio
async def test_list_channels_empty_without_links():
    adapter = _adapter(channels=[ChannelInfo(id="42", name="x", server_id="g1")])
    result = await outbound.list_channels(adapter, _api([]), "discord", "user-1")
    assert result == []
    adapter.list_text_channels.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_channels_drops_unlinked_server_channels():
    # Defense-in-depth: even if the adapter over-returns, channels outside the
    # user's linked servers are filtered out of the picker.
    adapter = _adapter(
        channels=[
            ChannelInfo(id="10", name="ours", server_id="g1"),
            ChannelInfo(id="20", name="leaked", server_id="other"),
        ]
    )
    result = await outbound.list_channels(adapter, _api(["g1"]), "discord", "user-1")
    assert [c.id for c in result] == ["10"]
