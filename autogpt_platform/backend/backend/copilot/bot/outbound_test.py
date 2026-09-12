"""Tests for proactive-output authorization + channel resolution."""

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot import outbound
from backend.copilot.bot.adapters.base import ChannelInfo, EditOutcome, PostedRef


class _FakeRedis:
    """Just enough Redis for the authorship record: get/set over a dict."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value


@pytest.fixture(autouse=True)
def sent_store():
    """Back `sent_messages` with an in-memory store for every test.

    Autouse because the edit path now refuses anything without an authorship
    record, so a real Redis client would otherwise be reached (and every edit
    test would fail on the record lookup rather than the check it is about).
    """
    fake = _FakeRedis()
    with patch("backend.copilot.bot.sent_messages.get_redis_async", return_value=fake):
        yield fake


def _seed_sent(
    store: _FakeRedis,
    platform: str,
    channel_id: str,
    ref_id: str,
    user_id: str,
    *,
    chunks: int = 1,
    editable: bool = True,
) -> None:
    """Pretend ``user_id`` already had the bot post ``ref_id`` there."""
    store.store[f"copilot-bot:sent:{platform}:{channel_id}:{ref_id}"] = json.dumps(
        {"user_id": user_id, "chunks": chunks, "editable": editable}
    )


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
    dm_channel: str | None = None,
    edit_outcome: EditOutcome = EditOutcome.OK,
) -> AsyncMock:
    adapter = AsyncMock()
    # Sync classifier — mirrors Discord's numeric-snowflake grammar so
    # _resolve_target routes IDs vs names the way the real adapter would.
    adapter.looks_like_channel_id = MagicMock(
        side_effect=lambda ref: bool(re.fullmatch(r"\d{15,21}", ref))
    )
    adapter.list_text_channels.return_value = channels or []
    adapter.get_channel_server_id.return_value = channel_server
    adapter.post_channel_message.return_value = posted
    adapter.create_channel_thread.return_value = thread
    adapter.open_dm_channel.return_value = dm_channel
    adapter.edit_channel_message.return_value = edit_outcome
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


def _dm_api(dm_user_id: str | None) -> AsyncMock:
    api = AsyncMock()
    api.get_dm_user_id.return_value = dm_user_id
    return api


@pytest.mark.asyncio
async def test_deliver_dm_happy_path():
    adapter = _adapter(dm_channel="dm-42", posted=PostedRef(id="100", url="https://x"))
    result = await outbound.deliver_dm(adapter, _dm_api("pu1"), "discord", "u1", "hi")
    assert result.ok is True
    assert result.kind == "dm"
    assert result.channel_id == "dm-42"
    assert result.ref_id == "100"
    adapter.open_dm_channel.assert_awaited_once_with("pu1")
    adapter.post_channel_message.assert_awaited_once_with("dm-42", "hi")


@pytest.mark.asyncio
async def test_deliver_dm_without_link_is_rejected():
    adapter = _adapter(dm_channel="dm-42", posted=PostedRef(id="100"))
    result = await outbound.deliver_dm(adapter, _dm_api(None), "discord", "u1", "hi")
    assert result.ok is False
    assert result.error == "no_dm_link"
    adapter.open_dm_channel.assert_not_awaited()
    adapter.post_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_dm_unopenable_channel():
    adapter = _adapter(dm_channel=None, posted=PostedRef(id="100"))
    result = await outbound.deliver_dm(adapter, _dm_api("pu1"), "discord", "u1", "hi")
    assert result.ok is False
    assert result.error == "dm_unavailable"
    adapter.post_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_dm_send_failure():
    adapter = _adapter(dm_channel="dm-42", posted=None)
    result = await outbound.deliver_dm(adapter, _dm_api("pu1"), "discord", "u1", "hi")
    assert result.ok is False
    assert result.error == "send_failed"
    assert result.channel_id == "dm-42"


@pytest.mark.asyncio
async def test_deliver_dm_empty_content_is_distinct_error():
    adapter = _adapter(dm_channel="dm-42")
    api = _dm_api("pu1")
    result = await outbound.deliver_dm(adapter, api, "discord", "u1", "  ")
    assert result.ok is False
    assert result.error == "empty_content"
    api.get_dm_user_id.assert_not_awaited()
    adapter.post_channel_message.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_edit_message_channel_happy_path(sent_store):
    adapter = _adapter(channel_server="g1")
    _seed_sent(sent_store, "discord", "42", "100", "user-1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "discord", "user-1", "channel", "42", "100", "updated"
    )
    assert result.ok is True
    assert result.error is None
    adapter.edit_channel_message.assert_awaited_once_with("42", "100", "updated")


@pytest.mark.asyncio
async def test_edit_message_channel_in_unlinked_server_is_rejected(sent_store):
    adapter = _adapter(channel_server="other-guild")
    _seed_sent(sent_store, "discord", "42", "100", "user-1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "discord", "user-1", "channel", "42", "100", "updated"
    )
    assert result.ok is False
    assert result.error == "not_authorized"
    adapter.edit_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_message_channel_unknown_id_is_not_found(sent_store):
    adapter = _adapter(channel_server=None)
    _seed_sent(sent_store, "discord", "42", "100", "user-1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "discord", "user-1", "channel", "42", "100", "updated"
    )
    assert result.ok is False
    assert result.error == "channel_not_found"


@pytest.mark.asyncio
async def test_edit_message_channel_no_linked_servers(sent_store):
    adapter = _adapter()
    _seed_sent(sent_store, "discord", "42", "100", "user-1")
    result = await outbound.edit_message(
        adapter, _api([]), "discord", "user-1", "channel", "42", "100", "updated"
    )
    assert result.ok is False
    assert result.error == "no_linked_servers"
    adapter.get_channel_server_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_message_dm_happy_path(sent_store):
    adapter = _adapter(dm_channel="dm-42")
    _seed_sent(sent_store, "discord", "dm-42", "100", "u1")
    result = await outbound.edit_message(
        adapter, _dm_api("pu1"), "discord", "u1", "dm", "dm-42", "100", "updated"
    )
    assert result.ok is True
    adapter.open_dm_channel.assert_awaited_once_with("pu1")
    adapter.edit_channel_message.assert_awaited_once_with("dm-42", "100", "updated")


@pytest.mark.asyncio
async def test_edit_message_dm_without_link_is_rejected(sent_store):
    adapter = _adapter(dm_channel="dm-42")
    _seed_sent(sent_store, "discord", "dm-42", "100", "u1")
    result = await outbound.edit_message(
        adapter, _dm_api(None), "discord", "u1", "dm", "dm-42", "100", "updated"
    )
    assert result.ok is False
    assert result.error == "no_dm_link"
    adapter.edit_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_message_dm_channel_id_mismatch_is_rejected(sent_store):
    # A caller-supplied channel_id that doesn't match this user's own DM
    # channel must never be trusted, even though it "looks like" a DM id —
    # this is the authorization check the caller-supplied id can't skip.
    adapter = _adapter(dm_channel="dm-42")
    # Seeded, so the authorship gate passes and the DM-channel check is the
    # one under test: the two gates are independent, not one behind the other.
    _seed_sent(sent_store, "discord", "someone-elses-dm", "100", "u1")
    result = await outbound.edit_message(
        adapter, _dm_api("pu1"), "discord", "u1", "dm", "someone-elses-dm", "100", "x"
    )
    assert result.ok is False
    assert result.error == "not_authorized"
    adapter.edit_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_message_empty_content_is_distinct_error():
    adapter = _adapter(channel_server="g1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "discord", "user-1", "channel", "42", "100", "   "
    )
    assert result.ok is False
    assert result.error == "empty_content"
    adapter.get_channel_server_id.assert_not_awaited()
    adapter.edit_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_message_unsupported_platform_is_surfaced(sent_store):
    adapter = _adapter(channel_server="g1", edit_outcome=EditOutcome.UNSUPPORTED)
    _seed_sent(sent_store, "teams", "42", "100", "user-1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "teams", "user-1", "channel", "42", "100", "updated"
    )
    assert result.ok is False
    assert result.error == "edit_unsupported"


@pytest.mark.asyncio
async def test_edit_message_not_found_is_surfaced(sent_store):
    adapter = _adapter(channel_server="g1", edit_outcome=EditOutcome.NOT_FOUND)
    _seed_sent(sent_store, "discord", "42", "100", "user-1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "discord", "user-1", "channel", "42", "100", "updated"
    )
    assert result.ok is False
    assert result.error == "message_not_found"


@pytest.mark.asyncio
async def test_edit_message_failed_is_surfaced(sent_store):
    adapter = _adapter(channel_server="g1", edit_outcome=EditOutcome.FAILED)
    _seed_sent(sent_store, "discord", "42", "100", "user-1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "discord", "user-1", "channel", "42", "100", "updated"
    )
    assert result.ok is False
    assert result.error == "edit_failed"


# -- Authorship: an edit needs proof *this* account had the bot post it --


@pytest.mark.asyncio
async def test_edit_message_without_authorship_record_is_refused():
    # The whole point: channel authorization alone used to be enough, and it
    # is shared by everyone linked to the server.
    adapter = _adapter(channel_server="g1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "discord", "user-1", "channel", "42", "100", "updated"
    )
    assert result.ok is False
    assert result.error == "not_sender"
    adapter.edit_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_message_by_another_linked_user_is_refused(sent_store):
    # Bob posted it; Alice is linked to the same guild, so every channel check
    # in this function passes for her. Only the record stops the rewrite.
    _seed_sent(sent_store, "discord", "42", "100", "bob")
    adapter = _adapter(channel_server="g1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "discord", "alice", "channel", "42", "100", "hijacked"
    )
    assert result.ok is False
    assert result.error == "not_sender"
    adapter.edit_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_message_dm_of_another_user_is_refused(sent_store):
    _seed_sent(sent_store, "discord", "dm-42", "100", "bob")
    adapter = _adapter(dm_channel="dm-42")
    result = await outbound.edit_message(
        adapter, _dm_api("pu1"), "discord", "alice", "dm", "dm-42", "100", "hijacked"
    )
    assert result.ok is False
    assert result.error == "not_sender"
    adapter.edit_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_message_chunked_post_is_refused(sent_store):
    # ref_id is the first chunk only, so editing would leave chunks 2..n stale
    # underneath a rewritten opening while reporting success.
    _seed_sent(sent_store, "discord", "42", "100", "user-1", chunks=3)
    adapter = _adapter(channel_server="g1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "discord", "user-1", "channel", "42", "100", "updated"
    )
    assert result.ok is False
    assert result.error == "edit_chunked"
    adapter.edit_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_message_thread_ref_is_refused(sent_store):
    # Discord/Telegram hand back a thread or chat id, which no edit call takes.
    _seed_sent(sent_store, "discord", "42", "100", "user-1", editable=False)
    adapter = _adapter(channel_server="g1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "discord", "user-1", "channel", "42", "100", "updated"
    )
    assert result.ok is False
    assert result.error == "edit_unsupported_ref"
    adapter.edit_channel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_message_records_authorship_so_the_sender_can_edit(sent_store):
    adapter = _adapter(channel_server="g1", posted=PostedRef(id="100"))
    posted = await outbound.deliver_message(
        adapter, _api(["g1"]), "discord", "user-1", "999888777666555444", "hi"
    )
    assert posted.ok is True

    edited = await outbound.edit_message(
        adapter,
        _api(["g1"]),
        "discord",
        "user-1",
        "channel",
        "999888777666555444",
        "100",
        "updated",
    )
    assert edited.ok is True


@pytest.mark.asyncio
async def test_deliver_message_records_chunk_count_from_the_adapter(sent_store):
    adapter = _adapter(channel_server="g1", posted=PostedRef(id="100", chunk_count=2))
    await outbound.deliver_message(
        adapter, _api(["g1"]), "discord", "user-1", "999888777666555444", "hi"
    )
    result = await outbound.edit_message(
        adapter,
        _api(["g1"]),
        "discord",
        "user-1",
        "channel",
        "999888777666555444",
        "100",
        "updated",
    )
    assert result.error == "edit_chunked"


@pytest.mark.asyncio
async def test_create_thread_records_the_adapters_editable_flag(sent_store):
    # A thread whose body never posted: the thread id still reaches the
    # caller so a retry can't duplicate it, but there is nothing to edit.
    adapter = _adapter(
        channel_server="g1",
        thread=PostedRef(id="t1", channel_id="t1", editable=False),
    )
    await outbound.create_thread(
        adapter, _api(["g1"]), "discord", "user-1", "999888777666555444", "n", "hi"
    )
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "discord", "user-1", "channel", "t1", "t1", "updated"
    )
    assert result.error == "edit_unsupported_ref"


@pytest.mark.asyncio
async def test_thread_body_is_editable_inside_the_new_thread(sent_store):
    # The body lives in the thread, not the channel it was created from, so
    # the edit target has to be the thread.
    adapter = _adapter(
        channel_server="g1", thread=PostedRef(id="body-1", channel_id="thread-9")
    )
    posted = await outbound.create_thread(
        adapter, _api(["g1"]), "discord", "user-1", "999888777666555444", "n", "hi"
    )
    assert posted.ok is True
    assert posted.channel_id == "thread-9"
    assert posted.ref_id == "body-1"

    edited = await outbound.edit_message(
        adapter,
        _api(["g1"]),
        "discord",
        "user-1",
        "channel",
        "thread-9",
        "body-1",
        "updated",
    )
    assert edited.ok is True
    adapter.edit_channel_message.assert_awaited_once_with(
        "thread-9", "body-1", "updated"
    )


@pytest.mark.asyncio
async def test_thread_without_its_own_channel_falls_back_to_the_parent(sent_store):
    # Teams delegates thread creation to a plain post, so its ref carries no
    # separate channel; the channel posted to is still the right one.
    adapter = _adapter(channel_server="g1", thread=PostedRef(id="m1"))
    posted = await outbound.create_thread(
        adapter, _api(["g1"]), "teams", "user-1", "999888777666555444", "n", "hi"
    )
    assert posted.channel_id == "999888777666555444"


@pytest.mark.asyncio
async def test_authorship_is_scoped_per_platform(sent_store):
    _seed_sent(sent_store, "discord", "42", "100", "user-1")
    adapter = _adapter(channel_server="g1")
    result = await outbound.edit_message(
        adapter, _api(["g1"]), "slack", "user-1", "channel", "42", "100", "updated"
    )
    assert result.error == "not_sender"
