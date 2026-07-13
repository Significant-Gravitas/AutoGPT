"""Tests for the Slack webhook adapter."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot.adapters.base import FileAttachment

from .adapter import SlackAdapter, _decode_target, _encode_target

_SIGN = "backend.copilot.bot.adapters.slack.adapter.signing.verify"


@pytest.fixture
def adapter() -> SlackAdapter:
    with patch("backend.copilot.bot.adapters.slack.adapter.AsyncWebClient"):
        a = SlackAdapter(MagicMock())
    client = MagicMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "111.222"})
    client.chat_postEphemeral = AsyncMock()
    client.chat_getPermalink = AsyncMock(return_value={"permalink": "https://x/p"})
    client.files_upload_v2 = AsyncMock()
    client.conversations_info = AsyncMock(return_value={"ok": True})
    client.users_info = AsyncMock(
        return_value={"user": {"profile": {"display_name": "Bently"}}}
    )
    client.auth_test = AsyncMock(return_value={"user_id": "UBOT", "team_id": "T1"})
    a._client = client
    return a


def _req(raw: bytes, headers: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.body = AsyncMock(return_value=raw)
    r.headers = headers or {}
    return r


class TestInboundRouting:
    @pytest.mark.asyncio
    async def test_invalid_signature_is_rejected(self, adapter):
        with patch(_SIGN, return_value=False):
            resp = await adapter._handle_event_request(_req(b"{}"))
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_url_verification_echoes_challenge(self, adapter):
        body = json.dumps({"type": "url_verification", "challenge": "xyz"}).encode()
        with patch(_SIGN, return_value=True):
            resp = await adapter._handle_event_request(_req(body))
        assert json.loads(bytes(resp.body).decode()) == {"challenge": "xyz"}

    @pytest.mark.asyncio
    async def test_app_mention_builds_channel_context(self, adapter):
        captured = {}

        async def cb(ctx, ad):
            captured["ctx"] = ctx

        adapter.on_message(cb)
        await adapter._dispatch_event(
            {
                "type": "app_mention",
                "channel": "C1",
                "ts": "1.2",
                "user": "U1",
                "text": "<@UBOT> hi there",
                "team": "T1",
            }
        )
        ctx = captured["ctx"]
        assert ctx.platform == "slack"
        assert ctx.channel_type == "channel"
        assert ctx.bot_mentioned is True
        assert ctx.text == "hi there"  # bot mention stripped

    @pytest.mark.asyncio
    async def test_thread_reply_is_not_bot_mentioned(self, adapter):
        captured = {}

        async def cb(ctx, ad):
            captured["ctx"] = ctx

        adapter.on_message(cb)
        await adapter._dispatch_event(
            {
                "type": "message",
                "channel_type": "channel",
                "channel": "C1",
                "ts": "2.0",
                "thread_ts": "1.0",
                "user": "U1",
                "text": "follow up",
            }
        )
        ctx = captured["ctx"]
        assert ctx.channel_type == "thread"
        assert ctx.bot_mentioned is False
        assert ctx.channel_id == "C1|1.0"

    @pytest.mark.asyncio
    async def test_channel_uses_top_level_team_id_for_server(self, adapter):
        # The event has no "team" — server_id must come from the payload's
        # top-level team_id (threaded via _dispatch_event).
        captured = {}

        async def cb(ctx, ad):
            captured["ctx"] = ctx

        adapter.on_message(cb)
        await adapter._dispatch_event(
            {
                "type": "app_mention",
                "channel": "C1",
                "ts": "1.2",
                "user": "U1",
                "text": "hi",
            },
            "T99",
        )
        assert captured["ctx"].server_id == "T99"

    @pytest.mark.asyncio
    async def test_dm_has_no_server_id(self, adapter):
        captured = {}

        async def cb(ctx, ad):
            captured["ctx"] = ctx

        adapter.on_message(cb)
        await adapter._dispatch_event(
            {
                "type": "message",
                "channel_type": "im",
                "channel": "D1",
                "ts": "1",
                "user": "U1",
                "text": "hi",
            },
            "T99",
        )
        assert captured["ctx"].channel_type == "dm"
        assert captured["ctx"].server_id is None

    @pytest.mark.asyncio
    async def test_bot_messages_are_skipped(self, adapter):
        called = []

        async def cb(ctx, ad):
            called.append(ctx)

        adapter.on_message(cb)
        await adapter._dispatch_event(
            {"type": "app_mention", "bot_id": "B9", "channel": "C1", "ts": "1"}
        )
        assert called == []

    @pytest.mark.asyncio
    async def test_inbound_files_become_attachments(self, adapter):
        adapter._download_slack_file = AsyncMock(return_value=b"filedata")
        captured = {}

        async def cb(ctx, ad):
            captured["ctx"] = ctx

        adapter.on_message(cb)
        await adapter._dispatch_event(
            {
                "type": "message",
                "channel_type": "im",
                "channel": "D1",
                "ts": "1",
                "user": "U1",
                "text": "look",
                "files": [{"name": "a.txt", "size": 8, "mimetype": "text/plain"}],
            }
        )
        ctx = captured["ctx"]
        assert ctx.channel_type == "dm"
        assert len(ctx.attachments) == 1
        assert ctx.attachments[0].filename == "a.txt"
        assert ctx.attachments[0].content == b"filedata"


class TestOutbound:
    @pytest.mark.asyncio
    async def test_send_message_renders_mrkdwn_and_mentions_and_threads(self, adapter):
        await adapter.send_message("C1|1.2", "**hi** @Bently", (("Bently", "U9"),))
        call = adapter._client.chat_postMessage.await_args.kwargs
        assert call["channel"] == "C1"
        assert call["thread_ts"] == "1.2"
        assert call["text"] == "*hi* <@U9>"

    @pytest.mark.asyncio
    async def test_non_allowlisted_mention_is_not_pinged(self, adapter):
        await adapter.send_message("C1", "hi @Ghost", ())
        assert adapter._client.chat_postMessage.await_args.kwargs["text"] == "hi @Ghost"

    @pytest.mark.asyncio
    async def test_send_file_uploads_into_thread(self, adapter):
        await adapter.send_file(
            "C1|1.2",
            "here you go",
            FileAttachment(filename="r.txt", mime_type="text/plain", content=b"x"),
        )
        call = adapter._client.files_upload_v2.await_args.kwargs
        assert call["channel"] == "C1"
        assert call["thread_ts"] == "1.2"
        assert call["filename"] == "r.txt"
        assert call["content"] == b"x"

    @pytest.mark.asyncio
    async def test_post_channel_message_returns_ref_with_permalink(self, adapter):
        ref = await adapter.post_channel_message("C1", "hello")
        assert ref is not None
        assert ref.id == "111.222"
        assert ref.url == "https://x/p"

    @pytest.mark.asyncio
    async def test_create_thread_encodes_target(self, adapter):
        assert await adapter.create_thread("C1", "9.9", "name") == "C1|9.9"

    @pytest.mark.asyncio
    async def test_rename_thread_is_noop(self, adapter):
        assert await adapter.rename_thread("C1|9.9", "x") is False


class TestChannelIdGrammar:
    def test_slack_ids_match(self, adapter):
        assert adapter.looks_like_channel_id("C01234567")
        assert adapter.looks_like_channel_id("D09ABCDEF")

    def test_channel_names_do_not_match(self, adapter):
        assert not adapter.looks_like_channel_id("general")
        assert not adapter.looks_like_channel_id("announcements")


class TestIdentityCaching:
    @pytest.mark.asyncio
    async def test_auth_failure_is_not_cached_and_retries(self, adapter):
        adapter._client.auth_test = AsyncMock(
            side_effect=[RuntimeError("blip"), {"user_id": "UBOT", "team_id": "T1"}]
        )
        # First call fails → not cached as "".
        assert await adapter._bot_user_id_cached() == ""
        assert adapter._bot_user_id is None
        # Next call recovers.
        assert await adapter._bot_user_id_cached() == "UBOT"
        assert adapter._team_id == "T1"

    @pytest.mark.asyncio
    async def test_empty_auth_response_is_not_cached_and_retries(self, adapter):
        adapter._client.auth_test = AsyncMock(
            side_effect=[
                {"user_id": "", "team_id": ""},
                {"user_id": "UBOT", "team_id": "T1"},
            ]
        )
        assert await adapter._bot_user_id_cached() == ""
        assert adapter._bot_user_id is None  # falsy response not cached
        assert await adapter._bot_user_id_cached() == "UBOT"


def test_target_encode_decode_roundtrip():
    assert _decode_target(_encode_target("C1", "1.2")) == ("C1", "1.2")
    assert _decode_target("D1") == ("D1", None)
