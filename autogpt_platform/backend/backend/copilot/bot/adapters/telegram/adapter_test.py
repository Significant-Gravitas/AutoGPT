"""Tests for the Telegram adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot.adapters.base import FileAttachment

from .adapter import (
    TelegramAdapter,
    _collect_mentionable_users,
    _decode_target,
    _encode_target,
    _verify_secret,
)

_ADAPTER = "backend.copilot.bot.adapters.telegram.adapter"


def _adapter() -> TelegramAdapter:
    with patch(f"{_ADAPTER}.config.get_bot_token", return_value="123:abc"):
        a = TelegramAdapter(MagicMock())
    a._bot_id = "999"
    a._bot_username = "OurBot"
    a._client = MagicMock()
    a._client.call = AsyncMock(return_value={"message_id": 77})
    return a


def _dm(text: str, **extra) -> dict:
    return {
        "message": {
            "message_id": 10,
            "text": text,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 42, "username": "bently"},
            **extra,
        }
    }


def _group(text: str, **extra) -> dict:
    return {
        "message": {
            "message_id": 11,
            "text": text,
            "chat": {"id": -100555, "type": "supergroup", "title": "Builders"},
            "from": {"id": 42, "username": "bently"},
            **extra,
        }
    }


class TestVerifySecret:
    def test_matching_header_passes(self):
        request = MagicMock()
        request.headers = {"X-Telegram-Bot-Api-Secret-Token": "s3cret"}
        with patch(f"{_ADAPTER}.config.get_webhook_secret", return_value="s3cret"):
            assert _verify_secret(request, b"") is True

    def test_wrong_or_missing_header_fails(self):
        request = MagicMock()
        request.headers = {"X-Telegram-Bot-Api-Secret-Token": "nope"}
        with patch(f"{_ADAPTER}.config.get_webhook_secret", return_value="s3cret"):
            assert _verify_secret(request, b"") is False
        request.headers = {}
        with patch(f"{_ADAPTER}.config.get_webhook_secret", return_value="s3cret"):
            assert _verify_secret(request, b"") is False

    def test_unconfigured_secret_rejects_everything(self):
        request = MagicMock()
        request.headers = {"X-Telegram-Bot-Api-Secret-Token": ""}
        with patch(f"{_ADAPTER}.config.get_webhook_secret", return_value=""):
            assert _verify_secret(request, b"") is False


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dm_dispatches_as_dm_context(self):
        a = _adapter()
        seen = []
        a.on_message(lambda ctx, adapter: seen.append(ctx) or _noop())
        await a._dispatch_update(_dm("hello there"))
        assert len(seen) == 1
        ctx = seen[0]
        assert ctx.channel_type == "dm"
        assert ctx.server_id is None
        assert ctx.channel_id == "42"
        assert ctx.text == "hello there"

    @pytest.mark.asyncio
    async def test_group_message_without_mention_is_ignored(self):
        a = _adapter()
        seen = []
        a.on_message(lambda ctx, adapter: seen.append(ctx) or _noop())
        await a._dispatch_update(_group("just chatting"))
        assert seen == []

    @pytest.mark.asyncio
    async def test_group_mention_dispatches_and_strips_the_mention(self):
        a = _adapter()
        seen = []
        a.on_message(lambda ctx, adapter: seen.append(ctx) or _noop())
        await a._dispatch_update(_group("@OurBot summarise this"))
        assert len(seen) == 1
        ctx = seen[0]
        assert ctx.channel_type == "channel"
        assert ctx.server_id == "-100555"
        assert ctx.text == "summarise this"
        assert ctx.bot_mentioned is True

    @pytest.mark.asyncio
    async def test_reply_to_bot_message_counts_as_engagement(self):
        a = _adapter()
        seen = []
        a.on_message(lambda ctx, adapter: seen.append(ctx) or _noop())
        await a._dispatch_update(
            _group("and now?", reply_to_message={"from": {"id": 999, "is_bot": True}})
        )
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_bot_messages_are_skipped(self):
        a = _adapter()
        seen = []
        a.on_message(lambda ctx, adapter: seen.append(ctx) or _noop())
        update = _dm("@OurBot hi")
        update["message"]["from"]["is_bot"] = True
        await a._dispatch_update(update)
        assert seen == []

    @pytest.mark.asyncio
    async def test_forum_topic_id_rides_in_the_target(self):
        a = _adapter()
        seen = []
        a.on_message(lambda ctx, adapter: seen.append(ctx) or _noop())
        await a._dispatch_update(_group("@OurBot hi", message_thread_id=7))
        assert seen[0].channel_id == "-100555|7"

    @pytest.mark.asyncio
    async def test_command_routes_to_command_handler_not_chat(self):
        a = _adapter()
        seen = []
        a.on_message(lambda ctx, adapter: seen.append(ctx) or _noop())
        update = _group("/setup")
        update["message"]["entities"] = [
            {"type": "bot_command", "offset": 0, "length": 6}
        ]
        with patch(f"{_ADAPTER}.commands.handle", new=AsyncMock()) as handle:
            await a._dispatch_update(update)
        handle.assert_awaited_once()
        assert seen == []

    @pytest.mark.asyncio
    async def test_command_handler_error_is_contained(self):
        # _dispatch_update runs as a fire-and-forget task — a raising command
        # handler must be caught and logged, not left to asyncio's deferred
        # "exception never retrieved".
        a = _adapter()
        update = _group("/setup")
        update["message"]["entities"] = [
            {"type": "bot_command", "offset": 0, "length": 6}
        ]
        with patch(
            f"{_ADAPTER}.commands.handle",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await a._dispatch_update(update)


class TestFeatures:
    @pytest.mark.asyncio
    async def test_engaging_message_gets_a_reaction_ack(self):
        a = _adapter()
        a.on_message(lambda ctx, adapter: _noop())
        await a._dispatch_update(_dm("hello"))
        methods = [c.args[0] for c in a._client.call.call_args_list]
        assert "setMessageReaction" in methods

    @pytest.mark.asyncio
    async def test_ignored_group_message_gets_no_reaction(self):
        a = _adapter()
        a.on_message(lambda ctx, adapter: _noop())
        await a._dispatch_update(_group("no mention here"))
        assert a._client.call.call_args_list == []

    @pytest.mark.asyncio
    async def test_command_menu_registered_on_startup(self):
        a = _adapter()
        await a._register_command_menu()
        assert a._client.call.call_args.args == ("setMyCommands",)
        menu = a._client.call.call_args.kwargs["commands"]
        assert {c["command"] for c in menu} >= {"setup", "new", "help", "unlink"}

    @pytest.mark.asyncio
    async def test_image_files_send_as_photos_documents_otherwise(self):
        a = _adapter()
        a._client.send_photo = AsyncMock()
        a._client.send_document = AsyncMock()
        await a.send_file(
            "42",
            "",
            FileAttachment(filename="x.png", mime_type="image/png", content=b"i"),
        )
        a._client.send_photo.assert_awaited_once()
        await a.send_file(
            "42",
            "",
            FileAttachment(filename="x.pdf", mime_type="application/pdf", content=b"d"),
        )
        a._client.send_document.assert_awaited_once()


class TestAnalytics:
    @pytest.mark.asyncio
    async def test_group_join_records_guild(self):
        a = _adapter()
        await a._dispatch_update(
            {
                "my_chat_member": {
                    "chat": {"id": -100777, "type": "supergroup", "title": "Builders"},
                    "new_chat_member": {"status": "member"},
                }
            }
        )
        a._api.track_guild_joined.assert_called_once_with(
            "telegram", "-100777", "Builders"
        )

    @pytest.mark.asyncio
    async def test_group_kick_records_guild_left(self):
        a = _adapter()
        await a._dispatch_update(
            {
                "my_chat_member": {
                    "chat": {"id": -100777, "type": "supergroup"},
                    "new_chat_member": {"status": "kicked"},
                }
            }
        )
        a._api.track_guild_left.assert_called_once_with("telegram", "-100777")

    @pytest.mark.asyncio
    async def test_private_membership_changes_are_ignored(self):
        a = _adapter()
        await a._dispatch_update(
            {
                "my_chat_member": {
                    "chat": {"id": 42, "type": "private"},
                    "new_chat_member": {"status": "member"},
                }
            }
        )
        a._api.track_guild_joined.assert_not_called()


class TestOutbound:
    @pytest.mark.asyncio
    async def test_send_message_renders_html_and_threads(self):
        a = _adapter()
        await a.send_message("-100555|7", "**bold** & plain")
        kwargs = a._client.call.call_args.kwargs
        assert a._client.call.call_args.args == ("sendMessage",)
        assert kwargs["chat_id"] == "-100555"
        assert kwargs["message_thread_id"] == 7
        assert kwargs["parse_mode"] == "HTML"
        assert kwargs["text"] == "<b>bold</b> &amp; plain"

    @pytest.mark.asyncio
    async def test_send_link_prefers_login_url_for_https(self):
        # Telegram attaches a signed identity when the user taps a login_url
        # button — the /link page verifies it for seamless linking.
        a = _adapter()
        await a.send_link("42", "Link it", "Open AutoGPT", "https://x/l")
        button = a._client.call.call_args.kwargs["reply_markup"]["inline_keyboard"][0][
            0
        ]
        assert button == {"text": "Open AutoGPT", "login_url": {"url": "https://x/l"}}

    @pytest.mark.asyncio
    async def test_send_link_falls_back_to_plain_url_when_login_rejected(self):
        # Unregistered domain (no BotFather /setdomain) → login_url send fails;
        # the link must still arrive as a plain URL button.
        a = _adapter()
        a._client.call = AsyncMock(
            side_effect=[RuntimeError("BUTTON_TYPE_INVALID"), {}]
        )
        await a.send_link("42", "Link it", "Open AutoGPT", "https://x/l")
        retry = a._client.call.call_args.kwargs["reply_markup"]["inline_keyboard"][0][0]
        assert retry == {"text": "Open AutoGPT", "url": "https://x/l"}

    @pytest.mark.asyncio
    async def test_send_link_degrades_to_text_when_button_rejected(self):
        # Telegram refuses localhost button URLs (local dev) — the link must
        # still arrive, appended as plain text.
        a = _adapter()
        a._client.call = AsyncMock(side_effect=[RuntimeError("Wrong HTTP URL"), {}])
        await a.send_link("42", "Link it", "Open", "http://localhost:3000/l?a=1&b=2")
        final = a._client.call.call_args.kwargs
        assert "reply_markup" not in final
        # parse_mode stays HTML, so the appended link must be entity-escaped.
        assert "Open: http://localhost:3000/l?a=1&amp;b=2" in final["text"]

    @pytest.mark.asyncio
    async def test_send_link_plain_url_for_non_https(self):
        a = _adapter()
        await a.send_link("42", "Link it", "Open", "http://localhost:3000/l")
        button = a._client.call.call_args.kwargs["reply_markup"]["inline_keyboard"][0][
            0
        ]
        assert button == {"text": "Open", "url": "http://localhost:3000/l"}

    @pytest.mark.asyncio
    async def test_create_thread_declines_so_replies_stay_in_chat(self):
        a = _adapter()
        assert await a.create_thread("-100555", "11", "AutoGPT: hi") is None


def test_short_chat_ids_count_as_channel_refs():
    a = _adapter()
    assert a.looks_like_channel_id("12345") is True
    assert a.looks_like_channel_id("-100987654321") is True
    assert a.looks_like_channel_id("general") is False


@pytest.mark.asyncio
async def test_attachments_without_declared_size_are_skipped():
    # The size cap is checked against the declared size pre-fetch; an
    # undeclared size can't be admitted.
    a = _adapter()
    attachments, _skipped = await a._extract_attachments(
        {"document": {"file_id": "f1", "file_name": "x.pdf"}}
    )
    assert attachments == ()


def test_target_codec_roundtrip():
    assert _decode_target(_encode_target("42")) == ("42", None)
    assert _decode_target(_encode_target("-100555", 7)) == ("-100555", 7)


def test_collect_mentionable_users_only_text_mentions_with_ids():
    message = {
        "entities": [
            {"type": "mention"},  # @username form — no numeric id, unusable
            {"type": "text_mention", "user": {"id": 5, "first_name": "Sam"}},
            {"type": "text_mention", "user": {"id": 9, "is_bot": True}},
        ]
    }
    assert _collect_mentionable_users(message) == (("Sam", "5"),)


async def _noop() -> None:
    return None
