"""Tests for the Slack adapter."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from .adapter import (
    COMMANDS_PATH,
    EVENTS_PATH,
    SlackAdapter,
    _decode_target,
    _encode_target,
)


def _adapter(bot_user_id: str = "UBOT") -> SlackAdapter:
    api = MagicMock()
    api.create_link_token = AsyncMock()
    with patch(
        "backend.copilot.bot.adapters.slack.adapter.AsyncWebClient"
    ) as client_cls:
        adapter = SlackAdapter(api)
        client_cls.assert_called_once()
    # Pre-cache identity so tests don't trigger auth.test calls.
    adapter._bot_user_id = bot_user_id
    adapter._client = AsyncMock()
    return adapter


def _app_with_adapter(adapter: SlackAdapter) -> FastAPI:
    app = FastAPI()
    adapter.register_routes(app)
    return app


class TestTargetEncoding:
    def test_round_trip_with_thread(self):
        encoded = _encode_target("C123", "1700000000.000")
        assert _decode_target(encoded) == ("C123", "1700000000.000")

    def test_dm_target_has_no_thread(self):
        assert _decode_target("D123") == ("D123", None)


class TestEventsRoute:
    def test_url_verification_returns_challenge(self):
        adapter = _adapter()
        client = TestClient(_app_with_adapter(adapter))
        with patch(
            "backend.copilot.bot.adapters.slack.adapter._verify_signature",
            return_value=True,
        ):
            r = client.post(
                EVENTS_PATH,
                content=json.dumps({"type": "url_verification", "challenge": "abc123"}),
                headers={"content-type": "application/json"},
            )
        assert r.status_code == 200
        assert r.json() == {"challenge": "abc123"}

    def test_invalid_signature_returns_401(self):
        adapter = _adapter()
        client = TestClient(_app_with_adapter(adapter))
        with patch(
            "backend.copilot.bot.adapters.slack.adapter._verify_signature",
            return_value=False,
        ):
            r = client.post(
                EVENTS_PATH,
                content=json.dumps({"type": "url_verification", "challenge": "x"}),
                headers={"content-type": "application/json"},
            )
        assert r.status_code == 401

    def test_event_callback_acks_immediately(self):
        adapter = _adapter()
        client = TestClient(_app_with_adapter(adapter))
        with patch(
            "backend.copilot.bot.adapters.slack.adapter._verify_signature",
            return_value=True,
        ):
            r = client.post(
                EVENTS_PATH,
                content=json.dumps(
                    {
                        "type": "event_callback",
                        "event": {"type": "app_mention"},
                    }
                ),
                headers={"content-type": "application/json"},
            )
        assert r.status_code == 200


class TestDispatch:
    @pytest.mark.asyncio
    async def test_skips_bot_messages(self):
        adapter = _adapter()
        callback = AsyncMock()
        adapter.on_message(callback)
        await adapter._dispatch_event(
            {"type": "message", "subtype": "bot_message", "channel_type": "im"}
        )
        callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_messages_with_bot_id(self):
        adapter = _adapter()
        callback = AsyncMock()
        adapter.on_message(callback)
        await adapter._dispatch_event(
            {"type": "message", "bot_id": "B999", "channel_type": "im"}
        )
        callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dm_message_builds_dm_context(self):
        adapter = _adapter()
        callback = AsyncMock()
        adapter.on_message(callback)
        adapter._user_name_cache["U42"] = "Bently"

        await adapter._dispatch_event(
            {
                "type": "message",
                "channel_type": "im",
                "channel": "D1",
                "ts": "1.0",
                "user": "U42",
                "text": "hello bot",
            }
        )
        callback.assert_awaited_once()
        await_args = callback.await_args
        assert await_args is not None
        ctx, _ = await_args.args
        assert ctx.platform == "slack"
        assert ctx.channel_type == "dm"
        assert ctx.channel_id == "D1"
        assert ctx.username == "Bently"
        assert ctx.text == "hello bot"

    @pytest.mark.asyncio
    async def test_channel_mention_uses_channel_target(self):
        adapter = _adapter()
        callback = AsyncMock()
        adapter.on_message(callback)
        adapter._user_name_cache["U42"] = "Bently"

        await adapter._dispatch_event(
            {
                "type": "app_mention",
                "channel": "C1",
                "ts": "1.0",
                "user": "U42",
                "team": "T1",
                "text": "<@UBOT> help",
            }
        )
        callback.assert_awaited_once()
        await_args = callback.await_args
        assert await_args is not None
        ctx, _ = await_args.args
        assert ctx.channel_type == "channel"
        assert ctx.channel_id == "C1"
        assert ctx.server_id == "T1"
        assert ctx.text == "help"  # bot mention stripped


class TestStripMentions:
    @pytest.mark.asyncio
    async def test_bot_mention_removed(self):
        adapter = _adapter("UBOT")
        result = await adapter._strip_mentions("<@UBOT> hello")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_user_mention_resolved_to_display_name(self):
        adapter = _adapter("UBOT")
        adapter._user_name_cache["U42"] = "Bently"
        result = await adapter._strip_mentions("thanks <@U42>")
        assert result == "thanks @Bently"

    @pytest.mark.asyncio
    async def test_handles_pipe_form(self):
        adapter = _adapter("UBOT")
        adapter._user_name_cache["U42"] = "Bently"
        result = await adapter._strip_mentions("ping <@U42|bently>")
        assert result == "ping @Bently"


class TestCollectMentionableUsers:
    @pytest.mark.asyncio
    async def test_excludes_bot_itself(self):
        adapter = _adapter("UBOT")
        adapter._user_name_cache["U42"] = "Bently"
        result = await adapter._collect_mentionable_users("<@UBOT> hey <@U42>")
        assert result == (("Bently", "U42"),)

    @pytest.mark.asyncio
    async def test_dedupes_repeats(self):
        adapter = _adapter("UBOT")
        adapter._user_name_cache["U42"] = "Bently"
        result = await adapter._collect_mentionable_users("<@U42> please <@U42> again")
        assert result == (("Bently", "U42"),)


class TestCommandRoute:
    def test_invalid_signature_returns_401(self):
        adapter = _adapter()
        client = TestClient(_app_with_adapter(adapter))
        with patch(
            "backend.copilot.bot.adapters.slack.adapter._verify_signature",
            return_value=False,
        ):
            r = client.post(
                COMMANDS_PATH,
                data={"command": "/setup"},
            )
        assert r.status_code == 401

    def test_valid_request_routes_to_commands(self):
        adapter = _adapter()
        client = TestClient(_app_with_adapter(adapter))
        with (
            patch(
                "backend.copilot.bot.adapters.slack.adapter._verify_signature",
                return_value=True,
            ),
            patch(
                "backend.copilot.bot.adapters.slack.adapter.commands.handle",
                new=AsyncMock(
                    return_value=MagicMock(status_code=200, body=b'{"ok":true}')
                ),
            ) as mock_handle,
        ):
            client.post(COMMANDS_PATH, data={"command": "/help"})
        mock_handle.assert_awaited_once()


class TestOutbound:
    @pytest.mark.asyncio
    async def test_send_message_uses_decoded_channel_and_thread_ts(self):
        adapter = _adapter()
        await adapter.send_message("C1|1700.0", "hi")
        adapter._client.chat_postMessage.assert_awaited_once_with(
            channel="C1", text="hi", thread_ts="1700.0"
        )

    @pytest.mark.asyncio
    async def test_send_message_dm_has_no_thread_ts(self):
        adapter = _adapter()
        await adapter.send_message("D1", "hi")
        adapter._client.chat_postMessage.assert_awaited_once_with(
            channel="D1", text="hi", thread_ts=None
        )

    @pytest.mark.asyncio
    async def test_send_reply_falls_back_to_message_id_when_no_thread_in_target(
        self,
    ):
        adapter = _adapter()
        await adapter.send_reply("C1", "hi", reply_to_message_id="1700.0")
        adapter._client.chat_postMessage.assert_awaited_once_with(
            channel="C1", text="hi", thread_ts="1700.0"
        )

    @pytest.mark.asyncio
    async def test_create_thread_packs_channel_and_message_ts(self):
        adapter = _adapter()
        result = await adapter.create_thread("C1", "1700.0", "name ignored")
        assert result == "C1|1700.0"

    @pytest.mark.asyncio
    async def test_rename_thread_returns_false(self):
        adapter = _adapter()
        assert await adapter.rename_thread("C1|1700.0", "new name") is False
