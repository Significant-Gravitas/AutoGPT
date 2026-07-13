"""Tests for the Slack webhook adapter (multi-workspace)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot.adapters.base import FileAttachment
from backend.data.bot_installs import BotInstallCredentials

from .adapter import SlackAdapter, _decode_target, _encode_target

_SIGN = "backend.copilot.bot.adapters.slack.adapter.signing.verify"


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.token = "xoxb-test"
    client.chat_postMessage = AsyncMock(return_value={"ts": "111.222"})
    client.chat_postEphemeral = AsyncMock()
    client.chat_getPermalink = AsyncMock(return_value={"permalink": "https://x/p"})
    client.files_upload_v2 = AsyncMock()
    client.conversations_info = AsyncMock(return_value={"ok": True})
    client.users_info = AsyncMock(
        return_value={"user": {"profile": {"display_name": "Bently"}}}
    )
    client.auth_test = AsyncMock(return_value={"user_id": "UBOT", "team_id": "T1"})
    return client


@pytest.fixture
def adapter() -> SlackAdapter:
    a = SlackAdapter(MagicMock())
    client = _mock_client()
    # Seed the per-workspace client cache so the real _client_for resolves to the
    # mock without a DB lookup — covers every team the behaviour tests use.
    for team in ("T1", "T99", ""):
        a._clients[team] = client
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
        # The opaque channel_id carries the workspace so sends pick its token.
        assert ctx.channel_id == "T1|C1|"

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
                "team": "T1",
            }
        )
        ctx = captured["ctx"]
        assert ctx.channel_type == "thread"
        assert ctx.bot_mentioned is False
        assert ctx.channel_id == "T1|C1|1.0"

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
        assert captured["ctx"].channel_id == "T99|C1|"

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
        # Even DMs carry the team in the target so the reply picks the token.
        assert captured["ctx"].channel_id == "T99|D1|"

    @pytest.mark.asyncio
    async def test_event_without_team_is_dropped(self, adapter):
        # No workspace → no token to reply with → ignore rather than guess.
        called = []

        async def cb(ctx, ad):
            called.append(ctx)

        adapter.on_message(cb)
        await adapter._dispatch_event(
            {"type": "app_mention", "channel": "C1", "ts": "1", "user": "U1"}
        )
        assert called == []

    @pytest.mark.asyncio
    async def test_bot_messages_are_skipped(self, adapter):
        called = []

        async def cb(ctx, ad):
            called.append(ctx)

        adapter.on_message(cb)
        # Carries a valid team so the only thing standing between this event
        # and the callback is the bot filter itself (not the missing-team guard).
        await adapter._dispatch_event(
            {
                "type": "app_mention",
                "bot_id": "B9",
                "channel": "C1",
                "ts": "1",
                "user": "U1",
                "text": "hi",
                "team": "T1",
            }
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
                "team": "T1",
                "files": [{"name": "a.txt", "size": 8, "mimetype": "text/plain"}],
            }
        )
        ctx = captured["ctx"]
        assert ctx.channel_type == "dm"
        assert len(ctx.attachments) == 1
        assert ctx.attachments[0].filename == "a.txt"
        assert ctx.attachments[0].content == b"filedata"


class TestUninstall:
    @pytest.mark.asyncio
    async def test_uninstall_event_revokes_and_evicts(self, adapter):
        adapter._clients["T5"] = MagicMock()
        adapter._bot_user_ids["T5"] = "UBOT"
        with patch(
            "backend.copilot.bot.adapters.slack.adapter.revoke_bot_install",
            new=AsyncMock(),
        ) as revoke:
            await adapter._dispatch_event({"type": "app_uninstalled"}, "T5")
        revoke.assert_awaited_once()
        assert "T5" not in adapter._clients
        assert "T5" not in adapter._bot_user_ids


class TestOutbound:
    @pytest.mark.asyncio
    async def test_send_message_renders_mrkdwn_and_mentions_and_threads(self, adapter):
        await adapter.send_message("T1|C1|1.2", "**hi** @Bently", (("Bently", "U9"),))
        call = adapter._clients["T1"].chat_postMessage.await_args.kwargs
        assert call["channel"] == "C1"
        assert call["thread_ts"] == "1.2"
        assert call["text"] == "*hi* <@U9>"

    @pytest.mark.asyncio
    async def test_non_allowlisted_mention_is_not_pinged(self, adapter):
        await adapter.send_message("T1|C1|", "hi @Ghost", ())
        assert (
            adapter._clients["T1"].chat_postMessage.await_args.kwargs["text"]
            == "hi @Ghost"
        )

    @pytest.mark.asyncio
    async def test_send_file_uploads_into_thread(self, adapter):
        await adapter.send_file(
            "T1|C1|1.2",
            "here you go",
            FileAttachment(filename="r.txt", mime_type="text/plain", content=b"x"),
        )
        call = adapter._clients["T1"].files_upload_v2.await_args.kwargs
        assert call["channel"] == "C1"
        assert call["thread_ts"] == "1.2"
        assert call["filename"] == "r.txt"
        assert call["content"] == b"x"

    @pytest.mark.asyncio
    async def test_post_channel_message_returns_ref_with_permalink(self, adapter):
        ref = await adapter.post_channel_message("T1|C1|", "hello")
        assert ref is not None
        assert ref.id == "111.222"
        assert ref.url == "https://x/p"

    @pytest.mark.asyncio
    async def test_create_thread_encodes_target(self, adapter):
        assert await adapter.create_thread("T1|C1|", "9.9", "name") == "T1|C1|9.9"

    @pytest.mark.asyncio
    async def test_rename_thread_is_noop(self, adapter):
        assert await adapter.rename_thread("T1|C1|9.9", "x") is False


class TestChannelIdGrammar:
    def test_slack_ids_match(self, adapter):
        assert adapter.looks_like_channel_id("C01234567")
        assert adapter.looks_like_channel_id("D09ABCDEF")

    def test_channel_names_do_not_match(self, adapter):
        assert not adapter.looks_like_channel_id("general")
        assert not adapter.looks_like_channel_id("announcements")


class TestPerWorkspaceClient:
    @pytest.mark.asyncio
    async def test_client_for_builds_from_stored_install_and_caches(self):
        a = SlackAdapter(MagicMock())
        install = BotInstallCredentials(
            team_id="T1", bot_token="xoxb-abc", bot_user_id="UBOT"
        )
        with (
            patch(
                "backend.copilot.bot.adapters.slack.adapter.get_bot_install",
                new=AsyncMock(return_value=install),
            ) as lookup,
            patch(
                "backend.copilot.bot.adapters.slack.adapter.AsyncWebClient"
            ) as web_client,
        ):
            first = await a._client_for("T1")
            second = await a._client_for("T1")
        web_client.assert_called_once_with(token="xoxb-abc")
        assert first is second  # cached — a single DB lookup + client build
        lookup.assert_awaited_once()
        assert a._bot_user_ids["T1"] == "UBOT"

    @pytest.mark.asyncio
    async def test_client_for_returns_none_when_uninstalled(self):
        a = SlackAdapter(MagicMock())
        with (
            patch(
                "backend.copilot.bot.adapters.slack.adapter.get_bot_install",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "backend.copilot.bot.adapters.slack.adapter.is_install_revoked",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "backend.copilot.bot.adapters.slack.adapter.config.get_bot_token",
                return_value="",
            ),
        ):
            assert await a._client_for("T1") is None

    @pytest.mark.asyncio
    async def test_client_for_rebuilds_after_ttl_so_reinstalled_tokens_apply(self):
        # A re-install replaces the workspace token in the DB; the TTL bounds
        # how long any replica keeps using a client built from the old token.
        from .adapter import _CLIENT_CACHE_TTL_SECONDS

        a = SlackAdapter(MagicMock())
        stale = MagicMock()
        a._clients["T1"] = stale
        a._client_cached_at["T1"] = -_CLIENT_CACHE_TTL_SECONDS  # long expired
        install = BotInstallCredentials(team_id="T1", bot_token="xoxb-new")
        with (
            patch(
                "backend.copilot.bot.adapters.slack.adapter.get_bot_install",
                new=AsyncMock(return_value=install),
            ),
            patch(
                "backend.copilot.bot.adapters.slack.adapter.AsyncWebClient"
            ) as web_client,
        ):
            fresh = await a._client_for("T1")
        assert fresh is not stale
        web_client.assert_called_once_with(token="xoxb-new")

    @pytest.mark.asyncio
    async def test_client_for_revoked_workspace_never_falls_back_to_static(self):
        # An uninstalled workspace must get None — the static token belongs to
        # the app's own workspace, not a workspace that revoked us.
        a = SlackAdapter(MagicMock())
        with (
            patch(
                "backend.copilot.bot.adapters.slack.adapter.get_bot_install",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "backend.copilot.bot.adapters.slack.adapter.is_install_revoked",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "backend.copilot.bot.adapters.slack.adapter.config.get_bot_token",
                return_value="xoxb-static",
            ),
        ):
            assert await a._client_for("TREVOKED") is None

    @pytest.mark.asyncio
    async def test_client_for_empty_team_falls_back_to_static_token(self):
        # A raw channel ref (no team) must resolve via the static token rather
        # than returning None — otherwise proactive posts fail (channel_not_found).
        a = SlackAdapter(MagicMock())
        lookup = AsyncMock(return_value=None)
        with (
            patch(
                "backend.copilot.bot.adapters.slack.adapter.get_bot_install",
                new=lookup,
            ),
            patch(
                "backend.copilot.bot.adapters.slack.adapter.config.get_bot_token",
                return_value="xoxb-static",
            ),
            patch(
                "backend.copilot.bot.adapters.slack.adapter.AsyncWebClient"
            ) as web_client,
        ):
            client = await a._client_for("")
        assert client is not None
        web_client.assert_called_once_with(token="xoxb-static")
        lookup.assert_not_awaited()  # no DB lookup for an empty team


class TestIdentityCaching:
    @pytest.mark.asyncio
    async def test_auth_failure_is_not_cached_and_retries(self, adapter):
        adapter._clients["T1"].auth_test = AsyncMock(
            side_effect=[RuntimeError("blip"), {"user_id": "UBOT"}]
        )
        # First call fails → not cached as "".
        assert await adapter._bot_user_id_for("T1") == ""
        assert "T1" not in adapter._bot_user_ids
        # Next call recovers and caches.
        assert await adapter._bot_user_id_for("T1") == "UBOT"
        assert adapter._bot_user_ids["T1"] == "UBOT"

    @pytest.mark.asyncio
    async def test_empty_auth_response_is_not_cached_and_retries(self, adapter):
        adapter._clients["T1"].auth_test = AsyncMock(
            side_effect=[{"user_id": ""}, {"user_id": "UBOT"}]
        )
        assert await adapter._bot_user_id_for("T1") == ""
        assert "T1" not in adapter._bot_user_ids  # falsy response not cached
        assert await adapter._bot_user_id_for("T1") == "UBOT"


def test_target_encode_decode_roundtrip():
    assert _decode_target(_encode_target("T1", "C1", "1.2")) == ("T1", "C1", "1.2")
    assert _decode_target(_encode_target("T1", "D1")) == ("T1", "D1", None)
    # Back-compat: the old single-workspace two-part form decodes with no team.
    assert _decode_target("C1|1.2") == ("", "C1", "1.2")
    assert _decode_target("D1") == ("", "D1", None)
