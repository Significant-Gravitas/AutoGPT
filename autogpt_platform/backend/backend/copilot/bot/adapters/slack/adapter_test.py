"""Tests for the Slack webhook adapter (multi-workspace)."""

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from slack_sdk.errors import SlackApiError

from backend.copilot.bot.adapters.base import FileAttachment
from backend.data.bot_installs import BotInstallCredentials

from . import config
from .adapter import SlackAdapter, _decode_target, _encode_target

_SIGN = "backend.copilot.bot.adapters.slack.adapter.signing.verify"
_SUBSCRIBED = "backend.copilot.bot.adapters.slack.adapter.threads.is_subscribed"


def _mention_in_thread() -> dict:
    return {
        "type": "app_mention",
        "channel": "C1",
        "ts": "2.0",
        "thread_ts": "1.0",
        "user": "U1",
        "text": "<@UBOT> summarize",
        "team": "T1",
    }


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.token = "xoxb-test"
    client.chat_postMessage = AsyncMock(return_value={"ts": "111.222"})
    client.chat_postEphemeral = AsyncMock()
    client.chat_getPermalink = AsyncMock(return_value={"permalink": "https://x/p"})
    client.files_upload_v2 = AsyncMock()
    client.conversations_info = AsyncMock(return_value={"ok": True})
    client.conversations_replies = AsyncMock(return_value={"messages": []})
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
    async def test_private_channel_thread_reply_is_forwarded(self, adapter):
        # Slack delivers private-channel messages as channel_type "group" —
        # dropping them killed every thread follow-up in private channels.
        captured = {}

        async def cb(ctx, ad):
            captured["ctx"] = ctx

        adapter.on_message(cb)
        await adapter._dispatch_event(
            {
                "type": "message",
                "channel_type": "group",
                "channel": "G1",
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
        assert ctx.channel_id == "T1|G1|1.0"

    @pytest.mark.asyncio
    async def test_mention_in_thread_does_not_double_fire(self, adapter):
        # Slack sends BOTH app_mention and message.channels for one @mention.
        # In a thread we own, the subscription gate passes both, so without a
        # dedupe the turn runs twice — two model calls, two replies.
        contexts = []

        async def cb(ctx, ad):
            contexts.append(ctx)

        adapter.on_message(cb)
        mention = {
            "type": "app_mention",
            "channel": "C1",
            "ts": "2.0",
            "thread_ts": "1.0",
            "user": "U1",
            "text": "<@UBOT> what now",
            "team": "T1",
        }
        with patch(_SUBSCRIBED, new=AsyncMock(return_value=True)):
            await adapter._dispatch_event(mention)
            await adapter._dispatch_event(
                {**mention, "type": "message", "channel_type": "channel"}
            )
        assert len(contexts) == 1
        assert contexts[0].bot_mentioned is True

    @pytest.mark.asyncio
    async def test_private_channel_mention_does_not_double_fire(self, adapter):
        contexts = []

        async def cb(ctx, ad):
            contexts.append(ctx)

        adapter.on_message(cb)
        mention = {
            "type": "app_mention",
            "channel": "G1",
            "ts": "2.0",
            "thread_ts": "1.0",
            "user": "U1",
            "text": "hey <@UBOT> look",
            "team": "T1",
        }
        with patch(_SUBSCRIBED, new=AsyncMock(return_value=True)):
            await adapter._dispatch_event(mention)
            await adapter._dispatch_event(
                {**mention, "type": "message", "channel_type": "group"}
            )
        assert len(contexts) == 1

    @pytest.mark.asyncio
    async def test_thread_reply_mentioning_someone_else_still_forwards(self, adapter):
        # Only OUR mention is a duplicate of an app_mention; a reply that pings
        # a teammate must still reach the handler.
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
                "text": "<@U9> can you look",
                "team": "T1",
            }
        )
        assert captured["ctx"].bot_mentioned is False

    @pytest.mark.asyncio
    async def test_unknown_bot_identity_keeps_forwarding_thread_replies(self, adapter):
        # Failing open on an unresolvable identity: a duplicate turn is a much
        # smaller failure than silently swallowing the user's message.
        adapter._clients["T1"].auth_test = AsyncMock(side_effect=RuntimeError("down"))
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
                "text": "<@UBOT> hi",
                "team": "T1",
            }
        )
        assert "ctx" in captured

    @pytest.mark.asyncio
    async def test_mention_in_unowned_thread_pulls_history(self, adapter):
        client = adapter._clients["T1"]
        client.conversations_replies = AsyncMock(
            return_value={
                "messages": [
                    {"ts": "1.0", "user": "U2", "text": "the parent"},
                    {"ts": "1.5", "user": "UBOT", "text": "bot noise"},
                    {"ts": "2.0", "user": "U1", "text": "<@UBOT> summarize"},
                ]
            }
        )
        captured = {}

        async def cb(ctx, ad):
            captured["ctx"] = ctx

        adapter.on_message(cb)
        with patch(_SUBSCRIBED, new=AsyncMock(return_value=False)):
            await adapter._dispatch_event(_mention_in_thread())
        client.conversations_replies.assert_awaited_once()
        # Bot's own message and the triggering mention are excluded.
        assert [e.text for e in captured["ctx"].thread_history] == ["the parent"]

    @pytest.mark.asyncio
    async def test_redis_blip_during_history_gate_does_not_drop_the_message(
        self, adapter
    ):
        # Optional context must never cost the user their turn — including a
        # failure in the subscription gate itself.
        captured = {}

        async def cb(ctx, ad):
            captured["ctx"] = ctx

        adapter.on_message(cb)
        with patch(_SUBSCRIBED, new=AsyncMock(side_effect=RuntimeError("redis down"))):
            await adapter._dispatch_event(_mention_in_thread())
        assert captured["ctx"].thread_history == ()

    @pytest.mark.asyncio
    async def test_mention_in_own_thread_skips_the_history_fetch(self, adapter):
        # The handler discards history for a subscribed thread, so the
        # adapter must not pay for the fetch.
        adapter.on_message(AsyncMock())
        with patch(_SUBSCRIBED, new=AsyncMock(return_value=True)):
            await adapter._dispatch_event(_mention_in_thread())
        adapter._clients["T1"].conversations_replies.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dm_and_unmentioned_thread_reply_fetch_no_history(self, adapter):
        adapter.on_message(AsyncMock())
        with patch(_SUBSCRIBED, new=AsyncMock(return_value=False)):
            await adapter._dispatch_event(
                {
                    "type": "message",
                    "channel_type": "im",
                    "channel": "D1",
                    "ts": "2.0",
                    "thread_ts": "1.0",
                    "user": "U1",
                    "text": "hi",
                    "team": "T1",
                }
            )
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
        adapter._clients["T1"].conversations_replies.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_group_dm_thread_reply_is_deliberately_dropped(self, adapter):
        # Multi-person DMs ("mpim") are not handled; pin the omission.
        cb = AsyncMock()
        adapter.on_message(cb)
        await adapter._dispatch_event(
            {
                "type": "message",
                "channel_type": "mpim",
                "channel": "G9",
                "ts": "2.0",
                "thread_ts": "1.0",
                "user": "U1",
                "text": "follow up",
                "team": "T1",
            }
        )
        cb.assert_not_awaited()

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
        revoke = AsyncMock()
        with patch(
            "backend.copilot.bot.adapters.slack.adapter.bot_installs_db",
            return_value=MagicMock(revoke_bot_install=revoke),
        ):
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
    async def test_raw_control_sequences_are_escaped_but_allowlist_pings(self, adapter):
        # A model-output <!channel> or raw <@Uid> must be neutralized; only
        # the allowlisted @Bently comes back as a live mention token.
        await adapter.send_message(
            "T1|C1|", "<!channel> <@U9> @Bently", (("Bently", "U9"),)
        )
        text = adapter._clients["T1"].chat_postMessage.await_args.kwargs["text"]
        assert text == "&lt;!channel&gt; &lt;@U9&gt; <@U9>"

    @pytest.mark.asyncio
    async def test_allowlisted_name_with_escapable_chars_still_pings(self, adapter):
        # Display names like "R&D" are raw; matching must happen before the
        # text is escaped or the ping silently disappears.
        await adapter.send_message("T1|C1|", "hey @R&D, see <@U7>", (("R&D", "U7"),))
        text = adapter._clients["T1"].chat_postMessage.await_args.kwargs["text"]
        assert text == "hey <@U7>, see &lt;@U7&gt;"

    @pytest.mark.asyncio
    async def test_post_channel_message_chunks_before_escaping(self, adapter):
        # A hard cut through escaped text can bisect an entity ("…ab&amp" /
        # ";ab…"); chunking the canonical text first can't.
        await adapter.post_channel_message("T1|C1|", "ab&" * 2000)
        calls = adapter._clients["T1"].chat_postMessage.await_args_list
        assert len(calls) >= 2
        for c in calls:
            sent = c.kwargs["text"]
            assert re.search(r"&(?!amp;)", sent) is None
            assert not sent.startswith(";")
            # Slack counts the escaped wire text, so expansion must re-split
            # rather than blow through the cap.
            assert len(sent) <= config.MAX_MESSAGE_LENGTH
        assert "".join(c.kwargs["text"] for c in calls) == "ab&amp;" * 2000

    @pytest.mark.asyncio
    async def test_send_message_resplits_when_escaping_expands_past_the_cap(
        self, adapter
    ):
        # The streamed reply path flushes canonical chunks; Slack counts the
        # escaped wire text, so expansion must re-split here too.
        await adapter.send_message("T1|C1|", ("a & b " * 640).strip())
        calls = adapter._clients["T1"].chat_postMessage.await_args_list
        assert len(calls) >= 2
        for c in calls:
            assert len(c.kwargs["text"]) <= config.MAX_MESSAGE_LENGTH

    @pytest.mark.asyncio
    async def test_mention_as_link_label_stays_plain_inside_the_link(self, adapter):
        # A live <@Uid> inside <url|label> would nest angle brackets and
        # truncate the label at Slack's first ">".
        await adapter.send_message(
            "T1|C1|", "see [@Bently](https://x.com) now", (("Bently", "U9"),)
        )
        text = adapter._clients["T1"].chat_postMessage.await_args.kwargs["text"]
        assert text == "see <https://x.com|@Bently> now"

    @pytest.mark.asyncio
    async def test_forged_nul_placeholder_cannot_ping(self, adapter):
        # NULs in model output are stripped before the stash, so a forged
        # placeholder can never be unwrapped into a live mention.
        await adapter.send_message("T1|C1|", "\x00U9\x00 hi", ())
        text = adapter._clients["T1"].chat_postMessage.await_args.kwargs["text"]
        assert "<@U9>" not in text and "\x00" not in text

    @pytest.mark.asyncio
    async def test_partial_chunked_post_keeps_what_landed(self, adapter):
        # Raising past the first chunk makes deliver_message report the whole
        # post failed, and the model reposts — duplicating the chunks that
        # already landed. Keep the partial result instead.
        client = adapter._clients["T1"]
        client.chat_postMessage = AsyncMock(
            side_effect=[
                {"ts": "111.222"},
                SlackApiError("ratelimited", MagicMock()),
            ]
        )
        ref = await adapter.post_channel_message("T1|C1|", "ab&" * 2000)
        assert ref is not None
        assert ref.id == "111.222"

    @pytest.mark.asyncio
    async def test_first_chunk_failure_still_raises(self, adapter):
        # Nothing landed, so the caller must hear about it and retry.
        client = adapter._clients["T1"]
        client.chat_postMessage = AsyncMock(
            side_effect=SlackApiError("ratelimited", MagicMock())
        )
        with pytest.raises(SlackApiError):
            await adapter.post_channel_message("T1|C1|", "ab&" * 2000)

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
        lookup = AsyncMock(return_value=install)
        with (
            patch(
                "backend.copilot.bot.adapters.slack.adapter.bot_installs_db",
                return_value=MagicMock(get_bot_install=lookup),
            ),
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
                "backend.copilot.bot.adapters.slack.adapter.bot_installs_db",
                return_value=MagicMock(
                    get_bot_install=AsyncMock(return_value=None),
                    is_install_revoked=AsyncMock(return_value=False),
                ),
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
                "backend.copilot.bot.adapters.slack.adapter.bot_installs_db",
                return_value=MagicMock(get_bot_install=AsyncMock(return_value=install)),
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
                "backend.copilot.bot.adapters.slack.adapter.bot_installs_db",
                return_value=MagicMock(
                    get_bot_install=AsyncMock(return_value=None),
                    is_install_revoked=AsyncMock(return_value=True),
                ),
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
                "backend.copilot.bot.adapters.slack.adapter.bot_installs_db",
                return_value=MagicMock(get_bot_install=lookup),
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
