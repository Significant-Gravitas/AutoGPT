"""Tests for the Teams adapter: inbound auth, activity mapping, and sends."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from backend.copilot.bot.adapters.teams import auth, commands, config
from backend.copilot.bot.adapters.teams.adapter import (
    MESSAGES_PATH,
    TeamsAdapter,
    _inbound_files,
)
from backend.copilot.bot.adapters.teams.text import mention_entities, to_teams_markdown
from backend.util.settings import AppEnvironment

_APP_ID = "11111111-2222-3333-4444-555555555555"
_SERVICE_URL = "https://smba.trafficmanager.net/teams/"
_CONFIG_PATH = "backend.copilot.bot.adapters.teams.config"


@pytest.fixture
def app_id():
    with patch(f"{_CONFIG_PATH}.get_app_id", return_value=_APP_ID):
        yield _APP_ID


@pytest.fixture(autouse=True)
def dedupe_redis():
    """Fake the dedupe store: every activity claims its id first try."""
    redis = MagicMock()
    redis.set = AsyncMock(return_value=True)
    with patch(
        "backend.copilot.bot.adapters.teams.adapter.get_redis_async",
        AsyncMock(return_value=redis),
    ):
        yield redis


@pytest.fixture(scope="module")
def signing_key():
    """A throwaway RSA key standing in for Microsoft's signing key."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(signing_key, **overrides) -> str:
    claims = {
        "iss": config.TOKEN_ISSUER,
        "aud": _APP_ID,
        "serviceurl": _SERVICE_URL,
        "exp": int(time.time()) + 600,
        "iat": int(time.time()),
        **overrides,
    }
    return jwt.encode(claims, signing_key, algorithm="RS256", headers={"kid": "test"})


def _validator_with(signing_key) -> auth.ConnectorTokenValidator:
    """A validator pre-seeded with our test key, skipping the JWKS fetch."""
    validator = auth.ConnectorTokenValidator()
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key(), as_dict=True)
    jwk["kid"] = "test"
    validator._keys = {"test": jwk}
    validator._fetched_at = time.monotonic()
    return validator


def _activity(**overrides) -> dict:
    return {
        "type": "message",
        "id": "activity-1",
        "serviceUrl": _SERVICE_URL,
        "text": "hello",
        "from": {"id": "29:user", "name": "Ada"},
        "conversation": {"id": "a:chat", "conversationType": "personal"},
        **overrides,
    }


# ── Inbound authentication ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_valid_token_is_accepted(app_id, signing_key):
    claims = await _validator_with(signing_key).validate(
        f"Bearer {_token(signing_key)}"
    )
    assert claims["aud"] == _APP_ID


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header", [None, "", "Basic abc", "Bearer", "Bearer   ", "token abc"]
)
async def test_non_bearer_headers_are_rejected(app_id, signing_key, header):
    with pytest.raises(auth.TeamsAuthError):
        await _validator_with(signing_key).validate(header)


@pytest.mark.asyncio
async def test_algorithm_is_pinned_before_key_lookup(app_id, signing_key):
    # A token asking to be verified with HMAC must be refused outright — never
    # let the token choose its own algorithm family.
    forged = jwt.encode({"aud": _APP_ID}, "secret", algorithm="HS256")
    with pytest.raises(auth.TeamsAuthError, match="algorithm"):
        await _validator_with(signing_key).validate(f"Bearer {forged}")


@pytest.mark.asyncio
async def test_wrong_audience_is_rejected(app_id, signing_key):
    other = _token(signing_key, aud="00000000-0000-0000-0000-000000000000")
    with pytest.raises(auth.TeamsAuthError):
        await _validator_with(signing_key).validate(f"Bearer {other}")


@pytest.mark.asyncio
async def test_expired_token_is_rejected(app_id, signing_key):
    stale = _token(signing_key, exp=int(time.time()) - 3600)
    with pytest.raises(auth.TeamsAuthError):
        await _validator_with(signing_key).validate(f"Bearer {stale}")


def test_service_url_claim_is_read_from_the_lowercase_key():
    # The wire claim is "serviceurl" even though the docs spell it serviceUrl.
    # Reading the camelCase spelling yields None, which would silently disable
    # this check entirely — hence a dedicated test.
    auth.verify_service_url({"serviceurl": _SERVICE_URL}, _SERVICE_URL)
    with pytest.raises(auth.TeamsAuthError):
        auth.verify_service_url({"serviceUrl": _SERVICE_URL}, _SERVICE_URL)


def test_service_url_mismatch_is_rejected():
    with pytest.raises(auth.TeamsAuthError):
        auth.verify_service_url({"serviceurl": _SERVICE_URL}, "https://evil.example/")


def test_service_url_trailing_slash_is_not_significant():
    auth.verify_service_url({"serviceurl": _SERVICE_URL.rstrip("/")}, _SERVICE_URL)


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("https://smba.trafficmanager.net/teams/", True),
        ("https://europe.botframework.com/", True),
        ("https://gov.botframework.azure.us/", True),
        ("https://evil.example.com/", False),
        # Suffix confusion: a lookalike host must not pass.
        ("https://smba.trafficmanager.net.evil.com/", False),
        # trafficmanager.net is customer-registerable — only the exact
        # Connector host is trusted, never the zone.
        ("https://evil-profile.trafficmanager.net/", False),
        # A bearer token never goes out in cleartext.
        ("http://smba.trafficmanager.net/teams/", False),
        ("", False),
        ("not-a-url", False),
    ],
)
def test_service_url_allowlist(url, allowed):
    assert auth.is_allowed_service_url(url) is allowed


# ── Activity -> MessageContext ─────────────────────────────────────


@pytest.mark.asyncio
async def test_personal_chat_maps_to_dm(app_id):
    ctx = await TeamsAdapter(MagicMock())._build_context(_activity())
    assert ctx is not None
    assert ctx.channel_type == "dm"
    assert ctx.server_id is None
    assert ctx.channel_id == "a:chat"


@pytest.mark.asyncio
async def test_channel_message_carries_team_as_server_id(app_id):
    activity = _activity(
        conversation={"id": "19:room@thread.tacv2", "conversationType": "channel"},
        channelData={"team": {"id": "19:team@thread.tacv2", "name": "Eng"}},
    )
    ctx = await TeamsAdapter(MagicMock())._build_context(activity)
    assert ctx is not None
    assert ctx.channel_type == "channel"
    assert ctx.server_id == "19:team@thread.tacv2"


@pytest.mark.asyncio
async def test_reply_chain_maps_to_thread(app_id):
    activity = _activity(
        conversation={
            "id": "19:room@thread.tacv2;messageid=1700",
            "conversationType": "channel",
        },
        channelData={"team": {"id": "19:team@thread.tacv2"}},
    )
    ctx = await TeamsAdapter(MagicMock())._build_context(activity)
    assert ctx is not None
    assert ctx.channel_type == "thread"


@pytest.mark.asyncio
async def test_top_level_post_with_own_messageid_maps_to_channel(app_id):
    # Real Teams suffixes even a brand-new top-level post with ;messageid= —
    # of the post itself. That must classify as a channel message, not a
    # thread reply, or the channel branch never runs in production.
    activity = _activity(
        id="1700",
        conversation={
            "id": "19:room@thread.tacv2;messageid=1700",
            "conversationType": "channel",
        },
        channelData={"team": {"id": "19:team@thread.tacv2"}},
    )
    ctx = await TeamsAdapter(MagicMock())._build_context(activity)
    assert ctx is not None
    assert ctx.channel_type == "channel"


def test_derived_thread_ids_reuse_the_learned_service_url(app_id):
    # create_thread mints "<base>;messageid=<n>" — replies there must go to
    # the regional host learned from inbound traffic, not the geo-routed
    # default, or an EMEA tenant served from a US deployment 404s.
    adapter = TeamsAdapter(MagicMock())
    adapter._remember_service_url(
        _activity(
            serviceUrl="https://emea.botframework.com/",
            conversation={"id": "19:room@thread.tacv2;messageid=5"},
        )
    )
    assert (
        adapter._service_url_for("19:room@thread.tacv2;messageid=999")
        == "https://emea.botframework.com/"
    )
    assert (
        adapter._service_url_for("19:room@thread.tacv2")
        == "https://emea.botframework.com/"
    )
    assert adapter._service_url_for("19:other@thread.tacv2") == (
        "https://smba.trafficmanager.net/teams/"
    )


@pytest.mark.asyncio
async def test_group_chat_is_skipped(app_id):
    # groupChat has no team identity, and the core handler drops any non-DM
    # turn without a server_id — so it is refused here rather than silently.
    activity = _activity(
        conversation={"id": "19:group@thread.v2", "conversationType": "groupChat"}
    )
    assert await TeamsAdapter(MagicMock())._build_context(activity) is None


@pytest.mark.asyncio
async def test_channel_without_team_is_skipped(app_id):
    activity = _activity(
        conversation={"id": "19:room@thread.tacv2", "conversationType": "channel"}
    )
    assert await TeamsAdapter(MagicMock())._build_context(activity) is None


@pytest.mark.asyncio
async def test_bot_mention_is_stripped_and_flagged(app_id):
    activity = _activity(
        text="<at>AutoGPT</at> summarise this",
        entities=[
            {
                "type": "mention",
                "text": "<at>AutoGPT</at>",
                "mentioned": {"id": _APP_ID, "name": "AutoGPT"},
            }
        ],
    )
    ctx = await TeamsAdapter(MagicMock())._build_context(activity)
    assert ctx is not None
    assert ctx.text == "summarise this"
    assert ctx.bot_mentioned is True


@pytest.mark.asyncio
@pytest.mark.parametrize("mentioned_id", [_APP_ID, f"28:{_APP_ID}"])
async def test_bot_mention_is_detected_in_either_spelling(app_id, mentioned_id):
    # Teams prefixes a participant id with its type, and the Bot Framework SDK
    # matches mentions against the prefixed ``recipient.id``. Reading only the
    # bare app id would mean the bot never sees an @mention — and channel turns
    # only run when mentioned, so it would be silent in every channel.
    activity = _activity(
        text="<at>AutoGPT</at> summarise this",
        recipient={"id": f"28:{_APP_ID}", "name": "AutoGPT"},
        entities=[
            {
                "type": "mention",
                "text": "<at>AutoGPT</at>",
                "mentioned": {"id": mentioned_id, "name": "AutoGPT"},
            }
        ],
    )
    ctx = await TeamsAdapter(MagicMock())._build_context(activity)
    assert ctx is not None
    assert ctx.bot_mentioned is True
    assert ctx.text == "summarise this"


@pytest.mark.asyncio
async def test_bot_mention_is_detected_from_recipient_when_app_id_is_unset():
    # Playground mode configures no app id; the activity's own recipient is
    # then the only thing identifying us.
    activity = _activity(
        text="<at>AutoGPT</at> hi",
        recipient={"id": "28:playground-bot", "name": "AutoGPT"},
        entities=[
            {
                "type": "mention",
                "text": "<at>AutoGPT</at>",
                "mentioned": {"id": "28:playground-bot", "name": "AutoGPT"},
            }
        ],
    )
    with patch(f"{_CONFIG_PATH}.get_app_id", return_value=""):
        ctx = await TeamsAdapter(MagicMock())._build_context(activity)
    assert ctx is not None
    assert ctx.bot_mentioned is True


@pytest.mark.asyncio
async def test_a_users_mention_is_not_mistaken_for_the_bot(app_id):
    activity = _activity(
        text="<at>Bob</at> hi",
        recipient={"id": f"28:{_APP_ID}", "name": "AutoGPT"},
        entities=[
            {
                "type": "mention",
                "text": "<at>Bob</at>",
                "mentioned": {"id": "29:bob", "name": "Bob"},
            }
        ],
    )
    ctx = await TeamsAdapter(MagicMock())._build_context(activity)
    assert ctx is not None
    assert ctx.bot_mentioned is False
    assert ctx.mentionable_users == (("Bob", "29:bob"),)


@pytest.mark.asyncio
async def test_mentionable_users_exclude_the_bot(app_id):
    activity = _activity(
        entities=[
            {
                "type": "mention",
                "text": "<at>AutoGPT</at>",
                "mentioned": {"id": _APP_ID, "name": "AutoGPT"},
            },
            {
                "type": "mention",
                "text": "<at>Grace</at>",
                "mentioned": {"id": "29:grace", "name": "Grace"},
            },
        ],
    )
    ctx = await TeamsAdapter(MagicMock())._build_context(activity)
    assert ctx is not None
    assert ctx.mentionable_users == (("Grace", "29:grace"),)


# ── Threading ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conversation_id,expected",
    [
        ("19:room@thread.tacv2", "19:room@thread.tacv2;messageid=1700"),
        # Already a reply-chain: re-wrapping would address a conversation that
        # does not exist.
        ("19:room@thread.tacv2;messageid=1", None),
        # Personal chats have no threads.
        ("a:chat", None),
    ],
)
async def test_create_thread_rules(app_id, conversation_id, expected):
    adapter = TeamsAdapter(MagicMock())
    assert await adapter.create_thread(conversation_id, "1700", "Title") == expected


# ── Sending ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_message_posts_markdown_activity(app_id):
    adapter = TeamsAdapter(MagicMock())
    adapter._client.send_activity = AsyncMock(return_value="activity-9")
    await adapter.send_message("a:chat", "# Title\n\nbody")
    activity = adapter._client.send_activity.await_args.args[2]
    assert activity["type"] == "message"
    assert activity["textFormat"] == "markdown"
    # Headings do not render in Teams; they are downgraded to bold.
    assert "**Title**" in activity["text"]
    assert "entities" not in activity


@pytest.mark.asyncio
async def test_send_message_attaches_entities_only_for_allowlisted_mentions(app_id):
    adapter = TeamsAdapter(MagicMock())
    adapter._client.send_activity = AsyncMock(return_value="activity-9")
    await adapter.send_message(
        "a:chat", "@Grace and @Mallory", mentionable_users=(("Grace", "29:grace"),)
    )
    activity = adapter._client.send_activity.await_args.args[2]
    assert activity["entities"] == [
        {
            "type": "mention",
            "text": "<at>Grace</at>",
            "mentioned": {"id": "29:grace", "name": "Grace"},
        }
    ]
    # The un-allowlisted name stays plain text and cannot ping.
    assert "<at>Mallory</at>" not in activity["text"]


@pytest.mark.asyncio
async def test_send_link_uses_an_adaptive_card_button(app_id):
    adapter = TeamsAdapter(MagicMock())
    adapter._client.send_activity = AsyncMock(return_value="activity-9")
    await adapter.send_link("a:chat", "Link up", "Connect", "https://example.com/link")
    card = adapter._client.send_activity.await_args.args[2]["attachments"][0]
    assert card["contentType"] == "application/vnd.microsoft.card.adaptive"
    # Pinned to 1.2 — the highest version every Teams client renders.
    assert card["content"]["version"] == "1.2"
    assert card["content"]["actions"][0]["url"] == "https://example.com/link"


@pytest.mark.asyncio
async def test_non_image_file_degrades_to_a_note(app_id):
    from backend.copilot.bot.adapters.base import FileAttachment

    adapter = TeamsAdapter(MagicMock())
    adapter._client.send_activity = AsyncMock(return_value="activity-9")
    await adapter.send_file(
        "a:chat",
        "here you go",
        FileAttachment(
            filename="report.pdf", mime_type="application/pdf", content=b"%PDF"
        ),
    )
    text = adapter._client.send_activity.await_args.args[2]["text"]
    assert "report.pdf" in text
    assert "attachments" not in adapter._client.send_activity.await_args.args[2]


@pytest.mark.asyncio
async def test_image_file_is_inlined(app_id):
    from backend.copilot.bot.adapters.base import FileAttachment

    adapter = TeamsAdapter(MagicMock())
    adapter._client.send_activity = AsyncMock(return_value="activity-9")
    await adapter.send_file(
        "a:chat",
        "chart",
        FileAttachment(filename="c.png", mime_type="image/png", content=b"\x89PNG"),
    )
    attachment = adapter._client.send_activity.await_args.args[2]["attachments"][0]
    assert attachment["contentUrl"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_oversized_image_degrades_to_a_note(app_id):
    # A Teams activity is capped at ~28KB total; an image too big to inline
    # must fall back to text, never a doomed multi-hundred-KB activity.
    from backend.copilot.bot.adapters.base import FileAttachment

    adapter = TeamsAdapter(MagicMock())
    adapter._client.send_activity = AsyncMock(return_value="activity-9")
    await adapter.send_file(
        "a:chat",
        "chart",
        FileAttachment(
            filename="big.png",
            mime_type="image/png",
            content=b"\x89" * (config.INLINE_IMAGE_BYTES + 1),
        ),
    )
    posted = adapter._client.send_activity.await_args.args[2]
    assert "attachments" not in posted
    assert "can't be attached" in posted["text"]


@pytest.mark.asyncio
async def test_rejected_inline_image_degrades_to_a_note(app_id):
    from backend.copilot.bot.adapters.base import FileAttachment
    from backend.copilot.bot.adapters.teams.api_client import TeamsApiError

    adapter = TeamsAdapter(MagicMock())
    adapter._client.send_activity = AsyncMock(
        side_effect=[TeamsApiError("413"), "activity-9"]
    )
    await adapter.send_file(
        "a:chat",
        "chart",
        FileAttachment(filename="c.png", mime_type="image/png", content=b"\x89PNG"),
    )
    posted = adapter._client.send_activity.await_args.args[2]
    assert "attachments" not in posted
    assert "can't be attached" in posted["text"]


@pytest.mark.asyncio
async def test_open_dm_channel_uses_the_participant_bot_id(app_id):
    adapter = TeamsAdapter(MagicMock())
    adapter._client.create_conversation = AsyncMock(return_value="a:new")
    with patch(f"{_CONFIG_PATH}.get_tenant_id", return_value="tenant-1"):
        assert await adapter.open_dm_channel("29:user") == "a:new"
    payload = adapter._client.create_conversation.await_args.args[1]
    assert payload["bot"]["id"] == f"28:{_APP_ID}"
    assert payload["members"] == [{"id": "29:user"}]


# ── Markup localization ────────────────────────────────────────────


def test_headings_become_bold():
    assert to_teams_markdown("## Results") == "**Results**"


def test_tables_are_fenced_to_keep_alignment():
    out = to_teams_markdown("| a | b |\n|---|---|\n| 1 | 2 |")
    assert out.startswith("```")
    assert out.rstrip().endswith("```")


def test_code_blocks_are_left_alone():
    src = "```python\n# comment\n| not | a table |\n```"
    assert to_teams_markdown(src) == src


def test_mention_entities_ignore_unknown_ids():
    assert mention_entities(["29:nobody"], (("Grace", "29:grace"),)) == []


# ── Commands ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("/setup", "setup"),
        ("/help extra words", "help"),
        ("/UNLINK", "unlink"),
        ("/unknown", None),
        ("no command here", None),
        ("", None),
    ],
)
def test_parse_command(text, expected):
    assert commands.parse_command(text) == expected


@pytest.mark.asyncio
async def test_setup_outside_a_team_explains_where_to_go(app_id):
    adapter = MagicMock()
    adapter.send_message = AsyncMock()
    adapter.send_link = AsyncMock()
    await commands.handle(MagicMock(), adapter, _activity(), "setup")
    message = adapter.send_message.await_args.args[1]
    assert "channel" in message.lower()
    adapter.send_link.assert_not_awaited()


@pytest.mark.asyncio
async def test_setup_in_a_team_links_the_team(app_id):
    from backend.copilot.bot.command_core import CommandReply

    adapter = MagicMock()
    adapter.send_message = AsyncMock()
    adapter.send_link = AsyncMock()
    activity = _activity(
        text="/setup",
        conversation={"id": "19:room@thread.tacv2", "conversationType": "channel"},
        channelData={"team": {"id": "19:team@thread.tacv2", "name": "Eng"}},
    )
    reply = CommandReply(
        text="Set up", button_label="Link Team", button_url="https://link.example/t"
    )
    with patch(
        "backend.copilot.bot.adapters.teams.commands.setup_reply",
        AsyncMock(return_value=reply),
    ) as setup:
        await commands.handle(MagicMock(), adapter, activity, "setup")
    kwargs = setup.await_args.kwargs
    assert kwargs["platform_server_id"] == "19:team@thread.tacv2"
    assert kwargs["server_name"] == "Eng"
    assert kwargs["server_noun"] == "team"
    adapter.send_link.assert_awaited_once_with(
        "19:room@thread.tacv2", "Set up", "Link Team", "https://link.example/t"
    )


@pytest.mark.asyncio
async def test_setup_asks_the_connector_for_a_missing_team_name():
    from backend.copilot.bot.command_core import CommandReply

    # Teams stamps the team name onto install activities but not onto the
    # message carrying /setup, so without the lookup the link is stored
    # nameless and settings can only show the raw thread id.
    adapter = MagicMock()
    adapter.send_message = AsyncMock()
    adapter.send_link = AsyncMock()
    adapter.client.get_team_details = AsyncMock(return_value={"name": "Engineering"})
    activity = _activity(
        text="/setup",
        conversation={"id": "19:room@thread.tacv2", "conversationType": "channel"},
        channelData={"team": {"id": "19:team@thread.tacv2"}},
    )
    with patch(
        "backend.copilot.bot.adapters.teams.commands.setup_reply",
        AsyncMock(return_value=CommandReply(text="Set up")),
    ) as setup:
        await commands.handle(MagicMock(), adapter, activity, "setup")

    adapter.client.get_team_details.assert_awaited_once_with(
        activity["serviceUrl"], "19:team@thread.tacv2"
    )
    assert setup.await_args.kwargs["server_name"] == "Engineering"


@pytest.mark.asyncio
async def test_setup_does_not_call_the_connector_when_the_name_is_present():
    from backend.copilot.bot.command_core import CommandReply

    adapter = MagicMock()
    adapter.send_message = AsyncMock()
    adapter.send_link = AsyncMock()
    adapter.client.get_team_details = AsyncMock()
    activity = _activity(
        text="/setup",
        conversation={"id": "19:room@thread.tacv2", "conversationType": "channel"},
        channelData={"team": {"id": "19:team@thread.tacv2", "name": "Eng"}},
    )
    with patch(
        "backend.copilot.bot.adapters.teams.commands.setup_reply",
        AsyncMock(return_value=CommandReply(text="Set up")),
    ) as setup:
        await commands.handle(MagicMock(), adapter, activity, "setup")

    adapter.client.get_team_details.assert_not_awaited()
    assert setup.await_args.kwargs["server_name"] == "Eng"


@pytest.mark.asyncio
async def test_setup_survives_a_failed_team_lookup():
    from backend.copilot.bot.command_core import CommandReply

    # A nameless link still works; the settings page falls back to the id.
    adapter = MagicMock()
    adapter.send_message = AsyncMock()
    adapter.send_link = AsyncMock()
    adapter.client.get_team_details = AsyncMock(side_effect=RuntimeError("connector"))
    activity = _activity(
        text="/setup",
        conversation={"id": "19:room@thread.tacv2", "conversationType": "channel"},
        channelData={"team": {"id": "19:team@thread.tacv2"}},
    )
    with patch(
        "backend.copilot.bot.adapters.teams.commands.setup_reply",
        AsyncMock(return_value=CommandReply(text="Set up")),
    ) as setup:
        await commands.handle(MagicMock(), adapter, activity, "setup")

    assert setup.await_args.kwargs["server_name"] == ""


# ── Factory gating ─────────────────────────────────────────────────


def test_adapter_is_not_built_without_credentials():
    from backend.copilot.bot.webhook_routes import build_webhook_adapters

    with (
        patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False),
        patch(f"{_CONFIG_PATH}.get_app_id", return_value=""),
        patch(f"{_CONFIG_PATH}.get_app_password", return_value=""),
        patch(f"{_CONFIG_PATH}.get_tenant_id", return_value=""),
        patch(
            "backend.copilot.bot.adapters.slack.config.get_signing_secret",
            return_value="",
        ),
        patch(
            "backend.copilot.bot.adapters.telegram.config.get_bot_token",
            return_value="",
        ),
    ):
        assert build_webhook_adapters(MagicMock()) == []


def _settings_with(app_env, flag):
    settings = MagicMock()
    settings.config.app_env = app_env
    settings.config.autopilot_bot_teams_allow_unverified = flag
    return settings


@pytest.mark.parametrize(
    "app_env,flag,expected",
    [
        (AppEnvironment.LOCAL, True, True),
        # The flag alone must never be enough on a deployed environment.
        (AppEnvironment.DEVELOPMENT, True, False),
        (AppEnvironment.PRODUCTION, True, False),
        # Nor is being local enough without the explicit opt-in.
        (AppEnvironment.LOCAL, False, False),
    ],
)
def test_playground_bypass_is_double_gated(app_env, flag, expected):
    with patch(f"{_CONFIG_PATH}.Settings", return_value=_settings_with(app_env, flag)):
        assert config.allow_unverified_requests() is expected


def test_loopback_service_url_only_allowed_in_playground_mode():
    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=True):
        assert auth.is_allowed_service_url("http://localhost:56150/") is True
    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False):
        assert auth.is_allowed_service_url("http://localhost:56150/") is False


_AUTH_PATH = "backend.copilot.bot.adapters.teams.auth"

_TEAM = {"id": "19:team@thread.tacv2", "name": "Eng"}


def _membership_activity(**overrides) -> dict:
    return {
        "type": "installationUpdate",
        "recipient": {"id": f"28:{_APP_ID}", "name": "AutoGPT"},
        "conversation": {"id": "19:room@thread.tacv2"},
        "channelData": {"team": _TEAM},
        **overrides,
    }


@pytest.mark.asyncio
async def test_install_puts_the_team_on_the_roster(app_id):
    # Without this the roster has no row, so admin analytics show the raw
    # team id instead of its name.
    api = MagicMock()
    adapter = TeamsAdapter(api)
    await adapter._dispatch_activity(_membership_activity(action="add"))
    api.track_guild_joined.assert_called_once_with("teams", _TEAM["id"], "Eng")
    api.track_guild_left.assert_not_called()


@pytest.mark.asyncio
async def test_uninstall_marks_the_team_left(app_id):
    api = MagicMock()
    adapter = TeamsAdapter(api)
    await adapter._dispatch_activity(_membership_activity(action="remove"))
    api.track_guild_left.assert_called_once_with("teams", _TEAM["id"])
    api.track_guild_joined.assert_not_called()


@pytest.mark.asyncio
async def test_bot_added_via_conversation_update_is_tracked(app_id):
    api = MagicMock()
    adapter = TeamsAdapter(api)
    await adapter._dispatch_activity(
        _membership_activity(
            type="conversationUpdate",
            membersAdded=[{"id": f"28:{_APP_ID}"}],
        )
    )
    api.track_guild_joined.assert_called_once_with("teams", _TEAM["id"], "Eng")


@pytest.mark.asyncio
async def test_other_people_joining_does_not_touch_the_roster(app_id):
    # membersAdded fires for every new team member; only the bot's own
    # add/remove should move the roster.
    api = MagicMock()
    adapter = TeamsAdapter(api)
    await adapter._dispatch_activity(
        _membership_activity(type="conversationUpdate", membersAdded=[{"id": "29:bob"}])
    )
    api.track_guild_joined.assert_not_called()
    api.track_guild_left.assert_not_called()


@pytest.mark.asyncio
async def test_personal_install_is_not_put_on_the_roster(app_id):
    api = MagicMock()
    adapter = TeamsAdapter(api)
    await adapter._dispatch_activity(_membership_activity(action="add", channelData={}))
    api.track_guild_joined.assert_not_called()
    api.track_guild_left.assert_not_called()


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:56150/", "http://host.docker.internal:56150/"),
        ("http://127.0.0.1:56150/", "http://host.docker.internal:56150/"),
        # No port to preserve.
        ("http://localhost/", "http://host.docker.internal/"),
        # Paths and queries must survive the host swap.
        (
            "http://localhost:56150/v3/x?a=1",
            "http://host.docker.internal:56150/v3/x?a=1",
        ),
        # A real Connector host is never touched.
        (
            "https://smba.trafficmanager.net/teams/",
            "https://smba.trafficmanager.net/teams/",
        ),
    ],
)
def test_loopback_is_rewritten_to_the_container_host(url, expected):
    with (
        patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=True),
        patch(f"{_AUTH_PATH}._in_container", return_value=True),
    ):
        assert auth.rewrite_loopback_for_container(url) == expected


@pytest.mark.parametrize(
    "unverified,in_container",
    [(False, True), (True, False), (False, False)],
)
def test_loopback_rewrite_needs_both_playground_mode_and_a_container(
    unverified, in_container
):
    url = "http://localhost:56150/"
    with (
        patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=unverified),
        patch(f"{_AUTH_PATH}._in_container", return_value=in_container),
    ):
        assert auth.rewrite_loopback_for_container(url) == url


def test_container_host_alias_is_dialable_only_in_playground_mode():
    url = "http://host.docker.internal:56150/"
    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=True):
        assert auth.is_allowed_service_url(url) is True
    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False):
        assert auth.is_allowed_service_url(url) is False


def test_rejection_bodies_do_not_echo_the_reason(app_id):
    """The caller is unauthenticated, so the body says nothing specific.

    The validator wraps the JWT parser's own exception text, which would
    otherwise be handed straight back to whoever is probing the endpoint.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    adapter = TeamsAdapter(MagicMock())
    adapter.register_routes(app)
    client = TestClient(app)

    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False):
        with patch.object(
            adapter._validator,
            "validate",
            AsyncMock(
                side_effect=auth.TeamsAuthError(
                    "malformed token: Invalid header string: codec can't decode 0x9e"
                )
            ),
        ):
            rejected = client.post(MESSAGES_PATH, json=_activity())
        malformed = client.post(
            MESSAGES_PATH,
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )

    assert rejected.status_code == 401
    assert "codec" not in rejected.text
    assert "malformed token" not in rejected.text
    # A parse failure must not hand back the payload or the parser's message.
    assert malformed.status_code in (400, 401)
    assert "not json" not in malformed.text


async def _drain(adapter: TeamsAdapter) -> None:
    """Await the adapter's fire-and-forget dispatch tasks.

    ``_handle_messages_request`` answers 200 and dispatches in a background
    task, so asserting on the handler's side effects immediately is a race.
    Polling from the test thread does not fix it: without a context manager
    the TestClient builds a portal per request and tears it down when the call
    returns, taking the loop that would run the dispatch with it.
    """
    while adapter._activity_tasks:
        await asyncio.gather(*tuple(adapter._activity_tasks))


def test_unauthenticated_request_is_rejected_when_not_in_playground_mode(app_id):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    adapter = TeamsAdapter(MagicMock())
    adapter.register_routes(app)
    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False):
        response = TestClient(app).post(MESSAGES_PATH, json=_activity())
    assert response.status_code == 401


def test_route_accepts_a_properly_signed_activity(app_id, signing_key):
    """End-to-end through the real HTTP route: signed token -> dispatch."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    adapter = TeamsAdapter(MagicMock())
    adapter._validator = _validator_with(signing_key)
    seen = []
    adapter.on_message(lambda ctx, _adapter: _record(seen, ctx))
    adapter.register_routes(app)

    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False):
        # One portal for both the request and the drain, so the dispatch has
        # a live loop to run on.
        with TestClient(app) as client:
            response = client.post(
                MESSAGES_PATH,
                json=_activity(),
                headers={"Authorization": f"Bearer {_token(signing_key)}"},
            )
            client.portal.call(_drain, adapter)
    assert response.status_code == 200
    assert [ctx.text for ctx in seen] == ["hello"]


@pytest.mark.asyncio
async def test_route_rejects_a_body_whose_service_url_was_swapped(app_id, signing_key):
    """A captured token replayed with a redirected serviceUrl must not pass."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    adapter = TeamsAdapter(MagicMock())
    adapter._validator = _validator_with(signing_key)
    adapter.register_routes(app)

    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False):
        response = TestClient(app).post(
            MESSAGES_PATH,
            json=_activity(serviceUrl="https://attacker.example/"),
            headers={"Authorization": f"Bearer {_token(signing_key)}"},
        )
    assert response.status_code == 401


async def _record(seen: list, ctx) -> None:
    seen.append(ctx)


# ── Activity dedupe ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_redelivered_activity_is_dispatched_once(app_id, dedupe_redis):
    # The Connector retries slow deliveries, and a captured request can be
    # replayed for the token's lifetime — both arrive as a repeated id.
    dedupe_redis.set = AsyncMock(side_effect=[True, None])
    adapter = TeamsAdapter(MagicMock())
    seen = []
    adapter.on_message(lambda ctx, _a: _record(seen, ctx))
    await adapter._dispatch_activity(_activity(text="original"))
    await adapter._dispatch_activity(_activity(text="redelivery"))
    # The FIRST delivery survives — dropping it and keeping the retry would
    # also pass a bare count assertion.
    assert [ctx.text for ctx in seen] == ["original"]


@pytest.mark.asyncio
async def test_activity_without_id_is_still_processed(app_id, dedupe_redis):
    adapter = TeamsAdapter(MagicMock())
    seen = []
    adapter.on_message(lambda ctx, _a: _record(seen, ctx))
    await adapter._dispatch_activity(_activity(id=""))
    assert len(seen) == 1
    dedupe_redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_dedupe_fails_open_when_redis_is_down(app_id, dedupe_redis):
    dedupe_redis.set = AsyncMock(side_effect=ConnectionError("redis down"))
    adapter = TeamsAdapter(MagicMock())
    seen = []
    adapter.on_message(lambda ctx, _a: _record(seen, ctx))
    await adapter._dispatch_activity(_activity())
    assert len(seen) == 1


# ── Inbound attachments ────────────────────────────────────────────


def _file_attachment(name: str, **content) -> dict:
    return {
        "contentType": "application/vnd.microsoft.teams.file.download.info",
        "name": name,
        "content": {"downloadUrl": f"https://files.example/{name}", **content},
    }


def test_inbound_files_are_not_capped_before_the_shared_bookkeeping(app_id):
    # Teams puts a non-file text/html attachment in the same array; it must
    # not consume cap slots, and the cap itself belongs to
    # collect_attachments, which owns the "too many files" note.
    activity = _activity(
        attachments=[{"contentType": "text/html", "content": "<div>hi</div>"}]
        + [_file_attachment(f"f{i}.txt") for i in range(10)]
    )
    files = _inbound_files(activity, MagicMock())
    assert [f.filename for f in files] == [f"f{i}.txt" for i in range(10)]


def test_pasted_image_is_ingested(app_id):
    activity = _activity(
        attachments=[
            {
                "contentType": "image/png",
                "contentUrl": "https://smba.trafficmanager.net/amer/attachment/1",
            }
        ]
    )
    files = _inbound_files(activity, MagicMock())
    assert len(files) == 1
    assert files[0].mime_type == "image/png"
    assert files[0].filename == "pasted-image"


def test_pasted_image_on_a_foreign_host_never_sees_the_bearer(app_id):
    """The activity body is unsigned, so contentUrl is attacker-choosable.

    Fetching it with the bot's Connector bearer would hand that token to
    whatever host the sender names, within the token's whole validity window.
    """
    activity = _activity(
        attachments=[
            {
                "contentType": "image/png",
                "contentUrl": "https://attacker.example/collect",
            }
        ]
    )
    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False):
        assert _inbound_files(activity, MagicMock()) == []


def test_pasted_image_host_must_be_a_connector_host_not_merely_public(app_id):
    # A public HTTPS host passes the looser attachment check; the bearer path
    # has to be stricter than that.
    activity = _activity(
        attachments=[
            {"contentType": "image/png", "contentUrl": "https://example.com/x.png"}
        ]
    )
    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False):
        assert _inbound_files(activity, MagicMock()) == []
        assert auth.is_fetchable_attachment_url("https://example.com/x.png") is True


@pytest.mark.parametrize(
    "url",
    [
        "http://files.example.com/doc.txt",  # cleartext
        "https://169.254.169.254/latest/meta-data/",  # cloud metadata
        "https://127.0.0.1/secret",
        "https://10.0.0.5/internal",
        "https://[::1]/secret",
        "file:///etc/passwd",
    ],
)
def test_file_download_urls_that_point_inward_are_skipped(app_id, url):
    activity = _activity(
        attachments=[
            {
                "contentType": ("application/vnd.microsoft.teams.file.download.info"),
                "name": "doc.txt",
                "content": {"downloadUrl": url, "fileType": "txt"},
            }
        ]
    )
    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False):
        assert _inbound_files(activity, MagicMock()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    [
        "127.1",  # shorthand: ipaddress rejects it, getaddrinfo resolves it
        "2130706433",  # integer form of 127.0.0.1
        "0177.0.0.1",  # octal form of 127.0.0.1
        "localhost",  # a name, not a literal
    ],
)
async def test_attachment_hosts_that_resolve_inward_are_refused(app_id, host):
    """The URL text cannot say where a name points — only resolving can."""
    with (
        patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False),
        pytest.raises(ValueError),
    ):
        await auth.ensure_attachment_host_is_external(f"https://{host}/x")


@pytest.mark.asyncio
async def test_an_unresolvable_attachment_host_is_refused(app_id):
    with (
        patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False),
        pytest.raises(ValueError),
    ):
        await auth.ensure_attachment_host_is_external("https://no-such-host.invalid/x")


@pytest.mark.asyncio
async def test_the_playground_may_still_serve_attachments_from_loopback(app_id):
    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=True):
        await auth.ensure_attachment_host_is_external("http://localhost:56150/x")


@pytest.mark.asyncio
async def test_the_attachment_fetch_stops_at_the_size_cap(app_id):
    """The only DoS bound on downloads — Teams declares no size for a file."""
    from backend.copilot.bot.adapters.teams.adapter import _bounded_fetch

    oversized = config.MAX_ATTACHMENT_BYTES + 1

    async def chunks():
        yield b"x" * oversized

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.aiter_bytes = chunks
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=response)
    stream.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.stream = MagicMock(return_value=stream)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    fetch = _bounded_fetch("https://files.example/big.bin")
    with (
        patch(
            "backend.copilot.bot.adapters.teams.auth."
            "ensure_attachment_host_is_external",
            new=AsyncMock(),
        ),
        patch(
            "backend.copilot.bot.adapters.teams.adapter.httpx.AsyncClient",
            return_value=client,
        ),
        pytest.raises(ValueError, match="size limit"),
    ):
        await fetch()


@pytest.mark.asyncio
async def test_the_attachment_fetch_never_follows_redirects(app_id):
    # A redirect would land the fetch on a host the SSRF check never saw.
    from backend.copilot.bot.adapters.teams.adapter import _bounded_fetch

    async def chunks():
        yield b"ok"

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.aiter_bytes = chunks
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=response)
    stream.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.stream = MagicMock(return_value=stream)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    made = MagicMock(return_value=client)

    fetch = _bounded_fetch("https://files.example/ok.txt")
    with (
        patch(
            "backend.copilot.bot.adapters.teams.auth."
            "ensure_attachment_host_is_external",
            new=AsyncMock(),
        ),
        patch("backend.copilot.bot.adapters.teams.adapter.httpx.AsyncClient", made),
    ):
        assert await fetch() == b"ok"

    assert made.call_args.kwargs["follow_redirects"] is False


def test_a_normal_sharepoint_download_still_works(app_id):
    activity = _activity(attachments=[_file_attachment("notes.txt")])
    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False):
        files = _inbound_files(activity, MagicMock())
    assert [f.filename for f in files] == ["notes.txt"]


def test_file_attachment_without_download_url_is_skipped(app_id):
    activity = _activity(
        attachments=[
            {
                "contentType": "application/vnd.microsoft.teams.file.download.info",
                "name": "ghost.txt",
                "content": {},
            }
        ]
    )
    assert _inbound_files(activity, MagicMock()) == []


def test_tenant_id_is_required_to_enable_the_adapter():
    # A single-tenant registration mints tokens against its own tenant
    # authority, so app id + password alone must NOT enable the adapter.
    # The bypass is pinned off so a developer's own Playground settings can't
    # decide the result.
    with (
        patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False),
        patch(f"{_CONFIG_PATH}.get_app_id", return_value=_APP_ID),
        patch(f"{_CONFIG_PATH}.get_app_password", return_value="secret"),
        patch(f"{_CONFIG_PATH}.get_tenant_id", return_value=""),
    ):
        assert config.is_configured() is False
