"""A bot-token send block with no token, for a user who linked that platform
to the AutoGPT bot, points the turn at post_to_chat_platform instead of a
setup card asking for a token they never need."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot.adapters.base import ChannelInfo
from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.ranking import ConnectionState
from backend.copilot.capabilities.registry import build_entries
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.tools import TOOL_GROUPS, TOOL_REGISTRY, chat_platform

from ._test_data import make_session
from .chat_platform_route import POST_TOOL
from .find_capability import FindCapabilityTool
from .helpers import prepare_block_for_execution
from .models import CapabilityListResponse, ErrorResponse, SetupRequirementsResponse

_USER = "route-user"
_SESSION = "route-session"
_ROUTE = "backend.copilot.tools.chat_platform_route"


def _block(name: str) -> MagicMock:
    block = MagicMock()
    block.id = "blk-send"
    block.name = name
    block.disabled = False
    block.description = ""
    block.block_type = MagicMock()
    block.input_schema.jsonschema.return_value = {
        "type": "object",
        "properties": {"message_content": {"type": "string"}},
        "required": [],
    }
    block.input_schema.get_credentials_fields.return_value = {}
    block.input_schema.get_credentials_fields_info.return_value = {}
    return block


# The bot is in two Discord servers; the user linked only "Team".
_GENERAL = ChannelInfo(id="111", name="general", server_id="1", server_name="Team")
_DISCORD = {"channel_name": "#general", "message_content": "test"}


def _links(
    *platforms: str, dm: tuple[str, ...] = (), server_id: str = "1"
) -> MagicMock:
    db = MagicMock()
    db.list_server_links = AsyncMock(
        return_value=[
            SimpleNamespace(platform=p, platform_server_id=server_id) for p in platforms
        ]
    )
    db.list_user_links = AsyncMock(
        return_value=[SimpleNamespace(platform=p) for p in dm]
    )
    return db


async def _prepare(
    block_name: str,
    links: MagicMock,
    *,
    bot_configured: bool = True,
    unattended: bool = False,
    permissions: CopilotPermissions | None = None,
    pins: dict | None = None,
    grant_hint: str = "",
    input_data: dict | None = None,
    channels: list[ChannelInfo] | Exception | None = None,
):
    bridge = MagicMock()
    if isinstance(channels, Exception):
        bridge.list_channels = AsyncMock(side_effect=channels)
    else:
        bridge.list_channels = AsyncMock(
            return_value=[_GENERAL] if channels is None else channels
        )
    session = MagicMock()
    session.session_id = _SESSION
    session.expert_id = None
    missing = MagicMock()
    missing.provider = "discord"
    with (
        patch(
            "backend.copilot.tools.helpers.get_block",
            return_value=_block(block_name),
        ),
        patch(
            "backend.copilot.capabilities.block_meta.COPILOT_EXCLUDED_BLOCK_IDS",
            new=set(),
        ),
        patch(
            "backend.copilot.capabilities.block_meta.COPILOT_EXCLUDED_BLOCK_TYPES",
            new=set(),
        ),
        patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
            AsyncMock(return_value=({}, [missing])),
        ),
        patch(
            "backend.copilot.tools.helpers.build_missing_credentials_from_field_info",
            return_value={"credentials": {"provider": "discord"}},
        ),
        patch(
            "backend.copilot.tools.helpers.is_unattended_turn",
            return_value=unattended,
        ),
        patch(f"{_ROUTE}.platform_linking_db", return_value=links),
        patch(f"{_ROUTE}._any_chat_platform_configured", return_value=bot_configured),
        patch(
            "backend.copilot.tools.helpers.turn_credential_pins",
            return_value=pins or {},
        ),
        patch(
            "backend.copilot.tools.helpers.ungranted_credential_hint",
            AsyncMock(return_value=grant_hint),
        ),
        patch(
            "backend.copilot.tools.helpers.unattended_missing_credentials_error",
            AsyncMock(side_effect=_unattended_error),
        ),
        patch(f"{_ROUTE}.get_current_permissions", return_value=permissions),
        patch(
            "backend.copilot.tools.helpers.get_current_permissions",
            return_value=permissions,
        ),
        patch(f"{_ROUTE}.get_copilot_chat_bridge_client", return_value=bridge),
    ):
        return await prepare_block_for_execution(
            block_id="blk-send",
            input_data=_DISCORD if input_data is None else input_data,
            user_id=_USER,
            session=session,
            session_id=_SESSION,
            dry_run=False,
        )


async def _unattended_error(subject, missing, session_id, user_id, expert_id):
    return ErrorResponse(
        message=f"{subject} has no discord credential connected, so this step "
        "did not run.",
        error="missing_credentials",
        session_id=session_id,
    )


@pytest.mark.asyncio
async def test_a_linked_platform_sends_the_turn_to_post_to_chat_platform():
    result = await _prepare("SendDiscordMessageBlock", _links("DISCORD"))

    assert isinstance(result, ErrorResponse)
    assert result.error == "use_post_to_chat_platform"
    assert "tool:post_to_chat_platform" in result.message
    assert "platform='discord'" in result.message
    assert "channel='#general'" in result.message


@pytest.mark.asyncio
async def test_a_dm_link_alone_does_not_send_a_channel_post_to_the_bot():
    """The bot posts into a channel only on a server the user linked; a DM
    link lets it DM the user, nothing more."""
    result = await _prepare("SendDiscordMessageBlock", _links(dm=("DISCORD",)))

    assert isinstance(result, SetupRequirementsResponse)


@pytest.mark.asyncio
async def test_the_dm_block_keeps_its_card_whatever_is_linked():
    """SendDiscordDMBlock DMs any user id; the bot can DM only the user."""
    links = _links("DISCORD", dm=("DISCORD",))
    result = await _prepare("SendDiscordDMBlock", links)

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message
    links.list_server_links.assert_not_called()


@pytest.mark.asyncio
async def test_a_scheduled_turn_is_sent_to_the_tool_not_told_to_skip():
    """The routine path: an unattended run used to end as "connect Discord,
    this step was skipped" while the bot could have posted for it."""
    result = await _prepare(
        "SendDiscordMessageBlock", _links("DISCORD"), unattended=True
    )

    assert isinstance(result, ErrorResponse)
    assert result.error == "use_post_to_chat_platform"


@pytest.mark.asyncio
async def test_no_link_keeps_the_card_and_names_the_bot_as_the_other_way():
    result = await _prepare("SendDiscordMessageBlock", _links("SLACK"))

    assert isinstance(result, SetupRequirementsResponse)
    assert "links the Discord server that channel is in" in result.message
    assert "tool:post_to_chat_platform" in result.message


@pytest.mark.asyncio
async def test_no_link_on_a_scheduled_turn_still_skips_and_names_the_bot():
    result = await _prepare("SendDiscordMessageBlock", _links(), unattended=True)

    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_credentials"
    assert "links the Discord server that channel is in" in result.message


@pytest.mark.asyncio
async def test_a_block_the_tool_cannot_replace_keeps_its_card_untouched():
    links = _links("DISCORD")
    result = await _prepare("DiscordGetCurrentUserBlock", links)

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message
    links.list_server_links.assert_not_called()


@pytest.mark.asyncio
async def test_no_bot_on_this_deployment_keeps_the_card_untouched():
    result = await _prepare(
        "SendDiscordMessageBlock", _links("DISCORD"), bot_configured=False
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_a_turn_that_may_not_post_keeps_the_card_untouched():
    result = await _prepare(
        "SendDiscordMessageBlock",
        _links("DISCORD"),
        permissions=CopilotPermissions(
            tools=["post_to_chat_platform"], tools_exclude=True
        ),
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_a_failed_link_lookup_falls_back_to_the_card():
    links = _links()
    links.list_server_links.side_effect = RuntimeError("db down")

    result = await _prepare("SendDiscordMessageBlock", links)

    assert isinstance(result, SetupRequirementsResponse)


async def _find(query: str, links: MagicMock):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(chat_platform, "_any_chat_platform_configured", lambda: True)
        index = CapabilityIndex(build_entries(TOOL_REGISTRY, TOOL_GROUPS))
    with (
        patch(
            "backend.copilot.tools.find_capability.load_connection_state",
            AsyncMock(return_value=ConnectionState()),
        ),
        patch(
            "backend.copilot.tools.find_capability.session_registry",
            AsyncMock(return_value=index),
        ),
        patch(f"{_ROUTE}.platform_linking_db", return_value=links),
    ):
        return await FindCapabilityTool()._execute(
            _USER, make_session(_USER), query=query
        )


@pytest.mark.asyncio
async def test_the_listing_says_which_platforms_the_bot_can_post_to():
    result = await _find("send a discord message", _links("DISCORD", dm=("SLACK",)))

    assert isinstance(result, CapabilityListResponse)
    tool = next(c for c in result.capabilities if c["name"] == POST_TOOL)
    assert tool["linked"] == ["discord"]
    assert tool["dm_linked"] == ["slack"]
    assert all("linked" not in c for c in result.capabilities if c is not tool)


@pytest.mark.asyncio
async def test_the_listing_says_nothing_is_linked():
    result = await _find("send a discord message", _links())

    assert isinstance(result, CapabilityListResponse)
    tool = next(c for c in result.capabilities if c["name"] == POST_TOOL)
    assert tool["linked"] == [] and tool["dm_linked"] == []


@pytest.mark.asyncio
async def test_a_search_without_the_tool_reads_no_links():
    links = _links("DISCORD")
    result = await _find("web search", links)

    assert isinstance(result, CapabilityListResponse)
    links.list_server_links.assert_not_called()


@pytest.mark.asyncio
async def test_a_schedule_pinned_to_a_discord_account_keeps_its_own_error():
    """The owner picked a Discord account for this routine; a run that cannot
    use it must say so, not quietly post as the bot instead."""
    links = _links("DISCORD")
    result = await _prepare(
        "SendDiscordMessageBlock",
        links,
        unattended=True,
        pins={"discord": MagicMock()},
    )

    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_credentials"
    links.list_server_links.assert_not_called()


@pytest.mark.asyncio
async def test_an_ungranted_discord_credential_keeps_the_grant_card():
    """The account owns a Discord credential this expert was not granted:
    granting it is the fix the owner set up, not the bot."""
    hint = " The account already has a discord credential: grant it."
    links = _links("DISCORD")
    result = await _prepare("SendDiscordMessageBlock", links, grant_hint=hint)

    assert isinstance(result, SetupRequirementsResponse)
    assert hint in result.message
    links.list_server_links.assert_not_called()


# The redirect says the user never needs a token, so it may fire only where
# the bot can post what the block was asked to post. Anywhere else the card
# stays, or the turn is left with no way to do the task.


@pytest.mark.asyncio
async def test_a_channel_in_a_server_the_user_did_not_link_keeps_the_card():
    """Server "Team" is linked; channel 222 is in another server the bot is
    in. The tool would answer channel_not_found."""
    result = await _prepare(
        "SendDiscordMessageBlock",
        _links("DISCORD"),
        input_data={"channel_name": "222", "message_content": "test"},
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "links the Discord server that channel is in" in result.message


@pytest.mark.asyncio
async def test_a_channel_name_in_another_server_keeps_the_card():
    result = await _prepare(
        "SendDiscordMessageBlock",
        _links("DISCORD"),
        input_data={
            "channel_name": "general",
            "server_name": "Other",
            "message_content": "test",
        },
    )

    assert isinstance(result, SetupRequirementsResponse)


@pytest.mark.asyncio
async def test_a_linked_channel_by_id_and_server_goes_to_the_tool():
    result = await _prepare(
        "SendDiscordMessageBlock",
        _links("DISCORD"),
        input_data={
            "channel_name": "111",
            "server_name": "team",
            "message_content": "test",
        },
    )

    assert isinstance(result, ErrorResponse)
    assert result.error == "use_post_to_chat_platform"
    assert "channel='#general'" in result.message


@pytest.mark.asyncio
async def test_a_name_two_linked_channels_share_is_posted_by_id():
    """The tool takes no server: an ambiguous name must go over as the ID."""
    other = ChannelInfo(id="333", name="general", server_id="3", server_name="Ops")
    result = await _prepare(
        "SendDiscordMessageBlock",
        _links("DISCORD"),
        input_data={
            "channel_name": "general",
            "server_name": "Ops",
            "message_content": "test",
        },
        channels=[_GENERAL, other],
    )

    assert isinstance(result, ErrorResponse)
    assert "channel='333'" in result.message


@pytest.mark.asyncio
async def test_a_bot_that_cannot_list_channels_keeps_the_card():
    result = await _prepare(
        "SendDiscordMessageBlock",
        _links("DISCORD"),
        channels=RuntimeError("bridge down"),
    )

    assert isinstance(result, SetupRequirementsResponse)


@pytest.mark.asyncio
async def test_a_private_discord_thread_keeps_the_card_and_no_bot_hint():
    """The bot opens public threads only."""
    result = await _prepare(
        "CreateDiscordThreadBlock",
        _links("DISCORD"),
        input_data={
            "channel_name": "#general",
            "thread_name": "Standup",
            "is_private": True,
        },
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_a_public_discord_thread_goes_to_the_tool_as_a_thread():
    result = await _prepare(
        "CreateDiscordThreadBlock",
        _links("DISCORD"),
        input_data={"channel_name": "#general", "thread_name": "Standup"},
    )

    assert isinstance(result, ErrorResponse)
    assert result.error == "use_post_to_chat_platform"
    assert "mode='thread'" in result.message


_SLACK_GENERAL = ChannelInfo(
    id="T1|C0GENERAL1|", name="general", server_id="T1", server_name="Acme"
)


@pytest.mark.asyncio
async def test_a_slack_dm_to_another_member_keeps_the_card_and_no_bot_hint():
    """The bot DMs only the user, never another member."""
    result = await _prepare(
        "SendSlackMessageBlock",
        _links("SLACK", server_id="T1"),
        input_data={"channel": "U0MEMBER01", "text": "hi"},
        channels=[_SLACK_GENERAL],
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_a_slack_reply_into_a_thread_keeps_the_card_and_no_bot_hint():
    """The tool cannot reply into an existing thread."""
    result = await _prepare(
        "SendSlackMessageBlock",
        _links("SLACK", server_id="T1"),
        input_data={"channel": "#general", "text": "hi", "thread_ts": "171.5"},
        channels=[_SLACK_GENERAL],
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_a_linked_slack_channel_by_id_goes_to_the_tool_by_name():
    result = await _prepare(
        "SendSlackMessageBlock",
        _links("SLACK", server_id="T1"),
        input_data={"channel": "C0GENERAL1", "text": "hi"},
        channels=[_SLACK_GENERAL],
    )

    assert isinstance(result, ErrorResponse)
    assert result.error == "use_post_to_chat_platform"
    assert "channel='#general'" in result.message


@pytest.mark.asyncio
async def test_a_telegram_chat_that_is_not_a_linked_group_keeps_the_card():
    result = await _prepare(
        "SendTelegramMessageBlock",
        _links("TELEGRAM", server_id="-100"),
        input_data={"chat_id": -200, "text": "hi"},
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "links the Telegram group" in result.message


@pytest.mark.asyncio
async def test_a_telegram_private_chat_keeps_the_card_and_no_bot_hint():
    result = await _prepare(
        "SendTelegramMessageBlock",
        _links("TELEGRAM", server_id="-100"),
        input_data={"chat_id": 4242, "text": "hi"},
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_a_linked_telegram_group_goes_to_the_tool():
    links = _links("TELEGRAM", server_id="-100")
    result = await _prepare(
        "SendTelegramMessageBlock",
        links,
        input_data={"chat_id": -100, "text": "hi"},
        channels=RuntimeError("Telegram cannot list chats"),
    )

    assert isinstance(result, ErrorResponse)
    assert result.error == "use_post_to_chat_platform"
    assert "channel='-100'" in result.message


@pytest.mark.asyncio
async def test_a_telegram_reply_keeps_the_card():
    result = await _prepare(
        "SendTelegramMessageBlock",
        _links("TELEGRAM", server_id="-100"),
        input_data={"chat_id": -100, "text": "hi", "reply_to_message_id": 7},
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_no_target_yet_keeps_the_card():
    """With no channel the target can't be known, so no redirect."""
    result = await _prepare(
        "SendDiscordMessageBlock",
        _links("DISCORD"),
        input_data={"message_content": "test"},
    )

    assert isinstance(result, SetupRequirementsResponse)
