"""A bot-token send block with no token names the AutoGPT bot as the other
way to post, where the bot could post it, and find_capability says where the
user linked a platform to the bot."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.ranking import ConnectionState
from backend.copilot.capabilities.registry import build_entries
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.tools import TOOL_GROUPS, TOOL_REGISTRY, chat_platform

from ._test_data import make_session
from .chat_platform_hint import POST_TOOL
from .find_capability import FindCapabilityTool
from .helpers import prepare_block_for_execution
from .models import CapabilityListResponse, ErrorResponse, SetupRequirementsResponse

_USER = "hint-user"
_SESSION = "hint-session"
_HINT = "backend.copilot.tools.chat_platform_hint"


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


_DISCORD = {"channel_name": "#general", "message_content": "test"}


def _links(*platforms: str, dm: tuple[str, ...] = ()) -> MagicMock:
    db = MagicMock()
    db.list_server_links = AsyncMock(
        return_value=[SimpleNamespace(platform=p) for p in platforms]
    )
    db.list_user_links = AsyncMock(
        return_value=[SimpleNamespace(platform=p) for p in dm]
    )
    return db


async def _prepare(
    block_name: str,
    *,
    bot_configured: bool = True,
    unattended: bool = False,
    permissions: CopilotPermissions | None = None,
    pins: dict | None = None,
    grant_hint: str = "",
    input_data: dict | None = None,
):
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
        patch(
            f"{_HINT}._configured_chat_platforms",
            return_value=(
                frozenset({"discord", "slack", "telegram"})
                if bot_configured
                else frozenset()
            ),
        ),
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
        patch(f"{_HINT}.get_current_permissions", return_value=permissions),
        patch(
            "backend.copilot.tools.helpers.get_current_permissions",
            return_value=permissions,
        ),
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
async def test_a_send_block_keeps_the_card_and_names_the_bot_as_the_other_way():
    result = await _prepare("SendDiscordMessageBlock")

    assert isinstance(result, SetupRequirementsResponse)
    assert "links the Discord server that channel is in" in result.message
    assert "tool:post_to_chat_platform" in result.message


@pytest.mark.asyncio
async def test_a_scheduled_turn_still_skips_and_names_the_bot():
    result = await _prepare("SendDiscordMessageBlock", unattended=True)

    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_credentials"
    assert "links the Discord server that channel is in" in result.message


@pytest.mark.asyncio
async def test_the_dm_block_gets_no_bot_hint():
    """SendDiscordDMBlock DMs any user id; the bot can DM only the user."""
    result = await _prepare("SendDiscordDMBlock")

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_a_block_the_tool_cannot_replace_gets_no_bot_hint():
    result = await _prepare("DiscordGetCurrentUserBlock")

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_no_bot_on_this_deployment_gets_no_bot_hint():
    result = await _prepare("SendDiscordMessageBlock", bot_configured=False)

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_a_turn_that_may_not_post_gets_no_bot_hint():
    result = await _prepare(
        "SendDiscordMessageBlock",
        permissions=CopilotPermissions(
            tools=["post_to_chat_platform"], tools_exclude=True
        ),
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_a_schedule_pinned_to_a_discord_account_keeps_its_own_error():
    """The owner picked a Discord account for this routine; a run that cannot
    use it must say so, not point at the bot instead."""
    result = await _prepare(
        "SendDiscordMessageBlock", unattended=True, pins={"discord": MagicMock()}
    )

    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_credentials"
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_an_ungranted_discord_credential_keeps_the_grant_card():
    """The account owns a Discord credential this expert was not granted:
    granting it is the fix the owner set up, not the bot."""
    hint = " The account already has a discord credential: grant it."
    result = await _prepare("SendDiscordMessageBlock", grant_hint=hint)

    assert isinstance(result, SetupRequirementsResponse)
    assert hint in result.message
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_an_ungranted_credential_on_a_scheduled_turn_gets_no_bot_hint():
    hint = " The account already has a discord credential: grant it."
    result = await _prepare("SendDiscordMessageBlock", unattended=True, grant_hint=hint)

    assert isinstance(result, ErrorResponse)
    assert "post_to_chat_platform" not in result.message


# The bot posts only into a public channel or thread of a linked server or
# group, or DMs the user: the hint stays off anything else it can't do.


@pytest.mark.asyncio
async def test_a_private_discord_thread_gets_no_bot_hint():
    """The bot opens public threads only."""
    result = await _prepare(
        "CreateDiscordThreadBlock",
        input_data={
            "channel_name": "#general",
            "thread_name": "Standup",
            "is_private": True,
        },
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "channel",
    [
        "U0MEMBER01",
        "W0MEMBER01",
        "D0DIRECT01",
        "@someone",
    ],
)
async def test_a_slack_dm_to_another_member_gets_no_bot_hint(channel):
    """The bot DMs only the user, never another member."""
    result = await _prepare(
        "SendSlackMessageBlock", input_data={"channel": channel, "text": "hi"}
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "channel",
    [
        "C0CHANNEL1",
        "#general",
        "deployments",
        "upcoming",
        "watercooler",
        "deployments2026",
        "u0member01",
        "w0member01",
        "d0direct01",
    ],
)
async def test_a_slack_channel_still_gets_a_bot_hint(channel):
    result = await _prepare(
        "SendSlackMessageBlock", input_data={"channel": channel, "text": "hi"}
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "links the Slack workspace" in result.message


@pytest.mark.asyncio
async def test_a_slack_reply_into_a_thread_gets_no_bot_hint():
    """The tool cannot reply into an existing thread."""
    result = await _prepare(
        "SendSlackMessageBlock",
        input_data={"channel": "#general", "text": "hi", "thread_ts": "171.5"},
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_a_telegram_group_names_the_bot_as_the_other_way():
    result = await _prepare(
        "SendTelegramMessageBlock", input_data={"chat_id": -200, "text": "hi"}
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "links the Telegram group" in result.message


@pytest.mark.asyncio
async def test_a_telegram_private_chat_gets_no_bot_hint():
    result = await _prepare(
        "SendTelegramMessageBlock", input_data={"chat_id": 4242, "text": "hi"}
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


@pytest.mark.asyncio
async def test_a_telegram_reply_gets_no_bot_hint():
    result = await _prepare(
        "SendTelegramMessageBlock",
        input_data={"chat_id": -100, "text": "hi", "reply_to_message_id": 7},
    )

    assert isinstance(result, SetupRequirementsResponse)
    assert "post_to_chat_platform" not in result.message


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
        patch(f"{_HINT}.platform_linking_db", return_value=links),
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
async def test_a_failed_link_lookup_lists_nothing_as_linked():
    links = _links()
    links.list_server_links.side_effect = RuntimeError("db down")
    result = await _find("send a discord message", links)

    assert isinstance(result, CapabilityListResponse)
    tool = next(c for c in result.capabilities if c["name"] == POST_TOOL)
    assert tool["linked"] == [] and tool["dm_linked"] == []


@pytest.mark.asyncio
async def test_a_search_without_the_tool_reads_no_links():
    links = _links("DISCORD")
    result = await _find("web search", links)

    assert isinstance(result, CapabilityListResponse)
    links.list_server_links.assert_not_called()
