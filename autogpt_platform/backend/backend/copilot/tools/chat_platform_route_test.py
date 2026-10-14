"""A bot-token send block with no token, for a user who linked that platform
to the AutoGPT bot, points the turn at post_to_chat_platform instead of a
setup card asking for a token they never need."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    links: MagicMock,
    *,
    bot_configured: bool = True,
    unattended: bool = False,
    permissions: CopilotPermissions | None = None,
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
        patch(f"{_ROUTE}.platform_linking_db", return_value=links),
        patch(f"{_ROUTE}._any_chat_platform_configured", return_value=bot_configured),
        patch(f"{_ROUTE}.get_current_permissions", return_value=permissions),
        patch(
            "backend.copilot.tools.helpers.get_current_permissions",
            return_value=permissions,
        ),
    ):
        return await prepare_block_for_execution(
            block_id="blk-send",
            input_data={"message_content": "test"},
            user_id=_USER,
            session=session,
            session_id=_SESSION,
            dry_run=False,
        )


@pytest.mark.asyncio
async def test_a_linked_platform_sends_the_turn_to_post_to_chat_platform():
    result = await _prepare("SendDiscordMessageBlock", _links("DISCORD"))

    assert isinstance(result, ErrorResponse)
    assert result.error == "use_post_to_chat_platform"
    assert "tool:post_to_chat_platform" in result.message
    assert "platform='discord'" in result.message


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
    assert "links Discord to the AutoGPT bot" in result.message
    assert "tool:post_to_chat_platform" in result.message


@pytest.mark.asyncio
async def test_no_link_on_a_scheduled_turn_still_skips_and_names_the_bot():
    result = await _prepare("SendDiscordMessageBlock", _links(), unattended=True)

    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_credentials"
    assert "links Discord to the AutoGPT bot" in result.message


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
