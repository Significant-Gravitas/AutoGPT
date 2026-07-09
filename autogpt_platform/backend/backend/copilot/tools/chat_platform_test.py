"""Tests for PostToChatPlatformTool and ListChatPlatformChannelsTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot.adapters.base import ChannelInfo
from backend.copilot.bot.outbound import DeliveryResult
from backend.copilot.tools.chat_platform import (
    ListChatPlatformChannelsTool,
    PostToChatPlatformTool,
)
from backend.copilot.tools.models import (
    ChatPlatformChannelListResponse,
    ChatPlatformPostedResponse,
    ErrorResponse,
)

from ._test_data import make_session

_USER = "test-user-chat"
_PATH = "backend.copilot.tools.chat_platform"


@pytest.fixture
def session():
    return make_session(_USER)


def _bridge() -> MagicMock:
    bridge = MagicMock()
    bridge.send_message_to_channel = AsyncMock()
    bridge.create_thread_in_channel = AsyncMock()
    bridge.list_channels = AsyncMock()
    return bridge


# ── PostToChatPlatformTool ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_requires_auth(session):
    result = await PostToChatPlatformTool()._execute(
        user_id=None, session=session, channel="#x", content="hi"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_post_missing_channel(session):
    result = await PostToChatPlatformTool()._execute(
        user_id=_USER, session=session, channel="  ", content="hi"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_channel"


@pytest.mark.asyncio
async def test_post_thread_requires_thread_name(session):
    result = await PostToChatPlatformTool()._execute(
        user_id=_USER, session=session, channel="#x", content="hi", mode="thread"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_thread_name"


@pytest.mark.asyncio
async def test_post_unsupported_platform(session):
    result = await PostToChatPlatformTool()._execute(
        user_id=_USER, session=session, platform="myspace", channel="#x", content="hi"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "unsupported_platform"


@pytest.mark.asyncio
async def test_post_message_happy_path_defaults_to_discord(session):
    bridge = _bridge()
    bridge.send_message_to_channel.return_value = DeliveryResult(
        ok=True,
        kind="message",
        channel_id="42",
        ref_id="100",
        url="https://discord.com/x",
    )
    with patch(f"{_PATH}.get_copilot_chat_bridge_client", return_value=bridge):
        result = await PostToChatPlatformTool()._execute(
            user_id=_USER, session=session, channel="#standup", content="hi"
        )
    assert isinstance(result, ChatPlatformPostedResponse)
    assert result.platform == "discord"
    assert result.kind == "message"
    assert result.channel_id == "42"
    assert result.url == "https://discord.com/x"
    # platform forwarded to the bridge as the enum value.
    assert (
        bridge.send_message_to_channel.await_args.kwargs["platform"].value == "DISCORD"
    )
    bridge.create_thread_in_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_thread_happy_path(session):
    bridge = _bridge()
    bridge.create_thread_in_channel.return_value = DeliveryResult(
        ok=True,
        kind="thread",
        channel_id="42",
        ref_id="t1",
        url="https://discord.com/t",
    )
    with patch(f"{_PATH}.get_copilot_chat_bridge_client", return_value=bridge):
        result = await PostToChatPlatformTool()._execute(
            user_id=_USER,
            session=session,
            platform="discord",
            channel="42",
            content="body",
            mode="thread",
            thread_name="Monday",
        )
    assert isinstance(result, ChatPlatformPostedResponse)
    assert result.kind == "thread"
    assert result.ref_id == "t1"
    bridge.create_thread_in_channel.assert_awaited_once()


@pytest.mark.asyncio
async def test_post_maps_authz_error(session):
    bridge = _bridge()
    bridge.send_message_to_channel.return_value = DeliveryResult(
        ok=False, kind="message", error="not_authorized"
    )
    with patch(f"{_PATH}.get_copilot_chat_bridge_client", return_value=bridge):
        result = await PostToChatPlatformTool()._execute(
            user_id=_USER, session=session, channel="999", content="hi"
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "not_authorized"
    assert "isn't linked" in result.message


@pytest.mark.asyncio
async def test_post_missing_content(session):
    result = await PostToChatPlatformTool()._execute(
        user_id=_USER, session=session, channel="#x", content="   "
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_content"


@pytest.mark.asyncio
async def test_post_invalid_mode(session):
    result = await PostToChatPlatformTool()._execute(
        user_id=_USER, session=session, channel="#x", content="hi", mode="shout"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "invalid_mode"


@pytest.mark.asyncio
async def test_post_message_send_failed_maps_error(session):
    bridge = _bridge()
    bridge.send_message_to_channel.return_value = DeliveryResult(
        ok=False, kind="message", channel_id="42", error="send_failed"
    )
    with patch(f"{_PATH}.get_copilot_chat_bridge_client", return_value=bridge):
        result = await PostToChatPlatformTool()._execute(
            user_id=_USER, session=session, channel="#x", content="hi"
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "send_failed"
    assert "permission" in result.message


@pytest.mark.asyncio
async def test_post_thread_failed_maps_error(session):
    bridge = _bridge()
    bridge.create_thread_in_channel.return_value = DeliveryResult(
        ok=False, kind="thread", channel_id="42", error="thread_failed"
    )
    with patch(f"{_PATH}.get_copilot_chat_bridge_client", return_value=bridge):
        result = await PostToChatPlatformTool()._execute(
            user_id=_USER,
            session=session,
            channel="#x",
            content="body",
            mode="thread",
            thread_name="Monday",
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "thread_failed"


@pytest.mark.asyncio
async def test_post_unavailable_without_token():
    with patch(f"{_PATH}._any_chat_platform_configured", return_value=False):
        assert PostToChatPlatformTool().is_available is False
    with patch(f"{_PATH}._any_chat_platform_configured", return_value=True):
        assert PostToChatPlatformTool().is_available is True


# ── ListChatPlatformChannelsTool ───────────────────────────────────


@pytest.mark.asyncio
async def test_list_channels_happy_path(session):
    bridge = _bridge()
    bridge.list_channels.return_value = [
        ChannelInfo(id="10", name="general", server_id="g1", server_name="Guild"),
    ]
    with patch(f"{_PATH}.get_copilot_chat_bridge_client", return_value=bridge):
        result = await ListChatPlatformChannelsTool()._execute(
            user_id=_USER, session=session
        )
    assert isinstance(result, ChatPlatformChannelListResponse)
    assert result.platform == "discord"
    assert result.count == 1
    assert result.channels[0].name == "general"


@pytest.mark.asyncio
async def test_list_channels_requires_auth(session):
    result = await ListChatPlatformChannelsTool()._execute(
        user_id=None, session=session
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"
