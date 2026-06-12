"""Tests for CoPilotChatBridge's outbound RPC dispatch.

Covers the seam that maps a ``Platform`` enum to the live adapter and routes
into ``outbound`` — including the casing round-trip (``platform.value`` vs
``platform_name``) and the no-adapter guard.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.platform_linking.models import Platform
from backend.util.service import UnhealthyServiceError

from .app import CoPilotChatBridge
from .outbound import DeliveryResult


def _bridge(*, with_adapter: bool = True) -> tuple[CoPilotChatBridge, MagicMock]:
    """Build a bridge without AppService.__init__, with a stubbed adapter+api."""
    bridge = CoPilotChatBridge.__new__(CoPilotChatBridge)
    adapter = MagicMock()
    adapter.platform_name = "discord"
    bridge._adapters_healthy = True
    bridge._api = MagicMock()
    bridge._adapters_by_platform = {"discord": adapter} if with_adapter else {}
    return bridge, adapter


@pytest.mark.asyncio
async def test_require_raises_when_adapter_missing():
    bridge, _ = _bridge(with_adapter=False)
    with pytest.raises(UnhealthyServiceError):
        bridge._require(Platform.DISCORD)


@pytest.mark.asyncio
async def test_require_raises_when_unhealthy():
    bridge, _ = _bridge()
    bridge._adapters_healthy = False
    with pytest.raises(UnhealthyServiceError):
        bridge._require(Platform.DISCORD)


@pytest.mark.asyncio
async def test_send_message_to_channel_routes_platform_value():
    bridge, _ = _bridge()
    expected = DeliveryResult(ok=True, kind="message", channel_id="42", ref_id="1")
    with patch(
        "backend.copilot.bot.app.outbound.deliver_message",
        new=AsyncMock(return_value=expected),
    ) as deliver:
        result = await bridge.send_message_to_channel(
            platform=Platform.DISCORD, user_id="u1", channel="#x", content="hi"
        )
    assert result is expected
    # platform passed through as the string value the lower layers expect.
    call = deliver.await_args
    assert call is not None
    _, _, platform_value, user_id, channel, content = call.args
    assert platform_value == "DISCORD"
    assert (user_id, channel, content) == ("u1", "#x", "hi")


@pytest.mark.asyncio
async def test_create_thread_in_channel_routes_to_outbound():
    bridge, _ = _bridge()
    expected = DeliveryResult(ok=True, kind="thread", channel_id="42", ref_id="t1")
    with patch(
        "backend.copilot.bot.app.outbound.create_thread",
        new=AsyncMock(return_value=expected),
    ) as create:
        result = await bridge.create_thread_in_channel(
            platform=Platform.DISCORD,
            user_id="u1",
            channel="#x",
            thread_name="Monday",
            content="body",
        )
    assert result is expected
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_channels_routes_to_outbound():
    bridge, _ = _bridge()
    with patch(
        "backend.copilot.bot.app.outbound.list_channels",
        new=AsyncMock(return_value=[]),
    ) as lister:
        result = await bridge.list_channels(platform=Platform.DISCORD, user_id="u1")
    assert result == []
    lister.assert_awaited_once()
