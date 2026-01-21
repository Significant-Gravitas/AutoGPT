"""
Tests for event_bus graceful degradation when Redis is unavailable.
"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from backend.data.event_bus import AsyncRedisEventBus


class TestEvent(BaseModel):
    """Test event model."""

    message: str


class TestNotificationBus(AsyncRedisEventBus[TestEvent]):
    """Test implementation of AsyncRedisEventBus."""

    Model = TestEvent

    @property
    def event_bus_name(self) -> str:
        return "test_event_bus"


@pytest.mark.asyncio
async def test_publish_event_handles_connection_failure_gracefully():
    """Test that publish_event logs warning instead of raising when Redis is unavailable."""
    bus = TestNotificationBus()
    event = TestEvent(message="test message")

    # Mock get_redis_async to raise connection error
    with patch(
        "backend.data.event_bus.redis.get_redis_async",
        side_effect=ConnectionError("Authentication required."),
    ):
        # Should not raise exception
        await bus.publish_event(event, "test_channel")


@pytest.mark.asyncio
async def test_listen_events_handles_connection_failure_gracefully():
    """Test that listen_events returns gracefully when Redis is unavailable."""
    bus = TestNotificationBus()

    # Mock get_redis_async to raise connection error
    with patch(
        "backend.data.event_bus.redis.get_redis_async",
        side_effect=ConnectionError("Authentication required."),
    ):
        # Should not raise exception, just return empty
        events = []
        async for event in bus.listen_events("test_channel"):
            events.append(event)

        assert len(events) == 0


@pytest.mark.asyncio
async def test_wait_for_event_returns_none_on_connection_failure():
    """Test that wait_for_event returns None when Redis is unavailable."""
    bus = TestNotificationBus()

    # Mock get_redis_async to raise connection error
    with patch(
        "backend.data.event_bus.redis.get_redis_async",
        side_effect=ConnectionError("Authentication required."),
    ):
        result = await bus.wait_for_event("test_channel", timeout=1.0)
        assert result is None


@pytest.mark.asyncio
async def test_publish_event_works_with_redis_available():
    """Test that publish_event works normally when Redis is available."""
    bus = TestNotificationBus()
    event = TestEvent(message="test message")

    # Mock successful Redis connection
    mock_redis = AsyncMock()
    mock_redis.publish = AsyncMock()

    with patch("backend.data.event_bus.redis.get_redis_async", return_value=mock_redis):
        await bus.publish_event(event, "test_channel")
        mock_redis.publish.assert_called_once()
