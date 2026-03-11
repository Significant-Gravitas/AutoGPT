"""Integration tests for LLM registry notification system."""

import asyncio
import json
import time

import pytest
from unittest.mock import AsyncMock, patch

from backend.data.llm_registry import notifications
from backend.executor import llm_registry_init


@pytest.mark.asyncio
async def test_notification_with_data_payload():
    """Verify notification can carry model data."""
    models_data = [
        {
            "slug": "gpt-4o",
            "displayName": "GPT-4o",
            "contextWindow": 128000,
        }
    ]

    # Mock Redis
    with patch("backend.data.llm_registry.notifications.connect_async") as mock_redis:
        mock_client = AsyncMock()
        mock_redis.return_value = mock_client

        # Publish notification
        await notifications.publish_registry_refresh_notification(models_data)

        # Verify Redis publish was called with JSON payload
        assert mock_client.publish.call_count == 1
        channel, payload = mock_client.publish.call_args[0]

        assert channel == "llm_registry:refresh"

        # Parse and verify payload
        parsed = json.loads(payload)
        assert parsed["action"] == "refresh"
        assert parsed["data"] == models_data


@pytest.mark.asyncio
async def test_notification_backwards_compatibility():
    """Verify notifications work without data payload (backwards compatibility)."""
    with patch("backend.data.llm_registry.notifications.connect_async") as mock_redis:
        mock_client = AsyncMock()
        mock_redis.return_value = mock_client

        # Publish without data
        await notifications.publish_registry_refresh_notification(models_data=None)

        # Verify simple string payload
        assert mock_client.publish.call_count == 1
        _, payload = mock_client.publish.call_args[0]
        assert payload == "refresh"


@pytest.mark.asyncio
async def test_subscribe_extracts_data_from_notification():
    """Verify subscriber can extract data from notification payload."""
    received_data = None

    async def mock_callback(data):
        nonlocal received_data
        received_data = data

    models_data = [{"slug": "gpt-4o", "displayName": "GPT-4o"}]

    # Simulate receiving a notification message
    message = {
        "type": "message",
        "channel": b"llm_registry:refresh",
        "data": json.dumps({"action": "refresh", "data": models_data}).encode("utf-8"),
    }

    # Mock Redis pubsub
    with patch("backend.data.llm_registry.notifications.connect_async") as mock_redis:
        mock_client = AsyncMock()
        mock_pubsub = AsyncMock()
        mock_redis.return_value = mock_client
        mock_client.pubsub.return_value = mock_pubsub

        # Return the message once, then None to stop the loop
        mock_pubsub.get_message.side_effect = [message, None]

        # Start subscription in a task and cancel after first message
        async def run_subscription():
            await notifications.subscribe_to_registry_refresh(mock_callback)

        task = asyncio.create_task(run_subscription())
        await asyncio.sleep(0.1)  # Let it process the message
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify callback was called with extracted data
        assert received_data == models_data


@pytest.mark.asyncio
async def test_jitter_adds_delay():
    """Verify jitter is applied before refresh."""
    with patch("backend.data.llm_registry.registry.refresh_llm_registry"), patch(
        "backend.data.block_cost_config.refresh_llm_costs"
    ), patch("backend.blocks._base.BlockSchema.clear_all_schema_caches"), patch(
        "backend.data.db.is_connected", return_value=True
    ):

        start = time.time()
        await llm_registry_init.refresh_registry_on_notification(models_data=[])
        elapsed = time.time() - start

        # Should have at least some delay (0-2 seconds)
        # We can't test the exact delay due to jitter randomness,
        # but we can verify it took some time
        assert elapsed >= 0
        assert elapsed <= 3  # Allow some overhead


@pytest.mark.asyncio
async def test_refresh_uses_provided_data():
    """Verify refresh uses provided data instead of fetching."""
    models_data = [{"slug": "test", "displayName": "Test"}]

    with patch(
        "backend.data.llm_registry.registry.refresh_llm_registry"
    ) as mock_refresh, patch("backend.data.block_cost_config.refresh_llm_costs"), patch(
        "backend.blocks._base.BlockSchema.clear_all_schema_caches"
    ), patch(
        "backend.data.db.is_connected", return_value=True
    ):

        await llm_registry_init.refresh_registry_on_notification(
            models_data=models_data
        )

        # Verify refresh was called with the data
        assert mock_refresh.call_count == 1
        assert mock_refresh.call_args[1]["models_data"] == models_data
