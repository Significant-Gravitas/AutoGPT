"""Unit tests for LLM registry caching and thundering herd protection."""

import asyncio

import pytest
from unittest.mock import AsyncMock, patch

from backend.data.llm_registry import registry


@pytest.mark.asyncio
async def test_fetch_registry_from_db_caching():
    """Verify @cached prevents duplicate DB calls."""
    with patch("backend.data.llm_registry.registry.prisma.models.LlmModel") as mock:
        mock.prisma().find_many = AsyncMock(return_value=[])

        # Clear cache first
        registry._fetch_registry_from_db.cache_clear()

        # Call twice
        await registry._fetch_registry_from_db()
        await registry._fetch_registry_from_db()

        # Verify only called once (cached)
        assert mock.prisma().find_many.call_count == 1


@pytest.mark.asyncio
async def test_thundering_herd_protection():
    """Verify only 1 DB call with 100 concurrent requests."""
    call_count = 0

    async def mock_db_fetch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)  # Simulate slow DB
        return []

    with patch("backend.data.llm_registry.registry.prisma.models.LlmModel") as mock:
        mock.prisma().find_many = mock_db_fetch

        # Clear cache first
        registry._fetch_registry_from_db.cache_clear()

        # Launch 100 concurrent fetches
        tasks = [registry._fetch_registry_from_db() for _ in range(100)]
        await asyncio.gather(*tasks)

        # Verify only 1 DB call due to thundering herd protection
        assert call_count == 1


@pytest.mark.asyncio
async def test_refresh_accepts_models_data():
    """Verify refresh_llm_registry can accept pre-fetched data."""
    models_data = [
        {
            "id": "test-id",
            "slug": "gpt-4o",
            "displayName": "GPT-4o",
            "description": "Test model",
            "providerId": "openai",
            "creatorId": None,
            "contextWindow": 128000,
            "maxOutputTokens": 4096,
            "priceTier": 2,
            "isEnabled": True,
            "isRecommended": False,
            "capabilities": {},
            "metadata": {},
            "Provider": {
                "name": "openai",
                "displayName": "OpenAI",
            },
            "Costs": [],
            "Creator": None,
        }
    ]

    with patch("backend.data.llm_registry.registry.prisma.models.LlmModel") as mock:
        # Should NOT call DB if data provided
        await registry.refresh_llm_registry(models_data=models_data)
        mock.prisma().find_many.assert_not_called()

        # Verify model was added to registry
        assert "gpt-4o" in registry._dynamic_models
        assert registry._dynamic_models["gpt-4o"].display_name == "GPT-4o"


@pytest.mark.asyncio
async def test_refresh_falls_back_to_cache():
    """Verify refresh_llm_registry fetches from cache when no data provided."""
    with patch(
        "backend.data.llm_registry.registry._fetch_registry_from_db"
    ) as mock_fetch:
        mock_fetch.return_value = []

        # Call without data - should fetch from cache
        await registry.refresh_llm_registry(models_data=None)

        # Verify cache fetch was called
        assert mock_fetch.call_count == 1


@pytest.mark.asyncio
async def test_cache_clear_forces_fresh_fetch():
    """
    Verify that cache_clear() forces a fresh DB fetch.

    This is CRITICAL for admin updates - when an admin changes a model,
    we must clear the cache before fetching to ensure fresh data is broadcast.
    """
    fetch_count = 0

    async def mock_db_fetch(*args, **kwargs):
        nonlocal fetch_count
        fetch_count += 1
        return [{"slug": f"model-{fetch_count}", "displayName": f"Model {fetch_count}"}]

    with patch("backend.data.llm_registry.registry.prisma.models.LlmModel") as mock:
        mock.prisma().find_many = mock_db_fetch

        # Clear cache and fetch first time
        registry._fetch_registry_from_db.cache_clear()
        result1 = await registry._fetch_registry_from_db()
        assert result1[0]["slug"] == "model-1"
        assert fetch_count == 1

        # Fetch second time (should use cache, no DB call)
        result2 = await registry._fetch_registry_from_db()
        assert result2[0]["slug"] == "model-1"  # Same cached data
        assert fetch_count == 1  # No additional DB call

        # Clear cache (simulating admin update)
        registry._fetch_registry_from_db.cache_clear()

        # Fetch third time (should hit DB for fresh data)
        result3 = await registry._fetch_registry_from_db()
        assert result3[0]["slug"] == "model-2"  # Fresh data!
        assert fetch_count == 2  # New DB call

        # Verify cache_clear() method exists and is callable
        assert callable(registry._fetch_registry_from_db.cache_clear)
