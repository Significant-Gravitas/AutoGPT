"""
Integration tests for platform cost logging.

These tests run actual database operations to verify that SafeJson metadata
round-trips correctly through Prisma — catching the DataError that occurred
when a plain Python dict was passed to the Prisma Json? field.
"""

import uuid

import pytest
from prisma.models import PlatformCostLog as PrismaLog
from prisma.models import User

from backend.util.json import SafeJson

from .platform_cost import PlatformCostEntry, log_platform_cost


@pytest.fixture
async def cost_log_user():
    """Create a throw-away user and clean up cost logs after the test."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"cost-test-{user_id}@example.com",
            "topUpConfig": SafeJson({}),
            "timezone": "UTC",
        }
    )
    yield user_id
    await PrismaLog.prisma().delete_many(where={"userId": user_id})
    await User.prisma().delete(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_log_platform_cost_metadata_round_trip(cost_log_user):
    """
    Verify that SafeJson metadata is persisted and read back correctly.

    This test would have caught the DataError that silently swallowed all cost
    log writes when a plain Python dict was passed to the Prisma Json? field.
    """
    user_id = cost_log_user
    entry = PlatformCostEntry(
        user_id=user_id,
        block_name="TestBlock",
        provider="openai",
        cost_microdollars=5000,
        input_tokens=100,
        output_tokens=50,
        model="gpt-4",
        metadata={"key": "val", "nested": {"x": 1}},
    )
    await log_platform_cost(entry)

    rows = await PrismaLog.prisma().find_many(where={"userId": user_id})
    assert len(rows) == 1
    assert rows[0].metadata == {"key": "val", "nested": {"x": 1}}
    assert rows[0].provider == "openai"
    assert rows[0].costMicrodollars == 5000


@pytest.mark.asyncio(loop_scope="session")
async def test_log_platform_cost_metadata_none(cost_log_user):
    """Verify that None metadata falls back to {} (not a DataError)."""
    user_id = cost_log_user
    entry = PlatformCostEntry(
        user_id=user_id,
        block_name="TestBlock",
        provider="anthropic",
        metadata=None,
    )
    await log_platform_cost(entry)

    rows = await PrismaLog.prisma().find_many(where={"userId": user_id})
    assert len(rows) == 1
    assert rows[0].metadata == {}


@pytest.mark.asyncio(loop_scope="session")
async def test_log_platform_cost_cache_tokens(cost_log_user):
    """Verify that cache_read_tokens and cache_creation_tokens are persisted."""
    user_id = cost_log_user
    entry = PlatformCostEntry(
        user_id=user_id,
        block_name="TestBlock",
        provider="anthropic",
        input_tokens=200,
        output_tokens=100,
        cache_read_tokens=50,
        cache_creation_tokens=25,
        model="claude-3-5-sonnet-20241022",
    )
    await log_platform_cost(entry)

    rows = await PrismaLog.prisma().find_many(where={"userId": user_id})
    assert len(rows) == 1
    assert rows[0].cacheReadTokens == 50
    assert rows[0].cacheCreationTokens == 25
