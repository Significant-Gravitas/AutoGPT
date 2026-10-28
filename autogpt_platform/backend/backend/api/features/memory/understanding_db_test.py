"""The Memory page's business-understanding routes against a real database row."""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.models import CoPilotUnderstanding
from redis.exceptions import ConnectionError as RedisConnectionError

from backend.api.rest_api import app
from backend.data import understanding as understanding_module
from backend.data.redis_client import get_redis_async
from backend.data.understanding import (
    FORGOTTEN_MARKER,
    BusinessUnderstandingInput,
    clear_business_understanding,
    get_business_understanding,
    upsert_business_understanding,
)


@pytest_asyncio.fixture(loop_scope="session")
async def client(
    setup_test_user, mock_jwt_user
) -> AsyncGenerator[httpx.AsyncClient, None]:
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client
    app.dependency_overrides.pop(get_jwt_payload, None)


@pytest_asyncio.fixture(loop_scope="session", autouse=True)
async def no_understanding(client, test_user_id: str):
    await clear_business_understanding(test_user_id)
    yield
    await clear_business_understanding(test_user_id)


async def _seed(user_id: str) -> None:
    await upsert_business_understanding(
        user_id,
        BusinessUnderstandingInput(
            user_name="Sam",
            business_name="Harbor Bikes",
            pain_points=["Chasing invoices"],
            additional_notes="Onboarding brain dump (typed): I run a bike shop.",
            suggested_prompts={"Automate": ["Send invoice reminders"]},
        ),
    )


async def test_shows_the_stored_understanding(client, test_user_id):
    await _seed(test_user_id)

    resp = await client.get("/api/memory/understanding")

    assert resp.status_code == 200
    body = resp.json()
    assert body["business_name"] == "Harbor Bikes"
    assert body["pain_points"] == ["Chasing invoices"]
    assert body["additional_notes"].endswith("I run a bike shop.")
    assert body["updated_at"] is not None
    assert "suggested_prompts" not in body


async def test_nothing_kept_reads_as_empty(client):
    resp = await client.get("/api/memory/understanding")

    assert resp.status_code == 200
    assert resp.json()["updated_at"] is None
    assert resp.json()["pain_points"] == []


async def test_forget_deletes_the_row_and_its_cached_copy(client, test_user_id):
    await _seed(test_user_id)
    # The read caches the row; a forget that kept the cache would keep serving it.
    assert await get_business_understanding(test_user_id) is not None

    resp = await client.delete("/api/memory/understanding")

    assert resp.status_code == 200
    assert resp.json() == {"forgotten": True}
    assert await get_business_understanding(test_user_id) is None
    again = await client.delete("/api/memory/understanding")
    assert again.json() == {"forgotten": False}


async def test_account_erase_forgets_the_understanding(client, test_user_id):
    await _seed(test_user_id)
    driver = AsyncMock()
    driver.execute_query.side_effect = [([{"c": 0}], None, None)]

    with patch("backend.api.features.memory.routes.open_driver", return_value=driver):
        resp = await client.delete("/api/memory")

    assert resp.status_code == 200
    assert await get_business_understanding(test_user_id) is None


async def test_a_read_that_raced_the_forget_cannot_cache_the_row_again(
    client, test_user_id, monkeypatch
):
    await _seed(test_user_id)
    redis = await get_redis_async()
    await redis.delete(understanding_module._cache_key(test_user_id))
    read_done, release = asyncio.Event(), asyncio.Event()
    real = CoPilotUnderstanding.prisma()

    class _ReadPausesBeforeCaching:
        async def find_unique(self, **kwargs):
            row = await real.find_unique(**kwargs)
            read_done.set()
            await release.wait()
            return row

        def __getattr__(self, name):
            return getattr(real, name)

    monkeypatch.setattr(
        understanding_module.CoPilotUnderstanding,
        "prisma",
        lambda: _ReadPausesBeforeCaching(),
    )
    reader = asyncio.create_task(get_business_understanding(test_user_id))
    await asyncio.wait_for(read_done.wait(), timeout=10)

    resp = await client.delete("/api/memory/understanding")
    release.set()
    await reader

    assert resp.json() == {"forgotten": True}
    cached = await redis.get(understanding_module._cache_key(test_user_id))
    assert cached == FORGOTTEN_MARKER
    assert await get_business_understanding(test_user_id) is None


async def test_forget_fails_when_the_cached_copy_cannot_be_replaced(test_user_id):
    await _seed(test_user_id)
    broken = AsyncMock()
    broken.set.side_effect = RedisConnectionError("redis down")

    with (
        patch(
            "backend.data.understanding.get_redis_async",
            new=AsyncMock(return_value=broken),
        ),
        pytest.raises(RedisConnectionError),
    ):
        await clear_business_understanding(test_user_id)
