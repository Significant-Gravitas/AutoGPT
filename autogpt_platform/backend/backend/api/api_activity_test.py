"""A real authenticated request through the app counts once against its account,
for a session on the internal API and an API key on the external one, and
answers the same while Redis fails or hangs."""

import asyncio
import random
from datetime import UTC, date, datetime, time, timedelta
from uuid import uuid4

import httpx
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import APIKeyPermission
from prisma.models import APIKey, User
from redis.exceptions import ConnectionError as RedisConnectionError

from backend.api.rest_api import app as rest_app
from backend.data import api_activity
from backend.data.auth import api_key as api_key_db
from backend.data.redis_client import HOST, PORT
from backend.util.test import SpinTestServer
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(HOST, PORT), reason="needs a live Redis"
)

SESSION_PATH = "/api/experiments/assignments"


@pytest.fixture
def day(mocker) -> date:
    """Midday on a day no other run writes to, so a run can't straddle midnight."""
    pinned = date(2001, 1, 1) + timedelta(days=random.randrange(5000))
    noon = datetime.combine(pinned, time(12), tzinfo=UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return noon

    mocker.patch.object(api_activity, "datetime", Clock)
    return pinned


@pytest.fixture(autouse=True)
def tally(mocker):
    """A tally of the test's own, so no other test's flusher writes its counts."""
    tally = api_activity._Tally()
    mocker.patch.object(api_activity, "_tally", tally)
    yield tally
    if tally._flusher is not None:
        tally._flusher.cancel()


@pytest.fixture
async def user_id(server: SpinTestServer, mocker):
    user_id = str(uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@example.com", "name": "Counted"}
    )
    # Only the token is faked: SpinTestServer's own override of `get_user_id`
    # would skip the resolver that counts.
    mocker.patch.dict(
        rest_app.dependency_overrides,
        {get_jwt_payload: lambda: {"sub": user_id, "role": "user"}},
        clear=True,
    )
    yield user_id
    await APIKey.prisma().delete_many(where={"userId": user_id})
    await User.prisma().delete_many(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_a_session_request_counts_once_as_internal(user_id, day):
    async with _client() as client:
        response = await client.get(SESSION_PATH)
    await api_activity.flush()

    assert response.status_code == 200
    counts = await api_activity.get_request_counts(day, [user_id])
    assert counts[user_id].model_dump() == {"internal": 1, "external": 0}


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("path", ["/external-api/v1/me", "/api/v2/me"])
async def test_an_api_key_request_counts_once_as_external(user_id, day, path):
    # v2 verifies the key in its rate limiter and again in the route.
    headers = await _api_key_headers(user_id)

    async with _client() as client:
        await client.get(path, headers=headers)
    await api_activity.flush()

    counts = await api_activity.get_request_counts(day, [user_id])
    assert counts[user_id].model_dump() == {"internal": 0, "external": 1}


@pytest.mark.asyncio(loop_scope="session")
async def test_counts_reach_redis_without_an_explicit_flush(mocker, user_id, day):
    mocker.patch.object(api_activity, "_FLUSH_INTERVAL_SECONDS", 0.05)

    async with _client() as client:
        await client.get(SESSION_PATH)
    counted = await asyncio.wait_for(_counted(day, user_id), timeout=10)

    assert counted.internal == 1


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("redis", ["fails", "hangs"])
@pytest.mark.parametrize("credential", ["session", "api_key"])
async def test_requests_answer_while_redis_fails_or_hangs(
    mocker, caplog, user_id, credential, redis
):
    async def unreachable():
        raise RedisConnectionError("Redis is unreachable")

    async def unanswered():
        await asyncio.Event().wait()

    get_redis = mocker.patch.object(
        api_activity,
        "get_redis_async",
        side_effect=unreachable if redis == "fails" else unanswered,
    )
    mocker.patch.object(api_activity, "_failures", api_activity._FailureLog())
    path, headers = (
        (SESSION_PATH, {})
        if credential == "session"
        else ("/external-api/v1/me", await _api_key_headers(user_id))
    )

    async with _client() as client:
        statuses = []
        for flushed in range(2):
            statuses.append((await client.get(path, headers=headers)).status_code)
            # The request itself never touched Redis.
            assert get_redis.await_count == flushed
            # Ten times the write's own timeout, so a lost timeout fails, not hangs.
            await asyncio.wait_for(api_activity.flush(), timeout=20)

    assert statuses == [200, 200]
    assert get_redis.await_count == 2
    warnings = [r for r in caplog.records if r.name == api_activity.logger.name]
    assert len(warnings) == 1
    assert api_activity._failures.held_back == 1


async def _api_key_headers(user_id: str) -> dict[str, str]:
    _, key = await api_key_db.create_api_key(
        "counted", user_id, [APIKeyPermission.IDENTITY]
    )
    return {"X-API-Key": key}


async def _counted(day: date, user_id: str) -> api_activity.APIRequestCounts:
    while not (counts := await api_activity.get_request_counts(day, [user_id]))[
        user_id
    ].total:
        await asyncio.sleep(0.05)
    return counts[user_id]


def _client() -> httpx.AsyncClient:
    # On the test's own loop: the session's Prisma client is bound to it.
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=rest_app), base_url="http://test"
    )
