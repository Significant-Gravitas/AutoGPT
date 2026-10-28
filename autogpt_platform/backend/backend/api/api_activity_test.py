"""A real authenticated request through the app counts once against its account,
for a session on the internal API and an API key on the external one."""

import random
from datetime import UTC, date, datetime, time, timedelta
from uuid import uuid4

import httpx
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import APIKeyPermission
from prisma.models import APIKey, User

from backend.api.rest_api import app as rest_app
from backend.data import api_activity
from backend.data.auth import api_key as api_key_db
from backend.data.redis_client import HOST, PORT
from backend.util.test import SpinTestServer
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(HOST, PORT), reason="needs a live Redis"
)


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


@pytest.fixture
async def user_id(server: SpinTestServer):
    user_id = str(uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@example.com", "name": "Counted"}
    )
    yield user_id
    await APIKey.prisma().delete_many(where={"userId": user_id})
    await User.prisma().delete_many(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_a_session_request_counts_once_as_internal(mocker, user_id, day):
    # Only the token is faked: SpinTestServer's own override of `get_user_id`
    # would skip the resolver that counts.
    mocker.patch.dict(
        rest_app.dependency_overrides,
        {get_jwt_payload: lambda: {"sub": user_id, "role": "user"}},
        clear=True,
    )

    async with _client() as client:
        response = await client.get("/api/desktop-preview?token=not-a-link")

    assert response.status_code == 404
    counts = await api_activity.get_request_counts(day, [user_id])
    assert counts[user_id].model_dump() == {"internal": 1, "external": 0}


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("path", ["/external-api/v1/me", "/api/v2/me"])
async def test_an_api_key_request_counts_once_as_external(user_id, day, path):
    # v2 verifies the key in its rate limiter and again in the route.
    _, key = await api_key_db.create_api_key(
        "counted", user_id, [APIKeyPermission.IDENTITY]
    )

    async with _client() as client:
        await client.get(path, headers={"X-API-Key": key})

    counts = await api_activity.get_request_counts(day, [user_id])
    assert counts[user_id].model_dump() == {"internal": 0, "external": 1}


def _client() -> httpx.AsyncClient:
    # On the test's own loop: the session's Prisma client is bound to it.
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=rest_app), base_url="http://test"
    )
