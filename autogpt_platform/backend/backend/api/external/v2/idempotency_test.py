"""
Tests for retry safety on the endpoints that spend money.

A run costs the caller credits, and a request that times out says nothing about
whether it started. These pin the three outcomes a client needs to be able to
tell apart: the run started, the run already started, and it is still starting.
"""

from typing import Optional
from unittest import mock

import fastapi
import pytest
import pytest_mock
from prisma.enums import APIKeyPermission

from backend.api.external.v2.idempotency import (
    IDEMPOTENCY_HEADER,
    idempotency_key,
    idempotent_run,
    replayed_run,
)
from backend.api.external.v2.tenancy import TenantContext
from backend.util.exceptions import NotFoundError

USER_ID = "user-1"
ORG_ID = "org-1"
KEY = "client-chosen-key"


async def test_a_first_request_claims_the_key_and_records_its_run(
    redis: mock.AsyncMock,
) -> None:
    async with idempotent_run(KEY, USER_ID) as claim:
        assert claim.existing_run_id is None
        await claim.record("run-1")

    redis.set.assert_any_await(
        f"v2:idem:{USER_ID}:{KEY}", "in-flight", nx=True, ex=86400
    )
    redis.set.assert_awaited_with(f"v2:idem:{USER_ID}:{KEY}", "run-1", ex=86400)


async def test_a_retry_with_the_same_key_reports_the_first_run(
    redis: mock.AsyncMock,
) -> None:
    """Without this the retry is a second execution and a second charge."""
    redis.set.return_value = None  # the key is already claimed
    redis.get.return_value = "run-1"

    async with idempotent_run(KEY, USER_ID) as claim:
        assert claim.existing_run_id == "run-1"


async def test_a_retry_while_the_first_is_still_running_is_a_conflict(
    redis: mock.AsyncMock,
) -> None:
    redis.set.return_value = None
    redis.get.return_value = "in-flight"

    with pytest.raises(fastapi.HTTPException) as raised:
        async with idempotent_run(KEY, USER_ID):
            pass

    assert raised.value.status_code == 409


async def test_a_failed_run_releases_its_key_for_a_retry(
    redis: mock.AsyncMock,
) -> None:
    """A key held by a request that never produced a run would lock the caller
    out of retrying with the value they chose, for 24 hours."""
    with pytest.raises(RuntimeError):
        async with idempotent_run(KEY, USER_ID):
            raise RuntimeError("enqueue failed")

    redis.delete.assert_awaited_once_with(f"v2:idem:{USER_ID}:{KEY}")


async def test_no_key_means_no_claim(redis: mock.AsyncMock) -> None:
    async with idempotent_run(None, USER_ID) as claim:
        assert claim.existing_run_id is None
        await claim.record("run-1")

    redis.set.assert_not_awaited()


async def test_an_unreachable_key_store_does_not_refuse_the_run(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Losing retry safety is where a caller without a key already is; losing
    the ability to run at all is not."""
    mocker.patch(
        "backend.api.external.v2.idempotency.get_redis_async",
        new_callable=mock.AsyncMock,
        side_effect=RuntimeError("redis is down"),
    )

    async with idempotent_run(KEY, USER_ID) as claim:
        assert claim.existing_run_id is None


async def test_a_replay_is_scoped_to_the_key_s_tenant(
    redis: mock.AsyncMock, mocker: pytest_mock.MockFixture
) -> None:
    """The recorded run id is a bare string; it must still pass the tenant check."""
    redis.set.return_value = None
    redis.get.return_value = "run-in-another-org"
    mocker.patch(
        "backend.data.execution.get_graph_execution",
        new_callable=mock.AsyncMock,
        return_value=mock.Mock(organization_id="another-org"),
    )

    async with idempotent_run(KEY, USER_ID) as claim:
        with pytest.raises(NotFoundError):
            await replayed_run(claim, _tenant())


@pytest.mark.parametrize(
    "sent,expected", [(None, None), ("", None), ("  ", None), (" k ", "k")]
)
def test_a_blank_header_is_no_key(sent: Optional[str], expected: Optional[str]) -> None:
    assert idempotency_key(sent) == expected


def test_both_run_endpoints_accept_the_header() -> None:
    """A key on one endpoint and not the other is worse than none on either."""
    from .app import v2_app

    schema = v2_app.openapi()
    for path in (
        "/library/agents/{agent_id}/runs",
        "/library/presets/{preset_id}/runs",
    ):
        names = {p["name"] for p in schema["paths"][path]["post"].get("parameters", [])}
        assert IDEMPOTENCY_HEADER in names, f"{path} takes no {IDEMPOTENCY_HEADER}"


@pytest.fixture
def redis(mocker: pytest_mock.MockFixture) -> mock.AsyncMock:
    client = mock.AsyncMock()
    client.set.return_value = True
    mocker.patch(
        "backend.api.external.v2.idempotency.get_redis_async",
        new_callable=mock.AsyncMock,
        return_value=client,
    )
    return client


def _tenant() -> TenantContext:
    return TenantContext(
        user_id=USER_ID,
        scopes=list(APIKeyPermission),
        type="api_key",
        organization_id=ORG_ID,
    )
