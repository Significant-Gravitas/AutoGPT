from unittest.mock import AsyncMock, patch

import pytest
from tenacity import wait_none

from backend.blocks.conductor._api import API_V0, RETRY_MAX_ATTEMPTS, ConductorClient
from backend.blocks.conductor.test_fixtures import TEST_CREDENTIALS, FakeResponse
from backend.sdk import Requests


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
@pytest.mark.parametrize("status", [429, 503])
async def test_mutations_are_attempted_only_once(method: str, status: int):
    request = AsyncMock(return_value=FakeResponse(status, {"message": "try later"}))
    client = ConductorClient(TEST_CREDENTIALS)
    with (
        patch.object(Requests, "_request", request),
        patch("backend.util.request.wait_exponential_jitter", return_value=wait_none()),
    ):
        with pytest.raises(ValueError, match=f"HTTP {status}"):
            await client._call(method, f"{API_V0}/workspaces", json_body={})

    assert request.await_count == 1


@pytest.mark.asyncio
async def test_reads_keep_bounded_retries():
    request = AsyncMock(return_value=FakeResponse(503, {"message": "try later"}))
    client = ConductorClient(TEST_CREDENTIALS)
    with (
        patch.object(Requests, "_request", request),
        patch("backend.util.request.wait_exponential_jitter", return_value=wait_none()),
    ):
        with pytest.raises(ValueError, match="HTTP 503"):
            await client.get_me()

    assert request.await_count == RETRY_MAX_ATTEMPTS
