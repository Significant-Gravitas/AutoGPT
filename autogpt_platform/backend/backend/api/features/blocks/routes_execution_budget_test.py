"""The block-execute route runs blocks away from the server's event loop and
under a time budget."""

import asyncio
import threading
from collections.abc import Callable
from unittest.mock import Mock

import fastapi
import pytest
import pytest_mock
from httpx import ASGITransport, AsyncClient

from backend.api.features.blocks.routes import router

app = fastapi.FastAPI()
app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


def _serve_block(mocker: pytest_mock.MockFixture, execute: Callable) -> None:
    block = Mock()
    block.disabled = False
    block.name = "TestBlock"
    block.execute = execute
    mocker.patch("backend.api.features.blocks.routes.get_block", return_value=block)
    user = Mock()
    user.timezone = "UTC"
    mocker.patch("backend.api.features.blocks.routes.get_user_by_id", return_value=user)
    mocker.patch(
        "backend.api.features.blocks.routes.execution_utils.block_usage_cost",
        return_value=(0, {}),
    )


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_server_answers_other_requests_while_a_block_runs(
    mocker: pytest_mock.MockFixture,
):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    async def blocking_execute(*args, **kwargs):
        started.set()
        # Synchronous wait: it holds whichever event loop runs the block.
        release.wait(timeout=10)
        finished.set()
        yield "output", "done"

    _serve_block(mocker, blocking_execute)

    async with _client() as client:
        execute = asyncio.create_task(
            client.post("/blocks/test-block/execute", json={})
        )
        try:
            assert await asyncio.to_thread(started.wait, 10)
            health = await client.get("/health")
            assert health.status_code == 200
            assert not finished.is_set()
        finally:
            release.set()
        response = await execute

    assert response.status_code == 200
    assert response.json() == {"output": ["done"]}


@pytest.mark.asyncio
async def test_block_runs_on_a_direct_execution_worker_thread(
    mocker: pytest_mock.MockFixture,
):
    thread_names: list[str] = []

    async def execute(*args, **kwargs):
        thread_names.append(threading.current_thread().name)
        yield "output", "done"

    _serve_block(mocker, execute)

    async with _client() as client:
        response = await client.post("/blocks/test-block/execute", json={})

    assert response.status_code == 200
    assert len(thread_names) == 1
    assert thread_names[0].startswith("direct-block-execution")


@pytest.mark.asyncio
async def test_block_past_the_time_budget_gets_a_504_and_is_cancelled(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        "backend.executor.direct_block_execution.DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS",
        0.5,
    )
    cancelled = threading.Event()

    async def never_finishes(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        yield "output", "unreachable"

    _serve_block(mocker, never_finishes)

    async with _client() as client:
        response = await asyncio.wait_for(
            client.post("/blocks/test-block/execute", json={}), timeout=10
        )
        health = await client.get("/health")

    assert response.status_code == 504
    assert "did not finish within 0.5 seconds" in response.json()["detail"]
    assert health.status_code == 200
    assert await asyncio.to_thread(cancelled.wait, 10)


@pytest.mark.asyncio
async def test_user_over_the_run_cap_gets_a_429_before_any_charge(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        "backend.executor.direct_block_execution.DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER",
        0,
    )
    executed: list[bool] = []

    async def execute(*args, **kwargs):
        executed.append(True)
        yield "output", "done"

    _serve_block(mocker, execute)
    charge = mocker.patch(
        "backend.api.features.blocks.routes.execution_utils.charge_for_direct_block_execution"
    )

    async with _client() as client:
        response = await client.post("/blocks/test-block/execute", json={})

    assert response.status_code == 429
    assert "Wait for one to finish" in response.json()["detail"]
    charge.assert_not_called()
    assert executed == []


@pytest.mark.asyncio
async def test_retired_worker_cap_gets_a_503_before_any_charge(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        "backend.executor.direct_block_execution.DIRECT_BLOCK_EXECUTION_MAX_RETIRED_WORKERS",
        0,
    )
    executed: list[bool] = []

    async def execute(*args, **kwargs):
        executed.append(True)
        yield "output", "done"

    _serve_block(mocker, execute)
    charge = mocker.patch(
        "backend.api.features.blocks.routes.execution_utils.charge_for_direct_block_execution"
    )

    async with _client() as client:
        response = await client.post("/blocks/test-block/execute", json={})

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]
    charge.assert_not_called()
    assert executed == []
