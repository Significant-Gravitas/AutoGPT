"""The block-execute route runs each block in a process of its own, ended at its
block's deadline."""

import asyncio
import gc
import os
import time
from pathlib import Path
from unittest.mock import Mock

import fastapi
import pytest
import pytest_mock
from httpx import ASGITransport, AsyncClient

from backend.api.features.blocks.routes import router
from backend.blocks._base import Block
from backend.executor import direct_block_run_parent
from backend.executor.direct_block_execution import (
    DirectBlockExecutionBusyError,
    start_run_parent,
    stop_run_parent,
)

app = fastapi.FastAPI()
app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


class _Block:
    """Has Block's default deadline unless it sets its own, as a real block does."""

    disabled = False

    @property
    def direct_run_timeout_seconds(self) -> float:
        return Block.direct_run_timeout_seconds


class _ReportsItsProcess(_Block):
    id = name = "reports-its-process"

    async def execute(self, input_data, **kwargs):
        yield "pid", os.getpid()


class _Computes(_Block):
    """Writes a beat every 20 ms and takes no cancel, so it runs until it is killed."""

    id = name = "computes"

    async def execute(self, input_data, **kwargs):
        while True:
            with open(input_data["beats"], "a") as beats:
                beats.write(".")
            time.sleep(0.02)
        yield "never", None


class _ComputesForLonger(_Computes):
    id = name = "computes-for-longer"
    direct_run_timeout_seconds = 3


class _TakesLongerThanTheDefault(_Block):
    id = name = "takes-longer-than-the-default"
    direct_run_timeout_seconds = 10

    async def execute(self, input_data, **kwargs):
        await asyncio.sleep(input_data["seconds"])
        yield "slept", input_data["seconds"]


_BLOCKS = {
    block.id: block
    for block in (
        _ReportsItsProcess,
        _Computes,
        _ComputesForLonger,
        _TakesLongerThanTheDefault,
    )
}


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def run_parent(monkeypatch: pytest.MonkeyPatch):
    # Set before the fork, so the run parent's children find these blocks.
    monkeypatch.setattr(direct_block_run_parent, "get_blocks", lambda: _BLOCKS)
    start_run_parent()
    yield
    stop_run_parent()
    gc.unfreeze()


def _serve_blocks(mocker: pytest_mock.MockFixture) -> Mock:
    mocker.patch(
        "backend.api.features.blocks.routes.get_block",
        side_effect=lambda block_id: _BLOCKS[block_id](),
    )
    user = Mock()
    user.timezone = "UTC"
    mocker.patch("backend.api.features.blocks.routes.get_user_by_id", return_value=user)
    return mocker.patch(
        "backend.api.features.blocks.routes.execution_utils.charge_for_direct_block_execution"
    )


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_the_block_runs_in_a_process_of_its_own(
    run_parent, mocker: pytest_mock.MockFixture
):
    _serve_blocks(mocker)

    async with _client() as client:
        response = await client.post("/blocks/reports-its-process/execute", json={})

    assert response.status_code == 200
    (pid,) = response.json()["pid"]
    assert pid != os.getpid()


@pytest.mark.asyncio
async def test_a_block_computing_past_its_budget_gets_a_504_and_is_killed(
    run_parent, mocker: pytest_mock.MockFixture, tmp_path: Path
):
    # Long enough for the child to start on a slow runner.
    mocker.patch.object(Block, "direct_run_timeout_seconds", 2)
    mocker.patch(
        "backend.executor.direct_block_execution.DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER",
        1,
    )
    # Past the cancel's grace and short of the child's own alarm, so only the run
    # parent's kill passes.
    mocker.patch(
        "backend.executor.direct_block_execution.DIRECT_BLOCK_EXECUTION_KILL_WAIT_SECONDS",
        4,
    )
    _serve_blocks(mocker)
    beats = tmp_path / "beats"

    async with _client() as client:
        run = asyncio.create_task(
            client.post("/blocks/computes/execute", json={"beats": str(beats)})
        )
        await _started(beats, run)
        try:
            health = await client.get("/health")
            response = await asyncio.wait_for(run, timeout=30)
        finally:
            await _stopped(run)
        stopped_at = beats.stat().st_size
        await asyncio.sleep(0.5)
        again = await client.post("/blocks/reports-its-process/execute", json={})

    assert health.status_code == 200
    assert response.status_code == 504
    assert "did not finish within 2 seconds" in response.json()["detail"]
    assert beats.stat().st_size == stopped_at
    # The user's one slot is free again.
    assert again.status_code == 200


@pytest.mark.asyncio
async def test_a_block_with_a_longer_deadline_runs_past_the_default(
    run_parent, mocker: pytest_mock.MockFixture
):
    mocker.patch.object(Block, "direct_run_timeout_seconds", 0.5)
    _serve_blocks(mocker)

    async with _client() as client:
        response = await asyncio.wait_for(
            client.post(
                "/blocks/takes-longer-than-the-default/execute", json={"seconds": 1.5}
            ),
            timeout=30,
        )

    assert response.status_code == 200
    assert response.json() == {"slept": [1.5]}


@pytest.mark.asyncio
async def test_a_block_with_a_longer_deadline_gets_a_504_at_its_own_deadline(
    run_parent, mocker: pytest_mock.MockFixture, tmp_path: Path
):
    mocker.patch.object(Block, "direct_run_timeout_seconds", 0.5)
    mocker.patch(
        "backend.executor.direct_block_execution.DIRECT_BLOCK_EXECUTION_KILL_WAIT_SECONDS",
        4,
    )
    _serve_blocks(mocker)
    beats = tmp_path / "beats"

    async with _client() as client:
        started = time.monotonic()
        response = await asyncio.wait_for(
            client.post(
                "/blocks/computes-for-longer/execute", json={"beats": str(beats)}
            ),
            timeout=30,
        )
        elapsed = time.monotonic() - started
    stopped_at = beats.stat().st_size
    await asyncio.sleep(0.5)

    assert response.status_code == 504
    assert "did not finish within 3 seconds" in response.json()["detail"]
    assert elapsed >= 3
    assert beats.stat().st_size == stopped_at


@pytest.mark.asyncio
async def test_a_run_parent_lost_after_the_claim_gets_a_503(
    run_parent, mocker: pytest_mock.MockFixture
):
    _serve_blocks(mocker)
    mocker.patch(
        "backend.api.features.blocks.routes.run_block_in_child",
        side_effect=DirectBlockExecutionBusyError(),
    )

    async with _client() as client:
        response = await client.post("/blocks/reports-its-process/execute", json={})

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


@pytest.mark.asyncio
async def test_user_over_the_run_cap_gets_a_429_before_any_charge(
    run_parent, mocker: pytest_mock.MockFixture
):
    mocker.patch(
        "backend.executor.direct_block_execution.DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER",
        0,
    )
    charge = _serve_blocks(mocker)

    async with _client() as client:
        response = await client.post("/blocks/reports-its-process/execute", json={})

    assert response.status_code == 429
    assert "Wait for one to finish" in response.json()["detail"]
    charge.assert_not_called()


@pytest.mark.asyncio
async def test_the_server_run_cap_gets_a_503_before_any_charge(
    run_parent, mocker: pytest_mock.MockFixture
):
    mocker.patch(
        "backend.executor.direct_block_execution.DIRECT_BLOCK_EXECUTION_MAX_RUNS", 0
    )
    charge = _serve_blocks(mocker)

    async with _client() as client:
        response = await client.post("/blocks/reports-its-process/execute", json={})

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]
    charge.assert_not_called()


@pytest.mark.asyncio
async def test_a_server_without_a_run_parent_gets_a_503_before_any_charge(
    mocker: pytest_mock.MockFixture,
):
    charge = _serve_blocks(mocker)

    async with _client() as client:
        response = await client.post("/blocks/reports-its-process/execute", json={})

    assert response.status_code == 503
    charge.assert_not_called()


async def _started(beats: Path, run: asyncio.Task) -> None:
    """Wait until the block has written its first beat, failing if the run ends
    first or the block never starts."""
    try:
        async with asyncio.timeout(30):
            while not beats.exists():
                if run.done():
                    run.result()
                    raise AssertionError("The run ended before its block started")
                await asyncio.sleep(0.02)
    except BaseException:
        await _stopped(run)
        raise


async def _stopped(run: asyncio.Task) -> None:
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)
