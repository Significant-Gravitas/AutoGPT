import asyncio
import signal
import sys

import pytest
from pydantic import SecretStr

from backend.util.link_checkout import runner
from backend.util.link_checkout.models import WorkerJob

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the worker runs in its own process group"
)


@pytest.fixture
def spawned(monkeypatch) -> list[asyncio.subprocess.Process]:
    processes: list[asyncio.subprocess.Process] = []
    spawn = asyncio.create_subprocess_exec

    async def record(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.asyncio, "create_subprocess_exec", record)
    return processes


def job(intent) -> WorkerJob:
    return WorkerJob(action="status", intent=intent, access_token=SecretStr("t"))


@pytest.mark.asyncio
async def test_cancelling_kills_the_worker_and_stays_a_cancellation(
    monkeypatch, intent, spawned
):
    monkeypatch.setattr(runner, "_ENTRY", "import time; time.sleep(60)")
    task = asyncio.create_task(runner.run_worker(job(intent)))
    while not spawned:
        await asyncio.sleep(0.05)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert spawned[0].returncode == -signal.SIGKILL


@pytest.mark.asyncio
async def test_a_failed_worker_says_nothing_of_its_output(monkeypatch, intent):
    monkeypatch.setattr(
        runner, "_ENTRY", "print('canary-4242424242424242'); raise SystemExit(3)"
    )

    with pytest.raises(RuntimeError) as failure:
        await runner.run_worker(job(intent))

    assert "canary" not in str(failure.value)
    assert "never retry payment" in str(failure.value)
