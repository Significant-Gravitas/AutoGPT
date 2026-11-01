import asyncio
import os
import resource
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.api.features.workspace import preview_render

# Stub renderers. The child process imports them from this module by name, so
# they stay at module level and this module imports nothing heavy.


def echo_renderer(content: bytes, width: int) -> bytes:
    return content + str(width).encode()


def hanging_renderer(_content: bytes, _width: int) -> bytes:
    time.sleep(120)
    return b"never"


def allocating_renderer(content: bytes, _width: int) -> bytes:
    # The content names how many bytes to allocate.
    block = bytearray(int(content))
    return bytes(block[:1]) or b"x"


def too_large_renderer(_content: bytes, _width: int) -> bytes:
    raise preview_render.PreviewTooLarge()


def failing_renderer(_content: bytes, _width: int) -> bytes:
    raise ValueError("cannot decode")


def crashing_renderer(_content: bytes, _width: int) -> bytes:
    os.abort()


def interval_renderer(_content: bytes, _width: int) -> bytes:
    # Report when the render ran, so the parent can count overlapping renders.
    start = time.time()
    time.sleep(0.5)
    return f"{start},{time.time()}".encode()


@pytest.fixture
def started_children(mocker) -> list[asyncio.subprocess.Process]:
    started: list[asyncio.subprocess.Process] = []
    start = preview_render._start_child

    async def record(*args, **kwargs):
        process = await start(*args, **kwargs)
        started.append(process)
        return process

    mocker.patch.object(preview_render, "_start_child", record)
    return started


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_render_returns_the_child_output():
    output = await preview_render.render_in_child(echo_renderer, b"bytes-", 64)

    assert output == b"bytes-64"


async def test_render_runs_in_a_separate_process(started_children):
    await preview_render.render_in_child(echo_renderer, b"x", 16)

    (child,) = started_children
    assert child.pid != os.getpid()


async def test_render_past_the_timeout_is_killed(mocker, started_children):
    mocker.patch.object(preview_render, "RENDER_TIMEOUT_SECONDS", 1)
    started = time.monotonic()

    with pytest.raises(preview_render.RenderFailed):
        await preview_render.render_in_child(hanging_renderer, b"x", 16)

    elapsed = time.monotonic() - started
    (child,) = started_children
    assert 1 <= elapsed < 30  # ended at the timeout, not when the render would
    assert child.returncode is not None
    assert not _is_running(child.pid)


async def test_render_over_the_memory_cap_is_too_large(mocker):
    cap = 256 * 1024 * 1024
    mocker.patch.object(preview_render, "RENDER_MAX_MEMORY_BYTES", cap)
    parent_peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    with pytest.raises(preview_render.PreviewTooLarge):
        await preview_render.render_in_child(
            allocating_renderer, str(2 * cap).encode(), 16
        )

    # ru_maxrss is in KiB on Linux; the allocation never touched this process.
    parent_peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert (parent_peak_after - parent_peak_before) * 1024 < cap


async def test_render_within_the_memory_cap_succeeds(mocker):
    cap = 256 * 1024 * 1024
    mocker.patch.object(preview_render, "RENDER_MAX_MEMORY_BYTES", cap)

    output = await preview_render.render_in_child(
        allocating_renderer, str(16 * 1024 * 1024).encode(), 16
    )

    assert output == b"\x00"


@pytest.mark.parametrize(
    "renderer, error",
    [
        (too_large_renderer, preview_render.PreviewTooLarge),
        (failing_renderer, preview_render.RenderFailed),
        (crashing_renderer, preview_render.RenderFailed),
    ],
)
async def test_render_outcomes_map_to_errors(renderer, error):
    with pytest.raises(error):
        await preview_render.render_in_child(renderer, b"x", 16)


async def test_render_output_above_the_limit_fails(mocker):
    mocker.patch.object(preview_render, "MAX_THUMBNAIL_BYTES", 1024)

    with pytest.raises(preview_render.RenderFailed):
        await preview_render.render_in_child(echo_renderer, b"x" * 4096, 16)


async def test_cancelled_render_kills_its_child(started_children):
    task = asyncio.create_task(
        preview_render.render_in_child(hanging_renderer, b"x", 16)
    )
    while not started_children:
        await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (child,) = started_children
    assert child.returncode is not None
    assert not _is_running(child.pid)


def test_hanging_renders_leave_the_default_executor_free(mocker):
    # Renders that hang must not hold the event loop's default thread pool,
    # which the API's token checks and other ``asyncio.to_thread`` calls share.
    # Run on a loop whose default pool is no bigger than the renders in flight.
    mocker.patch.object(preview_render, "RENDER_TIMEOUT_SECONDS", 5)
    renders = 4

    async def scenario() -> tuple[str, list[BaseException | bytes]]:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=renders))
        in_flight = [
            asyncio.create_task(
                preview_render.render_in_child(hanging_renderer, b"x", 16)
            )
            for _ in range(renders)
        ]
        await asyncio.sleep(0.5)
        answer = await asyncio.wait_for(asyncio.to_thread(lambda: "answered"), 3)
        assert not any(task.done() for task in in_flight)
        return answer, await asyncio.gather(*in_flight, return_exceptions=True)

    loop = asyncio.new_event_loop()
    try:
        with ThreadPoolExecutor(max_workers=1) as runner:
            answer, outcomes = runner.submit(
                loop.run_until_complete, scenario()
            ).result(timeout=60)
    finally:
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()

    assert answer == "answered"
    assert all(isinstance(o, preview_render.RenderFailed) for o in outcomes)
