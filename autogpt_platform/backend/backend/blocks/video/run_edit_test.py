"""A video block's edit runs in a child process: the event loop it shares with
other nodes stays free while it encodes, and the block's time limit, or a
stop, ends the encode itself rather than leaving it to run on."""

import asyncio
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from unittest.mock import AsyncMock, patch

import psutil
import pytest
from moviepy import ColorClip

from backend.blocks.video._render import clip_video
from backend.blocks.video._utils import run_edit
from backend.blocks.video.clip import VideoClipBlock
from backend.blocks.video.loop import LoopVideoBlock
from backend.data.execution import ExecutionContext


@pytest.fixture
def one_second_clip(tmp_path: Path) -> Path:
    path = tmp_path / "one_second.mp4"
    # Ten frames a second, so looping it to an hour encodes 36,000 frames
    ColorClip((16, 16), color=(0, 0, 0), duration=1).write_videofile(
        str(path), fps=10, logger=None
    )
    return path


async def test_a_long_encode_leaves_the_loop_free_and_the_time_limit_ends_it(
    one_second_clip: Path,
):
    # As the executor runs a block: under `asyncio.wait_for` with the block's
    # time limit, on a loop other nodes share. Looping a one-second clip to an
    # hour takes far longer to encode than the limit here.
    ticks = [0]
    started: dict[int, psutil.Process] = {}
    another_node = asyncio.ensure_future(_tick(ticks))
    watcher = asyncio.ensure_future(_note_child_processes(started))
    encode = asyncio.ensure_future(_loop_to_an_hour(LoopVideoBlock(), one_second_clip))
    try:
        await _until(lambda: any(_is_ffmpeg(p) for p in started.values()))
        ticks_while_encoding = ticks[0]
        limit_set = time.monotonic()

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(encode, timeout=1)

        # Ended at the limit, not when the encode would have finished
        assert time.monotonic() - limit_set < 10
        assert ticks[0] - ticks_while_encoding >= 5
    finally:
        await _cancel(another_node, watcher, encode)
    await _until_ended(started.values())


async def test_the_blocks_own_time_limit_ends_an_encode_nothing_else_limits(
    one_second_clip: Path,
):
    block = LoopVideoBlock()
    block.execution_timeout_seconds = 2
    started: dict[int, psutil.Process] = {}
    watcher = asyncio.ensure_future(_note_child_processes(started))
    began = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="did not finish within 2 seconds"):
            await _loop_to_an_hour(block, one_second_clip)
    finally:
        await _cancel(watcher)
    assert time.monotonic() - began < 20
    assert started
    await _until_ended(started.values())


async def test_an_edit_gets_its_arguments_and_returns_its_result(
    one_second_clip: Path,
):
    output = one_second_clip.with_name("first_half.mp4")

    duration = await run_edit(
        VideoClipBlock(),
        clip_video,
        [str(one_second_clip)],
        str(output),
        start_time=0.0,
        end_time=0.5,
    )

    assert duration == pytest.approx(0.5)
    assert output.stat().st_size > 0


async def test_an_edit_works_on_its_own_copy_of_its_inputs(one_second_clip: Path):
    # Two nodes of a run that take the same video store it at the same path,
    # and the second writes it over again while the first one's edit runs
    original = one_second_clip.read_bytes()
    edit = asyncio.ensure_future(
        run_edit(
            VideoClipBlock(),
            clip_video,
            [str(one_second_clip)],
            str(one_second_clip.with_name("first_half.mp4")),
            0.0,
            0.5,
        )
    )
    await asyncio.sleep(0)
    one_second_clip.write_bytes(b"")

    assert await edit == pytest.approx(0.5)
    one_second_clip.write_bytes(original)
    await run_edit(
        VideoClipBlock(),
        clip_video,
        [str(one_second_clip)],
        str(one_second_clip.with_name("again.mp4")),
        0.0,
        0.5,
    )
    # Its chapters are stripped from its copy, not from the stored file
    assert one_second_clip.read_bytes() == original


def test_an_edit_runs_from_an_event_loop_on_another_thread(one_second_clip: Path):
    # As the executor runs nodes: on a loop each graph worker runs in a thread
    # of its own, not the main thread
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        edit = run_edit(
            VideoClipBlock(),
            clip_video,
            [str(one_second_clip)],
            str(one_second_clip.with_name("first_half.mp4")),
            0.0,
            0.5,
        )
        duration = asyncio.run_coroutine_threadsafe(edit, loop).result(timeout=120)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert duration == pytest.approx(0.5)


async def _loop_to_an_hour(block: LoopVideoBlock, clip: Path) -> None:
    with (
        patch(
            "backend.blocks.video.loop.store_media_file",
            new=AsyncMock(side_effect=lambda file, **_: file),
        ),
        patch(
            "backend.blocks.video.loop.get_exec_file_path",
            side_effect=lambda _graph_exec_id, path: str(clip.parent / path),
        ),
    ):
        async for _ in block.run(
            LoopVideoBlock.Input(video_in=clip.name, duration=3600),
            execution_context=ExecutionContext(
                user_id="user-1", graph_exec_id="run-1", node_exec_id="node-1"
            ),
        ):
            pass


async def _tick(ticks: list[int]) -> None:
    while True:
        await asyncio.sleep(0.05)
        ticks[0] += 1


async def _note_child_processes(started: dict[int, psutil.Process]) -> None:
    """Note every process this test's process starts, however indirectly."""
    me = psutil.Process()
    while True:
        try:
            for process in me.children(recursive=True):
                started.setdefault(process.pid, process)
        except psutil.Error:
            pass  # One ended mid-walk
        await asyncio.sleep(0.02)


def _is_ffmpeg(process: psutil.Process) -> bool:
    try:
        return "ffmpeg" in process.name().lower()
    except psutil.Error:
        return False


async def _until(condition: Callable[[], bool], timeout: float = 60) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.02)


async def _until_ended(processes: Iterable[psutil.Process]) -> None:
    """Every process the edit started ends, its ffmpegs included."""
    processes = list(processes)
    await _until(lambda: not any(_alive(p) for p in processes), timeout=10)


def _alive(process: psutil.Process) -> bool:
    # A killed process stays listed as a zombie until whoever adopted it reaps it
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


async def _cancel(*tasks: asyncio.Future) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
