"""Shared utilities for video blocks."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import signal
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

import psutil

from backend.blocks.video import _render
from backend.blocks.video._render import EditRefused, check_frame_size, open_video
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError
from backend.util.file import MediaFileType, store_media_file

if TYPE_CHECKING:
    from moviepy.video.io.VideoFileClip import VideoFileClip

    from backend.blocks._base import AnyBlockSchema

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

# Run as a script, so the child doesn't import the blocks package
_EDIT_SCRIPT = os.path.abspath(_render.__file__)
# How much of a failed edit's stderr its log line keeps
_STDERR_TAIL = 2_000

# Known operation tags added by video blocks
_VIDEO_OPS = (
    r"(?:clip|overlay|narrated|looped|concat|audio_attached|with_audio|narration)"
)

# Matches: {node_exec_id}_{operation}_ where node_exec_id contains a UUID
_BLOCK_PREFIX_RE = re.compile(
    r"^[a-zA-Z0-9_-]*"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"[a-zA-Z0-9_-]*"
    r"_" + _VIDEO_OPS + r"_"
)

# Matches: a lone {node_exec_id}_ prefix (no operation keyword, e.g. download output)
_UUID_PREFIX_RE = re.compile(
    r"^[a-zA-Z0-9_-]*"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"[a-zA-Z0-9_-]*_"
)


async def run_edit(
    block: AnyBlockSchema,
    edit: Callable[Concatenate[list[str], P], T],
    inputs: list[str],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Do one of `_render`'s edits, on copies of its `inputs`, in a child
    process, and return its result.

    The event loop stays free while the edit runs, so the executor's time
    limit on the block and a stop request can end it; either kills the child
    with every ffmpeg it started. The block's own time limit does the same
    for a caller that sets none.

    Raises BlockExecutionError for an input the edit refuses, TimeoutError
    when the block's time limit passes, and RuntimeError for any other
    failure.
    """
    timeout = block.execution_timeout_seconds
    # A cleanup failure must not replace the edit's own outcome, a cancel included
    with tempfile.TemporaryDirectory(
        prefix="video-edit-", ignore_cleanup_errors=True
    ) as scratch:
        request = {
            "edit": edit.__name__,
            "args": [_copy_inputs(inputs, scratch), *args],
            "kwargs": kwargs,
            "answer_path": os.path.join(scratch, "answer.json"),
        }
        exit_code = await _run_child(request, scratch, timeout)
        answer = _read_answer(request["answer_path"])
        if answer is None:
            stderr = Path(scratch, "stderr.log").read_text(errors="replace")
            logger.warning(
                f"Video edit {edit.__name__} ended with exit code {exit_code} "
                f"and no answer: {stderr[-_STDERR_TAIL:]}"
            )
            raise RuntimeError(
                f"Video editing stopped unexpectedly (exit code {exit_code})"
            )
    if "refused" in answer:
        raise BlockExecutionError(
            message=answer["refused"], block_name=block.name, block_id=str(block.id)
        )
    if "error" in answer:
        raise RuntimeError(answer["error"])
    return answer["value"]


def _copy_inputs(inputs: list[str], scratch: str) -> list[str]:
    """Copy the edit's input files into its scratch directory, here on the
    event loop, so another node storing the same input can't change one
    mid-copy or later under the edit, and the edit, which strips chapters in
    place, changes no file another node reads."""
    return [
        shutil.copyfile(path, os.path.join(scratch, f"{i}-{os.path.basename(path)}"))
        for i, path in enumerate(inputs)
    ]


async def _run_child(
    request: dict[str, Any], scratch: str, timeout: float | None
) -> int | None:
    """Run the edit script on `request` and return its exit code, killing it
    on a cancel or once `timeout` passes."""
    with open(os.path.join(scratch, "stderr.log"), "wb") as stderr:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            _EDIT_SCRIPT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=stderr,
            # moviepy writes its temporary audio track to the working directory
            cwd=scratch,
            # The ffmpeg processes it starts join its group, so die with it
            start_new_session=True,
        )
        try:
            await asyncio.wait_for(
                child.communicate(json.dumps(request).encode()), timeout
            )
        except TimeoutError as e:
            await _kill(child)
            raise TimeoutError(
                f"Video editing did not finish within {timeout} seconds"
            ) from e
        except BaseException:
            await _kill(child)
            raise
    return child.returncode


async def _kill(child: asyncio.subprocess.Process) -> None:
    """Kill the child and every process it started, and reap it."""
    if child.returncode is None:
        if sys.platform == "win32":
            _kill_tree(child.pid)
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
    await child.wait()


def _kill_tree(pid: int) -> None:
    """Kill a process and its descendants, where there are no process groups."""
    try:
        parent = psutil.Process(pid)
        processes = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return
    for process in processes:
        with contextlib.suppress(psutil.NoSuchProcess):
            process.kill()


def _read_answer(answer_path: str) -> dict[str, Any] | None:
    try:
        with open(answer_path, encoding="utf-8") as answer_file:
            return json.load(answer_file)
    except (OSError, ValueError):
        return None


def extract_source_name(input_path: str, max_length: int = 50) -> str:
    """Extract the original source filename by stripping block-generated prefixes.

    Iteratively removes {node_exec_id}_{operation}_ prefixes that accumulate
    when chaining video blocks, recovering the original human-readable name.

    Safe for plain filenames (no UUID -> no stripping).
    Falls back to "video" if everything is stripped.
    """
    stem = Path(input_path).stem

    # Pass 1: strip {node_exec_id}_{operation}_ prefixes iteratively
    while _BLOCK_PREFIX_RE.match(stem):
        stem = _BLOCK_PREFIX_RE.sub("", stem, count=1)

    # Pass 2: strip a lone {node_exec_id}_ prefix (e.g. from download block)
    if _UUID_PREFIX_RE.match(stem):
        stem = _UUID_PREFIX_RE.sub("", stem, count=1)

    if not stem:
        return "video"

    return stem[:max_length]


def check_video_frame_size(video_path: str, *, block_name: str, block_id: str) -> None:
    """Raise if any video stream's frame (width x height, from the header) is over MAX_VIDEO_FRAME_PIXELS."""
    try:
        check_frame_size(video_path)
    except EditRefused as e:
        raise BlockExecutionError(
            message=str(e), block_name=block_name, block_id=block_id
        ) from e


def open_video_clip(
    video_path: str, *, block_name: str, block_id: str
) -> VideoFileClip:
    """Check the frame size, then open the video in this process, for a block
    that only reads it; an edit opens its videos in `run_edit`'s child."""
    try:
        return open_video(video_path)
    except EditRefused as e:
        raise BlockExecutionError(
            message=str(e), block_name=block_name, block_id=block_id
        ) from e


class VideoMediaStorage:
    """Storage steps shared by the video blocks, as methods so `test_mock` can replace them."""

    async def _store_input_video(
        self, execution_context: ExecutionContext, file: MediaFileType
    ) -> MediaFileType:
        return await store_media_file(
            file=file,
            execution_context=execution_context,
            return_format="for_local_processing",
        )

    async def _store_output_video(
        self, execution_context: ExecutionContext, file: MediaFileType
    ) -> MediaFileType:
        return await store_media_file(
            file=file,
            execution_context=execution_context,
            return_format="for_block_output",
        )
