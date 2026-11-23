"""Runs that share an execution id keep their local files apart, and each
removes its own when it ends. So does a block call with the copies it stages."""

import asyncio
import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.execution import ExecutionContext
from backend.util import file as file_module
from backend.util.file import (
    clean_exec_files,
    get_exec_file_path,
    remove_staged_copies_on_exit,
    separate_exec_files,
    store_media_file,
)
from backend.util.type import MediaFileType

_EXEC_ID = "exec-1"


@pytest.fixture(autouse=True)
def temp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(file_module, "TEMP_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def staging(monkeypatch: pytest.MonkeyPatch) -> None:
    """``store_media_file`` without the virus scanner or cloud storage."""
    monkeypatch.setattr(file_module, "scan_content_safe", AsyncMock())
    storage = MagicMock()
    storage.is_cloud_path.return_value = False
    monkeypatch.setattr(
        file_module, "get_cloud_storage_handler", AsyncMock(return_value=storage)
    )


def _write(name: str, content: str) -> Path:
    path = Path(get_exec_file_path(_EXEC_ID, name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


async def _stage(content: str) -> Path:
    """Stage a file for local processing as a block does, and return where
    its copy is."""
    encoded = base64.b64encode(content.encode()).decode()
    local_path = await store_media_file(
        MediaFileType(f"data:text/plain;base64,{encoded}"),
        ExecutionContext(user_id="user-1", graph_exec_id=_EXEC_ID),
        return_format="for_local_processing",
    )
    return Path(get_exec_file_path(_EXEC_ID, local_path))


def test_outside_a_run_files_go_in_the_execution_folder(temp_dir: Path) -> None:
    path = _write("a.txt", "graph run")

    assert path == temp_dir / "exec_file" / _EXEC_ID / "a.txt"
    clean_exec_files(_EXEC_ID)
    assert not (temp_dir / "exec_file" / _EXEC_ID).exists()


def test_a_run_s_files_are_removed_when_it_ends(temp_dir: Path) -> None:
    with separate_exec_files(_EXEC_ID):
        path = _write("a.txt", "call")
        assert path.parent.parent == temp_dir / "exec_file" / _EXEC_ID

    assert not path.exists()
    assert list((temp_dir / "exec_file").iterdir()) == []


def test_a_run_cannot_reach_into_another_run_s_folder() -> None:
    with separate_exec_files(_EXEC_ID):
        other_run = Path(get_exec_file_path(_EXEC_ID, "")).name

    with separate_exec_files(_EXEC_ID), pytest.raises(ValueError):
        get_exec_file_path(_EXEC_ID, f"../{other_run}/out.txt")


def test_a_run_s_files_are_removed_when_it_fails(temp_dir: Path) -> None:
    written: list[Path] = []

    with (
        pytest.raises(RuntimeError, match="block failed"),
        separate_exec_files(_EXEC_ID),
    ):
        written.append(_write("a.txt", "call"))
        raise RuntimeError("block failed")

    assert len(written) == 1 and not written[0].exists()
    assert list((temp_dir / "exec_file").iterdir()) == []


async def test_a_run_ending_leaves_a_concurrent_run_s_files_alone(
    temp_dir: Path,
) -> None:
    first_wrote = asyncio.Event()
    second_done = asyncio.Event()

    async def first_run() -> tuple[Path, str]:
        with separate_exec_files(_EXEC_ID):
            path = _write("out.txt", "first")
            first_wrote.set()
            await second_done.wait()
            return path, path.read_text()

    async def second_run() -> Path:
        await first_wrote.wait()
        with separate_exec_files(_EXEC_ID):
            path = _write("out.txt", "second")
            assert path.read_text() == "second"
        second_done.set()
        return path

    async with asyncio.timeout(30), asyncio.TaskGroup() as runs:
        first = runs.create_task(first_run())
        second = runs.create_task(second_run())
    (first_path, first_text), second_path = first.result(), second.result()

    assert first_text == "first"
    assert first_path != second_path
    assert not first_path.exists() and not second_path.exists()
    assert list((temp_dir / "exec_file").iterdir()) == []


async def test_a_block_call_s_copies_go_when_it_ends(
    temp_dir: Path, staging: None
) -> None:
    """Only its copies: one staged outside it, and a file a block wrote under
    a name of its own, stay until the run ends."""
    outside = await _stage("staged outside")
    output = _write("node-1_concat_clip.mp4", "a block's output")

    with remove_staged_copies_on_exit():
        inside = await _stage("staged inside")
        assert inside.read_text() == "staged inside"

    assert not inside.parent.exists()
    assert outside.read_text() == "staged outside"
    assert output.read_text() == "a block's output"
    clean_exec_files(_EXEC_ID)
    assert list((temp_dir / "exec_file").iterdir()) == []


async def test_a_failed_block_call_s_copies_go_too(staging: None) -> None:
    staged: list[Path] = []

    with (
        pytest.raises(RuntimeError, match="block failed"),
        remove_staged_copies_on_exit(),
    ):
        staged.append(await _stage("staged inside"))
        raise RuntimeError("block failed")

    assert len(staged) == 1 and not staged[0].parent.exists()


async def test_a_block_call_ending_leaves_a_running_one_s_copies_alone(
    staging: None,
) -> None:
    """The nodes of a run run at the same time, on one event loop."""
    first_staged = asyncio.Event()
    second_done = asyncio.Event()

    async def first_call() -> tuple[Path, str]:
        with remove_staged_copies_on_exit():
            copy = await _stage("first")
            first_staged.set()
            await second_done.wait()
            return copy, copy.read_text()

    async def second_call() -> Path:
        await first_staged.wait()
        with remove_staged_copies_on_exit():
            copy = await _stage("second")
        second_done.set()
        return copy

    # A task group cancels and joins the other call if one fails.
    async with asyncio.timeout(30), asyncio.TaskGroup() as calls:
        first = calls.create_task(first_call())
        second = calls.create_task(second_call())
    (first_copy, first_text), second_copy = first.result(), second.result()

    assert first_text == "first"
    assert not first_copy.exists() and not second_copy.exists()
