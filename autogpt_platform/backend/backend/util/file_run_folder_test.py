"""Runs that share an execution id keep their local files apart, and each
removes its own when it ends."""

import asyncio
from pathlib import Path

import pytest

from backend.util import file as file_module
from backend.util.file import clean_exec_files, get_exec_file_path, separate_exec_files

_EXEC_ID = "exec-1"


@pytest.fixture(autouse=True)
def temp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(file_module, "TEMP_DIR", tmp_path)
    return tmp_path


def _write(name: str, content: str) -> Path:
    path = Path(get_exec_file_path(_EXEC_ID, name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


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

    (first_path, first_text), second_path = await asyncio.wait_for(
        asyncio.gather(first_run(), second_run()), timeout=30
    )

    assert first_text == "first"
    assert first_path != second_path
    assert not first_path.exists() and not second_path.exists()
    assert list((temp_dir / "exec_file").iterdir()) == []
