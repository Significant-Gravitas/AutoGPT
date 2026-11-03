"""The host-side helpers open only what their check approved."""

import importlib
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from backend.copilot.baseline.service import _prepare_baseline_attachments
from backend.copilot.host_fs import open_no_follow, save_new_file, unlink_no_follow
from backend.copilot.sdk.e2b_file_tools import (
    _handle_edit_file,
    _handle_read_file,
    _handle_write_file,
)
from backend.copilot.sdk.file_ref import read_file_bytes
from backend.copilot.sdk.service import _prepare_file_attachments
from backend.copilot.sdk.tool_adapter import (
    get_copilot_tool_names,
    get_sdk_disallowed_tools,
)
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.workdir import (
    read_workdir_bytes,
    remove_from_workdir,
    save_to_workdir,
    set_executable,
)
from backend.copilot.tools.workspace_files import (
    _read_local_tool_result,
    _read_source_path,
)

_WORKSPACE_FILES = "backend.copilot.tools.workspace_files"
_WORKDIR = "backend.copilot.tools.workdir"
_E2B_FILE_TOOLS = "backend.copilot.sdk.e2b_file_tools"
_FILE_REF = "backend.copilot.sdk.file_ref"


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    session_dir = (tmp_path / "copilot-s1").resolve()
    session_dir.mkdir()
    monkeypatch.setattr(f"{_WORKDIR}.make_session_path", lambda _: str(session_dir))
    for module in (_E2B_FILE_TOOLS, _FILE_REF, _WORKSPACE_FILES):
        monkeypatch.setattr(f"{module}.get_sdk_cwd", lambda: str(session_dir))
    return session_dir


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    path = tmp_path / "outside.txt"
    path.write_text("outside old_text")
    path.chmod(0o600)
    return path


def _retarget(
    monkeypatch: pytest.MonkeyPatch, check: str, entry: Path, target: Path
) -> None:
    module, name = check.rsplit(".", 1)
    real = getattr(importlib.import_module(module), name)

    def retargeted(*args, **kwargs):
        result = real(*args, **kwargs)
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        entry.symlink_to(target)
        return result

    monkeypatch.setattr(check, retargeted)


def _unchanged(path: Path) -> bool:
    return path.read_text() == "outside old_text" and (
        stat.S_IMODE(path.stat().st_mode) == 0o600
    )


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


def test_open_no_follow_opens_a_regular_file(cwd):
    (cwd / "sub").mkdir()
    (cwd / "sub" / "a.txt").write_text("data")
    with open(cwd / "sub" / "a.txt", opener=open_no_follow) as f:
        assert f.read() == "data"


@pytest.mark.parametrize("mode", ["r", "w", "a"])
def test_open_no_follow_refuses_a_link_as_the_last_component(cwd, outside, mode):
    (cwd / "link").symlink_to(outside)
    with pytest.raises(OSError), open(cwd / "link", mode, opener=open_no_follow):
        pass
    assert _unchanged(outside)


@pytest.mark.parametrize(
    "below, flags, make_parents",
    [
        ("outside.txt", os.O_WRONLY | os.O_TRUNC, False),
        ("new/outside.txt", os.O_WRONLY | os.O_CREAT, True),
    ],
    ids=["existing-file", "new-parent"],
)
def test_open_no_follow_refuses_a_link_as_a_parent(
    cwd, outside, below, flags, make_parents
):
    (cwd / "dir").symlink_to(outside.parent)
    with pytest.raises(OSError):
        open_no_follow(str(cwd / "dir" / below), flags, make_parents=make_parents)
    assert _unchanged(outside)
    assert not (outside.parent / "new").exists()


def test_open_no_follow_makes_missing_parents(cwd):
    fd = open_no_follow(
        str(cwd / "a" / "b" / "c.txt"), os.O_WRONLY | os.O_CREAT, make_parents=True
    )
    os.close(fd)
    assert (cwd / "a" / "b" / "c.txt").is_file()


def test_open_no_follow_refuses_what_is_not_a_regular_file(cwd):
    fifo = cwd / "pipe"
    os.mkfifo(fifo)
    # A held writer keeps a regression from hanging the open instead of failing.
    writer = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
    try:
        with pytest.raises(OSError):
            open_no_follow(str(fifo), os.O_RDONLY)
    finally:
        os.close(writer)


def test_unlink_no_follow_removes_a_link_not_its_target(cwd, outside):
    (cwd / "link").symlink_to(outside)
    unlink_no_follow(str(cwd / "link"))
    assert not (cwd / "link").is_symlink()
    assert _unchanged(outside)


def test_unlink_no_follow_refuses_a_linked_parent(cwd, outside):
    (cwd / "dir").symlink_to(outside.parent)
    with pytest.raises(OSError):
        unlink_no_follow(str(cwd / "dir" / "outside.txt"))
    assert _unchanged(outside)


def test_save_new_file_never_writes_through_an_entry_already_there(cwd, tmp_path):
    (cwd / "doc.txt").symlink_to(tmp_path / "created.txt")
    (cwd / "doc_1.txt").write_text("taken")
    saved = save_new_file(str(cwd), "doc.txt", b"new")
    assert saved == str(cwd / "doc_2.txt")
    assert (cwd / "doc_2.txt").read_bytes() == b"new"
    assert (cwd / "doc_1.txt").read_text() == "taken"
    assert not (tmp_path / "created.txt").exists()


@pytest.mark.parametrize("name", ["..", ".", ""])
def test_save_new_file_keeps_a_dot_name_in_the_directory(cwd, name):
    saved = save_new_file(str(cwd), name, b"x")
    assert os.path.dirname(saved) == str(cwd)


# ---------------------------------------------------------------------------
# The helpers, one per entry point
# ---------------------------------------------------------------------------


async def test_write_workspace_file_source_path(cwd, outside, monkeypatch):
    (cwd / "f").write_text("inside")
    _retarget(
        monkeypatch, f"{_WORKSPACE_FILES}.validate_ephemeral_path", cwd / "f", outside
    )
    result = await _read_source_path(str(cwd / "f"), "s1")
    assert isinstance(result, ErrorResponse)


def test_read_workspace_file_local_fallback(cwd, outside, monkeypatch):
    (cwd / "f").write_text("inside")
    _retarget(
        monkeypatch,
        f"{_WORKSPACE_FILES}.resolve_allowed_local_path",
        cwd / "f",
        outside,
    )
    result = _read_local_tool_result(str(cwd / "f"), 0, None, "s1", sdk_cwd=str(cwd))
    assert isinstance(result, ErrorResponse)


async def test_save_to_workdir_last_component(cwd, outside, monkeypatch):
    (cwd / "f").write_text("inside")
    _retarget(monkeypatch, f"{_WORKDIR}.validate_ephemeral_path", cwd / "f", outside)
    result = await save_to_workdir(str(cwd / "f"), b"new", "s1")
    assert isinstance(result, ErrorResponse)
    assert _unchanged(outside)


async def test_save_to_workdir_parent(cwd, outside, monkeypatch):
    (cwd / "d").mkdir()
    _retarget(
        monkeypatch, f"{_WORKDIR}.validate_ephemeral_path", cwd / "d", outside.parent
    )
    result = await save_to_workdir(str(cwd / "d" / "outside.txt"), b"new", "s1")
    assert isinstance(result, ErrorResponse)
    assert _unchanged(outside)


async def test_read_workdir_bytes(cwd, outside, monkeypatch):
    (cwd / "f").write_text("inside")
    _retarget(monkeypatch, f"{_WORKDIR}.validate_ephemeral_path", cwd / "f", outside)
    assert await read_workdir_bytes(str(cwd / "f"), "s1") is None


async def test_remove_from_workdir_walks_its_parents_without_following(cwd, outside):
    (cwd / "skills").symlink_to(outside.parent)
    path = str(cwd / "skills" / "outside.txt")
    assert await remove_from_workdir([path], "s1") == [path]
    assert _unchanged(outside)


async def test_remove_from_workdir_removes_a_link_not_its_target(cwd, outside):
    (cwd / "link").symlink_to(outside)
    assert await remove_from_workdir([str(cwd / "link")], "s1") == []
    assert not (cwd / "link").is_symlink()
    assert _unchanged(outside)


async def test_remove_from_workdir_outside_the_directory(cwd, outside):
    assert await remove_from_workdir([str(outside)], "s1") == [str(outside)]
    assert _unchanged(outside)


async def test_set_executable_outside_the_directory(cwd, outside):
    assert await set_executable([str(outside)], True, "s1") == [str(outside)]
    assert _unchanged(outside)


async def test_set_executable_on_what_resolves_outside(cwd, outside):
    (cwd / "run.sh").symlink_to(outside)
    path = str(cwd / "run.sh")
    assert await set_executable([path], True, "s1") == [path]
    assert _unchanged(outside)


async def test_set_executable_acts_on_the_checked_entry(cwd, outside, monkeypatch):
    (cwd / "run.sh").write_text("inside")
    _retarget(
        monkeypatch, f"{_WORKDIR}.validate_ephemeral_path", cwd / "run.sh", outside
    )
    path = str(cwd / "run.sh")
    assert await set_executable([path], True, "s1") == [path]
    assert _unchanged(outside)


@pytest.mark.parametrize(
    "prepare, manager_at",
    [
        (_prepare_file_attachments, "backend.copilot.sdk.service"),
        (_prepare_baseline_attachments, "backend.copilot.baseline.service"),
    ],
)
async def test_attachment_copy(cwd, tmp_path, prepare, manager_at):
    (cwd / "doc.txt").symlink_to(tmp_path / "created.txt")
    manager = AsyncMock()
    manager.get_file_info.return_value = SimpleNamespace(
        id="f1", name="doc.txt", path="/doc.txt", mime_type="text/plain", size_bytes=3
    )
    manager.read_file_by_id.return_value = b"new"
    with patch(
        f"{manager_at}.get_workspace_manager",
        new_callable=AsyncMock,
        return_value=manager,
    ):
        await prepare(["f1"], "u", "s1", str(cwd))
    assert not (tmp_path / "created.txt").exists()
    assert (cwd / "doc_1.txt").read_bytes() == b"new"


async def test_agptfile_reference(cwd, outside, monkeypatch):
    (cwd / "f").write_text("inside")
    _retarget(
        monkeypatch, f"{_FILE_REF}.resolve_allowed_local_path", cwd / "f", outside
    )
    with pytest.raises(ValueError):
        await read_file_bytes(str(cwd / "f"), "u", make_session("u"))


async def test_local_read_file(cwd, outside, monkeypatch):
    (cwd / "f.txt").write_text("inside")
    _retarget(
        monkeypatch,
        f"{_E2B_FILE_TOOLS}.resolve_allowed_local_path",
        cwd / "f.txt",
        outside,
    )
    result = await _handle_read_file({"file_path": str(cwd / "f.txt")})
    assert result["isError"] is True
    assert "old_text" not in result["content"][0]["text"]


async def test_local_read_file_image(cwd, tmp_path, monkeypatch):
    # A valid image, so an error can come only from the refusal.
    Image.new("RGB", (2, 2)).save(tmp_path / "outside.png")
    (cwd / "f.png").write_text("inside")
    _retarget(
        monkeypatch,
        f"{_E2B_FILE_TOOLS}.resolve_allowed_local_path",
        cwd / "f.png",
        tmp_path / "outside.png",
    )
    result = await _handle_read_file({"file_path": str(cwd / "f.png")})
    assert result["isError"] is True


async def test_local_write_file(cwd, outside, monkeypatch):
    (cwd / "f").write_text("inside")
    _retarget(
        monkeypatch, f"{_E2B_FILE_TOOLS}.resolve_allowed_local_path", cwd / "f", outside
    )
    result = await _handle_write_file({"file_path": str(cwd / "f"), "content": "new"})
    assert result["isError"] is True
    assert _unchanged(outside)


async def test_local_edit_file(cwd, outside, monkeypatch):
    (cwd / "f").write_text("inside old_text")
    _retarget(
        monkeypatch, f"{_E2B_FILE_TOOLS}.resolve_allowed_local_path", cwd / "f", outside
    )
    result = await _handle_edit_file(
        {"file_path": str(cwd / "f"), "old_string": "old_text", "new_string": "new"}
    )
    assert result["isError"] is True
    assert _unchanged(outside)


@pytest.mark.parametrize("use_e2b", [False, True])
def test_cli_file_built_ins_are_off(use_e2b):
    allowed = set(get_copilot_tool_names(use_e2b=use_e2b))
    disallowed = set(get_sdk_disallowed_tools())
    for builtin in ("Glob", "Grep", "Read", "Write", "Edit"):
        assert builtin in disallowed
        assert builtin not in allowed


# ---------------------------------------------------------------------------
# A link that stays inside the directory keeps working
# ---------------------------------------------------------------------------


async def test_a_link_to_a_sibling_is_read_and_chmodded_but_removed_itself(cwd):
    (cwd / "real.sh").write_text("inside")
    (cwd / "real.sh").chmod(0o600)
    (cwd / "alias.sh").symlink_to(cwd / "real.sh")
    alias = str(cwd / "alias.sh")

    assert await _read_source_path(alias, "s1") == b"inside"
    assert await set_executable([alias], True, "s1") == []
    assert stat.S_IMODE((cwd / "real.sh").stat().st_mode) == 0o711
    assert await remove_from_workdir([alias], "s1") == []
    assert not (cwd / "alias.sh").is_symlink()
    assert (cwd / "real.sh").read_text() == "inside"
