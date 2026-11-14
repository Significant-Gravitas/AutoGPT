"""FILE_LIST's limits: the entry cap, the reply's byte budget and the scan budget."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import autogpt_local_executor.handlers as handlers_module
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import FileHandler
from autogpt_local_executor.protocol import (
    ErrorMessage,
    FileListMessage,
    FileListPayload,
    FileListResponseMessage,
    dump_message,
    new_id,
    now_ts,
)


def _list(tmp_path: Path, **payload) -> FileListMessage:
    return FileListMessage(
        id=new_id(), ts=now_ts(), payload=FileListPayload(path=str(tmp_path), **payload)
    )


async def _listing(tmp_path: Path, **payload) -> FileListResponseMessage:
    handler = FileHandler(ShimConfig(allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log"))
    response = await handler.handle_list(_list(tmp_path, **payload))
    assert isinstance(response, FileListResponseMessage)
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(("files", "truncated"), [(5, False), (6, True)])
async def test_truncated_means_a_match_was_left_out(
    tmp_path: Path, files: int, truncated: bool
) -> None:
    """Exactly max_entries matches used to come back marked truncated."""
    for index in range(files):
        (tmp_path / f"f{index}").write_text("", encoding="utf-8")

    response = await _listing(tmp_path, max_entries=5)

    assert len(response.payload.entries) == 5
    assert response.payload.truncated is truncated


@pytest.mark.asyncio
async def test_a_listing_stops_at_its_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """10,000 entries with long paths could pass the message limit, and then
    the whole listing was replaced with FILE_TOO_LARGE."""
    for index in range(50):
        (tmp_path / f"{'long-name-' * 8}{index:02d}").write_text("", encoding="utf-8")
    monkeypatch.setattr(handlers_module, "FILE_LIST_REPLY_BUDGET_BYTES", 4000)

    response = await _listing(tmp_path)

    assert 0 < len(response.payload.entries) < 50
    assert response.payload.truncated is True
    entry_bytes = sum(len(e.model_dump_json().encode()) + 1 for e in response.payload.entries)
    assert entry_bytes <= 4000
    assert len(dump_message(response).encode()) < 4000 + 1000


@pytest.mark.asyncio
async def test_a_walk_that_matches_nothing_stops_at_its_scan_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a glob that matches nothing, a recursive walk visited the whole
    tree, holding a request slot and a thread until it was done."""
    for folder in range(5):
        for index in range(10):
            (tmp_path / f"d{folder}").mkdir(exist_ok=True)
            (tmp_path / f"d{folder}" / f"f{index}").write_text("", encoding="utf-8")
    monkeypatch.setattr(handlers_module, "MAX_FILE_LIST_SCANNED", 12)

    response = await _listing(tmp_path, recursive=True, glob="*.nothing")

    assert response.payload.entries == []
    assert response.payload.truncated is True


@pytest.mark.asyncio
@pytest.mark.parametrize("recursive", [True, False], ids=["recursive", "flat"])
async def test_a_walk_stops_part_way_through_a_directory_larger_than_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recursive: bool
) -> None:
    """os.walk, and iterdir, read a whole directory before the walk counts
    a single entry, so one directory with millions of entries was read, and
    held in memory, in full, whatever the budget."""
    import os

    for index in range(50):
        (tmp_path / f"f{index:02d}").write_text("", encoding="utf-8")
    monkeypatch.setattr(handlers_module, "MAX_FILE_LIST_SCANNED", 12)
    read = 0
    real_scandir = os.scandir
    real_listdir = os.listdir

    class CountingScandir:
        def __init__(self, path):
            self._entries = real_scandir(path)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self._entries.close()

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal read
            entry = next(self._entries)
            read += 1
            return entry

        def close(self):
            self._entries.close()

    def counting_listdir(path="."):
        nonlocal read
        names = real_listdir(path)
        read += len(names)
        return names

    monkeypatch.setattr(os, "scandir", CountingScandir)
    monkeypatch.setattr(os, "listdir", counting_listdir)

    response = await _listing(tmp_path, recursive=recursive, glob="*.nothing")

    assert response.payload.truncated is True
    # The walk reads one entry past its budget to know it is spent, and the
    # glob's case rule is read from the folder's first entry, whose name has
    # a letter (platform_info.is_case_insensitive_fs).
    assert read <= 13 + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("max_entries", [0, -3])
async def test_max_entries_below_one_lists_nothing(tmp_path: Path, max_entries: int) -> None:
    """The cap was raised to at least one, so asking for none returned one."""
    (tmp_path / "match").write_text("", encoding="utf-8")

    response = await _listing(tmp_path, max_entries=max_entries)

    assert response.payload.entries == []
    assert response.payload.truncated is True


@pytest.mark.asyncio
async def test_a_recursive_listing_never_enters_a_junction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """To os.scandir a Windows junction is a folder like any other, so the
    walk went through one pointing outside the shared folder and listed
    what is there. It's listed itself, but never entered."""
    (tmp_path / "junction").mkdir()
    (tmp_path / "junction" / "outside.txt").write_text("", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    (tmp_path / "folder" / "inside.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        handlers_module, "_is_reparse_point", lambda path: Path(path).name == "junction"
    )

    response = await _listing(tmp_path, recursive=True)

    names = {entry.name for entry in response.payload.entries}
    assert {"junction", "folder", "inside.txt"} <= names
    assert "outside.txt" not in names


_posix_permissions = pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions, not root"
)


@_posix_permissions
@pytest.mark.asyncio
async def test_a_folder_that_cannot_be_read_is_an_error_not_an_empty_listing(
    tmp_path: Path,
) -> None:
    """A folder the shim may enter but not read (no read permission, or a
    macOS privacy prompt nobody answered) came back as an empty listing that
    wasn't truncated, so the folder looked empty."""
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden-by-the-error.txt").write_text("", encoding="utf-8")
    locked.chmod(0o300)
    handler = FileHandler(ShimConfig(allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log"))
    try:
        response = await handler.handle_list(_list(locked))
    finally:
        locked.chmod(0o700)

    assert isinstance(response, ErrorMessage)
    assert "locked" in response.payload.message


@_posix_permissions
@pytest.mark.asyncio
async def test_a_subfolder_that_cannot_be_read_marks_the_listing_truncated(
    tmp_path: Path,
) -> None:
    (tmp_path / "open").mkdir()
    (tmp_path / "open" / "a.txt").write_text("", encoding="utf-8")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "b.txt").write_text("", encoding="utf-8")
    locked.chmod(0o300)
    try:
        response = await _listing(tmp_path, recursive=True)
    finally:
        locked.chmod(0o700)

    assert {entry.name for entry in response.payload.entries} == {"open", "a.txt", "locked"}
    assert response.payload.truncated is True


_symlinks = pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")


@_symlinks
@pytest.mark.asyncio
async def test_a_listing_says_nothing_about_where_a_link_outside_points(tmp_path: Path) -> None:
    """is_file() and is_dir() follow a symlink, so a link to a path outside
    the shared folder told the platform whether that path exists, and
    whether it's a file or a folder."""
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("", encoding="utf-8")
    (tmp_path / "private").mkdir()
    (root / "to-file").symlink_to(tmp_path / "secret.txt")
    (root / "to-folder").symlink_to(tmp_path / "private")
    (root / "inside").mkdir()
    (root / "to-inside").symlink_to(root / "inside")

    response = await _listing(root)

    kinds = {e.name: (e.is_file, e.is_dir, e.is_symlink) for e in response.payload.entries}
    assert kinds["to-file"] == (False, False, True)
    assert kinds["to-folder"] == (False, False, True)
    assert kinds["to-inside"] == (False, True, True)  # inside the jail: followed
    assert kinds["inside"] == (False, True, False)


@_symlinks
@_posix_permissions
@pytest.mark.asyncio
async def test_a_link_into_a_folder_that_cannot_be_searched_is_listed(tmp_path: Path) -> None:
    """is_file() raised PermissionError for a link into a folder the shim
    can't search (Python 3.13 and earlier), and the whole FILE_LIST failed."""
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "target.txt").write_text("", encoding="utf-8")
    (tmp_path / "link").symlink_to(locked / "target.txt")
    (tmp_path / "plain.txt").write_text("", encoding="utf-8")
    locked.chmod(0)
    try:
        response = await _listing(tmp_path)
    finally:
        locked.chmod(0o700)

    kinds = {e.name: (e.is_file, e.is_dir, e.is_symlink) for e in response.payload.entries}
    assert kinds["link"] == (False, False, True)
    assert kinds["plain.txt"] == (True, False, False)


@pytest.mark.skipif(sys.platform != "linux", reason="a name that isn't UTF-8 needs Linux")
@pytest.mark.asyncio
async def test_a_name_that_is_not_utf8_is_left_out_of_the_listing(tmp_path: Path) -> None:
    """A POSIX name whose bytes aren't UTF-8 decodes to lone surrogates,
    which JSON can't carry, so model_dump_json raised and the whole
    FILE_LIST failed. The platform couldn't name that file anyway."""
    with open(os.path.join(os.fsencode(tmp_path), b"bad-\xff.txt"), "w", encoding="utf-8"):
        pass
    (tmp_path / "good.txt").write_text("", encoding="utf-8")

    response = await _listing(tmp_path)

    assert [entry.name for entry in response.payload.entries] == ["good.txt"]
    dump_message(response)  # the reply can be sent


@pytest.mark.skipif(os.name == "nt", reason="Windows names can't hold \" or \\")
@pytest.mark.asyncio
async def test_the_byte_budget_counts_names_as_json_escapes_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quotes and backslashes take two bytes each in JSON; a budget that
    counted raw bytes would let such names past it."""
    for index in range(50):
        (tmp_path / f"{chr(34) * 20}{chr(92) * 20}{index:02d}").write_text("", encoding="utf-8")
    monkeypatch.setattr(handlers_module, "FILE_LIST_REPLY_BUDGET_BYTES", 4000)

    response = await _listing(tmp_path)

    assert 0 < len(response.payload.entries) < 50
    entry_bytes = sum(len(e.model_dump_json().encode()) + 1 for e in response.payload.entries)
    assert entry_bytes <= 4000
