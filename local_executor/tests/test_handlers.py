"""Tests for FileHandler and CommandHandler."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import os
import signal
import stat
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import autogpt_local_executor.handlers as handlers_module
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import CommandHandler, FileHandler
from autogpt_local_executor.protocol import (
    MAX_WEBSOCKET_MESSAGE_BYTES,
    MAX_WIRE_TEXT_CONTENT_BYTES,
    AckMessage,
    CommandResultMessage,
    Encoding,
    ErrorCode,
    ErrorMessage,
    ExecuteCommandMessage,
    ExecuteCommandPayload,
    FileContentsMessage,
    FileDeleteMessage,
    FileDeletePayload,
    FileFormat,
    FileListMessage,
    FileListPayload,
    FileListResponseMessage,
    FileMoveMessage,
    FileMovePayload,
    FileReadMessage,
    FileReadPayload,
    FileStatMessage,
    FileStatPayload,
    FileStatResponseMessage,
    FileWriteMessage,
    FileWritePayload,
    Shell,
    dump_message,
    new_id,
    now_ts,
)


def make_config(tmp_path: Path) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        enable_shell=True,
    )


def _read_msg(path: Path, **kw) -> FileReadMessage:
    return FileReadMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileReadPayload(path=str(path), **kw),
    )


def _write_msg(path: Path, content: str, **kw) -> FileWriteMessage:
    return FileWriteMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileWritePayload(path=str(path), content=content, **kw),
    )


# ── FILE_READ / FILE_WRITE ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_then_read_text(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    resp = await h.handle_write(_write_msg(tmp_path / "a.txt", "hello"))
    assert isinstance(resp, AckMessage)
    resp = await h.handle_read(_read_msg(tmp_path / "a.txt"))
    assert isinstance(resp, FileContentsMessage)
    assert resp.payload.content == "hello"
    assert resp.payload.encoding == Encoding.UTF8


@pytest.mark.asyncio
async def test_read_small_range_from_file_larger_than_limit(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"0123456789")
    config = make_config(tmp_path)
    config.max_file_size_bytes = 4
    handler = FileHandler(config)

    response = await handler.handle_read(
        _read_msg(path, offset=3, length=4, format=FileFormat.BYTES)
    )

    assert isinstance(response, FileContentsMessage)
    assert base64.b64decode(response.payload.content) == b"3456"
    assert response.payload.truncated is True


@pytest.mark.asyncio
async def test_read_range_larger_than_limit_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"0123456789")
    config = make_config(tmp_path)
    config.max_file_size_bytes = 4

    response = await FileHandler(config).handle_read(
        _read_msg(path, offset=0, length=5, format=FileFormat.BYTES)
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.FILE_TOO_LARGE


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["read", "write"])
async def test_file_content_rejects_existing_hard_link(tmp_path: Path, operation: str) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret", encoding="utf-8")
    target = root / "linked.txt"
    os.link(outside, target)
    handler = FileHandler(make_config(root))

    response = (
        await handler.handle_read(_read_msg(target))
        if operation == "read"
        else await handler.handle_write(_write_msg(target, "overwrite"))
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert outside.read_text(encoding="utf-8") == "outside secret"
    assert target.read_text(encoding="utf-8") == "outside secret"


@pytest.mark.asyncio
async def test_file_content_checks_opened_descriptor_before_access(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "single.txt"
    target.write_text("must remain intact", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    original_open = os.open

    def link_before_open(*args, **kwargs):
        if Path(args[0]) == target:
            os.link(target, outside)
        return original_open(*args, **kwargs)

    handler = FileHandler(make_config(root))
    with patch("autogpt_local_executor.path_jail.os.open", side_effect=link_before_open):
        response = await handler.handle_read(_read_msg(target))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert target.read_text(encoding="utf-8") == "must remain intact"
    assert outside.read_text(encoding="utf-8") == "must remain intact"


@pytest.mark.asyncio
async def test_a_write_never_reaches_a_hard_link_made_after_the_check(tmp_path: Path) -> None:
    """A write puts a new file at the jail path rather than writing into the
    old one, so a hard link made from outside after the check keeps the old
    contents: nothing outside the jail is written."""
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "single.txt"
    target.write_text("must remain intact", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    original_replace = os.replace

    def link_before_replace(src, dst):
        if Path(dst) == target:
            os.link(target, outside)
        return original_replace(src, dst)

    handler = FileHandler(make_config(root))
    with patch("autogpt_local_executor.path_jail.os.replace", side_effect=link_before_replace):
        response = await handler.handle_write(_write_msg(target, "overwrite"))

    assert isinstance(response, AckMessage)
    assert outside.read_text(encoding="utf-8") == "must remain intact"
    assert target.read_text(encoding="utf-8") == "overwrite"


@pytest.mark.asyncio
async def test_an_in_place_write_checks_the_opened_descriptor_before_truncating(
    tmp_path: Path,
) -> None:
    """Where a new file can't take the old one's place (a folder that takes
    no new entries, say), a write overwrites the old file in place, opened
    without truncating: a hard link made from outside between the check and
    the open is refused on the opened descriptor, before a byte changes."""
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "single.txt"
    target.write_text("must remain intact", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    original_open = os.open

    def link_before_open(*args, **kwargs):
        if Path(args[0]) == target:
            os.link(target, outside)
        return original_open(*args, **kwargs)

    handler = FileHandler(make_config(root))
    with (
        patch(
            "autogpt_local_executor.path_jail._replace_with",
            side_effect=PermissionError(errno.EACCES, "no new entries here"),
        ),
        patch("autogpt_local_executor.path_jail.os.open", side_effect=link_before_open),
    ):
        response = await handler.handle_write(_write_msg(target, "overwrite"))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert target.read_text(encoding="utf-8") == "must remain intact"
    assert outside.read_text(encoding="utf-8") == "must remain intact"


@pytest.mark.asyncio
async def test_write_existing_single_link_truncates_after_validation(tmp_path: Path) -> None:
    target = tmp_path / "single.txt"
    target.write_text("a long original value", encoding="utf-8")

    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, AckMessage)
    assert target.read_text(encoding="utf-8") == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["read", "write"])
async def test_file_content_allows_symlink_to_single_link_inside_root(
    tmp_path: Path, operation: str
) -> None:
    target = tmp_path / "single.txt"
    target.write_text("original", encoding="utf-8")
    link = tmp_path / "symlink.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this platform/account")
    handler = FileHandler(make_config(tmp_path))

    if operation == "read":
        response = await handler.handle_read(_read_msg(link))
        assert isinstance(response, FileContentsMessage)
        assert response.payload.content == "original"
    else:
        response = await handler.handle_write(_write_msg(link, "new"))
        assert isinstance(response, AckMessage)
        assert target.read_text(encoding="utf-8") == "new"

    assert link.is_symlink()


@pytest.mark.asyncio
async def test_write_creates_parents(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    target = tmp_path / "a" / "b" / "c.txt"
    resp = await h.handle_write(_write_msg(target, "hi", create_parents=True))
    assert isinstance(resp, AckMessage)
    assert target.read_text(encoding="utf-8") == "hi"


@pytest.mark.asyncio
async def test_write_no_parents_returns_error(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    target = tmp_path / "missing-dir" / "c.txt"
    resp = await h.handle_write(_write_msg(target, "hi", create_parents=False))

    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.PATH_NOT_FOUND
    assert not target.parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("create_parents", [False, True])
async def test_write_rejects_dangling_symlink_outside_jail(
    tmp_path: Path, create_parents: bool
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    target = root / "link.txt"
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this platform/account")

    response = await FileHandler(make_config(root)).handle_write(
        _write_msg(target, "blocked", create_parents=create_parents)
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert not outside.exists()
    assert target.is_symlink()


@pytest.mark.asyncio
async def test_write_rejects_traversal_through_missing_parents(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "missing" / ".." / ".." / "outside" / "file.txt"

    response = await FileHandler(make_config(root)).handle_write(
        _write_msg(target, "blocked", create_parents=True)
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert not (tmp_path / "outside").exists()
    assert not (root / "missing").exists()


@pytest.mark.asyncio
async def test_read_base64_path(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    raw = bytes(range(256))
    (tmp_path / "bin.dat").write_bytes(raw)
    resp = await h.handle_read(
        _read_msg(tmp_path / "bin.dat", encoding=Encoding.BASE64, format=FileFormat.BYTES)
    )
    assert isinstance(resp, FileContentsMessage)
    assert resp.payload.encoding == Encoding.BASE64
    assert base64.b64decode(resp.payload.content) == raw


@pytest.mark.asyncio
async def test_read_crlf_preserved(tmp_path: Path) -> None:
    """Per PROTOCOL.md FILE_READ: shim does NOT auto-translate \\r\\n → \\n."""
    h = FileHandler(make_config(tmp_path))
    (tmp_path / "win.txt").write_bytes(b"line1\r\nline2\r\n")
    resp = await h.handle_read(_read_msg(tmp_path / "win.txt"))
    assert isinstance(resp, FileContentsMessage)
    assert "\r\n" in resp.payload.content


@pytest.mark.asyncio
async def test_read_offset_and_length(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    (tmp_path / "f.txt").write_text("abcdefghij", encoding="utf-8")
    resp = await h.handle_read(_read_msg(tmp_path / "f.txt", offset=2, length=3))
    assert isinstance(resp, FileContentsMessage)
    assert resp.payload.content == "cde"
    assert resp.payload.truncated is True


@pytest.mark.asyncio
async def test_read_missing_returns_path_not_found(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    resp = await h.handle_read(_read_msg(tmp_path / "nope.txt"))
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.PATH_NOT_FOUND


@pytest.mark.asyncio
async def test_read_outside_jail(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    resp = await h.handle_read(_read_msg(Path("/etc/passwd")))
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT


@pytest.mark.asyncio
async def test_write_too_large(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.max_file_size_bytes = 10
    h = FileHandler(config)
    resp = await h.handle_write(_write_msg(tmp_path / "big.txt", "x" * 20))
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.FILE_TOO_LARGE


# ── FILE_STAT ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stat_file(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    target = tmp_path / "x.txt"
    target.write_text("hi", encoding="utf-8")
    msg = FileStatMessage(id=new_id(), ts=now_ts(), payload=FileStatPayload(path=str(target)))
    resp = await h.handle_stat(msg)
    assert isinstance(resp, FileStatResponseMessage)
    assert resp.payload.exists is True
    assert resp.payload.is_file is True
    assert resp.payload.is_dir is False
    assert resp.payload.size_bytes == 2


@pytest.mark.asyncio
async def test_stat_missing_returns_exists_false(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    msg = FileStatMessage(
        id=new_id(), ts=now_ts(), payload=FileStatPayload(path=str(tmp_path / "nope"))
    )
    resp = await h.handle_stat(msg)
    assert isinstance(resp, FileStatResponseMessage)
    assert resp.payload.exists is False


@pytest.mark.asyncio
async def test_stat_outside_jail_is_distinguishable_from_missing(tmp_path: Path) -> None:
    """Per PROTOCOL.md: jail violation returns ERROR, not exists:false."""
    h = FileHandler(make_config(tmp_path))
    msg = FileStatMessage(id=new_id(), ts=now_ts(), payload=FileStatPayload(path="/etc/passwd"))
    resp = await h.handle_stat(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT


def _stat_msg(path: Path, **kw) -> FileStatMessage:
    return FileStatMessage(id=new_id(), ts=now_ts(), payload=FileStatPayload(path=str(path), **kw))


@pytest.mark.asyncio
async def test_stat_without_following_describes_the_symlink_itself(tmp_path: Path) -> None:
    target = tmp_path / "data.csv"
    target.write_text("a,b\n", encoding="utf-8")
    link = tmp_path / "latest.csv"
    _symlink_or_skip(link, target)
    handler = FileHandler(make_config(tmp_path))

    unfollowed = await handler.handle_stat(_stat_msg(link, follow_symlinks=False))
    followed = await handler.handle_stat(_stat_msg(link, follow_symlinks=True))

    assert isinstance(unfollowed, FileStatResponseMessage)
    assert unfollowed.payload.is_symlink is True
    assert unfollowed.payload.is_file is False
    assert unfollowed.payload.path == os.path.join(os.path.realpath(tmp_path), "latest.csv")
    assert isinstance(followed, FileStatResponseMessage)
    assert followed.payload.is_file is True
    assert followed.payload.path == os.path.realpath(target)


@pytest.mark.asyncio
async def test_stat_of_a_relative_path_describes_the_roots_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path named a place in the shim's working directory, and a
    followed one was checked for a symlink there too, not in the root."""
    target = tmp_path / "data.csv"
    target.write_text("a,b\n", encoding="utf-8")
    _symlink_or_skip(tmp_path / "latest.csv", target)
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")
    handler = FileHandler(make_config(tmp_path))

    followed = await handler.handle_stat(
        FileStatMessage(
            id=new_id(),
            ts=now_ts(),
            payload=FileStatPayload(path="latest.csv", follow_symlinks=True),
        )
    )

    assert isinstance(followed, FileStatResponseMessage)
    assert followed.payload.exists is True
    assert followed.payload.is_symlink is True
    assert followed.payload.path == os.path.realpath(target)


@pytest.mark.asyncio
async def test_stat_without_following_still_describes_the_root(tmp_path: Path) -> None:
    response = await FileHandler(make_config(tmp_path)).handle_stat(
        _stat_msg(tmp_path, follow_symlinks=False)
    )

    assert isinstance(response, FileStatResponseMessage)
    assert response.payload.exists is True
    assert response.payload.is_dir is True


# ── FILE_LIST ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_directory(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.csv").write_text("b", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    msg = FileListMessage(id=new_id(), ts=now_ts(), payload=FileListPayload(path=str(tmp_path)))
    resp = await h.handle_list(msg)
    assert isinstance(resp, FileListResponseMessage)
    names = {e.name for e in resp.payload.entries}
    assert {"a.txt", "b.csv", "sub"} <= names


@pytest.mark.asyncio
async def test_list_glob_csv(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.csv").write_text("b", encoding="utf-8")
    msg = FileListMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileListPayload(path=str(tmp_path), glob="*.csv"),
    )
    resp = await h.handle_list(msg)
    assert isinstance(resp, FileListResponseMessage)
    names = {e.name for e in resp.payload.entries}
    assert names == {"b.csv"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("folds", "listed"), [(True, {"data.csv"}), (False, set())])
async def test_a_glob_matches_by_the_folders_case_rule_where_the_shim_cant_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folds: bool, listed: set[str]
) -> None:
    """Whether a glob's case counts was told by a file made in the listed
    folder for a moment. Where none could be made (a read-only mount, say),
    the OS's default answered: on Linux `*.CSV` missed `data.csv` on a
    filesystem that folds case, and on macOS and Windows it matched
    `data.csv` on one that keeps case."""
    (tmp_path / "data.csv").write_text("a", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("b", encoding="utf-8")
    # As named and as resolved (macOS's /var is /private/var).
    folders = {os.path.normcase(str(tmp_path)), os.path.normcase(os.path.realpath(tmp_path))}
    real_stat = os.stat

    def stat_in_a_folder_with_that_rule(path, *args, **kwargs):
        parent, name = os.path.split(os.fspath(path))
        if os.path.normcase(parent) not in folders or name in os.listdir(parent):
            return real_stat(path, *args, **kwargs)
        same = [entry for entry in os.listdir(parent) if entry.lower() == name.lower()]
        if folds and same:
            return real_stat(os.path.join(parent, same[0]), *args, **kwargs)
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", os.fspath(path))

    def read_only(*_args, **_kwargs):
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(os, "stat", stat_in_a_folder_with_that_rule)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", read_only)
    response = await FileHandler(make_config(tmp_path)).handle_list(
        FileListMessage(
            id=new_id(), ts=now_ts(), payload=FileListPayload(path=str(tmp_path), glob="*.CSV")
        )
    )

    assert isinstance(response, FileListResponseMessage)
    assert {entry.name for entry in response.payload.entries} == listed


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [[], ["1", "22"]], ids=["empty", "no-letters"])
async def test_a_listing_never_writes_to_the_folder_to_tell_its_case_rule(
    tmp_path: Path, names: list[str]
) -> None:
    """Where none of the folder's entries could be looked up again with its
    case swapped (an empty folder, or names without a letter), a probe file
    was made in the folder for a moment, which sync clients and file
    watchers saw, though PROTOCOL.md calls FILE_LIST a pure read. The OS's
    default decides there now, and nothing is written."""
    folder = tmp_path / "shared"
    folder.mkdir()
    for name in names:
        (folder / name).write_text("x", encoding="utf-8")
    # Whole seconds long past, which any filesystem's timestamps hold.
    long_ago = 1_000_000_000 * 1_000_000_000
    os.utime(folder, ns=(long_ago, long_ago))

    response = await FileHandler(make_config(tmp_path)).handle_list(
        FileListMessage(
            id=new_id(), ts=now_ts(), payload=FileListPayload(path=str(folder), glob="*")
        )
    )

    assert isinstance(response, FileListResponseMessage)
    assert sorted(entry.name for entry in response.payload.entries) == names
    assert sorted(os.listdir(folder)) == names
    assert os.stat(folder).st_mtime_ns == long_ago


@pytest.mark.asyncio
async def test_a_folder_with_no_entry_to_look_up_is_looked_at_again_once_it_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OS's default stands in for a folder with no entry to look up only
    while it has none: kept, an empty folder's default would decide its
    globs after it gained one, on a filesystem whose rule is the other."""
    folds = sys.platform not in ("win32", "darwin")
    folder = tmp_path / "shared"
    folder.mkdir()
    folders = {os.path.normcase(str(folder)), os.path.normcase(os.path.realpath(folder))}
    real_stat = os.stat

    def stat_in_a_folder_with_the_other_rule(path, *args, **kwargs):
        parent, name = os.path.split(os.fspath(path))
        if os.path.normcase(parent) not in folders or name in os.listdir(parent):
            return real_stat(path, *args, **kwargs)
        same = [entry for entry in os.listdir(parent) if entry.lower() == name.lower()]
        if folds and same:
            return real_stat(os.path.join(parent, same[0]), *args, **kwargs)
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", os.fspath(path))

    monkeypatch.setattr(os, "stat", stat_in_a_folder_with_the_other_rule)
    handler = FileHandler(make_config(tmp_path))

    def listing(glob: str) -> FileListMessage:
        return FileListMessage(
            id=new_id(), ts=now_ts(), payload=FileListPayload(path=str(folder), glob=glob)
        )

    empty = await handler.handle_list(listing("*.CSV"))
    (folder / "data.csv").write_text("a", encoding="utf-8")
    response = await handler.handle_list(listing("*.CSV"))

    assert isinstance(empty, FileListResponseMessage)
    assert empty.payload.entries == []
    assert isinstance(response, FileListResponseMessage)
    assert {entry.name for entry in response.payload.entries} == ({"data.csv"} if folds else set())


@pytest.mark.asyncio
async def test_list_max_entries_truncates(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    for i in range(5):
        (tmp_path / f"f{i}").write_text("x", encoding="utf-8")
    msg = FileListMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileListPayload(path=str(tmp_path), max_entries=3),
    )
    resp = await h.handle_list(msg)
    assert isinstance(resp, FileListResponseMessage)
    assert len(resp.payload.entries) == 3
    assert resp.payload.truncated is True


@pytest.mark.asyncio
async def test_list_hidden_files_excluded_by_default(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    (tmp_path / ".hidden").write_text("x", encoding="utf-8")
    (tmp_path / "visible").write_text("x", encoding="utf-8")
    msg = FileListMessage(id=new_id(), ts=now_ts(), payload=FileListPayload(path=str(tmp_path)))
    resp = await h.handle_list(msg)
    assert isinstance(resp, FileListResponseMessage)
    names = {e.name for e in resp.payload.entries}
    assert ".hidden" not in names
    assert "visible" in names


# ── FILE_DELETE ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_file(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    target = tmp_path / "del.txt"
    target.write_text("x", encoding="utf-8")
    msg = FileDeleteMessage(id=new_id(), ts=now_ts(), payload=FileDeletePayload(path=str(target)))
    resp = await h.handle_delete(msg)
    assert isinstance(resp, AckMessage)
    assert not target.exists()


@pytest.mark.asyncio
async def test_delete_missing_without_missing_ok(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    msg = FileDeleteMessage(
        id=new_id(), ts=now_ts(), payload=FileDeletePayload(path=str(tmp_path / "nope"))
    )
    resp = await h.handle_delete(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.PATH_NOT_FOUND


@pytest.mark.asyncio
async def test_delete_missing_with_missing_ok(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    msg = FileDeleteMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileDeletePayload(path=str(tmp_path / "nope"), missing_ok=True),
    )
    resp = await h.handle_delete(msg)
    assert isinstance(resp, AckMessage)


@pytest.mark.asyncio
async def test_delete_nonempty_dir_without_recursive(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    d = tmp_path / "d"
    d.mkdir()
    (d / "x").write_text("x", encoding="utf-8")
    msg = FileDeleteMessage(id=new_id(), ts=now_ts(), payload=FileDeletePayload(path=str(d)))
    resp = await h.handle_delete(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.PATH_NOT_EMPTY


@pytest.mark.asyncio
async def test_delete_recursive(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "x").write_text("x", encoding="utf-8")
    msg = FileDeleteMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileDeletePayload(path=str(d), recursive=True),
    )
    resp = await h.handle_delete(msg)
    assert isinstance(resp, AckMessage)
    assert not d.exists()


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this platform/account")


def _delete_msg(path: Path, **kw) -> FileDeleteMessage:
    return FileDeleteMessage(
        id=new_id(), ts=now_ts(), payload=FileDeletePayload(path=str(path), **kw)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("recursive", [False, True])
async def test_delete_removes_a_symlink_not_its_target(tmp_path: Path, recursive: bool) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "report.txt").write_text("keep", encoding="utf-8")
    shortcut = tmp_path / "shortcut"
    _symlink_or_skip(shortcut, documents)

    response = await FileHandler(make_config(tmp_path)).handle_delete(
        _delete_msg(shortcut, recursive=recursive)
    )

    assert isinstance(response, AckMessage)
    assert not os.path.lexists(shortcut)
    assert (documents / "report.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
@pytest.mark.parametrize("recursive", [False, True])
async def test_delete_refuses_a_symlink_pointing_outside_the_root_and_leaves_its_target(
    tmp_path: Path, recursive: bool
) -> None:
    """The path must still resolve inside the root (CROSS_PLATFORM.md), so
    the link stays, and a recursive delete never reaches what it points at."""
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "documents"
    outside.mkdir()
    (outside / "report.txt").write_text("keep", encoding="utf-8")
    shortcut = root / "shortcut"
    _symlink_or_skip(shortcut, outside)

    response = await FileHandler(make_config(root)).handle_delete(
        _delete_msg(shortcut, recursive=recursive)
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert shortcut.is_symlink()
    assert (outside / "report.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
async def test_delete_removes_a_dangling_symlink(tmp_path: Path) -> None:
    link = tmp_path / "stale"
    _symlink_or_skip(link, tmp_path / "gone.txt")

    response = await FileHandler(make_config(tmp_path)).handle_delete(_delete_msg(link))

    assert isinstance(response, AckMessage)
    assert not os.path.lexists(link)


@pytest.mark.asyncio
async def test_delete_refuses_the_allowed_root_itself(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text("keep", encoding="utf-8")

    response = await FileHandler(make_config(root)).handle_delete(_delete_msg(root, recursive=True))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert (root / "notes.txt").read_text(encoding="utf-8") == "keep"


# ── FILE_MOVE ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_move_rename(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    src = tmp_path / "a.txt"
    src.write_text("x", encoding="utf-8")
    dst = tmp_path / "b.txt"
    msg = FileMoveMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileMovePayload(src=str(src), dst=str(dst)),
    )
    resp = await h.handle_move(msg)
    assert isinstance(resp, AckMessage)
    assert not src.exists()
    assert dst.read_text(encoding="utf-8") == "x"


@pytest.mark.asyncio
async def test_move_blocks_dst_outside_jail(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    src = tmp_path / "a.txt"
    src.write_text("x", encoding="utf-8")
    msg = FileMoveMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileMovePayload(src=str(src), dst="/etc/evil.txt"),
    )
    resp = await h.handle_move(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT


@pytest.mark.asyncio
async def test_move_rejects_dangling_destination_outside_jail(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "source.txt"
    source.write_text("original", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    target = root / "link.txt"
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this platform/account")

    response = await FileHandler(make_config(root)).handle_move(
        FileMoveMessage(
            id=new_id(),
            ts=now_ts(),
            payload=FileMovePayload(src=str(source), dst=str(target)),
        )
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert source.read_text(encoding="utf-8") == "original"
    assert not outside.exists()
    assert target.is_symlink()


@pytest.mark.asyncio
async def test_move_exists_without_overwrite(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    src = tmp_path / "a.txt"
    src.write_text("x", encoding="utf-8")
    dst = tmp_path / "b.txt"
    dst.write_text("y", encoding="utf-8")
    msg = FileMoveMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileMovePayload(src=str(src), dst=str(dst), overwrite=False),
    )
    resp = await h.handle_move(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.PATH_EXISTS


@pytest.mark.asyncio
async def test_move_overwrite(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    src = tmp_path / "a.txt"
    src.write_text("new", encoding="utf-8")
    dst = tmp_path / "b.txt"
    dst.write_text("old", encoding="utf-8")
    msg = FileMoveMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileMovePayload(src=str(src), dst=str(dst), overwrite=True),
    )
    resp = await h.handle_move(msg)
    assert isinstance(resp, AckMessage)
    assert dst.read_text(encoding="utf-8") == "new"


def _move_msg(src: Path, dst: Path, **kw) -> FileMoveMessage:
    return FileMoveMessage(
        id=new_id(), ts=now_ts(), payload=FileMovePayload(src=str(src), dst=str(dst), **kw)
    )


@pytest.mark.asyncio
async def test_move_renames_a_symlink_not_its_target(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "report.txt").write_text("keep", encoding="utf-8")
    shortcut = tmp_path / "shortcut"
    _symlink_or_skip(shortcut, documents)
    renamed = tmp_path / "renamed"

    response = await FileHandler(make_config(tmp_path)).handle_move(_move_msg(shortcut, renamed))

    assert isinstance(response, AckMessage)
    assert renamed.is_symlink()
    assert not os.path.lexists(shortcut)
    assert (documents / "report.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
async def test_move_overwrite_replaces_a_symlink_destination_not_its_target(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real.txt"
    real.write_text("keep", encoding="utf-8")
    link = tmp_path / "link.txt"
    _symlink_or_skip(link, real)
    source = tmp_path / "new.txt"
    source.write_text("new", encoding="utf-8")

    handler = FileHandler(make_config(tmp_path))
    refused = await handler.handle_move(_move_msg(source, link))
    replaced = await handler.handle_move(_move_msg(source, link, overwrite=True))

    assert isinstance(refused, ErrorMessage)
    assert refused.payload.code == ErrorCode.PATH_EXISTS
    assert isinstance(replaced, AckMessage)
    assert not link.is_symlink()
    assert link.read_text(encoding="utf-8") == "new"
    assert real.read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
async def test_move_cross_device_does_not_copy_or_remove_destination(
    tmp_path: Path, directory: bool
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    if directory:
        source.mkdir()
        destination.mkdir()
        (source / "source.txt").write_text("source", encoding="utf-8")
        (destination / "destination.txt").write_text("destination", encoding="utf-8")
    else:
        source.write_text("source", encoding="utf-8")
        destination.write_text("destination", encoding="utf-8")
    message = FileMoveMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileMovePayload(src=str(source), dst=str(destination), overwrite=True),
    )
    with (
        patch("os.rename", side_effect=OSError(errno.EXDEV, "cross-device move")),
        patch("os.replace", side_effect=OSError(errno.EXDEV, "cross-device move")),
    ):
        response = await FileHandler(make_config(tmp_path)).handle_move(message)

    assert isinstance(response, ErrorMessage)
    if directory:
        assert (source / "source.txt").read_text(encoding="utf-8") == "source"
        assert (destination / "destination.txt").read_text(encoding="utf-8") == "destination"
    else:
        assert source.read_text(encoding="utf-8") == "source"
        assert destination.read_text(encoding="utf-8") == "destination"


@pytest.mark.asyncio
async def test_move_overwrite_hard_link_does_not_change_outside_content(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret", encoding="utf-8")
    destination = root / "linked.txt"
    os.link(outside, destination)
    source = root / "source.txt"
    source.write_text("workspace content", encoding="utf-8")

    response = await FileHandler(make_config(root)).handle_move(
        FileMoveMessage(
            id=new_id(),
            ts=now_ts(),
            payload=FileMovePayload(src=str(source), dst=str(destination), overwrite=True),
        )
    )

    assert isinstance(response, AckMessage)
    assert outside.read_text(encoding="utf-8") == "outside secret"
    assert destination.read_text(encoding="utf-8") == "workspace content"
    assert not source.exists()


@pytest.mark.asyncio
async def test_move_hard_link_does_not_grant_content_access(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret", encoding="utf-8")
    source = root / "linked.txt"
    os.link(outside, source)
    destination = root / "moved.txt"
    handler = FileHandler(make_config(root))

    response = await handler.handle_move(
        FileMoveMessage(
            id=new_id(),
            ts=now_ts(),
            payload=FileMovePayload(src=str(source), dst=str(destination)),
        )
    )
    read_response = await handler.handle_read(_read_msg(destination))

    assert isinstance(response, AckMessage)
    assert isinstance(read_response, ErrorMessage)
    assert read_response.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert outside.read_text(encoding="utf-8") == "outside secret"
    assert not source.exists()


@pytest.mark.asyncio
async def test_move_cannot_overwrite_nonempty_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source", encoding="utf-8")
    (destination / "destination.txt").write_text("destination", encoding="utf-8")

    response = await FileHandler(make_config(tmp_path)).handle_move(
        FileMoveMessage(
            id=new_id(),
            ts=now_ts(),
            payload=FileMovePayload(src=str(source), dst=str(destination), overwrite=True),
        )
    )

    assert isinstance(response, ErrorMessage)
    assert (source / "source.txt").read_text(encoding="utf-8") == "source"
    assert (destination / "destination.txt").read_text(encoding="utf-8") == "destination"


# ── EXECUTE_COMMAND (mocked) ─────────────────────────────────────────────────


def _cmd_msg(argv=None, command=None, **kw) -> ExecuteCommandMessage:
    return ExecuteCommandMessage(
        id=new_id(),
        ts=now_ts(),
        payload=ExecuteCommandPayload(argv=argv, command=command, **kw),
    )


@pytest.mark.asyncio
async def test_command_execution_is_disabled_by_default(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.enable_shell = False
    response = await CommandHandler(config).handle(_cmd_msg(argv=["echo", "blocked"]))
    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.CAPABILITY_NOT_GRANTED


class _FakeProc:
    def __init__(self, *, stdout: bytes, stderr: bytes, returncode: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.pid = 12345

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_command_argv_form_calls_create_subprocess_exec(tmp_path: Path) -> None:
    """Per PROTOCOL.md, argv form must bypass shell parsing."""
    h = CommandHandler(make_config(tmp_path))
    fake = _FakeProc(stdout=b"out\n", stderr=b"", returncode=0)

    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        msg = _cmd_msg(argv=["echo", "hi"], cwd=str(tmp_path))
        resp = await h.handle(msg)

    assert isinstance(resp, CommandResultMessage)
    assert resp.payload.exit_code == 0
    assert resp.payload.stdout == "out\n"
    assert resp.payload.timed_out is False
    # First two args should be the binary and its first arg (we didn't pass through a shell).
    assert captured["args"][0] == "echo"
    assert captured["args"][1] == "hi"


@pytest.mark.asyncio
@pytest.mark.parametrize("plat", ["linux", "darwin", "windows"])
async def test_a_command_gets_its_own_session_without_a_preexec_fn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plat: str
) -> None:
    """A command got its own session from preexec_fn=os.setsid, Python code
    run in the forked child before exec. Python documents that as unsafe in
    a process with threads, which the shim has (a request's path checks run
    in worker threads just before the spawn): the child can deadlock before
    exec, where the command's timeout, which starts once the spawn returns,
    never reaches it. start_new_session makes the same setsid() call in C."""
    monkeypatch.setattr(handlers_module.platform_info, "detect_platform", lambda: plat)
    h = CommandHandler(make_config(tmp_path))
    captured: dict[str, Any] = {}

    async def fake_exec(*args, **kwargs):
        captured.update(kwargs)
        return _FakeProc(stdout=b"", stderr=b"", returncode=0)

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        resp = await h.handle(_cmd_msg(argv=["echo", "hi"], cwd=str(tmp_path)))

    assert isinstance(resp, CommandResultMessage)
    assert captured.get("preexec_fn") is None
    assert captured.get("start_new_session", False) is (plat != "windows")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sessions and process groups")
@pytest.mark.asyncio
async def test_a_command_leads_its_own_session_and_process_group(tmp_path: Path) -> None:
    """A timeout or a revoked session signals the command's process group by
    the command's PID (_terminate_tree), so the command leads a new session
    and the group in it."""
    h = CommandHandler(make_config(tmp_path))
    probe = "import os; print(os.getsid(0) == os.getpgid(0) == os.getpid())"

    resp = await h.handle(
        _cmd_msg(argv=[sys.executable, "-c", probe], cwd=str(tmp_path), timeout_seconds=30)
    )

    assert isinstance(resp, CommandResultMessage)
    assert resp.payload.stdout.strip() == "True"


@pytest.mark.asyncio
async def test_command_output_is_drained_and_truncated_to_shared_limit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.max_file_size_bytes = 1024
    handler = CommandHandler(config)
    command = "import os; os.write(1, b'o' * 100000); os.write(2, b'e' * 100000)"

    response = await handler.handle(
        _cmd_msg(
            argv=[sys.executable, "-c", command],
            cwd=str(tmp_path),
            timeout_seconds=10,
        )
    )

    assert isinstance(response, CommandResultMessage)
    combined = (response.payload.stdout + response.payload.stderr).encode("utf-8")
    assert len(combined) <= config.max_file_size_bytes
    assert "[command output truncated]" in response.payload.stderr
    assert response.payload.output_truncated is True
    assert response.payload.exit_code == 0


@pytest.mark.asyncio
async def test_command_output_is_safe_for_worst_case_json_escaping(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.max_file_size_bytes = MAX_WIRE_TEXT_CONTENT_BYTES + 1024
    handler = CommandHandler(config)
    fake = _FakeProc(
        stdout=b"\0" * (MAX_WIRE_TEXT_CONTENT_BYTES + 1),
        stderr=b"",
        returncode=0,
    )

    with patch("asyncio.create_subprocess_exec", return_value=fake):
        response = await handler.handle(_cmd_msg(argv=["emit-nuls"], cwd=str(tmp_path)))

    assert isinstance(response, CommandResultMessage)
    assert response.payload.output_truncated is True
    assert len(dump_message(response).encode("utf-8")) <= MAX_WEBSOCKET_MESSAGE_BYTES


@pytest.mark.asyncio
async def test_requested_command_timeout_is_clamped_to_local_ceiling(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.command_timeout_seconds = 2
    handler = CommandHandler(config)
    fake = _FakeProc(stdout=b"", stderr=b"", returncode=0)
    captured_timeout: float | None = None

    async def collect(_proc, *, timeout: float):
        nonlocal captured_timeout
        captured_timeout = timeout
        return b"", b"", False, False

    with patch("asyncio.create_subprocess_exec", return_value=fake):
        with patch.object(handler, "_collect_process_output", side_effect=collect):
            response = await handler.handle(
                _cmd_msg(argv=["slow"], cwd=str(tmp_path), timeout_seconds=300)
            )

    assert isinstance(response, CommandResultMessage)
    assert captured_timeout == 2


@pytest.mark.asyncio
async def test_command_rate_limit_rejects_excess_request(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.max_commands_per_minute = 1
    handler = CommandHandler(config)
    fake = _FakeProc(stdout=b"", stderr=b"", returncode=0)

    with patch("asyncio.create_subprocess_exec", return_value=fake):
        first = await handler.handle(_cmd_msg(argv=["one"], cwd=str(tmp_path)))
        second = await handler.handle(_cmd_msg(argv=["two"], cwd=str(tmp_path)))

    assert isinstance(first, CommandResultMessage)
    assert isinstance(second, ErrorMessage)
    assert second.payload.code == ErrorCode.SHIM_OVERLOADED


@pytest.mark.asyncio
async def test_concurrent_command_limit_rejects_without_queueing(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.max_concurrent_commands = 1
    handler = CommandHandler(config)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def execute(_msg):
        entered.set()
        await release.wait()
        return CommandResultMessage(
            id="first",
            ts=now_ts(),
            payload={
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "timed_out": False,
                "duration_seconds": 0,
            },
        )

    with patch.object(handler, "_execute", side_effect=execute):
        first_task = asyncio.create_task(handler.handle(_cmd_msg(argv=["one"])))
        await entered.wait()
        second = await handler.handle(_cmd_msg(argv=["two"]))
        release.set()
        await first_task

    assert isinstance(second, ErrorMessage)
    assert second.payload.code == ErrorCode.SHIM_OVERLOADED


@pytest.mark.asyncio
async def test_cancelling_command_handler_terminates_spawned_process(tmp_path: Path) -> None:
    handler = CommandHandler(make_config(tmp_path))
    fake = _FakeProc(stdout=b"", stderr=b"", returncode=0)
    entered = asyncio.Event()
    terminated = asyncio.Event()

    async def collect(_proc, *, timeout: float):
        entered.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def terminate(_proc):
        terminated.set()

    with patch("asyncio.create_subprocess_exec", return_value=fake):
        with patch.object(handler, "_collect_process_output", side_effect=collect):
            with patch.object(handler, "_terminate_tree", side_effect=terminate):
                task = asyncio.create_task(
                    handler.handle(_cmd_msg(argv=["long-running"], cwd=str(tmp_path)))
                )
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    assert terminated.is_set()


@pytest.mark.asyncio
async def test_a_cancelled_command_is_still_audited(tmp_path: Path) -> None:
    """A command stopped because its session ended or was revoked ran, but
    the cancel went straight past the audit write, so it left no
    EXECUTE_COMMAND record."""
    records: list[tuple[str, dict]] = []

    class Audit:
        async def write(self, op, *, request_id=None, details, result=None):
            records.append((op, result))

    handler = CommandHandler(make_config(tmp_path), Audit())
    fake = _FakeProc(stdout=b"", stderr=b"", returncode=None)
    entered = asyncio.Event()

    async def collect(_proc, *, timeout: float):
        entered.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def terminate(_proc):
        pass

    with patch("asyncio.create_subprocess_exec", return_value=fake):
        with patch.object(handler, "_collect_process_output", side_effect=collect):
            with patch.object(handler, "_terminate_tree", side_effect=terminate):
                task = asyncio.create_task(
                    handler.handle(_cmd_msg(argv=["long-running"], cwd=str(tmp_path)))
                )
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    [(op, result)] = [record for record in records if record[0] == "EXECUTE_COMMAND"]
    assert result["ok"] is False and result["error_code"] == "CANCELLED"


@pytest.mark.asyncio
async def test_a_command_cancelled_again_while_it_stops_is_still_stopped_and_audited(
    tmp_path: Path,
) -> None:
    """A second cancel while a cancelled command was being stopped (a detach
    and a revocation both stopping its chat) ended the stop where it waited,
    in the grace period before SIGKILL, and the CANCELLED record was never
    written."""
    records: list[tuple[str, dict]] = []

    class Audit:
        async def write(self, op, *, request_id=None, details, result=None):
            records.append((op, result))

    handler = CommandHandler(make_config(tmp_path), Audit())
    fake = _FakeProc(stdout=b"", stderr=b"", returncode=None)
    running = asyncio.Event()
    in_grace = asyncio.Event()
    grace_over = asyncio.Event()
    killed = False

    async def collect(_proc, *, timeout: float):
        running.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def terminate(_proc):
        nonlocal killed
        in_grace.set()
        await grace_over.wait()
        killed = True

    with patch("asyncio.create_subprocess_exec", return_value=fake):
        with patch.object(handler, "_collect_process_output", side_effect=collect):
            with patch.object(handler, "_terminate_tree", side_effect=terminate):
                task = asyncio.create_task(
                    handler.handle(_cmd_msg(argv=["ignores-sigterm"], cwd=str(tmp_path)))
                )
                await running.wait()
                task.cancel()
                await in_grace.wait()
                task.cancel()
                grace_over.set()
                with pytest.raises(asyncio.CancelledError):
                    await task

    assert killed
    [(op, result)] = [record for record in records if record[0] == "EXECUTE_COMMAND"]
    assert result["ok"] is False and result["error_code"] == "CANCELLED"


@pytest.mark.asyncio
async def test_command_shell_form_picks_shell(tmp_path: Path) -> None:
    h = CommandHandler(make_config(tmp_path))
    fake = _FakeProc(stdout=b"out\n", stderr=b"", returncode=0)
    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        return fake

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        with patch(
            "autogpt_local_executor.handlers.platform_info.resolve_shell",
            return_value=("/bin/bash", ["-c"]),
        ):
            msg = _cmd_msg(command="echo hi", cwd=str(tmp_path), shell=Shell.BASH)
            resp = await h.handle(msg)

    assert isinstance(resp, CommandResultMessage)
    # First three args: shell, -c, command string.
    assert captured["args"][0] == "/bin/bash"
    assert captured["args"][1] == "-c"
    assert captured["args"][2] == "echo hi"


@pytest.mark.asyncio
async def test_command_shell_not_available(tmp_path: Path) -> None:
    h = CommandHandler(make_config(tmp_path))
    with patch(
        "autogpt_local_executor.handlers.platform_info.resolve_shell",
        return_value=None,
    ):
        msg = _cmd_msg(command="echo hi", cwd=str(tmp_path), shell=Shell.ZSH)
        resp = await h.handle(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.SHELL_NOT_AVAILABLE


@pytest.mark.asyncio
async def test_command_neither_argv_nor_command(tmp_path: Path) -> None:
    h = CommandHandler(make_config(tmp_path))
    msg = _cmd_msg(argv=None, command=None, cwd=str(tmp_path))
    resp = await h.handle(msg)
    assert isinstance(resp, ErrorMessage)


@pytest.mark.asyncio
async def test_a_command_with_no_cwd_runs_in_a_relative_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command that named no cwd gave the jail the root itself as its
    path, and the jail takes a relative path from the root, so with a
    relative allowed_root (a legacy session's, as typed) the command ran in
    workspace/workspace, or failed where there was none."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()
    config = make_config(tmp_path)
    config.allowed_root = Path("workspace")
    captured: dict[str, object] = {}

    async def fake_exec(*args, **kwargs):
        captured.update(kwargs)
        return _FakeProc(stdout=b"", stderr=b"", returncode=0)

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        response = await CommandHandler(config).handle(_cmd_msg(argv=["echo", "hi"]))

    assert isinstance(response, CommandResultMessage)
    assert Path(str(captured["cwd"])).resolve() == (tmp_path / "workspace").resolve()


@pytest.mark.asyncio
async def test_command_cwd_outside_jail(tmp_path: Path) -> None:
    h = CommandHandler(make_config(tmp_path))
    msg = _cmd_msg(argv=["echo", "hi"], cwd="/etc")
    resp = await h.handle(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT


@pytest.mark.asyncio
async def test_command_timeout(tmp_path: Path) -> None:
    h = CommandHandler(make_config(tmp_path))

    class _HangingProc:
        def __init__(self) -> None:
            self.returncode = None
            self.pid = 12345
            self._calls = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            self._calls += 1
            if self._calls == 1:
                await asyncio.sleep(5)
            return b"", b""

        def send_signal(self, sig) -> None:
            self.returncode = -1

        def kill(self) -> None:
            self.returncode = -1

    hanging = _HangingProc()

    async def fake_exec(*args, **kwargs):
        return hanging

    async def fake_terminate(proc) -> None:
        proc.returncode = -1

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        with patch.object(CommandHandler, "_terminate_tree", staticmethod(fake_terminate)):
            msg = _cmd_msg(argv=["sleep", "10"], cwd=str(tmp_path), timeout_seconds=1)
            resp = await h.handle(msg)

    assert isinstance(resp, CommandResultMessage)
    assert resp.payload.timed_out is True


async def _wait_until(condition, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.05)
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.asyncio
async def test_terminating_a_command_kills_descendants_that_ignore_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command exits on SIGTERM; its child ignores SIGTERM and must still die."""
    import fcntl

    monkeypatch.setattr(handlers_module, "_TERMINATE_GRACE_SECONDS", 0.5)
    lock_path = tmp_path / "child.lock"
    pid_path = tmp_path / "child.pid"
    child = textwrap.dedent(
        f"""
        import fcntl, os, signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        lock = open({str(lock_path)!r}, "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open({str(pid_path)!r} + ".tmp", "w") as f:
            f.write(str(os.getpid()))
        os.replace({str(pid_path)!r} + ".tmp", {str(pid_path)!r})
        time.sleep(120)
        """
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(120)"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", parent, start_new_session=True
    )

    def child_released_its_lock() -> bool:
        with open(lock_path, "w", encoding="utf-8") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True

    try:
        assert await _wait_until(pid_path.exists, timeout=60)
        await CommandHandler._terminate_tree(proc)
        await asyncio.wait_for(proc.wait(), timeout=30)

        assert await _wait_until(child_released_its_lock, timeout=30)
    finally:
        if pid_path.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


def _watch_for_sigterm(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Set once a process group is sent SIGTERM: its grace period has begun."""
    sent = asyncio.Event()
    signal_group = handlers_module._signal_process_group

    def watched(pgid: int, sig: int) -> bool:
        if sig == signal.SIGTERM:
            sent.set()
        return signal_group(pgid, sig)

    monkeypatch.setattr(handlers_module, "_signal_process_group", watched)
    return sent


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and process groups")
@pytest.mark.asyncio
async def test_a_command_ignoring_sigterm_cancelled_twice_while_it_stops_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled command is sent SIGTERM, then SIGKILL once its grace period
    is out. A second cancel in that period ended its stop before SIGKILL, so
    a command that ignores SIGTERM ran on after its session ended, and its
    CANCELLED record was never written."""
    records: list[tuple[str, dict]] = []

    class Audit:
        async def write(self, op, *, request_id=None, details, result=None):
            records.append((op, result))

    config = make_config(tmp_path)
    config.command_timeout_seconds = 600
    handler = CommandHandler(config, Audit())
    spawned: list[asyncio.subprocess.Process] = []
    spawn = CommandHandler._spawn_argv

    async def spawn_and_keep(argv, **kwargs):
        proc = await spawn(argv, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(handler, "_spawn_argv", spawn_and_keep)
    sigterm_sent = _watch_for_sigterm(monkeypatch)
    ignoring = tmp_path / "ignoring-sigterm"
    script = textwrap.dedent(
        f"""
        import signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        open({str(ignoring)!r}, "w").close()
        time.sleep(120)
        """
    )
    task = asyncio.create_task(
        handler.handle(_cmd_msg(argv=[sys.executable, "-c", script], cwd=str(tmp_path)))
    )
    try:
        assert await _wait_until(ignoring.exists, timeout=60)
        task.cancel()
        await asyncio.wait_for(sigterm_sent.wait(), timeout=60)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        [(op, result)] = [record for record in records if record[0] == "EXECUTE_COMMAND"]
        assert result["error_code"] == "CANCELLED"
        assert result["exit_code"] == -signal.SIGKILL
        [proc] = spawned
        await asyncio.wait_for(proc.wait(), timeout=30)
        assert proc.returncode == -signal.SIGKILL
    finally:
        if not task.done():
            task.cancel()
        for proc in spawned:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.asyncio
async def test_a_command_cancelled_after_its_shell_exited_has_its_group_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command's shell can exit while a child it started keeps the output
    pipes open, which the shim reads until they close. A cancel landing then
    skipped the group's stop, since the group's leader had exited, and the
    child ran on after its session ended."""
    import fcntl

    records: list[tuple[str, dict]] = []

    class Audit:
        async def write(self, op, *, request_id=None, details, result=None):
            records.append((op, result))

    monkeypatch.setattr(handlers_module, "_OUTPUT_DRAIN_SECONDS", 120.0)
    config = make_config(tmp_path)
    config.command_timeout_seconds = 600
    handler = CommandHandler(config, Audit())
    spawned: list[asyncio.subprocess.Process] = []
    spawn = CommandHandler._spawn_argv

    async def spawn_and_keep(argv, **kwargs):
        proc = await spawn(argv, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(handler, "_spawn_argv", spawn_and_keep)
    lock_path = tmp_path / "child.lock"
    pid_path = tmp_path / "child.pid"
    child = textwrap.dedent(
        f"""
        import fcntl, os, time
        lock = open({str(lock_path)!r}, "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open({str(pid_path)!r} + ".tmp", "w") as f:
            f.write(str(os.getpid()))
        os.replace({str(pid_path)!r} + ".tmp", {str(pid_path)!r})
        time.sleep(120)
        """
    )
    # The shell: it starts the child, which inherits its stdout and stderr,
    # and exits once the child holds its lock.
    shell = textwrap.dedent(
        f"""
        import os, subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", {child!r}])
        while not os.path.exists({str(pid_path)!r}):
            time.sleep(0.01)
        """
    )

    def shell_exited() -> bool:
        return bool(spawned) and spawned[0].returncode is not None

    def child_released_its_lock() -> bool:
        with open(lock_path, "w", encoding="utf-8") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True

    task = asyncio.create_task(
        handler.handle(_cmd_msg(argv=[sys.executable, "-c", shell], cwd=str(tmp_path)))
    )
    try:
        assert await _wait_until(shell_exited, timeout=60)
        # Past the wait for the shell's exit, into the wait for the pipes.
        await asyncio.sleep(0.5)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert await _wait_until(child_released_its_lock, timeout=30)
        [(op, result)] = [record for record in records if record[0] == "EXECUTE_COMMAND"]
        assert result["error_code"] == "CANCELLED"
        assert result["exit_code"] == 0
    finally:
        if not task.done():
            task.cancel()
        if pid_path.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_path.read_text(encoding="utf-8")), signal.SIGKILL)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.asyncio
async def test_a_cancel_landing_as_the_command_exits_still_stops_its_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python 3.11's asyncio.wait_for drops a cancel that lands as the work
    it waits for ends. The command's exit was waited for that way, so a
    session revoked just as its command exited ended the request as if
    nothing had happened: the command was recorded as done, and a child it
    left holding its output ran on."""
    import fcntl

    records: list[tuple[str, dict]] = []

    class Audit:
        async def write(self, op, *, request_id=None, details, result=None):
            records.append((op, result))

    exited = handlers_module._exited
    requests: list[asyncio.Task] = []

    async def exited_as_the_request_is_cancelled(proc):
        await exited(proc)
        requests[0].cancel()

    monkeypatch.setattr(handlers_module, "_exited", exited_as_the_request_is_cancelled)
    config = make_config(tmp_path)
    config.command_timeout_seconds = 600
    handler = CommandHandler(config, Audit())
    lock_path = tmp_path / "child.lock"
    pid_path = tmp_path / "child.pid"
    child = textwrap.dedent(
        f"""
        import fcntl, os, time
        lock = open({str(lock_path)!r}, "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open({str(pid_path)!r} + ".tmp", "w") as f:
            f.write(str(os.getpid()))
        os.replace({str(pid_path)!r} + ".tmp", {str(pid_path)!r})
        time.sleep(120)
        """
    )
    # The command starts the child, which keeps its stdout and stderr, and
    # exits once the child holds its lock.
    command = textwrap.dedent(
        f"""
        import os, subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", {child!r}])
        while not os.path.exists({str(pid_path)!r}):
            time.sleep(0.01)
        """
    )

    def child_released_its_lock() -> bool:
        with open(lock_path, "w", encoding="utf-8") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True

    task = asyncio.create_task(
        handler.handle(_cmd_msg(argv=[sys.executable, "-c", command], cwd=str(tmp_path)))
    )
    requests.append(task)
    try:
        done, _ = await asyncio.wait({task}, timeout=60)
        assert task in done
        assert task.cancelled()
        assert await _wait_until(child_released_its_lock, timeout=30)
        [(op, result)] = [record for record in records if record[0] == "EXECUTE_COMMAND"]
        assert result["error_code"] == "CANCELLED"
    finally:
        if not task.done():
            task.cancel()
        if pid_path.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_path.read_text(encoding="utf-8")), signal.SIGKILL)


def _hold_pipe_connections(monkeypatch: pytest.MonkeyPatch) -> tuple[asyncio.Event, list[Any]]:
    """Have asyncio wait to connect a new command's pipes until the event
    returned is set, as a busy event loop might: its command was created
    before then. The list gets each command's Popen as its wait begins."""
    from asyncio import base_subprocess

    may_connect = asyncio.Event()
    created: list[Any] = []
    connect = base_subprocess.BaseSubprocessTransport._connect_pipes

    async def connect_when_allowed(self, waiter):
        created.append(self._proc)
        await may_connect.wait()
        await connect(self, waiter)

    monkeypatch.setattr(
        base_subprocess.BaseSubprocessTransport, "_connect_pipes", connect_when_allowed
    )
    return may_connect, created


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.asyncio
async def test_a_command_cancelled_as_it_starts_is_stopped_with_its_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A POSIX command runs from the moment it is created, and asyncio
    connects its output pipes after that. A cancel landing in between went
    to asyncio's own cleanup, which kills the command alone, so a child it
    had started ran on after its session ended, with no CANCELLED record:
    the request never had the process to stop."""
    import fcntl

    records: list[tuple[str, dict]] = []

    class Audit:
        async def write(self, op, *, request_id=None, details, result=None):
            records.append((op, result))

    may_connect, _created = _hold_pipe_connections(monkeypatch)
    config = make_config(tmp_path)
    config.command_timeout_seconds = 600
    handler = CommandHandler(config, Audit())
    lock_path = tmp_path / "child.lock"
    pid_path = tmp_path / "child.pid"
    child = textwrap.dedent(
        f"""
        import fcntl, os, time
        lock = open({str(lock_path)!r}, "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open({str(pid_path)!r} + ".tmp", "w") as f:
            f.write(str(os.getpid()))
        os.replace({str(pid_path)!r} + ".tmp", {str(pid_path)!r})
        time.sleep(120)
        """
    )
    command = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(120)"
    )

    def child_released_its_lock() -> bool:
        with open(lock_path, "w", encoding="utf-8") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True

    task = asyncio.create_task(
        handler.handle(_cmd_msg(argv=[sys.executable, "-c", command], cwd=str(tmp_path)))
    )
    try:
        # The child runs while the command's start still waits for its pipes.
        assert await _wait_until(pid_path.exists, timeout=60)
        assert not task.done()
        task.cancel()
        may_connect.set()

        assert await _wait_until(child_released_its_lock, timeout=30)
        with pytest.raises(asyncio.CancelledError):
            await task
        [(op, result)] = [record for record in records if record[0] == "EXECUTE_COMMAND"]
        assert result["error_code"] == "CANCELLED"
    finally:
        may_connect.set()
        if pid_path.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
        if not task.done():
            task.cancel()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows starts a command suspended")
@pytest.mark.asyncio
async def test_a_windows_command_cancelled_as_it_starts_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows a command is created suspended and runs only once it is in
    its job, after asyncio has connected its pipes. A cancel landing before
    then reaches asyncio's own cleanup, which ends the command while it is
    still suspended: nothing of it ran, so there is nothing to stop or
    record (on POSIX there is: the test above)."""
    records: list[tuple[str, dict]] = []

    class Audit:
        async def write(self, op, *, request_id=None, details, result=None):
            records.append((op, result))

    may_connect, created = _hold_pipe_connections(monkeypatch)
    handler = CommandHandler(make_config(tmp_path), Audit())
    ran = tmp_path / "ran"
    task = asyncio.create_task(
        handler.handle(
            _cmd_msg(
                argv=[sys.executable, "-c", f"open({str(ran)!r}, 'w').close()"],
                cwd=str(tmp_path),
            )
        )
    )
    try:
        assert await _wait_until(lambda: bool(created), timeout=60)
        task.cancel()
        may_connect.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        [popen] = created
        assert popen.poll() is not None
        assert not ran.exists()
        assert not handlers_module._command_jobs
        assert [record for record in records if record[0] == "EXECUTE_COMMAND"] == []
    finally:
        may_connect.set()
        if not task.done():
            task.cancel()


def _released(held: Path) -> bool:
    """Whether no process holds `held` open any more. Windows refuses to
    delete a file another process has open without sharing deletion, which
    Python's open() doesn't share."""
    try:
        held.unlink()
    except FileNotFoundError:
        return True
    except PermissionError:
        return False
    return True


def _holding_child(held: Path, ready: Path) -> str:
    """A child command that ignores CTRL_BREAK_EVENT, holds `held` open, and
    makes `ready` once it does."""
    return textwrap.dedent(
        f"""
        import os, signal, time
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        held = open({str(held)!r}, "w")
        open({str(ready)!r} + ".tmp", "w").close()
        os.replace({str(ready)!r} + ".tmp", {str(ready)!r})
        time.sleep(120)
        """
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
@pytest.mark.asyncio
async def test_a_stopped_windows_command_takes_the_processes_it_started_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows a command's stop sent CTRL_BREAK_EVENT, then
    TerminateProcess, which ends the command alone: a child it started that
    outlived the break ran on after its session ended. The command runs in a
    job of its own now, and the stop ends the job."""
    monkeypatch.setattr(handlers_module, "_WINDOWS_BREAK_GRACE_SECONDS", 0.5)
    config = make_config(tmp_path)
    config.command_timeout_seconds = 600
    handler = CommandHandler(config)
    held, ready = tmp_path / "held", tmp_path / "ready"
    command = (
        "import signal, subprocess, sys, time; "
        "signal.signal(signal.SIGBREAK, signal.SIG_IGN); "
        f"subprocess.Popen([sys.executable, '-c', {_holding_child(held, ready)!r}]); "
        "time.sleep(120)"
    )
    task = asyncio.create_task(
        handler.handle(_cmd_msg(argv=[sys.executable, "-c", command], cwd=str(tmp_path)))
    )
    try:
        assert await _wait_until(ready.exists, timeout=60)
        assert not _released(held)
        # The command got its job: without one, the stop ends it alone.
        assert len(handlers_module._command_jobs) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert await _wait_until(lambda: _released(held), timeout=30)
    finally:
        if not task.done():
            task.cancel()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
@pytest.mark.asyncio
async def test_a_windows_command_cancelled_after_it_exited_takes_its_child_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command can exit while a child it started holds the output pipes
    the shim reads. On Windows its stop then did nothing, since the
    command's own handle can't reach a child, and the child ran on."""
    records: list[tuple[str, dict]] = []

    class Audit:
        async def write(self, op, *, request_id=None, details, result=None):
            records.append((op, result))

    monkeypatch.setattr(handlers_module, "_OUTPUT_DRAIN_SECONDS", 120.0)
    config = make_config(tmp_path)
    config.command_timeout_seconds = 600
    handler = CommandHandler(config, Audit())
    spawned: list[asyncio.subprocess.Process] = []
    spawn = CommandHandler._spawn_argv

    async def spawn_and_keep(argv, **kwargs):
        proc = await spawn(argv, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(handler, "_spawn_argv", spawn_and_keep)
    held, ready = tmp_path / "held", tmp_path / "ready"
    # The command hands the child its stdout and stderr, and exits once the
    # child holds its file.
    command = textwrap.dedent(
        f"""
        import os, subprocess, sys, time
        subprocess.Popen(
            [sys.executable, "-c", {_holding_child(held, ready)!r}], stdout=1, stderr=2
        )
        while not os.path.exists({str(ready)!r}):
            time.sleep(0.01)
        """
    )
    task = asyncio.create_task(
        handler.handle(_cmd_msg(argv=[sys.executable, "-c", command], cwd=str(tmp_path)))
    )
    try:
        assert await _wait_until(
            lambda: bool(spawned) and spawned[0].returncode is not None, timeout=60
        )
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert await _wait_until(lambda: _released(held), timeout=30)
        [(op, result)] = [record for record in records if record[0] == "EXECUTE_COMMAND"]
        assert result["error_code"] == "CANCELLED"
        assert result["exit_code"] == 0
    finally:
        if not task.done():
            task.cancel()


def test_a_command_whose_job_cannot_be_made_is_never_resumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command whose job couldn't be made was resumed anyway, and its stop
    could then end it alone: what it started ran on after a timeout or a
    revoked session. It now stays suspended, for its request to refuse."""
    from autogpt_local_executor import windows_jobs

    resumed: list[int] = []

    def no_job(_kernel32: object, _pid: int) -> windows_jobs.CommandJob:
        raise OSError(errno.EACCES, "jobs are restricted here")

    monkeypatch.setattr(windows_jobs, "_kernel32", lambda: object())
    monkeypatch.setattr(windows_jobs, "_job_holding", no_job)
    monkeypatch.setattr(windows_jobs, "_resume", lambda _kernel32, pid: resumed.append(pid))

    with pytest.raises(OSError, match="not run"):
        windows_jobs.run_in_job(4321)

    assert resumed == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
@pytest.mark.asyncio
async def test_a_windows_command_that_cannot_get_a_job_is_refused_before_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When its job couldn't be made, a command ran without one, and a stop
    ended it alone. It is refused now, while still suspended, so nothing of
    it runs, and the refusal is recorded."""
    from autogpt_local_executor import windows_jobs

    records: list[tuple[str, dict]] = []

    class Audit:
        async def write(self, op, *, request_id=None, details, result=None):
            records.append((op, result))

    def no_job(_kernel32: object, _pid: int) -> windows_jobs.CommandJob:
        raise OSError(errno.EACCES, "jobs are restricted here")

    monkeypatch.setattr(windows_jobs, "_job_holding", no_job)
    created: list[asyncio.subprocess.Process] = []
    create = asyncio.create_subprocess_exec

    async def create_and_keep(*args, **kwargs):
        proc = await create(*args, **kwargs)
        created.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_and_keep)
    handler = CommandHandler(make_config(tmp_path), Audit())
    ran = tmp_path / "ran"

    response = await handler.handle(
        _cmd_msg(
            argv=[sys.executable, "-c", f"open({str(ran)!r}, 'w').close()"],
            cwd=str(tmp_path),
        )
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.INTERNAL_ERROR
    assert "job" in response.payload.message
    [proc] = created
    await asyncio.wait_for(proc.wait(), timeout=30)
    # Ended while it was still suspended, so it can't run now.
    assert not ran.exists()
    assert not handlers_module._command_jobs
    [(op, result)] = [record for record in records if record[0] == "EXECUTE_COMMAND"]
    assert result["error_code"] == "INTERNAL_ERROR"


@pytest.mark.asyncio
async def test_a_program_a_finished_command_started_in_the_background_keeps_running(
    tmp_path: Path,
) -> None:
    """A command that finishes leaves running what it started to outlive it
    (a server, say), on every OS: on Windows its job is let go of, not ended,
    once its request is done."""
    config = make_config(tmp_path)
    handler = CommandHandler(config)
    ready, go, survived = tmp_path / "ready", tmp_path / "go", tmp_path / "survived"
    child = textwrap.dedent(
        f"""
        import os, time
        open({str(ready)!r} + ".tmp", "w").close()
        os.replace({str(ready)!r} + ".tmp", {str(ready)!r})
        deadline = time.monotonic() + 60
        while not os.path.exists({str(go)!r}) and time.monotonic() < deadline:
            time.sleep(0.01)
        open({str(survived)!r}, "w").close()
        """
    )
    command = textwrap.dedent(
        f"""
        import os, subprocess, sys, time
        subprocess.Popen(
            [sys.executable, "-c", {child!r}],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        while not os.path.exists({str(ready)!r}):
            time.sleep(0.01)
        """
    )

    response = await handler.handle(
        _cmd_msg(argv=[sys.executable, "-c", command], cwd=str(tmp_path), timeout_seconds=30)
    )

    assert isinstance(response, CommandResultMessage)
    assert response.payload.exit_code == 0
    go.touch()
    assert await _wait_until(survived.exists, timeout=60)


@pytest.mark.asyncio
async def test_a_command_that_exits_leaving_a_child_on_its_output_is_done_when_it_exits(
    tmp_path: Path,
) -> None:
    """Process.wait() returns once the output pipes have closed as well as
    the process exited, so a command that exited after starting a child
    that kept its stdout and stderr (a server started in the background)
    was waited for until its timeout, reported as timed out, and stopped
    with the child. It is done once it exits now: its output is read for
    _OUTPUT_DRAIN_SECONDS more, and the child keeps running, as one with
    output of its own does."""
    config = make_config(tmp_path)
    config.command_timeout_seconds = 600
    handler = CommandHandler(config)
    ready, go, survived = tmp_path / "ready", tmp_path / "go", tmp_path / "survived"
    child = textwrap.dedent(
        f"""
        import os, time
        open({str(ready)!r} + ".tmp", "w").close()
        os.replace({str(ready)!r} + ".tmp", {str(ready)!r})
        deadline = time.monotonic() + 60
        while not os.path.exists({str(go)!r}) and time.monotonic() < deadline:
            time.sleep(0.01)
        open({str(survived)!r}, "w").close()
        """
    )
    # The command hands the child its stdout and stderr, and exits once the
    # child runs.
    command = textwrap.dedent(
        f"""
        import os, subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", {child!r}], stdout=1, stderr=2)
        while not os.path.exists({str(ready)!r}):
            time.sleep(0.01)
        print("started", flush=True)
        """
    )

    response = await handler.handle(
        _cmd_msg(argv=[sys.executable, "-c", command], cwd=str(tmp_path), timeout_seconds=40)
    )

    assert isinstance(response, CommandResultMessage)
    assert response.payload.timed_out is False
    assert response.payload.exit_code == 0
    assert "started" in response.payload.stdout
    go.touch()
    assert await _wait_until(survived.exists, timeout=60)


def _a_command_leaving_a_writer(folder: Path) -> list[str]:
    """A command that exits once it has started a program holding its
    stdout and stderr (a server started in the background, say). Once `go`
    is made in `folder`, the program writes 4 MiB to each, far past any
    pipe's buffer, then makes `wrote`; it exits once `stop` is made, or a
    minute after its start. Its PID is in `pid`."""
    pid, go, wrote, stop = (str(folder / name) for name in ("pid", "go", "wrote", "stop"))
    program = textwrap.dedent(
        f"""
        import os, time
        deadline = time.monotonic() + 60
        def wait_for(path):
            while not os.path.exists(path) and time.monotonic() < deadline:
                time.sleep(0.01)
        with open({pid!r} + ".tmp", "w") as f:
            f.write(str(os.getpid()))
        os.replace({pid!r} + ".tmp", {pid!r})
        wait_for({go!r})
        chunk = memoryview(b"x" * 65536)
        for _ in range(64):
            for fd in (1, 2):
                left = chunk
                while left:
                    left = left[os.write(fd, left):]
        open({wrote!r} + ".tmp", "w").close()
        os.replace({wrote!r} + ".tmp", {wrote!r})
        wait_for({stop!r})
        """
    )
    command = textwrap.dedent(
        f"""
        import os, subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", {program!r}], stdout=1, stderr=2)
        while not os.path.exists({pid!r}):
            time.sleep(0.01)
        print("started", flush=True)
        """
    )
    return [sys.executable, "-c", command]


def _let_the_writer_go(folder: Path) -> None:
    """Have _a_command_leaving_a_writer's program exit, and end it if it is
    still stuck writing."""
    (folder / "stop").touch()
    if not (folder / "wrote").exists():
        with contextlib.suppress(OSError, ValueError):
            os.kill(int((folder / "pid").read_text(encoding="utf-8")), signal.SIGTERM)


@pytest.mark.asyncio
async def test_a_program_left_on_a_commands_output_never_stops_on_a_full_pipe(
    tmp_path: Path,
) -> None:
    """A command that had exited was done once its output was read for
    _OUTPUT_DRAIN_SECONDS more, and nothing read its pipes after that,
    though a program it left running (a server started in the background)
    still held them. Once they filled, the program stopped on its next
    write, for good, or ended on a broken pipe once they were collected.
    What it writes after the result is read and dropped now, until it
    closes them."""
    config = make_config(tmp_path)
    config.command_timeout_seconds = 600
    handler = CommandHandler(config)
    try:
        response = await handler.handle(
            _cmd_msg(
                argv=_a_command_leaving_a_writer(tmp_path), cwd=str(tmp_path), timeout_seconds=40
            )
        )

        assert isinstance(response, CommandResultMessage)
        assert response.payload.timed_out is False
        assert response.payload.exit_code == 0
        assert response.payload.stdout.strip() == "started"
        (tmp_path / "go").touch()
        assert await _wait_until((tmp_path / "wrote").exists, timeout=60)
        (tmp_path / "stop").touch()
        # Its pipes closed as it exited, and the handler let go of them.
        assert await _wait_until(lambda: not handler._output_drains, timeout=60)
    finally:
        _let_the_writer_go(tmp_path)


@pytest.mark.asyncio
async def test_what_a_chats_command_left_running_is_read_until_the_machines_handler_closes(
    tmp_path: Path,
) -> None:
    """A chat ends long before a server one of its commands started, so a
    chat's handler leaves such a program's output to the machine's, whose
    close, as the shim stops, ends the reads at once: the program may never
    close its output."""
    config = make_config(tmp_path)
    config.command_timeout_seconds = 600
    machine, chat = CommandHandler(config), CommandHandler(config)
    machine.share_with(chat)
    try:
        response = await chat.handle(
            _cmd_msg(
                argv=_a_command_leaving_a_writer(tmp_path), cwd=str(tmp_path), timeout_seconds=40
            )
        )
        assert isinstance(response, CommandResultMessage)
        await chat.close()

        (tmp_path / "go").touch()
        assert await _wait_until((tmp_path / "wrote").exists, timeout=60)
        [(reading, command)] = machine._output_drains.items()
        await asyncio.wait_for(machine.close(), timeout=30)
        assert reading.cancelled()
        assert not machine._output_drains

        # The program exits, and its pipes, read to their end, close before
        # this test's event loop does.
        (tmp_path / "stop").touch()
        await asyncio.wait_for(
            handlers_module._discard_until_closed((command.stdout, command.stderr)), timeout=60
        )
    finally:
        _let_the_writer_go(tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.asyncio
async def test_a_group_stop_cut_short_in_its_grace_period_kills_the_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stop whose own task was cancelled in its grace period (the shim's
    event loop ending) never sent SIGKILL, and left the group running."""
    sigterm_sent = _watch_for_sigterm(monkeypatch)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ignoring', flush=True); time.sleep(120)",
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert proc.stdout is not None
        assert await asyncio.wait_for(proc.stdout.readline(), timeout=60) == b"ignoring\n"
        stopping = asyncio.create_task(handlers_module._terminate_process_group(proc.pid))
        await asyncio.wait_for(sigterm_sent.wait(), timeout=60)
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping

        await asyncio.wait_for(proc.wait(), timeout=30)
        assert proc.returncode == -signal.SIGKILL
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


@pytest.mark.asyncio
async def test_command_audit_log_written(tmp_path: Path) -> None:
    import json

    from autogpt_local_executor.audit import AuditWriter

    config = make_config(tmp_path)
    audit = AuditWriter(path=config.audit_log_path, audit_key=b"\0" * 32)
    h = CommandHandler(config, audit=audit)
    fake = _FakeProc(stdout=b"", stderr=b"", returncode=0)

    async def fake_exec(*args, **kwargs):
        return fake

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        msg = _cmd_msg(argv=["echo", "hi"], cwd=str(tmp_path))
        await h.handle(msg)

    assert config.audit_log_path.exists()
    lines = config.audit_log_path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1])
    assert rec["op"] == "EXECUTE_COMMAND"
    assert rec["details"]["argv"] == ["echo", "hi"]
    assert rec["result"]["ok"] is True
    assert rec["hmac"]


@pytest.mark.asyncio
async def test_command_env_merged_with_baseline(tmp_path: Path) -> None:
    h = CommandHandler(make_config(tmp_path))
    captured: dict = {}

    async def fake_exec(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeProc(stdout=b"", stderr=b"", returncode=0)

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        msg = _cmd_msg(argv=["echo", "x"], cwd=str(tmp_path), env={"MY_VAR": "v"})
        await h.handle(msg)

    env = captured["env"]
    assert env["MY_VAR"] == "v"
    assert "PYTHONIOENCODING" in env
    # PATH (or equivalent) is preserved from the baseline on every OS.
    assert any(k.upper() == "PATH" for k in env)


@pytest.mark.asyncio
async def test_recursive_list_without_hidden_skips_what_is_inside_hidden_dirs(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[remote]", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print()", encoding="utf-8")
    (tmp_path / "src" / ".env").write_text("TOKEN=x", encoding="utf-8")

    response = await FileHandler(make_config(tmp_path)).handle_list(
        FileListMessage(
            id=new_id(),
            ts=now_ts(),
            payload=FileListPayload(path=str(tmp_path), recursive=True, include_hidden=False),
        )
    )

    assert isinstance(response, FileListResponseMessage)
    assert sorted(entry.name for entry in response.payload.entries) == ["app.py", "src"]


# ── Writes that fail, moves that race, listings that ask for too much ───────


class _FullDisk:
    """A file object whose writes fail the way a full disk does: part of the
    data lands, then the disk is full."""

    def __init__(self, file) -> None:
        self._file = file

    def write(self, data: bytes) -> int:
        self._file.write(data[:1])
        self._file.flush()
        raise OSError(errno.ENOSPC, "No space left on device")

    def __getattr__(self, name: str):
        return getattr(self._file, name)

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        self._file.close()
        return False


@pytest.mark.asyncio
async def test_a_write_that_fails_part_way_leaves_the_old_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Truncating first meant a full disk destroyed the file being rewritten."""
    target = tmp_path / "notes.txt"
    target.write_text("the user's notes", encoding="utf-8")
    real_fdopen = os.fdopen
    monkeypatch.setattr(os, "fdopen", lambda fd, *a, **k: _FullDisk(real_fdopen(fd, *a, **k)))

    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))
    monkeypatch.undo()

    assert isinstance(response, ErrorMessage)
    assert target.read_text(encoding="utf-8") == "the user's notes"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"]


def _windows_acl(path: Path) -> list[str]:
    """`path`'s access entries as icacls prints them, one per line."""
    import subprocess

    printed = subprocess.run(
        ["icacls", str(path)], capture_output=True, text=True, check=True
    ).stdout
    return [line.replace(str(path), "").strip() for line in printed.splitlines() if ":(" in line]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DACLs")
@pytest.mark.asyncio
async def test_a_write_keeps_a_windows_files_own_permissions(tmp_path: Path) -> None:
    """A rewrite renamed a new file over the old one, and the new file had
    the folder's inherited DACL: a file the person had made private
    (inheritance off, their account alone) in a folder other accounts can
    read became readable to them. Such a file is written in place now."""
    import subprocess

    target = tmp_path / "private.txt"
    target.write_text("the person's notes", encoding="utf-8")
    account = subprocess.run(["whoami"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(
        ["icacls", str(target), "/inheritance:r", "/grant:r", f"{account}:(F)"],
        capture_output=True,
        check=True,
    )
    private = _windows_acl(target)
    assert private and not any("(I)" in entry for entry in private)

    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, AckMessage)
    assert target.read_text(encoding="utf-8") == "new"
    assert _windows_acl(target) == private
    assert sorted(p.name for p in tmp_path.iterdir()) == ["private.txt"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DACLs")
@pytest.mark.asyncio
async def test_a_write_still_replaces_a_windows_file_with_the_folders_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file whose DACL is the folder's, as most are, is still replaced, so a
    write that fails part way leaves it whole."""
    import autogpt_local_executor.path_jail as path_jail

    target = tmp_path / "notes.txt"
    target.write_text("old", encoding="utf-8")
    inherited = _windows_acl(target)
    replaced: list[Path] = []
    real_replace = path_jail.os.replace

    def replace(src, dst) -> None:
        replaced.append(Path(dst))
        real_replace(src, dst)

    monkeypatch.setattr(path_jail.os, "replace", replace)
    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, AckMessage)
    assert replaced == [target]
    assert target.read_text(encoding="utf-8") == "new"
    assert _windows_acl(target) == inherited


# A Linux POSIX ACL as its extended attribute holds it: a version, then each
# entry's tag, permissions and id, in order of tag and id (acl_ea.h).
_ACL_USER_OBJ, _ACL_USER, _ACL_GROUP_OBJ, _ACL_MASK, _ACL_OTHER = 0x01, 0x02, 0x04, 0x10, 0x20
_ACL_NO_ID = 0xFFFFFFFF
_ANOTHER_UID = 12345


def _linux_acl(*entries: tuple[int, int, int]) -> bytes:
    import struct

    return struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *entry) for entry in entries)


# Another account may read and write; the file's own group may not.
_ANOTHER_ACCOUNT_ONLY = (
    (_ACL_USER_OBJ, 0o6, _ACL_NO_ID),
    (_ACL_USER, 0o6, _ANOTHER_UID),
    (_ACL_GROUP_OBJ, 0o0, _ACL_NO_ID),
    (_ACL_MASK, 0o6, _ACL_NO_ID),
    (_ACL_OTHER, 0o0, _ACL_NO_ID),
)


def _set_linux_acl(path: Path, value: bytes, *, default: bool = False) -> None:
    name = "system.posix_acl_default" if default else "system.posix_acl_access"
    try:
        os.setxattr(path, name, value)
    except OSError as exc:
        if exc.errno in (errno.ENOTSUP, errno.EOPNOTSUPP):
            pytest.skip(f"{path.parent} keeps no POSIX ACLs")
        raise


def _linux_acl_of(path: Path) -> bytes | None:
    try:
        return os.getxattr(path, "system.posix_acl_access")
    except OSError as exc:
        if exc.errno == errno.ENODATA:
            return None
        raise


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux POSIX ACLs")
@pytest.mark.asyncio
async def test_a_write_keeps_a_linux_files_own_acl(tmp_path: Path) -> None:
    """A rewrite renamed a new file over the old one with the old mode, but
    not the old file's ACL. With an ACL the mode's group bits are its mask,
    so a file that let another account in and kept its own group out
    (group::---, mask::rw-) came back as group::rw-, open to the whole
    group, and the other account lost its entry. Such a file is written in
    place now."""
    target = tmp_path / "shared.txt"
    target.write_text("the person's notes", encoding="utf-8")
    target.chmod(0o600)
    _set_linux_acl(target, _linux_acl(*_ANOTHER_ACCOUNT_ONLY))
    acl = _linux_acl_of(target)
    inode = target.stat().st_ino

    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, AckMessage)
    assert target.read_text(encoding="utf-8") == "new"
    assert _linux_acl_of(target) == acl
    assert target.stat().st_ino == inode
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shared.txt"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux POSIX ACLs")
@pytest.mark.asyncio
async def test_a_write_doesnt_give_a_linux_file_its_folders_default_acl(tmp_path: Path) -> None:
    """The new file took the folder's default ACL, which the rename kept: a
    file made before it, which another account couldn't read, became
    readable to the account the default named. It is written in place,
    keeping its own permissions."""
    folder = tmp_path / "shared"
    folder.mkdir()
    target = folder / "notes.txt"
    target.write_text("the person's notes", encoding="utf-8")
    target.chmod(0o660)
    _set_linux_acl(folder, _linux_acl(*_ANOTHER_ACCOUNT_ONLY), default=True)

    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, AckMessage)
    assert target.read_text(encoding="utf-8") == "new"
    assert _linux_acl_of(target) is None
    assert target.stat().st_mode & 0o777 == 0o660
    assert sorted(p.name for p in folder.iterdir()) == ["notes.txt"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux POSIX ACLs")
@pytest.mark.asyncio
async def test_a_write_still_replaces_a_linux_file_with_its_folders_acl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file whose ACL is the one its folder gives a new file, as every file
    made there has, is still replaced, so a write that fails part way leaves
    it whole."""
    import autogpt_local_executor.path_jail as path_jail

    folder = tmp_path / "shared"
    folder.mkdir()
    _set_linux_acl(folder, _linux_acl(*_ANOTHER_ACCOUNT_ONLY), default=True)
    target = folder / "notes.txt"
    target.write_text("old", encoding="utf-8")
    acl = _linux_acl_of(target)
    assert acl is not None
    replaced: list[Path] = []
    real_replace = path_jail.os.replace

    def replace(src, dst) -> None:
        replaced.append(Path(dst))
        real_replace(src, dst)

    monkeypatch.setattr(path_jail.os, "replace", replace)
    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, AckMessage)
    assert replaced == [target]
    assert target.read_text(encoding="utf-8") == "new"
    assert _linux_acl_of(target) == acl


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux extended attributes")
@pytest.mark.asyncio
async def test_a_write_goes_in_place_when_the_files_acl_cant_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ACL the shim can't read may be one a replacement would drop, so the
    file is written in place."""
    target = tmp_path / "notes.txt"
    target.write_text("old", encoding="utf-8")
    inode = target.stat().st_ino
    real_getxattr = os.getxattr

    def getxattr(path, *args, **kwargs):
        if not isinstance(path, int) and Path(path) == target:
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        return real_getxattr(path, *args, **kwargs)

    monkeypatch.setattr(os, "getxattr", getxattr)
    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, AckMessage)
    assert target.read_text(encoding="utf-8") == "new"
    assert target.stat().st_ino == inode


def _macos_acl(path: Path) -> list[str]:
    """`path`'s ACL entries as `ls -le` prints them, one per line."""
    import subprocess

    printed = subprocess.run(
        ["ls", "-led", str(path)], capture_output=True, text=True, check=True
    ).stdout
    return [line.strip() for line in printed.splitlines()[1:] if line.strip()]


def _chmod_acl(path: Path, entry: str) -> None:
    import subprocess

    subprocess.run(["chmod", "+a", entry, str(path)], capture_output=True, check=True)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS ACLs")
@pytest.mark.asyncio
async def test_a_write_keeps_a_macos_files_own_acl(tmp_path: Path) -> None:
    """A rewrite renamed a new file over the old one with the old mode, but
    not its ACL: a file anyone could read but one account (a deny entry),
    came back readable to that account too. Such a file is written in
    place now."""
    target = tmp_path / "notes.txt"
    target.write_text("the person's notes", encoding="utf-8")
    target.chmod(0o644)
    _chmod_acl(target, "user:nobody deny read")
    acl = _macos_acl(target)
    assert acl
    inode = target.stat().st_ino

    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, AckMessage)
    assert target.read_text(encoding="utf-8") == "new"
    assert _macos_acl(target) == acl
    assert target.stat().st_ino == inode
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS ACLs")
@pytest.mark.asyncio
async def test_a_write_doesnt_give_a_macos_file_its_folders_inherited_acl(
    tmp_path: Path,
) -> None:
    """The new file took the folder's inherited entries, which the rename
    kept: a private file made before them became readable to the account
    they name. It is written in place, keeping its own permissions."""
    folder = tmp_path / "shared"
    folder.mkdir()
    target = folder / "notes.txt"
    target.write_text("the person's notes", encoding="utf-8")
    target.chmod(0o600)
    _chmod_acl(folder, "user:nobody allow read,file_inherit")
    assert _macos_acl(target) == []

    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, AckMessage)
    assert target.read_text(encoding="utf-8") == "new"
    assert _macos_acl(target) == []
    assert sorted(p.name for p in folder.iterdir()) == ["notes.txt"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS ACLs")
@pytest.mark.asyncio
async def test_a_write_still_replaces_a_macos_file_with_its_folders_acl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file whose entries are the ones its folder gives a new file is still
    replaced, so a write that fails part way leaves it whole."""
    import autogpt_local_executor.path_jail as path_jail

    folder = tmp_path / "shared"
    folder.mkdir()
    _chmod_acl(folder, "user:nobody allow read,file_inherit")
    target = folder / "notes.txt"
    target.write_text("old", encoding="utf-8")
    acl = _macos_acl(target)
    assert acl
    replaced: list[Path] = []
    real_replace = path_jail.os.replace

    def replace(src, dst) -> None:
        replaced.append(Path(dst))
        real_replace(src, dst)

    monkeypatch.setattr(path_jail.os, "replace", replace)
    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, AckMessage)
    assert replaced == [target]
    assert target.read_text(encoding="utf-8") == "new"
    assert _macos_acl(target) == acl


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes any file")
@pytest.mark.asyncio
async def test_a_write_refuses_a_file_the_person_made_read_only(tmp_path: Path) -> None:
    """A rename ignores the target's mode, so a read-only file was replaced
    whole, though writing it in place would have been refused."""
    target = tmp_path / "keep.txt"
    target.write_text("the user's notes", encoding="utf-8")
    target.chmod(0o444)

    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, ErrorMessage)
    assert target.read_text(encoding="utf-8") == "the user's notes"
    assert target.stat().st_mode & 0o777 == 0o444
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
@pytest.mark.asyncio
async def test_a_write_keeps_the_mode_and_a_new_file_follows_the_umask(tmp_path: Path) -> None:
    existing = tmp_path / "private.txt"
    existing.write_text("old", encoding="utf-8")
    existing.chmod(0o640)
    created = tmp_path / "created.txt"
    umask = os.umask(0o022)
    try:
        handler = FileHandler(make_config(tmp_path))
        await handler.handle_write(_write_msg(existing, "new"))
        await handler.handle_write(_write_msg(created, "new"))
    finally:
        os.umask(umask)

    assert existing.read_text(encoding="utf-8") == "new"
    assert existing.stat().st_mode & 0o777 == 0o640
    assert created.stat().st_mode & 0o777 == 0o644


def _another_group(directory: Path) -> int | None:
    """A group this process is in that a new file in `directory` wouldn't
    get: not the effective group (Linux) nor the directory's (BSD, macOS)."""
    if not hasattr(os, "getgroups"):
        return None
    taken = {os.getegid(), directory.stat().st_gid}
    return next((gid for gid in os.getgroups() if gid not in taken), None)


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership")
@pytest.mark.asyncio
async def test_a_write_keeps_the_files_group(tmp_path: Path) -> None:
    """The replacement was the shim's new file, in the shim's group (or the
    directory's), with the old mode: a file shared through a group lost it,
    and its group bits then applied to another group."""
    group = _another_group(tmp_path)
    if group is None:
        pytest.skip("needs a supplementary group")
    shared = tmp_path / "shared.txt"
    shared.write_text("old", encoding="utf-8")
    os.chown(shared, -1, group)
    shared.chmod(0o640)

    await FileHandler(make_config(tmp_path)).handle_write(_write_msg(shared, "new"))

    assert shared.read_text(encoding="utf-8") == "new"
    assert shared.stat().st_gid == group
    assert shared.stat().st_uid == os.geteuid()
    assert shared.stat().st_mode & 0o777 == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shared.txt"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership")
@pytest.mark.asyncio
async def test_a_write_the_shim_cant_give_the_owner_back_goes_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file another account owns, or in a group the shim isn't in, can't be
    handed back after a replace, so it is written in place, which keeps its
    owner and group."""
    import autogpt_local_executor.path_jail as path_jail

    group = _another_group(tmp_path)
    if group is None:
        pytest.skip("needs a supplementary group")
    shared = tmp_path / "shared.txt"
    shared.write_text("old", encoding="utf-8")
    os.chown(shared, -1, group)
    inode = shared.stat().st_ino

    def refuse(*_args: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(path_jail.os, "fchown", refuse)

    await FileHandler(make_config(tmp_path)).handle_write(_write_msg(shared, "new"))

    assert shared.read_text(encoding="utf-8") == "new"
    assert shared.stat().st_ino == inode
    assert shared.stat().st_gid == group
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shared.txt"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
@pytest.mark.asyncio
async def test_a_replacement_is_owner_only_until_it_takes_the_old_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file that replaces a private one was created under the umask's
    mode (0o644, say) and given the old mode only after: another account
    that opened it meanwhile, watching the directory, could read the new
    contents through that descriptor once they were written."""
    import autogpt_local_executor.path_jail as path_jail

    private = tmp_path / "private.txt"
    private.write_text("old", encoding="utf-8")
    private.chmod(0o600)
    before_mode: list[int] = []
    real_fchmod = os.fchmod

    def fchmod(fd: int, mode: int) -> None:
        before_mode.append(stat.S_IMODE(os.fstat(fd).st_mode))
        real_fchmod(fd, mode)

    monkeypatch.setattr(path_jail.os, "fchmod", fchmod)
    previous = os.umask(0o022)
    try:
        await FileHandler(make_config(tmp_path)).handle_write(_write_msg(private, "new"))
        created = tmp_path / "created.txt"
        await FileHandler(make_config(tmp_path)).handle_write(_write_msg(created, "x"))
    finally:
        os.umask(previous)

    assert private.read_text(encoding="utf-8") == "new"
    assert before_mode and all(mode & 0o077 == 0 for mode in before_mode)
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    # A new file still gets 0o666 less the umask.
    assert stat.S_IMODE(created.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
@pytest.mark.parametrize("privilege", [0o4000, 0o2000], ids=["setuid", "setgid"])
@pytest.mark.parametrize("in_place", [False, True], ids=["replaced", "in-place"])
@pytest.mark.asyncio
async def test_a_rewritten_file_never_keeps_set_id_bits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, privilege: int, in_place: bool
) -> None:
    """The mode went back whole, set-user-ID and set-group-ID included, so a
    shim with CAP_FSETID (root, say) left an agent-written program running
    with its owner's privileges. Linux drops those bits on an unprivileged
    write; the shim now drops them itself, wherever it runs."""
    import autogpt_local_executor.path_jail as path_jail

    program = tmp_path / "tool"
    program.write_text("old", encoding="utf-8")
    program.chmod(0o755 | privilege)
    applied: list[int] = []
    real_fchmod = os.fchmod

    def fchmod(fd: int, mode: int) -> None:
        applied.append(mode)
        real_fchmod(fd, mode)

    monkeypatch.setattr(path_jail.os, "fchmod", fchmod)
    if in_place:

        def refuse(*_args, **_kwargs):
            raise PermissionError("the directory takes no new entries")

        monkeypatch.setattr(path_jail, "_replace_with", refuse)
    handler = FileHandler(make_config(tmp_path))

    await handler.handle_write(_write_msg(program, "new"))

    assert program.read_text(encoding="utf-8") == "new"
    assert all(mode & 0o6000 == 0 for mode in applied)
    assert program.stat().st_mode & 0o7777 == 0o755


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_an_in_place_write_drops_set_id_bits_before_the_contents_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-place fallback truncated and wrote first and dropped the bits
    after, so a write that failed part way left a set-user-ID file holding
    partial contents, and a privileged shim, which the kernel doesn't strip
    the bits for, left them on the new contents until the chmod."""
    import contextlib
    import errno

    import autogpt_local_executor.path_jail as path_jail

    program = tmp_path / "tool"
    program.write_text("old", encoding="utf-8")
    program.chmod(0o4755)
    assert program.stat().st_mode & 0o4000
    modes_at_truncate: list[int] = []

    class FullDisk:
        def __init__(self, file) -> None:
            self._file = file

        def fileno(self) -> int:
            return self._file.fileno()

        def flush(self) -> None:
            self._file.flush()

        def truncate(self, size: int) -> int:
            # Before the truncate itself, which Linux strips the bits on
            # for an unprivileged process.
            modes_at_truncate.append(os.fstat(self._file.fileno()).st_mode)
            return self._file.truncate(size)

        def write(self, _data: bytes) -> int:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    real_open = path_jail.open_jail_file

    @contextlib.contextmanager
    def open_jail_file(*args, **kwargs):
        with real_open(*args, **kwargs) as file:
            yield FullDisk(file)

    def refuse(*_args, **_kwargs):
        raise PermissionError("the directory takes no new entries")

    monkeypatch.setattr(path_jail, "_replace_with", refuse)
    monkeypatch.setattr(path_jail, "open_jail_file", open_jail_file)

    with pytest.raises(OSError):
        path_jail.write_jail_file(program, tmp_path, b"new")

    assert [mode & 0o6000 for mode in modes_at_truncate] == [0]
    assert program.stat().st_mode & 0o7777 == 0o755


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
@pytest.mark.parametrize("privilege", [0o4000, 0o2000], ids=["setuid", "setgid"])
def test_an_in_place_write_that_cannot_drop_set_id_bits_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, privilege: int
) -> None:
    """The in-place fallback dropped the set-ID bits best effort and wrote
    anyway. A shim the kernel doesn't strip them for (CAP_FSETID) but that
    may not chmod the file (someone else's) left the new contents running
    with the owner's privileges."""
    import autogpt_local_executor.path_jail as path_jail

    program = tmp_path / "tool"
    program.write_text("old", encoding="utf-8")
    program.chmod(0o755 | privilege)

    def no_new_entries(*_args, **_kwargs):
        raise PermissionError("the directory takes no new entries")

    def not_the_owner(*_args) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(path_jail, "_replace_with", no_new_entries)
    monkeypatch.setattr(path_jail.os, "fchmod", not_the_owner)

    with pytest.raises(PermissionError):
        path_jail.write_jail_file(program, tmp_path, b"new")
    monkeypatch.undo()

    assert program.read_text(encoding="utf-8") == "old"
    assert program.stat().st_mode & 0o7777 == 0o755 | privilege


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_an_in_place_write_goes_ahead_when_only_the_mode_cannot_be_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogpt_local_executor.path_jail as path_jail

    notes = tmp_path / "notes.txt"
    notes.write_text("old", encoding="utf-8")

    def no_new_entries(*_args, **_kwargs):
        raise PermissionError("the directory takes no new entries")

    def not_the_owner(*_args) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(path_jail, "_replace_with", no_new_entries)
    monkeypatch.setattr(path_jail.os, "fchmod", not_the_owner)

    path_jail.write_jail_file(notes, tmp_path, b"new")
    monkeypatch.undo()

    assert notes.read_text(encoding="utf-8") == "new"


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_an_in_place_write_leaves_a_mode_with_no_set_id_bits_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-place fallback set the file's mode even when it had no set-ID
    bits to drop, so the mode it set was the one the file had. On an NFS
    mount a chmod reaches the server even when it changes nothing, and a
    server that drops ACL entries on a chmod (ZFS's aclmode=discard, say)
    dropped the file's own ACL, which writing in place is there to keep."""
    import autogpt_local_executor.path_jail as path_jail

    notes = tmp_path / "notes.txt"
    notes.write_text("old", encoding="utf-8")
    notes.chmod(0o640)
    applied: list[int] = []
    real_fchmod = os.fchmod

    def fchmod(fd: int, mode: int) -> None:
        applied.append(mode)
        real_fchmod(fd, mode)

    def acl_differs(*_args, **_kwargs):
        raise PermissionError("the file has an ACL a replacement wouldn't keep")

    monkeypatch.setattr(path_jail, "_replace_with", acl_differs)
    monkeypatch.setattr(path_jail.os, "fchmod", fchmod)

    path_jail.write_jail_file(notes, tmp_path, b"new")
    monkeypatch.undo()

    assert notes.read_text(encoding="utf-8") == "new"
    assert applied == []
    assert stat.S_IMODE(notes.stat().st_mode) == 0o640


@pytest.mark.parametrize("char", ["\u6587", "\U0001f600"], ids=["cjk", "emoji"])
@pytest.mark.asyncio
async def test_a_file_whose_name_is_long_in_bytes_can_be_written(tmp_path: Path, char: str) -> None:
    """The temporary file's name kept 100 characters of the target's, so a
    name of multi-byte characters, valid at up to 255 bytes, made a temporary
    name longer than that, and the write failed with ENAMETOOLONG."""
    stem = char * ((250 - len(".txt")) // len(char.encode("utf-8")))
    target = tmp_path / f"{stem}.txt"
    assert len(os.fsencode(target.name)) <= 255
    handler = FileHandler(make_config(tmp_path))

    response = await handler.handle_write(_write_msg(target, "new"))

    assert not isinstance(response, ErrorMessage), response
    assert target.read_text(encoding="utf-8") == "new"


def test_a_temporary_name_beside_a_file_stays_short_in_bytes() -> None:
    from autogpt_local_executor.fsops import temporary_sibling

    for name in ["\u6587" * 84, "\U0001f600" * 63, "a" * 255, "x.txt"]:
        temp = temporary_sibling(Path("/data") / name)
        assert temp.parent == Path("/data")
        assert temp.name.startswith(".") and temp.name.endswith(".tmp")
        assert len(os.fsencode(temp.name)) <= 120
        # Cut on a character boundary: the name is still valid UTF-8.
        os.fsencode(temp.name).decode("utf-8")


@pytest.mark.asyncio
async def test_a_write_overwrites_in_place_when_the_file_cannot_be_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows refuses to replace a file another program has open, and a
    directory may take no new entries; the old in-place write still works."""
    import autogpt_local_executor.path_jail as path_jail

    target = tmp_path / "open-elsewhere.txt"
    target.write_text("old", encoding="utf-8")

    def refuse(*_args) -> None:
        raise PermissionError(errno.EACCES, "The process cannot access the file")

    monkeypatch.setattr(path_jail.os, "replace", refuse)
    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))
    monkeypatch.undo()

    assert isinstance(response, AckMessage)
    assert target.read_text(encoding="utf-8") == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["open-elsewhere.txt"]


@pytest.mark.asyncio
async def test_a_move_refuses_a_destination_that_appears_after_the_check_where_the_os_can(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without overwrite, a file created between the existence check and the
    rename used to be replaced on POSIX, where rename overwrites silently.
    The rename itself refuses it where the OS can (renameat2, renamex_np,
    Windows' rename). Elsewhere fsops falls back to the check alone, which
    keeps that gap (see fsops.rename_no_replace), so this doesn't run there."""
    if sys.platform != "win32" and not _the_os_refuses_a_replacing_rename(tmp_path):
        pytest.skip("this OS or filesystem can't refuse in the rename itself")
    src = tmp_path / "draft.txt"
    src.write_text("draft", encoding="utf-8")
    dst = tmp_path / "final.txt"
    dst.write_text("someone else's file", encoding="utf-8")
    real_lexists = os.path.lexists
    monkeypatch.setattr(
        handlers_module.os.path,
        "lexists",
        lambda p: False if str(p) == str(dst) else real_lexists(p),
    )

    response = await FileHandler(make_config(tmp_path)).handle_move(_move_msg(src, dst))
    monkeypatch.undo()

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.PATH_EXISTS
    assert dst.read_text(encoding="utf-8") == "someone else's file"
    assert src.read_text(encoding="utf-8") == "draft"


def _the_os_refuses_a_replacing_rename(folder: Path) -> bool:
    """Whether this OS and filesystem refuse a replacing rename in the rename
    itself, asked of the C library directly. Asked of the code under test, a
    regression that made it report no support skipped these tests instead
    of failing them."""
    import ctypes
    import ctypes.util

    src, dst = folder / ".probe-src", folder / ".probe-dst"
    src.write_text("", encoding="utf-8")
    dst.write_text("", encoding="utf-8")
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if sys.platform == "linux" and hasattr(libc, "renameat2"):
            # renameat2(AT_FDCWD, src, AT_FDCWD, dst, RENAME_NOREPLACE)
            result = libc.renameat2(-100, os.fsencode(src), -100, os.fsencode(dst), 1)
        elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
            # renamex_np(src, dst, RENAME_EXCL)
            result = libc.renamex_np(os.fsencode(src), os.fsencode(dst), 4)
        else:
            return False
        return result != 0 and ctypes.get_errno() == errno.EEXIST
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows renames without libc")
def test_the_c_library_is_looked_up_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every no-replace move looked the C library up again, and on Linux
    ctypes.util.find_library runs `ldconfig -p` in a subprocess to do it."""
    import ctypes.util

    from autogpt_local_executor import fsops

    lookups: list[str] = []
    real_find_library = ctypes.util.find_library

    def find_library(name: str) -> str | None:
        lookups.append(name)
        return real_find_library(name)

    monkeypatch.setattr(ctypes.util, "find_library", find_library)
    fsops._libc.cache_clear()
    try:
        for name in ("a", "b", "c"):
            (tmp_path / name).write_text(name, encoding="utf-8")
            fsops.rename_no_replace(tmp_path / name, tmp_path / f"{name}-moved")
    finally:
        fsops._libc.cache_clear()

    assert len(lookups) <= 1


@pytest.mark.parametrize("native", [True, False], ids=["native", "fallback"])
def test_rename_no_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: bool) -> None:
    from autogpt_local_executor import fsops

    if not native:
        monkeypatch.setattr(fsops, "_native_no_replace_rename", lambda _src, _dst: False)
    src = tmp_path / "a"
    src.write_text("a", encoding="utf-8")
    taken = tmp_path / "b"
    taken.write_text("b", encoding="utf-8")

    with pytest.raises(FileExistsError):
        fsops.rename_no_replace(src, taken)
    if native and sys.platform in ("linux", "darwin"):
        if not _the_os_refuses_a_replacing_rename(tmp_path):
            pytest.skip("this filesystem can't refuse in the rename itself")
        # The atomic call itself must work here, not just the fallback.
        assert fsops._native_no_replace_rename(src, tmp_path / "c") is True
    else:
        fsops.rename_no_replace(src, tmp_path / "c")

    assert taken.read_text(encoding="utf-8") == "b"
    assert (tmp_path / "c").read_text(encoding="utf-8") == "a"
    assert not src.exists()


@pytest.mark.asyncio
async def test_a_listing_is_capped_whatever_max_entries_asks_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """max_entries had no bound, so a huge value walked the whole tree first."""
    for index in range(30):
        (tmp_path / f"f{index:02d}").write_text("", encoding="utf-8")
    monkeypatch.setattr(handlers_module, "MAX_FILE_LIST_ENTRIES", 10)
    msg = FileListMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileListPayload(path=str(tmp_path), max_entries=10**9),
    )

    response = await FileHandler(make_config(tmp_path)).handle_list(msg)

    assert isinstance(response, FileListResponseMessage)
    assert len(response.payload.entries) == 10
    assert response.payload.truncated is True


# ── The event loop and slow mounts ──────────────────────────────────────────


def _loop_stats_inside(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The paths in `root` that os.stat, os.lstat and os.path.realpath are
    called on from the calling thread, the event loop's. exists, is_dir,
    is_symlink and lexists all come down to the first two, and so does
    realpath on POSIX; on Windows realpath resolves a path that exists
    through nt._getfinalpathname, without them."""
    loop_thread = threading.get_ident()
    prefixes = (str(root), os.path.realpath(root))
    seen: list[str] = []

    def watched(real: Callable[..., Any]) -> Callable[..., Any]:
        def call(path: Any, *args: Any, **kwargs: Any) -> Any:
            if threading.get_ident() == loop_thread and not isinstance(path, int):
                name = os.fsdecode(os.fspath(path))
                if name.startswith(prefixes):
                    seen.append(name)
            return real(path, *args, **kwargs)

        return call

    monkeypatch.setattr(os, "stat", watched(os.stat))
    monkeypatch.setattr(os, "lstat", watched(os.lstat))
    monkeypatch.setattr(os.path, "realpath", watched(os.path.realpath))
    return seen


@pytest.mark.parametrize("call", ["realpath", "resolve", "exists", "lexists"])
def test_the_loop_watch_sees_a_path_in_the_root_checked_on_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: str
) -> None:
    """The check below passes only if its watch sees an in-root path checked
    on the calling thread. Watching os.stat and os.lstat alone missed one
    resolved on Windows, where realpath goes through nt._getfinalpathname
    for a path that exists."""
    root = tmp_path / "root"
    target = root / "dir"
    target.mkdir(parents=True)
    seen = _loop_stats_inside(root, monkeypatch)
    checks = {
        "realpath": lambda: os.path.realpath(target),
        "resolve": target.resolve,
        "exists": target.exists,
        "lexists": lambda: os.path.lexists(target),
    }

    checks[call]()

    assert seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["command", "read", "write", "stat", "stat-entry", "list", "delete", "move"]
)
async def test_a_request_checks_its_paths_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Each request resolved its paths in the jail, and list, delete and move
    checked them, on the event loop every session shares. A stalled mount
    (NFS, FUSE, a cloud-synced folder) holds a stat for as long as it stalls,
    and with it every session's work and keepalive."""
    root = tmp_path / "root"
    (root / "dir").mkdir(parents=True)
    a = root / "dir" / "a.txt"
    a.write_text("a", encoding="utf-8")
    config = ShimConfig(
        allowed_root=root,
        audit_log_path=tmp_path / "_audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        enable_shell=True,
    )
    files = FileHandler(config)
    listing = FileListMessage(id=new_id(), ts=now_ts(), payload=FileListPayload(path=str(a.parent)))
    requests = {
        "command": lambda: CommandHandler(config).handle(
            _cmd_msg(argv=[sys.executable, "-c", "pass"], cwd=str(a.parent))
        ),
        "read": lambda: files.handle_read(_read_msg(a)),
        "write": lambda: files.handle_write(_write_msg(a.parent / "b.txt", "b")),
        "stat": lambda: files.handle_stat(_stat_msg(a)),
        "stat-entry": lambda: files.handle_stat(_stat_msg(a, follow_symlinks=False)),
        "list": lambda: files.handle_list(listing),
        "delete": lambda: files.handle_delete(_delete_msg(a)),
        "move": lambda: files.handle_move(_move_msg(a, a.parent / "c.txt")),
    }
    seen = _loop_stats_inside(root, monkeypatch)

    response = await requests[kind]()

    assert not isinstance(response, ErrorMessage), response
    assert seen == []
