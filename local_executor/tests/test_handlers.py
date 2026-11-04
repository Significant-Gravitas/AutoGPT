"""Tests for FileHandler and CommandHandler."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import os
import signal
import sys
import textwrap
import time
from pathlib import Path
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
    outside.write_text("outside secret")
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
    assert outside.read_text() == "outside secret"
    assert target.read_text() == "outside secret"


@pytest.mark.asyncio
async def test_file_content_checks_opened_descriptor_before_access(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "single.txt"
    target.write_text("must remain intact")
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
    assert target.read_text() == "must remain intact"
    assert outside.read_text() == "must remain intact"


@pytest.mark.asyncio
async def test_a_write_never_reaches_a_hard_link_made_after_the_check(tmp_path: Path) -> None:
    """A write puts a new file at the jail path rather than writing into the
    old one, so a hard link made from outside after the check keeps the old
    contents: nothing outside the jail is written."""
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "single.txt"
    target.write_text("must remain intact")
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
    assert outside.read_text() == "must remain intact"
    assert target.read_text() == "overwrite"


@pytest.mark.asyncio
async def test_write_existing_single_link_truncates_after_validation(tmp_path: Path) -> None:
    target = tmp_path / "single.txt"
    target.write_text("a long original value")

    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))

    assert isinstance(response, AckMessage)
    assert target.read_text() == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["read", "write"])
async def test_file_content_allows_symlink_to_single_link_inside_root(
    tmp_path: Path, operation: str
) -> None:
    target = tmp_path / "single.txt"
    target.write_text("original")
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
        assert target.read_text() == "new"

    assert link.is_symlink()


@pytest.mark.asyncio
async def test_write_creates_parents(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    target = tmp_path / "a" / "b" / "c.txt"
    resp = await h.handle_write(_write_msg(target, "hi", create_parents=True))
    assert isinstance(resp, AckMessage)
    assert target.read_text() == "hi"


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
    (tmp_path / "f.txt").write_text("abcdefghij")
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
    target.write_text("hi")
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
    target.write_text("a,b\n")
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
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.csv").write_text("b")
    (tmp_path / "sub").mkdir()
    msg = FileListMessage(id=new_id(), ts=now_ts(), payload=FileListPayload(path=str(tmp_path)))
    resp = await h.handle_list(msg)
    assert isinstance(resp, FileListResponseMessage)
    names = {e.name for e in resp.payload.entries}
    assert {"a.txt", "b.csv", "sub"} <= names


@pytest.mark.asyncio
async def test_list_glob_csv(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.csv").write_text("b")
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
async def test_list_max_entries_truncates(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    for i in range(5):
        (tmp_path / f"f{i}").write_text("x")
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
    (tmp_path / ".hidden").write_text("x")
    (tmp_path / "visible").write_text("x")
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
    target.write_text("x")
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
    (d / "x").write_text("x")
    msg = FileDeleteMessage(id=new_id(), ts=now_ts(), payload=FileDeletePayload(path=str(d)))
    resp = await h.handle_delete(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.PATH_NOT_EMPTY


@pytest.mark.asyncio
async def test_delete_recursive(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "x").write_text("x")
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
    (documents / "report.txt").write_text("keep")
    shortcut = tmp_path / "shortcut"
    _symlink_or_skip(shortcut, documents)

    response = await FileHandler(make_config(tmp_path)).handle_delete(
        _delete_msg(shortcut, recursive=recursive)
    )

    assert isinstance(response, AckMessage)
    assert not os.path.lexists(shortcut)
    assert (documents / "report.txt").read_text() == "keep"


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
    (outside / "report.txt").write_text("keep")
    shortcut = root / "shortcut"
    _symlink_or_skip(shortcut, outside)

    response = await FileHandler(make_config(root)).handle_delete(
        _delete_msg(shortcut, recursive=recursive)
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert shortcut.is_symlink()
    assert (outside / "report.txt").read_text() == "keep"


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
    (root / "notes.txt").write_text("keep")

    response = await FileHandler(make_config(root)).handle_delete(_delete_msg(root, recursive=True))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert (root / "notes.txt").read_text() == "keep"


# ── FILE_MOVE ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_move_rename(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    src = tmp_path / "a.txt"
    src.write_text("x")
    dst = tmp_path / "b.txt"
    msg = FileMoveMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileMovePayload(src=str(src), dst=str(dst)),
    )
    resp = await h.handle_move(msg)
    assert isinstance(resp, AckMessage)
    assert not src.exists()
    assert dst.read_text() == "x"


@pytest.mark.asyncio
async def test_move_blocks_dst_outside_jail(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    src = tmp_path / "a.txt"
    src.write_text("x")
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
    source.write_text("original")
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
    assert source.read_text() == "original"
    assert not outside.exists()
    assert target.is_symlink()


@pytest.mark.asyncio
async def test_move_exists_without_overwrite(tmp_path: Path) -> None:
    h = FileHandler(make_config(tmp_path))
    src = tmp_path / "a.txt"
    src.write_text("x")
    dst = tmp_path / "b.txt"
    dst.write_text("y")
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
    src.write_text("new")
    dst = tmp_path / "b.txt"
    dst.write_text("old")
    msg = FileMoveMessage(
        id=new_id(),
        ts=now_ts(),
        payload=FileMovePayload(src=str(src), dst=str(dst), overwrite=True),
    )
    resp = await h.handle_move(msg)
    assert isinstance(resp, AckMessage)
    assert dst.read_text() == "new"


def _move_msg(src: Path, dst: Path, **kw) -> FileMoveMessage:
    return FileMoveMessage(
        id=new_id(), ts=now_ts(), payload=FileMovePayload(src=str(src), dst=str(dst), **kw)
    )


@pytest.mark.asyncio
async def test_move_renames_a_symlink_not_its_target(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "report.txt").write_text("keep")
    shortcut = tmp_path / "shortcut"
    _symlink_or_skip(shortcut, documents)
    renamed = tmp_path / "renamed"

    response = await FileHandler(make_config(tmp_path)).handle_move(_move_msg(shortcut, renamed))

    assert isinstance(response, AckMessage)
    assert renamed.is_symlink()
    assert not os.path.lexists(shortcut)
    assert (documents / "report.txt").read_text() == "keep"


@pytest.mark.asyncio
async def test_move_overwrite_replaces_a_symlink_destination_not_its_target(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real.txt"
    real.write_text("keep")
    link = tmp_path / "link.txt"
    _symlink_or_skip(link, real)
    source = tmp_path / "new.txt"
    source.write_text("new")

    handler = FileHandler(make_config(tmp_path))
    refused = await handler.handle_move(_move_msg(source, link))
    replaced = await handler.handle_move(_move_msg(source, link, overwrite=True))

    assert isinstance(refused, ErrorMessage)
    assert refused.payload.code == ErrorCode.PATH_EXISTS
    assert isinstance(replaced, AckMessage)
    assert not link.is_symlink()
    assert link.read_text() == "new"
    assert real.read_text() == "keep"


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
        (source / "source.txt").write_text("source")
        (destination / "destination.txt").write_text("destination")
    else:
        source.write_text("source")
        destination.write_text("destination")
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
        assert (source / "source.txt").read_text() == "source"
        assert (destination / "destination.txt").read_text() == "destination"
    else:
        assert source.read_text() == "source"
        assert destination.read_text() == "destination"


@pytest.mark.asyncio
async def test_move_overwrite_hard_link_does_not_change_outside_content(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret")
    destination = root / "linked.txt"
    os.link(outside, destination)
    source = root / "source.txt"
    source.write_text("workspace content")

    response = await FileHandler(make_config(root)).handle_move(
        FileMoveMessage(
            id=new_id(),
            ts=now_ts(),
            payload=FileMovePayload(src=str(source), dst=str(destination), overwrite=True),
        )
    )

    assert isinstance(response, AckMessage)
    assert outside.read_text() == "outside secret"
    assert destination.read_text() == "workspace content"
    assert not source.exists()


@pytest.mark.asyncio
async def test_move_hard_link_does_not_grant_content_access(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret")
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
    assert outside.read_text() == "outside secret"
    assert not source.exists()


@pytest.mark.asyncio
async def test_move_cannot_overwrite_nonempty_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source")
    (destination / "destination.txt").write_text("destination")

    response = await FileHandler(make_config(tmp_path)).handle_move(
        FileMoveMessage(
            id=new_id(),
            ts=now_ts(),
            payload=FileMovePayload(src=str(source), dst=str(destination), overwrite=True),
        )
    )

    assert isinstance(response, ErrorMessage)
    assert (source / "source.txt").read_text() == "source"
    assert (destination / "destination.txt").read_text() == "destination"


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
        with open(lock_path, "w") as probe:
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
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
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
    lines = config.audit_log_path.read_text().splitlines()
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
    (tmp_path / ".git" / "config").write_text("[remote]")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print()")
    (tmp_path / "src" / ".env").write_text("TOKEN=x")

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
    """A file object whose writes fail the way a full disk does."""

    def __init__(self, file) -> None:
        self._file = file

    def write(self, _data: bytes) -> int:
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
    target.write_text("the user's notes")
    real_fdopen = os.fdopen
    monkeypatch.setattr(os, "fdopen", lambda fd, *a, **k: _FullDisk(real_fdopen(fd, *a, **k)))

    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))
    monkeypatch.undo()

    assert isinstance(response, ErrorMessage)
    assert target.read_text() == "the user's notes"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
@pytest.mark.asyncio
async def test_a_write_keeps_the_mode_and_a_new_file_follows_the_umask(tmp_path: Path) -> None:
    existing = tmp_path / "private.txt"
    existing.write_text("old")
    existing.chmod(0o640)
    created = tmp_path / "created.txt"
    umask = os.umask(0o022)
    try:
        handler = FileHandler(make_config(tmp_path))
        await handler.handle_write(_write_msg(existing, "new"))
        await handler.handle_write(_write_msg(created, "new"))
    finally:
        os.umask(umask)

    assert existing.read_text() == "new"
    assert existing.stat().st_mode & 0o777 == 0o640
    assert created.stat().st_mode & 0o777 == 0o644


@pytest.mark.asyncio
async def test_a_write_overwrites_in_place_when_the_file_cannot_be_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows refuses to replace a file another program has open, and a
    directory may take no new entries; the old in-place write still works."""
    import autogpt_local_executor.path_jail as path_jail

    target = tmp_path / "open-elsewhere.txt"
    target.write_text("old")

    def refuse(*_args) -> None:
        raise PermissionError(errno.EACCES, "The process cannot access the file")

    monkeypatch.setattr(path_jail.os, "replace", refuse)
    response = await FileHandler(make_config(tmp_path)).handle_write(_write_msg(target, "new"))
    monkeypatch.undo()

    assert isinstance(response, AckMessage)
    assert target.read_text() == "new"
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
    src.write_text("draft")
    dst = tmp_path / "final.txt"
    dst.write_text("someone else's file")
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
    assert dst.read_text() == "someone else's file"
    assert src.read_text() == "draft"


def _the_os_refuses_a_replacing_rename(folder: Path) -> bool:
    from autogpt_local_executor import fsops

    src, dst = folder / ".probe-src", folder / ".probe-dst"
    src.write_text("")
    dst.write_text("")
    try:
        return fsops._native_no_replace_rename(src, dst)
    except FileExistsError:
        return True
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


@pytest.mark.parametrize("native", [True, False], ids=["native", "fallback"])
def test_rename_no_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: bool) -> None:
    from autogpt_local_executor import fsops

    if not native:
        monkeypatch.setattr(fsops, "_native_no_replace_rename", lambda _src, _dst: False)
    src = tmp_path / "a"
    src.write_text("a")
    taken = tmp_path / "b"
    taken.write_text("b")

    with pytest.raises(FileExistsError):
        fsops.rename_no_replace(src, taken)
    if native and sys.platform in ("linux", "darwin"):
        if not _the_os_refuses_a_replacing_rename(tmp_path):
            pytest.skip("this filesystem can't refuse in the rename itself")
        # The atomic call itself must work here, not just the fallback.
        assert fsops._native_no_replace_rename(src, tmp_path / "c") is True
    else:
        fsops.rename_no_replace(src, tmp_path / "c")

    assert taken.read_text() == "b"
    assert (tmp_path / "c").read_text() == "a"
    assert not src.exists()


@pytest.mark.asyncio
async def test_a_listing_is_capped_whatever_max_entries_asks_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """max_entries had no bound, so a huge value walked the whole tree first."""
    for index in range(30):
        (tmp_path / f"f{index:02d}").write_text("")
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
