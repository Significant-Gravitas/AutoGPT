"""An error names the path that was asked for, never the one it resolved to.

A refused path's reply read `Path '<asked>' (resolved to '<realpath>') is
outside allowed_root '<root's realpath>'`. For a link in the shared folder
that told the platform where the link points outside it, which FILE_LIST
takes care not to tell; on Windows the resolved prefix comes back in the
disk's own letter case, which also says whether a path exists. The replies
built from an OS error's text carried the file name the system reported,
which is the resolved path too, and the root's own real place with it.

Each reply names the path as asked now, and what the path resolved to goes
to the local log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pytest

import autogpt_local_executor.handlers as handlers_module
from autogpt_local_executor import path_jail
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import ShimDaemon
from autogpt_local_executor.handlers import CommandHandler, FileHandler
from autogpt_local_executor.path_jail import PathJailError
from autogpt_local_executor.protocol import (
    ErrorCode,
    ErrorMessage,
    ExecuteCommandMessage,
    ExecuteCommandPayload,
    FileDeleteMessage,
    FileDeletePayload,
    FileListMessage,
    FileListPayload,
    FileMoveMessage,
    FileMovePayload,
    FileReadMessage,
    FileReadPayload,
    FileStatMessage,
    FileStatPayload,
    FileWriteMessage,
    FileWritePayload,
    new_id,
    now_ts,
)

# The one name of the place a link leads to. No reply may hold it.
ELSEWHERE = "where-the-link-goes"


def _link_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this platform/account")


def _config(root: Path, tmp_path: Path) -> ShimConfig:
    return ShimConfig(
        allowed_root=root,
        audit_log_path=tmp_path / "_audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        enable_shell=True,
    )


def _envelope() -> dict[str, Any]:
    return {"id": new_id(), "ts": now_ts()}


async def _ask(config: ShimConfig, operation: str, path: Path) -> Any:
    """The reply to one request about `path`, as the platform gets it."""
    files = FileHandler(config)
    asked = str(path)
    if operation == "read":
        return await files.handle_read(
            FileReadMessage(**_envelope(), payload=FileReadPayload(path=asked))
        )
    if operation == "write":
        return await files.handle_write(
            FileWriteMessage(**_envelope(), payload=FileWritePayload(path=asked, content="x"))
        )
    if operation == "stat":
        return await files.handle_stat(
            FileStatMessage(**_envelope(), payload=FileStatPayload(path=asked))
        )
    if operation == "list":
        return await files.handle_list(
            FileListMessage(**_envelope(), payload=FileListPayload(path=asked))
        )
    if operation == "delete":
        return await files.handle_delete(
            FileDeleteMessage(**_envelope(), payload=FileDeletePayload(path=asked))
        )
    if operation == "move-from":
        return await files.handle_move(
            FileMoveMessage(
                **_envelope(),
                payload=FileMovePayload(src=asked, dst=str(config.allowed_root / "moved")),
            )
        )
    if operation == "move-to":
        (config.allowed_root / "to-move.txt").write_text("x", encoding="utf-8")
        return await files.handle_move(
            FileMoveMessage(
                **_envelope(),
                payload=FileMovePayload(src=str(config.allowed_root / "to-move.txt"), dst=asked),
            )
        )
    assert operation == "command-cwd"
    return await CommandHandler(config).handle(
        ExecuteCommandMessage(
            **_envelope(),
            payload=ExecuteCommandPayload(argv=["true"], cwd=asked),
        )
    )


OPERATIONS = [
    "read",
    "write",
    "stat",
    "list",
    "delete",
    "move-from",
    "move-to",
    "command-cwd",
]


# ── The jail's own refusal ───────────────────────────────────────────────────


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_a_link_that_leaves_the_root_is_refused_without_saying_where_it_goes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, operation: str
) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    elsewhere = tmp_path / ELSEWHERE
    elsewhere.mkdir()
    (elsewhere / "notes.txt").write_text("private", encoding="utf-8")
    _link_or_skip(root / "shortcut", elsewhere)
    asked = root / "shortcut" / "notes.txt"

    with caplog.at_level(logging.INFO, logger="autogpt_local_executor"):
        reply = await _ask(_config(root, tmp_path), operation, asked)

    assert isinstance(reply, ErrorMessage)
    assert reply.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert ELSEWHERE not in reply.payload.message
    assert "resolved to" not in reply.payload.message
    # It names what was asked for.
    assert repr(str(asked)) in reply.payload.message
    # And where the path leads is on record here, for the person.
    assert ELSEWHERE in caplog.text
    assert (elsewhere / "notes.txt").read_text(encoding="utf-8") == "private"


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_a_path_above_the_root_is_refused_as_it_was_asked(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, operation: str
) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    (tmp_path / ELSEWHERE).mkdir()
    asked = f"{root}{os.sep}..{os.sep}{ELSEWHERE}{os.sep}notes.txt"

    with caplog.at_level(logging.INFO, logger="autogpt_local_executor"):
        reply = await _ask(_config(root, tmp_path), operation, Path(asked))

    assert isinstance(reply, ErrorMessage)
    assert reply.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert "resolved to" not in reply.payload.message
    # The path as asked, `..` and all, and not the place that comes to,
    # which is in the log. (Each as it is written there: quoted.)
    comes_to = repr(os.path.normpath(asked))
    assert repr(asked) in reply.payload.message
    assert comes_to not in reply.payload.message
    assert comes_to in caplog.text


def test_the_refusal_reads_the_same_by_name_and_after_resolving(tmp_path: Path) -> None:
    """A path refused by its name alone (round 34) and one refused once it
    was resolved get one message, which names the path asked for and the
    root as the platform was told it: nothing in it says which check
    refused, or what a link in the root points at."""
    root = tmp_path / "shared"
    root.mkdir()
    elsewhere = tmp_path / ELSEWHERE
    elsewhere.mkdir()
    _link_or_skip(root / "shortcut", elsewhere)
    by_name = str(tmp_path / "another" / "x")
    by_link = str(root / "shortcut" / "x")

    messages = []
    for asked in (by_name, by_link):
        with pytest.raises(PathJailError) as refused:
            path_jail.assert_inside_jail(asked, root)
        assert refused.value.code == "PATH_OUTSIDE_ALLOWED_ROOT"
        messages.append(refused.value.message.replace(repr(asked), "<asked>"))
        assert refused.value.detail

    assert messages[0] == messages[1]
    assert messages[0] == f"Path <asked> is outside allowed_root {str(root)!r}"


def test_a_root_reached_through_a_link_is_named_as_it_was_configured(tmp_path: Path) -> None:
    """The root's own real place isn't the platform's to learn either: it
    was told the root as the person named it."""
    real = tmp_path / ELSEWHERE
    real.mkdir()
    alias = tmp_path / "shared"
    _link_or_skip(alias, real)

    with pytest.raises(PathJailError) as refused:
        path_jail.assert_inside_jail(tmp_path / "another" / "x", alias)

    assert ELSEWHERE not in refused.value.message
    assert repr(str(alias)) in refused.value.message
    assert ELSEWHERE in refused.value.detail


def test_a_root_that_is_not_there_is_said_without_the_systems_file_name(tmp_path: Path) -> None:
    """The system's error for a root that can't be resolved names the part
    it stopped at, with the links before it followed."""
    real = tmp_path / ELSEWHERE
    real.mkdir()
    _link_or_skip(tmp_path / "alias", real)
    root = tmp_path / "alias" / "not-there"

    with pytest.raises(PathJailError) as refused:
        path_jail.assert_inside_jail(root / "x", root)

    assert ELSEWHERE not in refused.value.message
    assert repr(str(root)) in refused.value.message
    assert refused.value.detail
    if sys.platform != "win32":
        # POSIX names the part it stopped at, with the links before it
        # followed. Windows names the path as it was given.
        assert ELSEWHERE in refused.value.detail


# ── Replies built from the system's own error ────────────────────────────────


def _the_system_refuses(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Have each named os function fail as the system does: with the path
    it was given, the resolved one, in its error."""

    def refuses(path: Any, *_args: Any, **_kwargs: Any) -> Any:
        raise PermissionError(13, "Permission denied", os.fspath(path))

    for name in names:
        monkeypatch.setattr(os, name, refuses)


@pytest.fixture
def linked_root(tmp_path: Path) -> tuple[Path, Path]:
    """A shared folder the person named by a link: (as named, its real place)."""
    real = tmp_path / ELSEWHERE
    real.mkdir()
    (real / "folder").mkdir()
    (real / "folder" / "inside.txt").write_text("x", encoding="utf-8")
    (real / "file.txt").write_text("x", encoding="utf-8")
    alias = tmp_path / "shared"
    _link_or_skip(alias, real)
    return alias, real


def _named_as_asked(reply: Any, asked: Path) -> None:
    assert isinstance(reply, ErrorMessage), reply
    assert ELSEWHERE not in reply.payload.message, reply.payload.message
    assert repr(str(asked)) in reply.payload.message, reply.payload.message


@pytest.mark.parametrize(
    ("operation", "name", "code"),
    [
        ("read", "missing.txt", ErrorCode.PATH_NOT_FOUND),
        ("list", "missing", ErrorCode.PATH_NOT_FOUND),
        ("list", "file.txt", ErrorCode.INTERNAL_ERROR),
        ("delete", "missing.txt", ErrorCode.PATH_NOT_FOUND),
        ("delete", "folder", ErrorCode.PATH_NOT_EMPTY),
        ("move-from", "missing.txt", ErrorCode.PATH_NOT_FOUND),
        ("move-to", "file.txt", ErrorCode.PATH_EXISTS),
    ],
)
async def test_a_reply_about_a_path_in_the_root_names_it_as_asked(
    tmp_path: Path, linked_root: tuple[Path, Path], operation: str, name: str, code: ErrorCode
) -> None:
    """These replies named the path the handler worked on: the resolved
    one, under the root's real place."""
    alias, _real = linked_root
    asked = alias / name

    reply = await _ask(_config(alias, tmp_path), operation, asked)

    _named_as_asked(reply, asked)
    assert reply.payload.code == code


@pytest.mark.parametrize(
    ("operation", "name", "refusing"),
    [
        ("read", "file.txt", ("open",)),
        ("write", "new.txt", ("open",)),
        ("stat", "file.txt", ("stat",)),
        ("list", "folder", ("scandir", "listdir")),
        ("delete", "file.txt", ("unlink",)),
        ("move-from", "file.txt", ()),
    ],
)
async def test_a_reply_built_from_the_systems_error_leaves_its_file_name_out(
    tmp_path: Path,
    linked_root: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    operation: str,
    name: str,
    refusing: tuple[str, ...],
) -> None:
    """An OSError's text ends with the file name the system was given, the
    resolved path. The reply keeps the system's reason and names the path
    as asked; the error as the system gave it goes to the log."""
    alias, _real = linked_root
    asked = alias / name
    config = _config(alias, tmp_path)

    def refused_rename(src: Any, dst: Any) -> None:
        raise PermissionError(13, "Permission denied", os.fspath(src), None, os.fspath(dst))

    with caplog.at_level(logging.INFO, logger="autogpt_local_executor"):
        with monkeypatch.context() as during:
            _the_system_refuses(during, *refusing)
            # A move goes through the rename that refuses to replace.
            during.setattr(handlers_module, "rename_no_replace", refused_rename)
            reply = await _ask(config, operation, asked)

    _named_as_asked(reply, asked)
    assert "Permission denied" in reply.payload.message
    assert ELSEWHERE in caplog.text


@pytest.mark.parametrize("operation", ["read", "write"])
async def test_a_refusal_at_the_second_check_of_a_files_content_names_the_path_as_asked(
    tmp_path: Path,
    linked_root: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    operation: str,
) -> None:
    """A read and a write check the path twice: once to place it, and again
    as the content is opened, by the resolved path the first check gave. A
    refusal at that second check named the path it had been given, the
    resolved one, under the root's real place. It takes a change between
    the two checks to be refused there: here the link the root is named by
    is pointed somewhere else."""
    alias, real = linked_root
    asked = alias / "file.txt"
    another = tmp_path / "another-place"
    another.mkdir()

    def after_the_link_moved(checked_again: Any) -> Any:
        def moved_first(*args: Any, **kwargs: Any) -> Any:
            alias.unlink()
            alias.symlink_to(another, target_is_directory=True)
            return checked_again(*args, **kwargs)

        return moved_first

    for name in ("open_jail_file", "write_jail_file"):
        monkeypatch.setattr(
            handlers_module, name, after_the_link_moved(getattr(handlers_module, name))
        )

    with caplog.at_level(logging.INFO, logger="autogpt_local_executor"):
        reply = await _ask(_config(alias, tmp_path), operation, asked)

    _named_as_asked(reply, asked)
    assert reply.payload.code == ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    # Where it had resolved to stays on this computer.
    assert ELSEWHERE in caplog.text
    assert (real / "file.txt").read_text(encoding="utf-8") == "x"
    assert list(another.iterdir()) == []


class _Socket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)


async def test_a_handler_that_fails_unexpectedly_says_no_more_than_what_kind_of_failure(
    tmp_path: Path,
    linked_root: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The daemon answered a handler's unexpected failure with the failure's
    own text: for an OSError, the resolved path again."""
    alias, real = linked_root
    config = _config(alias, tmp_path)
    config.session_id = "chat-1"
    daemon = ShimDaemon(
        config,
        token_store=object(),
        audit=AuditWriter(path=tmp_path / "_audit.log", audit_key=b"a" * 32),
    )
    daemon._semaphore = asyncio.Semaphore(config.max_concurrent)

    async def fails(_message: Any, *, send: Any = None) -> Any:
        raise PermissionError(13, "Permission denied", str(real / "file.txt"))

    monkeypatch.setattr(daemon, "_handle", fails)
    socket = _Socket()
    request = FileStatMessage(**_envelope(), payload=FileStatPayload(path=str(alias / "file.txt")))

    with caplog.at_level(logging.ERROR, logger="autogpt_local_executor"):
        await daemon._dispatch(socket, request)

    [reply] = [json.loads(frame) for frame in socket.sent]
    assert reply["type"] == "ERROR"
    assert reply["payload"]["code"] == ErrorCode.INTERNAL_ERROR.value
    assert ELSEWHERE not in reply["payload"]["message"]
    assert "PermissionError" in reply["payload"]["message"]
    assert "Permission denied" in reply["payload"]["message"]
    # The whole failure is in the log, as it was.
    assert ELSEWHERE in caplog.text
