"""Browsing a folder takes read access; choosing one for a chat takes write
access as well.

The folder dialog listed, and let a person open, only folders their account
could read, write and search. A writable folder under a read-only one
couldn't be reached. A Linux USB disk with ext4 or xfs is mounted with a top
folder that is root's, mode 0755, so the whole disk was missing from the
dialog's first list.

On Windows it went the other way: `os.access` looks at the read-only
attribute and no further, so every folder counted as writable. Another
account's profile under `C:\\Users` was listed and could be chosen, and then
every file operation in it failed.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import autogpt_local_executor.daemon as daemon_module
from autogpt_local_executor import fsops, session_grants
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import ShimDaemon
from autogpt_local_executor.directory_browser import DirectoryBrowseError, DirectoryBrowser
from autogpt_local_executor.protocol import (
    DirectoryListRequestMessage,
    DirectoryListRequestPayload,
    DirectoryListResponseMessage,
    ErrorCode,
    ErrorMessage,
    new_id,
    now_ts,
)
from autogpt_local_executor.session_grants import RootGrantError

posix_modes = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX file modes, as an account they apply to",
)
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows access lists")


def _platform() -> str:
    return "windows" if sys.platform == "win32" else "linux"


def _browser(home: Path, monkeypatch: pytest.MonkeyPatch) -> DirectoryBrowser:
    monkeypatch.setattr("autogpt_local_executor.directory_browser._windows_drive_roots", lambda: [])
    browser = DirectoryBrowser(platform_name=_platform(), home=home)
    monkeypatch.setattr(browser, "_children_of", lambda _parent: [])
    return browser


def _open(browser: DirectoryBrowser, browse_id: str, listing: Any, name: str) -> Any:
    """Open the listed folder called `name`: (its reference, its listing)."""
    [entry] = [entry for entry in listing.entries if entry.name == name]
    return entry, browser.list_directories(browse_id, entry.directory_ref)


def _names(listing: Any) -> list[str]:
    return [entry.name for entry in listing.entries]


# ── POSIX ────────────────────────────────────────────────────────────────────


@posix_modes
def test_a_writable_folder_under_a_read_only_one_can_be_reached_and_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "read-only"
    (parent / "mine").mkdir(parents=True)
    parent.chmod(0o555)
    try:
        browser = _browser(tmp_path, monkeypatch)
        roots = browser.list_directories(None, None)
        _home, at_home = _open(browser, roots.browse_id, roots, tmp_path.name)

        # Listed, and it opens.
        assert "read-only" in _names(at_home)
        read_only, inside = _open(browser, roots.browse_id, at_home, "read-only")
        assert _names(inside) == ["mine"]
        [mine] = inside.entries

        # It can't be chosen: nothing could be written in it.
        with pytest.raises(DirectoryBrowseError) as refused:
            browser.resolve(roots.browse_id, read_only.directory_ref)
        assert refused.value.code == "DIRECTORY_UNAVAILABLE"
        assert "write" in refused.value.message

        # The folder in it that is the person's own can.
        assert browser.resolve(roots.browse_id, mine.directory_ref) == parent / "mine"
    finally:
        parent.chmod(0o755)


@posix_modes
def test_a_disk_whose_top_folder_is_read_only_is_among_the_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A removable ext4 or xfs disk as udisks mounts it: the top folder is
    root's, 0755, and the person's folders are in it."""
    home = tmp_path / "home"
    home.mkdir()
    disk = tmp_path / "USB-DISK"
    (disk / "photos").mkdir(parents=True)
    disk.chmod(0o555)
    try:
        browser = DirectoryBrowser(platform_name="linux", home=home)
        monkeypatch.setattr(browser, "_linux_user_mounts", lambda: [disk])

        roots = browser.list_directories(None, None)

        assert "USB-DISK" in _names(roots)
        usb, on_the_disk = _open(browser, roots.browse_id, roots, "USB-DISK")
        assert _names(on_the_disk) == ["photos"]
        [photos] = on_the_disk.entries
        with pytest.raises(DirectoryBrowseError):
            browser.resolve(roots.browse_id, usb.directory_ref)
        assert browser.resolve(roots.browse_id, photos.directory_ref) == disk / "photos"
    finally:
        disk.chmod(0o755)


@posix_modes
def test_a_folder_that_cannot_be_read_or_searched_is_still_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, mode in (("no-read", 0o300), ("no-search", 0o600), ("fine", 0o755)):
        (tmp_path / name).mkdir()
        (tmp_path / name).chmod(mode)
    try:
        browser = _browser(tmp_path, monkeypatch)
        roots = browser.list_directories(None, None)
        _home, at_home = _open(browser, roots.browse_id, roots, tmp_path.name)

        assert _names(at_home) == ["fine"]
    finally:
        for name in ("no-read", "no-search"):
            (tmp_path / name).chmod(0o755)


@posix_modes
def test_a_grant_is_issued_and_restored_only_for_a_folder_that_can_be_written(
    tmp_path: Path,
) -> None:
    """The grant's own check keeps the rule the browser's choice has, and
    says which rule: write access that went after the folder was chosen was
    refused as "unavailable", which didn't tell the person to pick a folder
    the account can write to."""
    folder = tmp_path / "read-only"
    folder.mkdir()
    assert session_grants.canonical_directory(folder) == folder
    folder.chmod(0o555)
    try:
        with pytest.raises(RootGrantError, match="this account can't write to it"):
            session_grants.canonical_directory(folder)
    finally:
        folder.chmod(0o755)
    # A folder that can't be opened, or isn't there, is still unavailable.
    folder.chmod(0o300)
    try:
        with pytest.raises(RootGrantError, match="^Selected root is unavailable$"):
            session_grants.canonical_directory(folder)
    finally:
        folder.chmod(0o755)
    with pytest.raises(RootGrantError, match="^Selected root is unavailable$"):
        session_grants.canonical_directory(tmp_path / "not-there")


@pytest.mark.skipif(sys.platform == "win32", reason="what POSIX's access(2) is asked")
def test_posix_access_is_asked_and_the_folder_is_not_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking doesn't open the folder. On macOS, opening one the privacy
    settings guard (Desktop, Documents, Downloads) puts a prompt on the
    person's screen, for a folder they only browsed past."""
    opened: list[Any] = []
    monkeypatch.setattr(os, "scandir", lambda path: opened.append(path))
    monkeypatch.setattr(os, "listdir", lambda path: opened.append(path))

    assert fsops.can_open_folder(tmp_path) is True
    assert fsops.can_write_folder(tmp_path) is True
    assert opened == []


def test_a_folder_the_person_can_write_to_can_be_opened_and_written(tmp_path: Path) -> None:
    """On the OS the tests run on, for real."""
    assert fsops.can_open_folder(tmp_path) is True
    assert fsops.can_write_folder(tmp_path) is True
    assert list(tmp_path.iterdir()) == []
    assert fsops.can_open_folder(tmp_path / "not-there") is False
    assert fsops.can_write_folder(tmp_path / "not-there") is False


# ── Windows ──────────────────────────────────────────────────────────────────


def _this_account() -> str:
    """The SID of the account the tests run as, in icacls' form."""
    out = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return "*" + out.strip().rsplit(",", 1)[1].strip('"')


def _icacls(folder: Path, *arguments: str) -> None:
    subprocess.run(
        ["icacls", str(folder), *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    )


_BACKUP_RIGHTS = ("SeBackupPrivilege", "SeRestorePrivilege")


@contextlib.contextmanager
def _backup_rights(*, on: bool) -> Iterator[None]:
    """Run the block with this process's backup and restore privileges
    switched on or off, and put them back as they were after.

    An administrator's process holds both, switched off unless the program
    that started it switched them on, which Git Bash does and PowerShell
    doesn't. With them on, Windows lets the process open a folder for
    listing, or for adding a file, whatever the folder's access list says.
    So whether an access list stops these tests' account depended on the
    shell that started pytest. pytest.skip when they are to be on and the
    account doesn't hold them (one that is no administrator)."""
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class Luid(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.LONG)]

    class Privilege(ctypes.Structure):
        # TOKEN_PRIVILEGES with its one LUID_AND_ATTRIBUTES.
        _fields_ = [("count", wintypes.DWORD), ("luid", Luid), ("attributes", wintypes.DWORD)]

    adjust, query, enabled, not_all_assigned = 0x0020, 0x0008, 0x00000002, 1300
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.LookupPrivilegeValueW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        ctypes.POINTER(Luid),
    ]
    advapi32.AdjustTokenPrivileges.argtypes = [
        wintypes.HANDLE,
        wintypes.BOOL,
        ctypes.POINTER(Privilege),
        wintypes.DWORD,
        ctypes.POINTER(Privilege),
        ctypes.POINTER(wintypes.DWORD),
    ]

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), adjust | query, ctypes.byref(token)
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    def switch(name: str, attributes: int) -> int | None:
        """Set the privilege `name` to `attributes`. Returns what it was
        set to before, or None when the token doesn't hold it."""
        wanted, before, size = (
            Privilege(count=1, attributes=attributes),
            Privilege(),
            wintypes.DWORD(),
        )
        if not advapi32.LookupPrivilegeValueW(None, name, ctypes.byref(wanted.luid)):
            raise ctypes.WinError(ctypes.get_last_error())
        ctypes.set_last_error(0)
        if not advapi32.AdjustTokenPrivileges(
            token, False, ctypes.byref(wanted), ctypes.sizeof(before), ctypes.byref(before), size
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == not_all_assigned:
            return None
        # Nothing comes back for a privilege that was that way already.
        return before.attributes if before.count else attributes

    was: dict[str, int] = {}
    try:
        for name in _BACKUP_RIGHTS:
            before = switch(name, enabled if on else 0)
            if before is None:
                if on:
                    pytest.skip(f"this account doesn't hold {name}")
                continue
            was[name] = before
        yield
    finally:
        for name, before in was.items():
            switch(name, before & enabled)
        kernel32.CloseHandle(token)


@windows_only
def test_on_windows_a_folder_the_account_cannot_list_is_not_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another account's profile, say. `os.access` calls it readable and
    writable; opening it is refused."""
    locked = tmp_path / "locked"
    locked.mkdir()
    (tmp_path / "fine").mkdir()
    account = _this_account()
    # RD: list the folder. For this folder only, not what is made in it.
    _icacls(locked, "/deny", f"{account}:(RD)")
    try:
        with _backup_rights(on=False):
            assert os.access(locked, os.R_OK | os.W_OK | os.X_OK) is True
            assert fsops.can_open_folder(locked) is False

            browser = _browser(tmp_path, monkeypatch)
            roots = browser.list_directories(None, None)
            _home, at_home = _open(browser, roots.browse_id, roots, tmp_path.name)

            assert _names(at_home) == ["fine"]
    finally:
        _icacls(locked, "/remove:d", account)


@windows_only
def test_on_windows_a_folder_the_account_cannot_write_to_opens_but_cannot_be_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_only = tmp_path / "read-only"
    (read_only / "mine").mkdir(parents=True)
    account = _this_account()
    # WD: add a file to the folder. Not inherited, so `mine` keeps its own.
    _icacls(read_only, "/deny", f"{account}:(WD)")
    try:
        with _backup_rights(on=False):
            assert os.access(read_only, os.W_OK) is True
            assert fsops.can_open_folder(read_only) is True
            assert fsops.can_write_folder(read_only) is False
            assert fsops.can_write_folder(read_only / "mine") is True
            # Asking left nothing behind.
            assert [entry.name for entry in read_only.iterdir()] == ["mine"]

            browser = _browser(tmp_path, monkeypatch)
            roots = browser.list_directories(None, None)
            _home, at_home = _open(browser, roots.browse_id, roots, tmp_path.name)
            folder, inside = _open(browser, roots.browse_id, at_home, "read-only")
            assert _names(inside) == ["mine"]
            [mine] = inside.entries

            with pytest.raises(DirectoryBrowseError):
                browser.resolve(roots.browse_id, folder.directory_ref)
            with pytest.raises(RootGrantError, match="this account can't write to it"):
                session_grants.canonical_directory(read_only)
            assert browser.resolve(roots.browse_id, mine.directory_ref) == read_only / "mine"
    finally:
        _icacls(read_only, "/remove:d", account)


@windows_only
def test_on_windows_a_process_with_its_backup_rights_on_is_answered_as_windows_answers_it(
    tmp_path: Path,
) -> None:
    """An administrator's process started from Git Bash has its backup and
    restore privileges switched on, and Windows then lets it open a folder
    for listing or for adding a file whatever the access list says. The
    checks ask Windows, so there they say yes. A file made the ordinary way
    doesn't use those privileges, and is still refused."""
    locked = tmp_path / "locked"
    locked.mkdir()
    account = _this_account()
    _icacls(locked, "/deny", f"{account}:(RD,WD)")
    try:
        with _backup_rights(on=False):
            assert fsops.can_open_folder(locked) is False
            assert fsops.can_write_folder(locked) is False
        with _backup_rights(on=True):
            assert fsops.can_open_folder(locked) is True
            assert fsops.can_write_folder(locked) is True
            with pytest.raises(PermissionError):
                (locked / "made.txt").write_text("", encoding="utf-8")
        # And off again once the block is over.
        with _backup_rights(on=False):
            assert fsops.can_open_folder(locked) is False
    finally:
        _icacls(locked, "/remove:d", account)


# ── A folder whose check never returns ───────────────────────────────────────


async def test_a_folder_whose_access_check_never_returns_holds_the_listing_no_longer_than_its_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The look at whether a listed folder can be opened is a look at the
    disk like the others: on a share that stopped answering it doesn't
    return. The listing is answered within the control connection's bound
    for a folder that doesn't answer, as it is for the rest of its looks."""
    (tmp_path / "on-a-dead-share").mkdir()
    (tmp_path / "fine").mkdir()
    released = threading.Event()
    real = fsops.can_open_folder

    def never_for_the_dead_share(path: Any) -> bool:
        if Path(path).name == "on-a-dead-share":
            released.wait(60)
        return real(path)

    monkeypatch.setattr(fsops, "can_open_folder", never_for_the_dead_share)
    monkeypatch.setattr(daemon_module, "FOLDER_CHECK_SECONDS", 0.5)
    monkeypatch.setattr("autogpt_local_executor.directory_browser._windows_drive_roots", lambda: [])
    config = ShimConfig(
        allowed_root=tmp_path / "unused",
        audit_log_path=tmp_path / "_audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
    )
    daemon = ShimDaemon(
        config,
        token_store=object(),
        audit=AuditWriter(path=config.audit_log_path, audit_key=b"a" * 32),
    )
    daemon._directory_browser = DirectoryBrowser(platform_name=_platform(), home=tmp_path)
    monkeypatch.setattr(daemon._directory_browser, "_children_of", lambda _parent: [])
    daemon._granted_capabilities = frozenset({"directory_browse"})

    def ask(browse_id: str | None, directory_ref: str | None) -> DirectoryListRequestMessage:
        return DirectoryListRequestMessage(
            id=new_id(),
            ts=now_ts(),
            payload=DirectoryListRequestPayload(browse_id=browse_id, directory_ref=directory_ref),
        )

    try:
        roots = await daemon._handle(ask(None, None))
        assert isinstance(roots, DirectoryListResponseMessage)
        [home] = roots.payload.entries

        answer = await daemon._handle(ask(roots.payload.browse_id, home.directory_ref))

        assert isinstance(answer, ErrorMessage)
        assert answer.payload.code == ErrorCode.DIRECTORY_UNAVAILABLE
        assert answer.payload.details == {"reason": "not_answering"}
    finally:
        released.set()
