from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autogpt_local_executor import session_grants
from autogpt_local_executor.session_grants import (
    RootGrantError,
    RootGrantSigner,
    directory_birth,
    directory_fingerprint,
)


def test_root_grant_round_trip_and_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    signer = RootGrantSigner(b"a" * 32, "machine-1", now=lambda: 123.0)

    issued = signer.issue("session-1", root, 3)
    restored = RootGrantSigner(b"a" * 32, "machine-1").verify(issued.root_grant, "session-1")

    assert restored == issued
    assert restored.allowed_root == root.resolve()
    assert len(restored.fingerprint) == 64
    assert restored.revision == 3


def test_root_grant_rejects_tampering_and_wrong_context(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    signer = RootGrantSigner(b"a" * 32, "machine-1")
    grant = signer.issue("session-1", root, 1).root_grant
    tampered = grant[:-1] + ("A" if grant[-1] != "A" else "B")

    with pytest.raises(RootGrantError, match="signature"):
        signer.verify(tampered, "session-1")
    with pytest.raises(RootGrantError, match="another session"):
        signer.verify(grant, "session-2")
    with pytest.raises(RootGrantError, match="another machine"):
        RootGrantSigner(b"a" * 32, "machine-2").verify(grant, "session-1")


def test_root_grant_detects_directory_replacement(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    if directory_birth(root) is None:
        pytest.skip("this filesystem reports no directory creation time")
    signer = RootGrantSigner(b"a" * 32, "machine-1")
    grant = signer.issue("session-1", root, 1).root_grant
    root.rmdir()
    root.mkdir()

    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(grant, "session-1")


def test_root_grant_without_birth_claim_is_malformed(tmp_path: Path) -> None:
    """Every grant carries the creation-time claim, None where the platform
    keeps no creation time; one without it was never issued here."""
    root = tmp_path / "workspace"
    root.mkdir()
    key = b"a" * 32
    signer = RootGrantSigner(key, "machine-1")
    claims = {
        "allowed_root": str(root.resolve()),
        "fingerprint": directory_fingerprint(root),
        "issued_at": 1.0,
        "machine_id": "machine-1",
        "revision": 1,
        "session_id": "session-1",
    }
    encoded = (
        base64.urlsafe_b64encode(json.dumps(claims, sort_keys=True, separators=(",", ":")).encode())
        .rstrip(b"=")
        .decode()
    )
    grant_key = hmac.new(key, b"autogpt-root-grant-v1", hashlib.sha256).digest()
    signature = (
        base64.urlsafe_b64encode(hmac.new(grant_key, encoded.encode(), hashlib.sha256).digest())
        .rstrip(b"=")
        .decode()
    )

    with pytest.raises(RootGrantError, match="malformed"):
        signer.verify(f"v1.{encoded}.{signature}", "session-1")


def test_root_grant_rejects_symlink_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)

    with pytest.raises(RootGrantError, match="symbolic link"):
        RootGrantSigner(b"a" * 32, "machine-1").issue("session-1", link, 1)


class _Renumbered:
    """A stat result with another device number, the rest unchanged."""

    def __init__(self, result: os.stat_result) -> None:
        self._result = result

    @property
    def st_dev(self) -> int:
        return self._result.st_dev + 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)


def _renumber_the_device(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """From now on `root` reports another device number, its inode number
    and creation time unchanged, as a POSIX system numbers a device each time
    it is attached or mounted: a reboot can number a btrfs subvolume or an
    LVM volume afresh, and a share or a FUSE folder mounted again has a new
    number."""
    canonical = Path(os.path.realpath(root))
    stat = Path.stat

    def renumbered(self: Path, *, follow_symlinks: bool = True) -> Any:
        result = stat(self, follow_symlinks=follow_symlinks)
        return _Renumbered(result) if self == canonical else result

    monkeypatch.setattr(Path, "stat", renumbered)


def _creation_time(monkeypatch: pytest.MonkeyPatch, birth: str | None) -> None:
    """From now on every folder's creation time is `birth`: None as on a
    filesystem that reports none (an NFS share on Linux 6.12 or earlier,
    say)."""
    monkeypatch.setattr(session_grants, "directory_birth", lambda path: birth)


def _off_btrfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """From now on statfs says no folder is on Btrfs, as it does of ext4 or
    tmpfs, whatever the filesystem this test's folder is on."""
    monkeypatch.setattr(session_grants, "_on_btrfs", lambda _root: False)


def _mounts(monkeypatch: pytest.MonkeyPatch, table: Path, root: Path, source: str | None) -> None:
    """From now on the shim's mount table (/proc/self/mountinfo) has the
    folder that holds `root` mounted over NFS from `source`, on the device
    `root` reports, and the system's root folder on another device. With no
    `source`, no mount of that device."""
    number = Path(os.path.realpath(root)).stat().st_dev
    point = str(Path(os.path.realpath(root)).parent)
    for char, escape in (("\\", "\\134"), (" ", "\\040"), ("\t", "\\011"), ("\n", "\\012")):
        point = point.replace(char, escape)
    lines = [
        f"22 1 {os.major(number + 1)}:{os.minor(number + 1)} / / rw shared:1 - ext4 /dev/sda2 rw"
    ]
    if source is not None:
        lines.append(
            f"98 22 {os.major(number)}:{os.minor(number)} /export {point} rw,relatime "
            f"shared:60 - nfs4 {source} rw,vers=4.2"
        )
    # The kernel's table holds a path's bytes, as os.fsencode gives them.
    table.write_bytes(os.fsencode("\n".join(lines) + "\n"))
    monkeypatch.setattr(session_grants, "_MOUNTINFO", str(table), raising=False)


def _no_mount_table(monkeypatch: pytest.MonkeyPatch, missing: Path) -> None:
    """From now on the shim can't read its mount table (no /proc, say)."""
    monkeypatch.setattr(session_grants, "_MOUNTINFO", str(missing), raising=False)


_linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="on Linux a grant names its folder's mount"
)


def _grant(key: bytes, claims: dict[str, Any]) -> str:
    """A grant of `claims`, signed as RootGrantSigner signs one."""
    encoded = (
        base64.urlsafe_b64encode(json.dumps(claims, sort_keys=True, separators=(",", ":")).encode())
        .rstrip(b"=")
        .decode()
    )
    grant_key = hmac.new(key, b"autogpt-root-grant-v1", hashlib.sha256).digest()
    signature = (
        base64.urlsafe_b64encode(hmac.new(grant_key, encoded.encode(), hashlib.sha256).digest())
        .rstrip(b"=")
        .decode()
    )
    return f"v1.{encoded}.{signature}"


def test_a_grant_restores_once_its_folders_device_is_numbered_afresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fingerprint held st_dev, which a POSIX system gives a device when
    it is attached or mounted, so once a reboot or a new mount numbered the
    folder's device afresh, its grant never restored again, and its chat
    could no longer use this computer. Where the folder has a creation time,
    its inode number, creation time and path still tell one put in its place
    apart. Windows' st_dev is the volume's serial number, kept from when it
    was formatted, and still counts there.

    The folder is on a filesystem statfs knows isn't Btrfs. Where it can't
    tell, or on Btrfs with no subvolume to read, the device number counts,
    and the tests of those keep to it."""
    root = tmp_path / "workspace"
    root.mkdir()
    _creation_time(monkeypatch, "ns:1700000000123456789")
    _off_btrfs(monkeypatch)
    signer = RootGrantSigner(b"a" * 32, "machine-1")
    issued = signer.issue("session-1", root, 1)
    _renumber_the_device(monkeypatch, root)

    if sys.platform == "win32":
        with pytest.raises(RootGrantError, match="changed"):
            signer.verify(issued.root_grant, "session-1")
    else:
        assert signer.verify(issued.root_grant, "session-1") == issued


def test_a_grant_fingerprinted_with_its_device_number_restores_while_that_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grant issued while the fingerprint held the device number still
    restores, until the number changes, as it always did."""
    root = tmp_path / "workspace"
    root.mkdir()
    canonical = Path(os.path.realpath(root))
    stat = canonical.stat(follow_symlinks=False)
    identity = {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "path": os.path.normcase(str(canonical)),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    key = b"a" * 32
    grant = _grant(
        key,
        {
            "allowed_root": str(canonical),
            "birth": directory_birth(canonical),
            "fingerprint": fingerprint,
            "issued_at": 1.0,
            "machine_id": "machine-1",
            "revision": 1,
            "session_id": "session-1",
        },
    )
    signer = RootGrantSigner(key, "machine-1")

    assert signer.verify(grant, "session-1").fingerprint == fingerprint
    _renumber_the_device(monkeypatch, root)
    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(grant, "session-1")


def test_a_grant_for_a_folder_with_no_creation_time_keeps_to_its_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the filesystem reports no creation time, the path and inode
    number can't tell the selected folder from another filesystem mounted at
    its path later: the root folders of two filesystems of one type usually
    share an inode number (2 on ext4). Without the device number in its
    fingerprint, the folder's grant restored on a filesystem the person
    never chose. Where the folder's mount can't be found (or off Linux), it
    restores while the device number holds."""
    root = tmp_path / "workspace"
    root.mkdir()
    _creation_time(monkeypatch, None)
    _no_mount_table(monkeypatch, tmp_path / "no-mountinfo")
    signer = RootGrantSigner(b"a" * 32, "machine-1")
    issued = signer.issue("session-1", root, 1)

    assert signer.verify(issued.root_grant, "session-1") == issued
    # The same path and inode number on another device, as another
    # filesystem mounted at the path shows them.
    _renumber_the_device(monkeypatch, root)
    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(issued.root_grant, "session-1")


def test_a_grant_with_neither_a_device_number_nor_a_creation_time_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grant issued while a POSIX fingerprint held no device number names a
    folder with no creation time by its path and inode number alone, which
    another filesystem mounted at that path matches too, so it no longer
    restores, even unchanged: the person selects the folder again."""
    root = tmp_path / "workspace"
    root.mkdir()
    _creation_time(monkeypatch, None)
    canonical = Path(os.path.realpath(root))
    stat = canonical.stat(follow_symlinks=False)
    identity = {"inode": stat.st_ino, "path": os.path.normcase(str(canonical))}
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    key = b"a" * 32
    grant = _grant(
        key,
        {
            "allowed_root": str(canonical),
            "birth": None,
            "fingerprint": fingerprint,
            "issued_at": 1.0,
            "machine_id": "machine-1",
            "revision": 1,
            "session_id": "session-1",
        },
    )

    with pytest.raises(RootGrantError, match="changed"):
        RootGrantSigner(key, "machine-1").verify(grant, "session-1")


@_linux_only
def test_a_grant_for_a_folder_with_no_creation_time_keeps_to_its_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linux numbers a share or a FUSE folder from a pool, its number freed
    when it is unmounted, so another filesystem mounted at the folder's path
    later can be given that number, and its root folder the same inode
    number. With no creation time to tell them apart, the folder's grant
    restored on that filesystem, one the person never chose. The grant names
    the folder's mount now, and the same share mounted again restores."""
    root = tmp_path / "shared folder" / "workspace"
    root.mkdir(parents=True)
    _creation_time(monkeypatch, None)
    table = tmp_path / "mountinfo"
    _mounts(monkeypatch, table, root, "nas.local:/export/projects")
    signer = RootGrantSigner(b"a" * 32, "machine-1")
    issued = signer.issue("session-1", root, 1)

    assert signer.verify(issued.root_grant, "session-1") == issued
    # Another share mounted at the path, given the freed device number.
    _mounts(monkeypatch, table, root, "nas.local:/export/scratch")
    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(issued.root_grant, "session-1")
    # The folder's own share, mounted again with that number.
    _mounts(monkeypatch, table, root, "nas.local:/export/projects")
    assert signer.verify(issued.root_grant, "session-1") == issued


@_linux_only
def test_a_grant_naming_its_mount_is_refused_where_the_mount_cant_be_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grant that names its folder's mount isn't checked on the device
    number alone, which a reused number matches, once the mount can't be
    found: it is refused."""
    root = tmp_path / "workspace"
    root.mkdir()
    _creation_time(monkeypatch, None)
    table = tmp_path / "mountinfo"
    _mounts(monkeypatch, table, root, "nas.local:/export/projects")
    signer = RootGrantSigner(b"a" * 32, "machine-1")
    issued = signer.issue("session-1", root, 1)

    _mounts(monkeypatch, table, root, None)
    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(issued.root_grant, "session-1")
    _no_mount_table(monkeypatch, tmp_path / "no-mountinfo")
    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(issued.root_grant, "session-1")


@_linux_only
def test_a_grant_from_before_mounts_were_named_restores_while_its_device_number_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grant for a folder with no creation time issued before grants named
    the folder's mount holds its device number, inode number and path. It
    still restores while that number holds: a chat holding one can't select
    its folder again."""
    root = tmp_path / "workspace"
    root.mkdir()
    _creation_time(monkeypatch, None)
    _mounts(monkeypatch, tmp_path / "mountinfo", root, "nas.local:/export/projects")
    canonical = Path(os.path.realpath(root))
    stat = canonical.stat(follow_symlinks=False)
    identity = {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "path": os.path.normcase(str(canonical)),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    key = b"a" * 32
    grant = _grant(
        key,
        {
            "allowed_root": str(canonical),
            "birth": None,
            "fingerprint": fingerprint,
            "issued_at": 1.0,
            "machine_id": "machine-1",
            "revision": 1,
            "session_id": "session-1",
        },
    )
    signer = RootGrantSigner(key, "machine-1")

    assert signer.verify(grant, "session-1").fingerprint == fingerprint
    _renumber_the_device(monkeypatch, root)
    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(grant, "session-1")


_btrfs_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Btrfs is Linux's")


def _in_btrfs_subvolume(monkeypatch: pytest.MonkeyPatch, subvolume: str | None) -> None:
    """From now on every folder is on Btrfs, in the subvolume with the UUID
    `subvolume`: None for one whose subvolume can't be read (Linux before
    4.18)."""
    monkeypatch.setattr(session_grants, "_on_btrfs", lambda _root: True, raising=False)
    monkeypatch.setattr(session_grants, "_btrfs_subvolume", lambda _root: subvolume, raising=False)


@_btrfs_only
def test_a_grant_on_btrfs_is_refused_once_a_snapshot_takes_its_folders_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Btrfs snapshot keeps its subvolume's inode numbers and creation
    times, and only its device number differs, which the fingerprint of a
    folder with a creation time leaves out. So the subvolume's owner could
    snapshot it, put the snapshot in the folder's place, and the grant
    restored on the copy. The grant holds the subvolume's UUID now, and
    still restores once the folder's own device is numbered afresh."""
    root = tmp_path / "workspace"
    root.mkdir()
    _creation_time(monkeypatch, "ns:1700000000123456789")
    _in_btrfs_subvolume(monkeypatch, "5d1bc3a8a3524f2b9d0a41b1ab3c1d01")
    signer = RootGrantSigner(b"a" * 32, "machine-1")
    issued = signer.issue("session-1", root, 1)
    _renumber_the_device(monkeypatch, root)

    assert signer.verify(issued.root_grant, "session-1") == issued
    # A snapshot of the subvolume, in the folder's place.
    _in_btrfs_subvolume(monkeypatch, "93f0d1e3b7f04e7c8b5a7d6c2e1f0a99")
    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(issued.root_grant, "session-1")


@_btrfs_only
def test_a_grant_on_btrfs_whose_subvolume_cant_be_read_keeps_to_its_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where a Btrfs folder's subvolume can't be read, nothing else tells a
    snapshot from it but the device number, so that counts, as it does for
    a folder with no creation time: the grant restores while it holds."""
    root = tmp_path / "workspace"
    root.mkdir()
    _creation_time(monkeypatch, "ns:1700000000123456789")
    _in_btrfs_subvolume(monkeypatch, None)
    signer = RootGrantSigner(b"a" * 32, "machine-1")
    issued = signer.issue("session-1", root, 1)

    assert signer.verify(issued.root_grant, "session-1") == issued
    _renumber_the_device(monkeypatch, root)
    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(issued.root_grant, "session-1")


def _statfs_fails(monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    """From now on statfs(2) tells no folder's filesystem type: the call
    `fails`, the C library has no statfs (`missing`), or the C library is
    `unloadable`."""

    def statfs(_path: bytes, _result: Any) -> int:
        return -1

    def library(*_args: Any, **_kwargs: Any) -> Any:
        if how == "unloadable":
            raise OSError("the C library can't be loaded")
        return SimpleNamespace(statfs=statfs) if how == "fails" else SimpleNamespace()

    monkeypatch.setattr(session_grants.ctypes, "CDLL", library)


@_btrfs_only
@pytest.mark.parametrize("how", ["fails", "missing", "unloadable"])
def test_a_grant_where_the_filesystem_type_cant_be_read_keeps_to_its_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """Where statfs can't tell a folder's filesystem type, the folder may be
    on Btrfs, where a snapshot keeps its inode number and creation time. It
    was taken for another filesystem, so its grant held neither the device
    number nor the subvolume, and a snapshot put in its place matched it.
    The device number counts there, as where the subvolume can't be read."""
    root = tmp_path / "workspace"
    root.mkdir()
    _creation_time(monkeypatch, "ns:1700000000123456789")
    _statfs_fails(monkeypatch, how)
    signer = RootGrantSigner(b"a" * 32, "machine-1")
    issued = signer.issue("session-1", root, 1)

    assert signer.verify(issued.root_grant, "session-1") == issued
    # A snapshot in the folder's place: the same inode number and creation
    # time, and a device number of its own.
    _renumber_the_device(monkeypatch, root)
    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(issued.root_grant, "session-1")
    assert session_grants._on_btrfs(root) is None


def _fingerprint_of(canonical: Path, **identity: Any) -> str:
    stat = canonical.stat(follow_symlinks=False)
    held = {"inode": stat.st_ino, "path": os.path.normcase(str(canonical)), **identity}
    return hashlib.sha256(
        json.dumps(held, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _grant_for(key: bytes, canonical: Path, fingerprint: str, birth: str | None) -> str:
    return _grant(
        key,
        {
            "allowed_root": str(canonical),
            "birth": birth,
            "fingerprint": fingerprint,
            "issued_at": 1.0,
            "machine_id": "machine-1",
            "revision": 1,
            "session_id": "session-1",
        },
    )


@_btrfs_only
def test_a_grant_on_btrfs_with_neither_its_device_nor_its_subvolume_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grant issued for a Btrfs folder while the fingerprint of a folder
    with a creation time held neither its device number nor its subvolume is
    what a snapshot put in the folder's place matched, so it no longer
    restores, even unchanged."""
    root = tmp_path / "workspace"
    root.mkdir()
    birth = "ns:1700000000123456789"
    _creation_time(monkeypatch, birth)
    _in_btrfs_subvolume(monkeypatch, "5d1bc3a8a3524f2b9d0a41b1ab3c1d01")
    canonical = Path(os.path.realpath(root))
    key = b"a" * 32
    grant = _grant_for(key, canonical, _fingerprint_of(canonical), birth)

    with pytest.raises(RootGrantError, match="changed"):
        RootGrantSigner(key, "machine-1").verify(grant, "session-1")


@_btrfs_only
def test_a_grant_on_btrfs_with_its_device_number_restores_while_that_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grant issued while every fingerprint held the device number (as
    installs from the platform's setup card issue them) still restores on
    Btrfs while that number holds: a chat holding one can't select its
    folder again."""
    root = tmp_path / "workspace"
    root.mkdir()
    birth = "ns:1700000000123456789"
    _creation_time(monkeypatch, birth)
    _in_btrfs_subvolume(monkeypatch, "5d1bc3a8a3524f2b9d0a41b1ab3c1d01")
    canonical = Path(os.path.realpath(root))
    fingerprint = _fingerprint_of(canonical, device=canonical.stat().st_dev)
    key = b"a" * 32
    grant = _grant_for(key, canonical, fingerprint, birth)
    signer = RootGrantSigner(key, "machine-1")

    assert signer.verify(grant, "session-1").fingerprint == fingerprint
    _renumber_the_device(monkeypatch, root)
    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(grant, "session-1")


@_btrfs_only
def test_the_btrfs_check_reads_the_filesystem_type_statfs_reports(tmp_path: Path) -> None:
    """_on_btrfs calls statfs(2) through ctypes, and coreutils' stat -f
    reports the same call's type; off Btrfs no subvolume is read."""
    stat_command = shutil.which("stat")
    if stat_command is None:
        pytest.skip("needs coreutils' stat")
    reported = subprocess.run(
        [stat_command, "-f", "-c", "%t", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    on_btrfs = int(reported, 16) == 0x9123683E

    assert session_grants._on_btrfs(tmp_path) is on_btrfs
    subvolume = session_grants._btrfs_subvolume(tmp_path)
    if on_btrfs:
        assert subvolume is not None and len(subvolume) == 32
    else:
        assert subvolume is None


@_btrfs_only
def test_a_real_btrfs_snapshot_in_a_folders_place_is_refused(tmp_path: Path) -> None:
    """On a real Btrfs filesystem (skipped elsewhere): a subvolume's own user
    snapshots it and puts the snapshot in its place, which keeps its inode
    number and creation time."""
    btrfs = shutil.which("btrfs")
    if btrfs is None or not session_grants._on_btrfs(tmp_path):
        pytest.skip("needs a temporary folder on Btrfs and the btrfs command")
    folder = tmp_path / "workspace"
    subprocess.run([btrfs, "-q", "subvolume", "create", str(folder)], check=True)
    signer = RootGrantSigner(b"a" * 32, "machine-1")
    issued = signer.issue("session-1", folder, 1)
    assert signer.verify(issued.root_grant, "session-1") == issued

    moved = folder.rename(tmp_path / "workspace.orig")
    subprocess.run([btrfs, "-q", "subvolume", "snapshot", str(moved), str(folder)], check=True)
    assert folder.stat().st_ino == moved.stat().st_ino
    assert directory_birth(folder) == directory_birth(moved)

    with pytest.raises(RootGrantError, match="changed"):
        signer.verify(issued.root_grant, "session-1")
