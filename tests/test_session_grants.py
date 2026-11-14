from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

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
    could no longer use this computer. The folder's inode number, creation
    time and path still tell one put in its place apart. Windows' st_dev is
    the volume's serial number, kept from when it was formatted, and still
    counts there."""
    root = tmp_path / "workspace"
    root.mkdir()
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
