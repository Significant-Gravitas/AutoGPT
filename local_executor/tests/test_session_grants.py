from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

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
