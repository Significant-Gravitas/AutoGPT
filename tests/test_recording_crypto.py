"""
Tests for the recording key's passphrase fallback file (recording/crypto.py).

The key file must be private to this user from the moment it exists: other
local accounts must never be able to read it, not even briefly, and a failure
to keep it private must not be silently ignored.
"""

from __future__ import annotations

import os
import stat
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from autogpt_local_executor.recording import crypto

posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")


class _OsProxy(types.ModuleType):
    """The real `os` module with some functions replaced, for one module under test."""

    def __init__(self, **overrides: object) -> None:
        super().__init__("os")
        self.__dict__.update(overrides)

    def __getattr__(self, name: str) -> object:
        return getattr(os, name)


def _refuse_chmod(*_args: object, **_kwargs: object) -> None:
    raise PermissionError("chmod refused")


@pytest.fixture
def fallback_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "fallback"
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(directory))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "test-passphrase")
    return directory


@pytest.fixture
def permissive_umask() -> Iterator[None]:
    """A umask that would leave a new file readable by other accounts (0o644)."""
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_fallback_key_round_trips_and_leaves_no_temp_files(fallback_dir: Path) -> None:
    key = crypto._fallback_recording_key()

    assert len(key) == 32
    assert crypto._fallback_recording_key() == key
    assert [p.name for p in fallback_dir.iterdir()] == ["recording-key.enc"]


@posix_only
def test_new_fallback_key_file_is_owner_only_without_chmod(
    fallback_dir: Path, monkeypatch: pytest.MonkeyPatch, permissive_umask: None
) -> None:
    monkeypatch.setattr(crypto, "os", _OsProxy(chmod=_refuse_chmod))

    key = crypto._fallback_recording_key()

    assert _mode(fallback_dir / "recording-key.enc") == 0o600
    assert crypto._fallback_recording_key() == key


@posix_only
def test_regenerated_fallback_key_replaces_a_readable_file_with_an_owner_only_one(
    fallback_dir: Path, monkeypatch: pytest.MonkeyPatch, permissive_umask: None
) -> None:
    fallback_dir.mkdir()
    key_path = fallback_dir / "recording-key.enc"
    key_path.write_text("not-a-key-envelope", encoding="utf-8")
    key_path.chmod(0o644)
    monkeypatch.setattr(crypto, "os", _OsProxy(chmod=_refuse_chmod))

    key = crypto._fallback_recording_key()

    assert _mode(key_path) == 0o600
    assert crypto._fallback_recording_key() == key
    assert [p.name for p in fallback_dir.iterdir()] == ["recording-key.enc"]


@posix_only
def test_fallback_key_is_never_written_where_the_filesystem_ignores_the_mode(
    fallback_dir: Path, monkeypatch: pytest.MonkeyPatch, permissive_umask: None
) -> None:
    def open_ignoring_mode(path: object, flags: int, mode: int = 0o777) -> int:
        return os.open(path, flags, 0o644)  # type: ignore[arg-type]

    monkeypatch.setattr(crypto, "os", _OsProxy(open=open_ignoring_mode))

    with pytest.raises(RuntimeError, match="private"):
        crypto._fallback_recording_key()

    assert list(fallback_dir.iterdir()) == []
