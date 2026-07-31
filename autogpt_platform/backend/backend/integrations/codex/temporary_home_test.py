import asyncio
import os
import shutil
import time
from pathlib import Path

import pytest

from backend.integrations.codex import temporary_home
from backend.integrations.codex.temporary_home import (
    TemporaryCodexHome,
    TemporaryCodexHomeCleanupError,
)


def test_temporary_home_creates_private_workspace_and_cleans_up(tmp_path):
    with TemporaryCodexHome.create(tmp_path) as home:
        path = home.path
        assert home.root == tmp_path.resolve()
        assert path.parent == tmp_path.resolve()
        assert home.auth_path == path / "auth.json"
        assert home.workspace_path.is_dir()
        if os.name != "nt":
            assert path.stat().st_mode & 0o777 == 0o700
            assert home.workspace_path.stat().st_mode & 0o777 == 0o700

    assert not path.exists()


def test_temporary_home_cleans_up_after_failure(tmp_path):
    with pytest.raises(RuntimeError, match="expected"):
        with TemporaryCodexHome.create(tmp_path) as home:
            path = home.path
            raise RuntimeError("expected")

    assert not path.exists()


def test_temporary_home_cleans_up_if_initialization_fails(tmp_path, monkeypatch):
    mkdir = Path.mkdir

    def fail_workspace(path: Path, *args, **kwargs) -> None:
        if path.name == "workspace":
            raise PermissionError("workspace creation failed")
        mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_workspace)

    with pytest.raises(PermissionError, match="workspace creation failed"):
        TemporaryCodexHome.create(tmp_path)

    assert not list(tmp_path.glob("autogpt-codex-*"))


@pytest.mark.asyncio
async def test_temporary_home_cleans_up_after_cancellation(tmp_path):
    entered = asyncio.Event()

    async def wait_forever() -> None:
        async with TemporaryCodexHome.create(tmp_path) as home:
            nonlocal path
            path = home.path
            entered.set()
            await asyncio.Event().wait()

    path = tmp_path
    task = asyncio.create_task(wait_forever())
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not path.exists()


def test_cleanup_is_idempotent(tmp_path):
    home = TemporaryCodexHome.create(tmp_path)
    path = home.path

    home.cleanup()
    home.cleanup()

    assert not path.exists()


def test_cleanup_retries_transient_filesystem_locks(tmp_path, monkeypatch):
    home = TemporaryCodexHome.create(tmp_path)
    path = home.path
    attempts = 0
    remove_tree_once = temporary_home._remove_tree_once

    def fail_twice(target: Path, root: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient lock")
        remove_tree_once(target, root)

    monkeypatch.setattr(temporary_home, "_remove_tree_once", fail_twice)
    monkeypatch.setattr(temporary_home.time, "sleep", lambda _seconds: None)

    home.cleanup()

    assert attempts == 3
    assert not path.exists()


def test_cleanup_quarantines_a_locked_home_under_the_dedicated_root(
    tmp_path, monkeypatch
):
    home = TemporaryCodexHome.create(tmp_path)
    home.auth_path.write_text("fake-sensitive-auth", encoding="utf-8")
    scheduled: list[tuple[Path, Path]] = []
    monkeypatch.setattr(temporary_home, "_remove_with_retry", lambda *_args: False)
    monkeypatch.setattr(
        temporary_home,
        "_schedule_reap",
        lambda root, target: scheduled.append((root, target)),
    )

    home.cleanup()

    assert not home.path.exists()
    assert len(scheduled) == 1
    root, quarantined = scheduled[0]
    assert root == tmp_path.resolve()
    assert quarantined.parent == root
    assert quarantined.name.startswith(".quarantine-autogpt-codex-")
    assert (quarantined / "auth.json").read_text(encoding="utf-8") == (
        "fake-sensitive-auth"
    )
    shutil.rmtree(quarantined)


def test_cleanup_fails_loudly_if_home_cannot_be_removed_or_quarantined(
    tmp_path, monkeypatch
):
    home = TemporaryCodexHome.create(tmp_path)
    scheduled: list[tuple[Path, Path]] = []
    monkeypatch.setattr(temporary_home, "_remove_with_retry", lambda *_args: False)
    monkeypatch.setattr(temporary_home, "_quarantine", lambda *_args: None)
    monkeypatch.setattr(
        temporary_home,
        "_schedule_reap",
        lambda root, target: scheduled.append((root, target)),
    )

    with pytest.raises(TemporaryCodexHomeCleanupError):
        home.cleanup()

    assert scheduled == [(tmp_path.resolve(), home.path)]
    assert home.path.exists()
    shutil.rmtree(home.path)


def test_temporary_home_rejects_paths_outside_the_dedicated_root(tmp_path):
    root = tmp_path / "managed"
    root.mkdir()
    outside = tmp_path / "autogpt-codex-outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="direct child"):
        TemporaryCodexHome(outside, root.resolve())


def test_temporary_home_rejects_a_filesystem_root():
    filesystem_root = Path(Path.cwd().anchor)

    with pytest.raises(ValueError, match="filesystem root"):
        temporary_home._prepare_root(filesystem_root)


def test_reaper_recovers_a_marked_quarantine_after_restart(tmp_path):
    root = temporary_home._prepare_root(tmp_path)
    quarantined = root / ".quarantine-autogpt-codex-stale"
    quarantined.mkdir(mode=0o700)
    (quarantined / "auth.json").write_text("fake-sensitive-auth", encoding="utf-8")
    marker = root / f".reap-{quarantined.name}"
    marker.touch(mode=0o600)

    temporary_home._schedule_existing_reaps(root)

    deadline = time.monotonic() + 2
    while (quarantined.exists() or marker.exists()) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not quarantined.exists()
    assert not marker.exists()
