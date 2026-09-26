import os
import sys
import time
from pathlib import Path

import pytest

from backend.util.link_checkout import runtime
from backend.util.link_checkout.checkout_record import (
    assert_observable,
    mark_sensitive,
    unseal,
)


def test_sensitive_session_blocks_observation_after_process_restart(tmp_path):
    mark_sensitive(tmp_path)
    with pytest.raises(RuntimeError):
        assert_observable(tmp_path)


def test_unsealed_session_can_be_observed_again(tmp_path):
    mark_sensitive(tmp_path)
    unseal(tmp_path)
    assert_observable(tmp_path)


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="uses flock")
async def test_idle_browser_profiles_are_swept_but_checkouts_are_left_alone(
    tmp_path,
):
    now = time.time()

    def chat(name: str, idle_for: float, *markers: str) -> Path:
        directory = tmp_path / name
        (directory / "engine" / "profile").mkdir(parents=True)
        lock = directory / "operation.lock"
        lock.touch()
        os.utime(lock, (now - idle_for, now - idle_for))
        for marker in markers:
            (directory / marker).touch()
        return directory

    idle = chat("idle", 2 * 60 * 60)
    recent = chat("recent", 60)
    paying = chat("paying", 2 * 60 * 60, "checkout.json")
    sealed = chat("sealed", 2 * 60 * 60, "sensitive")

    await runtime.sweep_idle_browsers(tmp_path, now)

    assert not (idle / "engine").exists()
    assert all(
        (directory / "engine").exists() for directory in (recent, paying, sealed)
    )


@pytest.mark.asyncio
async def test_a_browser_whose_directory_fails_counts_as_not_retired(monkeypatch):
    """The caller keeps the chat sealed on False; an exception here used to
    skip that and leave the attempt without a receipt."""

    def unsafe(key: str) -> Path:
        raise RuntimeError("Private browser directory permissions are unsafe")

    monkeypatch.setattr(runtime, "session_home", unsafe)

    assert await runtime.retire_payment_browser("chat") is False
