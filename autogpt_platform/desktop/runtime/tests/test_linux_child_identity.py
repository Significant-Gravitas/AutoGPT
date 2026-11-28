import json
import signal
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest
from autogpt_desktop import process

BOOT_ID = "7f8cd886-54bf-4c08-98a0-d2758ca07e44"
IDENTITY = {"boot_id": BOOT_ID, "start_ticks": 1234}


@pytest.fixture
def linux_reaper(tmp_path, monkeypatch):
    child = Mock(pid=123)
    child.create_time.return_value = 100.0
    child.is_running.return_value = True
    monkeypatch.setattr(psutil, "Process", Mock(return_value=child))
    monkeypatch.setattr(psutil, "wait_procs", Mock(return_value=([], [])))
    monkeypatch.setattr(process.sys, "platform", "linux")
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(process.os, "getpgid", lambda pid: pid, raising=False)
    killpg = Mock()
    monkeypatch.setattr(process.os, "killpg", killpg, raising=False)
    identity = Mock(return_value=(BOOT_ID, 1234))
    monkeypatch.setattr(process, "_linux_process_identity", identity, raising=False)
    registry = process.ChildRegistry(tmp_path / "children.json")
    return registry, child, killpg, identity


@pytest.mark.parametrize(
    "identity,started,signals",
    [
        (IDENTITY, 99.0, True),
        (IDENTITY, 101.0, True),
        ({**IDENTITY, "start_ticks": 1235}, 100.0, False),
        ({**IDENTITY, "boot_id": "250d1495-8289-4e83-9100-b2baf4189869"}, 100.0, False),
    ],
)
def test_linux_reaper_uses_boot_and_ticks_not_wall_time(
    linux_reaper, identity, started, signals
):
    registry, _, killpg, _ = linux_reaper
    registry.path.write_text(
        json.dumps([{"pid": 123, "started": started, "linux_identity": identity}]),
        encoding="utf-8",
    )
    registry.reap_leftovers()
    if signals:
        killpg.assert_called_once_with(123, signal.SIGKILL)
    else:
        killpg.assert_not_called()
    assert not registry.path.exists()


def test_linux_legacy_drift_retains_registry_without_signaling(linux_reaper):
    registry, _, killpg, _ = linux_reaper
    saved = '[{"pid":123,"started":99.5}]'
    registry.path.write_text(saved, encoding="utf-8")
    with pytest.raises(RuntimeError, match="retained"):
        registry.reap_leftovers()
    killpg.assert_not_called()
    assert registry.path.read_text("utf-8") == saved


def test_linux_identity_is_rechecked_before_signal(linux_reaper):
    registry, child, killpg, _ = linux_reaper
    child.is_running.return_value = False
    registry.path.write_text(
        json.dumps([{"pid": 123, "started": 99.0, "linux_identity": IDENTITY}]),
        encoding="utf-8",
    )
    registry.reap_leftovers()
    killpg.assert_not_called()


@pytest.mark.parametrize(
    "identity",
    [
        None,
        {},
        {**IDENTITY, "start_ticks": True},
        {**IDENTITY, "start_ticks": -1},
        {**IDENTITY, "start_ticks": "1234"},
        {**IDENTITY, "boot_id": "invalid"},
    ],
)
def test_invalid_linux_identity_retains_registry_without_signaling(
    linux_reaper, identity
):
    registry, _, killpg, _ = linux_reaper
    saved = json.dumps([{"pid": 123, "started": 100.0, "linux_identity": identity}])
    registry.path.write_text(saved, encoding="utf-8")
    with pytest.raises(RuntimeError, match="retained"):
        registry.reap_leftovers()
    killpg.assert_not_called()
    assert registry.path.read_text("utf-8") == saved


@pytest.mark.parametrize(
    "failure", [PermissionError(), ValueError(), ProcessLookupError()]
)
def test_linux_identity_read_failure_never_signals(linux_reaper, failure):
    registry, _, killpg, identity = linux_reaper
    identity.side_effect = failure
    registry.path.write_text(
        json.dumps([{"pid": 123, "started": 100.0, "linux_identity": IDENTITY}]),
        encoding="utf-8",
    )
    if isinstance(failure, ProcessLookupError):
        registry.reap_leftovers()
        assert not registry.path.exists()
    else:
        with pytest.raises(RuntimeError, match="retained"):
            registry.reap_leftovers()
        assert registry.path.exists()
    killpg.assert_not_called()


@pytest.mark.parametrize("running", [True, False])
def test_record_linux_identity_rechecks_process_before_saving(linux_reaper, running):
    registry, child, _, identity = linux_reaper
    child.is_running.return_value = running
    managed = Mock(pid=123, exit_code=Mock(return_value=None))
    managed.name = "owned service"
    registry.record([managed])
    entries = json.loads(registry.path.read_text("utf-8"))
    if running:
        assert entries == [
            {
                "name": "owned service",
                "pid": 123,
                "started": 100.0,
                "linux_identity": IDENTITY,
            }
        ]
    else:
        assert entries == []
    identity.assert_called_once_with(123)


@pytest.mark.parametrize(
    "failure", [PermissionError(), ValueError(), ProcessLookupError()]
)
def test_record_linux_identity_failure_is_not_a_legacy_record(linux_reaper, failure):
    registry, _, _, identity = linux_reaper
    identity.side_effect = failure
    managed = Mock(pid=123, exit_code=Mock(return_value=None))
    managed.name = "owned service"
    registry.record([managed])
    entries = json.loads(registry.path.read_text("utf-8"))
    if isinstance(failure, ProcessLookupError):
        assert entries == []
    else:
        assert entries[0]["linux_identity"] is None


@pytest.mark.parametrize(
    "comm", [b"python", b"odd ) (name)\xff\n", b"name with spaces"]
)
def test_linux_identity_reads_raw_start_ticks_after_parenthesized_comm(
    monkeypatch, comm
):
    stat = b"123 (" + comm + b") S " + b"0 " * 18 + b"1234 0 0\n"
    reads = Mock(side_effect=[stat, BOOT_ID.encode() + b"\n"])
    monkeypatch.setattr(Path, "read_bytes", reads)
    assert process._linux_process_identity(123) == (BOOT_ID, 1234)
    assert reads.call_count == 2


@pytest.mark.parametrize(
    "stat",
    [
        b"wrong",
        b"123 (python S " + b"0 " * 20,
        b"124 (python) S " + b"0 " * 19,
        b"123 (python) S 0",
        b"123 (python) S " + b"0 " * 18 + b"-1",
    ],
)
def test_linux_identity_rejects_malformed_stat(monkeypatch, stat):
    monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=[stat, BOOT_ID.encode()]))
    with pytest.raises(ValueError):
        process._linux_process_identity(123)


@pytest.mark.parametrize("missing", ["process", "boot_id"])
def test_linux_identity_distinguishes_exit_from_missing_boot_id(monkeypatch, missing):
    stat = b"123 (python) S " + b"0 " * 18 + b"1234\n"
    results = (
        [FileNotFoundError()] if missing == "process" else [stat, FileNotFoundError()]
    )
    monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=results))
    with pytest.raises(
        ProcessLookupError if missing == "process" else FileNotFoundError
    ):
        process._linux_process_identity(123)
