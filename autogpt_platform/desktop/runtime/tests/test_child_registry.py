import json
import signal
import subprocess
import sys
from unittest.mock import Mock

import psutil
import pytest

from autogpt_desktop import process, reap
from autogpt_desktop.maintenance import DataBusyError, DataLock


@pytest.mark.parametrize(
    "entry",
    [
        {"pid": 123, "started": 99.5},
        {"pid": 123},
        {"pid": 123, "started": True},
        {"pid": 123, "started": float("nan")},
        {"pid": 123, "started": float("inf")},
        {"pid": 123, "started": 10**400},
        {"pid": 123, "started": 0},
        {"pid": True, "started": 100.0},
        {"pid": 0, "started": 100.0},
        {"pid": -123, "started": 100.0},
        {"pid": 2**31, "started": 100.0},
        {"pid": 123.0, "started": 100.0},
        {"pid": "123", "started": 100.0},
        "invalid entry",
    ],
)
def test_reaper_never_signals_unverified_identity(tmp_path, monkeypatch, entry):
    child = Mock()
    child.create_time.return_value = 100.0
    child.is_running.return_value = True
    child.children.return_value = []
    monkeypatch.setattr(psutil, "Process", Mock(return_value=child))
    wait = Mock(return_value=([], []))
    monkeypatch.setattr(psutil, "wait_procs", wait)
    monkeypatch.setattr(process.sys, "platform", "win32")
    registry = process.ChildRegistry(tmp_path / "children.json")
    saved = {"name": "service", **entry} if isinstance(entry, dict) else entry
    registry.path.write_text(json.dumps([saved]), encoding="utf-8")
    registry.reap_leftovers()
    child.kill.assert_not_called()
    child.children.assert_not_called()
    wait.assert_called_once_with([], timeout=10)


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_verified_identity_is_rechecked_before_signaling(
    tmp_path, monkeypatch, platform
):
    child = Mock()
    child.create_time.return_value = 100.0
    child.is_running.return_value = False
    monkeypatch.setattr(psutil, "Process", Mock(return_value=child))
    monkeypatch.setattr(psutil, "wait_procs", Mock(return_value=([], [])))
    monkeypatch.setattr(process.sys, "platform", platform)
    killpg = Mock()
    monkeypatch.setattr(process.os, "killpg", killpg, raising=False)
    registry = process.ChildRegistry(tmp_path / "children.json")
    registry.path.write_text('[{"pid":123,"started":100.0}]', encoding="utf-8")
    registry.reap_leftovers()
    child.kill.assert_not_called()
    killpg.assert_not_called()


@pytest.mark.parametrize("leader", [True, False])
def test_posix_reaper_only_kills_the_verified_services_own_group(
    tmp_path, monkeypatch, leader
):
    child = Mock()
    child.create_time.return_value = 100.0
    child.is_running.return_value = True
    monkeypatch.setattr(psutil, "Process", Mock(return_value=child))
    monkeypatch.setattr(psutil, "wait_procs", Mock(return_value=([], [])))
    monkeypatch.setattr(process.sys, "platform", "linux")
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        process.os, "getpgid", lambda pid: pid if leader else 456, raising=False
    )
    killpg = Mock()
    monkeypatch.setattr(process.os, "killpg", killpg, raising=False)
    registry = process.ChildRegistry(tmp_path / "children.json")
    registry.path.write_text('[{"pid":123,"started":100.0}]', encoding="utf-8")
    registry.reap_leftovers()
    if leader:
        killpg.assert_called_once_with(123, signal.SIGKILL)
    else:
        killpg.assert_not_called()


def test_shell_reaper_respects_runtime_and_restore_locks(tmp_path, monkeypatch):
    root = tmp_path / "data"
    cleanup = Mock()
    monkeypatch.setattr(process.ChildRegistry, "reap_leftovers", cleanup)
    with DataLock(root) as held:
        with pytest.raises(DataBusyError):
            reap.reap(root)
        cleanup.assert_not_called()
    held.journal.write_text("{}", encoding="utf-8")
    with pytest.raises(DataBusyError):
        reap.reap(root)
    cleanup.assert_not_called()
    held.journal.unlink()
    reap.reap(root)
    cleanup.assert_called_once_with()


@pytest.mark.parametrize(
    "failure", ["inspect", "parent-kill", "child-kill", "survivor"]
)
def test_uncertain_cleanup_retains_registry_for_retry(tmp_path, monkeypatch, failure):
    root = tmp_path / "data"
    registry = root / "run" / "children.json"
    registry.parent.mkdir(parents=True)
    registry.write_text('[{"pid":123,"started":100.0}]', encoding="utf-8")
    original = json.loads(registry.read_text("utf-8"))
    parent, child = Mock(pid=123), Mock(pid=456)
    child.create_time.return_value = 101.0
    parent.create_time.return_value = 100.0
    parent.is_running.return_value = True
    parent.children.return_value = [child]
    monkeypatch.setattr(psutil, "Process", Mock(return_value=parent))
    wait = Mock(return_value=([], [child] if failure == "survivor" else []))
    monkeypatch.setattr(psutil, "wait_procs", wait)
    monkeypatch.setattr(process.sys, "platform", "win32")
    if failure == "inspect":
        parent.create_time.side_effect = psutil.AccessDenied(123)
    elif failure == "parent-kill":
        parent.kill.side_effect = psutil.AccessDenied(123)
    elif failure == "child-kill":
        child.kill.side_effect = psutil.AccessDenied(456)
    with pytest.raises(RuntimeError, match="retained"):
        reap.reap(root)
    retained = json.loads(registry.read_text("utf-8"))
    assert retained[:1] == original
    if failure != "inspect":
        assert any(item["pid"] == 456 and item["started"] == 101.0 for item in retained)
        wait.assert_called_once_with([parent, child], timeout=10)
        parent.kill.assert_called_once_with()


def test_disappearing_descendant_does_not_prevent_parent_cleanup(tmp_path, monkeypatch):
    registry = process.ChildRegistry(tmp_path / "children.json")
    registry.path.write_text('[{"pid":123,"started":100.0}]', encoding="utf-8")
    parent, child = Mock(pid=123), Mock(pid=456)
    child.create_time.return_value = 101.0
    parent.create_time.return_value = 100.0
    parent.is_running.return_value = True
    parent.children.return_value = [child]
    child.kill.side_effect = psutil.NoSuchProcess(456)
    monkeypatch.setattr(psutil, "Process", Mock(return_value=parent))
    monkeypatch.setattr(psutil, "wait_procs", Mock(return_value=([parent, child], [])))
    monkeypatch.setattr(process.sys, "platform", "win32")
    registry.reap_leftovers()
    parent.kill.assert_called_once_with()
    assert not registry.path.exists()


def test_denied_descendant_remains_reapable_after_parent_exits(tmp_path, monkeypatch):
    registry = process.ChildRegistry(tmp_path / "children.json")
    registry.path.write_text('[{"pid":123,"started":100.0}]', encoding="utf-8")
    parent, child = Mock(pid=123), Mock(pid=456)
    for item, started in ((parent, 100.0), (child, 101.0)):
        item.create_time.return_value = started
        item.is_running.return_value = True
        item.children.return_value = []
    parent.children.return_value = [child]

    def lookup(pid):
        item = parent if pid == 123 else child
        if not item.is_running():
            raise psutil.NoSuchProcess(pid)
        return item

    def stop_parent():
        retained = json.loads(registry.path.read_text("utf-8"))
        assert any(item["pid"] == 456 and item["started"] == 101.0 for item in retained)
        parent.is_running.return_value = False

    def stop_child():
        if child.kill.call_count == 1:
            raise psutil.AccessDenied(456)
        child.is_running.return_value = False

    parent.kill.side_effect = stop_parent
    child.kill.side_effect = stop_child
    monkeypatch.setattr(psutil, "Process", lookup)
    monkeypatch.setattr(
        psutil,
        "wait_procs",
        lambda items, timeout: ([], [p for p in items if p.is_running()]),
    )
    monkeypatch.setattr(process.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="retained"):
        registry.reap_leftovers()
    assert not parent.is_running()
    assert child.is_running()
    registry.reap_leftovers()
    assert not child.is_running()
    assert not registry.path.exists()


@pytest.mark.parametrize("failure", ["inspect", "persist"])
def test_descendant_identity_is_saved_before_any_signal(tmp_path, monkeypatch, failure):
    registry = process.ChildRegistry(tmp_path / "children.json")
    registry.path.write_text('[{"pid":123,"started":100.0}]', encoding="utf-8")
    original = registry.path.read_bytes()
    parent, child = Mock(pid=123), Mock(pid=456)
    parent.create_time.return_value = 100.0
    parent.is_running.return_value = True
    parent.children.return_value = [child]
    child.create_time.return_value = 101.0
    if failure == "inspect":
        child.create_time.side_effect = psutil.AccessDenied(456)
    else:
        monkeypatch.setattr(process.os, "replace", Mock(side_effect=PermissionError))
    monkeypatch.setattr(psutil, "Process", Mock(return_value=parent))
    monkeypatch.setattr(psutil, "wait_procs", Mock(return_value=([], [])))
    monkeypatch.setattr(process.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="retained"):
        registry.reap_leftovers()
    parent.kill.assert_not_called()
    child.kill.assert_not_called()
    assert registry.path.read_bytes() == original


@pytest.mark.parametrize("content", ["{", "{}", "null"])
def test_helper_reports_unreadable_registry_without_deleting_it(tmp_path, content):
    root = tmp_path / "data"
    registry = root / "run" / "children.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(content, encoding="utf-8")
    with pytest.raises(RuntimeError, match="retained"):
        reap.reap(root)
    assert registry.read_text("utf-8") == content


@pytest.mark.parametrize("matching", [True, False])
def test_native_reaper_kills_owned_child_and_preserves_reused_pid(tmp_path, matching):
    root = tmp_path / "data"
    registry = root / "run" / "children.json"
    registry.parent.mkdir(parents=True)
    child = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=sys.platform != "win32",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        started = psutil.Process(child.pid).create_time()
        registry.write_text(
            json.dumps(
                [
                    {
                        "name": "owned test child",
                        "pid": child.pid,
                        "started": started if matching else started - 0.5,
                    }
                ]
            ),
            encoding="utf-8",
        )
        reap.reap(root)
        if matching:
            assert child.wait(timeout=5) is not None
        else:
            assert child.poll() is None
        assert not registry.exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_native_reaper_stops_the_service_process_tree(tmp_path):
    root = tmp_path / "data"
    registry = root / "run" / "children.json"
    registry.parent.mkdir(parents=True)
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import subprocess,sys,time; c=subprocess.Popen([sys.executable,'-I','-B','-c','import time; time.sleep(30)']); print(c.pid,flush=True); time.sleep(30)",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=sys.platform != "win32",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    child = None
    try:
        child = psutil.Process(int(launcher.stdout.readline()))
        registry.write_text(
            json.dumps(
                [
                    {
                        "name": "owned test launcher",
                        "pid": launcher.pid,
                        "started": psutil.Process(launcher.pid).create_time(),
                    }
                ]
            ),
            encoding="utf-8",
        )
        reap.reap(root)
        launcher.wait(timeout=5)
        try:
            child.wait(timeout=5)
        except psutil.TimeoutExpired:
            assert child.status() == psutil.STATUS_ZOMBIE
    finally:
        if launcher.poll() is None:
            launcher.kill()
        launcher.wait(timeout=5)
        if child is not None and child.is_running():
            child.kill()
        launcher.stdout.close()
