import json
import os
from pathlib import Path

import pytest

from autogpt_desktop import bootstrap, maintenance, migrations
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.maintenance import DataBusyError, DataLock, MaintenanceOperation
from autogpt_desktop.supervisor import Stack


def test_runtime_holds_lock_until_failed_start_is_fully_stopped(tmp_path, monkeypatch):
    stack = Stack(Bundle(tmp_path / "runtime"), DataDir(tmp_path / "data"))
    calls = []

    def locked():
        with pytest.raises(DataBusyError):
            with DataLock(stack.data.root):
                pytest.fail("data became available before the runtime finished")

    def fail():
        locked()
        calls.append("start")
        raise RuntimeError("failed startup")

    def stop():
        locked()
        calls.append("stop")

    monkeypatch.setattr(stack, "run_until_stopped", fail)
    monkeypatch.setattr(stack, "stop", stop)
    with pytest.raises(RuntimeError, match="failed startup"):
        stack.run()
    assert calls == ["start", "stop"]
    with DataLock(stack.data.root):
        pass


def test_contending_runtime_never_starts_or_cleans_up_other_runtime(
    tmp_path, monkeypatch
):
    stack = Stack(Bundle(tmp_path / "runtime"), DataDir(tmp_path / "data"))
    monkeypatch.setattr(
        stack, "run_until_stopped", lambda: pytest.fail("started without lock")
    )
    monkeypatch.setattr(
        stack, "stop", lambda: pytest.fail("cleaned another runtime's state")
    )
    with DataLock(stack.data.root):
        assert stack.run() == 1


def test_darwin_case_and_unicode_aliases_share_lock_name(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance.sys, "platform", "darwin")
    assert (
        DataLock(tmp_path / "Caf\u00e9").path.name
        == DataLock(tmp_path / "CAFE\u0301").path.name
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows extended path aliases")
@pytest.mark.parametrize("exists", [True, False])
def test_windows_extended_path_aliases_share_all_lock_names(tmp_path, exists):
    root = tmp_path / "AutoGPT"
    if exists:
        root.mkdir()
    plain = DataLock(root)
    extended = DataLock(Path("\\\\?\\" + str(root.resolve())))
    assert plain.key == extended.key
    assert plain.path.name == extended.path.name
    assert plain.journal.name == extended.journal.name
    with plain:
        with pytest.raises(DataBusyError):
            with extended:
                pytest.fail("extended path bypassed runtime lock")


@pytest.mark.parametrize(
    "content", ["{", '{"pid":0}', '{"pid":true}', '{"pid":99999999999999999999}', "[]"]
)
def test_invalid_shell_lease_blocks_maintenance_without_deleting_lease(
    tmp_path, content
):
    with DataLock(tmp_path / "data") as held:
        lease = held.root.parent / f".autogpt-shell-{held.key}-invalid.json"
        lease.write_text(content, encoding="utf-8")
        operation = MaintenanceOperation(held)
        with pytest.raises(DataBusyError, match="shell"):
            with operation:
                pytest.fail("invalid lease was accepted")
        assert lease.read_text("utf-8") == content
        assert not operation.path.exists()


def test_operation_publishes_before_checking_live_shell_lease(tmp_path, monkeypatch):
    with DataLock(tmp_path / "data") as held:
        lease = held.root.parent / f".autogpt-shell-{held.key}-{os.getpid()}-test.json"
        lease.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        operation = MaintenanceOperation(held)

        def owner(path):
            assert path == lease
            assert json.loads(operation.path.read_text("utf-8")) == {"pid": os.getpid()}
            return True

        monkeypatch.setattr(maintenance, "active_owner", owner)
        with pytest.raises(DataBusyError):
            with operation:
                pytest.fail("shell and maintenance overlapped")
        assert lease.exists() and not operation.path.exists()


def test_dead_shell_lease_is_ignored_without_deletion(tmp_path, monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "pid_exists", lambda pid: False)
    with DataLock(tmp_path / "data") as held:
        lease = held.root.parent / f".autogpt-shell-{held.key}-dead.json"
        lease.write_text('{"pid":123456}', encoding="utf-8")
        operation = MaintenanceOperation(held)
        with operation:
            assert operation.path.exists()
        assert lease.exists() and not operation.path.exists()


def test_pending_restore_journal_blocks_even_when_root_was_recreated(tmp_path):
    held = DataLock(tmp_path / "data")
    held.root.parent.mkdir(parents=True, exist_ok=True)
    held.journal.write_text("{", encoding="utf-8")
    with pytest.raises(DataBusyError, match="restore needs recovery"):
        with held:
            pytest.fail("opened data during interrupted restore")
    held.root.mkdir()
    with pytest.raises(DataBusyError, match="restore needs recovery"):
        with DataLock(held.root):
            pytest.fail("recreated root bypassed restore journal")


def test_live_operation_marker_is_never_replaced(tmp_path):
    with DataLock(tmp_path / "data") as held:
        operation = MaintenanceOperation(held)
        content = json.dumps({"pid": os.getpid()})
        operation.path.write_text(content, encoding="utf-8")
        with pytest.raises(DataBusyError, match="maintenance operation"):
            with operation:
                pytest.fail("replaced another operation marker")
        assert operation.path.read_text("utf-8") == content


def test_incompatible_schema_refuses_before_bootstrap_or_migration(
    tmp_path, monkeypatch
):
    stack = Stack(Bundle(tmp_path / "runtime"), DataDir(tmp_path / "data"))
    connector = object()
    monkeypatch.setattr(stack, "database_connector", lambda *args: connector)

    def reject(bundle, connect):
        assert bundle == stack.bundle and connect is connector
        raise migrations.IncompatibleSchemaError("newer schema")

    monkeypatch.setattr(migrations, "require_compatible_history", reject)
    for name in ("create_schemas", "remove_owner_trigger", "apply_migrations"):
        monkeypatch.setattr(
            bootstrap, name, lambda *args: pytest.fail("changed incompatible database")
        )
    with pytest.raises(migrations.IncompatibleSchemaError, match="newer schema"):
        stack.migrate({"postgres": 12345}, {"POSTGRES_PASSWORD": "synthetic"}, False)
    assert not stack.migrating
