import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from autogpt_desktop import backup, migrations
from autogpt_desktop.layout import Bundle
from autogpt_desktop.maintenance import DataBusyError, DataLock


@pytest.fixture
def populated(tmp_path, monkeypatch):
    bundle = Bundle(tmp_path / "runtime")
    migration = bundle.backend_dir / "migrations" / "001_initial" / "migration.sql"
    migration.parent.mkdir(parents=True)
    migration.write_text("CREATE TABLE example (id text);\n", encoding="utf-8")
    root = tmp_path / "AutoGPT"
    contents = {
        "autogpt-install.json": '{"install":"AutoGPT"}',
        "postgres/PG_VERSION": "16",
        "postgres/base/123/456": "synthetic populated database pages",
        "config/secrets.env": "ENCRYPTION_KEY=synthetic-test-key",
        "workspaces/user/nested/document.txt": "saved user work",
        "store-media/image.bin": "media",
        "rabbitmq/data/pending": "queued-work",
        "valkey/dump.rdb": "durable redis state",
        "cache/backend/user-data": "not assumed disposable",
        "run/stopped-runs.json": '{"runs":["do-not-resume"]}',
        "run/children.json": "[]",
    }
    for name, text in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def run_tool(argv, **kwargs):
        output = (
            b"postgres (PostgreSQL) 16.4\n"
            if argv[-1] == "--version"
            else b"Database cluster state:               shut down\n"
        )
        return subprocess.CompletedProcess(argv, 0, output, b"")

    monkeypatch.setattr(backup, "run_tool", run_tool)
    return bundle, root, tmp_path / "snapshot.agptbackup", contents


def test_full_round_trip_retains_original_and_recovery_data(populated):
    bundle, root, archive, contents = populated
    backup.create_backup(bundle, root, archive, "AutoGPT")
    (root / "workspaces/user/nested/document.txt").write_text(
        "newer work", encoding="utf-8"
    )
    previous = backup.restore_backup(bundle, root, archive, "AutoGPT")
    assert previous and previous.is_dir()
    assert (previous / "workspaces/user/nested/document.txt").read_text(
        "utf-8"
    ) == "newer work"
    for name, text in contents.items():
        if name not in backup.EXCLUDED:
            assert (root / name).read_text("utf-8") == text
    assert not (root / "run/children.json").exists()
    assert archive.is_file()


def test_restoring_corrupt_checksums_leaves_current_data_in_place(populated):
    bundle, root, archive, _ = populated
    backup.create_backup(bundle, root, archive, "AutoGPT")
    rewrite(
        archive,
        lambda manifest: manifest["files"]["config/secrets.env"].update(
            sha256="0" * 64
        ),
    )
    original = (root / "config/secrets.env").read_bytes()
    with pytest.raises(backup.BackupError, match="checksum"):
        backup.restore_backup(bundle, root, archive, "AutoGPT")
    assert (root / "config/secrets.env").read_bytes() == original
    assert not list(root.parent.glob("AutoGPT.before-restore-*"))
    assert not list(root.parent.glob(".AutoGPT.restore-*"))


@pytest.mark.parametrize("failure", ["rename", "parent-sync"])
def test_restore_publication_failure_restores_original_directory(
    populated, monkeypatch, failure
):
    bundle, root, archive, _ = populated
    backup.create_backup(bundle, root, archive, "AutoGPT")
    document = root / "workspaces/user/nested/document.txt"
    document.write_text("newer work", encoding="utf-8")
    rename = Path.rename
    failed = False

    def fail_rename(path, target):
        if path.name.startswith(".AutoGPT.restore-") and target == root:
            raise OSError("injected publication failure")
        return rename(path, target)

    def fail_sync(path):
        nonlocal failed
        if path == root.parent and not root.exists() and not failed:
            failed = True
            raise OSError("injected publication failure")

    if failure == "rename":
        monkeypatch.setattr(Path, "rename", fail_rename)
    else:
        monkeypatch.setattr(backup, "sync_directory", fail_sync)
    with pytest.raises(OSError, match="injected publication failure"):
        backup.restore_backup(bundle, root, archive, "AutoGPT")
    assert document.read_text("utf-8") == "newer work"
    assert not list(root.parent.glob("AutoGPT.before-restore-*"))
    assert not list(root.parent.glob(".AutoGPT.restore-*"))
    assert not DataLock(root).journal.exists()


def test_concurrent_root_recreation_preserves_all_data_and_blocks_startup(
    populated, monkeypatch
):
    bundle, root, archive, _ = populated
    backup.create_backup(bundle, root, archive, "AutoGPT")
    (root / "workspaces/user/nested/document.txt").write_text(
        "newer work", encoding="utf-8"
    )
    rename = Path.rename

    def recreate(path, target):
        result = rename(path, target)
        if path == root:
            root.mkdir()
            (root / "autogpt-install.json").write_text(
                '{"install":"AutoGPT"}', encoding="utf-8"
            )
        return result

    monkeypatch.setattr(Path, "rename", recreate)
    with pytest.raises(backup.BackupError, match="Restore needs recovery"):
        backup.restore_backup(bundle, root, archive, "AutoGPT")
    journal = json.loads(DataLock(root).journal.read_text("utf-8"))
    previous, stage = Path(journal["previous"]), Path(journal["stage"])
    assert (previous / "workspaces/user/nested/document.txt").read_text(
        "utf-8"
    ) == "newer work"
    assert (stage / "workspaces/user/nested/document.txt").read_text(
        "utf-8"
    ) == "saved user work"
    with pytest.raises(DataBusyError, match="restore needs recovery"):
        with DataLock(root):
            pytest.fail("runtime initialized a replacement data folder")


def test_interrupted_restore_retains_journal_old_data_and_staged_backup(
    populated, monkeypatch
):
    bundle, root, archive, _ = populated
    backup.create_backup(bundle, root, archive, "AutoGPT")
    rename = Path.rename

    def interrupt(path, target):
        if path.name.startswith(".AutoGPT.restore-"):
            raise KeyboardInterrupt("interrupted between renames")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", interrupt)
    with pytest.raises(KeyboardInterrupt):
        backup.restore_backup(bundle, root, archive, "AutoGPT")
    journal = json.loads(DataLock(root).journal.read_text("utf-8"))
    assert not root.exists()
    assert Path(journal["previous"]).is_dir()
    assert Path(journal["stage"]).is_dir()
    with pytest.raises(DataBusyError, match="restore needs recovery"):
        with DataLock(root):
            pytest.fail("runtime initialized data after interrupted restore")


@pytest.mark.parametrize(
    "marker", ["SingletonLock", "SingletonSocket", "SingletonCookie"]
)
def test_browser_profile_must_be_closed_for_backup_and_restore(populated, marker):
    bundle, root, archive, _ = populated
    backup.create_backup(bundle, root, archive, "AutoGPT")
    (root / marker).write_text("synthetic marker", encoding="utf-8")
    with pytest.raises(backup.BackupError, match="browser profile"):
        backup.create_backup(bundle, root, archive.with_suffix(".another"), "AutoGPT")
    with pytest.raises(backup.BackupError, match="browser profile"):
        backup.restore_backup(bundle, root, archive, "AutoGPT")
    assert not list(root.parent.glob("AutoGPT.before-restore-*"))


def rewrite(path, change):
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(files[backup.MANIFEST])
    change(manifest)
    files[backup.MANIFEST] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "C:/escape",
        "a\\escape",
        "/absolute",
        "a/../escape",
        "a:stream",
        "nul.txt",
        "trailing.",
    ],
)
def test_archive_paths_cannot_escape_or_alias_existing_files(populated, name):
    bundle, root, archive, _ = populated
    backup.create_backup(bundle, root, archive, "AutoGPT")
    rewrite(
        archive,
        lambda manifest: manifest["files"].update(
            {
                name: {
                    "size": 0,
                    "mode": 0o600,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                }
            }
        ),
    )
    with pytest.raises(backup.BackupError, match="path"):
        backup.restore_backup(bundle, root, archive, "AutoGPT")
    assert (root / "postgres/PG_VERSION").exists()


def test_old_bundle_cannot_restore_newer_schema_snapshot(populated):
    bundle, root, archive, _ = populated
    backup.create_backup(bundle, root, archive, "AutoGPT")
    (bundle.backend_dir / "migrations/001_initial/migration.sql").unlink()
    with pytest.raises(backup.BackupError, match="migrations"):
        backup.restore_backup(bundle, root, archive, "AutoGPT")


@pytest.mark.parametrize(
    "field,value",
    [
        ("install", "AutoGPT-other"),
        ("machine", "different-machine"),
        ("platform", "different-os"),
        ("postgres_major", "999"),
    ],
)
def test_incompatible_restore_is_refused_before_replacing_data(populated, field, value):
    bundle, root, archive, _ = populated
    backup.create_backup(bundle, root, archive, "AutoGPT")
    rewrite(archive, lambda manifest: manifest.update({field: value}))
    with pytest.raises(backup.BackupError):
        backup.restore_backup(bundle, root, archive, "AutoGPT")
    assert not list(root.parent.glob("AutoGPT.before-restore-*"))


def test_running_database_and_existing_archive_are_never_overwritten(populated):
    bundle, root, archive, _ = populated
    (root / "postgres/postmaster.pid").write_text("123", encoding="ascii")
    with pytest.raises(backup.BackupError, match="running"):
        backup.create_backup(bundle, root, archive, "AutoGPT")
    assert not archive.exists()
    (root / "postgres/postmaster.pid").unlink()
    archive.write_bytes(b"existing backup")
    with pytest.raises(backup.BackupError, match="already exists"):
        backup.create_backup(bundle, root, archive, "AutoGPT")
    assert archive.read_bytes() == b"existing backup"


def test_low_disk_space_refuses_backup_without_deleting_anything(
    populated, monkeypatch
):
    bundle, root, archive, _ = populated
    usage = backup.shutil.disk_usage(root)
    monkeypatch.setattr(
        backup.shutil,
        "disk_usage",
        lambda path: type(usage)(usage.total, usage.total, 0),
    )
    with pytest.raises(backup.BackupError, match="free space"):
        backup.create_backup(bundle, root, archive, "AutoGPT")
    assert not archive.exists()
    assert (root / "config/secrets.env").exists()


def test_data_mutated_during_copy_does_not_publish_backup(populated, monkeypatch):
    bundle, root, archive, _ = populated
    original = backup.inventory
    calls = 0

    def inventory(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / "new-file").write_text("concurrent writer", encoding="utf-8")
        return original(path)

    monkeypatch.setattr(backup, "inventory", inventory)
    with pytest.raises(backup.BackupError, match="changed"):
        backup.create_backup(bundle, root, archive, "AutoGPT")
    assert not archive.exists()


def test_lock_survives_data_rename_and_blocks_other_process(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    with DataLock(root):
        root.rename(tmp_path / "old")
        root.mkdir()
        with pytest.raises(DataBusyError):
            with DataLock(root):
                pytest.fail("lock was lost across restore")
        code = "from pathlib import Path; from autogpt_desktop.maintenance import DataLock; import sys; DataLock(Path(sys.argv[1])).__enter__()"
        result = subprocess.run(
            [sys.executable, "-c", code, str(root)],
            capture_output=True,
            timeout=20,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode != 0 and b"DataBusyError" in result.stderr
    with DataLock(root):
        pass


def test_backup_and_restore_cannot_run_while_runtime_holds_lock(populated):
    bundle, root, archive, _ = populated
    with DataLock(root):
        with pytest.raises(DataBusyError):
            backup.create_backup(bundle, root, archive, "AutoGPT")
        with pytest.raises(DataBusyError):
            backup.restore_backup(bundle, root, archive, "AutoGPT")


def test_alias_uses_same_lock(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks needs a Windows privilege")
    with DataLock(root):
        with pytest.raises(DataBusyError):
            with DataLock(alias):
                pytest.fail("alias bypassed the lock")


def test_links_fail_explicitly_instead_of_losing_data(populated):
    bundle, root, archive, _ = populated
    try:
        (root / "linked-workspace").symlink_to(
            root / "workspaces", target_is_directory=True
        )
    except OSError:
        pytest.skip("creating symlinks needs a Windows privilege")
    with pytest.raises(backup.BackupError, match="link or junction"):
        backup.create_backup(bundle, root, archive, "AutoGPT")
    assert not archive.exists()
