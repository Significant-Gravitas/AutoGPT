"""Fresh shell roots use the same atomic Windows ACL creation as the runtime."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from autogpt_desktop import shell_data, winlinks


def test_fresh_root_is_created_with_the_current_token_user(tmp_path, monkeypatch):
    monkeypatch.setattr(shell_data.sys, "platform", "win32")
    monkeypatch.setattr(winlinks, "current_user_sid", lambda: "user-sid")
    create = Mock(return_value=True)
    monkeypatch.setattr(winlinks, "create_private", create)
    root = tmp_path / "new-parent" / "profile"

    shell_data.create(root)

    assert root.parent.is_dir()
    assert not root.exists()
    create.assert_called_once_with(root, "user-sid")


def test_existing_root_is_never_claimed_or_repaired(tmp_path, monkeypatch):
    monkeypatch.setattr(shell_data.sys, "platform", "win32")
    monkeypatch.setattr(winlinks, "current_user_sid", lambda: "user-sid")
    create = Mock(return_value=False)
    monkeypatch.setattr(winlinks, "create_private", create)
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "marker"
    marker.write_bytes(b"existing data")
    before = root.stat()

    with pytest.raises(FileExistsError, match="appeared"):
        shell_data.create(root)

    assert marker.read_bytes() == b"existing data"
    assert root.stat().st_mode == before.st_mode
    assert root.stat().st_ino == before.st_ino
    create.assert_called_once_with(root, "user-sid")


@pytest.mark.parametrize("operation", ["identity", "create"])
def test_native_failure_is_not_ignored(tmp_path, monkeypatch, operation):
    monkeypatch.setattr(shell_data.sys, "platform", "win32")
    failure = OSError("injected native failure")
    identity = Mock(return_value="user-sid")
    create = Mock(return_value=True)
    (identity if operation == "identity" else create).side_effect = failure
    monkeypatch.setattr(winlinks, "current_user_sid", identity)
    monkeypatch.setattr(winlinks, "create_private", create)

    with pytest.raises(OSError) as caught:
        shell_data.create(tmp_path / "profile")

    assert caught.value is failure
    assert not (tmp_path / "profile").exists()


def test_non_windows_fails_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(shell_data.sys, "platform", "linux")
    create = Mock()
    monkeypatch.setattr(winlinks, "create_private", create)
    root = tmp_path / "absent" / "profile"

    with pytest.raises(OSError, match="requires Windows"):
        shell_data.create(root)

    assert not root.parent.exists()
    create.assert_not_called()


def test_parent_creation_failure_cannot_create_root(tmp_path, monkeypatch):
    monkeypatch.setattr(shell_data.sys, "platform", "win32")
    failure = OSError("parent refused")
    monkeypatch.setattr(Path, "mkdir", Mock(side_effect=failure))
    create = Mock()
    monkeypatch.setattr(winlinks, "create_private", create)

    with pytest.raises(OSError) as caught:
        shell_data.create(tmp_path / "profile")

    assert caught.value is failure
    create.assert_not_called()


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows ACLs")
def test_native_root_blocks_permissive_parent_inheritance(tmp_path):
    user = winlinks.current_user_sid()
    parent = tmp_path / "permissive"
    assert winlinks._create_directory(parent, (user, "SY", "WD"))
    root = parent / "profile"

    shell_data.create(root)
    config = root / "config"
    config.mkdir()
    marker = config / "synthetic.txt"
    marker.write_bytes(b"synthetic data")

    for path in (root, config, marker):
        handle = winlinks.open_as_it_is(path)
        try:
            access = winlinks.look(handle).access
        finally:
            winlinks._kernel32().CloseHandle(handle)
        assert access is not None
        assert {ace.sid for ace in access} == {user, winlinks.SYSTEM}
        assert all(ace.kind == winlinks.ACCESS_ALLOWED for ace in access)
