"""Fresh Windows roots use explicit access; existing roots are not migrated."""

import stat
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from autogpt_desktop import layout, winlinks

USER = "S-1-5-21-1-2-3-1001"
ROOT = Path("missing-parent") / "profile"
DIRECTORY = SimpleNamespace(
    st_mode=stat.S_IFDIR | 0o755, st_file_attributes=stat.FILE_ATTRIBUTE_DIRECTORY
)


@contextmanager
def preparation(monkeypatch, *, windows=True, existing=None, created=True):
    monkeypatch.setattr(layout, "WINDOWS", windows)
    user = Mock(return_value=USER)
    create = Mock(return_value=created)
    monkeypatch.setattr(winlinks, "current_user_sid", user)
    monkeypatch.setattr(winlinks, "create_private", create, raising=False)
    monkeypatch.setattr(
        winlinks, "refusal", Mock(side_effect=AssertionError("ACL audit"))
    )
    monkeypatch.setattr(
        winlinks, "_kernel32", Mock(side_effect=AssertionError("native"))
    )
    monkeypatch.setattr(
        winlinks, "_advapi32", Mock(side_effect=AssertionError("native"))
    )
    with (
        patch.object(Path, "mkdir", autospec=True) as mkdir,
        patch.object(Path, "chmod", autospec=True) as chmod,
        patch.object(Path, "lstat", autospec=True) as lstat,
    ):
        lstat.side_effect = existing if existing is not None else FileNotFoundError()
        yield SimpleNamespace(
            user=user, create=create, mkdir=mkdir, chmod=chmod, lstat=lstat
        )


def test_fresh_windows_root_uses_explicit_creator_and_ordinary_parents(monkeypatch):
    with preparation(monkeypatch) as calls:
        layout.DataDir(ROOT).prepare()
    calls.user.assert_called_once_with()
    calls.create.assert_called_once_with(ROOT, USER)
    calls.chmod.assert_not_called()
    calls.mkdir.assert_any_call(ROOT.parent, parents=True, exist_ok=True)
    assert all("mode" not in call.kwargs for call in calls.mkdir.call_args_list)


def test_existing_windows_root_keeps_its_permissions_and_contents(monkeypatch):
    with (
        preparation(monkeypatch, existing=[DIRECTORY]) as calls,
        patch.object(Path, "open", side_effect=AssertionError("content access")),
    ):
        layout.DataDir(ROOT).prepare()
    calls.user.assert_not_called()
    calls.create.assert_not_called()
    calls.chmod.assert_not_called()
    assert all(call.args[0] != ROOT.parent for call in calls.mkdir.call_args_list)


@pytest.mark.parametrize(
    "information",
    [
        SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_file_attributes=0),
        SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o755,
            st_file_attributes=stat.FILE_ATTRIBUTE_DIRECTORY
            | stat.FILE_ATTRIBUTE_REPARSE_POINT,
        ),
        SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0),
    ],
)
@pytest.mark.parametrize("raced", [False, True])
def test_existing_or_raced_in_non_directory_is_rejected(
    monkeypatch, information, raced
):
    observations = [FileNotFoundError(), information] if raced else [information]
    with (
        preparation(monkeypatch, existing=observations, created=False) as calls,
        pytest.raises(OSError),
    ):
        layout.DataDir(ROOT).prepare()
    calls.chmod.assert_not_called()
    assert all(call.args[0] == ROOT.parent for call in calls.mkdir.call_args_list)
    assert calls.create.call_count == int(raced)


def test_raced_in_directory_is_rechecked_without_repair(monkeypatch):
    with preparation(
        monkeypatch, existing=[FileNotFoundError(), DIRECTORY], created=False
    ) as calls:
        layout.DataDir(ROOT).prepare()
    calls.create.assert_called_once_with(ROOT, USER)
    assert calls.lstat.call_count == 2
    calls.chmod.assert_not_called()


@pytest.mark.parametrize("failure", ["lstat", "user", "create", "mkdir"])
def test_windows_root_errors_stop_before_children(monkeypatch, failure):
    error = PermissionError("mock access denied")
    with preparation(monkeypatch) as calls:
        getattr(calls, failure).side_effect = error
        with pytest.raises(PermissionError) as caught:
            layout.DataDir(ROOT).prepare()
    assert caught.value is error
    calls.chmod.assert_not_called()
    assert all(call.args[0] == ROOT.parent for call in calls.mkdir.call_args_list)


def test_raced_in_name_that_disappears_is_not_reported_as_created(monkeypatch):
    with (
        preparation(
            monkeypatch,
            existing=[FileNotFoundError(), FileNotFoundError()],
            created=False,
        ) as calls,
        pytest.raises(FileNotFoundError),
    ):
        layout.DataDir(ROOT).prepare()
    calls.create.assert_called_once_with(ROOT, USER)
    calls.chmod.assert_not_called()
    assert all(call.args[0] == ROOT.parent for call in calls.mkdir.call_args_list)


@pytest.mark.parametrize("existing", [None, [DIRECTORY]])
def test_posix_root_keeps_mode_0700_and_never_calls_windows(monkeypatch, existing):
    with preparation(monkeypatch, windows=False, existing=existing) as calls:
        layout.DataDir(ROOT).prepare()
    calls.mkdir.assert_any_call(ROOT, mode=0o700, parents=True, exist_ok=True)
    calls.chmod.assert_called_once_with(ROOT, 0o700)
    calls.user.assert_not_called()
    calls.create.assert_not_called()
