import ctypes
import errno
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from autogpt_desktop import atomic_file


def test_native_publication_preserves_completed_bytes_and_collision(tmp_path):
    pending = tmp_path / "pending"
    target = tmp_path / "backup"
    content = b"complete archive" * 4096
    pending.write_bytes(content)
    atomic_file.publish_file(pending, target)
    assert target.read_bytes() == content
    other = tmp_path / "other"
    other.write_bytes(b"replacement")
    with pytest.raises(FileExistsError):
        atomic_file.publish_file(other, target)
    assert target.read_bytes() == content
    assert other.read_bytes() == b"replacement"


def test_native_publication_without_hard_links_has_one_winner(tmp_path, monkeypatch):
    target = tmp_path / "backup"
    sources = [tmp_path / "first", tmp_path / "second"]
    contents = [b"first complete archive" * 4096, b"second complete archive" * 4096]
    for source, content in zip(sources, contents):
        source.write_bytes(content)
    monkeypatch.setattr(
        atomic_file.os,
        "link",
        Mock(side_effect=OSError(errno.EOPNOTSUPP, "hard links unsupported")),
    )
    barrier = threading.Barrier(2, timeout=5)

    def publish(index):
        barrier.wait()
        try:
            atomic_file.publish_file(sources[index], target)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, [0, 1]))
    assert results.count(True) == 1
    winner = results.index(True)
    loser = results.index(False)
    assert target.read_bytes() == contents[winner]
    assert not sources[winner].exists()
    assert sources[loser].read_bytes() == contents[loser]


@pytest.mark.parametrize(
    "code", [errno.EPERM, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP]
)
def test_unsupported_hard_link_uses_exclusive_rename(tmp_path, monkeypatch, code):
    pending = tmp_path / "pending"
    target = tmp_path / "backup"
    pending.write_bytes(b"completed")
    monkeypatch.setattr(atomic_file, "sys", SimpleNamespace(platform="linux"))
    link = Mock(side_effect=OSError(code, "no hard links"))
    rename = Mock()
    monkeypatch.setattr(atomic_file.os, "link", link)
    monkeypatch.setattr(atomic_file, "rename_noreplace", rename)
    atomic_file.publish_file(pending, target)
    rename.assert_called_once_with(pending, target)
    assert pending.read_bytes() == b"completed"
    assert not target.exists()


@pytest.mark.parametrize(
    "code",
    [
        errno.EEXIST,
        errno.EACCES,
        errno.EROFS,
        errno.ENOSPC,
        errno.EIO,
        errno.ENOENT,
        errno.EXDEV,
    ],
)
def test_other_link_errors_do_not_attempt_another_operation(
    tmp_path, monkeypatch, code
):
    pending = tmp_path / "pending"
    target = tmp_path / "backup"
    pending.write_bytes(b"completed")
    target.write_bytes(b"existing")
    failure = OSError(code, "injected")
    monkeypatch.setattr(atomic_file, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(atomic_file.os, "link", Mock(side_effect=failure))
    rename = Mock()
    monkeypatch.setattr(atomic_file, "rename_noreplace", rename)
    with pytest.raises(OSError) as caught:
        atomic_file.publish_file(pending, target)
    assert caught.value is failure
    rename.assert_not_called()
    assert pending.read_bytes() == b"completed"
    assert target.read_bytes() == b"existing"


@pytest.mark.parametrize("system", ["linux", "darwin"])
def test_native_rename_binding_requests_only_exclusive_publication(
    tmp_path, monkeypatch, system
):
    pending = tmp_path / "pending-é"
    target = tmp_path / "backup-é"
    rename = Mock(return_value=0)
    name = "renameat2" if system == "linux" else "renamex_np"
    library = SimpleNamespace(**{name: rename})
    monkeypatch.setattr(atomic_file, "sys", SimpleNamespace(platform=system))
    loader = Mock(return_value=library)
    monkeypatch.setattr(atomic_file.ctypes, "CDLL", loader)
    atomic_file.rename_noreplace(pending, target)
    loader.assert_called_once_with(None, use_errno=True)
    if system == "linux":
        rename.assert_called_once_with(
            -100, os.fsencode(pending), -100, os.fsencode(target), 1
        )
        assert rename.argtypes == [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
    else:
        rename.assert_called_once_with(os.fsencode(pending), os.fsencode(target), 4)
        assert rename.argtypes == [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    assert rename.restype is ctypes.c_int


@pytest.mark.parametrize("system", ["linux", "darwin"])
@pytest.mark.parametrize(
    "code",
    [
        errno.EEXIST,
        errno.EACCES,
        errno.ENOSPC,
        errno.EROFS,
        errno.EXDEV,
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
    ],
)
def test_native_error_never_falls_back_to_copy_or_replace(
    tmp_path, monkeypatch, system, code
):
    pending = tmp_path / "pending"
    target = tmp_path / "backup"
    pending.write_bytes(b"completed")
    target.write_bytes(b"existing")

    def fail(*args):
        ctypes.set_errno(code)
        return -1

    rename = Mock(side_effect=fail)
    name = "renameat2" if system == "linux" else "renamex_np"
    monkeypatch.setattr(atomic_file, "sys", SimpleNamespace(platform=system))
    monkeypatch.setattr(
        atomic_file.ctypes, "CDLL", Mock(return_value=SimpleNamespace(**{name: rename}))
    )
    with pytest.raises(OSError) as caught:
        atomic_file.rename_noreplace(pending, target)
    assert caught.value.errno == code
    assert pending.read_bytes() == b"completed"
    assert target.read_bytes() == b"existing"


@pytest.mark.parametrize("system", ["linux", "darwin", "unsupported"])
def test_missing_native_primitive_refuses_publication(tmp_path, monkeypatch, system):
    pending = tmp_path / "pending"
    target = tmp_path / "backup"
    pending.write_bytes(b"completed")
    monkeypatch.setattr(atomic_file, "sys", SimpleNamespace(platform=system))
    monkeypatch.setattr(
        atomic_file.ctypes, "CDLL", Mock(return_value=SimpleNamespace())
    )
    with pytest.raises(OSError, match="unavailable") as caught:
        atomic_file.rename_noreplace(pending, target)
    assert caught.value.errno == errno.ENOTSUP
    assert pending.read_bytes() == b"completed"
    assert not target.exists()


@pytest.mark.parametrize("source", [True, False])
def test_native_binding_rejects_embedded_null_paths(tmp_path, monkeypatch, source):
    loader = Mock()
    monkeypatch.setattr(atomic_file.ctypes, "CDLL", loader)
    paths = [tmp_path / "pending", tmp_path / "backup"]
    paths[0 if source else 1] = Path("invalid\0suffix")
    with pytest.raises(ValueError, match="null"):
        atomic_file.rename_noreplace(*paths)
    loader.assert_not_called()
