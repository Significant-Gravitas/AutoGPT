"""Dependency-light host filesystem contracts, on both native platforms."""

import os
from pathlib import Path

import pytest

from backend.copilot.host_fs import (
    cli_tmpdir,
    open_no_follow,
    save_new_file,
    unlink_no_follow,
)


@pytest.mark.parametrize("mode", ["r", "r+", "w", "w+", "a", "a+", "x", "x+"])
@pytest.mark.parametrize("binary", [False, True])
def test_open_modes(tmp_path: Path, mode: str, binary: bool):
    path = tmp_path / "file"
    if not mode.startswith("x"):
        path.write_bytes(b"original")
    with open(path, mode + ("b" if binary else ""), opener=open_no_follow) as file:
        assert not os.get_inheritable(file.fileno())
        if mode.startswith("r"):
            assert file.read() == (b"original" if binary else "original")
        if mode != "r":
            file.seek(0)
            file.write(b"new" if binary else "new")
    expected = b"originalnew" if mode.startswith("a") else b"new"
    if mode == "r":
        expected = b"original"
    elif mode == "r+":
        expected = b"newginal"
    assert path.read_bytes() == expected


def test_nested_exclusive_delete_and_names(tmp_path):
    path = tmp_path / "a" / "b" / "file"
    fd = open_no_follow(
        str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, make_parents=True
    )
    os.write(fd, b"bytes\r\n\x1a")
    os.close(fd)
    assert path.read_bytes() == b"bytes\r\n\x1a"
    with pytest.raises(FileExistsError):
        open_no_follow(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    assert path.read_bytes() == b"bytes\r\n\x1a"
    assert save_new_file(str(path.parent), "file", b"second").endswith("file_1")
    assert cli_tmpdir(str(path)) == str(path) + ".cli"
    unlink_no_follow(str(path))
    with pytest.raises(FileNotFoundError):
        open_no_follow(str(path), os.O_RDONLY)
    with pytest.raises(FileNotFoundError):
        unlink_no_follow(str(path))
    with pytest.raises(IsADirectoryError):
        open_no_follow(str(path.parent), os.O_RDONLY)
    with pytest.raises(OSError):
        unlink_no_follow(str(path.parent))


@pytest.mark.parametrize("path", ["relative", "../file"])
def test_absolute_required(path):
    with pytest.raises(ValueError):
        open_no_follow(path, os.O_RDONLY)


def test_unicode_and_long_path(tmp_path):
    path = tmp_path / ("nested-" + "a" * 90) / ("b" * 90) / "附件🚀.txt"
    fd = open_no_follow(str(path), os.O_CREAT | os.O_WRONLY, make_parents=True)
    os.write(fd, b"unicode name")
    os.close(fd)
    with open(path, "rb", opener=open_no_follow) as file:
        assert file.read() == b"unicode name"
    unlink_no_follow(str(path))


@pytest.mark.parametrize("mode", ["r", "w", "a", "r+", "w+", "a+"])
def test_links_preserve_outside(tmp_path, mode):
    outside = tmp_path / "outside"
    outside.write_bytes(b"untouched")
    link = tmp_path / "link"
    link.symlink_to(outside)
    with pytest.raises(OSError), open(link, mode, opener=open_no_follow):
        pass
    assert outside.read_bytes() == b"untouched"
    unlink_no_follow(str(link))
    assert not link.is_symlink()
    assert outside.read_bytes() == b"untouched"


def test_dangling_collision_and_parent_link(tmp_path):
    outside = tmp_path / "outside"
    link = tmp_path / "link"
    link.symlink_to(outside)
    with pytest.raises(FileExistsError):
        open_no_follow(str(link), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    assert save_new_file(str(tmp_path), "link", b"new").endswith("link_1")
    assert not outside.exists()
    unlink_no_follow(str(link))
    outside.mkdir()
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        open_no_follow(
            str(link / "child" / "file"), os.O_WRONLY | os.O_CREAT, make_parents=True
        )
    assert list(outside.iterdir()) == []
    unlink_no_follow(str(link))
    assert outside.is_dir()
