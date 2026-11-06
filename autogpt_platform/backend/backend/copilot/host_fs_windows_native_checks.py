"""Windows-only native tests. Run explicitly on a Windows test worker."""

import ctypes as c
import os
import struct
import subprocess
import sys
from contextlib import ExitStack
from ctypes import wintypes as w
from unittest.mock import patch

import pytest

from backend.copilot import host_fs_windows as fs
from backend.copilot import windows_handles as wh
from backend.copilot import windows_private as private
from backend.copilot.host_fs import open_no_follow, unlink_no_follow
from backend.copilot.host_private_directory import ensure_private_directory

if sys.platform != "win32":
    raise ImportError("Native Windows filesystem checks require Windows")


@pytest.mark.parametrize(
    "tail", ["file:stream", "file.", "file ", "CON", "nul.txt", "COM¹", "bad?name"]
)
def test_ambiguous_names_fail(tmp_path, tail):
    with pytest.raises(OSError):
        open_no_follow(str(tmp_path / tail), os.O_CREAT | os.O_WRONLY)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "path",
    [r"\\server\share\file", r"\\?\C:\file", r"\\.\C:\file", r"C:file", r"\file"],
)
def test_namespaces_fail_closed(path):
    with pytest.raises((OSError, ValueError)):
        open_no_follow(path, os.O_RDONLY)


def test_junction_parent_and_final_leaf(tmp_path):
    target = tmp_path / "outside"
    target.mkdir()
    (target / "file").write_bytes(b"untouched")
    link = tmp_path / "junction"
    subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        check=True,
        capture_output=True,
        timeout=10,
    )
    try:
        with pytest.raises(OSError):
            open_no_follow(str(link / "file"), os.O_WRONLY | os.O_TRUNC)
        with pytest.raises(OSError):
            unlink_no_follow(str(link / "file"))
        unlink_no_follow(str(link))
        assert (target / "file").read_bytes() == b"untouched"
        assert not link.exists()
    finally:
        if link.exists():
            os.rmdir(link)


def test_parent_rename_blocked_during_relative_open(tmp_path):
    directory = tmp_path / "parent"
    directory.mkdir()
    path = directory / "file"
    real = wh.relative
    attempted = []

    def raced(parent, name, *args, **kwargs):
        if name == "file":
            with pytest.raises(PermissionError):
                directory.rename(tmp_path / "moved")
            attempted.append(True)
        return real(parent, name, *args, **kwargs)

    with patch.object(wh, "relative", raced):
        fd = open_no_follow(str(path), os.O_CREAT | os.O_WRONLY)
        os.write(fd, b"inside")
        os.close(fd)
    assert attempted == [True]
    assert path.read_bytes() == b"inside"
    directory.rename(tmp_path / "moved")


def test_leaf_swap_blocked_before_truncation(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.write_bytes(b"untouched")
    moved = tmp_path / "moved"
    real = wh.inspect

    def raced(handle, checked, *, directory):
        result = real(handle, checked, directory=directory)
        if not directory:
            with pytest.raises(PermissionError):
                path.rename(moved)
        return result

    with patch.object(wh, "inspect", raced):
        fd = open_no_follow(str(path), os.O_WRONLY | os.O_TRUNC)
        os.write(fd, b"new")
        os.close(fd)
    assert path.read_bytes() == b"new"
    assert outside.read_bytes() == b"untouched"
    path.unlink()


@pytest.mark.parametrize("stage", ["inspect", "transfer", "setmode", "inherit"])
def test_exception_releases_resources(tmp_path, stage):
    path = tmp_path / "file"
    path.write_bytes(b"before")
    targets = {
        "inspect": (wh, "inspect"),
        "transfer": (fs.msvcrt, "open_osfhandle"),
        "setmode": (fs.msvcrt, "setmode"),
        "inherit": (fs.os, "set_inheritable"),
    }
    with (
        patch.object(*targets[stage], side_effect=RuntimeError("injected")),
        pytest.raises(RuntimeError),
    ):
        open_no_follow(str(path), os.O_RDWR)
    path.unlink()
    tmp_path.rename(tmp_path.with_name(tmp_path.name + "-released"))
    tmp_path.with_name(tmp_path.name + "-released").rename(tmp_path)


def test_private_fresh_reuse_and_path_lifetime(tmp_path):
    path = str(tmp_path / "private")
    with ExitStack() as lifetime:
        assert ensure_private_directory(path, lifetime=lifetime) == path
        assert ensure_private_directory(path, lifetime=lifetime) == path
        with pytest.raises(PermissionError):
            os.rename(tmp_path, str(tmp_path) + "-moved")
        with pytest.raises(PermissionError):
            os.rename(path, path + "-moved")
        child = os.path.join(path, "child")
        os.mkdir(child)
        with open(os.path.join(child, "file"), "wb") as file:
            file.write(b"private")
    os.rename(path, path + "-moved")


@pytest.mark.parametrize("destination", ["leaf", "ancestor"])
@pytest.mark.parametrize("operation", ["rename", "replace", "overwrite", "same"])
def test_private_lifetime_allows_file_renames(tmp_path, destination, operation):
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    leaf = ancestor / "private"
    outside = tmp_path / "outside"
    outside.mkdir()
    target_dir = leaf if destination == "leaf" else ancestor
    with ExitStack() as lifetime:
        ensure_private_directory(str(leaf), lifetime=lifetime)
        source = (target_dir if operation == "same" else outside) / "source"
        target = target_dir / "target"
        source.write_bytes(b"source")
        if operation == "overwrite":
            target.write_bytes(b"old")
        if operation == "rename":
            os.rename(source, target)
        else:
            os.replace(source, target)
        assert target.read_bytes() == b"source"
        assert not source.exists()
        for pinned in (ancestor, leaf):
            with pytest.raises(PermissionError):
                pinned.rename(pinned.with_name(pinned.name + "-moved"))
    leaf.rename(ancestor / "released")


def test_private_retains_only_after_the_entire_chain_is_validated(tmp_path):
    path = tmp_path / "private"
    real_verify = private.verify_acl
    real_retain = wh.retain_directory
    initial_chain = []
    replacements = []
    validated_leaf = False

    def verify(handle, user, *, private):
        nonlocal validated_leaf
        result = real_verify(handle, user, private=private)
        if not replacements:
            initial_chain.append(handle)
        if private:
            validated_leaf = True
        return result

    def retain(handle, checked_path):
        assert validated_leaf
        assert handle in initial_chain
        with pytest.raises(PermissionError):
            path.rename(tmp_path / "moved")
        replacement = real_retain(handle, checked_path)
        replacements.append(replacement)
        return replacement

    with (
        patch.object(private, "verify_acl", side_effect=verify),
        patch.object(wh, "retain_directory", side_effect=retain),
        ExitStack() as lifetime,
    ):
        ensure_private_directory(str(path), lifetime=lifetime)
        assert len(replacements) == len(initial_chain)
    path.rename(tmp_path / "released")


def test_private_revalidates_reopened_acl(tmp_path):
    path = tmp_path / "private"
    real_retain = wh.retain_directory
    user = private.current_user_sid()

    def retain(handle, checked_path):
        set_fixture_acl(path, f"D:P(A;OICI;FA;;;{user})(A;OICI;FR;;;WD)")
        return real_retain(handle, checked_path)

    with (
        patch.object(wh, "retain_directory", side_effect=retain),
        ExitStack() as lifetime,
        pytest.raises(PermissionError, match="Outsider access"),
    ):
        ensure_private_directory(str(path), lifetime=lifetime)
    path.rename(tmp_path / "released")


@pytest.mark.parametrize(
    "stage", ["reopen", "first_identity", "second_identity", "identity_mismatch"]
)
def test_private_retain_failure_releases_both_handle_chains(tmp_path, stage):
    path = tmp_path / "private"
    real_retain = wh.retain_directory
    real_info = wh.get_info
    calls = 0

    def retain(handle, checked_path):
        nonlocal calls
        calls += 1
        if stage == "reopen" and calls == 2:
            raise OSError("injected reopen failure")
        return real_retain(handle, checked_path)

    identity_calls = 0

    def get_info(handle, info_class, info, size):
        nonlocal identity_calls
        if info_class == 18:
            identity_calls += 1
            if (stage == "first_identity" and identity_calls == 3) or (
                stage == "second_identity" and identity_calls == 4
            ):
                c.set_last_error(5)
                return False
        result = real_info(handle, info_class, info, size)
        if stage == "identity_mismatch" and info_class == 18 and identity_calls == 4:
            c.cast(info, c.POINTER(c.c_ubyte))[0] ^= 1
        return result

    with (
        patch.object(wh, "retain_directory", side_effect=retain),
        patch.object(wh, "get_info", side_effect=get_info),
        ExitStack() as lifetime,
        pytest.raises(OSError),
    ):
        ensure_private_directory(str(path), lifetime=lifetime)
    path.rename(tmp_path / "released")
    tmp_path.rename(tmp_path.with_name(tmp_path.name + "-released"))
    tmp_path.with_name(tmp_path.name + "-released").rename(tmp_path)


def test_file_operation_parent_still_denies_write_sharing(tmp_path):
    path = tmp_path / "parent"
    path.mkdir()
    with fs.parents(str(path / "file")):
        handle = wh.create_file(str(path), 2, 7, None, 3, 0x02200000, None)
        if handle != c.c_void_p(-1).value:
            wh.close(handle)
            pytest.fail("Ordinary file-operation parents must deny FILE_ADD_FILE")
        assert c.get_last_error() == 32


def set_fixture_acl(path, sddl, *, protected=True):
    descriptor = w.LPVOID()
    assert private.convert_sd(sddl, 1, c.byref(descriptor), None)
    set_security = wh.bind(
        private.advapi, "SetFileSecurityW", w.BOOL, w.LPCWSTR, w.DWORD, w.LPVOID
    )
    try:
        assert set_security(
            str(path), 4 | (0x80000000 if protected else 0x20000000), descriptor
        )
    finally:
        wh.local_free(descriptor)


@pytest.mark.parametrize("rights", ["FR", "FW", "FA"])
def test_private_rejects_outsider_acl_without_repair(tmp_path, rights):
    path = tmp_path / "private"
    path.mkdir()
    user = private.current_user_sid()
    set_fixture_acl(path, f"D:P(A;OICI;FA;;;{user})(A;OICI;{rights};;;WD)")
    before = path.stat()
    with ExitStack() as lifetime, pytest.raises(PermissionError):
        ensure_private_directory(str(path), lifetime=lifetime)
    assert path.stat().st_ino == before.st_ino
    with fs.parents(str(path), security=True) as (parent, name, _):
        handle = wh.relative(
            parent,
            name,
            wh.DIRECTORY_ACCESS | wh.READ_CONTROL,
            wh.FILE_OPEN,
            directory=True,
        )
        try:
            with pytest.raises(PermissionError):
                private.verify_acl(handle, user, private=True)
        finally:
            wh.close(handle)


def test_private_rejects_file_and_link(tmp_path):
    path = tmp_path / "private"
    path.write_bytes(b"untouched")
    with ExitStack() as lifetime, pytest.raises(OSError):
        ensure_private_directory(str(path), lifetime=lifetime)
    assert path.read_bytes() == b"untouched"
    path.unlink()
    target = tmp_path / "target"
    target.mkdir()
    path.symlink_to(target, target_is_directory=True)
    with ExitStack() as lifetime, pytest.raises(OSError):
        ensure_private_directory(str(path), lifetime=lifetime)
    path.unlink()
    assert target.is_dir()


def test_private_rejects_unprotected_dacl(tmp_path):
    parent = tmp_path / "parent"
    with ExitStack() as lifetime:
        ensure_private_directory(str(parent), lifetime=lifetime)
    path = parent / "private"
    path.mkdir()
    user = private.current_user_sid()
    set_fixture_acl(
        path, f"D:(A;OICI;FA;;;{user})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)", protected=False
    )
    with ExitStack() as lifetime, pytest.raises(PermissionError, match="inheritance"):
        ensure_private_directory(str(path), lifetime=lifetime)


def test_parent_swap_before_open_is_refused(tmp_path):
    directory = tmp_path / "parent"
    directory.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "file"
    target.write_bytes(b"untouched")
    real = wh.relative

    def raced(parent, name, *args, **kwargs):
        if name == "parent":
            directory.rmdir()
            directory.symlink_to(outside, target_is_directory=True)
        return real(parent, name, *args, **kwargs)

    with patch.object(wh, "relative", raced), pytest.raises(OSError):
        open_no_follow(str(directory / "file"), os.O_WRONLY | os.O_TRUNC)
    directory.unlink()
    assert target.read_bytes() == b"untouched"


def test_parent_write_handle_blocked_and_handles_not_inherited(tmp_path):
    get_flags = wh.bind(
        wh.kernel, "GetHandleInformation", w.BOOL, w.HANDLE, c.POINTER(w.DWORD)
    )
    with fs.parents(str(tmp_path / "file")) as (_, _, handles):
        for handle in handles:
            flags = w.DWORD()
            assert get_flags(handle, c.byref(flags))
            assert flags.value & 1 == 0
        writer = wh.create_file(str(tmp_path), 0x40000000, 7, None, 3, 0x02200000, None)
        if writer != c.c_void_p(-1).value:
            wh.close(writer)
            pytest.fail("parent admitted GENERIC_WRITE while pinned")
        assert c.get_last_error() == 32


def test_validation_precedes_truncation(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"untouched")
    real = wh.inspect

    def refuse(handle, checked, *, directory):
        if not directory:
            raise OSError("injected rejection")
        return real(handle, checked, directory=directory)

    with patch.object(wh, "inspect", refuse), pytest.raises(OSError):
        open_no_follow(str(path), os.O_WRONLY | os.O_TRUNC)
    assert path.read_bytes() == b"untouched"


def test_native_open_refuses_success_without_a_handle():
    with patch.object(wh, "nt_create", return_value=0), pytest.raises(OSError):
        wh.relative(0, "file", wh.READ_ATTRIBUTES, wh.FILE_OPEN)


def test_sid_conversion_refuses_success_without_text():
    with patch.object(private, "sid_string", return_value=True), pytest.raises(OSError):
        private.sid_text(None)


def test_acl_validation_refuses_success_without_an_entry(tmp_path):
    with (
        patch.object(private, "get_ace", return_value=True),
        pytest.raises(PermissionError),
    ):
        ensure_private_directory(str(tmp_path / "private"))
    assert not (tmp_path / "private").exists()


def test_posix_directory_helper_refuses_windows():
    from backend.copilot.host_fs import _open_dir

    with pytest.raises(NotImplementedError):
        _open_dir([], make_parents=False, path="C:\\unused")


@pytest.mark.parametrize("operation", ["truncate", "delete"])
def test_native_mutation_failure_is_reported(tmp_path, operation):
    path = tmp_path / "file"
    path.write_bytes(b"untouched")

    def denied(*args):
        c.set_last_error(5)
        return 0

    with patch.object(wh, "set_info", denied), pytest.raises(PermissionError):
        if operation == "truncate":
            open_no_follow(str(path), os.O_WRONLY | os.O_TRUNC)
        else:
            unlink_no_follow(str(path))
    assert path.read_bytes() == b"untouched"
    path.unlink()


def test_private_error_releases_entire_chain(tmp_path):
    path = tmp_path / "private"
    real = private.verify_acl

    def refuse(handle, user, *, private):
        if private:
            raise RuntimeError("injected")
        return real(handle, user, private=private)

    with (
        patch.object(private, "verify_acl", side_effect=refuse),
        ExitStack() as lifetime,
        pytest.raises(RuntimeError),
    ):
        ensure_private_directory(str(path), lifetime=lifetime)
    path.rmdir()
    os.rename(tmp_path, str(tmp_path) + "-moved")
    os.rename(str(tmp_path) + "-moved", tmp_path)


def test_private_rejects_writable_ancestor_even_while_held(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    user = private.current_user_sid()
    set_fixture_acl(parent, f"D:P(A;OICI;FA;;;{user})(A;;FW;;;WD)")
    with ExitStack() as lifetime, pytest.raises(PermissionError):
        ensure_private_directory(str(parent / "private"), lifetime=lifetime)
    assert not (parent / "private").exists()


def test_private_allows_readable_ancestor_but_not_readable_leaf(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    user = private.current_user_sid()
    set_fixture_acl(parent, f"D:P(A;OICI;FA;;;{user})(A;OICI;FR;;;WD)")
    with ExitStack() as lifetime:
        path = str(parent / "private")
        assert ensure_private_directory(path, lifetime=lifetime) == path
        assert ensure_private_directory(path, lifetime=lifetime) == path


def test_bounded_repetition_has_no_handle_growth(tmp_path):
    count = wh.bind(
        wh.kernel, "GetProcessHandleCount", w.BOOL, w.HANDLE, c.POINTER(w.DWORD)
    )
    before, after = w.DWORD(), w.DWORD()
    path = str(tmp_path / "file")
    assert count(w.HANDLE(-1), c.byref(before))
    for _ in range(40):
        fd = open_no_follow(path, os.O_CREAT | os.O_RDWR)
        os.close(fd)
        with ExitStack() as lifetime:
            ensure_private_directory(str(tmp_path / "private"), lifetime=lifetime)
    assert count(w.HANDLE(-1), c.byref(after))
    assert after.value <= before.value


def convert_fixture_to_junction(parent, outside, access):
    substitute = ("\\??\\" + str(outside)).encode("utf-16-le")
    display = str(outside).encode("utf-16-le")
    names = substitute + b"\0\0" + display + b"\0\0"
    data = (
        struct.pack(
            "<IHHHHHH",
            0xA0000003,
            8 + len(names),
            0,
            0,
            len(substitute),
            len(substitute) + 2,
            len(display),
        )
        + names
    )
    ioctl = wh.bind(
        wh.kernel,
        "DeviceIoControl",
        w.BOOL,
        w.HANDLE,
        w.DWORD,
        w.LPVOID,
        w.DWORD,
        w.LPVOID,
        w.DWORD,
        c.POINTER(w.DWORD),
        w.LPVOID,
    )

    handle = wh.create_file(str(parent), access, 7, None, 3, 0x02200000, None)
    if handle == c.c_void_p(-1).value:
        assert c.get_last_error() in (5, 32)
        return False
    try:
        returned = w.DWORD()
        return bool(
            ioctl(
                handle,
                0x900A4,
                c.create_string_buffer(data),
                len(data),
                None,
                0,
                c.byref(returned),
                None,
            )
        )
    finally:
        wh.close(handle)


def test_relative_open_refuses_converted_parent(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker").write_bytes(b"untouched")
    (outside / "file").write_bytes(b"untouched")

    def convert(access):
        return convert_fixture_to_junction(parent, outside, access)

    assert convert(0x40000000), "fixture must prove native junction conversion works"
    os.rmdir(parent)
    parent.mkdir()
    try:
        with fs.parents(str(parent / "file")) as (held, name, _):
            assert convert(0x100), "write-attributes conversion must exercise the race"
            with pytest.raises(OSError):
                wh.relative(held, name, wh.READ_ATTRIBUTES | 2, wh.FILE_OPEN)
        assert (outside / "marker").read_bytes() == b"untouched"
        assert (outside / "file").read_bytes() == b"untouched"
    finally:
        os.rmdir(parent)


def test_private_retain_refuses_converted_directory(tmp_path):
    path = tmp_path / "private"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker").write_bytes(b"untouched")
    real_verify = private.verify_acl
    real_retain = wh.retain_directory
    leaf_handle = None
    converted = False

    def verify(handle, user, *, private):
        nonlocal leaf_handle
        result = real_verify(handle, user, private=private)
        if private:
            leaf_handle = handle
        return result

    def retain(handle, checked_path):
        nonlocal converted
        if handle == leaf_handle:
            converted = convert_fixture_to_junction(path, outside, 0x100)
            assert converted
        return real_retain(handle, checked_path)

    try:
        with (
            patch.object(private, "verify_acl", side_effect=verify),
            patch.object(wh, "retain_directory", side_effect=retain),
            ExitStack() as lifetime,
            pytest.raises(OSError),
        ):
            ensure_private_directory(str(path), lifetime=lifetime)
        assert converted
        assert (outside / "marker").read_bytes() == b"untouched"
    finally:
        if converted:
            os.rmdir(path)
