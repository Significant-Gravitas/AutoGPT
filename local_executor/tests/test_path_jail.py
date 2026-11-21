"""Path jail tests — every attack scenario from CROSS_PLATFORM.md."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from autogpt_local_executor.path_jail import (
    PathJailError,
    assert_inside_jail,
    is_inside_jail,
    open_jail_file,
)

# ── Happy path ───────────────────────────────────────────────────────────────


def test_a_path_outside_the_root_leaves_no_trace_outside_it(tmp_path: Path) -> None:
    """The check only reads: it never writes into a refused path's directory,
    nor into the root."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(PathJailError) as exc:
        assert_inside_jail(outside / "secret.txt", root)
    assert_inside_jail(root / "notes.txt", root)

    assert exc.value.code == "PATH_OUTSIDE_ALLOWED_ROOT"
    assert list(outside.iterdir()) == []
    assert list(root.iterdir()) == []


def test_allows_file_inside_root(tmp_path: Path) -> None:
    file = tmp_path / "ok.txt"
    file.write_text("hi", encoding="utf-8")
    assert is_inside_jail(file, tmp_path)


def test_allows_nested_file_inside_root(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("hi", encoding="utf-8")
    assert is_inside_jail(nested, tmp_path)


def test_allows_root_itself(tmp_path: Path) -> None:
    assert is_inside_jail(tmp_path, tmp_path)


# ── CROSS_PLATFORM.md "What the algorithm catches" — six attacks ─────────────


def test_blocks_dotdot_traversal(tmp_path: Path) -> None:
    target = tmp_path / "child" / ".." / ".." / "etc" / "passwd"
    with pytest.raises(PathJailError) as exc:
        assert_inside_jail(target, tmp_path)
    assert exc.value.code == "PATH_OUTSIDE_ALLOWED_ROOT"


def test_blocks_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir(exist_ok=True)
    secret = outside / "secret.txt"
    secret.write_text("nope", encoding="utf-8")
    link = tmp_path / "link"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this platform/account")
    try:
        with pytest.raises(PathJailError) as exc:
            assert_inside_jail(link, tmp_path)
        assert exc.value.code == "PATH_OUTSIDE_ALLOWED_ROOT"
    finally:
        link.unlink()
        secret.unlink()
        outside.rmdir()


def test_allows_uppercased_on_case_insensitive_fs(tmp_path: Path) -> None:
    """When the FS is case-insensitive, /WORKSPACE and /workspace are the
    same directory. We allow either spelling for the jail check."""
    if sys.platform not in ("darwin", "win32"):
        pytest.skip("Only relevant on case-insensitive FS by default")
    # Use the parent's case-toggled spelling and a real subpath.
    nested = tmp_path / "x.txt"
    nested.write_text("hi", encoding="utf-8")
    upper_root = Path(str(tmp_path).upper())
    if not upper_root.exists() or not upper_root.samefile(tmp_path):
        # Case sensitivity is per volume (and per directory on NTFS), so the
        # platform alone doesn't say whether the other spelling names tmp_path.
        pytest.skip("case-sensitive filesystem: the uppercased root is not tmp_path")

    assert_inside_jail(nested, upper_root)


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory modes")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
def test_a_case_variant_sibling_of_a_read_only_root_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where directories are case-sensitive, `WS` beside a root `Ws` is another
    directory. A case probe that cannot write into a read-only root must not
    fall back to folding case, which would let `WS` in."""
    root = tmp_path / "Ws"
    root.mkdir()
    sibling = tmp_path / "WS"
    try:
        sibling.mkdir()
    except FileExistsError:
        pytest.skip("this filesystem ignores case, so WS and Ws are one directory")
    (sibling / "secret.txt").write_text("not yours", encoding="utf-8")
    # macOS and Windows are where a failed probe used to assume insensitivity.
    monkeypatch.setattr(sys, "platform", "darwin")
    root.chmod(0o555)
    try:
        with pytest.raises(PathJailError):
            assert_inside_jail(sibling / "secret.txt", root)
    finally:
        root.chmod(0o755)


def test_a_root_with_a_nul_byte_is_refused_rather_than_raising(tmp_path: Path) -> None:
    bad_root = str(tmp_path) + "\x00evil"

    assert is_inside_jail(tmp_path / "notes.txt", bad_root) is False
    with pytest.raises(PathJailError):
        assert_inside_jail(tmp_path / "notes.txt", bad_root)


def test_blocks_prefix_lookalike_dir(tmp_path: Path) -> None:
    """/workspace2 must NOT be considered inside /workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace2 = tmp_path / "workspace2"
    workspace2.mkdir()
    file_in_lookalike = workspace2 / "evil.txt"
    file_in_lookalike.write_text("nope", encoding="utf-8")
    with pytest.raises(PathJailError) as exc:
        assert_inside_jail(file_in_lookalike, workspace)
    assert exc.value.code == "PATH_OUTSIDE_ALLOWED_ROOT"


@pytest.mark.skipif(sys.platform != "win32", reason="ADS only on Windows")
def test_blocks_alternate_data_stream(tmp_path: Path) -> None:
    bad = f"{tmp_path}\\file.txt:hidden"
    with pytest.raises(PathJailError) as exc:
        assert_inside_jail(bad, tmp_path)
    assert exc.value.code in ("PATH_INVALID_CHARS", "PATH_OUTSIDE_ALLOWED_ROOT")


@pytest.mark.skipif(sys.platform != "win32", reason="Reserved names only on Windows")
def test_blocks_windows_reserved_name(tmp_path: Path) -> None:
    bad = tmp_path / "CON"
    with pytest.raises(PathJailError) as exc:
        assert_inside_jail(bad, tmp_path)
    assert exc.value.code == "PATH_RESERVED_NAME"


@pytest.mark.skipif(sys.platform != "win32", reason="UNC prefix only on Windows")
def test_blocks_unc_long_path_escape(tmp_path: Path) -> None:
    r"""\\?\<drive>\..\..\etc should still fail the prefix check."""
    bad = f"\\\\?\\{tmp_path}\\..\\..\\Windows\\System32\\cmd.exe"
    with pytest.raises(PathJailError):
        assert_inside_jail(bad, tmp_path)


# ── Additional invariants ────────────────────────────────────────────────────


def test_assert_returns_resolved_path(tmp_path: Path) -> None:
    file = tmp_path / "ok.txt"
    file.write_text("hi", encoding="utf-8")
    resolved = assert_inside_jail(file, tmp_path)
    assert resolved.exists()


def test_missing_root_raises(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does-not-exist"
    file = tmp_path / "x.txt"
    file.write_text("hi", encoding="utf-8")
    with pytest.raises(PathJailError):
        assert_inside_jail(file, nonexistent)


def test_null_byte_in_path_rejected(tmp_path: Path) -> None:
    bad = str(tmp_path / "evil\x00.txt")
    with pytest.raises(PathJailError) as exc:
        assert_inside_jail(bad, tmp_path)
    assert exc.value.code in ("PATH_INVALID_CHARS", "PATH_OUTSIDE_ALLOWED_ROOT")


@pytest.mark.parametrize("write", [False, True])
def test_content_open_rejects_directory_before_opening(tmp_path: Path, write: bool) -> None:
    with patch("autogpt_local_executor.path_jail.os.open", wraps=os.open) as open_file:
        with pytest.raises(PathJailError) as error:
            with open_jail_file(tmp_path, tmp_path, write=write):
                pytest.fail("Directory was opened for content access")

    assert error.value.code == "CAPABILITY_NOT_GRANTED"
    assert all(Path(call.args[0]) != tmp_path for call in open_file.call_args_list)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Named pipes require POSIX")
@pytest.mark.parametrize("write", [False, True])
def test_content_open_rejects_fifo_before_opening(tmp_path: Path, write: bool) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with patch("autogpt_local_executor.path_jail.os.open", wraps=os.open) as open_file:
        with pytest.raises(PathJailError) as error:
            with open_jail_file(fifo, tmp_path, write=write):
                pytest.fail("FIFO was opened for content access")

    assert error.value.code == "CAPABILITY_NOT_GRANTED"
    assert all(Path(call.args[0]) != fifo for call in open_file.call_args_list)


@pytest.mark.parametrize("write", [False, True])
def test_content_open_conservatively_rejects_inside_hard_links(tmp_path: Path, write: bool) -> None:
    first = tmp_path / "first.txt"
    first.write_text("original", encoding="utf-8")
    second = tmp_path / "second.txt"
    os.link(first, second)

    with pytest.raises(PathJailError) as error:
        with open_jail_file(second, tmp_path, write=write):
            pytest.fail("Multi-linked file was opened for content access")

    assert error.value.code == "PATH_OUTSIDE_ALLOWED_ROOT"
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("write", [False, True])
def test_content_descriptor_is_closed_after_rejection(tmp_path: Path, write: bool) -> None:
    target = tmp_path / "single.txt"
    target.write_text("original", encoding="utf-8")
    opened = []
    original_open = os.open

    def link_on_open(*args, **kwargs):
        if Path(args[0]) != target:
            return original_open(*args, **kwargs)
        os.link(target, tmp_path / "linked.txt")
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    with patch("autogpt_local_executor.path_jail.os.open", side_effect=link_on_open):
        with pytest.raises(PathJailError):
            with open_jail_file(target, tmp_path, write=write):
                pytest.fail("Newly linked file was opened for content access")

    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
    assert target.read_text(encoding="utf-8") == "original"


def test_a_relative_path_is_taken_from_the_root_not_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path resolved against the directory the shim started in:
    refused when that was outside the root, and another file than the
    platform meant (it joins a relative path to allowed_root) when that was
    a folder inside it."""
    from autogpt_local_executor.path_jail import assert_entry_inside_jail

    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    expected = Path(os.path.realpath(root)) / "notes.txt"

    monkeypatch.chdir(elsewhere)
    assert assert_inside_jail("notes.txt", root) == expected
    monkeypatch.chdir(root / "sub")
    assert assert_inside_jail("notes.txt", root) == expected
    assert assert_entry_inside_jail("notes.txt", root) == expected
    with pytest.raises(PathJailError) as exc:
        assert_inside_jail(os.path.join("..", "escape.txt"), root)
    assert exc.value.code == "PATH_OUTSIDE_ALLOWED_ROOT"
