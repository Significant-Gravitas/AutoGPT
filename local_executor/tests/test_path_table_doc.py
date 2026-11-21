"""CROSS_PLATFORM.md's path-handling table says what the jail does.

Three of its rows described a jail that was never built, or one that had
since changed: that it compares case-folded on macOS and Windows (it never
folds case to let a path in), that the shim adds the `\\\\?\\` prefix for
long paths (the jail refuses every path that has it), and that reserved
names are refused only in FILE_WRITE (they are refused for every request
that names a path).
"""

from __future__ import annotations

import ntpath
import os
import sys
from pathlib import Path

import pytest

from autogpt_local_executor import path_jail
from autogpt_local_executor.path_jail import PathJailError, assert_inside_jail
from tests.test_path_jail_names import _Looks

ROOT = Path(__file__).resolve().parent.parent
_PIPE = "\x00pipe\x00"


def _rows(heading: str) -> dict[str, list[str]]:
    """The table under `heading` in CROSS_PLATFORM.md, by each row's first
    cell: its other cells, with an escaped `|` as a plain one."""
    document = (ROOT / "docs" / "CROSS_PLATFORM.md").read_text(encoding="utf-8")
    section = document[document.index(heading) :]
    rows: dict[str, list[str]] = {}
    for line in section.splitlines()[1:]:
        if line.startswith("#"):
            break
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [
            cell.strip().replace(_PIPE, "|")
            for cell in line.replace("\\|", _PIPE).strip("|").split("|")
        ]
        rows[cells[0]] = cells[1:]
    return rows


@pytest.fixture(scope="module")
def table() -> dict[str, list[str]]:
    rows = _rows("### Path handling")
    assert list(rows) == [
        "Dimension",
        "Path separator",
        "Drive letters / UNC",
        "Reserved filenames",
        "Invalid path chars",
        "Max path length",
        "Alternate data streams",
        "Case sensitivity (default FS)",
        "Home dir env var",
    ]
    return rows


def test_the_table_says_the_jail_never_folds_case(
    table: dict[str, list[str]], tmp_path: Path
) -> None:
    notes = table["Case sensitivity (default FS)"][-1]

    assert "never folds case" in notes
    assert "case-folded" not in notes
    assert "same device and inode" in notes

    # What it says, on whatever filesystem the tests run on: a differently
    # cased name beside the root is inside only if the filesystem says it is
    # the root.
    root = tmp_path / "Ws"
    root.mkdir()
    other = tmp_path / "WS"
    same_folder = other.exists() and os.path.samefile(root, other)
    if not same_folder:
        other.mkdir()
    if same_folder:
        assert assert_inside_jail(other / "file", root)
    else:
        with pytest.raises(PathJailError) as refused:
            assert_inside_jail(other / "file", root)
        assert refused.value.code == "PATH_OUTSIDE_ALLOWED_ROOT"


def test_the_table_says_the_long_path_prefix_is_refused_and_never_added(
    table: dict[str, list[str]],
) -> None:
    mac, linux, windows, notes = table["Max path length"]

    assert "transparently" not in notes
    assert "never adds the `\\\\?\\` prefix" in notes
    assert "refuses a path that has one" in notes
    # The jail's own limits are the row's numbers.
    limits = path_jail._PATH_MAX
    assert str(limits["darwin"]) in mac
    assert str(limits["linux"]) in linux and limits["wsl2"] == limits["linux"]
    assert str(limits["windows"]) in windows

    # A `\\?\` path is no path under any root, by its name alone.
    assert path_jail._named("\\\\?\\C:\\ws\\file.txt", True) is None
    assert path_jail._plainly_outside("\\\\?\\C:\\ws\\file.txt", ["C:\\ws"], pathmod=ntpath)


@pytest.mark.skipif(sys.platform != "win32", reason="the `\\\\?\\` prefix is Windows'")
def test_a_long_path_prefix_is_refused_on_windows(tmp_path: Path) -> None:
    with pytest.raises(PathJailError):
        assert_inside_jail("\\\\?\\" + str(tmp_path / "file.txt"), tmp_path)


def test_a_path_past_the_limit_is_refused_before_anything_looks_at_it(tmp_path: Path) -> None:
    """Its length alone refuses it: not the root, nor the path, is looked
    at. The code by itself would be the same from a jail that resolved the
    path first and then refused it."""
    limit = path_jail._PATH_MAX[path_jail.detect_platform()]

    with _Looks() as looks:
        with pytest.raises(PathJailError) as refused:
            assert_inside_jail("a" * (limit + 1), tmp_path)

    assert refused.value.code == "PATH_INVALID_CHARS"
    assert looks.seen == []


def test_the_table_says_reserved_names_are_refused_for_every_request(
    table: dict[str, list[str]],
) -> None:
    mac, linux, windows, notes = table["Reserved filenames"]

    assert "FILE_WRITE" not in notes.replace("FILE_*", "")
    assert "every request that names a path" in notes
    assert "command's `cwd`" in notes
    assert "`PATH_RESERVED_NAME`" in notes
    assert (mac, linux) == ("none", "none")
    for name in ("CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "COM1", "COM9", "LPT1", "LPT9"):
        assert f"`{name}`" in windows, name
        assert path_jail._is_windows_device_name(name.lower() + " .txt")
    for digit in "\u00b9\u00b2\u00b3":
        assert digit in windows
        assert path_jail._is_windows_device_name(f"com{digit}")


def test_the_table_lists_the_characters_the_jail_refuses(table: dict[str, list[str]]) -> None:
    mac, linux, windows, notes = table["Invalid path chars"]

    assert mac == linux == "NUL (`\\0`)"
    refused = [
        chr(code) for code in range(128) if path_jail._WINDOWS_INVALID_CHARS.search(chr(code))
    ]
    assert [character for character in refused if character.isprintable()] == list('"*<>?|')
    assert [ord(character) for character in refused if not character.isprintable()] == list(
        range(32)
    )
    listed = windows[: windows.index(" and control chars")]
    assert sorted(listed.replace("`", "").split()) == sorted('"*<>?|')
    assert "control chars 0–31" in windows
    assert "FILE_WRITE" not in notes
    assert "every request that names a path" in notes


def test_what_the_table_says_of_other_systems_characters_holds_here(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX's rule")
    with pytest.raises(PathJailError) as refused:
        assert_inside_jail("a\x00b", tmp_path)
    assert refused.value.code == "PATH_INVALID_CHARS"
    # A `\` and a `:` are ordinary characters of a name here.
    assert assert_inside_jail("a\\b:c", tmp_path) == Path(os.path.realpath(tmp_path)) / "a\\b:c"


def test_the_table_says_a_colon_is_refused_and_never_stripped(table: dict[str, list[str]]) -> None:
    notes = table["Alternate data streams"][-1]

    assert "refuses any path with a `:` after the drive letter" in notes
    assert "strips/" not in notes
    assert "never strips" in notes


def test_the_table_says_how_a_tilde_is_expanded(table: dict[str, list[str]]) -> None:
    notes = table["Home dir env var"][-1]

    assert "`Path.expanduser()`" in notes
    assert "Path.home()" not in notes


def test_the_table_says_how_a_wire_path_is_read(table: dict[str, list[str]]) -> None:
    notes = table["Path separator"][-1]

    assert "`pathlib.Path`" in notes
    assert "PurePath" not in notes
