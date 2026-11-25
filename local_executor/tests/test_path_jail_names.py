"""A path that is plainly outside the shared folder is refused by its name,
before the disk or the network is touched.

The jail resolved the requested path with os.path.realpath before it knew the
path was inside the root. On Windows resolving `\\\\host\\share\\a.txt` connects
to that host over SMB and signs in with the person's Windows account, and
`\\\\.\\pipe\\name` opens a device. On macOS `/net/<host>/...` has autofs
contact the host. One FILE_READ or FILE_STAT was enough.

The names are compared first now, and the resolved path is still what decides
for a path that passes: a link inside the folder is followed, to learn where
it points.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from autogpt_local_executor import path_jail
from autogpt_local_executor.path_jail import (
    PathJailError,
    assert_entry_inside_jail,
    assert_inside_jail,
)

# A host name that resolves nowhere (RFC 2606), for the paths that must
# never be looked at.
NOWHERE = "files.invalid"


def _outside(requested: str, spellings: list[str], pathmod: Any) -> bool:
    return path_jail._plainly_outside(requested, spellings, pathmod=pathmod)


# ── The names alone, with Windows paths, on every OS ─────────────────────────

DRIVE_ROOT = [r"C:\Users\me\proj"]
SHARE_ROOT = [r"\\srv\share\folder"]
# A mapped drive: as configured, and as Windows resolves it. The configured
# spelling is first: a relative path is taken from it.
MAPPED_ROOT = [r"Z:\ws", r"\\srv\share\ws"]
# As typed with an 8.3 name, and as resolved to the long one.
SHORT_NAME_ROOT = [r"C:\Users\RUNNER~1\ws", r"C:\Users\runneradmin\ws"]


@pytest.mark.parametrize(
    ("spellings", "requested"),
    [
        (DRIVE_ROOT, r"C:\Users\me\proj"),
        (DRIVE_ROOT, r"C:\Users\me\proj\a.txt"),
        (DRIVE_ROOT, r"C:\Users\me\proj\sub\..\a.txt"),
        (DRIVE_ROOT, r"C:\Users\me\proj\.\sub\\a.txt"),
        # Windows names a file whatever the case and either separator.
        (DRIVE_ROOT, "c:/users/ME/PROJ/a.txt"),
        (DRIVE_ROOT, r"c:\USERS\me\proj"),
        # Relative paths are taken from the root.
        (DRIVE_ROOT, "a.txt"),
        (DRIVE_ROOT, r"sub\deeper\..\a.txt"),
        (DRIVE_ROOT, "."),
        (DRIVE_ROOT, ""),
        # Relative on the root's own drive: joined to the root, as before.
        (DRIVE_ROOT, "C:a.txt"),
        (SHARE_ROOT, r"\\srv\share\folder"),
        (SHARE_ROOT, r"\\srv\share\folder\a.txt"),
        (SHARE_ROOT, "//srv/share/folder/a.txt"),
        (SHARE_ROOT, r"\\SRV\Share\FOLDER\a.txt"),
        (SHARE_ROOT, "a.txt"),
        (SHARE_ROOT, r"sub\..\a.txt"),
        # The root's own drive or share with no path after it is relative,
        # and nothing: the root itself, as before.
        (SHARE_ROOT, r"\\srv\share"),
        (DRIVE_ROOT, "C:"),
        (MAPPED_ROOT, "Z:"),
        (MAPPED_ROOT, r"Z:\ws\a.txt"),
        (MAPPED_ROOT, "z:/ws/a.txt"),
        (MAPPED_ROOT, r"\\srv\share\ws\a.txt"),
        (MAPPED_ROOT, "a.txt"),
        (SHORT_NAME_ROOT, r"C:\Users\RUNNER~1\ws\a.txt"),
        (SHORT_NAME_ROOT, r"C:\Users\runneradmin\ws\a.txt"),
    ],
)
def test_a_windows_path_under_a_spelling_of_the_root_goes_on_to_be_resolved(
    spellings: list[str], requested: str
) -> None:
    assert _outside(requested, spellings, ntpath) is False


@pytest.mark.parametrize(
    ("spellings", "requested"),
    [
        # Another drive, and a path relative to another drive's folder.
        (DRIVE_ROOT, r"D:\Users\me\proj\a.txt"),
        (DRIVE_ROOT, "D:a.txt"),
        (DRIVE_ROOT, "D:"),
        # Another place on the root's drive.
        (DRIVE_ROOT, r"C:\Users\me\proj2\a.txt"),
        (DRIVE_ROOT, r"C:\Users\me"),
        (DRIVE_ROOT, r"C:\Windows\System32\config\SAM"),
        (DRIVE_ROOT, r"\Windows\System32"),
        (DRIVE_ROOT, "/Windows/System32"),
        (DRIVE_ROOT, r"C:\Users\me\proj\..\other\a.txt"),
        (DRIVE_ROOT, r"C:\Users\me\proj\sub\..\..\..\a.txt"),
        (DRIVE_ROOT, r"..\other\a.txt"),
        (DRIVE_ROOT, r"sub\..\..\a.txt"),
        (DRIVE_ROOT, ".."),
        # A share, by either separator: Windows would sign in to the host.
        (DRIVE_ROOT, rf"\\{NOWHERE}\share\a.txt"),
        (DRIVE_ROOT, f"//{NOWHERE}/share/a.txt"),
        (DRIVE_ROOT, rf"\/{NOWHERE}/share/a.txt"),
        (DRIVE_ROOT, rf"\\{NOWHERE}\share"),
        (DRIVE_ROOT, rf"\\{NOWHERE}"),
        (DRIVE_ROOT, rf"\\\{NOWHERE}\share\a.txt"),
        # The device namespaces, the root's own drive in them included.
        (DRIVE_ROOT, r"\\.\pipe\name"),
        (DRIVE_ROOT, "//./pipe/name"),
        (DRIVE_ROOT, r"\\.\C:\Users\me\proj\a.txt"),
        (DRIVE_ROOT, r"\\?\C:\Users\me\proj\a.txt"),
        (DRIVE_ROOT, rf"\\?\UNC\{NOWHERE}\share\a.txt"),
        (DRIVE_ROOT, r"\\.\PhysicalDrive0"),
        (DRIVE_ROOT, r"\\."),
        # A root on a share takes its own paths only.
        (SHARE_ROOT, r"\\srv\other\folder\a.txt"),
        (SHARE_ROOT, rf"\\{NOWHERE}\share\folder\a.txt"),
        (SHARE_ROOT, r"\\srv\share\folder2\a.txt"),
        (SHARE_ROOT, r"\\srv\share\a.txt"),
        (SHARE_ROOT, "\\\\srv\\share\\"),
        (SHARE_ROOT, r"\\srv\share\folder\..\a.txt"),
        (SHARE_ROOT, r"\\srv\share\folder\..\..\..\a.txt"),
        (SHARE_ROOT, r"..\a.txt"),
        (SHARE_ROOT, r"C:\Users\me\a.txt"),
        (SHARE_ROOT, r"\\?\UNC\srv\share\folder\a.txt"),
        (MAPPED_ROOT, r"Z:\other\a.txt"),
        (MAPPED_ROOT, r"Y:\ws\a.txt"),
        (MAPPED_ROOT, "Y:a.txt"),
        (MAPPED_ROOT, r"\\srv\share\other\a.txt"),
        (MAPPED_ROOT, r"\\srv\other\ws\a.txt"),
        # The share the mapped drive stands for, named without a path: no
        # part of the configured root, and not under the resolved one.
        (MAPPED_ROOT, r"\\srv\share"),
    ],
)
def test_a_windows_path_that_is_plainly_outside_the_root_is_told_by_its_name(
    spellings: list[str], requested: str
) -> None:
    assert _outside(requested, spellings, ntpath) is True


# ── The names alone, with POSIX paths, on every OS ───────────────────────────

POSIX_ROOT = ["/private/tmp/ws", "/tmp/ws"]


@pytest.mark.parametrize(
    "requested",
    [
        "/private/tmp/ws",
        "/private/tmp/ws/a.txt",
        "/tmp/ws/a.txt",
        "/tmp/ws/sub/../a.txt",
        "/tmp/ws/./sub//a.txt",
        "//tmp/ws/a.txt",
        "a.txt",
        "sub/../a.txt",
        ".",
        "",
        # Another case, or another way to write an accented letter: a
        # filesystem may take either for the same folder, and the resolved
        # path decides.
        "/TMP/WS/a.txt",
        "/tmp/ws/caf\u00e9/a.txt",
        "/tmp/ws/cafe\u0301/a.txt",
    ],
)
def test_a_posix_path_under_a_spelling_of_the_root_goes_on_to_be_resolved(
    requested: str,
) -> None:
    assert _outside(requested, POSIX_ROOT, posixpath) is False


@pytest.mark.parametrize(
    "requested",
    [
        "/etc/passwd",
        "/",
        "/tmp",
        "/tmp/ws2/a.txt",
        "/private/tmp/wsx",
        f"/net/{NOWHERE}/share/a.txt",
        f"//net/{NOWHERE}/share/a.txt",
        "/tmp/ws/../other/a.txt",
        "/tmp/ws/sub/../../other/a.txt",
        # Out through `..` and on to a folder that mounts a host when looked at.
        f"/tmp/ws/../../net/{NOWHERE}/a.txt",
        "../other/a.txt",
        "sub/../../a.txt",
        "..",
        f"../../net/{NOWHERE}/a.txt",
    ],
)
def test_a_posix_path_that_is_plainly_outside_the_root_is_told_by_its_name(
    requested: str,
) -> None:
    assert _outside(requested, POSIX_ROOT, posixpath) is True


# ── A path that leaves the root and comes back ───────────────────────────────


@pytest.mark.parametrize(
    "requested",
    [
        # Through a folder that mounts a host when it is looked at.
        f"/net/{NOWHERE}/../../tmp/ws/a.txt",
        f"/net/{NOWHERE}/share/../../../tmp/ws/a.txt",
        "/etc/../tmp/ws/a.txt",
        # Above the root and back down into it, by either spelling.
        "/tmp/ws/../ws/a.txt",
        "/tmp/ws/sub/../../ws/a.txt",
        "/tmp/ws/../../private/tmp/ws/a.txt",
        "../ws/a.txt",
        "sub/../../ws/a.txt",
        "/tmp/../tmp/ws/a.txt",
    ],
)
def test_a_posix_path_that_leaves_the_root_on_its_way_is_outside_by_its_name(
    requested: str,
) -> None:
    """`..` was worked out on the names before the path was placed, so a
    path that went out and came back, as `/net/<host>/../../workspace/file`
    does for a root `/workspace`, came out as one inside the root. Resolving
    it then looked at each folder on the way, the one outside included:
    for `/net/<host>` that is a connection to the host."""
    assert _outside(requested, POSIX_ROOT, posixpath) is True


@pytest.mark.parametrize(
    "requested",
    [
        "/tmp/ws/sub/../a.txt",
        "/tmp/ws/sub/deeper/../../a.txt",
        "/tmp/ws/sub/..",
        "/private/tmp/ws/sub/../a.txt",
        "sub/../a.txt",
        "sub/deeper/../../a.txt",
        "./sub/./../a.txt",
    ],
)
def test_a_posix_path_whose_dots_stay_inside_the_root_goes_on_to_be_resolved(
    requested: str,
) -> None:
    assert _outside(requested, POSIX_ROOT, posixpath) is False


@pytest.mark.parametrize(
    ("spellings", "requested"),
    [
        (DRIVE_ROOT, r"C:\Windows\..\Users\me\proj\a.txt"),
        (DRIVE_ROOT, r"C:\Users\me\proj\..\proj\a.txt"),
        (DRIVE_ROOT, r"C:\Users\me\proj\sub\..\..\proj\a.txt"),
        (DRIVE_ROOT, r"..\proj\a.txt"),
        (DRIVE_ROOT, "c:/users/../Users/me/proj/a.txt"),
        (SHARE_ROOT, r"\\srv\share\other\..\folder\a.txt"),
        (SHARE_ROOT, r"\\srv\share\folder\..\folder\a.txt"),
        (MAPPED_ROOT, r"Z:\ws\..\ws\a.txt"),
    ],
)
def test_a_windows_path_that_leaves_the_root_on_its_way_is_outside_by_its_name(
    spellings: list[str], requested: str
) -> None:
    assert _outside(requested, spellings, ntpath) is True


def test_the_top_of_a_disk_as_the_root_has_nothing_above_it() -> None:
    """`..` at the top is the top, on every OS."""
    assert _outside("/../etc/hosts", ["/"], posixpath) is False
    assert _outside("../../etc", ["/"], posixpath) is False
    assert _outside(r"C:\..\Users", ["C:\\"], ntpath) is False
    assert _outside(r"D:\..\Users", ["C:\\"], ntpath) is True


def test_the_root_written_another_way_matches_one_letter_at_a_time() -> None:
    """Case and the form of an accented letter are set aside, and nothing
    else: a name that only starts like the root's is another folder."""
    nfc = unicodedata.normalize("NFC", "/Users/me/Caf\u00e9")
    nfd = unicodedata.normalize("NFD", nfc)

    assert _outside(nfd + "/a.txt", [nfc], posixpath) is False
    assert _outside(nfc.upper() + "/a.txt", [nfd], posixpath) is False
    assert _outside("/Users/me/Cafe/a.txt", [nfc], posixpath) is True
    assert _outside("/Users/me/Caf\u00e9s/a.txt", [nfc], posixpath) is True


# ── Device names ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "CON",
        "con",
        "nul.txt",
        "NUL .txt",
        "aux ",
        "PRN.tar.gz",
        "COM1",
        "lpt9.log",
        "CONIN$",
        "conout$",
        "COM\u00b9",
        "lpt\u00b3.txt",
    ],
)
def test_a_windows_device_name_is_one_in_any_folder_and_with_any_extension(name: str) -> None:
    assert path_jail._is_windows_device_name(name) is True


@pytest.mark.parametrize(
    "name", ["console", "com10", "nul2", "a.nul", "connection.txt", "lpt", "com", "$conin"]
)
def test_a_name_that_only_looks_like_a_device_is_a_file(name: str) -> None:
    assert path_jail._is_windows_device_name(name) is False


# ── The real jail: nothing but the root is looked at for a refused path ──────


class _Looks:
    """While entered, fails the test when anything looks at a path other
    than the ones the jail may: the root and the home folder, however
    spelled, and the folders above them, which resolving those two reads.
    It is left before the test ends, so that pytest reads its own files
    unwatched."""

    WATCHED = (
        (os.path, "realpath"),
        (os, "stat"),
        (os, "lstat"),
        (os, "open"),
        (os, "readlink"),
        (os, "scandir"),
        (os, "listdir"),
        (os, "access"),
    )

    def __init__(self, *allowed: os.PathLike[str] | str) -> None:
        self.allowed = {self._key(os.fspath(path)) for path in allowed}
        self.allowed |= {self._key(os.path.realpath(path)) for path in allowed}
        self.seen: list[str] = []
        self._real: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> _Looks:
        for owner, name in self.WATCHED:
            real = getattr(owner, name)
            self._real.append((owner, name, real))
            setattr(owner, name, self._watched(name, real))
        return self

    def __exit__(self, *_exc: object) -> None:
        for owner, name, real in self._real:
            setattr(owner, name, real)
        self._real.clear()

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.normpath(path))

    def _may_look_at(self, path: str) -> bool:
        key = self._key(path)
        return any(
            key == allowed or allowed.startswith(key.rstrip(os.sep) + os.sep)
            for allowed in self.allowed
        )

    def _watched(self, name: str, real: Any) -> Any:
        def watched(path: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(path, str | bytes | os.PathLike):
                text = os.fsdecode(path)
                self.seen.append(f"{name}({text!r})")
                if not self._may_look_at(text):
                    raise AssertionError(f"the jail looked at a refused path: {name}({text!r})")
            return real(path, *args, **kwargs)

        return watched


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("inside", encoding="utf-8")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("not yours", encoding="utf-8")
    return root


def _refused_everywhere(root: Path) -> Iterator[str]:
    """Paths outside `root`, as a request might write them."""
    above = root.parent
    yield str(above / "outside" / "secret.txt")
    yield str(above / "outside")
    yield str(above)
    yield str(above / "root2" / "a.txt")
    yield str(root) + "2"
    yield os.path.join(str(root), "..", "outside", "secret.txt")
    yield os.path.join(str(root), "sub", "..", "..", "outside", "secret.txt")
    yield os.path.join("..", "outside", "secret.txt")
    yield os.path.join("sub", "..", "..", "outside", "secret.txt")
    yield ".."


REFUSED_ON_POSIX = [
    "/etc/passwd",
    "/",
    f"/net/{NOWHERE}/share/a.txt",
    f"//net/{NOWHERE}/share/a.txt",
    f"/Volumes/{NOWHERE}/a.txt",
]
REFUSED_ON_WINDOWS = [
    rf"\\{NOWHERE}\share\a.txt",
    f"//{NOWHERE}/share/a.txt",
    rf"\\{NOWHERE}\share",
    r"\\.\pipe\autogpt-shim-test",
    "//./pipe/autogpt-shim-test",
    r"\\.\PhysicalDrive0",
    r"\Windows\System32\drivers\etc\hosts",
]


def _assert_refused_unseen(requested: str, root: Path, code: str) -> None:
    with _Looks(root, os.path.expanduser("~")) as looks:
        with pytest.raises(PathJailError) as refused:
            assert_inside_jail(requested, root)
        with pytest.raises(PathJailError):
            assert_entry_inside_jail(os.path.join(requested, "entry"), root)

    assert refused.value.code == code, (requested, looks.seen)
    # The answer names the root, and no place the request's path leads to.
    assert "resolved to" not in refused.value.message


def test_a_path_outside_the_root_is_refused_without_a_look_at_it(root: Path) -> None:
    for requested in _refused_everywhere(root):
        _assert_refused_unseen(requested, root, "PATH_OUTSIDE_ALLOWED_ROOT")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX paths")
@pytest.mark.parametrize("requested", REFUSED_ON_POSIX)
def test_a_posix_path_elsewhere_is_refused_without_a_look_at_it(root: Path, requested: str) -> None:
    _assert_refused_unseen(requested, root, "PATH_OUTSIDE_ALLOWED_ROOT")


def test_a_path_that_leaves_the_root_and_comes_back_is_refused_without_a_look_outside(
    root: Path,
) -> None:
    """Above the root and back down into it by its name. The jail took it
    for a path inside, which it resolves to, and resolved it: a look at
    each folder it passes through."""
    back = [
        os.path.join(str(root), "..", root.name, "a.txt"),
        os.path.join(str(root), "sub", "..", "..", root.name, "a.txt"),
        os.path.join("..", root.name, "a.txt"),
        os.path.join(str(root.parent), "outside", "..", root.name, "a.txt"),
    ]
    for requested in back:
        _assert_refused_unseen(requested, root, "PATH_OUTSIDE_ALLOWED_ROOT")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX paths")
def test_a_posix_path_through_another_host_and_back_is_refused_without_a_look_at_it(
    root: Path,
) -> None:
    up = "/".join([".."] * 3)
    for requested in (
        f"/net/{NOWHERE}/share/{up}{root}/a.txt",
        f"/Volumes/{NOWHERE}/../..{root}/a.txt",
    ):
        _assert_refused_unseen(requested, root, "PATH_OUTSIDE_ALLOWED_ROOT")


def test_dots_that_stay_inside_the_root_are_still_resolved(root: Path) -> None:
    inside = os.path.join(str(root), "sub", "..", "a.txt")

    assert assert_inside_jail(inside, root) == Path(os.path.realpath(root)) / "a.txt"
    assert assert_inside_jail(os.path.join("sub", "..", "a.txt"), root) == (
        Path(os.path.realpath(root)) / "a.txt"
    )


def test_a_path_from_the_home_folder_is_refused_without_a_look_at_it(
    monkeypatch: pytest.MonkeyPatch, root: Path, tmp_path: Path
) -> None:
    """`~` is the home folder, which this root isn't in."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    for requested in ("~/secret.txt", "~", os.path.join("~", "..", "outside", "secret.txt")):
        _assert_refused_unseen(requested, root, "PATH_OUTSIDE_ALLOWED_ROOT")


windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows paths")


@windows_only
@pytest.mark.parametrize("requested", REFUSED_ON_WINDOWS)
def test_a_windows_path_elsewhere_is_refused_without_a_look_at_it(
    root: Path, requested: str
) -> None:
    _assert_refused_unseen(requested, root, "PATH_OUTSIDE_ALLOWED_ROOT")


@windows_only
def test_another_drive_is_refused_without_a_look_at_it(root: Path) -> None:
    here = os.path.splitdrive(os.path.realpath(root))[0]
    other = "Q:" if here.upper() != "Q:" else "R:"

    for requested in (rf"{other}\a.txt", f"{other}a.txt", other, f"{other.lower()}/x/a.txt"):
        _assert_refused_unseen(requested, root, "PATH_OUTSIDE_ALLOWED_ROOT")


@windows_only
@pytest.mark.parametrize("name", ["CONIN$", "conout$", "NUL .txt", "COM\u00b9", "nul"])
def test_a_device_name_inside_the_root_is_refused_without_a_look_at_it(
    root: Path, name: str
) -> None:
    with _Looks(root, os.path.expanduser("~")) as looks:
        with pytest.raises(PathJailError) as refused:
            assert_inside_jail(root / name, root)

    assert refused.value.code == "PATH_RESERVED_NAME", looks.seen
    assert looks.seen == []


# ── Paths inside the root keep working ───────────────────────────────────────


def test_a_path_inside_the_root_is_resolved_as_before(root: Path) -> None:
    resolved = Path(os.path.realpath(root))

    assert assert_inside_jail(root / "a.txt", root) == resolved / "a.txt"
    assert assert_inside_jail("a.txt", root) == resolved / "a.txt"
    assert assert_inside_jail(os.path.join("sub", "..", "a.txt"), root) == resolved / "a.txt"
    assert assert_inside_jail(".", root) == resolved
    assert assert_inside_jail(root, root) == resolved
    assert assert_inside_jail(root / "not" / "made" / "yet.txt", root) == (
        resolved / "not" / "made" / "yet.txt"
    )


def _link_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this account can't make symlinks: {exc}")


def test_the_root_is_taken_as_configured_and_as_resolved(root: Path, tmp_path: Path) -> None:
    """A root reached through a link (`/tmp` on macOS, a home folder that is
    a link): the platform echoes the root as the shim reports it, and a
    command's output names it as resolved."""
    configured = tmp_path / "link-to-root"
    _link_or_skip(configured, root)
    resolved = Path(os.path.realpath(root))

    assert assert_inside_jail(configured / "a.txt", configured) == resolved / "a.txt"
    assert assert_inside_jail(resolved / "a.txt", configured) == resolved / "a.txt"
    assert assert_inside_jail("a.txt", configured) == resolved / "a.txt"
    with pytest.raises(PathJailError):
        assert_inside_jail(tmp_path / "outside" / "secret.txt", configured)


def test_the_root_is_taken_by_the_home_folders_own_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`~` is the home folder's own name, which may be a link, while the root
    the folder browser gave is the resolved one."""
    real_home = tmp_path / "volumes" / "homes" / "me"
    (real_home / "proj").mkdir(parents=True)
    (real_home / "proj" / "a.txt").write_text("inside", encoding="utf-8")
    home = tmp_path / "home-link"
    _link_or_skip(home, real_home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    root = Path(os.path.realpath(real_home / "proj"))

    assert assert_inside_jail(os.path.join("~", "proj", "a.txt"), root) == root / "a.txt"
    assert assert_inside_jail(home / "proj" / "a.txt", root) == root / "a.txt"
    with pytest.raises(PathJailError):
        assert_inside_jail(os.path.join("~", "other", "a.txt"), root)


def test_a_link_inside_the_root_is_followed_and_the_resolved_path_decides(
    root: Path, tmp_path: Path
) -> None:
    """Its name is inside the root, so it is resolved: that is how the jail
    learns where it points."""
    _link_or_skip(root / "inward", root / "sub")
    _link_or_skip(root / "outward", tmp_path / "outside")
    resolved = Path(os.path.realpath(root))

    assert assert_inside_jail(root / "inward" / "new.txt", root) == resolved / "sub" / "new.txt"
    with pytest.raises(PathJailError) as refused:
        assert_inside_jail(root / "outward" / "secret.txt", root)
    assert refused.value.code == "PATH_OUTSIDE_ALLOWED_ROOT"


@windows_only
def test_the_root_is_one_folder_whatever_its_case_or_separators(root: Path) -> None:
    resolved = Path(os.path.realpath(root))

    for spelling in (str(root).upper(), str(root).lower(), str(root).replace("\\", "/")):
        assert assert_inside_jail(os.path.join(spelling, "a.txt"), root) == resolved / "a.txt"
        assert assert_inside_jail(root / "a.txt", spelling) == resolved / "a.txt"


@windows_only
def test_a_root_given_by_its_short_name_takes_paths_by_that_name_and_the_long_one(
    root: Path,
) -> None:
    """A root typed with 8.3 names (`%TEMP%` often has them) is reported
    that way, so the platform sends paths that way, and a command's output
    names the long form the root resolves to. A root given by its long name
    isn't reached by short ones: the platform takes a path only under the
    root as reported, so it sends none."""
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(root), buffer, len(buffer)):
        pytest.skip("this folder's short name can't be read")
    short = buffer.value
    if short.casefold() == str(root).casefold():
        pytest.skip("this volume makes no 8.3 names")
    resolved = Path(os.path.realpath(root))

    assert Path(os.path.realpath(short)) == resolved
    assert assert_inside_jail(os.path.join(short, "a.txt"), short) == resolved / "a.txt"
    assert assert_inside_jail(resolved / "a.txt", short) == resolved / "a.txt"
    assert assert_inside_jail("a.txt", short) == resolved / "a.txt"
    with pytest.raises(PathJailError) as refused:
        assert_inside_jail(os.path.join(short, "a.txt"), root)
    assert refused.value.code == "PATH_OUTSIDE_ALLOWED_ROOT"


@windows_only
def test_a_root_on_a_share_takes_its_own_paths_and_no_other_shares(root: Path) -> None:
    """This computer's own administrative share stands in for a file
    server: the root is on \\\\localhost\\<drive>$."""
    drive, rest = os.path.splitdrive(os.path.realpath(root))
    share = rf"\\localhost\{drive[0]}$"
    on_share = share + rest
    if not os.path.isdir(on_share):
        pytest.skip(f"{share} isn't open to this account")
    resolved = Path(os.path.realpath(on_share))

    assert assert_inside_jail(on_share + r"\a.txt", on_share) == resolved / "a.txt"
    assert assert_inside_jail("a.txt", on_share) == resolved / "a.txt"
    assert assert_inside_jail(on_share.replace("\\", "/") + "/sub", on_share) == resolved / "sub"
    for elsewhere in (
        r"\\localhost\ADMIN$\win.ini",
        rf"\\127.0.0.1\{drive[0]}$" + rest + r"\a.txt",
        rf"\\{NOWHERE}\{drive[0]}$" + rest + r"\a.txt",
        share + r"\Windows\win.ini",
        on_share + r"\..\outside\secret.txt",
    ):
        with _Looks(on_share, os.path.expanduser("~")) as looks:
            with pytest.raises(PathJailError) as refused:
                assert_inside_jail(elsewhere, on_share)
        assert refused.value.code == "PATH_OUTSIDE_ALLOWED_ROOT", (elsewhere, looks.seen)


# ── A root that is itself written with dots ──────────────────────────────────


def test_a_root_configured_with_dots_takes_the_paths_written_under_it(tmp_path: Path) -> None:
    """`--allowed-root /work/staging/../shared` is kept as it was given, and
    HELLO reports it that way, so the platform's paths begin with it. The
    name check took each `..` of a path in its place but compared with the
    root's names worked out, so every such path was outside by its name."""
    (tmp_path / "staging").mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "notes.txt").write_text("here", encoding="utf-8")
    root = os.path.join(str(tmp_path), "staging", "..", "shared")

    assert assert_inside_jail(os.path.join(root, "notes.txt"), root) == (
        Path(os.path.realpath(shared)) / "notes.txt"
    )
    assert assert_inside_jail("notes.txt", root) == Path(os.path.realpath(shared)) / "notes.txt"
    # By the root's other spellings too, as before.
    assert assert_inside_jail(str(shared / "notes.txt"), root)
    # And what leaves it is still outside by its name. Resolving the root
    # itself passes through `staging`, as it is written; nothing else may be
    # looked at.
    (tmp_path / "staging" / "x").write_text("not shared", encoding="utf-8")
    for outside in (
        os.path.join(root, "..", "staging", "x"),
        os.path.join(str(tmp_path), "staging", "x"),
        os.path.join(str(tmp_path), "staging", "..", "shared2", "x"),
    ):
        with _Looks(root, tmp_path / "staging", os.path.expanduser("~")) as looks:
            with pytest.raises(PathJailError) as refused:
                assert_inside_jail(outside, root)
        assert refused.value.code == "PATH_OUTSIDE_ALLOWED_ROOT", (outside, looks.seen)
        assert not any("staging/x" in look or "staging\\x" in look for look in looks.seen)


def test_a_root_written_with_dots_is_a_spelling_of_its_own() -> None:
    spellings = ["/work/staging/../shared", "/work/shared"]

    assert _outside("/work/staging/../shared/notes.txt", spellings, posixpath) is False
    assert _outside("notes.txt", spellings, posixpath) is False
    assert _outside("/work/shared/notes.txt", spellings, posixpath) is False
    assert _outside("/work/staging/../shared/../staging/x", spellings, posixpath) is True
    assert _outside("/work/staging/x", spellings, posixpath) is True
    assert _outside("/work/staging/../other/x", spellings, posixpath) is True
