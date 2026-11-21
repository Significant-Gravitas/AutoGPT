from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from autogpt_local_executor.directory_browser import (
    MAX_DIRECTORY_ENTRIES,
    MAX_REFS_PER_BROWSE,
    DirectoryBrowseError,
    DirectoryBrowser,
)


def _root_listing(browser: DirectoryBrowser):
    listing = browser.list_directories(None, None)
    assert listing.entries
    return listing, listing.entries[0]


def test_lists_only_immediate_real_directories(tmp_path: Path) -> None:
    (tmp_path / "beta").mkdir()
    (tmp_path / "Alpha").mkdir()
    (tmp_path / "file.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "link").symlink_to(tmp_path / "Alpha", target_is_directory=True)
    (tmp_path / "Alpha" / "nested").mkdir()
    browser = DirectoryBrowser(platform_name="linux", home=tmp_path)

    roots, home = _root_listing(browser)
    listing = browser.list_directories(roots.browse_id, home.directory_ref)

    assert [entry.name for entry in listing.entries] == ["Alpha", "beta"]
    assert all(Path(entry.path).is_absolute() for entry in listing.entries)
    assert "nested" not in [entry.name for entry in listing.entries]
    assert all(not hasattr(entry, "size_bytes") for entry in listing.entries)


def test_listing_is_bounded_and_paginated(tmp_path: Path) -> None:
    for index in range(MAX_DIRECTORY_ENTRIES + 10):
        (tmp_path / f"dir-{index:03d}").mkdir()
    browser = DirectoryBrowser(platform_name="linux", home=tmp_path)

    roots, home = _root_listing(browser)
    first = browser.list_directories(roots.browse_id, home.directory_ref)

    assert len(first.entries) == MAX_DIRECTORY_ENTRIES
    assert first.next_cursor is not None
    assert first.truncated is False

    second = browser.list_directories(
        roots.browse_id,
        home.directory_ref,
        first.next_cursor,
    )

    assert len(second.entries) == 10
    assert second.next_cursor is None
    assert {entry.directory_ref for entry in first.entries}.isdisjoint(
        entry.directory_ref for entry in second.entries
    )
    assert [entry.name for entry in (*first.entries, *second.entries)] == [
        f"dir-{index:03d}" for index in range(MAX_DIRECTORY_ENTRIES + 10)
    ]


def test_listing_a_directory_again_reuses_its_references(tmp_path: Path) -> None:
    """Each listing used to mint new references, so going back and forth
    spent the browse's reference budget and later listings came back empty."""
    for index in range(40):
        (tmp_path / f"dir-{index:02d}").mkdir()
    browser = DirectoryBrowser(platform_name="linux", home=tmp_path)
    roots, home = _root_listing(browser)

    first = browser.list_directories(roots.browse_id, home.directory_ref)
    for _ in range(MAX_REFS_PER_BROWSE // 40 + 1):
        again = browser.list_directories(roots.browse_id, home.directory_ref)

    assert again.truncated is False
    assert [entry.directory_ref for entry in again.entries] == [
        entry.directory_ref for entry in first.entries
    ]


def test_reference_expires_and_reset_invalidates_it(tmp_path: Path) -> None:
    clock = [100.0]
    browser = DirectoryBrowser(platform_name="linux", home=tmp_path, now=lambda: clock[0])
    roots, home = _root_listing(browser)

    clock[0] = roots.expires_at
    with pytest.raises(DirectoryBrowseError, match="invalid or expired"):
        browser.resolve(roots.browse_id, home.directory_ref)

    roots, home = _root_listing(browser)
    browser.reset()
    with pytest.raises(DirectoryBrowseError, match="invalid or expired"):
        browser.resolve(roots.browse_id, home.directory_ref)


def test_request_requires_reference_pair(tmp_path: Path) -> None:
    browser = DirectoryBrowser(platform_name="linux", home=tmp_path)

    with pytest.raises(DirectoryBrowseError, match="both be null or both be set"):
        browser.list_directories("browse", None)

    with pytest.raises(DirectoryBrowseError, match="cursor requires"):
        browser.list_directories(None, None, "cursor")


def test_consume_makes_selection_single_use(tmp_path: Path) -> None:
    browser = DirectoryBrowser(platform_name="linux", home=tmp_path)
    roots, home = _root_listing(browser)
    assert browser.resolve(roots.browse_id, home.directory_ref) == tmp_path.resolve()

    browser.consume(roots.browse_id)

    with pytest.raises(DirectoryBrowseError):
        browser.resolve(roots.browse_id, home.directory_ref)


@pytest.mark.parametrize("platform_name", ["darwin", "linux", "wsl2", "windows"])
def test_every_platform_includes_canonical_home(
    tmp_path: Path, platform_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("autogpt_local_executor.directory_browser._windows_drive_roots", lambda: [])
    browser = DirectoryBrowser(platform_name=platform_name, home=tmp_path)

    listing = browser.list_directories(None, None)

    assert str(tmp_path.resolve()) in {entry.path for entry in listing.entries}


@pytest.mark.parametrize(
    ("platform_name", "extra_name"),
    [("darwin", "DataVolume"), ("linux", "MediaMount"), ("wsl2", "c")],
)
def test_posix_platform_roots_include_safe_mounts(
    tmp_path: Path,
    platform_name: str,
    extra_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mount = tmp_path / extra_name
    mount.mkdir()
    browser = DirectoryBrowser(platform_name=platform_name, home=tmp_path)
    monkeypatch.setattr(browser, "_children_of", lambda _parent: [mount])

    listing = browser.list_directories(None, None)

    assert str(mount.resolve()) in {entry.path for entry in listing.entries}


def test_windows_roots_include_fixed_or_removable_drives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive = tmp_path / "drive"
    drive.mkdir()
    monkeypatch.setattr(
        "autogpt_local_executor.directory_browser._windows_drive_roots", lambda: [drive]
    )
    browser = DirectoryBrowser(platform_name="windows", home=tmp_path)

    listing = browser.list_directories(None, None)

    assert str(drive.resolve()) in {entry.path for entry in listing.entries}


@pytest.mark.parametrize(
    ("tag", "redirects"),
    [
        (0xA000000C, True),  # IO_REPARSE_TAG_SYMLINK
        (0xA0000003, True),  # IO_REPARSE_TAG_MOUNT_POINT (junction)
        (0x9000001A, False),  # IO_REPARSE_TAG_CLOUD (OneDrive placeholder)
        (0x9000601A, False),  # IO_REPARSE_TAG_CLOUD_6
        (0x80000017, False),  # IO_REPARSE_TAG_WOF (compressed file)
    ],
)
def test_only_reparse_points_that_redirect_are_refused(tag: int, redirects: bool) -> None:
    """Refusing every reparse point dropped OneDrive folders, whose cloud
    placeholders are reparse points that redirect nowhere."""
    from autogpt_local_executor.directory_browser import _reparse_tag_redirects

    assert _reparse_tag_redirects(tag) is redirects


def test_browsing_shows_a_cloud_folder_and_hides_a_junction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test above checks the tag rule alone, so browsing would still
    pass it if every reparse point were refused and OneDrive's folders were
    hidden again. Here the listing itself runs as on Windows."""
    import os
    from types import SimpleNamespace

    from autogpt_local_executor import directory_browser

    tags = {"cloud": 0x9000601A, "junction": 0xA0000003}  # CLOUD_6, MOUNT_POINT

    class WindowsOs:
        """os as the browser sees it on Windows, with those reparse tags."""

        name = "nt"

        def __getattr__(self, attribute: str) -> object:
            return getattr(os, attribute)

        def lstat(self, path: str | os.PathLike[str]) -> object:
            real = os.lstat(path)
            tag = tags.get(Path(path).name, 0)
            return SimpleNamespace(
                st_mode=real.st_mode,
                st_file_attributes=0x400 if tag else 0,
                st_reparse_tag=tag,
            )

    for name in ("cloud", "junction", "plain"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(directory_browser, "os", WindowsOs())
    browser = DirectoryBrowser(platform_name="linux", home=tmp_path)

    roots, home = _root_listing(browser)
    listing = browser.list_directories(roots.browse_id, home.directory_ref)

    assert [entry.name for entry in listing.entries] == ["cloud", "plain"]


def test_sibling_directories_that_differ_only_in_case_get_their_own_refs() -> None:
    """A Windows path compares without case, and refs were reused by path,
    so in a case-sensitive folder (one WSL uses, say) `foo` got the ref
    already minted for its sibling `Foo`, and choosing it attached the
    chat to `Foo`."""
    from pathlib import PureWindowsPath

    from autogpt_local_executor.directory_browser import _BrowseSession

    session = _BrowseSession(expires_at=0.0)
    upper = DirectoryBrowser._add_ref(session, PureWindowsPath(r"C:\Work\Foo"), "parent")  # type: ignore[arg-type]
    lower = DirectoryBrowser._add_ref(session, PureWindowsPath(r"C:\Work\foo"), "parent")  # type: ignore[arg-type]
    again = DirectoryBrowser._add_ref(session, PureWindowsPath(r"C:\Work\foo"), "parent")  # type: ignore[arg-type]

    assert upper != lower
    assert again == lower
    assert str(session.refs[lower].path) == r"C:\Work\foo"


# ── Windows system folders ───────────────────────────────────────────────────

WINDOWS_SYSTEM_FOLDERS = [
    "$RECYCLE.BIN",
    "Documents and Settings",
    "Program Files",
    "Program Files (x86)",
    "ProgramData",
    "Recovery",
    "System Volume Information",
    "Windows",
]
# The two that are the system's own wherever they are: every volume has
# them, a volume mounted in a folder among them.
SYSTEM_FOLDERS_ANYWHERE = ["$RECYCLE.BIN", "System Volume Information"]


def _children_listed(
    folder: Path, monkeypatch: pytest.MonkeyPatch, *, drive_top: bool
) -> list[str]:
    """What a Windows browser lists in `folder`, taken for a drive's top
    folder (`C:\\`) or for one further down."""
    monkeypatch.setattr("autogpt_local_executor.directory_browser._windows_drive_roots", lambda: [])
    monkeypatch.setattr(
        "autogpt_local_executor.directory_browser._is_drive_top",
        lambda path: drive_top and Path(path) == folder.resolve(),
    )
    browser = DirectoryBrowser(platform_name="windows", home=folder)
    roots, home = _root_listing(browser)
    listing = browser.list_directories(roots.browse_id, home.directory_ref)
    return [entry.name for entry in listing.entries]


def test_a_projects_folder_with_a_system_folders_name_is_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`C:\\src\\app\\windows`, which Flutter makes in every project, was
    hidden like `C:\\Windows`, and so was any folder called Recovery or
    ProgramData: the names were left out at every depth."""
    app = tmp_path / "src" / "app"
    for name in ["lib", "windows", *WINDOWS_SYSTEM_FOLDERS[:-1]]:
        (app / name).mkdir(parents=True)

    listed = _children_listed(app, monkeypatch, drive_top=False)

    expected = {"lib", "windows", *WINDOWS_SYSTEM_FOLDERS[:-1]} - set(SYSTEM_FOLDERS_ANYWHERE)
    assert sorted(listed, key=str.casefold) == sorted(expected, key=str.casefold)


def test_the_system_folders_at_a_drives_top_are_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive = tmp_path / "C"
    for name in ["Users", "src", *WINDOWS_SYSTEM_FOLDERS]:
        (drive / name).mkdir(parents=True)

    assert _children_listed(drive, monkeypatch, drive_top=True) == ["src", "Users"]


def test_a_drives_top_is_told_by_its_path() -> None:
    from pathlib import PureWindowsPath

    from autogpt_local_executor.directory_browser import _is_drive_top

    assert _is_drive_top(PureWindowsPath("C:\\")) is True
    assert _is_drive_top(PureWindowsPath("d:\\")) is True
    # A share's top folder is a volume's top too.
    assert _is_drive_top(PureWindowsPath("\\\\server\\share\\")) is True
    assert _is_drive_top(PureWindowsPath("C:\\Windows")) is False
    assert _is_drive_top(PureWindowsPath("C:\\src\\app")) is False
    assert _is_drive_top(PureWindowsPath("\\\\server\\share\\projects")) is False


@pytest.mark.skipif(sys.platform != "win32", reason="the real system drive")
def test_the_real_system_drive_lists_users_and_not_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive = Path(os.environ.get("SystemDrive", "C:") + "\\")
    monkeypatch.setattr(
        "autogpt_local_executor.directory_browser._windows_drive_roots", lambda: [drive]
    )
    browser = DirectoryBrowser(platform_name="windows", home=tmp_path)
    roots = browser.list_directories(None, None)
    [top] = [entry for entry in roots.entries if Path(entry.path) == drive]

    listed = {
        entry.name.casefold()
        for entry in browser.list_directories(roots.browse_id, top.directory_ref).entries
    }

    assert "users" in listed
    assert not listed & {name.casefold() for name in WINDOWS_SYSTEM_FOLDERS}


# ── Removable drives, and a home folder that is a link ───────────────────────


def _link_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this platform/account")


@pytest.mark.parametrize("platform_name", ["linux", "wsl2"])
def test_removable_drives_are_looked_for_under_the_login_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform_name: str
) -> None:
    """udisks mounts a person's disks in a folder named for their login.
    They were looked for under the home folder's name, which differs for a
    home like `/home/kris@corp` or an overridden HOME: no disk showed."""
    home = tmp_path / "kris@corp"
    home.mkdir()
    disk = tmp_path / "USB-DISK"
    disk.mkdir()
    looked_in: list[Path] = []

    def children_of(parent: Path) -> list[Path]:
        looked_in.append(parent)
        return [disk] if parent == Path("/run/media/kris") else []

    monkeypatch.setattr("autogpt_local_executor.directory_browser._login_name", lambda: "kris")
    browser = DirectoryBrowser(platform_name=platform_name, home=home)
    monkeypatch.setattr(browser, "_children_of", children_of)

    roots = browser.list_directories(None, None)

    assert str(disk.resolve()) in {entry.path for entry in roots.entries}
    # Both places distributions use, for the login name, and for the home
    # folder's name as before.
    assert {Path("/media/kris"), Path("/run/media/kris")} <= set(looked_in)
    assert {Path("/media/kris@corp"), Path("/run/media/kris@corp")} <= set(looked_in)


def test_a_login_name_that_is_the_home_folders_name_is_looked_under_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "kris"
    home.mkdir()
    looked_in: list[Path] = []
    monkeypatch.setattr("autogpt_local_executor.directory_browser._login_name", lambda: "kris")
    browser = DirectoryBrowser(platform_name="linux", home=home)
    monkeypatch.setattr(browser, "_children_of", lambda parent: looked_in.append(parent) or [])

    browser.list_directories(None, None)

    assert looked_in == [Path("/media/kris"), Path("/run/media/kris")]


def test_with_no_login_name_to_read_the_home_folders_name_still_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "kris"
    home.mkdir()
    looked_in: list[Path] = []
    monkeypatch.setattr("autogpt_local_executor.directory_browser._login_name", lambda: None)
    browser = DirectoryBrowser(platform_name="linux", home=home)
    monkeypatch.setattr(browser, "_children_of", lambda parent: looked_in.append(parent) or [])

    browser.list_directories(None, None)

    assert looked_in == [Path("/media/kris"), Path("/run/media/kris")]


def test_the_login_name_is_the_accounts_own() -> None:
    from autogpt_local_executor.directory_browser import _login_name

    if sys.platform == "win32":
        # No user database of that kind; nothing here looks for Linux mounts.
        assert _login_name() is None
        return
    import pwd

    try:
        expected: str | None = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        # An account the user database has no entry for (a container run
        # under a numeric ID): it has no login name, and the test's own
        # lookup failed here before it compared anything.
        expected = None
    assert _login_name() == expected


def test_an_account_the_user_database_does_not_know_has_no_login_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autogpt_local_executor.directory_browser import _login_name

    if sys.platform == "win32":
        pytest.skip("no user database of that kind")
    import pwd

    def unknown(_uid: int) -> None:
        raise KeyError("getpwuid(): uid not found")

    monkeypatch.setattr(pwd, "getpwuid", unknown)

    assert _login_name() is None


@pytest.mark.parametrize("platform_name", ["darwin", "linux", "wsl2", "windows"])
def test_a_home_folder_that_is_a_link_is_a_root_by_its_real_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform_name: str
) -> None:
    """`/home/u -> /data/u`: the browser refuses links, and refused the home
    folder with them, so it was missing from the roots."""
    real = tmp_path / "data" / "u"
    (real / "projects").mkdir(parents=True)
    link = tmp_path / "home-u"
    _link_or_skip(link, real)
    monkeypatch.setattr("autogpt_local_executor.directory_browser._windows_drive_roots", lambda: [])
    browser = DirectoryBrowser(platform_name=platform_name, home=link)
    monkeypatch.setattr(browser, "_children_of", lambda _parent: [])

    roots = browser.list_directories(None, None)

    [home] = roots.entries
    assert home.path == str(real.resolve())
    assert home.name == "u"
    listing = browser.list_directories(roots.browse_id, home.directory_ref)
    assert [entry.name for entry in listing.entries] == ["projects"]
    assert browser.resolve(roots.browse_id, home.directory_ref) == real.resolve()


def test_a_link_among_a_folders_children_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the home folder's own link is followed. One inside a folder
    isn't listed, in the linked home either, and one among the places
    drives are mounted isn't a root."""
    real = tmp_path / "data" / "u"
    (real / "projects").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _link_or_skip(real / "shortcut", elsewhere)
    link = tmp_path / "home-u"
    _link_or_skip(link, real)
    media = tmp_path / "media"
    (media / "USB-DISK").mkdir(parents=True)
    _link_or_skip(media / "linked-disk", elsewhere)
    browser = DirectoryBrowser(platform_name="linux", home=link)
    monkeypatch.setattr(browser, "_linux_user_mounts", lambda: DirectoryBrowser._children_of(media))

    roots = browser.list_directories(None, None)

    assert sorted(entry.name for entry in roots.entries) == ["USB-DISK", "u"]
    [home] = [entry for entry in roots.entries if entry.name == "u"]
    listing = browser.list_directories(roots.browse_id, home.directory_ref)
    assert [entry.name for entry in listing.entries] == ["projects"]


def test_a_home_folder_that_is_not_there_is_no_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "home-u"
    _link_or_skip(link, tmp_path / "gone")
    browser = DirectoryBrowser(platform_name="linux", home=link)
    monkeypatch.setattr(browser, "_children_of", lambda _parent: [])

    assert browser.list_directories(None, None).entries == ()
