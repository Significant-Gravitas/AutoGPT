from __future__ import annotations

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
