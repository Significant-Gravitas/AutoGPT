from __future__ import annotations

from pathlib import Path

import pytest

from autogpt_local_executor.directory_browser import (
    MAX_DIRECTORY_ENTRIES,
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
