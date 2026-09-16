"""Tests for the one field of an expert that could make the backend fetch a
URL somebody else chose."""

from pathlib import Path

import pytest
import pytest_mock

from backend.api.features.experts import package_avatar
from backend.api.features.experts.package_avatar import packaged_avatar
from backend.api.features.experts.package_model import MAX_AVATAR_BYTES

LOCAL = "/api/store/media/user-1/images/a.png"


@pytest.fixture
def stored(mocker: pytest_mock.MockFixture, tmp_path: Path):
    def _store(name: str, content: bytes) -> Path:
        path = tmp_path / name
        path.write_bytes(content)
        mocker.patch.object(
            package_avatar.local_media, "get_media_path", return_value=path
        )
        return path

    return _store


async def test_media_we_host_is_embedded_as_bytes(stored):
    stored("a.png", b"\x89PNG\r\n\x1a\n")

    avatar, content = await packaged_avatar(LOCAL)

    assert avatar is not None
    assert avatar.kind == "file" and avatar.path == "avatar.png"
    assert content == b"\x89PNG\r\n\x1a\n"


async def test_an_absolute_url_to_our_own_media_is_the_same_avatar(
    stored, mocker: pytest_mock.MockFixture
):
    """``media_url`` prefixes the platform base URL, so the stored value may be
    absolute or site-relative for the very same file."""
    stored("a.png", b"\x89PNG")
    mocker.patch.object(
        package_avatar.settings.config, "platform_base_url", "https://app.example"
    )

    avatar, content = await packaged_avatar(f"https://app.example{LOCAL}")

    assert avatar is not None and avatar.path == "avatar.png"
    assert content == b"\x89PNG"


async def test_a_roster_path_travels_as_a_url():
    """It ships with the frontend, so a URL restores it anywhere this platform
    runs — there is nothing to embed."""
    avatar, content = await packaged_avatar("/experts/maria.svg")

    assert avatar is not None
    assert avatar.kind == "url" and avatar.url == "/experts/maria.svg"
    assert content is None


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/pixel.png",
        "https://storage.googleapis.com/not-our-bucket/a.png",
        "data:image/png;base64,AAAA",
    ],
)
async def test_media_we_do_not_host_is_dropped_rather_than_fetched(url: str):
    """Packaging must never be a way to make the backend request a URL of the
    caller's choosing."""
    assert await packaged_avatar(url) == (None, None)


async def test_a_format_we_cannot_name_is_dropped(stored):
    """The manifest only has names for the extensions the reader accepts."""
    stored("a.bmp", b"BM")
    assert await packaged_avatar("/api/store/media/user-1/images/a.bmp") == (None, None)


async def test_an_avatar_over_the_cap_is_dropped(stored):
    stored("a.png", b"\0" * (MAX_AVATAR_BYTES + 1))
    assert await packaged_avatar(LOCAL) == (None, None)


async def test_an_unreadable_file_is_dropped_rather_than_failing_the_package(
    mocker: pytest_mock.MockFixture,
):
    """A missing picture is a far smaller loss than a download that 500s."""
    mocker.patch.object(
        package_avatar.local_media, "get_media_path", side_effect=ValueError("nope")
    )
    assert await packaged_avatar(LOCAL) == (None, None)


@pytest.mark.parametrize("url", [None, "", "   "])
async def test_an_expert_with_no_picture_packages_without_one(url: str | None):
    assert await packaged_avatar(url) == (None, None)
