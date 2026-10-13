import unittest.mock
from unittest.mock import AsyncMock

import pytest

from backend.util.settings import Settings

from . import public_media

OWNER = "owner-1"
OWN_IMAGE = "users/owner-1/images/shot.png"
OWN_VIDEO = "users/owner-1/videos/demo.mp4"


@pytest.fixture
def mock_settings(monkeypatch):
    settings = Settings()
    monkeypatch.setattr(settings.config, "media_gcs_bucket_name", "test-bucket")
    monkeypatch.setattr(settings.config, "public_site_media_bucket", "")
    monkeypatch.setattr(settings.config, "private_user_data_bucket", "")
    monkeypatch.setattr(
        "backend.api.features.store.public_media.Settings", lambda: settings
    )
    return settings


@pytest.fixture
def mock_storage_client(mocker):
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    mocker.patch(
        "backend.api.features.store.public_media.async_storage.Storage",
        return_value=client,
    )
    return client


@pytest.fixture
def public_bucket(mock_settings):
    mock_settings.config.public_site_media_bucket = "public-bucket"
    return mock_settings


async def test_publish_urls_is_a_noop_when_unset(mock_settings, mock_storage_client):
    result = await public_media.publish_urls(
        [f"https://storage.googleapis.com/test-bucket/{OWN_IMAGE}"], [OWNER]
    )

    assert result == {}
    mock_storage_client.copy.assert_not_called()


async def test_publish_urls_copies_own_media_to_the_same_path(
    public_bucket, mock_storage_client
):
    image = f"https://storage.googleapis.com/test-bucket/{OWN_IMAGE}"
    video = f"https://storage.googleapis.com/test-bucket/{OWN_VIDEO}"

    result = await public_media.publish_urls([image, video, None, image], [OWNER])

    assert result == {
        image: f"https://storage.googleapis.com/public-bucket/{OWN_IMAGE}",
        video: f"https://storage.googleapis.com/public-bucket/{OWN_VIDEO}",
    }
    assert mock_storage_client.copy.await_args_list == [
        unittest.mock.call(
            "test-bucket", OWN_IMAGE, "public-bucket", new_name=OWN_IMAGE
        ),
        unittest.mock.call(
            "test-bucket", OWN_VIDEO, "public-bucket", new_name=OWN_VIDEO
        ),
    ]


@pytest.mark.parametrize(
    "url",
    [
        f"  https://storage.googleapis.com/test-bucket/{OWN_IMAGE}  ",
        f"https://storage.cloud.google.com/test-bucket/{OWN_IMAGE}?authuser=0",
        f"https://commondatastorage.googleapis.com/test-bucket/{OWN_IMAGE}",
        "https://storage.googleapis.com/download/storage/v1/b/test-bucket/o/"
        + "users%2Fowner-1%2Fimages%2Fshot.png?alt=media",
        f"https://test-bucket.storage.googleapis.com/{OWN_IMAGE}",
        f"gs://test-bucket/{OWN_IMAGE}",
        "https://storage.googleapis.com/test-bucket/users/owner-1/images/shot%2Epng",
        "/_next/image?url=https%3A%2F%2Fstorage.googleapis.com%2Ftest-bucket%2F"
        + "users%2Fowner-1%2Fimages%2Fshot.png&w=640&q=75",
        "/api/store/submissions/media/owner-1/images/shot.png",
    ],
)
async def test_publish_urls_accepts_every_managed_url_form(
    public_bucket, mock_storage_client, url
):
    result = await public_media.publish_urls([url], [OWNER])

    assert result == {url: f"https://storage.googleapis.com/public-bucket/{OWN_IMAGE}"}
    mock_storage_client.copy.assert_awaited_once_with(
        "test-bucket", OWN_IMAGE, "public-bucket", new_name=OWN_IMAGE
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://storage.googleapis.com/test-bucket/users/someone-else/images/a.png",
        "https://storage.googleapis.com/test-bucket/uploads/owner-1/secret.pdf",
        "https://storage.googleapis.com/test-bucket/workspaces/owner-1/notes.md",
        "https://storage.googleapis.com/test-bucket/users/owner-1/files/a.png",
        "https://storage.googleapis.com/test-bucket/users/owner-1/images/sub/a.png",
        "https://storage.googleapis.com/test-bucket/users/owner-1/images/..",
        "https://storage.googleapis.com/test-bucket/users/../images/a.png",
        "https://storage.googleapis.com/test-bucket/users/owner-1/images/a%20b.png",
        f"https://storage.googleapis.com/public-bucket/{OWN_IMAGE}",
        f"https://storage.googleapis.com/other-bucket/{OWN_IMAGE}",
        f"https://example.com/test-bucket/{OWN_IMAGE}",
        f"https://example.com/?u=https://storage.googleapis.com/test-bucket/{OWN_IMAGE}",
        "https://attacker.test/api/store/submissions/media/owner-1/images/shot.png",
        "/api/store/submissions/media/someone-else/images/shot.png",
        "/api/store/media/owner-1/images/shot.png",
        "",
    ],
)
async def test_publish_urls_refuses_everything_else(
    public_bucket, mock_storage_client, url
):
    assert await public_media.publish_urls([url], [OWNER]) == {}
    mock_storage_client.copy.assert_not_called()


async def test_publish_urls_skips_objects_that_fail_to_copy(
    public_bucket, mock_storage_client
):
    image = f"https://storage.googleapis.com/test-bucket/{OWN_IMAGE}"
    video = f"https://storage.googleapis.com/test-bucket/{OWN_VIDEO}"
    mock_storage_client.copy.side_effect = [Exception("404 not found"), {}]

    result = await public_media.publish_urls([image, video], [OWNER])

    assert result == {
        video: f"https://storage.googleapis.com/public-bucket/{OWN_VIDEO}"
    }


async def test_publish_urls_is_idempotent(public_bucket, mock_storage_client):
    image = f"https://storage.googleapis.com/test-bucket/{OWN_IMAGE}"

    first = await public_media.publish_urls([image], [OWNER])
    second = await public_media.publish_urls(list(first.values()), [OWNER])

    assert second == {}
    mock_storage_client.copy.assert_awaited_once()


async def test_publish_urls_accepts_media_of_every_listed_owner(
    public_bucket, mock_storage_client
):
    member_image = "users/member-2/images/shot.png"
    url = f"https://storage.googleapis.com/test-bucket/{member_image}"

    result = await public_media.publish_urls([url], [OWNER, "member-2"])

    assert result == {
        url: f"https://storage.googleapis.com/public-bucket/{member_image}"
    }


async def test_publish_urls_falls_back_to_the_legacy_bucket_as_public(
    mock_settings, mock_storage_client
):
    mock_settings.config.private_user_data_bucket = "private-bucket"
    image = f"https://storage.googleapis.com/private-bucket/{OWN_IMAGE}"

    result = await public_media.publish_urls([image], [OWNER])

    assert result == {image: f"https://storage.googleapis.com/test-bucket/{OWN_IMAGE}"}
    mock_storage_client.copy.assert_awaited_once_with(
        "private-bucket", OWN_IMAGE, "test-bucket", new_name=OWN_IMAGE
    )
