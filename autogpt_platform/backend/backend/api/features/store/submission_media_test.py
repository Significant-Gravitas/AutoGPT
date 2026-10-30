from unittest.mock import AsyncMock

import autogpt_libs.auth
import fastapi
import prisma.enums
import pytest

from backend.util.exceptions import NotFoundError
from backend.util.settings import Settings

from . import routes as store_routes
from . import submission_media


def _user(user_id: str, role: str = "authenticated") -> autogpt_libs.auth.User:
    return autogpt_libs.auth.User(user_id=user_id, email="", phone_number="", role=role)


OWNER = _user("owner")


@pytest.fixture
def mock_settings(monkeypatch):
    settings = Settings()
    monkeypatch.setattr(settings.config, "media_gcs_bucket_name", "test-bucket")
    monkeypatch.setattr(settings.config, "private_user_data_bucket", "private-media")
    monkeypatch.setattr(
        "backend.api.features.store.submission_media.Settings", lambda: settings
    )
    return settings


@pytest.fixture
def mock_storage_client(mocker):
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    mocker.patch(
        "backend.api.features.store.submission_media.async_storage.Storage",
        return_value=client,
    )
    return client


def test_private_media_route_requires_an_authenticated_user():
    route = next(
        route
        for route in store_routes.router.routes
        if getattr(route, "path", "")
        == "/submissions/media/{owner_user_id}/{media_type}/{filename}"
    )

    assert autogpt_libs.auth.requires_user in {
        dependency.call for dependency in route.dependant.dependencies
    }


async def test_private_media_read_serves_the_owner(mocker):
    mocker.patch.object(
        submission_media, "metadata", new=AsyncMock(return_value={"size": "13"})
    )

    async def chunks(*args):
        yield b"private-image"

    mocker.patch.object(submission_media, "stream", side_effect=chunks)

    response = await store_routes.get_private_submission_media(
        owner_user_id="owner",
        media_type="images",
        filename="image.jpeg",
        request=fastapi.Request({"type": "http", "headers": []}),
        user=OWNER,
    )

    assert (
        b"".join([chunk async for chunk in response.body_iterator]) == b"private-image"
    )
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-length"] == "13"


async def test_oversized_private_media_uses_short_lived_signed_redirect(mocker):
    size = submission_media.PRIVATE_MEDIA_PROXY_BUFFER_BYTES + 1
    mocker.patch.object(
        submission_media,
        "metadata",
        new=AsyncMock(return_value={"size": str(size)}),
    )
    signed_url = mocker.patch.object(
        submission_media,
        "signed_url",
        new=AsyncMock(return_value="https://storage.example/signed"),
    )
    stream = mocker.patch.object(submission_media, "stream")

    response = await store_routes.get_private_submission_media(
        owner_user_id="owner",
        media_type="images",
        filename="large.png",
        request=fastapi.Request({"type": "http", "headers": []}),
        user=OWNER,
    )

    assert response.status_code == 307
    assert response.headers["location"] == "https://storage.example/signed"
    assert response.headers["cache-control"] == "private, no-store"
    signed_url.assert_awaited_once_with("owner", "images", "large.png")
    stream.assert_not_called()


async def test_oversized_private_media_fails_closed_when_signing_fails(mocker):
    size = submission_media.PRIVATE_MEDIA_PROXY_BUFFER_BYTES + 1
    mocker.patch.object(
        submission_media,
        "metadata",
        new=AsyncMock(return_value={"size": str(size)}),
    )
    mocker.patch.object(
        submission_media,
        "signed_url",
        new=AsyncMock(side_effect=RuntimeError("signing unavailable")),
    )
    stream = mocker.patch.object(submission_media, "stream")

    with pytest.raises(fastapi.HTTPException) as error:
        await store_routes.get_private_submission_media(
            owner_user_id="owner",
            media_type="images",
            filename="large.png",
            request=fastapi.Request({"type": "http", "headers": []}),
            user=OWNER,
        )

    assert error.value.status_code == 503
    assert error.value.detail == "Private media is temporarily unavailable"
    stream.assert_not_called()


async def test_private_media_read_returns_not_found_for_missing_object(mocker):
    mocker.patch.object(
        submission_media,
        "metadata",
        new=AsyncMock(side_effect=FileNotFoundError("missing")),
    )

    with pytest.raises(NotFoundError):
        await store_routes.get_private_submission_media(
            owner_user_id="owner",
            media_type="images",
            filename="image.jpeg",
            request=fastapi.Request({"type": "http", "headers": []}),
            user=OWNER,
        )


@pytest.mark.parametrize(
    ("value", "total_size", "expected"),
    [
        ("bytes=0-4", 10, (0, 4)),
        ("bytes=5-", 10, (5, 9)),
        ("bytes=-3", 10, (7, 9)),
        ("bytes=0-99", 10, (0, 9)),
    ],
)
def test_private_media_range_parsing(value, total_size, expected):
    assert store_routes._parse_private_media_range(value, total_size) == expected


@pytest.mark.parametrize(
    ("value", "total_size"),
    [
        ("items=0-1", 10),
        ("bytes=", 10),
        ("bytes=5-4", 10),
        ("bytes=10-", 10),
        ("bytes=-0", 10),
        ("bytes=0-1,3-4", 10),
        ("bytes=0-0", 0),
    ],
)
def test_private_media_range_rejects_invalid_or_unsatisfiable_values(value, total_size):
    with pytest.raises(ValueError):
        store_routes._parse_private_media_range(value, total_size)


async def test_private_media_read_serves_single_byte_range(mocker):
    total_size = submission_media.PRIVATE_MEDIA_PROXY_BUFFER_BYTES + 1
    mocker.patch.object(
        submission_media,
        "metadata",
        new=AsyncMock(return_value={"size": str(total_size)}),
    )

    async def chunks(*args):
        assert args[-1] == (2, 5)
        yield b"ivat"

    stream = mocker.patch.object(submission_media, "stream", side_effect=chunks)
    request = fastapi.Request({"type": "http", "headers": [(b"range", b"bytes=2-5")]})

    response = await store_routes.get_private_submission_media(
        owner_user_id="owner",
        media_type="videos",
        filename="preview.mp4",
        request=request,
        user=OWNER,
    )

    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 2-5/{total_size}"
    assert response.headers["content-length"] == "4"
    assert response.headers["accept-ranges"] == "bytes"
    assert b"".join([chunk async for chunk in response.body_iterator]) == b"ivat"
    assert stream.call_count == 1


async def test_private_media_read_rejects_unsatisfiable_range(mocker):
    mocker.patch.object(
        submission_media, "metadata", new=AsyncMock(return_value={"size": "13"})
    )
    stream = mocker.patch.object(submission_media, "stream")
    request = fastapi.Request({"type": "http", "headers": [(b"range", b"bytes=99-")]})

    with pytest.raises(fastapi.HTTPException) as error:
        await store_routes.get_private_submission_media(
            owner_user_id="owner",
            media_type="videos",
            filename="preview.mp4",
            request=request,
            user=OWNER,
        )

    assert error.value.status_code == 416
    assert error.value.headers["Content-Range"] == "bytes */13"
    assert error.value.headers["Cache-Control"] == "private, no-store"
    stream.assert_not_called()


async def test_private_media_stream_reads_only_from_private_bucket(
    mock_settings, mock_storage_client
):
    class Stream:
        def __init__(self):
            self.chunks = [b"private", b"-image", b""]

        async def read(self, size):
            return self.chunks.pop(0)

    mock_storage_client.download_stream.return_value = Stream()

    chunks = [
        chunk
        async for chunk in submission_media.stream("owner", "images", "image.jpeg")
    ]

    assert b"".join(chunks) == b"private-image"
    mock_storage_client.download_stream.assert_awaited_once_with(
        "private-media",
        "users/owner/images/image.jpeg",
        headers=None,
        timeout=submission_media._STREAM_TIMEOUT,
    )
    # A total timeout would cut off any body the client reads slowly.
    assert submission_media._STREAM_TIMEOUT.total is None


async def test_private_media_range_stream_uses_gcs_range_header(
    mock_settings, mock_storage_client
):
    class Stream:
        def __init__(self):
            self.chunks = [b"part", b""]

        async def read(self, size):
            return self.chunks.pop(0)

    mock_storage_client.download_stream.return_value = Stream()

    chunks = [
        chunk
        async for chunk in submission_media.stream(
            "owner", "videos", "preview.mp4", (2, 5)
        )
    ]

    assert b"".join(chunks) == b"part"
    mock_storage_client.download_stream.assert_awaited_once_with(
        "private-media",
        "users/owner/videos/preview.mp4",
        headers={"Range": "bytes=2-5"},
        timeout=submission_media._STREAM_TIMEOUT,
    )


async def test_private_media_signed_url_targets_private_bucket(mock_settings, mocker):
    client = mocker.patch.object(
        submission_media.gcs_storage.Client, "create_anonymous_client"
    ).return_value
    generate = mocker.patch.object(
        submission_media,
        "generate_iam_signed_url",
        new=AsyncMock(return_value="https://storage.example/signed"),
    )

    result = await submission_media.signed_url("owner", "images", "image.jpeg")

    assert result == "https://storage.example/signed"
    generate.assert_awaited_once_with(
        client,
        "private-media",
        "users/owner/images/image.jpeg",
        submission_media.PRIVATE_MEDIA_SIGNED_URL_TTL_SECONDS,
    )


@pytest.mark.parametrize(
    "user, shares_an_org, allowed",
    [
        (_user("owner"), False, True),
        (_user("reviewer", role="admin"), False, True),
        (_user("colleague"), True, True),
        (_user("stranger"), False, False),
    ],
)
async def test_private_media_is_readable_by_owner_admins_and_colleagues(
    mocker, user, shares_an_org, allowed
):
    find_membership = mocker.patch("prisma.models.OrgMember.prisma")
    find_membership.return_value.find_first = AsyncMock(
        return_value=mocker.MagicMock() if shares_an_org else None
    )

    assert await submission_media.can_read(user, "owner") is allowed


async def test_private_media_read_hides_media_from_other_users(mocker):
    mocker.patch.object(submission_media, "can_read", new=AsyncMock(return_value=False))
    metadata = mocker.patch.object(submission_media, "metadata", new=AsyncMock())

    with pytest.raises(NotFoundError):
        await store_routes.get_private_submission_media(
            owner_user_id="owner",
            media_type="images",
            filename="image.jpeg",
            request=fastapi.Request({"type": "http", "headers": []}),
            user=_user("stranger"),
        )
    metadata.assert_not_awaited()


async def test_colleague_access_requires_active_memberships_in_a_live_org(mocker):
    find_membership = mocker.patch("prisma.models.OrgMember.prisma")
    find_membership.return_value.find_first = AsyncMock(return_value=None)

    await submission_media.can_read(_user("colleague"), "owner")

    assert find_membership.return_value.find_first.await_args.kwargs["where"] == {
        "userId": "colleague",
        "status": prisma.enums.OrgMemberStatus.ACTIVE,
        "Org": {
            "is": {
                "deletedAt": None,
                "Members": {
                    "some": {
                        "userId": "owner",
                        "status": prisma.enums.OrgMemberStatus.ACTIVE,
                    }
                },
            }
        },
    }
