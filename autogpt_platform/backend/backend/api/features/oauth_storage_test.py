import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from backend.api.features import oauth
from backend.api.features.oauth import _delete_app_current_logo_file, settings


@pytest.mark.asyncio
async def test_logo_url_updates_before_previous_object_is_deleted():
    app = SimpleNamespace(id="app-123", owner_id="owner", logo_url="old-logo")
    updated_app = SimpleNamespace(name="Test app")
    order = []

    async def update(**kwargs):
        order.append("update")
        return updated_app

    async def delete(previous):
        order.append("delete")

    with (
        patch.object(oauth, "get_oauth_application_by_id", AsyncMock(return_value=app)),
        patch.object(oauth, "update_oauth_application", side_effect=update),
        patch.object(oauth, "_delete_app_current_logo_file", side_effect=delete),
    ):
        result = await oauth.update_app_logo(
            "app-123", oauth.UpdateAppLogoRequest(logo_url="new-logo"), user_id="owner"
        )

    assert result is updated_app
    assert order == ["update", "delete"]


@pytest.mark.asyncio
async def test_unchanged_logo_url_does_not_delete_current_object():
    app = SimpleNamespace(id="app-123", owner_id="owner", logo_url="same-logo")
    updated_app = SimpleNamespace(name="Test app")

    with (
        patch.object(oauth, "get_oauth_application_by_id", AsyncMock(return_value=app)),
        patch.object(
            oauth, "update_oauth_application", AsyncMock(return_value=updated_app)
        ),
        patch.object(
            oauth, "_delete_app_current_logo_file", new_callable=AsyncMock
        ) as delete,
    ):
        result = await oauth.update_app_logo(
            "app-123", oauth.UpdateAppLogoRequest(logo_url="same-logo"), user_id="owner"
        )

    assert result is updated_app
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_logo_url_update_keeps_previous_object():
    app = SimpleNamespace(id="app-123", owner_id="owner", logo_url="old-logo")

    with (
        patch.object(oauth, "get_oauth_application_by_id", AsyncMock(return_value=app)),
        patch.object(oauth, "update_oauth_application", AsyncMock(return_value=None)),
        patch.object(
            oauth, "_delete_app_current_logo_file", new_callable=AsyncMock
        ) as delete,
        pytest.raises(oauth.HTTPException) as error,
    ):
        await oauth.update_app_logo(
            "app-123", oauth.UpdateAppLogoRequest(logo_url="new-logo"), user_id="owner"
        )

    assert error.value.status_code == 404
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_logo_reference_updates_before_previous_object_is_deleted(monkeypatch):
    monkeypatch.setattr(settings.config, "public_site_media_bucket", "public-media")
    monkeypatch.setattr(settings.config, "private_user_data_bucket", "private-data")
    monkeypatch.setattr(settings.config, "media_gcs_bucket_name", "legacy-media")
    app = SimpleNamespace(
        id="app-123",
        owner_id="owner",
        logo_url=(
            "https://storage.googleapis.com/legacy-media/"
            "oauth-apps/app-123/logo/old.png"
        ),
    )
    updated_app = SimpleNamespace(name="Test app")
    order = []
    update_arguments = {}

    async def update(**kwargs):
        order.append("update")
        update_arguments.update(kwargs)
        return updated_app

    async def delete(previous):
        order.append("delete")

    client = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    file = UploadFile(
        filename="logo.png",
        file=io.BytesIO(b"image"),
        headers=Headers({"content-type": "image/png"}),
    )

    with (
        patch.object(oauth, "get_oauth_application_by_id", AsyncMock(return_value=app)),
        patch.object(oauth, "update_oauth_application", side_effect=update),
        patch.object(oauth, "_delete_app_current_logo_file", side_effect=delete),
        patch.object(oauth, "scan_content_safe", AsyncMock()),
        patch.object(
            oauth.Image, "open", return_value=SimpleNamespace(size=(512, 512))
        ),
        patch.object(oauth.async_storage, "Storage", return_value=context),
    ):
        result = await oauth.upload_app_logo("app-123", file, user_id="owner")

    assert result is updated_app
    assert order == ["update", "delete"]
    assert client.upload.await_args.args[0] == "public-media"
    assert update_arguments["logo_url"].startswith(
        "https://storage.googleapis.com/public-media/oauth-apps/app-123/logo/"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bucket_name", "setting_name"),
    [
        ("public-media", "public_site_media_bucket"),
        ("legacy-media", "media_gcs_bucket_name"),
    ],
)
async def test_delete_app_logo_supports_current_and_legacy_buckets(
    monkeypatch, bucket_name, setting_name
):
    monkeypatch.setattr(settings.config, "public_site_media_bucket", "public-media")
    monkeypatch.setattr(settings.config, "private_user_data_bucket", "private-data")
    monkeypatch.setattr(settings.config, "media_gcs_bucket_name", "legacy-media")
    assert getattr(settings.config, setting_name) == bucket_name

    client = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    app = SimpleNamespace(
        id="app-123",
        logo_url=(
            f"https://storage.googleapis.com/{bucket_name}/"
            "oauth-apps/app-123/logo/logo.png"
        ),
    )

    with patch(
        "backend.api.features.oauth.async_storage.Storage", return_value=context
    ):
        await _delete_app_current_logo_file(app)

    client.delete.assert_awaited_once_with(
        bucket_name, "oauth-apps/app-123/logo/logo.png"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "logo_url",
    [
        "https://example.com/public-media/oauth-apps/app-123/logo/logo.png",
        "https://storage.googleapis.com/other-bucket/oauth-apps/app-123/logo/logo.png",
        "https://storage.googleapis.com/public-media/oauth-apps/other/logo/logo.png",
        "https://storage.googleapis.com/public-media/users/app-123/logo/logo.png",
    ],
)
async def test_delete_app_logo_rejects_unmanaged_locations(monkeypatch, logo_url):
    monkeypatch.setattr(settings.config, "public_site_media_bucket", "public-media")
    monkeypatch.setattr(settings.config, "private_user_data_bucket", "private-data")
    monkeypatch.setattr(settings.config, "media_gcs_bucket_name", "legacy-media")
    app = SimpleNamespace(id="app-123", logo_url=logo_url)

    storage = MagicMock()
    with patch("backend.api.features.oauth.async_storage.Storage", storage):
        await _delete_app_current_logo_file(app)

    storage.assert_not_called()
