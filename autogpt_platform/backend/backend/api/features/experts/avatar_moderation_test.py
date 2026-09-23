from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_unreviewed_external_avatar_is_rejected():
    from backend.api.features.experts.avatar_moderation import require_approved_avatar

    with patch(
        "backend.api.features.experts.avatar_moderation.get_redis_async",
        new=AsyncMock(return_value=AsyncMock(get=AsyncMock(return_value=None))),
    ):
        with pytest.raises(HTTPException) as error:
            await require_approved_avatar("owner", "https://example.com/avatar.png")
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_pending_moderation_does_not_approve_an_image():
    from backend.api.features.experts.avatar_moderation import moderate_avatar_image

    response = SimpleNamespace(
        json=lambda: {"success": True, "status": "pending", "content_id": "123"}
    )
    with patch(
        "backend.api.features.experts.avatar_moderation.settings"
    ) as settings, patch(
        "backend.api.features.experts.avatar_moderation.Requests"
    ) as requests:
        settings.config.automod_api_url = "https://moderation.example"
        settings.secrets.automod_api_key = "test"
        settings.config.automod_timeout = 5
        requests.return_value.post = AsyncMock(return_value=response)
        with pytest.raises(HTTPException) as error:
            await moderate_avatar_image("owner", b"image", "image/png")
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_approval_receipt_cannot_be_reused_by_another_owner():
    from backend.api.features.experts.avatar_moderation import (
        record_approved_avatar,
        require_approved_avatar,
    )

    receipts = {}
    redis = AsyncMock()
    redis.setex.side_effect = lambda key, ttl, value: receipts.update({key: value})
    redis.get.side_effect = lambda key: receipts.get(key)
    with patch(
        "backend.api.features.experts.avatar_moderation.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        await record_approved_avatar("owner", "https://example.com/upload.png")
        await require_approved_avatar("owner", "https://example.com/upload.png")
        with pytest.raises(HTTPException) as error:
            await require_approved_avatar(
                "other-owner", "https://example.com/upload.png"
            )
    assert error.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["rejected", "flagged", "pending"])
async def test_nonapproved_moderation_never_allows_upload(status):
    from backend.api.features.experts.avatar_moderation import moderate_avatar_image

    response = SimpleNamespace(
        json=lambda: {"success": True, "status": status, "content_id": "123"}
    )
    with patch(
        "backend.api.features.experts.avatar_moderation.settings"
    ) as settings, patch(
        "backend.api.features.experts.avatar_moderation.Requests"
    ) as requests:
        settings.config.automod_api_url = "https://moderation.example"
        settings.secrets.automod_api_key = "test"
        settings.config.automod_timeout = 5
        requests.return_value.post = AsyncMock(return_value=response)
        with pytest.raises(HTTPException) as error:
            await moderate_avatar_image("owner", b"image", "image/png")
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_moderation_outage_fails_closed():
    from backend.api.features.experts.avatar_moderation import moderate_avatar_image

    with patch(
        "backend.api.features.experts.avatar_moderation.settings"
    ) as settings, patch(
        "backend.api.features.experts.avatar_moderation.Requests"
    ) as requests:
        settings.config.automod_api_url = "https://moderation.example"
        settings.secrets.automod_api_key = "test"
        settings.config.automod_timeout = 5
        requests.return_value.post = AsyncMock(side_effect=TimeoutError)
        with pytest.raises(HTTPException) as error:
            await moderate_avatar_image("owner", b"image", "image/png")
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_approved_image_is_submitted_with_owner_and_image_type():
    from backend.api.features.experts.avatar_moderation import moderate_avatar_image

    response = SimpleNamespace(
        json=lambda: {"success": True, "status": "approved", "content_id": "123"}
    )
    with patch(
        "backend.api.features.experts.avatar_moderation.settings"
    ) as settings, patch(
        "backend.api.features.experts.avatar_moderation.Requests"
    ) as requests:
        settings.config.automod_api_url = "https://moderation.example"
        settings.secrets.automod_api_key = "test"
        settings.config.automod_timeout = 5
        requests.return_value.post = AsyncMock(return_value=response)
        await moderate_avatar_image("owner", b"image", "image/png")
        payload = requests.return_value.post.call_args.kwargs["json"]
    assert payload["type"] == "image"
    assert payload["content"].startswith("data:image/png;base64,")
    assert payload["metadata"]["user_id"] == "owner"


@pytest.mark.asyncio
@pytest.mark.parametrize("connection_fails", [False, True])
async def test_receipt_outage_returns_retryable_error(connection_fails):
    from backend.api.features.experts.avatar_moderation import record_approved_avatar

    redis = AsyncMock()
    redis.setex.side_effect = ConnectionError("unavailable")
    with patch(
        "backend.api.features.experts.avatar_moderation.get_redis_async",
        new=AsyncMock(
            return_value=redis,
            side_effect=ConnectionError("unavailable") if connection_fails else None,
        ),
    ):
        with pytest.raises(HTTPException) as error:
            await record_approved_avatar("owner", "https://example.com/upload.png")
    assert error.value.status_code == 503
    assert "try again" in error.value.detail
