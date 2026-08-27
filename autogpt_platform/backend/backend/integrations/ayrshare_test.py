import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.integrations import ayrshare


@pytest.mark.asyncio
async def test_create_post_does_not_log_or_retain_sensitive_request_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    original_key = ayrshare.settings.secrets.ayrshare_api_key
    object.__setattr__(
        ayrshare.settings.secrets,
        "ayrshare_api_key",
        "platform-api-secret",
    )
    try:
        response = MagicMock(ok=True, status=200)
        response.json.return_value = {
            "status": "success",
            "posts": [
                {
                    "status": "success",
                    "id": "post-1",
                    "refId": "ref-1",
                    "profileTitle": "profile",
                    "post": "private post body",
                }
            ],
        }
        requests = MagicMock(post=AsyncMock(return_value=response))
        client = ayrshare.AyrshareClient(custom_requests=requests)

        with caplog.at_level(logging.DEBUG, logger=ayrshare.__name__):
            await client.create_post(
                "private post body",
                [ayrshare.SocialPlatform.TWITTER],
                profile_key="profile-api-secret",
            )

        logs = caplog.text
        assert "platform-api-secret" not in logs
        assert "profile-api-secret" not in logs
        assert "private post body" not in logs
        assert "Profile-Key" not in client.headers
    finally:
        object.__setattr__(
            ayrshare.settings.secrets,
            "ayrshare_api_key",
            original_key,
        )
