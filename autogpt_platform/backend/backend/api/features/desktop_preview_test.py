"""The preview gateway checks ownership before disclosing the desktop URL."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest
from autogpt_libs.auth import get_user_id
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from pytest_snapshot.plugin import Snapshot

from backend.api.features.desktop_preview import router
from backend.copilot.tools.e2b_sandbox import _STREAM_STOPPED
from backend.util.desktop_preview import DesktopPreview, create_preview_link
from backend.util.encryption import JSONCryptor

_LIVE_URL = "https://6080-sandbox.e2b.app/vnc.html?password=private"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_user_id] = lambda: "owner"
    with TestClient(app) as client:
        yield client


def test_owner_redirect_is_private(client, configured_snapshot: Snapshot):
    with patch(
        "backend.api.features.desktop_preview.resolve_preview",
        return_value=DesktopPreview(
            purpose="e2b-desktop-preview", user_id="owner", url=_LIVE_URL
        ),
    ) as resolve:
        response = client.get(
            "/api/desktop-preview?token=encrypted", follow_redirects=False
        )
    resolve.assert_called_once_with("owner", "encrypted")
    assert response.status_code == 307
    assert response.headers["location"].endswith("?password=private")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "private" not in response.text
    configured_snapshot.assert_match(
        _response_json(response), "desktop_preview_owner_redirect"
    )


def test_wrong_owner_or_invalid_link_returns_no_credential(
    client, configured_snapshot: Snapshot
):
    with patch(
        "backend.api.features.desktop_preview.resolve_preview", return_value=None
    ):
        response = client.get(
            "/api/desktop-preview?token=encrypted", follow_redirects=False
        )
    assert response.status_code == 404
    assert "location" not in response.headers
    configured_snapshot.assert_match(
        _response_json(response), "desktop_preview_unavailable"
    )


def test_unauthenticated_request_is_rejected(configured_snapshot: Snapshot):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    with (
        TestClient(app) as client,
        patch("backend.api.features.desktop_preview.resolve_preview") as resolve,
    ):
        response = client.get(
            "/api/desktop-preview?token=encrypted", follow_redirects=False
        )
    assert response.status_code == 401
    resolve.assert_not_called()
    configured_snapshot.assert_match(
        _response_json(response), "desktop_preview_unauthenticated"
    )


@pytest.mark.parametrize(
    "stored, status",
    [
        ("private", 307),
        # A pause stopped the stream, a restart rotated the password, or the
        # box's running limit passed and the key expired.
        (_STREAM_STOPPED, 410),
        ("rotated", 410),
        (None, 410),
    ],
)
def test_a_link_opens_only_the_stream_the_box_still_serves(client, stored, status):
    redis = MagicMock()
    redis.get = AsyncMock(return_value=stored)
    cryptor = JSONCryptor(Fernet.generate_key().decode())
    with (
        patch("backend.util.desktop_preview.JSONCryptor", return_value=cryptor),
        patch(
            "backend.util.desktop_preview.get_redis_async",
            AsyncMock(return_value=redis),
        ),
    ):
        link = create_preview_link("owner", _LIVE_URL, stream_key="owner:stream")
        token = parse_qs(urlsplit(link).query)["token"][0]
        response = client.get(
            f"/api/desktop-preview?token={token}", follow_redirects=False
        )
    redis.get.assert_awaited_once_with("owner:stream")
    assert response.status_code == status
    if status == 410:
        assert "location" not in response.headers
        assert "private" not in response.text


def _response_json(response: Response) -> str:
    return json.dumps(
        {
            "status": response.status_code,
            "headers": dict(response.headers),
            "body": response.text,
        },
        indent=2,
        sort_keys=True,
    )
