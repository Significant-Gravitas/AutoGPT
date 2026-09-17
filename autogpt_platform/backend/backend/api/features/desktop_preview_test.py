"""The preview gateway checks ownership before disclosing the desktop URL."""

import json
from unittest.mock import patch

import pytest
from autogpt_libs.auth import get_user_id
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from pytest_snapshot.plugin import Snapshot

from backend.api.features.desktop_preview import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_user_id] = lambda: "owner"
    with TestClient(app) as client:
        yield client


def test_owner_redirect_is_private(client, configured_snapshot: Snapshot):
    with patch(
        "backend.api.features.desktop_preview.resolve_preview_link",
        return_value="https://6080-sandbox.e2b.app/vnc.html?password=private",
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
        "backend.api.features.desktop_preview.resolve_preview_link", return_value=None
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
        patch("backend.api.features.desktop_preview.resolve_preview_link") as resolve,
    ):
        response = client.get(
            "/api/desktop-preview?token=encrypted", follow_redirects=False
        )
    assert response.status_code == 401
    resolve.assert_not_called()
    configured_snapshot.assert_match(
        _response_json(response), "desktop_preview_unauthenticated"
    )


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
