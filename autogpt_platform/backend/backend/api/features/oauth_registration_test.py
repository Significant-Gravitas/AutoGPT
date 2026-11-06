"""Who may register an OAuth client: nobody unless the instance enables it, and
then only a public client that comes back to the loopback interface or https."""

import httpx
import pytest

from backend.api.features import oauth_registration
from backend.api.features.oauth_metadata import WELL_KNOWN_AUTHORIZATION_SERVER_PATH
from backend.api.features.oauth_registration import (
    ClientRegistrationRequest,
    registration_problem,
)
from backend.api.rest_api import app
from backend.util.settings import Config

LOOPBACK = "http://127.0.0.1:33418/callback"


async def test_registration_is_off_unless_the_instance_enables_it(
    monkeypatch: pytest.MonkeyPatch,
):
    assert not Config.model_fields["oauth_dynamic_client_registration"].default
    monkeypatch.setattr(
        oauth_registration.settings.config, "oauth_dynamic_client_registration", False
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/oauth/register", json={"redirect_uris": [LOOPBACK]}
        )
        metadata = (await client.get(WELL_KNOWN_AUTHORIZATION_SERVER_PATH)).json()

    assert response.status_code == 404
    assert "registration_endpoint" not in metadata
    assert "none" not in metadata["token_endpoint_auth_methods_supported"]


async def test_the_metadata_advertises_registration_once_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        oauth_registration.settings.config, "oauth_dynamic_client_registration", True
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        metadata = (await client.get(WELL_KNOWN_AUTHORIZATION_SERVER_PATH)).json()

    assert metadata["registration_endpoint"].endswith("/api/oauth/register")
    assert "none" in metadata["token_endpoint_auth_methods_supported"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"redirect_uris": ["http://evil.example/callback"]},
        {"redirect_uris": ["cursor://anysphere.cursor-retrieval/oauth/callback"]},
        {"redirect_uris": ["https://example.com/callback#fragment"]},
        {"redirect_uris": ["https://user:pass@example.com/callback"]},
        {"redirect_uris": []},
        {"redirect_uris": [LOOPBACK] * 6},
        {
            "redirect_uris": [LOOPBACK],
            "token_endpoint_auth_method": "client_secret_post",
        },
        {"redirect_uris": [LOOPBACK], "grant_types": ["implicit"]},
        {"redirect_uris": [LOOPBACK], "grant_types": ["refresh_token"]},
        {"redirect_uris": [LOOPBACK], "response_types": ["token"]},
        {"redirect_uris": [LOOPBACK], "scope": "READ_LIBRARY ADMIN"},
        {"redirect_uris": [LOOPBACK], "client_name": "x" * 101},
    ],
)
def test_a_client_outside_the_policy_is_refused(metadata: dict):
    assert registration_problem(ClientRegistrationRequest(**metadata)) is not None


@pytest.mark.parametrize(
    "redirect_uri",
    [
        "http://localhost:8765/callback",
        LOOPBACK,
        "http://[::1]:9000/callback",
        "https://vscode.dev/redirect",
    ],
)
def test_a_public_client_with_a_loopback_or_https_redirect_is_accepted(
    redirect_uri: str,
):
    metadata = ClientRegistrationRequest(
        redirect_uris=[redirect_uri], scope="READ_LIBRARY RUN_AGENT"
    )
    assert registration_problem(metadata) is None
