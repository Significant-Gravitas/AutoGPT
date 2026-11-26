"""Who may register an OAuth client: nobody unless the instance enables it, and
then only a public client that comes back to the loopback interface and asks
for no more than the MCP server's permissions."""

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


REDIRECT = "invalid_redirect_uri"
METADATA = "invalid_client_metadata"


@pytest.mark.parametrize(
    "metadata, error",
    [
        ({"redirect_uris": ["http://evil.example/callback"]}, REDIRECT),
        ({"redirect_uris": ["https://vscode.dev/redirect"]}, REDIRECT),
        ({"redirect_uris": ["https://localhost:8765/callback"]}, REDIRECT),
        ({"redirect_uris": ["cursor://anysphere.cursor-retrieval/callback"]}, REDIRECT),
        ({"redirect_uris": [f"{LOOPBACK}#fragment"]}, REDIRECT),
        ({"redirect_uris": ["http://user:pass@localhost:8765/callback"]}, REDIRECT),
        ({"redirect_uris": []}, REDIRECT),
        ({"redirect_uris": [LOOPBACK] * 6}, REDIRECT),
        (
            {
                "redirect_uris": [LOOPBACK],
                "token_endpoint_auth_method": "client_secret_post",
            },
            METADATA,
        ),
        ({"redirect_uris": [LOOPBACK], "grant_types": ["implicit"]}, METADATA),
        ({"redirect_uris": [LOOPBACK], "grant_types": ["refresh_token"]}, METADATA),
        ({"redirect_uris": [LOOPBACK], "response_types": ["token"]}, METADATA),
        ({"redirect_uris": [LOOPBACK], "scope": "READ_LIBRARY ADMIN"}, METADATA),
        (
            {"redirect_uris": [LOOPBACK], "scope": "READ_LIBRARY MANAGE_INTEGRATIONS"},
            METADATA,
        ),
        ({"redirect_uris": [LOOPBACK], "client_name": "x" * 101}, METADATA),
    ],
)
def test_a_client_outside_the_policy_is_refused(metadata: dict, error: str):
    problem = registration_problem(ClientRegistrationRequest(**metadata))
    assert problem is not None and problem.error == error


@pytest.mark.parametrize(
    "redirect_uri",
    [
        "http://localhost:8765/callback",
        LOOPBACK,
        "http://127.0.0.1/callback",
        "http://[::1]:9000/callback",
    ],
)
def test_a_public_client_with_a_loopback_redirect_is_accepted(
    redirect_uri: str,
):
    metadata = ClientRegistrationRequest(
        redirect_uris=[redirect_uri], scope="READ_LIBRARY RUN_AGENT"
    )
    assert registration_problem(metadata) is None
