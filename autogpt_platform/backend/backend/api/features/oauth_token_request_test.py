"""The token endpoint refuses a malformed request without writing the client
secret it carried to the logs, which go to Sentry."""

import base64
import logging

import httpx
import pytest

from backend.api.rest_api import app

SECRET = "agpt_secret_must-not-reach-the-logs"


@pytest.mark.parametrize("encoding", ["data", "json"])
async def test_a_malformed_token_request_does_not_log_its_client_secret(
    caplog: pytest.LogCaptureFixture, encoding: str
):
    body = {
        "grant_type": "authorization_code",
        "client_id": "some-client",
        "client_secret": SECRET,
    }
    with caplog.at_level(logging.INFO):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/oauth/token", **{encoding: body})

    assert response.status_code == 422, response.text
    assert "code_verifier" in response.text
    assert SECRET not in caplog.text
    assert SECRET not in response.text


@pytest.mark.parametrize(
    "authorization",
    [
        b"Basic not-base64!",
        b"Basic " + base64.b64encode(b"\xff:secret"),
        "Basic Y2xpZW50OnNlY3JldA==\u00e9".encode("latin-1"),
    ],
)
async def test_malformed_basic_client_credentials_get_a_401(authorization: bytes):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/oauth/token",
            data={"grant_type": "authorization_code", "code": "code"},
            headers={"Authorization": authorization},
        )

    assert response.status_code == 401, response.text


def test_the_published_token_request_lets_http_basic_carry_the_client_id():
    operation = app.openapi()["paths"]["/api/oauth/token"]["post"]
    schemas = [
        grant
        for content in operation["requestBody"]["content"].values()
        for grant in content["schema"]["oneOf"]
    ]

    assert len(schemas) == 4
    assert all("client_id" in s["properties"] for s in schemas)
    assert not [s["title"] for s in schemas if "client_id" in s["required"]]
