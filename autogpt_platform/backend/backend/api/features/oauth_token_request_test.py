"""The token endpoint refuses a malformed request without writing the client
secret it carried to the logs, which go to Sentry."""

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
