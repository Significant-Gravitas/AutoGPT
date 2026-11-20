"""GET /airtable/credentials/{cred_id}/bases: the bases a connection can reach,
for the picker card, from the caller's own credential only."""

from unittest.mock import AsyncMock, patch

import aiohttp
import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from pydantic import SecretStr

from backend.api.features.integrations.airtable import router
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.util.request import HTTPClientError, HTTPServerError

app = fastapi.FastAPI()
app.include_router(router, prefix="/airtable")
client = fastapi.testclient.TestClient(app)

_LIST = "backend.api.features.integrations.airtable.list_bases"
_CREDS = "backend.api.features.integrations.airtable.creds_manager"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[enforce_payment_paywall] = lambda: None
    yield
    app.dependency_overrides.clear()


def _airtable(cred_id: str = "cred-1") -> OAuth2Credentials:
    return OAuth2Credentials(
        id=cred_id,
        provider="airtable",
        title="Airtable",
        access_token=SecretStr("secret-token"),
        scopes=["schema.bases:read"],
    )


def test_lists_every_page_of_bases_writable_first():
    pages = [
        {
            "bases": [
                {"id": "app1", "name": "Old reports", "permissionLevel": "read"},
                {"id": "app2", "name": "Shop", "permissionLevel": "create"},
            ],
            "offset": "next",
        },
        {"bases": [{"id": "app3", "name": "CRM", "permissionLevel": "edit"}]},
    ]
    with (
        patch(_CREDS) as creds,
        patch(_LIST, AsyncMock(side_effect=pages)) as list_bases,
    ):
        creds.get = AsyncMock(return_value=_airtable())
        resp = client.get("/airtable/credentials/cred-1/bases")

    assert resp.status_code == 200
    assert resp.json() == {
        "bases": [
            {"id": "app2", "name": "Shop", "access": "create"},
            {"id": "app3", "name": "CRM", "access": "edit"},
            {"id": "app1", "name": "Old reports", "access": "read"},
        ],
        "truncated": False,
    }
    assert list_bases.await_args_list[1].kwargs == {"offset": "next"}
    assert "secret-token" not in resp.text


def test_listing_bases_checks_the_paywall():
    """It calls Airtable, an outside service, so an account without a plan
    is stopped like every other route that does."""
    route = next(r for r in router.routes if r.path.endswith("/bases"))
    assert enforce_payment_paywall in [d.call for d in route.dependant.dependencies]


def test_someone_elses_or_another_providers_credential_is_not_found():
    github = APIKeyCredentials(
        id="cred-gh", provider="github", title="GitHub", api_key=SecretStr("k")
    )
    for found in (None, github):
        with patch(_CREDS) as creds, patch(_LIST, AsyncMock()) as list_bases:
            creds.get = AsyncMock(return_value=found)
            resp = client.get("/airtable/credentials/cred-gh/bases")
        assert resp.status_code == 404
        list_bases.assert_not_awaited()


def test_a_refused_connection_says_to_reconnect():
    refused = HTTPClientError("Unauthorized", 401)
    with patch(_CREDS) as creds, patch(_LIST, AsyncMock(side_effect=refused)):
        creds.get = AsyncMock(return_value=_airtable())
        resp = client.get("/airtable/credentials/cred-1/bases")

    assert resp.status_code == 409
    assert resp.json()["detail"] == (
        "Airtable didn't accept this connection. Reconnect Airtable and try again."
    )


def test_a_refused_token_refresh_also_says_to_reconnect():
    """Getting an OAuth credential can refresh it; Airtable refusing that
    refresh (a revoked grant) is the same fix for the user."""
    refused = HTTPClientError("invalid_grant", 400)
    with patch(_CREDS) as creds, patch(_LIST, AsyncMock()) as list_bases:
        creds.get = AsyncMock(side_effect=refused)
        resp = client.get("/airtable/credentials/cred-1/bases")

    assert resp.status_code == 409
    assert "Reconnect Airtable" in resp.json()["detail"]
    list_bases.assert_not_awaited()


@pytest.mark.parametrize(
    "status, expected, words",
    [
        (403, 409, "Reconnect Airtable"),
        (429, 503, "Airtable is busy"),
        (422, 502, "Couldn't load your Airtable bases"),
    ],
)
def test_only_a_refusal_says_reconnect(status, expected, words):
    """Reconnecting fixes a refused connection, not a rate limit or a bad
    request, so each says what actually helps."""
    error = HTTPClientError("nope", status)
    with patch(_CREDS) as creds, patch(_LIST, AsyncMock(side_effect=error)):
        creds.get = AsyncMock(return_value=_airtable())
        resp = client.get("/airtable/credentials/cred-1/bases")

    assert resp.status_code == expected
    assert words in resp.json()["detail"]


@pytest.mark.parametrize(
    "error",
    [
        HTTPServerError("down", 503),
        aiohttp.ClientConnectionError("no route"),
        TimeoutError(),
    ],
    ids=["airtable-5xx", "connection", "timeout"],
)
def test_airtable_being_down_is_said_plainly(error):
    """An outage is not the user's to fix: no 500, no "reconnect"."""
    with patch(_CREDS) as creds, patch(_LIST, AsyncMock(side_effect=error)):
        creds.get = AsyncMock(return_value=_airtable())
        resp = client.get("/airtable/credentials/cred-1/bases")

    assert resp.status_code == 502
    assert resp.json()["detail"] == (
        "Airtable isn't answering right now. Try again in a minute."
    )
