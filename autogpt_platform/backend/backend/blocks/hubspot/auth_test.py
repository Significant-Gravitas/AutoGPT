import importlib
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import SecretStr

from backend.api.features.integrations.models import get_supported_auth_types
from backend.blocks._base import BlockSchemaInput
from backend.blocks.hubspot import _auth, _config
from backend.blocks.hubspot._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.hubspot.company import HubSpotCompanyBlock
from backend.blocks.hubspot.contact import HubSpotContactBlock
from backend.blocks.hubspot.engagement import HubSpotEngagementBlock
from backend.data.model import APIKeyCredentials, OAuth2Credentials

BLOCKS = [
    (HubSpotCompanyBlock, "company", {"operation": "create"}),
    (HubSpotContactBlock, "contact", {"operation": "create"}),
    (HubSpotEngagementBlock, "engagement", {"operation": "send_email"}),
]
TOKEN = "test-hubspot-access-token"
API_KEY_CREDENTIALS = APIKeyCredentials(
    id=TEST_CREDENTIALS.id,
    provider="hubspot",
    api_key=SecretStr(TOKEN),
    title="HubSpot private app",
    expires_at=None,
)
OAUTH_CREDENTIALS = OAuth2Credentials(
    id=TEST_CREDENTIALS.id,
    provider="hubspot",
    title="example.com",
    username="jdoe@example.com",
    access_token=SecretStr(TOKEN),
    refresh_token=SecretStr("na1-refresh-token"),
    access_token_expires_at=None,
    refresh_token_expires_at=None,
    scopes=["oauth", *_auth.CONTACT_SCOPES, *_auth.COMPANY_SCOPES],
)


@pytest.fixture
def hubspot_oauth_env(monkeypatch: pytest.MonkeyPatch):
    """Re-import HubSpot's auth with the OAuth app set (or not), then restore."""

    def configure(client_id: str, client_secret: str):
        monkeypatch.setenv("HUBSPOT_CLIENT_ID", client_id)
        monkeypatch.setenv("HUBSPOT_CLIENT_SECRET", client_secret)
        importlib.reload(_auth)
        importlib.reload(_config)
        return _auth

    yield configure
    monkeypatch.undo()
    importlib.reload(_auth)
    importlib.reload(_config)


def _credential_types(auth_module) -> frozenset[str]:
    class Input(BlockSchemaInput):
        credentials: auth_module.HubSpotCredentialsInput = (
            auth_module.HubSpotCredentialsField(auth_module.CONTACT_SCOPES)
        )

    return Input.get_credentials_fields_info()["credentials"].supported_types


def test_oauth_is_not_offered_without_the_hubspot_app(hubspot_oauth_env):
    auth = hubspot_oauth_env("", "")

    assert not auth.HUBSPOT_OAUTH_IS_CONFIGURED
    assert get_supported_auth_types("hubspot") == ["api_key"]
    assert _credential_types(auth) == frozenset({"api_key"})


def test_oauth_needs_both_the_client_id_and_secret(hubspot_oauth_env):
    auth = hubspot_oauth_env("fake-client-id", "")

    assert get_supported_auth_types("hubspot") == ["api_key"]
    assert _credential_types(auth) == frozenset({"api_key"})


def test_oauth_is_offered_next_to_the_api_key_when_configured(hubspot_oauth_env):
    auth = hubspot_oauth_env("fake-client-id", "fake-client-secret")

    assert auth.HUBSPOT_OAUTH_IS_CONFIGURED
    assert get_supported_auth_types("hubspot") == ["api_key", "oauth2"]
    assert _credential_types(auth) == frozenset({"api_key", "oauth2"})


@pytest.mark.parametrize(
    ("block_cls", "scopes"),
    [
        (HubSpotCompanyBlock, _auth.COMPANY_SCOPES),
        (HubSpotContactBlock, _auth.CONTACT_SCOPES),
        (HubSpotEngagementBlock, _auth.CONTACT_SCOPES),
    ],
)
def test_credential_field_asks_for_a_token_hubspot_still_issues(block_cls, scopes):
    schema = block_cls.Input.model_json_schema()["properties"]["credentials"]
    description = schema["description"]

    assert "Sign in with HubSpot" in description
    assert "service key" in description
    assert "private app access token" in description
    assert "Bearer" in description
    assert "crm.objects.companies.write" in description
    assert "crm.objects.contacts.write" in description
    assert "Legacy apps" not in description
    assert "requires an API Key" not in description
    field = block_cls.Input.get_credentials_fields_info()["credentials"]
    assert "api_key" in field.supported_types
    assert field.required_scopes == frozenset(scopes)


@pytest.mark.parametrize(
    "credentials", [API_KEY_CREDENTIALS, OAUTH_CREDENTIALS], ids=["api_key", "oauth2"]
)
@pytest.mark.parametrize(("block_cls", "module", "inputs"), BLOCKS)
async def test_token_is_sent_as_bearer(block_cls, module, inputs, credentials):
    block = block_cls()
    response = Mock()
    response.json.return_value = {"id": "1"}
    with patch(f"backend.blocks.hubspot.{module}.Requests") as requests:
        requests.return_value.post = AsyncMock(return_value=response)
        input_data = block_cls.Input(credentials=TEST_CREDENTIALS_INPUT, **inputs)
        outputs = [o async for o in block.run(input_data, credentials=credentials)]

    post = requests.return_value.post
    post.assert_awaited_once()
    headers = post.await_args.kwargs["headers"]
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert "hapikey" not in str(post.await_args)
    assert outputs[0][1] == {"id": "1"}
