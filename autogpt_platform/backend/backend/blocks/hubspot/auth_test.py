from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import SecretStr

from backend.api.features.integrations.models import get_supported_auth_types
from backend.blocks.hubspot._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.hubspot.company import HubSpotCompanyBlock
from backend.blocks.hubspot.contact import HubSpotContactBlock
from backend.blocks.hubspot.engagement import HubSpotEngagementBlock
from backend.data.model import APIKeyCredentials

BLOCKS = [
    (HubSpotCompanyBlock, "company", {"operation": "create"}),
    (HubSpotContactBlock, "contact", {"operation": "create"}),
    (HubSpotEngagementBlock, "engagement", {"operation": "send_email"}),
]


def test_hubspot_offers_no_oauth_sign_in():
    # There is no HubSpot OAuth handler, so an OAuth option can only 404.
    assert get_supported_auth_types("hubspot") == ["api_key"]


@pytest.mark.parametrize("block_cls", [b[0] for b in BLOCKS])
def test_credential_field_asks_for_a_token_hubspot_still_issues(block_cls):
    schema = block_cls.Input.model_json_schema()["properties"]["credentials"]
    description = schema["description"]

    assert "service key" in description
    assert "private app access token" in description
    assert "Bearer" in description
    assert "requires an API Key" not in description
    field = block_cls.Input.get_credentials_fields_info()["credentials"]
    assert field.supported_types == frozenset({"api_key"})


@pytest.mark.parametrize(("block_cls", "module", "inputs"), BLOCKS)
async def test_token_is_sent_as_bearer(block_cls, module, inputs):
    token = "test-hubspot-access-token"
    credentials = APIKeyCredentials(
        id=TEST_CREDENTIALS.id,
        provider="hubspot",
        api_key=SecretStr(token),
        title="HubSpot private app",
        expires_at=None,
    )
    block = block_cls()
    response = Mock()
    response.json.return_value = {"id": "1"}
    with patch(f"backend.blocks.hubspot.{module}.Requests") as requests:
        requests.return_value.post = AsyncMock(return_value=response)
        input_data = block_cls.Input(credentials=TEST_CREDENTIALS_INPUT, **inputs)
        async for _ in block.run(input_data, credentials=credentials):
            pass

    post = requests.return_value.post
    post.assert_awaited_once()
    headers = post.await_args.kwargs["headers"]
    assert headers["Authorization"] == f"Bearer {token}"
    assert "hapikey" not in str(post.await_args)
