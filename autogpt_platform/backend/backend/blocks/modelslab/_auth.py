from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

ModelsLabCredentials = APIKeyCredentials
ModelsLabCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.MODELSLAB],
    Literal["api_key"],
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="11223344-aabb-ccdd-1234-556677889900",
    provider="modelslab",
    api_key=SecretStr("mock-modelslab-api-key"),
    title="Mock ModelsLab API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def ModelsLabCredentialsField() -> ModelsLabCredentialsInput:
    """Creates a ModelsLab credentials input on a block."""
    return CredentialsField(
        description="The ModelsLab integration requires an API Key. "
        "Get yours at https://modelslab.com/dashboard.",
    )
