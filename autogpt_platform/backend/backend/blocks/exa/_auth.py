from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput

ExaCredentials = APIKeyCredentials
ExaCredentialsInput = CredentialsMetaInput[
    Literal["exa"],
    Literal["api_key"],
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="exa",
    api_key=SecretStr("mock-exa-api-key"),
    title="Mock Exa API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def ExaCredentialsField() -> ExaCredentialsInput:
    """Creates an Exa credentials input on a block."""
    return CredentialsField(
        provider="exa",
        supported_credential_types={"api_key"},
        description="The Exa integration requires an API Key.",
    )
