from typing import Literal

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName
from pydantic import SecretStr

SardisCredentials = APIKeyCredentials
SardisCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.SARDIS],
    Literal["api_key"],
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="sardis",
    api_key=SecretStr("mock-sardis-api-key"),
    title="Mock Sardis API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def SardisCredentialsField() -> SardisCredentialsInput:
    """Creates a Sardis credentials input on a block."""
    return CredentialsField(
        description="The Sardis integration requires an API key. "
        "Get one at https://sardis.sh",
    )
