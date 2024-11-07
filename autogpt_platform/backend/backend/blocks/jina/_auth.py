from typing import Literal

from autogpt_libs.supabase_integration_credentials_store.types import APIKeyCredentials
from pydantic import SecretStr

from backend.data.model import CredentialsField, CredentialsMetaInput

JinaCredentials = APIKeyCredentials
JinaCredentialsInput = CredentialsMetaInput[
    Literal["jina"],
    Literal["api_key"],
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="jina",
    api_key=SecretStr("mock-jina-api-key"),
    title="Mock Jina API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}


def JinaCredentialsField() -> JinaCredentialsInput:
    """
    Creates a Jina credentials input on a block.

    """
    return CredentialsField(
        provider="jina",
        supported_credential_types={"api_key"},
        description="The Jina integration can be used with an API Key.",
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="jina",
    api_key=SecretStr("mock-jina-api-key"),
    title="Mock Jina API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}
