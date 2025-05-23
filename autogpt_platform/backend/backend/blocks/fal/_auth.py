from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName
from backend.util.settings import Settings

FalCredentials = APIKeyCredentials
FalCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.FAL],
    Literal["api_key"],
]

DEFAULT_CREDENTIAL_ID = "6c0f5bd0-9008-4638-9d79-4b40b631803e"
ENV_VAR = "fal_api_key"
DEFAULT_TITLE = "Use Credits for FAL"


def default_credentials(settings: Settings = Settings()) -> APIKeyCredentials | None:
    key = getattr(settings.secrets, ENV_VAR, "")
    if not key and ENV_VAR:
        return None
    if not key:
        key = "FAKE_API_KEY"
    return APIKeyCredentials(
        id=DEFAULT_CREDENTIAL_ID,
        provider=ProviderName.FAL.value,
        api_key=SecretStr(key),
        title=DEFAULT_TITLE,
        expires_at=None,
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="fal",
    api_key=SecretStr("mock-fal-api-key"),
    title="Mock FAL API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def FalCredentialsField() -> FalCredentialsInput:
    """
    Creates a FAL credentials input on a block.
    """
    return CredentialsField(
        description="The FAL integration can be used with an API Key.",
    )
