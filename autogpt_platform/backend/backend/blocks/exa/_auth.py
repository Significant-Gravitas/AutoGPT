from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName
from backend.util.settings import Settings

ExaCredentials = APIKeyCredentials
ExaCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.EXA],
    Literal["api_key"],
]

DEFAULT_CREDENTIAL_ID = "96153e04-9c6c-4486-895f-5bb683b1ecec"
ENV_VAR = "exa_api_key"
DEFAULT_TITLE = "Use Credits for Exa search"


def default_credentials(settings: Settings = Settings()) -> APIKeyCredentials | None:
    key = getattr(settings.secrets, ENV_VAR, "")
    if not key and ENV_VAR:
        return None
    if not key:
        key = "FAKE_API_KEY"
    return APIKeyCredentials(
        id=DEFAULT_CREDENTIAL_ID,
        provider=ProviderName.EXA.value,
        api_key=SecretStr(key),
        title=DEFAULT_TITLE,
        expires_at=None,
    )


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
    return CredentialsField(description="The Exa integration requires an API Key.")
