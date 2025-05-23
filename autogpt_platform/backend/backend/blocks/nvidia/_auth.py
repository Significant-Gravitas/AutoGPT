from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName
from backend.util.settings import Settings

NvidiaCredentials = APIKeyCredentials
NvidiaCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.NVIDIA],
    Literal["api_key"],
]

DEFAULT_CREDENTIAL_ID = "96b83908-2789-4dec-9968-18f0ece4ceb3"
ENV_VAR = "nvidia_api_key"
DEFAULT_TITLE = "Use Credits for Nvidia"


def default_credentials(settings: Settings = Settings()) -> APIKeyCredentials | None:
    key = getattr(settings.secrets, ENV_VAR, "")
    if not key and ENV_VAR:
        return None
    if not key:
        key = "FAKE_API_KEY"
    return APIKeyCredentials(
        id=DEFAULT_CREDENTIAL_ID,
        provider=ProviderName.NVIDIA.value,
        api_key=SecretStr(key),
        title=DEFAULT_TITLE,
        expires_at=None,
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="nvidia",
    api_key=SecretStr("mock-nvidia-api-key"),
    title="Mock Nvidia API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def NvidiaCredentialsField() -> NvidiaCredentialsInput:
    """Creates an Nvidia credentials input on a block."""
    return CredentialsField(description="The Nvidia integration requires an API Key.")
