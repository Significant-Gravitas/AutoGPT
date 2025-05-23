from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName
from backend.util.settings import Settings

SmartLeadCredentials = APIKeyCredentials
SmartLeadCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.SMARTLEAD],
    Literal["api_key"],
]

DEFAULT_CREDENTIAL_ID = "3bcdbda3-84a3-46af-8fdb-bfd2472298b8"
ENV_VAR = "smartlead_api_key"
DEFAULT_TITLE = "Use Credits for SmartLead"


def default_credentials(settings: Settings = Settings()) -> APIKeyCredentials | None:
    key = getattr(settings.secrets, ENV_VAR, "")
    if not key and ENV_VAR:
        return None
    if not key:
        key = "FAKE_API_KEY"
    return APIKeyCredentials(
        id=DEFAULT_CREDENTIAL_ID,
        provider=ProviderName.SMARTLEAD.value,
        api_key=SecretStr(key),
        title=DEFAULT_TITLE,
        expires_at=None,
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="smartlead",
    api_key=SecretStr("mock-smartlead-api-key"),
    title="Mock SmartLead API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def SmartLeadCredentialsField() -> SmartLeadCredentialsInput:
    """
    Creates a SmartLead credentials input on a block.
    """
    return CredentialsField(
        description="The SmartLead integration can be used with an API Key.",
    )
