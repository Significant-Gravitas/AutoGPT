from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

# Define the type of credentials input expected for Example API
AyrshareCredentials = APIKeyCredentials
AyrshareCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.AYRSHARE], Literal["api_key"]
]

# Mock credentials for testing Example API integration
AYRSHARE_CREDENTIALS = APIKeyCredentials(
    id="8274be87-67a7-445d-bbc3-d27a017100f0",
    provider="ayrshare-provider",
    api_key=SecretStr("mock-ayrshare-api-key"),
    title="Mock Ayrshare API key",
    expires_at=None,
)

# Dictionary representation of test credentials for input fields
AYRSHARE_CREDENTIALS_INPUT = {
    "provider": AYRSHARE_CREDENTIALS.provider,
    "id": AYRSHARE_CREDENTIALS.id,
    "type": AYRSHARE_CREDENTIALS.type,
    "title": AYRSHARE_CREDENTIALS.title,
}


def AyrshareCredentialsField() -> AyrshareCredentialsInput:
    """Creates an Ayrshare credentials input on a block."""
    return CredentialsField(description="The Ayrshare integration requires an API Key.")
