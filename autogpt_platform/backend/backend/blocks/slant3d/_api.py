from enum import Enum
from typing import Literal

from pydantic import BaseModel, SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

Slant3DCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.SLANT3D], Literal["api_key"]
]


def Slant3DCredentialsField() -> Slant3DCredentialsInput:
    return CredentialsField(description="Slant3D API key for authentication")


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="slant3d",
    api_key=SecretStr("mock-slant3d-api-key"),
    title="Mock Slant3D API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


class CustomerDetails(BaseModel):
    name: str
    email: str
    phone: str
    address: str
    city: str
    state: str
    zip: str
    country_iso: str = "US"
    is_residential: bool = True


class Color(Enum):
    WHITE = "white"
    BLACK = "black"


class Profile(Enum):
    PLA = "PLA"
    PETG = "PETG"


class OrderItem(BaseModel):
    # filename: str
    file_url: str
    quantity: str  # String as per API spec
    color: Color = Color.WHITE
    profile: Profile = Profile.PLA
    # image_url: str = ""
    # sku: str = ""


class Filament(BaseModel):
    filament: str
    hexColor: str
    colorTag: str
    profile: str
