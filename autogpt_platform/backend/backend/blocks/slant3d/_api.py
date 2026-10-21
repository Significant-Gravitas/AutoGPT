from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName
from backend.util.type import MediaFileType

Slant3DCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.SLANT3D], Literal["api_key"]
]


def Slant3DCredentialsField() -> Slant3DCredentialsInput:
    return CredentialsField(description="Slant3D v2 API key for Bearer authentication")


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
    phone: str = ""
    address: str
    city: str
    state: str
    zip: str
    country_iso: str = "US"
    is_residential: bool = True
    address_line2: str = ""


class Color(Enum):
    WHITE = "white"
    BLACK = "black"


class Profile(Enum):
    PLA = "PLA"
    PETG = "PETG"
    OPM = "OPM"


class OrderItem(BaseModel):
    file_url: MediaFileType = Field(
        default=MediaFileType(""),
        description="STL file URL, workspace file, or data URI; ignored when file_id is set",
    )
    file_id: str = Field(
        default="", description="Uploaded Slant3D public file service ID"
    )
    quantity: int = Field(ge=1)
    color: str = "white"
    profile: Profile = Profile.PLA
    filament_id: str = Field(
        default="", description="Filament public ID; overrides color and profile"
    )

    @model_validator(mode="after")
    def require_file(self):
        if not self.file_id and not self.file_url:
            raise ValueError("Provide file_id or file_url for each print item")
        return self


class Filament(BaseModel):
    publicId: str
    name: str
    provider: str
    profile: str
    color: str
    hexValue: str
    available: bool | None = None
    imageURL: str | None = None
    filament: str
    hexColor: str
    colorTag: str
