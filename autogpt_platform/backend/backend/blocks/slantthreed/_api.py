from typing import Literal, Optional
from pydantic import BaseModel
from backend.data.model import CredentialsField, CredentialsMetaInput
from backend.data.block import BlockSchema

Slant3DCredentialsInput = CredentialsMetaInput[
    Literal["slant3d"], Literal["api_key"]
]


def Slant3DCredentialsField() -> Slant3DCredentialsInput:
    return CredentialsField(
        provider="slant3d",
        supported_credential_types={"api_key"},
        description="Slant3D API key for authentication",
    )


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


class OrderItem(BaseModel):
    filename: str
    file_url: str
    quantity: str  # String as per API spec
    color: str
    profile: str = "PLA"
    image_url: str = ""
    sku: str = ""
